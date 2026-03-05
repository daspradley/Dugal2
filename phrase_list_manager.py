"""
Azure Speech Phrase List Manager for Dugal

This module manages vocabulary extraction from Excel documents and loads
them into Azure Speech Services for dramatically improved recognition accuracy.

Features:
- Extracts vocabulary from any Excel file
- Caches phrases in memory for performance
- Optional JSON persistence for faster reopening
- Dynamic updates as inventory changes
- Document-agnostic - works with ANY Excel file!

Based on Microsoft Word's dictation approach.
"""

import logging
import json
import os
from typing import List, Set, Dict, Optional
from datetime import datetime
import azure.cognitiveservices.speech as speechsdk

logger = logging.getLogger(__name__)


class PhraseListManager:
    """Manages Azure Speech phrase lists from Excel document vocabulary."""
    
    def __init__(self, cache_dir: str = ".dugal_data"):
        """
        Initialize the phrase list manager.
        
        Args:
            cache_dir: Directory to store cached phrase lists
        """
        self.cache_dir = cache_dir
        self.phrases: Set[str] = set()
        self.phrase_variations: Dict[str, List[str]] = {}
        self.source_file: Optional[str] = None
        self.last_updated: Optional[datetime] = None
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info("PhraseListManager initialized")
    
    def extract_phrases_from_workbook(self, workbook, search_column: str = "A") -> Set[str]:
        """
        Extract all unique phrases from an Excel workbook.
        
        Args:
            workbook: openpyxl workbook object
            search_column: Column to extract phrases from (default: "A")
            
        Returns:
            Set of unique phrases
        """
        logger.info(f"Extracting phrases from workbook, column {search_column}")
        
        phrases = set()
        
        # Get all sheets (excluding hidden/system sheets)
        sheets_to_process = [
            sheet for sheet in workbook.sheetnames
            if not sheet.startswith('_') and sheet not in ['TOTALS', 'Free ', 'Kitchen Alcohol']
        ]
        
        for sheet_name in sheets_to_process:
            try:
                sheet = workbook[sheet_name]
                logger.debug(f"Processing sheet: {sheet_name}")
                
                # Find the search column
                col_letter = search_column.upper()
                
                # Iterate through rows (skip first 10 for headers)
                for row_idx, row in enumerate(sheet.iter_rows(min_row=11, values_only=True), start=11):
                    # Get the cell value from search column
                    col_idx = ord(col_letter) - ord('A')
                    
                    if col_idx < len(row) and row[col_idx]:
                        value = str(row[col_idx]).strip()
                        
                        # Skip empty or header-like entries
                        if value and len(value) >= 2:
                            # Skip entries with "MASTER INVENTORY" or similar
                            if "MASTER INVENTORY" not in value.upper() and \
                               "COST SHEET" not in value.upper():
                                phrases.add(value)
                
                logger.debug(f"Sheet {sheet_name}: extracted {len(phrases)} phrases so far")
                
            except Exception as e:
                logger.error(f"Error processing sheet {sheet_name}: {e}")
                continue
        
        logger.info(f"✅ Extracted {len(phrases)} unique phrases from workbook")
        return phrases
    
    def generate_variations(self, phrases: Set[str]) -> Dict[str, List[str]]:
        """
        Generate common variations of phrases for better recognition.
        
        Args:
            phrases: Set of base phrases
            
        Returns:
            Dictionary mapping base phrase to list of variations
        """
        logger.info(f"Generating variations for {len(phrases)} phrases")
        
        variations = {}
        
        for phrase in phrases:
            phrase_vars = [phrase]  # Always include original
            
            # Lowercase version
            if phrase != phrase.lower():
                phrase_vars.append(phrase.lower())
            
            # Remove punctuation variations
            no_punct = phrase.replace("'", "").replace("-", " ").replace(".", "")
            if no_punct != phrase:
                phrase_vars.append(no_punct)
                phrase_vars.append(no_punct.lower())
            
            # Split multi-word phrases and add individual words
            words = phrase.split()
            if len(words) > 1:
                # Add first word (brand name)
                phrase_vars.append(words[0])
                phrase_vars.append(words[0].lower())
                
                # Add last word (product type)
                phrase_vars.append(words[-1])
                phrase_vars.append(words[-1].lower())
            
            # Common abbreviation handling
            abbrev_map = {
                'Double': 'Dbl',
                'Reserve': 'Res',
                'Single': 'Sngl',
                'Barrel': 'Brl',
                'Whiskey': 'Whisky'
            }
            
            for full, abbrev in abbrev_map.items():
                if full in phrase:
                    phrase_vars.append(phrase.replace(full, abbrev))
                    phrase_vars.append(phrase.replace(full, abbrev).lower())
                if abbrev in phrase:
                    phrase_vars.append(phrase.replace(abbrev, full))
                    phrase_vars.append(phrase.replace(abbrev, full).lower())
            
            # Store unique variations
            variations[phrase] = list(set(phrase_vars))
        
        total_variations = sum(len(v) for v in variations.values())
        logger.info(f"✅ Generated {total_variations} total phrase variations")
        
        return variations
    
    def load_phrases_into_azure(self, recognizer: speechsdk.SpeechRecognizer, 
                                max_phrases: int = 5000) -> int:
        """
        Load cached phrases into Azure Speech recognizer.
        
        Args:
            recognizer: Azure SpeechRecognizer object
            max_phrases: Maximum phrases to load (Azure limit is ~5000)
            
        Returns:
            Number of phrases loaded
        """
        if not self.phrases:
            logger.warning("No phrases cached - cannot load into Azure")
            return 0
        
        logger.info(f"Loading phrases into Azure Speech recognizer")
        
        try:
            # Create phrase list grammar
            phrase_list_grammar = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
            
            # Flatten all variations
            all_phrases = []
            for base_phrase, variations in self.phrase_variations.items():
                all_phrases.extend(variations)
            
            # Remove duplicates and limit to max
            unique_phrases = list(set(all_phrases))[:max_phrases]
            
            # Add phrases to Azure
            for phrase in unique_phrases:
                phrase_list_grammar.addPhrase(phrase)
            
            logger.info(f"✅ Loaded {len(unique_phrases)} phrases into Azure Speech")
            return len(unique_phrases)
            
        except Exception as e:
            logger.error(f"Error loading phrases into Azure: {e}")
            return 0
    
    def cache_phrases(self, phrases: Set[str], source_file: str):
        """
        Cache phrases in memory and optionally save to JSON.
        
        Args:
            phrases: Set of phrases to cache
            source_file: Path to the source Excel file
        """
        self.phrases = phrases
        self.source_file = source_file
        self.last_updated = datetime.now()
        
        # Generate variations
        self.phrase_variations = self.generate_variations(phrases)
        
        logger.info(f"Cached {len(phrases)} phrases from {source_file}")
    
    def save_cache_to_json(self, filename: str = "phrase_cache.json"):
        """
        Save cached phrases to JSON file for faster reopening.
        
        Args:
            filename: Name of the cache file
        """
        if not self.phrases:
            logger.warning("No phrases to save")
            return
        
        cache_path = os.path.join(self.cache_dir, filename)
        
        cache_data = {
            "phrases": list(self.phrases),
            "source_file": self.source_file,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "variations": self.phrase_variations
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved phrase cache to {cache_path}")
            
        except Exception as e:
            logger.error(f"Error saving phrase cache: {e}")
    
    def load_cache_from_json(self, filename: str = "phrase_cache.json") -> bool:
        """
        Load cached phrases from JSON file.
        
        Args:
            filename: Name of the cache file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        cache_path = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(cache_path):
            logger.debug(f"No cache file found at {cache_path}")
            return False
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            self.phrases = set(cache_data.get("phrases", []))
            self.source_file = cache_data.get("source_file")
            self.phrase_variations = cache_data.get("variations", {})
            
            last_updated_str = cache_data.get("last_updated")
            if last_updated_str:
                self.last_updated = datetime.fromisoformat(last_updated_str)
            
            logger.info(f"✅ Loaded {len(self.phrases)} phrases from cache")
            return True
            
        except Exception as e:
            logger.error(f"Error loading phrase cache: {e}")
            return False
    
    def add_phrase(self, phrase: str):
        """
        Add a new phrase to the cache (for dynamic updates).
        
        Args:
            phrase: New phrase to add
        """
        if phrase and phrase not in self.phrases:
            self.phrases.add(phrase)
            
            # Generate variations for new phrase
            variations = self.generate_variations({phrase})
            self.phrase_variations.update(variations)
            
            self.last_updated = datetime.now()
            
            logger.debug(f"Added new phrase: {phrase}")
    
    def get_phrase_count(self) -> int:
        """Get the total number of cached phrases."""
        return len(self.phrases)
    
    def get_variation_count(self) -> int:
        """Get the total number of phrase variations."""
        return sum(len(v) for v in self.phrase_variations.values())
    
    def clear_cache(self):
        """Clear all cached phrases."""
        self.phrases.clear()
        self.phrase_variations.clear()
        self.source_file = None
        self.last_updated = None
        logger.info("Phrase cache cleared")


# Example usage / integration function
def setup_azure_phrases_from_workbook(workbook, recognizer: speechsdk.SpeechRecognizer,
                                     search_column: str = "A",
                                     cache_manager: Optional[PhraseListManager] = None) -> PhraseListManager:
    """
    Complete workflow: Extract phrases from workbook and load into Azure.
    
    Args:
        workbook: openpyxl workbook object
        recognizer: Azure SpeechRecognizer
        search_column: Column to extract from (default "A")
        cache_manager: Existing PhraseListManager (creates new if None)
        
    Returns:
        PhraseListManager with cached phrases
    """
    # Create manager if not provided
    if cache_manager is None:
        cache_manager = PhraseListManager()
    
    # Extract phrases from workbook
    phrases = cache_manager.extract_phrases_from_workbook(workbook, search_column)
    
    # Cache phrases
    cache_manager.cache_phrases(phrases, source_file=getattr(workbook, '_file_path', 'unknown'))
    
    # Load into Azure
    loaded_count = cache_manager.load_phrases_into_azure(recognizer)
    
    logger.info(f"✅ Setup complete: {loaded_count} phrases loaded into Azure Speech")
    
    return cache_manager
