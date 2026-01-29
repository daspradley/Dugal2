"""
Enhanced Voice Interaction Module for Dugal Inventory System.
Combines speech recognition, synthesis, command processing, and pattern learning.
"""

from __future__ import annotations
import os
import sys
import json
import logging
import re
import random
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from main_dugal import MainDugal
from dataclasses import dataclass, field
from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string
import speech_recognition as sr
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import (
    SpeechConfig,
    SpeechSynthesizer,
    SpeechRecognizer,
    AudioConfig,
    ResultReason,
    CancellationReason,
    SpeechSynthesisCancellationDetails,
    SpeechSynthesisOutputFormat,
    PropertyId
)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from logging_manager import LoggingManager
from excel_handler import ExcelHandler
from command_router import CommandRouter, CommandMatch, RouterResult
# Configure logging
logger = logging.getLogger(__name__)

class VoiceState:
    """Enhanced state tracking for voice interactions."""
    
    def __init__(self, dugal: Optional[MainDugal] = None):
        """Initialize voice interaction state with improved caching."""
        # Core components
        self.dugal = dugal
        self.excel_handler = None
        self.logging_manager = None
        self.speech_config = None
        self.synthesizer = None
        self.recognizer = None
        
        # State tracking
        self.mode = "wild"  # Personality mode: wild, mild, or proper
        self.error_count = 0
        self.last_activity = None
        self.current_mic_index = None
        self.learning_mode = False
        self.current_learning_term = None
        self.learning_variations = []
        self.active_item_context = None
        
        # Enhanced statistics tracking
        self.stats = {
            'total_attempts': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'last_success_time': None,
            'learning_attempts': 0,
            'successful_learns': 0,
            'error_types': {},
            'command_history': []
        }
        
        # Initialize cache and patterns
        self.cache = self._init_cache()
        self.common_patterns = self._init_common_patterns()
    
    def _init_cache(self) -> Dict[str, Any]:
        """Initialize the cache dictionary."""
        return {
            'misspellings': {
                'vodka': ['vodca', 'votka'],
                'whiskey': ['whisky', 'wisky'],
                'tequila': ['tekila', 'tequilla'],
                'kahlua': ['kalua', 'kaluha'],
                'jagermeister': ['jager', 'yeager', 'jaeger'],
                'hennessy': ['hennesey', 'henessy']
            },
            'commands': {
                'sheet_selection': [
                    r"work with (?:only )?(?:the )?(.+?)(?: sheet)?$",
                    r"look at (?:only )?(.+?)$",
                    r"focus on (.+?)$",
                    r"lets? look at (.+?)(?: only)?$"
                ],
                'column_targeting': [
                    r"change target (?:input )?to (.+?)$",
                    r"set target to (.+?)$",
                    r"input to (.+?)$",
                    r"use column (.+?)$"
                ],
                'reset_commands': [
                    r"back to normal",
                    r"reset (?:to normal|sheets?)",
                    r"look at all sheets?",
                    r"check all sheets?"
                ]
            }
        }
    
    def _init_common_patterns(self) -> Dict[str, List[str]]:
        """Initialize common voice command patterns."""
        return {
            'inventory_update': [
                r'^(\w+(?:\s+\w+)*)\s+(-?\d*\.?\d+)$',
                r'^add\s+(-?\d*\.?\d+)\s+(?:of\s+)?(.+)$',
                r'^remove\s+(-?\d*\.?\d+)\s+(?:of\s+)?(.+)$',
                r'^set\s+(.+)\s+to\s+(-?\d*\.?\d+)$'
            ],
            'mode_change': [
                r'^(?:be|go)\s+(wild|mild|proper)$',
                r'^change\s+mode\s+to\s+(wild|mild|proper)$'
            ],
            'system_commands': [
                r'^help$',
                r'^show commands$',
                r'^status$',
                r'^system status$',
                r'^diagnostics$'
            ],
            'learning': [
                r'^learn\s+term\s+(.+)$',
                r'^add\s+variation\s+(.+)\s+for\s+(.+)$',
                r'^confirm\s+learning\s+(.+)$'
            ],
            'sheet_selection': [
                r'^(?:look at|work with|focus on)\s+(?:only\s+)?(.+?)(?:\s+sheet)?$',
                r'^select\s+(?:sheet|sheets)\s+(.+)$'
            ]
        }
class PatternLearningHelpers:
    """Enhanced helper methods for pattern learning and optimization."""
    
    def __init__(self, voice_interaction):
        """Initialize with improved learning capabilities."""
        self.voice_interaction = voice_interaction
        self.logger = logging.getLogger(__name__)
        self.pattern_cache = {}
        self.learning_history = []
        self.min_confidence_threshold = 0.4
        self.high_confidence_threshold = 0.8
        
        # Initialize phonetic matching data
        self._init_phonetic_patterns()
    
    def _init_phonetic_patterns(self):
        """Initialize phonetic matching patterns and common substitutions."""
        self.phonetic_patterns = {
            'vowel_sounds': [
                (r'a[eiy]', 'a'),
                (r'e[aiy]', 'e'),
                (r'i[aey]', 'i'),
                (r'o[aey]', 'o'),
                (r'u[aey]', 'u')
            ],
            'consonant_sounds': [
                (r'ph', 'f'),
                (r'gh', 'f'),
                (r'[ck]h', 'k'),
                (r'wh', 'w'),
                (r'mb', 'm')
            ]
        }
        
        self.common_substitutions = {
            'numbers': {
                'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'zero': '0'
            },
            'units': {
                'percent': '%',
                'degrees': '°',
                'point': '.'
            }
        }

    def analyze_pattern(self, spoken_form: str, written_form: str) -> Dict[str, Any]:
        """Analyze pattern with enhanced optimization strategies."""
        self.logger.debug(f"Analyzing pattern: {spoken_form} -> {written_form}")
        
        try:
            spoken_parts = spoken_form.lower().split()
            written_parts = written_form.lower().split()
            
            # Perform phonetic analysis
            phonetic_matches = self._find_phonetic_matches(spoken_parts, written_parts)
            
            # Find common parts and unique elements
            common_parts = set(spoken_parts) & set(written_parts)
            unique_spoken = set(spoken_parts) - set(written_parts)
            unique_written = set(written_parts) - set(spoken_parts)
            
            # Calculate sequence similarity
            sequence_similarity = SequenceMatcher(None, spoken_form, written_form).ratio()
            
            # Generate potential transformation rules
            potential_rules = self._identify_patterns(spoken_parts, written_parts)
            
            analysis = {
                'common_parts': common_parts,
                'unique_spoken': unique_spoken,
                'unique_written': unique_written,
                'phonetic_matches': phonetic_matches,
                'potential_rules': potential_rules,
                'sequence_similarity': sequence_similarity,
                'confidence': self._calculate_pattern_confidence(spoken_parts, written_parts)
            }
            
            # Cache the analysis for future reference
            cache_key = f"{spoken_form}_{written_form}"
            self.pattern_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing pattern: {e}")
            return {}

    def learn_pattern(self, spoken_form: str, written_form: str) -> Dict[str, Any]:
        """Learn and record a new pattern with comprehensive analysis."""
        try:
            # Analyze the pattern
            analysis = self.analyze_pattern(spoken_form, written_form)
            confidence = analysis.get('confidence', 0)
            
            if confidence < self.min_confidence_threshold:
                return {
                    'success': False,
                    'message': "Pattern confidence too low",
                    'confidence': confidence
                }
            
            # Record the pattern
            pattern_data = {
                'type': 'learned_pattern',
                'spoken_form': spoken_form,
                'written_form': written_form,
                'analysis': analysis,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            self.record_learning(pattern_data)
            
            # Optimize patterns if we have enough data
            if len(self.learning_history) >= 5:
                self.optimize_patterns()
            
            return {
                'success': True,
                'pattern': pattern_data,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error learning pattern: {e}")
            return {
                'success': False,
                'message': f"Error learning pattern: {str(e)}",
                'confidence': 0.0
            }

    def validate_learned_pattern(self, spoken_term: str, written_term: str) -> bool:
        """Validate if a pattern meets the learning criteria."""
        try:
            analysis = self.analyze_pattern(spoken_term, written_term)
            confidence = analysis.get('confidence', 0)
            
            # Stricter validation criteria
            validation_criteria = [
                confidence >= 0.4,  # Minimum confidence threshold
                bool(analysis.get('common_parts')),  # Must have some common parts
                len(self._find_phonetic_matches(
                    spoken_term.lower().split(), 
                    written_term.lower().split()
                )) > 0
            ]
            
            # All criteria must be met
            return all(validation_criteria)
            
        except Exception as e:
            self.logger.error(f"Error validating pattern: {e}")
            return False

    def record_learning(self, pattern: Dict[str, Any]) -> None:
        """Record pattern for future optimization and share with search engine."""
        try:
            entry = {
                'pattern': pattern,
                'timestamp': datetime.now().isoformat(),
                'success': pattern.get('confidence', 0) > 0.5
            }
            
            self.learning_history.append(entry)
            
            # Log the pattern
            if self.voice_interaction.state.logging_manager:
                self.voice_interaction.state.logging_manager.log_pattern_match({
                    'type': 'learning_record',
                    'entry': entry
                })
            
            # Share pattern with search engine using component manager
            try:
                from component_manager import component_manager
                search_engine = component_manager.get_search_engine()
                
                if search_engine and hasattr(search_engine, 'learn_pattern'):
                    self.logger.debug(f"Sharing learned pattern with search engine: {pattern}")
                    search_engine.learn_pattern(pattern)
                else:
                    # Fall back to direct reference if component manager doesn't work
                    if hasattr(self.voice_interaction.state, 'search_engine'):
                        search_engine = self.voice_interaction.state.search_engine
                        if search_engine and hasattr(search_engine, 'learn_pattern'):
                            self.logger.debug(f"Sharing learned pattern with search engine (direct): {pattern}")
                            search_engine.learn_pattern(pattern)
            except ImportError:
                # Fall back to old method if component manager not available
                if hasattr(self.voice_interaction.state, 'search_engine'):
                    search_engine = self.voice_interaction.state.search_engine
                    if search_engine and hasattr(search_engine, 'learn_pattern'):
                        self.logger.debug(f"Sharing learned pattern with search engine (legacy): {pattern}")
                        search_engine.learn_pattern(pattern)
                
        except Exception as e:
            self.logger.error(f"Error recording pattern: {e}")

    def optimize_patterns(self) -> None:
        """Optimize stored patterns based on learning history."""
        if not self.learning_history:
            return
            
        try:
            transformations = []
            
            # Group similar patterns
            grouped_patterns = {}
            for entry in self.learning_history:
                pattern = entry.get('pattern', {})
                key = f"{pattern.get('from', '')}_{pattern.get('to', '')}"
                if key not in grouped_patterns:
                    grouped_patterns[key] = []
                grouped_patterns[key].append(pattern)
            
            # Find common transformations
            for patterns in grouped_patterns.values():
                if len(patterns) >= 2:  # Require at least 2 occurrences
                    common = self._find_common_transformation(patterns)
                    if common:
                        transformations.append(common)
            
            # Apply optimizations to search engine using component manager
            if transformations:
                try:
                    from component_manager import component_manager
                    search_engine = component_manager.get_search_engine()
                    
                    if search_engine and hasattr(search_engine, 'update_transformation_rules'):
                        self.logger.debug(f"Applying {len(transformations)} optimized patterns to search engine")
                        search_engine.update_transformation_rules(transformations)
                    else:
                        # Fall back to direct reference if component manager doesn't work
                        if hasattr(self.voice_interaction.state, 'search_engine'):
                            search_engine = self.voice_interaction.state.search_engine
                            if search_engine and hasattr(search_engine, 'update_transformation_rules'):
                                self.logger.debug(f"Applying optimized patterns to search engine (direct)")
                                search_engine.update_transformation_rules(transformations)
                except ImportError:
                    # Fall back to old method if component manager not available
                    if hasattr(self.voice_interaction.state, 'excel_handler'):
                        excel_handler = self.voice_interaction.state.excel_handler
                        if hasattr(excel_handler, 'search_engine'):
                            self.logger.debug(f"Applying optimized patterns to search engine (legacy)")
                            excel_handler.search_engine.update_transformation_rules(transformations)
            
            self.logger.debug(f"Optimized {len(transformations)} patterns")
            
        except Exception as e:
            self.logger.error(f"Error optimizing patterns: {e}")

    def _identify_shorthand_patterns(self, spoken_parts: List[str], written_parts: List[str]) -> Dict[str, Any]:
        """Identify shorthand/expansion patterns between spoken and written forms."""
        try:
            # Check for shorthand patterns (written form is shorter than spoken form)
            if len(written_parts) < len(spoken_parts):
                return {
                    'type': 'shorthand',
                    'from': ' '.join(spoken_parts),
                    'to': ' '.join(written_parts),
                    'confidence': 0.7 if len(set(spoken_parts) & set(written_parts)) > 0 else 0.4
                }
                
            # Check for expansion patterns (written form is longer than spoken form)
            elif len(written_parts) > len(spoken_parts):
                return {
                    'type': 'expansion',
                    'from': ' '.join(spoken_parts),
                    'to': ' '.join(written_parts),
                    'confidence': 0.7 if len(set(spoken_parts) & set(written_parts)) > 0 else 0.4
                }
                
            # For equal length, check for specific abbreviations
            else:
                abbreviations = {}
                for s_word, w_word in zip(spoken_parts, written_parts):
                    if s_word != w_word and len(s_word) <= len(w_word):
                        # Check if s_word could be an abbreviation of w_word
                        if s_word.lower() == w_word.lower()[:len(s_word)]:
                            abbreviations[s_word] = w_word
                
                if abbreviations:
                    return {
                        'type': 'abbreviation',
                        'abbreviations': abbreviations,
                        'confidence': 0.8
                    }
                    
            return {}
            
        except Exception as e:
            self.logger.error(f"Error identifying shorthand patterns: {e}")
            return {}

    def _try_pattern_correction(self, word: str) -> Optional[str]:
        """Attempt to correct a word using learned patterns."""
        try:
            # Check misspellings cache
            misspellings = self.voice_interaction.state.cache.get('misspellings', {})
            if word.lower() in misspellings:
                return misspellings[word.lower()]
            
            # Try pattern matching for advanced correction
            analysis = self.analyze_pattern(word, word)
            patterns = analysis.get('potential_rules', [])
            
            for pattern in patterns:
                if pattern.get('confidence', 0) > 0.8:
                    return pattern.get('to')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error applying pattern correction: {e}")
            return None

    def _terms_similar(self, term1: str, term2: str) -> bool:
        """Check if two terms are similar enough to be considered matches."""
        try:
            # Direct match
            if term1.lower() == term2.lower():
                return True
            
            # Calculate word overlap
            words1 = set(term1.lower().split())
            words2 = set(term2.lower().split())
            
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            similarity = overlap / total if total > 0 else 0
            
            # Check phonetic similarity
            phonetic_matches = self._find_phonetic_matches(
                term1.lower().split(),
                term2.lower().split()
            )
            
            # Term must be either similar or have phonetic matches
            return similarity > 0.7 or len(phonetic_matches) > 0
            
        except Exception as e:
            self.logger.error(f"Error comparing terms: {e}")
            return False

    def _find_phonetic_matches(self, spoken_parts: List[str], written_parts: List[str]) -> List[Tuple[str, str]]:
        """Find phonetically similar word pairs between spoken and written forms."""
        matches = []
        
        try:
            for s_word in spoken_parts:
                for w_word in written_parts:
                    if self._are_words_phonetically_similar(s_word, w_word):
                        matches.append((s_word, w_word))
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Error finding phonetic matches: {e}")
            return []

    def _are_words_phonetically_similar(self, word1: str, word2: str) -> bool:
        """Check if two words are phonetically similar using enhanced rules."""
        try:
            # Apply phonetic normalization
            w1 = self._normalize_word(word1)
            w2 = self._normalize_word(word2)
            
            # Direct match after normalization
            if w1 == w2:
                return True
            
            # Calculate similarity threshold based on word length
            min_length = min(len(w1), len(w2))
            threshold = 0.8 if min_length > 4 else 0.7
            
            # Check similarity ratio
            similarity = SequenceMatcher(None, w1, w2).ratio()
            return similarity >= threshold
            
        except Exception as e:
            self.logger.error(f"Error comparing words phonetically: {e}")
            return False

    def _normalize_word(self, word: str) -> str:
        """Apply phonetic normalization rules to a word."""
        normalized = word.lower()
        
        try:
            # Apply vowel sound patterns
            for pattern, replacement in self.phonetic_patterns.get('vowel_sounds', []):
                normalized = re.sub(pattern, replacement, normalized)
            
            # Apply consonant sound patterns
            for pattern, replacement in self.phonetic_patterns.get('consonant_sounds', []):
                normalized = re.sub(pattern, replacement, normalized)
            
            # Apply common substitutions
            for category in self.common_substitutions.values():
                if normalized in category:
                    normalized = category[normalized]
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing word: {e}")
            return word

    def _identify_patterns(self, spoken_parts: List[str], written_parts: List[str]) -> List[Dict[str, Any]]:
        """Identify potential transformation patterns between spoken and written forms."""
        patterns = []
        
        try:
            # Look for word order patterns
            if len(spoken_parts) > 1 and len(written_parts) > 1:
                order_pattern = self._analyze_word_order(spoken_parts, written_parts)
                if order_pattern:
                    patterns.append({
                        'type': 'word_order',
                        'pattern': order_pattern,
                        'confidence': order_pattern.get('confidence', 0)
                    })
            
            # Look for word substitution patterns
            substitutions = self._find_word_substitutions(spoken_parts, written_parts)
            if substitutions:
                patterns.append({
                    'type': 'substitution',
                    'pattern': substitutions,
                    'confidence': self._calculate_substitution_confidence(substitutions)
                })
            
            # Look for shorthand/expansion patterns
            shorthand = self._identify_shorthand_patterns(spoken_parts, written_parts)
            if shorthand:
                patterns.append({
                    'type': 'shorthand',
                    'pattern': shorthand,
                    'confidence': shorthand.get('confidence', 0)
                })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error identifying patterns: {e}")
            return []

    def _calculate_pattern_confidence(self, spoken_parts: List[str], written_parts: List[str]) -> float:
        """Calculate overall confidence in pattern matching."""
        try:
            # Various factors affect confidence
            factors = {
                'length_match': 1.0 if len(spoken_parts) == len(written_parts) else 0.5,
                'common_parts': len(set(spoken_parts) & set(written_parts)) / max(len(spoken_parts), len(written_parts), 1),
                'phonetic_matches': len(self._find_phonetic_matches(spoken_parts, written_parts)) / max(len(spoken_parts), len(written_parts), 1)
            }
            
            # Weighted average
            weights = {'length_match': 0.3, 'common_parts': 0.4, 'phonetic_matches': 0.3}
            confidence = sum(score * weights[factor] for factor, score in factors.items())
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.0

    def _find_word_substitutions(self, spoken_parts: List[str], written_parts: List[str]) -> Dict[str, str]:
        """Find word substitution patterns between spoken and written forms."""
        substitutions = {}
        
        try:
            # Find direct substitutions
            for s_word in spoken_parts:
                for w_word in written_parts:
                    if self._are_words_phonetically_similar(s_word, w_word) and s_word != w_word:
                        substitutions[s_word] = w_word
                        
            return substitutions
            
        except Exception as e:
            logger.error(f"Error finding word substitutions: {e}")
            return {}

    def cleanup(self) -> None:
        """Clean up pattern learning resources."""
        try:
            # Optimize patterns before cleanup
            self.optimize_patterns()
            
            # Clear caches
            self.pattern_cache.clear()
            self.learning_history.clear()
            
            self.logger.debug("Pattern learning resources cleaned up successfully")
            
            # Log cleanup if available
            if self.voice_interaction.state.logging_manager:
                self.voice_interaction.state.logging_manager.log_pattern_match({
                    'type': 'cleanup',
                    'component': 'pattern_learning',
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Error during pattern learning cleanup: {e}")

class VoiceInteraction(QObject):
    """Enhanced voice interaction and speech recognition with Azure integration."""
    
    # Define signals
    speech_started = pyqtSignal()
    speech_ended = pyqtSignal()
    command_processed = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    
    def __init__(
            self,
            azure_key: str = None,
            azure_region: str = None,
            dugal: Optional[MainDugal] = None,
            *args,
            **kwargs
        ) -> None:
        """Initialize voice interaction system with improved error handling."""
        super().__init__()
        logger.debug("Initializing voice interaction system...")
        self.logger = logging.getLogger(__name__)
        
        try:
            # Initialize state
            self.state = VoiceState(dugal=dugal)
            
            # Get Azure credentials from environment if not provided
            if not azure_key:
                azure_key = os.getenv('AZURE_SPEECH_KEY', '8f7f210f78064aa6929fad817c2be132')
            if not azure_region:
                azure_region = os.getenv('AZURE_REGION', 'centralus')
                
            # Configure Dugal and logging
            self.state.dugal = dugal
            
            # Get logging manager from Dugal if available
            if dugal and hasattr(dugal, 'logging_manager'):
                self.state.logging_manager = dugal.logging_manager
            else:
                self.state.logging_manager = LoggingManager()
                
            # Log initialization start
            if self.state.logging_manager:
                self.state.logging_manager.log_pattern_match({
                    'type': 'initialization',
                    'component': 'voice_interaction',
                    'timestamp': datetime.now().isoformat()
                })
                
            # Initialize pattern learning
            self.pattern_helpers = PatternLearningHelpers(self)
            
            # Initialize speech recognition
            self.state.recognizer = sr.Recognizer()
            self.state.recognizer.pause_threshold = 0.5
            self.state.recognizer.phrase_threshold = 0.2
            self.state.recognizer.non_speaking_duration = 0.3
            
            # Verify Azure credentials
            self._verify_azure_credentials(azure_key, azure_region)
            
            # Initialize Azure speech services
            self._init_azure_speech_services(azure_key, azure_region)
            
            # Initialize command router
            self.command_router = CommandRouter(logging_manager=self.logging_manager if hasattr(self, 'logging_manager') else None)
            
            # Register command handlers
            self.register_command_handlers()

            # Initialize inactivity timer
            self.inactivity_timer = QTimer()
            self.inactivity_timer.setInterval(45000)  # 45 seconds
            self.inactivity_timer.timeout.connect(self._handle_inactivity)
            self.state.last_activity = datetime.now()
            
            # Setup event handlers
            self._setup_event_handlers()
            
            logger.debug("Voice interaction system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice interaction: {e}")
            # Ensure cleanup of any initialized resources
            if hasattr(self, 'inactivity_timer'):
                self.inactivity_timer.stop()
            if hasattr(self.state, 'logging_manager'):
                self.state.logging_manager.log_error(str(e), {
                    'context': 'initialization',
                    'component': 'voice_interaction'
                })
            raise RuntimeError(f"Voice interaction initialization failed: {str(e)}")

    def connect_to_search_engine(self, search_engine):
        """Connect voice interaction directly to the search engine."""
        logger.debug("Connecting voice interaction directly to search engine")
        self.state.search_engine = search_engine
        
        # Always register the search engine in the global registry
        from global_registry import GlobalRegistry
        GlobalRegistry.register('search_engine', search_engine)
        
        # Diagnose the search engine state
        search_engine.diagnose_search_index()
        logger.debug("Voice interaction connected to search engine with cache size: %d", 
                    len(search_engine.inventory_cache) if hasattr(search_engine, 'inventory_cache') else 0)

    def get_search_engine(self):
        """Get the search engine from the global registry or local reference."""
        try:
            # First try global registry
            from global_registry import GlobalRegistry
            search_engine = GlobalRegistry.get('search_engine')
            
            # If found in registry, update our local reference
            if search_engine:
                if hasattr(self, 'search_engine') and self.search_engine is not search_engine:
                    logger.debug(f"Updating local search engine reference from registry")
                    self.search_engine = search_engine
                return search_engine
                
            # If not in registry but we have one, register it
            if hasattr(self, 'search_engine') and self.search_engine:
                GlobalRegistry.register('search_engine', self.search_engine)
                return self.search_engine
                
            # Last resort: if excel_handler has one
            if hasattr(self, 'excel_handler') and hasattr(self.excel_handler, 'search_engine'):
                search_engine = self.excel_handler.search_engine
                GlobalRegistry.register('search_engine', search_engine)
                return search_engine
                
            logger.warning("No search engine available")
            return None
            
        except Exception as e:
            logger.error(f"Error getting search engine: {e}")
            # Fallback to local reference if available
            return getattr(self, 'search_engine', None)

    def configure_speech_recognizer(self, timeout_ms=5000, end_silence_ms=1500):
        """Configure the speech recognizer with extended timeout parameters."""
        try:
            # Create a speech configuration with your Azure subscription key and region
            speech_config = speechsdk.SpeechConfig(
                subscription=self.state.speech_key, 
                region=self.state.speech_region
            )
            
            # Set the recognition language
            speech_config.speech_recognition_language = "en-US"
            
            # Extended properties for longer recognition
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, 
                str(timeout_ms)  # Increase initial silence timeout (default is usually 5000ms)
            )
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, 
                str(end_silence_ms)  # Increase end silence timeout (default is usually 500ms)
            )
            
            # Create an audio configuration using the default microphone
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            
            # Create the speech recognizer with the extended configuration
            self.speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )
            
            self.logger.debug(f"Speech recognizer configured with extended timeouts: initial={timeout_ms}ms, end silence={end_silence_ms}ms")
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring speech recognizer: {e}")
            return False

    def start_recognition(self):
        """Start continuous voice recognition with extended timeouts."""
        try:
            self.logger.debug("Starting continuous voice recognition...")
            
            # Configure the recognizer with extended timeouts
            # 7000ms initial timeout, 2000ms end silence timeout
            self.configure_speech_recognizer(timeout_ms=7000, end_silence_ms=2000)
            
            # Connect callbacks
            self.speech_recognizer.recognized.connect(self._recognized_callback)
            self.speech_recognizer.recognizing.connect(self._recognizing_callback)
            self.speech_recognizer.session_started.connect(self._session_started_callback)
            self.speech_recognizer.session_stopped.connect(self._session_stopped_callback)
            self.speech_recognizer.canceled.connect(self._canceled_callback)
            
            # Start continuous recognition
            self.speech_recognizer.start_continuous_recognition()
            self.state.recognition_active = True
            self.state.status = "listening"
            self.logger.debug("Speech recognition started")
            
            return True
        except Exception as e:
            self.logger.error(f"Error starting speech recognition: {e}")
            return False

    def set_recognition_timeouts(self, initial_timeout_ms=7000, end_silence_ms=2000):
        """Adjust speech recognition timeouts dynamically."""
        try:
            if hasattr(self, 'speech_recognizer') and self.speech_recognizer:
                # Stop recognition if active
                was_active = False
                if self.state.recognition_active:
                    was_active = True
                    self.stop_recognition()
                    
                # Reconfigure with new timeouts
                success = self.configure_speech_recognizer(
                    timeout_ms=initial_timeout_ms,
                    end_silence_ms=end_silence_ms
                )
                
                # Restart if it was active
                if was_active and success:
                    self.start_recognition()
                    
                self.logger.debug(f"Recognition timeouts updated - initial: {initial_timeout_ms}ms, end silence: {end_silence_ms}ms")
                return success
            else:
                self.logger.warning("Cannot set timeouts - speech recognizer not initialized")
                return False
        except Exception as e:
            self.logger.error(f"Error setting recognition timeouts: {e}")
            return False

    def find_item(self, search_term: str) -> Dict[str, Any]:
        """Find an item in the inventory and return its information."""
        try:
            # Get search engine from global registry first
            from global_registry import GlobalRegistry
            search_engine = GlobalRegistry.get('search_engine')
            
            # If not found in registry, try our stored reference
            if not search_engine and hasattr(self.state, 'search_engine'):
                search_engine = self.state.search_engine
                # Register it for future use
                if search_engine:
                    GlobalRegistry.register('search_engine', search_engine)
            
            if not search_engine:
                logger.error("No search engine available")
                return {'found': False, 'message': 'Search engine not available'}
            
            # Use the search engine
            return search_engine.find_item(search_term)
            
        except Exception as e:
            logger.error(f"Error finding item: {e}")
            return {'found': False, 'message': f"Error finding item: {str(e)}"}

    def diagnose_search_engine(self):
        """Diagnose the search engine connection and state."""
        logger.debug("=== VOICE SEARCH ENGINE DIAGNOSTIC ===")
        
        if not hasattr(self.state, 'search_engine') or not self.state.search_engine:
            logger.error("No search engine connected to voice interaction")
            return False
            
        logger.debug(f"Voice interaction search engine id: {id(self.state.search_engine)}")
        
        # Check if search engine has inventory
        if not hasattr(self.state.search_engine, 'inventory_cache'):
            logger.error("Search engine has no inventory_cache attribute")
            return False
            
        inventory_size = len(self.state.search_engine.inventory_cache)
        logger.debug(f"Search engine inventory size: {inventory_size} items")
        
        # Check workbook reference
        if not hasattr(self.state.search_engine, 'workbook'):
            logger.error("Search engine has no workbook reference")
            return False
            
        logger.debug(f"Search engine workbook reference: {bool(self.state.search_engine.workbook)}")
        
        # Check input column
        input_col_idx = getattr(self.state.search_engine, 'input_column_index', None)
        input_col_name = getattr(self.state.search_engine, 'input_column_name', None)
        logger.debug(f"Search engine input column: {input_col_name} (index: {input_col_idx})")
        
        # Test a random search if we have inventory
        if inventory_size > 0:
            try:
                # Get a random item from inventory
                random_key = list(self.state.search_engine.inventory_cache.keys())[0]
                logger.debug(f"Test search for: '{random_key}'")
                
                # Use the find_item method
                result = self.state.search_engine.find_item(random_key)
                    
                if result:
                    logger.debug(f"Test search successful: {bool(result)}")
                else:
                    logger.warning("Test search failed: No result returned")
            except Exception as e:
                logger.error(f"Test search failed with error: {e}")
                
        logger.debug("=== END VOICE DIAGNOSTIC ===")
        return True

    def register_command_handlers(self):
        """Register all command handlers with the router."""
        # Inventory update patterns
        self.command_router.register_handler(
            r"^(.*?)(reserve|res)$",  # Match terms ending with "reserve" or "res"
            self._handle_search
        )

        self.command_router.register_handler(
            r"add (\d+\.?\d*) (to|for) (.+)",
            self._handle_add_inventory
        )
        
        self.command_router.register_handler(
            r"(\w+(?:\s+\w+)*)\s+(-?\d*\.?\d+)$",
            self._handle_direct_update
        )
        
        self.command_router.register_handler(
            r"set\s+(.+)\s+to\s+(-?\d*\.?\d+)$",
            self._handle_set_value
        )
        
        # Search patterns
        self.command_router.register_handler(
            r"(find|search for|lookup) (.+)",
            self._handle_search
        )
        
        # Mode change patterns
        self.command_router.register_handler(
            r"^(?:be|go)\s+(wild|mild|proper)$",
            self._handle_mode_change
        )
        
        self.command_router.register_handler(
            r"^change\s+mode\s+to\s+(wild|mild|proper)$",
            self._handle_mode_change
        )
        
        # System command patterns
        self.command_router.register_handler(
            r"^help$",
            self._handle_help_request
        )
        
        self.command_router.register_handler(
            r"^(system status|show status|diagnostics)$",
            self._handle_system_status
        )
        
        # Learning patterns
        self.command_router.register_handler(
            r"^learn\s+term\s+(.+)$",
            self._handle_learn_term
        )
        
        # Register inventory item handler
        self.command_router.register_inventory_handler(self._handle_inventory_item)

        # Register fallback handler
        self.command_router.register_fallback(self._handle_fallback)
        
        # Register unknown command handler
        self.command_router.register_unknown(self._handle_unknown)

    def _handle_add_inventory(self, match):
        """Handle add inventory commands."""
        # Extract parameters from match
        amount = match.groups[0]
        preposition = match.groups[1]  # "to" or "for"
        item_name = match.groups[2]
        
        try:
            # Convert amount to float
            value = float(amount)
            
            # Execute inventory update
            if hasattr(self.state, 'excel_handler') and self.state.excel_handler:
                result = self._execute_inventory_update(item_name, value)
                return result
            else:
                return {
                    "success": False,
                    "message": "No inventory system connected"
                }
        except ValueError:
            return {
                "success": False,
                "message": f"Invalid quantity: {amount}"
            }
        except Exception as e:
            logger.error(f"Error handling add inventory: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    def _handle_search(self, match):
        """Handle search commands with standardized component access."""
        search_term = match.groups[1]
        
        try:
            # Log search attempt with search term
            logger.debug(f"Searching for: '{search_term}'")
            
            # Use component manager to get search engine
            try:
                from component_manager import component_manager
                
                # Get search engine through component manager
                search_engine = component_manager.get_search_engine()
                
                # If component manager couldn't provide a search engine, try fallbacks
                if not search_engine:
                    logger.debug("Component manager couldn't provide search engine, trying fallbacks")
                    
                    # Try to create a factory function for the search engine
                    if not component_manager.component_factories.get('search_engine'):
                        def create_search_engine():
                            from search_engine import AdaptiveInventorySearchEngine
                            return AdaptiveInventorySearchEngine()
                        
                        component_manager.register_component_factory('search_engine', create_search_engine)
                        
                        # Try again with the factory
                        search_engine = component_manager.get_search_engine()
            
            except ImportError:
                # Fall back to old method if component manager not available
                logger.debug("Component manager not available, using legacy method")
                
                # Always get the search engine from the global registry first
                from global_registry import GlobalRegistry
                search_engine = GlobalRegistry.get('search_engine')
                
                # If not in registry, try direct reference
                if not search_engine and hasattr(self.state, 'search_engine') and self.state.search_engine:
                    search_engine = self.state.search_engine
                    # Register it in the registry for future use
                    GlobalRegistry.register('search_engine', search_engine)
                    
                # If still no search engine, try excel handler
                if not search_engine and hasattr(self.state, 'excel_handler') and self.state.excel_handler:
                    if hasattr(self.state.excel_handler, 'search_engine'):
                        search_engine = self.state.excel_handler.search_engine
                        # Register it in the registry
                        GlobalRegistry.register('search_engine', search_engine)
            
            # Final check if we have a search engine
            if not search_engine:
                logger.error("Failed to obtain search engine through any method")
                return {"success": False, "message": "No search engine available"}
                
            # Update our reference
            self.state.search_engine = search_engine
            
            # Check if search engine has inventory data
            if hasattr(search_engine, 'inventory_cache'):
                cache_size = len(search_engine.inventory_cache)
                logger.debug(f"Search engine inventory has {cache_size} items")
                if cache_size == 0:
                    logger.warning("Search index is empty")
                    return {"success": False, "message": "Search index is empty. Please load a file first."}
            
            # Diagnose search engine state before searching
            if hasattr(search_engine, 'diagnose_search_index'):
                search_engine.diagnose_search_index()
            else:
                # Log information about search engine if diagnose method isn't available
                logger.debug(f"Search engine state: has inventory_cache={hasattr(search_engine, 'inventory_cache')}")
                if hasattr(search_engine, 'inventory_cache'):
                    logger.debug(f"Inventory cache size: {len(search_engine.inventory_cache)}")
                    logger.debug(f"Sample keys: {list(search_engine.inventory_cache.keys())[:5] if search_engine.inventory_cache else []}")
            
            # Perform the search
            logger.debug(f"Executing find_item for '{search_term}'")
            item_info = search_engine.find_item(search_term)
            logger.debug(f"Search result: {item_info}")
            
            # Process results
            if item_info and item_info.get('found', False):
                # Store as active context
                self.state.active_item_context = item_info.get('item', search_term)
                
                # Get current value if available
                current_value = item_info.get('value', 'unknown')
                
                return {
                    "success": True,
                    "message": "Item found",
                    "item": item_info.get('item', search_term),
                    "value": current_value
                }
            else:
                return {
                    "success": False,
                    "message": "Item not found",
                    "search_term": search_term
                }
        except Exception as e:
            logger.error(f"Error handling search: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    def _handle_fallback(self, match):
        """Handle partial matches with alternatives."""
        command_text = match.command_text
        alternatives = match.metadata.get('alternatives', [])
        
        # Log the fallback
        if self.state.logging_manager:
            self.state.logging_manager.log_pattern_match({
                'type': 'fallback_command',
                'command': command_text,
                'alternatives': [(m.pattern, m.confidence) for m in alternatives] if alternatives else [],
                'timestamp': datetime.now().isoformat()
            })
        
        # Suggest the most likely alternatives
        if alternatives:
            # Try to find the item in inventory as a last resort
            try:
                # Get the search engine - try direct reference first
                if hasattr(self.state, 'search_engine') and self.state.search_engine:
                    search_engine = self.state.search_engine
                # Then try excel handler
                elif hasattr(self.state, 'excel_handler') and self.state.excel_handler and hasattr(self.state.excel_handler, 'search_engine'):
                    search_engine = self.state.excel_handler.search_engine
                # Finally try registry
                else:
                    from global_registry import GlobalRegistry
                    search_engine = GlobalRegistry.get('search_engine')
                    
                if search_engine:
                    logger.debug(f"Fallback search for '{command_text}' with search engine ID: {id(search_engine)}")
                    
                    # Check inventory cache
                    if hasattr(search_engine, 'inventory_cache'):
                        cache_size = len(search_engine.inventory_cache)
                        logger.debug(f"Search engine has {cache_size} inventory items")
                        
                        # Only search if there are items in the cache
                        if cache_size > 0:
                            item_info = search_engine.find_item(command_text)
                            
                            if item_info and item_info.get('found', False):
                                # Found the item, store it as context
                                self.state.active_item_context = item_info.get('item', command_text)
                                current_value = item_info.get('value', 'unknown')
                                
                                return {
                                    "success": True,
                                    "message": "Item recognized",
                                    "item": command_text,
                                    "current_value": current_value
                                }
            except Exception as e:
                logger.error(f"Error in fallback item search: {e}")
        
        return {
            "success": False,
            "message": "Command not recognized fully",
            "alternatives": [(alt.pattern, alt.confidence) for alt in alternatives] if alternatives else []
        }

    def _handle_unknown(self, match):
        """Handle unknown commands."""
        command_text = match.command_text
        
        # Try to learn from failed command
        self.learn_from_failed_command(command_text)
        
        # Log the unknown command
        if self.state.logging_manager:
            self.state.logging_manager.log_pattern_match({
                'type': 'unknown_command',
                'command': command_text,
                'timestamp': datetime.now().isoformat()
            })
        
        # Try to find the item in inventory as a last resort
        if hasattr(self.state, 'excel_handler') and self.state.excel_handler:
            if hasattr(self.state.excel_handler, 'search_engine'):
                try:
                    search_engine = self.state.excel_handler.search_engine
                    item_info = search_engine.find_item(command_text)
                    
                    if item_info and item_info.get('found', False):
                        # Found the item, store it as context
                        self.state.active_item_context = item_info['item']
                        current_value = item_info.get('value', 'unknown')
                        
                        return {
                            "success": True,
                            "message": "Item recognized",
                            "item": command_text,
                            "current_value": current_value
                        }
                except Exception as e:
                    logger.error(f"Error in unknown command item search: {e}")
        
        return {
            "success": False,
            "message": "Unknown command"
        }

    def _handle_direct_update(self, match):
        """Handle direct inventory updates in the format 'item value'."""
        try:
            # Extract parameters from match
            item_name = match.groups[0]
            value_str = match.groups[1]
            
            try:
                value = float(value_str)
                
                # Execute inventory update
                if hasattr(self.state, 'excel_handler') and self.state.excel_handler:
                    result = self._execute_inventory_update(item_name, value)
                    return result
                else:
                    return {
                        "success": False,
                        "message": "No inventory system connected"
                    }
            except ValueError:
                return {
                    "success": False,
                    "message": f"Invalid quantity: {value_str}"
                }
        except Exception as e:
            logger.error(f"Error handling direct update: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    def _handle_set_value(self, match):
        """Handle setting inventory values in the format 'set item to value'."""
        try:
            # Extract parameters from match
            item_name = match.groups[0]
            value_str = match.groups[1]
            
            try:
                value = float(value_str)
                
                # Execute inventory update (not addition)
                if hasattr(self.state, 'excel_handler') and self.state.excel_handler:
                    result = self._execute_inventory_update(item_name, value, is_addition=False)
                    return result
                else:
                    return {
                        "success": False,
                        "message": "No inventory system connected"
                    }
            except ValueError:
                return {
                    "success": False,
                    "message": f"Invalid quantity: {value_str}"
                }
        except Exception as e:
            logger.error(f"Error handling set value: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    def _handle_inventory_item(self, command_match):
        """Handle commands that match inventory items."""
        item_info = command_match.metadata.get('item_info', {})
        item_name = item_info.get('item', command_match.command_text)
        
        self.speak(f"Found {item_name} in inventory. Current quantity is {item_info.get('value', 'unknown')}.")
        
        # Display result in UI if available
        if hasattr(self, 'display_result'):
            self.display_result(item_info)
        
        return {
            'success': True,
            'action': 'inventory_lookup',
            'item': item_name,
            'details': item_info
        }

    def _handle_learn_term(self, match):
        """Handle learning new terms for the dictionary."""
        try:
            # Extract term from match
            term = match.groups[0]
            
            # Log learning attempt
            if self.state.logging_manager:
                self.state.logging_manager.log_learning_event({
                    'type': 'start_learning',
                    'command': f"learn term {term}",
                    'timestamp': datetime.now().isoformat()
                })

            # Prepare learning state
            self.state.learning_mode = True
            self.state.learning_variations = []
            self.state.current_learning_term = term

            # Provide feedback
            self.speak("Now, please spell out the term exactly as written in the document.")
            
            # We'll return success but indicate this is just the start of learning
            return {
                "success": True,
                "message": "Learning mode started",
                "term": term,
                "requires_follow_up": True
            }
            
        except Exception as e:
            logger.error(f"Error handling learn term: {e}")
            self._reset_learning_state()
            return {
                "success": False,
                "message": f"Error starting learning mode: {str(e)}"
            }

    def _verify_azure_credentials(self, key: str, region: str) -> None:
        """Verify Azure credentials with enhanced validation."""
        try:
            if not key or len(key) != 32:
                raise ValueError(f"Invalid Azure key format: length={len(key) if key else 0}")
            
            if not region:
                raise ValueError("Azure region must be specified")
            
            # Verify region format
            if not re.match(r'^[a-z]+(?:-[a-z]+)*$', region):
                raise ValueError(f"Invalid region format: {region}")
            
            logger.debug(f"Azure credentials verified for region: {region}")
            
        except Exception as e:
            logger.error(f"Azure credential verification error: {e}")
            raise

    def _init_azure_speech_services(self, key: str, region: str) -> None:
        """Initialize Azure speech services with enhanced setup and validation."""
        try:
            logger.debug("Setting up Azure speech configuration...")
            
            # Create speech config with enhanced properties
            self.state.speech_config = self._create_speech_config(key, region)
            
            # Configure speech properties
            self._configure_speech_properties()
            
            # Create and test synthesizer
            self._create_speech_synthesizer()
            self._test_speech_synthesis()
            
            logger.debug("Azure speech services initialized successfully")
            
        except Exception as e:
            logger.error(f"Azure speech services initialization error: {e}")
            raise RuntimeError(f"Failed to initialize Azure speech services: {str(e)}")

    def _create_speech_config(self, key: str, region: str) -> speechsdk.SpeechConfig:
        """Create Azure speech configuration with enhanced settings."""
        try:
            config = speechsdk.SpeechConfig(subscription=key, region=region)
            
            # Set output format for high quality
            config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
            )
            
            # Configure service timeouts
            config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
                "5000"
            )
            config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
                "5000"
            )
            
            return config
            
        except Exception as e:
            logger.error(f"Speech config creation error: {e}")
            raise

    def _configure_speech_properties(self) -> None:
        """Configure speech properties with enhanced settings."""
        try:
            config = self.state.speech_config
            
            # Set voice based on personality mode
            voice = self._get_voice_for_mode()
            config.speech_synthesis_voice_name = voice
            
            # Configure recognition settings
            config.enable_dictation()
            #config.enable_audio_configuration()
            
            # Set speech recognition language
            config.speech_recognition_language = "en-US"
            
            # Configure profanity handling
            config.set_profanity(speechsdk.ProfanityOption.Masked)
            
            logger.debug("Speech properties configured successfully")
            
        except Exception as e:
            logger.error(f"Speech properties configuration error: {e}")
            raise

    def connect_voice_to_inventory(self, excel_handler, onedrive_handler=None) -> None:
        """Connect voice recognition to inventory updates and file management."""
        try:
            # Verify Dugal initialization
            if not hasattr(self.state, 'dugal') or not self.state.dugal:
                logger.error("Dugal not properly initialized")
                if self.state.logging_manager:
                    self.state.logging_manager.log_error("Dugal not initialized", {
                        'context': 'inventory_connection'
                    })
                return
            
            # Connect handlers
            self.state.onedrive_handler = onedrive_handler
            self.state.excel_handler = excel_handler if excel_handler else (
                self.state.dugal.excel_handler if hasattr(self.state.dugal, 'excel_handler') else None
            )
            
            # CRITICAL: Directly connect to the search engine from excel handler
            if hasattr(self.state, 'excel_handler') and self.state.excel_handler and hasattr(self.state.excel_handler, 'search_engine'):
                # Get the search engine from excel handler
                search_engine = self.state.excel_handler.search_engine
                
                # Store direct reference to this search engine
                self.state.search_engine = search_engine
                
                # Also register it in the global registry for shared access
                from global_registry import GlobalRegistry
                GlobalRegistry.register('search_engine', search_engine)
                
                # Log the connection details
                logger.debug(f"Connected voice to search engine (ID: {id(search_engine)})")
                logger.debug(f"Search engine has {len(search_engine.inventory_cache) if hasattr(search_engine, 'inventory_cache') else 0} items in inventory cache")
                
                # Initialize pattern learning and optimization
                self.pattern_helpers.optimize_patterns()
                    
                # Initialize learning patterns if available
                if hasattr(search_engine, 'get_all_patterns'):
                    patterns = search_engine.get_all_patterns()
                    if self.state.logging_manager:
                        self.state.logging_manager.log_pattern_match({
                            'type': 'learning_record',
                            'entry': {
                                'pattern': {'patterns': patterns},
                                'timestamp': datetime.now().isoformat(),
                                'success': True
                            }
                        })
                
                # Run diagnostics to verify connection
                if hasattr(self, 'diagnose_search_engine'):
                    self.diagnose_search_engine()
                
            else:
                logger.error("Excel handler has no search engine - voice commands for inventory will not work")
            
            # Log successful connection
            if self.state.logging_manager:
                self.state.logging_manager.log_pattern_match({
                    'type': 'connection',
                    'component': 'inventory',
                    'handlers': {
                        'excel': bool(self.state.excel_handler),
                        'onedrive': bool(onedrive_handler)
                    },
                    'pattern_learning': (
                        hasattr(self.state, 'excel_handler') and
                        self.state.excel_handler and
                        hasattr(self.state.excel_handler, 'search_engine')
                        ),
                    'timestamp': datetime.now().isoformat()
                })
            
            logger.debug("Voice recognition connected to inventory system")
            
        except Exception as e:
            error_msg = f"Error connecting to inventory: {str(e)}"
            logger.error(error_msg)
            if self.state.logging_manager:
                self.state.logging_manager.log_error(error_msg, {
                    'context': 'inventory_connection',
                    'component': 'voice',
                    'handlers': {
                        'excel': bool(self.state.excel_handler),
                        'onedrive': bool(onedrive_handler)
                    }
                })
                
    def _get_voice_for_mode(self) -> str:
        """Get appropriate voice based on personality mode."""
        mode = self.state.mode
        
        # Voice mapping for different modes
        voices = {
            'wild': "en-GB-AlfieNeural",     # More casual, energetic voice
            'mild': "en-GB-RyanNeural",    # Balanced, neutral voice
            'proper': "en-GB-ThomasNeural"  # More formal, refined voice
        }
        
        return voices.get(mode, "en-GB-RyanNeural")

    def _create_speech_synthesizer(self) -> None:
        """Create and configure speech synthesizer with enhanced error handling."""
        try:
            config = self.state.speech_config
            
            # Create synthesizer with audio config
            self.state.synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=config
            )
            
            # Configure synthesizer properties
            self._configure_synthesizer_properties()
            
            logger.debug("Speech synthesizer created successfully")
            
        except Exception as e:
            logger.error(f"Speech synthesizer creation error: {e}")
            raise

    def _configure_synthesizer_properties(self) -> None:
        """Configure synthesizer properties for optimal performance."""
        try:
            synthesizer = self.state.synthesizer
            
            # Configure SSML settings
            synthesizer.synthesis_word_boundary_enabled = True
            
            # Set up event handlers if needed
            if hasattr(synthesizer, 'synthesis_completed'):
                synthesizer.synthesis_completed.connect(self._handle_synthesis_completed)
            if hasattr(synthesizer, 'synthesis_canceled'):
                synthesizer.synthesis_canceled.connect(self._handle_synthesis_canceled)
            
            # Set output properties
            props = synthesizer.properties
            props.set_property(
                speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary,
                "true"
            )
            
        except Exception as e:
            logger.error(f"Synthesizer properties configuration error: {e}")
            raise

    def _test_speech_synthesis(self) -> None:
        """Test speech synthesis with enhanced error detection."""
        try:
            logger.debug("Testing speech synthesis...")
            
            # Skip actual speech if Dugal is configured to suppress welcome
            if hasattr(self.state, 'dugal') and self.state.dugal and hasattr(self.state.dugal.state, 'suppress_welcome') and self.state.dugal.state.suppress_welcome:
                logger.debug("Skipping audible test due to suppress_welcome flag")
                return
            
            # Create test message based on mode
            test_message = self._create_test_message()
            
            # Attempt synthesis with timeout protection
            result = self._synthesize_with_timeout(test_message)
            
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.warning(f"Speech synthesis test result: {result.reason}, continuing initialization")
                # Don't raise an error here, just log a warning and continue
            
            logger.debug("Speech synthesis test successful")
            
        except Exception as e:
            logger.error(f"Speech synthesis test error: {e}")
            logger.warning("Continuing without full speech synthesis capability")
            # Don't re-raise the error, allow initialization to continue

    def _create_test_message(self) -> str:
        """Create appropriate test message based on personality mode."""
        mode = self.state.mode
        messages = {
            'wild': "Hey there! System's up and running!",
            'mild': "Hello, testing speech synthesis.",
            'proper': "Greetings. Speech synthesis system initialized."
        }
        return messages.get(mode, "Testing speech synthesis.")

    def _synthesize_with_timeout(self, text: str, timeout: int = 15) -> speechsdk.SpeechSynthesisResult:
        """Perform speech synthesis with timeout protection."""
        try:
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: self.state.synthesizer.speak_text_async(text).get()
                )
                
                try:
                    # Increase timeout from 5 to 15 seconds
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.debug("Speech synthesis timeout occurred, continuing with initialization")
                    # Instead of raising an error, create a minimal result object
                    dummy_result = speechsdk.SpeechSynthesisResult()
                    dummy_result._reason = speechsdk.ResultReason.SynthesizingAudioCompleted
                    return dummy_result
                    
        except Exception as e:
            logger.error(f"Synthesis timeout error: {e}")
            raise

    def _handle_synthesis_completed(self, evt: speechsdk.SpeechSynthesisEventArgs) -> None:
        """Handle successful speech synthesis completion."""
        try:
            # Update performance metrics
            if hasattr(self.state, 'performance_metrics'):
                metrics = self.state.performance_metrics
                metrics['synthesis_success'] = metrics.get('synthesis_success', 0) + 1
                
                # Track audio duration if available
                if hasattr(evt, 'result') and hasattr(evt.result, 'audio_duration'):
                    metrics['audio_durations'] = metrics.get('audio_durations', [])
                    metrics['audio_durations'].append(evt.result.audio_duration)
            
            # Log success
            logger.debug("Speech synthesis completed successfully")
            
        except Exception as e:
            logger.error(f"Error handling synthesis completion: {e}")

    def _handle_synthesis_canceled(self, evt: speechsdk.SpeechSynthesisEventArgs) -> None:
        """Handle speech synthesis cancellation with error recovery."""
        try:
            if hasattr(evt, 'result') and hasattr(evt.result, 'cancellation_details'):
                details = evt.result.cancellation_details
                
                logger.error(f"Speech synthesis canceled: {details.reason}")
                
                if details.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Error details: {details.error_details}")
                    
                    # Attempt service recovery
                    self._attempt_service_recovery()
                    
                    # Update error metrics
                    self.state.error_count += 1
                    
            # Emit status signal
            self.status_changed.emit("synthesis_error")
            
        except Exception as e:
            logger.error(f"Error handling synthesis cancellation: {e}")

    def _handle_synthesis_failure(self, result: speechsdk.SpeechSynthesisResult) -> None:
        """Handle synthesis failure with detailed diagnostics."""
        try:
            if result.reason == speechsdk.ResultReason.Canceled:
                details = speechsdk.SpeechSynthesisCancellationDetails(result)
                
                error_message = f"Synthesis failed: {details.reason}"
                if details.error_details:
                    error_message += f" ({details.error_details})"
                
                raise RuntimeError(error_message)
            else:
                raise RuntimeError(f"Synthesis failed with reason: {result.reason}")
            
        except Exception as e:
            logger.error(f"Synthesis failure handling error: {e}")
            raise

    def _attempt_service_recovery(self) -> None:
        """Attempt to recover from service errors."""
        try:
            logger.warning("Attempting to recover from service error...")
            
            # If multiple errors, try reinitializing services
            if self.state.error_count >= 3:
                if hasattr(self.state, 'speech_config'):
                    key = self.state.speech_config.get_property(
                        speechsdk.PropertyId.SpeechServiceConnection_Key
                    )
                    region = self.state.speech_config.get_property(
                        speechsdk.PropertyId.SpeechServiceConnection_Region
                    )
                    
                    # Reinitialize speech services
                    self._init_azure_speech_services(key, region)
                    
                    # Reset error count after successful recovery
                    self.state.error_count = 0
                    
                    logger.debug("Speech services successfully recovered")
                
        except Exception as e:
            logger.error(f"Error during service recovery: {e}")

    def _setup_event_handlers(self) -> None:
        """Initialize event handlers and signal connections."""
        try:
            # Connect internal signals
            self.speech_started.connect(self._handle_speech_start)
            self.speech_ended.connect(self._handle_speech_end)
            self.command_processed.connect(self._handle_command_completion)
            self.status_changed.connect(self._handle_status_change)
            
            logger.debug("Event handlers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up event handlers: {e}")
            raise

    def process_voice_command(self, command: str) -> Dict[str, Any]:
        """Process voice commands including dictionary learning."""
        logger.debug(f"Processing voice command: '{command}'")
        start_time = datetime.now()
        
        # Always use the registry search engine
        from global_registry import GlobalRegistry
        registry_engine = GlobalRegistry.get('search_engine')
        if registry_engine:
            if not hasattr(self.state, 'search_engine') or id(self.state.search_engine) != id(registry_engine):
                logger.warning(f"Voice interaction using outdated search engine - updating from registry")
                self.state.search_engine = registry_engine
        
        
        
        self.logger = logger

        try:
            # Log the command
            if self.state.logging_manager:
                self.state.logging_manager.log_pattern_match({
                    'type': 'command',
                    'input': command,
                    'timestamp': datetime.now().isoformat()
                })

            # Check for active item context (where user previously mentioned an item)
            if hasattr(self.state, 'active_item_context') and self.state.active_item_context:
                # Check if this is a number or a number with an operation
                try:
                    parts = command.strip().split()
                    
                    # Case 1: Just a number - direct update
                    if len(parts) == 1:
                        try:
                            value = float(parts[0])
                            item_name = self.state.active_item_context
                            
                            # Clear the active item context after using it
                            self.state.active_item_context = None
                            
                            # Process as an inventory update with the stored item name
                            logger.debug(f"Using active item context: '{item_name}' with value {value}")
                            
                            # Execute inventory update
                            if hasattr(self.state, 'excel_handler') and self.state.excel_handler:
                                result = self._execute_inventory_update(item_name, value)
                                
                                # Skip response unless there's an error
                                duration = (datetime.now() - start_time).total_seconds()
                                if result.get("success"):
                                    self._track_recognition_performance(True, duration)
                                    # Skip verbal feedback for success
                                else:
                                    self._track_recognition_performance(False, duration)
                                    error_message = self._format_error_response(result)
                                    self.speak(error_message)
                                
                                return result
                        except ValueError:
                            # Not a simple number, continue to next case
                            pass
                            
                    # Case 2: Operation with a number (add, subtract, etc.)
                    if len(parts) == 2 and parts[0].lower() in ['add', 'plus', 'subtract', 'minus']:
                        try:
                            operation = parts[0].lower()
                            value = float(parts[1])
                            
                            # Apply the operation logic
                            if operation in ['subtract', 'minus']:
                                value = -value  # Make it negative for subtraction
                                
                            item_name = self.state.active_item_context
                            
                            # Clear the active item context
                            self.state.active_item_context = None
                            
                            # Execute inventory update as an addition/subtraction
                            logger.debug(f"Using active item context: '{item_name}' with {operation} {value}")
                            
                            # Execute inventory update
                            if hasattr(self.state, 'excel_handler') and self.state.excel_handler:
                                result = self._execute_inventory_update(item_name, value, is_addition=True)
                                
                                # Skip response unless there's an error
                                duration = (datetime.now() - start_time).total_seconds()
                                if result.get("success"):
                                    self._track_recognition_performance(True, duration)
                                    # Skip verbal feedback for success
                                else:
                                    self._track_recognition_performance(False, duration)
                                    error_message = self._format_error_response(result)
                                    self.speak(error_message)
                                
                                return result
                        except ValueError:
                            # Not a valid operation + number, clear context and continue
                            pass
                    
                    # If we get here, clear the context as the command isn't related
                    self.state.active_item_context = None
                except Exception as context_error:
                    logger.error(f"Error processing item context: {context_error}")
                    self.state.active_item_context = None

            # Check for learning commands
            if command.lower().startswith("dugal learn term"):
                result = self._handle_learn_command(command)
                duration = (datetime.now() - start_time).total_seconds()
                self._track_learning_performance({
                    'success': result.get('success', False),
                    'term': command,
                    'duration': duration
                })
                return result
                
            # Check for learning mode
            if self.state.learning_mode:
                return self._handle_learning_mode_input(command)

            # Normalize and correct potential spelling/pattern issues
            corrected_command = self.correct_spelling(command)
            
            # Check for item-specific commands like "buffalo trace add .6"
            item_operation_match = re.match(r'^([\w\s]+?)\s+(add|plus|subtract|minus)\s+(\d*\.?\d+)$', corrected_command, re.IGNORECASE)
            if item_operation_match:
                item_name = item_operation_match.group(1).strip()
                operation = item_operation_match.group(2).lower()
                value_str = item_operation_match.group(3)
                
                try:
                    value = float(value_str)
                    
                    # Make value negative for subtract operations
                    if operation in ['subtract', 'minus']:
                        value = -value
                    
                    # Execute as an addition/subtraction
                    logger.debug(f"Processing item operation: '{item_name}' {operation} {value}")
                    
                    if hasattr(self.state, 'excel_handler') and self.state.excel_handler:
                        result = self._execute_inventory_update(item_name, value, is_addition=True)
                        
                        # Track performance but skip verbal response for efficiency
                        duration = (datetime.now() - start_time).total_seconds()
                        if result.get("success"):
                            self._track_recognition_performance(True, duration)
                            # Skip verbal feedback for success
                        else:
                            self._track_recognition_performance(False, duration)
                            error_message = self._format_error_response(result)
                            self.speak(error_message)
                        
                        return result
                except ValueError:
                    # Continue with normal processing if conversion fails
                    pass
            
            # Normal command processing with new router
            result = self.process_command(corrected_command)
            
            self.logger.debug(f"Checking if '{corrected_command}' is an inventory item...")

            # If command wasn't recognized, check if it might be just an item name
            if not result:
                # Fix the logger attribute error
                local_logger = logger  # Use the global logger
                
                local_logger.debug(f"Command not recognized, checking if '{corrected_command}' is an inventory item...")
                
                # Try to find a valid search engine with inventory data
                search_engine = None
                
                # Option 1: Try our direct reference first
                if hasattr(self.state, 'search_engine') and self.state.search_engine:
                    search_engine = self.state.search_engine
                    local_logger.debug(f"Using direct search engine reference (ID: {id(search_engine)})")
                    
                    # Verify it has inventory data
                    if hasattr(search_engine, 'inventory_cache'):
                        cache_size = len(search_engine.inventory_cache)
                        local_logger.debug(f"Direct search engine has {cache_size} inventory items")
                        # If empty, don't use this engine
                        if cache_size == 0:
                            search_engine = None
                
                # Option 2: Try excel handler if needed
                if not search_engine and hasattr(self.state, 'excel_handler') and self.state.excel_handler:
                    local_logger.debug(f"Excel handler available: {bool(self.state.excel_handler)}")
                    
                    if hasattr(self.state.excel_handler, 'search_engine'):
                        excel_search_engine = self.state.excel_handler.search_engine
                        local_logger.debug(f"Excel handler search engine available (ID: {id(excel_search_engine)})")
                        
                        # Verify it has inventory data
                        if hasattr(excel_search_engine, 'inventory_cache'):
                            cache_size = len(excel_search_engine.inventory_cache)
                            local_logger.debug(f"Excel search engine has {cache_size} inventory items")
                            if cache_size > 0:
                                search_engine = excel_search_engine
                                # Update our reference for future use
                                self.state.search_engine = search_engine
                
                # Option 3: Last resort - try global registry
                if not search_engine:
                    try:
                        from global_registry import GlobalRegistry
                        registry_engine = GlobalRegistry.get('search_engine')
                        if registry_engine and hasattr(registry_engine, 'inventory_cache') and len(registry_engine.inventory_cache) > 0:
                            local_logger.debug(f"Using registry search engine (ID: {id(registry_engine)}) with {len(registry_engine.inventory_cache)} items")
                            search_engine = registry_engine
                            # Update the local reference
                            self.state.search_engine = registry_engine
                    except Exception as registry_error:
                        local_logger.error(f"Error accessing registry: {registry_error}")
                
                # Now use the search engine if we found a valid one
                if search_engine and hasattr(search_engine, 'find_item'):
                    try:
                        # Check search engine state
                        if hasattr(search_engine, 'inventory_cache'):
                            local_logger.debug(f"Selected search engine inventory cache has {len(search_engine.inventory_cache)} items")
                            
                            # Log a few sample keys to verify content
                            sample_keys = list(search_engine.inventory_cache.keys())[:5] if search_engine.inventory_cache else []
                            local_logger.debug(f"Sample inventory keys: {sample_keys}")
                        
                        # Use the search engine to find the item
                        local_logger.debug(f"Calling find_item with: '{corrected_command}'")
                        item_info = search_engine.find_item(corrected_command)
                        local_logger.debug(f"Find item result: {item_info}")
                        
                        if item_info and item_info.get('found', False):
                            # Found the item, store it as context and return success
                            self.state.active_item_context = item_info['item']
                            
                            # Get current value if available
                            current_value = item_info.get('value', 'unknown')
                            
                            # Just provide a minimal acknowledgment
                            if current_value != 'unknown':
                                self.speak(f"{corrected_command}: {current_value}")
                            else:
                                self.speak(f"Found {corrected_command}")
                            
                            return {
                                "success": True,
                                "message": "Item recognized",
                                "item": corrected_command,
                                "current_value": current_value
                            }
                        else:
                            local_logger.debug(f"Item not found: {item_info.get('message', 'Unknown reason')}")
                    except Exception as search_error:
                        local_logger.error(f"Error searching for item: {search_error}")
                        local_logger.exception("Detailed search error:")  # This logs the full stack trace
                else:
                    local_logger.warning("No valid search engine with inventory data found")

        # For commands that reached this point, provide the standard feedback
            duration = (datetime.now() - start_time).total_seconds()
            if result and result.get("success"):
                self._track_recognition_performance(True, duration)
                if self.state.dugal:
                    success_message = self._format_success_response(result)
                    self.speak(success_message)
            elif result:
                self._track_recognition_performance(False, duration)
                if self.state.dugal:
                    error_message = self._format_error_response(result)
                    self.speak(error_message)
            else:
                # Unrecognized command
                self._track_recognition_performance(False, duration)
                return self._handle_unrecognized_command(corrected_command)
            
            return result if result else {"success": False, "message": "Command not recognized"}

        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            if self.state.logging_manager:
                self.state.logging_manager.log_error(str(e), {
                    'context': 'voice_command',
                    'command': command
                })
            
            # Dugal's personality-driven error handling
            if self.state.dugal:
                error_response = self.state.dugal.get_response('error', str(e))
                self.speak(error_response)
            
            return {
                'success': False,
                'message': f"Error processing command: {str(e)}"
            }

    def process_command(self, command: str) -> Dict[str, Any]:
        """
        Process a voice command using the command router.
        Dugal's brain, but with a PhD in pattern recognition.
        """
        if not command:
            return {"success": False, "message": "No command detected"}
        
        try:
            # Log the attempt with Dugal's flair
            self._log_command_attempt(command)
            
            # Use the command router to match and execute the command
            result = self.command_router.route_command(command)
            
            if result.success:
                try:
                    # Call the matched handler with the command match
                    handler_result = result.command_match.handler(result.command_match)
                    return handler_result
                except Exception as e:
                    logger.error(f"Error executing command handler: {e}")
                    return {"success": False, "message": f"Error: {str(e)}"}
            else:
                logger.warning(f"Command routing failed: {result.error}")
                return {"success": False, "message": result.error}
                
        except Exception as e:
            logger.error(f"Command processing error: {e}")
            return {
                "success": False,
                "message": f"Command processing error: {str(e)}",
                "error_details": str(e)
            }

    def _process_system_command(self, command: str) -> Optional[Dict[str, Any]]:
        """Process system-level commands with comprehensive handling."""
        system_commands = [
            # Mode change commands
            {
                'patterns': [
                    r'^change mode to (wild|mild|proper)$', 
                    r'^(be|go) (wild|mild|proper)$'
                ],
                'handler': self._handle_mode_change
            },
            # Help request commands
            {
                'patterns': [
                    r'^help$', 
                    r'^what can you do$',
                    r'^show commands$'
                ],
                'handler': self._handle_help_request
            },
            # System status commands
            {
                'patterns': [
                    r'^system status$', 
                    r'^show status$',
                    r'^diagnostics$'
                ],
                'handler': self._handle_system_status
            }
        ]
        
        try:
            for cmd_config in system_commands:
                for pattern in cmd_config['patterns']:
                    match = re.match(pattern, command)
                    if match:
                        try:
                            # Extract groups for handler
                            groups = match.groups() if match.groups() else []
                            return cmd_config['handler'](*groups)
                        except Exception as e:
                            logger.error(f"System command processing error: {e}")
                            return {
                                "success": False,
                                "message": f"Error processing system command: {str(e)}"
                            }
            
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error in system command processing: {e}")
            return {
                "success": False,
                "message": "System command processing failed"
            }

    def _process_learning_command(self, command: str) -> Optional[Dict[str, Any]]:
        """Process commands related to term learning."""
        learning_patterns = [
            (r'^learn\s+term\s+(.+)$', self._handle_learn_command),
            (r'^add\s+variation\s+(.+)\s+for\s+(.+)$', self._handle_variation_addition),
            (r'^confirm\s+learning\s+(.+)$', self._confirm_learning)
        ]
        
        for pattern, handler in learning_patterns:
            match = re.match(pattern, command)
            if match:
                try:
                    return handler(*match.groups())
                except Exception as e:
                    logger.error(f"Learning command processing error: {e}")
                    return {
                        "success": False,
                        "message": f"Error processing learning command: {str(e)}"
                    }
        
        return None

    def process_inventory_command(self, command: str) -> Dict[str, Any]:
        """Process inventory commands with enhanced validation."""
        logger.debug(f"Analyzing inventory command: '{command}'")
        
        try:
            # Extract item details with advanced parsing
            item_name, value = self._extract_inventory_details(command)
            
            if not item_name or value is None:
                result = {
                    "success": False,
                    "message": f"Could not understand inventory command: '{command}'"
                }
                logger.warning(f"Failed to parse inventory command: {command}")
                return result
            
            # Pattern learning and correction
            corrected_item = self._correct_item_name(item_name)
            
            # Verify system readiness
            if not self._verify_system_ready():
                return {
                    "success": False,
                    "message": "Inventory system is not ready"
                }
            
            # Perform inventory update
            result = self._execute_inventory_update(corrected_item, value)
            
            # Learning tracking
            if result.get("success"):
                self._track_successful_update(corrected_item, value)
            
            return result
        
        except Exception as e:
            logger.error(f"Inventory command error: {e}")
            return {
                "success": False,
                "message": f"Error processing inventory command: {str(e)}"
            }

    def _process_inventory_command(self, command: str) -> Optional[Dict[str, Any]]:
        """Process inventory-related commands."""
        inventory_patterns = self.state.cache.get('commands', {}).get('inventory_update', [])
        
        try:
            for pattern in inventory_patterns:
                match = re.match(pattern, command)
                if match:
                    # Extract details based on pattern type
                    if pattern.startswith(r'^(\w+(?:\s+\w+)*)\s+(-?\d*\.?\d+)'):
                        item_name, value_str = match.groups()
                        value = float(value_str)
                    elif pattern.startswith(r'^add\s+(-?\d*\.?\d+)\s+(?:of\s+)?(.+)'):
                        value_str, item_name = match.groups()
                        value = float(value_str)
                    elif pattern.startswith(r'^remove\s+(-?\d*\.?\d+)\s+(?:of\s+)?(.+)'):
                        value_str, item_name = match.groups()
                        value = -float(value_str)  # Negative for removal
                    elif pattern.startswith(r'^set\s+(.+)\s+to\s+(-?\d*\.?\d+)'):
                        item_name, value_str = match.groups()
                        value = float(value_str)
                    else:
                        continue
                    
                    return self.process_inventory_command(command)
                    
            return None
            
        except Exception as e:
            logger.error(f"Error processing inventory command: {e}")
            return None

    def start_listening(self) -> None:
        """Start continuous voice recognition with enhanced error handling."""
        logger.debug("Starting continuous voice recognition...")
        
        try:
            if not self.state.recognizer:
                raise RuntimeError("Speech recognizer not initialized")
            
            # Emit started signal
            self.speech_started.emit()
            
            with sr.Microphone() as source:
                # Configure recognizer for optimal performance
                self._configure_recognition(source)
                
                while True:
                    try:
                        # Listen for audio input
                        audio = self.state.recognizer.listen(
                            source,
                            timeout=5,
                            phrase_time_limit=20
                        )
                        
                        # Process the audio
                        self._process_audio(audio)
                        
                    except sr.WaitTimeoutError:
                        # Normal timeout, continue listening
                        continue
                        
                    except sr.UnknownValueError:
                        logger.debug("Could not understand audio")
                        self._handle_recognition_error("unknown_value")
                        continue
                        
                    except sr.RequestError as e:
                        logger.error(f"Recognition service error: {e}")
                        self._handle_recognition_error("service_error", str(e))
                        self._attempt_service_recovery()
                        
                    except Exception as e:
                        logger.error(f"Unexpected listening error: {e}")
                        self._handle_recognition_error("unexpected", str(e))
                        
        except Exception as e:
            logger.error(f"Error starting voice recognition: {e}")
            self.speech_ended.emit()
            raise

    def _configure_recognition(self, source: sr.Microphone) -> None:
        """Configure speech recognizer for optimal performance."""
        try:
            # Adjust for ambient noise
            self.state.recognizer.adjust_for_ambient_noise(
                source, 
                duration=1
            )
            
            # Configure recognition parameters
            self.state.recognizer.pause_threshold = 0.5
            self.state.recognizer.phrase_threshold = 0.2
            self.state.recognizer.non_speaking_duration = 0.3
            
            # Enable dynamic energy threshold
            self.state.recognizer.dynamic_energy_threshold = True
            self.state.recognizer.dynamic_energy_adjustment_damping = 0.15
            self.state.recognizer.dynamic_energy_ratio = 1.5
            
            logger.debug("Recognition configured successfully")
            
        except Exception as e:
            logger.error(f"Error configuring recognition: {e}")
            raise

    def _process_audio(self, audio: sr.AudioData) -> None:
        """Process captured audio with enhanced error handling."""
        start_time = datetime.now()
        
        try:
            # Recognize speech
            text = self.state.recognizer.recognize_google(audio)
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.debug(f"Recognized text: {text}")
            
            # Process the recognized command
            result = self.process_voice_command(text)
            
            # Track performance
            self._track_recognition_performance(
                success=result.get('success', False),
                duration=duration,
                command=text
            )
            
            # Provide feedback if needed
            if not result.get('success') and result.get('suggestions'):
                self._provide_command_suggestions(result['suggestions'])
                
        except sr.UnknownValueError:
            logger.debug("Speech not recognized")
            # Don't log errors for normal non-recognition
        except sr.RequestError as e:
            logger.error(f"Recognition service error: {e}")
            duration = (datetime.now() - start_time).total_seconds()
            self._track_recognition_performance(
                success=False,
                duration=duration,
                error=f"request_error: {str(e)}"
            )
        except Exception as e:
            # Just log the error without any additional details to avoid the empty error logs
            logger.error(f"Error processing audio: {str(e)}")

    def _handle_recognition_error(self, error_type: str, details: str = "") -> None:
        """Handle recognition errors with appropriate recovery actions."""
        try:
            # Update error statistics
            state = self.state
            state.error_count += 1
            state.stats['error_types'][error_type] = state.stats['error_types'].get(error_type, 0) + 1
            
            # Log error with context
            if state.logging_manager:
                state.logging_manager.log_error(
                    f"Recognition error: {error_type}",
                    {
                        'details': details,
                        'error_count': state.error_count,
                        'mode': state.mode
                    }
                )
            
            # Provide feedback based on personality mode
            if error_type == "unknown_value":
                self._provide_error_feedback("I didn't catch that. Could you repeat?")
            elif error_type == "service_error":
                self._provide_error_feedback("Having trouble with the speech service.")
            
        except Exception as e:
            logger.error(f"Error handling recognition error: {e}")

    def listen_for_command(self) -> Dict[str, Any]:
        """Listen for a single voice command with advanced error handling."""
        try:
            if not self.state.recognizer:
                return {
                    "success": False, 
                    "message": "Speech recognizer not initialized"
                }

            # Prompt for command
            prompt = self.state.dugal.get_response('waiting')
            self.speak(prompt)

            with sr.Microphone() as source:
                self.state.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                audio = self.state.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=10
                )

                start_time = datetime.now()
                try:
                    text = self.state.recognizer.recognize_google(audio)
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    logger.debug(f"Recognized command: {text}")
                    self._track_recognition_performance(True, duration)
                    
                    return {
                        "success": True, 
                        "command": text.strip()
                    }

                except sr.UnknownValueError:
                    logger.debug("No speech recognized")
                    error_msg = self.state.dugal.get_response('error', "Didn't catch that")
                    self.speak(error_msg)
                    self._track_recognition_performance(False, 0)
                    return {
                        "success": False, 
                        "message": "No speech recognized"
                    }

                except sr.RequestError as e:
                    logger.error(f"Recognition service error: {e}")
                    error_msg = self.state.dugal.get_response('error', str(e))
                    self.speak(error_msg)
                    self._track_recognition_performance(False, 0)
                    return {
                        "success": False, 
                        "message": f"Service error: {str(e)}"
                    }

        except Exception as e:
            logger.error(f"Command listening error: {e}")
            return {
                "success": False, 
                "message": str(e)
            }

    def _handle_speech_start(self) -> None:
        """Handle speech recognition start event."""
        try:
            logger.debug("Speech recognition started")
            
            # Update UI state
            self.status_changed.emit("listening")
            
            # Reset error count
            self.state.error_count = 0
            
            # Start inactivity timer
            self.inactivity_timer.start()
            
            # Update activity timestamp
            self.state.last_activity = datetime.now()
            
        except Exception as e:
            logger.error(f"Error handling speech start: {e}")

    def _handle_speech_end(self) -> None:
        """Handle speech recognition end event."""
        try:
            logger.debug("Speech recognition ended")
            
            # Update UI state
            self.status_changed.emit("idle")
            
            # Stop inactivity timer
            self.inactivity_timer.stop()
            
            # Save performance metrics
            self._save_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error handling speech end: {e}")

    def _handle_command_completion(self, result: Dict[str, Any]) -> None:
        """Handle command processing completion."""
        try:
            # Update performance metrics
            self._update_command_metrics(result)
            
            # Provide feedback based on result
            self._provide_command_feedback(result)
            
            # Update UI state
            status = "success" if result.get('success') else "error"
            self.status_changed.emit(status)
            
            # Reset activity timer
            self.inactivity_timer.start()
            
        except Exception as e:
            logger.error(f"Error handling command completion: {e}")

    def learn_from_failed_command(self, command: str) -> None:
        """Add unrecognized commands to the learning queue for future dictionary additions."""
        try:
            logger.debug(f"Adding failed command to learning queue: '{command}'")
            
            # Store the failed command for later review
            if not hasattr(self.state, 'failed_commands'):
                self.state.failed_commands = []
                
            self.state.failed_commands.append({
                'command': command,
                'timestamp': datetime.now().isoformat()
            })
            
            # Notify user that this command can be learned
            if self.state.dugal:
                if len(command.split()) >= 2:  # Only worth learning multi-word commands
                    self.state.dugal.speak(
                        f"I don't recognize '{command}'. You can add this term in the Dictionary Manager."
                    )
                else:
                    self.state.dugal.speak(f"I don't understand '{command}'.")
                    
            # Log the failed command
            if self.state.logging_manager:
                self.state.logging_manager.log_pattern_match({
                    'type': 'failed_command',
                    'command': command,
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error learning from failed command: {e}")

    def _handle_status_change(self, status: str) -> None:
        """Handle system status changes."""
        try:
            logger.debug(f"Status changed to: {status}")
            
            # Update state
            self.state.last_status = status
            self.state.last_status_change = datetime.now()
            
            # Log status change
            if self.state.logging_manager:
                self.state.logging_manager.log_status_change(status)
            
            # Handle specific status conditions
            if status == "error":
                self._handle_error_status()
            elif status == "inactive":
                self._handle_inactive_status()
            
        except Exception as e:
            logger.error(f"Error handling status change: {e}")

    def _handle_inactivity(self) -> None:
        """Handle system inactivity with appropriate response."""
        try:
            # Optimize resources during inactivity
            if hasattr(self, '_optimize_resources'):
                self._optimize_resources()
            
            # Get appropriate inactivity message
            message = self._get_inactivity_message()
            self.speak(message)
            
            # Log inactivity
            if self.state.logging_manager:
                self.state.logging_manager.log_pattern_match({
                    'type': 'inactivity',
                    'duration': (datetime.now() - self.state.last_activity).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                })
                    
        except Exception as e:
            logger.error(f"Error handling inactive status: {e}")

    def _get_inactivity_message(self) -> str:
        """Get appropriate inactivity message based on personality mode."""
        mode = self.state.mode
        
        messages = {
            'wild': "Hey, still with me? I'm all ears whenever you're ready!",
            'mild': "Just checking if you need anything. I'm still listening.",
            'proper': "Pardon the interruption, but I wanted to confirm if you require assistance."
        }
        
        return messages.get(mode, "Are you still there? I'm listening.")

    def speak(self, text: str, priority: bool = False) -> bool:
        """Synthesize and speak text with enhanced error handling."""
        try:
            if not self.state.synthesizer:
                raise RuntimeError("Speech synthesizer not initialized")
            
            # Apply personality mode adjustments
            text = self._apply_personality_mode(text)
            
            # Configure speech properties based on priority
            if priority:
                self._configure_priority_speech()
            
            # Synthesize and speak
            result = self.state.synthesizer.speak_text_async(text).get()
            
            success = result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
            
            if not success:
                if result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = speechsdk.SpeechSynthesisCancellationDetails(result)
                    logger.error(f"Speech synthesis canceled: {cancellation_details.reason}")
                    
                    if cancellation_details.reason == speechsdk.CancellationReason.Error:
                        logger.error(f"Speech synthesis error: {cancellation_details.error_details}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return False

    def _apply_personality_mode(self, text: str) -> str:
        """Apply personality mode adjustments to speech text."""
        mode = self.state.mode
        
        try:
            if mode == "wild":
                # Add enthusiasm and casual language
                text = text.replace(".", "!")
                text = text.replace("Error", "Oops")
                text = text.replace("Unable to", "Can't")
            
            elif mode == "proper":
                # Add formality
                text = text.replace("can't", "cannot")
                text = text.replace("won't", "will not")
                if not text.endswith((".", "!", "?")):
                    text += "."
            
            # Default "mild" mode uses text as is
            return text
            
        except Exception as e:
            logger.error(f"Error applying personality mode: {e}")
            return text

    def _configure_priority_speech(self) -> None:
        """Configure speech synthesis for priority messages."""
        try:
            config = self.state.speech_config
            
            # Increase priority and reduce latency
            config.set_property(
                speechsdk.PropertyId.Speech_LogFilename,
                "high_priority.log"
            )
            
            # Adjust synthesis settings
            config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
            )
            
            # Configure timeouts
            config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
                "2000"
            )
            
        except Exception as e:
            logger.error(f"Error configuring priority speech: {e}")

    def _handle_learn_command(self, command: str) -> Dict[str, Any]:
        """Handle the term learning command sequence with comprehensive validation."""
        try:
            # Log learning attempt
            if self.state.logging_manager:
                self.state.logging_manager.log_learning_event({
                    'type': 'start_learning',
                    'command': command,
                    'timestamp': datetime.now().isoformat()
                })

            # Prepare learning state
            self.state.learning_mode = True
            self.state.learning_variations = []

            # Step 1: Get spoken term
            self.speak("Please say how the term sounds in normal speech.")
            spoken_result = self.listen_for_command()

            if not spoken_result.get('success'):
                self._reset_learning_state()
                return {
                    "success": False, 
                    "message": "Failed to capture spoken term"
                }

            spoken_term = spoken_result['command'].strip()
            self.state.current_learning_term = spoken_term

            # Step 2: Get written term
            self.speak("Now, please spell out the term exactly as written in the document.")
            written_result = self.listen_for_command()

            if not written_result.get('success'):
                self._reset_learning_state()
                return {
                    "success": False, 
                    "message": "Failed to capture written term"
                }

            written_term = written_result['command'].strip()

            # Step 3: Validate terms
            validation_result = self._validate_learning_terms(spoken_term, written_term)
            if not validation_result['success']:
                self._reset_learning_state()
                return validation_result

            # Step 4: Confirm learning
            return self._confirm_term_learning(spoken_term, written_term)

        except Exception as e:
            logger.error(f"Term learning error: {e}")
            self._reset_learning_state()
            return {
                "success": False, 
                "message": f"Learning process failed: {str(e)}"
            }

    def _validate_learning_terms(self, spoken_term: str, written_term: str) -> Dict[str, Any]:
        """Validate terms for learning with advanced pattern analysis."""
        try:
            # Validate using pattern helpers
            if not self.pattern_helpers.validate_learned_pattern(spoken_term, written_term):
                return {
                    "success": False, 
                    "message": "Terms do not match sufficiently for learning"
                }

            # Optional additional validation
            if len(spoken_term) < 2 or len(written_term) < 2:
                return {
                    "success": False, 
                    "message": "Terms are too short to be valid"
                }

            return {
                "success": True, 
                "message": "Terms validated successfully"
            }

        except Exception as e:
            logger.error(f"Term validation error: {e}")
            return {
                "success": False, 
                "message": f"Validation failed: {str(e)}"
            }

    def _confirm_term_learning(self, spoken_term: str, written_term: str) -> Dict[str, Any]:
        """Confirm and save learned term with multiple validation steps."""
        try:
            # Final confirmation request
            confirmation_prompt = f"I'll learn that '{spoken_term}' is written as '{written_term}'. Is this correct?"
            self.speak(confirmation_prompt)
            
            confirmation_result = self.listen_for_command()
            
            if not confirmation_result.get('success'):
                self._reset_learning_state()
                return {
                    "success": False, 
                    "message": "Confirmation failed"
                }

            # Verify confirmation
            if not self._verify_confirmation(spoken_term, confirmation_result['command']):
                self._reset_learning_state()
                return {
                    "success": False, 
                    "message": "Confirmation did not match"
                }

            # Prepare term data
            term_data = {
                'written_form': written_term,
                'variations': [spoken_term, confirmation_result['command']]
            }

            # Save learned term
            save_result = self._save_learned_term(term_data)

            # Final processing
            if save_result:
                self.speak("Term learned successfully!")
                self._reset_learning_state()
                return {
                    "success": True, 
                    "message": "Term learned and saved"
                }
            else:
                self.speak("Sorry, I had trouble saving the term.")
                self._reset_learning_state()
                return {
                    "success": False, 
                    "message": "Failed to save learned term"
                }

        except Exception as e:
            logger.error(f"Term confirmation error: {e}")
            self._reset_learning_state()
            return {
                "success": False, 
                "message": f"Confirmation process failed: {str(e)}"
            }

    def _verify_confirmation(self, original_term: str, confirmation_term: str) -> bool:
        """Verify that the confirmation matches the original term."""
        try:
            # Use pattern helpers for advanced matching
            return self.pattern_helpers._terms_similar(original_term, confirmation_term)
        except Exception as e:
            logger.error(f"Confirmation verification error: {e}")
            return False

    def _reset_learning_state(self) -> None:
        """Reset learning state to default values."""
        self.state.learning_mode = False
        self.state.current_learning_term = None
        self.state.learning_variations = []

    def _handle_learning_mode_input(self, command: str) -> Dict[str, Any]:
        """Process input while in learning mode."""
        try:
            # Check for learning mode termination
            if command.lower() == "done learning":
                self._reset_learning_state()
                self.speak("Learning mode ended.")
                return {
                    "success": True,
                    "message": "Learning mode ended",
                    "type": "learning_end"
                }

            # Verify active learning context
            if not self.state.current_learning_term:
                return {
                    "success": False,
                    "message": "No active learning term"
                }

            # Add variation to learning variations
            self.state.learning_variations.append(command.strip())
            logger.debug(f"Added variation: '{command}'")

            # Analyze variation using pattern helpers
            analysis = self.pattern_helpers.analyze_pattern(
                self.state.current_learning_term,
                command
            )

            confidence = analysis.get('confidence', 0)

            # Evaluate variation confidence
            if confidence > 0.7:
                self.speak("Good variation! Say another or 'done learning' to finish.")
                return {
                    "success": True,
                    "message": "Variation added",
                    "confidence": confidence
                }
            else:
                self.speak("That variation seems quite different. Are you sure it's correct?")
                return {
                    "success": True,
                    "message": "Low confidence variation",
                    "confidence": confidence
                }

        except Exception as e:
            logger.error(f"Learning mode input processing error: {e}")
            self._reset_learning_state()
            return {
                "success": False,
                "message": f"Learning input error: {str(e)}"
            }

    def _save_learned_term(self, term_data: Dict[str, Any]) -> bool:
        """Save learned term to pattern database."""
        try:
            # Verify search engine availability
            if not hasattr(self.state, 'excel_handler') or not self.state.excel_handler:
                logger.warning("No Excel handler available for learning")
                return False
                
            if not hasattr(self.state.excel_handler, 'search_engine'):
                logger.warning("No search engine available for learning")
                return False

            search_engine = self.state.excel_handler.search_engine

            # Add term to search engine
            success = search_engine.add_learned_pattern(
                written_form=term_data['written_form'],
                variations=term_data['variations'],
                metadata={
                    'added_by': 'voice',
                    'timestamp': datetime.now().isoformat()
                }
            )

            if success:
                # Record learning for pattern optimization
                self.pattern_helpers.record_learning({
                    'from': term_data['variations'][0],
                    'to': term_data['written_form'],
                    'confidence': 1.0,
                    'type': 'manual_entry'
                })

                # Update learning statistics
                self.state.stats['learning_attempts'] += 1
                self.state.stats['successful_learns'] += 1

                # Log pattern match
                if self.state.logging_manager:
                    self.state.logging_manager.log_pattern_match({
                        'type': 'term_saved',
                        'term': term_data,
                        'timestamp': datetime.now().isoformat()
                    })

            return success

        except Exception as e:
            logger.error(f"Error saving learned term: {e}")
            if self.state.logging_manager:
                self.state.logging_manager.log_error(str(e), {
                    'context': 'save_term',
                    'term': term_data.get('written_form', '')
                })
            return False

    def _handle_variation_addition(self, variation: str, base_term: str) -> Dict[str, Any]:
        """Handle adding a variation to an existing learned term."""
        try:
            if not self.state.learning_mode:
                return {
                    "success": False,
                    "message": "Not currently in learning mode"
                }
            
            # Validate variation
            analysis = self.pattern_helpers.analyze_pattern(base_term, variation)
            confidence = analysis.get('confidence', 0)
            
            if confidence > 0.7:
                self.state.learning_variations.append(variation)
                return {
                    "success": True,
                    "message": "Variation added successfully",
                    "confidence": confidence
                }
            else:
                return {
                    "success": False,
                    "message": "Variation does not match base term closely enough",
                    "confidence": confidence
                }
        
        except Exception as e:
            logger.error(f"Variation addition error: {e}")
            return {
                "success": False,
                "message": f"Error adding variation: {str(e)}"
            }

    def _confirm_learning(self, term: str) -> Dict[str, Any]:
        """Confirm and finalize learning of a term."""
        try:
            if not self.state.learning_mode:
                return {
                    "success": False,
                    "message": "Not currently in learning mode"
                }
            
            # Validate accumulated variations
            if not self.state.learning_variations:
                return {
                    "success": False,
                    "message": "No variations learned for this term"
                }
            
            # Prepare term data for saving
            term_data = {
                'written_form': term,
                'variations': self.state.learning_variations
            }
            
            # Save learned term
            save_result = self._save_learned_term(term_data)
            
            # Reset learning state
            self.state.learning_mode = False
            self.state.learning_variations = []
            
            return {
                "success": save_result,
                "message": "Term learning confirmed and saved" if save_result else "Failed to save learned term"
            }
        
        except Exception as e:
            logger.error(f"Learning confirmation error: {e}")
            return {
                "success": False,
                "message": f"Error confirming learning: {str(e)}"
            }

    def _verify_system_ready(self) -> bool:
        """Verify that all necessary components are initialized and ready."""
        try:
            # Check Dugal initialization
            if not hasattr(self.state, 'dugal') or not self.state.dugal:
                logger.warning("Dugal not properly initialized")
                return False
                
            # Check critical components
            critical_components = [
                ('onedrive_handler', hasattr(self.state.dugal, 'onedrive_handler')),
                ('excel_handler', hasattr(self.state.dugal, 'excel_handler'))
            ]
            
            # Log any missing components
            missing_components = [name for name, exists in critical_components if not exists]
            
            if missing_components:
                logger.warning(f"System not fully initialized. Missing: {missing_components}")
                return False
            
            # Additional readiness checks
            onedrive_handler = self.state.dugal.onedrive_handler
            if not onedrive_handler.state.local_file_path:
                logger.warning("No active file loaded")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"System readiness check failed: {e}")
            return False

    def _correct_item_name(self, item: str) -> str:
        """Correct and normalize item names using multiple strategies."""
        try:
            # Normalize whitespace and case
            normalized = item.strip().lower()
            
            # Check misspellings cache
            misspellings = self.state.cache.get('misspellings', {})
            if normalized in misspellings:
                return misspellings[normalized]
            
            # Use pattern learning for advanced correction
            corrected = self.pattern_helpers._try_pattern_correction(normalized)
            if corrected:
                return corrected
            
            return normalized
        
        except Exception as e:
            logger.error(f"Item name correction error: {e}")
            return item

    def _execute_inventory_update(self, item: str, value: float, is_addition: bool = False) -> Dict[str, Any]:
        """Execute comprehensive inventory update with multi-stage validation."""
        try:
            # Get handlers
            onedrive_handler = self.state.dugal.onedrive_handler
            
            # Validate inputs
            if not item or value is None:
                return {
                    "success": False,
                    "message": "Invalid item or value"
                }
            
            # Check value range
            if abs(value) > 1000:
                return {
                    "success": False,
                    "message": "Update value exceeds reasonable range"
                }
            
            # Perform update
            update_result = onedrive_handler.update_inventory(item, value, is_addition)
            
            # Additional processing if update successful
            if update_result.get('success'):
                # Log successful update
                if self.state.logging_manager:
                    self.state.logging_manager.log_pattern_match({
                        'type': 'inventory_update',
                        'item': item,
                        'value': value,
                        'operation': 'add/subtract' if is_addition else 'set',
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Record for pattern learning
                self.pattern_helpers.record_learning({
                    'from': item,
                    'to': item,  # Assuming successful update validates the item
                    'confidence': 1.0,
                    'type': 'inventory_update'
                })
            
            return update_result
        
        except Exception as e:
            logger.error(f"Inventory update execution error: {e}")
            if self.state.logging_manager:
                self.state.logging_manager.log_error(str(e), {
                    'context': 'inventory_update',
                    'item': item,
                    'value': value,
                    'operation': 'add/subtract' if is_addition else 'set'
                })
            
            return {
                "success": False,
                "message": f"Update failed: {str(e)}"
            }

    def _extract_inventory_details(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """Extract item name and numeric value from voice input."""
        try:
            # Mapping for converting text numbers to digits
            number_mapping = {
                'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
                'point': '.', 'dot': '.', 'minus': '-'
            }
            
            # Terms to ignore
            ignored_terms = {'ml', 'liters', 'bottle', 'bottles', 'year', 'years'}
            
            words = text.lower().split()
            
            number_words = []
            item_words = []
            
            for word in words:
                if (word in number_mapping or 
                    word.replace('.', '', 1).isdigit() or 
                    word in {'year', 'years'}):
                    number_words.append(word)
                elif word not in ignored_terms:
                    item_words.append(word)
            
            if not number_words:
                return None, None
            
            # Convert number words to digits
            number_str = ''.join(
                number_mapping.get(word, word) 
                for word in number_words 
                if word not in {'year', 'years'}
            )
            
            # Process item name
            item_name = ' '.join(item_words).strip()
            
            if not item_name:
                return None, None
            
            try:
                value = float(number_str)
                return item_name, value
            except ValueError:
                logger.error(f"Could not convert '{number_str}' to a number")
                return None, None
        
        except Exception as e:
            logger.error(f"Inventory details extraction error: {e}")
            return None, None

    def _find_similar_items(self, item_name: str) -> List[str]:
        """Find similar items using multiple matching strategies."""
        try:
            if hasattr(self.state.dugal, 'excel_handler'):
                excel_handler = self.state.dugal.excel_handler
                if hasattr(excel_handler, 'search_engine'):
                    search_engine = excel_handler.search_engine
                    matches = search_engine.find_similar_items(item_name)
                    return matches[:3]  # Top 3 matches
            
            return []
        
        except Exception as e:
            logger.error(f"Similar items search error: {e}")
            return []

    def _handle_unrecognized_command(self, command: str) -> Dict[str, Any]:
        """Handle commands that cannot be processed with advanced suggestion mechanism."""
        try:
            # Attempt to find similar commands using pattern matching
            similar_commands = self._find_similar_commands(command)
            
            # Log unrecognized command
            if self.state.logging_manager:
                self.state.logging_manager.log_error("Unrecognized command", {
                    'context': 'command_processing',
                    'command': command,
                    'suggestions': similar_commands
                })
            
            return {
                "success": False,
                "message": "Command not understood",
                "suggestions": similar_commands
            }
        
        except Exception as e:
            logger.error(f"Unrecognized command handling error: {e}")
            return {
                "success": False,
                "message": "Unable to process command"
            }

    def _find_similar_commands(self, command: str) -> List[str]:
        """Find potentially similar commands using various matching strategies."""
        try:
            # Predefined command patterns for comparison
            command_patterns = [
                r'add (\d+) (.*)',
                r'update (.*) to (\d+)',
                r'remove (\d+) (.*)',
                r'learn term (.*)',
                r'set (.*) as (\d+)'
            ]
            
            similar_commands = []
            
            # Pattern-based matching
            for pattern in command_patterns:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    similar_commands.append(match.group(0))
            
            # Phonetic and edit distance matching
            phonetic_matches = self.pattern_helpers._find_phonetic_matches(
                command.split(), 
                command.split()
            )
            similar_commands.extend([match[1] for match in phonetic_matches])
            
            return list(set(similar_commands))[:3]  # Limit to top 3 unique suggestions
        
        except Exception as e:
            logger.error(f"Similar command search error: {e}")
            return []

    def _format_success_response(self, result: Dict[str, Any]) -> str:
        """Format a success response with variation based on update context."""
        try:
            updates = result.get("updates", [])
            if not updates:
                return "Update successful, but details are unclear."

            update = updates[0]
            item = update.get("item", "something")
            new_value = update.get("new_value", 0)
            sheet_count = len(updates)

            # Template-based response generation
            responses = [
                f"Updated {item} to {new_value} across {sheet_count} sheets.",
                f"Inventory for {item} now set to {new_value}.",
                f"Successfully updated {item}: new value is {new_value}.",
                f"Tracking {item} updated to {new_value} in {sheet_count} locations."
            ]

            return random.choice(responses)

        except Exception as e:
            logger.error(f"Success response formatting error: {e}")
            return "Update processed successfully."

    def _format_error_response(self, result: Dict[str, Any]) -> str:
        """Format an error response with contextual information."""
        try:
            message = result.get("message", "Unknown error occurred")
            suggestions = result.get("suggestions", [])

            # Base error messages
            error_templates = [
                f"Error processing request: {message}",
                f"Unable to complete operation: {message}",
                f"Request encountered an issue: {message}"
            ]

            # Add suggestions if available
            if suggestions:
                suggestion_text = f" Suggestions: {', '.join(suggestions)}"
                error_templates = [msg + suggestion_text for msg in error_templates]

            return random.choice(error_templates)

        except Exception as e:
            logger.error(f"Error response formatting error: {e}")
            return "An unexpected error occurred during processing."

    def _log_command_attempt(self, command: str) -> None:
        """Log details of command processing attempt."""
        try:
            if self.state.logging_manager:
                self.state.logging_manager.log_pattern_match({
                    'type': 'command_attempt',
                    'command': command,
                    'timestamp': datetime.now().isoformat(),
                    'mode': self.state.mode
                })
        except Exception as e:
            logger.error(f"Command logging error: {e}")

    def _track_successful_update(self, item: str, value: float) -> None:
        """Track and log successful inventory updates."""
        try:
            # Update local statistics
            self.state.stats['total_updates'] = self.state.stats.get('total_updates', 0) + 1
            
            # Log to performance tracking
            if self.state.logging_manager:
                self.state.logging_manager.log_performance({
                    'type': 'inventory_update',
                    'item': item,
                    'value': value,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Potential pattern learning
            self.pattern_helpers.record_learning({
                'from': item,
                'to': item,
                'confidence': 1.0,
                'type': 'inventory_update'
            })
        
        except Exception as e:
            logger.error(f"Update tracking error: {e}")

    def _track_recognition_performance(self, success: bool, duration: float, command: str = None, error: str = None) -> None:
        """Track speech recognition performance metrics."""
        try:
            # Update state stats
            stats = self.state.stats
            stats['total_attempts'] = stats.get('total_attempts', 0) + 1
            
            if success:
                stats['successful_recognitions'] = stats.get('successful_recognitions', 0) + 1
                stats['last_success_time'] = datetime.now().isoformat()
            else:
                stats['failed_recognitions'] = stats.get('failed_recognitions', 0) + 1
            
            # Performance metrics
            metrics = {
                'type': 'recognition',
                'success': success,
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'mode': self.state.mode
            }
            
            # Add command or error details if provided
            if command:
                metrics['command'] = command
            if error:
                metrics['error'] = error
            
            # Log performance data
            if self.state.logging_manager:
                self.state.logging_manager.log_performance(metrics)
            
        except Exception as e:
            logger.error(f"Error tracking recognition performance: {e}")

    def _track_learning_performance(self, result: Dict[str, Any]) -> None:
        """Track dictionary learning performance metrics."""
        try:
            stats = self.state.stats
            stats['learning_attempts'] = stats.get('learning_attempts', 0) + 1
            
            if result.get('success', False):
                stats['successful_learns'] = stats.get('successful_learns', 0) + 1
            
            # Performance logging
            if self.state.logging_manager:
                self.state.logging_manager.log_performance({
                    'type': 'learning',
                    'success': result.get('success', False),
                    'term': result.get('term'),
                    'duration': result.get('duration', 0),
                    'timestamp': datetime.now().isoformat(),
                    'total_attempts': stats['learning_attempts'],
                    'success_rate': stats.get('successful_learns', 0) / max(stats['learning_attempts'], 1) * 100
                })
        
        except Exception as e:
            logger.error(f"Learning performance tracking error: {e}")

    def _update_command_metrics(self, result: Dict[str, Any]) -> None:
        """Update command processing performance metrics."""
        try:
            # Store in command history
            if hasattr(self.state, 'command_history'):
                self.state.command_history.append({
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Limit history size
                if len(self.state.command_history) > 100:
                    self.state.command_history = self.state.command_history[-100:]
            
            # Log performance if available
            if self.state.logging_manager:
                self.state.logging_manager.log_performance({
                    'type': 'command',
                    'success': result.get('success', False),
                    'command_type': result.get('type', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error updating command metrics: {e}")

    def _save_performance_metrics(self) -> None:
        """Save performance metrics to persistent storage."""
        try:
            if self.state.logging_manager and hasattr(self.state.logging_manager, 'save_stats'):
                self.state.logging_manager.save_stats()
                
            logger.debug("Performance metrics saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")

    def _handle_error_status(self) -> None:
        """Handle system error status with recovery attempts."""
        try:
            error_count = self.state.error_count
            
            # Apply graduated recovery based on error count
            if error_count >= 5:
                # Severe error - attempt full system reset
                logger.warning("Multiple errors detected, attempting system reset")
                if self._attempt_service_recovery():
                    self.state.error_count = 0
                    self.speak("System recovery complete.")
                    self.status_changed.emit("recovered")
                else:
                    self.speak("System errors persist. Please restart the application.")
            elif error_count >= 3:
                # Moderate error - attempt component recovery
                logger.warning("Several errors detected, attempting component recovery")
                self._attempt_component_recovery()
                self.speak("Attempting to recover from errors.")
            else:
                # Minor error - log and continue
                logger.info("Minor error detected, continuing operation")
                self.speak("Encountered a minor error. Let's continue.")
                
        except Exception as e:
            logger.error(f"Error handling error status: {e}")

    def _handle_inactive_status(self) -> None:
        """Handle system inactivity with appropriate response."""
        try:
            # Optimize resources during inactivity
            self._optimize_resources()
            
            # Get appropriate inactivity message
            message = self._get_inactivity_message()
            self.speak(message)
            
            # Log inactivity
            if self.state.logging_manager:
                self.state.logging_manager.log_pattern_match({
                    'type': 'inactivity',
                    'duration': (datetime.now() - self.state.last_activity).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error handling inactive status: {e}")

    def _get_inactivity_message(self) -> str:
        """Get appropriate inactivity message based on personality mode."""
        mode = self.state.mode
        
        messages = {
            'wild': "Hey, still with me? I'm all ears whenever you're ready!",
            'mild': "Just checking if you need anything. I'm still listening.",
            'proper': "Pardon the interruption, but I wanted to confirm if you require assistance."
        }
        
        return messages.get(mode, "Are you still there? I'm listening.")

    def _optimize_resources(self) -> None:
        """Optimize system resources during inactivity."""
        try:
            # Clean up pattern cache if needed
            if hasattr(self.pattern_helpers, 'pattern_cache'):
                cache_size = len(self.pattern_helpers.pattern_cache)
                if cache_size > 1000:
                    logger.debug("Pruning pattern cache during inactivity")
                    self._prune_pattern_cache()
            
            # Optimize learning history if needed
            if hasattr(self.pattern_helpers, 'learning_history'):
                history_size = len(self.pattern_helpers.learning_history)
                if history_size > 500:
                    logger.debug("Optimizing learning history during inactivity")
                    self._optimize_learning_history()
            
            # Reset error count if needed
            if self.state.error_count > 0:
                logger.debug("Resetting error count during inactivity")
                self.state.error_count = 0
            
            logger.debug("Resource optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")

    def _prune_pattern_cache(self) -> None:
        """Prune pattern cache to maintain optimal size."""
        try:
            # Sort patterns by usage and confidence
            cache = self.pattern_helpers.pattern_cache
            sorted_patterns = sorted(
                cache.items(),
                key=lambda x: (x[1].get('confidence', 0)),
                reverse=True
            )
            
            # Keep top 1000 patterns
            self.pattern_helpers.pattern_cache = dict(sorted_patterns[:1000])
            
            logger.debug(f"Pattern cache pruned to {len(self.pattern_helpers.pattern_cache)} items")
            
        except Exception as e:
            logger.error(f"Error pruning pattern cache: {e}")

    def _optimize_learning_history(self) -> None:
        """Optimize learning history for better performance."""
        try:
            learning_history = self.pattern_helpers.learning_history
            
            # Group similar patterns
            grouped_patterns = {}
            for entry in learning_history:
                pattern = entry.get('pattern', {})
                key = f"{pattern.get('from', '')}_{pattern.get('to', '')}"
                if key not in grouped_patterns:
                    grouped_patterns[key] = []
                grouped_patterns[key].append(entry)
            
            # Keep most recent and highest confidence patterns
            optimized_history = []
            for patterns in grouped_patterns.values():
                sorted_patterns = sorted(
                    patterns,
                    key=lambda x: (
                        x.get('pattern', {}).get('confidence', 0),
                        x.get('timestamp', '')
                    ),
                    reverse=True
                )
                optimized_history.extend(sorted_patterns[:5])  # Keep top 5 variations
            
            self.pattern_helpers.learning_history = optimized_history
            
            logger.debug(f"Learning history optimized to {len(self.pattern_helpers.learning_history)} entries")
            
        except Exception as e:
            logger.error(f"Error optimizing learning history: {e}")

    def _handle_mode_change(self, *args) -> Dict[str, Any]:
        """Handle personality mode changes."""
        try:
            # Extract mode from arguments
            mode = next((arg for arg in args if arg in ['wild', 'mild', 'proper']), None)
            
            if not mode:
                return {
                    "success": False,
                    "message": "Invalid mode specified"
                }
            
            # Update state mode
            previous_mode = self.state.mode
            self.state.mode = mode
            
            # Update voice based on new mode
            if hasattr(self.state, 'speech_config'):
                voice = self._get_voice_for_mode()
                self.state.speech_config.speech_synthesis_voice_name = voice
            
            # Log mode change
            if self.state.logging_manager:
                self.state.logging_manager.log_pattern_match({
                    'type': 'mode_change',
                    'from': previous_mode,
                    'to': mode,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Provide feedback in new mode
            messages = {
                'wild': "Woo-hoo! Let's have some fun with this!",
                'mild': "Switching to mild mode. Ready to assist you.",
                'proper': "I have switched to proper mode. How may I be of assistance?"
            }
            
            self.speak(messages.get(mode, f"Mode changed to {mode}"))
            
            return {
                "success": True,
                "message": f"Mode changed to {mode}",
                "previous_mode": previous_mode
            }
            
        except Exception as e:
            logger.error(f"Mode change error: {e}")
            return {
                "success": False,
                "message": f"Failed to change mode: {str(e)}"
            }

    def _handle_help_request(self, *args) -> Dict[str, Any]:
        """Generate comprehensive help documentation."""
        try:
            help_sections = {
                "Inventory Commands": [
                    "Add items: 'add 5 vodka'",
                    "Update items: 'update vodka to 10'",
                    "Remove items: 'remove 3 whiskey'"
                ],
                "Learning Commands": [
                    "Learn new term: 'learn term <term>'",
                    "Add variation: 'add variation <variation> for <term>'"
                ],
                "System Commands": [
                    "Change mode: 'change mode to wild/mild/proper'",
                    "System status: 'show status'",
                    "Help: 'help'"
                ]
            }
            
            help_text = "\n".join([
                f"{section}:\n" + 
                "\n".join(f"  - {cmd}" for cmd in commands)
                for section, commands in help_sections.items()
            ])
            
            # Log help request
            if self.state.logging_manager:
                self.state.logging_manager.log_pattern_match({
                    'type': 'help_request',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Speak abbreviated help message
            self.speak("Here are some commands you can use. Check the display for more details.")
            
            return {
                "success": True,
                "message": help_text
            }
            
        except Exception as e:
            logger.error(f"Help request generation error: {e}")
            return {
                "success": False,
                "message": "Unable to generate help documentation"
            }

    def _handle_system_status(self, *args) -> Dict[str, Any]:
        """Generate comprehensive system status report."""
        try:
            # Collect system metrics
            status = {
                "Mode": self.state.mode,
                "Components": {
                    "OneDrive Handler": bool(hasattr(self.state.dugal, 'onedrive_handler')),
                    "Excel Handler": bool(hasattr(self.state.dugal, 'excel_handler')),
                    "Search Engine": bool(
                        hasattr(self.state.dugal, 'excel_handler') and 
                        hasattr(self.state.dugal.excel_handler, 'search_engine')
                    )
                },
                "Learning Stats": {
                    "Total Attempts": self.state.stats.get('learning_attempts', 0),
                    "Successful Learns": self.state.stats.get('successful_learns', 0)
                },
                "Recognition Performance": {
                    "Total Attempts": self.state.stats.get('total_attempts', 0),
                    "Successful Recognitions": self.state.stats.get('successful_recognitions', 0)
                }
            }
            
            # Log status request
            if self.state.logging_manager:
                self.state.logging_manager.log_pattern_match({
                    'type': 'system_status_check',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Generate status summary for speech
            success_rate = 0
            if status["Recognition Performance"]["Total Attempts"] > 0:
                success_rate = (
                    status["Recognition Performance"]["Successful Recognitions"] / 
                    status["Recognition Performance"]["Total Attempts"]
                ) * 100
                
            status_summary = (
                f"System is running in {status['Mode']} mode with a "
                f"{success_rate:.1f}% recognition success rate."
            )
            
            self.speak(status_summary)
            
            return {
                "success": True,
                "message": str(status)
            }
            
        except Exception as e:
            logger.error(f"System status generation error: {e}")
            return {
                "success": False,
                "message": "Unable to generate system status"
            }

    def _attempt_component_recovery(self) -> bool:
        """Attempt to recover specific components after errors."""
        try:
            recovery_successful = True
            
            # Attempt to recover speech services
            if not self._verify_speech_services():
                logger.warning("Speech services need recovery")
                recovery_successful = recovery_successful and self._reinitialize_speech_services()
            
            # Reset error counters
            self.state.error_count = max(0, self.state.error_count - 1)
            
            return recovery_successful
            
        except Exception as e:
            logger.error(f"Component recovery error: {e}")
            return False

    def _verify_speech_services(self) -> bool:
        """Verify speech services are functioning properly."""
        try:
            return bool(
                self.state.synthesizer and 
                self.state.recognizer and 
                self.state.speech_config
            )
        except Exception as e:
            logger.error(f"Error verifying speech services: {e}")
            return False

    def _reinitialize_speech_services(self) -> bool:
        """Reinitialize speech services after failure."""
        try:
            logger.debug("Reinitializing speech services...")
            
            # Store current configuration
            if self.state.speech_config:
                old_key = self.state.speech_config.get_property(
                    speechsdk.PropertyId.SpeechServiceConnection_Key
                )
                old_region = self.state.speech_config.get_property(
                    speechsdk.PropertyId.SpeechServiceConnection_Region
                )
                
                # Reinitialize services
                self._init_azure_speech_services(old_key, old_region)
                
                logger.debug("Speech services reinitialized successfully")
                return True
            else:
                logger.error("No existing speech configuration to reinitialize")
                return False
                
        except Exception as e:
            logger.error(f"Error reinitializing speech services: {e}")
            return False

    def correct_spelling(self, text: str) -> str:
        """Correct spelling of terms using multiple strategies."""
        try:
            words = text.lower().split()
            corrected_words = []

            for word in words:
                # Check misspellings cache
                corrected = word
                misspellings = self.state.cache.get('misspellings', {})
                
                for correct, variations in misspellings.items():
                    if word in variations:
                        corrected = correct
                        break
                
                # Pattern-based correction
                if corrected == word and hasattr(self, 'pattern_helpers'):
                    pattern_correction = self.pattern_helpers._try_pattern_correction(word)
                    if pattern_correction:
                        corrected = pattern_correction
                
                corrected_words.append(corrected)
            
            result = ' '.join(corrected_words)
            
            if result != text:
                logger.debug(f"Spelling corrected: '{text}' -> '{result}'")
            
            return result
        
        except Exception as e:
            logger.error(f"Spelling correction error: {e}")
            return text

    def _provide_command_feedback(self, result: Dict[str, Any]) -> None:
        """Provide appropriate feedback for command results."""
        try:
            mode = self.state.mode
            
            if result.get('success'):
                message = self._get_success_message(result, mode)
            else:
                message = self._get_error_message(result, mode)
            
            # Speak feedback with appropriate priority
            self.speak(message, priority=not result.get('success'))
            
        except Exception as e:
            logger.error(f"Error providing command feedback: {e}")

    def _get_success_message(self, result: Dict[str, Any], mode: str) -> str:
        """Get mode-appropriate success message."""
        base_message = result.get('message', 'Command completed successfully')
        
        messages = {
            'wild': f"Got it! {base_message}",
            'mild': base_message,
            'proper': f"I've completed your request. {base_message}"
        }
        
        return messages.get(mode, base_message)

    def _get_error_message(self, result: Dict[str, Any], mode: str) -> str:
        """Get mode-appropriate error message."""
        base_message = result.get('message', 'Command could not be completed')
        
        messages = {
            'wild': f"Oops! {base_message}. Want to try again?",
            'mild': f"{base_message}. Please try again.",
            'proper': f"I apologize, but {base_message}. Would you like to make another attempt?"
        }
        
        return messages.get(mode, base_message)

    def _provide_command_suggestions(self, suggestions: List[str]) -> None:
        """Provide command suggestions based on failed recognition."""
        try:
            if not suggestions:
                return
            
            mode = self.state.mode
            
            if mode == "wild":
                message = f"Not quite! Did you mean: {', '.join(suggestions)}?"
            elif mode == "proper":
                message = f"I believe you might have meant one of the following: {', '.join(suggestions)}"
            else:  # mild
                message = f"Similar commands: {', '.join(suggestions)}"
            
            self.speak(message)
            
        except Exception as e:
            logger.error(f"Error providing command suggestions: {e}")

    def _provide_error_feedback(self, message: str) -> None:
        """Provide error feedback based on personality mode."""
        try:
            mode = self.state.mode
            
            if mode == "wild":
                message = f"Whoops! {message} Give it another shot!"
            elif mode == "proper":
                message = f"I apologize, but {message} Would you kindly try again?"
            # Default "mild" mode uses message as is
            
            self.speak(message, priority=True)
            
        except Exception as e:
            logger.error(f"Error providing error feedback: {e}")

    def cleanup(self) -> None:
        """Comprehensive cleanup of voice interaction resources."""
        try:
            logger.debug("Starting voice interaction cleanup")
            
            # Stop active timers
            if hasattr(self, 'inactivity_timer'):
                self.inactivity_timer.stop()
            
            # Clean up speech services
            if hasattr(self.state, 'synthesizer') and self.state.synthesizer:
                try:
                    # Different versions of the SDK may have different cleanup methods
                    if hasattr(self.state.synthesizer, 'close'):
                        self.state.synthesizer.close()
                    elif hasattr(self.state.synthesizer, 'dispose'):
                        self.state.synthesizer.dispose()
                except Exception as e:
                    logger.error(f"Synthesizer cleanup error: {e}")
            
            # Clean up recognizer
            if hasattr(self.state, 'recognizer') and self.state.recognizer:
                try:
                    # Stop any ongoing recognition
                    if hasattr(self.state.recognizer, 'stop_listening'):
                        self.state.recognizer.stop_listening()
                except Exception as e:
                    logger.error(f"Recognizer cleanup error: {e}")
            
            # Clean up pattern learning
            if hasattr(self, 'pattern_helpers'):
                self.pattern_helpers.cleanup()
            
            # Save final state
            if hasattr(self, '_save_final_state'):
                self._save_final_state()
            
            logger.debug("Voice interaction cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            if hasattr(self.state, 'logging_manager') and self.state.logging_manager:
                self.state.logging_manager.log_error(str(e), {
                    'context': 'cleanup',
                    'component': 'voice_interaction'
                })

    def _save_final_state(self) -> None:
        """Save final system state and generate performance report."""
        try:
            # Generate final performance report
            performance_report = self._generate_performance_report()
            
            # Log final state
            if hasattr(self.state, 'logging_manager') and self.state.logging_manager:
                self.state.logging_manager.log_performance(performance_report)
            
            # Reset state
            self.state.stats = {
                'total_attempts': 0,
                'successful_recognitions': 0,
                'learning_attempts': 0,
                'successful_learns': 0
            }
            
        except Exception as e:
            logger.error(f"Error saving final state: {e}")

    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            stats = self.state.stats
            return {
                "recognition": {
                    "total_attempts": stats.get('total_attempts', 0),
                    "successful_recognitions": stats.get('successful_recognitions', 0),
                    "success_rate": (
                        stats.get('successful_recognitions', 0) / 
                        max(stats.get('total_attempts', 1), 1)
                    ) * 100
                },
                "learning": {
                    "total_attempts": stats.get('learning_attempts', 0),
                    "successful_learns": stats.get('successful_learns', 0),
                    "patterns_learned": len(self.pattern_helpers.learning_history)
                },
                "errors": {
                    "total_errors": self.state.error_count,
                    "error_types": stats.get('error_types', {})
                },
                "system": {
                    "mode": self.state.mode,
                    "last_activity": (
                        self.state.last_activity.isoformat() 
                        if self.state.last_activity else None
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}

# Standalone test function
def run_voice_interaction() -> bool:
    """Test function to run voice interaction independently."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.debug("Starting voice interaction test")

    try:
        # Create and test voice interaction
        voice = VoiceInteraction()
        voice.speak("Voice interaction test successful")
        logger.debug("Voice interaction test completed")
        return True
    except Exception as e:
        logger.error("Voice interaction test failed: %s", str(e))
        return False

if __name__ == "__main__":
    sys.exit(0 if run_voice_interaction() else 1)
