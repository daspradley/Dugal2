"""
Data manager module for Dugal Inventory System.
Handles persistent storage of learned patterns and system configurations.
"""

import os
import json
import logging
import time
from typing import Dict, Set, Any, Optional
from datetime import datetime
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class DataManager:
    """Manages persistent data storage for Dugal."""
    
    def __init__(self, data_dir: str = ".dugal_data", logging_manager=None, search_engine=None):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Directory for storing persistent data
            logging_manager: Optional logging manager instance
            search_engine: Optional search engine instance
        """
        self.data_dir = data_dir
        self.patterns_file = os.path.join(data_dir, "learned_patterns.json")
        self.backup_dir = os.path.join(data_dir, "backups")
        self.max_backups = 5
        self.logging_manager = logging_manager
        self.search_engine = search_engine
        self._ensure_directories()
        logger.debug(f"Data manager initialized with directory: {data_dir}")
        
        if self.logging_manager:
            self.logging_manager.log_pattern_match({
                'type': 'data_manager_init',
                'data_dir': data_dir,
                'timestamp': datetime.now().isoformat()
            })

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.backup_dir, exist_ok=True)
            logger.debug("Data directories verified")
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            raise

    def save_patterns(self, patterns: Dict[str, Set[str]]) -> bool:
        """Save learned patterns to file with backup."""
        logger.debug("Saving patterns")
        if self.logging_manager:
            self.logging_manager.log_pattern_match({
                'type': 'save_patterns_start',
                'pattern_count': len(patterns),
                'timestamp': datetime.now().isoformat()
            })
        
        try:
            # Create backup first
            if os.path.exists(self.patterns_file):
                backup_path = self._create_backup()
                if self.logging_manager:
                    self.logging_manager.log_pattern_match({
                        'type': 'backup_created',
                        'backup_file': backup_path,
                        'timestamp': datetime.now().isoformat()
                    })

            # Convert sets to lists for JSON serialization
            serializable_patterns = {k: list(v) for k, v in patterns.items()}
            
            # Write to temporary file first
            temp_file = f"{self.patterns_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_patterns, f, indent=2)
            
            # Atomic replace
            os.replace(temp_file, self.patterns_file)
            
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'save_patterns_complete',
                    'pattern_count': len(patterns),
                    'timestamp': datetime.now().isoformat()
                })
                
            logger.debug(f"Successfully saved {len(patterns)} patterns")
            return True
            
        except Exception as e:
            error_msg = f"Error saving patterns: {e}"
            logger.error(error_msg)
            if self.logging_manager:
                self.logging_manager.log_error(error_msg, {
                    'context': 'save_patterns',
                    'pattern_count': len(patterns)
                })
            return False

    def load_state(self) -> Dict[str, Any]:
        """Load saved state from file."""
        state_file = os.path.join(self.data_dir, "state.json")
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state: {e}")
        return {}

    def save_state(self, state: Dict[str, Any]) -> bool:
        """Save current state to file."""
        state_file = os.path.join(self.data_dir, "state.json")
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False

    def load_patterns(self) -> Dict[str, Set[str]]:
        """
        Load learned patterns from file.
        
        Returns:
            Dict[str, Set[str]]: Loaded patterns
        """
        logger.debug("Loading patterns")
        try:
            if not os.path.exists(self.patterns_file):
                logger.debug("No patterns file found, starting fresh")
                return {}
                
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
            
            # Convert lists back to sets
            converted_patterns = {
                k: set(v) for k, v in patterns.items()
            }
            
            logger.debug(f"Successfully loaded {len(converted_patterns)} patterns")
            return converted_patterns
            
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            # Try to recover from backup
            backup_patterns = self._recover_from_backup()
            if backup_patterns is not None:
                return backup_patterns
            return {}

    def _create_backup(self) -> str:  # Change return type to str
        """Create a backup of the patterns file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(
                self.backup_dir,
                f"patterns_{timestamp}.json"
            )
            
            # Copy current file to backup
            with open(self.patterns_file, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
                    
            self._cleanup_old_backups()
            logger.debug(f"Created backup: {backup_path}")
            
            return backup_path  # Add this return statement
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise  # Raise the exception instead of implicitly returning None

    def _cleanup_old_backups(self) -> None:
        """Remove old backups keeping only the most recent ones."""
        try:
            backups = sorted(
                Path(self.backup_dir).glob("patterns_*.json"),
                key=os.path.getctime
            )
            
            while len(backups) > self.max_backups:
                oldest = backups.pop(0)
                os.remove(oldest)
                logger.debug(f"Removed old backup: {oldest}")
                
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")

    def _recover_from_backup(self) -> Optional[Dict[str, Set[str]]]:
        """Try to recover patterns from most recent backup."""
        try:
            backups = sorted(
                Path(self.backup_dir).glob("patterns_*.json"),
                key=os.path.getctime,
                reverse=True
            )
            
            for backup in backups:
                try:
                    with open(backup, 'r', encoding='utf-8') as f:
                        patterns = json.load(f)
                    
                    converted_patterns = {
                        k: set(v) for k, v in patterns.items()
                    }
                    
                    logger.debug(f"Successfully recovered from backup: {backup}")
                    return converted_patterns
                    
                except Exception as e:
                    logger.error(f"Error reading backup {backup}: {e}")
                    continue
                    
            logger.error("No valid backups found")
            return None
            
        except Exception as e:
            logger.error(f"Error during backup recovery: {e}")
            return None

    def verify_data_integrity(self) -> bool:
        """Verify integrity of saved data."""
        try:
            if not os.path.exists(self.patterns_file):
                return True
                
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
            
            # Basic validation
            if not isinstance(patterns, dict):
                raise ValueError("Invalid patterns format")
                
            for key, value in patterns.items():
                if not isinstance(key, str):
                    raise ValueError(f"Invalid key type: {type(key)}")
                if not isinstance(value, list):
                    raise ValueError(f"Invalid value type: {type(value)}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Data integrity check failed: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        stats = {
            "patterns_count": 0,
            "variations_count": 0,
            "file_size": 0,
            "last_modified": None,
            "backup_count": 0
        }
        
        try:
            if os.path.exists(self.patterns_file):
                stats["file_size"] = os.path.getsize(self.patterns_file)
                stats["last_modified"] = datetime.fromtimestamp(
                    os.path.getmtime(self.patterns_file)
                )
                
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    patterns = json.load(f)
                    stats["patterns_count"] = len(patterns)
                    stats["variations_count"] = sum(len(v) for v in patterns.values())
            
            stats["backup_count"] = len(list(Path(self.backup_dir).glob("patterns_*.json")))
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            
        return stats
