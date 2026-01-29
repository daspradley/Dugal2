"""
Configuration manager module for Dugal Inventory System.
Centralizes all configuration settings with file-based and environment variable overrides.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class DugalConfig:
    """Centralized configuration for the Dugal system."""
    
    # Search thresholds
    SIMILARITY_THRESHOLD = 0.35
    TOKEN_MATCH_THRESHOLD = 0.45
    
    # Phonetic matching
    GENERATE_PHONETIC_VARIATIONS = True
    MAX_VARIATIONS_PER_TERM = 10
    
    # File operations
    FILE_BACKUP_COUNT = 5
    REFRESH_INTERVAL = 2  # seconds
    
    # Voice recognition
    SPEECH_TIMEOUT = 5  # seconds
    COMMAND_CONFIDENCE_THRESHOLD = 0.7
    
    # Logging
    LOG_LEVEL = logging.DEBUG
    MAX_LOG_FILES = 10
    MAX_LOG_SIZE = 1024 * 1024 * 5  # 5 MB
    
    # Dictionary learning
    AUTO_LEARN_PATTERNS = True
    MIN_TERM_LENGTH = 3
    
    # Application paths
    DATA_DIR = ".dugal_data"
    TEMP_DIR = ".dugal_temp"
    
    @classmethod
    def load_from_file(cls, config_path: str) -> bool:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return False
                
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
            # Update class attributes from config file
            for key, value in config_data.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
                    
            logger.debug(f"Loaded configuration from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    @classmethod
    def load_from_env(cls) -> None:
        """Load configuration from environment variables."""
        try:
            # Look for environment variables with DUGAL_ prefix
            for key in dir(cls):
                if key.isupper():  # Only process constants
                    env_key = f"DUGAL_{key}"
                    if env_key in os.environ:
                        # Convert value to appropriate type
                        orig_value = getattr(cls, key)
                        env_value = os.environ[env_key]
                        
                        # Convert based on original type
                        if isinstance(orig_value, bool):
                            value = env_value.lower() in ('true', 'yes', '1')
                        elif isinstance(orig_value, int):
                            value = int(env_value)
                        elif isinstance(orig_value, float):
                            value = float(env_value)
                        else:
                            value = env_value
                            
                        setattr(cls, key, value)
                        logger.debug(f"Set {key}={value} from environment")
            
        except Exception as e:
            logger.error(f"Error loading configuration from environment: {e}")
    
    @classmethod
    def save_to_file(cls, config_path: str) -> bool:
        """
        Save current configuration to a JSON file.
        
        Args:
            config_path: Path to save configuration
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            config_dict = {}
            
            # Get all uppercase attributes (constants)
            for key in dir(cls):
                if key.isupper():
                    value = getattr(cls, key)
                    # Only serialize JSON-compatible types
                    if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                        config_dict[key] = value
            
            # Write to temporary file first
            temp_path = f"{config_path}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            # Atomic replace
            os.replace(temp_path, config_path)
            logger.debug(f"Saved configuration to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """
        Get current configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        config_dict = {}
        for key in dir(cls):
            if key.isupper():
                config_dict[key] = getattr(cls, key)
        return config_dict
    
    @classmethod
    def update(cls, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
        """
        for key, value in config_dict.items():
            if hasattr(cls, key) and key.isupper():
                setattr(cls, key, value)

# Initialize configuration
def init_config(config_path: Optional[str] = None) -> None:
    """
    Initialize configuration from file and environment.
    
    Args:
        config_path: Optional path to configuration file
    """
    if config_path:
        DugalConfig.load_from_file(config_path)
    
    # Environment variables override file settings
    DugalConfig.load_from_env()
    
    logger.debug("Configuration initialized")
