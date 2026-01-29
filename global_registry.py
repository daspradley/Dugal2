"""
Global registry for sharing component instances across the Dugal system.
"""

import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GlobalRegistry:
    """Central registry for sharing component instances across the application."""
    
    _instances = {}
    _initialized = False
    _lock = threading.Lock()
    
    @classmethod
    def register(cls, name: str, instance: Any) -> Any:
        """
        Register a component instance in the global registry.
        Returns the instance for convenient chaining.
        """
        with cls._lock:  # Add thread safety
            if name in cls._instances:
                old_id = id(cls._instances[name])
                new_id = id(instance)
                if old_id != new_id:
                    logger.warning(f"Replacing existing {name} in registry: {old_id} -> {new_id}")
            else:
                logger.debug(f"Registering new instance {name} in global registry (ID: {id(instance)})")
                
            cls._instances[name] = instance
            return instance
        
    @classmethod
    def get(cls, name: str, default: Any = None) -> Any:
        """
        Get a component from the registry.
        Returns default if the component is not found.
        """
        with cls._lock:  # Add thread safety
            if name in cls._instances:
                logger.debug(f"Retrieved {name} from registry (ID: {id(cls._instances[name])})")
                return cls._instances[name]
            else:
                logger.debug(f"Requested component '{name}' not found in registry")
                return default
    
    @classmethod
    def list_keys(cls) -> list:
        """List all keys in the registry."""
        with cls._lock:
            return list(cls._instances.keys())
            
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a component is registered."""
        with cls._lock:
            return name in cls._instances
        
    @classmethod
    def update(cls, name: str, instance: Any) -> Any:
        """
        Update an existing component in the registry.
        Creates a new entry if it doesn't exist.
        """
        old_id = id(cls._instances.get(name)) if name in cls._instances else None
        new_id = id(instance)
        
        if old_id != new_id:
            logger.info(f"Updating {name} in registry: {old_id} -> {new_id}")
            
        cls._instances[name] = instance
        return instance
        
    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if a component exists in the registry."""
        return name in cls._instances
        
    @classmethod
    def clear(cls) -> None:
        """Clear all registered instances."""
        cls._instances.clear()
        logger.debug("Global registry cleared")
        
    @classmethod
    def diagnose(cls) -> Dict[str, Any]:
        """Return diagnostic information about the registry."""
        result = {
            "registered_components": len(cls._instances),
            "components": {}
        }
        
        for name, instance in cls._instances.items():
            result["components"][name] = {
                "id": id(instance),
                "type": type(instance).__name__
            }
            
        logger.debug(f"Registry diagnosis: {len(cls._instances)} components registered")
        return result

# Create a global instance for easy imports
registry = GlobalRegistry()
