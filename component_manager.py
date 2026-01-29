"""
Component Manager for Dugal Inventory System.

Provides standardized access to shared components across the application.
Ensures consistent component lifecycle management and thread safety.
"""

import logging
import threading
from typing import Dict, Any, Optional, Type, Callable
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Import global registry
from global_registry import GlobalRegistry

class ComponentManager:
    """
    Centralized component manager that provides standardized access to shared components.
    
    Features:
    - Thread-safe component access
    - Consistent component lifecycle management
    - Automatic component registration
    - Component recovery and fallback mechanisms
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance of ComponentManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = ComponentManager()
            return cls._instance
    
    def __init__(self):
        """Initialize the component manager."""
        self.logger = logging.getLogger(__name__)
        self.component_factories = {}
        self.component_recovery_handlers = {}
        self.last_access_times = {}
        
        # Register default factories
        self._register_default_factories()
        
    def register_component_factory(self, component_name: str, factory_func: Callable) -> None:
        """
        Register a factory function for creating a component if it doesn't exist.
        
        Args:
            component_name: The name of the component
            factory_func: A function that creates a new instance of the component
        """
        self.component_factories[component_name] = factory_func
        logger.debug(f"Registered factory for component: {component_name}")
        
    def register_recovery_handler(self, component_name: str, recovery_func: Callable) -> None:
        """
        Register a recovery handler for a component.
        
        Args:
            component_name: The name of the component
            recovery_func: A function that attempts to recover the component
        """
        self.component_recovery_handlers[component_name] = recovery_func
        logger.debug(f"Registered recovery handler for component: {component_name}")
    
    def get_component(self, component_name: str, default: Any = None) -> Any:
        """
        Get a component by name with standardized fallback and recovery.
        
        Args:
            component_name: The name of the component to retrieve
            default: Default value to return if component cannot be found or created
            
        Returns:
            The component instance or default if not available
        """
        # Record access time for diagnostics
        self.last_access_times[component_name] = datetime.now()
        
        # First try to get from registry
        component = GlobalRegistry.get(component_name)
        
        # If component exists, return it
        if component is not None:
            logger.debug(f"Retrieved {component_name} from registry")
            return component
            
        # If no component but we have a factory, create it
        if component_name in self.component_factories:
            try:
                logger.debug(f"Creating {component_name} using factory")
                component = self.component_factories[component_name]()
                if component is not None:
                    # Register the newly created component
                    GlobalRegistry.register(component_name, component)
                    return component
            except Exception as e:
                logger.error(f"Error creating {component_name}: {e}")
        
        # Return default if we couldn't get or create the component
        logger.warning(f"Component {component_name} not available")
        return default
    
    def get_search_engine(self):
        """
        Standardized method to get the search engine instance.
        
        Returns:
            The search engine instance or None if not available
        """
        return self.get_component('search_engine')
    
    def recover_component(self, component_name: str) -> bool:
        """
        Attempt to recover a component using its registered recovery handler.
        
        Args:
            component_name: The name of the component to recover
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if component_name not in self.component_recovery_handlers:
            logger.warning(f"No recovery handler for {component_name}")
            return False
            
        try:
            logger.info(f"Attempting to recover {component_name}")
            recovery_func = self.component_recovery_handlers[component_name]
            result = recovery_func()
            
            if result:
                logger.info(f"Successfully recovered {component_name}")
            else:
                logger.warning(f"Failed to recover {component_name}")
                
            return result
        except Exception as e:
            logger.error(f"Error recovering {component_name}: {e}")
            return False
    
    def register_component(self, component_name: str, component: Any) -> Any:
        """
        Register a component in the global registry.
        
        Args:
            component_name: The name of the component
            component: The component instance
            
        Returns:
            The registered component
        """
        return GlobalRegistry.register(component_name, component)
    
    def _register_default_factories(self) -> None:
        """Register default factory functions for common components."""
        try:
            # Register search engine factory
            def create_search_engine():
                try:
                    from search_engine import AdaptiveInventorySearchEngine
                    self.logger.debug("Creating new search engine instance via factory")
                    return AdaptiveInventorySearchEngine()
                except ImportError:
                    self.logger.error("Failed to import AdaptiveInventorySearchEngine")
                    return None
                except Exception as e:
                    self.logger.error(f"Error creating search engine: {e}")
                    return None
                    
            self.register_component_factory('search_engine', create_search_engine)
            
            # Register search engine recovery handler
            def recover_search_engine():
                try:
                    # Try to get existing instance first
                    search_engine = GlobalRegistry.get('search_engine')
                    
                    # If no instance exists, create a new one
                    if not search_engine:
                        search_engine = create_search_engine()
                        if search_engine:
                            GlobalRegistry.register('search_engine', search_engine)
                            self.logger.info("Created new search engine during recovery")
                            return True
                        return False
                        
                    # If instance exists but might be corrupted, try to reinitialize it
                    if hasattr(search_engine, '_ensure_nltk_data'):
                        search_engine._ensure_nltk_data()
                    if hasattr(search_engine, '_load_learned_patterns'):
                        search_engine.learned_patterns = search_engine._load_learned_patterns()
                    if hasattr(search_engine, '_load_base_patterns'):
                        search_engine.base_patterns = search_engine._load_base_patterns()
                        
                    self.logger.info("Recovered existing search engine")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error recovering search engine: {e}")
                    return False
                    
            self.register_recovery_handler('search_engine', recover_search_engine)
            
        except Exception as e:
            self.logger.error(f"Error registering default factories: {e}")
    
    def diagnose_components(self) -> Dict[str, Any]:
        """
        Get diagnostic information about all components.
        
        Returns:
            A dictionary with diagnostic information
        """
        registry_info = GlobalRegistry.diagnose()
        
        # Add our own diagnostic information
        result = {
            "registry_info": registry_info,
            "component_factories": list(self.component_factories.keys()),
            "recovery_handlers": list(self.component_recovery_handlers.keys()),
            "last_access_times": {k: v.isoformat() for k, v in self.last_access_times.items()}
        }
        
        return result

# Create a singleton instance for easy imports
component_manager = ComponentManager.get_instance()