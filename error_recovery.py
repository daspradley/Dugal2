"""
Error recovery module for Dugal Inventory System.
Provides utilities for robust error handling and recovery mechanisms.
"""

import os
import time
import logging
import traceback
import json
from functools import wraps
from typing import Callable, Any, Dict, Optional, Tuple, List
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class RecoveryState:
    """Class to track recovery attempts and failures."""
    
    def __init__(self):
        self.recovery_attempts = {}
        self.last_errors = {}
        self.component_health = {}
    
    def record_attempt(self, operation: str) -> int:
        """
        Record a recovery attempt for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            int: Number of attempts for this operation
        """
        if operation not in self.recovery_attempts:
            self.recovery_attempts[operation] = 0
        self.recovery_attempts[operation] += 1
        return self.recovery_attempts[operation]
    
    def record_error(self, operation: str, error: Exception) -> None:
        """
        Record an error for an operation.
        
        Args:
            operation: Operation name
            error: Exception object
        """
        self.last_errors[operation] = {
            'error': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
    
    def mark_component_health(self, component: str, healthy: bool) -> None:
        """
        Mark a component as healthy or unhealthy.
        
        Args:
            component: Component name
            healthy: True if component is healthy, False otherwise
        """
        self.component_health[component] = healthy
    
    def reset_attempts(self, operation: str) -> None:
        """
        Reset recovery attempts for an operation.
        
        Args:
            operation: Operation name
        """
        if operation in self.recovery_attempts:
            self.recovery_attempts[operation] = 0
    
    def get_report(self) -> Dict[str, Any]:
        """
        Get a report of recovery state.
        
        Returns:
            Dict[str, Any]: Recovery report
        """
        return {
            'recovery_attempts': self.recovery_attempts.copy(),
            'last_errors': self.last_errors.copy(),
            'component_health': self.component_health.copy(),
            'timestamp': datetime.now().isoformat()
        }

# Global recovery state
recovery_state = RecoveryState()

def safe_operation(operation_func: Callable, *args, 
                   operation_name: Optional[str] = None,
                   max_retries: int = 3, 
                   retry_delay: float = 0.5,
                   recovery_func: Optional[Callable] = None,
                   **kwargs) -> Any:
    """
    Execute an operation with automatic retry and recovery.
    
    Args:
        operation_func: Function to execute
        *args: Arguments for the function
        operation_name: Name of the operation (defaults to function name)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        recovery_func: Optional recovery function to call on failure
        **kwargs: Keyword arguments for the function
        
    Returns:
        Any: Result of the operation function
        
    Raises:
        RuntimeError: If operation fails after all retries
    """
    if operation_name is None:
        operation_name = operation_func.__name__
    
    for attempt in range(max_retries):
        try:
            result = operation_func(*args, **kwargs)
            # Success - reset the attempt counter
            recovery_state.reset_attempts(operation_name)
            return result
            
        except Exception as e:
            logger.error(f"Operation '{operation_name}' failed (attempt {attempt+1}/{max_retries}): {e}")
            recovery_state.record_error(operation_name, e)
            recovery_state.record_attempt(operation_name)
            
            if attempt == max_retries - 1:
                # Last attempt, try recovery function if provided
                if recovery_func:
                    try:
                        logger.info(f"Attempting recovery for '{operation_name}'")
                        recovery_func()
                    except Exception as recovery_error:
                        logger.error(f"Recovery for '{operation_name}' failed: {recovery_error}")
                
                # Raise a more informative error
                raise RuntimeError(f"Operation '{operation_name}' failed after {max_retries} attempts: {e}")
            
            # Wait before retry with exponential backoff
            delay = retry_delay * (2 ** attempt)
            logger.debug(f"Retrying '{operation_name}' in {delay:.2f} seconds")
            time.sleep(delay)

def with_recovery(max_retries: int = 3, 
                 retry_delay: float = 0.5,
                 recovery_func: Optional[Callable] = None):
    """
    Decorator for functions to add automatic retry and recovery.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        recovery_func: Optional recovery function to call on failure
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return safe_operation(
                func, *args, 
                operation_name=func.__name__,
                max_retries=max_retries,
                retry_delay=retry_delay,
                recovery_func=recovery_func,
                **kwargs
            )
        return wrapper
    return decorator

def repair_json_file(file_path: str) -> bool:
    """
    Attempt to repair a corrupted JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        bool: True if repair was successful, False otherwise
    """
    try:
        logger.debug(f"Attempting to repair JSON file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return False
        
        # Try to load the file
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            logger.debug(f"File {file_path} is already valid JSON")
            return True
        except json.JSONDecodeError:
            pass
        
        # File is corrupted, try to recover
        # First, make a backup
        backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            with open(file_path, 'r') as src:
                with open(backup_path, 'w') as dst:
                    dst.write(src.read())
            logger.debug(f"Created backup at {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
        
        # Reset the file to a valid state
        if file_path.endswith('.json'):
            # Determine whether it should be an empty object or array
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                if content.startswith('['):
                    default_content = '[]'
                else:
                    default_content = '{}'
            except:
                default_content = '{}'
                
            with open(file_path, 'w') as f:
                f.write(default_content)
            logger.debug(f"Reset {file_path} to {default_content}")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error repairing JSON file: {e}")
        return False

class ComponentHealthMonitor:
    """
    Monitor the health of system components and manage recovery.
    """
    
    def __init__(self, logging_manager=None):
        self.components = {}
        self.logging_manager = logging_manager
        self.recovery_actions = {}
        
    def register_component(self, component_name: str, 
                          health_check_func: Callable[[], bool],
                          recovery_func: Optional[Callable[[], bool]] = None) -> None:
        """
        Register a component for health monitoring.
        
        Args:
            component_name: Name of the component
            health_check_func: Function to check component health
            recovery_func: Function to recover the component
        """
        self.components[component_name] = {
            'health_check': health_check_func,
            'recovery': recovery_func,
            'last_check': None,
            'healthy': None,
            'recovery_attempts': 0
        }
        
    def check_component(self, component_name: str) -> bool:
        """
        Check the health of a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            bool: True if component is healthy, False otherwise
        """
        if component_name not in self.components:
            logger.warning(f"Component {component_name} not registered")
            return False
            
        component = self.components[component_name]
        try:
            healthy = component['health_check']()
            component['last_check'] = datetime.now()
            component['healthy'] = healthy
            
            # Log component health state
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'component_health_check',
                    'component': component_name,
                    'healthy': healthy,
                    'timestamp': datetime.now().isoformat()
                })
                
            recovery_state.mark_component_health(component_name, healthy)
            return healthy
            
        except Exception as e:
            logger.error(f"Error checking component {component_name}: {e}")
            component['last_check'] = datetime.now()
            component['healthy'] = False
            
            if self.logging_manager:
                self.logging_manager.log_error(str(e), {
                    'context': 'component_health_check',
                    'component': component_name
                })
                
            recovery_state.mark_component_health(component_name, False)
            return False
    
    def recover_component(self, component_name: str) -> bool:
        """
        Attempt to recover a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            bool: True if recovery was successful, False otherwise
        """
        if component_name not in self.components:
            logger.warning(f"Component {component_name} not registered")
            return False
            
        component = self.components[component_name]
        if not component['recovery']:
            logger.warning(f"No recovery function for component {component_name}")
            return False
            
        try:
            logger.info(f"Attempting to recover component {component_name}")
            component['recovery_attempts'] += 1
            
            # Log recovery attempt
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'component_recovery_attempt',
                    'component': component_name,
                    'attempt': component['recovery_attempts'],
                    'timestamp': datetime.now().isoformat()
                })
                
            # Attempt recovery
            success = component['recovery']()
            
            # Check if recovery was successful
            if success:
                logger.info(f"Recovery successful for component {component_name}")
                # Verify with a health check
                if self.check_component(component_name):
                    if self.logging_manager:
                        self.logging_manager.log_pattern_match({
                            'type': 'component_recovery_success',
                            'component': component_name,
                            'timestamp': datetime.now().isoformat()
                        })
                    return True
                else:
                    logger.warning(f"Recovery reported success but health check failed for {component_name}")
            
            logger.error(f"Recovery failed for component {component_name}")
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'component_recovery_failed',
                    'component': component_name,
                    'timestamp': datetime.now().isoformat()
                })
            return False
            
        except Exception as e:
            logger.error(f"Error during recovery for component {component_name}: {e}")
            if self.logging_manager:
                self.logging_manager.log_error(str(e), {
                    'context': 'component_recovery',
                    'component': component_name
                })
            return False
    
    def check_all_components(self) -> Dict[str, bool]:
        """
        Check health of all registered components.
        
        Returns:
            Dict[str, bool]: Component health status
        """
        results = {}
        for component_name in self.components:
            results[component_name] = self.check_component(component_name)
        return results
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get detailed health report for all components.
        
        Returns:
            Dict[str, Any]: Health report
        """
        report = {}
        for name, component in self.components.items():
            report[name] = {
                'healthy': component['healthy'],
                'last_check': component['last_check'].isoformat() if component['last_check'] else None,
                'recovery_attempts': component['recovery_attempts']
            }
        return report
