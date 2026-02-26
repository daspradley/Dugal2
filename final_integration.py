"""
Final integration module for Dugal Inventory System.
Coordinates all component interactions and manages the main application lifecycle.
"""

import logging
import sys
import os
import subprocess
import json
import shutil
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import azure.cognitiveservices.speech as speechsdk
from config_manager import DugalConfig, init_config
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow
)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon
from error_recovery import safe_operation, with_recovery, ComponentHealthMonitor
from onedrive_handler import OneDriveHandler
from excel_handler import ExcelHandler
from voice_interaction import VoiceInteraction
from gui_module import EnhancedDugalGUI as DugalGUI
from main_dugal import MainDugal
from search_engine import AdaptiveInventorySearchEngine
from dictionary_manager import DictionaryManager
from data_manager import DataManager
from logging_manager import LoggingManager
from global_registry import GlobalRegistry

# Try to import component manager (with fallback to ensure backward compatibility)
try:
    from component_manager import component_manager
    logger = logging.getLogger(__name__)
    logger.debug("Component manager imported successfully")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.debug("Component manager not available, using legacy component management")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_global_search_engine():
    """Create the global search engine instance and register it in the global registry."""
    logger.debug("Creating global search engine instance")
    
    # Try to use component manager first
    try:
        from component_manager import component_manager
        
        # Check if search engine already exists in component manager
        search_engine = component_manager.get_search_engine()
        
        if not search_engine:
            # Create a new search engine and register it
            search_engine = AdaptiveInventorySearchEngine()
            component_manager.register_component('search_engine', search_engine)
            logger.debug(f"Created and registered new search engine via component manager (ID: {id(search_engine)})")
        else:
            logger.debug(f"Using existing search engine from component manager (ID: {id(search_engine)})")
            
        return search_engine
        
    except ImportError:
        # Fall back to legacy method if component manager is not available
        logger.debug("Component manager not available, using legacy method")
        search_engine = AdaptiveInventorySearchEngine()
        GlobalRegistry.register('search_engine', search_engine)
        logger.debug(f"Created and registered new search engine via global registry (ID: {id(search_engine)})")
        return search_engine

# Create the global instance if needed
try:
    # Try to get from component manager first
    from component_manager import component_manager
    search_engine = component_manager.get_search_engine()
    if not search_engine:
        search_engine = create_global_search_engine()
except ImportError:
    # Fall back to legacy method
    search_engine = create_global_search_engine()

def synchronize_search_engine_references_global(components=None):
    """Ensure all components use the same search engine reference (utility version)."""
    try:
        logger.debug("=== SYNCHRONIZING SEARCH ENGINE REFERENCES (GLOBAL) ===")
        
        # Try to use component manager first
        try:
            from component_manager import component_manager
            
            # Get search engine from component manager
            search_engine = component_manager.get_search_engine()
            
            if search_engine:
                logger.debug(f"Using search engine from component manager (ID: {id(search_engine)})")
                
                # If components were provided, update them
                if components:
                    for name, component in components.items():
                        if hasattr(component, 'search_engine'):
                            component.search_engine = search_engine
                            logger.debug(f"Updated {name} search engine reference via component manager")
                
                logger.debug("=== SEARCH ENGINE SYNCHRONIZATION COMPLETE (via component manager) ===")
                return True
                
        except ImportError:
            logger.debug("Component manager not available, falling back to legacy method")
        
        # Fall back to legacy method if component manager is not available
        from global_registry import GlobalRegistry
        
        # Get reference from registry
        registry_engine = GlobalRegistry.get('search_engine')
        if not registry_engine:
            logger.error("No search engine in global registry!")
            return False
            
        # If components were provided, update them
        if components:
            for name, component in components.items():
                if hasattr(component, 'search_engine'):
                    component.search_engine = registry_engine
                    logger.debug(f"Updated {name} search engine reference via global registry")
        
        logger.debug("=== SEARCH ENGINE SYNCHRONIZATION COMPLETE (via legacy method) ===")
        return True
        
    except Exception as e:
        logger.error(f"Error synchronizing search engine references: {e}")
        return False

def cleanup_json_files(force_reset_critical=True):
    """Check for and repair corrupted JSON files before application startup."""
    logger.debug("Checking for corrupted JSON files...")
    
    # Define the directory and files to check
    json_dir = '.dugal_data'
    critical_files = ['onedrive_state.json', 'settings.json', 'user_preferences.json']
    
    # Create directory if needed
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
        logger.debug(f"Created data directory: {json_dir}")
    
    # Check for pattern_matches.json in logs directory
    logs_dir = os.path.join(json_dir, 'logs')
    if os.path.exists(logs_dir):
        pattern_matches_file = os.path.join(logs_dir, 'pattern_matches.json')
        if os.path.exists(pattern_matches_file):
            try:
                # Backup the file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f"{pattern_matches_file}.backup_{timestamp}"
                shutil.copy2(pattern_matches_file, backup_file)
                logger.debug(f"Backed up pattern_matches.json to {backup_file}")
                
                # Reset the file to a valid empty array
                with open(pattern_matches_file, 'w') as f:
                    f.write('[]')
                logger.debug(f"Reset pattern_matches.json to empty array")
            except Exception as e:
                logger.error(f"Error fixing pattern_matches.json: {e}")
    
    # Force reset critical files if requested
    if force_reset_critical:
        for json_file in critical_files:
            file_path = os.path.join(json_dir, json_file)
            # Backup existing file first if it exists
            if os.path.exists(file_path):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = f"{file_path}.backup_{timestamp}"
                    shutil.copy2(file_path, backup_file)
                    logger.debug(f"Backed up critical file before reset: {backup_file}")
                except Exception as e:
                    logger.warning(f"Could not backup {file_path} before reset: {e}")
            
            # Write empty JSON
            with open(file_path, 'w') as f:
                f.write('{}')
            logger.debug(f"Force reset critical JSON file: {file_path}")
    
    # Check any other JSON files in the dugal_data directory
    for filename in os.listdir(json_dir):
        if filename.endswith('.json') and filename not in critical_files:
            file_path = os.path.join(json_dir, filename)
            try:
                # Try to parse the file
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    json.loads(content)
            except json.JSONDecodeError:
                # Backup and reset
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = f"{file_path}.backup_{timestamp}"
                    shutil.copy2(file_path, backup_file)
                    with open(file_path, 'w') as f:
                        f.write('{}')
                    logger.debug(f"Reset corrupted JSON file: {file_path}")
                except Exception as e:
                    logger.error(f"Error fixing corrupted file {file_path}: {e}")
    
    # Check for any JSON files in the current directory
    for filename in os.listdir('.'):
        if filename.endswith('.json'):
            try:
                with open(filename, 'r') as f:
                    content = f.read().strip()
                    json.loads(content)
            except json.JSONDecodeError:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = f"{filename}.backup_{timestamp}"
                    shutil.copy2(filename, backup_file)
                    with open(filename, 'w') as f:
                        f.write('{}')
                    logger.debug(f"Reset corrupted JSON file in root: {filename}")
                except Exception as e:
                    logger.error(f"Error fixing file in root {filename}: {e}")
    
    return True

def process_json_file(file_path):
    """Process a single JSON file - validate and fix if needed."""
    # If file doesn't exist, create an empty JSON file
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('{}')  # Write an empty JSON object
        logger.debug(f"Created new empty JSON file: {file_path}")
        return
        
    # Check if file exists and attempt to repair if corrupted
    try:
        # Try to open and parse the JSON file
        with open(file_path, 'r') as f:
            content = f.read().strip()
            # Skip empty files
            if not content:
                with open(file_path, 'w') as f2:
                    f2.write('{}')
                logger.debug(f"Fixed empty JSON file: {file_path}")
                return
            
            # Try to parse JSON
            json.loads(content)
            logger.debug(f"JSON file validated: {file_path}")
                
    except json.JSONDecodeError:
        # JSON is corrupted, make a backup and create a new empty file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{file_path}.backup_{timestamp}"
        
        try:
            # Create backup of corrupted file
            shutil.copy2(file_path, backup_file)
            logger.warning(f"Corrupted JSON file backed up to: {backup_file}")
            
            # Replace with empty JSON object
            with open(file_path, 'w') as f:
                f.write('{}')
            logger.debug(f"Replaced corrupted JSON file with empty object: {file_path}")
            
        except Exception as e:
            logger.error(f"Error backing up corrupted file {file_path}: {e}")
            
            # If backup fails, try to directly overwrite the file
            try:
                with open(file_path, 'w') as f:
                    f.write('{}')
                logger.warning(f"Overwrote corrupted JSON file without backup: {file_path}")
            except Exception as write_error:
                logger.critical(f"Failed to fix corrupted JSON file {file_path}: {write_error}")
    except Exception as e:
        logger.error(f"Unexpected error processing JSON file {file_path}: {e}")
        # Try to reset the file as a last resort
        try:
            with open(file_path, 'w') as f:
                f.write('{}')
            logger.warning(f"Reset JSON file after unexpected error: {file_path}")
        except:
            pass
@dataclass
class IntegrationState:
    """Tracks the state of the integration system."""
    dugal: Optional[MainDugal] = None
    gui: Optional[DugalGUI] = None
    logging_manager: Optional['LoggingManager'] = None
    voice_interaction: Optional[VoiceInteraction] = None
    one_drive_handler: Optional[OneDriveHandler] = None
    excel_handler: Optional[ExcelHandler] = None
    file_coordinator: Optional['FileCoordinator'] = None
    dictionary_manager: Optional[DictionaryManager] = None
    search_engine: Optional[AdaptiveInventorySearchEngine] = None
    data_manager: Optional[DataManager] = None
    initialized_components: Dict[str, bool] = field(default_factory=lambda: {
        'onedrive': False,
        'excel': False,
        'voice': False,
        'gui': False,
        'coordinator': False,
        'dictionary': False,
        'search': False,
        'data': False
    })

@dataclass
class IntegrationMonitor:
    """Monitors overall system performance and health."""
    component_health: Dict[str, bool] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)

@dataclass
class CoordinationState:
    """Tracks coordination state between OneDrive and Excel."""
    file_path: Optional[str] = None
    onedrive_synced: bool = False
    excel_loaded: bool = False
    last_refresh: Optional[datetime] = None
    refresh_count: int = 0
    sync_errors: int = 0

# This should be defined outside of any class, as a standalone function
def check_and_close_excel_instances():
    """Check for open Excel instances and check file locks."""
    logger.debug("Checking for open Excel instances...")
    
    try:
        import win32com.client
        import pythoncom
        import psutil
        import win32file
        import pywintypes
        import os
        
        # Ensure we're in the right thread for COM
        pythoncom.CoInitialize()
        
        # Check if Excel is running
        excel_running = False
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] and 'excel' in proc.info['name'].lower():
                excel_running = True
                break
        
        # Also check common file paths for locks
        file_locked = False
        # Use more dynamic paths that will work across different systems
        common_paths = [
            os.path.join(os.path.expanduser("~"), "Desktop", "Bar Master Inventory 2024.xlsx"),
            os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "Bar Master Inventory 2024.xlsx")
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                try:
                    # Try to open the file exclusively
                    handle = win32file.CreateFile(
                        path,
                        win32file.GENERIC_READ,
                        0,  # No sharing
                        None,
                        win32file.OPEN_EXISTING,
                        win32file.FILE_ATTRIBUTE_NORMAL,
                        None
                    )
                    win32file.CloseHandle(handle)
                except pywintypes.error:
                    file_locked = True
                    logger.debug(f"File {path} is locked by another process")
        
        if excel_running or file_locked:
            # Create a dialog asking for permission
            from PyQt5.QtWidgets import QMessageBox
            message = 'Excel is currently running' if excel_running else ''
            if file_locked:
                if message:
                    message += ' and inventory files are locked'
                else:
                    message = 'Inventory files are locked by another process'
            
            message += '.\nWould you like Dugal to close Excel and release locks before continuing?'
            
            reply = QMessageBox.question(
                None, 
                'Excel/File Locks Detected', 
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                try:
                    # Get Excel application
                    excel = win32com.client.Dispatch("Excel.Application")
                    
                    # Save and close all workbooks
                    for workbook in excel.Workbooks:
                        try:
                            workbook.Save()
                            logger.debug(f"Saved workbook: {workbook.Name}")
                        except:
                            logger.warning(f"Could not save workbook: {workbook.Name}")
                            
                        workbook.Close(True)  # True to save changes
                    
                    # Quit Excel
                    excel.Quit()
                    logger.debug("Excel application closed successfully")
                    
                    # Clean up
                    excel = None
                    
                    # Wait a moment for locks to release
                    import time
                    time.sleep(1)
                    
                    return True
                except Exception as e:
                    logger.error(f"Error closing Excel: {e}")
                    QMessageBox.warning(
                        None,
                        "Error",
                        f"Could not close Excel: {str(e)}\n"
                        "Please close Excel manually before proceeding."
                    )
                    return False
        else:
            logger.debug("No Excel instances or file locks detected")
            return True
            
    except Exception as e:
        logger.error(f"Error checking Excel instances: {e}")
        return False
    finally:
        try:
            pythoncom.CoUninitialize()
        except:
            pass
class FileCoordinator:
    """Coordinates file operations between OneDrive and Excel."""
    
    def __init__(self, excel_handler, onedrive_handler, logging_manager=None, dictionary_manager=None):
        self.excel_handler = excel_handler
        self.onedrive_handler = onedrive_handler
        self.logging_manager = logging_manager
        self.dictionary_manager = dictionary_manager
        self.state = CoordinationState()
        
        # Use the configuration value instead of hardcoded
        self.REFRESH_INTERVAL = DugalConfig.REFRESH_INTERVAL  # Replace the hardcoded 2
        
        # Add the health monitor
        self.health_monitor = ComponentHealthMonitor(logging_manager=logging_manager)
        
        # Register health checks for components
        self.health_monitor.register_component(
            "excel",
            lambda: self.excel_handler is not None and hasattr(self.excel_handler, 'state'),
            self._recover_excel_handler
        )
        
        self.health_monitor.register_component(
            "onedrive",
            lambda: self.onedrive_handler is not None,
            self._recover_onedrive_handler
        )
    
    # Add these recovery methods
    def _recover_excel_handler(self):
        """Attempt to recover Excel handler if it's not functioning."""
        logger.debug("Attempting to recover Excel handler")
        if self.excel_handler:
            try:
                # Try to reset Excel handler state
                self.excel_handler.cleanup()
                # Reinitialize if needed
                if hasattr(self, 'dugal') and self.dugal:
                    self.excel_handler = ExcelHandler(dugal=self.dugal)
                return True
            except Exception as e:
                logger.error(f"Failed to recover Excel handler: {e}")
        return False
    
    def _recover_onedrive_handler(self):
        """Attempt to recover OneDrive handler if it's not functioning."""
        logger.debug("Attempting to recover OneDrive handler")
        if self.onedrive_handler:
            try:
                # Try to reset OneDrive handler state
                self.onedrive_handler.cleanup()
                # Reinitialize if needed
                if hasattr(self, 'dugal') and self.dugal:
                    self.onedrive_handler = OneDriveHandler(dugal=self.dugal)
                return True
            except Exception as e:
                logger.error(f"Failed to recover OneDrive handler: {e}")
        return False

    def add_dictionary_manager_button(self, parent_widget):
        """Add the dictionary manager button to the file selection dialog."""
        if not hasattr(self, 'dictionary_button'):
            from PyQt5.QtWidgets import QPushButton
            self.dictionary_button = QPushButton("Dictionary Manager")
            self.dictionary_button.setStyleSheet("""
                QPushButton {
                    background-color: #4a6fa5;
                    color: white;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #5a80b6;
                }
            """)
            self.dictionary_button.clicked.connect(self.show_dictionary_manager)
            
            # Add to parent widget layout
            if hasattr(parent_widget, 'layout'):
                parent_widget.layout().addWidget(self.dictionary_button)

    def open_file(self, filepath: str) -> bool:
        """Coordinate opening file between OneDrive and Excel."""
        logger.debug("Coordinating file open: %s", filepath)
        
        try:
            # Set filepath in OneDrive handler explicitly
            self.onedrive_handler.state.local_file_path = filepath
            logger.debug(f"Set initial file path in OneDrive handler: {self.onedrive_handler.state.local_file_path}")
            
            # First ensure OneDrive sync and access with retries
            if not self.onedrive_handler.ensure_file_available_with_retry(filepath):
                # Show a message to the user if we couldn't get the file
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    None, 
                    "File Locked",
                    f"The file {filepath} is currently locked by another application.\n\n"
                    "Please close any applications that might be using this file and try again."
                )
                logger.error("Failed to access file - it may be locked by another application")
                return False
            
            # Verify filepath is still set after ensure_file_available_with_retry
            logger.debug(f"File path after ensure: {self.onedrive_handler.state.local_file_path}")
            
            # Re-set filepath just to be sure
            self.onedrive_handler.state.local_file_path = filepath
            
            # Lock the file in OneDrive
            if not self.onedrive_handler.lock_file():
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    None, 
                    "Lock Failed",
                    f"Could not obtain exclusive access to the file.\n\n"
                    "Another application may be using this file."
                )
                logger.error("Failed to lock file in OneDrive")
                return False
            
            logger.debug(f"File successfully locked: {self.onedrive_handler.state.local_file_path}")
            
            # Now open in Excel as read-only
            if not self.excel_handler.load_excel_file(filepath, read_only=True):
                logger.error("Failed to load file in Excel")
                self.onedrive_handler.unlock_file()  # Release lock if Excel fails
                return False
            
            logger.debug("=== PROCESSING INVENTORY TERMS FOR DICTIONARY ===")
            
            # After the file is successfully loaded - ALWAYS process inventory terms
            try:
                if hasattr(self, 'dictionary_manager') and self.dictionary_manager:
                    logger.debug("Dictionary manager available - processing inventory terms")
                    if hasattr(self.excel_handler, 'search_engine') and self.excel_handler.search_engine:
                        logger.debug(f"Search engine available with {len(self.excel_handler.search_engine.inventory_cache) if hasattr(self.excel_handler.search_engine, 'inventory_cache') else 0} items")
                        variations_added = self.dictionary_manager.process_inventory_terms(
                            workbook=self.excel_handler.state.workbook,
                            search_engine=self.excel_handler.search_engine
                        )
                        logger.info(f"✅ Added {variations_added} phonetic variations to search engine")
                    else:
                        logger.warning("Search engine not available for dictionary processing")
                else:
                    logger.warning("Dictionary manager not available")
            except Exception as e:
                logger.error(f"Error processing inventory terms: {e}")
            
            logger.debug("=== DICTIONARY PROCESSING COMPLETE ===")
                
            # After the file is successfully loaded
            if hasattr(self.excel_handler, 'search_engine') and hasattr(self, 'dictionary_manager'):
                logger.debug("Generating phonetic variations for inventory terms")
                variations_added = self.dictionary_manager.process_inventory_terms(
                    workbook=self.excel_handler.state.workbook,
                    search_engine=self.excel_handler.search_engine
                )
                logger.debug(f"Added {variations_added} phonetic variations to search engine")
                
            # Update state
            self.state.file_path = filepath
            self.state.onedrive_synced = True
            self.state.excel_loaded = True
            self.state.last_refresh = datetime.now()
            
            # Ensure OneDrive handler state is consistent
            self.onedrive_handler.state.local_file_path = filepath
            
            # Start periodic refresh
            self._start_refresh_timer()
            
            # Update voice interaction connection after file load
            if hasattr(self.excel_handler, 'dugal') and hasattr(self.excel_handler.dugal, 'voice_interaction'):
                voice = self.excel_handler.dugal.voice_interaction
                logger.debug("Refreshing voice-to-inventory connection after file load")
                voice.connect_voice_to_inventory(self.onedrive_handler, self.excel_handler)
                logger.debug("Voice connection refreshed")
                
                # NEW: Connect voice system to file via SyncManager
                logger.debug("Connecting voice system to file via SyncManager")
                try:
                    voice_result = voice.connect_to_file(filepath)
                    if voice_result['success']:
                        logger.info(f"✅ Voice system connected to file via SyncManager")
                        logger.info(f"   Temp file: {voice_result.get('temp_path')}")
                    else:
                        logger.warning(f"⚠️ Voice SyncManager connection failed: {voice_result.get('message')}")
                except Exception as e:
                    logger.error(f"Error connecting voice to file via SyncManager: {e}")
            
            logger.debug("File successfully opened and coordinated")
            logger.debug(f"Final check - OneDrive handler file path: {self.onedrive_handler.state.local_file_path}")
            
            # Ensure all components have the latest data - IMPORTANT: This must come BEFORE the return statement
            self.refresh_components_after_file_load()
            self.synchronize_search_engine_references()
            
            return True
                
        except Exception as e:
            logger.error("Coordination error: %s", str(e))
            self._cleanup_on_error()
            return False
        
        # This line will never be reached due to the return statement above
        # self.synchronize_search_engine_references()
        
    def get_search_engine(self):
        """Get the search engine from the global registry or local reference."""
        try:
            # First try global registry
            from global_registry import GlobalRegistry
            logger.debug("FileCoordinator requesting search engine from registry")
            search_engine = GlobalRegistry.get('search_engine')
            
            # If found in registry, update our local reference
            if search_engine:
                if hasattr(self, 'search_engine') and self.search_engine is not search_engine:
                    logger.debug(f"Updating FileCoordinator search engine reference (old ID: {id(self.search_engine)}, new ID: {id(search_engine)})")
                    self.search_engine = search_engine
                elif not hasattr(self, 'search_engine'):
                    logger.debug(f"Setting initial FileCoordinator search engine reference (ID: {id(search_engine)})")
                    self.search_engine = search_engine
                return search_engine
                
            # If not in registry but we have one, register it
            if hasattr(self, 'search_engine') and self.search_engine:
                logger.debug(f"Registering FileCoordinator's search engine in global registry (ID: {id(self.search_engine)})")
                GlobalRegistry.register('search_engine', self.search_engine)
                return self.search_engine
                
            # Last resort: if excel_handler has one
            if hasattr(self, 'excel_handler') and hasattr(self.excel_handler, 'search_engine'):
                search_engine = self.excel_handler.search_engine
                logger.debug(f"Using ExcelHandler's search engine as fallback (ID: {id(search_engine)})")
                GlobalRegistry.register('search_engine', search_engine)
                self.search_engine = search_engine  # Also set local reference
                return search_engine
                
            logger.warning("No search engine available in any location")
            return None
            
        except Exception as e:
            logger.error(f"Error in FileCoordinator.get_search_engine: {e}")
            # Fallback to local reference if available
            local_engine = getattr(self, 'search_engine', None)
            if local_engine:
                logger.debug(f"Falling back to local search engine reference (ID: {id(local_engine)})")
                return local_engine
            return None

    def refresh_components_after_file_load(self):
        """Ensure all components have access to the latest data."""
        try:
            logger.debug("=== REFRESHING COMPONENTS AFTER FILE LOAD ===")
            
            # First, synchronize search engine references across components
            self.synchronize_search_engine_references()
            
            if self.excel_handler and hasattr(self.excel_handler, 'search_engine'):
                # Get search engine from registry as the source of truth
                from global_registry import GlobalRegistry
                registry_engine = GlobalRegistry.get('search_engine')
                
                # If registry has a search engine, ensure we're using it
                if registry_engine:
                    if id(self.excel_handler.search_engine) != id(registry_engine):
                        logger.debug(f"Updating excel_handler's search engine to match registry")
                        self.excel_handler.search_engine = registry_engine
                    search_engine = registry_engine
                else:
                    # If no registry engine, register excel_handler's
                    search_engine = self.excel_handler.search_engine
                    GlobalRegistry.register('search_engine', search_engine)
                
                # Log the identity of the search engine
                logger.debug(f"Excel handler search engine ID: {id(search_engine)}")
                
                # Connect to Dugal's main object explicitly
                if hasattr(self.excel_handler.state, 'dugal'):
                    dugal = self.excel_handler.state.dugal
                    logger.debug(f"Connecting search engine to Dugal main object")
                    dugal.search_engine = search_engine
                
                # Check if search engine has data
                if not hasattr(search_engine, 'inventory_cache') or len(search_engine.inventory_cache) == 0:
                    logger.debug("Search engine has no inventory data, rebuilding index")
                    
                    # Get workbook and search column if available
                    if hasattr(self.excel_handler, 'state'):
                        workbook = getattr(self.excel_handler.state, 'workbook', None)
                        search_column = getattr(self.excel_handler.state, 'search_column', 'A')
                        
                        if workbook:
                            # Rebuild search index
                            search_engine.build_search_index(workbook, search_column)
                            
                            # Set input column if available
                            if hasattr(self.excel_handler.state, 'input_column'):
                                # Find column index for input column
                                for sheet_name in self.excel_handler.state.selected_sheets:
                                    sheet = workbook[sheet_name]
                                    for idx, cell in enumerate(sheet[1], start=1):
                                        if cell.value and str(cell.value).strip() == self.excel_handler.state.input_column:
                                            search_engine.input_column_index = idx
                                            search_engine.input_column_name = self.excel_handler.state.input_column
                                            logger.debug(f"Found input column '{self.excel_handler.state.input_column}' at index {idx}")
                                            break
                            
                            logger.debug(f"Rebuilt search index, now has {len(search_engine.inventory_cache) if hasattr(search_engine, 'inventory_cache') else 0} items")
                            
                            # Re-register in registry after rebuilding
                            GlobalRegistry.register('search_engine', search_engine)
                
                # Connect voice interaction to search engine
                if hasattr(self.excel_handler.state, 'dugal') and hasattr(self.excel_handler.state.dugal, 'voice_interaction'):
                    voice = self.excel_handler.state.dugal.voice_interaction
                    
                    # Log the identity before assignment
                    if hasattr(voice.state, 'search_engine'):
                        logger.debug(f"Voice search engine ID before: {id(voice.state.search_engine)}")
                    
                    # Enhanced: Use dedicated connect method if available
                    if hasattr(voice, 'connect_to_search_engine'):
                        voice.connect_to_search_engine(search_engine)
                        logger.debug("Used connect_to_search_engine method for direct connection")
                    else:
                        # Store direct reference to search engine in voice interaction
                        voice.state.search_engine = search_engine
                        logger.debug(f"Voice search engine ID after: {id(voice.state.search_engine)}")
                    
                    logger.debug(f"Connected voice interaction to search engine with {len(search_engine.inventory_cache) if hasattr(search_engine, 'inventory_cache') else 0} items")
                    
                    # Run voice diagnostics
                    if hasattr(voice, 'diagnose_search_engine'):
                        logger.debug("Running voice interaction diagnostics")
                        voice.diagnose_search_engine()
                    
                    # Diagnose search engine state
                    if hasattr(search_engine, 'diagnose_search_index'):
                        search_engine.diagnose_search_index()
                        
                    # Run a test search if there are items in the cache
                    if hasattr(search_engine, 'inventory_cache') and len(search_engine.inventory_cache) > 0:
                        test_item = next(iter(search_engine.inventory_cache.keys()))
                        logger.debug(f"Running test search for '{test_item}'")
                        try:
                            # Use find_item instead of search
                            result = search_engine.find_item(test_item)
                            logger.debug(f"Test search result: {bool(result.get('found', False)) if result else False}")
                        except Exception as e:
                            logger.error(f"Test search failed: {e}")
            
            # Verify all references are consistent after refresh
            self.verify_search_engine_references()
                
            logger.debug("=== COMPONENT REFRESH COMPLETE ===")
        
        # Enhanced error handling
        except AttributeError as attr_err:
            # Specific handling for attribute errors (common in reference issues)
            logger.error(f"Attribute error in component refresh: {attr_err}")
            logger.error(f"This may indicate missing component references")
            # Attempt recovery for common issues
            if hasattr(self, '_attempt_reference_recovery'):
                self._attempt_reference_recovery()
        except Exception as e:
            logger.error(f"Error in refresh_components_after_file_load: {e}")

        # Final verification check
        if (hasattr(self, 'excel_handler') and
            hasattr(self.excel_handler.state, 'dugal') and 
            hasattr(self.excel_handler.state.dugal, 'voice_interaction') and
            hasattr(self.excel_handler, 'search_engine')):
            
            voice = self.excel_handler.state.dugal.voice_interaction
            if hasattr(voice.state, 'search_engine'):
                excel_engine_id = id(self.excel_handler.search_engine)
                voice_engine_id = id(voice.state.search_engine)
                if excel_engine_id == voice_engine_id:
                    logger.debug(f"✓ Voice and Excel using same search engine instance: {excel_engine_id}")
                else:
                    logger.warning(f"✗ Reference mismatch! Excel: {excel_engine_id}, Voice: {voice_engine_id}")
                    
                    # Try to fix the mismatch
                    from global_registry import GlobalRegistry
                    registry_engine = GlobalRegistry.get('search_engine')
                    if registry_engine:
                        voice.state.search_engine = registry_engine
                        self.excel_handler.search_engine = registry_engine
                        logger.debug(f"Fixed reference mismatch using registry engine")
                        
    def _attempt_reference_recovery(self):
        """Attempt to recover from reference errors."""
        try:
            logger.debug("Attempting reference recovery")
            # Check if search engine is available
            if not hasattr(self.excel_handler, 'search_engine') or not self.excel_handler.search_engine:
                logger.error("Excel handler missing search engine reference")
                return False
                
            # Get the search engine reference
            search_engine = self.excel_handler.search_engine
            
            # Ensure Dugal has the reference
            if hasattr(self.excel_handler.state, 'dugal'):
                self.excel_handler.state.dugal.search_engine = search_engine
                logger.debug("Restored search engine reference to Dugal")
                
            # Ensure voice interaction has the reference
            if (hasattr(self.excel_handler.state, 'dugal') and 
                hasattr(self.excel_handler.state.dugal, 'voice_interaction')):
                voice = self.excel_handler.state.dugal.voice_interaction
                voice.state.search_engine = search_engine
                logger.debug("Restored search engine reference to voice interaction")
                
            return True
        except Exception as e:
            logger.error(f"Reference recovery failed: {e}")
            return False

    def synchronize_search_engine_references(self):
        """Ensure all components use the same search engine reference."""
        try:
            from global_registry import GlobalRegistry
            logger.debug("=== SYNCHRONIZING SEARCH ENGINE REFERENCES ===")
            
            # Check if registry has a search engine
            registry_engine = GlobalRegistry.get('search_engine')
            registry_id = id(registry_engine) if registry_engine else None
            logger.debug(f"Registry search engine ID: {registry_id}")
            
            # Determine the best search engine to use
            chosen_engine = None
            chosen_source = None
            
            # Option 1: Use registry engine if it has data
            if registry_engine and hasattr(registry_engine, 'inventory_cache') and len(registry_engine.inventory_cache) > 0:
                chosen_engine = registry_engine
                chosen_source = "registry"
                logger.debug(f"Using registry search engine with {len(registry_engine.inventory_cache)} items")
            
            # Option 2: Check excel handler
            elif hasattr(self, 'excel_handler') and hasattr(self.excel_handler, 'search_engine'):
                excel_engine = self.excel_handler.search_engine
                if excel_engine and hasattr(excel_engine, 'inventory_cache') and len(excel_engine.inventory_cache) > 0:
                    chosen_engine = excel_engine
                    chosen_source = "excel_handler"
                    logger.debug(f"Using excel_handler search engine with {len(excel_engine.inventory_cache)} items")
            
            # Option 3: If no engine with data is found, use registry engine anyway
            if not chosen_engine and registry_engine:
                chosen_engine = registry_engine
                chosen_source = "empty_registry"
                logger.debug("Using registry search engine (no data found)")
                
            # Option 4: Create new engine if none found
            if not chosen_engine:
                logger.debug("No search engine found, creating new one")
                from search_engine import AdaptiveInventorySearchEngine
                chosen_engine = AdaptiveInventorySearchEngine()
                chosen_source = "new"
            
            # Register the chosen engine in the registry
            GlobalRegistry.register('search_engine', chosen_engine)
            logger.debug(f"Registered search engine from {chosen_source} in global registry")
            
            # Update all component references
            if hasattr(self, 'excel_handler') and self.excel_handler:
                self.excel_handler.search_engine = chosen_engine
                logger.debug("Updated excel_handler search engine reference")
                
            # Update Dugal's search engine if available
            if hasattr(self.excel_handler, 'dugal') and self.excel_handler.dugal:
                self.excel_handler.dugal.search_engine = chosen_engine
                logger.debug("Updated dugal search engine reference")
                
            # Update voice interaction's search engine if available
            if (hasattr(self.excel_handler, 'dugal') and 
                hasattr(self.excel_handler.dugal, 'voice_interaction') and 
                hasattr(self.excel_handler.dugal.voice_interaction, 'state')):
                self.excel_handler.dugal.voice_interaction.state.search_engine = chosen_engine
                logger.debug("Updated voice_interaction search engine reference")
                    
            logger.debug("=== SEARCH ENGINE SYNCHRONIZATION COMPLETE ===")
            return True
        except Exception as e:
            logger.error(f"Error synchronizing search engine references: {e}")
            return False

    def verify_search_engine_references(self) -> bool:
        """Verify that all components have the same search engine reference."""
        try:
            logger.debug("=== VERIFYING SEARCH ENGINE REFERENCES ===")
            
            # Get search engine references
            from global_registry import GlobalRegistry
            registry_engine = GlobalRegistry.get('search_engine')
            registry_id = id(registry_engine) if registry_engine else None
            
            excel_id = id(self.excel_handler.search_engine) if hasattr(self.excel_handler, 'search_engine') else None
            
            voice_id = None
            if (hasattr(self.excel_handler, 'dugal') and 
                hasattr(self.excel_handler.dugal, 'voice_interaction') and 
                hasattr(self.excel_handler.dugal.voice_interaction, 'state') and 
                hasattr(self.excel_handler.dugal.voice_interaction.state, 'search_engine')):
                voice_id = id(self.excel_handler.dugal.voice_interaction.state.search_engine)
            
            dugal_id = id(self.excel_handler.dugal.search_engine) if hasattr(self.excel_handler, 'dugal') and hasattr(self.excel_handler.dugal, 'search_engine') else None
            
            # Log all IDs for diagnostic purposes
            logger.debug(f"Search engine IDs - Registry: {registry_id}, Excel: {excel_id}, Voice: {voice_id}, Dugal: {dugal_id}")
            
            # Check if all references match the registry
            excel_match = excel_id == registry_id if excel_id and registry_id else False
            voice_match = voice_id == registry_id if voice_id and registry_id else False
            dugal_match = dugal_id == registry_id if dugal_id and registry_id else False
            
            all_match = excel_match and voice_match and dugal_match
            
            if not all_match:
                logger.warning("✗ Search engine references do not match across components")
                mismatch_components = []
                if excel_id and not excel_match:
                    mismatch_components.append("excel_handler")
                if voice_id and not voice_match:
                    mismatch_components.append("voice_interaction")
                if dugal_id and not dugal_match:
                    mismatch_components.append("dugal")
                logger.warning(f"Mismatched components: {mismatch_components}")
            else:
                logger.debug("✓ All components using same search engine instance")
            
            logger.debug("=== VERIFICATION COMPLETE ===")
            return all_match
            
        except Exception as e:
            logger.error(f"Error verifying search engine references: {e}")
            return False

    def handle_sync_error(self, error_type: str, details: Dict[str, Any]) -> None:
        """Enhanced error handling for sync issues."""
        if self.logging_manager:
            self.logging_manager.log_error(
                f"Sync error: {error_type}",
                {
                    'context': 'file_coordination',
                    'details': details,
                    'sync_errors': self.state.sync_errors
                }
            )

    def _start_refresh_timer(self):
        """Start timer for periodic Excel view refresh."""
        try:
            if hasattr(self.excel_handler, 'control_panel') and self.excel_handler.control_panel:
                timer = self.excel_handler.control_panel.refresh_timer
                timer.timeout.connect(self.refresh_excel_view)
                timer.start(self.REFRESH_INTERVAL * 1000)  # Convert seconds to milliseconds
                logger.debug("Refresh timer started")
        except Exception as e:
            logger.error("Error starting refresh timer: %s", str(e))

    def refresh_excel_view(self) -> bool:
        """Refresh Excel view to show OneDrive updates."""
        if not self._can_refresh():
            return False
            
        try:
            # Verify OneDrive sync first
            if not self.onedrive_handler.refresh_file():
                logger.error("OneDrive refresh failed")
                self.state.sync_errors += 1
                # Try to recover OneDrive handler
                try:
                    self._recover_onedrive_handler()
                except Exception as recovery_error:
                    logger.error(f"Failed to recover OneDrive handler: {recovery_error}")
                return False
                
            # Now refresh Excel view
            if not self.excel_handler.refresh_view():
                logger.error("Excel refresh failed")
                # Try to recover Excel handler
                try:
                    self._recover_excel_handler()
                except Exception as recovery_error:
                    logger.error(f"Failed to recover Excel handler: {recovery_error}")
                return False
                
            self.state.refresh_count += 1
            self.state.last_refresh = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Refresh error: {e}")
            self.state.sync_errors += 1
            # Attempt comprehensive recovery
            try:
                self._attempt_reference_recovery()
            except Exception as recovery_error:
                logger.error(f"Failed to recover after refresh error: {recovery_error}")
            return False

    def _can_refresh(self) -> bool:
        """Check if refresh is allowed based on timing and state."""
        if not (self.state.onedrive_synced and self.state.excel_loaded):
            return False
            
        if not self.state.last_refresh:
            return True
            
        time_since_refresh = (datetime.now() - self.state.last_refresh).total_seconds()
        return time_since_refresh >= self.REFRESH_INTERVAL

    def _cleanup_on_error(self) -> None:
        """Clean up resources when an error occurs."""
        try:
            if self.onedrive_handler:
                self.onedrive_handler.unlock_file()
            if self.excel_handler:
                self.excel_handler.cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def cleanup(self) -> None:
        """Clean up resources before shutdown."""
        try:
            # Stop refresh timer
            if hasattr(self.excel_handler, 'control_panel'):
                if hasattr(self.excel_handler.control_panel, 'refresh_timer'):
                    self.excel_handler.control_panel.refresh_timer.stop()
            
            # Cleanup handlers
            if self.excel_handler:
                self.excel_handler.cleanup()
            if self.onedrive_handler:
                self.onedrive_handler.cleanup()
                
            logger.debug("File coordination cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during coordination cleanup: {e}")

    def generate_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive system diagnostic report."""
        report = {
            'component_status': self.state.initialized_components,
            'sync_status': self.state.file_coordinator.state if self.state.file_coordinator else None,
            'error_logs': self.state.logging_manager.analyze_errors() if self.state.logging_manager else None,
            'timestamp': datetime.now().isoformat()
        }
        return report

class FinalIntegration(QMainWindow):
    """
    Master class that integrates all components for running the Dugal application.
    Manages component lifecycle and inter-module communication.
    """
    # Signal definitions
    component_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)
    mode_changed = pyqtSignal(str)
    update_status = pyqtSignal(str)

    def __init__(self) -> None:
        """Initialize the final integration of Dugal's components."""
        super().__init__()
        self.state = IntegrationState()
        
        # Add the health monitor
        self.health_monitor = ComponentHealthMonitor()
        
        try:
            self.setup_components()
            
            # After components are set up, update the health monitor with logging manager
            if self.state.logging_manager:
                self.health_monitor.logging_manager = self.state.logging_manager
                
        except Exception as e:
            self.handle_fatal_error("Initialization Error", str(e))

        self.synchronize_search_engine_references()

    def setup_components(self) -> None:
        """Set up all components in the correct order with proper error handling."""
        try:
            # First initialize logging
            logger.debug("Initializing Logging Manager...")
            self.state.logging_manager = LoggingManager()
            # Register in global registry
            GlobalRegistry.register('logging_manager', self.state.logging_manager)
            self.state.initialized_components['logging'] = True

            # Initialize data manager
            logger.debug("Initializing Data Manager...")
            self.state.data_manager = DataManager()
            GlobalRegistry.register('data_manager', self.state.data_manager)
            self.state.initialized_components['data'] = True

            # Initialize search engine using component manager if available
            logger.debug("Initializing Search Engine...")
            try:
                from component_manager import component_manager
                
                # Get search engine from component manager
                search_engine = component_manager.get_search_engine()
                
                if not search_engine:
                    logger.debug("Component manager couldn't provide search engine, creating a new one")
                    search_engine = AdaptiveInventorySearchEngine()
                    component_manager.register_component('search_engine', search_engine)
                    
                logger.debug(f"Using search engine from component manager (ID: {id(search_engine)})")
                self.state.search_engine = search_engine
                
            except ImportError:
                # Fall back to legacy method if component manager is not available
                logger.debug("Component manager not available, using legacy method")
                global_search_engine = GlobalRegistry.get('search_engine')
                if global_search_engine:
                    logger.debug("Using existing global search engine instance")
                    self.state.search_engine = global_search_engine
                else:
                    logger.debug("Creating new search engine instance")
                    self.state.search_engine = AdaptiveInventorySearchEngine()
                    # Register search engine in global registry - critical for shared references
                    GlobalRegistry.register('search_engine', self.state.search_engine)

            # Load any saved patterns
            patterns = self.state.data_manager.load_patterns()
            if patterns and hasattr(self.state.search_engine, 'load_patterns'):
                self.state.search_engine.load_patterns(patterns)
            self.state.initialized_components['search'] = True

            # Initialize Main Dugal
            logger.debug("Initializing MainDugal...")
            self.state.dugal = MainDugal()
            self.state.dugal.logging_manager = self.state.logging_manager
            # Connect search engine to Dugal
            self.state.dugal.search_engine = GlobalRegistry.get('search_engine')
            GlobalRegistry.register('dugal', self.state.dugal)

            # Setup OneDrive Handler with robust error handling
            if not self._init_onedrive_handler():
                raise RuntimeError("Failed to initialize OneDrive component")
            # Register OneDrive handler in registry
            GlobalRegistry.register('onedrive_handler', self.state.one_drive_handler)

            # Setup Excel Handler w/ Search Engine
            logger.debug("Setting up ExcelHandler...")
            self.state.excel_handler = ExcelHandler(dugal=self.state.dugal)
            # Use the global search engine instance
            self.state.excel_handler.search_engine = GlobalRegistry.get('search_engine')
            GlobalRegistry.register('excel_handler', self.state.excel_handler)
            self.state.initialized_components['excel'] = True

            # Initialize Dictionary Manager
            logger.debug("Initializing Dictionary Manager...")
            self.state.dictionary_manager = DictionaryManager(
                search_engine=GlobalRegistry.get('search_engine'),
                dugal=self.state.dugal
            )
            GlobalRegistry.register('dictionary_manager', self.state.dictionary_manager)
            self.state.initialized_components['dictionary'] = True

            # Initialize Voice Interaction with Dugal reference
            logger.debug("Setting up voice components...")
            self.state.voice_interaction = VoiceInteraction(dugal=self.state.dugal)
            GlobalRegistry.register('voice_interaction', self.state.voice_interaction)
            self.state.initialized_components['voice'] = True
            self.state.dugal.voice_interaction = self.state.voice_interaction
            
            # Explicitly connect search engine to voice interaction's state
            if hasattr(self.state.voice_interaction, 'state'):
                self.state.voice_interaction.state.search_engine = GlobalRegistry.get('search_engine')
                logger.debug(f"Connected voice interaction to search engine (ID: {id(GlobalRegistry.get('search_engine'))})")
                
            # Register health checks
            self.health_monitor.register_component(
                "onedrive",
                lambda: hasattr(self.state, "one_drive_handler") and self.state.one_drive_handler is not None,
                self._recover_onedrive_handler
            )
            
            self.health_monitor.register_component(
                "voice",
                lambda: hasattr(self.state, "voice_interaction") and self.state.voice_interaction is not None,
                self._recover_voice_interaction
            )
            
            # Add search engine health check
            self.health_monitor.register_component(
                "search_engine",
                lambda: self.verify_search_engine_references(),
                self._fix_search_engine_references
            )

            # Setup File Coordinator
            logger.debug("Setting up FileCoordinator...")
            self.state.file_coordinator = FileCoordinator(
                self.state.excel_handler,
                self.state.one_drive_handler,
                logging_manager=self.state.logging_manager,
                dictionary_manager=self.state.dictionary_manager
            )
            GlobalRegistry.register('file_coordinator', self.state.file_coordinator)
            self.state.initialized_components['coordinator'] = True

            # Connect voice to inventory system
            self.state.voice_interaction.connect_voice_to_inventory(
                self.state.one_drive_handler,
                self.state.excel_handler
            )

            # Setup GUI
            logger.debug("Setting up GUI components...")
            self.state.gui = DugalGUI(
                voice_interaction=self.state.voice_interaction,
                one_drive_handler=self.state.one_drive_handler,
                excel_handler=self.state.excel_handler,
                file_coordinator=self.state.file_coordinator,
                logging_manager=self.state.logging_manager
            )
            GlobalRegistry.register('gui', self.state.gui)
            self.state.gui.dugal = self.state.dugal
            self.state.gui.file_coordinator = self.state.file_coordinator
            self.state.initialized_components['gui'] = True

            # Connect Signals
            logger.debug("Connecting signals...")
            self._connect_signals()
            self._connect_nlp_signals()
            
            # Verify system connectivity
            logger.debug("Verifying system connectivity...")
            self.verify_system_connectivity()
            
            logger.debug("All components initialized successfully")

        except Exception as e:
            failed_component = next(
                (comp for comp, init in self.state.initialized_components.items() 
                 if not init),
                "unknown"
            )
            logger.critical(
                "Critical setup error in %s component: %s", 
                failed_component, 
                str(e)
            )
            raise RuntimeError(
                f"Failed to initialize {failed_component} component: {str(e)}"
            ) from e

    def _verify_search_engine_references(self):
        """Verify all components are using the same search engine instance."""
        try:
            from global_registry import GlobalRegistry
            logger.debug("=== VERIFYING SEARCH ENGINE REFERENCES ===")
            
            # Get registry engine
            registry_engine = GlobalRegistry.get('search_engine')
            if not registry_engine:
                logger.error("No search engine in global registry!")
                return False
                
            # Collect ID references
            engine_ids = {
                'registry': id(registry_engine),
                'excel': id(self.excel_handler.search_engine) if hasattr(self.excel_handler, 'search_engine') else None,
                'voice': id(self.voice_interaction.state.search_engine) if hasattr(self.voice_interaction, 'state') and hasattr(self.voice_interaction.state, 'search_engine') else None,
                'dugal': id(self.dugal.search_engine) if hasattr(self.dugal, 'search_engine') else None
            }
            
            logger.debug(f"Search engine IDs - Registry: {engine_ids['registry']}, Excel: {engine_ids['excel']}, Voice: {engine_ids['voice']}, Dugal: {engine_ids['dugal']}")
            
            # Check for mismatches
            mismatches = []
            for component, engine_id in engine_ids.items():
                if engine_id is not None and engine_id != engine_ids['registry']:
                    mismatches.append(component)
                    
            if mismatches:
                logger.warning("✗ Search engine references do not match across components")
                logger.warning(f"Mismatched components: {mismatches}")
                return False
            else:
                logger.debug("✓ All components using same search engine instance")
                return True
                
        except Exception as e:
            logger.error(f"Error verifying search engine references: {e}")
            return False
        finally:
            logger.debug("=== VERIFICATION COMPLETE ===")

    def check_system_health(self):
        """Perform a comprehensive check of system health and connectivity."""
        logger.debug("Performing system health check...")
        
        # Check component connectivity
        connectivity_ok = self.verify_system_connectivity()
        
        # Check search engine data
        search_engine = GlobalRegistry.get('search_engine')
        if search_engine and hasattr(search_engine, 'diagnose_search_index'):
            search_engine.diagnose_search_index()
        
        # Check voice connection
        voice = GlobalRegistry.get('voice_interaction')
        if voice and hasattr(voice, 'diagnose_search_engine'):
            voice.diagnose_search_engine()
        
        # Diagnose global registry
        GlobalRegistry.diagnose()
        
        # If connectivity issues were found, refresh components
        if not connectivity_ok and hasattr(self.state, 'file_coordinator'):
            logger.debug("Attempting to fix connectivity issues...")
            self.state.file_coordinator.refresh_components_after_file_load()
        
        logger.debug("System health check complete")

    @with_recovery(max_retries=3, retry_delay=1.0)
    def _init_onedrive_handler(self):
        """Initialize OneDrive handler with robust error handling."""
        try:
            logger.debug("Setting up OneDriveHandler...")
            logger.debug("Creating OneDriveHandler instance")
            
            # First, check and fix the OneDrive state file specifically
            onedrive_state_file = os.path.join('.dugal_data', 'onedrive_state.json')
            try:
                # Don't try to validate, just overwrite
                with open(onedrive_state_file, 'w') as f:
                    f.write('{}')
                logger.debug(f"Reset OneDrive state file immediately before handler creation")
            except Exception as e:
                logger.error(f"Error resetting OneDrive state file: {e}")
            
            # Now create the handler with a try/except block to catch JSON errors
            try:
                # Create the handler once
                handler = OneDriveHandler(dugal=self.state.dugal)
                
                # Assign to both variable names for compatibility
                self.state.onedrive_handler = handler
                self.state.one_drive_handler = handler  # Same object, different variable name
            except json.JSONDecodeError as json_error:
                # If we get a JSON error during initialization, print debugging info
                logger.critical(f"JSON decode error initializing OneDriveHandler: {json_error}")
                
                # Try to find which file is causing the problem by checking all JSON files in directory
                for root, dirs, files in os.walk('.'):
                    for file in files:
                        if file.endswith('.json'):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r') as f:
                                    content = f.read()
                                    json.loads(content)
                            except json.JSONDecodeError as e:
                                logger.critical(f"Found corrupted JSON file: {file_path}, Error: {e}")
                                # Force overwrite this file
                                with open(file_path, 'w') as f:
                                    f.write('{}')
                                logger.debug(f"Force reset corrupted JSON file: {file_path}")
                
                # Try one more time after fixing all JSON files
                handler = OneDriveHandler(dugal=self.state.dugal)
                self.state.onedrive_handler = handler
                self.state.one_drive_handler = handler
                
            # Rest of your handler initialization code...
            logger.debug("Handler created, checking state")
            self.state.onedrive_handler.state.logging_manager = self.state.logging_manager
            self.state.onedrive_handler.state.data_manager = self.state.data_manager
            self.state.onedrive_handler.state.search_engine = self.state.search_engine
            self.state.onedrive_handler.state.excel_handler = self.state.excel_handler
            
            logger.debug("Handler assigned to state")
            self.state.initialized_components['onedrive'] = True
            logger.debug("OneDrive component marked as initialized")
            
            return True
        except Exception as e:
            error_msg = f"Failed to initialize onedrive component: {str(e)}"
            logger.critical(error_msg)
            if hasattr(self.state, 'logging_manager') and self.state.logging_manager:
                self.state.logging_manager.log_critical_error(error_msg, {
                    'context': 'component_initialization',
                    'component': 'onedrive'
                })
            return False

    def _connect_signals(self) -> None:
        """Connect inter-component signals."""
        try:
            # GUI signals
            if self.state.gui:
                self.component_ready.connect(self.state.gui.handle_component_ready)
                self.error_occurred.connect(self.state.gui.handle_error)
                self.mode_changed.connect(self.state.gui.handle_mode_change)

            # Connect Excel refresh signals
            if self.state.excel_handler and self.state.one_drive_handler:
                self.state.excel_handler.refresh_complete.connect(
                    self.state.one_drive_handler.refresh_file
                )

            # Connect voice processing signals
            if self.state.voice_interaction and self.state.one_drive_handler:
                self.state.voice_interaction.command_processed.connect(
                    self.handle_voice_command
                )

            logger.debug("Signals connected successfully")

        except Exception as e:
            logger.error(f"Error connecting signals: {e}")
            raise

    def _connect_nlp_signals(self) -> None:
        """Connect signals for NLP and dictionary features."""
        try:
            # Connect search engine signals
            if self.state.search_engine and self.state.excel_handler:
                self.state.excel_handler.refresh_complete.connect(self._save_learned_patterns)

            # Connect dictionary manager signals
            if self.state.dictionary_manager:
                self.state.dictionary_manager.pattern_learned.connect(self._handle_new_pattern)

            # Add debug logging of voice-search connectivity
            logger.debug("Voice connection status:")
            logger.debug(f"- Excel handler available: {hasattr(self.state, 'excel_handler')}")
            if hasattr(self.state, 'excel_handler'):
                logger.debug(f"- Search engine available: {hasattr(self.state.excel_handler, 'search_engine')}")
                if hasattr(self.state.excel_handler, 'search_engine'):
                    logger.debug(f"- Inventory cache size: {len(self.state.excel_handler.search_engine.inventory_cache) if hasattr(self.state.excel_handler.search_engine, 'inventory_cache') else 'None'}")
                    logger.debug(f"- Input column index: {getattr(self.state.excel_handler.search_engine, 'input_column_index', None)}")
                    logger.debug(f"- Workbook assigned: {hasattr(self.state.excel_handler.search_engine, 'workbook')}")
            
            logger.debug(f"- Voice interaction available: {hasattr(self.state, 'voice_interaction')}")
            if hasattr(self.state, 'voice_interaction'):
                logger.debug(f"- Voice connected to Excel: {hasattr(self.state.voice_interaction.state, 'excel_handler')}")
                logger.debug(f"- Voice connected to OneDrive: {hasattr(self.state.voice_interaction.state, 'onedrive_handler')}")
                
            logger.debug("NLP signals connected successfully")
        except Exception as e:
            logger.error(f"Error connecting NLP signals: {e}")
            raise

    def synchronize_search_engine_references(self):
        """Ensure all components use the same search engine reference."""
        try:
            logger.debug("=== SYNCHRONIZING SEARCH ENGINE REFERENCES ===")
            
            # Try to use component manager first
            try:
                from component_manager import component_manager
                
                # Get search engine from component manager
                search_engine = component_manager.get_search_engine()
                
                if not search_engine:
                    logger.debug("Component manager couldn't provide search engine, trying to create one")
                    
                    # Try to create a search engine using the factory
                    if 'search_engine' in component_manager.component_factories:
                        search_engine = component_manager.component_factories['search_engine']()
                        if search_engine:
                            component_manager.register_component('search_engine', search_engine)
                            logger.debug(f"Created and registered new search engine via factory")
                
                if search_engine:
                    # Update all component references
                    if hasattr(self.state, 'excel_handler'):
                        self.state.excel_handler.search_engine = search_engine
                        logger.debug("Updated excel_handler search engine reference")
                        
                    if hasattr(self.state, 'voice_interaction') and hasattr(self.state.voice_interaction, 'state'):
                        self.state.voice_interaction.state.search_engine = search_engine
                        logger.debug("Updated voice_interaction search engine reference")
                        
                    if hasattr(self.state, 'dugal'):
                        # Update direct references first
                        self.state.dugal.search_engine = search_engine
                        # Then check and update state references if it exists
                        if hasattr(self.state.dugal, 'state'):
                            self.state.dugal.state.search_engine = search_engine
                        logger.debug("Updated dugal search engine reference")
                    
                    # Verify references were fixed
                    self.verify_search_engine_references()
                    logger.debug("=== SEARCH ENGINE SYNCHRONIZATION COMPLETE (via component manager) ===")
                    return True
                    
            except ImportError:
                logger.debug("Component manager not available, falling back to legacy method")
            
            # Fall back to legacy method if component manager is not available or couldn't provide a search engine
            from global_registry import GlobalRegistry
            
            # Check if registry has a search engine
            registry_engine = GlobalRegistry.get('search_engine')
            registry_id = id(registry_engine) if registry_engine else None
            logger.debug(f"Registry search engine ID: {registry_id}")
            
            # Determine the best search engine to use
            chosen_engine = None
            chosen_source = None
            
            # Option 1: Use registry engine if it has data
            if registry_engine and hasattr(registry_engine, 'inventory_cache') and len(registry_engine.inventory_cache) > 0:
                chosen_engine = registry_engine
                chosen_source = "registry"
                logger.debug(f"Using registry search engine with {len(registry_engine.inventory_cache)} items")
            
            # Option 2: Check excel handler
            elif hasattr(self.state, 'excel_handler') and hasattr(self.state.excel_handler, 'search_engine'):
                excel_engine = self.state.excel_handler.search_engine
                if excel_engine and hasattr(excel_engine, 'inventory_cache') and len(excel_engine.inventory_cache) > 0:
                    chosen_engine = excel_engine
                    chosen_source = "excel_handler"
                    logger.debug(f"Using excel_handler search engine with {len(excel_engine.inventory_cache)} items")
            
            # Option 3: Check voice interaction
            elif (hasattr(self.state, 'voice_interaction') and 
                  hasattr(self.state.voice_interaction, 'state') and 
                  hasattr(self.state.voice_interaction.state, 'search_engine')):
                voice_engine = self.state.voice_interaction.state.search_engine
                if voice_engine and hasattr(voice_engine, 'inventory_cache') and len(voice_engine.inventory_cache) > 0:
                    chosen_engine = voice_engine
                    chosen_source = "voice_interaction"
                    logger.debug(f"Using voice_interaction search engine with {len(voice_engine.inventory_cache)} items")
            
            # Option 4: Create new engine if none found with data
            if not chosen_engine:
                logger.debug("No search engine with data found, creating new one")
                from search_engine import AdaptiveInventorySearchEngine
                chosen_engine = AdaptiveInventorySearchEngine()
                chosen_source = "new"
            
            # Register the chosen engine in the registry
            GlobalRegistry.register('search_engine', chosen_engine)
            logger.debug(f"Registered search engine from {chosen_source} in global registry")
            
            # Update all component references
            if hasattr(self.state, 'excel_handler'):
                self.state.excel_handler.search_engine = chosen_engine
                logger.debug("Updated excel_handler search engine reference")
                
            if hasattr(self.state, 'voice_interaction') and hasattr(self.state.voice_interaction, 'state'):
                self.state.voice_interaction.state.search_engine = chosen_engine
                logger.debug("Updated voice_interaction search engine reference")
                
            if hasattr(self.state, 'dugal'):
                # Update direct references first
                self.state.dugal.search_engine = chosen_engine
                # Then check and update state references if it exists
                if hasattr(self.state.dugal, 'state'):
                    self.state.dugal.state.search_engine = chosen_engine
                logger.debug("Updated dugal search engine reference")
                
            # Verify references were fixed
            self.verify_search_engine_references()
                
            logger.debug("=== SEARCH ENGINE SYNCHRONIZATION COMPLETE (via legacy method) ===")
            return True
        except Exception as e:
            logger.error(f"Error synchronizing search engine references: {e}")
            return False

    def verify_search_engine_references(self) -> bool:
        """Verify that all components have the same search engine reference."""
        try:
            logger.debug("=== VERIFYING SEARCH ENGINE REFERENCES ===")
            
            # Try to use component manager first
            try:
                from component_manager import component_manager
                
                # Get search engine from component manager
                cm_engine = component_manager.get_search_engine()
                cm_id = id(cm_engine) if cm_engine else None
                
                if cm_engine:
                    # Get component references
                    excel_id = id(self.state.excel_handler.search_engine) if hasattr(self.state.excel_handler, 'search_engine') else None
                    voice_id = (id(self.state.voice_interaction.state.search_engine) 
                              if hasattr(self.state.voice_interaction, 'state') and 
                                 hasattr(self.state.voice_interaction.state, 'search_engine') else None)
                    dugal_id = id(self.state.dugal.search_engine) if hasattr(self.state.dugal, 'search_engine') else None
                    
                    # Log all IDs for diagnostic purposes
                    logger.debug(f"Search engine IDs - Component Manager: {cm_id}, Excel: {excel_id}, Voice: {voice_id}, Dugal: {dugal_id}")
                    
                    # Check if all references match the component manager
                    excel_match = excel_id == cm_id if excel_id and cm_id else False
                    voice_match = voice_id == cm_id if voice_id and cm_id else False
                    dugal_match = dugal_id == cm_id if dugal_id and cm_id else False
                    
                    all_match = excel_match and voice_match and dugal_match
                    
                    if not all_match:
                        logger.warning("✗ Search engine references do not match component manager instance")
                        mismatch_components = []
                        if excel_id and not excel_match:
                            mismatch_components.append("excel_handler")
                        if voice_id and not voice_match:
                            mismatch_components.append("voice_interaction")
                        if dugal_id and not dugal_match:
                            mismatch_components.append("dugal")
                        logger.warning(f"Mismatched components: {mismatch_components}")
                    else:
                        logger.debug("✓ All components using same search engine instance from component manager")
                    
                    logger.debug("=== VERIFICATION COMPLETE (via component manager) ===")
                    return all_match
                    
            except ImportError:
                logger.debug("Component manager not available, falling back to legacy method")
            
            # Fall back to legacy method if component manager is not available
            from global_registry import GlobalRegistry
            registry_engine = GlobalRegistry.get('search_engine')
            registry_id = id(registry_engine) if registry_engine else None
            
            excel_id = id(self.state.excel_handler.search_engine) if hasattr(self.state.excel_handler, 'search_engine') else None
            voice_id = (id(self.state.voice_interaction.state.search_engine) 
                      if hasattr(self.state.voice_interaction, 'state') and 
                         hasattr(self.state.voice_interaction.state, 'search_engine') else None)
            dugal_id = id(self.state.dugal.search_engine) if hasattr(self.state.dugal, 'search_engine') else None
            
            # Log all IDs for diagnostic purposes
            logger.debug(f"Search engine IDs - Registry: {registry_id}, Excel: {excel_id}, Voice: {voice_id}, Dugal: {dugal_id}")
            
            # First check if registry has a search engine
            if not registry_id:
                logger.warning("No search engine found in global registry!")
                
                # Find the best candidate to register
                if excel_id:
                    logger.debug(f"Registering excel_handler's search engine in registry")
                    GlobalRegistry.register('search_engine', self.state.excel_handler.search_engine)
                elif voice_id:
                    logger.debug(f"Registering voice_interaction's search engine in registry")
                    GlobalRegistry.register('search_engine', self.state.voice_interaction.state.search_engine)
                elif dugal_id:
                    logger.debug(f"Registering dugal's search engine in registry")
                    GlobalRegistry.register('search_engine', self.state.dugal.search_engine)
                
                # Update registry_id if we registered something
                registry_engine = GlobalRegistry.get('search_engine')
                registry_id = id(registry_engine) if registry_engine else None
                
            # Check if all references match the registry
            excel_match = excel_id == registry_id if excel_id and registry_id else False
            voice_match = voice_id == registry_id if voice_id and registry_id else False
            dugal_match = dugal_id == registry_id if dugal_id and registry_id else False
            
            all_match = excel_match and voice_match and dugal_match
            
            if not all_match:
                logger.warning("✗ Search engine references do not match across components")
                mismatch_components = []
                if excel_id and not excel_match:
                    mismatch_components.append("excel_handler")
                if voice_id and not voice_match:
                    mismatch_components.append("voice_interaction")
                if dugal_id and not dugal_match:
                    mismatch_components.append("dugal")
                logger.warning(f"Mismatched components: {mismatch_components}")
            else:
                logger.debug("✓ All components using same search engine instance")
            
            logger.debug("=== VERIFICATION COMPLETE (via legacy method) ===")
            return all_match
            
        except Exception as e:
            logger.error(f"Error verifying search engine references: {e}")
            return False

    def _fix_search_engine_references(self) -> bool:
        """Fix search engine references to ensure all components use the same instance."""
        try:
            # Try to use component manager first
            try:
                from component_manager import component_manager
                
                # Try to recover the search engine using the component manager
                if 'search_engine' in component_manager.component_recovery_handlers:
                    logger.debug("Attempting to recover search engine via component manager")
                    recovery_success = component_manager.recover_component('search_engine')
                    if recovery_success:
                        logger.debug("Successfully recovered search engine via component manager")
                
                # Get the authoritative search engine instance from component manager
                search_engine = component_manager.get_search_engine()
                
                if search_engine:
                    logger.debug(f"Fixing search engine references to use component manager instance (ID: {id(search_engine)})")
                    
                    # Update all component references
                    if hasattr(self.state, 'excel_handler'):
                        self.state.excel_handler.search_engine = search_engine
                        logger.debug("Updated excel_handler search engine reference")
                        
                    if hasattr(self.state, 'voice_interaction') and hasattr(self.state.voice_interaction, 'state'):
                        self.state.voice_interaction.state.search_engine = search_engine
                        logger.debug("Updated voice_interaction search engine reference")
                        
                    if hasattr(self.state, 'dugal'):
                        self.state.dugal.search_engine = search_engine
                        logger.debug("Updated dugal search engine reference")
                        
                    if hasattr(self.state, 'dictionary_manager'):
                        self.state.dictionary_manager.search_engine = search_engine
                        logger.debug("Updated dictionary_manager search engine reference")
                    
                    # Verify the fix worked
                    fixed = self.verify_search_engine_references()
                    if fixed:
                        logger.debug("Search engine references successfully fixed via component manager")
                    return fixed
                    
            except ImportError:
                logger.debug("Component manager not available, falling back to legacy method")
            
            # Fall back to legacy method if component manager is not available or couldn't provide a search engine
            from global_registry import GlobalRegistry
            
            # Get the authoritative search engine instance from registry
            search_engine = GlobalRegistry.get('search_engine')
            if not search_engine:
                logger.error("No search engine found in registry")
                return False
                
            logger.debug(f"Fixing search engine references to use registry instance (ID: {id(search_engine)})")
            
            # Update all component references
            if hasattr(self.state, 'excel_handler'):
                self.state.excel_handler.search_engine = search_engine
                logger.debug("Updated excel_handler search engine reference")
                
            if hasattr(self.state, 'voice_interaction') and hasattr(self.state.voice_interaction, 'state'):
                self.state.voice_interaction.state.search_engine = search_engine
                logger.debug("Updated voice_interaction search engine reference")
                
            if hasattr(self.state, 'dugal'):
                self.state.dugal.search_engine = search_engine
                logger.debug("Updated dugal search engine reference")
                
            if hasattr(self.state, 'dictionary_manager'):
                self.state.dictionary_manager.search_engine = search_engine
                logger.debug("Updated dictionary_manager search engine reference")
            
            # Verify the fix worked
            fixed = self.verify_search_engine_references()
            if fixed:
                logger.debug("Search engine references successfully fixed via legacy method")
            return fixed
            
        except Exception as e:
            logger.error(f"Error fixing search engine references: {e}")
            return False

    def verify_system_connectivity(self):
        """Verify that all system components are properly connected."""
        logger.debug("=== SYSTEM CONNECTIVITY CHECK ===")
        
        # Track component references for verification
        component_refs = {}
        
        try:
            # Get search engine references from different components
            if hasattr(self.state, 'search_engine'):
                component_refs['main'] = id(self.state.search_engine)
                
            if hasattr(self.state, 'excel_handler') and hasattr(self.state.excel_handler, 'search_engine'):
                component_refs['excel'] = id(self.state.excel_handler.search_engine)
                
            if (hasattr(self.state, 'voice_interaction') and 
                hasattr(self.state.voice_interaction, 'state') and 
                hasattr(self.state.voice_interaction.state, 'search_engine')):
                component_refs['voice'] = id(self.state.voice_interaction.state.search_engine)
                
            if hasattr(self.state, 'dugal') and hasattr(self.state.dugal, 'search_engine'):
                component_refs['dugal'] = id(self.state.dugal.search_engine)
                
            # Log all component references
            for name, ref_id in component_refs.items():
                logger.debug(f"{name.capitalize()} search engine ID: {ref_id}")
                
            # Check for mismatches
            reference_ids = set(component_refs.values())
            if len(reference_ids) > 1:
                logger.error("CRITICAL: Search engine reference mismatch detected!")
                logger.error(f"Different components have different search engine instances: {component_refs}")
                
                # Attempt to fix by propagating the main reference
                self._fix_search_engine_references()
                return False
            else:
                logger.debug("System connectivity verified: All components share the same search engine instance")
                return True
                
        except Exception as e:
            logger.error(f"Error verifying system connectivity: {e}")
            return False
            
    def _fix_reference_mismatch(self, component_refs):
        """Attempt to fix reference mismatches by propagating the main reference."""
        try:
            # Determine which reference to use (prefer main > excel > voice > dugal)
            if 'main' in component_refs:
                source = 'main'
                reference = self.state.search_engine
            elif 'excel' in component_refs:
                source = 'excel'
                reference = self.state.excel_handler.search_engine
            else:
                logger.error("Cannot fix reference mismatch: No valid source reference")
                return False
                
            logger.debug(f"Fixing reference mismatch using {source} as source (ID: {component_refs[source]})")
            
            # Propagate the reference to all components
            if hasattr(self.state, 'excel_handler'):
                self.state.excel_handler.search_engine = reference
                logger.debug("Updated excel_handler reference")
                
            if hasattr(self.state, 'voice_interaction') and hasattr(self.state.voice_interaction, 'state'):
                self.state.voice_interaction.state.search_engine = reference
                logger.debug("Updated voice_interaction reference")
                
            if hasattr(self.state, 'dugal'):
                self.state.dugal.search_engine = reference
                logger.debug("Updated dugal reference")
                
            # Verify fix
            logger.debug("Reference mismatch fixed. New references:")
            if hasattr(self.state, 'excel_handler'):
                logger.debug(f"Excel: {id(self.state.excel_handler.search_engine)}")
            if hasattr(self.state.voice_interaction, 'state'):
                logger.debug(f"Voice: {id(self.state.voice_interaction.state.search_engine)}")
            if hasattr(self.state, 'dugal'):
                logger.debug(f"Dugal: {id(self.state.dugal.search_engine)}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error fixing reference mismatch: {e}")
            return False

    def _save_learned_patterns(self) -> None:
        """Save learned patterns after updates."""
        try:
            if self.state.search_engine and self.state.data_manager:
                patterns = self.state.search_engine.get_all_patterns()
                self.state.data_manager.save_patterns(patterns)
                logger.debug("Learned patterns saved successfully")
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")

    def _handle_new_pattern(self, base_term: str, variation: str) -> None:
        """Handle newly learned patterns."""
        try:
            if self.state.search_engine:
                self.state.search_engine.add_manual_pattern(base_term, variation)
                self._save_learned_patterns()
                logger.debug(f"New pattern added: {base_term} -> {variation}")
        except Exception as e:
            logger.error(f"Error handling new pattern: {e}")

    def handle_voice_command(self, result: Dict[str, Any]) -> None:
        """Handle processed voice commands."""
        try:
            if result.get("success", False):
                if result.get("action") == "update_inventory":
                    self._handle_inventory_update(result)
                elif result.get("action") == "select_sheets":
                    self._handle_sheet_selection(result)
                elif result.get("action") == "refresh_view":
                    self._handle_refresh_request(result)
            else:
                self.error_occurred.emit(
                    "Command Error",
                    result.get("message", "Unknown error processing command")
                )

        except Exception as e:
            logger.error(f"Error handling voice command: {e}")
            self.error_occurred.emit("Command Error", str(e))

    def _handle_inventory_update(self, result: Dict[str, Any]) -> None:
        """Handle inventory update commands."""
        try:
            item = result.get("item")
            value = result.get("value")
            if item and value is not None:
                update_result = self.state.one_drive_handler.update_inventory(item, value)
                if update_result.get("success", False):
                    self.state.file_coordinator.refresh_excel_view()
                else:
                    self.error_occurred.emit(
                        "Update Error",
                        update_result.get("message", "Failed to update inventory")
                    )
        except Exception as e:
            logger.error(f"Error handling inventory update: {e}")
            self.error_occurred.emit("Update Error", str(e))

    def _handle_sheet_selection(self, result: Dict[str, Any]) -> None:
        """Handle sheet selection commands."""
        try:
            sheets = result.get("sheets", [])
            if sheets:
                self.state.excel_handler.state.selected_sheets = sheets
                self.state.file_coordinator.refresh_excel_view()
        except Exception as e:
            logger.error(f"Error handling sheet selection: {e}")
            self.error_occurred.emit("Selection Error", str(e))

    def _handle_refresh_request(self, result: Dict[str, Any]) -> None:
        """Handle refresh requests."""
        try:
            self.state.file_coordinator.refresh_excel_view()
        except Exception as e:
            logger.error(f"Error handling refresh request: {e}")
            self.error_occurred.emit("Refresh Error", str(e))

    def handle_fatal_error(self, title: str, message: str) -> None:
        """Handle fatal errors by logging and cleaning up."""
        logger.critical("%s: %s", title, message)
        self.cleanup()
        sys.exit(1)

    def _recover_onedrive_handler(self) -> bool:
        """Attempt to recover the OneDrive handler if it's not functioning."""
        logger.debug("Attempting to recover OneDrive handler")
        try:
            # First try to clean up any existing handler
            if self.state.one_drive_handler:
                try:
                    self.state.one_drive_handler.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up existing OneDrive handler: {e}")
            
            # Reset the OneDrive state file
            onedrive_state_file = os.path.join('.dugal_data', 'onedrive_state.json')
            try:
                with open(onedrive_state_file, 'w') as f:
                    f.write('{}')
                logger.debug("Reset OneDrive state file for recovery")
            except Exception as e:
                logger.error(f"Error resetting OneDrive state file: {e}")
            
            # Create a new OneDrive handler
            self._init_onedrive_handler()
            
            # Check if initialization was successful
            if self.state.one_drive_handler and self.state.initialized_components.get('onedrive', False):
                logger.debug("OneDrive handler successfully recovered")
                return True
            else:
                logger.error("OneDrive handler recovery failed")
                return False
                
        except Exception as e:
            logger.error(f"Error recovering OneDrive handler: {e}")
            return False

    def _recover_voice_interaction(self) -> bool:
        """Attempt to recover the Voice Interaction component if it's not functioning."""
        logger.debug("Attempting to recover Voice Interaction system")
        try:
            # First try to clean up any existing voice interaction
            if self.state.voice_interaction:
                try:
                    self.state.voice_interaction.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up existing Voice Interaction: {e}")
            
            # Reinitialize voice interaction
            try:
                # Get Azure credentials from existing configuration if possible
                azure_key = None
                azure_region = None
                if hasattr(self.state, 'voice_interaction') and self.state.voice_interaction:
                    if hasattr(self.state.voice_interaction.state, 'speech_config'):
                        try:
                            azure_key = self.state.voice_interaction.state.speech_config.get_property(
                                speechsdk.PropertyId.SpeechServiceConnection_Key
                            )
                            azure_region = self.state.voice_interaction.state.speech_config.get_property(
                                speechsdk.PropertyId.SpeechServiceConnection_Region
                            )
                        except:
                            pass
                
                # Create new voice interaction instance
                self.state.voice_interaction = VoiceInteraction(
                    azure_key=azure_key,
                    azure_region=azure_region,
                    dugal=self.state.dugal
                )
                
                # Connect voice interaction to inventory system
                if self.state.one_drive_handler and self.state.excel_handler:
                    self.state.voice_interaction.connect_voice_to_inventory(
                        self.state.excel_handler,
                        self.state.one_drive_handler
                    )
                    
                # Mark component as initialized
                self.state.initialized_components['voice'] = True
                logger.debug("Voice Interaction successfully recovered")
                return True
                
            except Exception as e:
                logger.error(f"Failed to reinitialize Voice Interaction: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error recovering Voice Interaction: {e}")
            return False

    def _recover_excel_handler(self) -> bool:
        """Attempt to recover the Excel Handler if it's not functioning."""
        logger.debug("Attempting to recover Excel Handler")
        try:
            # First try to clean up any existing handler
            if self.state.excel_handler:
                try:
                    self.state.excel_handler.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up existing Excel Handler: {e}")
            
            # Create a new Excel handler
            self.state.excel_handler = ExcelHandler(dugal=self.state.dugal)
            
            # Reinitialize search engine connection
            if self.state.search_engine:
                self.state.excel_handler.search_engine = self.state.search_engine
            
            # Mark component as initialized
            self.state.initialized_components['excel'] = True
            logger.debug("Excel Handler successfully recovered")
            return True
                
        except Exception as e:
            logger.error(f"Error recovering Excel Handler: {e}")
            return False

    def _recover_dictionary_manager(self) -> bool:
        """Attempt to recover the Dictionary Manager if it's not functioning."""
        logger.debug("Attempting to recover Dictionary Manager")
        try:
            # First try to clean up any existing manager
            if self.state.dictionary_manager:
                try:
                    self.state.dictionary_manager.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up existing Dictionary Manager: {e}")
            
            # Create a new Dictionary Manager
            self.state.dictionary_manager = DictionaryManager(
                search_engine=self.state.search_engine,
                dugal=self.state.dugal
            )
            
            # Mark component as initialized
            self.state.initialized_components['dictionary'] = True
            logger.debug("Dictionary Manager successfully recovered")
            return True
                
        except Exception as e:
            logger.error(f"Error recovering Dictionary Manager: {e}")
            return False

    def _recover_file_coordinator(self) -> bool:
        """Attempt to recover the File Coordinator if it's not functioning."""
        logger.debug("Attempting to recover File Coordinator")
        try:
            # First try to clean up any existing coordinator
            if self.state.file_coordinator:
                try:
                    self.state.file_coordinator.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up existing File Coordinator: {e}")
            
            # Create a new File Coordinator
            self.state.file_coordinator = FileCoordinator(
                self.state.excel_handler,
                self.state.one_drive_handler,
                logging_manager=self.state.logging_manager
            )
            
            # Mark component as initialized
            self.state.initialized_components['coordinator'] = True
            logger.debug("File Coordinator successfully recovered")
            return True
                
        except Exception as e:
            logger.error(f"Error recovering File Coordinator: {e}")
            return False

    def _recover_gui(self) -> bool:
        """Attempt to recover the GUI component if it's not functioning."""
        logger.debug("Attempting to recover GUI component")
        try:
            # First try to clean up any existing GUI
            if self.state.gui:
                try:
                    self.state.gui.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up existing GUI: {e}")
            
            # Create a new GUI
            self.state.gui = DugalGUI(
                voice_interaction=self.state.voice_interaction,
                one_drive_handler=self.state.one_drive_handler,
                excel_handler=self.state.excel_handler,
                file_coordinator=self.state.file_coordinator,
                logging_manager=self.state.logging_manager
            )
            
            # Set up additional GUI references
            self.state.gui.dugal = self.state.dugal
            self.state.gui.file_coordinator = self.state.file_coordinator
            
            # Connect signals
            self.component_ready.connect(self.state.gui.handle_component_ready)
            self.error_occurred.connect(self.state.gui.handle_error)
            self.mode_changed.connect(self.state.gui.handle_mode_change)
            
            # Mark component as initialized
            self.state.initialized_components['gui'] = True
            logger.debug("GUI component successfully recovered")
            return True
                
        except Exception as e:
            logger.error(f"Error recovering GUI component: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up all components before shutdown."""
        logger.debug("Starting cleanup")
        try:
            # Clean up in reverse order of initialization
            
            # Save patterns before cleanup
            if self.state.search_engine and self.state.data_manager:
                try:
                    patterns = self.state.search_engine.get_all_patterns()
                    self.state.data_manager.save_patterns(patterns)
                    logger.debug("Final pattern save completed")
                except Exception as e:
                    logger.error(f"Error saving patterns during cleanup: {e}")

            # Dictionary manager cleanup
            if self.state.dictionary_manager:
                try:
                    self.state.dictionary_manager.cleanup()
                except Exception as e:
                    logger.error(f"Dictionary manager cleanup error: {e}")

            # GUI cleanup
            if self.state.gui:
                try:
                    self.state.gui.cleanup()
                except Exception as e:
                    logger.error(f"GUI cleanup error: {e}")

            # File coordinator cleanup
            if self.state.file_coordinator:
                try:
                    self.state.file_coordinator.cleanup()
                except Exception as e:
                    logger.error(f"File coordinator cleanup error: {e}")

            # Excel handler cleanup
            if self.state.excel_handler:
                try:
                    self.state.excel_handler.cleanup()
                except Exception as e:
                    logger.error(f"Excel handler cleanup error: {e}")

            # OneDrive handler cleanup
            if self.state.one_drive_handler:
                try:
                    self.state.one_drive_handler.cleanup()
                except Exception as e:
                    logger.error(f"OneDrive handler cleanup error: {e}")

            # Voice interaction cleanup
            if self.state.voice_interaction:
                try:
                    # NEW: Save any unsaved changes in SyncManager before cleanup
                    if hasattr(self.state.voice_interaction, 'sync_manager'):
                        try:
                            if self.state.voice_interaction.sync_manager.is_open:
                                if self.state.voice_interaction.sync_manager.update_handler.has_unsaved_changes():
                                    logger.warning("⚠️ Unsaved changes detected in SyncManager during shutdown")
                                    # Optionally auto-save or prompt user
                                    # self.state.voice_interaction.sync_manager.save_to_source()
                                self.state.voice_interaction.sync_manager.close_file()
                                logger.debug("SyncManager closed successfully")
                        except Exception as e:
                            logger.error(f"Error cleaning up SyncManager: {e}")
                    
                    self.state.voice_interaction.cleanup()
                except Exception as e:
                    logger.error(f"Voice interaction cleanup error: {e}")

            # Reset state
            self.state = IntegrationState()
            logger.debug("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def run_final_integration() -> None:
    """Run the final integration application."""
    try:
        # Initialize configuration
        init_config()

        # Clean up corrupted JSON files
        cleanup_json_files()

        app = QApplication(sys.argv)
        
        # Check and close Excel instances
        check_and_close_excel_instances()
             
        # Set application style and properties
        app.setStyle('Fusion')
        app.setApplicationName("Dugal Inventory System")
        app.setApplicationVersion("1.0.0")
        
        # Create and show main window
        final_integration = FinalIntegration()
        final_integration.show()
        
        # Set up exception handling for the Qt event loop
        def exception_hook(exctype, value, traceback):
            logger.critical("Unhandled exception", exc_info=(exctype, value, traceback))
            sys.__excepthook__(exctype, value, traceback)
            
        sys.excepthook = exception_hook
        
        # Run the application
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.critical("Fatal error during application runtime: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    run_final_integration()
