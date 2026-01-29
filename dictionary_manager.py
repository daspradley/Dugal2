"""
Dictionary manager module for Dugal Inventory System.
Handles pattern learning, term management, and NLP-enhanced search functionality.
"""

from __future__ import annotations

import os
import json
import logging
import shutil
import unicodedata
import re
from datetime import datetime
from typing import Dict, Set, List, Optional, Any, TYPE_CHECKING
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QTreeWidget, QTreeWidgetItem,
    QDialogButtonBox, QStatusBar, QProgressBar, QMessageBox, QStyle
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from metaphone import doublemetaphone

logger = logging.getLogger(__name__)

class DictionaryManager(QWidget):
    """Interface for managing learned patterns and terms."""
    term_added = pyqtSignal(dict)
    term_updated = pyqtSignal(dict)
    scan_completed = pyqtSignal(dict)
    pattern_learned = pyqtSignal(str, str)
    
    def __init__(self, search_engine=None, dugal=None):
        """Initialize the Dictionary Manager."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing Dictionary Manager")
        
        # Store references and initialize state
        self.search_engine = search_engine
        self.dugal = dugal
        if dugal:
            self.logging_manager = dugal.logging_manager if hasattr(dugal, 'logging_manager') else None
        else:
            self.logging_manager = None

        # Initialize UI base components
        self.dictionary_tree = QTreeWidget()  # Main tree view
        self.dictionary_tree.setHeaderLabels(["Base Term", "Variations"])  # Two columns
        self.dictionary_tree.setColumnCount(2)
        
        # Initialize additional UI components
        self.scan_button = None
        self.add_button = None
        self.search_bar = None
        self.progress_bar = None
        
        # Set up icons before UI construction
        try:
            self.setup_icons()
        except Exception as e:
            self.logger.error(f"Error setting up icons: {e}")
            # Continue with default icons if failed
            self.base_term_icon = QIcon()
            self.variation_icon = QIcon()

        # Initialize storage
        self.pattern_storage = None
        if hasattr(dugal, 'data_manager'):
            self.pattern_storage = dugal.data_manager
            
        # Initialize temp directory
        self.temp_dir = os.path.join(os.getcwd(), '.dugal_temp')
        os.makedirs(self.temp_dir, exist_ok=True)

        # Initialize pattern editor reference
        self.pattern_editor = None

        # Setup main UI
        self.logger.debug("Setting up Dictionary Manager UI")
        try:
            self.setup_ui()
        except Exception as e:
            self.logger.error(f"Error in UI setup: {e}")
            raise
            
        # Load existing patterns
        try:
            self.load_patterns()
            self.refresh_dictionary_view()
        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")
            self.show_status_message("Error loading patterns", error=True)

    def setup_icons(self):
        """Initialize icons for the dictionary tree."""
        try:
            # Default theme icons for terms and variations
            style = self.style()
            self.base_term_icon = QIcon(style.standardPixmap(QStyle.SP_DirIcon))
            self.variation_icon = QIcon(style.standardPixmap(QStyle.SP_FileIcon))
            
            # Alternative icons if we're using application-specific ones
            # icon_path = os.path.join(os.path.dirname(__file__), 'icons')
            # if os.path.exists(icon_path):
            #     base_icon_path = os.path.join(icon_path, 'base_term.png')
            #     var_icon_path = os.path.join(icon_path, 'variation.png')
            #     if os.path.exists(base_icon_path) and os.path.exists(var_icon_path):
            #         self.base_term_icon = QIcon(base_icon_path)
            #         self.variation_icon = QIcon(var_icon_path)
            
            # Verify icons were loaded properly
            if self.base_term_icon.isNull() or self.variation_icon.isNull():
                self.logger.warning("One or more icons failed to load, using fallbacks")
                # Fallback icons
                self.base_term_icon = QIcon(style.standardPixmap(QStyle.SP_DirIcon))
                self.variation_icon = QIcon(style.standardPixmap(QStyle.SP_FileIcon))
                
            self.logger.debug("Icons setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up icons: {e}")
            # Set empty icons as fallback
            self.base_term_icon = QIcon()
            self.variation_icon = QIcon()
            raise
     
    def setup_ui(self):
        """Set up the dictionary manager interface."""
        logger.debug("Setting up Dictionary Manager UI")
        layout = QVBoxLayout(self)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Top buttons panel
        button_panel = QHBoxLayout()
        
        self.scan_button = QPushButton("Scan Document for Terms")
        self.scan_button.setStyleSheet("""
            QPushButton {
                background-color: #6c5ce7;
                color: white;
                padding: 12px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #5f51e8;
            }
        """)
        
        self.add_button = QPushButton("Add New Term")
        self.add_button.setStyleSheet("""
            QPushButton {
                background-color: #00b894;
                color: white;
                padding: 12px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #00a885;
            }
        """)
        
        button_panel.addWidget(self.scan_button)
        button_panel.addWidget(self.add_button)
        button_panel.addStretch()

        # Search bar
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search terms...")
        self.search_bar.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
            }
        """)
        search_layout.addWidget(self.search_bar)

        # Dictionary view with enhanced styling
        self.dictionary_tree = QTreeWidget()
        self.dictionary_tree.setHeaderLabels(["Base Term", "Variations"])
        self.dictionary_tree.setStyleSheet("""
            QTreeWidget {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }
            QTreeWidget::item {
                padding: 5px;
            }
        """)

        # Add components to main layout
        layout.addLayout(button_panel)
        layout.addLayout(search_layout)
        layout.addWidget(self.dictionary_tree)

        # Set up icons for the tree items
        self.setup_icons()

        # Connect signals
        self.scan_button.clicked.connect(self.scan_document)
        self.add_button.clicked.connect(self.show_add_term_dialog)
        self.search_bar.textChanged.connect(self.filter_terms)
        self.dictionary_tree.itemDoubleClicked.connect(self.edit_term)

        # Set up the status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                border-top: 1px solid #ddd;
                padding: 5px;
            }
        """)
        layout.addWidget(self.status_bar)

        logger.debug("Dictionary Manager UI setup complete")

        # Initial population
        self.refresh_dictionary_view()

    def _setup_learning_section(self):
        """Set up the learning section for failed commands."""
        learning_group = QGroupBox("Learn From Failed Commands")
        layout = QVBoxLayout()
        
        self.failed_commands_list = QListWidget()
        
        learn_button = QPushButton("Add Selected to Dictionary")
        learn_button.clicked.connect(self._learn_selected_command)
        
        layout.addWidget(QLabel("Select a failed command to add:"))
        layout.addWidget(self.failed_commands_list)
        layout.addWidget(learn_button)
        
        learning_group.setLayout(layout)
        return learning_group

    def _populate_failed_commands(self):
        """Populate failed commands from voice interaction."""
        self.failed_commands_list.clear()
        
        if self.dugal and hasattr(self.dugal, 'voice_interaction'):
            vi = self.dugal.voice_interaction
            if hasattr(vi.state, 'failed_commands'):
                for item in vi.state.failed_commands:
                    self.failed_commands_list.addItem(item['command'])

    def show_status_message(self, message: str, error: bool = False) -> None:
        """Display a status message."""
        try:
            if error:
                self.status_bar.setStyleSheet("color: red")
            else:
                self.status_bar.setStyleSheet("")
            self.status_bar.showMessage(message, 3000)  # Show for 3 seconds
        except Exception as e:
            self.logger.error(f"Error showing status message: {e}")

    def get_search_engine(self):
        """Get the search engine from the component manager or local reference."""
        try:
            # Try to use component manager first
            try:
                from component_manager import component_manager
                
                # Get search engine from component manager
                search_engine = component_manager.get_search_engine()
                
                if search_engine:
                    # If found in component manager, update our local reference
                    if hasattr(self, 'search_engine') and self.search_engine is not search_engine:
                        logger.debug(f"Updating local search engine reference from component manager")
                        self.search_engine = search_engine
                    return search_engine
                    
            except ImportError:
                logger.debug("Component manager not available, falling back to legacy method")
            
            # Fall back to legacy method if component manager is not available
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

    def load_patterns(self) -> None:
        """Load existing patterns from storage."""
        try:
            if self.pattern_storage:
                patterns = self.pattern_storage.load_patterns()
                if patterns:
                    for base_term, variations in patterns.items():
                        base_item = self.find_or_create_base_term(base_term)
                        for variation in variations:
                            variation_item = QTreeWidgetItem(base_item)
                            variation_item.setText(0, variation)
                            variation_item.setIcon(0, self.variation_icon)
            
            self.logger.debug(f"Loaded {self.dictionary_tree.topLevelItemCount()} pattern sets into view")
        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")

    def add_pattern(self, base_term: str, variation: str) -> None:
        """Add a new pattern and emit signal."""
        try:
            # Validate inputs
            if not base_term or not variation:
                self.logger.error("Invalid pattern: empty base term or variation")
                return

            # Normalize terms
            base_term = base_term.strip().lower()
            variation = variation.strip().lower()

            # Check if pattern already exists
            if self.pattern_exists(base_term, variation):
                self.logger.debug(f"Pattern already exists: {base_term} -> {variation}")
                return

            # Add to search engine if available
            if self.search_engine:
                self.search_engine.add_manual_pattern(base_term, variation)

            # Update the pattern treeview
            base_item = self.find_or_create_base_term(base_term)
            variation_item = QTreeWidgetItem(base_item)
            variation_item.setText(0, variation)
            variation_item.setIcon(0, self.variation_icon)

            # Save pattern to storage
            if self.pattern_storage:
                self.pattern_storage.add_pattern(base_term, variation)

            # Emit the signal
            self.pattern_learned.emit(base_term, variation)

            # Log success
            self.logger.debug(f"Pattern added successfully: {base_term} -> {variation}")
            
            # Update UI if needed
            self.show_status_message(f"Added pattern: {variation} → {base_term}")
            self.refresh_dictionary_view()

        except Exception as e:
            self.logger.error(f"Error adding pattern: {e}")
            self.show_status_message(f"Error adding pattern: {str(e)}", error=True)

    def pattern_exists(self, base_term: str, variation: str) -> bool:
        """Check if a pattern already exists."""
        try:
            root = self.pattern_tree.invisibleRootItem()
            for i in range(root.childCount()):
                base_item = root.child(i)
                if base_item.text(0).lower() == base_term.lower():
                    for j in range(base_item.childCount()):
                        if base_item.child(j).text(0).lower() == variation.lower():
                            return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking pattern existence: {e}")
            return False

    def process_inventory_terms(self, workbook=None, search_engine=None):
        """Process inventory terms to generate phonetic variations."""
        logger.debug("Processing inventory terms for phonetic variations")
        
        if not search_engine:
            search_engine = self.search_engine
            
        if not search_engine or not hasattr(search_engine, 'inventory_cache'):
            logger.warning("No search engine available with inventory cache")
            return
        
        # Get all unique terms from inventory
        inventory_terms = set()
        for key, entries in search_engine.inventory_cache.items():
            for entry in entries:
                if 'original' in entry:
                    inventory_terms.add(entry['original'].lower())
        
        # Process each term
        variations_added = 0
        for term in inventory_terms:
            variations = self.generate_phonetic_variations(term)
            if variations:
                # Add to search engine patterns
                if term not in search_engine.learned_patterns:
                    search_engine.learned_patterns[term] = set()
                
                # Add phonetic variations
                for variation in variations:
                    if variation != term:  # Don't add self as variation
                        search_engine.learned_patterns[term].add(variation)
                        variations_added += 1
        
        # Save updated patterns
        search_engine.save_learned_patterns()
        logger.debug(f"Added {variations_added} phonetic variations for {len(inventory_terms)} terms")
        return variations_added

    def process_inventory_terms_async(self, workbook=None, search_engine=None):
        """Process inventory terms asynchronously."""
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.process_inventory_terms, workbook, search_engine)
            # Return future for caller to use
            return future

    def generate_phonetic_variations(self, term):
        """Generate phonetic variations of a term based on how it sounds."""
        if not term:
            return []
            
        variations = set()
        words = term.lower().split()
        
        # Process each word
        for word in words:
            # Get metaphone codes for the word
            primary, secondary = doublemetaphone(word)
            
            # Create variations with common vowel substitutions
            vowel_subs = {'a': ['e', 'o'], 'e': ['i', 'a'], 'i': ['e', 'y'], 
                          'o': ['a', 'u'], 'u': ['o', 'oo']}
            
            for i, char in enumerate(word):
                if char in vowel_subs:
                    for sub in vowel_subs[char]:
                        variation = word[:i] + sub + word[i+1:]
                        variations.add(variation)
            
            # Add variations with dropped or doubled consonants
            for i, char in enumerate(word):
                if char not in 'aeiou':
                    # Dropped consonant
                    variation = word[:i] + word[i+1:]
                    variations.add(variation)
                    
                    # Doubled consonant
                    variation = word[:i] + char + char + word[i+1:]
                    variations.add(variation)
        
        # Generate compound variations for multi-word terms
        result_variations = set()
        result_variations.add(term)  # Add the original term
        
        # For multi-word terms, create variations with different word combinations
        if len(words) > 1:
            # Add variations with different word orders
            for i in range(len(words)):
                # Skip a word
                subset = words[:i] + words[i+1:]
                result_variations.add(' '.join(subset))
        
        # Add variations with different spellings
        for variation in variations:
            result_variations.add(variation)
        
        # Add commonly misheard sounds (e.g., "th" vs "f", "s" vs "z")
        sound_pairs = [('th', 'f'), ('s', 'z'), ('v', 'f'), ('m', 'n'), ('ch', 'sh')]
        for pair in sound_pairs:
            if pair[0] in word:
                variation = word.replace(pair[0], pair[1])
                variations.add(variation)
            if pair[1] in word:
                variation = word.replace(pair[1], pair[0])
                variations.add(variation)
        
        # Add variations for numbers and homophones
        number_map = {'to': ['2', 'too'], 'for': ['4', 'fore'], 'one': ['1'], 'two': ['2']}
        for word in words:
            if word in number_map:
                for variant in number_map[word]:
                    new_words = words.copy()
                    new_words[words.index(word)] = variant
                    result_variations.add(' '.join(new_words))

        return list(result_variations)

    def find_or_create_base_term(self, base_term: str) -> QTreeWidgetItem:
        """Find existing base term or create new one."""
        root = self.dictionary_tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if item.text(0).lower() == base_term.lower():
                return item

        # Create new base term if not found
        new_item = QTreeWidgetItem(self.dictionary_tree)
        new_item.setText(0, base_term)
        new_item.setIcon(0, self.base_term_icon)
        return new_item

    def save_current_state(self):
        """Save current dictionary state."""
        try:
            if hasattr(self.search_engine, 'save_patterns'):
                self.search_engine.save_patterns()
                logger.debug("Dictionary state saved successfully")
                
                if self.logging_manager:
                    self.logging_manager.log_pattern_match({
                        'type': 'dictionary_state_saved',
                        'timestamp': datetime.now().isoformat()
                    })
        except Exception as e:
            logger.error(f"Error saving dictionary state: {e}")
            if self.logging_manager:
                self.logging_manager.log_error(str(e), {'context': 'dictionary_save'})

    def filter_terms(self, search_text: str) -> None:
        """Filter dictionary terms based on search text."""
        try:
            search_text = search_text.lower().strip()
            for i in range(self.dictionary_tree.topLevelItemCount()):
                base_item = self.dictionary_tree.topLevelItem(i)
                base_term = base_item.text(0).lower()
                matches = False

                # Check base term
                if search_text in base_term:
                    matches = True
                else:
                    # Check variations
                    for j in range(base_item.childCount()):
                        variation_item = base_item.child(j)
                        if search_text in variation_item.text(0).lower():
                            matches = True
                            break

                base_item.setHidden(not matches)
                
                # Show variations if there's a match
                if matches:
                    base_item.setExpanded(True)
        except Exception as e:
            self.logger.error(f"Error filtering terms: {e}")
            self.show_status_message("Error filtering terms", error=True)

    def edit_term(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle editing of terms and variations."""
        try:
            parent = item.parent()
            base_term = ""
            variation = ""
            
            if parent:  # Editing a variation
                base_term = parent.text(0)
                variation = item.text(0)
            else:  # Editing a base term
                base_term = item.text(0)
            
            dialog = TermEditorDialog(
                parent=self,
                base_term=base_term,
                edit_variation=variation
            )
            
            if dialog.exec_():
                new_base, variations = dialog.get_terms()
                if not new_base:
                    self.show_status_message("Base term cannot be empty", error=True)
                    return
                    
                # Update base term if changed
                if new_base != base_term:
                    if self.search_engine:
                        self.search_engine.update_base_term(base_term, new_base)
                        self.pattern_learned.emit(base_term, new_base)
                
                # Add/update variations
                for var in variations:
                    if var != variation:
                        self.add_pattern(new_base, var)
                
                self.refresh_dictionary_view()
                self.show_status_message("Terms updated successfully")
                
        except Exception as e:
            logger.error(f"Error editing term: {e}")
            self.show_status_message("Error updating terms", error=True)

    def show_add_term_dialog(self) -> None:
        """Show dialog for adding a new term."""
        try:
            dialog = AddTermDialog(parent=self)
            if dialog.exec_():
                base_term, variation = dialog.get_terms()
                if base_term:
                    self.add_pattern(base_term, variation)
        except Exception as e:
            self.logger.error(f"Error showing add term dialog: {e}")
            self.show_status_message("Error adding new term", error=True)

    def scan_document(self):
        """Initiate document scan for new terms."""
        logger.debug("Initiating document scan")
        if not hasattr(self.parent(), 'state') or not hasattr(self.parent().state, 'workbook'):
            QMessageBox.warning(self, "Error", "No workbook loaded")
            return
            
        reply = QMessageBox.question(self, 'Scan Document', 'Scan document for new terms?')
        if reply == QMessageBox.Yes:
            self.scan_button.setEnabled(False)
            self.progress_bar.show()
            try:
                self.progress_bar.setValue(25)
                self.search_engine.learn_from_document(
                    self.parent().state.workbook,
                    self.parent().state.search_column
                )
                self.progress_bar.setValue(75)
                self.refresh_dictionary_view()
                QMessageBox.information(self, "Success", "Scan completed!")
            except Exception as e:
                logger.error(f"Scan error: {e}")
                QMessageBox.warning(self, "Error", str(e))
            finally:
                self.scan_button.setEnabled(True)
                self.progress_bar.hide()

    def refresh_dictionary_view(self) -> None:
        """Refresh the dictionary tree view."""
        try:
            self.dictionary_tree.clear()
            
            if self.search_engine:
                patterns = self.search_engine.get_all_patterns()
                for base_term, variations in patterns.items():
                    base_item = QTreeWidgetItem(self.dictionary_tree)
                    base_item.setText(0, base_term)
                    base_item.setIcon(0, self.base_term_icon)
                    
                    for variation in variations:
                        var_item = QTreeWidgetItem(base_item)
                        var_item.setText(0, variation)
                        var_item.setIcon(0, self.variation_icon)
                
                self.logger.debug(f"Loaded {len(patterns)} pattern sets into view")
        except Exception as e:
            self.logger.error(f"Error refreshing dictionary view: {e}")
            self.show_status_message("Error refreshing view", error=True)

    def cleanup(self):
        """Clean up dictionary manager resources."""
        try:
            logger.debug("Starting dictionary manager cleanup")
            
            # Save current state
            self.save_current_state()
            
            # Clean up temp files
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.debug("Temp directory cleaned")
            
            # Close pattern editor if open
            if self.pattern_editor:
                self.pattern_editor.close()
                logger.debug("Pattern editor closed")
            
            # Clean up UI components
            for widget in self.findChildren(QWidget):
                try:
                    widget.close()
                    widget.deleteLater()
                except Exception as e:
                    logger.error(f"Error cleaning up widget: {e}")
            
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'dictionary_cleanup_complete',
                    'timestamp': datetime.now().isoformat()
                })
            
            logger.debug("Dictionary manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during dictionary manager cleanup: {e}")
            if self.logging_manager:
                self.logging_manager.log_error(str(e), {'context': 'dictionary_cleanup'})

    def closeEvent(self, event):
        """Handle window close event."""
        try:
            self.cleanup()
            event.accept()
        except Exception as e:
            logger.error(f"Error during close event: {e}")
            event.accept()


class TermEditorDialog(QDialog):
    """Dialog for adding or editing terms and variations."""
    
    def __init__(self, parent=None, base_term="", edit_variation=""):
        super().__init__(parent)
        logger.debug(f"Initializing TermEditorDialog - base_term: '{base_term}', edit_variation: '{edit_variation}'")
        self.setWindowTitle("Term Editor")
        self.setup_ui(base_term, edit_variation)

    def setup_ui(self, base_term, edit_variation):
        """Set up the term editor interface."""
        layout = QVBoxLayout(self)
        
        # Base term input
        base_layout = QHBoxLayout()
        base_label = QLabel("Base Term:")
        self.base_term = QLineEdit(base_term)
        base_layout.addWidget(base_label)
        base_layout.addWidget(self.base_term)
        
        # Variations input
        variations_label = QLabel("Variations (one per line):")
        self.variations = QTextEdit()
        if edit_variation:
            self.variations.setText(edit_variation)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        layout.addLayout(base_layout)
        layout.addWidget(variations_label)
        layout.addWidget(self.variations)
        layout.addWidget(buttons)
        
        logger.debug("Term editor UI setup complete")

    def get_terms(self) -> Tuple[str, List[str]]:
        """Get the base term and list of variations."""
        base = self.base_term.text().strip()
        # Split variations by newline and clean each one
        variations = [v.strip() for v in self.variations.toPlainText().split('\n') 
                     if v.strip()]  # Only keep non-empty variations
        return base, variations
    
    def accept(self):
        """Validate and accept the dialog."""
        base_term = self.base_term.text().strip()
        if not base_term:
            logger.warning("Attempted to save term with empty base term")
            QMessageBox.warning(self, "Validation Error", "Base term cannot be empty.")
            return
        logger.debug(f"Term editor accepted with base term: '{base_term}'")
        super().accept()

    def reject(self):
        """Cancel the dialog."""
        logger.debug("Term editor cancelled")
        super().reject()

    def cleanup(self) -> None:
        """Clean up resources before closing."""
        try:
            if self.parent() and self.parent().search_engine:
                self.parent().save_current_state()
            if self.parent() and self.parent().logging_manager:
                self.parent().logging_manager.log_pattern_match({
                    'type': 'term_editor_cleanup',
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Editor cleanup error: {e}")

class AddTermDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.setWindowTitle("Add New Term")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    dm = DictionaryManager(None)
    dm.show()
    sys.exit(app.exec_())
