"""
Enhanced GUI module for Dugal Inventory System.
Handles all graphical interface components, user interactions, and integrates logging.
"""

import sys
import os
import logging
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QLabel, QWidget,
    QHBoxLayout, QFrame, QApplication, QMessageBox, QToolBar,
    QStatusBar, QMenuBar, QMenu, QAction, QDialog, QProgressBar,
    QTextEdit
)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize

from voice_interaction import VoiceInteraction
from onedrive_handler import OneDriveHandler
from excel_handler import ExcelHandler
from dictionary_manager import DictionaryManager
from microphone_test import EnhancedMicrophoneTest

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class GUIState:
    """Enhanced state tracking for GUI components."""
    mode: str = "wild"
    is_test_mode: bool = False
    last_action: Optional[str] = None
    last_error: Optional[str] = None
    component_status: Dict[str, bool] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    active_windows: Dict[str, bool] = field(default_factory=dict)
    logging_manager: Optional['LoggingManager'] = None

@dataclass
class GUIComponents:
    """Stores GUI component references with enhanced tracking."""
    end_button: Optional[QPushButton] = None
    proper_button: Optional[QPushButton] = None
    mild_button: Optional[QPushButton] = None
    wild_button: Optional[QPushButton] = None
    status_label: Optional[QLabel] = None
    progress_bar: Optional[QProgressBar] = None
    log_viewer: Optional[QTextEdit] = None
    control_panel: Optional[QFrame] = None
class StatusMonitor(QFrame):
    """Real-time status monitoring widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the status monitor interface."""
        layout = QVBoxLayout(self)
        
        # Status header
        header = QLabel("System Status")
        header.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 14px;
                color: #2c3e50;
            }
        """)
        layout.addWidget(header)
        
        # Component status grid
        self.status_grid = QFrame()
        grid_layout = QHBoxLayout(self.status_grid)
        layout.addWidget(self.status_grid)
        
        # Performance metrics
        self.metrics_view = QTextEdit()
        self.metrics_view.setReadOnly(True)
        self.metrics_view.setMaximumHeight(100)
        layout.addWidget(self.metrics_view)
        
    def update_status(self, component: str, status: bool):
        """Update component status indicator."""
        color = "#27ae60" if status else "#c0392b"
        indicator = QLabel("●")
        indicator.setStyleSheet(f"QLabel {{ color: {color}; }}")
        label = QLabel(component)
        
        container = QFrame()
        layout = QHBoxLayout(container)
        layout.addWidget(indicator)
        layout.addWidget(label)
        
        self.status_grid.layout().addWidget(container)

class EnhancedDugalGUI(QMainWindow):
    """Enhanced main GUI window for Dugal Inventory System."""
    
    # Enhanced signals
    dictionary_opened = pyqtSignal()
    component_status_changed = pyqtSignal(str, bool)
    error_occurred = pyqtSignal(str, dict)
    mode_changed = pyqtSignal(str)
    action_completed = pyqtSignal(str, dict)

    def __init__(
        self,
        voice_interaction: Optional[VoiceInteraction] = None,
        one_drive_handler: Optional[OneDriveHandler] = None,
        excel_handler: Optional[ExcelHandler] = None,
        file_coordinator = None,  # Type hint omitted to avoid circular import
        logging_manager = None
    ):
        """Initialize the enhanced GUI with handlers and logging."""
        super().__init__()
        logger.debug("Initializing Enhanced Dugal GUI")
        
        # Core components
        self.voice_interaction = voice_interaction
        self.one_drive_handler = one_drive_handler
        self.excel_handler = excel_handler
        self.file_coordinator = file_coordinator
        self.logging_manager = logging_manager
        
        # State and components
        self.state = GUIState(logging_manager=logging_manager)
        self.components = GUIComponents()
        self.status_monitor = None
        self.microphone_test_window = None
        self.dugal = None
        self.dictionary_manager = None
        
        # Initialize image cache
        self._image_cache = {}
        
        # Initialize metrics display
        self.metrics_display = QTextEdit()
        self.metrics_display.setReadOnly(True)
        self.metrics_display.setMaximumHeight(200)

        # Initialize UI
        self.setup_enhanced_menu()
        self.setup_enhanced_statusbar()
        self.mode_buttons = []
        self.init_ui()
        self.setup_monitoring()
        
        if self.logging_manager:
            self.logging_manager.log_pattern_match({
                'type': 'gui_initialized',
                'timestamp': datetime.now().isoformat()
            })
            
        logger.debug("Enhanced GUI initialization complete")

    def setup_monitoring(self):
        """Setup monitoring components and timers."""
        try:
            # Initialize monitoring components
            self.monitoring_timer = QTimer(self)
            self.monitoring_timer.timeout.connect(self._update_monitoring_display)
            self.monitoring_timer.start(1000)  # Update every second
            
            # Initialize component status
            self._update_component_status()
        except Exception as e:
            logger.error(f"Error setting up monitoring: {e}")

    def init_ui(self) -> None:
        """Initialize the enhanced GUI window and components."""
        # Base styles for text elements and frames
        base_style = """
            QLabel {
                color: black;
                text-decoration: none;
                font-size: 12px;
                background-color: none;
                padding 2px;
            }
            QLineEdit {
                color: black;
                text-decoration: none;
                font-size: 12px;
                background-color: white;
                border: 1 px solid #ddd;
            }
            QFrame {
                background: none;
                border: none;
                margin: 2px;
            }
            QWidget {
                background-color: white;
            }
        """
        
        self.setStyleSheet(base_style)
        
        # Calculate screen size and set window size
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.8)    # Reduced to 80% of screen width
        height = int(screen.height() * 0.8)   # Reduced to 80% of screen height
        
        # Position window in center of screen
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        
        self.setGeometry(x, y, width, height)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
        # Create central widget with fixed size
        main_widget = QWidget(self)
        main_widget.setObjectName("mainWidget")
        main_widget.setFixedSize(width, height)
        self.setCentralWidget(main_widget)
            
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)  # Increased spacing
        main_layout.setContentsMargins(15,15,15,15)  # Increased margins
        
        # Header - smaller to give more room to main content
        header_frame = self._create_header_section()
        #header_frame.setMaximumHeight(int(height * 0.2))  # Reduced from 0.6
        header_frame.setStyleSheet("background: none;")
        main_layout.addWidget(header_frame)
        
        # Content area - give it more vertical space
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # Main content gets more horizontal space
        content_frame = QFrame()
        #content_frame.setMaximumHeight(int(height * 0.6))  # Increased for main content
        content_frame.setStyleSheet("background: none;")
        content_layout.addWidget(content_frame, stretch=8)  # Increased stretch
        self._setup_main_content(content_frame)
           
        main_layout.addLayout(content_layout)
        
        # Footer - reduced to give more space to content
        footer_frame = self._create_footer_section()
        #footer_frame.setMaximumHeight(int(height * 0.2))  # Reduced from 0.5
        footer_frame.setStyleSheet("background: none;")
        main_layout.addWidget(footer_frame)
        
        self.setWindowTitle("Dugal The Barman Inventory Service")
        self.show()  # Make sure to show the window
        
        # Log UI initialization
        if self.logging_manager:
            self.logging_manager.log_pattern_match({
                'type': 'ui_initialized',
                'components': list(self.state.component_status.keys()),
                'timestamp': datetime.now().isoformat()
            })

    def _create_header_section(self) -> QFrame:
        """Create simplified header with just title."""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        
        # Title and subtitle only
        title = QLabel("Dugal Inventory System")
        title.setStyleSheet("""
            QLabel {
                color: black;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        subtitle = QLabel("Intelligent Inventory Management")
        subtitle.setStyleSheet("QLabel { color: #bdc3c7; font-size: 14px; }")
        
        title_layout = QVBoxLayout()
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        layout.addLayout(title_layout)
        layout.addStretch()
        
        return frame

    def _setup_main_content(self, frame: QFrame) -> None:
        """Main content area with flag-style layout."""
        layout = QHBoxLayout(frame)
        
        # Left section for buttons - reduced width to avoid frame overlap
        button_section = QFrame()
        button_layout = QVBoxLayout(button_section)
        button_layout.setSpacing(10)
        button_layout.setContentsMargins(20, 20, 40, 20)  # Added right margin
        
        # Add smaller spacing at the top to push all buttons down a bit
        # Use an empty widget with fixed height instead of stretch
        spacer = QWidget()
        spacer.setFixedHeight(120)  # Adjust this value as needed (100 pixels down)
        button_layout.addWidget(spacer)

        # Common button size - slightly reduced width
        button_width = 525  # Temporary increase from 400/50
        button_height = 75
        
        # Store button references for later use
        self.mode_buttons = []  # For keeping track of mode buttons
        
        # Create mode buttons first
        mode_buttons = [
            ("The Professional Mixologist", "#f8f8f8", "proper"),
            ("The Weary Server", "#f8f8f8", "mild"),
            ("The Angry Bartender", "#f8f8f8", "wild")
        ]

        for text, color, mode in mode_buttons:
            btn = QPushButton(text)
            btn.setFixedSize(button_width, button_height)
            btn.mode = mode  # Add mode attribute using internal mode name
            btn.clicked.connect(lambda checked, btn=btn, m=mode: self._handle_mode_selection(m))
            self.mode_buttons.append(btn)  # Store for mode tracking
            self._setup_button_style(btn, color)
            button_layout.addWidget(btn)

        # Create End button
        end_button = QPushButton("End")
        end_button.setFixedSize(button_width, button_height)
        end_button.clicked.connect(self._handle_end_click)
        self._setup_button_style(end_button, "#ff4d4d")
        button_layout.addWidget(end_button)

        # Create action buttons
        #action_buttons = [
        #    ("Scan Document", "#00FF00", self._handle_scan_click),
        #    ("Dictionary Manager", "#DA706D", self._handle_dictionary_click),
        #    ("Clear", "#f8f8f8", self._clear_logs)
        #]

        #for text, color, handler in action_buttons:
        #    btn = QPushButton(text)
        #    btn.setFixedSize(button_width, button_height)
        #    btn.clicked.connect(handler)
        #    self._setup_button_style(btn, color)
        #    button_layout.addWidget(btn)

        button_layout.addStretch()
        layout.addWidget(button_section, stretch=3)
        
        # Right section for image
        image_section = QFrame()
        image_layout = QVBoxLayout(image_section)
        image_layout.setContentsMargins(10, 10, 10, 10)
        image_paths = ["dugal.jpg", "dugal2.jpeg", "bob.jpeg"]
        image_label = self._load_image(image_paths, width=800, height=800)
        if image_label:
            image_layout.addWidget(image_label, alignment=Qt.AlignCenter)
        layout.addWidget(image_section, stretch=2)

    def _setup_button_style(self, button: QPushButton, color: str) -> None:
        """Helper method to set up button styling."""
        is_end = button.text() == "End"
        style = f"""
            QPushButton {{
                font-size: 16px;
                font-weight: bold;
                padding: 0px;
                border-radius: 15px;
                border: 2px solid #808080;
                color: {('#ffffff' if is_end else '#333333')};
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 {self._lighten_color(color, 10)}, 
                                                stop:1 {color});
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 {color}, 
                                                stop:1 {self._darken_color(color, 10)});
                border: 2px solid #606060;
            }}
            QPushButton:pressed {{
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 {self._darken_color(color, 10)}, 
                                                stop:1 {self._darken_color(color, 20)});
                border: 2px solid #404040;
                padding-top: 2px;
                padding-left: 2px;
            }}
        """
        button.setStyleSheet(style)

    def _lighten_color(self, hex_color: str, percent: int) -> str:
        """Lighten a hex color by the specified percentage."""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        lighter = [int(min(255, value * (100 + percent) / 100)) for value in rgb]
        return f"#{lighter[0]:02x}{lighter[1]:02x}{lighter[2]:02x}"

    def _darken_color(self, hex_color: str, percent: int) -> str:
            """Darken a hex color by the specified percentage."""
            # Remove '#' if present
            hex_color = hex_color.lstrip('#')
            
            # Convert to RGB
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            # Darken each component
            darkened = [int(max(0, value * (100 - percent) / 100)) for value in rgb]
            
            # Convert back to hex
            return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"        

    def _create_footer_section(self) -> QFrame:
        """Create footer with controls and status."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                padding: 10px;
                margin: 5px;
                border-radius: 5px;
            }
        """)
        layout = QHBoxLayout(frame)
        
        # Status label
        status_label = QLabel("Ready")
        status_label.setStyleSheet("color: white;")
        layout.addWidget(status_label)
        
        # Spacer
        layout.addStretch()
        
        return frame

    def _load_image(self, image_paths: list, width: int = 64, height: int = 64) -> Optional[QLabel]:
        """Load and cache an image from a list of possible paths."""
        try:
            # Check cache first
            cache_key = f"{','.join(image_paths)}_{width}_{height}"
            if cache_key in self._image_cache:
                pixmap = self._image_cache[cache_key]
                label = QLabel()
                label.setPixmap(pixmap)
                return label

            # Try each path in common locations
            for img_path in image_paths:
                possible_locations = [
                    img_path,
                    os.path.join('resources', img_path),
                    os.path.join('images', img_path),
                    os.path.join('assets', img_path),
                    os.path.join(os.path.dirname(__file__), img_path),
                    os.path.join(os.path.dirname(__file__), 'resources', img_path),
                    os.path.join(os.path.dirname(__file__), 'images', img_path),
                    os.path.join(os.path.dirname(__file__), 'assets', img_path)
                ]

                for path in possible_locations:
                    if os.path.exists(path):
                        pixmap = QPixmap(path)
                        if not pixmap.isNull():
                            # Scale if dimensions provided
                            if width and height:
                                pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            
                            # Cache the pixmap
                            self._image_cache[cache_key] = pixmap
                            
                            # Create and return label
                            label = QLabel()
                            label.setPixmap(pixmap)
                            return label

            # If no image found, create placeholder
            return self._create_placeholder_image(width, height)

        except Exception as e:
            logger.error(f"Error loading images {image_paths}: {e}")
            return self._create_placeholder_image(width, height)

    def _create_placeholder_image(self, width: int, height: int) -> QLabel:
        """Create a placeholder image when actual image cannot be loaded."""
        try:
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.lightGray)
            
            painter = QPainter(pixmap)
            painter.setPen(Qt.darkGray)
            painter.drawRect(0, 0, width-1, height-1)
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "No Image")
            painter.end()
            
            label = QLabel()
            label.setPixmap(pixmap)
            return label
            
        except Exception as e:
            logger.error(f"Error creating placeholder image: {e}")
            return QLabel("Image")

    def _setup_monitoring_sidebar(self, frame: QFrame) -> None:
        """Set up the monitoring sidebar with real-time stats."""
        layout = QVBoxLayout(frame)
        frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        # Create status monitor
        self.status_monitor = StatusMonitor()
        layout.addWidget(self.status_monitor)
        
        # Add real-time metrics display
        metrics_frame = QFrame()
        metrics_layout = QVBoxLayout(metrics_frame)
        
        metrics_label = QLabel("System Metrics")
        metrics_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        metrics_layout.addWidget(metrics_label)
        
        self.metrics_display = QTextEdit()
        self.metrics_display.setReadOnly(True)
        self.metrics_display.setMaximumHeight(200)
        metrics_layout.addWidget(self.metrics_display)
        
        layout.addWidget(metrics_frame)
        
        # Start monitoring updates
        self._start_monitoring_updates()

    def _create_mode_buttons(self, layout: QHBoxLayout) -> None:
        """Create enhanced mode selection buttons with visual feedback."""
        button_style = (
            "QPushButton {"
            "    background-color: #2c3e50;"
            "    color: white;"
            "    border: none;"
            "    padding: 8px 16px;"  # Reduced padding
            "    font-size: 14px;"    # Slightly smaller font
            "    min-width: 120px;"   # Reduced minimum width
            "    min-height: 40px;"   # Reduced minimum height
            "    border-radius: 5px;"
            "    margin: 3px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #34495e;"
            "}"
            "QPushButton:disabled {"
            "    background-color: #7f8c8d;"
            "}"
            "QPushButton:checked {"
            "    background-color: #27ae60;"
            "}"
        )

        mode_container = QFrame()
        mode_container.setStyleSheet("QFrame { padding: 5px; }")  # Reduced padding
        mode_layout = QHBoxLayout(mode_container)
        mode_layout.setSpacing(10)  # Reduced spacing between buttons

        # Create mode buttons
        self.components.wild_button = QPushButton("Start Wild")
        self.components.mild_button = QPushButton("Start Mild")
        self.components.proper_button = QPushButton("Start Proper")
        
        mode_buttons = [
            (self.components.wild_button, "wild"),
            (self.components.mild_button, "mild"),
            (self.components.proper_button, "proper")
        ]

        for button, mode in mode_buttons:
            button.setStyleSheet(button_style)
            button.setCheckable(True)
            button.setFixedWidth(150)  # Set fixed width for consistency
            button.setFixedHeight(40)  # Set fixed height
            button.clicked.connect(lambda checked, m=mode: self._handle_mode_selection(m))
            mode_layout.addWidget(button)

        layout.addWidget(mode_container)

        # End button with special styling
        self.components.end_button = QPushButton("End")
        self.components.end_button.setStyleSheet(
            "QPushButton {"
            "    background-color: #c0392b;"
            "    color: white;"
            "    border: none;"
            "    padding: 8px 16px;"
            "    font-size: 14px;"
            "    min-width: 100px;"
            "    min-height: 40px;"
            "    border-radius: 5px;"
            "    margin: 3px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #e74c3c;"
            "}"
        )
        self.components.end_button.setFixedWidth(100)  # Smaller width for end button
        self.components.end_button.setFixedHeight(40)
        self.components.end_button.clicked.connect(self._handle_end_click)
        layout.addWidget(self.components.end_button)

    def setup_enhanced_menu(self):
        """Setup simplified enhanced menu bar."""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu('&File')
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self._handle_end_click)
        file_menu.addAction(exit_action)
        
        # Help Menu
        help_menu = menubar.addMenu('&Help')
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_enhanced_statusbar(self):
        """Setup enhanced statusbar with system info."""
        statusbar = self.statusBar()
        statusbar.setStyleSheet("""
            QStatusBar {
                background-color: #34495e;
                color: white;
            }
            QLabel {
                color: white;
                padding: 3px;
            }
        """)

        # Component status indicators
        for component in ['Voice', 'Excel', 'OneDrive', 'Dictionary']:
            status = QLabel(f"{component}: Inactive")
            status.setObjectName(f"{component.lower()}_status")
            statusbar.addPermanentWidget(status)

        # Connection status
        self.connection_status = QLabel("⚫ Offline")
        self.connection_status.setStyleSheet("color: #e74c3c;")
        statusbar.addPermanentWidget(self.connection_status)

        def update_component_status(component: str, active: bool):
            status_label = self.findChild(QLabel, f"{component.lower()}_status")
            if status_label:
                status = "Active" if active else "Inactive"
                color = "#2ecc71" if active else "#e74c3c"
                status_label.setText(f"{component}: {status}")
                status_label.setStyleSheet(f"color: {color};")

        self.component_status_changed.connect(update_component_status)
        
        # Update connection status when components change
        def check_connection():
            all_active = all(self.state.component_status.values())
            self.connection_status.setText("⚫ Online" if all_active else "⚫ Offline")
            self.connection_status.setStyleSheet(
                f"color: {'#2ecc71' if all_active else '#e74c3c'};"
            )

        self.component_status_changed.connect(lambda *args: check_connection())

    def refresh_data(self):
        """Refresh all data displays."""
        try:
            if hasattr(self, 'status_monitor'):
                self.status_monitor.update_all()
            if self.dugal and hasattr(self.dugal, 'refresh_data'):
                self.dugal.refresh_data()
        except Exception as e:
            self.logger.error(f"Error refreshing data: {e}")

    def toggle_monitoring(self):
        """Toggle monitoring sidebar visibility."""
        if hasattr(self, 'monitoring_widget'):
            self.monitoring_widget.setVisible(
                not self.monitoring_widget.isVisible()
            )

    def toggle_voice_control(self):
        """Toggle voice control system."""
        try:
            if self.dugal and hasattr(self.dugal, 'voice_interaction'):
                current_state = self.dugal.voice_interaction.is_active()
                self.dugal.voice_interaction.set_active(not current_state)
        except Exception as e:
            self.logger.error(f"Error toggling voice control: {e}")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, 'About Dugal',
            'Dugal Inventory Management System\n'
            'Version 2.0\n\n'
            'An intelligent inventory management solution.'
            'Created by D. Spradley, CSS, 2024.'
    )

    def _create_action_buttons(self, layout: QVBoxLayout) -> None:
        """Create enhanced action buttons with visual feedback."""
        action_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton[type="dictionary"] {
                background-color: #9b59b6;
            }
            QPushButton[type="dictionary"]:hover {
                background-color: #8e44ad;
            }
            QPushButton[type="scan"] {
                background-color: #2ecc71;
            }
            QPushButton[type="scan"]:hover {
                background-color: #27ae60;
            }
        """

        # Dictionary button
        dict_button = QPushButton("Dictionary Manager")
        dict_button.setProperty("type", "dictionary")
        dict_button.setStyleSheet(action_style)
        dict_button.clicked.connect(self._handle_dictionary_click)
        layout.addWidget(dict_button)

        # Scan button
        scan_button = QPushButton("Scan Document")
        scan_button.setProperty("type", "scan")
        scan_button.setStyleSheet(action_style)
        scan_button.clicked.connect(self._handle_scan_click)
        layout.addWidget(scan_button)

    def _setup_log_viewer(self, layout: QVBoxLayout) -> None:
        """Set up enhanced log viewer with filtering and search."""
        log_frame = QFrame()
        log_layout = QVBoxLayout(log_frame)
        
        # Log header with controls
        header_layout = QHBoxLayout()
        
        log_label = QLabel("System Logs")
        log_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        header_layout.addWidget(log_label)
        
        clear_button = QPushButton("Clear")
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        clear_button.clicked.connect(self._clear_logs)
        header_layout.addWidget(clear_button)
        
        log_layout.addLayout(header_layout)
        
        # Log viewer
        self.components.log_viewer = QTextEdit()
        self.components.log_viewer.setReadOnly(True)
        self.components.log_viewer.setMaximumHeight(200)
        self.components.log_viewer.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 5px;
                font-family: monospace;
            }
        """)
        log_layout.addWidget(self.components.log_viewer)
        
        layout.addWidget(log_frame)

    def synchronize_search_engine_references(self):
        """Ensure GUI uses the latest search engine reference from registry."""
        try:
            from global_registry import GlobalRegistry
            logger.debug("Synchronizing GUI search engine reference")
            
            # Get search engine from registry
            search_engine = GlobalRegistry.get('search_engine')
            if not search_engine:
                logger.warning("No search engine in global registry")
                return False
            
            # Update local reference if different
            if hasattr(self, 'search_engine') and self.search_engine is not search_engine:
                logger.debug(f"Updating GUI search engine reference (old: {id(self.search_engine)}, new: {id(search_engine)})")
                self.search_engine = search_engine
            elif not hasattr(self, 'search_engine'):
                logger.debug("Setting initial GUI search engine reference")
                self.search_engine = search_engine
                
            return True
        except Exception as e:
            logger.error(f"Error synchronizing GUI search engine reference: {e}")
            return False

    def _start_monitoring_updates(self) -> None:
        """Start periodic monitoring updates."""
        self.monitoring_timer = QTimer(self)
        self.monitoring_timer.timeout.connect(self._update_monitoring_display)
        self.monitoring_timer.start(1000)  # Update every second

    def handle_component_ready(self, component_name: str) -> None:
        """Handle component ready signal."""
        try:
            logger.debug(f"Component ready: {component_name}")
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"{component_name} ready", 2000)
        except Exception as e:
            logger.error(f"Error handling component ready: {e}")

    def _handle_test_complete(self, result) -> None:
        """Handle microphone test completion."""
        logger.debug("Handling microphone test completion")
        try:
            if result.get('success'):
                if self.voice_interaction:
                    self.voice_interaction.speak("Microphone test successful!")
                    
                # Hide the main GUI window
                self.hide()
                
                # Close microphone test window if still open
                if self.microphone_test_window and self.microphone_test_window.isVisible():
                    self.microphone_test_window.close()
                    
                # Now show file selection dialog
                self.show_file_selection()
            else:
                if self.voice_interaction:
                    self.voice_interaction.speak("Microphone test failed. Please try again.")
                self.show_mic_test(self.state.mode)  # Show test window again
        except Exception as e:
            logger.error(f"Error handling test completion: {e}")

        self.synchronize_search_engine_references()

    def handle_error(self, title: str, message: str) -> None:
        """Handle error signal."""
        try:
            logger.error(f"Error - {title}: {message}")
            QMessageBox.critical(self, title, message)
        except Exception as e:
            logger.error(f"Error handling error signal: {e}")

    def handle_mode_change(self, mode: str) -> None:
        """Handle mode change signal."""
        try:
            logger.debug(f"Mode changed to: {mode}")
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"Mode: {mode}", 2000)
            self._update_mode_buttons(mode)
        except Exception as e:
            logger.error(f"Error handling mode change: {e}")

    def _update_mode_buttons(self, current_mode: str) -> None:
        """Update mode buttons based on current mode."""
        try:
            for button in self.mode_buttons:
                if hasattr(button, 'mode'):
                    button.setChecked(button.mode == current_mode)
        except Exception as e:
            logger.error(f"Error updating mode buttons: {e}")

    def _update_monitoring_display(self) -> None:
        """Update monitoring display with current metrics."""
        try:
            if self.logging_manager:
                metrics = self.logging_manager.get_performance_metrics()
                self._update_metrics_display(metrics)
                
            # Update component status
            self._update_component_status()
            
        except Exception as e:
            logger.error(f"Error updating monitoring display: {e}")
            if self.logging_manager:
                self.logging_manager.log_error(str(e), {
                    'context': 'monitoring_update'
                })

    def _update_metrics_display(self, metrics: Dict[str, Any]) -> None:
        """Update metrics display with formatted data."""
        try:
            formatted_metrics = json.dumps(metrics, indent=2)
            self.metrics_display.setText(formatted_metrics)
            
            # Update progress bar if relevant metrics exist
            if 'recognition_rate' in metrics:
                self.components.progress_bar.setValue(int(metrics['recognition_rate']))
                
        except Exception as e:
            logger.error(f"Error updating metrics display: {e}")

    def _update_component_status(self) -> None:
        """Update component status display."""
        try:
            components = {
                'Voice': bool(self.voice_interaction),
                'Excel': bool(self.excel_handler),
                'OneDrive': bool(self.one_drive_handler),
                'Dictionary': bool(self.dictionary_manager),
                'Monitoring': self.monitoring_timer.isActive()
            }
            
            for component, status in components.items():
                if component not in self.state.component_status or self.state.component_status[component] != status:
                    self.state.component_status[component] = status
                    self.component_status_changed.emit(component, status)
                    
                    if self.status_monitor:
                        self.status_monitor.update_status(component, status)
                    
                    if self.logging_manager:
                        self.logging_manager.log_pattern_match({
                            'type': 'component_status_change',
                            'component': component,
                            'status': status,
                            'timestamp': datetime.now().isoformat()
                        })
                        
        except Exception as e:
            logger.error(f"Error updating component status: {e}")

    def _log_message(self, message: str, level: str = "info") -> None:
        """Add a message to the log viewer with enhanced formatting."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            color = {
                "info": "#2c3e50",
                "success": "#27ae60",
                "error": "#c0392b",
                "warning": "#f39c12"
            }.get(level, "#2c3e50")
            
            formatted_message = f'<span style="color: {color}"><b>[{timestamp}]</b> {message}</span>'
            self.components.log_viewer.append(formatted_message)
            
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'gui_log',
                    'message': message,
                    'level': level,
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error logging message: {e}")

    def _clear_logs(self) -> None:
        """Clear the log viewer."""
        try:
            self.components.log_viewer.clear()
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'logs_cleared',
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error clearing logs: {e}")

    def show_error(self, title: str, message: str) -> None:
        """Show enhanced error dialog with logging."""
        try:
            self._log_message(message, "error")
            
            error_box = QMessageBox(self)
            error_box.setIcon(QMessageBox.Critical)
            error_box.setWindowTitle(title)
            error_box.setText(message)
            error_box.setStandardButtons(QMessageBox.Ok)
            
            if self.logging_manager:
                self.logging_manager.log_error(message, {
                    'context': 'error_dialog',
                    'title': title
                })
            
            error_box.exec_()
            
        except Exception as e:
            logger.error(f"Error showing error dialog: {e}")

    def _handle_mode_selection(self, mode: str) -> None:
        """Handle mode selection with enhanced logging and feedback."""
        logger.debug(f"Mode selection: {mode}")
        try:
            self.state.mode = mode
            self._update_mode_buttons(mode)
            self.mode_changed.emit(mode)
            
            # Set Dugal's mode and trigger welcome
            if self.dugal:
                self.dugal.state.mode = mode
                
            if self.voice_interaction:
                if mode == "wild":
                    self.voice_interaction.speak("Ah, fook me runnin'!")
                elif mode == "mild":
                    self.voice_interaction.speak("Again? Well...I guess...")
                elif mode == "proper":
                    self.voice_interaction.speak("Welcome and let's begin.")
            
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'mode_change',
                    'mode': mode,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Minimize the window to taskbar
            self.showMinimized()
            
            # Give a little time for the window to minimize before showing mic test
            QTimer.singleShot(500, lambda: self.show_mic_test(mode))
            
        except Exception as e:
            error_msg = f"Error changing mode: {e}"
            logger.error(error_msg)
            if self.logging_manager:
                self.logging_manager.log_error(error_msg, {
                    'context': 'mode_selection',
                    'selected_mode': mode
                })
            self.show_error("Mode Change Error", error_msg)

    def _handle_dictionary_click(self) -> None:
        """Handle dictionary button click with enhanced error handling."""
        logger.debug("Opening dictionary manager")
        try:
            if not self.dictionary_manager:
                if hasattr(self.excel_handler, 'search_engine'):
                    self.dictionary_manager = DictionaryManager(
                        self.excel_handler.search_engine,
                        self
                    )
                else:
                    raise RuntimeError("Search engine not properly initialized")
            
            self.dictionary_manager.show()
            self.dictionary_manager.raise_()
            self.dictionary_manager.activateWindow()
            self.dictionary_opened.emit()
            
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'dictionary_opened',
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            error_msg = f"Could not open dictionary manager: {e}"
            logger.error(error_msg)
            if self.logging_manager:
                self.logging_manager.log_error(error_msg, {
                    'context': 'dictionary_manager'
                })
            self.show_error("Dictionary Error", error_msg)

    def _perform_document_scan(self) -> None:
        """Perform document scan with progress updates."""
        try:
            self.components.status_label.setText("Scanning document...")
            self.components.progress_bar.show()
            self.components.progress_bar.setValue(0)
            QApplication.processEvents()
            
            # Start scan in steps
            total_steps = 5
            for step in range(total_steps):
                # Perform scan step
                progress = ((step + 1) / total_steps) * 100
                self.components.progress_bar.setValue(int(progress))
                QApplication.processEvents()
                
                if step == 0:
                    self._log_message("Initializing scan...", "info")
                elif step == 1:
                    self._log_message("Analyzing document structure...", "info")
                elif step == 2:
                    self.excel_handler.search_engine.learn_from_document(
                        self.excel_handler.state.workbook,
                        self.excel_handler.state.search_column
                    )
                    self._log_message("Learning patterns...", "info")
                elif step == 3:
                    if self.dictionary_manager and self.dictionary_manager.isVisible():
                        self.dictionary_manager.refresh_dictionary_view()
                    self._log_message("Updating dictionary...", "info")
                elif step == 4:
                    self._log_message("Finalizing scan...", "info")
                
            self.components.status_label.setText("Scan complete")
            self._log_message("Document scan completed successfully!", "success")
            
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'document_scan_complete',
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            raise RuntimeError(f"Error during document scan: {e}")
        finally:
            self.components.progress_bar.hide()

    def show_mic_test(self, mode: str) -> None:
        """Show enhanced microphone test window with logging."""
        logger.debug(f"Running Microphone Test in {mode} mode")
        try:
            if not self.microphone_test_window:
                self.microphone_test_window = EnhancedMicrophoneTest(
                    self.voice_interaction,
                    excel_handler=self.excel_handler,
                    file_coordinator=self.file_coordinator
                )
                # Connect the test completion signal
                if hasattr(self.microphone_test_window, 'test_completed'):
                    self.microphone_test_window.test_completed.connect(self._handle_test_complete)
            
            self.microphone_test_window.set_mode(mode)
            self.microphone_test_window.setWindowFlags(
                self.microphone_test_window.windowFlags() | 
                Qt.WindowStaysOnTopHint
            )
            
            self.microphone_test_window.show()
            
        except Exception as e:
            error_msg = f"Failed to start microphone test: {e}"
            logger.error(error_msg)
            if self.logging_manager:
                self.logging_manager.log_error(error_msg, {
                    'context': 'microphone_test',
                    'mode': mode
                })
            self.show()
            self.show_error("Microphone Test Error", error_msg)
        
    def handle_voice_command(self, command: str) -> None:
        """Handle voice commands with enhanced feedback."""
        logger.debug(f"Processing voice command: {command}")
        try:
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'voice_command',
                    'command': command,
                    'timestamp': datetime.now().isoformat()
                })
            
            if command.lower().startswith("open dictionary"):
                self._handle_dictionary_click()
            elif command.lower().startswith("scan document"):
                self._handle_scan_click()
            else:
                self._log_message(f"Unhandled voice command: {command}", "warning")
                
        except Exception as e:
            error_msg = f"Error processing voice command: {e}"
            logger.error(error_msg)
            if self.logging_manager:
                self.logging_manager.log_error(error_msg, {
                    'context': 'voice_command',
                    'command': command
                })

    def _handle_scan_click(self) -> None:
        """Handle scan button click with progress tracking."""
        logger.debug("Initiating document scan")
        try:
            if not self.excel_handler or not self.excel_handler.state.workbook:
                raise RuntimeError("Please load a workbook first")
            
            reply = QMessageBox.question(
                self,
                'Scan Document',
                'This will scan the current document for new terms. Continue?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self._perform_document_scan()
                
        except Exception as e:
            error_msg = f"Could not start document scan: {e}"
            logger.error(error_msg)
            if self.logging_manager:
                self.logging_manager.log_error(error_msg, {
                    'context': 'document_scan'
                })
            self.show_error("Scan Error", error_msg)

    def _handle_end_click(self) -> None:
        """Handle end button click with cleanup."""
        try:
            logger.debug("End button clicked")
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'application_end',
                    'timestamp': datetime.now().isoformat()
                })
            
            reply = QMessageBox.question(
                self,
                'Confirm Exit',
                'Are you sure you want to exit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.cleanup()
                self.close()
                
        except Exception as e:
            logger.error(f"Error handling end click: {e}")
            self.show_error("Exit Error", str(e))

    def show_file_selection(self):
        """Show the file selection dialog with dictionary manager button."""
        try:
            # Make sure main GUI is visible and has focus
            self.show()
            self.raise_()
            self.activateWindow()
            
            # Short delay to ensure window is active before showing dialog
            QTimer.singleShot(100, self._show_file_dialog)
            
        except Exception as e:
            logger.error(f"Error in file selection: {e}")

    def _show_file_dialog(self):
        """Helper method to show file dialog with dictionary manager button."""
        try:
            from PyQt5.QtWidgets import QFileDialog, QVBoxLayout, QPushButton, QWidget
            
            # Create a custom file dialog
            file_dialog = QFileDialog(self, "Select Excel File for Dugal")
            file_dialog.setNameFilter("Excel Files (*.xlsx *.xlsm)")
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            
            # Create a custom layout for the dictionary button instead of using the dialog's layout directly
            # Most reliable approach is to add it to a separate widget and customize the dialog
            custom_widget = QWidget()
            custom_layout = QVBoxLayout(custom_widget)
            
            # Add dictionary manager button
            dict_button = QPushButton("Dictionary Manager")
            dict_button.setStyleSheet("""
                QPushButton {
                    background-color: #6c5ce7;
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #5f51e8;
                }
            """)
            dict_button.clicked.connect(self._handle_dictionary_click)
            custom_layout.addWidget(dict_button)
            
            # Set layout and show the widget
            file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # Important for custom widgets
            layout = file_dialog.layout()
            if layout:  # Safer check for layout existence
                layout.addWidget(custom_widget)
            
            # Show dialog and get result
            if file_dialog.exec_():
                file_path = file_dialog.selectedFiles()[0]
                
                if file_path:
                    # If you have a file coordinator
                    if hasattr(self, 'file_coordinator') and self.file_coordinator:
                        self.file_coordinator.open_file(file_path)
                    # Or if you're using excel_handler directly
                    elif hasattr(self, 'excel_handler') and self.excel_handler:
                        self.excel_handler.load_excel_file(file_path, read_only=True)
        except Exception as e:
            logger.error(f"Error opening file: {e}")

    def cleanup(self) -> None:
        """Clean up resources with enhanced error handling."""
        logger.debug("Starting GUI cleanup")
        try:
            # Stop monitoring timer
            if hasattr(self, 'monitoring_timer'):
                self.monitoring_timer.stop()
            
            # Clean up microphone test window
            if self.microphone_test_window:
                try:
                    self.microphone_test_window.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up microphone test: {e}")
            
            # Clean up dictionary manager
            if self.dictionary_manager:
                try:
                    self.dictionary_manager.close()
                except Exception as e:
                    logger.error(f"Error closing dictionary manager: {e}")
            
            # Clear image cache
            if hasattr(self, '_image_cache'):
                self._image_cache.clear()
            
            # Final logging
            if self.logging_manager:
                try:
                    self.logging_manager.log_pattern_match({
                        'type': 'gui_cleanup',
                        'components_cleaned': list(self.state.component_status.keys()),
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error in final logging: {e}")
            
            logger.debug("GUI cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during GUI cleanup: {e}")

    def closeEvent(self, event) -> None:
        """Handle window close event with cleanup."""
        try:
            self.cleanup()
            event.accept()
        except Exception as e:
            logger.error(f"Error during close event: {e}")
            event.accept()

def run_gui() -> None:
    """Run the enhanced GUI application."""
    try:
        # Set up exception handling
        def exception_hook(exctype, value, traceback):
            logger.critical("Unhandled exception:", exc_info=(exctype, value, traceback))
            sys._excepthook(exctype, value, traceback)
            sys.exit(1)
        
        # Initialize application
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  # Consistent modern style
        
        # Set up system-wide exception handling
        sys._excepthook = sys.excepthook
        sys.excepthook = exception_hook
        
        # Create and show GUI
        gui = EnhancedDugalGUI()
        gui.show()
        
        # Start event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.critical(f"Critical error running GUI: {e}")
        sys.exit(1)
    finally:
        # Ensure proper cleanup
        try:
            app.quit()
        except:
            pass

def create_gui(
    voice_interaction: Optional[VoiceInteraction] = None,
    one_drive_handler: Optional[OneDriveHandler] = None,
    excel_handler: Optional[ExcelHandler] = None,
    file_coordinator = None,
    logging_manager = None
) -> EnhancedDugalGUI:
    """Create and configure a new GUI instance with the specified components."""
    try:
        # Initialize GUI
        gui = EnhancedDugalGUI(
            voice_interaction=voice_interaction,
            one_drive_handler=one_drive_handler,
            excel_handler=excel_handler,
            file_coordinator=file_coordinator,
            logging_manager=logging_manager
        )
        
        # Set up exception handling for the GUI instance
        gui.error_occurred.connect(lambda title, details: logger.error(
            f"GUI Error - {title}: {details}"
        ))
        
        return gui
        
    except Exception as e:
        logger.critical(f"Failed to create GUI: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run application
    run_gui()

# Backward compatibility
DugalGUI = EnhancedDugalGUI
