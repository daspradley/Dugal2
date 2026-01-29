"""
Enhanced microphone test module for Dugal Inventory System.
Combines robust voice recognition with improved audio visualization and feedback.
"""

import sys
import json
import logging
import wave
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import speech_recognition as sr
import pyaudio
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel,
    QProgressBar, QMessageBox, QApplication, QFileDialog,
    QHBoxLayout, QFrame, QTextEdit, QShortcut
)
from PyQt5.QtCore import pyqtSignal, QTimer, Qt, QMetaObject, Q_ARG
from PyQt5.QtGui import QPainter, QColor, QLinearGradient, QIcon, QKeySequence

from audio_types import AudioResult
from voice_interaction import VoiceInteraction
from excel_handler import ExcelHandler
from dictionary_manager import DictionaryManager

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MicTestComponents:
    """Stores microphone test UI components."""
    microphone_dropdown: QComboBox = None
    level_indicator: 'EnhancedAudioVisualizer' = None
    status_label: QLabel = None
    heard_text_label: QLabel = None
    test_phrase_label: QLabel = None
    test_button: QPushButton = None
    refresh_button: QPushButton = None
    proceed_button: QPushButton = None
    dict_button: QPushButton = None
    play_button: QPushButton = None
    progress_bar: QProgressBar = None
    log_viewer: QTextEdit = None
    
class EnhancedAudioVisualizer(QWidget):
    """Enhanced audio level visualization with color gradients."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMinimumWidth(300)
        self.levels = [0] * 50
        self.setStyleSheet("background-color: black;")
        self._painter = None
        
    def update_level(self, level: float):
        """Update with new audio level with improved thresholding."""
        NOISE_THRESHOLD = 0.02
        SPEECH_SCALING = 2.0
        
        if level < NOISE_THRESHOLD:
            level = 0
        else:
            level = min(1.0, (level - NOISE_THRESHOLD) * SPEECH_SCALING)
        
        self.levels.pop(0)
        self.levels.append(level)
        self.update()
        
    def paintEvent(self, event):
        if self._painter is not None:
            self._painter.end()
            
        self._painter = QPainter(self)
        self._painter.setPen(Qt.NoPen)
        
        width = self.width()
        height = self.height()
        bar_width = int(width / len(self.levels))
        center_y = height // 2
        
        # Draw background
        self._painter.fillRect(0, 0, width, height, QColor(0, 0, 0))
        
        # Draw enhanced waveform with color transitions
        for i, level in enumerate(self.levels):
            bar_height = int(level * (height * 0.8))
            x = int(i * bar_width)
            
            # Color transitions based on audio level
            if level > 0.8:  # Too loud
                color1 = QColor(255, 50, 50)  # Red
                color2 = QColor(255, 100, 100)
            elif level > 0.5:  # Good level
                color1 = QColor(0, 255, 0)  # Green
                color2 = QColor(100, 255, 100)
            else:  # Soft
                color1 = QColor(0, 200, 255)  # Blue
                color2 = QColor(100, 200, 255)
            
            gradient = QLinearGradient(
                x, center_y - bar_height//2,
                x, center_y + bar_height//2
            )
            gradient.setColorAt(0, color1)
            gradient.setColorAt(0.5, color2)
            gradient.setColorAt(1, color1)
            
            self._painter.setBrush(gradient)
            self._painter.drawRect(
                x,
                center_y - bar_height//2,
                max(1, bar_width - 1),
                max(1, bar_height)
            )
            
        self._painter.end()
        self._painter = None

class EnhancedMicrophoneTest(QWidget):
    """Enhanced microphone test with improved visualization and feedback."""
    
    test_completed = pyqtSignal(object)  # Emits AudioResult
    proceed_clicked = pyqtSignal()  # Signal emitted when proceed is clicked
    level_update = pyqtSignal(float)  # Signal for audio level updates
    dictionary_opened = pyqtSignal()  # Signal for dictionary events
    
    def __init__(
        self, 
        voice_interaction: VoiceInteraction,
        file_coordinator = None,
        excel_handler: Optional[ExcelHandler] = None
    ) -> None:
        super().__init__()
        logger.debug("Initializing Enhanced Microphone Test")
        self.logger = logging.getLogger(__name__)
        self.voice_interaction = voice_interaction
        self.file_coordinator = file_coordinator
        self.excel_handler = excel_handler or ExcelHandler(dugal=self.voice_interaction.state.dugal)
        self.logging_manager = voice_interaction.state.logging_manager if voice_interaction else None
        self.test_history = []
        self.test_log = []
        self.recognizer = sr.Recognizer()
        self.is_testing = False
        self.test_successful = False
        self.current_mic_index = None
        self.mode = None
        self.log_display = None
        self.components = MicTestComponents()
        self.dictionary_manager = None
        
        # Audio recording setup
        self.recorded_audio = None
        self.RATE = 44100
        self.CHUNK = 1024
        self.AUDIO_FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.SMOOTHING = 0.7
        self.last_level = 0
        self.stream = None
        self.pyaudio_instance = None
        
        self.setup_ui()
        self.level_update.connect(self._update_level_indicator)
        
        # Add shortcut for bypassing test (e.g., Ctrl+B)
        self.bypass_shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        self.bypass_shortcut.activated.connect(self._bypass_test)

        if self.logging_manager:
            self.logging_manager.log_pattern_match({
                'type': 'microphone_test_init',
                'timestamp': datetime.now().isoformat()
            })

    def set_mode(self, mode: str):
        """Set the mode based on user selection."""
        self.mode = mode
        logger.debug(f"Microphone test mode set to: {mode}")

    def setup_ui(self) -> None:
        """Set up the enhanced user interface with improved feedback and controls."""
        self.setWindowTitle("Dugal's Enhanced Microphone Test")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Header section
        header_frame = self._create_header_section()
        layout.addWidget(header_frame)

        # Status and instructions
        status_frame = self._create_status_section()
        layout.addWidget(status_frame)

        # Microphone selection area
        mic_frame = self._create_microphone_section()
        layout.addWidget(mic_frame)

        # Test phrase and heard text
        phrase_frame = self._create_phrase_section()
        layout.addWidget(phrase_frame)

        # Progress and visualization
        progress_frame = self._create_progress_section()
        layout.addWidget(progress_frame)

        # Control buttons
        button_frame = self._create_button_section()
        layout.addWidget(button_frame)

        # Log viewer
        log_frame = self._create_log_section()
        layout.addWidget(log_frame)

    def _create_header_section(self) -> QFrame:
        """Create the header section with title and subtitle."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        layout = QVBoxLayout(frame)
        
        title = QLabel("Dugal's Microphone Setup")
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("Let's make sure I can hear you properly")
        subtitle.setStyleSheet("""
            QLabel {
                color: #bdc3c7;
                font-size: 16px;
            }
        """)
        subtitle.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return frame

    def _create_status_section(self) -> QFrame:
        """Create the status section with dynamic feedback."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        layout = QVBoxLayout(frame)
        
        self.components.status_label = QLabel("Select your microphone and test it below.")
        self.components.status_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        self.components.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.components.status_label)
        
        return frame

    def _create_microphone_section(self) -> QFrame:
        """Create the microphone selection section."""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        
        mic_label = QLabel("Select Microphone:")
        mic_label.setStyleSheet("font-weight: bold;")
        
        self.components.microphone_dropdown = QComboBox()
        self.components.microphone_dropdown.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                min-width: 300px;
            }
        """)
        
        self.components.refresh_button = QPushButton("⟳")
        self.components.refresh_button.setToolTip("Refresh microphone list")
        self.components.refresh_button.clicked.connect(self.populate_microphones)
        self.components.refresh_button.setStyleSheet("""
            QPushButton {
                padding: 5px 10px;
                border-radius: 3px;
                background-color: #3498db;
                color: white;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        layout.addWidget(mic_label)
        layout.addWidget(self.components.microphone_dropdown)
        layout.addWidget(self.components.refresh_button)
        
        self.populate_microphones()
        return frame

    def _create_phrase_section(self) -> QFrame:
        """Create the test phrase and feedback section."""
        frame = QFrame()
        layout = QVBoxLayout(frame)
        
        self.components.test_phrase_label = QLabel(
            "Test phrase: 'The quick brown fox jumped over the lazy silver dog.'"
        )
        self.components.test_phrase_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #e8f5e9;
                border-radius: 5px;
                font-style: italic;
            }
        """)
        
        self.components.heard_text_label = QLabel("Heard text: (waiting for input...)")
        self.components.heard_text_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 5px;
            }
        """)
        
        layout.addWidget(self.components.test_phrase_label)
        layout.addWidget(self.components.heard_text_label)
        
        return frame

    def _create_progress_section(self) -> QFrame:
        """Create the progress and visualization section."""
        frame = QFrame()
        layout = QVBoxLayout(frame)
        
        # Audio level visualizer
        self.components.level_indicator = EnhancedAudioVisualizer(self)
        layout.addWidget(self.components.level_indicator)
        
        # Progress bar for recording/listening
        progress_frame = QFrame()
        progress_layout = QHBoxLayout(progress_frame)
        
        self.components.progress_bar = QProgressBar()
        self.components.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                border-radius: 2px;
            }
        """)
        self.components.progress_bar.setMinimum(0)
        self.components.progress_bar.setMaximum(100)
        
        progress_layout.addWidget(QLabel("Progress:"))
        progress_layout.addWidget(self.components.progress_bar)
        layout.addWidget(progress_frame)
        
        return frame

    def _create_button_section(self) -> QFrame:
        """Create the control buttons section."""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setSpacing(10)
        
        # Test button
        self.components.test_button = self._create_styled_button(
            "Test Microphone",
            "#2ecc71",
            "#27ae60"
        )
        self.components.test_button.clicked.connect(self.test_microphone)
        
        # Play button
        self.components.play_button = self._create_styled_button(
            "Play Recording",
            "#e67e22",
            "#d35400"
        )
        self.components.play_button.clicked.connect(self.play_recorded_audio)
        self.components.play_button.setEnabled(False)
        
        # Dictionary button
        self.components.dict_button = self._create_styled_button(
            "Dictionary",
            "#6c5ce7",
            "#5f51e8"
        )
        self.components.dict_button.clicked.connect(self.show_dictionary_manager)
        
        # Proceed button
        self.components.proceed_button = self._create_styled_button(
            "Proceed",
            "#3498db",
            "#2980b9"
        )
        self.components.proceed_button.clicked.connect(self.on_proceed)
        self.components.proceed_button.setEnabled(False)
        
        # Add buttons to layout
        layout.addWidget(self.components.test_button)
        layout.addWidget(self.components.play_button)
        layout.addWidget(self.components.dict_button)
        layout.addWidget(self.components.proceed_button)
        
        return frame

    def _create_log_section(self) -> QFrame:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        
        # Create log viewer first
        if not hasattr(self.components, 'log_viewer') or not self.components.log_viewer:
            self.components.log_viewer = QTextEdit()
            self.components.log_viewer.setReadOnly(True)
            self.components.log_viewer.setMaximumHeight(100)
            self.components.log_viewer.setStyleSheet("""
                QTextEdit {
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 3px;
                    font-family: monospace;
                }
            """)
        
        # Then add to layout
        layout.addWidget(self.components.log_viewer)
        
        return frame

    def show_dictionary_manager(self) -> None:
        """Show the dictionary management interface with enhanced logging."""
        # Make sure logging components are initialized
        if not hasattr(self, 'components') or not self.components or not hasattr(self.components, 'log_viewer'):
            logger.error("Components not properly initialized")
            return
            
        self.log_message("Opening dictionary manager...", "info")
        try:
            if not self.dictionary_manager:
                # Get search engine from global registry first
                from global_registry import GlobalRegistry
                search_engine = GlobalRegistry.get('search_engine')
                
                # If not in registry, try excel handler
                if not search_engine and hasattr(self.excel_handler, 'search_engine'):
                    search_engine = self.excel_handler.search_engine
                    # Register it in the registry
                    GlobalRegistry.register('search_engine', search_engine)
                    
                if search_engine:
                    self.dictionary_manager = DictionaryManager(
                        search_engine,
                        self
                    )
                else:
                    self.log_message("Search engine not initialized", "error")
                    QMessageBox.warning(
                        self,
                        "Error",
                        "Search engine not properly initialized"
                    )
                    return
            
            self.dictionary_manager.show()
            self.dictionary_manager.raise_()
            self.dictionary_manager.activateWindow()
            self.dictionary_opened.emit()
            self.log_message("Dictionary manager opened", "success")
            
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'dictionary_opened',
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            error_msg = f"Could not open dictionary manager: {str(e)}"
            logger.error(error_msg)
            if hasattr(self, 'log_message'):
                self.log_message(error_msg, "error")
            QMessageBox.warning(self, "Error", error_msg)

    def log_message(self, message: str, level: str = "info") -> None:
        """Add a message to the log viewer with timestamp."""
        # Initialize log viewer if missing
        if not hasattr(self.components, 'log_viewer') or not self.components.log_viewer:
            self.components.log_viewer = QTextEdit()
            self.components.log_viewer.setReadOnly(True)
            self.components.log_viewer.setMaximumHeight(100)
            self.components.log_viewer.setStyleSheet("""
                QTextEdit {
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 3px;
                    font-family: monospace;
                }
            """)

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
                'type': 'mic_test_log',
                'message': message,
                'level': level,
                'timestamp': datetime.now().isoformat()
            })

    def _create_styled_button(self, text: str, base_color: str, hover_color: str) -> QPushButton:
        """Create a consistently styled button."""
        button = QPushButton(text)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {base_color};
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 120px;
                min-height: 40px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
            }}
        """)
        return button

    def _clear_logs(self) -> None:
        """Clear the log viewer."""
        self.components.log_viewer.clear()
        if self.logging_manager:
            self.logging_manager.log_pattern_match({
                'type': 'logs_cleared',
                'timestamp': datetime.now().isoformat()
            })

    def start_level_monitoring(self) -> None:
        """Start monitoring microphone levels with enhanced error handling."""
        try:
            # Clean up existing monitoring
            self.stop_level_monitoring()

            self.pyaudio_instance = pyaudio.PyAudio()
            
            def audio_callback(in_data, frame_count, time_info, status):
                try:
                    data = np.frombuffer(in_data, dtype=np.float32)
                    level = np.sqrt(np.mean(data**2))
                    self.last_level = (self.SMOOTHING * self.last_level + 
                                     (1 - self.SMOOTHING) * level)
                    self.level_update.emit(self.last_level)
                except Exception as e:
                    self.log_message(f"Audio callback error: {e}", "error")
                return (in_data, pyaudio.paContinue)
            
            mic_index = self.components.microphone_dropdown.currentData()
            self.stream = self.pyaudio_instance.open(
                format=self.AUDIO_FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=mic_index,
                frames_per_buffer=self.CHUNK,
                stream_callback=audio_callback
            )
            
            self.stream.start_stream()
            self.log_message("Audio monitoring started", "success")
            
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'level_monitoring_start',
                    'device_index': mic_index,
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            self.log_message(f"Failed to start audio monitoring: {e}", "error")
            self.stop_level_monitoring()

    def stop_level_monitoring(self) -> None:
        """Stop monitoring microphone levels."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                self.log_message(f"Error stopping audio stream: {e}", "error")
            self.stream = None
            
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception as e:
                self.log_message(f"Error terminating PyAudio: {e}", "error")
            self.pyaudio_instance = None

        if self.logging_manager:
            self.logging_manager.log_pattern_match({
                'type': 'level_monitoring_stop',
                'timestamp': datetime.now().isoformat()
            })
        self.log_message("Audio monitoring stopped", "info")

    def test_microphone(self) -> None:
        """Activate the microphone for testing with enhanced feedback."""
        if self.is_testing:
            return

        self.is_testing = True
        self.components.test_button.setEnabled(False)
        self.components.proceed_button.setEnabled(False)
        self.components.dict_button.setEnabled(False)
        self.components.play_button.setEnabled(False)
        self.components.progress_bar.setValue(0)

        if self.logging_manager:
            self.logging_manager.log_pattern_match({
                'type': 'microphone_test_start',
                'device_index': self.current_mic_index,
                'timestamp': datetime.now().isoformat()
            })

        try:
            self.log_message("Starting microphone test...", "info")
            self.components.status_label.setText("Preparing microphone test...")
            QApplication.processEvents()

            # Pause level monitoring during test
            current_stream = self.stream
            if current_stream:
                current_stream.stop_stream()

            self.current_mic_index = self.components.microphone_dropdown.currentData()
            mic_index = self.current_mic_index

            if self.check_microphone(mic_index):
                with sr.Microphone(device_index=mic_index) as source:
                    self.components.status_label.setText("Adjusting for ambient noise...")
                    self.log_message("Adjusting for ambient noise...", "info")
                    QApplication.processEvents()
                    
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    self.components.progress_bar.setValue(20)
                    
                    self.components.status_label.setText("Listening... please speak.")
                    self.log_message("Listening for speech...", "info")
                    QApplication.processEvents()
                    
                    try:
                        # Create a timer for the progress bar
                        total_time = 5  # seconds
                        steps = 50
                        step_time = total_time * 1000 / steps
                        current_progress = 20
                        
                        def update_progress():
                            nonlocal current_progress
                            if current_progress < 90:
                                current_progress += (70 / steps)
                                self.components.progress_bar.setValue(int(current_progress))
                        
                        # Start progress updates
                        timer = QTimer(self)
                        timer.timeout.connect(update_progress)
                        timer.start(int(step_time))
                        
                        audio = self.recognizer.listen(source, timeout=8)
                        self.recorded_audio = audio  # Save for playback
                        timer.stop()
                        self.components.progress_bar.setValue(90)
                        
                        self._process_test_audio(audio)
                        
                    except sr.WaitTimeoutError:
                        self.log_message("Listening timed out", "error")
                        self.show_error("Listening timed out. Please try again.")
            else:
                self.log_message("Failed to access microphone", "error")
                self.show_error("Failed to access microphone. Please check your device settings.")

        except Exception as e:
            logger.error(f"Microphone test error: {str(e)}")
            self.log_message(f"Test error: {str(e)}", "error")
            self.show_error(f"Microphone error: {str(e)}")
            
        finally:
            self.is_testing = False
            self.components.test_button.setEnabled(True)
            self.components.dict_button.setEnabled(True)
            self.components.progress_bar.setValue(100)
            
            # Resume level monitoring
            if current_stream:
                try:
                    current_stream.start_stream()
                except:
                    self.start_level_monitoring()

    def _process_test_audio(self, audio: sr.AudioData) -> None:
        """Process recorded audio with enhanced feedback and logging."""
        try:
            self.components.status_label.setText("Processing speech...")
            self.log_message("Processing recorded audio...", "info")
            QApplication.processEvents()
            
            heard_text = self.recognizer.recognize_google(audio)
            self.components.heard_text_label.setText(f"Heard text: {heard_text}")
            self.components.proceed_button.setEnabled(True)
            self.components.play_button.setEnabled(True)
            self.test_successful = True
            
            # Calculate similarity between test phrase and heard text
            similarity = self._calculate_text_similarity(
                "the quick brown fox jumped over the lazy silver dog",
                heard_text.lower()
            )
            
            if similarity > 0.8:
                self.log_message("Excellent speech recognition!", "success")
                status_message = "Test successful! Great pronunciation."
            elif similarity > 0.6:
                self.log_message("Good speech recognition", "success")
                status_message = "Test successful! Acceptable pronunciation."
            else:
                self.log_message("Speech recognized but accuracy could be improved", "warning")
                status_message = "Test completed. Consider speaking more clearly."

            self.components.status_label.setText(status_message)
            
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'speech_recognition_result',
                    'heard_text': heard_text,
                    'similarity_score': similarity,
                    'timestamp': datetime.now().isoformat()
                })
            
            #Create AudioResult but DON'T emit the signal here-- emit on_proceed
            self.audio_result = AudioResult.from_audio_data(audio)
            self.audio_result.recognized_text = heard_text
            
        except sr.UnknownValueError:
            self.components.heard_text_label.setText("Heard text: Could not understand speech.")
            self.test_successful = False
            self.log_message("Speech not recognized", "error")
            self.components.status_label.setText("Speech not recognized. Please try again.")
            
        except sr.RequestError as e:
            self.show_error(f"Recognition error: {str(e)}")
            self.test_successful = False
            self.log_message(f"Recognition service error: {str(e)}", "error")
            self.components.status_label.setText("Recognition service error. Please try again.")

    def play_recorded_audio(self) -> None:
        """Play back the recorded audio for verification."""
        if not self.recorded_audio:
            self.log_message("No recorded audio to play", "warning")
            return

        try:
            self.components.play_button.setEnabled(False)
            self.components.status_label.setText("Playing recorded audio...")
            self.log_message("Playing recorded audio...", "info")
            
            # Convert audio data to format suitable for playback
            CHUNK = 1024
            audio_data = self.recorded_audio.get_raw_data()
            
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(self.recorded_audio.sample_width),
                channels=1,
                rate=self.recorded_audio.sample_rate,
                output=True
            )

            # Break audio data into chunks and play
            start = 0
            while start < len(audio_data):
                chunk = audio_data[start:start + CHUNK]
                stream.write(chunk)
                start += CHUNK

            stream.stop_stream()
            stream.close()
            p.terminate()
            
            self.components.play_button.setEnabled(True)
            self.components.status_label.setText("Playback completed")
            self.log_message("Audio playback completed", "success")

        except Exception as e:
            self.log_message(f"Error playing audio: {str(e)}", "error")
            self.components.play_button.setEnabled(True)
            self.show_error(f"Playback error: {str(e)}")

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using character-level comparison."""
        text1 = ''.join(c.lower() for c in text1 if c.isalnum())
        text2 = ''.join(c.lower() for c in text2 if c.isalnum())
        
        if not text1 or not text2:
            return 0.0

        matches = sum(a == b for a, b in zip(text1, text2))
        length = max(len(text1), len(text2))
        return matches / length if length > 0 else 0.0

    def _update_level_indicator(self, level: float) -> None:
        """Update the level indicator with visual feedback."""
        if self.components.level_indicator:
            self.components.level_indicator.update_level(level)
            
            # Add visual feedback for audio levels
            if level > 0.8:
                self.components.status_label.setStyleSheet("QLabel { color: #c0392b; font-weight: bold; }")
                self.components.status_label.setText("Audio level too high!")
            elif level > 0.1:
                self.components.status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")
                if not self.is_testing:
                    self.components.status_label.setText("Good audio level")
            else:
                self.components.status_label.setStyleSheet("QLabel { color: #2c3e50; font-weight: bold; }")
                if not self.is_testing:
                    self.components.status_label.setText("Waiting for audio...")

    def check_microphone(self, mic_index: int) -> bool:
        """Check if the selected microphone is available and working."""
        p = None
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=self.AUDIO_FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=mic_index,
                frames_per_buffer=self.CHUNK
            )
            
            # Test reading from the microphone
            data = stream.read(self.CHUNK)
            if not data:
                raise Exception("No data received from microphone")
                
            stream.close()
            self.log_message("Microphone check successful", "success")
            return True
            
        except Exception as e:
            self.log_message(f"Microphone check failed: {e}", "error")
            if self.logging_manager:
                self.logging_manager.log_error(str(e), {
                    'context': 'microphone_check',
                    'device_index': mic_index
                })
            return False
            
        finally:
            if p:
                try:
                    p.terminate()
                except:
                    pass

    def populate_microphones(self) -> None:
        """Populate the dropdown with available microphone input devices only."""
        try:
            current_selection = self.components.microphone_dropdown.currentText()
            self.components.microphone_dropdown.clear()
            
            # Get PyAudio instance
            p = pyaudio.PyAudio()
            
            # Find input devices only
            microphones = []
            seen_names = set()  # Track names we've already seen
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                # Only include if it's an input device (has input channels)
                if device_info.get('maxInputChannels', 0) > 0:
                    name = device_info.get('name', f"Unknown Device {i}")
                    
                    # Handle duplicate names
                    if name in seen_names:
                        # Check if it's significantly different from devices we've seen
                        # (e.g., different sample rates might matter)
                        continue
                    
                    seen_names.add(name)
                    microphones.append((i, name))
            
            # Add device names to dropdown
            for idx, name in microphones:
                # Store the device index as item data
                self.components.microphone_dropdown.addItem(name, idx)
            
            # Try to restore previous selection
            if current_selection:
                index = self.components.microphone_dropdown.findText(current_selection)
                if index >= 0:
                    self.components.microphone_dropdown.setCurrentIndex(index)
            
            p.terminate()
            
            self.log_message(f"Found {len(microphones)} input devices", "info")
            
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'microphones_populated',
                    'count': len(microphones),
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            self.log_message(f"Error populating microphones: {e}", "error")
            self.components.microphone_dropdown.addItem("No microphones found")
            if self.logging_manager:
                self.logging_manager.log_error(str(e), {
                    'context': 'populate_microphones'
                })

    def on_proceed(self) -> None:
        """Handle actions when the 'Proceed' button is clicked."""
        try:
            self.log_message("Proceeding with setup...", "info")

            # Only show confirmation if mic hasn't been successfully tested
            if not self.test_successful:
                reply = QMessageBox.question(
                    self,
                    'Skip Microphone Test?',
                    'Are you sure you want to skip testing the microphone?',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    self.test_microphone()
                    return
                    
                self.log_message("User chose to skip microphone test", "warning")
                if self.voice_interaction:
                    self.voice_interaction.speak(
                        "You're skipping the test, are you? Ackh, looks like we have a "
                        "brave soul here. But you know better. Of course you do. So let's "
                        "just skip the wee test instead of saying one little bity phrase. Fookin' Idiot."
                    )

            # Now emit the test_completed signal with the saved audio result
            if hasattr(self, 'audio_result') and self.audio_result:
                self.test_completed.emit(self.audio_result)
            else:
                # Create a dummy result for skipped tests
                dummy_result = AudioResult(True, b'', 16000, 2)
                dummy_result.recognized_text = "Test skipped"
                self.test_completed.emit(dummy_result)

            # Stop audio monitoring
            self.stop_level_monitoring()
            
            # Important: make sure window is closed, not just hidden
            self.close()  # Close instead of hide
            
            # Save dictionary state if needed (keep this part for proper cleanup)
            if self.dictionary_manager and hasattr(self.excel_handler, 'search_engine'):
                try:
                    self.excel_handler.search_engine.save_current_state()
                    self.log_message("Dictionary state saved", "success")
                except Exception as e:
                    self.log_message(f"Error saving dictionary state: {e}", "error")
            
            # Clean up resources before emitting signal
            self.cleanup()
            
            # Emit signal AFTER cleanup
            self.proceed_clicked.emit()
            
            # Schedule deletion for after event processing completes
            self.deleteLater()

        except Exception as e:
            error_msg = f"Error in proceed operation: {str(e)}"
            self.log_message(error_msg, "error")
            QMessageBox.critical(self, "Error", error_msg)
            
    def show_error(self, message: str) -> None:
        """Display an error message with enhanced formatting and logging."""
        self.log_message(message, "error")
        
        error_box = QMessageBox(self)
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Error")
        error_box.setText(message)
        error_box.setStandardButtons(QMessageBox.Ok)
        
        # Update status label with error
        self.components.status_label.setText("Error: " + message)
        self.components.status_label.setStyleSheet("""
            QLabel {
                color: #e74c3c;
                font-weight: bold;
            }
        """)
        
        if self.logging_manager:
            self.logging_manager.log_error(message, {
                'context': 'microphone_test',
                'component': 'error_display'
            })
        
        error_box.exec_()

    def cleanup(self) -> None:
        """Clean up resources before shutdown."""
        self.log_message("Starting cleanup...", "info")
        try:
            # Stop audio monitoring
            self.stop_level_monitoring()
            
            # Clean up dictionary manager
            if self.dictionary_manager:
                try:
                    if not self.dictionary_manager.isDestroyed(): #Add check
                        self.dictionary_manager.close()
                    self.log_message("Dictionary manager closed", "success")
                except Exception as e:
                    self.log_message(f"Error closing dictionary manager: {e}", "error")
            
            # Clean up PyAudio
            if hasattr(self, 'pyaudio_instance') and self.pyaudio_instance:
                try:
                    self.pyaudio_instance.terminate()
                    self.log_message("Audio system terminated", "success")
                except Exception as e:
                    self.log_message(f"Error terminating PyAudio: {e}", "error")
                    
            # Save final state if needed
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'microphone_test_cleanup',
                    'final_state': {
                        'test_successful': self.test_successful,
                        'current_mic_index': self.current_mic_index,
                        'mode': self.mode
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            self.log_message("Cleanup completed successfully", "success")
            
        except Exception as e:
            error_msg = f"Error during cleanup: {e}"
            self.log_message(error_msg, "error")
            logger.error(error_msg)

    def _bypass_test(self) -> None:
        """Bypass the microphone test with a keyboard shortcut."""
        self.logger.debug("Microphone test bypassed with keyboard shortcut")
        self.log_message("Microphone test bypassed with keyboard shortcut", "warning")
        
        # Create a dummy result for skipped tests
        dummy_result = AudioResult(True, b'', 16000, 2)
        dummy_result.recognized_text = "Test bypassed with keyboard shortcut"
        self.test_completed.emit(dummy_result)
        
        # Stop audio monitoring
        self.stop_level_monitoring()
        
        # Close the window
        self.close()
        
        # Emit signal AFTER cleanup
        self.proceed_clicked.emit()

    def closeEvent(self, event) -> None:
        """Handle cleanup when window is closed."""
        try:
            self.log_message("Closing microphone test...", "info")
            self.cleanup()
            event.accept()
        except Exception as e:
            error_msg = f"Error during close: {e}"
            self.log_message(error_msg, "error")
            logger.error(error_msg)
            event.accept()


def run_mic_test() -> None:
    """Test function to run microphone test independently."""
    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)
    voice_interaction = VoiceInteraction()
    mic_test = EnhancedMicrophoneTest(voice_interaction)
    mic_test.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_mic_test()
