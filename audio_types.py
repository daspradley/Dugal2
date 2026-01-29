"""
Audio types and utilities for Dugal Inventory System.
Defines data structures for handling audio processing results.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import speech_recognition as sr

@dataclass
class AudioResult:
    """
    Class to hold audio processing results.
    
    Attributes:
        success: Whether audio processing was successful
        data: Raw audio data bytes
        error: Error message if processing failed
        sample_rate: Audio sample rate in Hz
        sample_width: Width of audio samples in bytes
        recognized_text: Optional text from speech recognition
        metadata: Optional dictionary for additional data
    """
    success: bool
    data: Optional[bytes] = None
    error: Optional[str] = None
    sample_rate: Optional[int] = None
    sample_width: Optional[int] = None
    recognized_text: Optional[str] = None
    metadata: Dict[str, Any] = None

    @classmethod
    def from_audio_data(cls, audio_data: sr.AudioData) -> 'AudioResult':
        """
        Create AudioResult from sr.AudioData object.
        
        Args:
            audio_data: Speech recognition audio data
            
        Returns:
            AudioResult: New instance with data from audio_data
        """
        return cls(
            success=True,
            data=audio_data.get_raw_data(),
            sample_rate=audio_data.sample_rate,
            sample_width=audio_data.sample_width,
            metadata={}
        )

    def get(self, param=None):
        """
        Compatibility method to support legacy .get() calls with an optional parameter.
        If param is specified, tries to access that attribute, otherwise returns data.
        """
        if param is None:
            return self.data
        elif hasattr(self, param):
            return getattr(self, param)
        else:
            return None

@dataclass
class DictationResult:
    """
    Enhanced result type for dictation with learning capabilities.
    
    Attributes:
        audio_result: The base AudioResult
        learned_terms: Dictionary of recognized terms to dictionary entries
        context_data: Additional context for the dictation
    """
    audio_result: AudioResult
    learned_terms: Dict[str, str] = None
    context_data: Dict[str, Any] = None

    def has_learned_terms(self) -> bool:
        """Check if dictation resulted in any learned terms."""
        return bool(self.learned_terms)
    
    def get_learned_variations(self) -> Dict[str, str]:
        """Get variations of terms learned during dictation."""
        variations = {}
        if self.learned_terms:
            for spoken, written in self.learned_terms.items():
                variations[spoken] = written
        return variations

@dataclass
class AudioConfig:
    """
    Configuration for audio processing.
    
    Attributes:
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        chunk_size: Size of audio chunks for processing
        format_type: Audio format (e.g., 'float32')
        noise_threshold: Threshold for noise detection
        speech_threshold: Threshold for speech detection
    """
    sample_rate: int = 44100
    channels: int = 1
    chunk_size: int = 1024
    format_type: str = 'float32'
    noise_threshold: float = 0.02
    speech_threshold: float = 0.1
    
    @classmethod
    def default_config(cls) -> 'AudioConfig':
        """Get default audio configuration."""
        return cls()
