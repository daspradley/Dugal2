"""
Logging manager for Dugal Inventory System.
Handles centralized logging, debugging, and performance tracking.
"""

import logging
logger = logging.getLogger(__name__)
import logging.handlers
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

@dataclass
class VoiceMonitoring:
    """Handles real-time monitoring and diagnostics for voice interaction."""
    
    def __init__(self, voice_interaction):
        """Initialize voice monitoring."""
        self.voice_interaction = voice_interaction
        self.logger = logging.getLogger(__name__)
        self.monitoring_active = False
        self.diagnostic_data = {}
        
    def start_monitoring(self) -> None:
        """Start real-time voice system monitoring."""
        try:
            self.monitoring_active = True
            if self.voice_interaction.state.logging_manager:
                self.voice_interaction.state.logging_manager.log_pattern_match({
                    'type': 'monitoring_start',
                    'timestamp': datetime.now().isoformat()
                })
            self._monitor_metrics()
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {e}")
            
    def _monitor_metrics(self) -> None:
        """Monitor key system metrics."""
        try:
            if not self.monitoring_active:
                return
                
            metrics = {
                'audio_quality': self._check_audio_quality(),
                'response_times': self._check_response_times(),
                'error_rates': self._check_error_rates(),
                'learning_effectiveness': self._check_learning_metrics()
            }
            
            self._update_diagnostic_data(metrics)
            
            # Log metrics
            if self.voice_interaction.state.logging_manager:
                self.voice_interaction.state.logging_manager.log_performance({
                    'type': 'monitoring_metrics',
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Error monitoring metrics: {e}")
            
    def _check_audio_quality(self) -> Dict[str, Any]:
        """Check audio system quality."""
        return {
            'signal_strength': self._get_signal_strength(),
            'noise_level': self._get_noise_level(),
            'clarity_score': self._calculate_clarity_score()
        }
        
    def _check_response_times(self) -> Dict[str, Any]:
        """Monitor system response times."""
        return {
            'recognition_latency': self._get_recognition_latency(),
            'processing_time': self._get_processing_time(),
            'synthesis_latency': self._get_synthesis_latency()
        }
        
    def _check_error_rates(self) -> Dict[str, Any]:
        """Monitor error rates across different operations."""
        return {
            'recognition_errors': self._get_recognition_errors(),
            'processing_errors': self._get_processing_errors(),
            'learning_errors': self._get_learning_errors()
        }
        
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        self.logger.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")
        
    def log_final_state(self, state: Dict[str, Any]) -> None:
        """Log final application state."""
        self.logger.info(f"Final state: {json.dumps(state, indent=2)}")
        self.save_stats()  # Save final statistics

    def handle_read_only_refresh(self, file_path: str) -> None:
        """Handle read-only file refresh events."""
        self.log_pattern_match({
            'type': 'read_only_refresh',
            'file': file_path,
            'timestamp': datetime.now().isoformat()
        })

    def get_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        try:
            if not self.diagnostic_data:
                return {'status': 'No diagnostic data available'}
                
            report = {
                'system_status': self._analyze_system_status(),
                'performance_metrics': self.diagnostic_data,
                'recommendations': self._generate_recommendations(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log report generation
            if self.voice_interaction.state.logging_manager:
                self.voice_interaction.state.logging_manager.log_pattern_match({
                    'type': 'diagnostic_report',
                    'report': report,
                    'timestamp': datetime.now().isoformat()
                })
                
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating diagnostic report: {e}")
            return {'error': str(e)}
            
    def _generate_recommendations(self) -> List[str]:
        """Generate system improvement recommendations."""
        recommendations = []
        try:
            if self.diagnostic_data.get('audio_quality', {}).get('noise_level', 0) > 0.7:
                recommendations.append("High noise level detected. Consider adjusting microphone settings.")
                
            if self.diagnostic_data.get('error_rates', {}).get('recognition_errors', 0) > 0.3:
                recommendations.append("High recognition error rate. Check audio input quality.")
                
            if self.diagnostic_data.get('response_times', {}).get('recognition_latency', 0) > 2.0:
                recommendations.append("Slow recognition response times. Check network connectivity.")
                
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            
        return recommendations
        
    def cleanup(self) -> None:
        """Clean up monitoring resources."""
        try:
            self.monitoring_active = False
            if self.voice_interaction.state.logging_manager:
                self.voice_interaction.state.logging_manager.log_pattern_match({
                    'type': 'monitoring_stop',
                    'timestamp': datetime.now().isoformat()
                })
            self.diagnostic_data = {}
            
        except Exception as e:
            self.logger.error(f"Error during monitoring cleanup: {e}")

class LoggingConfig:
    """Configuration for logging system."""
    log_dir: str = "logs"
    debug_log: str = "debug.log"
    error_log: str = "error.log"
    pattern_log: str = "patterns.log"
    stats_file: str = "stats.json"
    performance_log: str = "performance.log"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        'pattern_matches': 0,
        'successful_learns': 0,
        'recognition_rate': 0.0,
        'average_confidence': 0.0,
        'error_count': 0
    })

class LoggingManager:
    """Manages logging and debugging for Dugal system."""
    
    def __init__(self, base_dir=".dugal_data"):
        """Initialize logging manager."""
        self.logger = logging.getLogger(__name__)
        self.base_dir = base_dir
        self.pattern_log_path = os.path.join(base_dir, "logs", "pattern_matches.json")
        self.stats_path = os.path.join(base_dir, "logs", "performance_stats.json")
        
        # Initialize metrics and config
        self.config = SimpleNamespace()
        self.config.metrics = {
            'error_count': 0,
            'pattern_matches': 0,
            'recognition_rate': 0,
            'success_rate': 0
        }
        
        self.performance_metrics = {
            "pattern_matches": 0,
            "successful_commands": 0,
            "failed_commands": 0,
            "response_times": [],
            "error_count": 0,
            "warnings": 0,
            "patterns_matched": 0,
            "performance_index": 0
        }
        
        # Ensure log directories exist
        os.makedirs(os.path.dirname(self.pattern_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        
        # Initialize or load existing logs
        if os.path.exists(self.pattern_log_path):
            with open(self.pattern_log_path, 'r') as f:
                self.pattern_matches = json.load(f)
        else:
            self.pattern_matches = []
            with open(self.pattern_log_path, 'w') as f:
                json.dump(self.pattern_matches, f)
            
    def _serialize_data(self, data: Any) -> Any:
        """Convert data to JSON-serializable format."""
        if isinstance(data, dict):
            return {str(k): self._serialize_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        elif hasattr(data, '__dataclass_fields__'):
            fields = {}
            for field_name in data.__dataclass_fields__:
                value = getattr(data, field_name)
                fields[field_name] = self._serialize_data(value)
            return fields
        elif hasattr(data, '__dict__'):
            return self._serialize_data(data.__dict__)
        elif hasattr(data, '_asdict'):
            return self._serialize_data(data._asdict())
        else:
            try:
                json.dumps(data)  # Test if it's JSON serializable
                return data
            except (TypeError, ValueError):
                return str(data)

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        # Create logs directory if needed
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Configure handlers
        handlers = {
            'debug': (self.config.debug_log, logging.DEBUG),
            'error': (self.config.error_log, logging.ERROR),
            'pattern': (self.config.pattern_log, logging.INFO),
            'performance': (self.config.performance_log, logging.INFO)
        }
        
        # Set up each handler
        for name, (filename, level) in handlers.items():
            handler = logging.handlers.RotatingFileHandler(
                os.path.join(self.config.log_dir, filename),
                maxBytes=self.config.max_log_size,
                backupCount=self.config.backup_count
            )
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logging.getLogger().addHandler(handler)

    def log_pattern_match(self, data: Dict[str, Any]) -> None:
        """Log pattern match data with enhanced serialization."""
        try:
            serialized_data = self._serialize_data(data)
            with open(self.pattern_log_path, 'a') as f:
                json.dump(serialized_data, f, indent=2)
                f.write('\n')
            
            # Update performance metrics
            if serialized_data.get('type') == 'pattern_analysis':
                self.update_performance_metrics(serialized_data)
                
        except Exception as e:
            logger.error(f"Error logging pattern match: {e}")
        
    def log_learning_event(self, event: Dict[str, Any]) -> None:
        """Log learning event."""
        self.logger.info(f"Learning event: {json.dumps(event, indent=2)}")
        if event.get('success', False):
            self.config.metrics['successful_learns'] += 1
            
    def log_recognition_result(self, success: bool, confidence: float) -> None:
        """Log speech recognition result."""
        total = self.config.metrics.get('total_recognitions', 0) + 1
        successful = self.config.metrics.get('successful_recognitions', 0)
        
        if success:
            successful += 1
            
        self.config.metrics.update({
            'total_recognitions': total,
            'successful_recognitions': successful,
            'recognition_rate': (successful / total) * 100,
            'average_confidence': (
                (self.config.metrics.get('average_confidence', 0) * (total - 1) + confidence)
                / total
            )
        })
        
    def log_error(self, error: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log error with context."""
        self.logger.error(f"Error: {error}", extra={'context': context})
        self.config.metrics['error_count'] += 1
        
    def log_performance(self, performance_data: dict) -> None:
        """Log performance metrics."""
        try:
            self.performance_metrics.update(performance_data)
            with open(self.stats_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error logging performance: {e}")

    def log_status_change(self, status: str) -> None:
        """Log status change events."""
        try:
            self.log_pattern_match({
                'type': 'status_change',
                'status': status,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error logging status change: {e}")

    def save_stats(self) -> None:
        """Save statistics in a serialization-safe way."""
        try:
            stats = self._serialize_data(self.performance_metrics)
            with open(self.stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            return {
                'error_count': self.performance_metrics['error_count'],
                'pattern_matches': self.performance_metrics['pattern_matches'],
                'successful_commands': self.performance_metrics['successful_commands'],
                'failed_commands': self.performance_metrics['failed_commands'],
                'performance_index': self.performance_metrics['performance_index']
            }
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return {
                'error_count': 0,
                'pattern_matches': 0,
                'successful_commands': 0,
                'failed_commands': 0,
                'performance_index': 0
            }
            
    def analyze_errors(self) -> List[Dict[str, Any]]:
        """Analyze error patterns from logs."""
        error_patterns = []
        try:
            error_log_path = os.path.join(self.config.log_dir, self.config.error_log)
            if os.path.exists(error_log_path):
                with open(error_log_path, 'r') as f:
                    for line in f:
                        if 'ERROR' in line:
                            error_patterns.append(self._parse_error_line(line))
        except Exception as e:
            self.logger.error(f"Error analyzing logs: {e}")
            
        return error_patterns
        
    def _parse_error_line(self, line: str) -> Dict[str, Any]:
        """Parse error log line into structured data."""
        try:
            parts = line.split(' - ')
            return {
                'timestamp': parts[0],
                'level': parts[2],
                'message': parts[3].strip(),
                'context': json.loads(parts[4]) if len(parts) > 4 else {}
            }
        except Exception:
            return {'message': line.strip()}
            
    def cleanup_start(self):
        """Log start of cleanup process."""
        try:
            self.log_pattern_match({
                'type': 'cleanup_start',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error logging cleanup start: {e}")

    def cleanup(self) -> None:
        """Clean up logging resources."""
        try:
            self.save_stats()
            
            # Close all handlers
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)
                
            self.logger.debug("Logging cleanup completed")
            
        except Exception as e:
            print(f"Error during logging cleanup: {e}")
