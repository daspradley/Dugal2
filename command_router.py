"""
Command router module for Dugal Inventory System.
Provides a flexible system for routing voice commands to appropriate handlers.
"""

import re
import logging
from typing import Dict, Callable, Any, List, Optional, Pattern, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class CommandMatch:
    """Represents a matched command and its parameters."""
    handler: Callable
    command_text: str
    pattern: str
    groups: Tuple = field(default_factory=tuple)
    named_groups: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RouterResult:
    """Result of command routing."""
    success: bool
    command_match: Optional[CommandMatch] = None
    error: Optional[str] = None
    alternative_matches: List[CommandMatch] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CommandRouter:
    """Routes voice commands to appropriate handlers."""
    
    def __init__(self, logging_manager=None):
        """
        Initialize the command router.
        
        Args:
            logging_manager: Optional logging manager
        """
        self.handlers = {}  # Pattern string to handler function
        self.regex_patterns = {}  # Pattern string to compiled regex
        self.fallback_handler = None
        self.unknown_handler = None
        self.preprocessors = []
        self.logging_manager = logging_manager
        self.command_history = []
        self.recent_matches = []
        self.max_history = 100
        
        logger.debug("Command router initialized")
    
    def register_handler(self, command_pattern: str, handler_func: Callable,
                        preprocessor: Optional[Callable] = None) -> None:
        """
        Register a handler for a command pattern.
        
        Args:
            command_pattern: Regex pattern to match commands
            handler_func: Function to handle matching commands
            preprocessor: Optional function to preprocess commands before matching
        """
        try:
            # Compile the regex pattern
            regex = re.compile(command_pattern, re.IGNORECASE)
            
            self.handlers[command_pattern] = handler_func
            self.regex_patterns[command_pattern] = regex
            
            logger.debug(f"Registered handler for pattern: {command_pattern}")
            
        except Exception as e:
            logger.error(f"Error registering handler for pattern '{command_pattern}': {e}")
            raise
    
    def register_inventory_handler(self, handler_func: Callable) -> None:
        """
        Register a handler for inventory item commands.
        
        Args:
            handler_func: Function to handle inventory item commands
        """
        self.inventory_handler = handler_func
        logger.debug("Registered inventory item handler")

    def register_fallback(self, handler_func: Callable) -> None:
        """
        Register a fallback handler for when a command partially matches.
        
        Args:
            handler_func: Function to handle partial matches
        """
        self.fallback_handler = handler_func
        logger.debug("Registered fallback handler")
    
    def register_unknown(self, handler_func: Callable) -> None:
        """
        Register a handler for unknown commands.
        
        Args:
            handler_func: Function to handle unknown commands
        """
        self.unknown_handler = handler_func
        logger.debug("Registered unknown command handler")
    
    def register_preprocessor(self, preprocessor_func: Callable) -> None:
        """
        Register a preprocessor for commands.
        
        Args:
            preprocessor_func: Function to preprocess commands
        """
        self.preprocessors.append(preprocessor_func)
        logger.debug("Registered command preprocessor")
    
    def route_command(self, command_text: str) -> RouterResult:
        """
        Route a command to the appropriate handler.
        
        Args:
            command_text: Text of the command to route
            
        Returns:
            RouterResult: Result of command routing
        """
        if not command_text:
            return RouterResult(
                success=False,
                error="Empty command text"
            )
            
        original_command = command_text
        command_text = command_text.strip()
        
        # Add to history
        self.command_history.append({
            'text': command_text,
            'timestamp': datetime.now().isoformat(),
            'processed': False
        })
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)
        
        logger.debug(f"Routing command: '{command_text}'")
        
        # Apply preprocessors
        processed_text = command_text
        for preprocessor in self.preprocessors:
            try:
                processed_text = preprocessor(processed_text)
            except Exception as e:
                logger.error(f"Error in command preprocessor: {e}")
        
        # Track close/alternative matches for fallback
        alternative_matches = []
        
        # Try to match against registered patterns
        for pattern_str, regex in self.regex_patterns.items():
            try:
                match = regex.search(processed_text)
                if match:
                    handler = self.handlers[pattern_str]
                    
                    # Create match object with groups and named groups
                    command_match = CommandMatch(
                        handler=handler,
                        command_text=processed_text,
                        pattern=pattern_str,
                        groups=match.groups() if match.groups() else (),
                        named_groups=match.groupdict()
                    )
                    
                    # Update command history
                    self.command_history[-1]['processed'] = True
                    self.command_history[-1]['matched_pattern'] = pattern_str
                    
                    # Add to recent matches
                    self.recent_matches.append(command_match)
                    if len(self.recent_matches) > 10:
                        self.recent_matches.pop(0)
                    
                    # Log the match
                    if self.logging_manager:
                        self.logging_manager.log_pattern_match({
                            'type': 'command_matched',
                            'command': processed_text,
                            'pattern': pattern_str,
                            'groups': list(match.groups()) if match.groups() else [],
                            'named_groups': match.groupdict(),
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    logger.debug(f"Command matched pattern: {pattern_str}")
                    
                    # Return successful result
                    return RouterResult(
                        success=True,
                        command_match=command_match
                    )
                
                # Check for potential partial matches
                if pattern_str.lower() in processed_text.lower() or \
                   any(word in pattern_str.lower() for word in processed_text.lower().split()):
                    # This is a partial match, add to alternatives
                    confidence = self._calculate_match_confidence(pattern_str, processed_text)
                    if confidence > 0.3:  # Threshold for alternative matches
                        handler = self.handlers[pattern_str]
                        alt_match = CommandMatch(
                            handler=handler,
                            command_text=processed_text,
                            pattern=pattern_str,
                            confidence=confidence
                        )
                        alternative_matches.append(alt_match)
                
            except Exception as e:
                logger.error(f"Error matching pattern '{pattern_str}': {e}")
        
        # No direct match found
        
        # Check if command matches an inventory item
        if hasattr(self, 'inventory_handler') and hasattr(self.inventory_handler, '__call__'):
            try:
                # Try to use component manager first
                try:
                    from component_manager import component_manager
                    search_engine = component_manager.get_search_engine()
                except ImportError:
                    # Fall back to legacy method if component manager is not available
                    from global_registry import GlobalRegistry
                    search_engine = GlobalRegistry.get('search_engine')
                
                if search_engine and hasattr(search_engine, 'find_item'):
                    # Check if command is an inventory item
                    result = search_engine.find_item(processed_text)
                    
                    if result and result.get('found', False):
                        logger.debug(f"Command matches inventory item: {result['item']}")
                        
                        if self.logging_manager:
                            self.logging_manager.log_pattern_match({
                                'type': 'command_inventory_match',
                                'command': processed_text,
                                'item': result['item'],
                                'timestamp': datetime.now().isoformat()
                            })
                        
                        # Create a command match for the inventory item
                        inventory_match = CommandMatch(
                            handler=self.inventory_handler,
                            command_text=processed_text,
                            pattern="inventory_item",
                            metadata={'item_info': result}
                        )
                        
                        # Update command history
                        self.command_history[-1]['processed'] = True
                        self.command_history[-1]['matched_pattern'] = 'inventory_item'
                        
                        return RouterResult(
                            success=True,
                            command_match=inventory_match,
                            metadata={'match_type': 'inventory', 'item': result['item']}
                        )
            except Exception as e:
                logger.error(f"Error checking inventory match: {e}")

        # Sort alternative matches by confidence
        alternative_matches.sort(key=lambda m: m.confidence, reverse=True)
        
        # Check if we have a good alternative match
        if alternative_matches and alternative_matches[0].confidence > 0.7:
            # Use the best alternative match
            best_match = alternative_matches[0]
            
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'command_alternative_match',
                    'command': processed_text,
                    'pattern': best_match.pattern,
                    'confidence': best_match.confidence,
                    'timestamp': datetime.now().isoformat()
                })
            
            logger.debug(f"Using alternative match: {best_match.pattern} (confidence: {best_match.confidence:.2f})")
            
            # Update command history
            self.command_history[-1]['processed'] = True
            self.command_history[-1]['matched_pattern'] = best_match.pattern
            self.command_history[-1]['confidence'] = best_match.confidence
            
            return RouterResult(
                success=True,
                command_match=best_match,
                alternative_matches=alternative_matches[1:],
                metadata={'match_type': 'alternative', 'confidence': best_match.confidence}
            )
        
        # Try fallback handler if available and we have alternative matches
        if self.fallback_handler and alternative_matches:
            try:
                if self.logging_manager:
                    self.logging_manager.log_pattern_match({
                        'type': 'command_fallback',
                        'command': processed_text,
                        'alternatives': [(m.pattern, m.confidence) for m in alternative_matches],
                        'timestamp': datetime.now().isoformat()
                    })
                
                logger.debug(f"Using fallback handler with alternatives: {len(alternative_matches)}")
                
                # Create a fallback match
                fallback_match = CommandMatch(
                    handler=self.fallback_handler,
                    command_text=processed_text,
                    pattern="fallback",
                    metadata={'alternatives': alternative_matches}
                )
                
                # Update command history
                self.command_history[-1]['processed'] = True
                self.command_history[-1]['matched_pattern'] = 'fallback'
                
                return RouterResult(
                    success=True,
                    command_match=fallback_match,
                    alternative_matches=alternative_matches,
                    metadata={'match_type': 'fallback'}
                )
                
            except Exception as e:
                logger.error(f"Error in fallback handler: {e}")
        
        # No match found, use unknown handler if available
        if self.unknown_handler:
            try:
                if self.logging_manager:
                    self.logging_manager.log_pattern_match({
                        'type': 'command_unknown',
                        'command': processed_text,
                        'timestamp': datetime.now().isoformat()
                    })
                
                logger.debug("Using unknown command handler")
                
                # Create an unknown match
                unknown_match = CommandMatch(
                    handler=self.unknown_handler,
                    command_text=processed_text,
                    pattern="unknown"
                )
                
                # Update command history
                self.command_history[-1]['processed'] = True
                self.command_history[-1]['matched_pattern'] = 'unknown'
                
                return RouterResult(
                    success=True,
                    command_match=unknown_match,
                    alternative_matches=alternative_matches,
                    metadata={'match_type': 'unknown'}
                )
                
            except Exception as e:
                logger.error(f"Error in unknown handler: {e}")
        
        # No handler found
        if self.logging_manager:
            self.logging_manager.log_pattern_match({
                'type': 'command_no_handler',
                'command': processed_text,
                'timestamp': datetime.now().isoformat()
            })
        
        logger.warning(f"No handler found for command: '{processed_text}'")
        
        return RouterResult(
            success=False,
            error="No matching handler found",
            alternative_matches=alternative_matches
        )
    
    def _calculate_match_confidence(self, pattern: str, command: str) -> float:
        """
        Calculate confidence score for a partial match.
        
        Args:
            pattern: Pattern to match against
            command: Command text
            
        Returns:
            float: Confidence score (0-1)
        """
        # Basic text similarity approach
        pattern_words = set(pattern.lower().split())
        command_words = set(command.lower().split())
        
        if not pattern_words or not command_words:
            return 0.0
        
        # Calculate word overlap
        common_words = pattern_words.intersection(command_words)
        
        # Jaccard similarity
        similarity = len(common_words) / len(pattern_words.union(command_words))
        
        # Boost if key words match
        key_words = {'add', 'update', 'delete', 'find', 'search', 'open', 'close', 'save'}
        key_word_match = any(word in key_words for word in common_words)
        
        if key_word_match:
            similarity = min(1.0, similarity * 1.3)  # Boost by 30%
        
        return similarity
