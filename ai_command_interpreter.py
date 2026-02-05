"""
AI-powered command interpretation for Dugal Inventory System.
Uses Claude API to interpret natural language voice commands with context awareness.

Author: Dugal AI Integration
Date: January 2026
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import anthropic

logger = logging.getLogger(__name__)


@dataclass
class CommandInterpretation:
    """Structured interpretation of a voice command."""
    intent: str  # search, inventory_update, navigate_sheet, set_search_column, set_input_column, system_command, mode_change, unclear
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    ambiguities: List[str] = field(default_factory=list)
    suggested_clarification: Optional[str] = None
    raw_response: Optional[str] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class AICommandInterpreter:
    """
    Interprets voice commands using Claude API.
    
    Provides natural language understanding for inventory commands,
    maintaining context and resolving ambiguities intelligently.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        logging_manager=None,
        enable_caching: bool = True,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the AI interpreter.
        
        Args:
            api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
            logging_manager: Optional logging manager for tracking
            enable_caching: Whether to cache common command interpretations
            confidence_threshold: Minimum confidence for AI interpretation (0.0-1.0)
        """
        self.api_key = api_key or os.environ.get('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') or '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'
        if not self.api_key:
            logger.warning("No Anthropic API key found. AI interpretation will be unavailable.")
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.logging_manager = logging_manager
        self.confidence_threshold = confidence_threshold
        
        # Context management
        self.context_history = []
        self.max_context = 5  # Keep last 5 commands for context
        
        # Caching for common commands
        self.enable_caching = enable_caching
        self.cache = {}
        self.max_cache_size = 100
        
        # Statistics
        self.stats = {
            'total_interpretations': 0,
            'ai_successes': 0,
            'ai_failures': 0,
            'cache_hits': 0,
            'fallback_uses': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0
        }
        
        logger.info("AI Command Interpreter initialized successfully")
    
    def interpret_command(
        self, 
        command_text: str,
        context: Optional[Dict[str, Any]] = None,
        force_ai: bool = False
    ) -> CommandInterpretation:
        """
        Interpret a voice command using AI.
        
        Args:
            command_text: The spoken command text
            context: Current system state (active sheet, columns, last item, etc.)
            force_ai: Skip cache and force fresh AI interpretation
            
        Returns:
            CommandInterpretation with structured command data
        """
        start_time = datetime.now()
        self.stats['total_interpretations'] += 1
        
        # Normalize command text
        command_text = command_text.strip()
        
        # Check cache first (unless force_ai is True)
        if self.enable_caching and not force_ai:
            cache_key = self._generate_cache_key(command_text, context)
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache hit for command: '{command_text}'")
                cached_result = self.cache[cache_key]
                cached_result.processing_time = (datetime.now() - start_time).total_seconds()
                return cached_result
        
        # Build prompt with context
        prompt = self._build_interpretation_prompt(command_text, context)
        
        try:
            # Call Claude API
            logger.debug(f"Sending command to AI: '{command_text}'")
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.1,  # Low temperature for consistent parsing
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Parse response
            response_text = response.content[0].text
            interpretation = self._parse_response(response_text, command_text)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            interpretation.processing_time = processing_time
            
            # Update statistics
            self.stats['ai_successes'] += 1
            self._update_running_stats(interpretation)
            
            # Add to context history
            self._add_to_history(command_text, interpretation)
            
            # Cache result if enabled
            if self.enable_caching:
                cache_key = self._generate_cache_key(command_text, context)
                self._add_to_cache(cache_key, interpretation)
            
            # Log success
            if self.logging_manager:
                self.logging_manager.log_pattern_match({
                    'type': 'ai_interpretation_success',
                    'command': command_text,
                    'intent': interpretation.intent,
                    'confidence': interpretation.confidence,
                    'processing_time': processing_time,
                    'timestamp': datetime.now().isoformat()
                })
            
            logger.info(
                f"AI interpretation: '{command_text}' → {interpretation.intent} "
                f"(confidence: {interpretation.confidence:.2f}, time: {processing_time:.3f}s)"
            )
            
            return interpretation
            
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            self.stats['ai_failures'] += 1
            return self._fallback_interpretation(command_text, f"API error: {e}")
            
        except Exception as e:
            logger.error(f"AI interpretation error: {e}", exc_info=True)
            self.stats['ai_failures'] += 1
            return self._fallback_interpretation(command_text, str(e))
    
    def _build_interpretation_prompt(
        self, 
        command_text: str, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build the prompt for Claude API with comprehensive context."""
        
        # Build context section
        context_str = ""
        if context:
            context_parts = []
            
            if 'active_sheet' in context:
                context_parts.append(f"- Active Sheet: {context['active_sheet']}")
            
            if 'sheets' in context and context['sheets']:
                sheets_list = ', '.join(context['sheets'])
                context_parts.append(f"- Available Sheets: {sheets_list}")
            
            if 'search_column' in context:
                context_parts.append(f"- Search Column: {context['search_column']}")
            
            if 'input_column' in context:
                context_parts.append(f"- Input/Target Column: {context['input_column']}")
            
            if 'last_item' in context:
                context_parts.append(f"- Last Item Referenced: {context['last_item']}")
            
            if 'columns' in context and context['columns']:
                cols_list = ', '.join(str(c) for c in context['columns'])
                context_parts.append(f"- Available Columns: {cols_list}")
            
            if context_parts:
                context_str = "Current System State:\n" + '\n'.join(context_parts) + "\n"
        
        # Build recent commands section
        recent_commands = ""
        if self.context_history:
            recent_commands = "\nRecent Command History (for context):\n"
            for i, hist in enumerate(self.context_history[-3:], 1):
                intent = hist['interpretation'].intent
                entities_str = ", ".join(
                    f"{k}={v}" for k, v in hist['interpretation'].entities.items()
                )
                recent_commands += f"{i}. \"{hist['command']}\" → Intent: {intent}"
                if entities_str:
                    recent_commands += f", Entities: {entities_str}"
                recent_commands += "\n"
        
        prompt = f"""You are an expert command interpreter for a voice-controlled Excel inventory management system. Your role is to parse natural language commands into structured intents and entities.

{context_str}{recent_commands}
User's Voice Command: "{command_text}"

Analyze this command and respond ONLY with a JSON object following this exact structure:

{{
    "intent": "<intent_type>",
    "entities": {{<extracted_entities>}},
    "confidence": <0.0_to_1.0>,
    "ambiguities": [<list_of_ambiguous_aspects>],
    "suggested_clarification": "<clarifying_question_or_null>"
}}

INTENT TYPES:
1. "search" - User wants to find/look up an inventory item
2. "inventory_update" - User wants to add/subtract/set inventory quantity
3. "navigate_sheet" - User wants to switch to a specific Excel sheet
4. "set_search_column" - User wants to set which column to search in
5. "set_input_column" - User wants to set which column to input data to
6. "system_command" - System commands (help, status, diagnostics, etc.)
7. "mode_change" - Changing personality mode (wild, mild, proper)
8. "unclear" - Command is too ambiguous to interpret confidently

ENTITY EXTRACTION RULES BY INTENT:

For "search":
  {{"item_name": "name of item to search for"}}

For "inventory_update":
  {{
    "item_name": "name of inventory item",
    "quantity": <numeric_value>,
    "operation": "add" | "subtract" | "set"
  }}
  - "add", "plus", "increase", "more" → operation: "add"
  - "remove", "subtract", "minus", "less", "take away" → operation: "subtract"
  - "set to", "make it", "change to" → operation: "set"

For "navigate_sheet":
  {{
    "sheet_name": "name of sheet",
    "exclusive": true | false  // true if "only this sheet", false if just switching focus
  }}

For "set_search_column":
  {{"column_name": "name of column to search in"}}

For "set_input_column":
  {{"column_name": "name of column for data entry"}}

For "mode_change":
  {{"mode": "wild" | "mild" | "proper"}}

For "system_command":
  {{"command_type": "help" | "status" | "diagnostics" | "reset" | etc}}

INTERPRETATION GUIDELINES:

1. CONTEXT REFERENCES:
   - "it", "that", "this" → Use last_item from context
   - "the same", "previous" → Use last_item from context
   - If no context available, mark as ambiguous

2. NUMBER HANDLING:
   - Spelled numbers: "three" → 3, "twenty-five" → 25
   - Handle both: "3" and "three" identically
   - Decimals: "two point five" → 2.5
   
   CRITICAL: NUMBERS IN PRODUCT NAMES VS INVENTORY COUNTS
   
   Numbers 1-25 are AMBIGUOUS - could be age statements OR small inventory counts:
   - "knob creek 12" → AMBIGUOUS (12 Year vs 12 bottles)
   - "glenlivet 18" → AMBIGUOUS (18 Year vs 18 bottles)
   - "buffalo trace 15" → AMBIGUOUS (15 bottles is realistic)
   
   When ambiguous (number 1-25 with NO explicit operation word):
   - Mark as "unclear" intent
   - Confidence: 0.3-0.5
   - Suggested clarification: "Did you mean to search for '[Item] [Number]' or update [Item] quantity by [Number]?"
   
   Numbers 26-99 are MORE LIKELY inventory counts but STILL CHECK:
   - "makers mark 46" → LIKELY UPDATE (46 bottles reasonable)
   - "woodford reserve 90" → LIKELY UPDATE (90 bottles reasonable)
   - BUT "old forester 86" could be "Old Forester 86 Proof"
   
   Numbers 100+ are ALMOST ALWAYS inventory counts:
   - "titos 150" → UPDATE (150 bottles)
   - "well vodka 200" → UPDATE (200 bottles)
   
   Numbers 1000+ are PRODUCT NAMES (years, proof marks):
   - "glenlivet 1824" → SEARCH (product name)
   - "old grand dad 114" → SEARCH (114 proof)
   
   EXPLICIT operation words (add, subtract, set, plus, minus) = ALWAYS UPDATE:
   - "knob creek 12 add" → UPDATE (regardless of number)
   - "add 12 to knob creek" → UPDATE
   - "glenlivet 18 plus" → UPDATE

3. ITEM NAME MATCHING:
   - Accept partial names: "Tito's" is sufficient for "Tito's Vodka"
   - Brand names alone are valid: "Hendricks" for "Hendricks Gin"
   - Be flexible with spelling variations
   - Numbers at END of item name suggest product variant: "Knob Creek 12", "Glenlivet 18"
   - PRESERVE numbers as part of item name when ambiguous

4. IMPLICIT OPERATIONS DECISION TREE:
   
   Pattern: "[Item Name] [Number]"
   
   Step 1: Check for explicit operation word
   - If "add", "subtract", "plus", "minus", "set" present → ALWAYS inventory_update
   
   Step 2: Check number range
   - If number ≥ 1000 → SEARCH (likely product name/year)
   - If number 100-999 → LIKELY UPDATE (mark confidence 0.7, could clarify)
   - If number 26-99 → AMBIGUOUS (confidence 0.4, ASK FOR CLARIFICATION)
   - If number 1-25 → VERY AMBIGUOUS (confidence 0.3, ASK FOR CLARIFICATION)
   
   Step 3: Check decimal numbers
   - Decimals (0.1 - 9.9) → LIKELY UPDATE (fractional bottles common)
   
   Examples:
   - "makers mark 3" → unclear (confidence 0.3)
   - "makers mark 0.6" → inventory_update (confidence 0.8, decimals = bottles)
   - "makers mark 150" → inventory_update (confidence 0.9)
   - "glenlivet 1824" → search (confidence 0.95, year = product name)
   - "knob creek add 12" → inventory_update (confidence 1.0, explicit operation)
5. CONFIDENCE SCORING:
   - 0.9-1.0: Crystal clear, unambiguous
   - 0.7-0.9: Clear but minor ambiguity
   - 0.5-0.7: Somewhat clear, notable ambiguity
   - 0.3-0.5: Quite ambiguous, needs clarification
   - 0.0-0.3: Very unclear, definitely needs clarification

6. AMBIGUITY HANDLING:
   - If confidence < 0.5, provide suggested_clarification
   - List specific ambiguities in the "ambiguities" array
   - Offer specific clarifying questions

7. MULTI-WORD ITEMS:
   - Preserve full names: "Buffalo Trace" not just "Buffalo" or "Trace"
   - Keep brand + product: "Maker's Mark" not just "Maker"

EXAMPLES:

Command: "Find Tito's"
{{
  "intent": "search",
  "entities": {{"item_name": "Tito's"}},
  "confidence": 0.95,
  "ambiguities": [],
  "suggested_clarification": null
}}

Command: "Add 3 to it"
(Assuming last_item = "Hendricks Gin")
{{
  "intent": "inventory_update",
  "entities": {{
    "item_name": "Hendricks Gin",
    "quantity": 3,
    "operation": "add"
  }},
  "confidence": 0.9,
  "ambiguities": [],
  "suggested_clarification": null
}}

Command: "Makers Mark 12"
{{
  "intent": "unclear",
  "entities": {{}},
  "confidence": 0.4,
  "ambiguities": [
    "Could be searching for 'Makers Mark 12' (product variant)",
    "Could be updating 'Makers Mark' quantity by 12"
  ],
  "suggested_clarification": "Did you mean to search for 'Makers Mark 12' or update Makers Mark quantity by 12?"
}}

Command: "Work with only the spirits sheet"
{{
  "intent": "navigate_sheet",
  "entities": {{
    "sheet_name": "spirits",
    "exclusive": true
  }},
  "confidence": 0.95,
  "ambiguities": [],
  "suggested_clarification": null
}}

Command: "Use the price column"
{{
  "intent": "set_input_column",
  "entities": {{"column_name": "price"}},
  "confidence": 0.9,
  "ambiguities": [],
  "suggested_clarification": null
}}

NOW INTERPRET THE USER'S COMMAND. Respond ONLY with a valid JSON object, no other text, no markdown formatting, no explanation."""

        return prompt
    
    def _parse_response(self, response_text: str, original_command: str) -> CommandInterpretation:
        """Parse Claude's JSON response into CommandInterpretation."""
        try:
            # Clean response text
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                # Find the actual JSON content
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('```'):
                        in_json = not in_json
                        continue
                    if in_json or (line.strip().startswith('{') or json_lines):
                        json_lines.append(line)
                response_text = '\n'.join(json_lines).strip()
            
            # Parse JSON
            data = json.loads(response_text)
            
            # Validate required fields
            if 'intent' not in data:
                raise ValueError("Response missing 'intent' field")
            
            # Create interpretation object
            interpretation = CommandInterpretation(
                intent=data['intent'],
                entities=data.get('entities', {}),
                confidence=float(data.get('confidence', 0.5)),
                ambiguities=data.get('ambiguities', []),
                suggested_clarification=data.get('suggested_clarification'),
                raw_response=response_text
            )
            
            return interpretation
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}\nResponse: {response_text}")
            return CommandInterpretation(
                intent="unclear",
                entities={},
                confidence=0.0,
                ambiguities=["Failed to parse AI response - invalid JSON"],
                suggested_clarification="I had trouble understanding that command. Could you rephrase?",
                raw_response=response_text
            )
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return CommandInterpretation(
                intent="unclear",
                entities={},
                confidence=0.0,
                ambiguities=[f"Parse error: {str(e)}"],
                suggested_clarification="I didn't quite catch that. Could you try again?"
            )
    
    def _fallback_interpretation(
        self, 
        command_text: str, 
        error_msg: str = "AI service unavailable"
    ) -> CommandInterpretation:
        """
        Simple fallback interpretation when AI fails.
        Uses basic keyword matching.
        """
        self.stats['fallback_uses'] += 1
        logger.warning(f"Using fallback interpretation: {error_msg}")
        
        text_lower = command_text.lower().strip()
        
        # Very basic pattern matching
        if any(word in text_lower for word in ['find', 'search', 'look for', 'where', 'locate']):
            # Try to extract item name
            for keyword in ['find ', 'search for ', 'look for ', 'where is ', 'locate ']:
                if keyword in text_lower:
                    item_name = text_lower.split(keyword, 1)[1].strip()
                    return CommandInterpretation(
                        intent="search",
                        entities={'item_name': item_name},
                        confidence=0.3,
                        ambiguities=["Fallback mode - low confidence"],
                        suggested_clarification=None
                    )
        
        if any(word in text_lower for word in ['add', 'plus', 'increase', 'more']):
            return CommandInterpretation(
                intent="inventory_update",
                entities={'operation': 'add'},
                confidence=0.2,
                ambiguities=["Fallback mode - partial interpretation"],
                suggested_clarification="Could you specify what item and how much?"
            )
        
        if 'sheet' in text_lower or 'focus' in text_lower or 'work with' in text_lower:
            return CommandInterpretation(
                intent="navigate_sheet",
                entities={},
                confidence=0.2,
                ambiguities=["Fallback mode - sheet name not extracted"],
                suggested_clarification="Which sheet did you want to work with?"
            )
        
        # Complete fallback
        return CommandInterpretation(
            intent="unclear",
            entities={},
            confidence=0.0,
            ambiguities=[error_msg],
            suggested_clarification="I didn't understand that command. Could you rephrase?"
        )
    
    def _generate_cache_key(
        self, 
        command_text: str, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key from command and relevant context."""
        # Normalize command
        normalized = command_text.lower().strip()
        
        # Include relevant context in key
        context_key = ""
        if context:
            # Only include context that affects interpretation
            relevant_context = {
                'last_item': context.get('last_item'),
                'active_sheet': context.get('active_sheet')
            }
            context_key = json.dumps(relevant_context, sort_keys=True)
        
        return f"{normalized}|{context_key}"
    
    def _add_to_cache(self, cache_key: str, interpretation: CommandInterpretation) -> None:
        """Add interpretation to cache with size management."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = interpretation
    
    def _add_to_history(
        self, 
        command_text: str, 
        interpretation: CommandInterpretation
    ) -> None:
        """Add command and interpretation to history."""
        self.context_history.append({
            'command': command_text,
            'interpretation': interpretation,
            'timestamp': datetime.now().isoformat()
        })
        
        # Maintain max history size
        if len(self.context_history) > self.max_context:
            self.context_history.pop(0)
    
    def _update_running_stats(self, interpretation: CommandInterpretation) -> None:
        """Update running statistics."""
        # Update average confidence
        total = self.stats['total_interpretations']
        current_avg = self.stats['avg_confidence']
        new_avg = ((current_avg * (total - 1)) + interpretation.confidence) / total
        self.stats['avg_confidence'] = new_avg
        
        # Update average processing time if available
        if interpretation.processing_time:
            current_avg_time = self.stats['avg_processing_time']
            new_avg_time = ((current_avg_time * (total - 1)) + interpretation.processing_time) / total
            self.stats['avg_processing_time'] = new_avg_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get interpreter statistics."""
        return {
            **self.stats,
            'cache_size': len(self.cache),
            'history_size': len(self.context_history),
            'success_rate': (
                self.stats['ai_successes'] / max(self.stats['total_interpretations'], 1)
            ) * 100
        }
    
    def clear_cache(self) -> None:
        """Clear the interpretation cache."""
        self.cache.clear()
        logger.info("Interpretation cache cleared")
    
    def clear_history(self) -> None:
        """Clear the command history."""
        self.context_history.clear()
        logger.info("Command history cleared")
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats = {
            'total_interpretations': 0,
            'ai_successes': 0,
            'ai_failures': 0,
            'cache_hits': 0,
            'fallback_uses': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0
        }
        logger.info("Statistics reset")


# Standalone testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== AI Command Interpreter Test ===\n")
    
    # Initialize interpreter
    try:
        interpreter = AICommandInterpreter()
        print("✓ Interpreter initialized successfully\n")
    except ValueError as e:
        print(f"✗ Failed to initialize: {e}")
        print("\nTo test, set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        exit(1)
    
    # Test commands
    test_commands = [
        ("Find Tito's", {}),
        ("Add 3 to it", {'last_item': "Tito's Vodka"}),
        ("Work with the spirits sheet", {'sheets': ['spirits', 'wine', 'beer']}),
        ("Set target to price", {'columns': ['Item', 'Quantity', 'Price']}),
        ("Makers Mark 12", {}),  # Ambiguous
        ("How many Hendricks do we have?", {})
    ]
    
    print("Testing commands:\n")
    for command, context in test_commands:
        print(f"Command: \"{command}\"")
        if context:
            print(f"Context: {context}")
        
        result = interpreter.interpret_command(command, context)
        
        print(f"  Intent: {result.intent}")
        print(f"  Entities: {result.entities}")
        print(f"  Confidence: {result.confidence:.2f}")
        if result.ambiguities:
            print(f"  Ambiguities: {result.ambiguities}")
        if result.suggested_clarification:
            print(f"  Suggestion: {result.suggested_clarification}")
        print(f"  Processing time: {result.processing_time:.3f}s")
        print()
    
    # Show statistics
    print("\n=== Statistics ===")
    stats = interpreter.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
