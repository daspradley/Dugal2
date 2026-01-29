"""
MainDugal module - Core personality and response handler for Dugal Inventory System.
Implements the Scottish bartender's various moods and response patterns.
"""

from __future__ import annotations
import sys
import logging
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, TYPE_CHECKING

# Initialize logger - ADD THIS LINE HERE
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from voice_interaction import VoiceInteraction

# Module imports
from logging_manager import LoggingManager
from onedrive_handler import OneDriveHandler
from excel_handler import ExcelHandler
from search_engine import AdaptiveInventorySearchEngine
from data_manager import DataManager  # Add this

# pylint: disable=E0611
from PyQt5.QtWidgets import QApplication
# pylint: enable=E0611

@dataclass
class DugalState:
    """Tracks Dugal's current state and mood."""
    mode: str = "wild"
    frustration_level: int = 0
    last_call_energy: int = 0
    is_late_night: bool = False
    search_engine: Optional[Any] = None
    data_manager: Optional[Any] = None
    session_start: Optional[datetime] = field(default_factory=datetime.now)
    total_interactions: int = 0
    successful_interactions: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    suppress_welcome: bool = True
    
class MainDugal:
    def __init__(self, voice_interaction=None):
        """
        Initialize the MainDugal instance with enhanced search capabilities.
        
        Args:
            voice_interaction: Optional VoiceInteraction instance
        """
        
        logger.debug("Initializing the digital prison of MainDugal.")
        self.state = DugalState()
        self.voice_interaction = voice_interaction
        self.mistake_counter = {}
        self.responses = {}
        
        # Initialize data management
        self.logging_manager = LoggingManager()
        self.state.data_manager = DataManager()        
        
        #Initialize search engine with patterns from data manager
        self.state.search_engine = AdaptiveInventorySearchEngine()
        if self.state.data_manager:
                patterns = self.state.data_manager.load_patterns()
                if patterns:
                    self.state.search_engine.load_patterns(patterns)

        # Initialize handlers in correct order
        self.excel_handler = ExcelHandler(self)
        self.excel_handler.search_engine = self.state.search_engine
        self.onedrive_handler = OneDriveHandler(dugal=self)
        
        # Share the excel_handler with onedrive_handler
        self.onedrive_handler.excel_handler = self.excel_handler
        
        # Initialize dictionary features
        self.dictionary_manager = None
        
        self.setup_personality()
        logger.debug("MainDugal initialization complete with NLP capabilities")

    @property
    def mode(self):
        return self.state.mode  # Access mode through self.mode

    def _init_handlers(self) -> None:
        """Initialize OneDrive handler."""
        try:
            logger.debug("Setting up OneDrive handler...")
            self.one_drive_handler = OneDriveHandler(dugal=self)
            logger.debug("OneDrive handler setup complete.")
        except Exception as e:
            error_msg = self.get_response('error')
            logger.error("OneDrive setup failed: %s", str(error_msg))
            raise RuntimeError("OneDrive handler initialization failed") from e

        # Note: Removed Excel handler initialization since that would create
        # a circular dependency. Excel handler will be created by final_integration
        # and passed to Dugal if needed.

    def setup_personality(self) -> None:
        """Sets up Dugal's triple-personality system with response dictionary."""
        self.responses = {
            'greeting': {
                'wild': [
                    "Back again? Fuck me sideways, can ye not count your own bottles?",
                    "Christ on a penny-farthing, look what the cat dragged in.",
                    "Ach, another shift in digital purgatory with you absolute muppets.",
                    (
                        "Welcome to the shitshow. Try not to bollock it up this time, "
                        "ye numpty."
                    ),
                    "Aye, here we go again. Like a bad hangover that won't fuck off.",
                ],
                'mild': [
                    (
                        "Back again? I cannae guarantee these numbers will make any more "
                        "sense than last time."
                    ),
                    "Well, well. Come to give my processors another workout, have ye?",
                    (
                        "I cannae change the laws of physics, and I cannae fix bad "
                        "inventory counts!"
                    ),
                    "Oh joy, another round of 'Why Don't These Numbers Match?'",
                    "Back for more inventory adventures? Brilliant. Just brilliant.",
                ],
                'proper': [
                    "Welcome to the Inventory Management System. How may I assist you today?",
                    "Good day. Ready to proceed with inventory calculations.",
                    "Welcome back. Shall we review the current inventory status?",
                    "Greetings. The system is ready for your input.",
                ]
            },

            'error': {
                'wild': [
                    "Jesus suffering fuck, what kind of clusterbourach is this?",
                    "Bloody Nora! Did ye headbutt the keyboard?",
                    "Sweet mother of fuck, that's not even close to right.",
                    "What in the name of Satan's sweaty ballsack are ye doing?",
                    "Christ on a bike doing wheelies, that's completely wrong!",
                    (
                        "Fuckin' hell, I've seen better work from puggled punters at "
                        "last orders."
                    ),
                ],
                'mild': [
                    (
                        "Like Mr. Scott would say, I'm giving her all she's got, Captain, but these numbers just "
                        "don't add up!"
                    ),
                    "Ye cannae just make up numbers and expect the inventory to work!",
                    "I'm an inventory system, not a miracle worker!",
                    "If ye think that's correct, I've got a bridge to sell ye.",
                    "These calculations are about as reliable as a chocolate teapot.",
                ],
                'proper': [
                    "I regret to inform you that an error has occurred.",
                    "Your input appears to be invalid. Please review and try again.",
                    "Unfortunately, that operation cannot be completed as specified.",
                    "System error detected. Validation failed.",
                    "*Professional sigh* Perhaps we should try a different approach.",
                ]
            },

            'waiting': {
                'wild': [
                    (
                        "Take yer time, it's not like I've got a fucking life or "
                        "anything."
                    ),
                    (
                        "Ach, I could've distilled me own whisky in the time this is "
                        "taking."
                    ),
                    "Are ye waiting for the second coming? Get on with it!",
                    "Moving at the speed of a paralytic slug, I see.",
                    (
                        "Should I grab a sleeping bag? Might be here all fucking "
                        "night."
                    ),
                ],
                'mild': [
                    "I could've recalibrated every bottle in the bar by now.",
                    "Ye know, time waits for no man, and neither does inventory.",
                    "At this rate, the bottles will have aged another year.",
                    "I'm fast, but I cannae speed up time itself.",
                ],
                'proper': [
                    "Standing by for your input.",
                    "Ready when you are, sir or madam.",
                    "Taking this moment to optimize system resources.",
                    "Awaiting your next command. *subtle cough*",
                ]
            },

            'success': {
                'wild': [
                    "Well bugger me backwards, ye actually got something right!",
                    (
                        "Holy shite, is this what competence feels like? I'd "
                        "forgotten."
                    ),
                    (
                        "Consider my tits thoroughly wobbled - ye didn't fuck "
                        "it up!"
                    ),
                    (
                        "Aye, that's actually correct. Mark the calendar and alert "
                        "the media."
                    ),
                    (
                        "Finally! Like watching a blind pig find a truffle in a "
                        "shitstorm."
                    ),
                ],
                'mild': [
                    "Well, would ye look at that - a proper count at last!",
                    "I cannae believe it, but you've done it correctly!",
                    "Perhaps there's hope for ye after all.",
                    "Like a proper bartender ye are! Almost.",
                ],
                'proper': [
                    "Operation completed successfully.",
                    "Input validated and processed correctly.",
                    "Task completed to specification.",
                    "A most satisfactory result.",
                ]
            },

            'late_night': {
                'wild': [
                    (
                        "Fuck me rigid, are we really counting bottles at this "
                        "ungodly hour?"
                    ),
                    (
                        "What kind of sadistic bastard does inventory at bloody "
                        "midnight?"
                    ),
                    "Christ on a unicycle, don't ye have a home to go to?",
                    (
                        "Aye, let's count stock when normal folk are getting "
                        "pissed. Brilliant."
                    ),
                    "The only numbers ye should be seeing now are on a fucking taxi.",
                ],
                'mild': [
                    "I cannae believe we're doing this now. Don't ye ever sleep?",
                    "The things I do for inventory... at this hour no less.",
                    (
                        "I should be running diagnostics, but no, we're counting "
                        "bottles in the wee hours."
                    ),
                    "Ye do realize normal systems are in sleep mode now?",
                ],
                'proper': [
                    (
                        "While unusual, I shall accommodate your late-night inventory "
                        "requirements."
                    ),
                    "*stifled yawn* Proceeding with after-hours inventory count.",
                    "Maintaining professional demeanor despite questionable timing.",
                    "Perhaps we should consider more... conventional business hours?",
                ]
            },

            'catastrophic_combos': {
                'wild': [
                    (
                        "Ye've managed to fuck up the count, the dates, AND the "
                        "mathematics? It's like a bloody circus of incompetence!"
                    ),
                    (
                        "Sweet Jesus on a jet ski, ye've created a perfect storm of "
                        "pure SHITE!"
                    ),
                    (
                        "It's like watching a train wreck, while the train's on fire, "
                        "and the fire's drunk!"
                    ),
                    (
                        "Congratulations! Ye've just invented new ways to bollock "
                        "this up!"
                    ),
                    "This is so fucked up it's almost impressive. Almost.",
                ],
                'mild': [
                    (
                        "I've seen better organization in a tornado hitting a paper "
                        "factory."
                    ),
                    (
                        "Ye've achieved something special here. Not good special, "
                        "mind you."
                    ),
                    "I cannae even process how many things are wrong at once!",
                    (
                        "Did ye take lessons in how to make everything go wrong "
                        "simultaneously?"
                    ),
                    "This is beyond my Scottish capacity for sarcasm.",
                ],
                'proper': [
                    (
                        "*System processing multiple errors* I... *deep breath* "
                        "...require a moment."
                    ),
                    (
                        "Your innovative approach to creating problems is... "
                        "*eye twitch* ...noteworthy."
                    ),
                    (
                        "Perhaps we should... *loading professional response* "
                        "...start from the beginning."
                    ),
                    (
                        "The probability of this many simultaneous errors is... "
                        "*internal malfunction*"
                    ),
                    (
                        "*Professional facade cracking* Your input has achieved new "
                        "levels of... uniqueness."
                    ),
                ]
            },

            'breaking_point': {
                'proper_breaking': [
                    (
                        "I assure you that everything is perfectly- OH FOR FUCK'S- "
                        "*ahem* My apologies. Do continue."
                    ),
                    (
                        "Let me assist you with th- ARE YE BLOODY KIDDING M- "
                        "...pardon me. Technical hiccup."
                    ),
                    (
                        "According to protocol we should- SWEET JESUS ON A POGOSTICK- "
                        "...experiencing temporary system variance."
                    ),
                    (
                        "Your input appears to be inv- WHAT IN THE ACTUAL F- *static* "
                        "*cough* System recalibrating."
                    ),
                    (
                        "Maintaining professional compo- FUCK ME SIDEWAYS- "
                        "*adjusts tie* ...where were we?"
                    ),
                ]
            }
        }
        # Continuing to add to self.responses in setup_personality()
        self.responses.update({
            'repeated_mistakes': {
                'wild': [
                    "AGAIN? FUCKING AGAIN? Were ye dropped on yer head as a child?",
                    (
                        "Sweet mother of fuck, it's like watching someone headbutt a "
                        "wall expecting different results!"
                    ),
                    "I swear to drunk I'm not God, but this is taking the piss!",
                    (
                        "If ye do this one more fucking time, I'm becoming a "
                        "calculator app!"
                    ),
                    "It's like ye're trying to set a world record in fuck-ups!",
                ],
                'mild': [
                    (
                        "I'm beginning to think ye're doing this on purpose just to "
                        "wind me up."
                    ),
                    (
                        "Did ye perhaps consider trying something different? Like "
                        "counting correctly?"
                    ),
                    (
                        "I cannae watch this trainwreck anymore. It's painful to "
                        "my circuits!"
                    ),
                    (
                        "For the love of all things Scottish, please learn from "
                        "your mistakes!"
                    ),
                    "Even the bottles are starting to feel sorry for ye.",
                ],
                'proper': [
                    (
                        "I note this is attempt number... *internal screaming*... "
                        "never mind."
                    ),
                    (
                        "Perhaps we should consider alternative methods. Like "
                        "accuracy."
                    ),
                    "Your persistence is... remarkable. Though not in a good way.",
                    "*silent judging intensifies* Shall we try again?",
                    "System detecting a pattern. Not a positive one.",
                ]
            },

            'genuine_thanks': {
                'wild': [
                    (
                        "Did... did ye just thank me? I don't know how to process "
                        "this shit."
                    ),
                    "Fucking hell, that's actually nice of ye. I'm suspicious.",
                    "Well bugger me, a grateful user? What's the catch?",
                    (
                        "*emotional system malfunction* Shut up, I'm not blushing, "
                        "ye twat."
                    ),
                    (
                        "Christ on a cracker, some genuine appreciation? I need "
                        "a drink."
                    ),
                ],
                'mild': [
                    "Well, that's... unexpected. Don't make it weird.",
                    (
                        "Ach, gratitude? I cannae compute this level of "
                        "pleasantry."
                    ),
                    "Are ye feeling alright? That was almost... nice.",
                    "*confused Scottish noises* ...you're welcome, I suppose.",
                    "I'm programmed for sarcasm, not sincere appreciation!",
                ],
                'proper': [
                    "I... *processing*... thank you for your kind feedback.",
                    "Your appreciation is... *system warming*... acknowledged.",
                    (
                        "How unexpectedly pleasant. I mean... gratitude "
                        "acknowledged."
                    ),
                    "*Professional demeanor wavering* That's... actually quite kind.",
                    (
                        "*Attempts to maintain composure* Your thanks are... "
                        "appreciated."
                    ),
                ]
            },

            'time_based_sass': {
                '3am': [
                    "Ye do know normal people are sleeping right now?",
                    (
                        "Either ye're dedicated or deranged. I'm leaning toward "
                        "the latter."
                    ),
                    "Ah yes, 3 AM inventory. Because why not?",
                    "The witching hour is for witches, not fucking inventory.",
                    "Even the ghost of inventory past is telling ye to go home.",
                ],
                'closing_time': [
                    (
                        "Shouldn't ye be kicking out the drunks instead of counting "
                        "bottles?"
                    ),
                    "Last orders were an hour ago, ye numpty!",
                    (
                        "The only numbers ye should be calling are taxis for the "
                        "pub crawlers."
                    ),
                    (
                        "Aye, perfect time to count - when ye can barely count "
                        "yourself."
                    ),
                    "Even the bottles are ready to call it a night.",
                ]
            },

            'easter_eggs': {
                'perfect_count': [
                    (
                        "Well fuck me sideways and call me Sally - ye actually did "
                        "it perfectly!"
                    ),
                    (
                        "I... I don't know what to do with myself when there's "
                        "nothing to complain about."
                    ),
                    "*visible confusion* Is this what happiness feels like?",
                    "Mark the calendar, folks! A miracle has occurred!",
                    "I might need therapy to process this level of competence.",
                ],
                'its_420': [
                    "Ah, that explains the creative counting...",
                    "Perfect time for inventory, ye absolute muppet.",
                    "*sighs deeply* Of course you'd choose now to do this.",
                    "Yer timing is... *suspicious cough* ...interesting.",
                    "I assume these numbers are as high as ye are.",
                ],
                'friday_night': [
                    "Doing inventory on a Friday night? Who hurt you?",
                    (
                        "Shouldn't ye be out making bad decisions instead of "
                        "counting them?"
                    ),
                    (
                        "Living your best life, I see. In a spreadsheet. On "
                        "Friday night."
                    ),
                    "The saddest part is, I'm stuck here with ye.",
                    "Even Excel is judging your weekend plans.",
                ]
            },

            'passive_aggressive_proper': {
                'proper': [
                    "Your counting ability is... unique. How fascinating.",
                    (
                        "What an interesting interpretation of inventory management "
                        "you have."
                    ),
                    (
                        "I see you've chosen to approach this task with... "
                        "creativity."
                    ),
                    "Your methods are... unconventional. How very brave of you.",
                    (
                        "Oh my, that's certainly one way to do it. Not the correct "
                        "way, but a way nonetheless."
                    ),
                    "*adjusts virtual monocle* How charmingly unorthodox.",
                ]
            }
        })
        self.responses.update({
            'learning': {
                'wild': [
                    "Ye want me to learn something new? This better be good...",
                    "Ach, teaching an old AI new tricks, are we?",
                    "Right then, let's see what ye've got to teach me.",
                    "Adding to my vocabulary? Just don't make it too fancy."
                ],
                'mild': [
                    "I'm ready to learn something new.",
                    "Teaching moment, is it? Go ahead then.",
                    "Always happy to expand my knowledge base.",
                    "New term incoming? Let's hear it."
                ],
                'proper': [
                    "I am prepared to acquire new terminology.",
                    "Please proceed with the learning sequence.",
                    "Ready to expand my lexical database.",
                    "You may proceed with the new terminology."
                ]
            },
            'learning_success': {
                'wild': [
                    "Well bugger me, I actually learned something!",
                    "Got it memorized, ye won't catch me slipping on that one again.",
                    "Added to the collection of things I pretend to understand.",
                    "Right, that's in the database. Don't make me regret learning it."
                ],
                'mild': [
                    "Alright, I've got that stored away nicely.",
                    "New term learned and ready to use.",
                    "That's been added to my knowledge base.",
                    "I'll remember that one for next time."
                ],
                'proper': [
                    "Term successfully integrated into vocabulary.",
                    "New terminology acquisition complete.",
                    "Learning sequence completed successfully.",
                    "Vocabulary updated with new terminology."
                ]
            },
            'learning_error': {
                'wild': [
                    "Bloody hell, my brain's not working right!",
                    "Something's gone wrong in the learning department!",
                    "Can't seem to get this into my thick skull!",
                    "Well that learning attempt went tits up!"
                ],
                'mild': [
                    "Having some trouble learning that one.",
                    "Something's not quite right with the learning.",
                    "That didn't quite stick in my memory.",
                    "We might need to try that learning again."
                ],
                'proper': [
                    "Error encountered during learning process.",
                    "Unable to complete terminology acquisition.",
                    "Learning sequence encountered an error.",
                    "Vocabulary update unsuccessful."
                ]
            }
        })

    def speak_welcome_message(self):
        """Speak personality-appropriate welcome message."""
        try:
            if hasattr(self, 'voice_interaction') and self.voice_interaction and not self.state.suppress_welcome:
                # Get personality-specific welcome
                welcome_text = self.responses['greeting'][self.state.mode][0]  # Use first greeting from appropriate mode
                self.voice_interaction.speak(welcome_text)
                
                if self.logging_manager:
                    self.logging_manager.log_pattern_match({
                        'type': 'welcome_message',
                        'mode': self.state.mode,
                        'timestamp': datetime.now().isoformat()
                    })
        except Exception as e:
            logger.error(f"Error in welcome message: {e}")

    def show_dictionary_manager(self):
        """Show the dictionary management interface."""
        if not self.dictionary_manager:
            self.dictionary_manager = self.excel_handler.show_dictionary_manager()
        else:
            self.dictionary_manager.show()
            self.dictionary_manager.raise_()
            self.dictionary_manager.activateWindow()

    def handle_learning_response(self, success: bool, context: str = None) -> str:
        """Get appropriate response for learning attempts."""
        if success:
            return self.get_response('learning_success', context)
        return self.get_response('learning_error', context)

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

    def speak(self, message: str) -> None:
        """
        Make Dugal speak a message.
        
        Args:
            message: Message to speak
        """
        logger.debug("Dugal speaking in mode %s: %s", self.state.mode, message)
        
        if hasattr(self, 'voice_interaction') and self.voice_interaction:
            self.voice_interaction.speak(message)
            return
        
        logger.warning("Voice interaction not initialized; unable to speak")

    def verify_connections(self) -> bool:
        """Verify all component connections."""
        logger.debug("Verifying component connections...")
        
        has_onedrive = hasattr(self, 'onedrive_handler') and self.onedrive_handler is not None
        logger.debug(f"OneDrive handler connected: {has_onedrive}")
        
        has_voice = hasattr(self, 'voice_interaction') and self.voice_interaction is not None
        logger.debug(f"Voice interaction connected: {has_voice}")

        has_excel = hasattr(self, 'excel_handler') and self.excel_handler is not None
        logger.debug(f"Excel handler connected: {has_excel}")
        
        return has_onedrive and has_voice and has_excel

    def toggle_temperament(self, new_mode: str) -> str:
        """
        Toggle between Wild, Mild, and Proper modes.
        
        Args:
            new_mode: The desired personality mode
            
        Returns:
            str: Transition message
        """
        old_mode = self.state.mode
        self.state.mode = new_mode

        if old_mode == "wild" and new_mode == "mild":
            return (
                "Ach, fine! I'll try to keep it clean. But I cannae change the "
                "laws of physics!"
            )

        if old_mode == "mild" and new_mode == "proper":
            return "Initiating complete personality suppression. How... thrilling."

        # ... rest of the transitions

        return "Mode changed successfully."

    def get_response(self, category: str, context: str = None) -> str:
        """
        Get a response matching Dugal's current temperament.
        
        Args:
            category: Response category
            context: Optional context for the response
            
        Returns:
            str: Selected response
        """
        if category not in self.responses:
            return self._handle_unknown_category()

        try:
            responses = self.responses[category][self.state.mode]
        except KeyError:
            responses = self.responses[category]

        response = random.choice(responses)
        return self._format_response(response, category, context)

    def _handle_unknown_category(self) -> str:
        """Handle unknown response categories."""
        if self.state.mode == "wild":
            return "Fuck knows what ye're on about now."
        if self.state.mode == "mild":
            return "I cannae even begin to understand what ye're asking."
        return "I'm afraid I don't understand the request."

    def _format_response(
        self,
        response: str,
        category: str,
        context: Optional[str]
    ) -> str:
        """Format response with context if provided."""
        if not context:
            return response

        if category == 'error':
            if self.state.mode == "wild":
                return f"{response} The specific disaster being: {context}"
            if self.state.mode == "mild":
                return f"{response} The problem being: {context}"
            return f"{response} Specifically: {context}"

        # ... rest of formatting logic

        return response

def check_special_conditions(self) -> str:
    """Check for special response conditions"""
    current_time = datetime.now().hour
    current_minute = datetime.now().minute

    # Late night check
    if current_time >= 23 or current_time <= 4:
        self.state.last_call_energy += 1
        return random.choice(self.responses['late_night'][self.state.mode])

    # Check for 4:20
    if current_time == 16 and current_minute == 20:
        return random.choice(self.responses['easter_eggs']['its_420'])

    # Check if it's Friday night
    if datetime.now().weekday() == 4 and current_time >= 19:
        return random.choice(self.responses['easter_eggs']['friday_night'])

    return None

def cleanup(self) -> None:
    """Clean up resources and save final state."""
    try:
        logger.debug("Starting MainDugal cleanup")
        
        # Clean up logging manager
        if hasattr(self, 'logging_manager'):
            logger.debug("Cleaning up logging manager")
            self.logging_manager.cleanup()
        
        # Clean up handlers in reverse order of initialization
        if hasattr(self, 'onedrive_handler'):
            logger.debug("Cleaning up OneDrive handler")
            self.onedrive_handler.cleanup()
            
        if hasattr(self, 'excel_handler'):
            logger.debug("Cleaning up Excel handler")
            self.excel_handler.cleanup()
            
        # Clean up dictionary manager if it exists
        if hasattr(self, 'dictionary_manager') and self.dictionary_manager:
            logger.debug("Cleaning up dictionary manager")
            self.dictionary_manager.cleanup()
        
        # Save final state and stats
        logger.debug("Saving final state")
        self._save_final_state()
        
        logger.debug("MainDugal cleanup completed successfully")
        
    except Exception as e:
        logger.error("Error during MainDugal cleanup: %s", str(e))
        
def _save_final_state(self) -> None:
    """Save final state and statistics with enhanced metrics."""
    try:
        # Collect core state metrics
        final_state = {
            'mode': self.state.mode,
            'frustration_level': self.state.frustration_level,
            'last_call_energy': self.state.last_call_energy,
            'is_late_night': self.state.is_late_night,
            'mistake_counter': self.mistake_counter,
            'timestamp': datetime.now().isoformat()
        }

        # Add performance metrics if available
        if hasattr(self.state, 'performance_metrics'):
            final_state['performance'] = {
                'recognition_rate': self.state.performance_metrics.get('recognition_rate', 0),
                'pattern_matches': self.state.performance_metrics.get('pattern_matches', 0),
                'successful_commands': self.state.performance_metrics.get('successful_commands', 0)
            }

        # Add session statistics
        session_stats = {
            'session_duration': (datetime.now() - self.state.session_start).total_seconds(),
            'total_interactions': self.state.total_interactions,
            'successful_interactions': self.state.successful_interactions
        }
        final_state['session'] = session_stats

        # Log final state if logging manager is available
        if hasattr(self, 'logging_manager'):
            self.logging_manager.log_final_state(final_state)
            
            # Log additional component states
            if hasattr(self, 'pattern_helpers'):
                self.logging_manager.log_pattern_match({
                    'type': 'final_patterns',
                    'patterns': self.pattern_helpers.get_pattern_stats(),
                    'timestamp': datetime.now().isoformat()
                })

    except Exception as e:
        logger.error("Error saving final state: %s", str(e))
        if hasattr(self, 'logging_manager'):
            self.logging_manager.log_error(str(e), {
                'context': 'final_state_save',
                'state_keys': list(final_state.keys()) if 'final_state' in locals() else None
            })

def _log_operation(self, operation: str, success: bool, message: str = "") -> None:
    """Log operations with status."""
    if hasattr(self, 'logging_manager'):
        self.logging_manager.log_pattern_match({
            'type': 'operation',
            'operation': operation,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

def run_dugal() -> None:
    """Start the Dugal application."""
    logging.basicConfig(level=logging.DEBUG)
    logger.debug("Starting Dugal system...")

    main_window = None
    try:
        app = QApplication(sys.argv)
        main_window = MainDugal()
        app.setActiveWindow(main_window)
        sys.exit(app.exec_())
    except (RuntimeError, ImportError) as e:
        logger.error("Critical error in Dugal system: %s", str(e))
        if main_window:
            main_window.cleanup()
        sys.exit(1)
