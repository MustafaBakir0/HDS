"""
Enhanced conversation manager for the Drawing AI App
Provides dynamic, context-aware responses based on user interaction history
"""
import random
import time
from datetime import datetime
from collections import deque
from PyQt5.QtCore import QObject, pyqtSignal

import config
from core.tts import TextToSpeech


class ConversationState:
    """Tracks the state of the conversation for contextual responses"""

    def __init__(self):
        """Initialize the conversation state"""
        # Track consecutive successes and failures
        self.consecutive_correct_guesses = 0
        self.consecutive_high_confidence = 0
        self.consecutive_low_confidence = 0

        # Track shape counts
        self.shape_counts = {}

        # Recent history
        self.recent_guesses = deque(maxlen=5)

        # Interaction start time
        self.session_start = datetime.now()

        # Drawing metrics
        self.last_drawing_start = None
        self.last_drawing_end = None

        # First-time user flag
        self.is_first_time = True

    def record_guess(self, shape, confidence, is_correct):
        """
        Record a guess and update the state

        Args:
            shape: The shape that was guessed
            confidence: Confidence percentage
            is_correct: Whether the guess was correct
        """
        # Update shape counts
        if shape in self.shape_counts:
            self.shape_counts[shape] += 1
        else:
            self.shape_counts[shape] = 1

        # Update recent guesses
        self.recent_guesses.append({
            'shape': shape,
            'confidence': confidence,
            'is_correct': is_correct,
            'timestamp': datetime.now()
        })

        # Update consecutive counters
        if is_correct:
            self.consecutive_correct_guesses += 1
            self.consecutive_low_confidence = 0

            if confidence >= 80:
                self.consecutive_high_confidence += 1
            else:
                self.consecutive_high_confidence = 0
        else:
            self.consecutive_correct_guesses = 0
            self.consecutive_high_confidence = 0

            if confidence < 50:
                self.consecutive_low_confidence += 1
            else:
                self.consecutive_low_confidence = 0

    def start_drawing(self):
        """Record that drawing has started"""
        self.last_drawing_start = datetime.now()
        self.last_drawing_end = None

    def end_drawing(self):
        """Record that drawing has ended"""
        self.last_drawing_end = datetime.now()

    def drawing_duration(self):
        """Get the duration of the last drawing in seconds"""
        if self.last_drawing_start and self.last_drawing_end:
            return (self.last_drawing_end - self.last_drawing_start).total_seconds()
        return 0

    def get_most_common_shape(self):
        """Get the most commonly drawn shape"""
        if not self.shape_counts:
            return None
        return max(self.shape_counts.items(), key=lambda x: x[1])[0]

    def is_repeat_shape(self, shape):
        """Check if this shape was recently drawn"""
        recent_shapes = [g['shape'] for g in self.recent_guesses]
        return recent_shapes.count(shape) > 1

    def session_duration(self):
        """Get the session duration in minutes"""
        return (datetime.now() - self.session_start).total_seconds() / 60


class ConversationManager(QObject):
    """Enhanced manager for the AI's conversational responses"""

    # Signals
    message_ready = pyqtSignal(str)  # Emitted when a new message is ready
    thinking_started = pyqtSignal()  # Emitted when AI starts thinking
    thinking_finished = pyqtSignal()  # Emitted when AI finishes thinking

    def __init__(self, use_tts=True, parent=None):
        """
        Initialize the conversation manager

        Args:
            use_tts: Whether to use text-to-speech
            parent: Parent QObject
        """
        super().__init__(parent)

        # Initialize text-to-speech if needed
        self.use_tts = use_tts
        self.tts = TextToSpeech() if use_tts else None

        # Initialize conversation state
        self.state = ConversationState()

        # Keep track of conversation history
        self.history = []

        # Track when the last response was given
        self.last_response_time = None

        # Used templates to avoid repetition
        self.used_templates = {
            'greeting': set(),
            'thinking': set(),
            'correct_high': set(),
            'correct_medium': set(),
            'incorrect': set(),
            'low_confidence': set(),
            'canvas_cleared': set(),
            'easter_eggs': set()
        }

    def get_greeting(self, is_first_time=False):
        """
        Get a contextual greeting message

        Args:
            is_first_time: Whether this is the user's first time

        Returns:
            str: Greeting message
        """
        if is_first_time or self.state.is_first_time:
            self.state.is_first_time = False
            greeting = self._get_fresh_template('greeting', config.AI_FIRST_TIME_GREETING)
        else:
            greeting = self._get_fresh_template('greeting', config.AI_GREETING)

        self._speak(greeting)
        self.history.append({"role": "ai", "message": greeting})
        self.last_response_time = datetime.now()

        return greeting

    def get_thinking_message(self):
        """
        Get a random thinking message

        Returns:
            str: Thinking message
        """
        return self._get_fresh_template('thinking', config.AI_THINKING)

    def process_recognition(self, recognition_results):
        """
        Process recognition results and generate a dynamic response

        Args:
            recognition_results: Results from the DrawingRecognizer

        Returns:
            str: Response message
        """
        # Record end of drawing time
        self.state.end_drawing()

        # Start thinking
        self.thinking_started.emit()
        thinking_message = self.get_thinking_message()
        self.message_ready.emit(thinking_message)

        # Simulate thinking time (adjust based on complexity)
        delay = min(1.0 + (len(str(recognition_results)) / 5000), 2.5)
        time.sleep(delay)

        # Generate the response based on confidence level and history
        if recognition_results["success"]:
            top_guess = recognition_results["guesses"][0]
            shape_name = top_guess["name"]
            confidence = top_guess["confidence"]
            is_confident = recognition_results["is_confident"]

            response = self._generate_dynamic_response(shape_name, confidence, is_confident)

            # Update state with correct guess for now (user can correct later)
            self.state.record_guess(shape_name, confidence, is_confident)
        else:
            # Recognition failed
            response = self._get_fresh_template('low_confidence', config.AI_LOW_CONFIDENCE)
            response = response.format(confidence=25)  # Low default confidence

            # Update state with a failed guess
            self.state.record_guess("unknown", 25, False)

        # Add to history
        self.history.append({"role": "ai", "message": response})

        # Speak the response
        self._speak(response)

        # Update last response time
        self.last_response_time = datetime.now()

        # Signal that thinking is finished
        self.thinking_finished.emit()

        return response

    def _generate_dynamic_response(self, shape_name, confidence, is_confident):
        """
        Generate a dynamic response based on the current state

        Args:
            shape_name: Name of the recognized shape
            confidence: Confidence percentage
            is_confident: Whether the AI is confident in its guess

        Returns:
            str: Dynamic response
        """
        # Format the confidence value properly
        confidence_value = int(confidence)

        # Check for special sequences
        if self.state.consecutive_correct_guesses >= 5:
            return random.choice(config.AI_MULTIPLE_ATTEMPTS[-5:]).format(
                guess=shape_name, confidence=confidence_value)

        if self.state.consecutive_correct_guesses >= 3:
            return random.choice(config.AI_MULTIPLE_ATTEMPTS[-10:-5]).format(
                guess=shape_name, confidence=confidence_value)

        if self.state.consecutive_low_confidence >= 3:
            return random.choice(config.AI_MULTIPLE_ATTEMPTS[5:10]).format(
                guess=shape_name, confidence=confidence_value)

        if self.state.consecutive_low_confidence >= 2:
            return random.choice(config.AI_MULTIPLE_ATTEMPTS[:5]).format(
                guess=shape_name, confidence=confidence_value)

        # Check for special shapes
        if shape_name == "heart" and random.random() < 0.7:
            return config.AI_EASTER_EGGS[-4] if random.random() < 0.5 else config.AI_EASTER_EGGS[-3]

        if shape_name == "smiley" and random.random() < 0.7:
            return config.AI_EASTER_EGGS[-2] if random.random() < 0.5 else config.AI_EASTER_EGGS[-1]

        # Check if this is a repeat shape
        if self.state.is_repeat_shape(shape_name) and random.random() < 0.7:
            return config.AI_EASTER_EGGS[-6] if random.random() < 0.5 else config.AI_EASTER_EGGS[-5]

        # Check drawing size and time for easter eggs
        drawing_time = self.state.drawing_duration()
        if drawing_time > 30 and random.random() < 0.7:  # Long drawing time
            return config.AI_EASTER_EGGS[-8] if random.random() < 0.5 else config.AI_EASTER_EGGS[-7]

        # Default responses based on confidence
        if confidence_value >= 80:
            # High confidence
            template = self._get_fresh_template('correct_high', config.AI_CORRECT_GUESS_HIGH_CONFIDENCE)
            return template.format(guess=shape_name, confidence=confidence_value)
        elif confidence_value >= 60:
            # Medium confidence
            template = self._get_fresh_template('correct_medium', config.AI_CORRECT_GUESS_MEDIUM_CONFIDENCE)
            return template.format(guess=shape_name, confidence=confidence_value)
        else:
            # Low confidence
            template = self._get_fresh_template('incorrect', config.AI_INCORRECT_GUESS)
            return template.format(guess=shape_name, confidence=confidence_value)

    def get_canvas_cleared_message(self):
        """
        Get a dynamic message for when the canvas is cleared

        Returns:
            str: Canvas cleared message
        """
        message = self._get_fresh_template('canvas_cleared', config.AI_CANVAS_CLEARED)
        self._speak(message)
        self.history.append({"role": "ai", "message": message})
        self.last_response_time = datetime.now()
        return message

    def get_training_saved_message(self, shape_name):
        """
        Get a message for when a drawing is saved to training

        Args:
            shape_name: Name of the saved shape

        Returns:
            str: Training saved message
        """
        message = random.choice(config.AI_ENCOURAGE_TRAINING).format(guess=shape_name)
        self._speak(message)
        self.history.append({"role": "ai", "message": message})
        self.last_response_time = datetime.now()
        return message

    def get_confidence_colored_message(self, shape_name, confidence):
        """
        Get a confidence-colored message

        Args:
            shape_name: Name of the recognized shape
            confidence: Confidence percentage

        Returns:
            str: Confidence-colored message
        """
        # Determine confidence level
        if confidence >= 80:
            template = config.AI_CONFIDENCE_COLORS["high"]
        elif confidence >= 60:
            template = config.AI_CONFIDENCE_COLORS["medium"]
        else:
            template = config.AI_CONFIDENCE_COLORS["low"]

        return template.format(guess=shape_name, confidence=confidence)

    def update_guess_result(self, was_correct, shape_name=None):
        """
        Update the state with the actual correctness of the guess

        Args:
            was_correct: Whether the guess was actually correct
            shape_name: The correct shape name (if different)

        Returns:
            str: Follow-up response
        """
        if not self.state.recent_guesses:
            return ""

        # Get the last guess
        last_guess = self.state.recent_guesses[-1]

        # If we already have the correct status, no need to update
        if last_guess['is_correct'] == was_correct:
            return ""

        # Update the state
        last_guess['is_correct'] = was_correct

        # If it was the wrong shape, update the shape
        if not was_correct and shape_name:
            last_guess['shape'] = shape_name

        # Update consecutive counters
        if was_correct:
            self.state.consecutive_correct_guesses += 1
            self.state.consecutive_low_confidence = 0
        else:
            self.state.consecutive_correct_guesses = 0
            self.state.consecutive_high_confidence = 0

            if last_guess['confidence'] < 50:
                self.state.consecutive_low_confidence += 1

        # Generate a follow-up response
        if was_correct:
            response = random.choice(config.AI_FOLLOW_UP_CORRECT)
        else:
            response = random.choice(config.AI_FOLLOW_UP_INCORRECT)

        self._speak(response)
        self.history.append({"role": "ai", "message": response})
        self.last_response_time = datetime.now()

        return response

    def _get_fresh_template(self, category, templates):
        """
        Get a template that hasn't been used recently

        Args:
            category: Template category
            templates: List of templates

        Returns:
            str: Fresh template
        """
        # Get templates that haven't been used
        unused = [t for t in templates if t not in self.used_templates[category]]

        # If all have been used, reset the category
        if not unused:
            self.used_templates[category] = set()
            unused = templates

        # Get a random template
        template = random.choice(unused)

        # Mark as used
        self.used_templates[category].add(template)

        return template

    def _speak(self, text):
        """
        Speak the given text if TTS is enabled

        Args:
            text: Text to speak
        """
        # Remove emojis and formatting for speaking
        clean_text = ''.join(c for c in text if c.isprintable() and c not in "{}%()[]")

        # Replace common abbreviations
        clean_text = clean_text.replace("AI", "A.I.")

        if self.use_tts and self.tts:
            self.tts.speak(clean_text)

    def notify_drawing_started(self):
        """Notify that drawing has started"""
        self.state.start_drawing()

    def reset_state(self):
        """Reset the conversation state"""
        self.state = ConversationState()
        self.history = []
        self.used_templates = {
            'greeting': set(),
            'thinking': set(),
            'correct_high': set(),
            'correct_medium': set(),
            'incorrect': set(),
            'low_confidence': set(),
            'canvas_cleared': set(),
            'easter_eggs': set()
        }