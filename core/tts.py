"""
Enhanced text-to-speech module for the Drawing AI App with proper error handling
"""
import os
import threading
import random
import tempfile
import platform
import logging

# Set up logging - THIS MUST COME BEFORE ANY LOGGER USAGE
logger = logging.getLogger(__name__)

# For offline TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
    logger.info("pyttsx3 available for offline TTS")
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not available - offline TTS will be disabled")

# For audio playback
try:
    from pygame import mixer
    PYGAME_AVAILABLE = True
    # Initialize mixer
    try:
        mixer.init()
        logger.info("Pygame mixer initialized")
    except Exception as e:
        logger.error(f"Failed to initialize pygame mixer: {e}")
        PYGAME_AVAILABLE = False
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("pygame not available - alternative audio playback will be used")

import config


class TextToSpeech:
    """Class for text-to-speech functionality with enhanced error handling"""

    def __init__(self, api_key=None, use_openai=False):
        """
        Initialize the text-to-speech engine

        Args:
            api_key: OpenAI API key (if using OpenAI TTS) - Not used anymore
            use_openai: Whether to use OpenAI TTS instead of pyttsx3 - Ignored, always uses pyttsx3
        """
        # Always use pyttsx3, never use OpenAI
        self.use_openai = False
        self.client = None
        self.engine = None
        self.temp_files = []  # Track temp files for cleanup

        # Voice properties
        self.voice = "default"  # Default voice for pyttsx3
        self.volume = 1.0     # 0.0 to 1.0
        self.speed = 1.0      # 0.5 to 2.0

        # Initialize pyttsx3 if available
        if PYTTSX3_AVAILABLE:
            try:
                self.engine = pyttsx3.init()

                # Set properties
                self.engine.setProperty('rate', 175)  # Speed (words per minute)
                self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

                # Get available voices
                voices = self.engine.getProperty('voices')
                if voices:
                    # Try to find a female voice, otherwise use the first voice
                    female_voices = [v for v in voices if 'female' in v.name.lower()]
                    if female_voices:
                        self.engine.setProperty('voice', female_voices[0].id)
                    else:
                        self.engine.setProperty('voice', voices[0].id)

                logger.info("pyttsx3 TTS initialized")
            except Exception as e:
                logger.error(f"Error initializing pyttsx3 TTS: {e}")
                self.engine = None

    def speak(self, text, callback=None):
        """
        Speak the given text

        Args:
            text: Text to speak
            callback: Function to call when speech is completed
        """
        # Use a thread to avoid blocking the UI
        threading.Thread(target=self._speak_thread, args=(text, callback), daemon=True).start()

    def _speak_thread(self, text, callback=None):
        """
        Speak the text in a separate thread

        Args:
            text: Text to speak
            callback: Function to call when speech is completed
        """
        try:
            if self.engine:
                self._speak_pyttsx3(text)
            else:
                logger.info(f"TTS not available. Would speak: {text}")

            # Call the callback if provided
            if callback:
                callback()
        except Exception as e:
            logger.error(f"Error in speak thread: {e}")
            # Still call callback on error to prevent UI waiting
            if callback:
                callback()

    def _speak_pyttsx3(self, text):
        """
        Speak using pyttsx3

        Args:
            text: Text to speak
        """
        try:
            # Set properties based on current settings
            if self.engine:
                self.engine.setProperty('rate', int(175 * self.speed))
                self.engine.setProperty('volume', self.volume)

                # Speak the text
                self.engine.say(text)
                self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Error speaking with pyttsx3: {e}")

    def cleanup(self):
        """Clean up all temporary files"""
        for file_path in self.temp_files[:]:
            self._cleanup_temp_file(file_path)

    def _cleanup_temp_file(self, file_path):
        """
        Clean up a temporary file

        Args:
            file_path: Path to the file to clean up
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                if file_path in self.temp_files:
                    self.temp_files.remove(file_path)
        except Exception as e:
            logger.error(f"Error cleaning up temp file: {e}")

    def set_voice(self, voice):
        """
        Set the voice to use

        Args:
            voice: Voice name/identifier
        """
        if self.engine and PYTTSX3_AVAILABLE:
            try:
                voices = self.engine.getProperty('voices')
                for v in voices:
                    if voice.lower() in v.name.lower():
                        self.engine.setProperty('voice', v.id)
                        self.voice = voice
                        break
            except Exception as e:
                logger.error(f"Error setting voice: {e}")

    def set_volume(self, volume):
        """
        Set the speech volume

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.volume = max(0.0, min(1.0, volume))
        if self.engine and PYTTSX3_AVAILABLE:
            self.engine.setProperty('volume', self.volume)

    def set_speed(self, speed):
        """
        Set the speech speed

        Args:
            speed: Speech speed multiplier (0.5 to 2.0)
        """
        self.speed = max(0.5, min(2.0, speed))
        if self.engine and PYTTSX3_AVAILABLE:
            self.engine.setProperty('rate', int(175 * self.speed))