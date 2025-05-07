"""
User settings manager for the Drawing AI App
"""
from data.database import Database


class UserSettings:
    """Manager for user settings"""

    # Default settings
    DEFAULT_SETTINGS = {
        "dark_mode": False,
        "use_tts": True,
        "tts_volume": 0.8,
        "tts_speed": 1.0,
        "default_pen_color": "#000000",
        "default_pen_width": 3,
        "show_tutorial": True,
        "show_confidence": True,
        "save_drawings_automatically": False
    }

    def __init__(self, db=None):
        """
        Initialize the user settings manager

        Args:
            db: Database instance, will create one if not provided
        """
        self.db = db or Database()
        self.settings = self.DEFAULT_SETTINGS.copy()

        # Load settings from database
        self.load_settings()

    def load_settings(self):
        """Load all settings from the database"""
        for key in self.DEFAULT_SETTINGS:
            value = self.db.get_setting(key, self.DEFAULT_SETTINGS[key])
            self.settings[key] = value

    def save_settings(self):
        """Save all settings to the database"""
        for key, value in self.settings.items():
            self.db.save_setting(key, value)

    def get(self, key, default=None):
        """
        Get a setting value

        Args:
            key: Setting key
            default: Default value if the setting doesn't exist

        Returns:
            Setting value
        """
        value = self.settings.get(key, default)

        # Convert string values to appropriate types
        if isinstance(value, str):
            # Handle booleans
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False

            # Handle integers
            try:
                if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    return int(value)
            except:
                pass

            # Handle floats
            try:
                return float(value)
            except:
                pass

        return value

        return value

    def set(self, key, value):
        """
        Set a setting value

        Args:
            key: Setting key
            value: Setting value

        Returns:
            bool: True if successful, False otherwise
        """
        self.settings[key] = value
        return self.db.save_setting(key, value)

    def reset(self, key=None):
        """
        Reset a setting to its default value, or all settings if key is None

        Args:
            key: Setting key to reset, or None to reset all settings

        Returns:
            bool: True if successful, False otherwise
        """
        if key is None:
            # Reset all settings
            self.settings = self.DEFAULT_SETTINGS.copy()
            return self.save_settings()
        elif key in self.DEFAULT_SETTINGS:
            # Reset a specific setting
            self.settings[key] = self.DEFAULT_SETTINGS[key]
            return self.db.save_setting(key, self.DEFAULT_SETTINGS[key])
        else:
            return False

    def toggle_dark_mode(self):
        """
        Toggle dark mode setting

        Returns:
            bool: New dark mode value
        """
        dark_mode = not self.get("dark_mode")
        self.set("dark_mode", dark_mode)
        return dark_mode

    def toggle_tts(self):
        """
        Toggle text-to-speech setting

        Returns:
            bool: New TTS value
        """
        use_tts = not self.get("use_tts")
        self.set("use_tts", use_tts)
        return use_tts