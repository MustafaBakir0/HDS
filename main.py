#!/usr/bin/env python3
"""
Drawing AI App - An enhanced interactive drawing application with AI recognition

This module contains the main application entry point with improved error handling,
performance optimizations, and enhanced user experience.
"""
import sys
import os
import logging
import traceback
from datetime import datetime
from pathlib import Path
import argparse
from dotenv import load_dotenv
from PyQt5.QtGui import QMouseEvent, QCursor
from PyQt5.QtCore import QEvent, Qt
import config

from PyQt5.QtWidgets import (
    QApplication, QMessageBox, QSplashScreen, QStyleFactory
)
from PyQt5.QtGui import QIcon, QPixmap, QFontDatabase
from PyQt5.QtCore import Qt, QTimer, QSettings, QCoreApplication, QTranslator

# Load environment variables from .env file (for API keys)
load_dotenv("environment.env")  # Explicitly specify the file path

# Configure application-wide settings
QCoreApplication.setApplicationName("Drawing AI App")
QCoreApplication.setOrganizationName("Drawing AI")
QCoreApplication.setApplicationVersion("1.0.0")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Get logger for this module
logger = logging.getLogger(__name__)


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler for unhandled exceptions

    Args:
        exc_type: Exception type
        exc_value: Exception value
        exc_traceback: Exception traceback
    """
    # Skip KeyboardInterrupt
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Log the exception
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

    # Format the traceback for display
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    tb_text = ''.join(tb_lines)

    # Show error dialog if QApplication exists
    app = QApplication.instance()
    if app:
        # Create and show error dialog
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Critical Error")
        msg_box.setText("An unhandled error has occurred. The application will exit.")
        msg_box.setInformativeText(str(exc_value))
        msg_box.setDetailedText(tb_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


def run_application():
    """Main function to run the application with full error handling"""
    try:
        # Install the global exception handler
        sys.excepthook = handle_exception

        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Drawing AI App")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--style", help="Set application style (Fusion, Windows, etc.)")
        parser.add_argument("--lang", help="Set application language (e.g., en, es)")
        parser.add_argument("--no-arduino", action="store_true", help="Disable Arduino controller at startup")
        parser.add_argument("--port", help="Specify Arduino COM port (e.g., COM5)")
        args = parser.parse_args()

        # Set logging level based on debug flag or environment variable
        if args.debug or os.environ.get("DEBUG", "").lower() == "true":
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # Create the application
        app = QApplication(sys.argv)

        # Print important environment variables for debugging
        logger.debug(f"OPENAI_API_KEY exists: {bool(os.environ.get('OPENAI_API_KEY'))}")
        logger.debug(f"USE_OPENAI_TTS: {os.environ.get('USE_OPENAI_TTS')}")
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Working directory: {os.getcwd()}")

        # Load and set the application stylesheet
        _load_stylesheet(app)

        # Set application style
        if args.style:
            app.setStyle(args.style)
        else:
            # Use Fusion style by default for consistent cross-platform appearance
            app.setStyle("Fusion")

        # List available styles
        logger.debug("Available styles: %s", QStyleFactory.keys())
        logger.debug("Using style: %s", app.style().objectName())

        # Load custom fonts
        _load_fonts()

        # Set up translation if specified
        if args.lang:
            _setup_translation(app, args.lang)

        # Show splash screen while loading
        splash_pixmap = QPixmap("resources/splash.png")
        if splash_pixmap.isNull():
            # Create a default splash screen if the image is not found
            logger.warning("Splash screen image not found. Creating default.")
            splash_pixmap = QPixmap(400, 300)
            splash_pixmap.fill(Qt.white)

        splash = QSplashScreen(splash_pixmap)
        splash.show()
        app.processEvents()

        # Update splash screen with loading message
        splash.showMessage("Loading modules...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
        app.processEvents()

        # Import modules after splash screen is shown
        # This helps to avoid importing before exception handler is set up
        from ui.main_window import MainWindow
        from error_handler import ErrorHandler
        from data.database import Database
        from arduino_input import setup_arduino_integration

        # Create error handler
        error_handler = ErrorHandler(app)

        # Update splash screen
        splash.showMessage("Initializing database...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
        app.processEvents()

        # Initialize database with error handling
        try:
            db = Database()
            logger.debug("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

            # Show error message
            QMessageBox.critical(
                None,
                "Database Error",
                f"Failed to initialize the database: {str(e)}\n\n"
                "The application will continue without database support."
            )

            db = None
            
        # Auto-import dataset if available
        splash.showMessage("Checking for training dataset...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
        app.processEvents()
        
     

        # Update splash screen
        splash.showMessage("Creating main window...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
        app.processEvents()

        # Create and configure the main window
        window = MainWindow(db, error_handler)
        logger.debug("Main window created")

        # Set debug mode if specified
        if args.debug or os.environ.get("DEBUG", "").lower() == "true":
            if hasattr(window, 'set_debug_mode'):
                window.set_debug_mode(True)
        
        # Initialize Arduino controller (enabled by default unless --no-arduino specified)
        if not args.no_arduino:
            splash.showMessage("Connecting to Arduino...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
            app.processEvents()
            
            try:
                # Use specified port if provided
                port = args.port if hasattr(args, 'port') and args.port else None
                setup_arduino_integration(window, port)
                logger.info(f"Arduino controller initialized{f' on port {port}' if port else ''}")
            except Exception as e:
                logger.error(f"Arduino initialization failed: {e}")
                
                # Show warning but continue
                QMessageBox.warning(
                    None,
                    "Arduino Connection",
                    f"Failed to initialize Arduino controller: {str(e)}\n\n"
                    "The application will continue without Arduino support."
                )

        # Close splash screen and show main window
        splash.finish(window)
        window.show()
        logger.info("Application started successfully")

        # Start the application event loop
        exit_code = app.exec_()

        # Clean up before exit
        _cleanup(window, db)

        # Return the exit code
        return exit_code

    except Exception as e:
        # Log the exception
        logger.critical("Fatal error during application startup", exc_info=True)

        # Show error dialog
        app = QApplication.instance()
        if app:
            QMessageBox.critical(
                None,
                "Fatal Error",
                f"A fatal error occurred during application startup:\n\n{str(e)}\n\n"
                "The application will now exit."
            )

        # Return error code
        return 1


def _load_stylesheet(app):
    """
    Load and apply the application stylesheet

    Args:
        app: QApplication instance
    """
    try:
        # Check for custom stylesheet file
        style_path = Path("resources/style.css")
        if style_path.exists():
            with open(style_path, "r") as f:
                stylesheet = f.read()
                app.setStyleSheet(stylesheet)
                logger.debug("Applied custom stylesheet")
        else:
            logger.debug("No custom stylesheet found at resources/style.css")
    except Exception as e:
        logger.warning(f"Failed to load stylesheet: {e}")


def _load_fonts():
    """Load custom fonts for the application"""
    try:
        # Create fonts directory if it doesn't exist
        os.makedirs("resources/fonts", exist_ok=True)

        # Load all fonts in the fonts directory
        font_dir = Path("resources/fonts")
        font_count = 0

        if font_dir.exists():
            for font_file in font_dir.glob("*.ttf"):
                font_id = QFontDatabase.addApplicationFont(str(font_file))
                if font_id != -1:
                    font_count += 1

            if font_count > 0:
                logger.debug(f"Loaded {font_count} custom fonts")
            else:
                logger.debug("No fonts found in resources/fonts directory")
        else:
            logger.debug("Fonts directory does not exist")
    except Exception as e:
        logger.warning(f"Failed to load custom fonts: {e}")


def _setup_translation(app, lang_code):
    """
    Set up translation for the application

    Args:
        app: QApplication instance
        lang_code: Language code (e.g., 'en', 'es')
    """
    try:
        translation_path = Path(f"resources/translations/drawing_ai_{lang_code}.qm")

        # Check if file exists before trying to load
        if translation_path.exists():
            translator = QTranslator()
            if translator.load(str(translation_path)):
                app.installTranslator(translator)
                logger.debug(f"Loaded translation for {lang_code}")
            else:
                logger.warning(f"Failed to load translation file for {lang_code}")
        else:
            logger.warning(f"Translation file for {lang_code} not found at {translation_path}")
    except Exception as e:
        logger.warning(f"Failed to set up translation: {e}")


def _cleanup(window, db):
    """
    Clean up resources before exit

    Args:
        window: MainWindow instance
        db: Database instance
    """
    try:
        # Save application state
        if window and hasattr(window, 'save_settings'):
            window.save_settings()

        # Close database connection
        if db:
            db.close()

        logger.info("Application cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def create_dirs():
    """Create necessary directories for the application"""
    dirs = [
        "data",
        "saved_drawings",
        "training_shapes",
        "models",
        "logs",
        "resources",
        "resources/fonts",
        "resources/translations",
        "backups",
        "auto_save",
        "arduino"  # Add arduino directory for sketches
    ]

    for directory in dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")


if __name__ == "__main__":
    # Create necessary directories
    create_dirs()

    # Run the application
    exit_code = run_application()

    # Exit with the appropriate code
    sys.exit(exit_code)