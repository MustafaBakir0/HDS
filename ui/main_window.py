"""
main application window for the drawing ai app
"""
import os
import random
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStatusBar, QToolBar, QAction,
    QSplitter, QFileDialog, QMessageBox, QDialog, QCheckBox,
    QRadioButton, QGroupBox, QButtonGroup
)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon, QFont, QPixmap

import config
from ui.canvas import DrawingCanvas
from ui.toolbar import ToolBar
from ui.settings_dialog import SettingsDialog
from core.drawing_manager import DrawingManager
from core.conversation import ConversationManager
from core.recognition import DrawingRecognizer
from core.tts import TextToSpeech
from data.user_settings import UserSettings
from data.database import Database
from data.history import DrawingHistory
from arduino_input import ArduinoInput
from PyQt5.QtCore import Qt, QPoint
import logging

class LabelDialog(QDialog):
    """dialog for confirming and labeling a drawing"""

    def __init__(self, guess, parent=None):
        """initialize the label dialog"""
        super().__init__(parent)

        self.setWindowTitle("Confirm Shape")
        self.setMinimumWidth(300)

        # store the AI's guess
        self.ai_guess = guess
        self.selected_label = guess  # default to AI's guess
        self.is_correct = True       # default to correct

        # create UI
        self._init_ui()

    def _init_ui(self):
        """initialize the user interface"""
        layout = QVBoxLayout(self)

        # title label
        title = QLabel("Is this shape correct?")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ai guess info
        guess_label = QLabel(f"AI guessed: {self.ai_guess.capitalize()}")
        guess_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(guess_label)

        # correct/incorrect group
        correct_group = QGroupBox("Feedback")
        correct_layout = QVBoxLayout()

        self.correct_radio = QRadioButton("Yes, the guess is correct")
        self.incorrect_radio = QRadioButton("No, the guess is incorrect")

        self.correct_radio.setChecked(True)
        self.correct_radio.toggled.connect(self._on_correctness_changed)

        correct_layout.addWidget(self.correct_radio)
        correct_layout.addWidget(self.incorrect_radio)
        correct_group.setLayout(correct_layout)
        layout.addWidget(correct_group)

        # alternative label group
        self.alternative_group = QGroupBox("Select correct shape")
        alternative_layout = QVBoxLayout()

        # common shapes
        self.shape_buttons = {}
        common_shapes = [
            "circle", "rectangle", "square", "triangle",
            "star", "heart", "diamond", "oval", "line"
        ]

        for shape in common_shapes:
            radio = QRadioButton(shape.capitalize())
            if shape == self.ai_guess:
                radio.setChecked(True)
            radio.toggled.connect(lambda checked, s=shape: self._on_shape_selected(s, checked))
            alternative_layout.addWidget(radio)
            self.shape_buttons[shape] = radio

        self.alternative_group.setLayout(alternative_layout)
        self.alternative_group.setEnabled(False)  # disabled initially
        layout.addWidget(self.alternative_group)

        # buttons
        button_layout = QHBoxLayout()

        self.save_button = QPushButton("Save to Database")
        self.save_button.clicked.connect(self.accept)
        button_layout.addWidget(self.save_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def _on_correctness_changed(self, checked):
        """handle correctness radio button changes"""
        self.is_correct = self.correct_radio.isChecked()
        self.alternative_group.setEnabled(not self.is_correct)

        # if correct, use AI's guess
        if self.is_correct:
            self.selected_label = self.ai_guess
            if self.ai_guess in self.shape_buttons:
                self.shape_buttons[self.ai_guess].setChecked(True)

    def _on_shape_selected(self, shape, checked):
        """handle shape radio button selection"""
        if checked:
            self.selected_label = shape

    def get_result(self):
        """get the dialog result"""
        return {
            "label": self.selected_label,
            "is_correct": self.is_correct
        }


class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self, db=None, error_handler=None):
        super().__init__()

        # Initialize data components
        self.db = db or Database()
        self.error_handler = error_handler  # Store the error handler
        self.user_settings = UserSettings(self.db)
        self.drawing_history = DrawingHistory(self.db)

        # Initialize core components
        self.drawing_manager = DrawingManager()
        self.conversation = ConversationManager(use_tts=self.user_settings.get("use_tts"))

        # Set window properties
        self.setWindowTitle(config.APP_NAME)
        self.resize(config.DEFAULT_WINDOW_WIDTH, config.DEFAULT_WINDOW_HEIGHT)

        # Initialize UI
        self._init_ui()

        # Apply user settings
        self._apply_settings()

        # Show welcome message
        self.statusBar().showMessage("Welcome to Drawing AI App! Start drawing and I'll try to guess what it is.")

        # Current recognition results
        self.current_results = None

        # Display greeting
        greeting = self.conversation.get_greeting()
        self.response_label.setText(greeting)

        # Show tutorial if enabled
        if self.user_settings.get("show_tutorial"):
            QTimer.singleShot(1000, self.show_tutorial)

    def _init_ui(self):
        """Initialize the user interface"""
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout
        main_layout = QVBoxLayout(self.central_widget)

        # Create toolbar with drawing tools
        self.toolbar = ToolBar()
        main_layout.addWidget(self.toolbar)

        # Create splitter for canvas and info panel
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # Create drawing canvas
        self.canvas = DrawingCanvas(error_handler=self.error_handler)  # Pass error_handler to canvas
        splitter.addWidget(self.canvas)

        # Create info panel
        self.info_panel = self._create_info_panel()
        splitter.addWidget(self.info_panel)

        # Set splitter sizes (70% canvas, 30% info panel)
        splitter.setSizes([int(config.DEFAULT_WINDOW_WIDTH * 0.7), int(config.DEFAULT_WINDOW_WIDTH * 0.3)])

        # Connect the toolbar signals to canvas
        self.toolbar.color_selected.connect(self.canvas.set_pen_color)
        self.toolbar.size_selected.connect(self.canvas.set_pen_width)
        self.toolbar.eraser_selected.connect(self.canvas.toggle_eraser)
        self.toolbar.clear_selected.connect(self.on_canvas_clear)

        # Connect canvas signals
        self.canvas.canvas_updated.connect(self.on_canvas_updated)

        # Connect drawing manager signals
        self.drawing_manager.recognition_started.connect(self.on_recognition_started)
        self.drawing_manager.recognition_finished.connect(self.on_recognition_finished)

        # Connect conversation manager signals
        self.conversation.message_ready.connect(self.on_message_ready)
        self.conversation.thinking_started.connect(self.on_thinking_started)
        self.conversation.thinking_finished.connect(self.on_thinking_finished)

        # Create status bar
        self.statusBar().setFont(QFont("Arial", 10))

        # Create menu bar
        self._create_menu_bar()

    def _create_info_panel(self):
        """Create the information panel that shows AI responses"""
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        # Title for the panel
        title_label = QLabel("AI Assistant")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(title_label)

        # Add avatar placeholder (would be an image in real implementation)
        avatar_label = QLabel("🤖")  # Simple robot emoji as placeholder
        avatar_label.setFont(QFont("Arial", 48))
        avatar_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(avatar_label)

        # Add response area
        self.response_label = QLabel("Draw something and I'll try to guess what it is!")
        self.response_label.setFont(QFont("Arial", 12))
        self.response_label.setWordWrap(True)
        self.response_label.setAlignment(Qt.AlignCenter)
        self.response_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        info_layout.addWidget(self.response_label)

        # Add confidence display area
        self.confidence_label = QLabel("Confidence levels will appear here")
        self.confidence_label.setFont(QFont("Arial", 10))
        self.confidence_label.setWordWrap(True)
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("color: #606060;")
        info_layout.addWidget(self.confidence_label)

        # Add the "Guess My Drawing" button
        self.guess_button = QPushButton("Guess My Drawing!")
        self.guess_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.guess_button.setMinimumHeight(50)
        self.guess_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.guess_button.clicked.connect(self.guess_drawing)
        info_layout.addWidget(self.guess_button)

        # Add the "Save to Training Database" button
        self.save_training_button = QPushButton("Save to Training Database")
        self.save_training_button.setFont(QFont("Arial", 11))
        self.save_training_button.setMinimumHeight(40)
        self.save_training_button.setStyleSheet("background-color: #2196F3; color: white;")
        self.save_training_button.clicked.connect(self.save_to_training)
        self.save_training_button.setEnabled(False)  # Disabled until a guess is made
        info_layout.addWidget(self.save_training_button)

        # Add spacer at the bottom
        info_layout.addStretch(1)

        return info_widget

    def _create_menu_bar(self):
        """Create the application menu bar"""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        # New drawing action
        new_action = QAction("&New Drawing", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_drawing)
        file_menu.addAction(new_action)

        # Save action
        save_action = QAction("&Save Drawing", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_drawing)
        file_menu.addAction(save_action)

        # Open drawing action
        open_action = QAction("&Open Drawing History", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_drawing_history)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        # Settings action
        settings_action = QAction("Se&ttings", self)
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menu_bar.addMenu("&View")

        # Toggle dark mode action
        self.dark_mode_action = QAction("&Dark Mode", self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(self.user_settings.get("dark_mode"))
        self.dark_mode_action.triggered.connect(self.toggle_dark_mode)
        view_menu.addAction(self.dark_mode_action)

        # Training menu
        training_menu = menu_bar.addMenu("&Training")

        # View labeled shapes action
        view_training_action = QAction("&View Training Database", self)
        view_training_action.triggered.connect(self.view_training_database)
        training_menu.addAction(view_training_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        # Tutorial action
        tutorial_action = QAction("&Tutorial", self)
        tutorial_action.triggered.connect(self.show_tutorial)
        help_menu.addAction(tutorial_action)

        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _apply_settings(self):
        """Apply user settings to the application"""
        # Apply dark mode if enabled
        self.dark_mode_action.setChecked(self.user_settings.get("dark_mode"))
        if self.user_settings.get("dark_mode"):
            self._apply_dark_mode()
        else:
            self._apply_light_mode()

        # Set default pen color and width
        default_color = self.user_settings.get("default_pen_color")
        default_width = self.user_settings.get("default_pen_width")

        self.canvas.set_pen_color(default_color)
        self.canvas.set_pen_width(default_width)

        # Update TTS settings
        self.conversation.use_tts = self.user_settings.get("use_tts")
        if hasattr(self.conversation, 'tts') and self.conversation.tts:
            # If we had volume and speed controls in TTS, we would set them here
            pass

    def _apply_dark_mode(self):
        """Apply dark mode theme to the application"""
        # Set dark theme for the main window
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2D2D30;
                color: #E6E6E6;
            }
            QPushButton {
                background-color: #3F3F46;
                color: #E6E6E6;
                border: 1px solid #555555;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #606060;
            }
            QLabel {
                color: #E6E6E6;
            }
            QMenuBar {
                background-color: #2D2D30;
                color: #E6E6E6;
            }
            QMenuBar::item:selected {
                background-color: #3F3F46;
            }
            QMenu {
                background-color: #2D2D30;
                color: #E6E6E6;
            }
            QMenu::item:selected {
                background-color: #3F3F46;
            }
            QToolBar {
                background-color: #2D2D30;
                border: 1px solid #3F3F46;
            }
        """)

        # Update response label
        self.response_label.setStyleSheet("background-color: #3F3F46; color: #E6E6E6; padding: 10px; border-radius: 5px;")

        # Update confidence label
        self.confidence_label.setStyleSheet("color: #B0B0B0;")

        # Keep the button colors
        self.guess_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.save_training_button.setStyleSheet("background-color: #2196F3; color: white;")

    def _apply_light_mode(self):
        """Apply light mode theme to the application"""
        # Clear the style sheet to return to default
        self.setStyleSheet("")

        # Set light theme for specific components
        self.response_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.confidence_label.setStyleSheet("color: #606060;")

        # Keep the button colors
        self.guess_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.save_training_button.setStyleSheet("background-color: #2196F3; color: white;")

    def new_drawing(self):
        """Start a new drawing"""
        # Ask for confirmation if there's something on the canvas
        # In a real implementation, we would check if the canvas is empty
        reply = QMessageBox.question(
            self,
            "New Drawing",
            "Do you want to start a new drawing? This will clear the current canvas.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.on_canvas_clear()

    def on_canvas_clear(self):
        """Handle canvas clear event"""
        # Clear the canvas
        self.canvas.clear()

        # Get and display a canvas cleared message
        message = self.conversation.get_canvas_cleared_message()
        self.response_label.setText(message)

        # Reset confidence label
        self.confidence_label.setText("Confidence levels will appear here")

        # Disable save training button
        self.save_training_button.setEnabled(False)

        # Clear current results
        self.current_results = None

        # Update status bar
        self.statusBar().showMessage("Canvas cleared. Ready for a new drawing!")

    def on_canvas_updated(self):
        """Handle canvas update event"""
        # Auto-save if enabled
        if self.user_settings.get("save_drawings_automatically"):
            # This would save after every update, which might be too frequent
            # In a real implementation, we might use a timer to debounce
            pass

    def on_recognition_started(self):
        """Handle recognition start event"""
        # Update UI to show we're recognizing
        self.guess_button.setEnabled(False)
        self.save_training_button.setEnabled(False)
        self.statusBar().showMessage("AI is analyzing your drawing...")

    def on_recognition_finished(self, results):
        """Handle recognition finish event"""
        # Store the current results
        self.current_results = results

        # Update UI with results
        self.guess_button.setEnabled(True)

        # Enable save training button if recognition was successful
        if results["success"]:
            self.save_training_button.setEnabled(True)

        # Process the results with conversation manager
        self.conversation.process_recognition(results)

        # Show the confidence scores if enabled
        if self.user_settings.get("show_confidence"):
            self._update_confidence_display(results)
        else:
            self.confidence_label.setText("")

        # Update status bar
        self.statusBar().showMessage("AI has made a guess! How did it do?")

        # Save to history if auto-save is enabled
        if self.user_settings.get("save_drawings_automatically"):
            self.drawing_history.save_drawing(self.canvas.get_image(), results)

    def _update_confidence_display(self, results):
        """Update the confidence display with recognition results"""
        if results["success"] and results["guesses"]:
            # Format the confidence scores
            confidence_text = ""
            for guess in results["guesses"]:
                confidence_text += f"{guess['name'].capitalize()}: {guess['confidence']}%, "

            # Remove the trailing comma and space
            confidence_text = confidence_text[:-2]

            self.confidence_label.setText(confidence_text)
        else:
            self.confidence_label.setText("Could not determine confidence levels")

    def on_message_ready(self, message):
        """Handle new message from conversation manager"""
        self.response_label.setText(message)

    def on_thinking_started(self):
        """Handle thinking start event"""
        # We could add a thinking animation here
        pass

    def on_thinking_finished(self):
        """Handle thinking finish event"""
        # We could stop the thinking animation here
        pass

    def guess_drawing(self):
        """Handle the user clicking the guess button"""
        # Get the current drawing
        image = self.canvas.get_image()

        # Ask the drawing manager to recognize it
        self.drawing_manager.recognize_drawing(image)

    def save_to_training(self):
        """Save the current drawing to the training database"""
        # Make sure we have results and a drawing
        if not self.current_results or not self.current_results["success"]:
            QMessageBox.warning(
                self,
                "No Recognition Results",
                "Please draw something and have the AI guess it first."
            )
            return

        # Get the top guess
        top_guess = self.current_results["guesses"][0]["name"]
        confidence = self.current_results["guesses"][0]["confidence"]

        # Show the label confirmation dialog
        dialog = LabelDialog(top_guess, self)
        if dialog.exec_():
            # Get the result
            result = dialog.get_result()
            shape_name = result["label"]
            is_correct = result["is_correct"]

            # Save the image to a file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_{shape_name}_{timestamp}.png"

            # Make sure the directory exists
            training_dir = "training_shapes"
            os.makedirs(training_dir, exist_ok=True)

            # Full path to the file
            filepath = os.path.join(training_dir, filename)

            # Save the image
            image = self.canvas.get_image()
            if image.save(filepath):
                # Save to database
                self.db.save_labeled_shape(filepath, shape_name, confidence, is_correct)

                # Show confirmation
                QMessageBox.information(
                    self,
                    "Drawing Saved",
                    f"Your drawing has been saved to the training database as a {shape_name.capitalize()}."
                )

                # Update status bar
                self.statusBar().showMessage(f"Drawing saved to training database as {shape_name}")
            else:
                QMessageBox.warning(
                    self,
                    "Save Error",
                    "There was an error saving your drawing to the training database."
                )

    def view_training_database(self):
        """Open a dialog to view the training database"""
        try:
            from ui.training_dialog import TrainingDialog

            # Check if we have any shapes first
            shapes = self.db.get_labeled_shapes()

            if not shapes:
                QMessageBox.information(
                    self,
                    "Training Database",
                    "The training database is currently empty. Save some drawings first!"
                )
                return

            # Create and show the training dialog
            dialog = TrainingDialog(self.db, self)
            dialog.exec_()
        except Exception as e:
            if hasattr(self, 'error_handler') and self.error_handler:
                self.error_handler.handle_error(e, "Opening training database")
            else:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Could not open training database: {str(e)}"
                )

    def save_drawing(self):
        """Save the current drawing to a file"""
        # Get the current drawing
        image = self.canvas.get_image()

        # Ask for a filename
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Drawing",
            "",
            "PNG Files (*.png);;All Files (*)",
            options=options
        )

        if filename:
            # Ensure the filename has a .png extension
            if not filename.lower().endswith('.png'):
                filename += '.png'

            # Save the image
            if image.save(filename):
                self.statusBar().showMessage(f"Drawing saved to {filename}")

                # Also save to history
                self.drawing_history.save_drawing(image)
            else:
                self.statusBar().showMessage("Error saving drawing")

    def open_drawing_history(self):
        """Open the drawing history"""
        try:
            from data.history import HistoryDialog

            # Create and show the history dialog
            dialog = HistoryDialog(self.drawing_history, self)

            # Connect the dialog's signal
            dialog.drawing_selected.connect(self.on_history_drawing_selected)

            dialog.exec_()
        except Exception as e:
            if hasattr(self, 'error_handler') and self.error_handler:
                self.error_handler.handle_error(e, "Opening drawing history")
            else:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Could not open drawing history: {str(e)}"
                )

    def on_history_drawing_selected(self, pixmap, data):
        """Handle a drawing being selected from history"""
        # Set the canvas to the selected drawing
        self.canvas.set_image(pixmap)

        # Update the UI with the data
        if 'top_guess' in data and data['top_guess']:
            self.response_label.setText(f"This drawing was recognized as a {data['top_guess'].capitalize()}")

            if 'confidence' in data and data['confidence']:
                self.confidence_label.setText(f"{data['top_guess'].capitalize()}: {data['confidence']}%")

    def show_settings(self):
        """Show the settings dialog"""
        dialog = SettingsDialog(self.user_settings, self.db, self)
        dialog.settings_changed.connect(self._on_settings_changed)
        dialog.exec_()

    def _on_settings_changed(self, new_settings):
        """Handle settings changes"""
        # Apply the new settings
        self._apply_settings()

        # Show a confirmation message
        self.statusBar().showMessage("Settings updated")

    def toggle_dark_mode(self):
        """Toggle dark mode"""
        dark_mode = self.user_settings.toggle_dark_mode()

        if dark_mode:
            self._apply_dark_mode()
        else:
            self._apply_light_mode()

    def show_tutorial(self):
        """Show the tutorial dialog"""
        QMessageBox.information(
            self,
            "Tutorial",
            """
            <h3>Welcome to Drawing AI App!</h3>
            <p>This application lets you draw on the canvas and uses AI to recognize what you've drawn.</p>
            <h4>How to use:</h4>
            <ol>
                <li>Use the toolbar to select colors and pen sizes</li>
                <li>Draw something on the canvas</li>
                <li>Click the "Guess My Drawing" button to have the AI guess what you've drawn</li>
                <li>The AI will respond with its guess and confidence level</li>
                <li>If the AI guesses correctly, click "Save to Training Database" to help the AI learn</li>
            </ol>
            <p>The more drawings you save to the training database, the better the AI will get at recognizing shapes!</p>
            <p>Have fun drawing!</p>
            """
        )

    def show_about(self):
        """Show information about the application"""
        QMessageBox.about(
            self,
            "About Drawing AI App",
            f"""
            <h3>Drawing AI App v{config.APP_VERSION}</h3>
            <p>A fun drawing application with AI recognition</p>
            <p>This application demonstrates the integration of:</p>
            <ul>
                <li>PyQt5 for the user interface</li>
                <li>OpenCV for shape detection</li>
                <li>Text-to-speech for interactive communication</li>
                <li>Database for storing labeled training data</li>
            </ul>
            <p>Created for educational and demonstration purposes.</p>
            """
        )

    def closeEvent(self, event):
        """Handle window close event"""
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Exit",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Close the database connection
            if hasattr(self, 'db') and self.db:
                self.db.close()
            event.accept()
        else:
            event.ignore()

    def set_debug_mode(self, enabled):
        """
        Enable or disable debug mode

        Args:
            enabled: Whether to enable debug mode
        """
        self.debug_mode = enabled

        # Log the change
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG if enabled else logging.INFO)

        # Update any components that need to know about debug mode
        if hasattr(self, 'drawing_manager') and hasattr(self.drawing_manager, 'recognizer'):
            self.drawing_manager.recognizer.debug = enabled

        # Update status bar
        if enabled:
            self.statusBar().showMessage("Debug mode enabled")

    def save_settings(self):
        """Save application settings"""
        if hasattr(self, 'user_settings'):
            try:
                self.user_settings.save_settings()
            except Exception as e:
                if hasattr(self, 'error_handler') and self.error_handler:
                    self.error_handler.handle_error(e, "Saving settings")

                    