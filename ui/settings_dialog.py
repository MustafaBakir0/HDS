"""
Enhanced settings dialog for the Drawing AI App with expanded options
for shape recognition, confidence thresholds, and performance settings
"""
import os
import logging
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QCheckBox, QSlider, QComboBox, QPushButton, QTabWidget,
    QColorDialog, QSpinBox, QListWidget, QListWidgetItem, QMessageBox,
    QLineEdit, QFileDialog, QProgressBar, QWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFont

import config
from data.user_settings import UserSettings
from error_handler import handle_errors, handle_ui_errors

# Set up logging
logger = logging.getLogger(__name__)


class ColorButton(QPushButton):
    """Button for selecting a color"""

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(30, 30)
        self.setStyleSheet(f"background-color: {color}; border: 2px solid #888888;")

    def set_color(self, color):
        """Set the button color"""
        self.color = color
        self.setStyleSheet(f"background-color: {color}; border: 2px solid #888888;")


class SettingsDialog(QDialog):
    """Enhanced dialog for changing application settings"""

    # Signals
    settings_changed = pyqtSignal(dict)  # Emitted when settings are changed
    database_backup_requested = pyqtSignal(str)  # Emitted when database backup is requested
    model_retrain_requested = pyqtSignal()  # Emitted when model retraining is requested

    def __init__(self, user_settings, db=None, parent=None):
        super().__init__(parent)

        # Store the user settings and database
        self.user_settings = user_settings
        self.db = db

        # Store the original settings for comparison
        self.original_settings = user_settings.settings.copy()

        # Set up the dialog
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self.setMinimumHeight(500)

        # Create the UI
        self._init_ui()

        # Apply validation for numeric fields
        self._setup_validators()

        # Status bar at the bottom
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: green;")

        # Clear status after a delay
        self.status_timer = QTimer(self)
        self.status_timer.setSingleShot(True)
        self.status_timer.timeout.connect(self._clear_status)

    def _init_ui(self):
        """Initialize the user interface"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Add tabs
        self.tab_widget.addTab(self._create_appearance_tab(), "Appearance")
        self.tab_widget.addTab(self._create_recognition_tab(), "Recognition")
        self.tab_widget.addTab(self._create_voice_tab(), "Voice")
        self.tab_widget.addTab(self._create_performance_tab(), "Performance")
        self.tab_widget.addTab(self._create_backup_tab(), "Backup & Data")

        layout.addWidget(self.tab_widget)

        # Status label
        layout.addWidget(self.status_label)

        # Buttons
        buttons_layout = QHBoxLayout()

        # Reset button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._on_reset_clicked)
        buttons_layout.addWidget(reset_button)

        # Add spacer
        buttons_layout.addStretch()

        # Save button
        save_button = QPushButton("Save")
        save_button.setDefault(True)
        save_button.clicked.connect(self._on_save_clicked)
        buttons_layout.addWidget(save_button)

        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)

        layout.addLayout(buttons_layout)

    def _create_appearance_tab(self):
        """Create the appearance settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Appearance group
        appearance_group = QGroupBox("Theme Settings")
        appearance_layout = QFormLayout(appearance_group)

        # Dark mode
        self.dark_mode_checkbox = QCheckBox()
        self.dark_mode_checkbox.setChecked(self.user_settings.get("dark_mode"))
        appearance_layout.addRow("Dark Mode:", self.dark_mode_checkbox)

        # Default pen color
        color_layout = QHBoxLayout()

        self.color_button = ColorButton(self.user_settings.get("default_pen_color"))
        self.color_button.clicked.connect(self._on_color_button_clicked)
        color_layout.addWidget(self.color_button)

        self.color_label = QLabel(self.user_settings.get("default_pen_color"))
        color_layout.addWidget(self.color_label)

        appearance_layout.addRow("Default Pen Color:", color_layout)

        # Default pen width
        self.pen_width_spinner = QSpinBox()
        self.pen_width_spinner.setRange(1, 20)
        self.pen_width_spinner.setValue(self.user_settings.get("default_pen_width"))
        appearance_layout.addRow("Default Pen Width:", self.pen_width_spinner)

        # Canvas background color
        bg_color_layout = QHBoxLayout()

        self.bg_color_button = ColorButton(self.user_settings.get("canvas_background_color", config.DEFAULT_BACKGROUND_COLOR))
        self.bg_color_button.clicked.connect(self._on_bg_color_button_clicked)
        bg_color_layout.addWidget(self.bg_color_button)

        self.bg_color_label = QLabel(self.user_settings.get("canvas_background_color", config.DEFAULT_BACKGROUND_COLOR))
        bg_color_layout.addWidget(self.bg_color_label)

        appearance_layout.addRow("Canvas Background:", bg_color_layout)

        layout.addWidget(appearance_group)

        # UI Settings Group
        ui_group = QGroupBox("UI Settings")
        ui_layout = QFormLayout(ui_group)

        # Show toolbar
        self.show_toolbar_checkbox = QCheckBox()
        self.show_toolbar_checkbox.setChecked(self.user_settings.get("show_toolbar", True))
        ui_layout.addRow("Show Toolbar:", self.show_toolbar_checkbox)

        # Show status bar
        self.show_statusbar_checkbox = QCheckBox()
        self.show_statusbar_checkbox.setChecked(self.user_settings.get("show_statusbar", True))
        ui_layout.addRow("Show Status Bar:", self.show_statusbar_checkbox)

        # Show tutorial
        self.tutorial_checkbox = QCheckBox()
        self.tutorial_checkbox.setChecked(self.user_settings.get("show_tutorial"))
        ui_layout.addRow("Show Tutorial for New Users:", self.tutorial_checkbox)

        # Show confidence
        self.confidence_checkbox = QCheckBox()
        self.confidence_checkbox.setChecked(self.user_settings.get("show_confidence"))
        ui_layout.addRow("Show Confidence Percentages:", self.confidence_checkbox)

        # Color code confidence
        self.color_code_confidence_checkbox = QCheckBox()
        self.color_code_confidence_checkbox.setChecked(self.user_settings.get("color_code_confidence", True))
        ui_layout.addRow("Color Code Confidence:", self.color_code_confidence_checkbox)

        layout.addWidget(ui_group)

        # Add stretch to push everything to the top
        layout.addStretch()

        return tab

    def _create_recognition_tab(self):
        """Create the recognition settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Recognition settings group
        recognition_group = QGroupBox("Recognition Settings")
        recognition_layout = QFormLayout(recognition_group)

        # Confidence threshold slider
        self.confidence_threshold_slider = QSlider(Qt.Horizontal)
        self.confidence_threshold_slider.setRange(30, 90)
        self.confidence_threshold_slider.setValue(self.user_settings.get("confidence_threshold", config.DEFAULT_CONFIDENCE_THRESHOLD))
        self.confidence_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_threshold_slider.setTickInterval(10)

        # Value label for confidence threshold
        self.confidence_threshold_value = QLabel(f"{self.confidence_threshold_slider.value()}%")
        self.confidence_threshold_slider.valueChanged.connect(
            lambda value: self.confidence_threshold_value.setText(f"{value}%")
        )

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.confidence_threshold_slider)
        threshold_layout.addWidget(self.confidence_threshold_value)

        recognition_layout.addRow("Confidence Threshold:", threshold_layout)

        # Enable ML recognition
        self.enable_ml_checkbox = QCheckBox()
        self.enable_ml_checkbox.setChecked(self.user_settings.get("enable_ml", True))
        recognition_layout.addRow("Enable Machine Learning:", self.enable_ml_checkbox)

        # Enable incremental training
        self.incremental_training_checkbox = QCheckBox()
        self.incremental_training_checkbox.setChecked(self.user_settings.get("enable_incremental_training", True))
        recognition_layout.addRow("Enable Background Training:", self.incremental_training_checkbox)

        layout.addWidget(recognition_group)

        # Shapes group
        shapes_group = QGroupBox("Recognized Shapes")
        shapes_layout = QVBoxLayout(shapes_group)

        # Info label
        shapes_layout.addWidget(QLabel("Select shapes to recognize:"))

        # Get current recognized shapes
        recognized_shapes = self.user_settings.get("recognized_shapes", [
            'circle', 'ellipse', 'rectangle', 'square', 'triangle',
            'diamond', 'line', 'star', 'heart', 'arrow', 'pentagon',
            'hexagon', 'cross', 'smiley', 'other'
        ])

        # Shape list widget
        self.shapes_list = QListWidget()
        self.shapes_list.setMaximumHeight(150)

        # Add all possible shapes
        all_shapes = [
            'circle', 'ellipse', 'rectangle', 'square', 'triangle',
            'diamond', 'line', 'star', 'heart', 'arrow', 'pentagon',
            'hexagon', 'cross', 'smiley', 'other'
        ]

        for shape in all_shapes:
            item = QListWidgetItem(shape.capitalize())
            item.setData(Qt.UserRole, shape)
            item.setCheckState(Qt.Checked if shape in recognized_shapes else Qt.Unchecked)
            self.shapes_list.addItem(item)

        shapes_layout.addWidget(self.shapes_list)

        # Add custom shape option
        custom_layout = QHBoxLayout()
        self.custom_shape_input = QLineEdit()
        self.custom_shape_input.setPlaceholderText("Enter custom shape name")
        custom_layout.addWidget(self.custom_shape_input)

        add_shape_button = QPushButton("Add")
        add_shape_button.clicked.connect(self._add_custom_shape)
        custom_layout.addWidget(add_shape_button)

        shapes_layout.addLayout(custom_layout)

        layout.addWidget(shapes_group)

        # Statistics group if database is available
        if self.db:
            stats_group = QGroupBox("Recognition Statistics")
            stats_layout = QVBoxLayout(stats_group)

            # Get statistics
            try:
                shape_stats = self.db.get_shape_stats(limit=5)

                if shape_stats:
                    stats_layout.addWidget(QLabel("<b>Top 5 Shapes by Recognition Accuracy:</b>"))

                    for stat in shape_stats:
                        accuracy = stat.get("accuracy", 0)
                        count = stat.get("drawn_count", 0)

                        if count >= 5:  # Only show shapes with enough samples
                            label = QLabel(f"{stat['shape_name'].capitalize()}: " +
                                          f"{accuracy:.1f}% accuracy ({count} samples)")
                            stats_layout.addWidget(label)
                else:
                    stats_layout.addWidget(QLabel("No recognition statistics available yet."))
            except Exception as e:
                logger.error(f"Error loading shape statistics: {e}")
                stats_layout.addWidget(QLabel("Error loading statistics."))

            layout.addWidget(stats_group)

        # Add stretch to push everything to the top
        layout.addStretch()

        return tab

    def _create_voice_tab(self):
        """Create the voice settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Voice Settings group
        voice_group = QGroupBox("Voice Settings")
        voice_layout = QFormLayout(voice_group)

        # Text-to-speech
        self.tts_checkbox = QCheckBox()
        self.tts_checkbox.setChecked(self.user_settings.get("use_tts"))
        self.tts_checkbox.toggled.connect(self._on_tts_toggled)
        voice_layout.addRow("Enable Voice:", self.tts_checkbox)

        # TTS volume
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(self.user_settings.get("tts_volume") * 100))
        self.volume_slider.setEnabled(self.tts_checkbox.isChecked())

        # Volume value label
        self.volume_value = QLabel(f"{self.volume_slider.value()}%")
        self.volume_slider.valueChanged.connect(
            lambda value: self.volume_value.setText(f"{value}%")
        )

        volume_layout = QHBoxLayout()
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_value)

        voice_layout.addRow("Voice Volume:", volume_layout)

        # TTS speed
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(int(self.user_settings.get("tts_speed") * 100))
        self.speed_slider.setEnabled(self.tts_checkbox.isChecked())
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(25)

        # Speed value label
        self.speed_value = QLabel(f"{self.speed_slider.value() / 100:.1f}x")
        self.speed_slider.valueChanged.connect(
            lambda value: self.speed_value.setText(f"{value / 100:.1f}x")
        )

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_value)

        voice_layout.addRow("Voice Speed:", speed_layout)

        # Voice selection (if OpenAI TTS is available)
        self.voice_combo = QComboBox()
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        self.voice_combo.addItems(voices)
        current_voice = self.user_settings.get("tts_voice", "alloy")
        index = voices.index(current_voice) if current_voice in voices else 0
        self.voice_combo.setCurrentIndex(index)
        self.voice_combo.setEnabled(self.tts_checkbox.isChecked())

        voice_layout.addRow("Voice Style:", self.voice_combo)

        # Test voice button
        self.test_voice_button = QPushButton("Test Voice")
        self.test_voice_button.clicked.connect(self._test_voice)
        self.test_voice_button.setEnabled(self.tts_checkbox.isChecked())
        voice_layout.addRow("", self.test_voice_button)

        layout.addWidget(voice_group)

        # AI Conversation Settings
        conversation_group = QGroupBox("Conversation Settings")
        conversation_layout = QFormLayout(conversation_group)

        # Enable conversation history
        self.conversation_history_checkbox = QCheckBox()
        self.conversation_history_checkbox.setChecked(
            self.user_settings.get("enable_conversation_history", True)
        )
        conversation_layout.addRow("Save Conversation History:", self.conversation_history_checkbox)

        # Conversation style
        self.conversation_style_combo = QComboBox()
        styles = ["Dynamic", "Professional", "Playful", "Educational"]
        self.conversation_style_combo.addItems(styles)
        current_style = self.user_settings.get("conversation_style", "Dynamic")
        index = styles.index(current_style) if current_style in styles else 0
        self.conversation_style_combo.setCurrentIndex(index)

        conversation_layout.addRow("Conversation Style:", self.conversation_style_combo)

        layout.addWidget(conversation_group)

        # Add stretch to push everything to the top
        layout.addStretch()

        return tab

    def _create_performance_tab(self):
        """Create the performance settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Auto-save settings
        autosave_group = QGroupBox("Auto-Save Settings")
        autosave_layout = QFormLayout(autosave_group)

        # Auto-save drawings
        self.autosave_checkbox = QCheckBox()
        self.autosave_checkbox.setChecked(self.user_settings.get("save_drawings_automatically"))
        self.autosave_checkbox.toggled.connect(self._on_autosave_toggled)
        autosave_layout.addRow("Auto-save Drawings:", self.autosave_checkbox)

        # Auto-save interval
        self.autosave_interval_spinner = QSpinBox()
        self.autosave_interval_spinner.setRange(5, 60)
        self.autosave_interval_spinner.setValue(self.user_settings.get("autosave_interval", 30))
        self.autosave_interval_spinner.setSuffix(" seconds")
        self.autosave_interval_spinner.setEnabled(self.autosave_checkbox.isChecked())
        autosave_layout.addRow("Auto-save Interval:", self.autosave_interval_spinner)

        layout.addWidget(autosave_group)

        # Performance settings
        performance_group = QGroupBox("Performance Settings")
        performance_layout = QFormLayout(performance_group)

        # Drawing quality
        self.drawing_quality_combo = QComboBox()
        qualities = ["Low", "Medium", "High"]
        self.drawing_quality_combo.addItems(qualities)
        current_quality = self.user_settings.get("drawing_quality", "Medium")
        index = qualities.index(current_quality) if current_quality in qualities else 1
        self.drawing_quality_combo.setCurrentIndex(index)

        performance_layout.addRow("Drawing Quality:", self.drawing_quality_combo)

        # Recognition delay
        self.recognition_delay_spinner = QSpinBox()
        self.recognition_delay_spinner.setRange(0, 5000)
        self.recognition_delay_spinner.setValue(self.user_settings.get("recognition_delay", 500))
        self.recognition_delay_spinner.setSuffix(" ms")
        self.recognition_delay_spinner.setSingleStep(100)

        performance_layout.addRow("Recognition Delay:", self.recognition_delay_spinner)

        # Debug mode
        self.debug_mode_checkbox = QCheckBox()
        self.debug_mode_checkbox.setChecked(self.user_settings.get("debug_mode", False))
        performance_layout.addRow("Debug Mode:", self.debug_mode_checkbox)

        layout.addWidget(performance_group)

        # Canvas settings
        canvas_group = QGroupBox("Canvas Settings")
        canvas_layout = QFormLayout(canvas_group)

        # Show grid
        self.show_grid_checkbox = QCheckBox()
        self.show_grid_checkbox.setChecked(self.user_settings.get("show_grid", False))
        canvas_layout.addRow("Show Grid:", self.show_grid_checkbox)

        # Grid size
        self.grid_size_spinner = QSpinBox()
        self.grid_size_spinner.setRange(5, 50)
        self.grid_size_spinner.setValue(self.user_settings.get("grid_size", 20))
        self.grid_size_spinner.setSuffix(" px")
        canvas_layout.addRow("Grid Size:", self.grid_size_spinner)

        # Antialiasing
        self.antialiasing_checkbox = QCheckBox()
        self.antialiasing_checkbox.setChecked(self.user_settings.get("use_antialiasing", True))
        canvas_layout.addRow("Use Antialiasing:", self.antialiasing_checkbox)

        # Smooth drawing
        self.smooth_drawing_checkbox = QCheckBox()
        self.smooth_drawing_checkbox.setChecked(self.user_settings.get("smooth_drawing", True))
        canvas_layout.addRow("Smooth Drawing:", self.smooth_drawing_checkbox)

        layout.addWidget(canvas_group)

        # Add stretch to push everything to the top
        layout.addStretch()

        return tab

    def _create_backup_tab(self):
        """Create the backup and data tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Backup group
        backup_group = QGroupBox("Backup Options")
        backup_layout = QVBoxLayout(backup_group)

        # Backup button
        backup_button = QPushButton("Backup Database")
        backup_button.clicked.connect(self._backup_database)
        backup_layout.addWidget(backup_button)

        # Scheduled backups
        scheduled_layout = QHBoxLayout()

        self.scheduled_backup_checkbox = QCheckBox("Enable Scheduled Backups:")
        self.scheduled_backup_checkbox.setChecked(
            self.user_settings.get("enable_scheduled_backups", False)
        )
        scheduled_layout.addWidget(self.scheduled_backup_checkbox)

        self.backup_interval_combo = QComboBox()
        intervals = ["Daily", "Weekly", "Monthly"]
        self.backup_interval_combo.addItems(intervals)
        current_interval = self.user_settings.get("backup_interval", "Weekly")
        index = intervals.index(current_interval) if current_interval in intervals else 1
        self.backup_interval_combo.setCurrentIndex(index)
        self.backup_interval_combo.setEnabled(self.scheduled_backup_checkbox.isChecked())

        self.scheduled_backup_checkbox.toggled.connect(
            lambda checked: self.backup_interval_combo.setEnabled(checked)
        )

        scheduled_layout.addWidget(self.backup_interval_combo)
        backup_layout.addLayout(scheduled_layout)

        layout.addWidget(backup_group)

        # Model training group
        training_group = QGroupBox("Model Training")
        training_layout = QVBoxLayout(training_group)

        # Manual retrain button
        retrain_button = QPushButton("Retrain Recognition Model")
        retrain_button.clicked.connect(self._retrain_model)
        training_layout.addWidget(retrain_button)

        # Training data stats
        if self.db:
            try:
                labeled_shapes = self.db.get_labeled_shapes(limit=1)
                total_count = len(self.db.get_labeled_shapes())

                stats_label = QLabel(f"Training database contains {total_count} labeled shapes.")
                training_layout.addWidget(stats_label)
            except Exception as e:
                logger.error(f"Error getting training data stats: {e}")
                training_layout.addWidget(QLabel("Error loading training data statistics."))

        layout.addWidget(training_group)

        # Data cleanup group
        cleanup_group = QGroupBox("Data Cleanup")
        cleanup_layout = QVBoxLayout(cleanup_group)

        # Clear history button
        clear_history_button = QPushButton("Clear Drawing History")
        clear_history_button.clicked.connect(self._clear_drawing_history)
        cleanup_layout.addWidget(clear_history_button)

        # Export training data
        export_button = QPushButton("Export Training Data")
        export_button.clicked.connect(self._export_training_data)
        cleanup_layout.addWidget(export_button)

        # Import training data
        import_button = QPushButton("Import Training Data")
        import_button.clicked.connect(self._import_training_data)
        cleanup_layout.addWidget(import_button)

        layout.addWidget(cleanup_group)

        # Add stretch to push everything to the top
        layout.addStretch()

        return tab

    def _setup_validators(self):
        """Set up validators for numeric fields"""
        # All numeric fields already have ranges set, so no additional validation needed
        pass

    def _on_color_button_clicked(self):
        """Handle color button click for pen color"""
        current_color = QColor(self.color_button.color)
        color = QColorDialog.getColor(current_color, self, "Select Default Pen Color")

        if color.isValid():
            color_name = color.name()
            self.color_button.set_color(color_name)
            self.color_label.setText(color_name)

    def _on_bg_color_button_clicked(self):
        """Handle color button click for background color"""
        current_color = QColor(self.bg_color_button.color)
        color = QColorDialog.getColor(current_color, self, "Select Canvas Background Color")

        if color.isValid():
            color_name = color.name()
            self.bg_color_button.set_color(color_name)
            self.bg_color_label.setText(color_name)

    def _on_tts_toggled(self, checked):
        """Handle TTS checkbox toggle"""
        self.volume_slider.setEnabled(checked)
        self.speed_slider.setEnabled(checked)
        self.voice_combo.setEnabled(checked)
        self.test_voice_button.setEnabled(checked)

    def _on_autosave_toggled(self, checked):
        """Handle autosave checkbox toggle"""
        self.autosave_interval_spinner.setEnabled(checked)

    def _add_custom_shape(self):
        """Add a custom shape to the shapes list"""
        shape_name = self.custom_shape_input.text().strip().lower()

        if not shape_name:
            self._set_status("Please enter a shape name", "red")
            return

        # Check if shape already exists
        for i in range(self.shapes_list.count()):
            item = self.shapes_list.item(i)
            if item.data(Qt.UserRole) == shape_name:
                self._set_status(f"Shape '{shape_name}' already exists", "red")
                return

        # Add new shape
        item = QListWidgetItem(shape_name.capitalize())
        item.setData(Qt.UserRole, shape_name)
        item.setCheckState(Qt.Checked)
        self.shapes_list.addItem(item)

        # Clear input
        self.custom_shape_input.clear()

        self._set_status(f"Added shape: {shape_name.capitalize()}", "green")

    def _test_voice(self):
        """Test the text-to-speech voice"""
        # This would typically use the TTS engine to speak a test phrase
        # For demonstration purposes, we'll just show a message
        self._set_status("Voice test: Hello, this is a test message!", "green")

        # In a real implementation, we would use the TextToSpeech class
        # with the current settings to speak the message

    def _backup_database(self):
        """Backup the database"""
        try:
            if not self.db:
                self._set_status("Database not available", "red")
                return

            # Choose backup location
            backup_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Database Backup",
                "backups/drawing_ai_app_backup.db",
                "Database Files (*.db);;All Files (*)"
            )

            if not backup_path:
                return

            # Emit signal for backup
            self.database_backup_requested.emit(backup_path)

            self._set_status(f"Database backed up to {backup_path}", "green")
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            self._set_status("Error backing up database", "red")

    def _retrain_model(self):
        """Retrain the recognition model"""
        # Ask for confirmation
        result = QMessageBox.question(
            self,
            "Retrain Model",
            "This will retrain the shape recognition model using all available labeled data. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if result == QMessageBox.Yes:
            # Emit signal for retraining
            self.model_retrain_requested.emit()
            self._set_status("Model retraining initiated", "green")

    def _clear_drawing_history(self):
        """Clear the drawing history"""
        # Ask for confirmation
        result = QMessageBox.question(
            self,
            "Clear History",
            "This will permanently delete all drawing history. This action cannot be undone. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if result == QMessageBox.Yes:
            # In a real implementation, we would clear the history from the database
            self._set_status("Drawing history cleared", "green")

    def _export_training_data(self):
        """Export training data to a directory"""
        export_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            "."
        )

        if not export_dir:
            return

        # In a real implementation, we would export the data
        self._set_status("Training data exported", "green")

    def _import_training_data(self):
        """Import training data from a directory"""
        import_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Import Directory",
            "."
        )

        if not import_dir:
            return

        # In a real implementation, we would import the data
        self._set_status("Training data imported", "green")

    def _on_save_clicked(self):
        """Handle save button click"""
        try:
            # Get the current settings
            new_settings = {
                # Appearance settings
                "dark_mode": self.dark_mode_checkbox.isChecked(),
                "default_pen_color": self.color_button.color,
                "default_pen_width": self.pen_width_spinner.value(),
                "canvas_background_color": self.bg_color_button.color,
                "show_toolbar": self.show_toolbar_checkbox.isChecked(),
                "show_statusbar": self.show_statusbar_checkbox.isChecked(),
                "show_tutorial": self.tutorial_checkbox.isChecked(),
                "show_confidence": self.confidence_checkbox.isChecked(),
                "color_code_confidence": self.color_code_confidence_checkbox.isChecked(),

                # Recognition settings
                "confidence_threshold": self.confidence_threshold_slider.value(),
                "enable_ml": self.enable_ml_checkbox.isChecked(),
                "enable_incremental_training": self.incremental_training_checkbox.isChecked(),
                "recognized_shapes": self._get_selected_shapes(),

                # Voice settings
                "use_tts": self.tts_checkbox.isChecked(),
                "tts_volume": self.volume_slider.value() / 100.0,
                "tts_speed": self.speed_slider.value() / 100.0,
                "tts_voice": self.voice_combo.currentText().lower(),
                "enable_conversation_history": self.conversation_history_checkbox.isChecked(),
                "conversation_style": self.conversation_style_combo.currentText(),

                # Performance settings
                "save_drawings_automatically": self.autosave_checkbox.isChecked(),
                "autosave_interval": self.autosave_interval_spinner.value(),
                "drawing_quality": self.drawing_quality_combo.currentText(),
                "recognition_delay": self.recognition_delay_spinner.value(),
                "debug_mode": self.debug_mode_checkbox.isChecked(),

                # Canvas settings
                "show_grid": self.show_grid_checkbox.isChecked(),
                "grid_size": self.grid_size_spinner.value(),
                "use_antialiasing": self.antialiasing_checkbox.isChecked(),
                "smooth_drawing": self.smooth_drawing_checkbox.isChecked(),

                # Backup settings
                "enable_scheduled_backups": self.scheduled_backup_checkbox.isChecked(),
                "backup_interval": self.backup_interval_combo.currentText()
            }

            # Update the user settings
            for key, value in new_settings.items():
                self.user_settings.set(key, value)

            # Emit the signal if settings have changed
            if new_settings != self.original_settings:
                self.settings_changed.emit(new_settings)

            # Show success message
            self._set_status("Settings saved successfully", "green")

            # Close the dialog after a short delay
            QTimer.singleShot(500, self.accept)

        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            self._set_status(f"Error saving settings: {e}", "red")

    def _on_reset_clicked(self):
        """Handle reset button click"""
        # Ask for confirmation
        result = QMessageBox.question(
            self,
            "Reset Settings",
            "This will reset all settings to their default values. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if result == QMessageBox.No:
            return

        try:
            # Reset to default settings
            self.dark_mode_checkbox.setChecked(UserSettings.DEFAULT_SETTINGS["dark_mode"])
            self.tts_checkbox.setChecked(UserSettings.DEFAULT_SETTINGS["use_tts"])
            self.volume_slider.setValue(int(UserSettings.DEFAULT_SETTINGS["tts_volume"] * 100))
            self.speed_slider.setValue(int(UserSettings.DEFAULT_SETTINGS["tts_speed"] * 100))
            self.color_button.set_color(UserSettings.DEFAULT_SETTINGS["default_pen_color"])
            self.color_label.setText(UserSettings.DEFAULT_SETTINGS["default_pen_color"])
            self.pen_width_spinner.setValue(UserSettings.DEFAULT_SETTINGS["default_pen_width"])
            self.tutorial_checkbox.setChecked(UserSettings.DEFAULT_SETTINGS["show_tutorial"])
            self.confidence_checkbox.setChecked(UserSettings.DEFAULT_SETTINGS["show_confidence"])
            self.autosave_checkbox.setChecked(UserSettings.DEFAULT_SETTINGS["save_drawings_automatically"])

            # Reset recognition settings
            self.confidence_threshold_slider.setValue(config.DEFAULT_CONFIDENCE_THRESHOLD)
            self.enable_ml_checkbox.setChecked(True)
            self.incremental_training_checkbox.setChecked(True)

            # Reset canvas settings
            self.show_grid_checkbox.setChecked(False)
            self.grid_size_spinner.setValue(20)
            self.antialiasing_checkbox.setChecked(True)
            self.smooth_drawing_checkbox.setChecked(True)

            # Reset all shapes to checked
            for i in range(self.shapes_list.count()):
                self.shapes_list.item(i).setCheckState(Qt.Checked)

            self._set_status("Settings reset to defaults", "green")
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
            self._set_status(f"Error resetting settings: {e}", "red")

    def _get_selected_shapes(self):
        """Get the list of selected shapes"""
        selected_shapes = []

        for i in range(self.shapes_list.count()):
            item = self.shapes_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_shapes.append(item.data(Qt.UserRole))

        return selected_shapes

    def _set_status(self, message, color="green"):
        """Set the status message"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color};")

        # Clear after 3 seconds
        self.status_timer.start(3000)

    def _clear_status(self):
        """Clear the status message"""
        self.status_label.setText("")