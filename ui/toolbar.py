"""
Toolbar with drawing tools for the Drawing AI App
"""
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QComboBox, QLabel, QFrame, QColorDialog
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor, QIcon, QFont

import config


class ColorButton(QPushButton):
    """Custom button for color selection"""

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(30, 30)
        self.setStyleSheet(f"background-color: {color}; border: 2px solid #888888;")
        self.setToolTip(f"Select {color} color")


class ToolBar(QWidget):
    """Toolbar with drawing tools"""

    # Signals
    color_selected = pyqtSignal(str)  # Emitted when a color is selected
    size_selected = pyqtSignal(int)  # Emitted when a pen size is selected
    eraser_selected = pyqtSignal(bool)  # Emitted when eraser is toggled
    clear_selected = pyqtSignal()  # Emitted when clear button is clicked

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up the layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)

        # Create color selection section
        color_section = self._create_color_section()
        layout.addWidget(color_section)

        # Add vertical separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.VLine)
        separator1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator1)

        # Create size selection section
        size_section = self._create_size_section()
        layout.addWidget(size_section)

        # Add vertical separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)

        # Create tools section
        tools_section = self._create_tools_section()
        layout.addWidget(tools_section)

        # Add stretch to align everything to the left
        layout.addStretch(1)

        # Default values
        self.current_color = config.DEFAULT_PEN_COLOR
        self.current_size = config.DEFAULT_PEN_WIDTH
        self.eraser_active = False

    def _create_color_section(self):
        """Create the color selection section"""
        section = QWidget()
        layout = QVBoxLayout(section)

        # Label
        label = QLabel("Colors")
        label.setFont(QFont("Arial", 10, QFont.Bold))
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # Color buttons in horizontal layout
        colors_layout = QHBoxLayout()

        # Add color buttons
        for color_info in config.COLOR_PALETTE:
            color_value = color_info["value"]
            btn = ColorButton(color_value)
            btn.clicked.connect(lambda checked, c=color_value: self._on_color_clicked(c))
            colors_layout.addWidget(btn)

        # Add custom color button
        custom_btn = QPushButton("...")
        custom_btn.setFixedSize(30, 30)
        custom_btn.setToolTip("Custom color")
        custom_btn.clicked.connect(self._on_custom_color_clicked)
        colors_layout.addWidget(custom_btn)

        layout.addLayout(colors_layout)
        return section

    def _create_size_section(self):
        """Create the pen size selection section"""
        section = QWidget()
        layout = QVBoxLayout(section)

        # Label
        label = QLabel("Pen Size")
        label.setFont(QFont("Arial", 10, QFont.Bold))
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # Size combo box
        size_combo = QComboBox()
        for size_info in config.PEN_SIZES:
            size_combo.addItem(size_info["name"], size_info["value"])

        # Set default size
        default_index = next(
            (i for i, size_info in enumerate(config.PEN_SIZES)
             if size_info["value"] == config.DEFAULT_PEN_WIDTH),
            0
        )
        size_combo.setCurrentIndex(default_index)

        # Connect signal
        size_combo.currentIndexChanged.connect(
            lambda index: self._on_size_changed(size_combo.itemData(index))
        )

        layout.addWidget(size_combo)
        return section

    def _create_tools_section(self):
        """Create the drawing tools section"""
        section = QWidget()
        layout = QVBoxLayout(section)

        # Label
        label = QLabel("Tools")
        label.setFont(QFont("Arial", 10, QFont.Bold))
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # Tools in horizontal layout
        tools_layout = QHBoxLayout()

        # Eraser button
        self.eraser_btn = QPushButton("Eraser")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.setFixedHeight(30)
        self.eraser_btn.setToolTip("Toggle eraser tool")
        self.eraser_btn.clicked.connect(self._on_eraser_toggled)
        tools_layout.addWidget(self.eraser_btn)

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(30)
        clear_btn.setToolTip("Clear the canvas")
        clear_btn.clicked.connect(self._on_clear_clicked)
        tools_layout.addWidget(clear_btn)

        layout.addLayout(tools_layout)
        return section

    def _on_color_clicked(self, color):
        """Handle color button click"""
        self.current_color = color

        # Disable eraser if it was active
        if self.eraser_active:
            self.eraser_active = False
            self.eraser_btn.setChecked(False)
            self.eraser_selected.emit(False)

        # Emit the signal
        self.color_selected.emit(color)

    def _on_custom_color_clicked(self):
        """Handle custom color button click"""
        initial_color = QColor(self.current_color)
        color = QColorDialog.getColor(initial_color, self, "Select Color")

        if color.isValid():
            self.current_color = color.name()

            # Disable eraser if it was active
            if self.eraser_active:
                self.eraser_active = False
                self.eraser_btn.setChecked(False)
                self.eraser_selected.emit(False)

            # Emit the signal
            self.color_selected.emit(self.current_color)

    def _on_size_changed(self, size):
        """Handle pen size change"""
        self.current_size = size
        self.size_selected.emit(size)

    def _on_eraser_toggled(self, checked):
        """Handle eraser button toggle"""
        self.eraser_active = checked
        self.eraser_selected.emit(checked)

    def _on_clear_clicked(self):
        """Handle clear button click"""
        self.clear_selected.emit()