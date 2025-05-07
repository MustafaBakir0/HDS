"""
training database dialog for the drawing ai app
"""
import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QComboBox, QTabWidget,
    QGridLayout, QScrollArea, QWidget, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QIcon

from data.database import Database


class TrainingDialog(QDialog):
    """dialog for viewing and managing the training database"""

    def __init__(self, db, parent=None):
        """initialize the training dialog"""
        super().__init__(parent)

        self.db = db
        self.current_shape = None
        self.shapes = []

        # set up the dialog
        self.setWindowTitle("Training Database")
        self.setMinimumSize(800, 600)

        # create the UI
        self._init_ui()

        # load the data
        self._load_data()

    def _init_ui(self):
        """initialize the user interface"""
        # main layout
        layout = QVBoxLayout(self)

        # header
        header_layout = QHBoxLayout()

        title_label = QLabel("Shape Training Database")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        header_layout.addWidget(title_label)

        self.shape_filter = QComboBox()
        self.shape_filter.addItem("All Shapes")
        self.shape_filter.currentIndexChanged.connect(self._on_filter_changed)
        header_layout.addWidget(self.shape_filter)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._load_data)
        header_layout.addWidget(refresh_button)

        layout.addLayout(header_layout)

        # tabs
        self.tabs = QTabWidget()

        # grid view
        self.grid_tab = QWidget()
        self.tabs.addTab(self.grid_tab, "Grid View")

        # grid layout
        grid_layout = QVBoxLayout(self.grid_tab)

        # scroll area for grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        grid_layout.addWidget(scroll_area)

        # container for grid
        self.grid_container = QWidget()
        scroll_area.setWidget(self.grid_container)

        # grid to display shapes
        self.grid = QGridLayout(self.grid_container)
        self.grid.setSpacing(10)

        # list view
        self.list_tab = QWidget()
        self.tabs.addTab(self.list_tab, "List View")

        # list layout
        list_layout = QVBoxLayout(self.list_tab)

        # list widget
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        list_layout.addWidget(self.list_widget)

        # add tabs to main layout
        layout.addWidget(self.tabs)

        # stats panel
        stats_layout = QHBoxLayout()

        self.stats_label = QLabel("Loading statistics...")
        stats_layout.addWidget(self.stats_label)

        # add delete button
        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self._delete_selected)
        stats_layout.addWidget(delete_button)

        # close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        stats_layout.addWidget(close_button)

        layout.addLayout(stats_layout)

    def _load_data(self):
        """load data from the database"""
        # get all shapes
        self.shapes = self.db.get_labeled_shapes()

        # clear existing items
        self.list_widget.clear()

        # clear the grid
        for i in reversed(range(self.grid.count())):
            widget = self.grid.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # gather shape names for filter
        shape_names = set()
        for shape in self.shapes:
            shape_names.add(shape["shape_name"])

        # update shape filter
        current_filter = self.shape_filter.currentText()
        self.shape_filter.clear()
        self.shape_filter.addItem("All Shapes")
        for name in sorted(shape_names):
            self.shape_filter.addItem(name.capitalize())

        # try to restore previous filter
        index = self.shape_filter.findText(current_filter)
        if index >= 0:
            self.shape_filter.setCurrentIndex(index)

        # filter and display the data
        self._apply_filter()

    def _apply_filter(self):
        """apply the current filter and update the display"""
        filter_text = self.shape_filter.currentText()
        filtered_shapes = self.shapes

        if filter_text != "All Shapes":
            # convert back to lowercase for comparison
            filter_name = filter_text.lower()
            filtered_shapes = [s for s in self.shapes if s["shape_name"] == filter_name]

        # update list view
        self.list_widget.clear()
        for shape in filtered_shapes:
            item = QListWidgetItem(f"{shape['shape_name'].capitalize()} ({shape['confidence']}%)")
            item.setData(Qt.UserRole, shape)
            self.list_widget.addItem(item)

        # update grid view
        for i in reversed(range(self.grid.count())):
            widget = self.grid.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # create grid items
        row, col = 0, 0
        max_cols = 4  # adjust based on your layout preferences

        for i, shape in enumerate(filtered_shapes):
            # create item widget
            item = self._create_grid_item(shape)

            # add to grid
            self.grid.addWidget(item, row, col)

            # update position
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # update stats
        shape_counts = {}
        for shape in self.shapes:
            name = shape["shape_name"]
            if name in shape_counts:
                shape_counts[name] += 1
            else:
                shape_counts[name] = 1

        # build stats text
        stats_text = f"Total shapes: {len(self.shapes)} | Filtered: {len(filtered_shapes)} | "
        stats_text += " | ".join(f"{name.capitalize()}: {count}" for name, count in sorted(shape_counts.items()))

        self.stats_label.setText(stats_text)

    def _create_grid_item(self, shape):
        """create a grid item for the shape"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # image thumbnail
        thumbnail = QLabel()
        pixmap = QPixmap(shape["image_path"])
        if not pixmap.isNull():
            # scale to thumbnail size
            thumbnail_size = 120
            pixmap = pixmap.scaled(thumbnail_size, thumbnail_size,
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
            thumbnail.setPixmap(pixmap)

        thumbnail.setAlignment(Qt.AlignCenter)
        layout.addWidget(thumbnail)

        # shape name and confidence
        name_label = QLabel(f"{shape['shape_name'].capitalize()}")
        name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_label)

        conf_label = QLabel(f"Confidence: {shape['confidence']}%")
        conf_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(conf_label)

        # make the widget selectable
        widget.mousePressEvent = lambda e, s=shape: self._on_grid_item_clicked(s)
        widget.setStyleSheet("QWidget:hover { background-color: #e0e0e0; } "
                             "QWidget { border: 1px solid #cccccc; border-radius: 5px; padding: 5px; }")

        # store shape data in the widget
        widget.setProperty("shape_data", shape)

        return widget

    def _on_filter_changed(self, index):
        """handle filter selection change"""
        self._apply_filter()

    def _on_item_clicked(self, item):
        """handle list item click"""
        shape = item.data(Qt.UserRole)
        self.current_shape = shape
        self._show_shape_details(shape)

    def _on_grid_item_clicked(self, shape):
        """handle grid item click"""
        self.current_shape = shape
        self._show_shape_details(shape)

        # highlight the item in the grid
        for i in range(self.grid.count()):
            widget = self.grid.itemAt(i).widget()
            if widget:
                widget_shape = widget.property("shape_data")
                if widget_shape and widget_shape["id"] == shape["id"]:
                    widget.setStyleSheet("QWidget { background-color: #d0e0ff; "
                                         "border: 2px solid #3f51b5; border-radius: 5px; padding: 5px; }")
                else:
                    widget.setStyleSheet("QWidget:hover { background-color: #e0e0e0; } "
                                         "QWidget { border: 1px solid #cccccc; border-radius: 5px; padding: 5px; }")

    def _show_shape_details(self, shape):
        """show details for the selected shape"""
        # This would display a detailed view of the shape
        # For now, we'll just highlight the item in the list
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            shape_data = item.data(Qt.UserRole)
            if shape_data and shape_data["id"] == shape["id"]:
                self.list_widget.setCurrentItem(item)
                break

    def _delete_selected(self):
        """delete the selected shape"""
        if not self.current_shape:
            QMessageBox.warning(self, "No Selection", "Please select a shape to delete.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete this {self.current_shape['shape_name']} shape?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Delete the shape (in a real implementation)
            # self.db.delete_labeled_shape(self.current_shape["id"])

            # Delete the image file
            try:
                os.remove(self.current_shape["image_path"])
            except Exception as e:
                print(f"Error deleting file: {e}")

            # Reload the data
            self._load_data()

            # Clear the selection
            self.current_shape = None