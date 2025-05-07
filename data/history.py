"""
drawing history management for the drawing ai app
"""
import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QPushButton,
    QLabel, QHBoxLayout, QGridLayout, QScrollArea, QWidget
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


from data.database import Database


class DrawingHistory:
    """manager for drawing history"""

    def __init__(self, db=None):
        """
        initialize the drawing history manager

        args:
            db: database instance
        """
        self.db = db or Database()

    def save_drawing(self, image, results=None):
        """
        save a drawing to the history

        args:
            image: qpixmap with the drawing
            results: recognition results from ai

        returns:
            str: path to the saved file
        """
        # generate a filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drawing_{timestamp}.png"

        # make sure the directory exists
        os.makedirs("saved_drawings", exist_ok=True)

        # full path to the file
        filepath = os.path.join("saved_drawings", filename)

        # save the image
        image.save(filepath)

        # save to database if we have results
        if results and results.get("success") and results.get("guesses"):
            top_guess = results["guesses"][0]["name"]
            confidence = results["guesses"][0]["confidence"]
            self.db.save_drawing(filepath, top_guess, confidence, results["guesses"])
        else:
            self.db.save_drawing(filepath)

        return filepath

    def get_history(self, limit=10, offset=0):
        """
        get drawing history

        args:
            limit: max number of records to return
            offset: offset for pagination

        returns:
            list: drawing history records
        """
        return self.db.get_drawing_history(limit, offset)

    def delete_drawing(self, drawing_id):
        """
        delete a drawing from history

        args:
            drawing_id: id of the drawing to delete

        returns:
            bool: true if successful
        """
        return self.db.delete_drawing(drawing_id)


class HistoryDialog(QDialog):
    """dialog for displaying drawing history"""

    # signals
    drawing_selected = pyqtSignal(QPixmap, dict)  # emitted when a drawing is selected

    def __init__(self, drawing_history, parent=None):
        """
        initialize the history dialog

        args:
            drawing_history: drawing history manager
            parent: parent widget
        """
        super().__init__(parent)

        self.drawing_history = drawing_history
        self.history_items = []

        # set up the dialog
        self.setWindowTitle("Drawing History")
        self.setMinimumSize(800, 600)

        # create the ui
        self._init_ui()

        # load the history
        self._load_history()

    def _init_ui(self):
        """initialize the user interface"""
        # main layout
        layout = QVBoxLayout(self)

        # title
        title_label = QLabel("Your Drawing History")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # scroll area for drawings
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # container for the grid
        container = QWidget()
        scroll_area.setWidget(container)

        # grid layout for the drawings
        self.grid_layout = QGridLayout(container)
        self.grid_layout.setSpacing(20)

        # buttons
        buttons_layout = QHBoxLayout()

        # load more button
        self.load_more_btn = QPushButton("Load More")
        self.load_more_btn.clicked.connect(self._load_more)
        buttons_layout.addWidget(self.load_more_btn)

        # add spacer
        buttons_layout.addStretch()

        # close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(close_btn)

        layout.addLayout(buttons_layout)

    def _load_history(self, offset=0):
        """
        load drawing history

        args:
            offset: offset for pagination
        """
        # get history from the database
        history = self.drawing_history.get_history(limit=12, offset=offset)

        if not history:
            # no more items, disable load more button
            self.load_more_btn.setEnabled(False)

            # show message if no items at all
            if offset == 0:
                label = QLabel("No drawings in history yet.")
                label.setAlignment(Qt.AlignCenter)
                self.grid_layout.addWidget(label, 0, 0)

            return

        # add items to the grid
        row, col = divmod(offset, 3)

        for item in history:
            # create thumbnail
            pixmap = QPixmap(item["image_path"])
            if not pixmap.isNull():
                # scale down to thumbnail size
                pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # create item widget
                item_widget = self._create_item_widget(pixmap, item)

                # add to grid
                self.grid_layout.addWidget(item_widget, row, col)

                # store in our list
                self.history_items.append({
                    "widget": item_widget,
                    "data": item,
                    "pixmap": pixmap
                })

                # update grid position
                col += 1
                if col >= 3:
                    col = 0
                    row += 1

    def _create_item_widget(self, pixmap, item_data):
        """
        create a widget for a history item

        args:
            pixmap: thumbnail pixmap
            item_data: data for the item

        returns:
            widget: the created widget
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # thumbnail
        thumbnail = QLabel()
        thumbnail.setPixmap(pixmap)
        thumbnail.setAlignment(Qt.AlignCenter)
        thumbnail.setFixedSize(200, 200)
        thumbnail.setStyleSheet("border: 1px solid #cccccc;")
        layout.addWidget(thumbnail)

        # info
        info_layout = QVBoxLayout()

        # date
        date_label = QLabel(item_data["timestamp"])
        date_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(date_label)

        # guess if available
        if item_data["top_guess"]:
            guess_label = QLabel(f"AI Guess: {item_data['top_guess'].capitalize()}")
            guess_label.setAlignment(Qt.AlignCenter)
            info_layout.addWidget(guess_label)

            # confidence if available
            if item_data["confidence"]:
                confidence = int(item_data["confidence"])
                conf_label = QLabel(f"Confidence: {confidence}%")
                conf_label.setAlignment(Qt.AlignCenter)
                info_layout.addWidget(conf_label)

        layout.addLayout(info_layout)

        # make clickable
        widget.setStyleSheet("QWidget:hover { background-color: #f0f0f0; }")
        widget.mousePressEvent = lambda e, d=item_data, p=pixmap: self._on_item_clicked(d, p)

        return widget

    def _on_item_clicked(self, item_data, pixmap):
        """
        handle item click

        args:
            item_data: data for the clicked item
            pixmap: thumbnail pixmap
        """
        # load the full-size image
        full_pixmap = QPixmap(item_data["image_path"])

        # emit the signal
        self.drawing_selected.emit(full_pixmap, item_data)

        # close the dialog
        self.accept()

    def _load_more(self):
        """load more history items"""
        self._load_history(offset=len(self.history_items))