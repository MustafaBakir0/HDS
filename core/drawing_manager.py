"""
drawing manager for the drawing ai app
"""
import os
import time
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter

from core.recognition import DrawingRecognizer


class DrawingManager(QObject):
    """manages drawing operations and interactions with the ai"""

    # signals
    recognition_started = pyqtSignal()  # emitted when recognition starts
    recognition_finished = pyqtSignal(dict)  # emitted when recognition finishes

    def __init__(self, api_key=None, parent=None):
        """
        initialize the drawing manager

        args:
            api_key: openai api key
            parent: parent qobject
        """
        super().__init__(parent)

        # initialize the recognizer
        self.recognizer = DrawingRecognizer(api_key=api_key)

        # keep track of drawing history
        self.drawing_history = []

    def recognize_drawing(self, image):
        """
        recognize the content of a drawing

        args:
            image: qpixmap with the drawing

        returns:
            dict: recognition results
        """
        # signal that recognition is starting
        self.recognition_started.emit()

        # call the recognizer
        results = self.recognizer.recognize(image)

        # add to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            "timestamp": timestamp,
            "image": image,
            "results": results
        }
        self.drawing_history.append(history_entry)

        # signal that recognition is finished
        self.recognition_finished.emit(results)

        return results

    def save_drawing(self, image, filename=None):
        """
        save the drawing to a file

        args:
            image: qpixmap with the drawing
            filename: optional filename, will generate one if not provided

        returns:
            str: path to the saved file
        """
        # generate a filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drawing_{timestamp}.png"

        # make sure the directory exists
        os.makedirs("saved_drawings", exist_ok=True)

        # full path to the file
        filepath = os.path.join("saved_drawings", filename)

        # save the image
        image.save(filepath)

        return filepath

    def get_history_entry(self, index):
        """
        get an entry from the drawing history

        args:
            index: index of the history entry

        returns:
            dict: history entry or none if index is out of range
        """
        if 0 <= index < len(self.drawing_history):
            return self.drawing_history[index]
        return None

    def get_history_length(self):
        """
        get the number of entries in the drawing history

        returns:
            int: number of history entries
        """
        return len(self.drawing_history)

    def clear_history(self):
        """clear the drawing history"""
        self.drawing_history = []