"""
Enhanced drawing canvas for the Drawing AI App with improved error handling,
responsiveness, and additional features like undo/redo, zoom, and grid.
"""
import os
import time
import logging
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QPixmap, QPainterPath, QBrush,
    QCursor, QTransform, QImage, QPainterPathStroker
)
from PyQt5.QtCore import (
    Qt, QPoint, QPointF, QRect, QRectF, QSize, QTimer,
    pyqtSignal, pyqtSlot, QObject, QByteArray, QBuffer
)

import config
from error_handler import handle_errors, handle_ui_errors

# Set up logging
logger = logging.getLogger(__name__)


class DrawingAction:
    """Class representing a drawing action for undo/redo"""

    def __init__(self, action_type, data=None):
        """
        Initialize the drawing action

        Args:
            action_type: Type of action (e.g., 'draw', 'clear')
            data: Action-specific data
        """
        self.action_type = action_type
        self.data = data
        self.timestamp = time.time()


class StrokeData:
    """Class representing stroke data for drawing actions"""

    def __init__(self, points=None, pen=None, eraser=False):
        """
        Initialize the stroke data

        Args:
            points: List of points in the stroke
            pen: QPen object for the stroke
            eraser: Whether this is an eraser stroke
        """
        self.points = points or []
        self.pen = pen
        self.eraser = eraser

    def add_point(self, point):
        """
        Add a point to the stroke

        Args:
            point: Point to add
        """
        self.points.append(point)


class DrawingCanvas(QWidget):
    """Enhanced widget for drawing with mouse with undo/redo and zoom support"""

    # Signals
    canvas_updated = pyqtSignal()  # Emitted when the canvas content changes
    stroke_started = pyqtSignal()  # Emitted when a stroke starts
    stroke_finished = pyqtSignal()  # Emitted when a stroke ends
    undo_available = pyqtSignal(bool)  # Emitted when undo availability changes
    redo_available = pyqtSignal(bool)  # Emitted when redo availability changes
    zoom_changed = pyqtSignal(float)  # Emitted when zoom level changes
    error_occurred = pyqtSignal(str)  # Emitted when an error occurs

    def __init__(self, parent=None, error_handler=None):
        """
        Initialize the drawing canvas

        Args:
            parent: Parent widget
            error_handler: Error handler object
        """
        super().__init__(parent)

        # Store error handler
        self.error_handler = error_handler

        # Canvas dimensions
        self.canvas_width = config.CANVAS_WIDTH
        self.canvas_height = config.CANVAS_HEIGHT

        # Set fixed size for the canvas
        self.setMinimumSize(self.canvas_width, self.canvas_height)

        # Create the canvas drawing surface
        self.image = QPixmap(self.canvas_width, self.canvas_height)
        self.image.fill(QColor(config.DEFAULT_BACKGROUND_COLOR))

        # Create a buffer for temporary drawing
        self.buffer = QPixmap(self.canvas_width, self.canvas_height)

        # Initialize drawing variables
        self.drawing = False
        self.eraser_mode = False
        self.last_point = QPoint()
        self.current_stroke = None

        # Pen settings
        self.pen_color = QColor(config.DEFAULT_PEN_COLOR)
        # Ensure pen_width is an integer
        if isinstance(config.DEFAULT_PEN_WIDTH, str):
            try:
                self.pen_width = int(config.DEFAULT_PEN_WIDTH)
            except ValueError:
                self.pen_width = 3  # Default to 3 if conversion fails
        else:
            self.pen_width = config.DEFAULT_PEN_WIDTH

        # Background color
        self.background_color = QColor(config.DEFAULT_BACKGROUND_COLOR)

        # Undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 50

        # Pressure sensitivity
        self.use_pressure = False
        self.min_pressure_width = 0.5
        self.max_pressure_multiplier = 2.0

        # Zoom and pan
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.panning = False
        self.pan_start_pos = QPointF()
        self.min_zoom = 0.25
        self.max_zoom = 5.0

        # Grid settings
        self.show_grid = False
        self.grid_size = 20
        self.grid_color = QColor(200, 200, 200, 100)

        # Cursor settings
        self.custom_cursor = False
        self.cursor_size = self.pen_width + 4  # Slightly larger than pen

        # Performance settings
        self.smooth_drawing = True
        self.optimize_redraw = True
        self.use_antialiasing = True

        # Auto-save settings
        self.auto_save_enabled = False
        self.auto_save_interval = 30  # seconds
        self.last_auto_save = time.time()
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self._auto_save)

        # Dirty flag (changes since last save)
        self.is_dirty = False

        # Set focus policy to enable keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Set mouse tracking
        self.setMouseTracking(True)

        # Update cursor
        self._update_cursor()

        # Connect to itself for error handling
        self.error_occurred.connect(self._on_error)

    @handle_ui_errors(show_dialog=True)
    def paintEvent(self, event):
        """Handle paint events to display the canvas"""
        painter = QPainter(self)

        # Apply zoom and pan
        painter.translate(self.pan_offset)
        painter.scale(self.zoom_factor, self.zoom_factor)

        # Enable antialiasing if requested
        if self.use_antialiasing:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Draw background (only needed if transparent)
        if self.background_color.alpha() < 255:
            painter.fillRect(0, 0, self.canvas_width, self.canvas_height,
                             QColor(255, 255, 255))

        # Draw the grid if enabled
        if self.show_grid and self.zoom_factor > 0.5:
            self._draw_grid(painter)

        # Draw the main image
        painter.drawPixmap(0, 0, self.image)

        # Draw the temporary buffer if drawing
        if self.drawing:
            painter.drawPixmap(0, 0, self.buffer)

    @handle_ui_errors(show_dialog=False)
    def _draw_grid(self, painter):
        """
        Draw a grid on the canvas

        Args:
            painter: QPainter object
        """
        # Save painter state
        painter.save()

        # Set up pen for grid
        grid_pen = QPen(self.grid_color)
        grid_pen.setWidth(0)  # Cosmetic pen
        painter.setPen(grid_pen)

        # Draw horizontal lines
        for y in range(0, self.canvas_height + 1, self.grid_size):
            painter.drawLine(0, y, self.canvas_width, y)

        # Draw vertical lines
        for x in range(0, self.canvas_width + 1, self.grid_size):
            painter.drawLine(x, 0, x, self.canvas_height)

        # Restore painter state
        painter.restore()

    @handle_ui_errors(show_dialog=False)
    def mousePressEvent(self, event):
        """Handle mouse press events to start drawing or panning"""
        # Map the mouse position to canvas coordinates
        canvas_pos = self._map_to_canvas(event.pos())

        # Check if the position is within the canvas
        if not self._is_within_canvas(canvas_pos):
            return

        # Middle button or Alt+Left button for panning
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier):
            self.panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        # Left button for drawing
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = canvas_pos

            # Create a new stroke
            self.current_stroke = StrokeData(
                points=[canvas_pos],
                pen=self._create_pen(event),
                eraser=self.eraser_mode
            )

            # Clear the buffer
            self.buffer.fill(Qt.transparent)

            # Emit stroke started signal
            self.stroke_started.emit()

    @handle_ui_errors(show_dialog=False)
    def mouseMoveEvent(self, event):
        """Handle mouse move events to continue drawing or panning"""
        # Handle panning
        if self.panning:
            delta = event.pos() - self.pan_start_pos
            self.pan_offset += delta
            self.pan_start_pos = event.pos()
            self.update()
            return

        # Update cursor for drawing (when not panning)
        if not self.drawing and self.custom_cursor:
            self.update()  # Update to redraw the cursor

        # Handle drawing
        if self.drawing:
            # Map the mouse position to canvas coordinates
            canvas_pos = self._map_to_canvas(event.pos())

            # Check if the position is within the canvas
            if not self._is_within_canvas(canvas_pos):
                return

            # Draw on the buffer
            painter = QPainter(self.buffer)

            # Enable antialiasing
            if self.use_antialiasing:
                painter.setRenderHint(QPainter.Antialiasing, True)

            # Set up the pen
            pen = self._create_pen(event)
            painter.setPen(pen)

            # Add point to current stroke
            self.current_stroke.points.append(canvas_pos)

            # Draw line from last position to current position
            if self.smooth_drawing and len(self.current_stroke.points) > 2:
                # Use a smooth path for better quality
                path = QPainterPath()
                path.moveTo(self.current_stroke.points[-3])

                # Add a quadratic curve
                mid_point = (self.current_stroke.points[-2] + self.current_stroke.points[-1]) / 2
                path.quadTo(self.current_stroke.points[-2], mid_point)

                painter.drawPath(path)
            else:
                # Simple line for performance or when we don't have enough points
                painter.drawLine(self.last_point, canvas_pos)

            # Update the last point
            self.last_point = canvas_pos

            # Update the widget (only the affected area if optimization is enabled)
            if self.optimize_redraw:
                # Convert float values to integers for QRect
                min_x = int(min(self.last_point.x(), canvas_pos.x()) - self.pen_width * 2)
                min_y = int(min(self.last_point.y(), canvas_pos.y()) - self.pen_width * 2)
                width = int(abs(self.last_point.x() - canvas_pos.x()) + self.pen_width * 4)
                height = int(abs(self.last_point.y() - canvas_pos.y()) + self.pen_width * 4)

                rect = QRect(min_x, min_y, width, height)

                # Apply zoom and pan to the update rect
                mapped_rect = self._map_to_widget(rect)
                self.update(mapped_rect)
            else:
                self.update()

    @handle_ui_errors(show_dialog=False)
    def mouseReleaseEvent(self, event):
        """Handle mouse release events to stop drawing or panning"""
        # Handle panning end
        if self.panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self.panning = False
            self._update_cursor()  # Restore the appropriate cursor
            return

        # Handle drawing end
        if self.drawing and event.button() == Qt.LeftButton:
            # Finalize the drawing by applying the buffer to the main image
            painter = QPainter(self.image)

            # Enable antialiasing
            if self.use_antialiasing:
                painter.setRenderHint(QPainter.Antialiasing, True)

            painter.drawPixmap(0, 0, self.buffer)

            # Add the action to the undo stack
            self._add_to_undo_stack('draw', self.current_stroke)

            # Reset drawing state
            self.drawing = False
            self.current_stroke = None

            # Clear the buffer
            self.buffer.fill(Qt.transparent)

            # Mark as dirty
            self.is_dirty = True

            # Update the widget
            self.update()

            # Emit signals
            self.stroke_finished.emit()
            self.canvas_updated.emit()

    @handle_ui_errors(show_dialog=False)
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        # Calculate zoom factor change
        delta = event.angleDelta().y()
        zoom_change = 1.0 + (delta / 1200.0)  # Adjust sensitivity

        # Calculate new zoom factor
        new_zoom = self.zoom_factor * zoom_change

        # Clamp to limits
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))

        # Only proceed if the zoom actually changed
        if new_zoom != self.zoom_factor:
            # Calculate the point under the cursor in scene coordinates
            old_pos = self._map_to_canvas(event.pos())

            # Update zoom factor
            self.zoom_factor = new_zoom

            # Calculate the new position under the cursor
            new_pos = self._map_to_canvas(event.pos())

            # Adjust the pan offset to keep the point under the cursor
            self.pan_offset += (new_pos - old_pos) * self.zoom_factor

            # Update the cursor
            self._update_cursor()

            # Update the widget
            self.update()

            # Emit signal
            self.zoom_changed.emit(self.zoom_factor)

    @handle_ui_errors(show_dialog=False)
    def keyPressEvent(self, event):
        """Handle key press events"""
        # Undo: Ctrl+Z
        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo()

        # Redo: Ctrl+Y or Ctrl+Shift+Z
        elif (event.key() == Qt.Key_Y and event.modifiers() & Qt.ControlModifier) or \
             (event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier and
              event.modifiers() & Qt.ShiftModifier):
            self.redo()

        # Toggle grid: G
        elif event.key() == Qt.Key_G:
            self.show_grid = not self.show_grid
            self.update()

        # Reset zoom and pan: R
        elif event.key() == Qt.Key_R:
            self.reset_view()

        # Clear canvas: Delete or Backspace
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.clear()

        # Toggle eraser: E
        elif event.key() == Qt.Key_E:
            self.toggle_eraser(not self.eraser_mode)

        # Pass to parent for other keys
        else:
            super().keyPressEvent(event)

    @handle_errors(context="Creating pen")
    def _create_pen(self, event=None):
        """
        Create a pen based on current settings and optional event

        Args:
            event: Optional mouse event for pressure sensitivity

        Returns:
            QPen: Configured pen
        """
        # Set up the pen
        pen = QPen()

        # Set color based on mode
        if self.eraser_mode:
            pen.setColor(self.background_color)
        else:
            pen.setColor(self.pen_color)

        # Set width
        width = self.pen_width

        # Apply pressure sensitivity if enabled and event has it
        if self.use_pressure and event and hasattr(event, 'pressure'):
            pressure = max(event.pressure(), 0.01)  # Ensure positive pressure
            width_range = self.pen_width * self.max_pressure_multiplier - self.min_pressure_width
            width = self.min_pressure_width + width_range * pressure

        pen.setWidth(int(width))

        # Set cap and join styles for nice lines
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)

        return pen

    @handle_errors(context="Mapping coordinates")
    def _map_to_canvas(self, pos):
        """
        Map widget coordinates to canvas coordinates

        Args:
            pos: Position in widget coordinates

        Returns:
            QPointF: Position in canvas coordinates
        """
        # Apply inverse transform
        transform = QTransform()
        transform.translate(self.pan_offset.x(), self.pan_offset.y())
        transform.scale(self.zoom_factor, self.zoom_factor)

        # Get the inverse transform
        ok = True
        inverse, ok = transform.inverted()

        if not ok:
            # Fallback if inversion fails
            logger.warning("Transform inversion failed")
            return QPointF(pos) / self.zoom_factor - self.pan_offset / self.zoom_factor

        # Apply the inverse transform
        return inverse.map(QPointF(pos))

    @handle_errors(context="Mapping coordinates")
    def _map_to_widget(self, rect):
        """
        Map canvas coordinates to widget coordinates

        Args:
            rect: Rectangle in canvas coordinates

        Returns:
            QRect: Rectangle in widget coordinates
        """
        # Apply transform
        transform = QTransform()
        transform.translate(self.pan_offset.x(), self.pan_offset.y())
        transform.scale(self.zoom_factor, self.zoom_factor)

        # Map the rectangle
        return transform.mapRect(QRectF(rect)).toRect()

    @handle_errors(context="Checking canvas bounds")
    def _is_within_canvas(self, pos):
        """
        Check if a position is within the canvas bounds

        Args:
            pos: Position to check

        Returns:
            bool: True if within bounds
        """
        return (0 <= pos.x() <= self.canvas_width and
                0 <= pos.y() <= self.canvas_height)

    @handle_errors(context="Updating cursor")
    def _update_cursor(self):
        """Update the cursor based on current mode"""
        if self.panning:
            # Panning cursor
            self.setCursor(Qt.OpenHandCursor)
        elif self.custom_cursor:
            # Create a custom cursor for drawing
            cursor_size = int(self.pen_width * self.zoom_factor) + 4
            cursor_size = max(8, min(cursor_size, 64))  # Constrain size

            cursor_pixmap = QPixmap(cursor_size, cursor_size)
            cursor_pixmap.fill(Qt.transparent)

            # Draw cursor
            painter = QPainter(cursor_pixmap)
            painter.setPen(QPen(Qt.black, 1))

            if self.eraser_mode:
                # Eraser cursor
                painter.setBrush(QBrush(Qt.white))
                painter.drawRect(0, 0, cursor_size - 1, cursor_size - 1)
                painter.drawLine(0, 0, cursor_size - 1, cursor_size - 1)
                painter.drawLine(0, cursor_size - 1, cursor_size - 1, 0)
            else:
                # Pen cursor
                painter.setBrush(QBrush(self.pen_color))
                painter.drawEllipse(0, 0, cursor_size - 1, cursor_size - 1)

            painter.end()

            # Create and set the cursor
            cursor = QCursor(cursor_pixmap)
            self.setCursor(cursor)
        else:
            # Default cursor
            self.setCursor(Qt.CrossCursor)

    @pyqtSlot(str)
    def _on_error(self, error_message):
        """
        Handle errors emitted by this widget

        Args:
            error_message: Error message
        """
        logger.error(f"Canvas error: {error_message}")

        # If we have an error handler, let it handle it
        if self.error_handler:
            self.error_handler.handle_error(Exception(error_message), "Canvas")

    @handle_errors(context="Adding to undo stack")
    def _add_to_undo_stack(self, action_type, data=None):
        """
        Add an action to the undo stack

        Args:
            action_type: Type of action
            data: Action data
        """
        # Create the action
        action = DrawingAction(action_type, data)

        # Add to undo stack
        self.undo_stack.append(action)

        # Limit the undo stack size
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)

        # Clear the redo stack
        self.redo_stack.clear()

        # Emit signals
        self.undo_available.emit(True)
        self.redo_available.emit(False)

    @handle_ui_errors(show_dialog=False)
    def _auto_save(self):
        """Auto-save the canvas if enabled"""
        if not self.auto_save_enabled or not self.is_dirty:
            return

        # Check if enough time has passed
        now = time.time()
        if now - self.last_auto_save < self.auto_save_interval:
            return

        # Save the canvas
        try:
            # Create auto-save directory if needed
            os.makedirs("auto_save", exist_ok=True)

            # Generate filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"auto_save/drawing_{timestamp}.png"

            # Save the image
            self.image.save(filename)

            # Update the timestamp
            self.last_auto_save = now

            # Reset the dirty flag
            self.is_dirty = False

            logger.info(f"Auto-saved canvas to {filename}")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")

    @handle_ui_errors()
    def set_pen_color(self, color):
        """
        Set the pen color

        Args:
            color: New pen color (string or QColor)
        """
        if isinstance(color, str):
            self.pen_color = QColor(color)
        else:
            self.pen_color = color

        # Update the cursor
        self._update_cursor()

    @handle_ui_errors()
    def set_pen_width(self, width):
        """
        Set the pen width

        Args:
            width: New pen width
        """
        # Ensure width is an integer
        if isinstance(width, str):
            try:
                width = int(width)
            except ValueError:
                width = 3  # Default to 3 if conversion fails

        self.pen_width = width
        self.cursor_size = width + 4

        # Update the cursor
        self._update_cursor()

    @handle_ui_errors()
    def set_background_color(self, color):
        """
        Set the background color

        Args:
            color: New background color (string or QColor)
        """
        if isinstance(color, str):
            self.background_color = QColor(color)
        else:
            self.background_color = color

        # Create a new image with the new background
        new_image = QPixmap(self.canvas_width, self.canvas_height)
        new_image.fill(self.background_color)

        # Draw the old image on top
        painter = QPainter(new_image)
        painter.drawPixmap(0, 0, self.image)
        painter.end()

        # Replace the old image
        self.image = new_image

        # Update the widget
        self.update()

    @handle_ui_errors()
    def toggle_eraser(self, enabled):
        """
        Toggle eraser mode

        Args:
            enabled: Whether to enable eraser mode
        """
        self.eraser_mode = enabled

        # Update the cursor
        self._update_cursor()

    @handle_ui_errors()
    def toggle_grid(self, enabled):
        """
        Toggle the grid

        Args:
            enabled: Whether to enable the grid
        """
        self.show_grid = enabled
        self.update()

    @handle_ui_errors()
    def set_grid_size(self, size):
        """
        Set the grid size

        Args:
            size: New grid size
        """
        self.grid_size = max(5, size)

        if self.show_grid:
            self.update()

    @handle_ui_errors()
    def clear(self):
        """Clear the canvas"""
        # Save current state to undo stack
        self._add_to_undo_stack('clear', self.get_image())

        # Clear the image
        self.image.fill(self.background_color)

        # Clear the buffer
        self.buffer.fill(Qt.transparent)

        # Mark as dirty
        self.is_dirty = True

        # Update the widget
        self.update()

        # Emit signal
        self.canvas_updated.emit()

    @handle_ui_errors()
    def undo(self):
        """Undo the last action"""
        if not self.undo_stack:
            return

        # Get the last action
        action = self.undo_stack.pop()

        # Add current state to redo stack
        if action.action_type == 'draw':
            # For draw actions, save the current image
            self.redo_stack.append(DrawingAction('state', self.get_image()))
        elif action.action_type == 'clear':
            # For clear actions, just push a clear action
            self.redo_stack.append(DrawingAction('clear', None))
        else:
            # For other actions, push a state action
            self.redo_stack.append(DrawingAction('state', self.get_image()))

        # Apply the undo
        if action.action_type == 'draw':
            # Redraw everything except this stroke
            self.image.fill(self.background_color)

            # Redraw all strokes from the undo stack
            painter = QPainter(self.image)

            # Enable antialiasing
            if self.use_antialiasing:
                painter.setRenderHint(QPainter.Antialiasing, True)

            for undo_action in self.undo_stack:
                if undo_action.action_type == 'draw':
                    stroke = undo_action.data

                    # Set up pen
                    painter.setPen(stroke.pen)

                    # Draw the stroke
                    if len(stroke.points) >= 2:
                        for i in range(1, len(stroke.points)):
                            painter.drawLine(stroke.points[i-1], stroke.points[i])

            painter.end()
        elif action.action_type == 'clear':
            # Restore the saved image
            self.image = action.data
        elif action.action_type == 'state':
            # Restore the saved state
            self.image = action.data

        # Update undo/redo availability
        self.undo_available.emit(bool(self.undo_stack))
        self.redo_available.emit(bool(self.redo_stack))

        # Mark as dirty
        self.is_dirty = True

        # Update the widget
        self.update()

        # Emit signal
        self.canvas_updated.emit()

    @handle_ui_errors()
    def redo(self):
        """Redo the last undone action"""
        if not self.redo_stack:
            return

        # Get the last action
        action = self.redo_stack.pop()

        # Add current state to undo stack
        self._add_to_undo_stack('state', self.get_image())

        # Apply the redo
        if action.action_type == 'state':
            # Restore the saved state
            self.image = action.data
        elif action.action_type == 'clear':
            # Clear the canvas
            self.image.fill(self.background_color)

        # Update redo availability
        self.redo_available.emit(bool(self.redo_stack))

        # Mark as dirty
        self.is_dirty = True

        # Update the widget
        self.update()

        # Emit signal
        self.canvas_updated.emit()

    @handle_ui_errors()
    def reset_view(self):
        """Reset zoom and pan to default"""
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)

        # Update the cursor
        self._update_cursor()

        # Update the widget
        self.update()

        # Emit signal
        self.zoom_changed.emit(self.zoom_factor)

    @handle_ui_errors()
    def set_zoom(self, zoom):
        """
        Set the zoom level

        Args:
            zoom: New zoom level
        """
        # Clamp zoom level
        zoom = max(self.min_zoom, min(self.max_zoom, zoom))

        if zoom != self.zoom_factor:
            # Store the center of the view
            center = QRect(0, 0, self.width(), self.height()).center()
            center_in_scene = self._map_to_canvas(center)

            # Set the new zoom
            self.zoom_factor = zoom

            # Recalculate the pan offset to keep the center
            new_center_in_scene = self._map_to_canvas(center)
            self.pan_offset += (new_center_in_scene - center_in_scene) * self.zoom_factor

            # Update the cursor
            self._update_cursor()

            # Update the widget
            self.update()

            # Emit signal
            self.zoom_changed.emit(self.zoom_factor)

    @handle_ui_errors()
    def get_image(self):
        """
        Get the current canvas image

        Returns:
            QPixmap: Canvas image
        """
        return self.image.copy()

    @handle_ui_errors()
    def set_image(self, image):
        """
        Set the canvas image

        Args:
            image: New canvas image (QPixmap or QImage)
        """
        if isinstance(image, QImage):
            self.image = QPixmap.fromImage(image)
        else:
            self.image = image.copy()

        # Clear the buffer
        self.buffer.fill(Qt.transparent)

        # Mark as dirty
        self.is_dirty = True

        # Update the widget
        self.update()

        # Emit signal
        self.canvas_updated.emit()

    @handle_ui_errors()
    def set_use_pressure(self, enabled):
        """
        Enable or disable pressure sensitivity

        Args:
            enabled: Whether to enable pressure sensitivity
        """
        self.use_pressure = enabled

    @handle_ui_errors()
    def set_custom_cursor(self, enabled):
        """
        Enable or disable custom cursor

        Args:
            enabled: Whether to enable custom cursor
        """
        self.custom_cursor = enabled
        self._update_cursor()

    @handle_ui_errors()
    def set_smooth_drawing(self, enabled):
        """
        Enable or disable smooth drawing

        Args:
            enabled: Whether to enable smooth drawing
        """
        self.smooth_drawing = enabled

    @handle_ui_errors()
    def set_use_antialiasing(self, enabled):
        """
        Enable or disable antialiasing

        Args:
            enabled: Whether to enable antialiasing
        """
        self.use_antialiasing = enabled
        self.update()

    @handle_ui_errors()
    def save_image(self, filename, format=None):
        """
        Save the canvas image to a file

        Args:
            filename: File path to save to
            format: Optional image format (e.g., 'PNG', 'JPG')

        Returns:
            bool: True if successful
        """
        result = self.image.save(filename, format)

        if result:
            # Reset the dirty flag
            self.is_dirty = False

        return result

    @handle_ui_errors()
    def to_base64(self, format='PNG'):
        """
        Convert the image to base64 data

        Args:
            format: Image format (e.g., 'PNG', 'JPG')

        Returns:
            str: Base64 encoded image data
        """
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QBuffer.WriteOnly)

        self.image.save(buffer, format)

        # Get the base64 data
        data = byte_array.toBase64().data().decode()

        return data

    @handle_ui_errors()
    def set_auto_save(self, enabled, interval=None):
        """
        Enable or disable auto-save

        Args:
            enabled: Whether to enable auto-save
            interval: Optional interval in seconds
        """
        self.auto_save_enabled = enabled

        if interval is not None:
            self.auto_save_interval = max(5, interval)

        if enabled:
            self.auto_save_timer.start(1000)  # Check every second
        else:
            self.auto_save_timer.stop()

    @handle_ui_errors()
    def is_empty(self):
        """
        Check if the canvas is empty

        Returns:
            bool: True if the canvas is empty
        """
        # Create a blank image with the background color
        blank = QPixmap(self.canvas_width, self.canvas_height)
        blank.fill(self.background_color)

        # Compare with current image
        return self._pixmaps_equal(self.image, blank)

    @handle_errors(context="Comparing pixmaps")
    def _pixmaps_equal(self, p1, p2):
        """
        Compare two pixmaps

        Args:
            p1: First pixmap
            p2: Second pixmap

        Returns:
            bool: True if the pixmaps are equal
        """
        if p1.size() != p2.size():
            return False

        # Convert to images
        img1 = p1.toImage()
        img2 = p2.toImage()

        # Compare bytes
        return img1 == img2

    @handle_ui_errors()
    def resize_canvas(self, width, height, keep_content=True, background_color=None):
        """
        Resize the canvas

        Args:
            width: New width
            height: New height
            keep_content: Whether to keep existing content
            background_color: Optional new background color
        """
        # Use current or specified background color
        bg_color = background_color or self.background_color

        # Create a new image with the new size
        new_image = QPixmap(width, height)
        new_image.fill(bg_color)

        if keep_content:
            # Draw the old image on top
            painter = QPainter(new_image)
            painter.drawPixmap(0, 0, self.image)
            painter.end()

        # Update the canvas
        self.canvas_width = width
        self.canvas_height = height
        self.image = new_image

        # Create a new buffer
        self.buffer = QPixmap(width, height)
        self.buffer.fill(Qt.transparent)

        # If background color was specified, update it
        if background_color:
            self.background_color = QColor(background_color)

        # Resize the widget
        self.setMinimumSize(width, height)

        # Reset view
        self.reset_view()

        # Mark as dirty
        self.is_dirty = True

        # Update the widget
        self.update()

        # Emit signal
        self.canvas_updated.emit()