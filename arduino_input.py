"""
Arduino input handler for the Drawing AI App.
Provides serial communication with Arduino joystick and buttons.
Translates Arduino input into canvas interactions with intuitive joystick control.
"""
import sys
import time
import logging
import threading
import serial
import serial.tools.list_ports
from queue import Queue, Empty
from PyQt5.QtCore import QObject, pyqtSignal, QPoint, QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QCursor

# config logging
logger = logging.getLogger(__name__)


class ArduinoInput(QObject):
    """
    Class for handling Arduino input via serial connection.
    Translates Arduino joystick and button commands into canvas interactions.
    """
    # signals for arduino events
    connected = pyqtSignal(bool)  # emitted when arduino connection status changes
    joystick_moved = pyqtSignal(int, int)  # emitted when joystick moves (dx, dy)
    
    # button action signals
    trigger_guess = pyqtSignal()  # trigger shape recognition
    trigger_clear = pyqtSignal()  # clear canvas
    trigger_draw_start = pyqtSignal()  # start drawing
    trigger_draw_stop = pyqtSignal()  # stop drawing
    trigger_color_change = pyqtSignal()  # change color
    trigger_thickness_change = pyqtSignal()  # change thickness

    def __init__(self, parent=None):
        """init arduino input handler"""
        super().__init__(parent)
        
        # serial connection
        self.port = None
        self.serial = None
        self.connected_port = None
        self.baud_rate = 115200
        self.is_connected = False
        
        # threading for serial reading
        self.read_thread = None
        self.running = False
        self.command_queue = Queue()
        
        # drawing state
        self.is_drawing = False
        self.cursor_position = QPoint(0, 0)
        
        # joystick settings
        self.sensitivity = 0.4
        self.acceleration = 0.4
        self.joystick_deadzone = 0 # ignore small movements (0-100 scale)
        
        # joystick last values for continuous movement
        self.last_joystick_x = 0
        self.last_joystick_y = 0
        self.joystick_timer = QTimer(self)
        self.joystick_timer.timeout.connect(self._apply_continuous_movement)
        self.joystick_timer.start(10)  # update at 100Hz for smooth motion
        
        # for auto reconnection
        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.timeout.connect(self.try_reconnect)
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

    def connect_to_arduino(self, port=None):
        """
        Connect to Arduino on specified port or auto-detect.
        
        Args:
            port: COM port to connect to (None for auto-detect)
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        # if already connected, disconnect first
        if self.is_connected:
            self.disconnect()
        
        # try COM5 port by default if no port is specified
        if port is None:
            # first try COM5
            if "COM5" in [p.device for p in serial.tools.list_ports.comports()]:
                port = "COM5"
                logger.info("Using default COM5 port for Arduino")
            else:
                # fall back to auto-detection
                port = self.find_arduino_port()
            
            if port is None:
                logger.error("No Arduino port found")
                return False
        
        try:
            logger.info(f"Connecting to Arduino on port {port}")
            self.serial = serial.Serial(port, self.baud_rate, timeout=1)
            time.sleep(2)  # give arduino time to reset
            
            # clear buffers
            self.serial.flushInput()
            self.serial.flushOutput()
            
            # consider connected immediately
            self.is_connected = True
            self.connected_port = port
            logger.info(f"Arduino connected on {port}")
            
            # start reader thread
            self.running = True
            self.read_thread = threading.Thread(target=self._read_serial)
            self.read_thread.daemon = True
            self.read_thread.start()
            
            # emit connected signal
            self.connected.emit(True)
            
            # reset reconnection counter
            self.reconnect_attempts = 0
            self.reconnect_timer.stop()
            
            return True
            
        except (serial.SerialException, OSError) as e:
            logger.error(f"Error connecting to Arduino: {e}")
            if self.serial:
                try:
                    self.serial.close()
                except:
                    pass
            return False

    def disconnect(self):
        """Disconnect from Arduino"""
        if self.is_connected:
            # stop reader thread
            self.running = False
            if self.read_thread:
                self.read_thread.join(timeout=1.0)
            
            # close serial connection
            try:
                self.serial.close()
            except:
                pass
            
            # update state
            self.is_connected = False
            self.connected_port = None
            
            # emit disconnected signal
            self.connected.emit(False)
            logger.info("Disconnected from Arduino")

    def find_arduino_port(self):
        """
        Find Arduino serial port automatically.
        
        Returns:
            str: Port name or None if not found
        """
        ports = list(serial.tools.list_ports.comports())
        logger.info(f"Available ports: {[p.device for p in ports]}")
        
        # look for arduino indicators in port descriptions
        for port in ports:
            desc = port.description.lower()
            if any(name in desc for name in ["arduino", "uno", "ch340", "usb serial"]):
                logger.info(f"Found likely Arduino on {port.device}: {port.description}")
                return port.device
        
        # if only one port available, assume it's arduino
        if len(ports) == 1:
            logger.info(f"Only one port available, assuming Arduino: {ports[0].device}")
            return ports[0].device
        
        return None

    def set_sensitivity(self, value):
        """
        Set joystick sensitivity.
        
        Args:
            value: Sensitivity value (1.0 is normal)
        """
        self.sensitivity = max(0.1, min(5.0, value))
        logger.info(f"Joystick sensitivity set to {self.sensitivity}")

    def set_acceleration(self, value):
        """
        Set joystick movement acceleration.
        
        Args:
            value: Acceleration factor (1.0 is linear)
        """
        self.acceleration = max(1.0, min(2.0, value))
        logger.info(f"Joystick acceleration set to {self.acceleration}")
        
    def set_deadzone(self, value):
        """
        Set joystick deadzone to ignore small movements.
        
        Args:
            value: Deadzone value (0-100)
        """
        self.joystick_deadzone = max(0, min(50, value))
        logger.info(f"Joystick deadzone set to {self.joystick_deadzone}")

    def try_reconnect(self):
        """Try to reconnect to Arduino after connection loss"""
        self.reconnect_attempts += 1
        logger.info(f"Attempting to reconnect (attempt {self.reconnect_attempts})")
        
        if self.connect_to_arduino(self.connected_port):
            logger.info("Reconnection successful")
            return
            
        # if max attempts reached, stop trying
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.warning("Max reconnection attempts reached, giving up")
            self.reconnect_timer.stop()
            self.reconnect_attempts = 0
            return
            
        # exponential backoff for retry interval
        backoff = min(30, 2 ** self.reconnect_attempts)
        self.reconnect_timer.start(backoff * 1000)

    def get_available_ports(self):
        """
        Get list of available COM ports.
        
        Returns:
            list: List of available port names
        """
        return [p.device for p in serial.tools.list_ports.comports()]

    def _read_serial(self):
        """Background thread to read data from Arduino"""
        logger.info("Serial read thread started")
        
        # buffer for incomplete data
        buffer = ""
        
        while self.running and self.serial and self.serial.is_open:
            try:
                # check for data
                if self.serial.in_waiting:
                    # read raw data
                    data = self.serial.read(self.serial.in_waiting).decode('utf-8', errors='replace')
                    
                    # add to buffer
                    buffer += data
                    
                    # process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        message = line.strip()
                        
                        if message:
                            # add to command queue
                            self.command_queue.put(message)
                            # process command
                            self._process_command(message)
                
                # process partial data after timeout
                elif buffer:
                    # if no newline received in a while, process buffer content
                    message = buffer.strip()
                    if message:
                        self.command_queue.put(message)
                        self._process_command(message)
                    buffer = ""
                    
            except (serial.SerialException, OSError) as e:
                logger.error(f"Error reading from Arduino: {e}")
                self.is_connected = False
                self.connected.emit(False)
                
                # start reconnection attempts
                if not self.reconnect_timer.isActive():
                    self.reconnect_timer.start(2000)  # try again in 2 seconds
                break
                
            except UnicodeDecodeError as e:
                # just log decode errors and continue
                logger.warning(f"Unicode decode error (ignoring): {e}")
                buffer = ""  # clear buffer on decode error
                
            # small delay to prevent cpu overuse
            time.sleep(0.01)
            
        logger.info("Serial read thread stopped")

    def _process_command(self, message):
        """
        Process command from Arduino.
        
        Args:
            message: Command string from Arduino
        """
        if not message:
            return
            
        logger.debug(f"Arduino command: {message}")
        
        # more flexible message parsing - accept different formats
        
        # check for joystick data
        if "JOY:" in message or "joy:" in message or "X:" in message:
            # extract numbers from the message
            import re
            numbers = re.findall(r'-?\d+', message)
            
            if len(numbers) >= 2:
                try:
                    x = int(numbers[0])
                    y = int(numbers[1])
                    
                    # update last joystick values for continuous movement
                    # store raw values (will be processed in continuous timer)
                    self.last_joystick_x = x
                    self.last_joystick_y = y
                    
                except (ValueError, IndexError) as e:
                    logger.error(f"Error parsing joystick data: {e}")
        
        # process button commands
        elif "BTN:" in message or "btn:" in message or "BUTTON:" in message:
            # extract command - everything after the prefix
            command = None
            if "BTN:" in message:
                command = message.split("BTN:")[1].strip()
            elif "btn:" in message:
                command = message.split("btn:")[1].strip()
            elif "BUTTON:" in message:
                command = message.split("BUTTON:")[1].strip()
                
            if command:
                self._handle_button_command(command)
                
        # check for simple keyword commands without prefixes
        elif any(keyword in message.upper() for keyword in ["GUESS", "CLEAR", "DRAW", "COLOR", "THICKNESS"]):
            # extract the command from the message
            command = message.strip().upper()
            self._handle_button_command(command)

    def _handle_button_command(self, command):
        """
        Handle button command from Arduino.
        
        Args:
            command: Button command string
        """
        logger.debug(f"Button command: {command}")
        
        # normalize command to uppercase and remove extra whitespace
        cmd = command.upper().strip()
        
        # check for different variations of commands
        if any(guess in cmd for guess in ["GUESS", "RECOGNIZE", "IDENTIFY"]):
            self.trigger_guess.emit()
            
        elif any(clear in cmd for clear in ["CLEAR", "ERASE", "RESET"]):
            self.trigger_clear.emit()
            
        elif any(draw_start in cmd for draw_start in ["DRAW_START", "DRAWING_START", "START_DRAWING", "DRAW START"]):
            self.is_drawing = True
            self.trigger_draw_start.emit()
            
        elif any(draw_stop in cmd for draw_stop in ["DRAW_STOP", "DRAWING_STOP", "STOP_DRAWING", "DRAW STOP"]):
            self.is_drawing = False
            self.trigger_draw_stop.emit()
            
        # just "DRAW" can toggle drawing state
        elif cmd == "DRAW":
            self.is_drawing = not self.is_drawing
            if self.is_drawing:
                self.trigger_draw_start.emit()
            else:
                self.trigger_draw_stop.emit()
                
        elif any(color in cmd for color in ["COLOR", "COLOUR", "CHANGE_COLOR"]):
            self.trigger_color_change.emit()
            
        elif any(thick in cmd for thick in ["THICKNESS", "WIDTH", "SIZE"]):
            self.trigger_thickness_change.emit()
            
    def _apply_continuous_movement(self):
        """Timer callback to apply continuous joystick movement"""
        # skip if not connected
        if not self.is_connected:
            return
            
        # apply deadzone to raw joystick values
        x = self.last_joystick_x
        y = self.last_joystick_y
        
        # only move if outside deadzone
        if abs(x) <= self.joystick_deadzone and abs(y) <= self.joystick_deadzone:
            return
            
        # apply deadzone (values below deadzone become 0)
        if abs(x) <= self.joystick_deadzone:
            x = 0
        if abs(y) <= self.joystick_deadzone:
            y = 0
            
        # calculate movement based on joystick position
        # the further from center, the faster it moves
        dx = self._calculate_movement(x)
        dy = self._calculate_movement(y)
        
        # update cursor if there's any movement
        if dx != 0 or dy != 0:
            self._update_cursor_position(dx, dy)
            
            # emit signal for other components
            self.joystick_moved.emit(dx, dy)

    def _calculate_movement(self, value):
        """
        Calculate movement amount based on joystick position.
        Implements intuitive control where position determines speed.
        
        Args:
            value: Raw joystick value (-100 to 100)
            
        Returns:
            Movement delta (pixels)
        """
        # normalize to -1.0 to 1.0
        normalized = value / 100.0
        
        # apply deadzone
        if abs(normalized) <= (self.joystick_deadzone / 100.0):
            return 0
            
        # apply acceleration curve (higher acceleration = faster response)
        sign = 1 if normalized >= 0 else -1
        accelerated = sign * (abs(normalized) ** self.acceleration)
        
        # apply sensitivity multiplier
        # higher value = more pixels moved per update
        movement = accelerated * self.sensitivity * 5.0  # base speed factor
        
        return int(movement)

    def _update_cursor_position(self, dx, dy):
        """
        Update cursor position based on joystick movement.
        
        Args:
            dx: X movement delta
            dy: Y movement delta
        """
        # get current position
        current_pos = QCursor.pos()
        
        # calculate new position
        new_x = current_pos.x() + dx
        new_y = current_pos.y() + dy
        
        # update position
        QCursor.setPos(new_x, new_y)
        
        # store position
        self.cursor_position = QPoint(new_x, new_y)


def setup_arduino_integration(main_window, port=None):
    """
    Set up Arduino integration with the main window.
    
    Args:
        main_window: MainWindow instance
        port: Optional COM port to connect to (None for auto-detect)
    
    Returns:
        ArduinoInput: Configured Arduino input handler
    """
    # create arduino input handler
    arduino_input = ArduinoInput(main_window)
    
    # store reference in main window
    main_window.arduino_input = arduino_input
    
    # connect signals to main window functions
    
    # connection status changes
    def on_arduino_connection_changed(connected):
        if connected:
            main_window.statusBar().showMessage("Arduino controller connected")
        else:
            main_window.statusBar().showMessage("Arduino controller disconnected")
    
    arduino_input.connected.connect(on_arduino_connection_changed)
    
    # button actions
    arduino_input.trigger_guess.connect(main_window.guess_drawing)
    arduino_input.trigger_clear.connect(main_window.on_canvas_clear)
    arduino_input.trigger_color_change.connect(_cycle_colors(main_window))
    arduino_input.trigger_thickness_change.connect(_cycle_thickness(main_window))
    
    # drawing control
    arduino_input.trigger_draw_start.connect(
        lambda: _simulate_mouse_press(main_window.canvas)
    )
    arduino_input.trigger_draw_stop.connect(
        lambda: _simulate_mouse_release(main_window.canvas)
    )
    
    # try to connect to arduino with specified port
    QTimer.singleShot(1000, lambda: arduino_input.connect_to_arduino(port))
    
    return arduino_input


def _cycle_colors(main_window):
    """
    Create a function to cycle through color palette.
    
    Args:
        main_window: MainWindow instance
        
    Returns:
        Function to cycle colors
    """
    import config
    
    # access to color palette
    color_palette = config.COLOR_PALETTE
    color_index = [0]  # use list for mutable closure
    
    def cycle_color():
        # increment index and wrap around
        color_index[0] = (color_index[0] + 1) % len(color_palette)
        
        # get new color
        new_color = color_palette[color_index[0]]["value"]
        color_name = color_palette[color_index[0]]["name"]
        
        # set color on canvas and toolbar
        main_window.canvas.set_pen_color(new_color)
        main_window.toolbar.color_selected.emit(new_color)
        
        # update status bar
        main_window.statusBar().showMessage(f"Color changed to {color_name}")
        
    return cycle_color


def _cycle_thickness(main_window):
    """
    Create a function to cycle through pen thickness options.
    
    Args:
        main_window: MainWindow instance
        
    Returns:
        Function to cycle thickness
    """
    # thickness options
    thickness_options = [2, 5, 10, 15, 20]
    thickness_index = [0]  # use list for mutable closure
    
    def cycle_thickness():
        # increment index and wrap around
        thickness_index[0] = (thickness_index[0] + 1) % len(thickness_options)
        
        # get new thickness
        new_thickness = thickness_options[thickness_index[0]]
        
        # set thickness on canvas and toolbar
        main_window.canvas.set_pen_width(new_thickness)
        main_window.toolbar.size_selected.emit(new_thickness)
        
        # update status bar
        main_window.statusBar().showMessage(f"Pen thickness changed to {new_thickness}")
        
    return cycle_thickness


def _simulate_mouse_press(canvas):
    """
    Simulate mouse press event on canvas at current cursor position.
    
    Args:
        canvas: DrawingCanvas instance
    """
    # get current cursor position relative to canvas
    cursor_pos = canvas.mapFromGlobal(QCursor.pos())
    
    # create fake mouse press event
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtCore import Qt
    
    event = QMouseEvent(
        QMouseEvent.MouseButtonPress,
        cursor_pos,
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier
    )
    
    # send event to canvas
    canvas.mousePressEvent(event)


def _simulate_mouse_release(canvas):
    """
    Simulate mouse release event on canvas at current cursor position.
    
    Args:
        canvas: DrawingCanvas instance
    """
    # get current cursor position relative to canvas
    cursor_pos = canvas.mapFromGlobal(QCursor.pos())
    
    # create fake mouse release event
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtCore import Qt
    
    event = QMouseEvent(
        QMouseEvent.MouseButtonRelease,
        cursor_pos,
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier
    )
    
    # send event to canvas
    canvas.mouseReleaseEvent(event)