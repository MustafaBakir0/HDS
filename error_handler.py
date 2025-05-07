"""
Error handling module for the Drawing AI App
Provides robust exception handling, logging, and user-friendly error messages
"""
import os
import sys
import time
import traceback
import logging
from datetime import datetime
from functools import wraps
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtCore import Qt, QObject, pyqtSignal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ErrorHandler")


class ErrorHandler(QObject):
    """Global application error handler"""

    # Signals
    error_occurred = pyqtSignal(str, str)  # Error message, details

    def __init__(self, app=None):
        """
        Initialize the error handler

        Args:
            app: QApplication instance
        """
        super().__init__()

        self.app = app
        self.error_count = 0
        self.last_error_time = 0
        self.error_log = []

        # Set up global exception hook if app is provided
        if app is not None:
            self._setup_exception_hook()

    def _setup_exception_hook(self):
        """Set up global exception handling"""
        sys.excepthook = self._handle_exception

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """
        Global exception handler

        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        # Don't catch KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log the exception
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        # Format traceback for details
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        # Add to error log
        error_entry = {
            'time': datetime.now(),
            'type': exc_type.__name__,
            'message': str(exc_value),
            'traceback': tb_str
        }
        self.error_log.append(error_entry)

        # Show error dialog to user
        self.show_error_dialog(
            "An unexpected error occurred",
            f"Error: {exc_value}\n\nThe application will attempt to continue.",
            tb_str
        )

    def handle_error(self, e, context=""):
        """
        Handle an exception and show error message

        Args:
            e: Exception object
            context: Context description

        Returns:
            bool: True if handled, False if critical
        """
        # Update error count and time
        self.error_count += 1
        now = time.time()

        # Check if we're in an error storm (many errors in rapid succession)
        if now - self.last_error_time < 1.0 and self.error_count > 5:
            logger.critical(f"Error storm detected! {self.error_count} errors in less than a second.")
            self.show_error_dialog(
                "Critical Error",
                "Multiple errors are occurring rapidly. The application may be unstable.",
                f"Last error: {str(e)}\nContext: {context}"
            )
            return False

        self.last_error_time = now

        # Log the error
        logger.error(f"{context} error: {e}", exc_info=True)

        # Add to error log
        error_entry = {
            'time': datetime.now(),
            'type': type(e).__name__,
            'message': str(e),
            'context': context,
            'traceback': traceback.format_exc()
        }
        self.error_log.append(error_entry)

        # Emit signal
        self.error_occurred.emit(f"{context} error: {str(e)}", traceback.format_exc())

        # Determine if error is critical
        is_critical = any(critical_term in str(e).lower() for critical_term in [
            "database is locked", "permission denied", "out of memory",
            "segmentation fault", "access violation"
        ])

        # Show error message if it's a critical error
        if is_critical:
            self.show_error_dialog(
                "Critical Error",
                f"A critical error occurred: {str(e)}",
                traceback.format_exc()
            )
            return False

        return True

    def show_error_dialog(self, title, message, details=None):
        """
        Show an error dialog

        Args:
                        title: Title of the dialog
            message: Error message
            details: Optional error details
        """
        # Create the message box
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)

        # Add details if provided
        if details:
            msg_box.setDetailedText(details)

        # Add buttons
        msg_box.setStandardButtons(QMessageBox.Ok)

        # Show modal dialog
        msg_box.exec_()

    def save_error_log(self, filename=None):
        """
        Save the error log to a file

        Args:
            filename: Optional filename, defaults to error_log_[timestamp].txt

        Returns:
            str: Path to the log file, or None if failed
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"error_log_{timestamp}.txt"

            # Make sure the logs directory exists
            os.makedirs("logs", exist_ok=True)
            filepath = os.path.join("logs", filename)

            # Write the log to the file
            with open(filepath, 'w') as f:
                f.write(f"Error Log ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
                f.write("=" * 80 + "\n\n")

                for i, error in enumerate(self.error_log):
                    f.write(f"Error {i + 1}: {error['type']} at {error['time']}\n")
                    f.write(f"Context: {error.get('context', 'Unknown')}\n")
                    f.write(f"Message: {error['message']}\n")
                    f.write("Traceback:\n")
                    f.write(error.get('traceback', 'No traceback available'))
                    f.write("\n" + "-" * 80 + "\n\n")

            return filepath
        except Exception as e:
            logger.error(f"Failed to save error log: {e}")
            return None

    def clear_error_log(self):
        """Clear the error log"""
        self.error_log = []
        self.error_count = 0

    def get_error_count(self):
        """Get the number of errors"""
        return self.error_count


# Function decorators for error handling

def handle_errors(context="Operation"):
    """
    Decorator for handling errors in functions

    Args:
        context: Context description for the operation

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get the instance of the class if this is a method
                instance = args[0] if args and isinstance(args[0], QObject) else None

                # Try to get the error handler from the instance or its parent
                error_handler = None
                if instance:
                    if hasattr(instance, 'error_handler'):
                        error_handler = instance.error_handler
                    elif hasattr(instance, 'parent') and instance.parent() and hasattr(instance.parent(),
                                                                                       'error_handler'):
                        error_handler = instance.parent().error_handler

                # If no error handler found, create a temporary one
                if not error_handler:
                    error_handler = ErrorHandler()

                # Handle the error
                if error_handler.handle_error(e, context):
                    # Non-critical error, return None or default value
                    return None
                else:
                    # Critical error, re-raise
                    raise

        return wrapper

    return decorator


def handle_db_errors(error_value=None):
    """
    Decorator for handling database errors

    Args:
        error_value: Value to return on error

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get the instance of the class
                instance = args[0]

                # Log the error
                logger.error(f"Database error in {func.__name__}: {e}", exc_info=True)

                # Try to close and reopen the database connection if it's locked
                if "database is locked" in str(e).lower() and hasattr(instance, 'connection'):
                    try:
                        if instance.connection:
                            instance.connection.close()
                        time.sleep(0.5)  # Give it a moment
                        instance._init_db()  # Reinitialize the connection

                        # Try the operation again
                        return func(*args, **kwargs)
                    except Exception as retry_error:
                        logger.error(f"Failed to retry database operation: {retry_error}")

                # Return the error value
                return error_value

        return wrapper

    return decorator


def handle_ui_errors(show_dialog=True):
    """
    Decorator for handling UI errors

    Args:
        show_dialog: Whether to show a dialog to the user

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get the instance of the class
                instance = args[0]

                # Get context
                context = f"UI action '{func.__name__}'"

                # Log the error
                logger.error(f"{context} error: {e}", exc_info=True)

                # Show dialog if requested
                if show_dialog:
                    QMessageBox.warning(
                        instance,
                        "Operation Failed",
                        f"The operation could not be completed: {str(e)}"
                    )

                # Signal the error if the instance has an error_occurred signal
                if hasattr(instance, 'error_occurred') and isinstance(instance.error_occurred, pyqtSignal):
                    instance.error_occurred.emit(str(e))

                # Return None
                return None

        return wrapper

    return decorator


# Application recovery functions

def recover_corrupt_database(db_file):
    """
    Attempt to recover a corrupt database

    Args:
        db_file: Path to the database file

    Returns:
        bool: True if recovery was successful
    """
    import sqlite3
    import shutil

    try:
        # Create a backup first
        backup_file = f"{db_file}.backup_{int(time.time())}"
        shutil.copy2(db_file, backup_file)
        logger.info(f"Created database backup: {backup_file}")

        # Try to connect and recover
        conn = sqlite3.connect(db_file)

        # Run integrity check
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]

        if result == "ok":
            logger.info("Database integrity check passed")
            conn.close()
            return True

        # Try to recover with dump and reload
        logger.warning(f"Database integrity check failed: {result}")

        # Dump the database to SQL
        dump_file = f"{db_file}.dump_{int(time.time())}.sql"
        os.system(f"sqlite3 {db_file} .dump > {dump_file}")

        # Create a new database
        recovered_file = f"{db_file}.recovered_{int(time.time())}"
        os.system(f"sqlite3 {recovered_file} < {dump_file}")

        # Check if recovery was successful
        conn = sqlite3.connect(recovered_file)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        conn.close()

        if result == "ok":
            logger.info("Database recovery successful")

            # Replace the original with the recovered version
            shutil.copy2(recovered_file, db_file)
            return True
        else:
            logger.error("Database recovery failed")
            return False
    except Exception as e:
        logger.error(f"Error recovering database: {e}", exc_info=True)
        return False


def check_filesystem_permissions():
    """
    Check if the application has proper filesystem permissions

    Returns:
        dict: Status of various filesystem operations
    """
    results = {
        'can_read_app_dir': False,
        'can_write_app_dir': False,
        'can_create_files': False,
        'can_write_temp': False
    }

    try:
        # Check if we can read the application directory
        app_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
        os.listdir(app_dir)
        results['can_read_app_dir'] = True

        # Check if we can write to the application directory
        test_file = os.path.join(app_dir, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            results['can_write_app_dir'] = True
        except Exception:
            pass

        # Check if we can create files
        try:
            os.makedirs('data', exist_ok=True)
            test_file = os.path.join('data', '.create_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            results['can_create_files'] = True
        except Exception:
            pass

        # Check if we can write to the temp directory
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=True) as f:
            f.write('test')
        results['can_write_temp'] = True

        return results
    except Exception as e:
        logger.error(f"Error checking filesystem permissions: {e}")
        return results


def system_health_check():
    """
    Perform a basic health check of the system

    Returns:
        dict: Health check results
    """
    health = {
        'status': 'healthy',
        'memory_usage': 0,
        'disk_space': 0,
        'filesystem_status': {},
        'issues': []
    }

    try:
        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        health['memory_usage'] = memory.percent

        if memory.percent > 90:
            health['status'] = 'warning'
            health['issues'].append(f"High memory usage: {memory.percent}%")

        # Check disk space
        disk = psutil.disk_usage('.')
        health['disk_space'] = disk.percent

        if disk.percent > 90:
            health['status'] = 'warning'
            health['issues'].append(f"Low disk space: {disk.percent}%")

        # Check filesystem permissions
        health['filesystem_status'] = check_filesystem_permissions()

        if not health['filesystem_status']['can_write_app_dir']:
            health['status'] = 'warning'
            health['issues'].append("Limited filesystem write permissions")

        return health
    except Exception as e:
        logger.error(f"Error performing health check: {e}")

        health['status'] = 'unknown'
        health['issues'].append(f"Health check failed: {str(e)}")

        return health