"""
Enhanced database operations for the Drawing AI App with robust
error handling and schema migration support
"""
import os
import json
import logging
import threading
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

# Try using sqlite3
try:
    import sqlite3 as sqlite
    print("Using sqlite3 for database operations")
except ImportError:
    print("Error: sqlite3 module not available")
    raise

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("database.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Database")


class DatabaseError(Exception):
    """Custom exception for database errors"""
    pass


class Database:
    """Enhanced database manager for the application with error handling and migration support"""

    # Lock for thread safety
    _lock = threading.RLock()
    # Schema version
    SCHEMA_VERSION = 2

    def __init__(self, db_file=None):
        """
        Initialize the database

        Args:
            db_file: Path to the database file
        """
        self.db_file = db_file or config.DATABASE_FILE
        self.connection = None
        self._connected = False

        # Initialize the database
        try:
            self._init_db()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")

    @contextmanager
    def _get_connection(self):
        """
        Context manager for getting a database connection

        Yields:
            sqlite.Connection: Database connection
        """
        with self._lock:
            if not self._connected:
                self._connect()

            try:
                yield self.connection
            except sqlite.Error as e:
                self.connection.rollback()
                logger.error(f"Database error: {e}")
                raise DatabaseError(f"Database operation failed: {e}")
            except Exception as e:
                self.connection.rollback()
                logger.error(f"Unexpected error: {e}")
                raise

    @contextmanager
    def _get_cursor(self):
        """
        Context manager for getting a database cursor

        Yields:
            sqlite.Cursor: Database cursor
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def _connect(self):
        """Connect to the database"""
        try:
            # Create the database directory if it doesn't exist
            db_dir = os.path.dirname(self.db_file)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)

            # Connect to the database
            self.connection = sqlite.connect(self.db_file, check_same_thread=False)

            # Enable foreign keys
            self.connection.execute("PRAGMA foreign_keys = ON")

            self._connected = True
            logger.info(f"Connected to database: {self.db_file}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self._connected = False
            raise DatabaseError(f"Database connection failed: {e}")

    def _init_db(self):
        """Initialize the database with error handling and migrations"""
        try:
            self._connect()

            # Check if we need to create the schema or migrate
            self._check_and_run_migrations()

            return True
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            return False

    def _check_and_run_migrations(self):
        """Check schema version and run migrations if needed"""
        with self._get_cursor() as cursor:
            # Check if schema_info table exists
            cursor.execute(
                '''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_info'
                '''
            )
            has_schema_info = cursor.fetchone() is not None

            current_version = 0
            if has_schema_info:
                # Get current schema version
                cursor.execute('SELECT version FROM schema_info')
                result = cursor.fetchone()
                current_version = result[0] if result else 0

            if current_version < self.SCHEMA_VERSION:
                logger.info(f"Database schema version: {current_version}, " +
                           f"latest version: {self.SCHEMA_VERSION}")
                self._run_migrations(current_version)

    def _run_migrations(self, current_version):
        """
        Run database migrations

        Args:
            current_version: Current schema version
        """
        with self._get_cursor() as cursor:
            # Create schema_info table if it doesn't exist
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER PRIMARY KEY
                )
                '''
            )

            # Run migrations based on current version
            if current_version < 1:
                logger.info("Running migration to version 1")
                self._create_base_schema(cursor)

            if current_version < 2:
                logger.info("Running migration to version 2")
                self._migrate_to_v2(cursor)

            # Update schema version
            if current_version == 0:
                cursor.execute('INSERT INTO schema_info (version) VALUES (?)',
                              (self.SCHEMA_VERSION,))
            else:
                cursor.execute('UPDATE schema_info SET version = ?',
                              (self.SCHEMA_VERSION,))

            logger.info(f"Database migrated to version {self.SCHEMA_VERSION}")

    def _create_base_schema(self, cursor):
        """
        Create the base database schema (version 1)

        Args:
            cursor: Database cursor
        """
        # Create the drawing history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS drawing_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            top_guess TEXT,
            confidence REAL,
            all_guesses TEXT,
            created_at TEXT NOT NULL
        )
        ''')

        # Create the user settings table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_key TEXT UNIQUE NOT NULL,
            setting_value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')

        # Create the labeled shapes table for training data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS labeled_shapes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shape_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            confidence REAL,
            timestamp TEXT NOT NULL,
            is_correct BOOLEAN NOT NULL DEFAULT 1
        )
        ''')

    def _migrate_to_v2(self, cursor):
        """
        Migrate schema from version 1 to 2

        Args:
            cursor: Database cursor
        """
        # Add features column to labeled_shapes table
        cursor.execute('''
        ALTER TABLE labeled_shapes ADD COLUMN features TEXT
        ''')

        # Add model_version column to labeled_shapes table
        cursor.execute('''
        ALTER TABLE labeled_shapes ADD COLUMN model_version INTEGER DEFAULT 1
        ''')

        # Add usage_stats table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS usage_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            session_id TEXT NOT NULL,
            drawings_count INTEGER DEFAULT 0,
            correct_guesses INTEGER DEFAULT 0,
            incorrect_guesses INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
        ''')

        # Add shape_stats table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS shape_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shape_name TEXT NOT NULL,
            drawn_count INTEGER DEFAULT 0,
            correct_count INTEGER DEFAULT 0,
            incorrect_count INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        ''')

    def close(self):
        """Close the database connection with error handling"""
        if self.connection:
            try:
                self.connection.close()
                self._connected = False
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")

    # Add methods for get_drawing_history, get_labeled_shapes, etc.
    def save_drawing(self, image_path, top_guess=None, confidence=None, all_guesses=None):
        """
        Save a drawing to the history

        Args:
            image_path: Path to the saved image
            top_guess: Top guess from recognition
            confidence: Confidence score of top guess
            all_guesses: All guesses from recognition

        Returns:
            bool: Success
        """
        try:
            with self._get_cursor() as cursor:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Convert all_guesses to JSON if it exists
                guesses_json = None
                if all_guesses:
                    guesses_json = json.dumps(all_guesses)

                cursor.execute('''
                INSERT INTO drawing_history
                (timestamp, image_path, top_guess, confidence, all_guesses, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp, image_path, top_guess, confidence, guesses_json, timestamp))

                return True
        except Exception as e:
            logger.error(f"Error saving drawing: {e}")
            return False

    def get_drawing_history(self, limit=10, offset=0):
        """
        Get drawing history

        Args:
            limit: Max number of records
            offset: Offset for pagination

        Returns:
            list: Drawing history records
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute('''
                SELECT id, timestamp, image_path, top_guess, confidence, all_guesses
                FROM drawing_history
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                ''', (limit, offset))

                rows = cursor.fetchall()

                result = []
                for row in rows:
                    # Parse JSON if all_guesses is not None
                    all_guesses = None
                    if row[5]:  # all_guesses column
                        try:
                            all_guesses = json.loads(row[5])
                        except:
                            pass

                    result.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'image_path': row[2],
                        'top_guess': row[3],
                        'confidence': row[4],
                        'all_guesses': all_guesses
                    })

                return result
        except Exception as e:
            logger.error(f"Error getting drawing history: {e}")
            return []

    def delete_drawing(self, drawing_id):
        """
        Delete a drawing from history

        Args:
            drawing_id: ID of drawing to delete

        Returns:
            bool: Success
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute('DELETE FROM drawing_history WHERE id = ?', (drawing_id,))
                return True
        except Exception as e:
            logger.error(f"Error deleting drawing: {e}")
            return False

    def get_setting(self, key, default=None):
        """
        Get a setting value

        Args:
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value or default
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute('SELECT setting_value FROM user_settings WHERE setting_key = ?', (key,))
                row = cursor.fetchone()

                if row:
                    return row[0]
                return default
        except Exception as e:
            logger.error(f"Error getting setting: {e}")
            return default

    def save_setting(self, key, value):
        """
        Save a setting

        Args:
            key: Setting key
            value: Setting value

        Returns:
            bool: Success
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Convert value to string
            if not isinstance(value, str):
                value = str(value)

            with self._get_cursor() as cursor:
                # Check if the setting already exists
                cursor.execute('SELECT id FROM user_settings WHERE setting_key = ?', (key,))
                row = cursor.fetchone()

                if row:
                    # Update existing setting
                    cursor.execute('''
                    UPDATE user_settings
                    SET setting_value = ?, updated_at = ?
                    WHERE setting_key = ?
                    ''', (value, timestamp, key))
                else:
                    # Insert new setting
                    cursor.execute('''
                    INSERT INTO user_settings
                    (setting_key, setting_value, updated_at)
                    VALUES (?, ?, ?)
                    ''', (key, value, timestamp))

                return True
        except Exception as e:
            logger.error(f"Error saving setting: {e}")
            return False

    def save_labeled_shape(self, image_path, shape_name, confidence=None, is_correct=True, features=None):
        """
        Save a labeled shape for training

        Args:
            image_path: Path to the image
            shape_name: Shape name/label
            confidence: Recognition confidence
            is_correct: Whether the recognition was correct
            features: Shape features for ML

        Returns:
            bool: Success
        """
        try:
            with self._get_cursor() as cursor:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Convert features to JSON if it exists
                features_json = None
                if features:
                    features_json = json.dumps(features)

                cursor.execute('''
                INSERT INTO labeled_shapes
                (shape_name, image_path, confidence, timestamp, is_correct, features)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (shape_name, image_path, confidence, timestamp, is_correct, features_json))

                return True
        except Exception as e:
            logger.error(f"Error saving labeled shape: {e}")
            return False

    def get_labeled_shapes(self, shape_name=None, limit=None):
        """
        Get labeled shapes

        Args:
            shape_name: Optional shape name filter
            limit: Optional limit on number of records

        Returns:
            list: Labeled shapes
        """
        try:
            with self._get_cursor() as cursor:
                query = '''
                SELECT id, shape_name, image_path, confidence, timestamp, is_correct, features
                FROM labeled_shapes
                '''

                params = []

                if shape_name:
                    query += ' WHERE shape_name = ?'
                    params.append(shape_name)

                query += ' ORDER BY timestamp DESC'

                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                result = []
                for row in rows:
                    # Parse features JSON if not None
                    features = None
                    if row[6]:  # features column
                        try:
                            features = json.loads(row[6])
                        except:
                            pass

                    result.append({
                        'id': row[0],
                        'shape_name': row[1],
                        'image_path': row[2],
                        'confidence': row[3],
                        'timestamp': row[4],
                        'is_correct': bool(row[5]),
                        'features': features
                    })

                return result
        except Exception as e:
            logger.error(f"Error getting labeled shapes: {e}")
            return []

    def get_shape_stats(self, limit=None):
        """
        Get shape statistics

        Args:
            limit: Optional limit on number of records

        Returns:
            list: Shape statistics
        """
        try:
            with self._get_cursor() as cursor:
                query = '''
                SELECT shape_name, drawn_count, correct_count, incorrect_count, avg_confidence
                FROM shape_stats
                ORDER BY correct_count DESC
                '''

                params = []

                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                result = []
                for row in rows:
                    # Calculate accuracy
                    drawn = row[1] or 0
                    correct = row[2] or 0
                    accuracy = (correct / drawn * 100) if drawn > 0 else 0

                    result.append({
                        'shape_name': row[0],
                        'drawn_count': drawn,
                        'correct_count': correct,
                        'incorrect_count': row[3] or 0,
                        'avg_confidence': row[4] or 0,
                        'accuracy': accuracy
                    })

                return result
        except Exception as e:
            logger.error(f"Error getting shape stats: {e}")
            return []