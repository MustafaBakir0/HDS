# Hand-Drawn Shapes Recognition

An interactive application that recognizes hand-drawn shapes using computer vision and machine learning techniques. This system combines geometric analysis via OpenCV with machine learning classification through scikit-learn to provide accurate recognition of various shapes.

## Project Overview

Hand-Drawn Shapes Recognition is a desktop application built with PyQt5 that allows users to draw shapes on a canvas and receive real-time recognition of what they've drawn. The system analyzes the drawings using a dual-approach that combines traditional computer vision techniques with machine learning to achieve higher accuracy.

The application features a user-friendly interface with drawing tools, an AI assistant that communicates recognition results, and a comprehensive training system that allows the model to improve over time. It also includes optional Arduino integration for alternative input methods.

## Technical Architecture

The project is structured into several modules, each handling specific aspects of the application:

### Core Components

#### Shape Recognition System (`core/recognition.py`)

The heart of the application is its shape recognition system, which employs a multi-faceted approach:

1. **Geometric Analysis**: The application first analyzes the drawn shape using traditional computer vision techniques through the `GeometricAnalyzer` class. This involves:
   - Contour extraction and analysis using OpenCV
   - Calculation of shape properties like circularity, aspect ratio, convexity
   - Approximation of polygon vertices at different thresholds
   - Fitting shapes to geometric primitives (circles, rectangles, triangles)
   - Analysis of symmetry, angles, and corners
   
2. **Feature Extraction**: The `ShapeFeatureExtractor` class extracts a comprehensive set of features from the shape:
   - Basic metrics like area, perimeter, bounding box
   - Shape properties like circularity, aspect ratio, convexity
   - Moment-based features including Hu Moments (scale, rotation, and translation invariant)
   - Contour approximations at different epsilon values
   - Corner angles and side lengths analysis
   - Symmetry metrics (vertical and horizontal)
   - Minimum enclosing shape analysis

3. **Machine Learning Classification**: The `ShapeClassifier` class implements a machine learning approach:
   - Uses scikit-learn's LinearSVC (Support Vector Classification)
   - Includes feature normalization through StandardScaler
   - Implements feature selection to use the most discriminative features
   - Provides confidence scores for different shape classes
   
4. **Combined Decision Making**: The final recognition combines both geometric and ML approaches:
   - Geometric scores provide baseline recognition
   - ML classification results are weighted higher when available
   - Confidence scores are normalized and ranked to produce top guesses

This dual approach allows the system to leverage both rule-based knowledge and learned patterns, making it more robust to variations in drawing styles.

#### Drawing Management (`core/drawing_manager.py`)

The `DrawingManager` class serves as a bridge between the UI and the recognition system. It manages:
- Recognition requests and results transmission
- Drawing history tracking
- Drawing saving functionality

#### Conversation Management (`core/conversation.py`)

This module handles the AI assistant's interactions with the user, providing natural language responses based on recognition results. It features:
- Dynamic and varied responses based on confidence levels
- Text-to-speech capabilities
- Sequential response templates for multiple drawing attempts

#### Arduino Integration (`core/arduino_controller.py` and `arduino_input.py`)

For users who want to use alternative input methods, the system includes Arduino integration, allowing drawings to be created using physical controllers or sensors.

### User Interface

#### Main Window (`ui/main_window.py`)

The central component of the UI, featuring:
- Drawing canvas
- Tool selection panel
- AI response display
- Recognition and training controls
- Settings and file management

#### Canvas (`ui/canvas.py`)

An interactive drawing area implemented with PyQt5 that:
- Handles mouse events for drawing
- Manages pen properties (color, width)
- Supports eraser functionality
- Provides image export capabilities

#### Dialog Windows

Several dialog windows for interaction:
- `SettingsDialog` for application configuration
- `LabelDialog` for confirming and labeling drawings
- `TrainingDialog` for managing training data

### Data Management

#### Database Interface (`data/database.py`)

Handles data persistence using SQLite, including:
- User settings storage
- Labeled shapes for training
- Drawing history

#### User Settings (`data/user_settings.py`)

Manages application configuration preferences such as:
- UI theme (dark/light mode)
- Drawing tool defaults
- AI interaction preferences
- Training settings

#### Drawing History (`data/history.py`)

Tracks and manages past drawings and recognition results.

## Key Algorithms and Techniques

### Contour Analysis

The system begins recognition by processing the drawn image:
1. Converting to grayscale and applying blur to reduce noise
2. Using adaptive thresholding to handle different drawing styles
3. Finding contours in the binary image
4. Focusing on the largest contour (primary shape)
5. Applying morphological operations to clean up the image

### Shape Feature Extraction

A rich set of features is extracted for each shape:
- **Area and Perimeter**: Basic size characteristics
- **Circularity**: How close the shape is to a perfect circle
- **Aspect Ratio**: Width to height ratio of the bounding box
- **Convexity**: Ratio of contour area to its convex hull area
- **Hu Moments**: Shape descriptors invariant to rotation, scale, and translation
- **Vertex Approximation**: Multiple levels of polygon approximation
- **Minimum Enclosing Shapes**: How well the shape fits into primitive shapes
- **Corner Analysis**: Distribution of internal angles
- **Symmetry Metrics**: Vertical and horizontal reflection similarity
- **Side Length Analysis**: Variation in polygon side lengths

### Machine Learning Pipeline

The ML component uses a pipeline with:
1. **StandardScaler**: Normalizes features to have zero mean and unit variance
2. **SelectKBest**: Selects the most discriminative features using ANOVA F-statistics
3. **LinearSVC**: A linear Support Vector Classifier optimized for multi-class classification

The model is continually improved through incremental training from user feedback.

### Vertex Estimation Algorithm

For certain applications, the system estimates the vertices of shapes:
1. Determining shape type based on geometric properties
2. Applying shape-specific vertex extraction methods:
   - For rectangles/squares: Using minimum area rectangle
   - For triangles: Using contour approximation or minimum enclosing triangle
   - For ellipses: Finding major and minor axis endpoints

## File Structure and Functionality

### Main Files

- **`main.py`**: The application entry point that initializes all components and handles error management.
- **`config.py`**: Contains application-wide settings and configurable parameters.
- **`error_handler.py`**: Manages error handling and logging throughout the application.

### Core Modules

- **`core/recognition.py`**: Implements the shape recognition system with geometric and ML approaches.
- **`core/drawing_manager.py`**: Manages drawing operations and interactions with the recognition system.
- **`core/conversation.py`**: Handles the AI assistant's responses and interactions.
- **`core/tts.py`**: Provides text-to-speech capabilities for the AI assistant.
- **`core/arduino_controller.py`**: Interfaces with Arduino hardware for alternative input.

### UI Components

- **`ui/main_window.py`**: Implements the main application window.
- **`ui/canvas.py`**: Provides the drawing canvas functionality.
- **`ui/toolbar.py`**: Implements the drawing tool selection interface.
- **`ui/settings_dialog.py`**: Creates the settings configuration interface.
- **`ui/training_dialog.py`**: Manages the training data interface.

### Data Management

- **`data/database.py`**: Handles database operations for storing settings and training data.
- **`data/user_settings.py`**: Manages user preferences.
- **`data/history.py`**: Tracks drawing history.

### Utility Files

- **`setup_app.py`**: Performs initial setup for the application.
- **`train_model.py`**: Command-line utility for training the recognition model.
- **`import_hds_dataset.py`**: Utility for importing datasets for training.

### Arduino Components

- **`arduino/arduino_controller.ino`**: Arduino sketch for hardware integration.
- **`arduino_input.py`**: Python interface for Arduino communication.

## Training and Recognition Workflow

The application's shape recognition follows this workflow:

1. **Drawing Capture**: The user draws a shape on the canvas.
2. **Image Processing**: The drawing is converted to a format suitable for analysis.
3. **Contour Extraction**: The system identifies the contours in the drawing.
4. **Feature Extraction**: A comprehensive set of features is extracted from the contours.
5. **Dual Analysis**:
   - Geometric analysis scores the shape against known geometric patterns
   - Machine learning classifier (if trained) provides probability scores
6. **Combined Decision**: The results are combined with weighting to produce a ranked list of guesses.
7. **Feedback Loop**: User confirms or corrects the recognition, which can be saved for training.
8. **Incremental Training**: The system can retrain in the background to incorporate new examples.

## Conclusion

Hand-Drawn Shapes Recognition demonstrates an effective hybrid approach to shape recognition, combining traditional computer vision techniques with machine learning. This dual approach allows the system to have strong baseline performance via geometric rules while continuously improving through learning from user feedback.

The interactive nature of the application makes it both a useful tool and an educational platform for understanding shape recognition techniques. The system's modular architecture also makes it extensible for future enhancements or adaptation to specific recognition tasks.