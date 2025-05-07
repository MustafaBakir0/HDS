"""
Advanced shape recognition module for Drawing AI App
Combines OpenCV contour analysis with machine learning for accurate shape detection
"""
import datetime
import os
import numpy as np
from PIL import Image
import random
import math
import time
import json
import threading
from pathlib import Path
import pickle
from collections import Counter
import logging
import multiprocessing
from functools import partial

# Fix for numpy versions
if not hasattr(np, 'int0'):
    np.int0 = np.int64

# Import OpenCV for shape detection
try:
    import cv2
    CV2_AVAILABLE = True
    print("opencv available for shape detection")
except ImportError:
    CV2_AVAILABLE = False
    print("opencv not installed. run: pip install opencv-python")

# In recognition.py, near the beginning of the file

# Try to import ML libraries
try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_classif
    ML_AVAILABLE = True
    print("scikit-learn available for ML classification")
except ImportError:
    ML_AVAILABLE = False
    print("scikit-learn not installed. Install with: pip install scikit-learn")

import config

# Set up logging
logger = logging.getLogger(__name__)


class DrawingRecognizer:
    """Class for recognizing drawings using combined geometric and ML approaches"""

    def __init__(self, api_key=None):
        """Initialize the recognizer"""
        # Create models directory
        os.makedirs("models", exist_ok=True)
        os.makedirs("training_shapes", exist_ok=True)
        
        # Cache directory for preprocessed features
        self.cache_dir = "feature_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Enable debugging to see what's happening
        self.debug = False
        self.debug_dir = "debug_images"
        if self.debug and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        # Shape recognition components
        self.feature_extractor = ShapeFeatureExtractor()
        self.geometric_analyzer = GeometricAnalyzer()
        self.classifier = ShapeClassifier()

        # Load settings
        self.load_settings()

        # Background training flag
        self.training_queued = False
        
        # Determine number of CPU cores for parallel processing
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free

    def load_settings(self):
        """Load recognition settings"""
        # Default settings
        self.settings = {
            'confidence_threshold': 50,
            'enable_ml': True,
            'enable_incremental_training': True,
            'recognized_shapes': [
                'triangle', 'ellipse', 'rectangle', 'square', 'cross', 
                'other', 'hexagon', 'line', 'airplane', 'bush'
            ]
        }

        # Try to load from database or config
        # Implementation would depend on how settings are stored

    def save_settings(self):
        """Save recognition settings"""
        # Implementation would depend on how settings are stored
        pass

    def recognize(self, image):
        """
        Recognize the content of a drawing

        Args:
            image: QPixmap with the drawing

        Returns:
            dict: Recognition results with top guesses and confidence scores
        """
        # Convert the QPixmap to image
        pil_image = self._prepare_image(image)

        # If OpenCV is available, use shape detection
        if CV2_AVAILABLE:
            try:
                return self._detect_shape(pil_image)
            except Exception as e:
                print(f"Error in shape detection: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to mock recognition if there's an error
                return self._mock_recognition()
        else:
            print("Using mock recognition (OpenCV not available)")
            # Otherwise, use mock recognition
            return self._mock_recognition()

    def _prepare_image(self, pixmap):
        """
        Convert QPixmap to PIL Image for processing

        Args:
            pixmap: QPixmap with the drawing

        Returns:
            PIL.Image: Image object
        """
        # Create a temporary file path
        temp_path = "temp_drawing.png"

        # Save the pixmap to a temporary file
        pixmap.save(temp_path, "PNG")

        # Open the image with PIL
        image = Image.open(temp_path)

        # Try to remove the temporary file
        try:
            os.remove(temp_path)
        except:
            pass  # Ignore errors in cleanup

        return image

    def _detect_shape(self, pil_image):
        """
        Detect shape using combined geometric and ML approaches

        Args:
            pil_image: PIL image with the drawing

        Returns:
            dict: Recognition results
        """
        # Convert PIL image to OpenCV format
        cv_image = np.array(pil_image)
        if cv_image.ndim == 3:  # Color image
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:  # Grayscale image
            gray = cv_image

        # Save original for debugging
        if self.debug:
            timestamp = int(time.time())
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_01_original.png", cv_image)
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_02_gray.png", gray)

        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive threshold to handle different lighting/drawing styles
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Optional: Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_03_binary.png", binary)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours for debugging
        if self.debug:
            contour_img = cv_image.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_04_contours.png", contour_img)

        # Filter small contours
        min_area = 500  # Adjust as needed
        contours = [c for c in contours if cv2.contourArea(c) > min_area]

        # If no contours found, return mock recognition
        if not contours:
            return self._mock_recognition()

        # Get the largest contour - usually the main drawing
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw the largest contour for debugging
        if self.debug:
            main_contour_img = cv_image.copy()
            cv2.drawContours(main_contour_img, [largest_contour], -1, (0, 0, 255), 2)
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_05_largest_contour.png", main_contour_img)

        # Extract features from the contour
        features = self.feature_extractor.extract_features(largest_contour, cv_image.shape[:2])
        
        # Estimate vertices from contour if we can
        estimated_vertices = self._estimate_vertices(largest_contour, features)
        if estimated_vertices:
            features['vertices'] = estimated_vertices
            
            # Add normalized vertices (0-1 range)
            img_height, img_width = cv_image.shape[:2]
            normalized_vertices = []
            for px, py in estimated_vertices:
                nx = px / img_width
                ny = py / img_height
                normalized_vertices.append([nx, ny])
            features['normalized_vertices'] = normalized_vertices

        # Use geometric analysis
        geometric_scores = self.geometric_analyzer.analyze(largest_contour, features)

        # Use ML classification if available and enabled
        ml_scores = {}
        if self.settings['enable_ml'] and self.classifier.is_trained and ML_AVAILABLE:
            ml_results = self.classifier.predict(features)
            if ml_results:
                ml_scores = ml_results

        # Combine scores (weighted average)
        combined_scores = {}
        for shape in self.settings['recognized_shapes']:
            geo_score = geometric_scores.get(shape, 0)
            ml_score = ml_scores.get(shape, 0)

            # If we have an ML score, weight it higher
            if shape in ml_scores and self.classifier.is_trained:
                combined_scores[shape] = (geo_score * 0.1 + ml_score * 0.9)
            else:
                combined_scores[shape] = geo_score

        # Sort by score and get top guesses
        sorted_shapes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Filter out zero scores
        sorted_shapes = [(shape, score) for shape, score in sorted_shapes if score > 0]

        # Get top 3 guesses
        guesses = []
        for shape, score in sorted_shapes[:3]:
            guesses.append({"name": shape, "confidence": int(score)})

        # If we have fewer than 3 guesses, add generic ones
        while len(guesses) < 3:
            guesses.append({"name": "unknown", "confidence": 1})

        # Normalize confidence scores to percentages
        total_score = sum(guess["confidence"] for guess in guesses)
        if total_score > 0:
            for guess in guesses:
                guess["confidence"] = int((guess["confidence"] / total_score) * 100)

        # Ensure they sum to 100%
        total = sum(guess["confidence"] for guess in guesses[:-1])
        guesses[-1]["confidence"] = max(1, 100 - total)

        # Determine if we're confident
        is_confident = guesses[0]["confidence"] >= self.settings['confidence_threshold']

        # Schedule incremental training if needed
        if self.settings['enable_incremental_training'] and not self.training_queued:
            self._schedule_background_training()

        # Create the result dictionary with vertex information if available
        result = {
            "success": True,
            "confidence_threshold": self.settings['confidence_threshold'],
            "guesses": guesses,
            "is_confident": is_confident,
            "features": features  # Include features for saving with training data
        }
        
        # Add vertices to the result if we have them
        if 'vertices' in features:
            result['vertices'] = features['vertices']
            
        return result
        
    def _estimate_vertices(self, contour, features):
        """
        Estimate the vertices of a shape based on its contour
        
        Args:
            contour: OpenCV contour
            features: Dictionary of extracted features
            
        Returns:
            list: Estimated vertices as [[x1, y1], [x2, y2], ...]
        """
        try:
            # Get the shape name from features or determine from features
            shape_type = None
            
            # Try to determine based on geometric properties
            circularity = features.get('circularity', 0)
            aspect_ratio = features.get('aspect_ratio', 1)
            vertices_count = features.get('vertices_3', 0)
            
            if circularity > 0.85:
                # For ellipses/circles
                shape_type = "ellipse"
            elif vertices_count == 3:
                # For triangles
                shape_type = "triangle"
            elif vertices_count == 4:
                # Could be rectangle or square
                if 0.9 < aspect_ratio < 1.1:
                    shape_type = "square"
                else:
                    shape_type = "rectangle"
            
            # Based on shape type, use different methods to extract vertices
            if shape_type == "rectangle" or shape_type == "square":
                # Get minimum area rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                vertices = np.int0(box).tolist()
                return vertices
                
            elif shape_type == "triangle":
                # Approximate contour with fewer points
                epsilon = 0.03 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 3:
                    return [point[0].tolist() for point in approx]
                else:
                    # If approximation didn't give triangle, use minimum enclosing triangle
                    _, triangle = cv2.minEnclosingTriangle(contour)
                    if triangle is not None:
                        return np.int0(triangle).reshape(-1, 2).tolist()
                
            elif shape_type == "ellipse":
                # For ellipses, get points along major and minor axes
                if len(contour) >= 5:  # Need at least 5 points for ellipse fitting
                    ellipse = cv2.fitEllipse(contour)
                    (center_x, center_y), (major_axis, minor_axis), angle = ellipse
                    
                    # Convert angle to radians
                    angle_rad = np.deg2rad(angle)
                    
                    # Calculate the four vertices (2 on major axis, 2 on minor axis)
                    vertices = []
                    
                    # Major axis points
                    x1 = center_x + major_axis/2 * np.cos(angle_rad)
                    y1 = center_y + major_axis/2 * np.sin(angle_rad)
                    x2 = center_x - major_axis/2 * np.cos(angle_rad)
                    y2 = center_y - major_axis/2 * np.sin(angle_rad)
                    
                    # Minor axis points
                    x3 = center_x + minor_axis/2 * np.cos(angle_rad + np.pi/2)
                    y3 = center_y + minor_axis/2 * np.sin(angle_rad + np.pi/2)
                    x4 = center_x - minor_axis/2 * np.cos(angle_rad + np.pi/2)
                    y4 = center_y - minor_axis/2 * np.sin(angle_rad + np.pi/2)
                    
                    vertices = [[int(x1), int(y1)], [int(x2), int(y2)], 
                               [int(x3), int(y3)], [int(x4), int(y4)]]
                    return vertices
            
            # If we don't have a specific shape or the specific extraction failed,
            # use a general approximation
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = [point[0].tolist() for point in approx]
            return vertices
            
        except Exception as e:
            print(f"Error estimating vertices: {e}")
            return None

    def _mock_recognition(self):
        """
        Generate mock recognition results for testing

        Returns:
            dict: Mock recognition results
        """
        # List of possible objects to recognize
        objects = [
            "circle", "rectangle", "square", "triangle", "star",
            "heart", "arrow", "diamond", "ellipse", "line"
        ]

        # Generate random results
        primary_guess = random.choice(objects)
        secondary_guesses = random.sample([obj for obj in objects if obj != primary_guess], 2)

        # Random confidence scores that sum to 100%
        primary_confidence = random.randint(60, 90)
        secondary_conf_1 = random.randint(5, (100 - primary_confidence) - 5)
        secondary_conf_2 = 100 - primary_confidence - secondary_conf_1

        results = {
            "success": True,
            "confidence_threshold": self.settings['confidence_threshold'],
            "guesses": [
                {"name": primary_guess, "confidence": primary_confidence},
                {"name": secondary_guesses[0], "confidence": secondary_conf_1},
                {"name": secondary_guesses[1], "confidence": secondary_conf_2}
            ],
            "is_confident": primary_confidence >= self.settings['confidence_threshold'],
            "features": {}  # Empty features for mock recognition
        }

        return results

    def save_for_training(self, image, label, recognition_results=None):
        """
        Save an image for training

        Args:
            image: QPixmap with the drawing
            label: Shape label (string)
            recognition_results: Optional recognition results containing features

        Returns:
            bool: Success
        """
        try:
            # Generate a filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_{label}_{timestamp}.png"

            # Save path
            filepath = os.path.join("training_shapes", filename)

            # Save the image
            image.save(filepath)

            # If we have recognition results with features, save them too
            if recognition_results and 'features' in recognition_results:
                features = recognition_results['features']
                features_file = os.path.join("training_shapes", f"{os.path.splitext(filename)[0]}.json")

                with open(features_file, 'w') as f:
                    json.dump(features, f)

            # Schedule training
            if self.settings['enable_incremental_training']:
                self._schedule_background_training()

            return True
        except Exception as e:
            print(f"Error saving for training: {e}")
            return False

    def _schedule_background_training(self):
        """Schedule background training"""
        # Only schedule if not already queued
        if not self.training_queued:
            self.training_queued = True
            threading.Thread(target=self._perform_background_training, daemon=True).start()

    def _perform_background_training(self):
        """Perform background training"""
        try:
            # Load training data
            training_data = self._load_training_data()

            # Train the model
            if training_data:
                self.classifier.train(training_data, background=False)
        except Exception as e:
            print(f"Error in background training: {e}")
        finally:
            self.training_queued = False

    def _load_training_data(self):
        """
        Load training data from training_shapes directory
        Uses parallel processing and caching for better performance

        Returns:
            list: List of (features, label) tuples
        """
        # Check if we already have cached feature data first
        training_data = []
        
        # Get all JSON files and images
        json_files = list(Path("training_shapes").glob("*.json"))
        png_files = list(Path("training_shapes").glob("*.png"))
        
        # Create a map of images to labels
        image_label_pairs = []
        
        print(f"Found {len(json_files)} JSON files and {len(png_files)} PNG files")
        
        # First try to load from JSON files (pre-extracted features)
        json_loaded = 0
        for json_path in json_files:
            try:
                # Get the corresponding image
                img_path = json_path.with_suffix('.png')
                
                if not img_path.exists():
                    continue
                
                # Extract the label from the filename
                # Format: training_LABEL_TIMESTAMP.json
                parts = json_path.stem.split('_')
                if len(parts) < 2:
                    continue
                
                label = parts[1]
                
                # Check cache first
                cache_path = os.path.join(self.cache_dir, f"{json_path.stem}.pkl")
                features = None
                
                if os.path.exists(cache_path):
                    # Load from cache
                    try:
                        with open(cache_path, 'rb') as f:
                            features = pickle.load(f)
                        json_loaded += 1
                    except:
                        # If cache loading fails, load from JSON
                        features = None
                
                # If not in cache, load from JSON
                if features is None:
                    with open(json_path, 'r') as f:
                        features = json.load(f)
                    
                    # Save to cache for next time
                    with open(cache_path, 'wb') as f:
                        pickle.dump(features, f)
                    json_loaded += 1
                
                # Add to training data
                training_data.append((features, label))
                
                # Remove this image from our list to process
                if img_path in png_files:
                    png_files.remove(img_path)
                    
            except Exception as e:
                print(f"Error loading training data from {json_path}: {e}")
        
        print(f"Loaded {json_loaded} samples from JSON files")
        
        # For remaining PNG files, extract features in parallel
        remaining_images = []
        for img_path in png_files:
            # Extract the label from the filename
            # Format: training_LABEL_TIMESTAMP.png
            parts = img_path.stem.split('_')
            if len(parts) < 2:
                continue
                
            label = parts[1]
            image_label_pairs.append((str(img_path), label))
        
        if image_label_pairs:
            print(f"Extracting features for {len(image_label_pairs)} additional images...")
            # Process remaining images in parallel
            additional_data = self._parallel_extract_features(image_label_pairs)
            training_data.extend(additional_data)
            
        print(f"Total training samples loaded: {len(training_data)}")
        return training_data

    def update_settings(self, new_settings):
        """
        Update recognition settings

        Args:
            new_settings: Dictionary of new settings

        Returns:
            bool: Success
        """
        # Update settings
        for key, value in new_settings.items():
            if key in self.settings:
                self.settings[key] = value

        # Save settings
        self.save_settings()

        return True
        
    def _extract_features_for_training(self, cv_image):
        """
        Extract features from an image for training purposes
        
        Args:
            cv_image: OpenCV image
            
        Returns:
            dict: Dictionary of features, or None if extraction failed
        """
        try:
            # Convert to grayscale if needed
            if cv_image.ndim == 3:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv_image
                
            # Preprocess the image (similar to _detect_shape method)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter small contours
            min_area = 500
            contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            # If no contours found, return None
            if not contours:
                return None
                
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Extract features
            features = self.feature_extractor.extract_features(largest_contour, cv_image.shape[:2])
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _parallel_extract_features(self, image_paths_labels, batch_size=10):
        """
        Extract features from multiple images in parallel
        
        Args:
            image_paths_labels: List of (image_path, label) tuples
            batch_size: Number of images to process in each batch
            
        Returns:
            list: List of (features, label) tuples
        """
        print(f"Extracting features from {len(image_paths_labels)} images using {self.num_workers} workers")
        
        # Process images in batches to avoid memory issues
        results = []
        total_images = len(image_paths_labels)
        
        for i in range(0, total_images, batch_size):
            batch = image_paths_labels[i:i+batch_size]
            
            # Create a process pool
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                # Define the worker function
                def process_image(img_path_label):
                    img_path, label = img_path_label
                    try:
                        # Check if we have a cached version
                        cache_path = os.path.join(self.cache_dir, f"{Path(img_path).stem}.pkl")
                        
                        if os.path.exists(cache_path):
                            # Load from cache
                            with open(cache_path, 'rb') as f:
                                features = pickle.load(f)
                        else:
                            # Load and process the image
                            image = Image.open(img_path)
                            cv_image = np.array(image)
                            if cv_image.ndim == 3:
                                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                            
                            # Extract features
                            features = self._extract_features_for_training(cv_image)
                            
                            # Cache the results
                            if features:
                                with open(cache_path, 'wb') as f:
                                    pickle.dump(features, f)
                        
                        return (features, label) if features else None
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        return None
                
                # Process the batch in parallel
                batch_results = pool.map(process_image, batch)
                
                # Filter out None results and extend our list
                valid_results = [r for r in batch_results if r is not None]
                results.extend(valid_results)
                
                print(f"Processed {min(i+batch_size, total_images)}/{total_images} images ({len(valid_results)}/{len(batch)} successful in this batch)")
        
        print(f"Feature extraction complete: {len(results)}/{total_images} successful")
        return results
    
    def import_dataset(self, dataset_dir, target_shape=None, limit=None):
        """
        Import training images from a dataset directory with shape subdirectories
        
        Args:
            dataset_dir: Path to dataset directory with structure {dataset_dir}/{shapeName}/*.jpg
                         where shapeName can be: "angleCross", "ellipse", "hexagon", "line", 
                         "square", "straightCross", and "triangle"
            target_shape: Optional shape to filter and import
            limit: Optional limit on number of samples per shape
                         
        Returns:
            int: Number of imported samples
        """
        import os
        from PIL import Image
        from pathlib import Path
        from datetime import datetime
        
        # Shape name mapping: maps dataset directory names to recognized shape names
        shape_mapping = {
            "angleCross": "cross",     # Map both cross types to "cross"
            "straightCross": "cross",
            "ellipse": "ellipse",      # Keep these the same
            "hexagon": "hexagon",
            "line": "line",
            "square": "square",
            "triangle": "triangle",
            "rectangle": "rectangle"   # Added for HDS dataset
        }
        
        # Create training_shapes directory if it doesn't exist
        os.makedirs("training_shapes", exist_ok=True)
        
        # Track import counts
        import_count = 0
        
        # Process each shape subdirectory
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            print(f"Dataset directory not found: {dataset_dir}")
            return import_count
            
        for shape_dir in dataset_path.iterdir():
            if not shape_dir.is_dir():
                continue
                
            # Get shape name from directory
            shape_name = shape_dir.name
            
            # Skip if not in our supported shapes
            if shape_name not in shape_mapping:
                print(f"Skipping unsupported shape category: {shape_name}")
                continue
                
            # Map to recognized shape name
            recognized_shape = shape_mapping[shape_name]
            
            # Skip if we're filtering for a specific shape
            if target_shape and recognized_shape != target_shape:
                continue
                
            print(f"Processing shape: {recognized_shape} from directory {shape_name}")
            
            # Process all jpg files in the directory (with optional limit)
            jpg_files = list(shape_dir.glob("*.jpg"))
            if limit:
                jpg_files = jpg_files[:limit]
                
            for img_path in jpg_files:
                try:
                    # Load the image
                    image = Image.open(img_path)
                    
                    # Generate a filename for saving
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"training_{recognized_shape}_{timestamp}.png"
                    
                    # Save path
                    filepath = os.path.join("training_shapes", filename)
                    
                    # Save the image as PNG
                    image.save(filepath, "PNG")
                    
                    # Extract features if OpenCV is available
                    if CV2_AVAILABLE:
                        cv_image = np.array(image)
                        if cv_image.ndim == 3:  # Color image
                            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                            
                        # Process similar to _detect_shape method
                        features = self._extract_features_for_training(cv_image)
                        
                        # Save features to JSON
                        if features:
                            features_file = os.path.join("training_shapes", f"{os.path.splitext(filename)[0]}.json")
                            with open(features_file, 'w') as f:
                                json.dump(features, f)
                    
                    # Increment counter
                    import_count += 1
                    if import_count % 10 == 0:
                        print(f"Imported {import_count} images so far...")
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    
        print(f"Total images imported: {import_count}")
        
        # If we imported any images, schedule training
        if import_count > 0 and self.settings['enable_incremental_training']:
            self._schedule_background_training()
        
        return import_count
    
    def import_hds_dataset(self, dataset_dir, target_shape=None, limit=None):
        """
        Import training images from Hand-drawn Shapes (HDS) Dataset with vertices
        
        Args:
            dataset_dir: Path to dataset directory with structure:
                         - {dataset_dir}/user.xxx/images/{shapeName}/*.png (images)
                         - {dataset_dir}/user.xxx/vertices/{shapeName}/*.csv (vertex data)
            target_shape: Optional shape to filter and import
            limit: Optional limit on number of samples per shape (per user directory)
                         
        Returns:
            int: Number of imported samples
        """
        import os
        import csv
        import numpy as np
        from PIL import Image
        from pathlib import Path
        from datetime import datetime
        
        # Shape name mapping for HDS dataset (specifically for the 4 shape categories)
        shape_mapping = {
            "rectangle": "rectangle",
            "ellipse": "ellipse",
            "triangle": "triangle",
            "other": "other"     # "other" is a catch-all category in HDS
        }
        
        # Create training_shapes directory if it doesn't exist
        os.makedirs("training_shapes", exist_ok=True)
        
        # Also create a directory for vertex data
        os.makedirs("training_vertices", exist_ok=True)
        
        # Track import counts
        import_count = 0
        total_files_found = 0
        
        # Process the dataset
        dataset_path = Path(dataset_dir)
        print(f"Processing HDS dataset at path: {dataset_path}")
        if not dataset_path.exists():
            print(f"Dataset directory not found: {dataset_dir}")
            return import_count
            
        # Check for user directories (user.xxx)
        print(f"Looking for user directories in {dataset_path}")
        user_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("user")]
        print(f"Found {len(user_dirs)} user directories: {[d.name for d in user_dirs]}")
        
        if not user_dirs:
            print(f"No user directories found in {dataset_path}")
            # Try to look for images and vertices in the main directory
            images_dir = dataset_path / "images"
            vertices_dir = dataset_path / "vertices"
            
            # Check the main directory structure
            if images_dir.exists() and vertices_dir.exists():
                print(f"Found direct images and vertices directories in {dataset_path}")
                user_dirs = [dataset_path]  # Treat the main directory as a user directory
            else:
                print(f"No valid dataset structure found in {dataset_path}")
                return import_count
        
        # Process each user directory
        for user_dir in user_dirs:
            print(f"Processing user directory: {user_dir}")
            user_id = user_dir.name  # Extract user ID (e.g., "user.abc" -> "user.abc")
            
            # Check for images and vertices directories in this user directory
            images_dir = user_dir / "images"
            vertices_dir = user_dir / "vertices"
            
            print(f"Checking for images directory: {images_dir} (exists: {images_dir.exists()})")
            print(f"Checking for vertices directory: {vertices_dir} (exists: {vertices_dir.exists()})")
            
            if not images_dir.exists() or not vertices_dir.exists():
                print(f"Invalid HDS dataset structure in {user_dir}. Missing images or vertices directory.")
                continue
                
            print(f"Both images and vertices directories found in {user_dir}. Continuing import...")
                
            # Process each shape subdirectory
            print(f"Looking for shape subdirectories in: {images_dir}")
            shape_dirs = list(images_dir.iterdir())
            print(f"Found {len(shape_dirs)} items in images directory")
            
            for shape_dir in shape_dirs:
                print(f"Checking item: {shape_dir} (is directory: {shape_dir.is_dir()})")
                if not shape_dir.is_dir():
                    print(f"Skipping non-directory item: {shape_dir}")
                    continue
                    
                # Get shape name from directory
                shape_name = shape_dir.name.lower()
                print(f"Processing shape directory: {shape_name}")
                
                # Skip if not in our supported shapes
                if shape_name not in shape_mapping:
                    print(f"Skipping unsupported shape category: {shape_name}")
                    continue
                    
                # Map to recognized shape name
                recognized_shape = shape_mapping[shape_name]
                
                # Skip if we're filtering for a specific shape
                if target_shape and recognized_shape != target_shape:
                    print(f"Skipping {shape_name} due to target_shape filter: {target_shape}")
                    continue
                    
                print(f"Processing shape: {recognized_shape} from directory {shape_name}")
                
                # Get corresponding vertices directory
                vertices_shape_dir = vertices_dir / shape_name
                print(f"Looking for vertices directory: {vertices_shape_dir} (exists: {vertices_shape_dir.exists()})")
                if not vertices_shape_dir.exists():
                    print(f"Warning: Vertices directory not found for {shape_name}")
                
                # Process all png files in the directory (with optional limit)
                png_files = list(shape_dir.glob("*.png"))
                total_files_found += len(png_files)
                print(f"Found {len(png_files)} PNG files in {shape_dir}")
                
                # Apply limit if specified
                shape_files = png_files
                if limit:
                    shape_files = png_files[:limit]
                    print(f"Limited to {len(shape_files)} files due to limit parameter")
                
                for img_path in shape_files:
                    try:
                        # Extract the base filename (e.g., ellipse.aly.0001)
                        base_name = img_path.stem
                        
                        # Look for corresponding vertex file in vertices directory
                        vertex_path = vertices_shape_dir / f"{base_name}.csv"
                        vertices = []
                        
                        # Read vertex data if available
                        if vertex_path.exists():
                            try:
                                with open(vertex_path, 'r') as csvfile:
                                    csv_reader = csv.reader(csvfile)
                                    for row in csv_reader:
                                        # Parse vertex coordinates (normalize from 0-1 to pixel coordinates)
                                        if len(row) >= 2:
                                            try:
                                                x = float(row[0])  # x-coordinate (0-1)
                                                y = float(row[1])  # y-coordinate (0-1)
                                                vertices.append([x, y])
                                            except ValueError:
                                                # Skip header or invalid rows
                                                continue
                            except Exception as e:
                                print(f"Error reading vertex data from {vertex_path}: {e}")
                        else:
                            print(f"Warning: No vertex file found for {base_name}")
                        
                        # Load the image
                        image = Image.open(img_path)
                        
                        # Generate a filename for saving that includes user ID
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"training_{recognized_shape}_{timestamp}_{user_id}_{base_name}.png"
                        
                        # Save path
                        filepath = os.path.join("training_shapes", filename)
                        
                        # Save the image as PNG
                        image.save(filepath, "PNG")
                        
                        # Extract features if OpenCV is available
                        if CV2_AVAILABLE:
                            cv_image = np.array(image)
                            if cv_image.ndim == 3:  # Color image
                                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                            
                            # Get image dimensions for vertex conversion
                            img_height, img_width = cv_image.shape[:2]
                            
                            # Convert normalized vertex coordinates to pixel coordinates
                            pixel_vertices = []
                            for x, y in vertices:
                                px = int(x * img_width)
                                py = int(y * img_height)
                                pixel_vertices.append([px, py])
                            
                            # Process similar to _detect_shape method
                            features = self._extract_features_for_training(cv_image)
                            
                            # Add vertices to features
                            if features and pixel_vertices:
                                features['vertices'] = pixel_vertices
                                
                                # Also store normalized vertices for consistent training
                                features['normalized_vertices'] = vertices
                                
                                # Add user and file information
                                features['user_id'] = user_id
                                features['original_filename'] = base_name
                            
                            # Save features to JSON
                            if features:
                                features_file = os.path.join("training_shapes", f"{os.path.splitext(filename)[0]}.json")
                                with open(features_file, 'w') as f:
                                    json.dump(features, f)
                        
                        # Increment counter
                        import_count += 1
                        if import_count % 100 == 0:
                            print(f"Imported {import_count} images so far...")
                    
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        import traceback
                        traceback.print_exc()
        
        print(f"Total files found: {total_files_found}")
        print(f"Total images successfully imported: {import_count}")
        
        # If we imported any images, schedule training
        if import_count > 0 and self.settings['enable_incremental_training']:
            self._schedule_background_training()
        
        return import_count


class ShapeFeatureExtractor:
    """Extract features from shape contours for machine learning"""

    def __init__(self):
        """Initialize the feature extractor"""
        self.debug = False

    def extract_features(self, contour, img_shape):
        """
        Extract a comprehensive set of shape features

        Args:
            contour: OpenCV contour of the shape
            img_shape: Tuple of image dimensions (height, width)

        Returns:
            dict: Dictionary of features
        """
        features = {}

        # Basic metrics
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        img_area = img_shape[0] * img_shape[1]

        # Shape properties
        features['area_ratio'] = area / img_area if img_area > 0 else 0
        features['circularity'] = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        features['aspect_ratio'] = w / h if h > 0 else 0
        features['extent'] = area / (w * h) if (w * h) > 0 else 0

        # Convex hull features
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        features['convexity'] = area / hull_area if hull_area > 0 else 0
        features['solidity'] = area / hull_area if hull_area > 0 else 0

        # Moments and center of mass
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w//2, y + h//2

        # Center of mass offset from geometric center
        geom_center_x, geom_center_y = x + w//2, y + h//2
        features['center_offset'] = math.sqrt((cx - geom_center_x)**2 + (cy - geom_center_y)**2) / math.sqrt(w**2 + h**2)

        # Hu Moments (scale, rotation and translation invariant)
        hu_moments = cv2.HuMoments(M)
        for i, hu in enumerate(hu_moments.flatten()):
            # Log transform to handle small values
            if hu != 0:
                features[f'hu_moment_{i}'] = -np.sign(hu) * np.log10(abs(hu))
            else:
                features[f'hu_moment_{i}'] = 0

        # Contour approximation at different thresholds
        epsilons = [0.01, 0.02, 0.03, 0.05]
        for eps in epsilons:
            epsilon = eps * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            features[f'vertices_{int(eps*100)}'] = len(approx)

        # Minimum enclosing shapes
        # Circle
        (x_circle, y_circle), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius**2
        features['circle_fit'] = area / circle_area if circle_area > 0 else 0

        # Rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        rect_area = cv2.contourArea(box)
        features['rect_fit'] = area / rect_area if rect_area > 0 else 0

        # Triangle
        triangle = cv2.minEnclosingTriangle(contour)[1]
        if triangle is not None:
            triangle_area = cv2.contourArea(triangle)
            features['triangle_fit'] = area / triangle_area if triangle_area > 0 else 0
        else:
            features['triangle_fit'] = 0

        # Convexity defects
        try:
            hull_idx = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull_idx)
            if defects is not None:
                defect_depths = []
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    defect_depths.append(d / 256.0)

                features['defect_count'] = len(defects)
                features['max_defect_depth'] = max(defect_depths) if defect_depths else 0
                features['avg_defect_depth'] = np.mean(defect_depths) if defect_depths else 0
            else:
                features['defect_count'] = 0
                features['max_defect_depth'] = 0
                features['avg_defect_depth'] = 0
        except:
            features['defect_count'] = 0
            features['max_defect_depth'] = 0
            features['avg_defect_depth'] = 0

        # Calculate corner angles
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        angles = self._calculate_corner_angles(approx)
        if angles:
            features['min_angle'] = min(angles)
            features['max_angle'] = max(angles)
            features['avg_angle'] = sum(angles) / len(angles)
            features['angle_std'] = np.std(angles) if len(angles) > 1 else 0

            # Check for right angles (90° ± 15°)
            right_angles = sum(1 for angle in angles if 75 <= angle <= 105)
            features['right_angle_ratio'] = right_angles / len(angles) if len(angles) > 0 else 0
        else:
            features['min_angle'] = 0
            features['max_angle'] = 0
            features['avg_angle'] = 0
            features['angle_std'] = 0
            features['right_angle_ratio'] = 0

        # Check for equal sides
        side_lengths = self._get_side_lengths(approx)
        if side_lengths:
            features['min_side'] = min(side_lengths)
            features['max_side'] = max(side_lengths)
            features['side_ratio'] = min(side_lengths) / max(side_lengths) if max(side_lengths) > 0 else 0
            features['side_std'] = np.std(side_lengths) / np.mean(side_lengths) if np.mean(side_lengths) > 0 else 0
        else:
            features['min_side'] = 0
            features['max_side'] = 0
            features['side_ratio'] = 0
            features['side_std'] = 0

        # Symmetry metrics
        features['vertical_symmetry'] = self._calculate_symmetry(contour, 'vertical', img_shape)
        features['horizontal_symmetry'] = self._calculate_symmetry(contour, 'horizontal', img_shape)

        # Calculate top/bottom area ratio (useful for heart shapes)
        top_area, bottom_area = self._get_vertical_half_areas(contour, y, h)
        features['top_bottom_ratio'] = top_area / (top_area + bottom_area) if (top_area + bottom_area) > 0 else 0.5

        # Create feature vector (converting dict to list in a consistent order)
        return features

    def _calculate_corner_angles(self, approx):
        """Calculate corner angles in degrees"""
        if len(approx) < 3:
            return []

        angles = []
        for i in range(len(approx)):
            prev_i = (i - 1) % len(approx)
            next_i = (i + 1) % len(approx)

            p1 = approx[prev_i][0]
            p2 = approx[i][0]
            p3 = approx[next_i][0]

            # Vectors
            v1 = [p1[0] - p2[0], p1[1] - p2[1]]
            v2 = [p3[0] - p2[0], p3[1] - p2[1]]

            # Dot product
            dot = v1[0] * v2[0] + v1[1] * v2[1]

            # Magnitudes
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

            # Angle in degrees
            if mag1 * mag2 == 0:
                continue

            cos_angle = dot / (mag1 * mag2)
            # Handle numerical issues
            cos_angle = min(1.0, max(-1.0, cos_angle))
            angle_rad = math.acos(cos_angle)
            angle_deg = math.degrees(angle_rad)

            angles.append(angle_deg)

        return angles

    def _get_side_lengths(self, approx):
        """Get the lengths of all sides in a polygon"""
        if len(approx) < 3:
            return []

        # Calculate side lengths
        sides = []
        for i in range(len(approx)):
            next_i = (i + 1) % len(approx)
            p1 = approx[i][0]
            p2 = approx[next_i][0]
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            sides.append(distance)

        return sides

    def _calculate_symmetry(self, contour, axis='vertical', img_shape=None):
        """
        Calculate symmetry score along vertical or horizontal axis

        Args:
            contour: OpenCV contour
            axis: 'vertical' or 'horizontal'
            img_shape: (height, width) of the image

        Returns:
            float: Symmetry score (0-1, higher is more symmetric)
        """
        # Create a blank mask
        if img_shape is None:
            x, y, w, h = cv2.boundingRect(contour)
            img_shape = (y+h+10, x+w+10)

        mask = np.zeros(img_shape, dtype=np.uint8)

        # Draw the contour on the mask
        cv2.drawContours(mask, [contour], -1, 255, -1)

        if axis == 'vertical':
            # Flip horizontally and compare
            flip_axis = 1  # Horizontal flip
            center = mask.shape[1] // 2
        else:
            # Flip vertically and compare
            flip_axis = 0  # Vertical flip
            center = mask.shape[0] // 2

        # Flip the mask
        flipped = cv2.flip(mask, flip_axis)

        # Calculate the intersection and union for IoU metric
        intersection = cv2.bitwise_and(mask, flipped)
        union = cv2.bitwise_or(mask, flipped)

        # IoU as symmetry score
        if np.sum(union) > 0:
            symmetry = np.sum(intersection) / np.sum(union)
        else:
            symmetry = 0

        return symmetry

    def _get_vertical_half_areas(self, contour, y, h):
        """Get areas of top and bottom halves of shape"""
        # Create a mask for the contour
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros((y+h+10, x+w+10), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Split into top and bottom halves
        mid_y = y + h // 2
        top_half = mask[y:mid_y, x:x+w]
        bottom_half = mask[mid_y:y+h, x:x+w]

        # Calculate areas
        top_area = np.sum(top_half > 0)
        bottom_area = np.sum(bottom_half > 0)

        return top_area, bottom_area


class ShapeClassifier:
    """Machine learning classifier for shapes"""

    def __init__(self, model_path=None):
        """
        Initialize the shape classifier

        Args:
            model_path: Path to saved model
        """
        self.model = None
        self.feature_extractor = ShapeFeatureExtractor()
        self.model_path = model_path or os.path.join("models", "shape_classifier.pkl")
        self.feature_names = None
        self.training_in_progress = False
        self.is_trained = False

        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Try to load existing model
        self.load_model()

    def load_model(self):
        """Load model from disk if it exists"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.feature_names = model_data.get('feature_names', None)
                    self.is_trained = True
                    print(f"Loaded shape classifier from {self.model_path}")
                    return True
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None

        return False

    def save_model(self):
        """Save model to disk"""
        if self.model is not None:
            try:
                model_data = {
                    'model': self.model,
                    'feature_names': self.feature_names
                }
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                print(f"Saved shape classifier to {self.model_path}")
                return True
            except Exception as e:
                print(f"Error saving model: {e}")

        return False

    def train(self, shapes_data, background=False):
        """
        Train the model on labeled shapes

        Args:
            shapes_data: List of (features_dict, label) tuples
            background: Whether to train in background thread

        Returns:
            bool: True if training started
        """
        if not ML_AVAILABLE:
            print("scikit-learn not available, cannot train model")
            return False

        if self.training_in_progress:
            print("Training already in progress")
            return False

        if background:
            threading.Thread(target=self._train_model, args=(shapes_data,), daemon=True).start()
            return True
        else:
            return self._train_model(shapes_data)

    def _train_model(self, shapes_data):
        """Internal method to train the model"""
        try:
            self.training_in_progress = True

            if not shapes_data:
                print("No training data provided")
                self.training_in_progress = False
                return False

            print(f"Training shape classifier on {len(shapes_data)} samples")

            # Extract features and labels
            X_dicts = [item[0] for item in shapes_data]
            y = [item[1] for item in shapes_data]

            # Check if we have enough samples
            if len(set(y)) < 2:
                print("Need samples from at least 2 classes for training")
                self.training_in_progress = False
                return False

            # Preprocess feature dictionaries to remove nested arrays and non-numerical values
            cleaned_X_dicts = []
            for feature_dict in X_dicts:
                # Create a clean copy without vertex data, non-numerical values, or other nested arrays
                clean_dict = {}
                for key, value in feature_dict.items():
                    # Skip vertex data, user_id, filename, and any other non-numerical features
                    if (key in ['vertices', 'normalized_vertices', 'user_id', 'original_filename'] or 
                        (isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], (list, tuple))) or
                        not isinstance(value, (int, float, bool))):
                        continue
                    clean_dict[key] = value
                cleaned_X_dicts.append(clean_dict)
            
            # Convert to DataFrame or structured array
            # First ensure all feature dictionaries have same keys
            all_features = set()
            for features in cleaned_X_dicts:
                all_features.update(features.keys())

            # Create a list of feature vectors
            X = []
            for features in cleaned_X_dicts:
                feature_vector = [features.get(feat, 0) for feat in all_features]
                X.append(feature_vector)

            # Store feature names for prediction
            self.feature_names = list(all_features)
            print(f"Using {len(self.feature_names)} features for training (excluded vertex data)")

            # Create and train model with feature selection and linear SVM
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(f_classif, k=min(50, len(self.feature_names)))),
                ('classifier', LinearSVC(dual=False, C=1.0, max_iter=5000))
            ])

            # If we have enough data, use train/test split
            if len(X) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)

                pipeline.fit(X_train, y_train)

                # Evaluate model
                train_accuracy = pipeline.score(X_train, y_train)
                test_accuracy = pipeline.score(X_test, y_test)

                print(f"Model training complete. Train accuracy: {train_accuracy:.2f}, Test accuracy: {test_accuracy:.2f}")
            else:
                # Just use all data for training if we have very few samples
                pipeline.fit(X, y)
                accuracy = pipeline.score(X, y)
                print(f"Model training complete. Accuracy on all data: {accuracy:.2f}")

            # Save the model
            self.model = pipeline
            self.save_model()
            self.is_trained = True

            return True

        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.training_in_progress = False

    # In the ShapeClassifier class in recognition.py

    def predict(self, features_dict):
        """
        Predict shape class from features

        Args:
            features_dict: Dictionary of shape features

        Returns:
            dict: Prediction results with class probabilities
        """
        # Check if scikit-learn is available
        if not ML_AVAILABLE:
            # Return a fallback with empty predictions if scikit-learn is not available
            logger.warning("scikit-learn not available, using geometric analysis only")
            return {}

        if not self.is_trained or self.model is None or not self.feature_names:
            return None

        try:
            # Convert features dict to vector, using only features the model was trained with
            feature_vector = []
            for feat in self.feature_names:
                # Use get with default 0.0 to handle missing features
                feature_vector.append(features_dict.get(feat, 0.0))

            # Get class names
            classes = self.model.classes_
            
            # For LinearSVC, we don't have direct probability outputs
            # We'll use decision function to get confidence scores
            decision_values = self.model.decision_function([feature_vector])
            
            # Convert decision values to pseudo-probabilities using softmax
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            
            # Apply softmax if we have multiple classes, otherwise manual normalization
            if len(classes) > 2:
                probs = softmax(decision_values[0])
            else:
                # For binary classification, convert single score to probability
                score = decision_values[0]
                pos_prob = 1 / (1 + np.exp(-score))  # sigmoid function
                probs = np.array([1 - pos_prob, pos_prob])
            
            # Create results
            results = {}
            for i, cls in enumerate(classes):
                results[cls] = float(probs[i])

            return results

        except Exception as e:
            logger.error(f"Error predicting: {e}")
            return None


class GeometricAnalyzer:
    """Analyzer for geometric shape detection using traditional computer vision"""

    def __init__(self):
        """Initialize the geometric analyzer"""
        self.shape_detectors = {
            'circle': self._detect_circle,
            'ellipse': self._detect_ellipse,
            'triangle': self._detect_triangle,
            'rectangle': self._detect_rectangle,
            'square': self._detect_square,
            'pentagon': self._detect_pentagon,
            'hexagon': self._detect_hexagon,
            'diamond': self._detect_diamond,
            'line': self._detect_line,
            'arrow': self._detect_arrow,
            'star': self._detect_star,
            'heart': self._detect_heart,
            'cross': self._detect_cross,
            'smiley': self._detect_smiley,
            'airplane': self._detect_airplane,
            'bush': self._detect_bush
        }

    def analyze(self, contour, features):
        """
        Analyze a contour using geometric methods

        Args:
            contour: OpenCV contour of the shape
            features: Dictionary of extracted features

        Returns:
            dict: Shape scores
        """
        # Initialize scores
        scores = {shape: 0 for shape in self.shape_detectors.keys()}

        # Run each detector
        for shape, detector in self.shape_detectors.items():
            score = detector(contour, features)
            scores[shape] = score

        return scores

    def _detect_circle(self, contour, features):
        """Detect if the shape is a circle"""
        score = 0

        # Circularity is a key metric for circles
        circularity = features['circularity']
        if circularity > 0.9:
            score += 85
        elif circularity > 0.85:
            score += 70
        elif circularity > 0.8:
            score += 50

        # Circle fit
        circle_fit = features['circle_fit']
        if circle_fit > 0.9:
            score += 15
        elif circle_fit > 0.85:
            score += 10

        # Aspect ratio should be close to 1
        aspect_ratio = features['aspect_ratio']
        if 0.9 < aspect_ratio < 1.1:
            score += 10
        elif 0.8 < aspect_ratio < 1.2:
            score += 5

        # Symmetric shapes
        if features['vertical_symmetry'] > 0.9 and features['horizontal_symmetry'] > 0.9:
            score += 10

        return min(100, score)

    def _detect_ellipse(self, contour, features):
        """Detect if the shape is an ellipse/oval"""
        score = 0

        # High circularity but not a perfect circle
        circularity = features['circularity']
        if 0.8 < circularity < 0.9:
            score += 70
        elif 0.7 < circularity < 0.95:
            score += 50

        # Aspect ratio not close to 1 (to differentiate from circle)
        aspect_ratio = features['aspect_ratio']
        if aspect_ratio < 0.8 or aspect_ratio > 1.2:
            score += 20

        # Symmetry
        if features['vertical_symmetry'] > 0.85 and features['horizontal_symmetry'] > 0.85:
            score += 10

        # Convexity
        if features['convexity'] > 0.95:
            score += 10

        # Circle fit not too high (to differentiate from circle)
        if features['circle_fit'] < 0.95 and features['circle_fit'] > 0.8:
            score += 10

        return min(100, score)

    def _detect_triangle(self, contour, features):
        """Detect if the shape is a triangle"""
        score = 0

        # Vertex count is key for triangles
        if features['vertices_3'] == 3:
            score += 70
        elif features['vertices_5'] == 3:
            score += 50

        # Triangle fit
        if features['triangle_fit'] > 0.9:
            score += 20
        elif features['triangle_fit'] > 0.8:
            score += 10

        # For equilateral triangles, sides should be equal
        if features['side_std'] < 0.1:
            score += 10

        return min(100, score)

    def _detect_rectangle(self, contour, features):
        """Detect if the shape is a rectangle"""
        score = 0

        # 4 vertices is key
        if features['vertices_3'] == 4:
            score += 50
        elif features['vertices_5'] == 4:
            score += 40

        # Right angles
        if features['right_angle_ratio'] > 0.9:
            score += 30
        elif features['right_angle_ratio'] > 0.7:
            score += 20

        # Aspect ratio not close to 1 (to differentiate from square)
        aspect_ratio = features['aspect_ratio']
        if aspect_ratio < 0.85 or aspect_ratio > 1.15:
            score += 20

        # Rectangle fit
        if features['rect_fit'] > 0.95:
            score += 10

        return min(100, score)

    def _detect_square(self, contour, features):
        """Detect if the shape is a square"""
        score = 0

        # 4 vertices is key
        if features['vertices_3'] == 4:
            score += 50
        elif features['vertices_5'] == 4:
            score += 40

        # Right angles
        if features['right_angle_ratio'] > 0.9:
            score += 20
        elif features['right_angle_ratio'] > 0.7:
            score += 10

        # Aspect ratio close to 1
        aspect_ratio = features['aspect_ratio']
        if 0.9 < aspect_ratio < 1.1:
            score += 30
        elif 0.85 < aspect_ratio < 1.15:
            score += 20

        # Equal sides
        if features['side_std'] < 0.1:
            score += 20
        elif features['side_std'] < 0.2:
            score += 10

        return min(100, score)

    def _detect_pentagon(self, contour, features):
        """Detect if the shape is a pentagon"""
        score = 0

        # 5 vertices is key
        if features['vertices_3'] == 5:
            score += 70
        elif features['vertices_5'] == 5 or features['vertices_2'] == 5:
            score += 50

        # Regular pentagon has similar sides
        if features['side_std'] < 0.15:
            score += 20
        elif features['side_std'] < 0.25:
            score += 10

        # Convexity
        if features['convexity'] > 0.9:
            score += 10

        return min(100, score)

    def _detect_hexagon(self, contour, features):
        """Detect if the shape is a hexagon"""
        score = 0

        # 6 vertices is key
        if features['vertices_3'] == 6:
            score += 70
        elif features['vertices_5'] == 6 or features['vertices_2'] == 6:
            score += 50

        # Regular hexagon has similar sides
        if features['side_std'] < 0.15:
            score += 20
        elif features['side_std'] < 0.25:
            score += 10

        # Convexity
        if features['convexity'] > 0.9:
            score += 10

        return min(100, score)

    def _detect_diamond(self, contour, features):
        """Detect if the shape is a diamond (rhombus)"""
        score = 0

        # 4 vertices is key
        if features['vertices_3'] == 4:
            score += 50
        elif features['vertices_5'] == 4:
            score += 40

        # Not having right angles (to differentiate from square/rectangle)
        if features['right_angle_ratio'] < 0.5:
            score += 20

        # Equal sides
        if features['side_std'] < 0.15:
            score += 30
        elif features['side_std'] < 0.25:
            score += 20

        # Aspect ratio not too extreme
        aspect_ratio = features['aspect_ratio']
        if 0.7 < aspect_ratio < 1.3:
            score += 10

        return min(100, score)

    def _detect_line(self, contour, features):
        """Detect if the shape is a line"""
        score = 0

        # Low vertex count
        if features['vertices_1'] <= 2:
            score += 50
        elif features['vertices_3'] <= 2:
            score += 30

        # Extreme aspect ratio
        aspect_ratio = features['aspect_ratio']
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            score += 50
        elif aspect_ratio > 5 or aspect_ratio < 0.2:
            score += 30

        return min(100, score)

    def _detect_arrow(self, contour, features):
        """Detect if the shape is an arrow"""
        score = 0

        # Arrows typically have 5-7 vertices
        if 5 <= features['vertices_3'] <= 7:
            score += 40
        elif 5 <= features['vertices_5'] <= 7:
            score += 30

        # Arrows have medium convexity due to the arrowhead
        if 0.7 < features['convexity'] < 0.9:
            score += 20

        # Arrows typically have an asymmetric center of mass
        if features['center_offset'] > 0.1:
            score += 20

        # Elongated shape
        aspect_ratio = features['aspect_ratio']
        if aspect_ratio > 2 or aspect_ratio < 0.5:
            score += 20

        return min(100, score)

    def _detect_star(self, contour, features):
        """Detect if the shape is a star"""
        score = 0

        # Stars typically have many convexity defects
        if features['defect_count'] >= 5:
            score += 50
        elif features['defect_count'] >= 3:
            score += 30

        # Stars have low convexity
        if features['convexity'] < 0.7:
            score += 30
        elif features['convexity'] < 0.8:
            score += 20

        # Stars are typically symmetric
        if features['vertical_symmetry'] > 0.8 and features['horizontal_symmetry'] > 0.8:
            score += 20

        return min(100, score)

    def _detect_heart(self, contour, features):
        """Detect if the shape is a heart"""
        score = 0

        # Hearts have characteristic top/bottom ratio
        if 0.55 < features['top_bottom_ratio'] < 0.7:
            score += 50
        elif 0.5 < features['top_bottom_ratio'] < 0.75:
            score += 30

        # Hearts have low convexity due to the top dip
        if 0.75 < features['convexity'] < 0.9:
            score += 30
        elif 0.7 < features['convexity'] < 0.95:
            score += 20

        # Hearts typically have vertical symmetry but not horizontal
        if features['vertical_symmetry'] > 0.85 and features['horizontal_symmetry'] < 0.8:
            score += 20

        return min(100, score)

    def _detect_cross(self, contour, features):
        """Detect if the shape is a cross"""
        score = 0

        # Crosses typically have 8-12 vertices
        if 8 <= features['vertices_2'] <= 12:
            score += 40
        elif 8 <= features['vertices_3'] <= 12:
            score += 30

        # Crosses have low convexity
        if features['convexity'] < 0.7:
            score += 30
        elif features['convexity'] < 0.8:
            score += 20

        # Crosses typically have both vertical and horizontal symmetry
        if features['vertical_symmetry'] > 0.85 and features['horizontal_symmetry'] > 0.85:
            score += 30

        return min(100, score)

    def _detect_smiley(self, contour, features):
        """Detect if the shape is a smiley face"""
        score = 0

        # Smileys are circular
        if features['circularity'] > 0.85:
            score += 30
        elif features['circularity'] > 0.8:
            score += 20

        # But they have lower convexity than a perfect circle due to the smile
        if 0.8 < features['convexity'] < 0.95:
            score += 30
        elif 0.75 < features['convexity'] < 0.97:
            score += 20

        # Smileys have a few significant convexity defects (eyes and mouth)
        if 2 <= features['defect_count'] <= 4:
            score += 40

        return min(100, score)
        
    def _detect_airplane(self, contour, features):
        """Detect if the shape is an airplane"""
        score = 0
        
        # Airplanes typically have elongated shape
        aspect_ratio = features['aspect_ratio']
        if 2.0 < aspect_ratio < 4.0:
            score += 30
        elif 1.5 < aspect_ratio < 5.0:
            score += 20
            
        # Airplanes have multiple convexity defects (wings, tail)
        if 4 <= features['defect_count'] <= 8:
            score += 30
        elif 2 <= features['defect_count'] <= 10:
            score += 20
            
        # Airplanes have medium to low convexity due to wings
        if 0.6 < features['convexity'] < 0.8:
            score += 20
        elif 0.5 < features['convexity'] < 0.9:
            score += 10
            
        # Airplanes typically have 6-10 vertices for main features
        if 6 <= features['vertices_3'] <= 10:
            score += 20
        elif 5 <= features['vertices_5'] <= 12:
            score += 10
            
        return min(100, score)
        
    def _detect_bush(self, contour, features):
        """Detect if the shape is a bush"""
        score = 0
        
        # Bushes are somewhat circular but irregular
        circularity = features['circularity']
        if 0.6 < circularity < 0.8:
            score += 30
        elif 0.5 < circularity < 0.85:
            score += 20
            
        # Bushes have rough edges with many convexity defects
        if features['defect_count'] >= 8:
            score += 30
        elif features['defect_count'] >= 5:
            score += 20
            
        # Bushes often have many vertices
        if features['vertices_2'] >= 10:
            score += 20
        elif features['vertices_5'] >= 6:
            score += 10
            
        # Bushes typically have low convexity
        if 0.5 < features['convexity'] < 0.7:
            score += 20
        elif 0.4 < features['convexity'] < 0.8:
            score += 10
            
        # Aspect ratio for bushes is often close to 1 (somewhat round)
        aspect_ratio = features['aspect_ratio']
        if 0.8 < aspect_ratio < 1.2:
            score += 10
            
        return min(100, score)