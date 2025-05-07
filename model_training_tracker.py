#!/usr/bin/env python3
"""
Script to count the number of training examples for each shape
in the training_shapes directory
"""

import os
import json
from pathlib import Path
from collections import Counter
import pickle
import sys
import time

def count_shapes_from_files():
    """
    Count shapes based on training file names
    """
    # Get all JSON and PNG files
    json_files = list(Path("training_shapes").glob("*.json"))
    png_files = list(Path("training_shapes").glob("*.png"))
    
    print(f"Found {len(json_files)} JSON files and {len(png_files)} PNG files")
    
    # Count based on JSON files first (these should have features)
    shape_counter = Counter()
    error_files = []
    
    print("Analyzing files...")
    start_time = time.time()
    processed = 0
    
    # Process JSON files
    for json_path in json_files:
        try:
            # Extract the label from the filename
            # Format: training_LABEL_TIMESTAMP.json
            parts = json_path.stem.split('_')
            if len(parts) < 2:
                error_files.append(str(json_path))
                continue
                
            label = parts[1]
            shape_counter[label] += 1
            
            processed += 1
            if processed % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {processed}/{len(json_files)} files ({processed/len(json_files)*100:.1f}%) in {elapsed:.1f} seconds")
                
        except Exception as e:
            error_files.append(str(json_path))
            print(f"Error processing {json_path}: {e}")
    
    # Check for PNG files without corresponding JSON
    png_only_shapes = Counter()
    for png_path in png_files:
        json_path = png_path.with_suffix('.json')
        if not json_path.exists():
            try:
                parts = png_path.stem.split('_')
                if len(parts) < 2:
                    continue
                    
                label = parts[1]
                png_only_shapes[label] += 1
            except Exception:
                pass
    
    elapsed = time.time() - start_time
    print(f"\nProcessed {len(json_files)} files in {elapsed:.1f} seconds")
    
    return shape_counter, png_only_shapes, error_files

def count_shapes_from_model():
    """
    Try to get shape information from the trained model
    """
    model_path = "models/shape_classifier.pkl"
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        if 'model' not in model_data:
            print("Model data doesn't contain a 'model' key")
            return None
            
        model = model_data['model']
        
        # Check if this is a pipeline
        if hasattr(model, 'steps'):
            # Find the classifier in the pipeline
            classifier = None
            for name, estimator in model.steps:
                if hasattr(estimator, 'classes_'):
                    classifier = estimator
                    break
            
            if classifier is None:
                print("Couldn't find a classifier with classes_ in the pipeline")
                return None
        else:
            # Direct classifier
            classifier = model
        
        # Check if classifier has classes_
        if not hasattr(classifier, 'classes_'):
            print("Classifier doesn't have classes_ attribute")
            return None
            
        # Get the classes
        classes = classifier.classes_
        
        # Try to get class weights or sample counts if available
        class_weights = {}
        
        if hasattr(classifier, 'class_weight_'):
            class_weights = {cls: weight for cls, weight in zip(classes, classifier.class_weight_)}
        
        return {'classes': classes, 'weights': class_weights}
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def display_results(shape_counts, png_only_counts, model_info):
    """
    Display shape counts in a formatted table
    """
    # Combine counts
    total_counts = shape_counts.copy()
    for label, count in png_only_counts.items():
        if label in total_counts:
            total_counts[label] += count
        else:
            total_counts[label] = count
    
    # Sort by count (highest first)
    sorted_shapes = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get list of shapes from model if available
    model_shapes = []
    if model_info and 'classes' in model_info:
        model_shapes = model_info['classes']
    
    # Calculate total for percentage
    total_samples = sum(total_counts.values())
    
    # Display table header
    print("\n=== Shape Training Data Distribution ===\n")
    print(f"{'Shape':<20} {'Count':<10} {'Percentage':<10} {'With JSON':<10} {'PNG Only':<10} {'In Model':<10}")
    print("-" * 70)
    
    # Display each shape with its count
    for shape, count in sorted_shapes:
        json_count = shape_counts.get(shape, 0)
        png_only = png_only_counts.get(shape, 0)
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        in_model = "Yes" if shape in model_shapes else "No"
        
        print(f"{shape:<20} {count:<10} {percentage:<10.2f} {json_count:<10} {png_only:<10} {in_model:<10}")
    
    # Display summary
    print("-" * 70)
    print(f"Total Shapes: {len(sorted_shapes)}")
    print(f"Total Training Samples: {total_samples}")
    
    # Show model shapes not in training data
    if model_info and 'classes' in model_info:
        missing_in_training = [shape for shape in model_shapes if shape not in total_counts]
        if missing_in_training:
            print("\nShapes in model but not in training data:")
            for shape in missing_in_training:
                print(f"  - {shape}")

def main():
    print("Counting shapes in training data...")
    
    # Count shapes from training files
    shape_counts, png_only_counts, error_files = count_shapes_from_files()
    
    # Get information from model if available
    model_info = count_shapes_from_model()
    
    # Display results
    display_results(shape_counts, png_only_counts, model_info)
    
    # Report errors if any
    if error_files:
        print(f"\nEncountered {len(error_files)} errors while processing files.")
        if len(error_files) <= 10:
            print("Error files:")
            for file in error_files:
                print(f"  - {file}")
        else:
            print(f"First 10 error files (out of {len(error_files)}):")
            for file in error_files[:10]:
                print(f"  - {file}")

if __name__ == "__main__":
    main()