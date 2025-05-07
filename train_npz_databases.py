#!/usr/bin/env python3
"""
Script to import and train NPZ shape files

This is a convenience wrapper around import_npz_shapes.py that automatically
runs the import process followed by training the model.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("npz_training.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def import_and_train_npz(npz_dir="training_set", output_dir="training_shapes", 
                        samples_per_shape=100, image_size=256, shapes=None,
                        verbose=False, test_mode=False):
    """
    Import NPZ files and train the model
    
    Args:
        npz_dir: Directory containing NPZ files
        output_dir: Directory to save training data
        samples_per_shape: Number of samples to generate per shape
        image_size: Size of the generated images
        shapes: List of shape names to process (if None, process all)
        verbose: Enable verbose output
        test_mode: Process only a few samples for testing
        
    Returns:
        bool: True if successful
    """
    try:
        print("\n" + "="*80)
        print(f"NPZ SHAPE IMPORTER AND TRAINER")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Settings:")
        print(f"  NPZ directory:     {npz_dir}")
        print(f"  Output directory:  {output_dir}")
        print(f"  Samples per shape: {samples_per_shape if samples_per_shape > 0 else 'ALL'}")
        print(f"  Image size:        {image_size}px")
        if shapes:
            print(f"  Shapes to process: {', '.join(shapes)}")
        print(f"  Verbose mode:      {verbose}")
        print(f"  Test mode:         {test_mode}")
        print("="*80 + "\n")
        
        if test_mode and samples_per_shape != 0:
            samples_per_shape = 3
            print("TEST MODE ENABLED: Processing only 3 samples per shape")
        
        # Build the command to run import_npz_shapes.py with training
        cmd = [
            sys.executable,  # Current Python interpreter
            "import_npz_shapes.py",
            f"--npz_dir={npz_dir}",
            f"--output_dir={output_dir}",
            f"--samples={samples_per_shape}",
            f"--size={image_size}",
            "--train"  # Always train after importing
        ]
        
        if shapes:
            cmd.extend(["--shapes"] + shapes)
            
        if verbose:
            cmd.append("--verbose")
            
        if test_mode:
            cmd.append("--test")
            
        # Log the command
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the import_npz_shapes.py script as a subprocess
        print("\nRunning NPZ import and training process...")
        start_time = datetime.now()
        
        result = subprocess.run(cmd, text=True, capture_output=False)
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        
        if result.returncode == 0:
            print("\n" + "="*80)
            print("IMPORT AND TRAINING COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Process completed in: {elapsed_str}")
            
            # Extract shape information from the recognized_shapes list in recognition.py
            print("\nChecking core/recognition.py for recognized shapes...")
            try:
                with open("core/recognition.py", "r") as f:
                    content = f.read()
                    
                # Look for the recognized_shapes list
                import re
                shapes_match = re.search(r"'recognized_shapes':\s*\[(.*?)\]", content, re.DOTALL)
                if shapes_match:
                    shapes_str = shapes_match.group(1)
                    current_shapes = [s.strip().strip("'\"") for s in shapes_str.split(',') if s.strip()]
                    print(f"Current recognized shapes: {', '.join(current_shapes)}")
                    
                    # Check if our shapes are already in the list
                    if shapes:
                        missing_shapes = [s for s in shapes if s not in current_shapes]
                        if missing_shapes:
                            print("\nIMPORTANT: The following shapes need to be added to recognition.py:")
                            for shape in missing_shapes:
                                print(f"  '{shape}',")
                        else:
                            print("\nAll requested shapes are already in the recognized_shapes list.")
            except Exception as e:
                logger.error(f"Error checking recognition.py: {e}")
                
            return True
        else:
            print("\n" + "="*80)
            print("IMPORT AND TRAINING FAILED")
            print("="*80)
            print(f"Process failed after: {elapsed_str}")
            print("Check the logs for details.")
            return False
        
    except Exception as e:
        logger.exception(f"Error during import and training: {e}")
        print(f"\nError: {e}")
        print("See npz_training.log for details.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Import NPZ files and train model')
    parser.add_argument('--npz_dir', type=str, default="training_set",
                      help='Directory containing NPZ files (default: training_set)')
    parser.add_argument('--output_dir', type=str, default="training_shapes",
                      help='Directory to save training data (default: training_shapes)')
    parser.add_argument('--samples', type=int, default=100,
                      help='Number of samples per shape (default: 100, use 0 for all samples)')
    parser.add_argument('--size', type=int, default=256,
                      help='Image size (default: 256)')
    parser.add_argument('--shapes', type=str, nargs='+',
                      help='Specific shapes to process (e.g., airplane bush)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--test', action='store_true',
                      help='Test mode: process only a few samples')
    
    args = parser.parse_args()
    
    # Run the import and training process
    success = import_and_train_npz(
        args.npz_dir,
        args.output_dir,
        args.samples,
        args.size,
        args.shapes,
        args.verbose,
        args.test
    )
    
    if success:
        print("\nImport and training completed successfully!")
        
        # Provide instructions for next steps
        print("\nNext steps:")
        print("1. Make sure any new shapes are added to the recognized_shapes list in core/recognition.py")
        print("2. Add geometric detector methods for new shapes in core/recognition.py")
        print("3. Restart the application to use the new shapes")
    else:
        print("\nImport and training failed. Check the logs for details.")
        
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()