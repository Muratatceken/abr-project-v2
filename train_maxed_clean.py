#!/usr/bin/env python3
"""
Clean training script for maxed out L4 GPU configuration.
Suppresses CUDA library registration warnings.
"""

import os
import sys

# Suppress CUDA/cuDNN/cuBLAS warnings BEFORE importing torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Don't block CUDA operations
os.environ['PYTHONWARNINGS'] = 'ignore'   # Suppress Python warnings

# Set multiprocessing start method to avoid CUDA factory conflicts
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Now run the actual training
if __name__ == "__main__":
    # Add the project root to the path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import and run the main training script
    from train import main
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Clean ABR Training - No CUDA Warnings")
    parser.add_argument("--config", type=str, 
                       default="configs/config_sequential_training_maxed_clean.yaml",
                       help="Config file path")
    
    args = parser.parse_args()
    
    # Override sys.argv to pass the config to main()
    sys.argv = ["train.py", "--config", args.config]
    
    print("🚀 Starting clean maxed out training (no CUDA warnings)...")
    print(f"📁 Config: {args.config}")
    print("💫 Expected: ~1-2 min/epoch, ~25-50 min total")
    
    # Run training
    main()