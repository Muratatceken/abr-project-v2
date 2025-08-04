#!/usr/bin/env python3
"""
Suppress CUDA and TensorFlow warnings before training

This script sets up the environment to suppress common CUDA library
registration warnings that don't affect training performance.
"""

import os
import sys
import warnings
import logging

def suppress_cuda_warnings():
    """Suppress CUDA and TensorFlow warnings."""
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Suppress specific CUDA warnings
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Set logging level to reduce noise
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    
    # Try to import and configure TensorFlow if present
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
    except ImportError:
        pass  # TensorFlow not installed
    
    print("✅ CUDA warnings suppressed")

if __name__ == "__main__":
    suppress_cuda_warnings()
    
    # Import and run the training script
    if len(sys.argv) > 1:
        import subprocess
        cmd = ['python'] + sys.argv[1:]
        subprocess.run(cmd)
    else:
        print("Usage: python suppress_warnings.py train.py [args...]")