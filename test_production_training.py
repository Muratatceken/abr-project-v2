#!/usr/bin/env python3
"""
Test Production Training
Quick test script to run the production training for 100 epochs.
"""

import subprocess
import sys
import os

def main():
    """Run production training test."""
    print("🚀 Starting ABR Production Training Test")
    print("=" * 60)
    
    # Command to run production training
    cmd = [
        sys.executable, "run_production_training.py",
        "--epochs", "100",
        "--batch-size", "32", 
        "--learning-rate", "0.0001",
        "--save-every", "10",
        "--plot-every", "5",
        "--experiment-name", f"abr_production_100epochs_test"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Production training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 