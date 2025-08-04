#!/usr/bin/env python3
"""
ABR Training Demo Script

Quick demonstration of the complete training pipeline.
This script shows how to run a fast training session for testing.

Usage:
    python run_training_demo.py

Author: AI Assistant
Date: January 2025
"""

import os
import sys
import subprocess
import logging
from pathlib import Path


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def check_requirements():
    """Check if required files exist."""
    logger = logging.getLogger(__name__)
    
    required_files = [
        "configs/config.yaml",
        "data/processed/ultimate_dataset_with_clinical_thresholds.pkl",
        "train.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("Missing required files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    return True


def run_fast_training():
    """Run a fast training demo."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ABR Training Demo")
    logger.info("=" * 50)
    
    # Training command with fast development settings
    cmd = [
        sys.executable, "train.py",
        "--config", "configs/config.yaml",
        "--experiment", "demo_training",
        "--epochs", "3",
        "--batch_size", "8",
        "--fast_dev_run",
        "--log_level", "INFO"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        logger.info("Training completed successfully!")
        logger.info("Training output:")
        logger.info(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        logger.error("Error output:")
        logger.error(e.stderr)
        return False


def run_evaluation_demo():
    """Run evaluation demo if training succeeded."""
    logger = logging.getLogger(__name__)
    
    # Check if model exists
    model_path = "checkpoints/demo_training/latest_model.pt"
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}, skipping evaluation demo")
        return
    
    logger.info("Running evaluation demo...")
    
    cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint", model_path,
        "--output_dir", "outputs/demo_evaluation",
        "--log_level", "INFO"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Evaluation completed successfully!")
        logger.info("Evaluation output:")
        logger.info(result.stdout[-1000:])  # Last 1000 characters
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(e.stderr)


def run_inference_demo():
    """Run inference demo."""
    logger = logging.getLogger(__name__)
    
    # Check if model exists
    model_path = "checkpoints/demo_training/latest_model.pt"
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}, skipping inference demo")
        return
    
    logger.info("Running inference demo...")
    
    cmd = [
        sys.executable, "inference.py",
        "--checkpoint", model_path,
        "--age", "35",
        "--intensity", "80",
        "--rate", "30",
        "--fmp", "0.8",
        "--output_dir", "outputs/demo_inference",
        "--confidence"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Inference completed successfully!")
        logger.info("Inference output:")
        logger.info(result.stdout[-500:])  # Last 500 characters
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference failed: {e}")
        logger.error(e.stderr)


def main():
    """Main demo function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("ABR Project Training Pipeline Demo")
    logger.info("=" * 60)
    
    # Check requirements
    if not check_requirements():
        logger.error("Please ensure all required files are present")
        sys.exit(1)
    
    # Run training demo
    logger.info("Step 1: Training Demo")
    if run_fast_training():
        logger.info("✓ Training demo completed successfully")
        
        # Run evaluation demo
        logger.info("\nStep 2: Evaluation Demo")
        run_evaluation_demo()
        logger.info("✓ Evaluation demo completed")
        
        # Run inference demo
        logger.info("\nStep 3: Inference Demo")
        run_inference_demo()
        logger.info("✓ Inference demo completed")
        
        logger.info("\n" + "=" * 60)
        logger.info("Demo completed successfully!")
        logger.info("Check the following directories for outputs:")
        logger.info("  - checkpoints/demo_training/")
        logger.info("  - outputs/demo_evaluation/")
        logger.info("  - outputs/demo_inference/")
        logger.info("  - logs/demo_training/")
        
    else:
        logger.error("Training demo failed")
        sys.exit(1)


if __name__ == "__main__":
    main()