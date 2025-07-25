#!/usr/bin/env python3
"""
Production ABR Training Script
Runs full training for 150 epochs with comprehensive model saving and visualization.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.enhanced_train import main as train_main
from training.config_loader import ConfigLoader


def create_parser():
    """Create argument parser for production training."""
    parser = argparse.ArgumentParser(description='ABR Production Training')
    
    # Core training arguments
    parser.add_argument('--config', type=str, default='training/config_production.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    
    # Model saving arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/production',
                       help='Directory to save model checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--save-best', action='store_true', default=True,
                       help='Save best models based on validation metrics')
    
    # Visualization arguments
    parser.add_argument('--plot-dir', type=str, default='plots/production',
                       help='Directory to save training plots')
    parser.add_argument('--plot-every', type=int, default=5,
                       help='Create plots every N epochs')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plotting')
    
    # Logging arguments
    parser.add_argument('--log-dir', type=str, default='logs/production',
                       help='Directory for log files')
    parser.add_argument('--tensorboard-dir', type=str, default='runs/production',
                       help='TensorBoard log directory')
    parser.add_argument('--experiment-name', type=str, 
                       default=f'abr_production_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='Experiment name for logging')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--num-workers', type=int, default=6,
                       help='Number of data loader workers')
    
    # Training control
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation on existing model')
    
    return parser


def setup_directories(config: Dict[str, Any]):
    """Create necessary directories for training."""
    dirs_to_create = [
        config.get('checkpoint_dir', 'checkpoints/production'),
        config.get('plot_dir', 'plots/production'),
        config.get('log_dir', 'logs/production'),
        config.get('tensorboard_dir', 'runs/production'),
        config.get('output_dir', 'outputs/production')
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging for production training."""
    log_dir = config.get('log_dir', 'logs/production')
    experiment_name = config.get('experiment_name', 'abr_production')
    
    # Create log file path
    log_file = Path(log_dir) / f"{experiment_name}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('ABR_Production_Training')
    logger.info(f"Starting production training: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def setup_environment(config: Dict[str, Any]):
    """Setup training environment and optimizations."""
    # Set random seeds for reproducibility
    seed = config.get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # PyTorch optimizations
    if config.get('benchmark', True):
        torch.backends.cudnn.benchmark = True
        print("✓ Enabled CUDNN benchmark")
    
    if config.get('deterministic', False):
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        print("✓ Enabled deterministic mode")
    
    # Enable JIT fusion for faster training
    if hasattr(torch.jit, 'set_fusion_strategy'):
        torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
        print("✓ Enabled JIT fusion optimizations")


def apply_production_overrides(config: Dict[str, Any], args: argparse.Namespace):
    """Apply command line overrides to configuration."""
    # Core training parameters
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['num_workers'] = args.num_workers
    
    # Directory settings
    config['checkpoint_dir'] = args.checkpoint_dir
    config['plot_dir'] = args.plot_dir
    config['log_dir'] = args.log_dir
    config['tensorboard_dir'] = args.tensorboard_dir
    config['experiment_name'] = args.experiment_name
    
    # Saving settings
    config['save_every'] = args.save_every
    config['save_best_only'] = not args.save_best
    
    # Visualization settings
    if args.no_plots:
        config['visualization'] = {'enabled': False}
    else:
        config.setdefault('visualization', {})
        config['visualization']['enabled'] = True
        config['visualization']['save_plots'] = True
        config['visualization']['plot_dir'] = args.plot_dir
        config['visualization']['plot_every'] = args.plot_every
    
    # Resume training
    config['resume_from'] = args.resume
    config['validate_only'] = args.validate_only
    
    return config


def print_training_summary(config: Dict[str, Any]):
    """Print a comprehensive training configuration summary."""
    print("\n" + "=" * 80)
    print("🚀 ABR PRODUCTION TRAINING CONFIGURATION")
    print("=" * 80)
    
    # Core parameters
    print(f"Experiment: {config.get('experiment_name', 'N/A')}")
    print(f"Dataset: {config.get('data_path', 'N/A')}")
    print(f"Epochs: {config.get('num_epochs', 'N/A')}")
    print(f"Batch size: {config.get('batch_size', 'N/A')}")
    print(f"Learning rate: {config.get('learning_rate', 'N/A')}")
    print(f"Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}")
    print(f"Workers: {config.get('num_workers', 'N/A')}")
    
    # Model architecture
    model_config = config.get('model', {})
    print(f"Model channels: {model_config.get('base_channels', 'N/A')}")
    print(f"Model levels: {model_config.get('n_levels', 'N/A')}")
    print(f"Classes: {model_config.get('n_classes', 'N/A')}")
    
    # Training features
    print(f"Mixed precision: {config.get('use_amp', False)}")
    print(f"Early stopping patience: {config.get('patience', 'N/A')}")
    print(f"Curriculum learning: {config.get('curriculum', {}).get('enabled', False)}")
    
    # Validation
    val_config = config.get('validation', {})
    print(f"Fast validation: {val_config.get('fast_mode', False)}")
    print(f"Full validation every: {val_config.get('full_validation_every', 'N/A')} epochs")
    
    # Saving and visualization
    print(f"Save every: {config.get('save_every', 'N/A')} epochs")
    print(f"Visualization: {config.get('visualization', {}).get('enabled', False)}")
    print(f"Checkpoint dir: {config.get('checkpoint_dir', 'N/A')}")
    print(f"Plot dir: {config.get('plot_dir', 'N/A')}")
    
    print("=" * 80)


def main():
    """Main production training function."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration
    print(f"📋 Loading production configuration from: {args.config}")
    config_loader = ConfigLoader(args.config)
    config = config_loader.to_dict()
    
    # Apply command line overrides
    config = apply_production_overrides(config, args)
    
    # Setup environment
    setup_environment(config)
    
    # Setup directories
    setup_directories(config)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Print configuration summary
    print_training_summary(config)
    
    # Check device availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🔥 CUDA available: {gpu_count} GPU{'s' if gpu_count > 1 else ''}")
        print(f"  GPU 0: {gpu_name} ({gpu_memory:.0f} GB)")
        config['device'] = 'cuda'
    else:
        print("💻 Using CPU training")
        config['device'] = 'cpu'
    
    print("\n🎯 Starting Production ABR Training...")
    
    try:
        # Convert config dict to args-like object for compatibility
        class ConfigArgs:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        config_args = ConfigArgs(config)
        
        # Start training
        logger.info("Starting production training...")
        train_main(config_args)
        
        print("\n✅ Production training completed successfully!")
        logger.info("Production training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        logger.info("Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 