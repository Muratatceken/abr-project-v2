#!/usr/bin/env python3
"""
ABR Hierarchical U-Net Training Script

Professional training pipeline for ABR signal generation using
hierarchical U-Net with multi-task learning and diffusion training.

Usage:
    python train.py --config configs/config.yaml --experiment my_experiment
    python train.py --config configs/config.yaml --resume checkpoints/latest_model.pt
    python train.py --config configs/config.yaml --experiment ablation --batch_size 64

Author: AI Assistant
Date: January 2025
"""

# CRITICAL: Suppress CUDA/TensorFlow warnings BEFORE any imports
import os
import warnings
import sys

# Suppress TensorFlow/CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress Python warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Redirect stderr to suppress low-level CUDA messages
import io
import contextlib

class SuppressOutput:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self._original_stderr

import argparse
import logging
import random
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from omegaconf import DictConfig
from training.config_loader import load_config, save_config, create_experiment_config, print_config
from training.trainer import ABRTrainer
from data.dataset import ABRDataset, stratified_patient_split, create_optimized_dataloaders
from models.hierarchical_unet import OptimizedHierarchicalUNet


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Reduce noise from external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(config: DictConfig, device: torch.device) -> OptimizedHierarchicalUNet:
    """Create and initialize the ABR model."""
    model_config = config.model.architecture
    
    model = OptimizedHierarchicalUNet(
        signal_length=model_config.signal_length,
        static_dim=model_config.static_dim,
        base_channels=model_config.base_channels,
        n_levels=model_config.n_levels,
        num_classes=model_config.n_classes,  # Correct parameter name
        dropout=model_config.dropout,
        num_s4_layers=model_config.encoder.n_s4_layers,  # Correct parameter name
        s4_state_size=model_config.encoder.d_state,      # Correct parameter name
        num_transformer_layers=model_config.decoder.n_transformer_layers,  # Correct parameter name
        num_heads=model_config.decoder.n_heads  # Correct parameter name
    )
    
    model.to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
    
    model.apply(init_weights)
    
    return model


def create_data_loaders(config: DictConfig, device: torch.device) -> tuple:
    """Create train, validation, and test data loaders."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading dataset from: {config.data.dataset_path}")
    
    # Use the optimized dataloaders function
    train_loader, val_loader, test_loader, full_dataset = create_optimized_dataloaders(
        data_path=config.data.dataset_path,
        batch_size=config.data.dataloader.batch_size,
        train_ratio=config.data.splits.train_ratio,
        val_ratio=config.data.splits.val_ratio,
        test_ratio=config.data.splits.test_ratio,
        num_workers=config.data.dataloader.num_workers,
        pin_memory=config.data.dataloader.pin_memory,
        random_state=config.data.splits.random_seed
    )
    
    logger.info(f"Total samples: {len(full_dataset)}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="ABR Hierarchical U-Net Training")
    
    # Configuration arguments
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        default=None,
        help="Experiment name for organized logging"
    )
    
    # Training arguments
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=None,
        help="Learning rate (overrides config)"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Training device (cuda/cpu, overrides config)"
    )
    parser.add_argument(
        "--mixed_precision", 
        action="store_true",
        help="Enable mixed precision training"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=None,
        help="Number of data loader workers"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Reduce logging output"
    )
    
    # Development arguments
    parser.add_argument(
        "--fast_dev_run", 
        action="store_true",
        help="Run training for only a few steps for testing"
    )
    parser.add_argument(
        "--overfit_batch", 
        action="store_true",
        help="Overfit on a single batch for debugging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        log_level = "WARNING"
    else:
        log_level = args.log_level
    
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("ABR Hierarchical U-Net Training Pipeline")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        
        # Create configuration overrides from command line arguments
        overrides = {}
        if args.epochs is not None:
            overrides['training.epochs'] = args.epochs
        if args.batch_size is not None:
            overrides['data.dataloader.batch_size'] = args.batch_size
        if args.learning_rate is not None:
            overrides['training.optimizer.learning_rate'] = args.learning_rate
        if args.device is not None:
            overrides['hardware.device'] = args.device
        if args.mixed_precision:
            overrides['hardware.mixed_precision'] = True
        if args.num_workers is not None:
            overrides['data.dataloader.num_workers'] = args.num_workers
        
        # Fast development run
        if args.fast_dev_run:
            overrides.update({
                'training.epochs': 2,
                'data.dataloader.batch_size': 8,
                'training.checkpointing.save_frequency': 1,
                'training.validation.frequency': 1
            })
            logger.info("Fast development run enabled")
        
        # Load config with overrides
        if args.experiment:
            config = create_experiment_config(args.config, args.experiment, overrides)
        else:
            config = load_config(args.config, overrides)
        
        # Print configuration
        if not args.quiet:
            print_config(config, "Training Configuration")
        
        # Set reproducibility
        set_seed(config.reproducibility.seed)
        logger.info(f"Random seed set to: {config.reproducibility.seed}")
        
        # Setup device
        if torch.cuda.is_available():
            device = torch.device(config.hardware.device)
            logger.info(f"Using device: {device} ({torch.cuda.get_device_name()})")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            logger.info("CUDA not available, using CPU")
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config, device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(config, device)
        
        # Overfit on single batch for debugging
        if args.overfit_batch:
            logger.info("Overfitting on single batch for debugging")
            single_batch = next(iter(train_loader))
            train_loader = [single_batch] * 100  # Repeat single batch
            val_loader = [single_batch] * 10
            config.training.epochs = 50
        
        # Create trainer (sequential or standard)
        logger.info("Initializing trainer...")
        
        # Check if sequential training is enabled
        use_sequential = (hasattr(config, 'sequential_training') and 
                         config.sequential_training.get('enabled', False))
        
        if use_sequential:
            from training.sequential_trainer import SequentialTrainer
            trainer = SequentialTrainer(
                config=config,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device
            )
            logger.info("Sequential trainer initialized for curriculum learning")
        else:
            trainer = ABRTrainer(
                config=config,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device
            )
            logger.info("Standard trainer initialized")
        
        # Save training configuration
        config_save_path = Path(config.paths.checkpoint_dir) / "training_config.yaml"
        save_config(config, str(config_save_path))
        logger.info(f"Training configuration saved to: {config_save_path}")
        
        # Start training
        logger.info("Starting training...")
        trainer.train(resume_from=args.resume)
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method
    if torch.cuda.is_available():
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    main()