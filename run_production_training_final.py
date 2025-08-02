#!/usr/bin/env python3
"""
Final Production ABR Training Script

This script configures and runs the final production training for the ABR model
with all optimizations, early stopping, comprehensive checkpointing, and monitoring.

Usage:
    python run_production_training_final.py                    # Default production training
    python run_production_training_final.py --resume latest    # Resume from latest checkpoint
    python run_production_training_final.py --wandb           # Enable Weights & Biases logging

Author: AI Assistant
Date: January 2025
"""

import argparse
import os
import sys
import time
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from training.config_loader import ConfigLoader
from training.enhanced_train import ABRTrainer, create_model
from data.dataset import create_optimized_dataloaders


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for production training."""
    parser = argparse.ArgumentParser(
        description='Final Production ABR Model Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='training/config_production_final.yaml',
                       help='Path to production configuration file')
    
    # Training control
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint (latest, best_loss, best_f1, or path)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    # Logging and monitoring
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Override experiment name')
    
    # Validation and testing
    parser.add_argument('--validate_only', action='store_true',
                       help='Only run validation without training')
    parser.add_argument('--test_run', action='store_true',
                       help='Quick test run with 3 epochs')
    
    # Resource management
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Override number of data loading workers')
    
    return parser


def setup_environment():
    """Setup the training environment and optimizations."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Memory optimizations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable TensorFloat-32 for better performance on RTX 30xx+ cards
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, device):
    """Load model checkpoint and restore training state."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler state
    if 'scaler_state_dict' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Get resume information
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    best_val_f1 = checkpoint.get('best_val_f1', 0.0)
    
    print(f"✓ Resumed from epoch {start_epoch}")
    print(f"✓ Best validation loss: {best_val_loss:.6f}")
    print(f"✓ Best validation F1: {best_val_f1:.6f}")
    
    return start_epoch, best_val_loss, best_val_f1


def find_checkpoint_path(resume_arg, experiment_name):
    """Find the checkpoint path based on resume argument."""
    if resume_arg is None:
        return None
    
    # Check if it's a direct path
    if os.path.exists(resume_arg):
        return resume_arg
    
    # Check predefined checkpoints
    checkpoint_dir = Path("checkpoints") / experiment_name
    
    checkpoint_map = {
        'latest': 'latest_epoch_*.pth',
        'best_loss': 'best_loss.pth',
        'best_f1': 'best_f1.pth',
        'best_threshold': 'best_threshold.pth'
    }
    
    if resume_arg in checkpoint_map:
        if resume_arg == 'latest':
            # Find the latest epoch checkpoint
            pattern = checkpoint_map[resume_arg]
            checkpoints = list(checkpoint_dir.glob(pattern))
            if checkpoints:
                # Sort by epoch number and get the latest
                latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
                return str(latest)
        else:
            checkpoint_path = checkpoint_dir / checkpoint_map[resume_arg]
            if checkpoint_path.exists():
                return str(checkpoint_path)
    
    print(f"❌ Checkpoint not found: {resume_arg}")
    return None


def print_training_summary(config, model, device, resume_info=None):
    """Print comprehensive training configuration summary."""
    print('\n🚀 FINAL PRODUCTION ABR TRAINING')
    print('=' * 80)
    print(f'Experiment: {config["experiment"]["name"]}')
    print(f'Description: {config["experiment"]["description"]}')
    print(f'Model: {config["model"]["type"]}')
    print(f'Dataset: {config["data"]["data_path"]}')
    print(f'Device: {device}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Training configuration
    training_config = config.get('training', {})
    print(f'\nTraining Configuration:')
    print(f'  Batch Size: {training_config.get("batch_size", "N/A")}')
    print(f'  Epochs: {training_config.get("num_epochs", "N/A")}')
    print(f'  Learning Rate: {training_config.get("learning_rate", "N/A")}')
    print(f'  Mixed Precision: {training_config.get("use_amp", False)}')
    print(f'  Gradient Accumulation: {training_config.get("gradient_accumulation_steps", 1)}')
    
    # Early stopping
    early_stopping = config.get('early_stopping', {})
    if early_stopping.get('enabled', False):
        print(f'\nEarly Stopping:')
        print(f'  Monitor: {early_stopping.get("monitor", "val_loss")}')
        print(f'  Patience: {early_stopping.get("patience", 25)}')
        print(f'  Min Delta: {early_stopping.get("min_delta", 1e-5)}')
    
    # Checkpointing
    checkpointing = config.get('checkpointing', {})
    print(f'\nCheckpointing:')
    print(f'  Save Best: {checkpointing.get("save_best", True)}')
    print(f'  Save Top-K: {checkpointing.get("save_top_k", 3)}')
    print(f'  Strategies: {len(checkpointing.get("checkpoint_strategies", []))}')
    
    # Resume information
    if resume_info:
        print(f'\nResuming Training:')
        print(f'  Start Epoch: {resume_info[0]}')
        print(f'  Best Val Loss: {resume_info[1]:.6f}')
        print(f'  Best Val F1: {resume_info[2]:.6f}')
    
    print('=' * 80)


def main():
    """Main production training function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(args.config)
        
        # Override experiment name if provided
        if args.experiment_name:
            config['experiment']['name'] = args.experiment_name
        
        # Override for test run
        if args.test_run:
            config['training']['num_epochs'] = 3
            config['training']['batch_size'] = min(config['training']['batch_size'], 8)
            config['validation']['ddim_steps'] = 10
            config['experiment']['name'] += '_test'
            print("🧪 Running in test mode (3 epochs)")
        
        # Override batch size and workers if specified
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        if args.num_workers:
            config['training']['num_workers'] = args.num_workers
        
        # Enable W&B if requested
        if args.wandb:
            config['logging']['use_wandb'] = True
            print("📊 Weights & Biases logging enabled")
        
        # Set device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        print(f"🔧 Using device: {device}")
        
        # Create data loaders
        print('\n📊 Loading dataset...')
        train_loader, val_loader, _, _ = create_optimized_dataloaders(
            data_path=config['data']['data_path'],
            config=config
        )
        
        print(f"✓ Training samples: {len(train_loader.dataset)}")
        print(f"✓ Validation samples: {len(val_loader.dataset)}")
        
        # Create model
        print('\n🏗️  Creating model...')
        model = create_model(config)
        model = model.to(device)
        
        # Create trainer
        trainer = ABRTrainer(model, train_loader, val_loader, config, device)
        
        # Handle checkpoint resuming
        resume_info = None
        if args.resume:
            checkpoint_path = find_checkpoint_path(args.resume, config['experiment']['name'])
            if checkpoint_path:
                resume_info = load_checkpoint(
                    model, trainer.optimizer, trainer.scheduler, trainer.scaler, 
                    checkpoint_path, device
                )
                # Update trainer state
                trainer.epoch = resume_info[0]
                trainer.best_val_loss = resume_info[1]
                trainer.best_val_f1 = resume_info[2]
        
        # Print training summary
        print_training_summary(config, model, device, resume_info)
        
        # Validation only mode
        if args.validate_only:
            print('\n🔍 Running validation only...')
            val_metrics = trainer.validate_epoch()
            print(f"\nValidation Results:")
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.6f}")
            return 0
        
        # Start training
        print('\n🎯 Starting production training...')
        start_time = time.time()
        
        try:
            trainer.train()
            training_success = True
        except KeyboardInterrupt:
            print('\n⚠️  Training interrupted by user')
            training_success = False
        except Exception as e:
            print(f'\n❌ Training failed: {e}')
            raise
        
        elapsed_time = time.time() - start_time
        
        # Training completed
        if training_success:
            print(f'\n✅ Production training completed successfully!')
        else:
            print(f'\n⚠️  Training interrupted but checkpoints saved')
        
        print(f'⏱️  Total training time: {elapsed_time/3600:.2f} hours')
        
        # Final results
        if hasattr(trainer, 'best_val_loss'):
            print(f'\n📊 Final Results:')
            print(f'   Best Validation Loss: {trainer.best_val_loss:.6f}')
            print(f'   Best Validation F1: {trainer.best_val_f1:.6f}')
        
        # Checkpoint information
        checkpoint_dir = Path("checkpoints") / config['experiment']['name']
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            print(f'\n💾 Saved {len(checkpoints)} checkpoints to: {checkpoint_dir}')
            
            # List important checkpoints
            important_checkpoints = ['best_loss.pth', 'best_f1.pth', 'best_threshold.pth']
            for checkpoint_name in important_checkpoints:
                checkpoint_path = checkpoint_dir / checkpoint_name
                if checkpoint_path.exists():
                    print(f'   ✓ {checkpoint_name}')
        
        return 0
        
    except KeyboardInterrupt:
        print('\n⚠️  Training interrupted by user')
        return 1
    except Exception as e:
        print(f'\n❌ Training failed: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)