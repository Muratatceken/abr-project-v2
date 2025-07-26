#!/usr/bin/env python3
"""
Unified ABR Training Runner

Simple and reliable script for training ABR models with clinical thresholds.

Usage:
    python run_training.py                                    # Default: optimized model with clinical thresholds
    python run_training.py --config training/config.yaml     # Professional model
    python run_training.py --model_type professional         # Force professional model
    python run_training.py --epochs 100 --batch_size 16      # Override config parameters

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
    """Create argument parser for unified training script."""
    parser = argparse.ArgumentParser(
        description='Unified ABR Model Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration and model selection
    parser.add_argument('--config', type=str, default='training/config_optimized_architecture.yaml',
                       help='Path to YAML configuration file')
    parser.add_argument('--model_type', type=str, default='auto',
                       choices=['professional', 'optimized', 'auto'],
                       help='Model architecture type')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default=None,
                       help='Override dataset path')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Override learning rate')
    
    # Device selection
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    # Utility flags
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with 2 epochs')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    return parser


def detect_model_type_from_config(config: dict) -> str:
    """Detect model type from configuration."""
    model_type = config.get('model', {}).get('type', '').lower()
    
    if 'optimized' in model_type:
        return 'optimized'
    elif 'professional' in model_type:
        return 'professional'
    else:
        # Default to optimized for clinical thresholds
        return 'optimized'


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply command line overrides to configuration."""
    
    # Data overrides
    if args.data_path:
        config['data']['data_path'] = args.data_path
    
    # Training overrides
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Quick test mode
    if args.quick_test:
        config['training']['num_epochs'] = 2
        config['training']['batch_size'] = min(config['training']['batch_size'], 8)
        config['experiment']['name'] = f"{config['experiment']['name']}_quick_test"
    
    return config


def print_training_summary(config: dict, model_type: str, device: torch.device):
    """Print training configuration summary."""
    print('\n🚀 ABR TRAINING CONFIGURATION')
    print('=' * 60)
    print(f'Model Type: {model_type.upper()}')
    print(f'Model Architecture: {config["model"]["type"]}')
    print(f'Dataset: {config["data"]["data_path"]}')
    print(f'Experiment: {config["experiment"]["name"]}')
    print(f'Device: {device}')
    print(f'Batch Size: {config["training"]["batch_size"]}')
    print(f'Epochs: {config["training"]["num_epochs"]}')
    print(f'Learning Rate: {config["training"]["learning_rate"]}')
    
    if 'loss' in config and 'loss_weights' in config['loss']:
        print(f'Clinical Threshold Weight: {config["loss"]["loss_weights"].get("threshold", "N/A")}')
    
    print('=' * 60)


def main():
    """Main training function."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(args.config)
        
        # Detect or set model type
        if args.model_type == 'auto':
            model_type = detect_model_type_from_config(config)
        else:
            model_type = args.model_type
            
        # Update model type in config if needed
        if model_type == 'optimized':
            config['model']['type'] = 'optimized_hierarchical_unet_v2'
        elif model_type == 'professional':
            config['model']['type'] = 'professional_hierarchical_unet'
        
        # Apply command line overrides
        config = apply_overrides(config, args)
        
        # Set device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        # Print summary
        if args.verbose:
            print_training_summary(config, model_type, device)
        
        # Validate dataset path
        data_path = config['data']['data_path']
        if not os.path.exists(data_path):
            print(f'❌ Dataset not found: {data_path}')
            return 1
        
        # Create data loaders
        if args.verbose:
            print('\n📊 Loading dataset...')
        
        train_loader, val_loader, _, _ = create_optimized_dataloaders(
            data_path=data_path,
            config=config
        )
        
        # Create model
        if args.verbose:
            print('🏗️  Creating model...')
        
        model = create_model(config)
        model = model.to(device)
        
        if args.verbose:
            print(f'✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters')
        
        # Create trainer
        trainer = ABRTrainer(model, train_loader, val_loader, config, device)
        
        # Start training
        if args.verbose:
            print('\n🎯 Starting training...')
        
        start_time = time.time()
        trainer.train()
        elapsed_time = time.time() - start_time
        
        # Print results
        print(f'\n✅ Training completed successfully in {elapsed_time/60:.1f} minutes!')
        
        if hasattr(trainer, 'best_metrics') and trainer.best_metrics:
            print(f'\n📊 Best Validation Metrics:')
            for metric, value in trainer.best_metrics.items():
                if isinstance(value, (int, float)):
                    print(f'   {metric}: {value:.4f}')
        
        if hasattr(trainer, 'best_model_path'):
            print(f'\n💾 Best model saved to: {trainer.best_model_path}')
        
        return 0
        
    except KeyboardInterrupt:
        print('\n⚠️  Training interrupted by user')
        return 1
    except Exception as e:
        print(f'\n❌ Training failed: {e}')
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 