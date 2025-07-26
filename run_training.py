#!/usr/bin/env python3
"""
Unified ABR Training Runner

Enhanced script to run ABR training pipeline supporting both ProfessionalHierarchicalUNet 
and OptimizedHierarchicalUNet architectures with comprehensive features.

Usage:
    python run_training.py --config training/config.yaml
    python run_training.py --config training/config_optimized_v2.yaml --model_type optimized
    python run_training.py --quick_test --debug

Author: AI Assistant  
Date: January 2025
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from training.config_loader import ConfigLoader, load_config_with_overrides
from training.enhanced_train import main as enhanced_train_main


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for unified training script."""
    parser = argparse.ArgumentParser(
        description='Unified ABR Model Training with Architecture Selection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file
    parser.add_argument(
        '--config', type=str, default='training/config.yaml',
        help='Path to YAML configuration file'
    )
    
    # Model architecture selection
    parser.add_argument(
        '--model_type', type=str, default='professional', 
        choices=['professional', 'optimized', 'auto'],
        help='Model architecture type (professional=ProfessionalHierarchicalUNet, optimized=OptimizedHierarchicalUNet, auto=detect from config)'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_path', type=str, default='data/processed/ultimate_dataset.pkl',
        help='Path to dataset file'
    )
    parser.add_argument(
        '--valid_peaks_only', action='store_true',
        help='Train only on samples with valid V peaks'
    )
    parser.add_argument(
        '--val_split', type=float, default=0.2,
        help='Validation split ratio'
    )
    
    # Model arguments (compatible with both architectures)
    parser.add_argument(
        '--base_channels', type=int, default=64,
        help='Base number of channels'
    )
    parser.add_argument(
        '--n_levels', type=int, default=4,
        help='Number of U-Net levels'
    )
    parser.add_argument(
        '--num_transformer_layers', type=int, default=3,
        help='Number of transformer layers'
    )
    parser.add_argument(
        '--use_cross_attention', action='store_true', default=True,
        help='Use cross-attention between encoder and decoder'
    )
    parser.add_argument(
        '--film_dropout', type=float, default=0.15,
        help='FiLM dropout rate for robustness'
    )
    parser.add_argument(
        '--num_classes', type=int, default=5,
        help='Number of output classes'
    )
    
    # Optimized architecture specific arguments
    parser.add_argument(
        '--use_multi_scale_attention', action='store_true', default=False,
        help='Use multi-scale attention for peak detection (optimized model only)'
    )
    parser.add_argument(
        '--use_task_specific_extractors', action='store_true', default=False,
        help='Use task-specific feature extractors (optimized model only)'
    )
    parser.add_argument(
        '--use_attention_skip_connections', action='store_true', default=False,
        help='Use attention-based skip connections (optimized model only)'
    )
    parser.add_argument(
        '--enable_joint_generation', action='store_true', default=False,
        help='Enable joint signal and parameter generation (optimized model only)'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.01,
        help='Weight decay for AdamW optimizer'
    )
    parser.add_argument(
        '--use_amp', action='store_true', default=True,
        help='Use automatic mixed precision training'
    )
    parser.add_argument(
        '--patience', type=int, default=15,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--gradient_clip_norm', type=float, default=1.0,
        help='Gradient clipping norm'
    )
    
    # Loss function arguments
    parser.add_argument(
        '--use_focal_loss', action='store_true',
        help='Use focal loss for classification (helps with class imbalance)'
    )
    parser.add_argument(
        '--use_class_weights', action='store_true', default=True,
        help='Use class weights for imbalanced data'
    )
    parser.add_argument(
        '--signal_loss_weight', type=float, default=1.0,
        help='Weight for signal reconstruction loss'
    )
    parser.add_argument(
        '--classification_loss_weight', type=float, default=1.5,
        help='Weight for classification loss'
    )
    
    # Data augmentation and sampling
    parser.add_argument(
        '--augment', action='store_true', default=True,
        help='Use data augmentation'
    )
    parser.add_argument(
        '--use_balanced_sampler', action='store_true',
        help='Use balanced sampling for imbalanced classes'
    )
    parser.add_argument(
        '--cfg_dropout_prob', type=float, default=0.1,
        help='CFG dropout probability for unconditional training'
    )
    
    # Scheduler arguments
    parser.add_argument(
        '--scheduler', type=str, default='cosine_warm_restarts',
        choices=['cosine_warm_restarts', 'reduce_on_plateau', 'step', 'none'],
        help='Learning rate scheduler type'
    )
    parser.add_argument(
        '--T_0', type=int, default=10,
        help='Initial restart period for cosine warm restarts'
    )
    parser.add_argument(
        '--eta_min', type=float, default=1e-6,
        help='Minimum learning rate for cosine scheduler'
    )
    
    # Output and logging
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--experiment_name', type=str, default=None,
        help='Experiment name for logging'
    )
    parser.add_argument(
        '--use_wandb', action='store_true',
        help='Use Weights & Biases for logging'
    )
    parser.add_argument(
        '--wandb_project', type=str, default='abr-unified-training',
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--save_every', type=int, default=10,
        help='Save checkpoint every N epochs'
    )
    
    # System arguments
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loader workers'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--gpu_ids', type=int, nargs='+', default=[0],
        help='GPU IDs to use for training'
    )
    
    # Cross-validation arguments
    parser.add_argument(
        '--use_cv', action='store_true',
        help='Use cross-validation instead of single train/val split'
    )
    parser.add_argument(
        '--cv_folds', type=int, default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--cv_strategy', type=str, default='StratifiedGroupKFold',
        choices=['StratifiedKFold', 'StratifiedGroupKFold', 'GroupKFold'],
        help='Cross-validation strategy'
    )

    # Resume training
    parser.add_argument(
        '--resume_from', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )
    
    # Quick training modes
    parser.add_argument(
        '--quick_test', action='store_true',
        help='Quick test mode with reduced epochs and batch size'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Debug mode with minimal data and epochs'
    )
    
    # Architecture testing
    parser.add_argument(
        '--test_architecture', action='store_true',
        help='Test the architecture before training'
    )
    
    return parser


def setup_environment(args):
    """Setup training environment."""
    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Set CUDA settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available, using CPU")
    
    # Set number of threads
    torch.set_num_threads(4)


def apply_quick_modes(args):
    """Apply quick test or debug mode settings."""
    if args.debug:
        print("🐛 DEBUG MODE: Using minimal settings for debugging")
        args.num_epochs = 2
        args.batch_size = 4
        args.num_workers = 0
        args.save_every = 1
        args.patience = 2
        args.base_channels = 32
        args.n_levels = 2
        args.num_transformer_layers = 1
        
    elif args.quick_test:
        print("⚡ QUICK TEST MODE: Using reduced settings for quick validation")
        args.num_epochs = 10
        args.batch_size = 16
        args.patience = 5
        args.save_every = 2
        args.base_channels = 48
        args.n_levels = 3


def detect_model_type_from_config(config):
    """Detect model type from configuration."""
    model_config = config.get('model', {})
    
    # Check for optimized model indicators
    optimized_indicators = [
        'use_multi_scale_attention',
        'use_task_specific_extractors', 
        'use_attention_skip_connections',
        'enable_joint_generation'
    ]
    
    # Check if any optimized features are enabled
    for indicator in optimized_indicators:
        if model_config.get(indicator, False):
            return 'optimized'
    
    # Check model type directly
    model_type = model_config.get('type', '').lower()
    if 'optimized' in model_type:
        return 'optimized'
    elif 'professional' in model_type:
        return 'professional'
    
    # Default to professional
    return 'professional'


def create_optimized_model_config(args):
    """Create configuration for optimized model."""
    return {
        'input_channels': 1,
        'static_dim': 4,
        'base_channels': args.base_channels,
        'n_levels': args.n_levels,
        'sequence_length': 200,
        'signal_length': 200,
        'num_classes': args.num_classes,
        
        # S4 configuration
        's4_state_size': 64,
        'num_s4_layers': 2,
        'use_enhanced_s4': True,
        
        # Transformer configuration
        'num_transformer_layers': args.num_transformer_layers,
        'num_heads': 8,
        'use_multi_scale_attention': args.use_multi_scale_attention,
        'use_cross_attention': args.use_cross_attention,
        
        # FiLM and conditioning
        'dropout': 0.1,
        'film_dropout': args.film_dropout,
        'use_cfg': True,
        
        # Output configuration
        'use_attention_heads': True,
        'predict_uncertainty': True,
        
        # Joint generation
        'enable_joint_generation': args.enable_joint_generation,
        'static_param_ranges': {
            'age': [-2.0, 2.0],
            'intensity': [-2.0, 2.0], 
            'stimulus_rate': [-2.0, 2.0],
            'fmp': [0.0, 150.0]
        },
        
        # Optimization features
        'use_task_specific_extractors': args.use_task_specific_extractors,
        'use_attention_skip_connections': args.use_attention_skip_connections,
        'channel_multiplier': 2.0
    }


def test_optimized_architecture(model_config, device):
    """Test the optimized architecture before training."""
    print("\n🧪 Testing optimized architecture...")
    
    try:
        from models.hierarchical_unet import OptimizedHierarchicalUNet
        
        model = OptimizedHierarchicalUNet(**model_config)
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            # Test input
            x = torch.randn(2, 1, 200, device=device)
            static_params = torch.randn(2, 4, device=device)
            
            # Test conditional generation
            outputs = model(x, static_params, generation_mode='conditional')
            print(f"✅ Conditional generation: {list(outputs.keys())}")
            
            # Test joint generation if enabled
            if model_config.get('enable_joint_generation', False):
                joint_outputs = model.generate_joint(batch_size=2, device=device)
                print(f"✅ Joint generation: {list(joint_outputs.keys())}")
            
            # Get model info
            model_info = model.get_model_info()
            print(f"✅ Model parameters: {model_info['total_parameters']:,}")
            print(f"✅ Architecture features: {len(model_info['features'])}")
            
            print("🎉 Optimized architecture test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Optimized architecture test failed: {e}")
        return False


def main():
    """Main training function."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Apply quick modes
    apply_quick_modes(args)
    
    # Setup environment
    setup_environment(args)
    
    # Load configuration
    if os.path.exists(args.config):
        print(f"📋 Loading configuration from: {args.config}")
        config = load_config_with_overrides(args.config, args=args)
    else:
        print(f"Configuration file not found: {args.config}")
        print("Using command line arguments only")
        from training.config_loader import create_config_from_args
        config = create_config_from_args(args)
    
    # Determine model type
    if args.model_type == 'auto':
        detected_type = detect_model_type_from_config(config)
        print(f"🔍 Auto-detected model type: {detected_type}")
        model_type = detected_type
    else:
        model_type = args.model_type
    
    # Update configuration based on model type
    if model_type == 'optimized':
        print("🚀 Using OptimizedHierarchicalUNet architecture")
        
        # Set optimized model indicators in config
        if 'model' not in config:
            config['model'] = {}
        
        config['model']['type'] = 'optimized_hierarchical_unet_v2'
        config['model']['use_multi_scale_attention'] = args.use_multi_scale_attention
        config['model']['use_task_specific_extractors'] = args.use_task_specific_extractors
        config['model']['use_attention_skip_connections'] = args.use_attention_skip_connections
        config['model']['enable_joint_generation'] = args.enable_joint_generation
        
        # Test optimized architecture if requested
        if args.test_architecture:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_config = create_optimized_model_config(args)
            if not test_optimized_architecture(model_config, device):
                print("❌ Architecture test failed, aborting training")
                return 1
                
        # Update project name for wandb
        if args.use_wandb:
            args.wandb_project = 'abr-optimized-v2'
        
    else:
        print("🏛️ Using ProfessionalHierarchicalUNet architecture")
        if 'model' not in config:
            config['model'] = {}
        config['model']['type'] = 'professional_hierarchical_unet'
    
    # Validate required paths
    data_path = config.get('data', {}).get('data_path', args.data_path)
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        print("Please ensure the ultimate_dataset.pkl file exists")
        return 1
    
    # Print configuration summary
    print("\n" + "="*70)
    print("🚀 UNIFIED ABR TRAINING CONFIGURATION")
    print("="*70)
    print(f"Model Type: {model_type.upper()}")
    print(f"Dataset: {data_path}")
    print(f"Valid peaks only: {config.get('data', {}).get('valid_peaks_only', args.valid_peaks_only)}")
    print(f"Batch size: {config.get('training', {}).get('batch_size', args.batch_size)}")
    print(f"Learning rate: {config.get('training', {}).get('learning_rate', args.learning_rate)}")
    print(f"Epochs: {config.get('training', {}).get('num_epochs', args.num_epochs)}")
    print(f"Model channels: {config.get('model', {}).get('base_channels', args.base_channels)}")
    print(f"Model levels: {config.get('model', {}).get('n_levels', args.n_levels)}")
    print(f"Transformer layers: {config.get('model', {}).get('num_transformer_layers', args.num_transformer_layers)}")
    print(f"Cross-attention: {config.get('model', {}).get('use_cross_attention', args.use_cross_attention)}")
    print(f"FiLM dropout: {config.get('model', {}).get('film_dropout', args.film_dropout)}")
    
    if model_type == 'optimized':
        print(f"Multi-scale attention: {args.use_multi_scale_attention}")
        print(f"Task-specific extractors: {args.use_task_specific_extractors}")
        print(f"Attention skip connections: {args.use_attention_skip_connections}")
        print(f"Joint generation: {args.enable_joint_generation}")
    
    print(f"Mixed precision: {config.get('training', {}).get('use_amp', args.use_amp)}")
    print(f"Output directory: {config.get('logging', {}).get('output_dir', args.output_dir or 'auto-generated')}")
    print("="*70)
    
    # Convert config back to args format for compatibility with enhanced_train.py
    class ConfigArgs:
        def __init__(self, config_dict, original_args):
            self.config = config_dict
            
            # Copy original args
            for key, value in vars(original_args).items():
                setattr(self, key, value)
            
            # Override with config values, flattening nested structure
            self._flatten_config(config_dict)
        
        def _flatten_config(self, config, prefix=''):
            for key, value in config.items():
                if isinstance(value, dict):
                    self._flatten_config(value, f"{prefix}{key}_" if prefix else f"{key}_")
                else:
                    attr_name = f"{prefix}{key}" if prefix else key
                    setattr(self, attr_name, value)
    
    # Create args object from config
    config_args = ConfigArgs(config, args)
    
    # Set specific attributes that the training script expects
    for key in ['data_path', 'valid_peaks_only', 'batch_size', 'learning_rate', 
                'num_epochs', 'base_channels', 'n_levels', 'num_transformer_layers',
                'use_cross_attention', 'film_dropout', 'use_amp', 'patience',
                'use_focal_loss', 'use_class_weights', 'augment', 'use_balanced_sampler',
                'cfg_dropout_prob', 'output_dir', 'use_wandb', 'wandb_project',
                'experiment_name', 'num_workers', 'random_seed', 'num_classes']:
        
        # Get value from nested config
        value = config
        for part in key.split('_'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = getattr(args, key, None)
                break
        
        if value is not None:
            setattr(config_args, key, value)
    
    # Start training
    try:
        print(f"\n🎯 Starting {model_type.title()} ABR Training...")
        
        # Check if cross-validation is requested
        if config.get('validation', {}).get('use_cv', False) or getattr(config_args, 'use_cv', False):
            print("🔄 Running Cross-Validation...")
            from training.enhanced_train import run_cross_validation
            
            # Update config with CV settings
            if not config.get('validation'):
                config['validation'] = {}
            
            config['validation'].update({
                'use_cv': True,
                'cv_folds': getattr(config_args, 'cv_folds', 5),
                'cv_strategy': getattr(config_args, 'cv_strategy', 'StratifiedGroupKFold')
            })
            
            cv_results = run_cross_validation(config)
            
            print("\n✅ Cross-validation completed successfully!")
            print(f"📊 Results saved to: {config.get('output_dir', 'outputs')}/cv_results.json")
            
            # Print summary
            if cv_results['mean_metrics']:
                print(f"\n📈 Cross-Validation Summary:")
                for metric in ['f1_macro', 'f1_weighted', 'balanced_accuracy']:
                    if metric in cv_results['mean_metrics']:
                        mean_val = cv_results['mean_metrics'][metric]
                        std_val = cv_results['std_metrics'][metric]
                        print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
            
        else:
            print("🚀 Running Single Training...")
            enhanced_train_main(config_args)
            print(f"\n✅ {model_type.title()} training completed successfully!")
            
        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 