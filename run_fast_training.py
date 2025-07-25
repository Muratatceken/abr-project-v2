#!/usr/bin/env python3
"""
Fast Training Runner for ABR Model

Optimized training script that uses the fast configuration for maximum speed
while maintaining model complexity.

Usage:
    python run_fast_training.py
    python run_fast_training.py --batch_size 128
    python run_fast_training.py --gpu_ids 0 1  # Multi-GPU
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from training.config_loader import load_config_with_overrides
from training.enhanced_train import main as train_main


def create_parser():
    """Create argument parser for fast training."""
    parser = argparse.ArgumentParser(
        description='Fast ABR Model Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Override options for fast training
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--num_epochs', type=int, help='Override number of epochs')
    parser.add_argument('--num_workers', type=int, help='Override number of data workers')
    parser.add_argument('--gpu_ids', type=int, nargs='+', help='GPU IDs to use')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    
    # Quick modes
    parser.add_argument('--ultra_fast', action='store_true', 
                       help='Ultra fast mode: even larger batch size and fewer epochs')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark mode: single epoch for speed testing')
    
    return parser


def apply_fast_overrides(args, config):
    """Apply fast training overrides to config."""
    
    if args.ultra_fast:
        print("🚀 ULTRA FAST MODE: Maximum speed optimizations")
        config['training']['batch_size'] = 128
        config['training']['num_epochs'] = 25
        config['training']['gradient_accumulation_steps'] = 4
        config['training']['num_workers'] = 12
        config['training']['validate_every'] = 3
        config['training']['patience'] = 5
        config['data']['val_split'] = 0.1
        
    elif args.benchmark:
        print("⏱️ BENCHMARK MODE: Single epoch for speed testing")
        config['training']['num_epochs'] = 1
        config['training']['batch_size'] = 64
        config['evaluation']['validate_every'] = 1
        config['training']['save_every'] = 1
        
    # Apply command line overrides
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.num_workers:
        config['training']['num_workers'] = args.num_workers
    if args.gpu_ids:
        config['hardware']['gpu_ids'] = args.gpu_ids
    if args.output_dir:
        config['logging']['output_dir'] = args.output_dir


def setup_environment():
    """Setup optimized environment for fast training."""
    # Set optimal PyTorch settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set number of threads for CPU operations
    torch.set_num_threads(4)
    
    # Enable JIT compilation if available
    try:
        torch.jit.set_fusion_strategy([('STATIC', 2), ('DYNAMIC', 20)])
        print("✓ Enabled JIT fusion optimizations")
    except:
        pass
    
    # Print system info
    if torch.cuda.is_available():
        print(f"🔥 CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    else:
        print("💻 Using CPU training")


def main():
    """Main fast training function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Load fast configuration
    config_path = 'training/config_fast.yaml'
    if not os.path.exists(config_path):
        print(f"❌ Fast config not found: {config_path}")
        print("Please ensure the fast training configuration exists")
        return 1
    
    print(f"📋 Loading fast configuration from: {config_path}")
    config = load_config_with_overrides(config_path, args=args)
    
    # Apply fast training overrides
    apply_fast_overrides(args, config)
    
    # Validate dataset
    if not os.path.exists(config['data']['data_path']):
        print(f"❌ Dataset not found: {config['data']['data_path']}")
        return 1
    
    # Print optimized configuration summary
    print("\n" + "="*70)
    print("🚀 FAST ABR TRAINING CONFIGURATION")
    print("="*70)
    print(f"Dataset: {config['data']['data_path']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Data workers: {config['training']['num_workers']}")
    print(f"Validation every: {config['evaluation']['validate_every']} epochs")
    print(f"Early stopping patience: {config['training']['patience']}")
    print(f"Mixed precision: {config['training']['use_amp']}")
    print(f"Memory optimization: {config.get('memory', {}).get('use_memory_efficient_attention', False)}")
    print("="*70)
    
    # Convert config to args format for compatibility
    class ConfigArgs:
        def __init__(self, config_dict):
            self.config = config_dict
            # Flatten for attribute access
            for section, values in config_dict.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        setattr(self, f"{section}_{key}", value)
                        setattr(self, key, value)  # Also set without prefix
                else:
                    setattr(self, section, values)
    
    config_args = ConfigArgs(config)
    
    try:
        print("\n🎯 Starting Fast ABR Training...")
        train_main(config_args)
        print("\n✅ Fast training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 