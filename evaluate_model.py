#!/usr/bin/env python3
"""
Comprehensive ABR Signal Generation Model Evaluation

This script provides a complete evaluation pipeline for the trained ABR signal generation model.
It evaluates signal quality, generation consistency, conditional control, and produces detailed reports.

Usage:
    python evaluate_model.py --config configs/config_colab_a100.yaml --checkpoint checkpoints/pro/best.pth
    python evaluate_model.py --config configs/config_colab_a100.yaml --checkpoint checkpoints/pro/best.pth --num_samples 100
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import json
import time
from omegaconf import OmegaConf

from training.config_loader import load_config
from models.hierarchical_unet import OptimizedHierarchicalUNet
from data.dataset import create_optimized_dataloaders
from evaluation import SignalGenerationEvaluator
import warnings
warnings.filterwarnings('ignore')


def load_model(config: dict, checkpoint_path: str, device: str = 'cuda') -> nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        config: Model configuration
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"📦 Loading model from {checkpoint_path}")
    
    # Initialize model
    model_cfg = config.get('model', {})
    model = OptimizedHierarchicalUNet(
        input_channels=model_cfg.get('input_channels', 1),  # Ensure correct input channels
        signal_length=model_cfg.get('signal_length', 200),
        static_dim=model_cfg.get('static_dim', 4),
        base_channels=model_cfg.get('base_channels', 64),
        n_levels=model_cfg.get('n_levels', 4),
        dropout=model_cfg.get('dropout', 0.1),
        s4_state_size=model_cfg.get('s4_state_size', 64),
        num_s4_layers=model_cfg.get('num_s4_layers', 2),
        num_transformer_layers=model_cfg.get('num_transformer_layers', 2),
        num_heads=model_cfg.get('num_heads', 8),
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 0)
        
        # Filter out classification head weights if they exist in checkpoint but not in model
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        # Remove task heads that exist in checkpoint but not in current model
        filtered_state_dict = {}
        removed_heads = set()
        for key, value in state_dict.items():
            # Skip all task heads that don't exist in current model
            task_heads = ['class_head.', 'peak_head.', 'threshold_head.', 'static_param_head.']
            is_task_head = any(key.startswith(head) for head in task_heads)
            
            if is_task_head and key not in model_keys:
                head_type = next(head for head in task_heads if key.startswith(head))
                removed_heads.add(head_type.rstrip('.'))
                continue
            if key in model_keys:
                filtered_state_dict[key] = value
            elif not is_task_head:
                print(f"⚠️  Skipping unknown weight: {key}")
        
        if removed_heads:
            print(f"⚠️  Removed task heads: {', '.join(removed_heads)}")
        
        # Check if EMA weights are available (preferred for generation)
        if 'ema_shadow' in checkpoint:
            print("🎯 Using EMA weights for better generation quality")
            ema_state_dict = checkpoint['ema_shadow']
            
            # Filter EMA weights the same way
            filtered_ema_state_dict = {}
            for key, value in ema_state_dict.items():
                task_heads = ['class_head.', 'peak_head.', 'threshold_head.', 'static_param_head.']
                is_task_head = any(key.startswith(head) for head in task_heads)
                if not is_task_head and key in model_keys:
                    filtered_ema_state_dict[key] = value
            
            missing_keys, unexpected_keys = model.load_state_dict(filtered_ema_state_dict, strict=False)
            print(f"✅ Loaded EMA model from epoch {epoch} (filtered classification weights)")
        else:
            # Fallback to regular weights
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✅ Loaded regular model from epoch {epoch} (filtered classification weights)")
        
        if missing_keys:
            print(f"⚠️  Missing keys in checkpoint: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
    else:
        # Direct state dict - also filter if needed
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        
        filtered_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('class_head.') and key not in model_keys:
                print(f"⚠️  Skipping classification head weight: {key}")
                continue
            if key in model_keys:
                filtered_state_dict[key] = value
                
        model.load_state_dict(filtered_state_dict, strict=False)
        print("✅ Loaded model state dict (filtered)")
    
    model = model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate ABR Signal Generation Model')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None for all)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_ddim_steps', type=int, default=50,
                       help='Number of DDIM sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                       help='Classifier-free guidance scale')
    parser.add_argument('--detailed_analysis', action='store_true',
                       help='Run detailed analysis (slower but more comprehensive)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 Starting ABR Model Evaluation")
    print(f"📱 Device: {device}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Load configuration
    config = load_config(args.config)
    print(f"⚙️ Loaded config from {args.config}")
    
    # Override data config for evaluation
    config['data']['batch_size'] = args.batch_size
    
    # Load model
    model = load_model(config, args.checkpoint, device)
    
    # Create data loaders
    print("📊 Creating data loaders...")
    train_loader, val_loader, test_loader, dataset = create_optimized_dataloaders(
        data_path=config['data'].get('path', 'data/processed/ultimate_dataset_with_clinical_thresholds.pkl'),
        batch_size=args.batch_size,
        train_ratio=config['data'].get('train_ratio', 0.7),
        val_ratio=config['data'].get('val_ratio', 0.15),
        test_ratio=config['data'].get('test_ratio', 0.15),
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True),
        random_state=config['training'].get('random_seed', 42),
    )
    
    print(f"✅ Created data loaders:")
    print(f"   - Test samples: {len(test_loader.dataset)}")
    print(f"   - Batch size: {args.batch_size}")
    
    # Initialize evaluator
    evaluator = SignalGenerationEvaluator(
        model=model,
        config=config,
        device=device,
        output_dir=args.output_dir
    )
    
    print("🔍 Starting comprehensive evaluation...")
    start_time = time.time()
    
    # 1. Evaluate on test dataset
    print("\n📈 Phase 1: Dataset Evaluation")
    dataset_metrics = evaluator.evaluate_on_dataset(
        test_loader=test_loader,
        num_samples=args.num_samples,
        save_samples=True
    )
    print(f"✅ Dataset evaluation completed")
    
    # 2. Evaluate generation quality
    print("\n🎯 Phase 2: Generation Quality Evaluation")
    # Get some static conditions for generation testing
    test_batch = next(iter(test_loader))
    test_static = test_batch['static_params'][:min(20, len(test_batch['static_params']))]
    
    generation_stats = evaluator.evaluate_generation_quality(
        static_conditions=test_static,
        num_steps=args.num_ddim_steps,
        cfg_scale=args.cfg_scale
    )
    print(f"✅ Generation quality evaluation completed")
    
    # 3. Evaluate conditional control (if detailed analysis requested)
    if args.detailed_analysis:
        print("\n🎛️ Phase 3: Conditional Control Evaluation")
        condition_variations = [
            {'hearing_loss': 0.0},   # Normal hearing
            {'hearing_loss': 0.5},   # Moderate hearing loss
            {'hearing_loss': 1.0},   # Severe hearing loss
            {'age': 20.0},           # Young
            {'age': 60.0},           # Older
        ]
        
        control_results = evaluator.evaluate_conditional_control(
            test_loader=test_loader,
            condition_variations=condition_variations
        )
        print(f"✅ Conditional control evaluation completed")
        
        # 4. Evaluate consistency
        print("\n🔄 Phase 4: Generation Consistency Evaluation")
        consistency_results = evaluator.evaluate_consistency(
            static_conditions=test_static[:5],  # Test with fewer samples for consistency
            num_generations=10
        )
        print(f"✅ Consistency evaluation completed")
    
    # 5. Generate comprehensive report
    print("\n📊 Phase 5: Generating Evaluation Report")
    report = evaluator.generate_evaluation_report(save_report=True)
    
    total_time = time.time() - start_time
    print(f"\n🎉 Evaluation completed in {total_time:.2f} seconds")
    
    # Print key results
    print("\n📋 Key Results Summary:")
    print("=" * 50)
    
    if 'metrics' in dataset_metrics:
        metrics = dataset_metrics
        if 'snr' in metrics:
            print(f"📊 Signal-to-Noise Ratio: {metrics['snr']['mean']:.2f} ± {metrics['snr']['std']:.2f} dB")
        if 'correlation' in metrics and 'pearson_r' in metrics['correlation']:
            corr = metrics['correlation']['pearson_r']['mean']
            print(f"🔗 Signal Correlation: {corr:.3f}")
        if 'rmse' in metrics:
            print(f"📏 RMSE: {metrics['rmse']['mean']:.4f} ± {metrics['rmse']['std']:.4f}")
    
    if 'avg_generation_time' in generation_stats:
        print(f"⏱️ Avg Generation Time: {generation_stats['avg_generation_time']:.3f}s per sample")
    
    print(f"\n💾 Detailed results saved to: {args.output_dir}/")
    print(f"📄 Evaluation report: {args.output_dir}/evaluation_report.json")
    print(f"📈 Summary metrics: {args.output_dir}/summary_metrics.csv")
    print(f"🖼️ Visualizations: {args.output_dir}/plots/")
    
    # Print recommendations
    if 'recommendations' in report:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    print("\n✨ Evaluation pipeline completed successfully!")


if __name__ == '__main__':
    main()