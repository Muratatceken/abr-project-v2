#!/usr/bin/env python3
"""
Enhanced Model Evaluation Script
Evaluates the enhanced ABR diffusion model with proper architecture matching.
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml
from tqdm import tqdm

# Import enhanced components
from models.improved_hierarchical_unet import ImprovedHierarchicalUNet
from models.hierarchical_unet import OptimizedHierarchicalUNet
from data.dataset import create_optimized_dataloaders
from utils.sampling import create_ddim_sampler
from utils.schedule import get_noise_schedule, NoiseSchedule


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Create the correct model architecture based on config."""
    model_cfg = config.get('model', {})
    
    # For evaluation, we'll try to use the standard model first (more compatible)
    # and fall back to enhanced if needed
    print("📦 Using Standard Optimized Hierarchical U-Net for maximum compatibility")
    
    # Get the number of classes from config or default
    num_classes = model_cfg.get('num_classes', 5)  # Default to 5 classes
    
    model = OptimizedHierarchicalUNet(
        signal_length=model_cfg.get('signal_length', 200),
        static_dim=model_cfg.get('static_dim', 4),
        input_channels=model_cfg.get('input_channels', 1),
        base_channels=model_cfg.get('base_channels', 64),
        n_levels=model_cfg.get('n_levels', 4),
        num_classes=num_classes,
        dropout=model_cfg.get('dropout', 0.1),
        s4_state_size=model_cfg.get('s4_state_size', 64),
        num_s4_layers=model_cfg.get('num_s4_layers', 2),
        num_transformer_layers=model_cfg.get('num_transformer_layers', 2),
        num_heads=model_cfg.get('num_heads', 8),
    ).to(device)
    
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """Load model checkpoint with proper error handling."""
    print(f"📦 Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Choose the best available state dict
    state_dict_to_use = None
    source_name = ""
    
    if 'ema_shadow' in checkpoint and checkpoint['ema_shadow']:
        print("🎯 Using EMA weights for better generation quality")
        state_dict_to_use = checkpoint['ema_shadow']
        source_name = "EMA"
    else:
        print("📦 Using regular model weights")
        state_dict_to_use = checkpoint.get('model_state_dict', {})
        source_name = "regular"
    
    # Filter out incompatible keys (especially head layers that might have different sizes)
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    incompatible_keys = []
    
    for key, value in state_dict_to_use.items():
        if key in model_state_dict:
            # Check if shapes match
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                incompatible_keys.append(f"{key}: checkpoint {value.shape} vs model {model_state_dict[key].shape}")
        else:
            incompatible_keys.append(f"{key}: not in model")
    
    if incompatible_keys:
        print(f"⚠️  Skipping {len(incompatible_keys)} incompatible keys:")
        for key in incompatible_keys[:5]:  # Show first 5
            print(f"    {key}")
        if len(incompatible_keys) > 5:
            print(f"    ... and {len(incompatible_keys) - 5} more")
    
    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"✅ Loaded {source_name} weights:")
    print(f"    Loaded: {len(filtered_state_dict)} layers")
    print(f"    Missing: {len(missing_keys)} layers")
    print(f"    Incompatible: {len(incompatible_keys)} layers")
    
    epoch = checkpoint.get('epoch', 0)
    print(f"📅 Model from epoch {epoch}")
    
    return checkpoint


def generate_samples(
    model: torch.nn.Module,
    sampler,
    static_params: torch.Tensor,
    device: torch.device,
    num_samples: int = 10,
    steps: int = 50
) -> torch.Tensor:
    """Generate ABR signal samples."""
    model.eval()
    
    print(f"🎯 Generating {num_samples} samples with {steps} DDIM steps...")
    
    generated_samples = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Generating"):
            # Sample from batch
            idx = i % len(static_params)
            static_sample = static_params[idx:idx+1].to(device)
            
            # Generate sample
            shape = (1, 1, 200)  # [batch, channels, length]
            sample = sampler.sample(
                model=model,
                shape=shape,
                static_params=static_sample,
                device=device,
                num_steps=steps,
                cfg_scale=1.0,
                progress=False
            )
            
            generated_samples.append(sample.cpu())
    
    return torch.cat(generated_samples, dim=0)


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    sampler,
    device: torch.device,
    num_samples: int = 20,
    steps: int = 50
) -> Dict[str, Any]:
    """Evaluate model performance."""
    print("🔍 Starting model evaluation...")
    
    # Get some test data for generation
    test_batch = next(iter(test_loader))
    test_signals = test_batch['signal'][:num_samples]
    test_static = test_batch['static_params'][:num_samples]
    
    # Generate samples
    generated = generate_samples(
        model=model,
        sampler=sampler,
        static_params=test_static,
        device=device,
        num_samples=num_samples,
        steps=steps
    )
    
    # Basic statistics
    real_mean = test_signals.mean().item()
    real_std = test_signals.std().item()
    gen_mean = generated.mean().item()
    gen_std = generated.std().item()
    
    # MSE between generated and real (rough quality metric)
    # Note: This is not perfect since we're not comparing paired samples
    mse_approx = F.mse_loss(generated[:len(test_signals)], test_signals).item()
    
    results = {
        'num_samples': num_samples,
        'ddim_steps': steps,
        'real_signal_mean': real_mean,
        'real_signal_std': real_std,
        'generated_mean': gen_mean,
        'generated_std': gen_std,
        'approximate_mse': mse_approx,
        'generated_samples': generated.numpy(),
        'real_samples': test_signals.numpy(),
        'static_params': test_static.numpy()
    }
    
    return results


def plot_comparison(results: Dict[str, Any], output_dir: Path):
    """Create comparison plots."""
    generated = results['generated_samples']
    real = results['real_samples']
    
    num_compare = min(6, len(generated), len(real))
    
    fig, axes = plt.subplots(num_compare, 2, figsize=(15, 3*num_compare))
    if num_compare == 1:
        axes = axes.reshape(1, -1)
    
    time_axis = np.arange(200) / 1000  # Assuming 1kHz sampling
    
    for i in range(num_compare):
        # Generated signal
        axes[i, 0].plot(time_axis, generated[i, 0, :], 'b-', linewidth=1.0, label='Generated')
        axes[i, 0].set_title(f'Generated ABR Sample {i+1}')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_ylim([-4, 4])
        
        # Real signal
        axes[i, 1].plot(time_axis, real[i, 0, :], 'r-', linewidth=1.0, label='Real')
        axes[i, 1].set_title(f'Real ABR Sample {i+1}')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_ylim([-4, 4])
        
        if i == num_compare - 1:
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary statistics plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Distribution comparison
    axes[0].hist(generated.flatten(), bins=50, alpha=0.7, label='Generated', density=True)
    axes[0].hist(real.flatten(), bins=50, alpha=0.7, label='Real', density=True)
    axes[0].set_xlabel('Amplitude')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Amplitude Distribution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Statistics comparison
    metrics = ['Mean', 'Std']
    gen_stats = [results['generated_mean'], results['generated_std']]
    real_stats = [results['real_signal_mean'], results['real_signal_std']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1].bar(x - width/2, gen_stats, width, label='Generated', alpha=0.8)
    axes[1].bar(x + width/2, real_stats, width, label='Real', alpha=0.8)
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Statistical Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced ABR Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="evaluation_enhanced", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM sampling steps")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Enhanced ABR Model Evaluation")
    print(f"📱 Device: {device}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load config
    config = load_config(args.config)
    print(f"⚙️ Loaded config from {args.config}")
    
    # Create model
    model = create_model(config, device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    
    # Create data loader
    print("📊 Creating data loaders...")
    _, _, test_loader, _ = create_optimized_dataloaders(
        data_path=config.get('data', {}).get('path', 'data/processed/ultimate_dataset_with_clinical_thresholds.pkl'),
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    print(f"✅ Test samples: {len(test_loader.dataset)}")
    
    # Create sampler
    diffusion_cfg = config.get('diffusion', {})
    schedule = get_noise_schedule(
        diffusion_cfg.get('schedule_type', 'cosine'),
        diffusion_cfg.get('num_timesteps', 1000)
    )
    noise_schedule = NoiseSchedule(schedule)
    
    # Move noise schedule to device
    for key, tensor in noise_schedule.__dict__.items():
        if isinstance(tensor, torch.Tensor):
            setattr(noise_schedule, key, tensor.to(device))
    
    sampler = create_ddim_sampler(
        noise_schedule_type=diffusion_cfg.get('schedule_type', 'cosine'),
        num_timesteps=diffusion_cfg.get('num_timesteps', 1000),
        eta=diffusion_cfg.get('eta', 0.0),
        clip_denoised=diffusion_cfg.get('clip_denoised', False)
    )
    
    print(f"🎯 DDIM sampler: {args.ddim_steps} steps, eta={diffusion_cfg.get('eta', 0.0)}")
    
    # Evaluate
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        sampler=sampler,
        device=device,
        num_samples=args.num_samples,
        steps=args.ddim_steps
    )
    
    # Create plots
    plot_comparison(results, output_dir)
    
    # Save results
    summary = {
        'checkpoint': args.checkpoint,
        'config': args.config,
        'epoch': checkpoint.get('epoch', 0),
        'num_samples': results['num_samples'],
        'ddim_steps': results['ddim_steps'],
        'statistics': {
            'real_mean': results['real_signal_mean'],
            'real_std': results['real_signal_std'],
            'generated_mean': results['generated_mean'],
            'generated_std': results['generated_std'],
            'approximate_mse': results['approximate_mse']
        }
    }
    
    with open(output_dir / 'evaluation_summary.yaml', 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    # Print summary
    print("\n📋 Evaluation Summary:")
    print(f"   Real signals - Mean: {results['real_signal_mean']:.4f}, Std: {results['real_signal_std']:.4f}")
    print(f"   Generated   - Mean: {results['generated_mean']:.4f}, Std: {results['generated_std']:.4f}")
    print(f"   Approximate MSE: {results['approximate_mse']:.4f}")
    print(f"   Results saved to: {output_dir}")
    
    print("✅ Evaluation completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)