#!/usr/bin/env python3
"""
Demo script for joint generation of ABR signals and static parameters using CVAE.

This script demonstrates the new joint generation capabilities where the model
can generate both ABR signals and their corresponding static parameters from
the latent space alone.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Add project root to path
import sys
sys.path.append('.')

from models.cvae import CVAE
from training.dataset import ABRDataset
from utils.losses import joint_cvae_loss, static_params_loss
from utils.data_utils import get_dataset_info


def test_joint_generation():
    """Test the joint generation functionality."""
    print("Testing Joint Generation CVAE")
    print("=" * 50)
    
    # Load dataset info
    data_path = "data/processed/processed_data_categorical_200ts.pkl"
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        print("Please ensure the data is preprocessed first.")
        return
    
    # Create dataset to get dimensions
    dataset = ABRDataset(data_path, return_peaks=True)
    sample_info = dataset.get_sample_info()
    
    print(f"Dataset Info:")
    print(f"  Signal length: {sample_info['signal_length']}")
    print(f"  Static params dim: {sample_info['static_params_dim']}")
    print(f"  Number of peaks: {sample_info['num_peaks']}")
    print(f"  Number of samples: {sample_info['num_samples']}")
    
    # Model parameters
    latent_dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models for comparison
    print("\nCreating models...")
    
    # Legacy model (conditions decoder on static params)
    legacy_model = CVAE(
        signal_length=sample_info['signal_length'],
        static_dim=sample_info['static_params_dim'],
        latent_dim=latent_dim,
        predict_peaks=True,
        num_peaks=sample_info['num_peaks'],
        joint_generation=False  # Legacy mode
    ).to(device)
    
    # Joint generation model (generates both signals and static params)
    joint_model = CVAE(
        signal_length=sample_info['signal_length'],
        static_dim=sample_info['static_params_dim'],
        latent_dim=latent_dim,
        predict_peaks=True,
        num_peaks=sample_info['num_peaks'],
        joint_generation=True  # Joint generation mode
    ).to(device)
    
    print(f"Legacy model parameters: {sum(p.numel() for p in legacy_model.parameters()):,}")
    print(f"Joint model parameters: {sum(p.numel() for p in joint_model.parameters()):,}")
    
    # Test forward pass with sample data
    print("\nTesting forward pass...")
    
    # Get a sample batch
    sample_batch = dataset[0]
    batch_size = 4
    
    # Create a mini-batch
    signals = torch.stack([dataset[i]['signal'] for i in range(batch_size)]).to(device)
    static_params = torch.stack([dataset[i]['static_params'] for i in range(batch_size)]).to(device)
    peaks = torch.stack([dataset[i]['peaks'] for i in range(batch_size)]).to(device)
    peak_masks = torch.stack([dataset[i]['peak_mask'] for i in range(batch_size)]).to(device)
    
    print(f"Input shapes:")
    print(f"  Signals: {signals.shape}")
    print(f"  Static params: {static_params.shape}")
    print(f"  Peaks: {peaks.shape}")
    
    # Test legacy model
    print("\nTesting legacy model...")
    legacy_model.eval()
    with torch.no_grad():
        legacy_output = legacy_model(signals, static_params)
        if len(legacy_output) == 4:  # With peaks
            recon_signal_legacy, mu_legacy, logvar_legacy, predicted_peaks_legacy = legacy_output
            print(f"  Legacy output shapes: signal={recon_signal_legacy.shape}, peaks={predicted_peaks_legacy.shape}")
        else:
            recon_signal_legacy, mu_legacy, logvar_legacy = legacy_output
            print(f"  Legacy output shapes: signal={recon_signal_legacy.shape}")
    
    # Test joint generation model
    print("\nTesting joint generation model...")
    joint_model.eval()
    with torch.no_grad():
        joint_output = joint_model(signals, static_params)
        if len(joint_output) == 5:  # With peaks
            recon_signal_joint, recon_static_joint, mu_joint, logvar_joint, predicted_peaks_joint = joint_output
            print(f"  Joint output shapes: signal={recon_signal_joint.shape}, static={recon_static_joint.shape}, peaks={predicted_peaks_joint.shape}")
        else:
            recon_signal_joint, recon_static_joint, mu_joint, logvar_joint = joint_output
            print(f"  Joint output shapes: signal={recon_signal_joint.shape}, static={recon_static_joint.shape}")
    
    # Test loss functions
    print("\nTesting loss functions...")
    
    # Joint loss
    joint_total_loss, signal_recon_loss, static_recon_loss, kl_loss = joint_cvae_loss(
        recon_signal_joint, signals, recon_static_joint, static_params, 
        mu_joint, logvar_joint, beta=1.0, static_loss_weight=1.0
    )
    
    print(f"Joint loss components:")
    print(f"  Total loss: {joint_total_loss.item():.4f}")
    print(f"  Signal reconstruction loss: {signal_recon_loss.item():.4f}")
    print(f"  Static reconstruction loss: {static_recon_loss.item():.4f}")
    print(f"  KL loss: {kl_loss.item():.4f}")
    
    # Test sampling
    print("\nTesting sampling...")
    
    # Legacy sampling (requires static params)
    print("  Legacy sampling (with static params):")
    with torch.no_grad():
        legacy_samples = legacy_model.sample(static_params[:2], n_samples=1)
        if isinstance(legacy_samples, tuple):
            print(f"    Generated shapes: signal={legacy_samples[0].shape}, peaks={legacy_samples[1].shape}")
        else:
            print(f"    Generated shapes: signal={legacy_samples.shape}")
    
    # Joint sampling (no static params needed)
    print("  Joint sampling (no static params needed):")
    with torch.no_grad():
        joint_samples = joint_model.sample(n_samples=4)
        if len(joint_samples) == 3:  # With peaks
            gen_signals, gen_static_params, gen_peaks = joint_samples
            print(f"    Generated shapes: signal={gen_signals.shape}, static={gen_static_params.shape}, peaks={gen_peaks.shape}")
        else:
            gen_signals, gen_static_params = joint_samples
            print(f"    Generated shapes: signal={gen_signals.shape}, static={gen_static_params.shape}")
    
    # Visualize results
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Original vs Joint Reconstruction (Signal)
    sample_idx = 0
    axes[0, 0].plot(signals[sample_idx].cpu().numpy(), label='Original Signal', alpha=0.7)
    axes[0, 0].plot(recon_signal_joint[sample_idx].cpu().numpy(), label='Joint Reconstruction', alpha=0.7)
    axes[0, 0].set_title('Signal Reconstruction Comparison')
    axes[0, 0].set_xlabel('Time (samples)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Static Parameters Comparison
    static_names = ['Age', 'Intensity', 'Stimulus Rate', 'FMP', 'Res No', 'Hearing Loss']
    x_pos = np.arange(len(static_names))
    
    original_static = static_params[sample_idx].cpu().numpy()
    recon_static = recon_static_joint[sample_idx].cpu().numpy()
    
    width = 0.35
    axes[0, 1].bar(x_pos - width/2, original_static, width, label='Original', alpha=0.7)
    axes[0, 1].bar(x_pos + width/2, recon_static, width, label='Reconstructed', alpha=0.7)
    axes[0, 1].set_title('Static Parameters Reconstruction')
    axes[0, 1].set_xlabel('Parameter')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(static_names, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Generated Signals
    axes[1, 0].plot(gen_signals[0].cpu().numpy(), label='Generated 1', alpha=0.7)
    axes[1, 0].plot(gen_signals[1].cpu().numpy(), label='Generated 2', alpha=0.7)
    axes[1, 0].plot(gen_signals[2].cpu().numpy(), label='Generated 3', alpha=0.7)
    axes[1, 0].set_title('Generated Signals (Joint Mode)')
    axes[1, 0].set_xlabel('Time (samples)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Generated Static Parameters
    gen_static_np = gen_static_params.cpu().numpy()
    for i in range(min(3, len(gen_static_np))):
        axes[1, 1].plot(x_pos, gen_static_np[i], marker='o', label=f'Generated {i+1}', alpha=0.7)
    
    axes[1, 1].set_title('Generated Static Parameters')
    axes[1, 1].set_xlabel('Parameter')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(static_names, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_joint_generation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nDemo completed successfully!")
    print("Results saved to: demo_joint_generation_results.png")
    
    # Summary
    print("\n" + "=" * 50)
    print("JOINT GENERATION SUMMARY")
    print("=" * 50)
    print("✓ Legacy mode: Conditions decoder on static parameters")
    print("✓ Joint mode: Generates both signals and static parameters from latent space")
    print("✓ Backward compatibility: Both modes work with same codebase")
    print("✓ Loss functions: Added static parameter reconstruction loss")
    print("✓ Sampling: Joint mode can generate without providing static parameters")
    print("✓ Training: Updated trainer supports both modes")
    print("=" * 50)


if __name__ == "__main__":
    test_joint_generation() 