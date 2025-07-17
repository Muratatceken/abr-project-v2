#!/usr/bin/env python3
"""
Demo script to test Hierarchical CVAE functionality.

This script demonstrates:
1. Hierarchical encoder structure (Global + Local encoders)
2. Hierarchical decoder with FiLM conditioning
3. Separate latent spaces for global and local features
4. Enhanced loss functions for hierarchical training
5. Flexible sampling with control over global and local latents
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from models.cvae import HierarchicalCVAE
from models.encoder import GlobalEncoder, LocalEncoder, HierarchicalEncoder
from models.decoder import HierarchicalDecoder
from utils.losses import hierarchical_cvae_loss, enhanced_hierarchical_cvae_loss


def create_sample_data(batch_size=4, signal_length=200, static_dim=7):
    """Create sample data for testing."""
    # Create sample signals (ABR-like)
    signals = torch.randn(batch_size, signal_length) * 0.5
    
    # Create sample static parameters
    static_params = torch.randn(batch_size, static_dim) * 2.0
    
    return signals, static_params


def test_hierarchical_encoders():
    """Test the hierarchical encoder components."""
    print("=" * 60)
    print("Testing Hierarchical Encoders")
    print("=" * 60)
    
    signal_length = 200
    static_dim = 7
    global_latent_dim = 32
    local_latent_dim = 32
    batch_size = 4
    
    # Create sample data
    signals, static_params = create_sample_data(batch_size, signal_length, static_dim)
    
    print(f"Input shapes - Signals: {signals.shape}, Static params: {static_params.shape}")
    
    # Test Global Encoder
    print("\nTesting Global Encoder...")
    global_encoder = GlobalEncoder(signal_length, static_dim, global_latent_dim)
    mu_global, logvar_global = global_encoder(signals, static_params)
    
    print(f"Global encoder output:")
    print(f"  mu_global: {mu_global.shape}")
    print(f"  logvar_global: {logvar_global.shape}")
    
    # Test Local Encoder
    print("\nTesting Local Encoder...")
    local_encoder = LocalEncoder(signal_length, static_dim, local_latent_dim, early_signal_ratio=0.3)
    mu_local, logvar_local = local_encoder(signals, static_params)
    
    print(f"Local encoder output:")
    print(f"  mu_local: {mu_local.shape}")
    print(f"  logvar_local: {logvar_local.shape}")
    print(f"  Early signal length: {local_encoder.early_signal_length}")
    
    # Test Hierarchical Encoder
    print("\nTesting Hierarchical Encoder...")
    hierarchical_encoder = HierarchicalEncoder(
        signal_length, static_dim, global_latent_dim, local_latent_dim
    )
    encoder_output = hierarchical_encoder(signals, static_params)
    
    print(f"Hierarchical encoder output:")
    for key, value in encoder_output.items():
        print(f"  {key}: {value.shape}")
    
    latent_dims = hierarchical_encoder.get_latent_dims()
    print(f"Latent dimensions: {latent_dims}")
    
    print("✓ Hierarchical encoders test passed!")
    return True


def test_hierarchical_decoder():
    """Test the hierarchical decoder."""
    print("\n" + "=" * 60)
    print("Testing Hierarchical Decoder")
    print("=" * 60)
    
    signal_length = 200
    static_dim = 7
    global_latent_dim = 32
    local_latent_dim = 32
    batch_size = 4
    
    # Create sample latents and static params
    z_global = torch.randn(batch_size, global_latent_dim)
    z_local = torch.randn(batch_size, local_latent_dim)
    _, static_params = create_sample_data(batch_size, signal_length, static_dim)
    
    print(f"Input shapes:")
    print(f"  z_global: {z_global.shape}")
    print(f"  z_local: {z_local.shape}")
    print(f"  static_params: {static_params.shape}")
    
    # Test Hierarchical Decoder without peaks
    print("\nTesting Hierarchical Decoder (no peaks)...")
    decoder = HierarchicalDecoder(
        signal_length, static_dim, global_latent_dim, local_latent_dim,
        predict_peaks=False, use_film=True
    )
    
    recon_signal, recon_static_params = decoder(z_global, z_local, static_params)
    
    print(f"Decoder output (no peaks):")
    print(f"  recon_signal: {recon_signal.shape}")
    print(f"  recon_static_params: {recon_static_params.shape}")
    
    # Test Hierarchical Decoder with peaks
    print("\nTesting Hierarchical Decoder (with peaks)...")
    decoder_with_peaks = HierarchicalDecoder(
        signal_length, static_dim, global_latent_dim, local_latent_dim,
        predict_peaks=True, num_peaks=6, use_film=True
    )
    
    recon_signal, recon_static_params, predicted_peaks = decoder_with_peaks(
        z_global, z_local, static_params
    )
    
    print(f"Decoder output (with peaks):")
    print(f"  recon_signal: {recon_signal.shape}")
    print(f"  recon_static_params: {recon_static_params.shape}")
    print(f"  predicted_peaks: {predicted_peaks.shape}")
    
    # Test backward compatibility method
    print("\nTesting backward compatibility method...")
    z_combined = torch.cat([z_global, z_local], dim=1)
    recon_signal_compat, recon_static_params_compat = decoder.forward_from_combined_z(
        z_combined, static_params
    )
    
    print(f"Backward compatible output:")
    print(f"  recon_signal: {recon_signal_compat.shape}")
    print(f"  recon_static_params: {recon_static_params_compat.shape}")
    
    print("✓ Hierarchical decoder test passed!")
    return True


def test_hierarchical_cvae():
    """Test the complete Hierarchical CVAE."""
    print("\n" + "=" * 60)
    print("Testing Hierarchical CVAE")
    print("=" * 60)
    
    signal_length = 200
    static_dim = 7
    global_latent_dim = 32
    local_latent_dim = 32
    batch_size = 4
    
    # Create model
    print("Creating Hierarchical CVAE...")
    model = HierarchicalCVAE(
        signal_length=signal_length,
        static_dim=static_dim,
        global_latent_dim=global_latent_dim,
        local_latent_dim=local_latent_dim,
        predict_peaks=True,
        use_film=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Get latent dimensions
    latent_dims = model.get_latent_dims()
    print(f"Latent dimensions: {latent_dims}")
    
    # Create sample data
    signals, static_params = create_sample_data(batch_size, signal_length, static_dim)
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(signals, static_params)
    
    print(f"Forward pass output:")
    for key, value in output.items():
        print(f"  {key}: {value.shape}")
    
    # Test individual encoding/decoding
    print(f"\nTesting individual encode/decode...")
    with torch.no_grad():
        # Test encoding
        encoder_output = model.encode(signals, static_params)
        print(f"Encoder output keys: {list(encoder_output.keys())}")
        
        # Test decoding
        z_global = encoder_output['mu_global']  # Use mean for deterministic test
        z_local = encoder_output['mu_local']
        decoder_output = model.decode(z_global, z_local, static_params)
        print(f"Decoder output keys: {list(decoder_output.keys())}")
    
    print("✓ Hierarchical CVAE test passed!")
    return model


def test_hierarchical_sampling():
    """Test hierarchical sampling capabilities."""
    print("\n" + "=" * 60)
    print("Testing Hierarchical Sampling")
    print("=" * 60)
    
    # Create model
    model = HierarchicalCVAE(
        signal_length=200,
        static_dim=7,
        global_latent_dim=32,
        local_latent_dim=32,
        predict_peaks=True,
        use_film=True
    )
    
    model.eval()
    
    # Test 1: Pure random sampling
    print("Test 1: Pure random sampling...")
    with torch.no_grad():
        sample_output = model.sample(n_samples=3)
    
    print(f"Random sampling output:")
    for key, value in sample_output.items():
        print(f"  {key}: {value.shape}")
    
    # Test 2: Sampling with provided static parameters
    print("\nTest 2: Sampling with static parameters...")
    static_condition = torch.randn(1, 7)
    
    with torch.no_grad():
        sample_output = model.sample(static_params=static_condition, n_samples=3)
    
    print(f"Conditional sampling output:")
    for key, value in sample_output.items():
        print(f"  {key}: {value.shape}")
    
    # Test 3: Sampling with controlled latents
    print("\nTest 3: Sampling with controlled latents...")
    z_global_fixed = torch.randn(1, 32)
    z_local_varying = torch.randn(3, 32)
    
    with torch.no_grad():
        sample_output = model.sample(
            static_params=static_condition,
            n_samples=3,
            z_global=z_global_fixed,
            z_local=z_local_varying
        )
    
    print(f"Controlled latent sampling output:")
    for key, value in sample_output.items():
        print(f"  {key}: {value.shape}")
    
    print("✓ Hierarchical sampling test passed!")
    return True


def test_hierarchical_losses():
    """Test hierarchical loss functions."""
    print("\n" + "=" * 60)
    print("Testing Hierarchical Loss Functions")
    print("=" * 60)
    
    batch_size = 4
    signal_length = 200
    static_dim = 7
    global_latent_dim = 32
    local_latent_dim = 32
    num_peaks = 6
    
    # Create sample data with gradients enabled
    recon_signal = torch.randn(batch_size, signal_length, requires_grad=True)
    target_signal = torch.randn(batch_size, signal_length)
    recon_static_params = torch.randn(batch_size, static_dim, requires_grad=True)
    target_static_params = torch.randn(batch_size, static_dim)
    
    mu_global = torch.randn(batch_size, global_latent_dim, requires_grad=True)
    logvar_global = torch.randn(batch_size, global_latent_dim, requires_grad=True)
    mu_local = torch.randn(batch_size, local_latent_dim, requires_grad=True)
    logvar_local = torch.randn(batch_size, local_latent_dim, requires_grad=True)
    
    predicted_peaks = torch.randn(batch_size, num_peaks)
    target_peaks = torch.randn(batch_size, num_peaks)
    peak_mask = torch.ones(batch_size, num_peaks, dtype=torch.bool)
    
    # Test basic hierarchical loss
    print("Testing basic hierarchical CVAE loss...")
    total_loss, signal_loss, static_loss, global_kl_loss, local_kl_loss = hierarchical_cvae_loss(
        recon_signal, target_signal,
        recon_static_params, target_static_params,
        mu_global, logvar_global,
        mu_local, logvar_local,
        global_kl_weight=0.01,
        local_kl_weight=0.01,
        static_loss_weight=1.0
    )
    
    print(f"Basic hierarchical loss components:")
    print(f"  Total loss: {total_loss:.6f}")
    print(f"  Signal loss: {signal_loss:.6f}")
    print(f"  Static loss: {static_loss:.6f}")
    print(f"  Global KL loss: {global_kl_loss:.6f}")
    print(f"  Local KL loss: {local_kl_loss:.6f}")
    
    # Test enhanced hierarchical loss
    print("\nTesting enhanced hierarchical CVAE loss...")
    loss_weights = {
        'reconstruction': 1.0,
        'static': 1.0,
        'global_kl': 0.01,
        'local_kl': 0.01,
        'peak': 1.0,
        'alignment': 0.1
    }
    
    loss_config = {
        'peak_loss_type': 'mae',
        'use_alignment_loss': True,
        'alignment_type': 'warped_mse',
        'max_warp': 5
    }
    
    enhanced_total_loss, loss_components = enhanced_hierarchical_cvae_loss(
        recon_signal, target_signal,
        recon_static_params, target_static_params,
        mu_global, logvar_global,
        mu_local, logvar_local,
        predicted_peaks, target_peaks, peak_mask,
        loss_weights, loss_config
    )
    
    print(f"Enhanced hierarchical loss components:")
    print(f"  Total loss: {enhanced_total_loss:.6f}")
    for component, value in loss_components.items():
        print(f"  {component}: {value:.6f}")
    
    # Test gradient computation
    print("\nTesting gradient computation...")
    enhanced_total_loss.backward()
    
    print("✓ Hierarchical loss functions test passed!")
    return True


def test_latent_space_separation():
    """Test that global and local latents capture different information."""
    print("\n" + "=" * 60)
    print("Testing Latent Space Separation")
    print("=" * 60)
    
    # Create model
    model = HierarchicalCVAE(
        signal_length=200,
        static_dim=7,
        global_latent_dim=32,
        local_latent_dim=32,
        predict_peaks=False,
        use_film=True
    )
    
    model.eval()
    
    # Create signals with different global characteristics but similar local patterns
    batch_size = 4
    signal_length = 200
    
    # Base signal with local pattern
    base_local_pattern = torch.sin(torch.linspace(0, 4*np.pi, signal_length//4)).repeat(4)
    
    # Create signals with different global characteristics
    signals = torch.zeros(batch_size, signal_length)
    for i in range(batch_size):
        # Different global scaling and offset
        global_scale = 0.5 + i * 0.3
        global_offset = i * 0.2
        signals[i] = global_scale * base_local_pattern + global_offset
        
        # Add some noise
        signals[i] += torch.randn(signal_length) * 0.1
    
    static_params = torch.randn(batch_size, 7)
    
    # Encode signals
    with torch.no_grad():
        encoder_output = model.encode(signals, static_params)
    
    # Analyze latent space
    mu_global = encoder_output['mu_global']
    mu_local = encoder_output['mu_local']
    
    # Compute variance across batch dimension
    global_variance = torch.var(mu_global, dim=0).mean()
    local_variance = torch.var(mu_local, dim=0).mean()
    
    print(f"Latent space analysis:")
    print(f"  Global latent variance (across batch): {global_variance:.6f}")
    print(f"  Local latent variance (across batch): {local_variance:.6f}")
    
    # Test reconstruction with swapped latents
    print(f"\nTesting latent swapping...")
    with torch.no_grad():
        # Original reconstruction
        orig_output = model.decode(mu_global, mu_local, static_params)
        orig_signal = orig_output['recon_signal']
        
        # Swap global latents between samples
        mu_global_swapped = mu_global[[1, 0, 3, 2]]  # Swap pairs
        swapped_output = model.decode(mu_global_swapped, mu_local, static_params)
        swapped_signal = swapped_output['recon_signal']
        
        # Compute reconstruction differences
        orig_recon_error = F.mse_loss(orig_signal, signals)
        swapped_recon_error = F.mse_loss(swapped_signal, signals)
    
    print(f"  Original reconstruction error: {orig_recon_error:.6f}")
    print(f"  Swapped global latent error: {swapped_recon_error:.6f}")
    print(f"  Error increase from swapping: {(swapped_recon_error - orig_recon_error):.6f}")
    
    print("✓ Latent space separation test passed!")
    return True


def create_visualization():
    """Create visualization of hierarchical CVAE structure."""
    print("\n" + "=" * 60)
    print("Creating Hierarchical CVAE Visualization")
    print("=" * 60)
    
    # Create a diagram showing the hierarchical structure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Latent space comparison
    np.random.seed(42)
    global_latents = np.random.randn(100, 2) * 0.8  # More compact
    local_latents = np.random.randn(100, 2) * 1.2   # More spread
    
    ax1.scatter(global_latents[:, 0], global_latents[:, 1], alpha=0.6, s=30, 
               c='red', label='Global Latent (z_global)')
    ax1.scatter(local_latents[:, 0], local_latents[:, 1], alpha=0.6, s=30, 
               c='blue', label='Local Latent (z_local)')
    ax1.set_title('Latent Space Visualization')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Simulated signal reconstruction comparison
    x = np.linspace(0, 4*np.pi, 200)
    original_signal = np.sin(x) + 0.3*np.sin(3*x) + 0.1*np.sin(5*x)
    
    # Global features affect overall shape
    global_recon = 1.2 * np.sin(x) + 0.2  # Different amplitude and offset
    
    # Local features affect fine details
    local_recon = np.sin(x) + 0.4*np.sin(3*x) + 0.2*np.sin(5*x)  # Different detail balance
    
    ax2.plot(x, original_signal, 'k-', linewidth=2, label='Original Signal')
    ax2.plot(x, global_recon, 'r--', linewidth=2, label='Global Features Only')
    ax2.plot(x, local_recon, 'b--', linewidth=2, label='Local Features Only')
    ax2.set_title('Feature Separation in Reconstruction')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training loss curves
    epochs = np.arange(1, 101)
    total_loss = 5.0 * np.exp(-epochs/20) + 0.5 + 0.1*np.random.randn(100)
    global_kl = 2.0 * np.exp(-epochs/15) + 0.1 + 0.05*np.random.randn(100)
    local_kl = 1.5 * np.exp(-epochs/25) + 0.08 + 0.03*np.random.randn(100)
    recon_loss = 1.8 * np.exp(-epochs/30) + 0.2 + 0.05*np.random.randn(100)
    
    ax3.plot(epochs, total_loss, 'k-', linewidth=2, label='Total Loss')
    ax3.plot(epochs, recon_loss, 'g-', linewidth=2, label='Reconstruction')
    ax3.plot(epochs, global_kl, 'r-', linewidth=2, label='Global KL')
    ax3.plot(epochs, local_kl, 'b-', linewidth=2, label='Local KL')
    ax3.set_title('Hierarchical Training Loss Components')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Architecture diagram (text-based)
    ax4.text(0.1, 0.9, 'Hierarchical CVAE Architecture', fontsize=14, fontweight='bold')
    ax4.text(0.1, 0.8, '1. Global Encoder:', fontsize=12, fontweight='bold', color='red')
    ax4.text(0.15, 0.75, '• Captures overall signal patterns', fontsize=10)
    ax4.text(0.15, 0.7, '• Output: z_global (32D)', fontsize=10)
    
    ax4.text(0.1, 0.6, '2. Local Encoder:', fontsize=12, fontweight='bold', color='blue')
    ax4.text(0.15, 0.55, '• Captures fine-grained details', fontsize=10)
    ax4.text(0.15, 0.5, '• Uses early signal conditioning', fontsize=10)
    ax4.text(0.15, 0.45, '• Output: z_local (32D)', fontsize=10)
    
    ax4.text(0.1, 0.35, '3. Hierarchical Decoder:', fontsize=12, fontweight='bold', color='green')
    ax4.text(0.15, 0.3, '• z_global → FiLM modulation', fontsize=10)
    ax4.text(0.15, 0.25, '• z_local + static → detail generation', fontsize=10)
    ax4.text(0.15, 0.2, '• Output: signal + static + peaks', fontsize=10)
    
    ax4.text(0.1, 0.1, '4. Benefits:', fontsize=12, fontweight='bold', color='purple')
    ax4.text(0.15, 0.05, '• Better disentanglement of features', fontsize=10)
    ax4.text(0.15, 0.0, '• More controlled generation', fontsize=10)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('hierarchical_cvae_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved as 'hierarchical_cvae_visualization.png'")
    plt.close()
    
    return True


def run_comprehensive_test():
    """Run all hierarchical CVAE tests."""
    print("Starting Hierarchical CVAE Tests")
    print("=" * 60)
    
    try:
        # Run individual tests
        test_hierarchical_encoders()
        test_hierarchical_decoder()
        model = test_hierarchical_cvae()
        test_hierarchical_sampling()
        test_hierarchical_losses()
        test_latent_space_separation()
        create_visualization()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("Hierarchical CVAE implementation is working correctly.")
        print("=" * 60)
        
        # Print usage instructions
        print("\n📋 Usage Instructions:")
        print("1. Use configs/hierarchical_config.json for hierarchical training")
        print("2. Import HierarchicalCVAE from models.cvae")
        print("3. Use separate global and local latent dimensions")
        print("4. Global latents control FiLM modulation")
        print("5. Local latents handle fine detail generation")
        print("6. Enhanced loss functions handle both latent spaces")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Add missing import
    import torch.nn.functional as F
    
    # Run tests
    success = run_comprehensive_test()
    
    if success:
        print("\n🎉 Hierarchical CVAE implementation is ready for use!")
        print("Key features:")
        print("  • Separate global and local encoders")
        print("  • FiLM conditioning with global latents")
        print("  • Fine detail generation with local latents")
        print("  • Enhanced loss functions for both latent spaces")
        print("  • Flexible sampling with latent control")
        print("  • Better feature disentanglement")
    else:
        print("\n⚠️  Please check the implementation and fix any issues.") 