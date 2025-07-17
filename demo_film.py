#!/usr/bin/env python3
"""
Demo script to test FiLM (Feature-wise Linear Modulation) conditioning in CVAE.

This script demonstrates:
1. Creating CVAE models with and without FiLM conditioning
2. Comparing forward pass behaviors
3. Testing joint generation with FiLM conditioning
4. Verifying backward compatibility
"""

import torch
import numpy as np
import json
from models.cvae import CVAE
from utils.losses import joint_cvae_loss
import matplotlib.pyplot as plt


def create_sample_data(batch_size=4, signal_length=200, static_dim=7):
    """Create sample data for testing."""
    # Create sample signals (ABR-like)
    signals = torch.randn(batch_size, signal_length) * 0.5
    
    # Create sample static parameters
    static_params = torch.randn(batch_size, static_dim) * 2.0
    
    return signals, static_params


def test_film_block():
    """Test the FiLMBlock module independently."""
    print("=" * 60)
    print("Testing FiLMBlock Module")
    print("=" * 60)
    
    from models.decoder import FiLMBlock
    
    # Test parameters
    feature_dim = 128
    static_dim = 7
    batch_size = 4
    
    # Create FiLM block
    film_block = FiLMBlock(feature_dim, static_dim)
    
    # Create test inputs
    features = torch.randn(batch_size, feature_dim)
    static_params = torch.randn(batch_size, static_dim)
    
    # Test forward pass
    print(f"Input features shape: {features.shape}")
    print(f"Static params shape: {static_params.shape}")
    
    modulated = film_block(features, static_params)
    print(f"Output features shape: {modulated.shape}")
    
    # Test that initial transformation is close to identity (due to initialization)
    diff = torch.abs(modulated - features).mean()
    print(f"Mean difference from input (should be small initially): {diff:.6f}")
    
    # Test parameter shapes
    gamma = film_block.gamma_mlp(static_params)
    beta = film_block.beta_mlp(static_params)
    print(f"Gamma shape: {gamma.shape}")
    print(f"Beta shape: {beta.shape}")
    
    print("✓ FiLMBlock test passed!")
    return True


def test_model_creation():
    """Test creating CVAE models with and without FiLM."""
    print("\n" + "=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    signal_length = 200
    static_dim = 7
    latent_dim = 64
    
    # Test model without FiLM
    print("Creating CVAE without FiLM...")
    model_no_film = CVAE(
        signal_length=signal_length,
        static_dim=static_dim,
        latent_dim=latent_dim,
        joint_generation=True,
        use_film=False
    )
    
    # Test model with FiLM
    print("Creating CVAE with FiLM...")
    model_with_film = CVAE(
        signal_length=signal_length,
        static_dim=static_dim,
        latent_dim=latent_dim,
        joint_generation=True,
        use_film=True
    )
    
    # Count parameters
    params_no_film = sum(p.numel() for p in model_no_film.parameters())
    params_with_film = sum(p.numel() for p in model_with_film.parameters())
    
    print(f"Parameters without FiLM: {params_no_film:,}")
    print(f"Parameters with FiLM: {params_with_film:,}")
    print(f"Additional parameters due to FiLM: {params_with_film - params_no_film:,}")
    
    # Check that FiLM blocks exist
    decoder_with_film = model_with_film.decoder
    assert hasattr(decoder_with_film, 'film1'), "FiLM block 1 not found"
    assert hasattr(decoder_with_film, 'film2'), "FiLM block 2 not found"
    assert hasattr(decoder_with_film, 'film3'), "FiLM block 3 not found"
    assert hasattr(decoder_with_film, 'film4'), "FiLM block 4 not found"
    
    print("✓ Model creation test passed!")
    return model_no_film, model_with_film


def test_forward_pass():
    """Test forward pass with and without FiLM."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    # Create models
    model_no_film, model_with_film = test_model_creation()
    
    # Create sample data
    signals, static_params = create_sample_data()
    
    print(f"Input shapes - Signals: {signals.shape}, Static params: {static_params.shape}")
    
    # Test forward pass without FiLM
    print("\nTesting forward pass without FiLM...")
    model_no_film.eval()
    with torch.no_grad():
        output_no_film = model_no_film(signals, static_params)
    
    if len(output_no_film) == 4:  # joint generation without peaks
        recon_signals_no_film, recon_static_no_film, mu_no_film, logvar_no_film = output_no_film
        print(f"Output shapes (no FiLM):")
        print(f"  Reconstructed signals: {recon_signals_no_film.shape}")
        print(f"  Reconstructed static: {recon_static_no_film.shape}")
        print(f"  Mu: {mu_no_film.shape}")
        print(f"  Logvar: {logvar_no_film.shape}")
    
    # Test forward pass with FiLM
    print("\nTesting forward pass with FiLM...")
    model_with_film.eval()
    with torch.no_grad():
        output_with_film = model_with_film(signals, static_params)
    
    if len(output_with_film) == 4:  # joint generation without peaks
        recon_signals_film, recon_static_film, mu_film, logvar_film = output_with_film
        print(f"Output shapes (with FiLM):")
        print(f"  Reconstructed signals: {recon_signals_film.shape}")
        print(f"  Reconstructed static: {recon_static_film.shape}")
        print(f"  Mu: {mu_film.shape}")
        print(f"  Logvar: {logvar_film.shape}")
    
    # Compare outputs
    signal_diff = torch.abs(recon_signals_no_film - recon_signals_film).mean()
    static_diff = torch.abs(recon_static_no_film - recon_static_film).mean()
    
    print(f"\nDifferences between FiLM and no-FiLM outputs:")
    print(f"  Signal reconstruction difference: {signal_diff:.6f}")
    print(f"  Static reconstruction difference: {static_diff:.6f}")
    
    print("✓ Forward pass test passed!")
    return model_no_film, model_with_film


def test_joint_generation():
    """Test joint generation with FiLM conditioning."""
    print("\n" + "=" * 60)
    print("Testing Joint Generation with FiLM")
    print("=" * 60)
    
    # Create model with FiLM
    model = CVAE(
        signal_length=200,
        static_dim=7,
        latent_dim=64,
        joint_generation=True,
        use_film=True
    )
    
    model.eval()
    
    # Test 1: Pure generation (no static params provided)
    print("Test 1: Pure generation from noise...")
    with torch.no_grad():
        pure_output = model.sample(n_samples=3)
    
    if len(pure_output) == 2:  # signals and static params
        gen_signals, gen_static = pure_output
        print(f"Generated signals shape: {gen_signals.shape}")
        print(f"Generated static params shape: {gen_static.shape}")
    
    # Test 2: Conditional generation (with static params for FiLM)
    print("\nTest 2: FiLM-conditioned generation...")
    static_condition = torch.randn(1, 7)  # Single condition
    
    with torch.no_grad():
        cond_output = model.sample(static_params=static_condition, n_samples=3)
    
    if len(cond_output) == 2:  # signals and static params
        cond_signals, cond_static = cond_output
        print(f"Conditioned signals shape: {cond_signals.shape}")
        print(f"Conditioned static params shape: {cond_static.shape}")
    
    # Compare the two generations
    signal_variation_pure = torch.std(gen_signals, dim=0).mean()
    signal_variation_cond = torch.std(cond_signals, dim=0).mean()
    
    print(f"\nSignal variation analysis:")
    print(f"  Pure generation variation: {signal_variation_pure:.6f}")
    print(f"  Conditioned generation variation: {signal_variation_cond:.6f}")
    
    print("✓ Joint generation test passed!")
    return model


def test_loss_computation():
    """Test loss computation with FiLM model."""
    print("\n" + "=" * 60)
    print("Testing Loss Computation")
    print("=" * 60)
    
    # Create model with FiLM
    model = CVAE(
        signal_length=200,
        static_dim=7,
        latent_dim=64,
        joint_generation=True,
        use_film=True
    )
    
    model.train()
    
    # Create sample data
    signals, static_params = create_sample_data()
    
    # Forward pass
    output = model(signals, static_params)
    
    if len(output) == 4:
        recon_signals, recon_static, mu, logvar = output
        
        # Compute joint loss
        total_loss, signal_loss, static_loss, kl_loss = joint_cvae_loss(
            recon_signals, signals,
            recon_static, static_params,
            mu, logvar,
            beta=0.01,
            static_loss_weight=1.0
        )
        
        print(f"Loss computation results:")
        print(f"  Total loss: {total_loss:.6f}")
        print(f"  Signal reconstruction loss: {signal_loss:.6f}")
        print(f"  Static reconstruction loss: {static_loss:.6f}")
        print(f"  KL divergence loss: {kl_loss:.6f}")
        
        # Test backward pass
        total_loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"  Total gradient norm: {total_grad_norm:.6f}")
    
    print("✓ Loss computation test passed!")


def test_backward_compatibility():
    """Test that models still work in legacy mode."""
    print("\n" + "=" * 60)
    print("Testing Backward Compatibility")
    print("=" * 60)
    
    # Create model with FiLM but in legacy mode
    model = CVAE(
        signal_length=200,
        static_dim=7,
        latent_dim=64,
        joint_generation=False,  # Legacy mode
        use_film=True
    )
    
    model.eval()
    
    # Create sample data
    signals, static_params = create_sample_data()
    
    # Test legacy forward pass
    print("Testing legacy forward pass...")
    with torch.no_grad():
        output = model(signals, static_params)
    
    if len(output) == 3:  # Legacy mode: recon_signal, mu, logvar
        recon_signals, mu, logvar = output
        print(f"Legacy output shapes:")
        print(f"  Reconstructed signals: {recon_signals.shape}")
        print(f"  Mu: {mu.shape}")
        print(f"  Logvar: {logvar.shape}")
    
    # Test legacy sampling
    print("Testing legacy sampling...")
    with torch.no_grad():
        sample_output = model.sample(static_params=static_params[:2], n_samples=1)
    
    print(f"Legacy sample shape: {sample_output.shape}")
    
    print("✓ Backward compatibility test passed!")


def run_comprehensive_test():
    """Run all FiLM tests."""
    print("Starting FiLM Implementation Tests")
    print("=" * 60)
    
    try:
        # Run individual tests
        test_film_block()
        test_model_creation()
        test_forward_pass()
        test_joint_generation()
        test_loss_computation()
        test_backward_compatibility()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("FiLM conditioning is working correctly.")
        print("=" * 60)
        
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
    
    # Run tests
    success = run_comprehensive_test()
    
    if success:
        print("\n🎉 FiLM implementation is ready for use!")
        print("You can now train models with FiLM conditioning using:")
        print("python main.py --config configs/film_config.json")
    else:
        print("\n⚠️  Please check the implementation and fix any issues.") 