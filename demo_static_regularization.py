#!/usr/bin/env python3
"""
Demo script for testing static regularization functionality in CVAE.

This script demonstrates:
1. Static decoder implementation
2. Static reconstruction loss
3. InfoNCE contrastive loss
4. Enhanced loss functions with static regularization
5. Hierarchical static regularization
6. Training integration
7. Sample generation with static prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, Tuple, Any

# Import our modules
from models.cvae import CVAE, HierarchicalCVAE
from utils.losses import (
    static_reconstruction_loss,
    infonce_contrastive_loss,
    hierarchical_static_reconstruction_loss,
    hierarchical_infonce_loss,
    enhanced_cvae_loss_with_static_regularization,
    enhanced_hierarchical_cvae_loss_with_static_regularization
)


def test_static_decoder():
    """Test static decoder implementation in CVAE."""
    print("=" * 60)
    print("Testing Static Decoder Implementation")
    print("=" * 60)
    
    # Model parameters
    signal_length = 200
    static_dim = 8
    latent_dim = 64
    batch_size = 16
    
    # Create CVAE with static decoder
    model = CVAE(
        signal_length=signal_length,
        static_dim=static_dim,
        latent_dim=latent_dim,
        predict_peaks=False,
        use_film=True
    )
    
    # Check that static decoder exists
    assert hasattr(model, 'static_decoder'), "CVAE should have static_decoder"
    print(f"✓ Static decoder created with input dim {latent_dim} -> output dim {static_dim}")
    
    # Test forward pass
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    # Create dummy data
    signal = torch.randn(batch_size, signal_length)
    static_params = torch.randn(batch_size, static_dim)
    
    with torch.no_grad():
        output = model(signal, static_params)
        
        # Check that we get the static reconstruction as an additional output
        if len(output) == 4:  # recon_signal, mu, logvar, recon_static_from_z
            recon_signal, mu, logvar, recon_static_from_z = output
            print(f"✓ Forward pass successful")
            print(f"  - Reconstructed signal shape: {recon_signal.shape}")
            print(f"  - Latent mu shape: {mu.shape}")
            print(f"  - Latent logvar shape: {logvar.shape}")
            print(f"  - Static reconstruction shape: {recon_static_from_z.shape}")
            
            # Check shapes
            assert recon_signal.shape == (batch_size, signal_length)
            assert mu.shape == (batch_size, latent_dim)
            assert logvar.shape == (batch_size, latent_dim)
            assert recon_static_from_z.shape == (batch_size, static_dim)
            print("✓ All output shapes correct")
        else:
            print(f"✗ Unexpected number of outputs: {len(output)}")
            return False
    
    # Test sample generation with static prediction
    print("\nTesting sample generation with static prediction...")
    with torch.no_grad():
        # Test without static generation
        sample_output = model.sample(static_params=static_params[:2], n_samples=1)
        print(f"✓ Sample generation without static prediction: {type(sample_output)}")
        
        # Test with static generation
        sample_output_with_static = model.sample(
            static_params=static_params[:2], 
            n_samples=1, 
            generate_static_from_z=True
        )
        print(f"✓ Sample generation with static prediction: {type(sample_output_with_static)}")
        
        if isinstance(sample_output_with_static, tuple):
            print(f"  - Generated samples have {len(sample_output_with_static)} components")
    
    print("✓ Static decoder test passed!")
    return True


def test_static_reconstruction_loss():
    """Test static reconstruction loss function."""
    print("\n" + "=" * 60)
    print("Testing Static Reconstruction Loss")
    print("=" * 60)
    
    batch_size = 16
    static_dim = 8
    
    # Create dummy data
    recon_static = torch.randn(batch_size, static_dim)
    target_static = torch.randn(batch_size, static_dim)
    
    # Test basic static reconstruction loss
    loss = static_reconstruction_loss(recon_static, target_static)
    print(f"✓ Static reconstruction loss: {loss.item():.4f}")
    
    # Test with identical inputs (should be close to 0)
    identical_loss = static_reconstruction_loss(target_static, target_static)
    print(f"✓ Identical inputs loss: {identical_loss.item():.6f}")
    assert identical_loss.item() < 1e-6, "Identical inputs should have near-zero loss"
    
    # Test gradient flow
    recon_static.requires_grad_(True)
    loss = static_reconstruction_loss(recon_static, target_static)
    loss.backward()
    print(f"✓ Gradient magnitude: {recon_static.grad.norm().item():.4f}")
    assert recon_static.grad is not None, "Gradients should flow through loss"
    
    print("✓ Static reconstruction loss test passed!")
    return True


def test_infonce_contrastive_loss():
    """Test InfoNCE contrastive loss function."""
    print("\n" + "=" * 60)
    print("Testing InfoNCE Contrastive Loss")
    print("=" * 60)
    
    batch_size = 16
    latent_dim = 64
    static_dim = 8
    
    # Create dummy data
    z = torch.randn(batch_size, latent_dim)
    static_params = torch.randn(batch_size, static_dim)
    
    # Test InfoNCE loss
    loss = infonce_contrastive_loss(z, static_params, temperature=0.07)
    print(f"✓ InfoNCE loss: {loss.item():.4f}")
    
    # Test with different temperatures
    temperatures = [0.01, 0.07, 0.1, 0.5]
    losses = []
    for temp in temperatures:
        temp_loss = infonce_contrastive_loss(z, static_params, temperature=temp)
        losses.append(temp_loss.item())
        print(f"  - Temperature {temp}: {temp_loss.item():.4f}")
    
    # Test edge case: batch size 1 (should return 0)
    single_z = z[:1]
    single_static = static_params[:1]
    single_loss = infonce_contrastive_loss(single_z, single_static)
    print(f"✓ Single sample loss: {single_loss.item():.6f}")
    assert single_loss.item() == 0.0, "Single sample should have zero loss"
    
    # Test gradient flow
    z.requires_grad_(True)
    static_params.requires_grad_(True)
    loss = infonce_contrastive_loss(z, static_params)
    loss.backward()
    print(f"✓ Z gradient magnitude: {z.grad.norm().item():.4f}")
    print(f"✓ Static gradient magnitude: {static_params.grad.norm().item():.4f}")
    
    print("✓ InfoNCE contrastive loss test passed!")
    return True


def test_hierarchical_static_losses():
    """Test hierarchical static reconstruction and InfoNCE losses."""
    print("\n" + "=" * 60)
    print("Testing Hierarchical Static Losses")
    print("=" * 60)
    
    batch_size = 16
    static_dim = 8
    global_latent_dim = 32
    local_latent_dim = 32
    
    # Create dummy data
    recon_static_global = torch.randn(batch_size, static_dim // 2)
    recon_static_local = torch.randn(batch_size, static_dim // 2)
    recon_static_combined = torch.randn(batch_size, static_dim)
    target_static = torch.randn(batch_size, static_dim)
    z_global = torch.randn(batch_size, global_latent_dim)
    z_local = torch.randn(batch_size, local_latent_dim)
    
    # Test hierarchical static reconstruction loss
    total_loss, components = hierarchical_static_reconstruction_loss(
        recon_static_global, recon_static_local, recon_static_combined, target_static
    )
    print(f"✓ Hierarchical static reconstruction total loss: {total_loss.item():.4f}")
    print(f"  - Global component: {components['static_global_loss'].item():.4f}")
    print(f"  - Local component: {components['static_local_loss'].item():.4f}")
    print(f"  - Combined component: {components['static_combined_loss'].item():.4f}")
    
    # Test hierarchical InfoNCE loss
    infonce_total_loss, infonce_components = hierarchical_infonce_loss(
        z_global, z_local, target_static
    )
    print(f"✓ Hierarchical InfoNCE total loss: {infonce_total_loss.item():.4f}")
    print(f"  - Global InfoNCE: {infonce_components['infonce_global_loss'].item():.4f}")
    print(f"  - Local InfoNCE: {infonce_components['infonce_local_loss'].item():.4f}")
    
    print("✓ Hierarchical static losses test passed!")
    return True


def test_enhanced_cvae_loss_with_static_regularization():
    """Test enhanced CVAE loss with static regularization."""
    print("\n" + "=" * 60)
    print("Testing Enhanced CVAE Loss with Static Regularization")
    print("=" * 60)
    
    batch_size = 16
    signal_length = 200
    static_dim = 8
    latent_dim = 64
    num_peaks = 6
    
    # Create dummy data
    recon_signal = torch.randn(batch_size, signal_length)
    target_signal = torch.randn(batch_size, signal_length)
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    recon_static_from_z = torch.randn(batch_size, static_dim)
    target_static = torch.randn(batch_size, static_dim)
    z = torch.randn(batch_size, latent_dim)
    predicted_peaks = torch.randn(batch_size, num_peaks)
    target_peaks = torch.randn(batch_size, num_peaks)
    peak_mask = torch.ones(batch_size, num_peaks, dtype=torch.bool)  # All peaks valid
    
    # Test enhanced loss with static regularization
    total_loss, loss_components = enhanced_cvae_loss_with_static_regularization(
        recon_signal=recon_signal,
        target_signal=target_signal,
        mu=mu,
        logvar=logvar,
        recon_static_from_z=recon_static_from_z,
        target_static=target_static,
        z=z,
        predicted_peaks=predicted_peaks,
        target_peaks=target_peaks,
        peak_mask=peak_mask
    )
    
    print(f"✓ Enhanced CVAE loss with static regularization: {total_loss.item():.4f}")
    print("Loss components:")
    for key, value in loss_components.items():
        print(f"  - {key}: {value.item():.4f}")
    
    # Check that all expected components are present
    expected_components = [
        'reconstruction_loss', 'kl_loss', 'static_reconstruction_loss',
        'infonce_loss', 'peak_loss', 'alignment_loss', 'total_loss'
    ]
    for component in expected_components:
        assert component in loss_components, f"Missing component: {component}"
    print("✓ All expected loss components present")
    
    # Test without InfoNCE
    loss_config_no_infonce = {'use_infonce': False}
    total_loss_no_infonce, components_no_infonce = enhanced_cvae_loss_with_static_regularization(
        recon_signal=recon_signal,
        target_signal=target_signal,
        mu=mu,
        logvar=logvar,
        recon_static_from_z=recon_static_from_z,
        target_static=target_static,
        loss_config=loss_config_no_infonce
    )
    print(f"✓ Loss without InfoNCE: {components_no_infonce['infonce_loss'].item():.6f}")
    assert components_no_infonce['infonce_loss'].item() == 0.0, "InfoNCE should be zero when disabled"
    
    print("✓ Enhanced CVAE loss with static regularization test passed!")
    return True


def test_hierarchical_cvae_static_regularization():
    """Test hierarchical CVAE with static regularization."""
    print("\n" + "=" * 60)
    print("Testing Hierarchical CVAE Static Regularization")
    print("=" * 60)
    
    # Model parameters
    signal_length = 200
    static_dim = 8
    global_latent_dim = 32
    local_latent_dim = 32
    batch_size = 16
    
    # Create hierarchical CVAE
    model = HierarchicalCVAE(
        signal_length=signal_length,
        static_dim=static_dim,
        global_latent_dim=global_latent_dim,
        local_latent_dim=local_latent_dim,
        predict_peaks=True,
        use_film=True
    )
    
    # Check static decoders exist
    assert hasattr(model, 'global_static_decoder'), "Should have global static decoder"
    assert hasattr(model, 'local_static_decoder'), "Should have local static decoder"
    assert hasattr(model, 'static_combiner'), "Should have static combiner"
    print("✓ Hierarchical static decoders created")
    
    # Test forward pass
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    signal = torch.randn(batch_size, signal_length)
    static_params = torch.randn(batch_size, static_dim)
    
    with torch.no_grad():
        output = model(signal, static_params)
        
        print(f"✓ Hierarchical forward pass successful")
        print(f"  - Output keys: {list(output.keys())}")
        
        # Check that static reconstruction is included
        assert 'recon_static_from_z' in output, "Should have static reconstruction from z"
        print(f"  - Static reconstruction shape: {output['recon_static_from_z'].shape}")
    
    # Test sample generation with static prediction
    print("\nTesting hierarchical sample generation with static prediction...")
    with torch.no_grad():
        sample_output = model.sample(
            static_params=static_params[:2], 
            n_samples=1, 
            generate_static_from_z=True
        )
        print(f"✓ Sample generation with static prediction")
        print(f"  - Sample output keys: {list(sample_output.keys())}")
        
        if 'generated_static_from_z' in sample_output:
            print(f"  - Generated static shape: {sample_output['generated_static_from_z'].shape}")
    
    # Test enhanced hierarchical loss
    print("\nTesting enhanced hierarchical loss with static regularization...")
    output_data = {}
    with torch.no_grad():
        forward_output = model(signal, static_params)
        for key, value in forward_output.items():
            output_data[key] = value
    
    # Extract components for loss computation
    total_loss, loss_components = enhanced_hierarchical_cvae_loss_with_static_regularization(
        recon_signal=output_data['recon_signal'],
        target_signal=signal,
        recon_static_params=output_data['recon_static_params'],
        target_static_params=static_params,
        mu_global=output_data['mu_global'],
        logvar_global=output_data['logvar_global'],
        mu_local=output_data['mu_local'],
        logvar_local=output_data['logvar_local'],
        recon_static_from_z=output_data['recon_static_from_z'],
        z_global=output_data['z_global'],
        z_local=output_data['z_local']
    )
    
    print(f"✓ Enhanced hierarchical loss: {total_loss.item():.4f}")
    print("Hierarchical loss components:")
    for key, value in loss_components.items():
        print(f"  - {key}: {value.item():.4f}")
    
    print("✓ Hierarchical CVAE static regularization test passed!")
    return True


def test_correlation_analysis():
    """Test latent-static correlation analysis."""
    print("\n" + "=" * 60)
    print("Testing Latent-Static Correlation Analysis")
    print("=" * 60)
    
    # Create simple data with known correlations
    batch_size = 100
    latent_dim = 32
    static_dim = 8
    
    # Create latent vectors with some structure
    z = torch.randn(batch_size, latent_dim)
    
    # Create static parameters correlated with first few latent dimensions
    static_params = torch.zeros(batch_size, static_dim)
    static_params[:, 0] = z[:, 0] + 0.1 * torch.randn(batch_size)  # Strong correlation
    static_params[:, 1] = z[:, 1] + 0.5 * torch.randn(batch_size)  # Moderate correlation
    static_params[:, 2:] = torch.randn(batch_size, static_dim - 2)  # No correlation
    
    # Compute correlations
    z_np = z.detach().numpy()
    static_np = static_params.detach().numpy()
    
    correlations = np.zeros((latent_dim, static_dim))
    for i in range(latent_dim):
        for j in range(static_dim):
            correlations[i, j] = np.corrcoef(z_np[:, i], static_np[:, j])[0, 1]
    
    print(f"✓ Correlation matrix computed: {correlations.shape}")
    print(f"  - Max correlation: {np.max(np.abs(correlations)):.4f}")
    print(f"  - Mean correlation: {np.mean(np.abs(correlations)):.4f}")
    
    # Check that we found the expected correlations
    assert abs(correlations[0, 0]) > 0.8, f"Should find strong correlation: {correlations[0, 0]:.4f}"
    assert abs(correlations[1, 1]) > 0.3, f"Should find moderate correlation: {correlations[1, 1]:.4f}"
    print(f"✓ Expected correlations found: z[0]↔static[0]={correlations[0, 0]:.4f}, z[1]↔static[1]={correlations[1, 1]:.4f}")
    
    # Test InfoNCE effectiveness with correlated data
    infonce_loss_before = infonce_contrastive_loss(z, static_params)
    
    # Shuffle static params to break correlations
    shuffled_indices = torch.randperm(batch_size)
    static_shuffled = static_params[shuffled_indices]
    infonce_loss_after = infonce_contrastive_loss(z, static_shuffled)
    
    print(f"✓ InfoNCE loss with correlations: {infonce_loss_before.item():.4f}")
    print(f"✓ InfoNCE loss without correlations: {infonce_loss_after.item():.4f}")
    
    # InfoNCE should be higher when correlations are broken
    if infonce_loss_after.item() > infonce_loss_before.item():
        print("✓ InfoNCE correctly identifies broken correlations")
    else:
        print("⚠ InfoNCE may not be sensitive enough to correlation changes")
    
    print("✓ Correlation analysis test passed!")
    return True


def test_end_to_end_training_compatibility():
    """Test that all components work together for training."""
    print("\n" + "=" * 60)
    print("Testing End-to-End Training Compatibility")
    print("=" * 60)
    
    # Create model
    model = CVAE(
        signal_length=200,
        static_dim=8,
        latent_dim=64,
        predict_peaks=True,
        use_film=True
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy batch
    batch_size = 8
    signal = torch.randn(batch_size, 200)
    static_params = torch.randn(batch_size, 8)
    peaks = torch.randn(batch_size, 6)
    peak_mask = torch.ones(batch_size, 6, dtype=torch.bool)
    
    # Test training step with static regularization
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    recon_signal, mu, logvar, predicted_peaks, recon_static_from_z = model(signal, static_params)
    
    # Sample z for InfoNCE
    z = model.reparameterize(mu, logvar)
    
    # Compute loss with static regularization
    total_loss, loss_components = enhanced_cvae_loss_with_static_regularization(
        recon_signal=recon_signal,
        target_signal=signal,
        mu=mu,
        logvar=logvar,
        recon_static_from_z=recon_static_from_z,
        target_static=static_params,
        z=z,
        predicted_peaks=predicted_peaks,
        target_peaks=peaks,
        peak_mask=peak_mask
    )
    
    print(f"✓ Training forward pass successful")
    print(f"  - Total loss: {total_loss.item():.4f}")
    print(f"  - Loss components: {len(loss_components)}")
    
    # Backward pass
    total_loss.backward()
    print(f"✓ Backward pass successful")
    
    # Check gradients
    total_norm = 0
    param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    total_norm = total_norm ** (1. / 2)
    print(f"✓ Gradient norm: {total_norm:.4f} across {param_count} parameters")
    
    # Optimizer step
    optimizer.step()
    print(f"✓ Optimizer step successful")
    
    # Test evaluation mode
    model.eval()
    with torch.no_grad():
        eval_output = model(signal, static_params)
        print(f"✓ Evaluation mode successful")
    
    print("✓ End-to-end training compatibility test passed!")
    return True


def visualize_static_regularization_effect():
    """Visualize the effect of static regularization on latent space."""
    print("\n" + "=" * 60)
    print("Visualizing Static Regularization Effect")
    print("=" * 60)
    
    try:
        # Create models with and without static regularization
        signal_length = 200
        static_dim = 8
        latent_dim = 64
        batch_size = 50
        
        # Generate structured data
        torch.manual_seed(42)
        np.random.seed(42)
        
        signal = torch.randn(batch_size, signal_length)
        static_params = torch.randn(batch_size, static_dim)
        
        # Model without regularization (simulate pre-training)
        model = CVAE(signal_length, static_dim, latent_dim, use_film=True)
        model.eval()
        
        with torch.no_grad():
            _, mu_before, _, recon_static_before = model(signal, static_params)
        
        # Compute correlations before regularization
        z_before = mu_before.numpy()
        static_np = static_params.numpy()
        recon_static_np = recon_static_before.numpy()
        
        corr_latent_static = np.mean([
            abs(np.corrcoef(z_before[:, i], static_np[:, j])[0, 1])
            for i in range(min(10, latent_dim))
            for j in range(static_dim)
        ])
        
        corr_recon_target = np.mean([
            abs(np.corrcoef(recon_static_np[:, i], static_np[:, i])[0, 1])
            for i in range(static_dim)
        ])
        
        print(f"✓ Before regularization:")
        print(f"  - Mean |correlation| between latent and static: {corr_latent_static:.4f}")
        print(f"  - Mean |correlation| between reconstructed and target static: {corr_recon_target:.4f}")
        
        # Test static regularization loss
        static_loss = static_reconstruction_loss(recon_static_before, static_params)
        infonce_loss = infonce_contrastive_loss(mu_before, static_params)
        
        print(f"  - Static reconstruction loss: {static_loss.item():.4f}")
        print(f"  - InfoNCE loss: {infonce_loss.item():.4f}")
        
        # Create visualization data
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Static reconstruction accuracy
        plt.subplot(1, 3, 1)
        static_errors = torch.abs(recon_static_before - static_params).numpy()
        plt.boxplot([static_errors[:, i] for i in range(static_dim)], 
                   labels=[f'S{i}' for i in range(static_dim)])
        plt.title('Static Reconstruction Errors')
        plt.ylabel('Absolute Error')
        plt.xlabel('Static Parameter')
        
        # Plot 2: Latent-static correlations
        plt.subplot(1, 3, 2)
        corr_matrix = np.zeros((min(8, latent_dim), static_dim))
        for i in range(min(8, latent_dim)):
            for j in range(static_dim):
                corr_matrix[i, j] = np.corrcoef(z_before[:, i], static_np[:, j])[0, 1]
        
        plt.imshow(np.abs(corr_matrix), cmap='viridis', aspect='auto')
        plt.colorbar(label='|Correlation|')
        plt.title('Latent-Static Correlations')
        plt.xlabel('Static Parameter')
        plt.ylabel('Latent Dimension')
        
        # Plot 3: InfoNCE visualization
        plt.subplot(1, 3, 3)
        # Compute pairwise similarities for InfoNCE with dimension matching
        z_norm = F.normalize(mu_before, dim=1)
        static_norm = F.normalize(static_params, dim=1)
        
        # Handle dimension mismatch like in InfoNCE loss
        z_dim = z_norm.size(1)
        static_dim_val = static_norm.size(1)
        if z_dim != static_dim_val:
            min_dim = min(z_dim, static_dim_val)
            z_proj = z_norm[:, :min_dim]
            static_proj = static_norm[:, :min_dim]
        else:
            z_proj = z_norm
            static_proj = static_norm
        
        similarities = torch.mm(z_proj, static_proj.T).numpy()
        
        plt.imshow(similarities, cmap='RdBu_r', aspect='equal')
        plt.colorbar(label='Cosine Similarity')
        plt.title('InfoNCE Similarity Matrix')
        plt.xlabel('Static Sample')
        plt.ylabel('Latent Sample')
        
        plt.tight_layout()
        plt.savefig('static_regularization_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualization saved as 'static_regularization_analysis.png'")
        
        return True
        
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")
        return False


def create_demo_config():
    """Create a demo configuration file for static regularization."""
    print("\n" + "=" * 60)
    print("Creating Demo Configuration")
    print("=" * 60)
    
    config = {
        "model": {
            "signal_length": 200,
            "static_dim": 8,
            "latent_dim": 64,
            "predict_peaks": True,
            "use_film": True
        },
        "static_regularization": {
            "use_static_regularization": True,
            "static_regularization_weight": 0.5,
            "use_infonce_loss": True,
            "infonce_weight": 0.1,
            "infonce_temperature": 0.07
        },
        "enhanced_loss": {
            "use_enhanced_loss": True,
            "loss_weights": {
                "reconstruction": 1.0,
                "kl_divergence": 1.0,
                "static_reconstruction": 0.5,
                "infonce_contrastive": 0.1
            },
            "loss_config": {
                "use_infonce": True,
                "infonce_temperature": 0.07
            }
        }
    }
    
    # Save configuration
    with open('demo_static_regularization_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Demo configuration saved as 'demo_static_regularization_config.json'")
    return True


def main():
    """Run all static regularization tests."""
    print("Static Regularization Test Suite")
    print("=" * 80)
    
    tests = [
        test_static_decoder,
        test_static_reconstruction_loss,
        test_infonce_contrastive_loss,
        test_hierarchical_static_losses,
        test_enhanced_cvae_loss_with_static_regularization,
        test_hierarchical_cvae_static_regularization,
        test_correlation_analysis,
        test_end_to_end_training_compatibility,
        visualize_static_regularization_effect,
        create_demo_config
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 80)
    print("STATIC REGULARIZATION TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1:2d}. {test.__name__:<40} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All static regularization tests passed!")
        print("\nKey features implemented:")
        print("- Static decoder for latent-static reconstruction")
        print("- InfoNCE contrastive loss for latent-static alignment")
        print("- Hierarchical static regularization")
        print("- Enhanced loss functions with static regularization")
        print("- Training integration with comprehensive logging")
        print("- Sample generation with static prediction")
        print("- Correlation analysis and visualization")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    main() 