"""
Comprehensive tests for ABRTransformerGenerator

Tests for shape, behavior, and functionality after the refinements.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ABRTransformerGenerator, MultiLayerTransformerBlock


def test_forward_shape_signal_only():
    """Test basic forward pass with correct output shapes."""
    B, T, S = 4, 200, 6
    model = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True
    )
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    # Test with return_dict=True
    out = model(x, static_params=s, timesteps=t, return_dict=True)
    assert isinstance(out, dict) and "signal" in out
    assert out["signal"].shape == (B, 1, T)
    
    # Test with return_dict=False
    out_tensor = model(x, static_params=s, timesteps=t, return_dict=False)
    assert isinstance(out_tensor, torch.Tensor)
    assert out_tensor.shape == (B, 1, T)


def test_length_assertion():
    """Test that model raises assertion for wrong input length."""
    model = ABRTransformerGenerator(sequence_length=200)
    x = torch.randn(2, 1, 180)  # wrong length
    
    with pytest.raises(AssertionError, match="Expected input length"):
        _ = model(x, return_dict=True)


def test_unconditional_ok():
    """Test unconditional generation (no static params or timesteps)."""
    model = ABRTransformerGenerator(
        sequence_length=200, static_dim=0, 
        use_timestep_cond=False, use_static_film=False
    )
    x = torch.randn(2, 1, 200)
    out = model(x, static_params=None, timesteps=None, return_dict=True)
    assert out["signal"].shape == (2, 1, 200)


def test_conditional_with_static_params():
    """Test conditional generation with static parameters."""
    model = ABRTransformerGenerator(
        sequence_length=200, static_dim=4,
        use_timestep_cond=False, use_static_film=True
    )
    x = torch.randn(3, 1, 200)
    s = torch.randn(3, 4)
    out = model(x, static_params=s, timesteps=None, return_dict=True)
    assert out["signal"].shape == (3, 1, 200)


def test_conditional_with_timesteps():
    """Test conditional generation with timestep conditioning."""
    model = ABRTransformerGenerator(
        sequence_length=200, static_dim=0,
        use_timestep_cond=True, use_static_film=False
    )
    x = torch.randn(2, 1, 200)
    t = torch.randint(0, 1000, (2,))
    out = model(x, static_params=None, timesteps=t, return_dict=True)
    assert out["signal"].shape == (2, 1, 200)


def test_multiscale_stem_functionality():
    """Test that the MultiScaleStem processes different scales correctly."""
    from models.abr_transformer import MultiScaleStem
    
    d_model = 256
    stem = MultiScaleStem(d_model)
    
    x = torch.randn(4, 1, 200)
    out = stem(x)
    
    assert out.shape == (4, d_model, 200)
    
    # Test that all branches are used (different from single-scale)
    assert hasattr(stem, 'b3')
    assert hasattr(stem, 'b7') 
    assert hasattr(stem, 'b15')
    assert hasattr(stem, 'fuse')


def test_conv_module_integration():
    """Test that ConvModule is properly integrated in transformer blocks."""
    model = ABRTransformerGenerator(
        input_channels=1, static_dim=4, sequence_length=200,
        d_model=128, n_layers=2, n_heads=4,
        use_timestep_cond=True, use_static_film=True
    )
    
    # Check that conv modules are present in transformer layers
    for layer in model.transformer.layers:
        assert hasattr(layer, 'use_conv_module')
        assert layer.use_conv_module == True
        assert hasattr(layer, 'conv_module')


def test_relative_position_only():
    """Test that model uses relative position encoding only."""
    model = ABRTransformerGenerator(sequence_length=200)
    
    # Check that pos is Identity (no absolute positional encoding)
    assert isinstance(model.pos, torch.nn.Identity)
    
    # Check that transformer uses relative position
    assert model.transformer.relative_position is not None


def test_no_interpolation_in_forward():
    """Test that no interpolation happens during forward pass."""
    model = ABRTransformerGenerator(sequence_length=200)
    
    # This should work fine (exact length)
    x = torch.randn(2, 1, 200)
    out = model(x, return_dict=True)
    assert out["signal"].shape == (2, 1, 200)
    
    # This should raise an assertion error (wrong length)
    x_wrong = torch.randn(2, 1, 150)
    with pytest.raises(AssertionError):
        _ = model(x_wrong, return_dict=True)


def test_gradient_flow():
    """Test that gradients flow properly through the refined model."""
    model = ABRTransformerGenerator(
        input_channels=1, static_dim=4, sequence_length=200,
        d_model=128, n_layers=2, n_heads=4
    )
    
    x = torch.randn(2, 1, 200, requires_grad=True)
    s = torch.randn(2, 4)
    t = torch.randint(0, 100, (2,))
    
    out = model(x, static_params=s, timesteps=t, return_dict=True)
    loss = out["signal"].sum()
    loss.backward()
    
    # Check that input gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # Check that model parameters have gradients
    param_with_grad = 0
    total_params = 0
    for param in model.parameters():
        total_params += 1
        if param.grad is not None:
            param_with_grad += 1
    
    gradient_ratio = param_with_grad / total_params
    assert gradient_ratio > 0.9, f"Only {gradient_ratio:.1%} of parameters have gradients"


def test_parameter_count_reasonable():
    """Test that model has reasonable parameter count."""
    model = ABRTransformerGenerator(
        input_channels=1, static_dim=4, sequence_length=200,
        d_model=256, n_layers=6, n_heads=8
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Should have reasonable number of parameters (not too many, not too few)
    assert 1_000_000 < total_params < 50_000_000, f"Unexpected parameter count: {total_params:,}"
    assert trainable_params == total_params, "All parameters should be trainable"


def test_device_consistency():
    """Test that model handles device placement correctly."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = ABRTransformerGenerator(sequence_length=200, d_model=128, n_layers=2).to(device)
        
        x = torch.randn(2, 1, 200, device=device)
        s = torch.randn(2, 4, device=device)
        
        out = model(x, static_params=s, return_dict=True)
        assert out["signal"].device == device
    
    # Test CPU
    device = torch.device('cpu')
    model = ABRTransformerGenerator(sequence_length=200, d_model=128, n_layers=2).to(device)
    
    x = torch.randn(2, 1, 200, device=device)
    s = torch.randn(2, 4, device=device)
    
    out = model(x, static_params=s, return_dict=True)
    assert out["signal"].device == device


def test_deterministic_output():
    """Test that model produces deterministic output when in eval mode."""
    model = ABRTransformerGenerator(sequence_length=200, d_model=128, n_layers=2)
    model.eval()
    
    x = torch.randn(2, 1, 200)
    s = torch.randn(2, 4)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    out1 = model(x, static_params=s, return_dict=True)
    
    torch.manual_seed(42)
    out2 = model(x, static_params=s, return_dict=True)
    
    # Outputs should be identical
    torch.testing.assert_close(out1["signal"], out2["signal"])


def test_batch_size_flexibility():
    """Test that model works with different batch sizes."""
    model = ABRTransformerGenerator(sequence_length=200, d_model=128, n_layers=2)
    
    batch_sizes = [1, 2, 4, 8, 16]
    for B in batch_sizes:
        x = torch.randn(B, 1, 200)
        s = torch.randn(B, 4)
        
        out = model(x, static_params=s, return_dict=True)
        assert out["signal"].shape == (B, 1, 200)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
