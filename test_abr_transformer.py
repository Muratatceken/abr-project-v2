"""
Shape Test for ABR Transformer Generator

Quick test to verify the new ABR Transformer architecture works correctly
and produces the expected output shapes.
"""

import torch
from models import ABRTransformerGenerator
from models.abr_transformer import create_abr_transformer


def test_abr_transformer_shapes():
    """Test ABR Transformer Generator forward pass and output shapes."""
    print("Testing ABR Transformer Generator...")
    
    # Test parameters
    B, T, S = 4, 200, 10  # batch=4, sequence=200, static_params=10
    
    # Create model
    model = ABRTransformerGenerator(
        input_channels=1, 
        static_dim=S, 
        sequence_length=T,
        d_model=256, 
        n_layers=6, 
        n_heads=8, 
        ff_mult=4, 
        dropout=0.1,
        use_timestep_cond=True, 
        use_static_film=True
    )
    
    print(f"Model created with parameters:")
    print(f"  - Input channels: 1")
    print(f"  - Static dim: {S}")
    print(f"  - Sequence length: {T}")
    print(f"  - Model dimension: 256")
    print(f"  - Layers: 6, Heads: 8")
    
    # Test inputs
    x = torch.randn(B, 1, T)
    static_params = torch.randn(B, S)
    timesteps = torch.randint(0, 1000, (B,))
    
    print(f"\nInput shapes:")
    print(f"  - x: {x.shape}")
    print(f"  - static_params: {static_params.shape}")
    print(f"  - timesteps: {timesteps.shape}")
    
    # Test with return_dict=True
    print(f"\nTesting with return_dict=True...")
    with torch.no_grad():
        out_dict = model(x, static_params=static_params, timesteps=timesteps, return_dict=True)
    
    assert isinstance(out_dict, dict), f"Expected dict, got {type(out_dict)}"
    assert 'signal' in out_dict, f"Expected 'signal' key in output dict"
    assert out_dict['signal'].shape == (B, 1, T), f"Expected shape {(B, 1, T)}, got {out_dict['signal'].shape}"
    
    print(f"  ✓ Output dict: {list(out_dict.keys())}")
    print(f"  ✓ Signal shape: {out_dict['signal'].shape}")
    
    # Test with return_dict=False
    print(f"\nTesting with return_dict=False...")
    with torch.no_grad():
        out_tensor = model(x, static_params=static_params, timesteps=timesteps, return_dict=False)
    
    assert isinstance(out_tensor, torch.Tensor), f"Expected tensor, got {type(out_tensor)}"
    assert out_tensor.shape == (B, 1, T), f"Expected shape {(B, 1, T)}, got {out_tensor.shape}"
    
    print(f"  ✓ Output tensor shape: {out_tensor.shape}")
    
    # Test without conditioning
    print(f"\nTesting without conditioning...")
    with torch.no_grad():
        out_uncond = model(x, static_params=None, timesteps=None, return_dict=True)
    
    assert out_uncond['signal'].shape == (B, 1, T), f"Expected shape {(B, 1, T)}, got {out_uncond['signal'].shape}"
    print(f"  ✓ Unconditional output shape: {out_uncond['signal'].shape}")
    
    # Test with different sequence length (should fail with assertion)
    print(f"\nTesting length assertion...")
    x_diff = torch.randn(B, 1, 150)  # Different length
    try:
        with torch.no_grad():
            out_diff = model(x_diff, static_params=static_params, timesteps=timesteps, return_dict=True)
        print(f"  ❌ ERROR: Model should have rejected length 150")
        assert False, "Model should reject inputs with wrong length"
    except AssertionError as e:
        if "Expected input length" in str(e):
            print(f"  ✓ Correctly rejected wrong input length: {str(e)[:50]}...")
        else:
            raise e
    
    print(f"\n✓ All tests passed! ABRTransformerGenerator works correctly.")


def test_factory_function():
    """Test the create_abr_transformer factory function."""
    print(f"\nTesting factory function...")
    
    model = create_abr_transformer(
        static_dim=4,
        sequence_length=200,
        d_model=128,
        n_layers=4,
        n_heads=4
    )
    
    B, T = 2, 200
    x = torch.randn(B, 1, T)
    static_params = torch.randn(B, 4)
    
    with torch.no_grad():
        out = model(x, static_params=static_params, return_dict=True)
    
    assert out['signal'].shape == (B, 1, T), f"Expected shape {(B, 1, T)}, got {out['signal'].shape}"
    print(f"  ✓ Factory function creates working model")


def test_model_parameters():
    """Test model parameter count and memory usage."""
    print(f"\nTesting model parameters...")
    
    model = ABRTransformerGenerator(
        input_channels=1,
        static_dim=4,
        sequence_length=200,
        d_model=256,
        n_layers=6,
        n_heads=8
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    
    # Verify model is trainable
    assert trainable_params > 0, "Model should have trainable parameters"
    assert trainable_params == total_params, "All parameters should be trainable"
    
    print(f"  ✓ Model has appropriate parameter count")


def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    print(f"\nTesting gradient flow...")
    
    model = ABRTransformerGenerator(
        input_channels=1,
        static_dim=4,
        sequence_length=200,
        d_model=128,  # Smaller for faster test
        n_layers=2,
        n_heads=4
    )
    
    # Create inputs and target
    B, T = 2, 200
    x = torch.randn(B, 1, T)
    static_params = torch.randn(B, 4)
    target = torch.randn(B, 1, T)
    
    # Forward pass
    out = model(x, static_params=static_params, return_dict=True)
    
    # Compute loss and backward pass
    loss = torch.nn.functional.mse_loss(out['signal'], target)
    loss.backward()
    
    # Check that gradients exist
    has_gradients = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            has_gradients += 1
    
    gradient_ratio = has_gradients / total_params
    print(f"  - Parameters with gradients: {has_gradients}/{total_params} ({gradient_ratio:.1%})")
    
    assert gradient_ratio > 0.9, f"Expected >90% of parameters to have gradients, got {gradient_ratio:.1%}"
    print(f"  ✓ Gradients flow properly through the model")


if __name__ == "__main__":
    print("=" * 60)
    print("ABR Transformer Generator - Shape and Functionality Tests")
    print("=" * 60)
    
    try:
        test_abr_transformer_shapes()
        test_factory_function()
        test_model_parameters()
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! ABR Transformer Generator is ready to use.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
