"""
Unit tests for enhanced ABR Transformer model architecture.

Tests for hearing loss classification head, joint generation capabilities,
and backward compatibility with existing model usage.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ABRTransformerGenerator
from models.abr_transformer import create_abr_transformer


class TestHearingLossClassificationHead:
    """Test hearing loss classification head initialization and functionality."""
    
    def test_hearing_loss_head_initialization(self):
        """Test that hearing loss head is properly initialized."""
        # Test with default hearing loss classes (5)
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            d_model=256,
            hearing_loss_classes=5
        )
        
        # Check that hearing loss head exists and has correct dimensions
        assert hasattr(model, 'hearing_loss_head')
        assert isinstance(model.hearing_loss_head, torch.nn.Linear)
        assert model.hearing_loss_head.in_features == 256  # d_model
        assert model.hearing_loss_head.out_features == 5   # hearing_loss_classes
        assert model.hearing_loss_classes == 5
        
    def test_custom_hearing_loss_classes(self):
        """Test initialization with custom number of hearing loss classes."""
        # Test with 3 classes (Normal, Mild, Severe)
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            d_model=128,
            hearing_loss_classes=3
        )
        
        assert model.hearing_loss_head.out_features == 3
        assert model.hearing_loss_classes == 3
        
        # Test with 7 classes
        model_7 = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            d_model=256,
            hearing_loss_classes=7
        )
        
        assert model_7.hearing_loss_head.out_features == 7
        assert model_7.hearing_loss_classes == 7
        
    def test_hearing_loss_head_parameters(self):
        """Test that hearing loss head parameters are properly initialized."""
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            d_model=256,
            hearing_loss_classes=5
        )
        
        # Check that parameters exist and have correct shapes
        weight = model.hearing_loss_head.weight
        bias = model.hearing_loss_head.bias
        
        assert weight.shape == (5, 256)  # (out_features, in_features)
        assert bias.shape == (5,)        # (out_features,)
        
        # Check that parameters are not zero (proper initialization)
        assert not torch.allclose(weight, torch.zeros_like(weight))
        assert torch.allclose(bias, torch.zeros_like(bias))  # Bias should be zero-initialized


class TestEnhancedForwardPass:
    """Test enhanced forward pass with joint outputs."""
    
    def test_hearing_loss_logits_output_shape(self):
        """Test that hearing loss logits have correct output shape."""
        B, T, S = 4, 200, 4
        model = ABRTransformerGenerator(
            input_channels=1,
            static_dim=S,
            sequence_length=T,
            d_model=256,
            hearing_loss_classes=5
        )
        
        x = torch.randn(B, 1, T)
        s = torch.randn(B, S)
        t = torch.randint(0, 1000, (B,))
        
        # Test with return_dict=True
        out = model(x, static_params=s, timesteps=t, return_dict=True)
        
        assert isinstance(out, dict)
        assert "hearing_loss_logits" in out
        assert out["hearing_loss_logits"].shape == (B, 5)  # [batch_size, num_classes]
        
    def test_hearing_loss_logits_different_batch_sizes(self):
        """Test hearing loss logits with different batch sizes."""
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            d_model=128,
            hearing_loss_classes=3
        )
        
        batch_sizes = [1, 2, 8, 16]
        for B in batch_sizes:
            x = torch.randn(B, 1, 200)
            s = torch.randn(B, 4)
            
            out = model(x, static_params=s, return_dict=True)
            assert out["hearing_loss_logits"].shape == (B, 3)
            
    def test_hearing_loss_logits_values_reasonable(self):
        """Test that hearing loss logits produce reasonable values."""
        B, T, S = 4, 200, 4
        model = ABRTransformerGenerator(
            sequence_length=T,
            static_dim=S,
            d_model=256,
            hearing_loss_classes=5
        )
        
        x = torch.randn(B, 1, T)
        s = torch.randn(B, S)
        
        out = model(x, static_params=s, return_dict=True)
        logits = out["hearing_loss_logits"]
        
        # Check that logits are not all zeros or all the same
        assert not torch.allclose(logits, torch.zeros_like(logits))
        assert not torch.allclose(logits, logits[0:1].expand_as(logits))
        
        # Check that logits are finite
        assert torch.isfinite(logits).all()
        
        # Check that softmax probabilities sum to 1
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-6)
        
    def test_hearing_loss_different_inputs_different_outputs(self):
        """Test that different inputs produce different hearing loss predictions."""
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            d_model=256,
            hearing_loss_classes=5
        )
        
        # Create two different inputs
        x1 = torch.randn(2, 1, 200)
        x2 = torch.randn(2, 1, 200)
        s = torch.randn(2, 4)
        
        out1 = model(x1, static_params=s, return_dict=True)
        out2 = model(x2, static_params=s, return_dict=True)
        
        # Hearing loss predictions should be different for different inputs
        assert not torch.allclose(out1["hearing_loss_logits"], out2["hearing_loss_logits"], atol=1e-6)


class TestOutputDictionaryStructure:
    """Test output dictionary structure and completeness."""
    
    def test_complete_output_dictionary(self):
        """Test that output dictionary contains all expected keys."""
        B, T, S = 4, 200, 4
        model = ABRTransformerGenerator(
            sequence_length=T,
            static_dim=S,
            d_model=256,
            hearing_loss_classes=5,
            joint_static_generation=True
        )
        
        x = torch.randn(B, 1, T)
        s = torch.randn(B, S)
        t = torch.randint(0, 1000, (B,))
        
        out = model(x, static_params=s, timesteps=t, return_dict=True)
        
        # Check all expected keys are present
        expected_keys = {"signal", "peak_5th_exists", "hearing_loss_logits", "static_recon"}
        assert set(out.keys()) == expected_keys
        
        # Check shapes
        assert out["signal"].shape == (B, 1, T)
        assert out["peak_5th_exists"].shape == (B,)
        assert out["hearing_loss_logits"].shape == (B, 5)
        assert out["static_recon"].shape == (B, S)
        
    def test_output_dictionary_without_joint_generation(self):
        """Test output dictionary when joint static generation is disabled."""
        B, T, S = 4, 200, 4
        model = ABRTransformerGenerator(
            sequence_length=T,
            static_dim=S,
            d_model=256,
            hearing_loss_classes=3,
            joint_static_generation=False
        )
        
        x = torch.randn(B, 1, T)
        s = torch.randn(B, S)
        
        out = model(x, static_params=s, return_dict=True)
        
        # Check expected keys (no static_recon)
        expected_keys = {"signal", "peak_5th_exists", "hearing_loss_logits"}
        assert set(out.keys()) == expected_keys
        
        # Check shapes
        assert out["signal"].shape == (B, 1, T)
        assert out["peak_5th_exists"].shape == (B,)
        assert out["hearing_loss_logits"].shape == (B, 3)
        
    def test_output_dictionary_consistency(self):
        """Test that output dictionary structure is consistent across calls."""
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            d_model=128,
            hearing_loss_classes=5
        )
        
        x = torch.randn(2, 1, 200)
        s = torch.randn(2, 4)
        
        # Multiple forward passes should have consistent structure
        out1 = model(x, static_params=s, return_dict=True)
        out2 = model(x, static_params=s, return_dict=True)
        
        assert set(out1.keys()) == set(out2.keys())
        for key in out1.keys():
            assert out1[key].shape == out2[key].shape


class TestBackwardCompatibility:
    """Test backward compatibility with existing model usage."""
    
    def test_return_dict_false_compatibility(self):
        """Test that return_dict=False maintains backward compatibility."""
        B, T, S = 4, 200, 4
        model = ABRTransformerGenerator(
            sequence_length=T,
            static_dim=S,
            d_model=256,
            hearing_loss_classes=5
        )
        
        x = torch.randn(B, 1, T)
        s = torch.randn(B, S)
        t = torch.randint(0, 1000, (B,))
        
        # Test with return_dict=False (backward compatibility)
        out_tensor = model(x, static_params=s, timesteps=t, return_dict=False)
        
        # Should return only the signal tensor
        assert isinstance(out_tensor, torch.Tensor)
        assert not isinstance(out_tensor, (dict, tuple))
        assert out_tensor.shape == (B, 1, T)
        
    def test_existing_model_parameters_unchanged(self):
        """Test that existing model parameters and behavior are unchanged."""
        # Create model with original parameters
        model = ABRTransformerGenerator(
            input_channels=1,
            static_dim=4,
            sequence_length=200,
            d_model=256,
            n_layers=6,
            n_heads=8,
            ff_mult=4,
            dropout=0.1,
            use_timestep_cond=True,
            use_static_film=True
        )
        
        # Check that all original components still exist
        assert hasattr(model, 'stem')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'out_proj')
        assert hasattr(model, 'peak5_head')
        assert hasattr(model, 'attn_pool')
        
        # Check that new component is added without breaking existing ones
        assert hasattr(model, 'hearing_loss_head')
        
        # Test forward pass works with original interface
        x = torch.randn(2, 1, 200)
        s = torch.randn(2, 4)
        t = torch.randint(0, 1000, (2,))
        
        # Original interface should still work
        out = model(x, static_params=s, timesteps=t, return_dict=True)
        assert "signal" in out
        assert "peak_5th_exists" in out
        assert out["signal"].shape == (2, 1, 200)
        assert out["peak_5th_exists"].shape == (2,)
        
    def test_default_hearing_loss_classes(self):
        """Test that default hearing loss classes value works correctly."""
        # Test with default value (should be 5)
        model = ABRTransformerGenerator(sequence_length=200, static_dim=4)
        
        assert model.hearing_loss_classes == 5
        assert model.hearing_loss_head.out_features == 5
        
        x = torch.randn(2, 1, 200)
        out = model(x, return_dict=True)
        assert out["hearing_loss_logits"].shape == (2, 5)
        
    def test_no_static_params_compatibility(self):
        """Test that model works without static parameters (backward compatibility)."""
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=0,  # No static parameters
            use_static_film=False,
            hearing_loss_classes=3
        )
        
        x = torch.randn(2, 1, 200)
        
        # Should work without static_params
        out = model(x, static_params=None, return_dict=True)
        assert "signal" in out
        assert "peak_5th_exists" in out
        assert "hearing_loss_logits" in out
        assert out["hearing_loss_logits"].shape == (2, 3)


class TestCreateABRTransformerFactory:
    """Test the create_abr_transformer factory function with hearing loss support."""
    
    def test_factory_function_hearing_loss_parameter(self):
        """Test that factory function accepts hearing_loss_classes parameter."""
        model = create_abr_transformer(
            static_dim=4,
            sequence_length=200,
            d_model=256,
            hearing_loss_classes=7
        )
        
        assert isinstance(model, ABRTransformerGenerator)
        assert model.hearing_loss_classes == 7
        assert model.hearing_loss_head.out_features == 7
        
    def test_factory_function_default_hearing_loss(self):
        """Test that factory function uses default hearing loss classes."""
        model = create_abr_transformer(
            static_dim=4,
            sequence_length=200,
            d_model=256
        )
        
        assert model.hearing_loss_classes == 5  # Default value
        assert model.hearing_loss_head.out_features == 5
        
    def test_factory_function_backward_compatibility(self):
        """Test that factory function maintains backward compatibility."""
        # Should work with original parameters
        model = create_abr_transformer(
            static_dim=4,
            sequence_length=200,
            d_model=256,
            n_layers=6,
            n_heads=8
        )
        
        # Should have all original functionality plus new hearing loss head
        x = torch.randn(2, 1, 200)
        s = torch.randn(2, 4)
        
        out = model(x, static_params=s, return_dict=True)
        assert "signal" in out
        assert "peak_5th_exists" in out
        assert "hearing_loss_logits" in out


class TestGradientFlow:
    """Test gradient flow through hearing loss classification head."""
    
    def test_hearing_loss_gradient_flow(self):
        """Test that gradients flow properly through hearing loss head."""
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            d_model=128,
            n_layers=2,
            hearing_loss_classes=3
        )
        
        x = torch.randn(2, 1, 200, requires_grad=True)
        s = torch.randn(2, 4)
        
        out = model(x, static_params=s, return_dict=True)
        
        # Test hearing loss classification path gradients
        hearing_loss = out["hearing_loss_logits"].sum()
        hearing_loss.backward()
        
        # Check that input gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check that hearing loss head parameters have gradients
        assert model.hearing_loss_head.weight.grad is not None
        assert model.hearing_loss_head.bias.grad is not None
        
        # Check that shared components (attention pooling) have gradients
        assert model.attn_pool.attention[0].weight.grad is not None  # First linear layer
        
    def test_multi_task_gradient_flow(self):
        """Test gradient flow when multiple tasks are used simultaneously."""
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            d_model=128,
            n_layers=2,
            hearing_loss_classes=3,
            joint_static_generation=True
        )
        
        x = torch.randn(2, 1, 200, requires_grad=True)
        s = torch.randn(2, 4)
        
        out = model(x, static_params=s, return_dict=True)
        
        # Combined loss from all tasks
        signal_loss = out["signal"].sum()
        peak_loss = out["peak_5th_exists"].sum()
        hearing_loss = out["hearing_loss_logits"].sum()
        static_loss = out["static_recon"].sum()
        
        total_loss = signal_loss + peak_loss + hearing_loss + static_loss
        total_loss.backward()
        
        # Check that all task-specific heads have gradients
        assert model.out_proj.weight.grad is not None  # Signal generation
        assert model.peak5_head.weight.grad is not None  # Peak classification
        assert model.hearing_loss_head.weight.grad is not None  # Hearing loss
        assert model.static_recon_head.weight.grad is not None  # Static reconstruction
        
        # Check that shared attention pooling has gradients
        assert model.attn_pool.attention[0].weight.grad is not None  # First linear layer


class TestModelConsistency:
    """Test model consistency and deterministic behavior."""
    
    def test_deterministic_hearing_loss_output(self):
        """Test that hearing loss output is deterministic in eval mode."""
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            d_model=128,
            hearing_loss_classes=5
        )
        model.eval()
        
        x = torch.randn(2, 1, 200)
        s = torch.randn(2, 4)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        out1 = model(x, static_params=s, return_dict=True)
        
        torch.manual_seed(42)
        out2 = model(x, static_params=s, return_dict=True)
        
        # Hearing loss outputs should be identical
        torch.testing.assert_close(out1["hearing_loss_logits"], out2["hearing_loss_logits"])
        
    def test_hearing_loss_output_device_consistency(self):
        """Test that hearing loss output is consistent across devices."""
        if torch.cuda.is_available():
            # Test CUDA
            device = torch.device('cuda')
            model = ABRTransformerGenerator(
                sequence_length=200,
                static_dim=4,
                d_model=128,
                hearing_loss_classes=3
            ).to(device)
            
            x = torch.randn(2, 1, 200, device=device)
            s = torch.randn(2, 4, device=device)
            
            out = model(x, static_params=s, return_dict=True)
            assert out["hearing_loss_logits"].device == device
            assert out["hearing_loss_logits"].shape == (2, 3)
        
        # Test CPU
        device = torch.device('cpu')
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            d_model=128,
            hearing_loss_classes=3
        ).to(device)
        
        x = torch.randn(2, 1, 200, device=device)
        s = torch.randn(2, 4, device=device)
        
        out = model(x, static_params=s, return_dict=True)
        assert out["hearing_loss_logits"].device == device
        assert out["hearing_loss_logits"].shape == (2, 3)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_hearing_loss_class(self):
        """Test model with single hearing loss class."""
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            hearing_loss_classes=1
        )
        
        x = torch.randn(2, 1, 200)
        s = torch.randn(2, 4)
        
        out = model(x, static_params=s, return_dict=True)
        assert out["hearing_loss_logits"].shape == (2, 1)
        
    def test_large_hearing_loss_classes(self):
        """Test model with large number of hearing loss classes."""
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=4,
            d_model=256,
            hearing_loss_classes=20
        )
        
        x = torch.randn(2, 1, 200)
        s = torch.randn(2, 4)
        
        out = model(x, static_params=s, return_dict=True)
        assert out["hearing_loss_logits"].shape == (2, 20)
        
    def test_zero_static_dim_with_hearing_loss(self):
        """Test model with no static parameters but with hearing loss classification."""
        model = ABRTransformerGenerator(
            sequence_length=200,
            static_dim=0,
            use_static_film=False,
            hearing_loss_classes=3
        )
        
        x = torch.randn(2, 1, 200)
        
        out = model(x, static_params=None, return_dict=True)
        assert "hearing_loss_logits" in out
        assert out["hearing_loss_logits"].shape == (2, 3)


if __name__ == "__main__":
    pytest.main([__file__])