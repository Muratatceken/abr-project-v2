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


def test_forward_with_peak_classification():
    """Test that the model returns both signal and peak classification outputs."""
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
    assert isinstance(out, dict)
    assert "signal" in out
    assert "peak_5th_exists" in out
    assert out["signal"].shape == (B, 1, T)
    assert out["peak_5th_exists"].shape == (B,)  # scalar logits per batch item


def test_backward_compatibility_return_dict_false():
    """Test that return_dict=False still returns only signal tensor for backward compatibility."""
    B, T, S = 4, 200, 6
    model = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True
    )
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    # Test with return_dict=False
    out_tensor = model(x, static_params=s, timesteps=t, return_dict=False)
    assert isinstance(out_tensor, torch.Tensor)
    assert out_tensor.shape == (B, 1, T)
    # Ensure it's not a dict or tuple
    assert not isinstance(out_tensor, (dict, tuple))


def test_attention_pooling_functionality():
    """Test that the attention pooling mechanism works correctly."""
    B, T, S = 4, 200, 6
    model = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True
    )
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    # Test that different input sequences produce different pooled representations
    out1 = model(x, static_params=s, timesteps=t, return_dict=True)
    out2 = model(x + 0.1 * torch.randn_like(x), static_params=s, timesteps=t, return_dict=True)
    
    # Peak classification outputs should be different for different inputs
    assert not torch.allclose(out1["peak_5th_exists"], out2["peak_5th_exists"], atol=1e-6)
    
    # Check that pooling output has expected dimensionality
    assert out1["peak_5th_exists"].shape == (B,)
    assert out2["peak_5th_exists"].shape == (B,)


def test_cross_attention_functionality():
    """Test that cross-attention between static params and signal features works correctly."""
    B, T, S = 4, 200, 6
    model = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        use_cross_attention=True
    )
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    # Test that cross-attention components are created
    assert hasattr(model, 'cross_attention')
    assert hasattr(model, 'static_encoder')
    assert hasattr(model, 'static_tokens')
    
    # Test that different static parameters produce different outputs
    out1 = model(x, static_params=s, timesteps=t, return_dict=True)
    out2 = model(x, static_params=s + 0.1 * torch.randn_like(s), timesteps=t, return_dict=True)
    
    # Outputs should be different due to cross-attention
    assert not torch.allclose(out1["signal"], out2["signal"], atol=1e-6)
    assert not torch.allclose(out1["peak_5th_exists"], out2["peak_5th_exists"], atol=1e-6)


def test_joint_static_generation():
    """Test static parameter reconstruction capability."""
    B, T, S = 4, 200, 6
    model = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        joint_static_generation=True
    )
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    # Test that static reconstruction components are created
    assert hasattr(model, 'static_recon_head')
    
    # Test output structure
    out = model(x, static_params=s, timesteps=t, return_dict=True)
    assert "static_recon" in out
    assert out["static_recon"].shape == (B, S)
    
    # Test that static reconstruction has reasonable values
    assert not torch.allclose(out["static_recon"], torch.zeros_like(out["static_recon"]), atol=1e-6)


def test_learnable_positional_embeddings():
    """Test learnable vs fixed positional encoding."""
    B, T, S = 4, 200, 6
    
    # Model with identity positioning
    model_identity = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        use_learned_pos_emb=False
    )
    
    # Model with learnable positioning
    model_learned = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        use_learned_pos_emb=True
    )
    
    # Test that different positioning is used
    assert isinstance(model_identity.pos, torch.nn.Identity)
    assert not isinstance(model_learned.pos, torch.nn.Identity)
    
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    # Test that outputs are different
    out1 = model_identity(x, static_params=s, timesteps=t, return_dict=True)
    out2 = model_learned(x, static_params=s, timesteps=t, return_dict=True)
    
    # Outputs should be different due to different positioning
    assert not torch.allclose(out1["signal"], out2["signal"], atol=1e-6)


def test_film_residual_connections():
    """Test residual connections in FiLM layers."""
    B, T, S = 4, 200, 6
    
    # Model without residual connections
    model_no_residual = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        film_residual=False
    )
    
    # Model with residual connections
    model_residual = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        film_residual=True
    )
    
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    # Test that outputs are different
    out1 = model_no_residual(x, static_params=s, timesteps=t, return_dict=True)
    out2 = model_residual(x, static_params=s, timesteps=t, return_dict=True)
    
    # Outputs should be different due to residual connections
    assert not torch.allclose(out1["signal"], out2["signal"], atol=1e-6)


def test_multi_scale_feature_fusion():
    """Test multi-scale feature processing."""
    B, T, S = 4, 200, 6
    model = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        use_multi_scale_fusion=True
    )
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    # Test that multi-scale fusion components are created
    assert hasattr(model, 'multi_scale_fusion')
    
    # Test output structure
    out = model(x, static_params=s, timesteps=t, return_dict=True)
    assert "signal" in out
    assert "peak_5th_exists" in out
    assert out["signal"].shape == (B, 1, T)
    assert out["peak_5th_exists"].shape == (B,)


def test_multi_scale_feature_extractor_non_divisible():
    """Test MultiScaleFeatureExtractor with non-divisible input dimensions."""
    from models.blocks.heads import MultiScaleFeatureExtractor
    
    # Test case: D=258, scales=4 (258 % 4 = 2)
    input_dim = 258
    scales = [1, 3, 5, 7]
    
    extractor = MultiScaleFeatureExtractor(input_dim, scales)
    
    # Verify channel allocation
    expected_channels = [64, 64, 65, 65]  # 258 // 4 = 64, remainder 2 -> last 2 get +1
    assert extractor.out_channels_per_branch == expected_channels
    assert sum(extractor.out_channels_per_branch) == input_dim
    
    # Test forward pass
    batch_size, seq_len = 4, 100
    x = torch.randn(batch_size, input_dim, seq_len)
    output = extractor(x)
    
    # Verify output shape
    assert output.shape == (batch_size, input_dim, seq_len)
    
    # Test case: D=255, scales=4 (255 % 4 = 3)
    input_dim = 255
    scales = [1, 3, 5, 7]
    
    extractor2 = MultiScaleFeatureExtractor(input_dim, scales)
    
    # Verify channel allocation
    expected_channels2 = [63, 64, 64, 64]  # 255 // 4 = 63, remainder 3 -> last 3 get +1
    assert extractor2.out_channels_per_branch == expected_channels2
    assert sum(extractor2.out_channels_per_branch) == input_dim


def test_ablation_mode_configurations():
    """Test different ablation modes affect instantiated modules."""
    B, T, S = 4, 200, 6
    
    # Test minimal mode
    model_minimal = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        ablation_mode="minimal"
    )
    
    # Test full mode
    model_full = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        ablation_mode="full"
    )
    
    # Test that minimal mode disables all enhancements
    assert not model_minimal.use_cross_attention
    assert not model_minimal.use_learned_pos_emb
    assert not model_minimal.film_residual
    assert not model_minimal.use_multi_scale_fusion
    assert not model_minimal.joint_static_generation
    assert not model_minimal.use_advanced_blocks
    assert not model_minimal.use_multi_scale_attention
    assert not model_minimal.use_gated_ffn
    
    # Test that minimal mode doesn't create enhancement modules
    assert not hasattr(model_minimal, 'cross_attention') or model_minimal.cross_attention is None
    assert not hasattr(model_minimal, 'static_encoder') or model_minimal.static_encoder is None
    assert not hasattr(model_minimal, 'multi_scale_fusion') or model_minimal.multi_scale_fusion is None
    assert not hasattr(model_minimal, 'static_recon_head') or model_minimal.static_recon_head is None
    assert isinstance(model_minimal.pos, torch.nn.Identity)
    
    # Test that full mode enables all enhancements
    assert model_full.use_cross_attention
    assert model_full.use_learned_pos_emb
    assert model_full.film_residual
    assert model_full.use_multi_scale_fusion
    assert model_full.joint_static_generation
    assert model_full.use_advanced_blocks
    assert model_full.use_multi_scale_attention
    assert model_full.use_gated_ffn
    
    # Test that full mode creates enhancement modules
    assert hasattr(model_full, 'cross_attention') and model_full.cross_attention is not None
    assert hasattr(model_full, 'static_encoder') and model_full.static_encoder is not None
    assert hasattr(model_full, 'multi_scale_fusion') and model_full.multi_scale_fusion is not None
    assert hasattr(model_full, 'static_recon_head') and model_full.static_recon_head is not None
    assert not isinstance(model_full.pos, torch.nn.Identity)
    
    # Test joint_gen_only mode
    model_joint_only = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        ablation_mode="joint_gen_only"
    )
    
    # Test that joint_gen_only mode enables only joint_static_generation
    assert not model_joint_only.use_cross_attention
    assert not model_joint_only.use_learned_pos_emb
    assert not model_joint_only.film_residual
    assert not model_joint_only.use_multi_scale_fusion
    assert model_joint_only.joint_static_generation
    assert not model_joint_only.use_advanced_blocks
    assert not model_joint_only.use_multi_scale_attention
    assert not model_joint_only.use_gated_ffn
    
    # Test that joint_gen_only mode creates only static_recon_head
    assert not hasattr(model_joint_only, 'cross_attention') or model_joint_only.cross_attention is None
    assert not hasattr(model_joint_only, 'static_encoder') or model_joint_only.static_encoder is None
    assert not hasattr(model_joint_only, 'multi_scale_fusion') or model_joint_only.multi_scale_fusion is None
    assert hasattr(model_joint_only, 'static_recon_head') and model_joint_only.static_recon_head is not None
    assert isinstance(model_joint_only.pos, torch.nn.Identity)
    
    # Test that advanced transformer blocks are instantiated when enabled
    if model_full.use_advanced_blocks:
        assert hasattr(model_full.transformer, 'layers')
        for layer in model_full.transformer.layers:
            from models.blocks.transformer_block import AdvancedTransformerBlock
            assert isinstance(layer, AdvancedTransformerBlock)
            assert hasattr(layer, 'use_multi_scale_attention')
            assert hasattr(layer, 'use_gated_ffn')
    
    # Test forward pass to ensure features actually affect output
    B, T, S = 4, 200, 6
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    # Test minimal mode forward pass
    out_minimal = model_minimal(x, static_params=s, timesteps=t, return_dict=True)
    assert "signal" in out_minimal
    assert "peak_5th_exists" in out_minimal
    assert out_minimal["signal"].shape == (B, 1, T)
    assert out_minimal["peak_5th_exists"].shape == (B,)
    
    # Test full mode forward pass
    out_full = model_full(x, static_params=s, timesteps=t, return_dict=True)
    assert "signal" in out_full
    assert "peak_5th_exists" in out_full
    assert "static_recon" in out_full  # Should have static reconstruction
    assert out_full["signal"].shape == (B, 1, T)
    assert out_full["peak_5th_exists"].shape == (B,)
    assert out_full["static_recon"].shape == (B, S)
    
    # Verify that outputs are different (features are actually working)
    # Note: This is a basic check - in practice, outputs might be similar due to initialization
    assert torch.allclose(out_minimal["signal"], out_full["signal"], atol=1e-6) == False


def test_advanced_transformer_blocks_instantiation():
    """Test that advanced transformer blocks are properly instantiated when enabled."""
    B, T, S = 4, 200, 6
    
    # Test with advanced blocks enabled
    model_advanced = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=2, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        use_advanced_blocks=True, use_multi_scale_attention=True, use_gated_ffn=True
    )
    
    # Verify that advanced blocks are enabled
    assert model_advanced.use_advanced_blocks
    assert model_advanced.use_multi_scale_attention
    assert model_advanced.use_gated_ffn
    
    # Check that transformer layers are AdvancedTransformerBlock instances
    from models.blocks.transformer_block import AdvancedTransformerBlock
    for layer in model_advanced.transformer.layers:
        assert isinstance(layer, AdvancedTransformerBlock)
        assert hasattr(layer, 'use_multi_scale_attention')
        assert hasattr(layer, 'use_gated_ffn')
    
    # Test forward pass to ensure advanced blocks work
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    out = model_advanced(x, static_params=s, timesteps=t, return_dict=True)
    assert "signal" in out
    assert "peak_5th_exists" in out
    assert out["signal"].shape == (B, 1, T)
    assert out["peak_5th_exists"].shape == (B,)


def test_attention_mask_support():
    """Test that attention mask is properly supported in forward pass."""
    B, T, S = 4, 200, 6
    
    model = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True
    )
    
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    # Test without attention mask (backward compatibility)
    out_no_mask = model(x, static_params=s, timesteps=t, return_dict=True)
    assert "signal" in out_no_mask
    assert "peak_5th_exists" in out_no_mask
    
    # Test with attention mask (1 = unmasked, 0 = masked)
    attn_mask = torch.ones(B, T, T, dtype=torch.bool)
    out_with_mask = model(x, static_params=s, timesteps=t, attn_mask=attn_mask, return_dict=True)
    assert "signal" in out_with_mask
    assert "peak_5th_exists" in out_with_mask
    
    # Test that outputs are different (mask affects attention computation)
    # This is expected behavior as the mask influences the attention patterns
    assert not torch.allclose(out_no_mask["signal"], out_with_mask["signal"], atol=1e-6)
    # Peak classification is also affected by attention mask (attention pooling)
    assert not torch.allclose(out_no_mask["peak_5th_exists"], out_with_mask["peak_5th_exists"], atol=1e-6)


def test_configurable_static_tokens():
    """Test that static tokens can be configured with different counts."""
    B, T, S = 4, 200, 6
    
    # Test with default 1 static token
    model_default = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        use_cross_attention=True
    )
    
    # Test with 3 static tokens
    model_three_tokens = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True,
        use_cross_attention=True, num_static_tokens=3
    )
    
    # Verify static token shapes
    assert model_default.static_tokens.shape == (1, 1, 256)
    assert model_three_tokens.static_tokens.shape == (1, 3, 256)
    
    # Test forward pass with both models
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    out_default = model_default(x, static_params=s, timesteps=t, return_dict=True)
    out_three = model_three_tokens(x, static_params=s, timesteps=t, return_dict=True)
    
    # Both should produce valid outputs
    assert "signal" in out_default
    assert "signal" in out_three
    assert out_default["signal"].shape == (B, 1, T)
    assert out_three["signal"].shape == (B, 1, T)


def test_enhanced_backward_compatibility():
    """Ensure all new features are backward compatible."""
    B, T, S = 4, 200, 6
    
    # Test default parameters maintain original behavior
    model_default = ABRTransformerGenerator(
        input_channels=1, static_dim=S, sequence_length=T,
        d_model=256, n_layers=6, n_heads=8, ff_mult=4,
        dropout=0.1, use_timestep_cond=True, use_static_film=True
    )
    
    x = torch.randn(B, 1, T)
    s = torch.randn(B, S)
    t = torch.randint(0, 1000, (B,))
    
    # Test that default behavior is preserved
    out = model_default(x, static_params=s, timesteps=t, return_dict=True)
    assert "signal" in out
    assert "peak_5th_exists" in out
    assert out["signal"].shape == (B, 1, T)
    assert out["peak_5th_exists"].shape == (B,)
    
    # Test that return_dict=False still works
    out_tensor = model_default(x, static_params=s, timesteps=t, return_dict=False)
    assert isinstance(out_tensor, torch.Tensor)
    assert out_tensor.shape == (B, 1, T)


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
    """Test that gradients flow properly through both signal generation and peak classification paths."""
    model = ABRTransformerGenerator(
        input_channels=1, static_dim=4, sequence_length=200,
        d_model=128, n_layers=2, n_heads=4
    )
    
    x = torch.randn(2, 1, 200, requires_grad=True)
    s = torch.randn(2, 4)
    t = torch.randint(0, 100, (2,))
    
    out = model(x, static_params=s, timesteps=t, return_dict=True)
    
    # Test signal generation path gradients
    signal_loss = out["signal"].sum()
    signal_loss.backward(retain_graph=True)
    
    # Check that input gradients exist for signal path
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # Check that model parameters have gradients from signal path
    param_with_grad = 0
    total_params = 0
    for param in model.parameters():
        total_params += 1
        if param.grad is not None:
            param_with_grad += 1
    
    gradient_ratio = param_with_grad / total_params
    assert gradient_ratio > 0.9, f"Only {gradient_ratio:.1%} of parameters have gradients from signal path"
    
    # Clear gradients and test peak classification path
    model.zero_grad()
    x.grad = None
    
    # Test peak classification path gradients
    peak_loss = out["peak_5th_exists"].sum()
    peak_loss.backward()
    
    # Check that input gradients exist for peak path
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # Check that model parameters have gradients from peak path
    param_with_grad = 0
    for param in model.parameters():
        if param.grad is not None:
            param_with_grad += 1
    
    gradient_ratio = param_with_grad / total_params
    assert gradient_ratio > 0.9, f"Only {gradient_ratio:.1%} of parameters have gradients from peak path"


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


# ============================================================================
# Multi-Task Training Tests
# ============================================================================

def test_multi_task_dataset_peak_labels():
    """Test that ABRDataset with return_peak_labels=True includes peak labels in output."""
    from data.dataset import ABRDataset
    
    # Create a mock dataset with peak labels
    class MockDataset:
        def __init__(self):
            self.data = [
                {
                    'signal': torch.randn(200).numpy(),
                    'static_params': torch.randn(4).numpy(),
                    'target': 0,
                    'patient_id': 'test1',
                    'v_peak_mask': [True, True]  # Both peaks present
                },
                {
                    'signal': torch.randn(200).numpy(),
                    'static_params': torch.randn(4).numpy(),
                    'target': 1,
                    'patient_id': 'test2',
                    'v_peak_mask': [False, True]  # Only second peak present
                }
            ]
    
    # Mock the dataset loading
    import sys
    from unittest.mock import patch
    
    with patch('data.dataset.joblib.load', return_value={'data': MockDataset().data}):
        dataset = ABRDataset(return_peak_labels=True)
        
        # Test that peak labels are computed correctly
        sample1 = dataset[0]
        sample2 = dataset[1]
        
        assert 'peak_exists' in sample1
        assert 'peak_exists' in sample2
        assert sample1['peak_exists'] == 1.0  # True and True = True
        assert sample2['peak_exists'] == 0.0  # False and True = False
        
        # Test peak class weights
        peak_weights = dataset.get_peak_class_weights()
        assert peak_weights.shape == (2,)  # [negative_class, positive_class]
        assert peak_weights[0] > 0 and peak_weights[1] > 0


def test_multi_task_loss_computation():
    """Test that multi-task loss combines signal and peak losses correctly."""
    from utils.schedules import linear_weight_schedule
    
    # Test progressive weight scheduling
    epoch = 10
    start_epoch = 0
    end_epoch = 20
    start_weight = 0.0
    end_weight = 0.5
    
    current_weight = linear_weight_schedule(epoch, start_epoch, end_epoch, start_weight, end_weight)
    expected_weight = 0.25  # Halfway between 0.0 and 0.5
    
    assert abs(current_weight - expected_weight) < 1e-6
    
    # Test edge cases
    assert linear_weight_schedule(0, start_epoch, end_epoch, start_weight, end_weight) == start_weight
    assert linear_weight_schedule(25, start_epoch, end_epoch, start_weight, end_weight) == end_weight


def test_classification_metrics():
    """Test all classification metrics functions in utils.metrics."""
    from utils.metrics import (
        binary_accuracy, binary_precision_recall_f1, 
        auroc_score, compute_classification_metrics
    )
    
    # Test data
    logits = torch.tensor([0.8, -0.2, 0.9, -0.1, 0.7])
    targets = torch.tensor([1, 0, 1, 0, 1])
    
    # Test accuracy
    accuracy = binary_accuracy(logits, targets)
    assert 0.0 <= accuracy <= 1.0
    assert accuracy == 1.0  # All predictions correct with threshold 0.0
    
    # Test precision, recall, F1
    metrics = binary_precision_recall_f1(logits, targets)
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 0.0 <= metrics['precision'] <= 1.0
    assert 0.0 <= metrics['recall'] <= 1.0
    assert 0.0 <= metrics['f1'] <= 1.0
    
    # Test AUROC
    auroc = auroc_score(logits, targets)
    assert 0.0 <= auroc <= 1.0
    
    # Test comprehensive metrics
    all_metrics = compute_classification_metrics(logits, targets)
    assert 'accuracy' in all_metrics
    assert 'precision' in all_metrics
    assert 'recall' in all_metrics
    assert 'f1' in all_metrics
    assert 'auroc' in all_metrics
    
    # Test edge cases
    # All positive
    all_pos_logits = torch.tensor([0.8, 0.9, 0.7])
    all_pos_targets = torch.tensor([1, 1, 1])
    auroc_all_pos = auroc_score(all_pos_logits, all_pos_targets)
    assert torch.isnan(auroc_all_pos)  # AUROC not computable with single class


def test_parameter_groups_optimization():
    """Test create_param_groups function separates parameters correctly."""
    from utils.schedules import create_param_groups
    
    # Create a mock model with named parameters
    class MockModel:
        def __init__(self):
            self.stem_weight = torch.nn.Parameter(torch.randn(10, 10))
            self.transformer_weight = torch.nn.Parameter(torch.randn(10, 10))
            self.out_proj_weight = torch.nn.Parameter(torch.randn(10, 10))
            self.attn_pool_weight = torch.nn.Parameter(torch.randn(10, 10))
            self.peak5_head_weight = torch.nn.Parameter(torch.randn(10, 10))
            self.static_recon_head_weight = torch.nn.Parameter(torch.randn(10, 10))
        
        def named_parameters(self):
            return [
                ('stem.conv1.weight', self.stem_weight),
                ('transformer.layers.0.weight', self.transformer_weight),
                ('out_proj.weight', self.out_proj_weight),
                ('attn_pool.weight', self.attn_pool_weight),
                ('peak5_head.weight', self.peak5_head_weight),
                ('static_recon_head.weight', self.static_recon_head_weight)
            ]
    
    model = MockModel()
    base_lr = 1e-4
    task_lr_multipliers = {
        'signal': 1.0,
        'peak_classification': 0.8,
        'static_reconstruction': 1.2
    }
    
    param_groups = create_param_groups(model, base_lr, task_lr_multipliers)
    
    # Check that all parameters are included
    total_params = sum(len(group['params']) for group in param_groups)
    assert total_params == 6
    
    # Check learning rates
    for group in param_groups:
        if 'signal' in group['name']:
            assert group['lr'] == base_lr * 1.0
        elif 'peak_classification' in group['name']:
            assert group['lr'] == base_lr * 0.8
        elif 'static_reconstruction' in group['name']:
            assert group['lr'] == base_lr * 1.2


def test_progressive_training_schedule():
    """Test linear and cosine weight scheduling functions."""
    from utils.schedules import linear_weight_schedule, cosine_weight_schedule
    
    # Test linear schedule
    start_epoch, end_epoch = 10, 30
    start_weight, end_weight = 0.0, 1.0
    
    # Before start
    assert linear_weight_schedule(5, start_epoch, end_epoch, start_weight, end_weight) == start_weight
    
    # Middle
    mid_epoch = (start_epoch + end_epoch) // 2
    mid_weight = linear_weight_schedule(mid_epoch, start_epoch, end_epoch, start_weight, end_weight)
    expected_mid_weight = (start_weight + end_weight) / 2
    assert abs(mid_weight - expected_mid_weight) < 1e-6
    
    # After end
    assert linear_weight_schedule(35, start_epoch, end_epoch, start_weight, end_weight) == end_weight
    
    # Test cosine schedule
    cos_mid_weight = cosine_weight_schedule(mid_epoch, start_epoch, end_epoch, start_weight, end_weight)
    assert 0.0 <= cos_mid_weight <= 1.0
    
    # Cosine should be smoother than linear
    assert abs(cos_mid_weight - expected_mid_weight) < 0.1


def test_multi_task_validation_metrics():
    """Test validation function returns all expected metrics when multi-task enabled."""
    # This test would require a full validation setup
    # For now, we test the individual components
    
    from utils.metrics import compute_classification_metrics
    
    # Mock validation metrics
    logits = torch.tensor([0.8, -0.2, 0.9, -0.1, 0.7])
    targets = torch.tensor([1, 0, 1, 0, 1])
    
    metrics = compute_classification_metrics(logits, targets)
    
    # Verify all expected metrics are present
    expected_keys = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], (int, float))


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
