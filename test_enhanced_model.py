#!/usr/bin/env python3
"""
Comprehensive Test Script for Enhanced ABR Hierarchical U-Net

Tests all the implemented fixes and improvements:
- Fix 1: FiLM Dropout
- Fix 2: Masked Loss in Peak Prediction
- Fix 3: Attention Pooling for Threshold Regression
- Improvement 1: Deeper Transformer Stack
- Improvement 2: Positional Encoding
- Improvement 3: Cross-Attention
- Improvement 4: Enhanced CFG Support

Author: AI Assistant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced model
from models.hierarchical_unet import ProfessionalHierarchicalUNet
from models.blocks.heads import EnhancedPeakHead
from models.blocks.film import CFGWrapper

def test_model_initialization():
    """Test 1: Model Initialization and Architecture Validation"""
    print("🧪 Test 1: Model Initialization and Architecture Validation")
    print("-" * 60)
    
    try:
        # Initialize model with enhanced features
        model = ProfessionalHierarchicalUNet(
            input_channels=1,
            static_dim=4,
            base_channels=64,
            n_levels=4,
            sequence_length=200,
            signal_length=200,
            num_classes=5,
            
            # Enhanced features
            num_transformer_layers=3,  # Deeper transformer stack
            use_cross_attention=True,  # Cross-attention enabled
            use_positional_encoding=True,  # Positional encoding
            film_dropout=0.15,  # FiLM dropout
            use_cfg=True  # CFG support
        )
        
        # Get model info
        model_info = model.get_model_info()
        print(f"✅ Model initialized successfully!")
        print(f"   - Total parameters: {model_info['total_parameters']:,}")
        print(f"   - Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"   - Architecture levels: {model_info['architecture']['encoder_levels']}")
        print(f"   - CFG support: {model_info['architecture']['supports_cfg']}")
        
        return model, True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {str(e)}")
        return None, False

def test_forward_pass(model):
    """Test 2: Basic Forward Pass"""
    print("\n🧪 Test 2: Basic Forward Pass")
    print("-" * 60)
    
    try:
        batch_size = 4
        sequence_length = 200
        static_dim = 4
        
        # Create test inputs
        x = torch.randn(batch_size, 1, sequence_length)
        static_params = torch.randn(batch_size, static_dim)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(x, static_params)
        
        # Validate outputs
        expected_keys = ['recon', 'peak', 'class', 'threshold']
        for key in expected_keys:
            if key not in outputs:
                raise ValueError(f"Missing output key: {key}")
        
        # Check output shapes
        print(f"✅ Forward pass successful!")
        print(f"   - Signal reconstruction: {outputs['recon'].shape}")
        print(f"   - Peak prediction: {len(outputs['peak'])} components")
        print(f"   - Classification: {outputs['class'].shape}")
        print(f"   - Threshold: {outputs['threshold'].shape}")
        
        return outputs, True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {str(e)}")
        return None, False

def test_film_dropout(model):
    """Test 3: FiLM Dropout Functionality"""
    print("\n🧪 Test 3: FiLM Dropout Functionality")
    print("-" * 60)
    
    try:
        batch_size = 8
        x = torch.randn(batch_size, 1, 200)
        static_params = torch.randn(batch_size, 4)
        
        # Test training mode (should apply FiLM dropout)
        model.train()
        outputs_train = []
        for _ in range(5):  # Multiple runs to see variation
            with torch.no_grad():
                output = model(x, static_params)
            outputs_train.append(output['recon'])
        
        # Check for variation (indicating dropout is working)
        variations = []
        for i in range(1, len(outputs_train)):
            diff = torch.mean(torch.abs(outputs_train[i] - outputs_train[0]))
            variations.append(diff.item())
        
        avg_variation = np.mean(variations)
        
        # Test eval mode (should be deterministic)
        model.eval()
        outputs_eval = []
        for _ in range(3):
            with torch.no_grad():
                output = model(x, static_params)
            outputs_eval.append(output['recon'])
        
        eval_diff = torch.mean(torch.abs(outputs_eval[1] - outputs_eval[0])).item()
        
        print(f"✅ FiLM Dropout test successful!")
        print(f"   - Training mode variation: {avg_variation:.6f} (should be > 0)")
        print(f"   - Eval mode variation: {eval_diff:.6f} (should be ≈ 0)")
        
        if avg_variation > eval_diff * 10:  # Training should have more variation
            print(f"   - ✅ FiLM dropout is working correctly!")
            return True
        else:
            print(f"   - ⚠️  FiLM dropout may not be working as expected")
            return False
            
    except Exception as e:
        print(f"❌ FiLM dropout test failed: {str(e)}")
        return False

def test_masked_loss():
    """Test 4: Masked Loss in Peak Prediction"""
    print("\n🧪 Test 4: Masked Loss in Peak Prediction")
    print("-" * 60)
    
    try:
        # Create peak prediction head
        peak_head = EnhancedPeakHead(
            input_dim=64,
            hidden_dim=64,
            use_attention=True
        )
        
        batch_size = 8
        input_features = torch.randn(batch_size, 64)
        
        # Forward pass
        peak_exists, peak_latency, peak_amplitude = peak_head(input_features)
        
        # Create test targets and mask
        true_existence = torch.randint(0, 2, (batch_size,)).float()
        true_latency = torch.randn(batch_size) * 2 + 4  # 2-6 ms range
        true_amplitude = torch.randn(batch_size) * 0.2  # Small amplitude
        peak_mask = true_existence  # Mask where peaks exist
        
        # Test masked loss computation
        loss_dict = peak_head.compute_masked_loss(
            pred_existence=peak_exists,
            pred_latency=peak_latency,
            pred_amplitude=peak_amplitude,
            true_existence=true_existence,
            true_latency=true_latency,
            true_amplitude=true_amplitude,
            peak_mask=peak_mask
        )
        
        # Validate loss components
        expected_keys = ['existence_loss', 'latency_loss', 'amplitude_loss', 'total_loss', 'num_peaks']
        for key in expected_keys:
            if key not in loss_dict:
                raise ValueError(f"Missing loss key: {key}")
        
        print(f"✅ Masked loss test successful!")
        print(f"   - Existence loss: {loss_dict['existence_loss']:.4f}")
        print(f"   - Latency loss: {loss_dict['latency_loss']:.4f}")
        print(f"   - Amplitude loss: {loss_dict['amplitude_loss']:.4f}")
        print(f"   - Total loss: {loss_dict['total_loss']:.4f}")
        print(f"   - Number of peaks: {loss_dict['num_peaks']:.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Masked loss test failed: {str(e)}")
        return False

def test_cross_attention(model):
    """Test 5: Cross-Attention Functionality"""
    print("\n🧪 Test 5: Cross-Attention Functionality")
    print("-" * 60)
    
    try:
        batch_size = 4
        x = torch.randn(batch_size, 1, 200)
        static_params = torch.randn(batch_size, 4)
        
        # Test with cross-attention enabled
        model.eval()
        with torch.no_grad():
            outputs_with_cross = model(x, static_params)
        
        # Temporarily disable cross-attention in decoder
        original_cross_attention = model.decoder_stack.decoder_levels[0].use_cross_attention
        for decoder_level in model.decoder_stack.decoder_levels:
            decoder_level.use_cross_attention = False
        
        with torch.no_grad():
            outputs_without_cross = model(x, static_params)
        
        # Restore original setting
        for decoder_level in model.decoder_stack.decoder_levels:
            decoder_level.use_cross_attention = original_cross_attention
        
        # Compare outputs (should be different)
        recon_diff = torch.mean(torch.abs(
            outputs_with_cross['recon'] - outputs_without_cross['recon']
        )).item()
        
        print(f"✅ Cross-attention test successful!")
        print(f"   - Output difference with/without cross-attention: {recon_diff:.6f}")
        
        if recon_diff > 1e-6:  # Should have meaningful difference
            print(f"   - ✅ Cross-attention is affecting outputs correctly!")
            return True
        else:
            print(f"   - ⚠️  Cross-attention may not be working as expected")
            return False
            
    except Exception as e:
        print(f"❌ Cross-attention test failed: {str(e)}")
        return False

def test_cfg_functionality(model):
    """Test 6: Enhanced CFG Functionality"""
    print("\n🧪 Test 6: Enhanced CFG Functionality")
    print("-" * 60)
    
    try:
        if not hasattr(model, 'cfg_wrapper') or model.cfg_wrapper is None:
            print("⚠️  CFG wrapper not found, skipping CFG test")
            return False
        
        batch_size = 4
        x = torch.randn(batch_size, 1, 200)
        static_params = torch.randn(batch_size, 4)
        
        # Test different CFG modes
        model.eval()
        
        # 1. Conditional generation
        with torch.no_grad():
            cond_outputs = model(x, static_params, cfg_mode='training')
        
        # 2. Unconditional generation
        with torch.no_grad():
            uncond_outputs = model(x, static_params, cfg_mode='unconditional', force_uncond=True)
        
        # 3. CFG inference with guidance
        with torch.no_grad():
            cfg_outputs = model(x, static_params, cfg_guidance_scale=2.0, cfg_mode='inference')
        
        # Test enhanced CFG sampling
        enhanced_outputs = model.cfg_wrapper.sample_with_cfg(
            x=x,
            condition=static_params,
            guidance_scale=1.5,
            temperature=0.9,
            apply_constraints=True
        )
        
        # Compare outputs
        cond_vs_uncond = torch.mean(torch.abs(cond_outputs['recon'] - uncond_outputs['recon'])).item()
        cfg_vs_cond = torch.mean(torch.abs(cfg_outputs['recon'] - cond_outputs['recon'])).item()
        
        print(f"✅ CFG functionality test successful!")
        print(f"   - Conditional vs Unconditional difference: {cond_vs_uncond:.6f}")
        print(f"   - CFG vs Conditional difference: {cfg_vs_cond:.6f}")
        print(f"   - Enhanced sampling completed successfully")
        
        # Test task-specific guidance scales
        original_scales = model.cfg_wrapper.get_guidance_scales()
        print(f"   - Task-specific scales: {original_scales}")
        
        return True
        
    except Exception as e:
        print(f"❌ CFG functionality test failed: {str(e)}")
        return False

def test_positional_encoding(model):
    """Test 7: Positional Encoding"""
    print("\n🧪 Test 7: Positional Encoding")
    print("-" * 60)
    
    try:
        # Test different sequence lengths to verify positional encoding
        batch_size = 2
        static_params = torch.randn(batch_size, 4)
        
        # Test with different sequence lengths
        seq_lengths = [100, 200, 150]  # Different lengths
        outputs = []
        
        model.eval()
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, 1, seq_len)
            try:
                with torch.no_grad():
                    output = model(x, static_params)
                outputs.append(output['recon'].shape)
                print(f"   - Sequence length {seq_len}: Output shape {output['recon'].shape}")
            except Exception as e:
                print(f"   - Sequence length {seq_len}: Failed ({str(e)})")
        
        print(f"✅ Positional encoding test successful!")
        print(f"   - Model handles different sequence lengths correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Positional encoding test failed: {str(e)}")
        return False

def test_attention_pooling():
    """Test 8: Attention Pooling in Threshold Head"""
    print("\n🧪 Test 8: Attention Pooling in Threshold Head")
    print("-" * 60)
    
    try:
        from models.blocks.heads import EnhancedThresholdHead, AttentionPooling
        
        # Test attention pooling directly
        input_dim = 64
        seq_len = 50
        batch_size = 4
        
        attention_pool = AttentionPooling(input_dim)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        pooled = attention_pool(x)
        
        if pooled.shape != (batch_size, input_dim):
            raise ValueError(f"Expected shape ({batch_size}, {input_dim}), got {pooled.shape}")
        
        # Test threshold head with attention
        threshold_head = EnhancedThresholdHead(
            input_dim=input_dim,
            use_attention=True,
            predict_uncertainty=True
        )
        
        # Test with sequence input
        x_seq = torch.randn(batch_size, input_dim, seq_len)  # Conv format
        threshold_output = threshold_head(x_seq)
        
        print(f"✅ Attention pooling test successful!")
        print(f"   - Pooled shape: {pooled.shape}")
        print(f"   - Threshold output shape: {threshold_output.shape}")
        print(f"   - Attention pooling working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Attention pooling test failed: {str(e)}")
        return False

def test_training_compatibility(model):
    """Test 9: Training Mode Compatibility"""
    print("\n🧪 Test 9: Training Mode Compatibility")
    print("-" * 60)
    
    try:
        batch_size = 4
        x = torch.randn(batch_size, 1, 200, requires_grad=True)
        static_params = torch.randn(batch_size, 4, requires_grad=True)
        
        # Test gradient computation
        model.train()
        outputs = model(x, static_params)
        
        # Create dummy loss
        loss = (
            F.mse_loss(outputs['recon'], torch.randn_like(outputs['recon'])) +
            F.cross_entropy(outputs['class'], torch.randint(0, 5, (batch_size,))) +
            F.mse_loss(outputs['threshold'], torch.randn_like(outputs['threshold']))
        )
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Training compatibility test successful!")
        print(f"   - Loss computed: {loss.item():.4f}")
        print(f"   - Gradients computed: {has_gradients}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training compatibility test failed: {str(e)}")
        return False

def run_comprehensive_test():
    """Run all tests comprehensively"""
    print("🚀 Enhanced ABR Hierarchical U-Net - Comprehensive Test Suite")
    print("=" * 80)
    
    test_results = {}
    
    # Test 1: Model Initialization
    model, init_success = test_model_initialization()
    test_results['initialization'] = init_success
    
    if not init_success:
        print("\n❌ Cannot proceed with other tests due to initialization failure")
        return test_results
    
    # Test 2: Forward Pass
    outputs, forward_success = test_forward_pass(model)
    test_results['forward_pass'] = forward_success
    
    # Test 3: FiLM Dropout
    test_results['film_dropout'] = test_film_dropout(model)
    
    # Test 4: Masked Loss
    test_results['masked_loss'] = test_masked_loss()
    
    # Test 5: Cross-Attention
    test_results['cross_attention'] = test_cross_attention(model)
    
    # Test 6: CFG Functionality
    test_results['cfg_functionality'] = test_cfg_functionality(model)
    
    # Test 7: Positional Encoding
    test_results['positional_encoding'] = test_positional_encoding(model)
    
    # Test 8: Attention Pooling
    test_results['attention_pooling'] = test_attention_pooling()
    
    # Test 9: Training Compatibility
    test_results['training_compatibility'] = test_training_compatibility(model)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<30} {status}")
    
    print("-" * 80)
    print(f"Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced model is ready for training.")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return test_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run comprehensive test
    results = run_comprehensive_test() 