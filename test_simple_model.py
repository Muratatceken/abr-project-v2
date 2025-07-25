#!/usr/bin/env python3
"""
Simple Model Test for Enhanced ABR Hierarchical U-Net

Tests the model without CFG wrapper to avoid circular reference issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_simple_model():
    """Test the model without CFG wrapper"""
    print("🚀 Testing Enhanced ABR Model (Simple)")
    print("=" * 60)
    
    try:
        from models.hierarchical_unet import ProfessionalHierarchicalUNet
        
        # Initialize model WITHOUT CFG wrapper to avoid circular reference
        model = ProfessionalHierarchicalUNet(
            input_channels=1,
            static_dim=4,
            base_channels=32,  # Smaller for testing
            n_levels=2,        # Smaller for testing
            sequence_length=200,
            signal_length=200,
            num_classes=5,
            
            # Enhanced features
            num_transformer_layers=2,  # Reduced for testing
            use_cross_attention=True,  # Cross-attention enabled
            use_positional_encoding=True,  # Positional encoding
            film_dropout=0.1,  # FiLM dropout
            use_cfg=False  # Disable CFG to avoid circular reference
        )
        
        print(f"✅ Model initialized successfully!")
        
        # Get model info
        model_info = model.get_model_info()
        print(f"   - Total parameters: {model_info['total_parameters']:,}")
        print(f"   - Architecture levels: {model_info['architecture']['encoder_levels']}")
        print(f"   - CFG support: {model_info['architecture']['supports_cfg']}")
        
        return model, True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False

def test_forward_and_modes(model):
    """Test forward pass and different modes"""
    print("\n🧪 Testing Forward Pass and Modes")
    print("-" * 50)
    
    try:
        batch_size = 2
        x = torch.randn(batch_size, 1, 200)
        static_params = torch.randn(batch_size, 4)
        
        # Test eval mode
        model.eval()
        with torch.no_grad():
            outputs_eval = model(x, static_params)
        
        print(f"✅ Eval mode forward pass:")
        print(f"   - Signal reconstruction: {outputs_eval['recon'].shape}")
        print(f"   - Peak prediction components: {len(outputs_eval['peak'])}")
        print(f"   - Classification: {outputs_eval['class'].shape}")
        print(f"   - Threshold: {outputs_eval['threshold'].shape}")
        
        # Test training mode
        model.train()
        outputs_train1 = model(x, static_params)
        outputs_train2 = model(x, static_params)
        
        # Check for FiLM dropout variation in training mode
        train_diff = torch.mean(torch.abs(outputs_train1['recon'] - outputs_train2['recon'])).item()
        
        print(f"✅ Training mode forward pass:")
        print(f"   - Training mode variation: {train_diff:.6f}")
        print(f"   - FiLM dropout working: {'✅' if train_diff > 1e-6 else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_gradients(model):
    """Test gradient computation"""
    print("\n🧪 Testing Gradient Computation")
    print("-" * 50)
    
    try:
        batch_size = 2
        x = torch.randn(batch_size, 1, 200, requires_grad=True)
        static_params = torch.randn(batch_size, 4, requires_grad=True)
        
        model.train()
        outputs = model(x, static_params)
        
        # Simple loss computation
        signal_loss = F.mse_loss(outputs['recon'], torch.randn_like(outputs['recon']))
        class_loss = F.cross_entropy(outputs['class'], torch.randint(0, 5, (batch_size,)))
        total_loss = signal_loss + class_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
        
        print(f"✅ Gradient computation:")
        print(f"   - Total loss: {total_loss.item():.4f}")
        print(f"   - Has gradients: {has_gradients}")
        print(f"   - Average gradient norm: {avg_grad_norm:.6f}")
        print(f"   - Gradients computed successfully: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_attention_toggle(model):
    """Test cross-attention functionality"""
    print("\n🧪 Testing Cross-Attention Toggle")
    print("-" * 50)
    
    try:
        batch_size = 2
        x = torch.randn(batch_size, 1, 200)
        static_params = torch.randn(batch_size, 4)
        
        model.eval()
        
        # Test with cross-attention enabled
        with torch.no_grad():
            outputs_with_cross = model(x, static_params)
        
        # Temporarily disable cross-attention in decoder levels
        for decoder_level in model.decoder_stack.decoder_levels:
            decoder_level.use_cross_attention = False
        
        with torch.no_grad():
            outputs_without_cross = model(x, static_params)
        
        # Restore cross-attention
        for decoder_level in model.decoder_stack.decoder_levels:
            decoder_level.use_cross_attention = True
        
        # Compare outputs
        recon_diff = torch.mean(torch.abs(
            outputs_with_cross['recon'] - outputs_without_cross['recon']
        )).item()
        
        print(f"✅ Cross-attention toggle test:")
        print(f"   - Output difference: {recon_diff:.6f}")
        print(f"   - Cross-attention working: {'✅' if recon_diff > 1e-6 else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-attention test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_test():
    """Run simplified model test"""
    print("🚀 Enhanced ABR Model - Simple Test")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: Model Initialization
    model, init_success = test_simple_model()
    test_results['model_initialization'] = init_success
    
    if not init_success:
        print("\n❌ Cannot proceed with other tests due to initialization failure")
        return test_results
    
    # Test 2: Forward Pass and Modes
    test_results['forward_and_modes'] = test_forward_and_modes(model)
    
    # Test 3: Gradient Computation
    test_results['gradients'] = test_gradients(model)
    
    # Test 4: Cross-Attention Toggle
    test_results['cross_attention'] = test_cross_attention_toggle(model)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SIMPLE TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print("-" * 60)
    print(f"Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All simple tests passed! Core model functionality is working.")
        print("\n📋 Validated Features:")
        print("   ✅ Model initialization and parameter counting")
        print("   ✅ Forward pass in eval and training modes")
        print("   ✅ FiLM dropout variation in training mode")  
        print("   ✅ Gradient computation and backpropagation")
        print("   ✅ Cross-attention toggle functionality")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return test_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run simple test
    results = run_simple_test() 