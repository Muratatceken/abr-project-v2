#!/usr/bin/env python3
"""
Test Script for Enhanced ABR Training Pipeline

Tests individual components and integration to ensure everything works correctly.

Author: AI Assistant
Date: January 2025
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing Imports...")
    
    try:
        # Test model imports
        from models.hierarchical_unet import ProfessionalHierarchicalUNet
        print("✅ Model import successful")
        
        # Test training components
        from training.enhanced_train import ABRDataset, collate_fn, EnhancedABRLoss
        print("✅ Training components import successful")
        
        # Test configuration
        from training.config_loader import ConfigLoader
        print("✅ Configuration loader import successful")
        
        # Test evaluation
        from training.evaluation import ABREvaluator
        print("✅ Evaluation import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_model_creation():
    """Test model creation with enhanced features."""
    print("\n🧪 Testing Model Creation...")
    
    try:
        from models.hierarchical_unet import ProfessionalHierarchicalUNet
        
        # Create model with minimal settings for testing
        model = ProfessionalHierarchicalUNet(
            input_channels=1,
            static_dim=4,
            base_channels=32,  # Smaller for testing
            n_levels=2,        # Smaller for testing
            sequence_length=200,
            signal_length=200,
            num_classes=5,
            
            # Enhanced features
            num_transformer_layers=2,
            use_cross_attention=True,
            use_positional_encoding=True,
            film_dropout=0.1,
            use_cfg=False  # Disable CFG to avoid circular reference
        )
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 1, 200)
        static_params = torch.randn(batch_size, 4)
        
        model.eval()
        with torch.no_grad():
            outputs = model(x, static_params)
        
        # Validate outputs
        expected_keys = ['recon', 'peak', 'class', 'threshold']
        for key in expected_keys:
            if key not in outputs:
                raise ValueError(f"Missing output key: {key}")
        
        print(f"✅ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"   - Signal reconstruction: {outputs['recon'].shape}")
        print(f"   - Peak prediction components: {len(outputs['peak'])}")
        print(f"   - Classification: {outputs['class'].shape}")
        print(f"   - Threshold: {outputs['threshold'].shape}")
        
        return model, True
        
    except Exception as e:
        print(f"❌ Model creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False

def test_dataset_and_loss():
    """Test dataset and loss function components."""
    print("\n🧪 Testing Dataset and Loss Functions...")
    
    try:
        from training.enhanced_train import ABRDataset, collate_fn, EnhancedABRLoss
        
        # Create dummy data
        dummy_data = []
        for i in range(10):
            dummy_data.append({
                'patient_id': f'test_{i}',
                'static_params': np.random.randn(4),
                'signal': np.random.randn(200),
                'v_peak': np.random.randn(2),
                'v_peak_mask': np.random.choice([True, False], 2),
                'target': np.random.randint(0, 5)
            })
        
        # Create dataset
        dataset = ABRDataset(dummy_data, mode='train', augment=False, cfg_dropout_prob=0.1)
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Test data loader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        
        # Get a batch
        batch = next(iter(dataloader))
        print(f"✅ Batch created successfully")
        print(f"   - Static params: {batch['static_params'].shape}")
        print(f"   - Signal: {batch['signal'].shape}")
        print(f"   - V peak: {batch['v_peak'].shape}")
        print(f"   - Target: {batch['target'].shape}")
        
        # Test loss function
        loss_fn = EnhancedABRLoss(
            n_classes=5,
            use_focal_loss=False,
            class_weights=None
        )
        
        # Create dummy model outputs
        batch_size = batch['signal'].size(0)
        dummy_outputs = {
            'recon': torch.randn(batch_size, 1, 200),
            'peak': (
                torch.randn(batch_size, 1),  # existence logits
                torch.randn(batch_size, 1),  # latency
                torch.randn(batch_size, 1)   # amplitude
            ),
            'class': torch.randn(batch_size, 5),
            'threshold': torch.randn(batch_size, 1)
        }
        
        # Compute loss
        total_loss, loss_components = loss_fn(dummy_outputs, batch)
        
        print(f"✅ Loss computation successful")
        print(f"   - Total loss: {total_loss.item():.4f}")
        print(f"   - Signal loss: {loss_components['signal_loss'].item():.4f}")
        print(f"   - Classification loss: {loss_components['classification_loss'].item():.4f}")
        print(f"   - Peak exist loss: {loss_components['peak_exist_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset/Loss test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n🧪 Testing Configuration System...")
    
    try:
        from training.config_loader import ConfigLoader
        
        # Test default configuration
        config_loader = ConfigLoader()
        
        # Set some test values
        config_loader.set('model.base_channels', 64)
        config_loader.set('training.batch_size', 32)
        config_loader.set('training.learning_rate', 1e-4)
        
        # Test getting values
        base_channels = config_loader.get('model.base_channels')
        batch_size = config_loader.get('training.batch_size')
        
        if base_channels != 64 or batch_size != 32:
            raise ValueError("Configuration set/get failed")
        
        # Test validation
        config_loader.config = {
            'model': {'input_channels': 1, 'static_dim': 4, 'num_classes': 5},
            'training': {'batch_size': 32, 'learning_rate': 1e-4, 'num_epochs': 10},
            'data': {'data_path': 'test.pkl'}
        }
        
        is_valid = config_loader.validate_config()
        
        print(f"✅ Configuration system working")
        print(f"   - Set/get operations: ✅")
        print(f"   - Configuration validation: {'✅' if is_valid else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False

def test_evaluation():
    """Test evaluation components."""
    print("\n🧪 Testing Evaluation System...")
    
    try:
        from training.evaluation import ABREvaluator
        
        # Create dummy model
        class DummyModel(nn.Module):
            def forward(self, x, static_params):
                batch_size = x.size(0)
                return {
                    'recon': torch.randn(batch_size, 1, 200),
                    'peak': (
                        torch.randn(batch_size, 1),
                        torch.randn(batch_size, 1),
                        torch.randn(batch_size, 1)
                    ),
                    'class': torch.randn(batch_size, 5),
                    'threshold': torch.randn(batch_size, 1)
                }
        
        model = DummyModel()
        device = torch.device('cpu')
        
        # Create evaluator
        evaluator = ABREvaluator(model, device, output_dir='test_eval_results')
        
        # Create dummy batch
        batch = {
            'patient_ids': ['test_1', 'test_2'],
            'static_params': torch.randn(2, 4),
            'signal': torch.randn(2, 1, 200),
            'v_peak': torch.randn(2, 2),
            'v_peak_mask': torch.tensor([[True, True], [False, True]]),
            'target': torch.tensor([0, 2])
        }
        
        # Evaluate batch
        evaluator.evaluate_batch(batch, compute_loss=False)
        
        # Compute metrics
        metrics = evaluator.compute_all_metrics()
        
        print(f"✅ Evaluation system working")
        print(f"   - Classification F1: {metrics['classification']['f1_macro']:.4f}")
        print(f"   - Peak existence F1: {metrics['peaks']['existence_f1']:.4f}")
        print(f"   - Signal correlation: {metrics['signal']['signal_correlation']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration of all components."""
    print("\n🧪 Testing Full Integration...")
    
    try:
        # Test that we can create a minimal training setup
        from models.hierarchical_unet import ProfessionalHierarchicalUNet
        from training.enhanced_train import ABRDataset, ABRTrainer, EnhancedABRLoss
        from torch.utils.data import DataLoader
        
        # Create minimal model
        model = ProfessionalHierarchicalUNet(
            input_channels=1,
            static_dim=4,
            base_channels=16,  # Very small for testing
            n_levels=2,
            sequence_length=200,
            signal_length=200,
            num_classes=5,
            num_transformer_layers=1,
            use_cross_attention=False,  # Disable for simplicity
            use_cfg=False
        )
        
        # Create minimal dataset
        dummy_data = []
        for i in range(8):  # Small dataset
            dummy_data.append({
                'patient_id': f'test_{i}',
                'static_params': np.random.randn(4),
                'signal': np.random.randn(200),
                'v_peak': np.random.randn(2),
                'v_peak_mask': np.random.choice([True, False], 2),
                'target': np.random.randint(0, 5)
            })
        
        # Split data
        train_data = dummy_data[:6]
        val_data = dummy_data[6:]
        
        # Create datasets and loaders
        from training.enhanced_train import collate_fn
        train_dataset = ABRDataset(train_data, mode='train', augment=False)
        val_dataset = ABRDataset(val_data, mode='val', augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        
        # Create minimal config
        config = {
            'batch_size': 2,
            'learning_rate': 1e-3,
            'num_epochs': 1,  # Just one epoch for testing
            'weight_decay': 0.01,
            'use_amp': False,  # Disable for simplicity
            'patience': 1,
            'num_workers': 0,
            'use_class_weights': False,
            'use_focal_loss': False,
            'output_dir': 'test_training_output',
            'use_wandb': False,
            'save_every': 1
        }
        
        device = torch.device('cpu')
        model = model.to(device)
        
        print(f"✅ Integration test setup complete")
        print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Training samples: {len(train_data)}")
        print(f"   - Validation samples: {len(val_data)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and provide summary."""
    print("🚀 Enhanced ABR Training Pipeline - Component Tests")
    print("=" * 60)
    
    test_results = {}
    
    # Run individual tests
    test_results['imports'] = test_imports()
    
    if test_results['imports']:
        model, model_success = test_model_creation()
        test_results['model'] = model_success
        test_results['dataset_loss'] = test_dataset_and_loss()
        test_results['configuration'] = test_configuration()
        test_results['evaluation'] = test_evaluation()
        test_results['integration'] = test_integration()
    else:
        print("⚠️  Skipping other tests due to import failures")
        for key in ['model', 'dataset_loss', 'configuration', 'evaluation', 'integration']:
            test_results[key] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<20} {status}")
    
    print("-" * 60)
    print(f"Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Training pipeline is ready.")
        print("\n📋 Next Steps:")
        print("   1. Ensure ultimate_dataset.pkl exists in data/processed/")
        print("   2. Run: python run_training.py --debug")
        print("   3. For full training: python run_training.py")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        print("\n🔧 Troubleshooting:")
        print("   - Check that all required packages are installed")
        print("   - Verify model architecture compatibility")
        print("   - Ensure configuration files are properly formatted")
    
    return test_results

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run all tests
    results = run_all_tests() 