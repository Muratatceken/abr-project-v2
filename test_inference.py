#!/usr/bin/env python3
"""
Test Script for ABR Inference Pipeline

Tests the complete inference functionality with synthetic model and data.

Author: AI Assistant
Date: January 2025
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from inference.inference import ABRInferenceEngine, create_sample_input


class MockABRModel(nn.Module):
    """Mock ABR model for testing inference pipeline."""
    
    def __init__(self, static_dim=4, signal_length=200, num_classes=5):
        super().__init__()
        self.static_dim = static_dim
        self.signal_length = signal_length
        self.num_classes = num_classes
        
        # Simple linear layers for testing
        self.signal_head = nn.Linear(static_dim, signal_length)
        self.peak_exist_head = nn.Linear(static_dim, 1)
        self.peak_latency_head = nn.Linear(static_dim, 1)
        self.peak_amplitude_head = nn.Linear(static_dim, 1)
        self.class_head = nn.Linear(static_dim, num_classes)
        self.threshold_head = nn.Linear(static_dim, 1)
    
    def forward(self, x, static_params, t=None):
        """Mock forward pass."""
        batch_size = static_params.size(0)
        
        # Generate mock outputs based on static parameters
        signal_out = self.signal_head(static_params).unsqueeze(1)  # [B, 1, L]
        
        # Add some realistic ABR-like patterns
        time_axis = torch.linspace(0, 10, self.signal_length, device=x.device)
        for i in range(batch_size):
            # Simple gaussian peak around 5ms
            peak_time = 4 + torch.randn(1, device=x.device) * 0.5
            peak_amp = 0.5 + torch.randn(1, device=x.device) * 0.1
            peak_signal = peak_amp * torch.exp(-((time_axis - peak_time) ** 2) / (2 * 0.5 ** 2))
            signal_out[i, 0] += peak_signal
        
        # Peak predictions
        peak_exist = self.peak_exist_head(static_params)  # [B, 1]
        peak_latency = torch.relu(self.peak_latency_head(static_params)) * 8 + 1  # 1-9ms
        peak_amplitude = torch.relu(self.peak_amplitude_head(static_params)) * 0.8 + 0.1  # 0.1-0.9
        
        # Classification (based on intensity)
        class_logits = self.class_head(static_params)
        
        # Threshold (based on intensity with some noise)
        threshold = torch.relu(self.threshold_head(static_params)) * 100 + 20  # 20-120 dB
        
        return {
            'recon': signal_out,
            'peak': (peak_exist, peak_latency, peak_amplitude),
            'class': class_logits,
            'threshold': threshold
        }


def create_mock_checkpoint(save_path: str):
    """Create a mock model checkpoint for testing."""
    model = MockABRModel()
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'input_channels': 1,
            'static_dim': 4,
            'base_channels': 64,
            'n_levels': 4,
            'sequence_length': 200,
            'signal_length': 200,
            'num_classes': 5,
            'num_transformer_layers': 3,
            'use_cross_attention': True,
            'use_positional_encoding': True,
            'film_dropout': 0.15,
            'use_cfg': True
        }
    }
    
    torch.save(checkpoint, save_path)
    print(f"📦 Mock checkpoint saved: {save_path}")


def test_inference_engine():
    """Test the ABR inference engine."""
    print("🧪 Testing ABR Inference Engine...")
    
    # Create mock checkpoint
    checkpoint_path = "test_model.pth"
    create_mock_checkpoint(checkpoint_path)
    
    # Create sample input
    input_path = create_sample_input("test_input.json")
    
    try:
        # Initialize inference engine
        print("\n1. Initializing inference engine...")
        engine = ABRInferenceEngine(
            model_path=checkpoint_path,
            device='cpu',  # Use CPU for testing
            cfg_scale=2.0,
            sampling_steps=10,  # Reduce steps for faster testing
            batch_size=4
        )
        print("✅ Inference engine initialized successfully")
        
        # Test JSON input
        print("\n2. Testing JSON input...")
        results = engine.run_inference(
            inputs=input_path,
            use_cfg=True,
            signal_length=200
        )
        print(f"✅ JSON inference completed: {len(results)} samples processed")
        
        # Test tensor input
        print("\n3. Testing tensor input...")
        static_params = torch.tensor([
            [30, 75, 25, 2.5],
            [50, 90, 15, 1.8]
        ], dtype=torch.float32)
        
        tensor_results = engine.run_inference(
            inputs=static_params,
            patient_ids=['T001', 'T002'],
            use_cfg=False,
            signal_length=200
        )
        print(f"✅ Tensor inference completed: {len(tensor_results)} samples processed")
        
        # Test result saving
        print("\n4. Testing result saving...")
        output_dir = "test_inference_output"
        saved_files = engine.save_results(
            results=results,
            output_dir=output_dir,
            save_json=True,
            save_csv=True,
            save_signals=True,
            save_visualizations=True
        )
        print(f"✅ Results saved: {list(saved_files.keys())}")
        
        # Validate results structure
        print("\n5. Validating results structure...")
        sample_result = results[0]
        required_fields = ['patient_id', 'static_parameters', 'generated_signal', 
                          'v_peak', 'predicted_class', 'threshold_dB']
        
        missing_fields = [field for field in required_fields if field not in sample_result]
        if missing_fields:
            print(f"❌ Missing fields: {missing_fields}")
            return False
        
        print("✅ All required fields present in results")
        
        # Print sample result
        print(f"\n📋 Sample Result for {sample_result['patient_id']}:")
        print(f"   Static params: {sample_result['static_parameters']}")
        print(f"   Predicted class: {sample_result['predicted_class']}")
        print(f"   Class confidence: {sample_result.get('class_confidence', 'N/A'):.3f}")
        print(f"   Threshold: {sample_result['threshold_dB']:.1f} dB")
        print(f"   Peak exists: {sample_result['v_peak']['exists']}")
        if sample_result['v_peak']['exists']:
            print(f"   Peak latency: {sample_result['v_peak']['latency']:.2f} ms")
            print(f"   Peak amplitude: {sample_result['v_peak']['amplitude']:.3f} μV")
        print(f"   Signal length: {len(sample_result['generated_signal'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        for file_path in [checkpoint_path, input_path]:
            if Path(file_path).exists():
                Path(file_path).unlink()


def test_cli_interface():
    """Test the CLI interface functionality."""
    print("\n🧪 Testing CLI Interface...")
    
    # Create test files
    checkpoint_path = "test_model_cli.pth"
    input_path = "test_input_cli.json"
    
    create_mock_checkpoint(checkpoint_path)
    create_sample_input(input_path)
    
    try:
        # Import main function
        from inference.inference import main
        
        # Mock command line arguments
        import sys
        original_argv = sys.argv.copy()
        
        sys.argv = [
            'inference.py',
            '--model_path', checkpoint_path,
            '--input_json', input_path,
            '--output_dir', 'test_cli_output',
            '--cfg_scale', '1.5',
            '--steps', '5',
            '--batch_size', '2',
            '--device', 'cpu',
            '--save_json',
            '--save_csv'
        ]
        
        # Run CLI
        exit_code = main()
        
        # Restore original argv
        sys.argv = original_argv
        
        if exit_code == 0:
            print("✅ CLI interface test passed")
            return True
        else:
            print("❌ CLI interface test failed")
            return False
    
    except Exception as e:
        print(f"❌ CLI test failed: {str(e)}")
        return False
    
    finally:
        # Cleanup
        for file_path in [checkpoint_path, input_path]:
            if Path(file_path).exists():
                Path(file_path).unlink()


def run_all_tests():
    """Run all inference tests."""
    print("🚀 ABR Inference Pipeline - Test Suite")
    print("=" * 50)
    
    test_results = {}
    
    # Test inference engine
    test_results['inference_engine'] = test_inference_engine()
    
    # Test CLI interface
    test_results['cli_interface'] = test_cli_interface()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<20} {status}")
    
    print("-" * 50)
    print(f"Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Inference pipeline is ready for use.")
        print("\n📋 Next Steps:")
        print("   1. Replace MockABRModel with your trained model")
        print("   2. Run: python inference/inference.py --model_path model.pth --input_json input.json")
        print("   3. Check outputs in specified output directory")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n✅ Inference pipeline test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Inference pipeline test failed!")
        sys.exit(1) 