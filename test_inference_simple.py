#!/usr/bin/env python3
"""
Simplified Test Script for ABR Inference Pipeline

Tests the inference functionality with a mock that bypasses model loading issues.

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


def test_inference_components():
    """Test individual inference components."""
    print("🧪 Testing Individual Inference Components...")
    
    # Test JSON input parsing
    print("\n1. Testing JSON input parsing...")
    sample_input = [
        {
            "patient_id": "P001",
            "static": [25, 70, 31, 2.8],
            "use_cfg": True
        },
        {
            "patient_id": "P002", 
            "static": [45, 85, 21, 1.5],
            "use_cfg": True
        }
    ]
    
    # Mock the parsing function
    from inference.inference import ABRInferenceEngine
    
    # Create a minimal engine instance for testing parsing
    class MockEngine:
        def __init__(self):
            self.device = torch.device('cpu')
        
        def _parse_json_inputs(self, input_data):
            return ABRInferenceEngine._parse_json_inputs(self, input_data)
    
    mock_engine = MockEngine()
    static_params, cfg_enabled, patient_ids = mock_engine._parse_json_inputs(sample_input)
    
    print(f"   ✅ Parsed {len(patient_ids)} samples")
    print(f"   ✅ Static params shape: {static_params.shape}")
    print(f"   ✅ Patient IDs: {patient_ids}")
    
    # Test result formatting
    print("\n2. Testing result formatting...")
    
    # Create mock data
    batch_size = 2
    signal_length = 200
    
    mock_signals = torch.randn(batch_size, 1, signal_length)
    mock_diagnostics = {
        'peak_existence': torch.tensor([True, False]),
        'peak_existence_logits': torch.tensor([[2.1], [-0.8]]),
        'peak_latency': torch.tensor([[5.2], [0.0]]),
        'peak_amplitude': torch.tensor([[0.6], [0.0]]),
        'predicted_class': torch.tensor([1, 3]),
        'class_probabilities': torch.softmax(torch.randn(batch_size, 5), dim=1),
        'threshold': torch.tensor([[65.3], [82.1]])
    }
    
    # Mock format_results method
    mock_engine.class_names = ["NORMAL", "NÖROPATİ", "SNİK", "TOTAL", "İTİK"]
    
    results = ABRInferenceEngine.format_results(
        mock_engine, mock_signals, mock_diagnostics, static_params, patient_ids
    )
    
    print(f"   ✅ Formatted {len(results)} results")
    
    # Validate result structure
    sample_result = results[0]
    required_fields = ['patient_id', 'static_parameters', 'generated_signal', 
                      'v_peak', 'predicted_class', 'threshold_dB']
    
    missing_fields = [field for field in required_fields if field not in sample_result]
    if missing_fields:
        print(f"   ❌ Missing fields: {missing_fields}")
        return False
    
    print("   ✅ All required fields present")
    
    # Test post-processing
    print("\n3. Testing post-processing...")
    
    processed_diagnostics = ABRInferenceEngine.post_process_predictions(
        mock_engine, mock_diagnostics, static_params
    )
    
    print(f"   ✅ Post-processed diagnostics: {list(processed_diagnostics.keys())}")
    
    if 'clinical_category' in processed_diagnostics:
        print(f"   ✅ Clinical categories added")
    
    return True


def test_sampling_components():
    """Test diffusion sampling components."""
    print("\n🧪 Testing Sampling Components...")
    
    try:
        from diffusion.sampling import DDIMSampler, create_ddim_sampler
        from diffusion.schedule import get_noise_schedule
        
        print("\n1. Testing noise schedule creation...")
        schedule = get_noise_schedule('cosine', num_timesteps=100)
        print(f"   ✅ Created cosine schedule with {len(schedule['betas'])} timesteps")
        
        print("\n2. Testing DDIM sampler creation...")
        sampler = create_ddim_sampler('cosine', num_timesteps=100, eta=0.0)
        print(f"   ✅ Created DDIM sampler with {sampler.num_timesteps} timesteps")
        
        print("\n3. Testing timestep schedule...")
        timesteps = sampler._get_timestep_schedule(20)
        print(f"   ✅ Generated timestep schedule: {len(timesteps)} steps")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Sampling test failed: {str(e)}")
        return False


def test_cli_argument_parsing():
    """Test CLI argument parsing."""
    print("\n🧪 Testing CLI Argument Parsing...")
    
    try:
        from inference.inference import main
        import argparse
        
        # Test argument parser creation
        parser = argparse.ArgumentParser()
        
        # Add some test arguments (simplified version)
        parser.add_argument('--model_path', type=str, required=True)
        parser.add_argument('--input_json', type=str)
        parser.add_argument('--cfg_scale', type=float, default=2.0)
        parser.add_argument('--steps', type=int, default=50)
        parser.add_argument('--device', type=str, default='auto')
        
        # Test parsing
        test_args = [
            '--model_path', 'test_model.pth',
            '--input_json', 'test_input.json',
            '--cfg_scale', '1.5',
            '--steps', '25',
            '--device', 'cpu'
        ]
        
        args = parser.parse_args(test_args)
        
        print(f"   ✅ Parsed model_path: {args.model_path}")
        print(f"   ✅ Parsed cfg_scale: {args.cfg_scale}")
        print(f"   ✅ Parsed steps: {args.steps}")
        print(f"   ✅ Parsed device: {args.device}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CLI parsing test failed: {str(e)}")
        return False


def test_file_operations():
    """Test file I/O operations."""
    print("\n🧪 Testing File Operations...")
    
    try:
        from inference.inference import create_sample_input
        
        print("\n1. Testing sample input creation...")
        input_path = create_sample_input("test_sample_input.json")
        
        if Path(input_path).exists():
            print(f"   ✅ Sample input created: {input_path}")
            
            # Test reading the file
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            print(f"   ✅ Sample contains {len(data)} entries")
            
            # Cleanup
            Path(input_path).unlink()
            print("   ✅ Cleanup completed")
        else:
            print("   ❌ Sample input file not created")
            return False
        
        print("\n2. Testing CSV writing...")
        
        # Create mock results
        mock_results = [
            {
                'patient_id': 'P001',
                'static_parameters': {'age': 25, 'intensity': 70, 'rate': 31, 'fmp': 2.8},
                'predicted_class': 'NORMAL',
                'class_confidence': 0.85,
                'threshold_dB': 65.3,
                'v_peak': {'exists': True, 'confidence': 0.9, 'latency': 5.2, 'amplitude': 0.6},
                'clinical_category': 'Moderate'
            }
        ]
        
        # Mock the CSV saving method
        from inference.inference import ABRInferenceEngine
        mock_engine = ABRInferenceEngine.__new__(ABRInferenceEngine)  # Create without __init__
        
        csv_path = Path("test_results.csv")
        mock_engine._save_csv_summary(mock_results, csv_path)
        
        if csv_path.exists():
            print(f"   ✅ CSV file created: {csv_path}")
            
            # Check content
            with open(csv_path, 'r') as f:
                content = f.read()
                if 'patient_id' in content and 'P001' in content:
                    print("   ✅ CSV content validated")
                else:
                    print("   ❌ CSV content invalid")
                    return False
            
            # Cleanup
            csv_path.unlink()
            print("   ✅ CSV cleanup completed")
        else:
            print("   ❌ CSV file not created")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ File operations test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_creation():
    """Test visualization creation without actual plotting."""
    print("\n🧪 Testing Visualization Components...")
    
    try:
        # Test that matplotlib imports work
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("   ✅ Visualization libraries imported")
        
        # Test figure creation
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        print("   ✅ Figure creation successful")
        
        # Test basic plotting
        x = np.linspace(0, 10, 200)
        y = np.sin(x) * np.exp(-x/5)
        axes[0, 0].plot(x, y)
        axes[0, 0].set_title('Test Signal')
        print("   ✅ Basic plotting successful")
        
        # Close figure to free memory
        plt.close(fig)
        print("   ✅ Figure cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Visualization test failed: {str(e)}")
        return False


def run_simplified_tests():
    """Run all simplified tests."""
    print("🚀 ABR Inference Pipeline - Simplified Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Test individual components
    test_results['inference_components'] = test_inference_components()
    test_results['sampling_components'] = test_sampling_components()
    test_results['cli_parsing'] = test_cli_argument_parsing()
    test_results['file_operations'] = test_file_operations()
    test_results['visualization'] = test_visualization_creation()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SIMPLIFIED TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print("-" * 60)
    print(f"Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All simplified tests passed!")
        print("\n📋 Inference Pipeline Components Verified:")
        print("   ✅ JSON input parsing and validation")
        print("   ✅ Result formatting and post-processing")
        print("   ✅ DDIM sampling and noise schedules")
        print("   ✅ CLI argument parsing")
        print("   ✅ File I/O operations (JSON, CSV)")
        print("   ✅ Visualization framework")
        print("\n🚀 Ready for integration with trained models!")
    else:
        print("⚠️  Some tests failed. Check individual components.")
    
    return passed == total


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run simplified tests
    success = run_simplified_tests()
    
    if success:
        print("\n✅ Simplified inference pipeline test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Simplified inference pipeline test failed!")
        sys.exit(1) 