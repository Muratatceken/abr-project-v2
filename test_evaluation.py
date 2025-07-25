#!/usr/bin/env python3
"""
Test Script for Comprehensive ABR Evaluation Pipeline

Tests all evaluation components with synthetic data to ensure functionality.

Author: AI Assistant
Date: January 2025
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from evaluation.comprehensive_eval import ABRComprehensiveEvaluator, create_evaluation_config


def create_synthetic_data(batch_size: int = 8, signal_length: int = 200) -> dict:
    """Create synthetic ABR data for testing."""
    
    # Generate synthetic signals
    time = np.linspace(0, 10, signal_length)  # 10ms
    signals = []
    
    for i in range(batch_size):
        # Create synthetic ABR with peak around 5ms
        peak_time = 4 + np.random.normal(0, 0.5)  # Peak around 4-6ms
        peak_amplitude = 0.5 + np.random.normal(0, 0.1)
        
        # Generate signal with peak
        signal = np.exp(-((time - peak_time) ** 2) / (2 * 0.5 ** 2)) * peak_amplitude
        signal += np.random.normal(0, 0.05, signal_length)  # Add noise
        signals.append(signal)
    
    signals = np.array(signals)
    
    # Create batch data
    batch_data = {
        'signal': torch.tensor(signals, dtype=torch.float32).unsqueeze(1),  # [B, 1, L]
        'static_params': torch.randn(batch_size, 4),  # [B, 4]
        'v_peak': torch.tensor([
            [4 + np.random.normal(0, 0.3), 0.5 + np.random.normal(0, 0.1)]  # latency, amplitude
            for _ in range(batch_size)
        ], dtype=torch.float32),  # [B, 2]
        'v_peak_mask': torch.ones(batch_size, 2, dtype=torch.bool),  # [B, 2]
        'target': torch.randint(0, 5, (batch_size,)),  # [B]
        'threshold': torch.tensor([60 + np.random.normal(0, 10) for _ in range(batch_size)], 
                                 dtype=torch.float32)  # [B]
    }
    
    return batch_data


def create_synthetic_model_outputs(batch_data: dict) -> dict:
    """Create synthetic model outputs that roughly match the input data."""
    batch_size = batch_data['signal'].size(0)
    
    # Add some noise to ground truth to simulate model predictions
    model_outputs = {
        'recon': batch_data['signal'] + torch.randn_like(batch_data['signal']) * 0.1,
        'peak': (
            torch.randn(batch_size, 1),  # existence logits
            batch_data['v_peak'][:, 0:1] + torch.randn(batch_size, 1) * 0.2,  # latency with noise
            batch_data['v_peak'][:, 1:2] + torch.randn(batch_size, 1) * 0.05  # amplitude with noise
        ),
        'class': torch.randn(batch_size, 5),  # classification logits
        'threshold': batch_data['threshold'].unsqueeze(1) + torch.randn(batch_size, 1) * 5  # threshold with noise
    }
    
    return model_outputs


def test_individual_components():
    """Test individual evaluation components."""
    print("🧪 Testing Individual Evaluation Components...")
    
    # Create evaluator
    config = create_evaluation_config()
    evaluator = ABRComprehensiveEvaluator(
        config=config,
        save_dir="test_evaluation_output"
    )
    
    # Create synthetic data
    batch_data = create_synthetic_data(batch_size=10)
    model_outputs = create_synthetic_model_outputs(batch_data)
    
    # Test signal reconstruction evaluation
    print("   Testing signal reconstruction evaluation...")
    recon_metrics = evaluator.evaluate_reconstruction(
        batch_data['signal'], model_outputs['recon']
    )
    print(f"   ✅ Signal reconstruction: {len(recon_metrics)} metrics computed")
    
    # Test peak estimation evaluation
    print("   Testing peak estimation evaluation...")
    peak_metrics = evaluator.evaluate_peak_estimation(
        model_outputs['peak'][0],  # existence logits
        batch_data['v_peak_mask'].any(dim=1).float(),  # existence ground truth
        model_outputs['peak'][1],  # latency predictions
        batch_data['v_peak'][:, 0],  # true latency
        model_outputs['peak'][2],  # amplitude predictions
        batch_data['v_peak'][:, 1],  # true amplitude
        batch_data['v_peak_mask']
    )
    print(f"   ✅ Peak estimation: {len(peak_metrics)} metrics computed")
    
    # Test classification evaluation
    print("   Testing classification evaluation...")
    class_metrics = evaluator.evaluate_classification(
        model_outputs['class'], batch_data['target']
    )
    print(f"   ✅ Classification: {len(class_metrics)} metrics computed")
    
    # Test threshold evaluation
    print("   Testing threshold estimation evaluation...")
    threshold_metrics = evaluator.evaluate_threshold_estimation(
        model_outputs['threshold'], batch_data['threshold']
    )
    print(f"   ✅ Threshold estimation: {len(threshold_metrics)} metrics computed")
    
    # Test failure mode detection
    print("   Testing clinical failure mode detection...")
    failure_modes = evaluator.compute_failure_modes(
        model_outputs['peak'][0],
        batch_data['v_peak_mask'].any(dim=1).float(),
        model_outputs['threshold'],
        batch_data['threshold'],
        torch.argmax(model_outputs['class'], dim=1),
        batch_data['target']
    )
    print(f"   ✅ Failure modes: {len(failure_modes)} modes detected")
    
    return True


def test_batch_evaluation():
    """Test complete batch evaluation."""
    print("\n🧪 Testing Complete Batch Evaluation...")
    
    # Create evaluator
    config = create_evaluation_config()
    evaluator = ABRComprehensiveEvaluator(
        config=config,
        save_dir="test_evaluation_output"
    )
    
    # Process multiple batches
    total_samples = 0
    for batch_idx in range(3):
        print(f"   Processing batch {batch_idx + 1}/3...")
        
        batch_data = create_synthetic_data(batch_size=8)
        model_outputs = create_synthetic_model_outputs(batch_data)
        
        # Evaluate batch
        batch_results = evaluator.evaluate_batch(batch_data, model_outputs, batch_idx)
        total_samples += batch_results['batch_size']
        
        print(f"   ✅ Batch {batch_idx + 1}: {batch_results['batch_size']} samples processed")
    
    # Compute aggregate metrics
    print("   Computing aggregate metrics...")
    aggregate_metrics = evaluator.compute_aggregate_metrics()
    
    print(f"   ✅ Aggregate metrics: {total_samples} total samples")
    print(f"   ✅ Categories evaluated: {len([k for k in aggregate_metrics.keys() if k not in ['total_batches', 'total_samples']])}")
    
    return aggregate_metrics


def test_visualizations():
    """Test diagnostic visualizations."""
    print("\n🧪 Testing Diagnostic Visualizations...")
    
    # Create evaluator
    config = create_evaluation_config()
    evaluator = ABRComprehensiveEvaluator(
        config=config,
        save_dir="test_evaluation_output"
    )
    
    # Create test data
    batch_data = create_synthetic_data(batch_size=5)
    model_outputs = create_synthetic_model_outputs(batch_data)
    
    # Create visualizations
    print("   Creating diagnostic visualizations...")
    visualizations = evaluator.create_batch_diagnostics(
        batch_data, model_outputs, batch_idx=0, n_samples=3
    )
    
    print(f"   ✅ Created {len(visualizations)} visualization types:")
    for viz_name in visualizations.keys():
        print(f"      - {viz_name}")
    
    return len(visualizations) > 0


def test_save_and_load():
    """Test saving and loading results."""
    print("\n🧪 Testing Save and Load Functionality...")
    
    # Create evaluator and run evaluation
    config = create_evaluation_config()
    evaluator = ABRComprehensiveEvaluator(
        config=config,
        save_dir="test_evaluation_output"
    )
    
    # Process a batch
    batch_data = create_synthetic_data(batch_size=10)
    model_outputs = create_synthetic_model_outputs(batch_data)
    evaluator.evaluate_batch(batch_data, model_outputs, 0)
    
    # Compute metrics
    aggregate_metrics = evaluator.compute_aggregate_metrics()
    
    # Save results
    print("   Saving evaluation results...")
    saved_files = evaluator.save_results("test_evaluation")
    
    print(f"   ✅ Saved {len(saved_files)} file types:")
    for file_type, file_path in saved_files.items():
        print(f"      - {file_type}: {file_path}")
        
        # Verify file exists
        if Path(file_path).exists():
            print(f"        ✅ File exists and readable")
        else:
            print(f"        ❌ File not found")
            return False
    
    return True


def run_all_tests():
    """Run all evaluation pipeline tests."""
    print("🚀 Comprehensive ABR Evaluation Pipeline - Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    try:
        # Test individual components
        test_results['components'] = test_individual_components()
        
        # Test batch evaluation
        aggregate_metrics = test_batch_evaluation()
        test_results['batch_evaluation'] = aggregate_metrics is not None
        
        # Test visualizations
        test_results['visualizations'] = test_visualizations()
        
        # Test save/load
        test_results['save_load'] = test_save_and_load()
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Print summary
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
        print("🎉 All tests passed! Evaluation pipeline is ready for use.")
        print("\n📋 Next Steps:")
        print("   1. Run: python evaluate.py --checkpoint model.pth --split test")
        print("   2. Check outputs in: test_evaluation_output/")
        print("   3. Review generated visualizations and metrics")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return passed == total


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run all tests
    success = run_all_tests()
    
    if success:
        print("\n✅ Evaluation pipeline test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Evaluation pipeline test failed!")
        sys.exit(1) 