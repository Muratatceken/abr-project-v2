#!/usr/bin/env python3
"""
Test script to verify evaluation fixes work correctly
"""

import torch
import numpy as np
import json
from pathlib import Path

def test_evaluation_fixes():
    """Test the fixed evaluation components with sample data."""
    
    print("🧪 Testing Evaluation Fixes")
    print("=" * 40)
    
    # Create synthetic test data
    batch_size = 100
    signal_length = 200
    
    # Synthetic predictions and ground truth
    predictions = {
        'signals': torch.randn(batch_size, signal_length),
        'classifications': torch.randn(batch_size, 5),  # 5 classes
        'peak_predictions': torch.randn(batch_size, 5),  # [existence, latency, amplitude, ...]
        'thresholds': torch.randn(batch_size, 2)  # Multiple threshold outputs
    }
    
    # Synthetic ground truth
    ground_truth = {
        'signals': torch.randn(batch_size, signal_length),
        'classifications': torch.randint(0, 5, (batch_size,)),
        'peaks': torch.randn(batch_size, 2, 2),  # [latency, amplitude], [value, mask]
        'thresholds': torch.randn(batch_size)
    }
    
    # Test signal quality evaluation
    print("\n1. Testing Signal Quality Evaluation...")
    try:
        from evaluation.comprehensive_evaluator import ComprehensiveEvaluationMethods
        
        class TestEvaluator(ComprehensiveEvaluationMethods):
            def __init__(self):
                self.predictions = predictions
                self.ground_truth = ground_truth
        
        evaluator = TestEvaluator()
        signal_results = evaluator._evaluate_signal_quality()
        
        print(f"   ✅ Signal correlation: {signal_results['basic_metrics']['correlation_mean']:.3f}")
        print(f"   ✅ Signal SNR: {signal_results['basic_metrics']['snr_mean_db']:.1f} dB")
        
        # Check for NaN/Inf values
        for key, value in signal_results['basic_metrics'].items():
            if np.isnan(value) or np.isinf(value):
                print(f"   ❌ {key} is NaN/Inf: {value}")
            else:
                print(f"   ✅ {key}: {value:.3f}")
                
    except Exception as e:
        print(f"   ❌ Signal quality evaluation failed: {e}")
    
    # Test peak detection evaluation  
    print("\n2. Testing Peak Detection Evaluation...")
    try:
        peak_results = evaluator._evaluate_peak_detection()
        
        latency_corr = peak_results.get('latency_metrics', {}).get('correlation', 'N/A')
        amplitude_corr = peak_results.get('amplitude_metrics', {}).get('correlation', 'N/A')
        
        print(f"   ✅ Peak latency correlation: {latency_corr}")
        print(f"   ✅ Peak amplitude correlation: {amplitude_corr}")
        
        # Check for NaN values
        if isinstance(latency_corr, float) and (np.isnan(latency_corr) or np.isinf(latency_corr)):
            print(f"   ❌ Latency correlation is NaN/Inf")
        if isinstance(amplitude_corr, float) and (np.isnan(amplitude_corr) or np.isinf(amplitude_corr)):
            print(f"   ❌ Amplitude correlation is NaN/Inf")
            
    except Exception as e:
        print(f"   ❌ Peak detection evaluation failed: {e}")
    
    # Test threshold evaluation
    print("\n3. Testing Threshold Evaluation...")
    try:
        threshold_results = evaluator._evaluate_threshold_regression()
        
        r2_score = threshold_results['regression_metrics']['r2_score']
        correlation = threshold_results['regression_metrics']['correlation']
        
        print(f"   ✅ Threshold R²: {r2_score:.3f}")
        print(f"   ✅ Threshold correlation: {correlation:.3f}")
        
    except Exception as e:
        print(f"   ❌ Threshold evaluation failed: {e}")
    
    print(f"\n🎯 Evaluation fixes test completed!")

if __name__ == "__main__":
    test_evaluation_fixes()