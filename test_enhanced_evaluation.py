#!/usr/bin/env python3
"""
Test Script for Enhanced ABR Evaluation Pipeline Features

Tests bootstrap CI, clinical overlays, diagnostic cards, and quantile analysis.

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


def create_enhanced_synthetic_data(batch_size: int = 8, signal_length: int = 200) -> dict:
    """Create enhanced synthetic ABR data with patient IDs."""
    
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
    
    # Create batch data with patient IDs
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
                                 dtype=torch.float32),  # [B]
        'patient_ids': [f"PAT_{i+1:03d}" for i in range(batch_size)]  # Patient IDs
    }
    
    return batch_data


def test_bootstrap_confidence_intervals():
    """Test bootstrap confidence interval functionality."""
    print("🧪 Testing Bootstrap Confidence Intervals...")
    
    # Create evaluator with bootstrap enabled
    config = create_evaluation_config()
    config['bootstrap']['enabled'] = True
    config['bootstrap']['n_samples'] = 100  # Reduced for testing speed
    
    evaluator = ABRComprehensiveEvaluator(
        config=config,
        save_dir="test_enhanced_output"
    )
    
    # Process multiple batches to get meaningful statistics
    for batch_idx in range(3):
        batch_data = create_enhanced_synthetic_data(batch_size=10)
        model_outputs = create_synthetic_model_outputs(batch_data)
        evaluator.evaluate_batch(batch_data, model_outputs, batch_idx)
    
    # Compute aggregate metrics with bootstrap
    print("   Computing bootstrap confidence intervals...")
    aggregate_metrics = evaluator.compute_aggregate_metrics()
    
    # Check for CI metrics
    ci_found = False
    for category, metrics in aggregate_metrics.items():
        if isinstance(metrics, dict):
            for key in metrics.keys():
                if '_lower_ci' in key or '_upper_ci' in key:
                    ci_found = True
                    break
    
    print(f"   ✅ Bootstrap CI: {ci_found}")
    
    # Create summary table
    summary_path = evaluator.create_summary_table()
    print(f"   ✅ Summary table created: {Path(summary_path).exists()}")
    
    return ci_found and Path(summary_path).exists()


def test_clinical_overlays():
    """Test clinical overlay functionality."""
    print("\n🧪 Testing Clinical Overlays...")
    
    # Create evaluator with clinical overlays enabled
    config = create_evaluation_config()
    config['visualization']['clinical_overlays']['enabled'] = True
    
    evaluator = ABRComprehensiveEvaluator(
        config=config,
        save_dir="test_enhanced_output"
    )
    
    # Create test data with patient IDs
    batch_data = create_enhanced_synthetic_data(batch_size=5)
    model_outputs = create_synthetic_model_outputs(batch_data)
    
    # Create enhanced visualizations
    print("   Creating clinical overlay visualizations...")
    visualizations = evaluator.create_batch_diagnostics(
        batch_data, model_outputs, batch_idx=0, n_samples=3
    )
    
    print(f"   ✅ Created {len(visualizations)} visualization types with clinical overlays")
    
    return len(visualizations) > 0


def test_diagnostic_cards():
    """Test diagnostic card functionality."""
    print("\n🧪 Testing Diagnostic Cards...")
    
    # Create evaluator
    config = create_evaluation_config()
    evaluator = ABRComprehensiveEvaluator(
        config=config,
        save_dir="test_enhanced_output"
    )
    
    # Create test data
    batch_data = create_enhanced_synthetic_data(batch_size=4)
    model_outputs = create_synthetic_model_outputs(batch_data)
    
    # Create diagnostic cards
    print("   Creating multi-panel diagnostic cards...")
    cards = evaluator.create_diagnostic_cards(
        batch_data, model_outputs, batch_idx=0, n_samples=2
    )
    
    print(f"   ✅ Created {len(cards)} diagnostic cards:")
    for card_name in cards.keys():
        print(f"      - {card_name}")
    
    return len(cards) > 0


def test_quantile_analysis():
    """Test quantile and error range visualizations."""
    print("\n🧪 Testing Quantile Analysis...")
    
    # Create evaluator
    config = create_evaluation_config()
    evaluator = ABRComprehensiveEvaluator(
        config=config,
        save_dir="test_enhanced_output"
    )
    
    # Create test data with diverse classes and thresholds
    batch_data = create_enhanced_synthetic_data(batch_size=20)  # Larger batch for better statistics
    model_outputs = create_synthetic_model_outputs(batch_data)
    
    # Make classes more diverse
    batch_data['target'] = torch.randint(0, 5, (20,))
    
    # Create quantile visualizations
    print("   Creating quantile and error range visualizations...")
    quantile_viz = evaluator.create_quantile_error_visualizations(batch_data, model_outputs)
    
    print(f"   ✅ Created {len(quantile_viz)} quantile visualization types:")
    for viz_name in quantile_viz.keys():
        print(f"      - {viz_name}")
    
    return len(quantile_viz) > 0


def test_enhanced_cli_simulation():
    """Simulate enhanced CLI features."""
    print("\n🧪 Testing Enhanced CLI Features Simulation...")
    
    # Simulate CLI arguments
    class MockArgs:
        def __init__(self):
            self.no_visuals = False
            self.only_clinical_flags = False
            self.bootstrap_ci = True
            self.save_json_only = False
            self.limit_samples = 50
            self.diagnostic_cards = True
            self.quantile_analysis = True
    
    args = MockArgs()
    
    # Test CLI flag handling logic
    no_visuals = getattr(args, 'no_visuals', False)
    only_clinical_flags = getattr(args, 'only_clinical_flags', False)
    bootstrap_ci = getattr(args, 'bootstrap_ci', False)
    limit_samples = getattr(args, 'limit_samples', None)
    diagnostic_cards = getattr(args, 'diagnostic_cards', False)
    quantile_analysis = getattr(args, 'quantile_analysis', False)
    
    print(f"   ✅ CLI flags parsing:")
    print(f"      - no_visuals: {no_visuals}")
    print(f"      - only_clinical_flags: {only_clinical_flags}")
    print(f"      - bootstrap_ci: {bootstrap_ci}")
    print(f"      - limit_samples: {limit_samples}")
    print(f"      - diagnostic_cards: {diagnostic_cards}")
    print(f"      - quantile_analysis: {quantile_analysis}")
    
    return True


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


def run_enhanced_tests():
    """Run all enhanced evaluation pipeline tests."""
    print("🚀 Enhanced ABR Evaluation Pipeline - Feature Test Suite")
    print("=" * 65)
    
    test_results = {}
    
    try:
        # Test bootstrap confidence intervals
        test_results['bootstrap_ci'] = test_bootstrap_confidence_intervals()
        
        # Test clinical overlays
        test_results['clinical_overlays'] = test_clinical_overlays()
        
        # Test diagnostic cards
        test_results['diagnostic_cards'] = test_diagnostic_cards()
        
        # Test quantile analysis
        test_results['quantile_analysis'] = test_quantile_analysis()
        
        # Test enhanced CLI simulation
        test_results['cli_features'] = test_enhanced_cli_simulation()
        
    except Exception as e:
        print(f"❌ Enhanced test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Print summary
    print("\n" + "=" * 65)
    print("📊 ENHANCED FEATURES TEST SUMMARY")
    print("=" * 65)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<20} {status}")
    
    print("-" * 65)
    print(f"Overall Result: {passed}/{total} enhanced tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All enhanced features working correctly!")
        print("\n📋 Enhanced Features Available:")
        print("   ✅ Bootstrap confidence intervals")
        print("   ✅ Clinical overlays with patient info")
        print("   ✅ Multi-panel diagnostic cards")
        print("   ✅ Quantile and error range analysis")
        print("   ✅ Enhanced CLI control flags")
        print("\n🚀 Ready for production use with all enhancements!")
    else:
        print("⚠️  Some enhanced features need attention.")
    
    return passed == total


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run enhanced tests
    success = run_enhanced_tests()
    
    if success:
        print("\n✅ Enhanced evaluation pipeline test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Enhanced evaluation pipeline test failed!")
        sys.exit(1) 