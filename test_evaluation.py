#!/usr/bin/env python3
"""
Test script for the evaluation pipeline

This script tests the evaluation components with synthetic data
to ensure everything works correctly before running on real models.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.append('.')

from evaluation import SignalMetrics, SpectralMetrics, PerceptualMetrics, ABRSpecificMetrics
from evaluation import SignalGenerationEvaluator, EvaluationVisualizer, SignalAnalyzer
from evaluation.metrics import compute_all_metrics


def create_synthetic_abr_signal(length: int = 200, noise_level: float = 0.1) -> torch.Tensor:
    """Create a synthetic ABR-like signal for testing."""
    t = np.linspace(0, 20, length)  # 20ms signal
    
    # Create ABR-like waveform with typical waves I, III, V
    signal = np.zeros_like(t)
    
    # Wave I (around 1.5ms)
    wave1_center = 1.5
    signal += 0.3 * np.exp(-((t - wave1_center) / 0.3)**2) * np.sin(2 * np.pi * 1000 * (t - wave1_center) / 1000)
    
    # Wave III (around 3.8ms)  
    wave3_center = 3.8
    signal += 0.5 * np.exp(-((t - wave3_center) / 0.4)**2) * np.sin(2 * np.pi * 800 * (t - wave3_center) / 1000)
    
    # Wave V (around 5.6ms)
    wave5_center = 5.6
    signal += 0.7 * np.exp(-((t - wave5_center) / 0.5)**2) * np.sin(2 * np.pi * 600 * (t - wave5_center) / 1000)
    
    # Add some background activity
    signal += 0.1 * np.random.randn(length) * np.exp(-t / 10)
    
    # Add noise
    signal += noise_level * np.random.randn(length)
    
    return torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def test_metrics():
    """Test all metric functions."""
    print("🧪 Testing metrics...")
    
    # Create test signals
    real_signal = create_synthetic_abr_signal(200, 0.05)
    generated_signal = create_synthetic_abr_signal(200, 0.1) + 0.1 * torch.randn_like(real_signal)
    
    # Test individual metrics
    print("  Testing time-domain metrics...")
    mse = SignalMetrics.mse(generated_signal, real_signal)
    mae = SignalMetrics.mae(generated_signal, real_signal)
    snr = SignalMetrics.snr(generated_signal, real_signal)
    correlation = SignalMetrics.correlation(generated_signal, real_signal)
    
    print(f"    MSE: {mse:.4f}")
    print(f"    MAE: {mae:.4f}")
    print(f"    SNR: {snr:.2f} dB")
    print(f"    Correlation: {correlation['pearson_r']:.3f}")
    
    print("  Testing frequency-domain metrics...")
    freq_metrics = SpectralMetrics.frequency_response_error(generated_signal, real_signal)
    psd_metrics = SpectralMetrics.power_spectral_density_comparison(generated_signal, real_signal)
    
    print(f"    Frequency MSE: {freq_metrics['frequency_mse']:.4f}")
    print(f"    PSD Correlation: {psd_metrics['psd_correlation']:.3f}")
    
    print("  Testing perceptual metrics...")
    morph_sim = PerceptualMetrics.morphological_similarity(generated_signal, real_signal)
    envelope_sim = PerceptualMetrics.amplitude_envelope_similarity(generated_signal, real_signal)
    phase_coh = PerceptualMetrics.phase_coherence(generated_signal, real_signal)
    
    print(f"    Shape correlation: {morph_sim['shape_correlation']:.3f}")
    print(f"    Envelope similarity: {envelope_sim:.3f}")
    print(f"    Phase coherence: {phase_coh:.3f}")
    
    print("  Testing ABR-specific metrics...")
    wave_analysis = ABRSpecificMetrics.wave_component_analysis(generated_signal)
    print(f"    Wave I amplitude: {wave_analysis.get('wave_I_amplitude', 0):.3f}")
    print(f"    Wave III amplitude: {wave_analysis.get('wave_III_amplitude', 0):.3f}")
    print(f"    Wave V amplitude: {wave_analysis.get('wave_V_amplitude', 0):.3f}")
    
    # Test comprehensive metrics
    print("  Testing comprehensive metrics...")
    all_metrics = compute_all_metrics(generated_signal, real_signal)
    print(f"    Total metrics computed: {len(all_metrics)}")
    
    print("✅ Metrics test completed!")
    return True


def test_analysis():
    """Test signal analysis functions."""
    print("🔬 Testing signal analysis...")
    
    analyzer = SignalAnalyzer()
    test_signal = create_synthetic_abr_signal(200, 0.05)
    
    # Test signal properties analysis
    props = analyzer.analyze_signal_properties(test_signal)
    print(f"  Signal RMS: {props['rms']:.3f}")
    print(f"  Peak amplitude: {props['peak_amplitude']:.3f}")
    print(f"  Dynamic range: {props['dynamic_range']:.3f}")
    print(f"  Dominant frequency: {props['dominant_frequency']:.1f} Hz")
    
    # Test ABR wave analysis
    abr_analysis = analyzer.analyze_abr_waves(test_signal)
    detected_waves = abr_analysis['detected_waves']
    print(f"  Detected waves: {list(detected_waves.keys())}")
    
    # Test consistency analysis
    samples = [create_synthetic_abr_signal(200, 0.05) for _ in range(5)]
    consistency = analyzer.analyze_generation_consistency(samples)
    print(f"  Generation consistency: {consistency['overall_consistency_score']:.3f}")
    
    print("✅ Analysis test completed!")
    return True


def test_visualization():
    """Test visualization functions."""
    print("🎨 Testing visualization...")
    
    visualizer = EvaluationVisualizer(output_dir='test_plots')
    
    # Create test data
    generated_samples = [create_synthetic_abr_signal(200, 0.1) for _ in range(10)]
    real_samples = [create_synthetic_abr_signal(200, 0.05) for _ in range(10)]
    
    # Test sample comparison plots
    print("  Creating sample comparison plots...")
    plot_path = visualizer.plot_sample_comparisons(generated_samples[:3], real_samples[:3])
    print(f"    Saved to: {plot_path}")
    
    # Test overlay comparison
    print("  Creating overlay comparison...")
    overlay_path = visualizer.plot_overlay_comparison(generated_samples[:5], real_samples[:5])
    print(f"    Saved to: {overlay_path}")
    
    # Test frequency analysis
    print("  Creating frequency analysis...")
    freq_path = visualizer.plot_frequency_analysis(generated_samples[:3], real_samples[:3])
    print(f"    Saved to: {freq_path}")
    
    # Test metrics distribution (with mock metrics)
    mock_metrics = {
        'snr': {'mean': 15.2, 'std': 2.1, 'min': 10.1, 'max': 20.3, 'median': 15.0},
        'correlation': {'mean': 0.82, 'std': 0.05, 'min': 0.72, 'max': 0.91, 'median': 0.83},
        'rmse': {'mean': 0.08, 'std': 0.02, 'min': 0.05, 'max': 0.12, 'median': 0.08}
    }
    print("  Creating metrics distribution...")
    metrics_path = visualizer.plot_metrics_distribution(mock_metrics)
    print(f"    Saved to: {metrics_path}")
    
    # Test interactive dashboard
    print("  Creating interactive dashboard...")
    dashboard_path = visualizer.create_interactive_dashboard(
        generated_samples[:5], real_samples[:5], mock_metrics
    )
    print(f"    Saved to: {dashboard_path}")
    
    print("✅ Visualization test completed!")
    return True


def main():
    """Run all tests."""
    print("🚀 Testing ABR Evaluation Pipeline")
    print("=" * 50)
    
    success = True
    
    try:
        success &= test_metrics()
        success &= test_analysis()
        success &= test_visualization()
        
        if success:
            print("\n🎉 All tests passed!")
            print("The evaluation pipeline is ready to use.")
            print("\nNext steps:")
            print("1. Train your model using train.py")
            print("2. Run evaluation using evaluate_model.py")
            print("3. Check results in the output directory")
        else:
            print("\n❌ Some tests failed!")
            print("Please check the error messages above.")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)