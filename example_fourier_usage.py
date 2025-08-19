#!/usr/bin/env python3
"""
Example Usage of ABR Fourier Analysis Script

This script demonstrates how to use the ABRFourierAnalyzer class
for analyzing ABR signals with Fourier transforms.
"""

from abr_fourier_analysis import ABRFourierAnalyzer
import numpy as np

def main():
    """Example usage of the ABR Fourier analyzer."""
    
    print("🚀 ABR Fourier Analysis Example")
    print("=" * 40)
    
    # Initialize the analyzer
    pkl_file = "data/processed/ultimate_dataset_with_clinical_thresholds.pkl"
    analyzer = ABRFourierAnalyzer(pkl_file, total_duration=13.7)
    
    # Example 1: Analyze 5 random samples
    print("\n📊 Example 1: Analyzing 5 random samples")
    samples = analyzer.get_sample_signals(n_samples=5, random_seed=123)
    
    # Create basic Fourier plots
    analyzer.create_fourier_plots(samples, save_path="test_plots/example_fourier_5_samples.png")
    
    # Example 2: Analyze 15 samples with frequency analysis
    print("\n📊 Example 2: Comprehensive frequency analysis with 15 samples")
    samples_large = analyzer.get_sample_signals(n_samples=15, random_seed=456)
    
    # Create comprehensive frequency analysis
    analyzer.create_frequency_analysis_plot(samples_large, 
                                           save_path="test_plots/example_frequency_analysis.png")
    
    # Print detailed summary
    analyzer.print_analysis_summary(samples_large)
    
    # Example 3: Manual Fourier transform on a single signal
    print("\n📊 Example 3: Manual Fourier transform on single signal")
    single_sample = samples[0]
    signal = single_sample['signal']
    
    # Apply Fourier transform manually
    frequencies, magnitude, phase = analyzer.apply_fourier_transform(signal)
    
    print(f"   Signal length: {len(signal)} samples")
    print(f"   Frequency resolution: {frequencies[1] - frequencies[0]:.2f} Hz")
    print(f"   Maximum frequency: {frequencies[-1]:.1f} Hz")
    print(f"   Peak magnitude: {np.max(magnitude):.2f}")
    print(f"   Frequency at peak magnitude: {frequencies[np.argmax(magnitude)]:.1f} Hz")
    
    # Find dominant frequencies (top 5)
    top_indices = np.argsort(magnitude)[-5:][::-1]
    print(f"   Top 5 dominant frequencies:")
    for i, idx in enumerate(top_indices):
        print(f"     {i+1}. {frequencies[idx]:.1f} Hz (magnitude: {magnitude[idx]:.2f})")
    
    print("\n✅ All examples completed successfully!")
    print("📁 Check the 'test_plots' directory for generated plots.")

if __name__ == "__main__":
    main()
