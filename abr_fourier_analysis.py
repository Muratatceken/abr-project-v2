#!/usr/bin/env python3
"""
ABR Fourier Transform Analysis Script

This script loads ABR signals from the pkl file, applies Fourier transforms,
and creates comprehensive plots comparing original and transformed signals.

Features:
- Load ABR signals from ultimate dataset
- Apply Fast Fourier Transform (FFT)
- Generate comparison plots for original vs frequency domain
- Support for multiple samples analysis
- Interactive plotting with matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import seaborn as sns

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class ABRFourierAnalyzer:
    """Class for analyzing ABR signals using Fourier transforms."""
    
    def __init__(self, pkl_file_path: str, total_duration: float = 13.7):
        """
        Initialize the ABR Fourier Analyzer.
        
        Args:
            pkl_file_path (str): Path to the pkl file containing ABR data
            total_duration (float): Total signal duration in seconds (default: 13.7 seconds)
        """
        self.pkl_file_path = pkl_file_path
        self.total_duration = total_duration
        self.data = None
        self.metadata = None
        self.label_encoder = None
        
        # Calculate sampling rate based on signal length and duration
        # Will be updated after loading the first sample
        
        # Load the dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the ABR dataset from the pkl file."""
        try:
            print(f"🔄 Loading ABR dataset from {self.pkl_file_path}")
            
            # Load the main dataset
            dataset = joblib.load(self.pkl_file_path)
            
            if isinstance(dataset, dict):
                # Ultimate dataset format with metadata
                self.data = dataset.get('data', dataset)
                self.metadata = dataset.get('metadata', {})
                self.label_encoder = dataset.get('label_encoder', None)
            else:
                # Simple list format
                self.data = dataset
                self.metadata = {}
                self.label_encoder = None
            
            print(f"✅ Successfully loaded {len(self.data)} ABR samples")
            
            # Print sample structure for verification and calculate sampling rate
            if len(self.data) > 0:
                sample = self.data[0]
                signal_length = len(sample['signal'])
                self.sampling_rate = signal_length / self.total_duration
                
                print(f"📊 Sample structure: {list(sample.keys())}")
                print(f"   Signal shape: {sample['signal'].shape}")
                print(f"   Signal duration: {self.total_duration} seconds")
                print(f"   Calculated sampling rate: {self.sampling_rate:.2f} Hz")
                if 'static_params' in sample:
                    print(f"   Static params shape: {sample['static_params'].shape}")
                if 'target' in sample:
                    print(f"   Target: {sample['target']}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ PKL file not found: {self.pkl_file_path}")
        except Exception as e:
            raise RuntimeError(f"❌ Error loading dataset: {e}")
    
    def apply_fourier_transform(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Fast Fourier Transform to an ABR signal.
        
        Args:
            signal (np.ndarray): Input ABR signal
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - frequencies: Frequency bins
                - magnitude: Magnitude spectrum
                - phase: Phase spectrum
        """
        # Apply FFT
        fft_result = np.fft.fft(signal)
        
        # Calculate frequency bins
        n_samples = len(signal)
        frequencies = np.fft.fftfreq(n_samples, 1/self.sampling_rate)
        
        # Calculate magnitude and phase
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        
        # Return only positive frequencies (due to symmetry)
        positive_freq_mask = frequencies >= 0
        
        return (frequencies[positive_freq_mask], 
                magnitude[positive_freq_mask], 
                phase[positive_freq_mask])
    
    def get_sample_signals(self, n_samples: int = 10, random_seed: int = 42) -> List[Dict]:
        """
        Get a specified number of sample signals for analysis.
        
        Args:
            n_samples (int): Number of samples to extract
            random_seed (int): Random seed for reproducibility
            
        Returns:
            List[Dict]: List of sample dictionaries with signal data
        """
        np.random.seed(random_seed)
        
        if n_samples > len(self.data):
            print(f"⚠️  Requested {n_samples} samples, but only {len(self.data)} available.")
            n_samples = len(self.data)
        
        # Randomly select samples
        indices = np.random.choice(len(self.data), n_samples, replace=False)
        selected_samples = [self.data[i] for i in indices]
        
        print(f"📝 Selected {len(selected_samples)} samples for Fourier analysis")
        
        return selected_samples
    
    def create_fourier_plots(self, samples: List[Dict], save_path: str = None):
        """
        Create comprehensive plots comparing original signals and their Fourier transforms.
        
        Args:
            samples (List[Dict]): List of sample dictionaries
            save_path (str, optional): Path to save the plots
        """
        n_samples = len(samples)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 4 * n_samples))
        
        for i, sample in enumerate(samples):
            signal = sample['signal']
            patient_id = sample.get('patient_id', f'Sample_{i}')
            target = sample.get('target', 'Unknown')
            
            # Get target name if label encoder is available
            target_name = target
            if self.label_encoder is not None and isinstance(target, (int, np.integer)):
                try:
                    target_name = self.label_encoder.classes_[target]
                except (IndexError, AttributeError):
                    target_name = f'Class_{target}'
            
            # Apply Fourier transform
            frequencies, magnitude, phase = self.apply_fourier_transform(signal)
            
            # Time axis for original signal (0 to 13.7 seconds)
            time_axis = np.linspace(0, self.total_duration, len(signal))
            
            # Plot original signal
            plt.subplot(n_samples, 4, i*4 + 1)
            plt.plot(time_axis, signal, 'b-', linewidth=1)
            plt.title(f'Original Signal - {patient_id}\nTarget: {target_name}', fontsize=10)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude (μV)')
            plt.grid(True, alpha=0.3)
            
            # Plot magnitude spectrum
            plt.subplot(n_samples, 4, i*4 + 2)
            plt.plot(frequencies, magnitude, 'r-', linewidth=1)
            plt.title(f'Magnitude Spectrum - {patient_id}', fontsize=10)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.xlim(0, min(5, self.sampling_rate/2))  # Limit to 5Hz for better visualization with low sampling rate
            plt.grid(True, alpha=0.3)
            
            # Plot phase spectrum
            plt.subplot(n_samples, 4, i*4 + 3)
            plt.plot(frequencies, phase, 'g-', linewidth=1)
            plt.title(f'Phase Spectrum - {patient_id}', fontsize=10)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Phase (radians)')
            plt.xlim(0, min(5, self.sampling_rate/2))
            plt.grid(True, alpha=0.3)
            
            # Plot log magnitude spectrum for better visualization of small components
            plt.subplot(n_samples, 4, i*4 + 4)
            log_magnitude = 20 * np.log10(magnitude + 1e-12)  # Add small value to avoid log(0)
            plt.plot(frequencies, log_magnitude, 'm-', linewidth=1)
            plt.title(f'Log Magnitude (dB) - {patient_id}', fontsize=10)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.xlim(0, min(5, self.sampling_rate/2))
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {save_path}")
        
        plt.show()
    
    def create_frequency_analysis_plot(self, samples: List[Dict], save_path: str = None):
        """
        Create a comprehensive frequency analysis plot showing ABR-relevant frequency bands.
        
        Args:
            samples (List[Dict]): List of sample dictionaries
            save_path (str, optional): Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define frequency bands appropriate for the sampling rate
        nyquist_freq = self.sampling_rate / 2
        freq_bands = {
            'Very Low': (0, nyquist_freq * 0.25),
            'Low': (nyquist_freq * 0.25, nyquist_freq * 0.5),
            'Mid': (nyquist_freq * 0.5, nyquist_freq * 0.75),
            'High': (nyquist_freq * 0.75, nyquist_freq)
        }
        
        all_frequencies = []
        all_magnitudes = []
        band_powers = {band: [] for band in freq_bands.keys()}
        
        # Process all samples
        for sample in samples:
            signal = sample['signal']
            frequencies, magnitude, _ = self.apply_fourier_transform(signal)
            
            all_frequencies.append(frequencies)
            all_magnitudes.append(magnitude)
            
            # Calculate power in each frequency band
            for band_name, (low_freq, high_freq) in freq_bands.items():
                band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
                band_power = np.sum(magnitude[band_mask] ** 2)
                band_powers[band_name].append(band_power)
        
        # Plot 1: Average magnitude spectrum
        axes[0, 0].set_title('Average Magnitude Spectrum Across Samples', fontsize=12, weight='bold')
        if all_frequencies:
            # Find common frequency range
            min_len = min(len(freqs) for freqs in all_frequencies)
            avg_frequencies = all_frequencies[0][:min_len]
            avg_magnitude = np.mean([mag[:min_len] for mag in all_magnitudes], axis=0)
            
            axes[0, 0].plot(avg_frequencies, avg_magnitude, 'b-', linewidth=2, label='Average')
            
            # Highlight ABR-relevant frequency bands
            colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'orange']
            for i, (band_name, (low_freq, high_freq)) in enumerate(freq_bands.items()):
                axes[0, 0].axvspan(low_freq, high_freq, alpha=0.3, color=colors[i], label=f'{band_name} ({low_freq}-{high_freq} Hz)')
        
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Magnitude')
        axes[0, 0].set_xlim(0, min(5, self.sampling_rate/2))
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Power distribution by frequency bands
        axes[0, 1].set_title('Power Distribution by Frequency Bands', fontsize=12, weight='bold')
        band_names = list(freq_bands.keys())
        band_power_means = [np.mean(band_powers[band]) for band in band_names]
        band_power_stds = [np.std(band_powers[band]) for band in band_names]
        
        bars = axes[0, 1].bar(band_names, band_power_means, yerr=band_power_stds, 
                             capsize=5, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Average Power')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, band_power_means):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{mean_val:.2e}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Individual sample spectrograms (first 5 samples)
        axes[1, 0].set_title('Individual Sample Magnitude Spectra', fontsize=12, weight='bold')
        for i, sample in enumerate(samples[:5]):
            signal = sample['signal']
            frequencies, magnitude, _ = self.apply_fourier_transform(signal)
            patient_id = sample.get('patient_id', f'Sample_{i}')
            
            # Normalize magnitude for better visualization
            normalized_magnitude = magnitude / np.max(magnitude)
            axes[1, 0].plot(frequencies, normalized_magnitude, alpha=0.7, 
                           linewidth=1, label=f'{patient_id}')
        
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Normalized Magnitude')
        axes[1, 0].set_xlim(0, min(5, self.sampling_rate/2))
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Frequency band power correlation
        axes[1, 1].set_title('Frequency Band Power Correlations', fontsize=12, weight='bold')
        
        # Create correlation matrix
        band_data = np.array([band_powers[band] for band in band_names]).T
        correlation_matrix = np.corrcoef(band_data.T)
        
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_xticks(range(len(band_names)))
        axes[1, 1].set_yticks(range(len(band_names)))
        axes[1, 1].set_xticklabels(band_names)
        axes[1, 1].set_yticklabels(band_names)
        
        # Add correlation values as text
        for i in range(len(band_names)):
            for j in range(len(band_names)):
                text = axes[1, 1].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                      ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 1], label='Correlation Coefficient')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Frequency analysis plot saved to: {save_path}")
        
        plt.show()
    
    def print_analysis_summary(self, samples: List[Dict]):
        """
        Print a summary of the Fourier analysis results.
        
        Args:
            samples (List[Dict]): List of analyzed samples
        """
        print("\n" + "="*60)
        print("📊 ABR FOURIER TRANSFORM ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📈 Dataset Information:")
        print(f"   Total samples in dataset: {len(self.data):,}")
        print(f"   Samples analyzed: {len(samples)}")
        print(f"   Sampling rate: {self.sampling_rate:.2f} Hz")
        print(f"   Signal length: {len(samples[0]['signal'])} samples")
        print(f"   Signal duration: {self.total_duration} seconds")
        
        # Analyze frequency content
        freq_stats = []
        for sample in samples:
            signal = sample['signal']
            frequencies, magnitude, _ = self.apply_fourier_transform(signal)
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(magnitude)
            dominant_freq = frequencies[dominant_freq_idx]
            
            # Calculate power in signal-relevant bands (0 to Nyquist/2)
            nyquist_freq = self.sampling_rate / 2
            signal_band_mask = (frequencies >= 0) & (frequencies <= nyquist_freq * 0.5)
            signal_power = np.sum(magnitude[signal_band_mask] ** 2)
            total_power = np.sum(magnitude ** 2)
            signal_power_ratio = signal_power / total_power if total_power > 0 else 0
            
            freq_stats.append({
                'dominant_freq': dominant_freq,
                'signal_power_ratio': signal_power_ratio,
                'total_power': total_power
            })
        
        # Calculate summary statistics
        dominant_freqs = [stat['dominant_freq'] for stat in freq_stats]
        signal_ratios = [stat['signal_power_ratio'] for stat in freq_stats]
        total_powers = [stat['total_power'] for stat in freq_stats]
        
        print(f"\n🔍 Frequency Analysis Results:")
        print(f"   Dominant frequency range: {np.min(dominant_freqs):.1f} - {np.max(dominant_freqs):.1f} Hz")
        print(f"   Average dominant frequency: {np.mean(dominant_freqs):.1f} ± {np.std(dominant_freqs):.1f} Hz")
        print(f"   Low frequency power ratio (0-{self.sampling_rate/4:.1f} Hz): {np.mean(signal_ratios):.3f} ± {np.std(signal_ratios):.3f}")
        print(f"   Average total power: {np.mean(total_powers):.2e} ± {np.std(total_powers):.2e}")
        
        # Analyze by class if available
        if self.label_encoder is not None:
            print(f"\n📊 Analysis by Hearing Loss Type:")
            class_stats = {}
            for sample in samples:
                target = sample.get('target', -1)
                if target in class_stats:
                    class_stats[target].append(sample)
                else:
                    class_stats[target] = [sample]
            
            for target, class_samples in class_stats.items():
                if target >= 0 and target < len(self.label_encoder.classes_):
                    class_name = self.label_encoder.classes_[target]
                    print(f"   {class_name}: {len(class_samples)} samples")


def main():
    """Main function to run the ABR Fourier analysis."""
    parser = argparse.ArgumentParser(description='ABR Fourier Transform Analysis')
    parser.add_argument('--pkl_file', type=str, 
                       default='data/processed/ultimate_dataset_with_clinical_thresholds.pkl',
                       help='Path to the PKL file containing ABR data')
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of samples to analyze')
    parser.add_argument('--duration', type=float, default=13.7,
                       help='Total signal duration in seconds')
    parser.add_argument('--save_plots', type=str, default=None,
                       help='Directory to save plots (optional)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.save_plots:
        Path(args.save_plots).mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize analyzer
        analyzer = ABRFourierAnalyzer(args.pkl_file, args.duration)
        
        # Get sample signals
        samples = analyzer.get_sample_signals(args.n_samples, args.seed)
        
        # Create plots
        print("\n🎨 Creating Fourier transform comparison plots...")
        plot_path = Path(args.save_plots) / 'fourier_comparison.png' if args.save_plots else None
        analyzer.create_fourier_plots(samples, plot_path)
        
        print("\n🎨 Creating frequency analysis plots...")
        freq_plot_path = Path(args.save_plots) / 'frequency_analysis.png' if args.save_plots else None
        analyzer.create_frequency_analysis_plot(samples, freq_plot_path)
        
        # Print summary
        analyzer.print_analysis_summary(samples)
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
