"""
Signal Analysis Tools

This module provides specialized analysis tools for ABR signals,
including physiological analysis, statistical tests, and quality assessments.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats, signal
from scipy.signal import find_peaks
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


class SignalAnalyzer:
    """Comprehensive signal analysis tools for ABR evaluation."""
    
    def __init__(self):
        """Initialize the signal analyzer."""
        self.abr_wave_latencies = {
            'wave_I': (1.0, 2.0),    # ms
            'wave_III': (3.3, 4.3),  # ms  
            'wave_V': (5.1, 6.1)     # ms
        }
    
    def analyze_signal_properties(self, signal: torch.Tensor, sr: int = 1000) -> Dict[str, float]:
        """
        Analyze basic signal properties.
        
        Args:
            signal: Input signal tensor
            sr: Sampling rate
            
        Returns:
            Dictionary of signal properties
        """
        signal_np = signal.squeeze().cpu().numpy()
        
        properties = {
            # Amplitude properties
            'rms': float(np.sqrt(np.mean(signal_np**2))),
            'peak_amplitude': float(np.max(np.abs(signal_np))),
            'dynamic_range': float(np.max(signal_np) - np.min(signal_np)),
            'mean_amplitude': float(np.mean(signal_np)),
            'std_amplitude': float(np.std(signal_np)),
            
            # Shape properties
            'skewness': float(stats.skew(signal_np)),
            'kurtosis': float(stats.kurtosis(signal_np)),
            'zero_crossings': int(len(np.where(np.diff(np.signbit(signal_np)))[0])),
            
            # Energy properties
            'total_energy': float(np.sum(signal_np**2)),
            'signal_power': float(np.mean(signal_np**2)),
            
            # Frequency properties
            'dominant_frequency': self._estimate_dominant_frequency(signal_np, sr),
            'bandwidth': self._estimate_bandwidth(signal_np, sr),
        }
        
        return properties
    
    def analyze_abr_waves(self, signal: torch.Tensor, sr: int = 1000) -> Dict[str, Any]:
        """
        Analyze ABR-specific wave components.
        
        Args:
            signal: ABR signal tensor
            sr: Sampling rate
            
        Returns:
            Dictionary of ABR wave analysis results
        """
        signal_np = signal.squeeze().cpu().numpy()
        
        analysis = {
            'detected_waves': {},
            'wave_intervals': {},
            'interpeak_latencies': {},
            'amplitude_ratios': {}
        }
        
        # Detect peaks in typical ABR wave regions
        for wave_name, (start_ms, end_ms) in self.abr_wave_latencies.items():
            start_idx = int(start_ms * sr / 1000)
            end_idx = int(end_ms * sr / 1000)
            
            if end_idx < len(signal_np):
                wave_segment = signal_np[start_idx:end_idx]
                
                # Find peaks in this segment
                min_distance = max(1, int(0.5 * sr / 1000))  # Ensure minimum distance >= 1
                peaks, properties = find_peaks(np.abs(wave_segment), 
                                             height=np.std(signal_np) * 0.5,
                                             distance=min_distance)
                
                if len(peaks) > 0:
                    # Get the most prominent peak
                    peak_heights = np.abs(wave_segment[peaks])
                    main_peak_idx = peaks[np.argmax(peak_heights)]
                    peak_latency_ms = (start_idx + main_peak_idx) / sr * 1000
                    peak_amplitude = wave_segment[main_peak_idx]
                    
                    analysis['detected_waves'][wave_name] = {
                        'latency_ms': float(peak_latency_ms),
                        'amplitude': float(peak_amplitude),
                        'relative_position': float(main_peak_idx / len(wave_segment))
                    }
        
        # Calculate interpeak latencies
        waves = list(analysis['detected_waves'].keys())
        for i in range(len(waves)):
            for j in range(i+1, len(waves)):
                wave1, wave2 = waves[i], waves[j]
                if wave1 in analysis['detected_waves'] and wave2 in analysis['detected_waves']:
                    latency_diff = (analysis['detected_waves'][wave2]['latency_ms'] - 
                                  analysis['detected_waves'][wave1]['latency_ms'])
                    analysis['interpeak_latencies'][f'{wave1}_to_{wave2}'] = float(latency_diff)
        
        # Calculate amplitude ratios
        for i in range(len(waves)):
            for j in range(i+1, len(waves)):
                wave1, wave2 = waves[i], waves[j]
                if wave1 in analysis['detected_waves'] and wave2 in analysis['detected_waves']:
                    amp1 = analysis['detected_waves'][wave1]['amplitude']
                    amp2 = analysis['detected_waves'][wave2]['amplitude']
                    ratio = amp1 / max(abs(amp2), 1e-6)
                    analysis['amplitude_ratios'][f'{wave1}_to_{wave2}'] = float(ratio)
        
        return analysis
    
    def analyze_conditional_response(self, 
                                   generated_signal: torch.Tensor,
                                   conditions: torch.Tensor,
                                   condition_variation: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze how the generated signal responds to conditional inputs.
        
        Args:
            generated_signal: Generated signal tensor
            conditions: Condition parameters tensor
            condition_variation: Dictionary of condition variations applied
            
        Returns:
            Dictionary of conditional response analysis
        """
        signal_props = self.analyze_signal_properties(generated_signal)
        
        response_analysis = {
            'condition_params': conditions.squeeze().cpu().numpy().tolist(),
            'applied_variation': condition_variation,
            'signal_properties': signal_props,
            'response_strength': 0.0,
            'expected_response': {}
        }
        
        # Analyze expected responses based on condition variations
        for param_name, param_value in condition_variation.items():
            if param_name == 'hearing_loss':
                # Higher hearing loss should correlate with reduced amplitude
                expected_amplitude_reduction = param_value * 0.1  # Simple heuristic
                actual_amplitude = signal_props['peak_amplitude']
                response_analysis['expected_response'][param_name] = {
                    'expected_effect': 'amplitude_reduction',
                    'expected_value': expected_amplitude_reduction,
                    'actual_amplitude': actual_amplitude
                }
            elif param_name == 'age':
                # Older age might correlate with increased latency
                expected_latency_increase = (param_value - 30) * 0.01  # Simple heuristic
                response_analysis['expected_response'][param_name] = {
                    'expected_effect': 'latency_increase',
                    'expected_value': expected_latency_increase
                }
        
        return response_analysis
    
    def analyze_generation_consistency(self, generated_samples: List[torch.Tensor]) -> Dict[str, float]:
        """
        Analyze consistency across multiple generations with same conditions.
        
        Args:
            generated_samples: List of generated signal tensors
            
        Returns:
            Dictionary of consistency metrics
        """
        if len(generated_samples) < 2:
            return {'error': 'Need at least 2 samples for consistency analysis'}
        
        # Convert to numpy arrays
        samples_np = [s.squeeze().cpu().numpy() for s in generated_samples]
        sample_matrix = np.array(samples_np)
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(samples_np)):
            for j in range(i+1, len(samples_np)):
                corr, _ = stats.pearsonr(samples_np[i], samples_np[j])
                correlations.append(corr)
        
        # Calculate signal property consistency
        properties_list = [self.analyze_signal_properties(torch.tensor(s)) for s in samples_np]
        
        property_consistency = {}
        for prop_name in properties_list[0].keys():
            values = [props[prop_name] for props in properties_list]
            property_consistency[f'{prop_name}_mean'] = float(np.mean(values))
            property_consistency[f'{prop_name}_std'] = float(np.std(values))
            property_consistency[f'{prop_name}_cv'] = float(np.std(values) / (np.mean(values) + 1e-6))
        
        consistency_analysis = {
            'num_samples': len(generated_samples),
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'min_correlation': float(np.min(correlations)),
            'max_correlation': float(np.max(correlations)),
            'property_consistency': property_consistency,
            'overall_consistency_score': float(np.mean(correlations))
        }
        
        return consistency_analysis
    
    def statistical_comparison(self, 
                             generated_samples: List[torch.Tensor],
                             real_samples: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Perform statistical tests comparing generated and real samples.
        
        Args:
            generated_samples: List of generated signal tensors
            real_samples: List of real signal tensors
            
        Returns:
            Dictionary of statistical test results
        """
        # Convert to numpy
        gen_props = [self.analyze_signal_properties(s) for s in generated_samples]
        real_props = [self.analyze_signal_properties(s) for s in real_samples]
        
        statistical_tests = {}
        
        # Test each property
        for prop_name in gen_props[0].keys():
            gen_values = [props[prop_name] for props in gen_props]
            real_values = [props[prop_name] for props in real_props]
            
            # Two-sample t-test
            t_stat, t_p = stats.ttest_ind(gen_values, real_values)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p = stats.mannwhitneyu(gen_values, real_values, alternative='two-sided')
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(gen_values, real_values)
            
            statistical_tests[prop_name] = {
                't_test': {'statistic': float(t_stat), 'p_value': float(t_p)},
                'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_p)},
                'ks_test': {'statistic': float(ks_stat), 'p_value': float(ks_p)},
                'generated_mean': float(np.mean(gen_values)),
                'real_mean': float(np.mean(real_values)),
                'effect_size': float(abs(np.mean(gen_values) - np.mean(real_values)) / 
                                   np.sqrt((np.var(gen_values) + np.var(real_values)) / 2))
            }
        
        return statistical_tests
    
    def dimensionality_analysis(self, 
                              generated_samples: List[torch.Tensor],
                              real_samples: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze dimensionality and distribution of generated vs real samples.
        
        Args:
            generated_samples: List of generated signal tensors
            real_samples: List of real signal tensors
            
        Returns:
            Dictionary of dimensionality analysis results
        """
        # Combine and flatten samples
        gen_matrix = torch.stack(generated_samples).cpu().numpy()
        real_matrix = torch.stack(real_samples).cpu().numpy()
        
        gen_flat = gen_matrix.reshape(gen_matrix.shape[0], -1)
        real_flat = real_matrix.reshape(real_matrix.shape[0], -1)
        
        # PCA analysis
        combined_data = np.vstack([gen_flat, real_flat])
        pca = PCA(n_components=min(10, combined_data.shape[1]))
        pca_transformed = pca.fit_transform(combined_data)
        
        gen_pca = pca_transformed[:len(gen_flat)]
        real_pca = pca_transformed[len(gen_flat):]
        
        # Clustering analysis
        labels_true = np.concatenate([np.zeros(len(gen_flat)), np.ones(len(real_flat))])
        
        # K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels_pred = kmeans.fit_predict(pca_transformed[:, :5])  # Use first 5 PCs
        
        # Silhouette score
        silhouette_avg = silhouette_score(pca_transformed[:, :5], labels_true)
        silhouette_pred = silhouette_score(pca_transformed[:, :5], labels_pred)
        
        analysis = {
            'pca_explained_variance': pca.explained_variance_ratio_[:5].tolist(),
            'pca_cumulative_variance': np.cumsum(pca.explained_variance_ratio_[:5]).tolist(),
            'silhouette_score_true': float(silhouette_avg),
            'silhouette_score_predicted': float(silhouette_pred),
            'cluster_separation': float(np.mean([
                np.linalg.norm(np.mean(gen_pca, axis=0) - np.mean(real_pca, axis=0)),
                np.mean(np.linalg.norm(gen_pca - np.mean(gen_pca, axis=0), axis=1)),
                np.mean(np.linalg.norm(real_pca - np.mean(real_pca, axis=0), axis=1))
            ])),
        }
        
        return analysis
    
    def _estimate_dominant_frequency(self, signal: np.ndarray, sr: int) -> float:
        """Estimate dominant frequency using FFT."""
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sr)
        magnitude = np.abs(fft_result)
        
        # Find dominant frequency (excluding DC component)
        dominant_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        return float(freqs[dominant_idx])
    
    def _estimate_bandwidth(self, signal: np.ndarray, sr: int) -> float:
        """Estimate signal bandwidth."""
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sr)
        magnitude = np.abs(fft_result[:len(fft_result)//2])
        
        # Find -3dB bandwidth
        max_mag = np.max(magnitude)
        threshold = max_mag / np.sqrt(2)  # -3dB
        
        above_threshold = magnitude > threshold
        if np.any(above_threshold):
            freq_indices = np.where(above_threshold)[0]
            bandwidth = freqs[freq_indices[-1]] - freqs[freq_indices[0]]
            return float(abs(bandwidth))
        else:
            return 0.0