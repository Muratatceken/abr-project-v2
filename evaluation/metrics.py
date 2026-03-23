"""
Comprehensive Metrics for ABR Signal Generation Evaluation

This module provides various metrics to evaluate the quality of generated ABR signals,
including time-domain, frequency-domain, and perceptual metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.signal
from scipy.stats import pearsonr, spearmanr
from scipy.fft import fft, fftfreq
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, precision_recall_curve, auc
from scipy.interpolate import interp1d
import json
import logging

# Import robust SNR functions for consistency
from utils.metrics import compute_robust_snr, validate_snr_inputs

# Optional imports with fallbacks
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available. Some spectral metrics will be disabled.")


class SignalMetrics:
    """Time-domain signal quality metrics."""
    
    @staticmethod
    def mse(predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Squared Error."""
        return F.mse_loss(predicted, target).item()
    
    @staticmethod
    def mae(predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Absolute Error."""
        return F.l1_loss(predicted, target).item()
    
    @staticmethod
    def rmse(predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Root Mean Squared Error."""
        return torch.sqrt(F.mse_loss(predicted, target)).item()
    
    @staticmethod
    def snr(predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Signal-to-Noise Ratio in dB with robust calculation."""
        # Use the robust SNR implementation from utils.metrics for consistency
        try:
            return compute_robust_snr(predicted, target, log_edge_cases=False).item()
        except Exception as e:
            # Fallback to original implementation with enhanced epsilon
            signal_power = torch.mean(target ** 2)
            noise_power = torch.mean((predicted - target) ** 2)
            
            # Enhanced epsilon protection
            eps = 1e-6
            if noise_power < eps:
                if signal_power < eps:
                    return 0.0  # Both signals are essentially zero
                else:
                    return 60.0  # High SNR, bounded
            
            snr_value = 10 * torch.log10(signal_power / noise_power).item()
            # Apply bounds checking
            return max(-60.0, min(60.0, snr_value))
    
    @staticmethod
    def psnr(predicted: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
        """Peak Signal-to-Noise Ratio in dB with robust handling."""
        mse = F.mse_loss(predicted, target)
        
        # Enhanced epsilon protection
        eps = 1e-6
        if mse < eps:
            return 60.0  # High PSNR, bounded
        
        psnr_value = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse).item()
        # Apply bounds checking
        return max(-60.0, min(60.0, psnr_value))
    
    @staticmethod
    def correlation(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute correlation metrics."""
        pred_np = predicted.detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()
        
        pearson_r, pearson_p = pearsonr(pred_np, target_np)
        spearman_r, spearman_p = spearmanr(pred_np, target_np)
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }
    
    @staticmethod
    def dynamic_range(signal: torch.Tensor) -> float:
        """Compute dynamic range of signal."""
        return (torch.max(signal) - torch.min(signal)).item()
    
    @staticmethod
    def rms(signal: torch.Tensor) -> float:
        """Root Mean Square amplitude."""
        return torch.sqrt(torch.mean(signal ** 2)).item()


class SpectralMetrics:
    """Frequency-domain signal quality metrics."""
    
    @staticmethod
    def spectral_centroid(signal: torch.Tensor, sr: int = 1000) -> float:
        """Compute spectral centroid."""
        if not LIBROSA_AVAILABLE:
            return SpectralMetrics._spectral_centroid_fallback(signal, sr)
        signal_np = signal.detach().cpu().numpy().squeeze()
        centroid = librosa.feature.spectral_centroid(y=signal_np, sr=sr)
        return float(np.mean(centroid))
    
    @staticmethod
    def spectral_bandwidth(signal: torch.Tensor, sr: int = 1000) -> float:
        """Compute spectral bandwidth."""
        if not LIBROSA_AVAILABLE:
            return SpectralMetrics._spectral_bandwidth_fallback(signal, sr)
        signal_np = signal.detach().cpu().numpy().squeeze()
        bandwidth = librosa.feature.spectral_bandwidth(y=signal_np, sr=sr)
        return float(np.mean(bandwidth))
    
    @staticmethod
    def spectral_rolloff(signal: torch.Tensor, sr: int = 1000, roll_percent: float = 0.85) -> float:
        """Compute spectral rolloff point."""
        if not LIBROSA_AVAILABLE:
            return SpectralMetrics._spectral_rolloff_fallback(signal, sr, roll_percent)
        signal_np = signal.detach().cpu().numpy().squeeze()
        rolloff = librosa.feature.spectral_rolloff(y=signal_np, sr=sr, roll_percent=roll_percent)
        return float(np.mean(rolloff))
    
    @staticmethod
    def _spectral_centroid_fallback(signal: torch.Tensor, sr: int = 1000) -> float:
        """Fallback implementation of spectral centroid."""
        signal_np = signal.detach().cpu().numpy().squeeze()
        fft_result = np.abs(fft(signal_np))
        freqs = np.fft.fftfreq(len(signal_np), 1/sr)
        freqs = freqs[:len(freqs)//2]
        fft_result = fft_result[:len(fft_result)//2]
        centroid = np.sum(freqs * fft_result) / (np.sum(fft_result) + 1e-6)
        return float(centroid)
    
    @staticmethod
    def _spectral_bandwidth_fallback(signal: torch.Tensor, sr: int = 1000) -> float:
        """Fallback implementation of spectral bandwidth."""
        signal_np = signal.detach().cpu().numpy().squeeze()
        fft_result = np.abs(fft(signal_np))
        freqs = np.fft.fftfreq(len(signal_np), 1/sr)
        freqs = freqs[:len(freqs)//2]
        fft_result = fft_result[:len(fft_result)//2]
        centroid = np.sum(freqs * fft_result) / (np.sum(fft_result) + 1e-6)
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_result) / (np.sum(fft_result) + 1e-6))
        return float(bandwidth)
    
    @staticmethod
    def _spectral_rolloff_fallback(signal: torch.Tensor, sr: int = 1000, roll_percent: float = 0.85) -> float:
        """Fallback implementation of spectral rolloff."""
        signal_np = signal.detach().cpu().numpy().squeeze()
        fft_result = np.abs(fft(signal_np))
        freqs = np.fft.fftfreq(len(signal_np), 1/sr)
        freqs = freqs[:len(freqs)//2]
        fft_result = fft_result[:len(fft_result)//2]
        cumsum = np.cumsum(fft_result)
        rolloff_idx = np.where(cumsum >= roll_percent * cumsum[-1])[0]
        if len(rolloff_idx) > 0:
            return float(freqs[rolloff_idx[0]])
        return 0.0
    
    @staticmethod
    def frequency_response_error(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compare frequency responses between predicted and target signals."""
        pred_np = predicted.detach().cpu().numpy().squeeze()
        target_np = target.detach().cpu().numpy().squeeze()
        
        # Compute FFT
        pred_fft = np.abs(fft(pred_np))
        target_fft = np.abs(fft(target_np))
        
        # Frequency response metrics
        freq_mse = mean_squared_error(target_fft, pred_fft)
        freq_mae = mean_absolute_error(target_fft, pred_fft)
        
        # Magnitude spectrum correlation
        freq_corr, _ = pearsonr(target_fft, pred_fft)
        
        return {
            'frequency_mse': freq_mse,
            'frequency_mae': freq_mae,
            'frequency_correlation': freq_corr
        }
    
    @staticmethod
    def power_spectral_density_comparison(predicted: torch.Tensor, target: torch.Tensor, sr: int = 1000) -> Dict[str, float]:
        """Compare power spectral densities."""
        pred_np = predicted.detach().cpu().numpy().squeeze()
        target_np = target.detach().cpu().numpy().squeeze()
        
        # Compute PSDs
        freqs_pred, psd_pred = scipy.signal.welch(pred_np, fs=sr, nperseg=min(len(pred_np), 256))
        freqs_target, psd_target = scipy.signal.welch(target_np, fs=sr, nperseg=min(len(target_np), 256))
        
        # PSD comparison metrics
        psd_mse = mean_squared_error(psd_target, psd_pred)
        psd_mae = mean_absolute_error(psd_target, psd_pred)
        psd_corr, _ = pearsonr(psd_target, psd_pred)
        
        return {
            'psd_mse': psd_mse,
            'psd_mae': psd_mae,
            'psd_correlation': psd_corr,
            'frequencies': freqs_target,
            'target_psd': psd_target,
            'predicted_psd': psd_pred
        }


class PerceptualMetrics:
    """Perceptual and physiological signal quality metrics specific to ABR."""
    
    @staticmethod
    def morphological_similarity(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Assess morphological similarity of ABR waveforms."""
        # Peak detection and analysis
        pred_peaks = PerceptualMetrics._detect_peaks(predicted)
        target_peaks = PerceptualMetrics._detect_peaks(target)
        
        # Waveform shape correlation
        shape_corr, _ = pearsonr(
            predicted.detach().cpu().numpy().flatten(),
            target.detach().cpu().numpy().flatten()
        )
        
        # Peak alignment quality
        peak_alignment = PerceptualMetrics._peak_alignment_score(pred_peaks, target_peaks)
        
        return {
            'shape_correlation': shape_corr,
            'peak_alignment': peak_alignment,
            'predicted_peaks': pred_peaks,
            'target_peaks': target_peaks
        }
    
    @staticmethod
    def _detect_peaks(signal: torch.Tensor, height: float = 0.1, distance: int = 10) -> List[int]:
        """Detect peaks in ABR signal."""
        signal_np = signal.detach().cpu().numpy().squeeze()
        peaks, _ = scipy.signal.find_peaks(np.abs(signal_np), height=height, distance=distance)
        return peaks.tolist()
    
    @staticmethod
    def _peak_alignment_score(pred_peaks: List[int], target_peaks: List[int], tolerance: int = 5) -> float:
        """Score how well predicted peaks align with target peaks."""
        if not target_peaks:
            return 1.0 if not pred_peaks else 0.0
        
        aligned_peaks = 0
        for target_peak in target_peaks:
            for pred_peak in pred_peaks:
                if abs(pred_peak - target_peak) <= tolerance:
                    aligned_peaks += 1
                    break
        
        return aligned_peaks / len(target_peaks)
    
    @staticmethod
    def amplitude_envelope_similarity(predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Compare amplitude envelopes using Hilbert transform."""
        pred_np = predicted.detach().cpu().numpy().squeeze()
        target_np = target.detach().cpu().numpy().squeeze()
        
        # Extract amplitude envelopes
        pred_envelope = np.abs(scipy.signal.hilbert(pred_np))
        target_envelope = np.abs(scipy.signal.hilbert(target_np))
        
        # Correlation between envelopes
        envelope_corr, _ = pearsonr(target_envelope, pred_envelope)
        return envelope_corr
    
    @staticmethod
    def phase_coherence(predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Compute phase coherence between signals."""
        pred_np = predicted.detach().cpu().numpy().squeeze()
        target_np = target.detach().cpu().numpy().squeeze()
        
        # Compute analytic signals
        pred_analytic = scipy.signal.hilbert(pred_np)
        target_analytic = scipy.signal.hilbert(target_np)
        
        # Phase difference
        pred_phase = np.angle(pred_analytic)
        target_phase = np.angle(target_analytic)
        phase_diff = np.abs(pred_phase - target_phase)
        
        # Phase coherence (1 - normalized phase difference)
        coherence = 1 - np.mean(phase_diff) / np.pi
        return float(coherence)


class ABRSpecificMetrics:
    """ABR-specific evaluation metrics."""
    
    @staticmethod
    def wave_component_analysis(signal: torch.Tensor, sr: int = 1000) -> Dict[str, float]:
        """Analyze ABR wave components (I, III, V waves)."""
        signal_np = signal.detach().cpu().numpy().squeeze()
        
        # Typical ABR wave latencies (in samples, assuming 1kHz sampling)
        # Wave I: ~1.5ms, Wave III: ~3.8ms, Wave V: ~5.6ms
        wave_windows = {
            'wave_I': (int(1.0 * sr / 1000), int(2.0 * sr / 1000)),  # 1-2ms
            'wave_III': (int(3.3 * sr / 1000), int(4.3 * sr / 1000)),  # 3.3-4.3ms
            'wave_V': (int(5.1 * sr / 1000), int(6.1 * sr / 1000))   # 5.1-6.1ms
        }
        
        wave_amplitudes = {}
        for wave_name, (start, end) in wave_windows.items():
            if end < len(signal_np):
                wave_segment = signal_np[start:end]
                wave_amplitudes[f'{wave_name}_amplitude'] = float(np.max(np.abs(wave_segment)))
            else:
                wave_amplitudes[f'{wave_name}_amplitude'] = 0.0
        
        return wave_amplitudes
    
    @staticmethod
    def threshold_estimation_accuracy(predicted: torch.Tensor, target: torch.Tensor, 
                                    true_threshold: float, predicted_threshold: float) -> Dict[str, float]:
        """Evaluate threshold estimation accuracy."""
        threshold_error = abs(predicted_threshold - true_threshold)
        threshold_relative_error = threshold_error / max(true_threshold, 1e-6)
        
        return {
            'threshold_absolute_error': threshold_error,
            'threshold_relative_error': threshold_relative_error,
            'threshold_accuracy': 1.0 / (1.0 + threshold_relative_error)
        }
    
    @staticmethod
    def signal_to_noise_improvement(predicted: torch.Tensor, target: torch.Tensor, 
                                  noise_baseline: torch.Tensor) -> float:
        """Measure SNR improvement over baseline noise."""
        target_snr = SignalMetrics.snr(target, noise_baseline)
        predicted_snr = SignalMetrics.snr(predicted, noise_baseline)
        
        return predicted_snr - target_snr


def compute_all_metrics(predicted: torch.Tensor, target: torch.Tensor, 
                       sr: int = 1000, **kwargs) -> Dict[str, Union[float, Dict]]:
    """Compute all available metrics for signal evaluation with robust error handling."""
    
    results = {}
    
    # Time-domain metrics with enhanced error handling
    try:
        results.update({
            'mse': SignalMetrics.mse(predicted, target),
            'mae': SignalMetrics.mae(predicted, target),
            'rmse': SignalMetrics.rmse(predicted, target),
        })
    except Exception as e:
        print(f"Warning: Basic metrics calculation failed: {e}")
        results.update({
            'mse': float('nan'),
            'mae': float('nan'),
            'rmse': float('nan')
        })
    
    # SNR and PSNR with robust handling
    try:
        results['snr'] = SignalMetrics.snr(predicted, target)
        results['psnr'] = SignalMetrics.psnr(predicted, target)
    except Exception as e:
        print(f"Warning: SNR/PSNR calculation failed: {e}")
        results['snr'] = float('nan')
        results['psnr'] = float('nan')
    
    # Other metrics
    try:
        results.update({
            'correlation': SignalMetrics.correlation(predicted, target),
            'predicted_dynamic_range': SignalMetrics.dynamic_range(predicted),
            'target_dynamic_range': SignalMetrics.dynamic_range(target),
            'predicted_rms': SignalMetrics.rms(predicted),
            'target_rms': SignalMetrics.rms(target)
        })
    except Exception as e:
        print(f"Warning: Additional metrics calculation failed: {e}")
        results.update({
            'correlation': {'pearson_r': float('nan'), 'pearson_p': float('nan'),
                          'spearman_r': float('nan'), 'spearman_p': float('nan')},
            'predicted_dynamic_range': float('nan'),
            'target_dynamic_range': float('nan'),
            'predicted_rms': float('nan'),
            'target_rms': float('nan')
        })
    
    # Frequency-domain metrics  
    results.update({
        'predicted_spectral_centroid': SpectralMetrics.spectral_centroid(predicted, sr),
        'target_spectral_centroid': SpectralMetrics.spectral_centroid(target, sr),
        'predicted_spectral_bandwidth': SpectralMetrics.spectral_bandwidth(predicted, sr),
        'target_spectral_bandwidth': SpectralMetrics.spectral_bandwidth(target, sr),
        'frequency_response': SpectralMetrics.frequency_response_error(predicted, target),
        'psd_comparison': SpectralMetrics.power_spectral_density_comparison(predicted, target, sr)
    })
    
    # Perceptual metrics
    results.update({
        'morphological_similarity': PerceptualMetrics.morphological_similarity(predicted, target),
        'amplitude_envelope_similarity': PerceptualMetrics.amplitude_envelope_similarity(predicted, target),
        'phase_coherence': PerceptualMetrics.phase_coherence(predicted, target)
    })
    
    # ABR-specific metrics
    results.update({
        'predicted_wave_analysis': ABRSpecificMetrics.wave_component_analysis(predicted, sr),
        'target_wave_analysis': ABRSpecificMetrics.wave_component_analysis(target, sr)
    })
    
    return results


class ThresholdOptimizer:
    """
    Comprehensive threshold optimization for ABR peak classification.
    
    Supports multiple optimization strategies including F1-optimal, Youden's J statistic,
    and constrained optimization for precision/recall targets.
    """
    
    def __init__(self, roc_data: Optional[Dict[str, Any]] = None):
        """
        Initialize threshold optimizer.
        
        Args:
            roc_data: Optional pre-loaded ROC data dictionary
        """
        self.roc_data = roc_data
        self.logger = logging.getLogger(__name__)
    
    def analyze_roc_data(self, roc_file_path: str) -> Dict[str, Any]:
        """
        Load and parse ROC classification data from JSON files.
        
        Args:
            roc_file_path: Path to ROC classification JSON file
            
        Returns:
            Dictionary containing parsed ROC data
        """
        try:
            with open(roc_file_path, 'r') as f:
                data = json.load(f)
            
            # Extract ROC curve data
            if 'roc_analysis' in data:
                roc_curve_data = data['roc_analysis']['roc_curve']
                fpr = np.array(roc_curve_data['fpr'])
                tpr = np.array(roc_curve_data['tpr'])
                thresholds = np.array(roc_curve_data['thresholds'])
                
                # Check for shape mismatches in ROC data
                self.logger.debug(f"ROC array shapes - fpr: {fpr.shape}, tpr: {tpr.shape}, thresholds: {thresholds.shape}")
                
                # Validate and fix array shapes if needed
                if not (len(fpr) == len(tpr) == len(thresholds)):
                    self.logger.warning(f"ROC array length mismatch: fpr={len(fpr)}, tpr={len(tpr)}, thresholds={len(thresholds)}")
                    # Truncate to shortest length to ensure compatibility
                    min_length = min(len(fpr), len(tpr), len(thresholds))
                    self.logger.warning(f"Truncating ROC arrays to length {min_length}")
                    fpr = fpr[:min_length]
                    tpr = tpr[:min_length]
                    thresholds = thresholds[:min_length]
            else:
                raise KeyError("No 'roc_analysis' found in ROC data")
            
            # Extract precision-recall data
            if 'precision_recall_analysis' in data:
                pr_curve_data = data['precision_recall_analysis']['pr_curve']
                precision = np.array(pr_curve_data['precision'])
                recall = np.array(pr_curve_data['recall'])
                pr_thresholds = np.array(pr_curve_data['thresholds'])
                
                # Check for shape mismatches before processing
                self.logger.debug(f"PR array shapes - precision: {precision.shape}, recall: {recall.shape}, thresholds: {pr_thresholds.shape}")
                
                # Harden precision-recall data: remove NaNs and ensure monotonicity
                if precision is not None and pr_thresholds is not None:
                    # Validate array shapes first
                    if not (len(precision) == len(recall) == len(pr_thresholds)):
                        self.logger.warning(f"PR array length mismatch: precision={len(precision)}, recall={len(recall)}, thresholds={len(pr_thresholds)}")
                        # Truncate to shortest length to ensure compatibility
                        min_length = min(len(precision), len(recall), len(pr_thresholds))
                        self.logger.warning(f"Truncating PR arrays to length {min_length}")
                        precision = precision[:min_length]
                        recall = recall[:min_length]
                        pr_thresholds = pr_thresholds[:min_length]
                    
                    # Remove NaN values
                    valid_mask = ~(np.isnan(precision) | np.isnan(recall) | np.isnan(pr_thresholds))
                    if np.any(~valid_mask):
                        self.logger.warning(f"Removed {np.sum(~valid_mask)} NaN values from precision-recall data")
                        precision = precision[valid_mask]
                        recall = recall[valid_mask]
                        pr_thresholds = pr_thresholds[valid_mask]
                    
                    # Sort by thresholds to ensure monotonicity for interpolation
                    if len(pr_thresholds) > 1:
                        sort_indices = np.argsort(pr_thresholds)
                        precision = precision[sort_indices]
                        recall = recall[sort_indices]
                        pr_thresholds = pr_thresholds[sort_indices]
            else:
                self.logger.warning("No precision-recall data found")
                precision = None
                recall = None
                pr_thresholds = None
            
            # Validate data integrity
            if len(fpr) != len(tpr) or len(fpr) != len(thresholds):
                raise ValueError("ROC curve arrays have mismatched lengths")
            
            if precision is not None and len(precision) != len(recall):
                raise ValueError("Precision-recall arrays have mismatched lengths")
            
            parsed_data = {
                'fpr': fpr,
                'tpr': tpr, 
                'thresholds': thresholds,
                'precision': precision,
                'recall': recall,
                'pr_thresholds': pr_thresholds,
                'auroc': data.get('roc_analysis', {}).get('auroc', None),
                'average_precision': data.get('precision_recall_analysis', {}).get('average_precision', None)
            }
            
            self.roc_data = parsed_data
            self.logger.info(f"✓ Loaded ROC data with {len(fpr)} threshold points")
            return parsed_data
            
        except Exception as e:
            self.logger.error(f"Failed to load ROC data from {roc_file_path}: {e}")
            raise
    
    def find_optimal_threshold_constrained(self, 
                                         min_recall: float = None,
                                         min_precision: float = None,
                                         min_specificity: float = None) -> Dict[str, Any]:
        """
        Find optimal threshold satisfying user-defined constraints.
        
        Args:
            min_recall: Minimum required recall (sensitivity)
            min_precision: Minimum required precision (PPV)
            min_specificity: Minimum required specificity
            
        Returns:
            Dictionary with optimal threshold and performance metrics
        """
        if self.roc_data is None:
            raise ValueError("No ROC data loaded. Call analyze_roc_data() first.")
        
        fpr = self.roc_data['fpr']
        tpr = self.roc_data['tpr']
        thresholds = self.roc_data['thresholds']
        precision = self.roc_data['precision']
        recall = self.roc_data['recall']
        
        # Calculate specificity
        specificity = 1 - fpr
        
        # Find feasible thresholds
        feasible_mask = np.ones(len(thresholds), dtype=bool)
        
        # Apply recall constraint
        if min_recall is not None:
            feasible_mask &= (tpr >= min_recall)
        
        # Apply specificity constraint
        if min_specificity is not None:
            feasible_mask &= (specificity >= min_specificity)
        
        # Apply precision constraint (if precision-recall data available)
        if min_precision is not None:
            if precision is None or self.roc_data['pr_thresholds'] is None:
                # Cannot apply precision constraint without PR data
                return {
                    'success': False,
                    'message': 'Precision constraint cannot be applied: precision-recall data unavailable',
                    'constraints': {
                        'min_recall': min_recall,
                        'min_precision': min_precision,
                        'min_specificity': min_specificity
                    },
                    'recommendations': [
                        'Ensure precision-recall analysis was performed during ROC data generation',
                        'Check that precision-recall curve data is included in input file',
                        'Consider using only recall and specificity constraints',
                        'Rerun evaluation with precision-recall analysis enabled'
                    ]
                }
            
            # Interpolate precision at ROC thresholds
            pr_thresh = self.roc_data['pr_thresholds']
            # Handle edge cases in interpolation
            if len(pr_thresh) > 1:
                try:
                    # Additional robustness checks before interpolation
                    # Ensure arrays are clean and monotonic
                    pr_thresh_clean = np.array(pr_thresh)
                    precision_clean = np.array(precision)
                    
                    # Remove any remaining NaN values
                    valid_mask = ~(np.isnan(pr_thresh_clean) | np.isnan(precision_clean))
                    if not np.all(valid_mask):
                        pr_thresh_clean = pr_thresh_clean[valid_mask]
                        precision_clean = precision_clean[valid_mask]
                        self.logger.debug(f"Removed {np.sum(~valid_mask)} additional NaN values before interpolation")
                    
                    # Ensure sufficient points for interpolation
                    if len(pr_thresh_clean) < 2:
                        raise ValueError(f"Insufficient valid precision-recall points for interpolation: {len(pr_thresh_clean)}")
                    
                    # Sort by thresholds if not already sorted (additional safety)
                    if not np.all(pr_thresh_clean[:-1] <= pr_thresh_clean[1:]):
                        sort_indices = np.argsort(pr_thresh_clean)
                        pr_thresh_clean = pr_thresh_clean[sort_indices]
                        precision_clean = precision_clean[sort_indices]
                        self.logger.debug("Re-sorted precision-recall data for interpolation")
                    
                    prec_interp = interp1d(pr_thresh_clean, precision_clean, bounds_error=False, 
                                         fill_value='extrapolate')
                    interpolated_precision = prec_interp(thresholds)
                    
                    # Handle potential interpolation artifacts
                    interpolated_precision = np.clip(interpolated_precision, 0.0, 1.0)
                    feasible_mask &= (interpolated_precision >= min_precision)
                    
                except Exception as e:
                    self.logger.warning(f"Precision interpolation failed: {e}")
                    return {
                        'success': False,
                        'message': f'Precision constraint application failed due to interpolation error: {str(e)}',
                        'constraints': {
                            'min_recall': min_recall,
                            'min_precision': min_precision,
                            'min_specificity': min_specificity
                        },
                        'recommendations': [
                            'Check precision-recall data quality and completeness',
                            'Verify threshold arrays are monotonic and contain no NaN values',
                            'Consider using only recall and specificity constraints',
                            'Try relaxing precision constraint slightly'
                        ]
                    }
        
        feasible_indices = np.where(feasible_mask)[0]
        
        if len(feasible_indices) == 0:
            return {
                'success': False,
                'message': 'No thresholds satisfy the specified constraints',
                'constraints': {
                    'min_recall': min_recall,
                    'min_precision': min_precision,
                    'min_specificity': min_specificity
                },
                'recommendations': self._suggest_relaxed_constraints(min_recall, min_precision, min_specificity)
            }
        
        # Among feasible thresholds, select the one maximizing F1-score
        feasible_tpr = tpr[feasible_indices]
        feasible_fpr = fpr[feasible_indices]
        feasible_thresholds = thresholds[feasible_indices]
        
        # Calculate F1 scores for feasible points
        feasible_precision_approx = feasible_tpr / (feasible_tpr + feasible_fpr + 1e-10)
        feasible_f1 = 2 * (feasible_precision_approx * feasible_tpr) / (feasible_precision_approx + feasible_tpr + 1e-10)
        
        # Select best F1 score
        best_idx = feasible_indices[np.argmax(feasible_f1)]
        optimal_threshold = thresholds[best_idx]
        
        return {
            'success': True,
            'optimal_threshold': optimal_threshold,
            'performance': {
                'sensitivity': tpr[best_idx],
                'recall': tpr[best_idx],  # Mirror sensitivity for consistency
                'specificity': specificity[best_idx],
                'precision': feasible_precision_approx[np.argmax(feasible_f1)],
                'f1_score': feasible_f1[np.argmax(feasible_f1)],
                'fpr': fpr[best_idx],
                'tpr': tpr[best_idx]
            },
            'constraints_satisfied': {
                'min_recall': min_recall is None or tpr[best_idx] >= min_recall,
                'min_precision': min_precision is None or feasible_precision_approx[np.argmax(feasible_f1)] >= min_precision,
                'min_specificity': min_specificity is None or specificity[best_idx] >= min_specificity
            },
            'num_feasible_thresholds': len(feasible_indices)
        }
    
    def find_optimal_threshold_f1(self) -> Dict[str, Any]:
        """
        Find F1-optimal threshold with detailed performance metrics.
        
        Returns:
            Dictionary with F1-optimal threshold and performance
        """
        if self.roc_data is None:
            raise ValueError("No ROC data loaded. Call analyze_roc_data() first.")
        
        fpr = self.roc_data['fpr']
        tpr = self.roc_data['tpr']
        thresholds = self.roc_data['thresholds']
        
        # Calculate F1 scores
        precision_approx = tpr / (tpr + fpr + 1e-10)
        f1_scores = 2 * (precision_approx * tpr) / (precision_approx + tpr + 1e-10)
        
        # Find optimal F1
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'optimal_threshold': optimal_threshold,
            'performance': {
                'f1_score': f1_scores[optimal_idx],
                'sensitivity': tpr[optimal_idx],
                'recall': tpr[optimal_idx],  # Mirror sensitivity for consistency
                'specificity': 1 - fpr[optimal_idx],
                'precision': precision_approx[optimal_idx],
                'fpr': fpr[optimal_idx],
                'tpr': tpr[optimal_idx]
            },
            'optimization_method': 'f1_optimal'
        }
    
    def find_optimal_threshold_youden(self) -> Dict[str, Any]:
        """
        Find Youden's J statistic optimal threshold.
        
        Returns:
            Dictionary with Youden-optimal threshold and performance
        """
        if self.roc_data is None:
            raise ValueError("No ROC data loaded. Call analyze_roc_data() first.")
        
        fpr = self.roc_data['fpr']
        tpr = self.roc_data['tpr']
        thresholds = self.roc_data['thresholds']
        
        # Calculate Youden's J statistic
        youden_j = tpr - fpr
        
        # Find optimal Youden's J
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'optimal_threshold': optimal_threshold,
            'performance': {
                'youden_j': youden_j[optimal_idx],
                'sensitivity': tpr[optimal_idx],
                'recall': tpr[optimal_idx],  # Mirror sensitivity for consistency
                'specificity': 1 - fpr[optimal_idx],
                'fpr': fpr[optimal_idx],
                'tpr': tpr[optimal_idx],
                'precision': tpr[optimal_idx] / (tpr[optimal_idx] + fpr[optimal_idx] + 1e-10)
            },
            'optimization_method': 'youden_j'
        }
    
    def threshold_performance_analysis(self, threshold_range: Tuple[float, float] = None, 
                                     num_points: int = 100) -> Dict[str, Any]:
        """
        Comprehensive analysis of threshold performance across operating points.
        
        Args:
            threshold_range: Range of thresholds to analyze
            num_points: Number of threshold points to evaluate
            
        Returns:
            Dictionary with performance analysis across thresholds
        """
        if self.roc_data is None:
            raise ValueError("No ROC data loaded. Call analyze_roc_data() first.")
        
        fpr = self.roc_data['fpr']
        tpr = self.roc_data['tpr']
        thresholds = self.roc_data['thresholds']
        
        if threshold_range is None:
            threshold_range = (thresholds.min(), thresholds.max())
        
        # Create analysis grid
        analysis_thresholds = np.linspace(threshold_range[0], threshold_range[1], num_points)
        
        # Interpolate performance metrics
        try:
            tpr_interp = interp1d(thresholds, tpr, bounds_error=False, fill_value='extrapolate')
            fpr_interp = interp1d(thresholds, fpr, bounds_error=False, fill_value='extrapolate')
            
            analysis_tpr = tpr_interp(analysis_thresholds)
            analysis_fpr = fpr_interp(analysis_thresholds)
            analysis_specificity = 1 - analysis_fpr
            analysis_precision = analysis_tpr / (analysis_tpr + analysis_fpr + 1e-10)
            analysis_f1 = 2 * (analysis_precision * analysis_tpr) / (analysis_precision + analysis_tpr + 1e-10)
            analysis_youden = analysis_tpr - analysis_fpr
            
            return {
                'thresholds': analysis_thresholds.tolist(),
                'sensitivity': analysis_tpr.tolist(),
                'recall': analysis_tpr.tolist(),  # Mirror sensitivity for consistency
                'specificity': analysis_specificity.tolist(),
                'precision': analysis_precision.tolist(),
                'f1_score': analysis_f1.tolist(),
                'youden_j': analysis_youden.tolist(),
                'fpr': analysis_fpr.tolist(),
                'tpr': analysis_tpr.tolist()
            }
        except Exception as e:
            self.logger.error(f"Threshold performance analysis failed: {e}")
            return {'error': str(e)}
    
    def validate_threshold_constraints(self, constraints: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate that proposed constraints are achievable.
        
        Args:
            constraints: Dictionary of constraint values
            
        Returns:
            Dictionary with validation results and recommendations
        """
        if self.roc_data is None:
            raise ValueError("No ROC data loaded. Call analyze_roc_data() first.")
        
        fpr = self.roc_data['fpr']
        tpr = self.roc_data['tpr']
        specificity = 1 - fpr
        
        validation_results = {
            'constraints_feasible': True,
            'constraint_analysis': {},
            'recommendations': []
        }
        
        # Check each constraint
        if 'min_recall' in constraints:
            max_achievable_recall = tpr.max()
            is_feasible = constraints['min_recall'] <= max_achievable_recall
            validation_results['constraint_analysis']['recall'] = {
                'requested': constraints['min_recall'],
                'max_achievable': max_achievable_recall,
                'feasible': is_feasible
            }
            if not is_feasible:
                validation_results['constraints_feasible'] = False
                validation_results['recommendations'].append(
                    f"Reduce recall constraint to ≤{max_achievable_recall:.3f}"
                )
        
        if 'min_specificity' in constraints:
            max_achievable_specificity = specificity.max()
            is_feasible = constraints['min_specificity'] <= max_achievable_specificity
            validation_results['constraint_analysis']['specificity'] = {
                'requested': constraints['min_specificity'],
                'max_achievable': max_achievable_specificity,
                'feasible': is_feasible
            }
            if not is_feasible:
                validation_results['constraints_feasible'] = False
                validation_results['recommendations'].append(
                    f"Reduce specificity constraint to ≤{max_achievable_specificity:.3f}"
                )
        
        return validation_results
    
    def _suggest_relaxed_constraints(self, min_recall: float = None, 
                                   min_precision: float = None,
                                   min_specificity: float = None) -> List[str]:
        """
        Suggest relaxed constraints when original constraints cannot be satisfied.
        
        Args:
            min_recall: Original minimum recall constraint
            min_precision: Original minimum precision constraint  
            min_specificity: Original minimum specificity constraint
            
        Returns:
            List of recommendation strings
        """
        if self.roc_data is None:
            return ["Load ROC data first"]
        
        recommendations = []
        fpr = self.roc_data['fpr']
        tpr = self.roc_data['tpr']
        specificity = 1 - fpr
        
        if min_recall is not None:
            achievable_recall_80pct = np.percentile(tpr, 80)
            recommendations.append(f"Consider recall ≥{achievable_recall_80pct:.3f} (80th percentile)")
        
        if min_specificity is not None:
            achievable_spec_80pct = np.percentile(specificity, 80)
            recommendations.append(f"Consider specificity ≥{achievable_spec_80pct:.3f} (80th percentile)")
        
        if min_precision is not None:
            recommendations.append("Consider relaxing precision constraint by 5-10%")
        
        return recommendations


def analyze_roc_data(roc_file_path: str) -> Dict[str, Any]:
    """
    Load and analyze ROC classification data from JSON files.
    
    Args:
        roc_file_path: Path to ROC classification JSON file
        
    Returns:
        Dictionary containing analyzed ROC data
    """
    optimizer = ThresholdOptimizer()
    return optimizer.analyze_roc_data(roc_file_path)


def find_optimal_threshold_constrained(roc_data: Dict[str, Any],
                                     min_recall: float = None,
                                     min_precision: float = None,
                                     min_specificity: float = None) -> Dict[str, Any]:
    """
    Find optimal threshold with user-defined constraints.
    
    Args:
        roc_data: ROC curve data dictionary
        min_recall: Minimum required recall
        min_precision: Minimum required precision
        min_specificity: Minimum required specificity
        
    Returns:
        Dictionary with optimization results
    """
    optimizer = ThresholdOptimizer(roc_data)
    return optimizer.find_optimal_threshold_constrained(min_recall, min_precision, min_specificity)


def find_optimal_threshold_f1(roc_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find F1-optimal threshold.
    
    Args:
        roc_data: ROC curve data dictionary
        
    Returns:
        Dictionary with F1-optimal threshold and performance
    """
    optimizer = ThresholdOptimizer(roc_data)
    return optimizer.find_optimal_threshold_f1()


def find_optimal_threshold_youden(roc_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find Youden's J statistic optimal threshold.
    
    Args:
        roc_data: ROC curve data dictionary
        
    Returns:
        Dictionary with Youden-optimal threshold and performance
    """
    optimizer = ThresholdOptimizer(roc_data)
    return optimizer.find_optimal_threshold_youden()


def threshold_performance_analysis(roc_data: Dict[str, Any], 
                                 threshold_range: Tuple[float, float] = None,
                                 num_points: int = 100) -> Dict[str, Any]:
    """
    Analyze threshold performance across operating points.
    
    Args:
        roc_data: ROC curve data dictionary
        threshold_range: Range of thresholds to analyze
        num_points: Number of analysis points
        
    Returns:
        Dictionary with performance analysis
    """
    optimizer = ThresholdOptimizer(roc_data)
    return optimizer.threshold_performance_analysis(threshold_range, num_points)


def validate_threshold_constraints(roc_data: Dict[str, Any], 
                                 constraints: Dict[str, float]) -> Dict[str, Any]:
    """
    Validate that threshold constraints are achievable.
    
    Args:
        roc_data: ROC curve data dictionary
        constraints: Dictionary of constraint values
        
    Returns:
        Dictionary with validation results
    """
    return ThresholdOptimizer(roc_data).validate_threshold_constraints(constraints)