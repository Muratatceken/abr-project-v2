"""
Comprehensive Metrics for ABR Signal Generation Evaluation

This module provides various metrics to evaluate the quality of generated ABR signals,
including time-domain, frequency-domain, and perceptual metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import scipy.signal
from scipy.stats import pearsonr, spearmanr
from scipy.fft import fft, fftfreq
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
        """Signal-to-Noise Ratio in dB."""
        signal_power = torch.mean(target ** 2)
        noise_power = torch.mean((predicted - target) ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * torch.log10(signal_power / noise_power).item()
    
    @staticmethod
    def psnr(predicted: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
        """Peak Signal-to-Noise Ratio in dB."""
        mse = F.mse_loss(predicted, target)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse).item()
    
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
    """Compute all available metrics for signal evaluation."""
    
    results = {}
    
    # Time-domain metrics
    results.update({
        'mse': SignalMetrics.mse(predicted, target),
        'mae': SignalMetrics.mae(predicted, target),
        'rmse': SignalMetrics.rmse(predicted, target),
        'snr': SignalMetrics.snr(predicted, target),
        'psnr': SignalMetrics.psnr(predicted, target),
        'correlation': SignalMetrics.correlation(predicted, target),
        'predicted_dynamic_range': SignalMetrics.dynamic_range(predicted),
        'target_dynamic_range': SignalMetrics.dynamic_range(target),
        'predicted_rms': SignalMetrics.rms(predicted),
        'target_rms': SignalMetrics.rms(target)
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