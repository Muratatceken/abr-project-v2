"""
Peak detection and analysis for ABR signals.

Provides utilities for detecting ABR peaks (waves I, III, V) and computing
latency/amplitude metrics for evaluation.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
import logging


def detect_peaks_1d(x: np.ndarray, height_sigma: float = 1.0, min_distance: int = 6) -> List[int]:
    """
    Simple peak detection using local maxima with height threshold.
    
    Args:
        x: 1D signal array
        height_sigma: Height threshold as multiple of signal standard deviation
        min_distance: Minimum samples between peaks
        
    Returns:
        List of peak indices
    """
    if len(x) < 3:
        return []
    
    # Compute height threshold
    signal_std = np.std(x)
    signal_mean = np.mean(x)
    height_threshold = signal_mean + height_sigma * signal_std
    
    # Find local maxima above threshold
    peaks = []
    
    for i in range(1, len(x) - 1):
        # Check if it's a local maximum above threshold
        if (x[i] > x[i-1] and x[i] > x[i+1] and x[i] > height_threshold):
            # Check minimum distance constraint
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
    
    return peaks


def match_peaks_to_labels(detected_peaks: List[int], 
                         true_latencies: Dict[str, float],
                         latency_windows: Dict[str, List[int]],
                         signal: np.ndarray,
                         sample_rate_hz: float = 20000.0) -> Dict[str, Dict[str, Any]]:
    """
    Match detected peaks to ABR wave labels (I, III, V).
    
    Args:
        detected_peaks: List of detected peak indices
        true_latencies: Dict with keys like "I_Latency", "III_Latency", "V_Latency" (in ms)
        latency_windows: Dict with keys "I", "III", "V" and [start, end] sample ranges
        signal: Original signal for amplitude extraction
        sample_rate_hz: Sampling rate for time conversion
        
    Returns:
        Dict with wave-specific metrics
    """
    results = {}
    
    # Convert sample indices to time in ms
    def samples_to_ms(samples):
        return samples * 1000.0 / sample_rate_hz
    
    def ms_to_samples(ms):
        return int(ms * sample_rate_hz / 1000.0)
    
    for wave in ['I', 'III', 'V']:
        wave_key = f"{wave}_Latency"
        
        # Initialize results for this wave
        wave_results = {
            'detected': False,
            'latency_error_ms': float('nan'),
            'latency_error_samples': float('nan'),
            'amplitude_error': float('nan'),
            'detected_latency_ms': float('nan'),
            'detected_amplitude': float('nan'),
            'true_latency_ms': float('nan'),
            'true_amplitude': float('nan')
        }
        
        # Check if we have ground truth for this wave
        if wave_key not in true_latencies or np.isnan(true_latencies[wave_key]):
            results[wave] = wave_results
            continue
            
        true_latency_ms = true_latencies[wave_key]
        true_latency_samples = ms_to_samples(true_latency_ms)
        wave_results['true_latency_ms'] = true_latency_ms
        
        # Get expected amplitude at true latency (if within signal bounds)
        if 0 <= true_latency_samples < len(signal):
            wave_results['true_amplitude'] = signal[true_latency_samples]
        
        # Find peaks within the expected window for this wave
        if wave in latency_windows:
            window_start, window_end = latency_windows[wave]
            window_peaks = [p for p in detected_peaks if window_start <= p <= window_end]
            
            if window_peaks:
                # Find the peak closest to the true latency
                closest_peak = min(window_peaks, key=lambda p: abs(p - true_latency_samples))
                
                wave_results['detected'] = True
                wave_results['detected_latency_ms'] = samples_to_ms(closest_peak)
                wave_results['detected_amplitude'] = signal[closest_peak]
                
                # Compute errors
                wave_results['latency_error_ms'] = abs(samples_to_ms(closest_peak) - true_latency_ms)
                wave_results['latency_error_samples'] = abs(closest_peak - true_latency_samples)
                
                if not np.isnan(wave_results['true_amplitude']):
                    wave_results['amplitude_error'] = abs(signal[closest_peak] - wave_results['true_amplitude'])
        
        results[wave] = wave_results
    
    return results


def peak_metrics(ref: torch.Tensor, gen: torch.Tensor, 
                labels_list: List[Dict[str, Any]], 
                config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute peak-based metrics for ABR evaluation.
    
    Args:
        ref: Reference signals [B, 1, T] or [B, T]
        gen: Generated signals [B, 1, T] or [B, T]
        labels_list: List of label dicts with peak information
        config: Configuration dict with peak detection parameters
        
    Returns:
        Dictionary with aggregated peak metrics
    """
    # Convert tensors to numpy
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()
    if isinstance(gen, torch.Tensor):
        gen = gen.detach().cpu().numpy()
    
    # Handle dimensions
    if ref.ndim == 3:
        ref = ref.squeeze(1)
    if gen.ndim == 3:
        gen = gen.squeeze(1)
    
    # Extract config parameters
    peak_config = config.get('peaks', {})
    find_peaks_config = peak_config.get('find_peaks', {})
    height_sigma = find_peaks_config.get('height_sigma', 1.0)
    min_distance = find_peaks_config.get('min_distance', 6)
    latency_windows = peak_config.get('latency_windows', {
        'I': [20, 50], 'III': [70, 110], 'V': [130, 170]
    })
    
    # Initialize metrics accumulator
    wave_metrics = {wave: {'latency_errors': [], 'amplitude_errors': [], 'detected_count': 0, 'total_count': 0} 
                   for wave in ['I', 'III', 'V']}
    
    batch_size = ref.shape[0]
    
    for i in range(batch_size):
        # Get labels for this sample if available
        if i < len(labels_list):
            labels = labels_list[i]
        else:
            labels = {}
        
        # Detect peaks in reference and generated signals
        ref_peaks = detect_peaks_1d(ref[i], height_sigma, min_distance)
        gen_peaks = detect_peaks_1d(gen[i], height_sigma, min_distance)
        
        # Match peaks to labels for both signals
        ref_matches = match_peaks_to_labels(ref_peaks, labels, latency_windows, ref[i])
        gen_matches = match_peaks_to_labels(gen_peaks, labels, latency_windows, gen[i])
        
        # Accumulate metrics
        for wave in ['I', 'III', 'V']:
            ref_wave = ref_matches[wave]
            gen_wave = gen_matches[wave]
            
            # Count total samples with ground truth for this wave
            if not np.isnan(ref_wave['true_latency_ms']):
                wave_metrics[wave]['total_count'] += 1
                
                # If both reference and generated detected peaks
                if ref_wave['detected'] and gen_wave['detected']:
                    wave_metrics[wave]['detected_count'] += 1
                    
                    # Latency error between generated and reference detections
                    latency_error = abs(gen_wave['detected_latency_ms'] - ref_wave['detected_latency_ms'])
                    wave_metrics[wave]['latency_errors'].append(latency_error)
                    
                    # Amplitude error between generated and reference detections
                    amplitude_error = abs(gen_wave['detected_amplitude'] - ref_wave['detected_amplitude'])
                    wave_metrics[wave]['amplitude_errors'].append(amplitude_error)
    
    # Compute summary statistics
    summary_metrics = {}
    
    for wave in ['I', 'III', 'V']:
        metrics = wave_metrics[wave]
        
        # Detection rate
        detection_rate = metrics['detected_count'] / max(metrics['total_count'], 1)
        summary_metrics[f'{wave}_detection_rate'] = detection_rate
        
        # Latency MAE
        if metrics['latency_errors']:
            latency_mae = np.mean(metrics['latency_errors'])
            summary_metrics[f'{wave}_latency_mae_ms'] = latency_mae
        else:
            summary_metrics[f'{wave}_latency_mae_ms'] = float('nan')
        
        # Amplitude MAE  
        if metrics['amplitude_errors']:
            amplitude_mae = np.mean(metrics['amplitude_errors'])
            summary_metrics[f'{wave}_amplitude_mae'] = amplitude_mae
        else:
            summary_metrics[f'{wave}_amplitude_mae'] = float('nan')
    
    # Overall metrics
    all_latency_errors = []
    all_amplitude_errors = []
    total_detected = 0
    total_samples = 0
    
    for wave in ['I', 'III', 'V']:
        metrics = wave_metrics[wave]
        all_latency_errors.extend(metrics['latency_errors'])
        all_amplitude_errors.extend(metrics['amplitude_errors'])
        total_detected += metrics['detected_count']
        total_samples += metrics['total_count']
    
    summary_metrics['overall_detection_rate'] = total_detected / max(total_samples, 1)
    summary_metrics['overall_latency_mae_ms'] = np.mean(all_latency_errors) if all_latency_errors else float('nan')
    summary_metrics['overall_amplitude_mae'] = np.mean(all_amplitude_errors) if all_amplitude_errors else float('nan')
    
    logging.info(f"Peak analysis: {total_detected}/{total_samples} peaks detected across all waves")
    
    return summary_metrics


def check_peak_labels_available(dataset_or_labels: Any) -> bool:
    """
    Check if peak labels are available in the dataset or label structure.
    
    Args:
        dataset_or_labels: Dataset object or sample labels dict/list
        
    Returns:
        True if peak labels are available
    """
    # Expected peak label columns
    peak_columns = [
        'I_Latency', 'III_Latency', 'V_Latency',
        'I_Amplitude', 'III_Amplitude', 'V_Amplitude'
    ]
    
    # Check if it's a dataset with data samples
    if hasattr(dataset_or_labels, 'data') and len(dataset_or_labels.data) > 0:
        sample = dataset_or_labels.data[0]
        available_columns = set(sample.keys()) if isinstance(sample, dict) else set()
        return any(col in available_columns for col in peak_columns)
    
    # Check if it's a list of label dicts
    elif isinstance(dataset_or_labels, list) and len(dataset_or_labels) > 0:
        sample = dataset_or_labels[0]
        if isinstance(sample, dict):
            return any(col in sample for col in peak_columns)
    
    # Check if it's a single label dict
    elif isinstance(dataset_or_labels, dict):
        return any(col in dataset_or_labels for col in peak_columns)
    
    return False


def extract_peak_labels(meta_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract peak labels from metadata batch.
    
    Args:
        meta_batch: List of metadata dictionaries
        
    Returns:
        List of dictionaries with peak label information
    """
    peak_labels = []
    
    for meta in meta_batch:
        labels = {}
        
        # Extract latency labels (in ms)
        for wave in ['I', 'III', 'V']:
            latency_key = f'{wave}_Latency'
            if latency_key in meta:
                labels[latency_key] = meta[latency_key]
        
        # Extract amplitude labels
        for wave in ['I', 'III', 'V']:
            amplitude_key = f'{wave}_Amplitude'
            if amplitude_key in meta:
                labels[amplitude_key] = meta[amplitude_key]
        
        peak_labels.append(labels)
    
    return peak_labels
