#!/usr/bin/env python3
"""
Advanced Metrics Utilities for ABR Model Evaluation

This module provides comprehensive metric computation functions for:
- Signal reconstruction metrics
- Peak detection metrics  
- Classification metrics
- Threshold estimation metrics
- Clinical error analysis
- Bootstrap confidence intervals

Author: AI Assistant
Date: January 2025
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, mean_absolute_error, 
    mean_squared_error, r2_score
)
import scipy.stats
from collections import defaultdict
import warnings

# Optional imports
try:
    from fastdtw import fastdtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class MetricsCalculator:
    """Comprehensive metrics calculator for ABR model evaluation."""
    
    def __init__(self, class_names: List[str] = None):
        """Initialize metrics calculator.
        
        Args:
            class_names: List of class names for classification tasks
        """
        self.class_names = class_names or ["NORMAL", "NÖROPATİ", "SNİK", "TOTAL", "İTİK"]
        
    # ==================== SIGNAL RECONSTRUCTION METRICS ====================
    
    def compute_signal_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        compute_dtw: bool = False,
        compute_fft_mse: bool = False
    ) -> Dict[str, float]:
        """
        Compute comprehensive signal reconstruction metrics.
        
        Args:
            y_true: Ground truth signals [batch, channels, length]
            y_pred: Predicted signals [batch, channels, length]
            compute_dtw: Whether to compute DTW distance
            compute_fft_mse: Whether to compute FFT-based MSE
            
        Returns:
            Dictionary of signal metrics
        """
        # Flatten for easier computation
        y_true_flat = y_true.reshape(y_true.shape[0], -1)
        y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
        
        metrics = {}
        
        # Basic reconstruction metrics
        mse_per_sample = np.mean((y_pred_flat - y_true_flat) ** 2, axis=1)
        mae_per_sample = np.mean(np.abs(y_pred_flat - y_true_flat), axis=1)
        rmse_per_sample = np.sqrt(mse_per_sample)
        
        metrics['mse_mean'] = np.mean(mse_per_sample)
        metrics['mse_std'] = np.std(mse_per_sample)
        metrics['mae_mean'] = np.mean(mae_per_sample)
        metrics['mae_std'] = np.std(mae_per_sample)
        metrics['rmse_mean'] = np.mean(rmse_per_sample)
        metrics['rmse_std'] = np.std(rmse_per_sample)
        
        # Signal-to-Noise Ratio
        signal_power = np.mean(y_true_flat ** 2, axis=1)
        noise_power = mse_per_sample
        snr_per_sample = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        metrics['snr_mean'] = np.mean(snr_per_sample)
        metrics['snr_std'] = np.std(snr_per_sample)
        
        # Pearson correlation per sample
        correlations = []
        for i in range(y_true_flat.shape[0]):
            if np.std(y_pred_flat[i]) > 1e-8 and np.std(y_true_flat[i]) > 1e-8:
                try:
                    corr, _ = scipy.stats.pearsonr(y_true_flat[i], y_pred_flat[i])
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    pass
        
        if correlations:
            metrics['correlation_mean'] = np.mean(correlations)
            metrics['correlation_std'] = np.std(correlations)
        else:
            metrics['correlation_mean'] = 0.0
            metrics['correlation_std'] = 0.0
        
        # DTW distance (if available and requested)
        if compute_dtw and DTW_AVAILABLE:
            dtw_distances = []
            for i in range(min(100, y_true_flat.shape[0])):  # Limit for performance
                try:
                    distance, _ = fastdtw(y_true_flat[i], y_pred_flat[i])
                    dtw_distances.append(distance)
                except:
                    pass
            
            if dtw_distances:
                metrics['dtw_mean'] = np.mean(dtw_distances)
                metrics['dtw_std'] = np.std(dtw_distances)
        
        # FFT-based MSE (frequency domain)
        if compute_fft_mse:
            fft_mse_per_sample = []
            for i in range(y_true_flat.shape[0]):
                try:
                    fft_true = np.fft.fft(y_true_flat[i])
                    fft_pred = np.fft.fft(y_pred_flat[i])
                    fft_mse = np.mean(np.abs(fft_true - fft_pred) ** 2)
                    fft_mse_per_sample.append(fft_mse)
                except:
                    pass
            
            if fft_mse_per_sample:
                metrics['fft_mse_mean'] = np.mean(fft_mse_per_sample)
                metrics['fft_mse_std'] = np.std(fft_mse_per_sample)
        
        # Store individual sample metrics for distribution analysis
        metrics['mse_samples'] = mse_per_sample
        metrics['mae_samples'] = mae_per_sample
        metrics['snr_samples'] = snr_per_sample
        metrics['correlation_samples'] = np.array(correlations) if correlations else np.array([])
        
        return metrics
    
    # ==================== PEAK DETECTION METRICS ====================
    
    def compute_peak_metrics(
        self,
        peak_existence_pred: np.ndarray,
        peak_existence_true: np.ndarray,
        peak_latency_pred: np.ndarray,
        peak_latency_true: np.ndarray,
        peak_amplitude_pred: np.ndarray,
        peak_amplitude_true: np.ndarray,
        peak_masks: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive peak detection and estimation metrics.
        
        Args:
            peak_existence_pred: Predicted peak existence [batch]
            peak_existence_true: True peak existence [batch]
            peak_latency_pred: Predicted peak latencies [batch]
            peak_latency_true: True peak latencies [batch]
            peak_amplitude_pred: Predicted peak amplitudes [batch]
            peak_amplitude_true: True peak amplitudes [batch]
            peak_masks: Masks indicating valid peaks [batch]
            
        Returns:
            Dictionary of peak metrics
        """
        metrics = {}
        
        # Peak existence metrics (binary classification)
        if len(peak_existence_pred) > 0 and len(peak_existence_true) > 0:
            # Convert to binary if needed
            if peak_existence_pred.dtype != bool:
                peak_existence_pred_binary = peak_existence_pred > 0.5
            else:
                peak_existence_pred_binary = peak_existence_pred
            
            if peak_existence_true.dtype != bool:
                peak_existence_true_binary = peak_existence_true > 0.5
            else:
                peak_existence_true_binary = peak_existence_true
            
            metrics['existence_accuracy'] = accuracy_score(peak_existence_true_binary, peak_existence_pred_binary)
            metrics['existence_precision'] = precision_score(peak_existence_true_binary, peak_existence_pred_binary, zero_division=0)
            metrics['existence_recall'] = recall_score(peak_existence_true_binary, peak_existence_pred_binary, zero_division=0)
            metrics['existence_f1'] = f1_score(peak_existence_true_binary, peak_existence_pred_binary, zero_division=0)
            
            # Clinical error flags
            false_positives = np.sum((peak_existence_pred_binary == True) & (peak_existence_true_binary == False))
            false_negatives = np.sum((peak_existence_pred_binary == False) & (peak_existence_true_binary == True))
            
            metrics['false_peak_detections'] = int(false_positives)
            metrics['missed_peaks'] = int(false_negatives)
        
        # Peak parameter metrics (only for existing peaks)
        if peak_masks is not None:
            valid_mask = peak_masks.any(axis=1) if len(peak_masks.shape) > 1 else peak_masks
        else:
            valid_mask = peak_existence_true > 0.5
        
        if np.sum(valid_mask) > 0:
            # Latency metrics
            latency_true_valid = peak_latency_true[valid_mask]
            latency_pred_valid = peak_latency_pred[valid_mask]
            
            if len(latency_true_valid) > 0:
                metrics['latency_mae'] = mean_absolute_error(latency_true_valid, latency_pred_valid)
                metrics['latency_mse'] = mean_squared_error(latency_true_valid, latency_pred_valid)
                metrics['latency_rmse'] = np.sqrt(metrics['latency_mse'])
                
                try:
                    metrics['latency_r2'] = r2_score(latency_true_valid, latency_pred_valid)
                except:
                    metrics['latency_r2'] = 0.0
                
                try:
                    if np.std(latency_pred_valid) > 1e-8 and np.std(latency_true_valid) > 1e-8:
                        corr, _ = scipy.stats.pearsonr(latency_true_valid, latency_pred_valid)
                        metrics['latency_correlation'] = corr if not np.isnan(corr) else 0.0
                    else:
                        metrics['latency_correlation'] = 0.0
                except:
                    metrics['latency_correlation'] = 0.0
                
                # Store samples for distribution analysis
                metrics['latency_errors'] = np.abs(latency_true_valid - latency_pred_valid)
            
            # Amplitude metrics
            amplitude_true_valid = peak_amplitude_true[valid_mask]
            amplitude_pred_valid = peak_amplitude_pred[valid_mask]
            
            if len(amplitude_true_valid) > 0:
                metrics['amplitude_mae'] = mean_absolute_error(amplitude_true_valid, amplitude_pred_valid)
                metrics['amplitude_mse'] = mean_squared_error(amplitude_true_valid, amplitude_pred_valid)
                metrics['amplitude_rmse'] = np.sqrt(metrics['amplitude_mse'])
                
                try:
                    metrics['amplitude_r2'] = r2_score(amplitude_true_valid, amplitude_pred_valid)
                except:
                    metrics['amplitude_r2'] = 0.0
                
                try:
                    if np.std(amplitude_pred_valid) > 1e-8 and np.std(amplitude_true_valid) > 1e-8:
                        corr, _ = scipy.stats.pearsonr(amplitude_true_valid, amplitude_pred_valid)
                        metrics['amplitude_correlation'] = corr if not np.isnan(corr) else 0.0
                    else:
                        metrics['amplitude_correlation'] = 0.0
                except:
                    metrics['amplitude_correlation'] = 0.0
                
                # Store samples for distribution analysis
                metrics['amplitude_errors'] = np.abs(amplitude_true_valid - amplitude_pred_valid)
                
                # Binned amplitude classification (optional)
                amplitude_bins = self._create_amplitude_bins(amplitude_true_valid)
                if amplitude_bins is not None:
                    pred_bins = self._classify_amplitude_bins(amplitude_pred_valid, amplitude_bins)
                    true_bins = self._classify_amplitude_bins(amplitude_true_valid, amplitude_bins)
                    
                    if len(pred_bins) > 0 and len(true_bins) > 0:
                        metrics['amplitude_bin_accuracy'] = accuracy_score(true_bins, pred_bins)
        
        return metrics
    
    def _create_amplitude_bins(self, amplitudes: np.ndarray) -> Optional[np.ndarray]:
        """Create amplitude bins for classification."""
        if len(amplitudes) < 10:  # Need sufficient data
            return None
        
        # Create tertile bins (low, medium, high)
        percentiles = np.percentile(amplitudes, [33.33, 66.67])
        return percentiles
    
    def _classify_amplitude_bins(self, amplitudes: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Classify amplitudes into bins."""
        return np.digitize(amplitudes, bins)
    
    # ==================== CLASSIFICATION METRICS ====================
    
    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred_logits: np.ndarray,
        y_pred_probs: np.ndarray = None,
        compute_auc: bool = False
    ) -> Dict[str, Any]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True class labels [batch]
            y_pred_logits: Predicted logits [batch, num_classes]
            y_pred_probs: Predicted probabilities [batch, num_classes] (optional)
            compute_auc: Whether to compute AUC scores
            
        Returns:
            Dictionary of classification metrics
        """
        # Convert logits to predictions
        y_pred = np.argmax(y_pred_logits, axis=1)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # All possible labels
        all_labels = list(range(len(self.class_names)))
        
        # Per-class metrics with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress sklearn classification warnings
            
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=all_labels)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=all_labels)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=all_labels)
            
            # Macro and weighted averages
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=all_labels)
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=all_labels)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=all_labels)
            
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_labels)
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_labels)
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_labels)
        
        # Store per-class metrics
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Confusion matrix with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress sklearn classification warnings
            cm = confusion_matrix(y_true, y_pred, labels=all_labels)
            metrics['confusion_matrix'] = cm
            metrics['confusion_matrix_normalized'] = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Class support counts
        unique, counts = np.unique(y_true, return_counts=True)
        support_dict = dict(zip(unique, counts))
        for i, class_name in enumerate(self.class_names):
            metrics[f'support_{class_name}'] = support_dict.get(i, 0)
        
        # Check for zero prediction coverage
        predicted_classes = set(y_pred)
        all_classes = set(range(len(self.class_names)))
        unpredicted_classes = all_classes - predicted_classes
        
        metrics['unpredicted_classes'] = [self.class_names[i] for i in unpredicted_classes]
        metrics['num_unpredicted_classes'] = len(unpredicted_classes)
        
        # AUC scores (if probabilities available)
        if compute_auc and y_pred_probs is not None:
            try:
                # Multi-class AUC (one-vs-rest)
                metrics['auc_macro'] = roc_auc_score(y_true, y_pred_probs, average='macro', multi_class='ovr')
                metrics['auc_weighted'] = roc_auc_score(y_true, y_pred_probs, average='weighted', multi_class='ovr')
                
                # Per-class AUC
                for i, class_name in enumerate(self.class_names):
                    if i < y_pred_probs.shape[1]:
                        y_true_binary = (y_true == i).astype(int)
                        if len(np.unique(y_true_binary)) > 1:  # Need both classes
                            auc = roc_auc_score(y_true_binary, y_pred_probs[:, i])
                            metrics[f'auc_{class_name}'] = auc
            except Exception as e:
                warnings.warn(f"Could not compute AUC scores: {e}")
        
        return metrics
    
    # ==================== THRESHOLD ESTIMATION METRICS ====================
    
    def compute_threshold_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        clinical_bands: bool = True
    ) -> Dict[str, Any]:
        """
        Compute comprehensive threshold estimation metrics.
        
        Args:
            y_true: True thresholds [batch]
            y_pred: Predicted thresholds [batch]
            clinical_bands: Whether to compute clinical error bands
            
        Returns:
            Dictionary of threshold metrics
        """
        metrics = {}
        
        # Flatten arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Ensure same length
        min_len = min(len(y_true_flat), len(y_pred_flat))
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]
        
        if len(y_true_flat) == 0:
            return metrics
        
        # Basic regression metrics
        metrics['mae'] = mean_absolute_error(y_true_flat, y_pred_flat)
        metrics['mse'] = mean_squared_error(y_true_flat, y_pred_flat)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        try:
            metrics['r2'] = r2_score(y_true_flat, y_pred_flat)
        except:
            metrics['r2'] = 0.0
        
        try:
            if np.std(y_pred_flat) > 1e-8 and np.std(y_true_flat) > 1e-8:
                corr, _ = scipy.stats.pearsonr(y_true_flat, y_pred_flat)
                metrics['correlation'] = corr if not np.isnan(corr) else 0.0
            else:
                metrics['correlation'] = 0.0
        except:
            metrics['correlation'] = 0.0
        
        # Clinical error analysis
        if clinical_bands:
            errors = y_pred_flat - y_true_flat  # Positive = overestimate, Negative = underestimate
            
            # Clinical error flags (>20 dB threshold)
            false_clear = np.sum(errors < -20)  # Underestimate > 20 dB
            false_impairment = np.sum(errors > 20)  # Overestimate > 20 dB
            
            metrics['false_clear_count'] = int(false_clear)
            metrics['false_impairment_count'] = int(false_impairment)
            metrics['false_clear_rate'] = false_clear / len(errors)
            metrics['false_impairment_rate'] = false_impairment / len(errors)
            metrics['clinical_error_rate'] = (false_clear + false_impairment) / len(errors)
            
            # Error band analysis
            error_bands = {
                'excellent': np.sum(np.abs(errors) <= 5),    # ±5 dB
                'good': np.sum((np.abs(errors) > 5) & (np.abs(errors) <= 10)),  # 5-10 dB
                'acceptable': np.sum((np.abs(errors) > 10) & (np.abs(errors) <= 20)),  # 10-20 dB
                'poor': np.sum(np.abs(errors) > 20)  # >20 dB
            }
            
            for band, count in error_bands.items():
                metrics[f'error_band_{band}_count'] = int(count)
                metrics[f'error_band_{band}_rate'] = count / len(errors)
        
        # Store error samples for distribution analysis
        metrics['threshold_errors'] = np.abs(y_true_flat - y_pred_flat)
        metrics['threshold_residuals'] = y_pred_flat - y_true_flat
        
        return metrics
    
    # ==================== BOOTSTRAP CONFIDENCE INTERVALS ====================
    
    def bootstrap_ci(
        self, 
        values: np.ndarray, 
        n_bootstrap: int = 1000, 
        confidence_level: float = 0.95,
        statistic: str = 'mean'
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence intervals.
        
        Args:
            values: Array of values to bootstrap
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            statistic: Statistic to compute ('mean', 'median', 'std')
            
        Returns:
            Tuple of (statistic_value, lower_ci, upper_ci)
        """
        if len(values) == 0:
            return 0.0, 0.0, 0.0
        
        # Choose statistic function
        if statistic == 'mean':
            stat_func = np.mean
        elif statistic == 'median':
            stat_func = np.median
        elif statistic == 'std':
            stat_func = np.std
        else:
            stat_func = np.mean
        
        # Original statistic
        original_stat = stat_func(values)
        
        # Bootstrap sampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_stats.append(stat_func(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_ci = np.percentile(bootstrap_stats, lower_percentile)
        upper_ci = np.percentile(bootstrap_stats, upper_percentile)
        
        return original_stat, lower_ci, upper_ci
    
    # ==================== STRATIFIED EVALUATION ====================
    
    def compute_stratified_metrics(
        self,
        data: Dict[str, np.ndarray],
        predictions: Dict[str, np.ndarray],
        stratification_keys: List[str] = ['class', 'age_bin', 'intensity_bin']
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute metrics stratified by different variables.
        
        Args:
            data: Dictionary containing ground truth data
            predictions: Dictionary containing model predictions
            stratification_keys: Keys to stratify by
            
        Returns:
            Dictionary of stratified metrics
        """
        stratified_metrics = {}
        
        for key in stratification_keys:
            if key in data:
                strata_values = np.unique(data[key])
                stratified_metrics[key] = {}
                
                for stratum in strata_values:
                    mask = data[key] == stratum
                    
                    if np.sum(mask) > 0:  # Ensure we have samples in this stratum
                        # Extract data for this stratum
                        stratum_data = {k: v[mask] for k, v in data.items()}
                        stratum_preds = {k: v[mask] for k, v in predictions.items()}
                        
                        # Compute metrics for this stratum
                        stratum_metrics = {}
                        
                        # Signal metrics
                        if 'signal' in stratum_data and 'recon' in stratum_preds:
                            signal_metrics = self.compute_signal_metrics(
                                stratum_data['signal'], stratum_preds['recon']
                            )
                            stratum_metrics.update({f'signal_{k}': v for k, v in signal_metrics.items()})
                        
                        # Classification metrics
                        if 'target' in stratum_data and 'class' in stratum_preds:
                            class_metrics = self.compute_classification_metrics(
                                stratum_data['target'], stratum_preds['class']
                            )
                            stratum_metrics.update({f'class_{k}': v for k, v in class_metrics.items()})
                        
                        # Threshold metrics
                        if 'threshold' in stratum_data and 'threshold' in stratum_preds:
                            thresh_metrics = self.compute_threshold_metrics(
                                stratum_data['threshold'], stratum_preds['threshold']
                            )
                            stratum_metrics.update({f'threshold_{k}': v for k, v in thresh_metrics.items()})
                        
                        # Peak metrics
                        if all(k in stratum_data for k in ['v_peak', 'v_peak_mask']) and 'peak' in stratum_preds:
                            peak_existence_true = stratum_data['v_peak_mask'].any(axis=1)
                            peak_metrics = self.compute_peak_metrics(
                                stratum_preds['peak'][0] if len(stratum_preds['peak']) > 0 else np.zeros(len(peak_existence_true)),
                                peak_existence_true,
                                stratum_preds['peak'][1] if len(stratum_preds['peak']) > 1 else np.zeros(len(peak_existence_true)),
                                stratum_data['v_peak'][:, 0],
                                stratum_preds['peak'][2] if len(stratum_preds['peak']) > 2 else np.zeros(len(peak_existence_true)),
                                stratum_data['v_peak'][:, 1],
                                stratum_data['v_peak_mask']
                            )
                            stratum_metrics.update({f'peak_{k}': v for k, v in peak_metrics.items()})
                        
                        stratified_metrics[key][str(stratum)] = stratum_metrics
        
        return stratified_metrics 