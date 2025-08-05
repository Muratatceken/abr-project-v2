#!/usr/bin/env python3
"""
ABR Evaluation Metrics

Comprehensive metrics for evaluating ABR signal generation quality,
peak prediction accuracy, classification performance, and threshold estimation.

Author: AI Assistant
Date: January 2025
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from scipy import signal as scipy_signal
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class ABRMetrics:
    """Container for all ABR evaluation metrics."""
    
    # Signal quality metrics
    signal_mse: float = 0.0
    signal_mae: float = 0.0
    signal_correlation: float = 0.0
    signal_snr: float = 0.0
    spectral_similarity: float = 0.0
    morphological_similarity: float = 0.0
    
    # Peak prediction metrics
    peak_existence_accuracy: float = 0.0
    peak_existence_f1: float = 0.0
    peak_latency_mae: float = 0.0
    peak_latency_rmse: float = 0.0
    peak_amplitude_mae: float = 0.0
    peak_amplitude_rmse: float = 0.0
    peak_correlation: float = 0.0
    
    # Classification metrics
    classification_accuracy: float = 0.0
    classification_f1_macro: float = 0.0
    classification_f1_weighted: float = 0.0
    classification_precision_macro: float = 0.0
    classification_recall_macro: float = 0.0
    classification_auc: float = 0.0
    
    # Threshold metrics
    threshold_mae: float = 0.0
    threshold_rmse: float = 0.0
    threshold_r2: float = 0.0
    threshold_correlation: float = 0.0
    
    # Clinical relevance metrics
    clinical_concordance: float = 0.0
    diagnostic_agreement: float = 0.0
    severity_correlation: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float))}
    
    def __str__(self) -> str:
        """String representation of metrics."""
        lines = ["ABR Evaluation Metrics:"]
        lines.append(f"  Signal Quality:")
        lines.append(f"    MSE: {self.signal_mse:.6f}")
        lines.append(f"    MAE: {self.signal_mae:.6f}")
        lines.append(f"    Correlation: {self.signal_correlation:.4f}")
        lines.append(f"    SNR: {self.signal_snr:.2f} dB")
        
        lines.append(f"  Peak Prediction:")
        lines.append(f"    Existence Accuracy: {self.peak_existence_accuracy:.4f}")
        lines.append(f"    Existence F1: {self.peak_existence_f1:.4f}")
        lines.append(f"    Latency MAE: {self.peak_latency_mae:.4f} ms")
        lines.append(f"    Amplitude MAE: {self.peak_amplitude_mae:.4f} μV")
        
        lines.append(f"  Classification:")
        lines.append(f"    Accuracy: {self.classification_accuracy:.4f}")
        lines.append(f"    F1 (macro): {self.classification_f1_macro:.4f}")
        lines.append(f"    AUC: {self.classification_auc:.4f}")
        
        lines.append(f"  Threshold:")
        lines.append(f"    MAE: {self.threshold_mae:.2f} dB nHL")
        lines.append(f"    R²: {self.threshold_r2:.4f}")
        
        return "\n".join(lines)


def compute_signal_metrics(
    predicted_signals: np.ndarray,
    true_signals: np.ndarray,
    sampling_rate: float = 40000.0
) -> Dict[str, float]:
    """
    Compute signal quality metrics.
    
    Args:
        predicted_signals: Predicted signals [batch, seq_len]
        true_signals: True signals [batch, seq_len]
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of signal metrics
    """
    metrics = {}
    
    # Basic error metrics
    metrics['mse'] = float(np.mean((predicted_signals - true_signals) ** 2))
    metrics['mae'] = float(np.mean(np.abs(predicted_signals - true_signals)))
    
    # Correlation metrics
    correlations = []
    for pred, true in zip(predicted_signals, true_signals):
        if np.std(pred) > 1e-8 and np.std(true) > 1e-8:
            corr, _ = pearsonr(pred, true)
            correlations.append(corr if not np.isnan(corr) else 0.0)
        else:
            correlations.append(0.0)
    
    metrics['correlation'] = float(np.mean(correlations))
    
    # Signal-to-Noise Ratio
    signal_power = np.mean(true_signals ** 2, axis=1)
    noise_power = np.mean((predicted_signals - true_signals) ** 2, axis=1)
    snr_values = []
    for sp, np_val in zip(signal_power, noise_power):
        if np_val > 1e-12:
            snr_db = 10 * np.log10(sp / np_val)
            snr_values.append(snr_db)
        else:
            snr_values.append(100.0)  # Very high SNR for perfect match
    
    metrics['snr'] = float(np.mean(snr_values))
    
    # Spectral similarity
    spectral_similarities = []
    for pred, true in zip(predicted_signals, true_signals):
        # Compute power spectral density
        freqs, pred_psd = scipy_signal.welch(pred, fs=sampling_rate, nperseg=min(64, len(pred)//4))
        _, true_psd = scipy_signal.welch(true, fs=sampling_rate, nperseg=min(64, len(true)//4))
        
        # Compute cosine similarity
        if np.sum(pred_psd) > 1e-12 and np.sum(true_psd) > 1e-12:
            sim = 1 - cosine(pred_psd, true_psd)
            spectral_similarities.append(sim if not np.isnan(sim) else 0.0)
        else:
            spectral_similarities.append(0.0)
    
    metrics['spectral_similarity'] = float(np.mean(spectral_similarities))
    
    # Morphological similarity (envelope correlation)
    morphological_similarities = []
    for pred, true in zip(predicted_signals, true_signals):
        # Compute signal envelopes
        pred_env = np.abs(scipy_signal.hilbert(pred))
        true_env = np.abs(scipy_signal.hilbert(true))
        
        if np.std(pred_env) > 1e-8 and np.std(true_env) > 1e-8:
            corr, _ = pearsonr(pred_env, true_env)
            morphological_similarities.append(corr if not np.isnan(corr) else 0.0)
        else:
            morphological_similarities.append(0.0)
    
    metrics['morphological_similarity'] = float(np.mean(morphological_similarities))
    
    return metrics


def compute_peak_metrics(
    predicted_peaks: Dict[str, np.ndarray],
    true_peaks: np.ndarray,
    peak_masks: np.ndarray
) -> Dict[str, float]:
    """
    Compute peak prediction metrics.
    
    Args:
        predicted_peaks: Dict with 'existence', 'latency', 'amplitude'
        true_peaks: True peak values [batch, 2] for [latency, amplitude]
        peak_masks: Peak existence masks [batch, 2]
        
    Returns:
        Dictionary of peak metrics
    """
    metrics = {}
    
    # Extract predictions
    pred_existence = predicted_peaks['existence']  # [batch]
    pred_latency = predicted_peaks['latency']      # [batch]
    pred_amplitude = predicted_peaks['amplitude']  # [batch]
    
    # True values
    true_latency = true_peaks[:, 0]    # [batch]
    true_amplitude = true_peaks[:, 1]  # [batch]
    true_existence = (peak_masks[:, 0] & peak_masks[:, 1]).astype(float)  # Both latency and amplitude valid
    
    # Peak existence metrics
    pred_existence_binary = (pred_existence > 0.5).astype(float)
    metrics['existence_accuracy'] = float(accuracy_score(true_existence, pred_existence_binary))
    
    # F1 score for existence
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_existence, pred_existence_binary, average='binary', zero_division=0
    )
    metrics['existence_f1'] = float(f1)
    metrics['existence_precision'] = float(precision)
    metrics['existence_recall'] = float(recall)
    
    # Peak regression metrics (only for existing peaks)
    existing_mask = true_existence.astype(bool)
    
    if np.sum(existing_mask) > 0:
        # Latency metrics
        valid_pred_latency = pred_latency[existing_mask]
        valid_true_latency = true_latency[existing_mask]
        
        metrics['latency_mae'] = float(np.mean(np.abs(valid_pred_latency - valid_true_latency)))
        metrics['latency_rmse'] = float(np.sqrt(np.mean((valid_pred_latency - valid_true_latency) ** 2)))
        
        if len(valid_pred_latency) > 1 and np.std(valid_pred_latency) > 1e-8:
            latency_corr, _ = pearsonr(valid_pred_latency, valid_true_latency)
            metrics['latency_correlation'] = float(latency_corr if not np.isnan(latency_corr) else 0.0)
        else:
            metrics['latency_correlation'] = 0.0
        
        # Amplitude metrics
        valid_pred_amplitude = pred_amplitude[existing_mask]
        valid_true_amplitude = true_amplitude[existing_mask]
        
        metrics['amplitude_mae'] = float(np.mean(np.abs(valid_pred_amplitude - valid_true_amplitude)))
        metrics['amplitude_rmse'] = float(np.sqrt(np.mean((valid_pred_amplitude - valid_true_amplitude) ** 2)))
        
        if len(valid_pred_amplitude) > 1 and np.std(valid_pred_amplitude) > 1e-8:
            amplitude_corr, _ = pearsonr(valid_pred_amplitude, valid_true_amplitude)
            metrics['amplitude_correlation'] = float(amplitude_corr if not np.isnan(amplitude_corr) else 0.0)
        else:
            metrics['amplitude_correlation'] = 0.0
    else:
        # No existing peaks
        metrics['latency_mae'] = 0.0
        metrics['latency_rmse'] = 0.0
        metrics['latency_correlation'] = 0.0
        metrics['amplitude_mae'] = 0.0
        metrics['amplitude_rmse'] = 0.0
        metrics['amplitude_correlation'] = 0.0
    
    return metrics


def compute_classification_metrics(
    predicted_logits: np.ndarray,
    true_labels: np.ndarray,
    n_classes: int = 5
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predicted_logits: Prediction logits [batch, n_classes]
        true_labels: True class labels [batch]
        n_classes: Number of classes
        
    Returns:
        Dictionary of classification metrics
    """
    metrics = {}
    
    # Convert logits to predictions
    predicted_probs = F.softmax(torch.from_numpy(predicted_logits), dim=1).numpy()
    predicted_labels = np.argmax(predicted_logits, axis=1)
    
    # Basic accuracy
    metrics['accuracy'] = float(accuracy_score(true_labels, predicted_labels))
    
    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro', zero_division=0
    )
    metrics['precision_macro'] = float(precision)
    metrics['recall_macro'] = float(recall)
    metrics['f1_macro'] = float(f1)
    
    # Weighted F1 (accounts for class imbalance)
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted', zero_division=0
    )
    metrics['f1_weighted'] = float(f1_weighted)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0
    )
    
    for i in range(min(n_classes, len(precision_per_class))):
        metrics[f'precision_class_{i}'] = float(precision_per_class[i])
        metrics[f'recall_class_{i}'] = float(recall_per_class[i])
        metrics[f'f1_class_{i}'] = float(f1_per_class[i])
    
    # AUC (if we have probability predictions)
    try:
        if len(np.unique(true_labels)) > 1:
            if n_classes == 2:
                # Binary classification
                auc = roc_auc_score(true_labels, predicted_probs[:, 1])
            else:
                # Multi-class classification
                auc = roc_auc_score(true_labels, predicted_probs, multi_class='ovr', average='macro')
            metrics['auc'] = float(auc)
        else:
            metrics['auc'] = 0.0
    except ValueError:
        metrics['auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def compute_threshold_metrics(
    predicted_thresholds: np.ndarray,
    true_thresholds: np.ndarray
) -> Dict[str, float]:
    """
    Compute threshold prediction metrics.
    
    Args:
        predicted_thresholds: Predicted threshold values [batch]
        true_thresholds: True threshold values [batch]
        
    Returns:
        Dictionary of threshold metrics
    """
    metrics = {}
    
    # Basic error metrics
    metrics['mae'] = float(np.mean(np.abs(predicted_thresholds - true_thresholds)))
    metrics['rmse'] = float(np.sqrt(np.mean((predicted_thresholds - true_thresholds) ** 2)))
    
    # R-squared
    if np.std(true_thresholds) > 1e-8:
        metrics['r2'] = float(r2_score(true_thresholds, predicted_thresholds))
    else:
        metrics['r2'] = 0.0
    
    # Correlation
    if np.std(predicted_thresholds) > 1e-8 and np.std(true_thresholds) > 1e-8:
        corr, _ = pearsonr(predicted_thresholds, true_thresholds)
        metrics['correlation'] = float(corr if not np.isnan(corr) else 0.0)
        
        # Spearman correlation (rank-based)
        spearman_corr, _ = spearmanr(predicted_thresholds, true_thresholds)
        metrics['spearman_correlation'] = float(spearman_corr if not np.isnan(spearman_corr) else 0.0)
    else:
        metrics['correlation'] = 0.0
        metrics['spearman_correlation'] = 0.0
    
    # Clinical thresholds accuracy (within ±5 dB, ±10 dB)
    diff = np.abs(predicted_thresholds - true_thresholds)
    metrics['accuracy_5db'] = float(np.mean(diff <= 5.0))
    metrics['accuracy_10db'] = float(np.mean(diff <= 10.0))
    metrics['accuracy_15db'] = float(np.mean(diff <= 15.0))
    
    return metrics


def compute_clinical_metrics(
    predicted_outputs: Dict[str, np.ndarray],
    true_outputs: Dict[str, np.ndarray],
    static_params: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute clinical relevance metrics.
    
    Args:
        predicted_outputs: Dictionary of model predictions
        true_outputs: Dictionary of true values
        static_params: Static parameters [batch, static_dim] (age, intensity, etc.)
        
    Returns:
        Dictionary of clinical metrics
    """
    metrics = {}
    
    # Get predictions and targets
    pred_class = np.argmax(predicted_outputs['classification'], axis=1)
    true_class = true_outputs['classification']
    pred_threshold = predicted_outputs['threshold']
    true_threshold = true_outputs['threshold']
    
    # Clinical concordance: agreement on hearing loss severity
    # Map classes to severity levels
    severity_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}  # Normal, Mild/Moderate, Severe/Profound
    
    pred_severity = np.array([severity_mapping.get(c, c) for c in pred_class])
    true_severity = np.array([severity_mapping.get(c, c) for c in true_class])
    
    metrics['clinical_concordance'] = float(accuracy_score(true_severity, pred_severity))
    
    # Diagnostic agreement: threshold-based diagnosis
    # Normal hearing: < 25 dB, Hearing loss: >= 25 dB
    pred_diagnosis = (pred_threshold >= 25).astype(int)
    true_diagnosis = (true_threshold >= 25).astype(int)
    
    metrics['diagnostic_agreement'] = float(accuracy_score(true_diagnosis, pred_diagnosis))
    
    # Severity correlation with thresholds
    if np.std(pred_threshold) > 1e-8 and np.std(true_threshold) > 1e-8:
        severity_corr, _ = pearsonr(pred_threshold, true_threshold)
        metrics['severity_correlation'] = float(severity_corr if not np.isnan(severity_corr) else 0.0)
    else:
        metrics['severity_correlation'] = 0.0
    
    # Age-stratified performance (if static params available)
    if static_params is not None and static_params.shape[1] > 0:
        age = static_params[:, 0]  # Assuming age is first parameter
        
        # Young adults (< 40), Middle-aged (40-65), Elderly (> 65)
        young_mask = age < 40
        middle_mask = (age >= 40) & (age <= 65)
        elderly_mask = age > 65
        
        for name, mask in [('young', young_mask), ('middle', middle_mask), ('elderly', elderly_mask)]:
            if np.sum(mask) > 0:
                subset_pred = pred_class[mask]
                subset_true = true_class[mask]
                metrics[f'accuracy_{name}'] = float(accuracy_score(subset_true, subset_pred))
            else:
                metrics[f'accuracy_{name}'] = 0.0
    
    return metrics


def compute_all_metrics(
    model_outputs: Dict[str, torch.Tensor],
    batch_targets: Dict[str, torch.Tensor],
    static_params: Optional[torch.Tensor] = None
) -> ABRMetrics:
    """
    Compute all ABR evaluation metrics.
    
    Args:
        model_outputs: Dictionary of model outputs
        batch_targets: Dictionary of true targets  
        static_params: Optional static parameters
        
    Returns:
        ABRMetrics object with all computed metrics
    """
    # Convert tensors to numpy arrays
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    # Convert outputs to numpy with shape handling
    pred_signals = to_numpy(model_outputs.get('signal', model_outputs.get('recon')))
    true_signals = to_numpy(batch_targets['signal'])
    
    # Ensure signal shapes match
    if pred_signals.ndim == 2 and true_signals.ndim == 3:
        # pred: [batch, seq_len], true: [batch, 1, seq_len]
        if true_signals.shape[1] == 1:
            true_signals = true_signals.squeeze(1)
    elif pred_signals.ndim == 3 and true_signals.ndim == 2:
        # pred: [batch, 1, seq_len], true: [batch, seq_len]
        if pred_signals.shape[1] == 1:
            pred_signals = pred_signals.squeeze(1)
    elif pred_signals.ndim == 3 and true_signals.ndim == 3:
        # Both 3D - ensure same shape
        if pred_signals.shape[1] == 1 and true_signals.shape[1] != 1:
            pred_signals = pred_signals.squeeze(1)
        elif true_signals.shape[1] == 1 and pred_signals.shape[1] != 1:
            true_signals = true_signals.squeeze(1)
    
    # Peak predictions
    if 'peak' in model_outputs:
        peak_outputs = model_outputs['peak']
        if isinstance(peak_outputs, (list, tuple)):
            pred_peaks = {
                'existence': to_numpy(torch.sigmoid(peak_outputs[0])),
                'latency': to_numpy(peak_outputs[1]), 
                'amplitude': to_numpy(peak_outputs[2])
            }
        else:
            # Single tensor output - split appropriately
            pred_peaks = {
                'existence': to_numpy(torch.sigmoid(peak_outputs[:, 0])),
                'latency': to_numpy(peak_outputs[:, 1]),
                'amplitude': to_numpy(peak_outputs[:, 2])
            }
    else:
        # Create dummy peak predictions
        batch_size = pred_signals.shape[0]
        pred_peaks = {
            'existence': np.zeros(batch_size),
            'latency': np.zeros(batch_size),
            'amplitude': np.zeros(batch_size)
        }
    
    true_peaks = to_numpy(batch_targets['v_peak'])
    peak_masks = to_numpy(batch_targets['v_peak_mask'])
    
    # Classification predictions
    pred_class_logits = to_numpy(model_outputs.get('class', model_outputs.get('classification_logits')))
    true_class = to_numpy(batch_targets['target'])
    
    # Threshold predictions with shape handling
    pred_threshold = to_numpy(model_outputs.get('threshold'))
    true_threshold = to_numpy(batch_targets['threshold'])
    
    # Handle threshold shape mismatch (uncertainty predictions)
    if pred_threshold is not None and pred_threshold.ndim > 1:
        if pred_threshold.shape[-1] == 2:
            # Model predicts [mean, std] - use only mean for evaluation
            pred_threshold = pred_threshold[..., 0]
        elif pred_threshold.shape[-1] == 1:
            # Squeeze single dimension
            pred_threshold = pred_threshold.squeeze(-1)
    
    # Ensure true threshold is 1D
    if true_threshold is not None and true_threshold.ndim > 1:
        true_threshold = true_threshold.squeeze()
    
    # Compute metrics
    metrics = ABRMetrics()
    
    # Signal metrics
    signal_metrics = compute_signal_metrics(pred_signals, true_signals)
    metrics.signal_mse = signal_metrics['mse']
    metrics.signal_mae = signal_metrics['mae']
    metrics.signal_correlation = signal_metrics['correlation']
    metrics.signal_snr = signal_metrics['snr']
    metrics.spectral_similarity = signal_metrics['spectral_similarity']
    metrics.morphological_similarity = signal_metrics['morphological_similarity']
    
    # Peak metrics
    peak_metrics = compute_peak_metrics(pred_peaks, true_peaks, peak_masks)
    metrics.peak_existence_accuracy = peak_metrics['existence_accuracy']
    metrics.peak_existence_f1 = peak_metrics['existence_f1']
    metrics.peak_latency_mae = peak_metrics['latency_mae']
    metrics.peak_latency_rmse = peak_metrics['latency_rmse']
    metrics.peak_amplitude_mae = peak_metrics['amplitude_mae']
    metrics.peak_amplitude_rmse = peak_metrics['amplitude_rmse']
    metrics.peak_correlation = peak_metrics.get('latency_correlation', 0.0)
    
    # Classification metrics
    if pred_class_logits is not None and pred_class_logits.size > 0:
        class_metrics = compute_classification_metrics(pred_class_logits, true_class)
        metrics.classification_accuracy = class_metrics['accuracy']
        metrics.classification_f1_macro = class_metrics['f1_macro']
        metrics.classification_f1_weighted = class_metrics['f1_weighted']
        metrics.classification_precision_macro = class_metrics['precision_macro']
        metrics.classification_recall_macro = class_metrics['recall_macro']
        metrics.classification_auc = class_metrics['auc']
    
    # Threshold metrics
    if pred_threshold is not None and pred_threshold.size > 0:
        threshold_metrics = compute_threshold_metrics(pred_threshold, true_threshold)
        metrics.threshold_mae = threshold_metrics['mae']
        metrics.threshold_rmse = threshold_metrics['rmse']
        metrics.threshold_r2 = threshold_metrics['r2']
        metrics.threshold_correlation = threshold_metrics['correlation']
    
    # Clinical metrics
    clinical_outputs = {
        'classification': pred_class_logits,
        'threshold': pred_threshold
    }
    clinical_targets = {
        'classification': true_class,
        'threshold': true_threshold
    }
    
    if static_params is not None:
        static_np = to_numpy(static_params)
    else:
        static_np = None
    
    try:
        clinical_metrics = compute_clinical_metrics(clinical_outputs, clinical_targets, static_np)
        metrics.clinical_concordance = clinical_metrics['clinical_concordance']
        metrics.diagnostic_agreement = clinical_metrics['diagnostic_agreement']
        metrics.severity_correlation = clinical_metrics['severity_correlation']
    except Exception as e:
        logger.warning(f"Failed to compute clinical metrics: {e}")
        metrics.clinical_concordance = 0.0
        metrics.diagnostic_agreement = 0.0
        metrics.severity_correlation = 0.0
    
    return metrics