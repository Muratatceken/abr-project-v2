#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline for Multi-Task ABR Signal Generation

This module provides a complete evaluation framework for assessing the performance
of ABR signal generation models across multiple tasks:
- Signal reconstruction quality
- Peak detection and estimation
- Hearing loss classification
- Hearing threshold estimation

Author: AI Assistant
Date: January 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import csv
from collections import defaultdict
import warnings
from io import BytesIO
import base64

# Scientific computing
import scipy.stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# Data handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("pandas not available. Summary table will use basic CSV.")

# Optional imports
try:
    from fastdtw import fastdtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    warnings.warn("fastdtw not available. DTW metrics will be skipped.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False


class ABRComprehensiveEvaluator:
    """
    Comprehensive evaluator for multi-task ABR signal generation models.
    
    Provides detailed assessment across all model outputs with visual diagnostics,
    clinical performance flags, and comprehensive metrics.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        class_names: Optional[List[str]] = None,
        save_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize comprehensive evaluator.
        
        Args:
            config: Evaluation configuration dictionary
            class_names: List of class names for classification
            save_dir: Directory to save evaluation results
            device: Device for computations
        """
        self.config = config or self._get_default_config()
        self.class_names = class_names or ["NORMAL", "NÖROPATİ", "SNİK", "TOTAL", "İTİK"]
        self.save_dir = Path(save_dir) if save_dir else Path("outputs/evaluation")
        self.device = device or torch.device('cpu')
        
        # Create save directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "figures").mkdir(exist_ok=True)
        (self.save_dir / "data").mkdir(exist_ok=True)
        
        # Storage for batch results
        self.batch_results = []
        self.aggregate_metrics = {}
        self.failure_modes = defaultdict(int)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default evaluation configuration."""
        return {
            'dtw': DTW_AVAILABLE,
            'fft_mse': True,
            'waveform_samples': 5,
            'classification_metrics': ['accuracy', 'f1_macro', 'confusion_matrix'],
            'clinical_thresholds': {
                'threshold_overestimate': 15.0,  # dB
                'peak_latency_tolerance': 0.5,   # ms
                'peak_amplitude_tolerance': 0.1  # μV
            },
            'visualization': {
                'figsize': (15, 10),
                'dpi': 150,
                'save_format': 'png',
                'clinical_overlays': {
                    'show_patient_id': True,
                    'show_class_info': True,
                    'show_threshold_info': True,
                    'peak_markers': True
                },
                'diagnostic_cards': {
                    'layout': '2x2',
                    'include_text_overlay': True,
                    'card_figsize': [12, 10]
                }
            }
        }
    
    # ==================== TASK 1: RECONSTRUCTION QUALITY ====================
    
    def evaluate_reconstruction(
        self, 
        y_true: torch.Tensor, 
        y_pred: torch.Tensor
    ) -> Dict[str, float]:
        """
        Comprehensive signal reconstruction evaluation.
        
        Args:
            y_true: Ground truth signals [batch, channels, length]
            y_pred: Predicted signals [batch, channels, length]
            
        Returns:
            Dictionary of reconstruction metrics
        """
        # Ensure tensors are on CPU for numpy operations
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        
        # Flatten for easier computation
        y_true_flat = y_true_np.reshape(y_true_np.shape[0], -1)
        y_pred_flat = y_pred_np.reshape(y_pred_np.shape[0], -1)
        
        metrics = {}
        
        # Basic reconstruction metrics
        mse_per_sample = np.mean((y_pred_flat - y_true_flat) ** 2, axis=1)
        mae_per_sample = np.mean(np.abs(y_pred_flat - y_true_flat), axis=1)
        rmse_per_sample = np.sqrt(mse_per_sample)
        
        metrics['mse'] = np.mean(mse_per_sample)
        metrics['mse_std'] = np.std(mse_per_sample)
        metrics['mae'] = np.mean(mae_per_sample)
        metrics['mae_std'] = np.std(mae_per_sample)
        metrics['rmse'] = np.mean(rmse_per_sample)
        metrics['rmse_std'] = np.std(rmse_per_sample)
        
        # Signal-to-Noise Ratio
        signal_power = np.mean(y_true_flat ** 2, axis=1)
        noise_power = mse_per_sample
        snr_per_sample = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        metrics['snr'] = np.mean(snr_per_sample)
        metrics['snr_std'] = np.std(snr_per_sample)
        
        # Pearson correlation per sample
        correlations = []
        for i in range(y_true_flat.shape[0]):
            if np.std(y_pred_flat[i]) > 1e-8 and np.std(y_true_flat[i]) > 1e-8:
                corr, _ = scipy.stats.pearsonr(y_true_flat[i], y_pred_flat[i])
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if correlations:
            metrics['pearson_corr'] = np.mean(correlations)
            metrics['pearson_corr_std'] = np.std(correlations)
        else:
            metrics['pearson_corr'] = 0.0
            metrics['pearson_corr_std'] = 0.0
        
        # Dynamic Time Warping (if available)
        if self.config['dtw'] and DTW_AVAILABLE:
            dtw_distances = []
            # Limit to first 20 samples for efficiency
            n_samples = min(20, y_true_flat.shape[0])
            
            for i in range(n_samples):
                try:
                    distance, _ = fastdtw(y_true_flat[i], y_pred_flat[i])
                    dtw_distances.append(distance)
                except:
                    continue
            
            if dtw_distances:
                metrics['dtw'] = np.mean(dtw_distances)
                metrics['dtw_std'] = np.std(dtw_distances)
        
        # Spectral analysis (FFT-based)
        if self.config['fft_mse']:
            y_true_fft = np.fft.fft(y_true_flat, axis=1)
            y_pred_fft = np.fft.fft(y_pred_flat, axis=1)
            
            # Magnitude spectrum MSE
            true_mag = np.abs(y_true_fft)
            pred_mag = np.abs(y_pred_fft)
            spectral_mse_per_sample = np.mean((pred_mag - true_mag) ** 2, axis=1)
            
            metrics['spectral_mse'] = np.mean(spectral_mse_per_sample)
            metrics['spectral_mse_std'] = np.std(spectral_mse_per_sample)
            
            # Phase coherence
            true_phase = np.angle(y_true_fft)
            pred_phase = np.angle(y_pred_fft)
            phase_diff = np.abs(true_phase - pred_phase)
            phase_coherence = np.cos(phase_diff)
            
            metrics['phase_coherence'] = np.mean(phase_coherence)
            metrics['phase_coherence_std'] = np.std(phase_coherence)
        
        return metrics
    
    # ==================== TASK 2: PEAK ESTIMATION ====================
    
    def evaluate_peak_estimation(
        self,
        peak_exists_pred: torch.Tensor,
        peak_exists_true: torch.Tensor,
        peak_latency_pred: torch.Tensor,
        peak_latency_true: torch.Tensor,
        peak_amplitude_pred: torch.Tensor,
        peak_amplitude_true: torch.Tensor,
        peak_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Comprehensive peak estimation evaluation.
        
        Args:
            peak_exists_pred: Peak existence predictions [batch]
            peak_exists_true: Ground truth peak existence [batch]
            peak_latency_pred: Peak latency predictions [batch]
            peak_latency_true: Ground truth peak latencies [batch]
            peak_amplitude_pred: Peak amplitude predictions [batch]
            peak_amplitude_true: Ground truth peak amplitudes [batch]
            peak_mask: Mask for valid peaks [batch, 2] (latency, amplitude)
            
        Returns:
            Dictionary of peak estimation metrics
        """
        metrics = {}
        
        # Convert to numpy
        exists_pred_np = torch.sigmoid(peak_exists_pred).detach().cpu().numpy()
        exists_true_np = peak_exists_true.detach().cpu().numpy()
        
        # Peak existence evaluation (binary classification)
        exists_pred_binary = (exists_pred_np > 0.5).astype(int)
        
        metrics['existence_accuracy'] = accuracy_score(exists_true_np, exists_pred_binary)
        metrics['existence_f1'] = f1_score(exists_true_np, exists_pred_binary)
        
        # AUC if we have both classes
        if len(np.unique(exists_true_np)) > 1:
            metrics['existence_auc'] = roc_auc_score(exists_true_np, exists_pred_np)
        
        # Peak value evaluation (masked)
        latency_mask = peak_mask[:, 0].detach().cpu().numpy().astype(bool)
        amplitude_mask = peak_mask[:, 1].detach().cpu().numpy().astype(bool)
        
        if np.any(latency_mask):
            lat_pred = peak_latency_pred[peak_mask[:, 0]].detach().cpu().numpy()
            lat_true = peak_latency_true[peak_mask[:, 0]].detach().cpu().numpy()
            
            metrics['latency_mae'] = mean_absolute_error(lat_true, lat_pred)
            metrics['latency_rmse'] = np.sqrt(mean_squared_error(lat_true, lat_pred))
            metrics['latency_r2'] = r2_score(lat_true, lat_pred)
            
            # Error histogram data
            latency_errors = lat_pred - lat_true
            metrics['latency_error_mean'] = np.mean(latency_errors)
            metrics['latency_error_std'] = np.std(latency_errors)
        
        if np.any(amplitude_mask):
            amp_pred = peak_amplitude_pred[peak_mask[:, 1]].detach().cpu().numpy()
            amp_true = peak_amplitude_true[peak_mask[:, 1]].detach().cpu().numpy()
            
            metrics['amplitude_mae'] = mean_absolute_error(amp_true, amp_pred)
            metrics['amplitude_rmse'] = np.sqrt(mean_squared_error(amp_true, amp_pred))
            metrics['amplitude_r2'] = r2_score(amp_true, amp_pred)
            
            # Error histogram data
            amplitude_errors = amp_pred - amp_true
            metrics['amplitude_error_mean'] = np.mean(amplitude_errors)
            metrics['amplitude_error_std'] = np.std(amplitude_errors)
        
        return metrics
    
    # ==================== TASK 3: CLASSIFICATION EVALUATION ====================
    
    def evaluate_classification(
        self,
        pred_class_logits: torch.Tensor,
        true_class: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation.
        
        Args:
            pred_class_logits: Predicted class logits [batch, n_classes]
            true_class: Ground truth class labels [batch]
            
        Returns:
            Dictionary of classification metrics
        """
        # Convert to numpy
        pred_probs = F.softmax(pred_class_logits, dim=1).detach().cpu().numpy()
        pred_classes = np.argmax(pred_probs, axis=1)
        true_classes = true_class.detach().cpu().numpy()
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(true_classes, pred_classes)
        metrics['balanced_accuracy'] = balanced_accuracy_score(true_classes, pred_classes)
        metrics['macro_f1'] = f1_score(true_classes, pred_classes, average='macro')
        metrics['micro_f1'] = f1_score(true_classes, pred_classes, average='micro')
        metrics['weighted_f1'] = f1_score(true_classes, pred_classes, average='weighted')
        
        # Per-class F1 scores
        f1_per_class = f1_score(true_classes, pred_classes, average=None)
        for i, class_name in enumerate(self.class_names):
            if i < len(f1_per_class):
                metrics[f'f1_{class_name.lower()}'] = f1_per_class[i]
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, pred_classes)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Class distribution analysis
        true_dist = np.bincount(true_classes, minlength=len(self.class_names))
        pred_dist = np.bincount(pred_classes, minlength=len(self.class_names))
        
        metrics['true_distribution'] = (true_dist / len(true_classes)).tolist()
        metrics['pred_distribution'] = (pred_dist / len(pred_classes)).tolist()
        
        # Classification report
        report = classification_report(
            true_classes, pred_classes,
            target_names=self.class_names,
            labels=list(range(len(self.class_names))),  # Explicitly specify all labels
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = report
        
        return metrics
    
    # ==================== TASK 4: THRESHOLD ESTIMATION ====================
    
    def evaluate_threshold_estimation(
        self,
        threshold_pred: torch.Tensor,
        threshold_true: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Comprehensive threshold estimation evaluation.
        
        Args:
            threshold_pred: Predicted thresholds [batch]
            threshold_true: Ground truth thresholds [batch]
            
        Returns:
            Dictionary of threshold estimation metrics
        """
        # Convert to numpy
        pred_np = threshold_pred.detach().cpu().numpy().flatten()
        true_np = threshold_true.detach().cpu().numpy().flatten()
        
        metrics = {}
        
        # Basic regression metrics
        metrics['mae'] = mean_absolute_error(true_np, pred_np)
        metrics['mse'] = mean_squared_error(true_np, pred_np)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(true_np, pred_np)
        
        # Log-scale MAE (for threshold values)
        log_true = np.log1p(np.maximum(true_np, 0))
        log_pred = np.log1p(np.maximum(pred_np, 0))
        metrics['log_mae'] = mean_absolute_error(log_true, log_pred)
        
        # Pearson correlation
        if np.std(pred_np) > 1e-8 and np.std(true_np) > 1e-8:
            corr, p_value = scipy.stats.pearsonr(true_np, pred_np)
            metrics['pearson_corr'] = corr
            metrics['pearson_p_value'] = p_value
        else:
            metrics['pearson_corr'] = 0.0
            metrics['pearson_p_value'] = 1.0
        
        # Error analysis
        errors = pred_np - true_np
        metrics['error_mean'] = np.mean(errors)
        metrics['error_std'] = np.std(errors)
        metrics['error_median'] = np.median(errors)
        
        # Percentile analysis
        metrics['error_25th'] = np.percentile(errors, 25)
        metrics['error_75th'] = np.percentile(errors, 75)
        
        return metrics
    
    # ==================== TASK 5: CLINICAL FAILURE FLAGS ====================
    
    def compute_failure_modes(
        self,
        peak_exists_pred: torch.Tensor,
        peak_exists_true: torch.Tensor,
        threshold_pred: torch.Tensor,
        threshold_true: torch.Tensor,
        pred_class: torch.Tensor,
        true_class: torch.Tensor
    ) -> Dict[str, int]:
        """
        Compute clinical failure mode flags.
        
        Args:
            peak_exists_pred: Peak existence predictions [batch]
            peak_exists_true: Ground truth peak existence [batch]
            threshold_pred: Predicted thresholds [batch]
            threshold_true: Ground truth thresholds [batch]
            pred_class: Predicted classes [batch]
            true_class: Ground truth classes [batch]
            
        Returns:
            Dictionary of failure mode counts
        """
        failures = {}
        
        # Convert to numpy
        exists_pred = (torch.sigmoid(peak_exists_pred) > 0.5).detach().cpu().numpy()
        exists_true = peak_exists_true.detach().cpu().numpy()
        thresh_pred = threshold_pred.detach().cpu().numpy().flatten()
        thresh_true = threshold_true.detach().cpu().numpy().flatten()
        pred_classes = pred_class.detach().cpu().numpy()
        true_classes = true_class.detach().cpu().numpy()
        
        # False peak detection (predicted peak when none exists)
        false_peaks = np.sum((exists_pred == 1) & (exists_true == 0))
        failures['false_peak_detected'] = int(false_peaks)
        
        # Missed peak detection (no peak predicted when one exists)
        missed_peaks = np.sum((exists_pred == 0) & (exists_true == 1))
        failures['missed_peak_detected'] = int(missed_peaks)
        
        # Threshold overestimation (> 15 dB error)
        threshold_overest = np.sum(
            (thresh_pred - thresh_true) > self.config['clinical_thresholds']['threshold_overestimate']
        )
        failures['threshold_overestimated'] = int(threshold_overest)
        
        # Severe threshold underestimation (< -15 dB error)
        threshold_underest = np.sum(
            (thresh_true - thresh_pred) > self.config['clinical_thresholds']['threshold_overestimate']
        )
        failures['threshold_underestimated'] = int(threshold_underest)
        
        # Class mismatch in severe/profound cases
        # Assuming classes 3, 4 are severe/profound hearing loss
        severe_classes = [3, 4]  # TOTAL, İTİK
        severe_mask = np.isin(true_classes, severe_classes)
        severe_mismatches = np.sum(
            severe_mask & (pred_classes != true_classes)
        )
        failures['severe_class_mismatch'] = int(severe_mismatches)
        
        # Normal misclassified as severe
        normal_class = 0  # NORMAL
        normal_as_severe = np.sum(
            (true_classes == normal_class) & np.isin(pred_classes, severe_classes)
        )
        failures['normal_as_severe'] = int(normal_as_severe)
        
        return failures
    
    # ==================== TASK 6: BATCH DIAGNOSTICS VISUALIZATION ====================
    
    def create_batch_diagnostics(
        self,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        batch_idx: int = 0,
        n_samples: int = None
    ) -> Dict[str, bytes]:
        """
        Create comprehensive diagnostic visualizations for a batch with enhanced features.
        
        Args:
            batch_data: Ground truth batch data
            model_outputs: Model predictions
            batch_idx: Batch index for naming
            n_samples: Number of samples to visualize
            
        Returns:
            Dictionary of plot bytes for logging
        """
        if n_samples is None:
            n_samples = min(self.config['waveform_samples'], batch_data['signal'].size(0))
        
        visualizations = {}
        
        # Check visualization configuration
        viz_config = self.config.get('visualization', {})
        plot_config = viz_config.get('plots', {})
        
        # 1. Enhanced signal reconstruction with clinical overlays
        if plot_config.get('signal_reconstruction', True):
            if viz_config.get('clinical_overlays', {}).get('enabled', True):
                fig = self._plot_signal_reconstruction_with_clinical_overlays(
                    batch_data, model_outputs, n_samples
                )
            else:
                fig = self._plot_signal_reconstruction(batch_data, model_outputs, n_samples)
            visualizations['signal_reconstruction'] = self._fig_to_bytes(fig)
            plt.close(fig)
        
        # 2. Peak prediction scatter plots
        if 'peak' in model_outputs and plot_config.get('peak_predictions', True):
            fig = self._plot_peak_predictions(batch_data, model_outputs)
            visualizations['peak_predictions'] = self._fig_to_bytes(fig)
            plt.close(fig)
        
        # 3. Classification confusion matrix
        if 'class' in model_outputs and plot_config.get('classification_matrix', True):
            fig = self._plot_classification_results(batch_data, model_outputs)
            visualizations['classification'] = self._fig_to_bytes(fig)
            plt.close(fig)
        
        # 4. Threshold prediction scatter
        if 'threshold' in model_outputs and plot_config.get('threshold_scatter', True):
            fig = self._plot_threshold_predictions(batch_data, model_outputs)
            visualizations['threshold_predictions'] = self._fig_to_bytes(fig)
            plt.close(fig)
        
        # 5. Error distributions
        if plot_config.get('error_distributions', True):
            fig = self._plot_error_distributions(batch_data, model_outputs)
            visualizations['error_distributions'] = self._fig_to_bytes(fig)
            plt.close(fig)
        
        return visualizations
    
    def _plot_signal_reconstruction(
        self, 
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        n_samples: int
    ) -> plt.Figure:
        """Plot signal reconstruction with peak annotations."""
        fig, axes = plt.subplots(n_samples, 1, figsize=(15, 3*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        true_signals = batch_data['signal'][:n_samples].detach().cpu().numpy()
        pred_signals = model_outputs['recon'][:n_samples].detach().cpu().numpy()
        
        time_axis = np.linspace(0, 10, true_signals.shape[-1])  # 10ms ABR
        
        for i in range(n_samples):
            true_sig = true_signals[i].squeeze()
            pred_sig = pred_signals[i].squeeze()
            
            # Plot signals
            axes[i].plot(time_axis, true_sig, 'b-', label='True Signal', linewidth=2, alpha=0.8)
            axes[i].plot(time_axis, pred_sig, 'r--', label='Predicted Signal', linewidth=2, alpha=0.8)
            
            # Add peak annotations
            if 'peak' in model_outputs and i < len(batch_data.get('v_peak', [])):
                self._add_peak_annotations(axes[i], batch_data, model_outputs, i, time_axis)
            
            # Add classification and threshold info
            title_parts = [f'Sample {i+1}']
            
            if 'class' in model_outputs:
                pred_class = torch.argmax(model_outputs['class'][i]).item()
                true_class = batch_data['target'][i].item()
                title_parts.append(f'Class: {self.class_names[true_class]} → {self.class_names[pred_class]}')
            
            if 'threshold' in model_outputs:
                pred_thresh = model_outputs['threshold'][i].item()
                title_parts.append(f'Threshold: {pred_thresh:.1f} dB')
            
            # Compute correlation
            corr = np.corrcoef(true_sig, pred_sig)[0, 1] if np.std(pred_sig) > 1e-8 else 0.0
            mse = np.mean((true_sig - pred_sig) ** 2)
            title_parts.append(f'Corr: {corr:.3f}, MSE: {mse:.4f}')
            
            axes[i].set_title(' | '.join(title_parts))
            axes[i].set_xlabel('Time (ms)')
            axes[i].set_ylabel('Amplitude (μV)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_signal_reconstruction_with_clinical_overlays(
        self, 
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        n_samples: int
    ) -> plt.Figure:
        """Plot signal reconstruction with enhanced clinical overlays."""
        fig, axes = plt.subplots(n_samples, 1, figsize=(15, 3*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        true_signals = batch_data['signal'][:n_samples].detach().cpu().numpy()
        pred_signals = model_outputs['recon'][:n_samples].detach().cpu().numpy()
        
        time_axis = np.linspace(0, 10, true_signals.shape[-1])  # 10ms ABR
        
        # Get overlay configuration
        overlay_config = self.config.get('visualization', {}).get('clinical_overlays', {})
        show_patient_id = overlay_config.get('show_patient_id', True)
        show_class_info = overlay_config.get('show_class_info', True)
        show_threshold_info = overlay_config.get('show_threshold_info', True)
        peak_markers = overlay_config.get('peak_markers', True)
        
        for i in range(n_samples):
            true_sig = true_signals[i].squeeze()
            pred_sig = pred_signals[i].squeeze()
            
            # Plot signals
            axes[i].plot(time_axis, true_sig, 'b-', label='True Signal', linewidth=2, alpha=0.8)
            axes[i].plot(time_axis, pred_sig, 'r--', label='Predicted Signal', linewidth=2, alpha=0.8)
            
            # Enhanced clinical overlays
            title_parts = []
            
            # Patient ID
            if show_patient_id and 'patient_ids' in batch_data:
                patient_id = batch_data['patient_ids'][i] if i < len(batch_data['patient_ids']) else f"P{i+1}"
                title_parts.append(f"Patient: {patient_id}")
            
            # Class information
            if show_class_info and 'class' in model_outputs and 'target' in batch_data:
                pred_class = torch.argmax(model_outputs['class'][i]).item()
                true_class = batch_data['target'][i].item()
                pred_class_name = self.class_names[pred_class] if pred_class < len(self.class_names) else f"Class{pred_class}"
                true_class_name = self.class_names[true_class] if true_class < len(self.class_names) else f"Class{true_class}"
                title_parts.append(f"Class: GT={true_class_name} / Pred={pred_class_name}")
            
            # Threshold information
            if show_threshold_info and 'threshold' in model_outputs:
                pred_thresh = model_outputs['threshold'][i].item()
                if 'threshold' in batch_data:
                    true_thresh = batch_data['threshold'][i].item()
                else:
                    # Use intensity as proxy
                    true_thresh = batch_data['static_params'][i, 1].item() * 100
                title_parts.append(f"Thr: GT={true_thresh:.1f} / Pred={pred_thresh:.1f} dB")
            
            # Peak annotations with enhanced markers
            if peak_markers and 'peak' in model_outputs and i < len(batch_data.get('v_peak', [])):
                self._add_enhanced_peak_annotations(axes[i], batch_data, model_outputs, i, time_axis)
            
            # Compute and display correlation
            corr = np.corrcoef(true_sig, pred_sig)[0, 1] if np.std(pred_sig) > 1e-8 else 0.0
            mse = np.mean((true_sig - pred_sig) ** 2)
            title_parts.append(f'Corr: {corr:.3f}, MSE: {mse:.4f}')
            
            # Set enhanced title
            if title_parts:
                axes[i].set_title(' | '.join(title_parts), fontsize=10)
            
            axes[i].set_xlabel('Time (ms)')
            axes[i].set_ylabel('Amplitude (μV)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _add_enhanced_peak_annotations(
        self, 
        ax: plt.Axes,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        sample_idx: int,
        time_axis: np.ndarray
    ):
        """Add enhanced peak annotations with clinical markers."""
        peak_outputs = model_outputs['peak']
        
        if len(peak_outputs) >= 3:  # existence, latency, amplitude
            # Predicted peak (red)
            pred_exists = torch.sigmoid(peak_outputs[0][sample_idx]).item() > 0.5
            pred_latency = peak_outputs[1][sample_idx].item()
            pred_amplitude = peak_outputs[2][sample_idx].item()
            
            if pred_exists and 0 <= pred_latency <= 10:
                ax.axvline(pred_latency, color='red', linestyle=':', alpha=0.8, linewidth=2,
                          label=f'Pred Peak: {pred_latency:.2f}ms')
                ax.plot(pred_latency, pred_amplitude, 'ro', markersize=10, alpha=0.8, 
                       markeredgecolor='darkred', markeredgewidth=2)
        
        # True peak (green) - enhanced visibility
        if 'v_peak' in batch_data and 'v_peak_mask' in batch_data:
            true_peak = batch_data['v_peak'][sample_idx]
            peak_mask = batch_data['v_peak_mask'][sample_idx]
            
            if peak_mask[0] and peak_mask[1]:  # Both latency and amplitude valid
                true_latency = true_peak[0].item()
                true_amplitude = true_peak[1].item()
                
                if 0 <= true_latency <= 10:
                    ax.axvline(true_latency, color='green', linestyle=':', alpha=0.8, linewidth=2,
                              label=f'True Peak: {true_latency:.2f}ms')
                    ax.plot(true_latency, true_amplitude, 'go', markersize=10, alpha=0.8,
                           markeredgecolor='darkgreen', markeredgewidth=2)
                    
                    # Add peak error annotation
                    if len(peak_outputs) >= 3:
                        pred_latency = peak_outputs[1][sample_idx].item()
                        latency_error = abs(pred_latency - true_latency)
                        if latency_error > 0.5:  # Clinical significance threshold
                            ax.annotate(f'ΔT: {latency_error:.2f}ms', 
                                      xy=(true_latency, true_amplitude),
                                      xytext=(true_latency + 1, true_amplitude + 0.1),
                                      arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7),
                                      fontsize=8, color='orange', weight='bold')
    
    def _plot_peak_predictions(
        self,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor]
    ) -> plt.Figure:
        """Plot peak prediction scatter plots."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        peak_outputs = model_outputs['peak']
        if len(peak_outputs) < 3:
            return fig
        
        # Extract data
        pred_latency = peak_outputs[1].detach().cpu().numpy().flatten()
        pred_amplitude = peak_outputs[2].detach().cpu().numpy().flatten()
        
        if 'v_peak' in batch_data and 'v_peak_mask' in batch_data:
            true_peaks = batch_data['v_peak'].detach().cpu().numpy()
            peak_masks = batch_data['v_peak_mask'].detach().cpu().numpy()
            
            # Latency scatter plot
            latency_mask = peak_masks[:, 0].astype(bool)
            if np.any(latency_mask):
                true_lat = true_peaks[latency_mask, 0]
                pred_lat = pred_latency[latency_mask]
                
                axes[0].scatter(true_lat, pred_lat, alpha=0.6, s=50)
                
                # Perfect prediction line
                lat_min = min(np.min(true_lat), np.min(pred_lat))
                lat_max = max(np.max(true_lat), np.max(pred_lat))
                axes[0].plot([lat_min, lat_max], [lat_min, lat_max], 'r--', alpha=0.8)
                
                # R² score
                r2 = r2_score(true_lat, pred_lat)
                axes[0].set_title(f'Peak Latency Prediction (R² = {r2:.3f})')
                axes[0].set_xlabel('True Latency (ms)')
                axes[0].set_ylabel('Predicted Latency (ms)')
                axes[0].grid(True, alpha=0.3)
            
            # Amplitude scatter plot
            amplitude_mask = peak_masks[:, 1].astype(bool)
            if np.any(amplitude_mask):
                true_amp = true_peaks[amplitude_mask, 1]
                pred_amp = pred_amplitude[amplitude_mask]
                
                axes[1].scatter(true_amp, pred_amp, alpha=0.6, s=50)
                
                # Perfect prediction line
                amp_min = min(np.min(true_amp), np.min(pred_amp))
                amp_max = max(np.max(true_amp), np.max(pred_amp))
                axes[1].plot([amp_min, amp_max], [amp_min, amp_max], 'r--', alpha=0.8)
                
                # R² score
                r2 = r2_score(true_amp, pred_amp)
                axes[1].set_title(f'Peak Amplitude Prediction (R² = {r2:.3f})')
                axes[1].set_xlabel('True Amplitude (μV)')
                axes[1].set_ylabel('Predicted Amplitude (μV)')
                axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_classification_results(
        self,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor]
    ) -> plt.Figure:
        """Plot classification confusion matrix."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        pred_classes = torch.argmax(model_outputs['class'], dim=1).detach().cpu().numpy()
        true_classes = batch_data['target'].detach().cpu().numpy()
        
        cm = confusion_matrix(true_classes, pred_classes)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        
        ax.set_title('Classification Confusion Matrix (Normalized)')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        
        return fig
    
    def _plot_threshold_predictions(
        self,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor]
    ) -> plt.Figure:
        """Plot threshold prediction scatter plot."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        pred_thresh = model_outputs['threshold'].detach().cpu().numpy().flatten()
        
        # Use intensity as proxy for true threshold if not available
        if 'threshold' in batch_data:
            true_thresh = batch_data['threshold'].detach().cpu().numpy().flatten()
        else:
            # Use static params intensity as proxy
            intensity = batch_data['static_params'][:, 1].detach().cpu().numpy()
            true_thresh = intensity * 100  # Scale to dB range
        
        ax.scatter(true_thresh, pred_thresh, alpha=0.6, s=50)
        
        # Perfect prediction line
        thresh_min = min(np.min(true_thresh), np.min(pred_thresh))
        thresh_max = max(np.max(true_thresh), np.max(pred_thresh))
        ax.plot([thresh_min, thresh_max], [thresh_min, thresh_max], 'r--', alpha=0.8)
        
        # R² score
        r2 = r2_score(true_thresh, pred_thresh)
        ax.set_title(f'Threshold Prediction (R² = {r2:.3f})')
        ax.set_xlabel('True Threshold (dB SPL)')
        ax.set_ylabel('Predicted Threshold (dB SPL)')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_error_distributions(
        self,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor]
    ) -> plt.Figure:
        """Plot error distribution analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Signal reconstruction errors
        true_signals = batch_data['signal'].detach().cpu().numpy()
        pred_signals = model_outputs['recon'].detach().cpu().numpy()
        
        mse_errors = np.mean((pred_signals - true_signals) ** 2, axis=(1, 2))
        mae_errors = np.mean(np.abs(pred_signals - true_signals), axis=(1, 2))
        
        axes[0, 0].hist(mse_errors, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(mse_errors), color='red', linestyle='--')
        axes[0, 0].set_title('Signal MSE Error Distribution')
        axes[0, 0].set_xlabel('MSE Error')
        axes[0, 0].set_ylabel('Count')
        
        axes[0, 1].hist(mae_errors, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(mae_errors), color='red', linestyle='--')
        axes[0, 1].set_title('Signal MAE Error Distribution')
        axes[0, 1].set_xlabel('MAE Error')
        axes[0, 1].set_ylabel('Count')
        
        # Peak errors (if available)
        if 'peak' in model_outputs and len(model_outputs['peak']) >= 3:
            peak_outputs = model_outputs['peak']
            if 'v_peak' in batch_data and 'v_peak_mask' in batch_data:
                true_peaks = batch_data['v_peak'].detach().cpu().numpy()
                peak_masks = batch_data['v_peak_mask'].detach().cpu().numpy()
                
                # Latency errors
                latency_mask = peak_masks[:, 0].astype(bool)
                if np.any(latency_mask):
                    pred_lat = peak_outputs[1].detach().cpu().numpy()[latency_mask]
                    true_lat = true_peaks[latency_mask, 0]
                    lat_errors = pred_lat - true_lat
                    
                    axes[1, 0].hist(lat_errors, bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 0].axvline(np.mean(lat_errors), color='red', linestyle='--')
                    axes[1, 0].set_title('Peak Latency Error Distribution')
                    axes[1, 0].set_xlabel('Latency Error (ms)')
                    axes[1, 0].set_ylabel('Count')
        
        # Threshold errors (if available)
        if 'threshold' in model_outputs:
            pred_thresh = model_outputs['threshold'].detach().cpu().numpy().flatten()
            if 'threshold' in batch_data:
                true_thresh = batch_data['threshold'].detach().cpu().numpy().flatten()
            else:
                intensity = batch_data['static_params'][:, 1].detach().cpu().numpy()
                true_thresh = intensity * 100
            
            thresh_errors = pred_thresh - true_thresh
            
            axes[1, 1].hist(thresh_errors, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(np.mean(thresh_errors), color='red', linestyle='--')
            axes[1, 1].set_title('Threshold Error Distribution')
            axes[1, 1].set_xlabel('Threshold Error (dB)')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        return fig
    
    def _fig_to_bytes(self, fig: plt.Figure) -> bytes:
        """Convert matplotlib figure to bytes."""
        buffer = BytesIO()
        fig.savefig(buffer, format=self.config['visualization']['save_format'], 
                   dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        buffer.seek(0)
        return buffer.getvalue()
    
    # ==================== TASK 7: AGGREGATE & SAVE RESULTS ====================
    
    def evaluate_batch(
        self,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        batch_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single batch.
        
        Args:
            batch_data: Ground truth batch data
            model_outputs: Model predictions
            batch_idx: Batch index
            
        Returns:
            Dictionary of batch evaluation results
        """
        batch_results = {
            'batch_idx': batch_idx,
            'batch_size': batch_data['signal'].size(0)
        }
        
        # Signal reconstruction evaluation
        if 'recon' in model_outputs:
            recon_metrics = self.evaluate_reconstruction(
                batch_data['signal'], model_outputs['recon']
            )
            batch_results['reconstruction'] = recon_metrics
        
        # Peak estimation evaluation
        if 'peak' in model_outputs and len(model_outputs['peak']) >= 3:
            peak_outputs = model_outputs['peak']
            if 'v_peak' in batch_data and 'v_peak_mask' in batch_data:
                # Extract peak existence from mask
                peak_exists_true = batch_data['v_peak_mask'].any(dim=1).float()
                
                peak_metrics = self.evaluate_peak_estimation(
                    peak_outputs[0],  # existence logits
                    peak_exists_true,
                    peak_outputs[1],  # latency
                    batch_data['v_peak'][:, 0],  # true latency
                    peak_outputs[2],  # amplitude
                    batch_data['v_peak'][:, 1],  # true amplitude
                    batch_data['v_peak_mask']
                )
                batch_results['peaks'] = peak_metrics
        
        # Classification evaluation
        if 'class' in model_outputs and 'target' in batch_data:
            class_metrics = self.evaluate_classification(
                model_outputs['class'], batch_data['target']
            )
            batch_results['classification'] = class_metrics
        
        # Threshold evaluation
        if 'threshold' in model_outputs:
            if 'threshold' in batch_data:
                true_thresh = batch_data['threshold']
            else:
                # Use intensity as proxy
                true_thresh = batch_data['static_params'][:, 1] * 100
            
            thresh_metrics = self.evaluate_threshold_estimation(
                model_outputs['threshold'], true_thresh
            )
            batch_results['threshold'] = thresh_metrics
        
        # Clinical failure modes
        if all(key in model_outputs for key in ['peak', 'class', 'threshold']):
            failures = self.compute_failure_modes(
                model_outputs['peak'][0] if len(model_outputs['peak']) > 0 else torch.zeros(batch_data['signal'].size(0)),
                batch_data['v_peak_mask'].any(dim=1).float() if 'v_peak_mask' in batch_data else torch.zeros(batch_data['signal'].size(0)),
                model_outputs['threshold'],
                batch_data.get('threshold', batch_data['static_params'][:, 1] * 100),
                torch.argmax(model_outputs['class'], dim=1),
                batch_data['target']
            )
            batch_results['failure_modes'] = failures
            
            # Update aggregate failure counts
            for key, value in failures.items():
                self.failure_modes[key] += value
        
        # Store batch results
        self.batch_results.append(batch_results)
        
        return batch_results
    
    def compute_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics across all evaluated batches with enhanced features."""
        if not self.batch_results:
            return {}
        
        aggregate = {
            'total_batches': len(self.batch_results),
            'total_samples': sum(r['batch_size'] for r in self.batch_results)
        }
        
        # Aggregate metrics by category
        categories = ['reconstruction', 'peaks', 'classification', 'threshold']
        
        for category in categories:
            category_metrics = []
            for batch_result in self.batch_results:
                if category in batch_result:
                    category_metrics.append(batch_result[category])
            
            if category_metrics:
                aggregate[category] = self._aggregate_metrics_with_bootstrap(category_metrics)
        
        # Add failure mode summary
        aggregate['failure_modes'] = dict(self.failure_modes)
        
        # Compute failure rates
        total_samples = aggregate['total_samples']
        if total_samples > 0:
            failure_rates = {}
            for mode, count in self.failure_modes.items():
                failure_rates[f'{mode}_rate'] = count / total_samples
            aggregate['failure_rates'] = failure_rates
        
        self.aggregate_metrics = aggregate
        return aggregate
    
    def compute_bootstrap_ci(
        self, 
        metric_values: List[float], 
        n_samples: int = 500, 
        ci_percentile: float = 95
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence intervals for a metric.
        
        Args:
            metric_values: List of metric values
            n_samples: Number of bootstrap samples
            ci_percentile: Confidence interval percentile
            
        Returns:
            Tuple of (mean, lower_ci, upper_ci)
        """
        if not metric_values:
            return 0.0, 0.0, 0.0
        
        # Bootstrap sampling
        bootstrap_means = []
        for _ in range(n_samples):
            sample = np.random.choice(metric_values, size=len(metric_values), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Compute confidence intervals
        alpha = (100 - ci_percentile) / 2
        lower_ci = np.percentile(bootstrap_means, alpha)
        upper_ci = np.percentile(bootstrap_means, 100 - alpha)
        mean_val = np.mean(metric_values)
        
        return mean_val, lower_ci, upper_ci
    
    def _aggregate_metrics_with_bootstrap(self, metrics_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics with bootstrap confidence intervals."""
        if not metrics_list:
            return {}
        
        # Check if bootstrap is enabled
        bootstrap_config = self.config.get('bootstrap', {})
        use_bootstrap = bootstrap_config.get('enabled', False)
        n_samples = bootstrap_config.get('n_samples', 500)
        ci_percentile = bootstrap_config.get('ci_percentile', 95)
        
        # Get all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        aggregated = {}
        for key in all_keys:
            values = []
            for metrics in metrics_list:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        values.append(value)
            
            if values:
                if use_bootstrap:
                    mean_val, lower_ci, upper_ci = self.compute_bootstrap_ci(
                        values, n_samples, ci_percentile
                    )
                    aggregated[f'{key}_mean'] = mean_val
                    aggregated[f'{key}_lower_ci'] = lower_ci
                    aggregated[f'{key}_upper_ci'] = upper_ci
                    aggregated[f'{key}_ci_width'] = upper_ci - lower_ci
                else:
                    aggregated[f'{key}_mean'] = np.mean(values)
                    aggregated[f'{key}_std'] = np.std(values)
                
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
                aggregated[f'{key}_median'] = np.median(values)
        
        return aggregated
    
    def _aggregate_metrics_list(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Aggregate a list of metric dictionaries with optional bootstrap CI."""
        return self._aggregate_metrics_with_bootstrap(metrics_list)
    
    def save_results(self, filename_prefix: str = "evaluation") -> Dict[str, str]:
        """
        Save all evaluation results to files.
        
        Args:
            filename_prefix: Prefix for saved files
            
        Returns:
            Dictionary of saved file paths
        """
        saved_files = {}
        
        # Save aggregate metrics as JSON
        if self.aggregate_metrics:
            json_path = self.save_dir / "data" / f"{filename_prefix}_metrics.json"
            with open(json_path, 'w') as f:
                json.dump(self.aggregate_metrics, f, indent=2, default=str)
            saved_files['metrics_json'] = str(json_path)
        
        # Save batch results as JSON
        if self.batch_results:
            batch_json_path = self.save_dir / "data" / f"{filename_prefix}_batch_results.json"
            with open(batch_json_path, 'w') as f:
                json.dump(self.batch_results, f, indent=2, default=str)
            saved_files['batch_results_json'] = str(batch_json_path)
        
        # Save summary as CSV
        if self.aggregate_metrics:
            csv_path = self.save_dir / "data" / f"{filename_prefix}_summary.csv"
            self._save_summary_csv(csv_path)
            saved_files['summary_csv'] = str(csv_path)
        
        return saved_files
    
    def _save_summary_csv(self, csv_path: Path):
        """Save summary metrics as CSV."""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'Metric', 'Value'])
            
            for category, metrics in self.aggregate_metrics.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            writer.writerow([category, metric, value])
                elif isinstance(metrics, (int, float)):
                    writer.writerow(['aggregate', category, metrics])
    
    def log_to_tensorboard(self, writer, global_step: int):
        """Log results to TensorBoard."""
        if not TENSORBOARD_AVAILABLE or not self.aggregate_metrics:
            return
        
        # Log scalar metrics
        for category, metrics in self.aggregate_metrics.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        writer.add_scalar(f'eval/{category}/{metric}', value, global_step)
        
        # Log failure rates
        if 'failure_rates' in self.aggregate_metrics:
            for mode, rate in self.aggregate_metrics['failure_rates'].items():
                writer.add_scalar(f'eval/failure_rates/{mode}', rate, global_step)
    
    def log_to_wandb(self, step: int):
        """Log results to Weights & Biases."""
        if not WANDB_AVAILABLE or not self.aggregate_metrics:
            return
        
        log_dict = {'eval_step': step}
        
        # Flatten metrics for wandb
        for category, metrics in self.aggregate_metrics.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        log_dict[f'eval_{category}_{metric}'] = value
        
        wandb.log(log_dict)
    
    def print_summary(self):
        """Print comprehensive evaluation summary to console."""
        if not self.aggregate_metrics:
            print("No evaluation results available.")
            return
        
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE ABR MODEL EVALUATION SUMMARY")
        print("="*80)
        
        print(f"📊 Dataset: {self.aggregate_metrics['total_samples']} samples, {self.aggregate_metrics['total_batches']} batches")
        
        # Signal reconstruction
        if 'reconstruction' in self.aggregate_metrics:
            recon = self.aggregate_metrics['reconstruction']
            print(f"\n📈 SIGNAL RECONSTRUCTION:")
            print(f"   MSE: {recon.get('mse_mean', 0):.6f} ± {recon.get('mse_std', 0):.6f}")
            print(f"   MAE: {recon.get('mae_mean', 0):.6f} ± {recon.get('mae_std', 0):.6f}")
            print(f"   SNR: {recon.get('snr_mean', 0):.2f} ± {recon.get('snr_std', 0):.2f} dB")
            print(f"   Correlation: {recon.get('pearson_corr_mean', 0):.4f} ± {recon.get('pearson_corr_std', 0):.4f}")
        
        # Peak estimation
        if 'peaks' in self.aggregate_metrics:
            peaks = self.aggregate_metrics['peaks']
            print(f"\n🎯 PEAK ESTIMATION:")
            print(f"   Existence F1: {peaks.get('existence_f1_mean', 0):.4f} ± {peaks.get('existence_f1_std', 0):.4f}")
            print(f"   Latency MAE: {peaks.get('latency_mae_mean', 0):.4f} ± {peaks.get('latency_mae_std', 0):.4f} ms")
            print(f"   Amplitude MAE: {peaks.get('amplitude_mae_mean', 0):.4f} ± {peaks.get('amplitude_mae_std', 0):.4f} μV")
        
        # Classification
        if 'classification' in self.aggregate_metrics:
            classif = self.aggregate_metrics['classification']
            print(f"\n🏷️  CLASSIFICATION:")
            print(f"   Accuracy: {classif.get('accuracy_mean', 0):.4f} ± {classif.get('accuracy_std', 0):.4f}")
            print(f"   Macro F1: {classif.get('macro_f1_mean', 0):.4f} ± {classif.get('macro_f1_std', 0):.4f}")
            print(f"   Balanced Accuracy: {classif.get('balanced_accuracy_mean', 0):.4f} ± {classif.get('balanced_accuracy_std', 0):.4f}")
        
        # Threshold estimation
        if 'threshold' in self.aggregate_metrics:
            thresh = self.aggregate_metrics['threshold']
            print(f"\n📏 THRESHOLD ESTIMATION:")
            print(f"   MAE: {thresh.get('mae_mean', 0):.2f} ± {thresh.get('mae_std', 0):.2f} dB")
            print(f"   RMSE: {thresh.get('rmse_mean', 0):.2f} ± {thresh.get('rmse_std', 0):.2f} dB")
            print(f"   R²: {thresh.get('r2_mean', 0):.4f} ± {thresh.get('r2_std', 0):.4f}")
        
        # Clinical failure modes
        if 'failure_modes' in self.aggregate_metrics:
            failures = self.aggregate_metrics['failure_modes']
            rates = self.aggregate_metrics.get('failure_rates', {})
            print(f"\n⚠️  CLINICAL FAILURE MODES:")
            print(f"   False peaks detected: {failures.get('false_peak_detected', 0)} ({rates.get('false_peak_detected_rate', 0)*100:.2f}%)")
            print(f"   Missed peaks: {failures.get('missed_peak_detected', 0)} ({rates.get('missed_peak_detected_rate', 0)*100:.2f}%)")
            print(f"   Threshold overestimated: {failures.get('threshold_overestimated', 0)} ({rates.get('threshold_overestimated_rate', 0)*100:.2f}%)")
            print(f"   Severe class mismatches: {failures.get('severe_class_mismatch', 0)} ({rates.get('severe_class_mismatch_rate', 0)*100:.2f}%)")
        
        print("="*80)

    def create_summary_table(self) -> str:
        """
        Create a comprehensive summary table in CSV format.
        
        Returns:
            Path to saved summary table
        """
        if not self.aggregate_metrics:
            return ""
        
        # Prepare summary data
        summary_data = []
        
        # Check if bootstrap is enabled
        bootstrap_config = self.config.get('bootstrap', {})
        use_bootstrap = bootstrap_config.get('enabled', False)
        ci_percentile = bootstrap_config.get('ci_percentile', 95)
        
        # Define metric mappings
        metric_mappings = {
            'reconstruction': {
                'mse_mean': ('MSE', '—'),
                'mae_mean': ('MAE', '—'),
                'snr_mean': ('SNR', 'dB'),
                'pearson_corr_mean': ('Correlation', '—'),
                'spectral_mse_mean': ('Spectral MSE', '—')
            },
            'peaks': {
                'existence_f1_mean': ('Peak Existence F1', '—'),
                'latency_mae_mean': ('Latency MAE', 'ms'),
                'amplitude_mae_mean': ('Amplitude MAE', 'μV'),
                'latency_r2_mean': ('Latency R²', '—'),
                'amplitude_r2_mean': ('Amplitude R²', '—')
            },
            'classification': {
                'accuracy_mean': ('Accuracy', '—'),
                'balanced_accuracy_mean': ('Balanced Accuracy', '—'),
                'macro_f1_mean': ('Macro F1', '—'),
                'weighted_f1_mean': ('Weighted F1', '—')
            },
            'threshold': {
                'mae_mean': ('MAE', 'dB'),
                'rmse_mean': ('RMSE', 'dB'),
                'r2_mean': ('R²', '—'),
                'log_mae_mean': ('Log-scale MAE', '—'),
                'pearson_corr_mean': ('Correlation', '—')
            }
        }
        
        # Build summary table
        for task, metrics in metric_mappings.items():
            if task in self.aggregate_metrics:
                task_data = self.aggregate_metrics[task]
                
                for metric_key, (metric_name, unit) in metrics.items():
                    if metric_key in task_data:
                        mean_val = task_data[metric_key]
                        
                        if use_bootstrap:
                            lower_key = metric_key.replace('_mean', '_lower_ci')
                            upper_key = metric_key.replace('_mean', '_upper_ci')
                            lower_ci = task_data.get(lower_key, mean_val)
                            upper_ci = task_data.get(upper_key, mean_val)
                            
                            summary_data.append({
                                'Task': task.title(),
                                'Metric': metric_name,
                                'Mean': f"{mean_val:.4f}",
                                'Lower_CI': f"{lower_ci:.4f}",
                                'Upper_CI': f"{upper_ci:.4f}",
                                'CI_Width': f"{upper_ci - lower_ci:.4f}",
                                'Unit': unit
                            })
                        else:
                            std_key = metric_key.replace('_mean', '_std')
                            std_val = task_data.get(std_key, 0)
                            
                            summary_data.append({
                                'Task': task.title(),
                                'Metric': metric_name,
                                'Mean': f"{mean_val:.4f}",
                                'Std': f"{std_val:.4f}",
                                'Unit': unit
                            })
        
        # Save summary table
        summary_path = self.save_dir / "data" / "summary_table.csv"
        
        if summary_data:
            if PANDAS_AVAILABLE:
                import pandas as pd
                df = pd.DataFrame(summary_data)
                df.to_csv(summary_path, index=False)
            else:
                # Fallback to basic CSV writing
                with open(summary_path, 'w', newline='') as f:
                    if summary_data:
                        fieldnames = summary_data[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(summary_data)
        
        return str(summary_path)
    
    def create_quantile_error_visualizations(
        self,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, bytes]:
        """
        Create quantile and error range visualizations.
        
        Args:
            batch_data: Ground truth batch data
            model_outputs: Model predictions
            
        Returns:
            Dictionary of visualization bytes
        """
        visualizations = {}
        
        # 1. Threshold Error vs Ground Truth (binned)
        if 'threshold' in model_outputs:
            fig = self._plot_threshold_error_by_range(batch_data, model_outputs)
            visualizations['threshold_error_by_range'] = self._fig_to_bytes(fig)
            plt.close(fig)
        
        # 2. Peak Latency Error per Class
        if 'peak' in model_outputs and len(model_outputs['peak']) >= 3:
            fig = self._plot_peak_error_by_class(batch_data, model_outputs)
            visualizations['peak_error_by_class'] = self._fig_to_bytes(fig)
            plt.close(fig)
        
        # 3. Signal Quality by Class (boxplot)
        if 'recon' in model_outputs:
            fig = self._plot_signal_quality_by_class(batch_data, model_outputs)
            visualizations['signal_quality_by_class'] = self._fig_to_bytes(fig)
            plt.close(fig)
        
        return visualizations
    
    def _plot_threshold_error_by_range(
        self,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor]
    ) -> plt.Figure:
        """Plot threshold prediction errors binned by ground truth ranges."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Extract data
        pred_thresh = model_outputs['threshold'].detach().cpu().numpy().flatten()
        
        if 'threshold' in batch_data:
            true_thresh = batch_data['threshold'].detach().cpu().numpy().flatten()
        else:
            # Use intensity as proxy
            intensity = batch_data['static_params'][:, 1].detach().cpu().numpy()
            true_thresh = intensity * 100
        
        # Compute errors
        errors = pred_thresh - true_thresh
        
        # Create threshold bins
        thresh_bins = [(0, 25), (25, 40), (40, 55), (55, 70), (70, 90), (90, 120)]
        bin_labels = ['Normal\n(0-25)', 'Mild\n(25-40)', 'Moderate\n(40-55)', 
                     'Mod-Severe\n(55-70)', 'Severe\n(70-90)', 'Profound\n(90-120)']
        
        # Group errors by bins
        binned_errors = []
        bin_names = []
        
        for (min_thresh, max_thresh), label in zip(thresh_bins, bin_labels):
            mask = (true_thresh >= min_thresh) & (true_thresh < max_thresh)
            if np.any(mask):
                binned_errors.append(errors[mask])
                bin_names.append(label)
        
        # Create boxplot
        if binned_errors:
            bp = ax.boxplot(binned_errors, labels=bin_names, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Clinical Significance (+15 dB)')
        ax.axhline(y=-15, color='red', linestyle='--', alpha=0.7, label='Clinical Significance (-15 dB)')
        
        ax.set_title('Threshold Prediction Error by Ground Truth Range')
        ax.set_xlabel('Ground Truth Threshold Range (dB SPL)')
        ax.set_ylabel('Prediction Error (dB)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_peak_error_by_class(
        self,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor]
    ) -> plt.Figure:
        """Plot peak latency errors by hearing loss class."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        peak_outputs = model_outputs['peak']
        if len(peak_outputs) < 3:
            return fig
        
        # Extract data
        pred_latency = peak_outputs[1].detach().cpu().numpy().flatten()
        
        if 'v_peak' in batch_data and 'v_peak_mask' in batch_data:
            true_peaks = batch_data['v_peak'].detach().cpu().numpy()
            peak_masks = batch_data['v_peak_mask'].detach().cpu().numpy()
            true_classes = batch_data['target'].detach().cpu().numpy()
            
            # Filter for valid peaks
            latency_mask = peak_masks[:, 0].astype(bool)
            if np.any(latency_mask):
                valid_pred_lat = pred_latency[latency_mask]
                valid_true_lat = true_peaks[latency_mask, 0]
                valid_classes = true_classes[latency_mask]
                
                # Compute errors
                latency_errors = valid_pred_lat - valid_true_lat
                
                # Group by class
                class_errors = []
                class_labels = []
                
                for class_idx, class_name in enumerate(self.class_names):
                    class_mask = valid_classes == class_idx
                    if np.any(class_mask):
                        class_errors.append(latency_errors[class_mask])
                        class_labels.append(class_name)
                
                # Create violin plot
                if class_errors:
                    positions = range(len(class_errors))
                    parts = ax.violinplot(class_errors, positions=positions, showmeans=True, showmedians=True)
                    
                    # Color violin plots
                    colors = plt.cm.Set3(np.linspace(0, 1, len(parts['bodies'])))
                    for pc, color in zip(parts['bodies'], colors):
                        pc.set_facecolor(color)
                        pc.set_alpha(0.7)
                    
                    ax.set_xticks(positions)
                    ax.set_xticklabels(class_labels, rotation=45)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Clinical Tolerance (+0.5 ms)')
        ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7, label='Clinical Tolerance (-0.5 ms)')
        
        ax.set_title('Peak Latency Prediction Error by Hearing Loss Class')
        ax.set_xlabel('Hearing Loss Class')
        ax.set_ylabel('Latency Prediction Error (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_signal_quality_by_class(
        self,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor]
    ) -> plt.Figure:
        """Plot signal reconstruction quality metrics by class."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        true_signals = batch_data['signal'].detach().cpu().numpy()
        pred_signals = model_outputs['recon'].detach().cpu().numpy()
        true_classes = batch_data['target'].detach().cpu().numpy()
        
        # Compute per-sample metrics
        mse_per_sample = np.mean((pred_signals - true_signals) ** 2, axis=(1, 2))
        mae_per_sample = np.mean(np.abs(pred_signals - true_signals), axis=(1, 2))
        
        # Group by class
        class_mse = []
        class_mae = []
        class_labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = true_classes == class_idx
            if np.any(class_mask):
                class_mse.append(mse_per_sample[class_mask])
                class_mae.append(mae_per_sample[class_mask])
                class_labels.append(class_name)
        
        # MSE boxplot
        if class_mse:
            bp1 = axes[0].boxplot(class_mse, labels=class_labels, patch_artist=True)
            colors = plt.cm.viridis(np.linspace(0, 1, len(bp1['boxes'])))
            for patch, color in zip(bp1['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        axes[0].set_title('Signal Reconstruction MSE by Class')
        axes[0].set_xlabel('Hearing Loss Class')
        axes[0].set_ylabel('MSE')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # MAE boxplot
        if class_mae:
            bp2 = axes[1].boxplot(class_mae, labels=class_labels, patch_artist=True)
            colors = plt.cm.plasma(np.linspace(0, 1, len(bp2['boxes'])))
            for patch, color in zip(bp2['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        axes[1].set_title('Signal Reconstruction MAE by Class')
        axes[1].set_xlabel('Hearing Loss Class')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig 

    def create_diagnostic_cards(
        self,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        batch_idx: int = 0,
        n_samples: int = None
    ) -> Dict[str, bytes]:
        """
        Create multi-panel diagnostic cards for comprehensive patient assessment.
        
        Args:
            batch_data: Ground truth batch data
            model_outputs: Model predictions
            batch_idx: Batch index
            n_samples: Number of diagnostic cards to create
            
        Returns:
            Dictionary of diagnostic card bytes
        """
        if n_samples is None:
            n_samples = min(3, batch_data['signal'].size(0))
        
        cards = {}
        
        # Get configuration
        card_config = self.config.get('visualization', {}).get('diagnostic_cards', {})
        layout = card_config.get('layout', '2x2')
        include_text = card_config.get('include_text_overlay', True)
        card_figsize = card_config.get('card_figsize', [12, 10])
        
        for sample_idx in range(n_samples):
            # Create patient ID
            if 'patient_ids' in batch_data and sample_idx < len(batch_data['patient_ids']):
                patient_id = batch_data['patient_ids'][sample_idx]
            else:
                patient_id = f"B{batch_idx}_S{sample_idx}"
            
            # Create diagnostic card
            fig = self._create_single_diagnostic_card(
                batch_data, model_outputs, sample_idx, patient_id, 
                layout, include_text, card_figsize
            )
            
            cards[f'patient_{patient_id}_card'] = self._fig_to_bytes(fig)
            plt.close(fig)
        
        return cards
    
    def _create_single_diagnostic_card(
        self,
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        sample_idx: int,
        patient_id: str,
        layout: str = '2x2',
        include_text: bool = True,
        figsize: List[int] = [12, 10]
    ) -> plt.Figure:
        """Create a single diagnostic card with multiple panels."""
        fig = plt.figure(figsize=figsize)
        
        if layout == '2x2':
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        else:
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)  # Default to 2x2
        
        # Panel 1: Signal reconstruction
        ax1 = fig.add_subplot(gs[0, :])  # Top row, full width
        self._add_signal_panel(ax1, batch_data, model_outputs, sample_idx)
        
        # Panel 2: Peak analysis
        ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
        self._add_peak_panel(ax2, batch_data, model_outputs, sample_idx)
        
        # Panel 3: Threshold analysis
        ax3 = fig.add_subplot(gs[1, 1])  # Bottom right
        self._add_threshold_panel(ax3, batch_data, model_outputs, sample_idx)
        
        # Add overall title and patient information
        if include_text:
            self._add_card_header(fig, batch_data, model_outputs, sample_idx, patient_id)
        
        return fig
    
    def _add_signal_panel(
        self, 
        ax: plt.Axes, 
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        sample_idx: int
    ):
        """Add signal reconstruction panel to diagnostic card."""
        true_signal = batch_data['signal'][sample_idx].squeeze().detach().cpu().numpy()
        pred_signal = model_outputs['recon'][sample_idx].squeeze().detach().cpu().numpy()
        
        time_axis = np.linspace(0, 10, len(true_signal))
        
        ax.plot(time_axis, true_signal, 'b-', label='True Signal', linewidth=2, alpha=0.8)
        ax.plot(time_axis, pred_signal, 'r--', label='Predicted Signal', linewidth=2, alpha=0.8)
        
        # Add peak markers
        self._add_enhanced_peak_annotations(ax, batch_data, model_outputs, sample_idx, time_axis)
        
        # Compute correlation
        corr = np.corrcoef(true_signal, pred_signal)[0, 1] if np.std(pred_signal) > 1e-8 else 0.0
        mse = np.mean((true_signal - pred_signal) ** 2)
        
        ax.set_title(f'Signal Reconstruction (Corr: {corr:.3f}, MSE: {mse:.4f})')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (μV)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _add_peak_panel(
        self, 
        ax: plt.Axes, 
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        sample_idx: int
    ):
        """Add peak analysis panel to diagnostic card."""
        if 'peak' in model_outputs and len(model_outputs['peak']) >= 3:
            peak_outputs = model_outputs['peak']
            
            # Extract peak data
            pred_exists = torch.sigmoid(peak_outputs[0][sample_idx]).item()
            pred_latency = peak_outputs[1][sample_idx].item()
            pred_amplitude = peak_outputs[2][sample_idx].item()
            
            # True peak data
            if 'v_peak' in batch_data and 'v_peak_mask' in batch_data:
                true_peak = batch_data['v_peak'][sample_idx].detach().cpu().numpy()
                peak_mask = batch_data['v_peak_mask'][sample_idx].detach().cpu().numpy()
                
                if peak_mask[0] and peak_mask[1]:
                    true_latency = true_peak[0]
                    true_amplitude = true_peak[1]
                    
                    # Create scatter plot
                    ax.scatter([true_latency], [true_amplitude], c='green', s=100, 
                              alpha=0.8, label='True Peak', marker='o', edgecolors='darkgreen')
                    ax.scatter([pred_latency], [pred_amplitude], c='red', s=100, 
                              alpha=0.8, label='Predicted Peak', marker='s', edgecolors='darkred')
                    
                    # Add error lines
                    ax.plot([true_latency, pred_latency], [true_amplitude, pred_amplitude], 
                           'k--', alpha=0.5, linewidth=1)
                    
                    # Calculate errors
                    lat_error = abs(pred_latency - true_latency)
                    amp_error = abs(pred_amplitude - true_amplitude)
                    
                    ax.set_title(f'Peak Analysis\nΔLat: {lat_error:.2f}ms, ΔAmp: {amp_error:.3f}μV')
                else:
                    ax.text(0.5, 0.5, 'No Valid Peak Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title('Peak Analysis - No Data')
            else:
                ax.scatter([pred_latency], [pred_amplitude], c='red', s=100, 
                          alpha=0.8, label='Predicted Peak', marker='s')
                ax.set_title(f'Peak Analysis\nPred: {pred_latency:.2f}ms, {pred_amplitude:.3f}μV')
            
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Amplitude (μV)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Peak Analysis\nNot Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def _add_threshold_panel(
        self, 
        ax: plt.Axes, 
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        sample_idx: int
    ):
        """Add threshold analysis panel to diagnostic card."""
        if 'threshold' in model_outputs:
            pred_thresh = model_outputs['threshold'][sample_idx].item()
            
            if 'threshold' in batch_data:
                true_thresh = batch_data['threshold'][sample_idx].item()
            else:
                # Use intensity as proxy
                true_thresh = batch_data['static_params'][sample_idx, 1].item() * 100
            
            # Create bar plot
            categories = ['True Threshold', 'Predicted Threshold']
            values = [true_thresh, pred_thresh]
            colors = ['green', 'red']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f} dB', ha='center', va='bottom', fontweight='bold')
            
            # Add error information
            error = abs(pred_thresh - true_thresh)
            ax.text(0.5, 0.8, f'Error: {error:.1f} dB', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Add clinical significance line
            if error > 15:
                ax.text(0.5, 0.6, 'CLINICALLY SIGNIFICANT', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       color='red', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            
            ax.set_title('Threshold Analysis')
            ax.set_ylabel('Threshold (dB SPL)')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Threshold Analysis\nNot Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def _add_card_header(
        self, 
        fig: plt.Figure, 
        batch_data: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        sample_idx: int,
        patient_id: str
    ):
        """Add header information to diagnostic card."""
        header_info = []
        
        # Patient ID
        header_info.append(f"Patient ID: {patient_id}")
        
        # Classification info
        if 'class' in model_outputs and 'target' in batch_data:
            pred_class = torch.argmax(model_outputs['class'][sample_idx]).item()
            true_class = batch_data['target'][sample_idx].item()
            pred_class_name = self.class_names[pred_class] if pred_class < len(self.class_names) else f"Class{pred_class}"
            true_class_name = self.class_names[true_class] if true_class < len(self.class_names) else f"Class{true_class}"
            header_info.append(f"Classification: {true_class_name} → {pred_class_name}")
        
        # Peak existence
        if 'peak' in model_outputs and len(model_outputs['peak']) >= 3:
            pred_exists = torch.sigmoid(model_outputs['peak'][0][sample_idx]).item() > 0.5
            if 'v_peak_mask' in batch_data:
                true_exists = batch_data['v_peak_mask'][sample_idx].any().item()
                header_info.append(f"Peak Detected: GT={true_exists}, Pred={pred_exists}")
            else:
                header_info.append(f"Peak Detected: Pred={pred_exists}")
        
        # Add header text
        header_text = " | ".join(header_info)
        fig.suptitle(header_text, fontsize=14, fontweight='bold', y=0.95) 

def create_evaluation_config() -> Dict[str, Any]:
    """Create default evaluation configuration with enhanced features."""
    return {
        'dtw': DTW_AVAILABLE,
        'fft_mse': True,
        'waveform_samples': 5,
        'classification_metrics': ['accuracy', 'f1_macro', 'confusion_matrix'],
        'clinical_thresholds': {
            'threshold_overestimate': 15.0,
            'peak_latency_tolerance': 0.5,
            'peak_amplitude_tolerance': 0.1
        },
        'bootstrap': {
            'enabled': False,
            'n_samples': 500,
            'ci_percentile': 95
        },
        'visualization': {
            'figsize': (15, 10),
            'dpi': 150,
            'save_format': 'png',
            'plots': {
                'signal_reconstruction': True,
                'peak_predictions': True,
                'classification_matrix': True,
                'threshold_scatter': True,
                'error_distributions': True,
                'clinical_overlays': True,
                'diagnostic_cards': True,
                'quantile_analysis': True
            },
            'clinical_overlays': {
                'enabled': True,
                'show_patient_id': True,
                'show_class_info': True,
                'show_threshold_info': True,
                'peak_markers': True
            },
            'diagnostic_cards': {
                'layout': '2x2',
                'include_text_overlay': True,
                'card_figsize': [12, 10]
            }
        },
        'save_dir': 'outputs/evaluation'
    }