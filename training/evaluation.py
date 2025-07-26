#!/usr/bin/env python3
"""
Enhanced ABR Model Evaluation

Comprehensive evaluation utilities for the ProfessionalHierarchicalUNet model
including multi-task metrics, clinical evaluation, and visualization.

Author: AI Assistant
Date: January 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    balanced_accuracy_score, precision_recall_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import label_binarize
import joblib
import os
import scipy.stats
try:
    from fastdtw import fastdtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False


class ABREvaluator:
    """
    Comprehensive evaluator for ABR model with multi-task metrics.
    Supports both ProfessionalHierarchicalUNet and OptimizedHierarchicalUNet.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: Optional[List[str]] = None,
        output_dir: str = 'evaluation_results',
        model_type: str = 'auto'
    ):
        """
        Initialize ABR evaluator.
        
        Args:
            model: Trained ABR model
            device: Device for evaluation
            class_names: List of class names
            output_dir: Directory to save evaluation results
            model_type: Type of model ('professional', 'optimized', 'auto')
        """
        self.model = model
        self.device = device
        self.class_names = class_names or ["NORMAL", "NÖROPATİ", "SNİK", "TOTAL", "İTİK"]
        self.output_dir = output_dir
        
        # Detect model type if auto
        if model_type == 'auto':
            self.model_type = self._detect_model_type()
        else:
            self.model_type = model_type
        
        print(f"🔍 Detected model type: {self.model_type}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.reset_metrics()
    
    def _detect_model_type(self) -> str:
        """Detect model type from model attributes."""
        if hasattr(self.model, 'get_model_info'):
            try:
                model_info = self.model.get_model_info()
                if 'optimized' in model_info.get('model_name', '').lower():
                    return 'optimized'
            except:
                pass
        
        # Check for optimized model specific attributes
        optimized_indicators = [
            'use_multi_scale_attention',
            'use_task_specific_extractors',
            'enable_joint_generation'
        ]
        
        for indicator in optimized_indicators:
            if hasattr(self.model, indicator) and getattr(self.model, indicator, False):
                return 'optimized'
        
        return 'professional'
    
    def reset_metrics(self):
        """Reset all accumulated metrics."""
        self.predictions = {
            'class': [],
            'peak_exists': [],
            'peak_latency': [],
            'peak_amplitude': [],
            'threshold': [],
            'signal': [],
            'static_params': []  # For joint generation in optimized model
        }
        
        self.targets = {
            'class': [],
            'peak_exists': [],
            'peak_latency': [],
            'peak_amplitude': [],
            'threshold': [],
            'signal': [],
            'v_peak_mask': [],
            'static_params': []  # For joint generation
        }
        
        self.losses = defaultdict(list)
        self.sample_info = []
    
    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        compute_loss: bool = True,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single batch.
        
        Args:
            batch: Input batch
            compute_loss: Whether to compute loss
            loss_fn: Loss function for computing loss
            
        Returns:
            Batch metrics dictionary
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(
                x=batch['signal'],
                static_params=batch['static_params']
            )
            
            # Handle different output formats based on model type
            if self.model_type == 'optimized':
                # OptimizedHierarchicalUNet outputs
                class_key = 'class' if 'class' in outputs else 'classification_logits'
                signal_key = 'recon' if 'recon' in outputs else 'signal'
                
                class_pred = torch.argmax(outputs[class_key], dim=1)
                signal_pred = outputs[signal_key]
                
                # Handle peak outputs
                if 'peak' in outputs:
                    peak_outputs = outputs['peak']
                    if len(peak_outputs) >= 3:
                        peak_exists, peak_latency, peak_amplitude = peak_outputs[:3]
                    else:
                        peak_exists = peak_latency = peak_amplitude = torch.zeros_like(batch['target']).float()
                else:
                    peak_exists = peak_latency = peak_amplitude = torch.zeros_like(batch['target']).float()
                
                # Handle threshold outputs
                if 'threshold' in outputs:
                    threshold_pred = outputs['threshold']
                    if threshold_pred.dim() > 1:
                        threshold_pred = threshold_pred.squeeze(-1)
                else:
                    threshold_pred = torch.zeros_like(batch['target']).float()
                
                # Handle static parameter outputs (joint generation)
                if 'static_params' in outputs:
                    static_pred = outputs['static_params']
                    self.predictions['static_params'].extend(static_pred.cpu().numpy())
                    self.targets['static_params'].extend(batch['static_params'].cpu().numpy())
                
            else:
                # ProfessionalHierarchicalUNet outputs
                class_pred = torch.argmax(outputs['class'], dim=1)
                peak_exists, peak_latency, peak_amplitude = outputs['peak'][:3]
                threshold_pred = outputs['threshold']
                signal_pred = outputs['recon']
            
            # Process peak predictions
            peak_exists_pred = torch.sigmoid(peak_exists) > 0.5
            
            # Store predictions
            self.predictions['class'].extend(class_pred.cpu().numpy())
            self.predictions['peak_exists'].extend(peak_exists_pred.cpu().numpy().flatten())
            self.predictions['peak_latency'].extend(peak_latency.cpu().numpy().flatten())
            self.predictions['peak_amplitude'].extend(peak_amplitude.cpu().numpy().flatten())
            self.predictions['threshold'].extend(threshold_pred.cpu().numpy().flatten())
            self.predictions['signal'].extend(signal_pred.cpu().numpy())
            
            # Store targets
            self.targets['class'].extend(batch['target'].cpu().numpy())
            peak_exists_target = (batch['v_peak_mask'][:, 0] & batch['v_peak_mask'][:, 1]).float()
            self.targets['peak_exists'].extend(peak_exists_target.cpu().numpy())
            self.targets['peak_latency'].extend(batch['v_peak'][:, 0].cpu().numpy())
            self.targets['peak_amplitude'].extend(batch['v_peak'][:, 1].cpu().numpy())
            
            # Handle threshold targets - use explicit threshold if available
            if 'threshold' in batch:
                self.targets['threshold'].extend(batch['threshold'].cpu().numpy())
            else:
                # Use intensity as threshold proxy if no explicit threshold
                intensity_proxy = torch.sigmoid(batch['static_params'][:, 1])
                self.targets['threshold'].extend(intensity_proxy.cpu().numpy())
            
            self.targets['signal'].extend(batch['signal'].cpu().numpy())
            self.targets['v_peak_mask'].extend(batch['v_peak_mask'].cpu().numpy())
            
            # Store sample info
            for i in range(len(batch['target'])):
                sample_info = {
                    'patient_id': batch['patient_ids'][i] if 'patient_ids' in batch else i,
                    'true_class': batch['target'][i].item(),
                    'pred_class': class_pred[i].item(),
                    'static_params': batch['static_params'][i].cpu().numpy(),
                    'model_type': self.model_type
                }
                
                # Add optimized model specific info
                if self.model_type == 'optimized' and 'static_params' in outputs:
                    sample_info['pred_static_params'] = static_pred[i].cpu().numpy()
                
                self.sample_info.append(sample_info)
            
            # Compute loss if requested
            batch_metrics = {}
            if compute_loss and loss_fn is not None:
                loss, loss_components = loss_fn(outputs, batch)
                batch_metrics['total_loss'] = loss.item()
                for key, value in loss_components.items():
                    if torch.is_tensor(value):
                        batch_metrics[key] = value.item()
                        self.losses[key].append(value.item())
                    else:
                        batch_metrics[key] = value
                        self.losses[key].append(value)
        
        return batch_metrics
    
    def compute_classification_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive classification metrics.
        
        Returns:
            Dictionary of classification metrics
        """
        y_true = np.array(self.targets['class'])
        y_pred = np.array(self.predictions['class'])
        
        # Define all possible labels to ensure consistency
        labels = list(range(len(self.class_names)))  # Use all possible class labels
        
        # Basic metrics with explicit labels
        f1_macro = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Confusion matrix with explicit labels
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            labels=labels,  # Specify all possible labels
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class.tolist(),
            'balanced_accuracy': balanced_acc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'accuracy': (y_pred == y_true).mean()
        }
    
    def compute_peak_metrics(self) -> Dict[str, Any]:
        """
        Compute peak prediction metrics with masking.
        
        Returns:
            Dictionary of peak metrics
        """
        # Peak existence metrics
        exists_true = np.array(self.targets['peak_exists'])
        exists_pred = np.array(self.predictions['peak_exists'])
        
        # Binary f1_score with explicit labels
        exists_f1 = f1_score(exists_true, exists_pred, labels=[0, 1], zero_division=0)
        exists_acc = (exists_pred == exists_true).mean()
        
        # Peak regression metrics (only for valid peaks)
        v_peak_mask = np.array(self.targets['v_peak_mask'])
        latency_mask = v_peak_mask[:, 0]
        amplitude_mask = v_peak_mask[:, 1]
        
        # Latency metrics
        if latency_mask.any():
            latency_true = np.array(self.targets['peak_latency'])[latency_mask]
            latency_pred = np.array(self.predictions['peak_latency'])[latency_mask]
            
            latency_mae = mean_absolute_error(latency_true, latency_pred)
            latency_mse = mean_squared_error(latency_true, latency_pred)
            latency_r2 = r2_score(latency_true, latency_pred)
        else:
            latency_mae = latency_mse = latency_r2 = 0.0
        
        # Amplitude metrics
        if amplitude_mask.any():
            amplitude_true = np.array(self.targets['peak_amplitude'])[amplitude_mask]
            amplitude_pred = np.array(self.predictions['peak_amplitude'])[amplitude_mask]
            
            amplitude_mae = mean_absolute_error(amplitude_true, amplitude_pred)
            amplitude_mse = mean_squared_error(amplitude_true, amplitude_pred)
            amplitude_r2 = r2_score(amplitude_true, amplitude_pred)
        else:
            amplitude_mae = amplitude_mse = amplitude_r2 = 0.0
        
        return {
            'existence_f1': exists_f1,
            'existence_accuracy': exists_acc,
            'latency_mae': latency_mae,
            'latency_mse': latency_mse,
            'latency_r2': latency_r2,
            'amplitude_mae': amplitude_mae,
            'amplitude_mse': amplitude_mse,
            'amplitude_r2': amplitude_r2,
            'valid_peaks_ratio': latency_mask.mean()
        }
    
    def compute_signal_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive signal reconstruction metrics.
        
        Returns:
            Dictionary of signal reconstruction metrics
        """
        if not self.predictions['recon'] or not self.targets['signal']:
            return {}
        
        # Convert lists to tensors
        pred_signals = torch.stack(self.predictions['recon'])  # [N, 1, L]
        true_signals = torch.stack(self.targets['signal'])     # [N, 1, L]
        
        # Flatten for easier computation
        pred_flat = pred_signals.view(pred_signals.size(0), -1)  # [N, L]
        true_flat = true_signals.view(true_signals.size(0), -1)  # [N, L]
        
        metrics = {}
        
        # MSE and MAE
        mse_per_sample = torch.mean((pred_flat - true_flat) ** 2, dim=1)
        mae_per_sample = torch.mean(torch.abs(pred_flat - true_flat), dim=1)
        
        metrics['signal_mse'] = mse_per_sample.mean().item()
        metrics['signal_mse_std'] = mse_per_sample.std().item()
        metrics['signal_mae'] = mae_per_sample.mean().item()
        metrics['signal_mae_std'] = mae_per_sample.std().item()
        
        # RMSE
        rmse_per_sample = torch.sqrt(mse_per_sample)
        metrics['signal_rmse'] = rmse_per_sample.mean().item()
        metrics['signal_rmse_std'] = rmse_per_sample.std().item()
        
        # Pearson correlation per sample
        correlations = []
        for i in range(pred_flat.size(0)):
            pred_np = pred_flat[i].cpu().numpy()
            true_np = true_flat[i].cpu().numpy()
            
            # Check for constant signals (would cause correlation issues)
            if np.std(pred_np) > 1e-8 and np.std(true_np) > 1e-8:
                corr, _ = scipy.stats.pearsonr(pred_np, true_np)
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if correlations:
            metrics['signal_correlation'] = np.mean(correlations)
            metrics['signal_correlation_std'] = np.std(correlations)
        else:
            metrics['signal_correlation'] = 0.0
            metrics['signal_correlation_std'] = 0.0
        
        # Signal-to-Noise Ratio (SNR)
        signal_power = torch.mean(true_flat ** 2, dim=1)
        noise_power = torch.mean((pred_flat - true_flat) ** 2, dim=1)
        snr_per_sample = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        
        metrics['signal_snr'] = snr_per_sample.mean().item()
        metrics['signal_snr_std'] = snr_per_sample.std().item()
        
        # Dynamic Time Warping (DTW) distance if available
        if DTW_AVAILABLE:
            dtw_distances = []
            for i in range(min(pred_flat.size(0), 50)):  # Limit to 50 samples for efficiency
                pred_np = pred_flat[i].cpu().numpy()
                true_np = true_flat[i].cpu().numpy()
                
                try:
                    distance, _ = fastdtw(pred_np, true_np)
                    dtw_distances.append(distance)
                except:
                    continue
            
            if dtw_distances:
                metrics['signal_dtw'] = np.mean(dtw_distances)
                metrics['signal_dtw_std'] = np.std(dtw_distances)
        
        # Spectral similarity (FFT-based)
        pred_fft = torch.fft.fft(pred_flat, dim=1)
        true_fft = torch.fft.fft(true_flat, dim=1)
        
        # Magnitude spectrum similarity
        pred_mag = torch.abs(pred_fft)
        true_mag = torch.abs(true_fft)
        
        spectral_mse = torch.mean((pred_mag - true_mag) ** 2, dim=1)
        metrics['spectral_mse'] = spectral_mse.mean().item()
        metrics['spectral_mse_std'] = spectral_mse.std().item()
        
        # Phase coherence
        pred_phase = torch.angle(pred_fft)
        true_phase = torch.angle(true_fft)
        phase_diff = torch.abs(pred_phase - true_phase)
        phase_coherence = torch.cos(phase_diff)
        
        metrics['phase_coherence'] = torch.mean(phase_coherence).item()
        metrics['phase_coherence_std'] = torch.std(phase_coherence).item()
        
        return metrics
    
    def compute_threshold_metrics(self) -> Dict[str, Any]:
        """
        Compute threshold prediction metrics.
        
        Returns:
            Dictionary of threshold metrics
        """
        threshold_true = np.array(self.targets['threshold'])
        threshold_pred = np.array(self.predictions['threshold'])
        
        mae = mean_absolute_error(threshold_true, threshold_pred)
        mse = mean_squared_error(threshold_true, threshold_pred)
        r2 = r2_score(threshold_true, threshold_pred)
        
        return {
            'threshold_mae': mae,
            'threshold_mse': mse,
            'threshold_r2': r2
        }
    
    def compute_all_metrics(self) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Classification metrics
        metrics['classification'] = self.compute_classification_metrics()
        
        # Peak metrics
        metrics['peaks'] = self.compute_peak_metrics()
        
        # Signal metrics
        metrics['signal'] = self.compute_signal_metrics()
        
        # Threshold metrics
        metrics['threshold'] = self.compute_threshold_metrics()
        
        # Loss metrics
        if self.losses:
            metrics['losses'] = {
                key: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                for key, values in self.losses.items()
            }
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        y_true = np.array(self.targets['class'])
        y_pred = np.array(self.predictions['class'])
        
        # Use explicit labels for consistency
        labels = list(range(len(self.class_names)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Safe normalization of confusion matrix
        row_sums = cm.sum(axis=1)
        row_sums_safe = np.where(row_sums == 0, 1, row_sums)
        cm_normalized = cm.astype('float') / row_sums_safe[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_peak_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot peak prediction analysis.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Peak existence
        exists_true = np.array(self.targets['peak_exists'])
        exists_pred = np.array(self.predictions['peak_exists'])
        
        exists_cm = confusion_matrix(exists_true, exists_pred)
        sns.heatmap(exists_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Peak', 'Peak'], yticklabels=['No Peak', 'Peak'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Peak Existence Prediction')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # Peak latency scatter (only valid peaks)
        v_peak_mask = np.array(self.targets['v_peak_mask'])
        latency_mask = v_peak_mask[:, 0]
        
        if latency_mask.any():
            latency_true = np.array(self.targets['peak_latency'])[latency_mask]
            latency_pred = np.array(self.predictions['peak_latency'])[latency_mask]
            
            axes[0, 1].scatter(latency_true, latency_pred, alpha=0.6)
            axes[0, 1].plot([latency_true.min(), latency_true.max()], 
                           [latency_true.min(), latency_true.max()], 'r--', lw=2)
            axes[0, 1].set_xlabel('True Latency (ms)')
            axes[0, 1].set_ylabel('Predicted Latency (ms)')
            axes[0, 1].set_title('Peak Latency Prediction')
            
            # Add R² score
            r2 = r2_score(latency_true, latency_pred)
            axes[0, 1].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 1].transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Peak amplitude scatter (only valid peaks)
        amplitude_mask = v_peak_mask[:, 1]
        
        if amplitude_mask.any():
            amplitude_true = np.array(self.targets['peak_amplitude'])[amplitude_mask]
            amplitude_pred = np.array(self.predictions['peak_amplitude'])[amplitude_mask]
            
            axes[1, 0].scatter(amplitude_true, amplitude_pred, alpha=0.6)
            axes[1, 0].plot([amplitude_true.min(), amplitude_true.max()], 
                           [amplitude_true.min(), amplitude_true.max()], 'r--', lw=2)
            axes[1, 0].set_xlabel('True Amplitude (μV)')
            axes[1, 0].set_ylabel('Predicted Amplitude (μV)')
            axes[1, 0].set_title('Peak Amplitude Prediction')
            
            # Add R² score
            r2 = r2_score(amplitude_true, amplitude_pred)
            axes[1, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1, 0].transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Peak error distribution
        if latency_mask.any() and amplitude_mask.any():
            latency_errors = latency_pred - latency_true
            amplitude_errors = amplitude_pred - amplitude_true
            
            axes[1, 1].hist(latency_errors, bins=30, alpha=0.7, label='Latency Error', density=True)
            axes[1, 1].hist(amplitude_errors, bins=30, alpha=0.7, label='Amplitude Error', density=True)
            axes[1, 1].set_xlabel('Prediction Error')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Peak Prediction Error Distribution')
            axes[1, 1].legend()
            axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_signal_reconstruction(self, num_samples: int = 6, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot signal reconstruction examples.
        
        Args:
            num_samples: Number of samples to plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        signal_true = np.array(self.targets['signal'])
        signal_pred = np.array(self.predictions['signal'])
        
        # Select random samples
        indices = np.random.choice(len(signal_true), min(num_samples, len(signal_true)), replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
                
            true_signal = signal_true[idx].flatten()
            pred_signal = signal_pred[idx].flatten()
            
            # Plot signals
            time_axis = np.arange(len(true_signal))
            axes[i].plot(time_axis, true_signal, label='True', linewidth=2)
            axes[i].plot(time_axis, pred_signal, label='Predicted', linewidth=2, alpha=0.8)
            
            # Compute correlation
            corr = np.corrcoef(true_signal, pred_signal)[0, 1]
            mse = np.mean((true_signal - pred_signal) ** 2)
            
            axes[i].set_title(f'Sample {idx}\nCorr: {corr:.3f}, MSE: {mse:.4f}')
            axes[i].set_xlabel('Time (samples)')
            axes[i].set_ylabel('Amplitude')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        metrics = self.compute_all_metrics()
        
        report = ["=" * 80]
        report.append("ENHANCED ABR MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Classification metrics
        report.append("📊 CLASSIFICATION METRICS")
        report.append("-" * 40)
        cls_metrics = metrics['classification']
        report.append(f"Overall Accuracy: {cls_metrics['accuracy']:.4f}")
        report.append(f"Balanced Accuracy: {cls_metrics['balanced_accuracy']:.4f}")
        report.append(f"F1-Score (Macro): {cls_metrics['f1_macro']:.4f}")
        report.append(f"F1-Score (Weighted): {cls_metrics['f1_weighted']:.4f}")
        report.append("")
        
        report.append("Per-Class F1 Scores:")
        for i, (class_name, f1) in enumerate(zip(self.class_names, cls_metrics['f1_per_class'])):
            report.append(f"  {class_name}: {f1:.4f}")
        report.append("")
        
        # Peak metrics
        report.append("🎯 PEAK PREDICTION METRICS")
        report.append("-" * 40)
        peak_metrics = metrics['peaks']
        report.append(f"Peak Existence F1: {peak_metrics['existence_f1']:.4f}")
        report.append(f"Peak Existence Accuracy: {peak_metrics['existence_accuracy']:.4f}")
        report.append(f"Valid Peaks Ratio: {peak_metrics['valid_peaks_ratio']:.4f}")
        report.append("")
        
        report.append("Peak Latency Metrics:")
        report.append(f"  MAE: {peak_metrics['latency_mae']:.4f} ms")
        report.append(f"  MSE: {peak_metrics['latency_mse']:.4f} ms²")
        report.append(f"  R²: {peak_metrics['latency_r2']:.4f}")
        report.append("")
        
        report.append("Peak Amplitude Metrics:")
        report.append(f"  MAE: {peak_metrics['amplitude_mae']:.4f} μV")
        report.append(f"  MSE: {peak_metrics['amplitude_mse']:.4f} μV²")
        report.append(f"  R²: {peak_metrics['amplitude_r2']:.4f}")
        report.append("")
        
        # Signal metrics
        report.append("🌊 SIGNAL RECONSTRUCTION METRICS")
        report.append("-" * 40)
        signal_metrics = metrics['signal']
        report.append(f"Signal MSE: {signal_metrics['signal_mse']:.6f}")
        report.append(f"Signal MAE: {signal_metrics['signal_mae']:.6f}")
        report.append(f"Signal Correlation: {signal_metrics['signal_correlation']:.4f}")
        report.append(f"Signal SNR: {signal_metrics['signal_snr']:.2f} dB")
        report.append("")
        
        # Threshold metrics
        report.append("🎚️ THRESHOLD PREDICTION METRICS")
        report.append("-" * 40)
        threshold_metrics = metrics['threshold']
        report.append(f"Threshold MAE: {threshold_metrics['threshold_mae']:.4f}")
        report.append(f"Threshold MSE: {threshold_metrics['threshold_mse']:.4f}")
        report.append(f"Threshold R²: {threshold_metrics['threshold_r2']:.4f}")
        report.append("")
        
        # Loss metrics
        if 'losses' in metrics:
            report.append("📉 LOSS METRICS")
            report.append("-" * 40)
            for loss_name, loss_stats in metrics['losses'].items():
                report.append(f"{loss_name}:")
                report.append(f"  Mean: {loss_stats['mean']:.6f}")
                report.append(f"  Std: {loss_stats['std']:.6f}")
                report.append(f"  Range: [{loss_stats['min']:.6f}, {loss_stats['max']:.6f}]")
            report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_results(self, output_dir: Optional[str] = None) -> None:
        """
        Save all evaluation results including metrics, plots, and raw data.
        
        Args:
            output_dir: Directory to save results
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute and save metrics
        metrics = self.compute_all_metrics()
        
        # Save metrics as JSON
        import json
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(metrics), f, indent=2)
        
        # Save detailed report
        report = self.generate_report()
        with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
            f.write(report)
        
        # Save plots
        confusion_fig = self.plot_confusion_matrix()
        confusion_fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'), 
                             dpi=300, bbox_inches='tight')
        plt.close(confusion_fig)
        
        peak_fig = self.plot_peak_analysis()
        peak_fig.savefig(os.path.join(output_dir, 'peak_analysis.png'), 
                        dpi=300, bbox_inches='tight')
        plt.close(peak_fig)
        
        signal_fig = self.plot_signal_reconstruction()
        signal_fig.savefig(os.path.join(output_dir, 'signal_reconstruction.png'), 
                          dpi=300, bbox_inches='tight')
        plt.close(signal_fig)
        
        # Save raw predictions and targets
        results_data = {
            'predictions': self.predictions,
            'targets': self.targets,
            'sample_info': self.sample_info
        }
        
        joblib.dump(results_data, os.path.join(output_dir, 'raw_results.pkl'))
        
        print(f"Evaluation results saved to: {output_dir}")

    def create_diagnostic_visualizations(self, epoch: int, num_samples: int = 5) -> Dict[str, Any]:
        """
        Create comprehensive diagnostic visualizations for monitoring training progress.
        
        Args:
            epoch: Current training epoch
            num_samples: Number of samples to visualize
            
        Returns:
            Dictionary containing visualization data and plots
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from io import BytesIO
        import base64
        
        if not self.predictions['recon'] or not self.targets['signal']:
            return {}
        
        # Set style for better plots
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        visualizations = {}
        
        # 1. Signal Reconstruction Visualization
        pred_signals = torch.stack(self.predictions['recon'][:num_samples])
        true_signals = torch.stack(self.targets['signal'][:num_samples])
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            pred_signal = pred_signals[i].squeeze().cpu().numpy()
            true_signal = true_signals[i].squeeze().cpu().numpy()
            
            time_axis = np.linspace(0, 10, len(pred_signal))  # Assuming 10ms ABR
            
            axes[i].plot(time_axis, true_signal, 'b-', label='True Signal', linewidth=2, alpha=0.8)
            axes[i].plot(time_axis, pred_signal, 'r--', label='Predicted Signal', linewidth=2, alpha=0.8)
            
            # Add peak annotations if available
            if i < len(self.targets['v_peak']) and i < len(self.predictions['peak']):
                true_peak = self.targets['v_peak'][i]
                pred_peak = self.predictions['peak'][i]
                
                if len(pred_peak) >= 3:  # existence, latency, amplitude
                    pred_latency = pred_peak[1].item() if hasattr(pred_peak[1], 'item') else pred_peak[1]
                    pred_amplitude = pred_peak[2].item() if hasattr(pred_peak[2], 'item') else pred_peak[2]
                    
                    # Mark predicted peak
                    if 0 <= pred_latency <= 10:
                        axes[i].axvline(pred_latency, color='red', linestyle=':', alpha=0.7, label=f'Pred Peak: {pred_latency:.2f}ms')
                        axes[i].plot(pred_latency, pred_amplitude, 'ro', markersize=8, alpha=0.7)
                
                # Mark true peak if valid
                if len(true_peak) >= 2 and i < len(self.targets['v_peak_mask']):
                    mask = self.targets['v_peak_mask'][i]
                    if mask[0] and mask[1]:  # Both latency and amplitude are valid
                        true_latency = true_peak[0].item() if hasattr(true_peak[0], 'item') else true_peak[0]
                        true_amplitude = true_peak[1].item() if hasattr(true_peak[1], 'item') else true_peak[1]
                        
                        if 0 <= true_latency <= 10:
                            axes[i].axvline(true_latency, color='blue', linestyle=':', alpha=0.7, label=f'True Peak: {true_latency:.2f}ms')
                            axes[i].plot(true_latency, true_amplitude, 'bo', markersize=8, alpha=0.7)
            
            # Compute correlation for this sample
            corr = np.corrcoef(true_signal, pred_signal)[0, 1] if np.std(pred_signal) > 1e-8 else 0.0
            mse = np.mean((true_signal - pred_signal) ** 2)
            
            axes[i].set_title(f'Sample {i+1} - Correlation: {corr:.3f}, MSE: {mse:.4f}')
            axes[i].set_xlabel('Time (ms)')
            axes[i].set_ylabel('Amplitude (μV)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to bytes for logging
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        visualizations['signal_reconstruction'] = buffer.getvalue()
        plt.close()
        
        # 2. Peak Prediction Scatter Plots
        if self.predictions['peak'] and self.targets['v_peak']:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Extract peak data
            pred_latencies = []
            pred_amplitudes = []
            true_latencies = []
            true_amplitudes = []
            
            for i, (pred_peak, true_peak, mask) in enumerate(zip(
                self.predictions['peak'], self.targets['v_peak'], self.targets['v_peak_mask']
            )):
                if len(pred_peak) >= 3 and len(true_peak) >= 2:
                    if mask[0]:  # Valid latency
                        pred_lat = pred_peak[1].item() if hasattr(pred_peak[1], 'item') else pred_peak[1]
                        true_lat = true_peak[0].item() if hasattr(true_peak[0], 'item') else true_peak[0]
                        pred_latencies.append(pred_lat)
                        true_latencies.append(true_lat)
                    
                    if mask[1]:  # Valid amplitude
                        pred_amp = pred_peak[2].item() if hasattr(pred_peak[2], 'item') else pred_peak[2]
                        true_amp = true_peak[1].item() if hasattr(true_peak[1], 'item') else true_peak[1]
                        pred_amplitudes.append(pred_amp)
                        true_amplitudes.append(true_amp)
            
            # Latency scatter plot
            if pred_latencies and true_latencies:
                axes[0].scatter(true_latencies, pred_latencies, alpha=0.6, s=50)
                
                # Perfect prediction line
                lat_min = min(min(true_latencies), min(pred_latencies))
                lat_max = max(max(true_latencies), max(pred_latencies))
                axes[0].plot([lat_min, lat_max], [lat_min, lat_max], 'r--', alpha=0.8, label='Perfect Prediction')
                
                # Compute R²
                r2_lat = 1 - np.sum((np.array(true_latencies) - np.array(pred_latencies))**2) / \
                         np.sum((np.array(true_latencies) - np.mean(true_latencies))**2)
                
                axes[0].set_xlabel('True Peak Latency (ms)')
                axes[0].set_ylabel('Predicted Peak Latency (ms)')
                axes[0].set_title(f'Peak Latency Prediction (R² = {r2_lat:.3f})')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Amplitude scatter plot
            if pred_amplitudes and true_amplitudes:
                axes[1].scatter(true_amplitudes, pred_amplitudes, alpha=0.6, s=50)
                
                # Perfect prediction line
                amp_min = min(min(true_amplitudes), min(pred_amplitudes))
                amp_max = max(max(true_amplitudes), max(pred_amplitudes))
                axes[1].plot([amp_min, amp_max], [amp_min, amp_max], 'r--', alpha=0.8, label='Perfect Prediction')
                
                # Compute R²
                r2_amp = 1 - np.sum((np.array(true_amplitudes) - np.array(pred_amplitudes))**2) / \
                         np.sum((np.array(true_amplitudes) - np.mean(true_amplitudes))**2)
                
                axes[1].set_xlabel('True Peak Amplitude (μV)')
                axes[1].set_ylabel('Predicted Peak Amplitude (μV)')
                axes[1].set_title(f'Peak Amplitude Prediction (R² = {r2_amp:.3f})')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            visualizations['peak_predictions'] = buffer.getvalue()
            plt.close()
        
        # 3. Classification Confusion Matrix
        if self.predictions['class'] and self.targets['class']:
            from sklearn.metrics import confusion_matrix
            
            pred_classes = [torch.argmax(pred).item() for pred in self.predictions['class']]
            true_classes = [target.item() if hasattr(target, 'item') else target for target in self.targets['class']]
            
            # Use explicit labels for consistency
            labels = list(range(len(self.class_names)))
            cm = confusion_matrix(true_classes, pred_classes, labels=labels)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')
            
            # Save plot
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            visualizations['confusion_matrix'] = buffer.getvalue()
            plt.close()
        
        # 4. Threshold Prediction Curve
        if self.predictions['threshold'] and len(self.predictions['threshold']) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Extract threshold predictions (handle different formats)
            pred_thresholds = []
            for pred in self.predictions['threshold']:
                if hasattr(pred, 'item'):
                    pred_thresholds.append(pred.item())
                elif isinstance(pred, (int, float)):
                    pred_thresholds.append(pred)
                elif hasattr(pred, '__len__') and len(pred) > 0:
                    pred_thresholds.append(pred[0])
            
            if pred_thresholds:
                # Plot threshold distribution
                ax.hist(pred_thresholds, bins=20, alpha=0.7, edgecolor='black')
                ax.axvline(np.mean(pred_thresholds), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(pred_thresholds):.1f} dB')
                ax.set_xlabel('Predicted Threshold (dB SPL)')
                ax.set_ylabel('Count')
                ax.set_title(f'Threshold Prediction Distribution - Epoch {epoch}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Save plot
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            visualizations['threshold_distribution'] = buffer.getvalue()
            plt.close()
        
        # 5. Error Distribution Analysis
        if self.predictions['recon'] and self.targets['signal']:
            pred_signals = torch.stack(self.predictions['recon'])
            true_signals = torch.stack(self.targets['signal'])
            
            # Compute per-sample errors
            mse_errors = torch.mean((pred_signals - true_signals) ** 2, dim=[1, 2]).cpu().numpy()
            mae_errors = torch.mean(torch.abs(pred_signals - true_signals), dim=[1, 2]).cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # MSE distribution
            axes[0].hist(mse_errors, bins=20, alpha=0.7, edgecolor='black')
            axes[0].axvline(np.mean(mse_errors), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(mse_errors):.4f}')
            axes[0].set_xlabel('MSE Error')
            axes[0].set_ylabel('Count')
            axes[0].set_title('MSE Error Distribution')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # MAE distribution
            axes[1].hist(mae_errors, bins=20, alpha=0.7, edgecolor='black')
            axes[1].axvline(np.mean(mae_errors), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(mae_errors):.4f}')
            axes[1].set_xlabel('MAE Error')
            axes[1].set_ylabel('Count')
            axes[1].set_title('MAE Error Distribution')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            visualizations['error_distributions'] = buffer.getvalue()
            plt.close()
        
        return visualizations
    
    def log_to_tensorboard(self, writer, epoch: int):
        """
        Log comprehensive metrics and visualizations to TensorBoard.
        
        Args:
            writer: TensorBoard SummaryWriter
            epoch: Current epoch
        """
        # Compute all metrics
        metrics = self.compute_all_metrics()
        
        # Log scalar metrics
        for category, category_metrics in metrics.items():
            if isinstance(category_metrics, dict):
                for metric_name, metric_value in category_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        writer.add_scalar(f'{category}/{metric_name}', metric_value, epoch)
        
        # Create and log visualizations
        visualizations = self.create_diagnostic_visualizations(epoch)
        
        for viz_name, viz_data in visualizations.items():
            if viz_data:
                # Convert bytes to PIL Image for TensorBoard
                from PIL import Image
                image = Image.open(BytesIO(viz_data))
                
                # Convert PIL to tensor
                import torchvision.transforms as transforms
                transform = transforms.Compose([transforms.ToTensor()])
                image_tensor = transform(image)
                
                writer.add_image(f'diagnostics/{viz_name}', image_tensor, epoch)
        
        # Log histograms of predictions vs targets
        if self.predictions['recon'] and self.targets['signal']:
            pred_signals = torch.stack(self.predictions['recon']).flatten()
            true_signals = torch.stack(self.targets['signal']).flatten()
            
            writer.add_histogram('signals/predictions', pred_signals, epoch)
            writer.add_histogram('signals/targets', true_signals, epoch)
            writer.add_histogram('signals/errors', pred_signals - true_signals, epoch)


def evaluate_model_on_dataloader(
    model: nn.Module,
    dataloader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    class_names: Optional[List[str]] = None,
    output_dir: str = 'evaluation_results'
) -> Dict[str, Any]:
    """
    Evaluate model on a complete dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader for evaluation
        device: Device for evaluation
        loss_fn: Loss function
        class_names: List of class names
        output_dir: Output directory for results
        
    Returns:
        Complete evaluation metrics
    """
    evaluator = ABREvaluator(model, device, class_names, output_dir)
    
    # Evaluate all batches
    print("Evaluating model...")
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        for key in ['static_params', 'signal', 'v_peak', 'v_peak_mask', 'target']:
            if key in batch:
                batch[key] = batch[key].to(device)
        
        # Evaluate batch
        evaluator.evaluate_batch(batch, compute_loss=(loss_fn is not None), loss_fn=loss_fn)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Compute final metrics
    metrics = evaluator.compute_all_metrics()
    
    # Generate and save results
    evaluator.save_results()
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Classification F1 (Macro): {metrics['classification']['f1_macro']:.4f}")
    print(f"Peak Existence F1: {metrics['peaks']['existence_f1']:.4f}")
    print(f"Signal Correlation: {metrics['signal']['signal_correlation']:.4f}")
    print(f"Results saved to: {output_dir}")
    
    return metrics


if __name__ == '__main__':
    # Example usage
    print("ABR Evaluation Module")
    print("This module provides comprehensive evaluation utilities for ABR models.")
    print("Use evaluate_model_on_dataloader() to evaluate a trained model.") 