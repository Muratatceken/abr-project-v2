#!/usr/bin/env python3
"""
Advanced Visualization Utilities for ABR Model Evaluation

This module provides comprehensive visualization functions for:
- Signal reconstruction plots
- Peak detection visualizations
- Classification analysis plots
- Threshold estimation plots
- Error distribution plots
- Clinical diagnostic visualizations

Author: AI Assistant
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from io import BytesIO
import warnings

# Set plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class VisualizationEngine:
    """Comprehensive visualization engine for ABR model evaluation."""
    
    def __init__(
        self, 
        class_names: List[str] = None,
        figsize: Tuple[int, int] = (15, 10),
        dpi: int = 150,
        save_format: str = 'png'
    ):
        """Initialize visualization engine.
        
        Args:
            class_names: List of class names
            figsize: Default figure size
            dpi: DPI for saved figures
            save_format: Format for saved figures
        """
        self.class_names = class_names or ["NORMAL", "NÖROPATİ", "SNİK", "TOTAL", "İTİK"]
        self.figsize = figsize
        self.dpi = dpi
        self.save_format = save_format
        
        # Color schemes
        self.colors = {
            'true_signal': '#1f77b4',
            'pred_signal': '#ff7f0e',
            'true_peak': '#2ca02c',
            'pred_peak': '#d62728',
            'perfect_line': '#7f7f7f',
            'error_zone': '#ffcccc',
            'excellent_zone': '#ccffcc',
            'good_zone': '#ffffcc',
            'acceptable_zone': '#ffddcc',
            'poor_zone': '#ffcccc'
        }
    
    # ==================== SIGNAL RECONSTRUCTION PLOTS ====================
    
    def plot_signal_reconstruction_detailed(
        self,
        signals_true: np.ndarray,
        signals_pred: np.ndarray,
        sample_indices: List[int] = None,
        class_labels: np.ndarray = None,
        threshold_true: np.ndarray = None,
        threshold_pred: np.ndarray = None,
        peak_info: Dict[str, np.ndarray] = None,
        n_samples: int = 6,
        time_axis: np.ndarray = None
    ) -> plt.Figure:
        """
        Create detailed signal reconstruction plots with clinical annotations.
        
        Args:
            signals_true: True signals [batch, channels, length]
            signals_pred: Predicted signals [batch, channels, length]
            sample_indices: Specific sample indices to plot
            class_labels: Class labels for each sample
            threshold_true: True hearing thresholds
            threshold_pred: Predicted hearing thresholds
            peak_info: Dictionary with peak information
            n_samples: Number of samples to plot
            time_axis: Time axis for plotting
            
        Returns:
            Matplotlib figure
        """
        if sample_indices is None:
            sample_indices = list(range(min(n_samples, len(signals_true))))
        
        n_samples = len(sample_indices)
        if time_axis is None:
            time_axis = np.linspace(0, 10, signals_true.shape[-1])
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(15, 3*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for i, sample_idx in enumerate(sample_indices):
            ax = axes[i]
            
            # Get signals
            true_signal = signals_true[sample_idx].squeeze()
            pred_signal = signals_pred[sample_idx].squeeze()
            
            # Plot signals
            ax.plot(time_axis, true_signal, color=self.colors['true_signal'], 
                   linewidth=2, alpha=0.8, label='Ground Truth')
            ax.plot(time_axis, pred_signal, color=self.colors['pred_signal'], 
                   linewidth=2, alpha=0.8, linestyle='--', label='Predicted')
            
            # Add peak annotations if available
            if peak_info is not None:
                self._add_peak_annotations(ax, peak_info, sample_idx, time_axis)
            
            # Compute and display metrics
            corr = np.corrcoef(true_signal, pred_signal)[0, 1] if np.std(pred_signal) > 1e-8 else 0.0
            mse = np.mean((true_signal - pred_signal) ** 2)
            snr = 10 * np.log10(np.mean(true_signal**2) / (mse + 1e-8))
            
            # Create title with clinical information
            title_parts = [f'Sample {sample_idx}']
            
            if class_labels is not None and sample_idx < len(class_labels):
                class_idx = int(class_labels[sample_idx])
                if class_idx < len(self.class_names):
                    title_parts.append(f'Class: {self.class_names[class_idx]}')
            
            if threshold_true is not None and threshold_pred is not None:
                if sample_idx < len(threshold_true) and sample_idx < len(threshold_pred):
                    true_thresh = threshold_true[sample_idx]
                    pred_thresh = threshold_pred[sample_idx]
                    thresh_error = abs(pred_thresh - true_thresh)
                    title_parts.append(f'Threshold: {true_thresh:.1f}→{pred_thresh:.1f} dB (±{thresh_error:.1f})')
            
            title_parts.append(f'Corr: {corr:.3f}, MSE: {mse:.4f}, SNR: {snr:.1f} dB')
            
            ax.set_title(' | '.join(title_parts), fontsize=10)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude (μV)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add clinical zones if thresholds available
            if threshold_true is not None and sample_idx < len(threshold_true):
                self._add_clinical_zones(ax, threshold_true[sample_idx])
        
        plt.suptitle('Detailed Signal Reconstruction Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        return fig
    
    def _add_peak_annotations(
        self, 
        ax: plt.Axes, 
        peak_info: Dict[str, np.ndarray], 
        sample_idx: int, 
        time_axis: np.ndarray
    ):
        """Add peak annotations to signal plot."""
        if 'latency_true' in peak_info and sample_idx < len(peak_info['latency_true']):
            true_latency = peak_info['latency_true'][sample_idx]
            if true_latency > 0:  # Valid peak
                ax.axvline(true_latency, color=self.colors['true_peak'], 
                          linestyle='-', alpha=0.7, label='True Peak')
        
        if 'latency_pred' in peak_info and sample_idx < len(peak_info['latency_pred']):
            pred_latency = peak_info['latency_pred'][sample_idx]
            if pred_latency > 0:  # Valid peak
                ax.axvline(pred_latency, color=self.colors['pred_peak'], 
                          linestyle='--', alpha=0.7, label='Pred Peak')
    
    def _add_clinical_zones(self, ax: plt.Axes, threshold: float):
        """Add clinical interpretation zones to plot."""
        y_min, y_max = ax.get_ylim()
        
        # Add threshold reference line
        ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Add subtle background zones based on threshold severity
        if threshold <= 25:  # Normal
            ax.axhspan(y_min, y_max, alpha=0.05, color='green')
        elif threshold <= 40:  # Mild
            ax.axhspan(y_min, y_max, alpha=0.05, color='yellow')
        elif threshold <= 70:  # Moderate
            ax.axhspan(y_min, y_max, alpha=0.05, color='orange')
        else:  # Severe
            ax.axhspan(y_min, y_max, alpha=0.05, color='red')
    
    # ==================== ERROR DISTRIBUTION PLOTS ====================
    
    def plot_error_distributions(
        self,
        metrics_dict: Dict[str, Dict[str, Any]],
        stratify_by: str = None
    ) -> plt.Figure:
        """
        Plot comprehensive error distribution analysis.
        
        Args:
            metrics_dict: Dictionary of computed metrics
            stratify_by: Variable to stratify by (e.g., 'class')
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Signal reconstruction errors
        if 'signal' in metrics_dict:
            self._plot_signal_error_distributions(fig, gs, metrics_dict['signal'], stratify_by)
        
        # 2. Peak estimation errors
        if 'peaks' in metrics_dict:
            self._plot_peak_error_distributions(fig, gs, metrics_dict['peaks'], stratify_by)
        
        # 3. Threshold estimation errors
        if 'threshold' in metrics_dict:
            self._plot_threshold_error_distributions(fig, gs, metrics_dict['threshold'], stratify_by)
        
        # 4. Classification confusion matrix
        if 'classification' in metrics_dict:
            self._plot_classification_analysis(fig, gs, metrics_dict['classification'])
        
        plt.suptitle('Comprehensive Error Distribution Analysis', fontsize=18, y=0.98)
        return fig
    
    def _plot_signal_error_distributions(
        self, 
        fig: plt.Figure, 
        gs: GridSpec, 
        signal_metrics: Dict[str, Any], 
        stratify_by: str
    ):
        """Plot signal reconstruction error distributions."""
        # MSE distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if 'mse_samples' in signal_metrics:
            mse_samples = signal_metrics['mse_samples']
            ax1.hist(mse_samples, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(np.mean(mse_samples), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(mse_samples):.4f}')
            ax1.fill_betweenx([0, ax1.get_ylim()[1]], 
                             np.mean(mse_samples) - np.std(mse_samples),
                             np.mean(mse_samples) + np.std(mse_samples),
                             alpha=0.2, color='red', label='±1 STD')
            ax1.set_xlabel('MSE')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Signal MSE Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # SNR distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if 'snr_samples' in signal_metrics:
            snr_samples = signal_metrics['snr_samples']
            ax2.hist(snr_samples, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.axvline(np.mean(snr_samples), color='red', linestyle='--',
                       label=f'Mean: {np.mean(snr_samples):.1f} dB')
            ax2.set_xlabel('SNR (dB)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Signal-to-Noise Ratio Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Correlation distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if 'correlation_samples' in signal_metrics and len(signal_metrics['correlation_samples']) > 0:
            corr_samples = signal_metrics['correlation_samples']
            ax3.hist(corr_samples, bins=50, alpha=0.7, color='salmon', edgecolor='black')
            ax3.axvline(np.mean(corr_samples), color='red', linestyle='--',
                       label=f'Mean: {np.mean(corr_samples):.3f}')
            ax3.set_xlabel('Correlation')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Signal Correlation Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    def _plot_peak_error_distributions(
        self, 
        fig: plt.Figure, 
        gs: GridSpec, 
        peak_metrics: Dict[str, Any], 
        stratify_by: str
    ):
        """Plot peak estimation error distributions."""
        # Latency errors
        ax1 = fig.add_subplot(gs[1, 0])
        if 'latency_errors' in peak_metrics:
            latency_errors = peak_metrics['latency_errors']
            ax1.hist(latency_errors, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            ax1.axvline(np.mean(latency_errors), color='red', linestyle='--',
                       label=f'Mean: {np.mean(latency_errors):.3f} ms')
            ax1.set_xlabel('Latency Error (ms)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Peak Latency Error Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Amplitude errors
        ax2 = fig.add_subplot(gs[1, 1])
        if 'amplitude_errors' in peak_metrics:
            amplitude_errors = peak_metrics['amplitude_errors']
            ax2.hist(amplitude_errors, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.axvline(np.mean(amplitude_errors), color='red', linestyle='--',
                       label=f'Mean: {np.mean(amplitude_errors):.3f} μV')
            ax2.set_xlabel('Amplitude Error (μV)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Peak Amplitude Error Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    def _plot_threshold_error_distributions(
        self, 
        fig: plt.Figure, 
        gs: GridSpec, 
        threshold_metrics: Dict[str, Any], 
        stratify_by: str
    ):
        """Plot threshold estimation error distributions with clinical zones."""
        # Threshold errors
        ax1 = fig.add_subplot(gs[2, 0])
        if 'threshold_errors' in threshold_metrics:
            threshold_errors = threshold_metrics['threshold_errors']
            ax1.hist(threshold_errors, bins=40, alpha=0.7, color='gold', edgecolor='black')
            ax1.axvline(np.mean(threshold_errors), color='red', linestyle='--',
                       label=f'Mean: {np.mean(threshold_errors):.1f} dB')
            
            # Add clinical error zones
            ax1.axvspan(0, 5, alpha=0.2, color=self.colors['excellent_zone'], label='Excellent (≤5 dB)')
            ax1.axvspan(5, 10, alpha=0.2, color=self.colors['good_zone'], label='Good (5-10 dB)')
            ax1.axvspan(10, 20, alpha=0.2, color=self.colors['acceptable_zone'], label='Acceptable (10-20 dB)')
            ax1.axvspan(20, ax1.get_xlim()[1], alpha=0.2, color=self.colors['poor_zone'], label='Poor (>20 dB)')
            
            ax1.set_xlabel('Threshold Error (dB)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Hearing Threshold Error Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Residuals (bias analysis)
        ax2 = fig.add_subplot(gs[2, 1])
        if 'threshold_residuals' in threshold_metrics:
            residuals = threshold_metrics['threshold_residuals']
            ax2.hist(residuals, bins=40, alpha=0.7, color='mediumpurple', edgecolor='black')
            ax2.axvline(0, color='black', linestyle='-', alpha=0.5, label='Perfect')
            ax2.axvline(np.mean(residuals), color='red', linestyle='--',
                       label=f'Bias: {np.mean(residuals):.1f} dB')
            
            # Add clinical error zones
            ax2.axvspan(-20, -20, alpha=0.3, color='red', label='False Clear (< -20 dB)')
            ax2.axvspan(20, ax2.get_xlim()[1], alpha=0.3, color='orange', label='False Impairment (> +20 dB)')
            
            ax2.set_xlabel('Threshold Residual (Pred - True, dB)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Hearing Threshold Bias Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    def _plot_classification_analysis(
        self, 
        fig: plt.Figure, 
        gs: GridSpec, 
        class_metrics: Dict[str, Any]
    ):
        """Plot classification analysis including confusion matrix."""
        # Confusion matrix
        ax = fig.add_subplot(gs[0:2, 3])
        if 'confusion_matrix_normalized' in class_metrics:
            cm_norm = class_metrics['confusion_matrix_normalized']
            im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
            
            # Add text annotations
            thresh = cm_norm.max() / 2.
            for i in range(cm_norm.shape[0]):
                for j in range(cm_norm.shape[1]):
                    ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if cm_norm[i, j] > thresh else "black")
            
            ax.set_xlabel('Predicted Class')
            ax.set_ylabel('True Class')
            ax.set_title('Normalized Confusion Matrix')
            ax.set_xticks(range(len(self.class_names)))
            ax.set_yticks(range(len(self.class_names)))
            ax.set_xticklabels(self.class_names, rotation=45)
            ax.set_yticklabels(self.class_names)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # ==================== THRESHOLD SCATTER PLOTS ====================
    
    def plot_threshold_scatter_with_clinical_zones(
        self,
        threshold_true: np.ndarray,
        threshold_pred: np.ndarray,
        class_labels: np.ndarray = None,
        sample_ids: List[str] = None
    ) -> plt.Figure:
        """
        Create threshold scatter plot with clinical error zones.
        
        Args:
            threshold_true: True thresholds
            threshold_pred: Predicted thresholds
            class_labels: Class labels for color coding
            sample_ids: Sample IDs for annotation
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Flatten arrays
        threshold_true_flat = threshold_true.flatten()
        threshold_pred_flat = threshold_pred.flatten()
        
        # Ensure same length
        min_len = min(len(threshold_true_flat), len(threshold_pred_flat))
        threshold_true_flat = threshold_true_flat[:min_len]
        threshold_pred_flat = threshold_pred_flat[:min_len]
        
        # Main scatter plot
        if class_labels is not None:
            class_labels_flat = class_labels.flatten()[:min_len]
            scatter = ax1.scatter(threshold_true_flat, threshold_pred_flat, 
                                c=class_labels_flat, cmap='tab10', alpha=0.6, s=50)
            
            # Add colorbar with class names
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Hearing Loss Class')
            if len(self.class_names) <= 10:  # Only if reasonable number of classes
                cbar.set_ticks(range(len(self.class_names)))
                cbar.set_ticklabels(self.class_names)
        else:
            ax1.scatter(threshold_true_flat, threshold_pred_flat, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(threshold_true_flat.min(), threshold_pred_flat.min())
        max_val = max(threshold_true_flat.max(), threshold_pred_flat.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        # Clinical error zones
        ax1.fill_between([min_val, max_val], [min_val-20, max_val-20], [min_val, max_val], 
                        alpha=0.2, color='red', label='False Clear Zone (>20 dB under)')
        ax1.fill_between([min_val, max_val], [min_val, max_val], [min_val+20, max_val+20], 
                        alpha=0.2, color='orange', label='False Impairment Zone (>20 dB over)')
        ax1.fill_between([min_val, max_val], [min_val-5, max_val-5], [min_val+5, max_val+5], 
                        alpha=0.3, color='green', label='Excellent Zone (±5 dB)')
        
        # Compute and display metrics
        mae = np.mean(np.abs(threshold_true_flat - threshold_pred_flat))
        r2 = np.corrcoef(threshold_true_flat, threshold_pred_flat)[0, 1] ** 2 if len(threshold_true_flat) > 1 else 0
        
        ax1.set_xlabel('True Hearing Threshold (dB HL)')
        ax1.set_ylabel('Predicted Hearing Threshold (dB HL)')
        ax1.set_title(f'Hearing Threshold Predictions\nMAE: {mae:.1f} dB, R²: {r2:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Residual plot
        residuals = threshold_pred_flat - threshold_true_flat
        ax2.scatter(threshold_true_flat, residuals, alpha=0.6, s=50)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5, label='Perfect')
        ax2.axhline(np.mean(residuals), color='red', linestyle='--', 
                   label=f'Bias: {np.mean(residuals):.1f} dB')
        
        # Clinical error lines
        ax2.axhline(20, color='orange', linestyle=':', alpha=0.7, label='False Impairment (+20 dB)')
        ax2.axhline(-20, color='red', linestyle=':', alpha=0.7, label='False Clear (-20 dB)')
        ax2.fill_between([min_val, max_val], -5, 5, alpha=0.3, color='green', label='Excellent Zone (±5 dB)')
        
        ax2.set_xlabel('True Hearing Threshold (dB HL)')
        ax2.set_ylabel('Residual (Predicted - True, dB)')
        ax2.set_title('Hearing Threshold Residual Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # ==================== STRATIFIED ANALYSIS PLOTS ====================
    
    def plot_stratified_performance(
        self,
        stratified_metrics: Dict[str, Dict[str, Any]],
        metric_name: str = 'accuracy',
        stratify_by: str = 'class'
    ) -> plt.Figure:
        """
        Plot performance metrics stratified by different variables.
        
        Args:
            stratified_metrics: Dictionary of stratified metrics
            metric_name: Name of metric to plot
            stratify_by: Variable used for stratification
            
        Returns:
            Matplotlib figure
        """
        if stratify_by not in stratified_metrics:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, f'No stratified data available for {stratify_by}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        strata_data = stratified_metrics[stratify_by]
        strata_names = list(strata_data.keys())
        
        # Extract metric values
        metric_values = []
        error_bars = []
        
        for stratum in strata_names:
            stratum_metrics = strata_data[stratum]
            
            # Look for the metric in different categories
            metric_value = None
            for category in ['signal', 'class', 'threshold', 'peak']:
                full_metric_name = f'{category}_{metric_name}'
                if full_metric_name in stratum_metrics:
                    metric_value = stratum_metrics[full_metric_name]
                    break
            
            if metric_value is None and metric_name in stratum_metrics:
                metric_value = stratum_metrics[metric_name]
            
            if metric_value is not None:
                if isinstance(metric_value, (list, np.ndarray)):
                    metric_values.append(np.mean(metric_value))
                    error_bars.append(np.std(metric_value))
                else:
                    metric_values.append(metric_value)
                    error_bars.append(0)
            else:
                metric_values.append(0)
                error_bars.append(0)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        bars = ax.bar(strata_names, metric_values, yerr=error_bars, 
                     capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(error_bars)*0.1,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel(stratify_by.replace('_', ' ').title())
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} by {stratify_by.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if needed
        if len(strata_names) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def fig_to_bytes(self, fig: plt.Figure) -> bytes:
        """Convert matplotlib figure to bytes."""
        buffer = BytesIO()
        fig.savefig(buffer, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        return buffer.getvalue()
    
    def save_figure(self, fig: plt.Figure, filepath: Union[str, Path]) -> None:
        """Save figure to file."""
        fig.savefig(filepath, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig) 