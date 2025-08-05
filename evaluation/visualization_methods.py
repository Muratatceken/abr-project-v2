#!/usr/bin/env python3
"""
Comprehensive Visualization Methods for ABR Model Evaluation

Professional visualization suite for detailed analysis of ABR model performance
with static plots, interactive visualizations, and clinical-grade reports.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Rectangle
from pathlib import Path
import json
from typing import Dict, Any, List
from scipy.signal import spectrogram
from sklearn.metrics import confusion_matrix

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class VisualizationMethods:
    """Methods for creating comprehensive visualizations."""
    
    def _create_comprehensive_visualizations(self) -> None:
        """Create comprehensive static visualizations."""
        print("📊 Creating comprehensive static visualizations...")
        
        # 1. Overview dashboard
        self._create_overview_dashboard()
        
        # 2. Signal quality analysis plots
        self._create_signal_quality_plots()
        
        # 3. Classification analysis plots
        self._create_classification_plots()
        
        # 4. Peak detection analysis plots
        self._create_peak_detection_plots()
        
        # 5. Threshold regression plots
        self._create_threshold_regression_plots()
        
        # 6. Uncertainty analysis plots
        self._create_uncertainty_plots()
        
        # 7. Clinical correlation plots
        self._create_clinical_plots()
        
        print(f"✅ Static visualizations saved to {self.plots_dir}")
    
    def _create_overview_dashboard(self) -> None:
        """Create an overview dashboard with key metrics."""
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('ABR Model Evaluation Dashboard', fontsize=24, fontweight='bold', y=0.95)
        
        # 1. Signal Quality Metrics (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'signal_quality' in self.results:
            signal_metrics = self.results['signal_quality']['basic_metrics']
            metrics = ['Correlation', 'SNR (dB)', 'RMSE']
            values = [
                signal_metrics.get('correlation_mean', 0),
                signal_metrics.get('snr_mean_db', 0),
                1 - min(signal_metrics.get('rmse', 1), 1)  # Invert RMSE for bar chart
            ]
            bars = ax1.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
            ax1.set_title('Signal Quality', fontweight='bold')
            ax1.set_ylim(0, 1)
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Classification Metrics (top-center-left)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'classification' in self.results:
            class_metrics = self.results['classification']['basic_metrics']
            metrics = ['Accuracy', 'F1-Macro', 'F1-Weighted']
            values = [
                class_metrics.get('accuracy', 0),
                class_metrics.get('f1_macro', 0),
                class_metrics.get('f1_weighted', 0)
            ]
            bars = ax2.bar(metrics, values, color=['lightcoral', 'lightsalmon', 'peachpuff'])
            ax2.set_title('Classification Performance', fontweight='bold')
            ax2.set_ylim(0, 1)
            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Peak Detection Metrics (top-center-right)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'peak_detection' in self.results:
            peak_metrics = self.results['peak_detection']
            metrics = ['Existence F1', 'Latency Corr', 'Amplitude Corr']
            values = [
                peak_metrics.get('existence_metrics', {}).get('f1_score', 0),
                peak_metrics.get('latency_metrics', {}).get('correlation', 0),
                peak_metrics.get('amplitude_metrics', {}).get('correlation', 0)
            ]
            bars = ax3.bar(metrics, values, color=['lightblue', 'lightsteelblue', 'powderblue'])
            ax3.set_title('Peak Detection', fontweight='bold')
            ax3.set_ylim(0, 1)
            for bar, value in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Threshold Regression Metrics (top-right)
        ax4 = fig.add_subplot(gs[0, 3])
        if 'threshold_regression' in self.results:
            thresh_metrics = self.results['threshold_regression']['regression_metrics']
            metrics = ['R² Score', 'Correlation', 'Accuracy (5dB)']
            values = [
                max(0, thresh_metrics.get('r2_score', 0)),
                thresh_metrics.get('correlation', 0),
                self.results['threshold_regression']['error_analysis'].get('within_5db_percent', 0) / 100
            ]
            bars = ax4.bar(metrics, values, color=['gold', 'khaki', 'lightyellow'])
            ax4.set_title('Threshold Regression', fontweight='bold')
            ax4.set_ylim(0, 1)
            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Sample Signal Reconstructions (second row, span 2 columns)
        ax5 = fig.add_subplot(gs[1, :2])
        if len(self.predictions['signals']) > 0:
            # Show 3 sample reconstructions
            for i in range(min(3, len(self.predictions['signals']))):
                pred_signal = self.predictions['signals'][i].numpy().flatten()
                true_signal = self.ground_truth['signals'][i].numpy().flatten()
                time_axis = np.linspace(0, 10, len(pred_signal))  # Assume 10ms duration
                
                ax5.plot(time_axis, true_signal + i*2, 'k-', linewidth=2, label=f'True {i+1}' if i == 0 else '')
                ax5.plot(time_axis, pred_signal + i*2, 'r--', linewidth=2, label=f'Pred {i+1}' if i == 0 else '')
            
            ax5.set_xlabel('Time (ms)')
            ax5.set_ylabel('Amplitude (µV)')
            ax5.set_title('Sample Signal Reconstructions', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Confusion Matrix (second row, right columns)
        ax6 = fig.add_subplot(gs[1, 2:])
        if 'classification' in self.results:
            cm = np.array(self.results['classification']['confusion_matrix']['matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                       xticklabels=[f'Class {i}' for i in range(cm.shape[1])],
                       yticklabels=[f'Class {i}' for i in range(cm.shape[0])])
            ax6.set_title('Classification Confusion Matrix', fontweight='bold')
            ax6.set_xlabel('Predicted Class')
            ax6.set_ylabel('True Class')
        
        # 7. Threshold Prediction Scatter (third row, left)
        ax7 = fig.add_subplot(gs[2, :2])
        if len(self.predictions['thresholds']) > 0:
            pred_thresh = self.predictions['thresholds'].numpy()
            true_thresh = self.ground_truth['thresholds'].numpy()
            
            # Handle shape mismatch - if predictions have multiple outputs, use first one
            if len(pred_thresh.shape) > 1 and pred_thresh.shape[1] > 1:
                pred_thresh = pred_thresh[:, 0]  # Use first threshold output
            
            # Ensure both are 1D
            pred_thresh = pred_thresh.flatten()
            true_thresh = true_thresh.flatten()
            
            ax7.scatter(true_thresh, pred_thresh, alpha=0.6, s=30)
            
            # Perfect prediction line
            min_val, max_val = min(true_thresh.min(), pred_thresh.min()), max(true_thresh.max(), pred_thresh.max())
            ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            ax7.set_xlabel('True Threshold (dB)')
            ax7.set_ylabel('Predicted Threshold (dB)')
            ax7.set_title('Threshold Prediction Accuracy', fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Error Distribution (third row, right)
        ax8 = fig.add_subplot(gs[2, 2:])
        if len(self.predictions['thresholds']) > 0:
            pred_thresh = self.predictions['thresholds'].numpy()
            true_thresh = self.ground_truth['thresholds'].numpy()
            
            # Handle shape mismatch - if predictions have multiple outputs, use first one
            if len(pred_thresh.shape) > 1 and pred_thresh.shape[1] > 1:
                pred_thresh = pred_thresh[:, 0]  # Use first threshold output
            
            # Ensure both are 1D
            pred_thresh = pred_thresh.flatten()
            true_thresh = true_thresh.flatten()
            errors = pred_thresh - true_thresh
            
            ax8.hist(errors, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            ax8.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
            ax8.set_xlabel('Prediction Error (dB)')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Threshold Prediction Error Distribution', fontweight='bold')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Performance Summary (bottom row)
        ax9 = fig.add_subplot(gs[3, :])
        summary_metrics = self._compute_summary_metrics()
        
        # Create performance radar chart (simplified as bar chart)
        metrics = list(summary_metrics.keys())
        values = list(summary_metrics.values())
        
        # Filter numeric values
        numeric_metrics = [(m, v) for m, v in zip(metrics, values) if isinstance(v, (int, float))]
        if numeric_metrics:
            metrics, values = zip(*numeric_metrics)
            bars = ax9.bar(metrics, values, color=plt.cm.Set3(np.linspace(0, 1, len(metrics))))
            ax9.set_title('Overall Performance Summary', fontweight='bold', fontsize=16)
            ax9.set_ylim(0, 1)
            
            for bar, value in zip(bars, values):
                ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add evaluation timestamp
        timestamp = self.results.get('timestamp', 'Unknown')
        fig.text(0.02, 0.02, f'Evaluation completed: {timestamp}', fontsize=10, style='italic')
        
        plt.savefig(self.plots_dir / 'overview_dashboard.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'overview_dashboard.pdf', bbox_inches='tight')
        plt.close()
    
    def _create_signal_quality_plots(self) -> None:
        """Create detailed signal quality analysis plots."""
        if 'signal_quality' not in self.results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Signal Quality Analysis', fontsize=16, fontweight='bold')
        
        pred_signals = self.predictions['signals'].numpy()
        true_signals = self.ground_truth['signals'].numpy()
        
        # 1. Signal comparison examples
        ax = axes[0, 0]
        for i in range(min(5, len(pred_signals))):
            time_axis = np.linspace(0, 10, pred_signals.shape[-1])
            ax.plot(time_axis, true_signals[i].flatten() + i*1.5, 'k-', alpha=0.7, linewidth=1)
            ax.plot(time_axis, pred_signals[i].flatten() + i*1.5, 'r--', alpha=0.7, linewidth=1)
        ax.set_title('Signal Comparisons\n(Black: True, Red: Predicted)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV) + Offset')
        
        # 2. Correlation distribution
        ax = axes[0, 1]
        correlations = []
        for i in range(len(pred_signals)):
            corr = np.corrcoef(true_signals[i].flatten(), pred_signals[i].flatten())[0, 1]
            correlations.append(corr)
        
        ax.hist(correlations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2)
        ax.set_title(f'Signal Correlations\nMean: {np.mean(correlations):.3f}')
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('Frequency')
        
        # 3. SNR distribution
        ax = axes[0, 2]
        signal_power = np.mean(true_signals ** 2, axis=(1, 2))
        noise_power = np.mean((true_signals - pred_signals) ** 2, axis=(1, 2))
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        ax.hist(snr, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.axvline(np.mean(snr), color='red', linestyle='--', linewidth=2)
        ax.set_title(f'SNR Distribution\nMean: {np.mean(snr):.1f} dB')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Frequency')
        
        # 4. Spectral analysis
        ax = axes[1, 0]
        sample_idx = 0
        f, t, Sxx_true = spectrogram(true_signals[sample_idx].flatten(), nperseg=32)
        f, t, Sxx_pred = spectrogram(pred_signals[sample_idx].flatten(), nperseg=32)
        
        im1 = ax.imshow(10 * np.log10(Sxx_true + 1e-10), aspect='auto', origin='lower', cmap='viridis')
        ax.set_title('True Signal Spectrogram')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        
        # 5. Predicted spectrogram
        ax = axes[1, 1]
        im2 = ax.imshow(10 * np.log10(Sxx_pred + 1e-10), aspect='auto', origin='lower', cmap='viridis')
        ax.set_title('Predicted Signal Spectrogram')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        
        # 6. Error analysis
        ax = axes[1, 2]
        errors = np.mean((pred_signals - true_signals) ** 2, axis=(1, 2))
        ax.hist(errors, bins=30, alpha=0.7, color='salmon', edgecolor='black')
        ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2)
        ax.set_title(f'MSE Distribution\nMean: {np.mean(errors):.4f}')
        ax.set_xlabel('Mean Squared Error')
        ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'signal_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_classification_plots(self) -> None:
        """Create detailed classification analysis plots."""
        if 'classification' not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Classification Performance Analysis', fontsize=16, fontweight='bold')
        
        pred_classes = torch.softmax(self.predictions['classifications'], dim=1)
        pred_labels = torch.argmax(pred_classes, dim=1).numpy()
        true_labels = self.ground_truth['classifications'].numpy()
        
        # 1. Confusion Matrix
        ax = axes[0, 0]
        cm = confusion_matrix(true_labels, pred_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=[f'Class {i}' for i in range(cm.shape[1])],
                   yticklabels=[f'Class {i}' for i in range(cm.shape[0])])
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        
        # 2. Class-wise F1 Scores
        ax = axes[0, 1]
        class_report = self.results['classification']['per_class_metrics']
        classes = [k for k in class_report.keys() if k.isdigit()]
        f1_scores = [class_report[c]['f1-score'] for c in classes]
        
        bars = ax.bar(classes, f1_scores, color='lightcoral')
        ax.set_title('Per-Class F1 Scores')
        ax.set_xlabel('Class')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1)
        
        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Prediction Confidence Distribution
        ax = axes[1, 0]
        max_probs = torch.max(pred_classes, dim=1)[0].numpy()
        correct_predictions = (pred_labels == true_labels)
        
        ax.hist(max_probs[correct_predictions], bins=30, alpha=0.7, 
               label='Correct', color='lightgreen', density=True)
        ax.hist(max_probs[~correct_predictions], bins=30, alpha=0.7, 
               label='Incorrect', color='lightcoral', density=True)
        ax.set_title('Prediction Confidence Distribution')
        ax.set_xlabel('Maximum Probability')
        ax.set_ylabel('Density')
        ax.legend()
        
        # 4. Class Distribution
        ax = axes[1, 1]
        unique_true, counts_true = np.unique(true_labels, return_counts=True)
        unique_pred, counts_pred = np.unique(pred_labels, return_counts=True)
        
        x = np.arange(len(unique_true))
        width = 0.35
        
        ax.bar(x - width/2, counts_true, width, label='True', color='skyblue')
        ax.bar(x + width/2, counts_pred, width, label='Predicted', color='lightcoral')
        ax.set_title('Class Distribution Comparison')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {i}' for i in unique_true])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'classification_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_peak_detection_plots(self) -> None:
        """Create detailed peak detection analysis plots."""
        if 'peak_detection' not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Peak Detection Performance Analysis', fontsize=16, fontweight='bold')
        
        pred_peaks = self.predictions['peak_predictions'].numpy()
        true_peaks = self.ground_truth['peak_labels'].numpy()
        
        if pred_peaks.shape[-1] >= 3:
            # Extract components
            pred_exists = (pred_peaks[..., 0] > 0.5).astype(int)
            true_exists = (true_peaks[..., 0] > 0.5).astype(int)
            
            mask = true_exists.astype(bool)
            
            # 1. Peak Existence Accuracy
            ax = axes[0, 0]
            from sklearn.metrics import confusion_matrix
            cm_exists = confusion_matrix(true_exists.flatten(), pred_exists.flatten())
            sns.heatmap(cm_exists, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Peak', 'Peak'], yticklabels=['No Peak', 'Peak'])
            ax.set_title('Peak Existence Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            if np.any(mask):
                # 2. Latency Accuracy
                ax = axes[0, 1]
                pred_latencies = pred_peaks[..., 1][mask]
                true_latencies = true_peaks[..., 1][mask]
                
                ax.scatter(true_latencies, pred_latencies, alpha=0.6)
                min_lat, max_lat = min(true_latencies.min(), pred_latencies.min()), max(true_latencies.max(), pred_latencies.max())
                ax.plot([min_lat, max_lat], [min_lat, max_lat], 'r--', linewidth=2)
                ax.set_xlabel('True Latency (ms)')
                ax.set_ylabel('Predicted Latency (ms)')
                ax.set_title('Peak Latency Prediction')
                
                # 3. Amplitude Accuracy
                ax = axes[1, 0]
                pred_amplitudes = pred_peaks[..., 2][mask]
                true_amplitudes = true_peaks[..., 2][mask]
                
                ax.scatter(true_amplitudes, pred_amplitudes, alpha=0.6)
                min_amp, max_amp = min(true_amplitudes.min(), pred_amplitudes.min()), max(true_amplitudes.max(), pred_amplitudes.max())
                ax.plot([min_amp, max_amp], [min_amp, max_amp], 'r--', linewidth=2)
                ax.set_xlabel('True Amplitude (µV)')
                ax.set_ylabel('Predicted Amplitude (µV)')
                ax.set_title('Peak Amplitude Prediction')
                
                # 4. Timing Error Distribution
                ax = axes[1, 1]
                timing_errors = np.abs(pred_latencies - true_latencies)
                ax.hist(timing_errors, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
                ax.axvline(np.mean(timing_errors), color='red', linestyle='--', linewidth=2)
                ax.set_xlabel('Absolute Timing Error (ms)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Timing Error Distribution\nMean: {np.mean(timing_errors):.3f} ms')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'peak_detection_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_threshold_regression_plots(self) -> None:
        """Create detailed threshold regression analysis plots."""
        if 'threshold_regression' not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Threshold Regression Analysis', fontsize=16, fontweight='bold')
        
        pred_thresholds = self.predictions['thresholds'].numpy()
        true_thresholds = self.ground_truth['thresholds'].numpy()
        
        # Handle shape mismatch - if predictions have multiple outputs, use first one
        if len(pred_thresholds.shape) > 1 and pred_thresholds.shape[1] > 1:
            pred_thresholds = pred_thresholds[:, 0]  # Use first threshold output
        
        # Ensure both are 1D
        pred_thresholds = pred_thresholds.flatten()
        true_thresholds = true_thresholds.flatten()
        
        # 1. Scatter plot with perfect prediction line
        ax = axes[0, 0]
        ax.scatter(true_thresholds, pred_thresholds, alpha=0.6, s=20)
        min_val, max_val = min(true_thresholds.min(), pred_thresholds.min()), max(true_thresholds.max(), pred_thresholds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('True Threshold (dB)')
        ax.set_ylabel('Predicted Threshold (dB)')
        ax.set_title('Threshold Prediction Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Bland-Altman plot
        ax = axes[0, 1]
        mean_thresh = (pred_thresholds + true_thresholds) / 2
        diff_thresh = pred_thresholds - true_thresholds
        
        ax.scatter(mean_thresh, diff_thresh, alpha=0.6, s=20)
        
        bias = np.mean(diff_thresh)
        std_diff = np.std(diff_thresh)
        
        ax.axhline(bias, color='red', linestyle='-', linewidth=2, label=f'Bias: {bias:.2f}')
        ax.axhline(bias + 1.96 * std_diff, color='red', linestyle='--', linewidth=1, label=f'+1.96σ: {bias + 1.96 * std_diff:.2f}')
        ax.axhline(bias - 1.96 * std_diff, color='red', linestyle='--', linewidth=1, label=f'-1.96σ: {bias - 1.96 * std_diff:.2f}')
        
        ax.set_xlabel('Mean Threshold (dB)')
        ax.set_ylabel('Difference (Pred - True) (dB)')
        ax.set_title('Bland-Altman Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Error distribution
        ax = axes[1, 0]
        errors = np.abs(pred_thresholds - true_thresholds)
        ax.hist(errors, bins=30, alpha=0.7, color='salmon', edgecolor='black')
        ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
        ax.set_xlabel('Absolute Error (dB)')
        ax.set_ylabel('Frequency')
        ax.set_title('Threshold Prediction Error Distribution')
        ax.legend()
        
        # 4. Clinical category accuracy
        ax = axes[1, 1]
        def classify_hearing_loss(threshold):
            if threshold <= 20:
                return 'Normal'
            elif threshold <= 40:
                return 'Mild'
            elif threshold <= 70:
                return 'Moderate'
            elif threshold <= 90:
                return 'Severe'
            else:
                return 'Profound'
        
        true_categories = [classify_hearing_loss(t) for t in true_thresholds]
        pred_categories = [classify_hearing_loss(t) for t in pred_thresholds]
        
        from sklearn.metrics import confusion_matrix
        categories = ['Normal', 'Mild', 'Moderate', 'Severe', 'Profound']
        cm_clinical = confusion_matrix(true_categories, pred_categories, labels=categories)
        
        sns.heatmap(cm_clinical, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=categories, yticklabels=categories)
        ax.set_title('Clinical Category Confusion Matrix')
        ax.set_xlabel('Predicted Category')
        ax.set_ylabel('True Category')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'threshold_regression_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_uncertainty_plots(self) -> None:
        """Create uncertainty analysis plots."""
        if 'uncertainty' not in self.results or self.results['uncertainty'].get('message'):
            return
        
        uncertainties = self.predictions['uncertainties'].numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Uncertainty Analysis', fontsize=16, fontweight='bold')
        
        # 1. Uncertainty distribution
        ax = axes[0]
        if len(uncertainties.shape) > 1:
            uncertainty_values = np.mean(uncertainties, axis=tuple(range(1, len(uncertainties.shape))))
        else:
            uncertainty_values = uncertainties
        
        ax.hist(uncertainty_values, bins=30, alpha=0.7, color='lightpurple', edgecolor='black')
        ax.axvline(np.mean(uncertainty_values), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Uncertainty Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Uncertainty Distribution\nMean: {np.mean(uncertainty_values):.4f}')
        
        # 2. Uncertainty vs Error correlation
        ax = axes[1]
        pred_signals = self.predictions['signals'].numpy()
        true_signals = self.ground_truth['signals'].numpy()
        signal_errors = np.mean((pred_signals - true_signals) ** 2, axis=(1, 2))
        
        ax.scatter(uncertainty_values, signal_errors, alpha=0.6)
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('Signal MSE')
        ax.set_title('Uncertainty vs Prediction Error')
        
        # Add correlation coefficient
        corr = np.corrcoef(uncertainty_values, signal_errors)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_clinical_plots(self) -> None:
        """Create clinical correlation plots."""
        if 'clinical_analysis' not in self.results:
            return
        
        # Create clinical analysis plots here
        # This would include audiogram-style plots, clinical correlation analysis, etc.
        pass
    
    def _create_interactive_visualizations(self) -> None:
        """Create interactive visualizations using Plotly."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping interactive visualizations")
            return
        
        print("🎨 Creating interactive visualizations...")
        
        # 1. Interactive dashboard
        self._create_interactive_dashboard()
        
        # 2. Interactive signal explorer
        self._create_interactive_signal_explorer()
        
        # 3. Interactive performance explorer
        self._create_interactive_performance_explorer()
        
        print(f"✅ Interactive visualizations saved to {self.plots_dir}")
    
    def _create_interactive_dashboard(self) -> None:
        """Create interactive dashboard with Plotly."""
        if not PLOTLY_AVAILABLE:
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Signal Quality', 'Classification Performance', 
                          'Peak Detection', 'Threshold Regression'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Add plots for each metric type
        # (Implementation would continue here...)
        
        fig.update_layout(
            title="ABR Model Interactive Evaluation Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save interactive plot
        pyo.plot(fig, filename=str(self.plots_dir / 'interactive_dashboard.html'), auto_open=False)
    
    def _create_interactive_signal_explorer(self) -> None:
        """Create interactive signal exploration tool."""
        if not PLOTLY_AVAILABLE:
            return
        
        # Implementation for interactive signal explorer
        pass
    
    def _create_interactive_performance_explorer(self) -> None:
        """Create interactive performance exploration tool."""
        if not PLOTLY_AVAILABLE:
            return
        
        # Implementation for interactive performance explorer
        pass
    
    def _save_comprehensive_report(self, results: Dict[str, Any]) -> None:
        """Save comprehensive evaluation report."""
        # Save JSON report
        with open(self.reports_dir / 'evaluation_report.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary report
        self._save_summary_report(results)
        
        # Save detailed metrics
        self._save_detailed_metrics(results)
    
    def _save_summary_report(self, results: Dict[str, Any]) -> None:
        """Save human-readable summary report."""
        report_content = f"""
# ABR Model Evaluation Report

## Executive Summary
Evaluation completed on: {results['timestamp']}
Evaluation time: {results['evaluation_time_seconds']:.2f} seconds

## Model Information
- Model: {results['model_info']['model_class']}
- Total Parameters: {results['model_info']['total_parameters']:,}
- Trainable Parameters: {results['model_info']['trainable_parameters']:,}

## Dataset Information
- Number of samples: {results['dataset_info']['num_samples']}
- Signal shape: {results['dataset_info']['signal_shape']}
- Number of classes: {results['dataset_info']['num_classes']}

## Performance Summary
"""
        
        # Add performance metrics
        for task, metrics in results.items():
            if task in ['signal_quality', 'classification', 'peak_detection', 'threshold_regression']:
                report_content += f"\n### {task.replace('_', ' ').title()}\n"
                if isinstance(metrics, dict):
                    for category, values in metrics.items():
                        if isinstance(values, dict):
                            report_content += f"**{category.replace('_', ' ').title()}:**\n"
                            for metric, value in values.items():
                                if isinstance(value, (int, float)):
                                    report_content += f"- {metric}: {value:.4f}\n"
        
        # Add recommendations
        if 'recommendations' in results:
            report_content += "\n## Recommendations\n"
            for i, rec in enumerate(results['recommendations'], 1):
                report_content += f"{i}. {rec}\n"
        
        with open(self.reports_dir / 'summary_report.md', 'w') as f:
            f.write(report_content)
    
    def _save_detailed_metrics(self, results: Dict[str, Any]) -> None:
        """Save detailed metrics as CSV files."""
        # Save summary metrics
        summary_df = pd.DataFrame([results['summary_metrics']])
        summary_df.to_csv(self.data_dir / 'summary_metrics.csv', index=False)
        
        # Save predictions and ground truth
        if len(self.predictions['signals']) > 0:
            # Handle threshold shape mismatch
            pred_thresholds = self.predictions['thresholds'].numpy()
            true_thresholds = self.ground_truth['thresholds'].numpy()
            
            # Handle shape mismatch - if predictions have multiple outputs, use first one
            if len(pred_thresholds.shape) > 1 and pred_thresholds.shape[1] > 1:
                pred_thresholds = pred_thresholds[:, 0]  # Use first threshold output
            
            # Flatten and save predictions
            pred_data = {
                'sample_id': range(len(self.predictions['signals'])),
                'predicted_class': torch.argmax(self.predictions['classifications'], dim=1).numpy(),
                'true_class': self.ground_truth['classifications'].numpy(),
                'predicted_threshold': pred_thresholds.flatten(),
                'true_threshold': true_thresholds.flatten()
            }
            
            pred_df = pd.DataFrame(pred_data)
            pred_df.to_csv(self.data_dir / 'predictions.csv', index=False)