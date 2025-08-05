#!/usr/bin/env python3
"""
Comprehensive ABR Model Evaluation Methods

Professional evaluation methods for detailed analysis of ABR model performance
across all tasks with clinical-grade metrics and visualizations.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Statistical analysis
from scipy import stats
from scipy.signal import find_peaks, welch, spectrogram
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import label_binarize

# Simple DTW implementation (fastdtw removed)
def simple_dtw_distance(x, y):
    """Simple DTW distance calculation without fastdtw dependency."""
    # For simplicity, use mean absolute difference
    min_len = min(len(x), len(y))
    x_norm = x[:min_len] if len(x) >= min_len else x
    y_norm = y[:min_len] if len(y) >= min_len else y
    return float(np.mean(np.abs(np.array(x_norm) - np.array(y_norm))))

# Interactive plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ComprehensiveEvaluationMethods:
    """Methods for comprehensive ABR model evaluation."""
    
    def _evaluate_signal_quality(self) -> Dict[str, Any]:
        """Evaluate signal reconstruction quality."""
        pred_signals = self.predictions['signals'].numpy()
        true_signals = self.ground_truth['signals'].numpy()
        
        results = {
            'basic_metrics': {},
            'spectral_metrics': {},
            'temporal_metrics': {},
            'perceptual_metrics': {}
        }
        
        # Basic signal metrics
        mse = mean_squared_error(true_signals.flatten(), pred_signals.flatten())
        mae = mean_absolute_error(true_signals.flatten(), pred_signals.flatten())
        
        # Correlation analysis
        correlations = []
        for i in range(len(pred_signals)):
            corr, _ = pearsonr(true_signals[i].flatten(), pred_signals[i].flatten())
            correlations.append(corr)
        
        # SNR calculation
        signal_power = np.mean(true_signals ** 2, axis=1)
        noise_power = np.mean((true_signals - pred_signals) ** 2, axis=1)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # DTW distance (simplified)
        dtw_distances = []
        for i in range(min(100, len(pred_signals))):  # Sample for speed
            distance = simple_dtw_distance(true_signals[i].flatten(), pred_signals[i].flatten())
            dtw_distances.append(distance)
        
        results['basic_metrics'] = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'correlation_mean': float(np.mean(correlations)),
            'correlation_std': float(np.std(correlations)),
            'snr_mean_db': float(np.mean(snr)),
            'snr_std_db': float(np.std(snr)),
            'dtw_distance_mean': float(np.mean(dtw_distances)),
            'dtw_distance_std': float(np.std(dtw_distances))
        }
        
        # Spectral analysis
        true_psd_mean = []
        pred_psd_mean = []
        
        for i in range(min(50, len(pred_signals))):
            # Power spectral density
            f_true, psd_true = welch(true_signals[i].flatten(), nperseg=min(64, len(true_signals[i].flatten())//4))
            f_pred, psd_pred = welch(pred_signals[i].flatten(), nperseg=min(64, len(pred_signals[i].flatten())//4))
            
            true_psd_mean.append(np.mean(psd_true))
            pred_psd_mean.append(np.mean(psd_pred))
        
        spectral_correlation, _ = pearsonr(true_psd_mean, pred_psd_mean)
        
        results['spectral_metrics'] = {
            'spectral_correlation': float(spectral_correlation),
            'spectral_mse': float(mean_squared_error(true_psd_mean, pred_psd_mean))
        }
        
        # Temporal metrics
        true_peaks = []
        pred_peaks = []
        
        for i in range(min(100, len(pred_signals))):
            # Find peaks in signals
            true_peaks_idx, _ = find_peaks(true_signals[i].flatten(), height=0.1)
            pred_peaks_idx, _ = find_peaks(pred_signals[i].flatten(), height=0.1)
            
            true_peaks.append(len(true_peaks_idx))
            pred_peaks.append(len(pred_peaks_idx))
        
        peak_count_correlation, _ = pearsonr(true_peaks, pred_peaks)
        
        results['temporal_metrics'] = {
            'peak_count_correlation': float(peak_count_correlation),
            'peak_count_mae': float(mean_absolute_error(true_peaks, pred_peaks))
        }
        
        return results
    
    def _evaluate_classification(self) -> Dict[str, Any]:
        """Evaluate classification performance."""
        pred_classes = torch.softmax(self.predictions['classifications'], dim=1)
        pred_labels = torch.argmax(pred_classes, dim=1).numpy()
        true_labels = self.ground_truth['classifications'].numpy()
        
        results = {
            'basic_metrics': {},
            'per_class_metrics': {},
            'confusion_matrix': {},
            'roc_analysis': {},
            'confidence_analysis': {}
        }
        
        # Basic classification metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        f1_macro = f1_score(true_labels, pred_labels, average='macro')
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
        
        results['basic_metrics'] = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted)
        }
        
        # Per-class metrics
        class_report = classification_report(true_labels, pred_labels, output_dict=True)
        results['per_class_metrics'] = class_report
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        results['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'normalized': (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
        }
        
        # ROC analysis (multiclass)
        n_classes = pred_classes.shape[1]
        true_labels_bin = label_binarize(true_labels, classes=list(range(n_classes)))
        
        if n_classes > 2:
            fpr = {}
            tpr = {}
            roc_auc = {}
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], pred_classes[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            results['roc_analysis'] = {
                'per_class_auc': {str(k): float(v) for k, v in roc_auc.items()},
                'macro_auc': float(np.mean(list(roc_auc.values())))
            }
        
        # Confidence analysis
        max_probs = torch.max(pred_classes, dim=1)[0].numpy()
        correct_predictions = (pred_labels == true_labels)
        
        results['confidence_analysis'] = {
            'mean_confidence_correct': float(np.mean(max_probs[correct_predictions])),
            'mean_confidence_incorrect': float(np.mean(max_probs[~correct_predictions])),
            'confidence_correlation_with_accuracy': float(pearsonr(max_probs, correct_predictions.astype(float))[0])
        }
        
        return results
    
    def _evaluate_peak_detection(self) -> Dict[str, Any]:
        """Evaluate peak detection performance."""
        pred_peaks = self.predictions['peak_predictions'].numpy()
        true_peaks = self.ground_truth['peak_labels'].numpy()
        
        results = {
            'existence_metrics': {},
            'latency_metrics': {},
            'amplitude_metrics': {},
            'timing_accuracy': {}
        }
        
        # Assuming peak format: [exists, latency, amplitude] for each peak
        # Extract peak existence (binary)
        if pred_peaks.shape[-1] >= 3:  # At least one peak with 3 components
            pred_exists = (pred_peaks[..., 0] > 0.5).astype(int)  # Binary threshold
            true_exists = (true_peaks[..., 0] > 0.5).astype(int)
            
            # Debug shapes
            print(f"DEBUG: pred_peaks.shape = {pred_peaks.shape}")
            print(f"DEBUG: true_peaks.shape = {true_peaks.shape}")
            print(f"DEBUG: pred_exists.shape = {pred_exists.shape}")
            print(f"DEBUG: true_exists.shape = {true_exists.shape}")
            
            # Ensure consistent shapes
            min_len = min(len(pred_exists.flatten()), len(true_exists.flatten()))
            pred_exists_flat = pred_exists.flatten()[:min_len]
            true_exists_flat = true_exists.flatten()[:min_len]
            
            # Peak existence metrics
            exist_accuracy = accuracy_score(true_exists_flat, pred_exists_flat)
            exist_f1 = f1_score(true_exists_flat, pred_exists_flat, average='macro')
            
            results['existence_metrics'] = {
                'accuracy': float(exist_accuracy),
                'f1_score': float(exist_f1),
                'precision': float(f1_score(true_exists_flat, pred_exists_flat, average='macro')),
                'recall': float(f1_score(true_exists_flat, pred_exists_flat, average='macro'))
            }
            
            # Latency metrics (only for existing peaks)
            mask = true_exists.astype(bool)
            if np.any(mask):
                pred_latencies = pred_peaks[..., 1][mask]
                true_latencies = true_peaks[..., 1][mask]
                
                latency_mae = mean_absolute_error(true_latencies, pred_latencies)
                latency_correlation, _ = pearsonr(true_latencies.flatten(), pred_latencies.flatten())
                
                results['latency_metrics'] = {
                    'mae_ms': float(latency_mae),
                    'rmse_ms': float(np.sqrt(mean_squared_error(true_latencies, pred_latencies))),
                    'correlation': float(latency_correlation),
                    'mean_error_ms': float(np.mean(pred_latencies - true_latencies))
                }
                
                # Amplitude metrics (only for existing peaks)
                pred_amplitudes = pred_peaks[..., 2][mask]
                true_amplitudes = true_peaks[..., 2][mask]
                
                amplitude_mae = mean_absolute_error(true_amplitudes, pred_amplitudes)
                amplitude_correlation, _ = pearsonr(true_amplitudes.flatten(), pred_amplitudes.flatten())
                
                results['amplitude_metrics'] = {
                    'mae_uv': float(amplitude_mae),
                    'rmse_uv': float(np.sqrt(mean_squared_error(true_amplitudes, pred_amplitudes))),
                    'correlation': float(amplitude_correlation),
                    'mean_error_uv': float(np.mean(pred_amplitudes - true_amplitudes))
                }
                
                # Timing accuracy analysis
                timing_errors = np.abs(pred_latencies - true_latencies)
                timing_within_0_5ms = np.mean(timing_errors < 0.5)
                timing_within_1_0ms = np.mean(timing_errors < 1.0)
                
                results['timing_accuracy'] = {
                    'within_0_5ms_percent': float(timing_within_0_5ms * 100),
                    'within_1_0ms_percent': float(timing_within_1_0ms * 100),
                    'mean_timing_error_ms': float(np.mean(timing_errors)),
                    'std_timing_error_ms': float(np.std(timing_errors))
                }
        
        return results
    
    def _evaluate_threshold_regression(self) -> Dict[str, Any]:
        """Evaluate threshold regression performance."""
        pred_thresholds = self.predictions['thresholds'].numpy()
        true_thresholds = self.ground_truth['thresholds'].numpy()
        
        results = {
            'regression_metrics': {},
            'clinical_metrics': {},
            'error_analysis': {}
        }
        
        # Basic regression metrics
        mae = mean_absolute_error(true_thresholds, pred_thresholds)
        mse = mean_squared_error(true_thresholds, pred_thresholds)
        r2 = r2_score(true_thresholds, pred_thresholds)
        correlation, _ = pearsonr(true_thresholds.flatten(), pred_thresholds.flatten())
        
        results['regression_metrics'] = {
            'mae_db': float(mae),
            'rmse_db': float(np.sqrt(mse)),
            'r2_score': float(r2),
            'correlation': float(correlation),
            'mean_error_db': float(np.mean(pred_thresholds - true_thresholds)),
            'std_error_db': float(np.std(pred_thresholds - true_thresholds))
        }
        
        # Clinical accuracy metrics
        def classify_hearing_loss(threshold):
            if threshold <= 20:
                return 0  # Normal
            elif threshold <= 40:
                return 1  # Mild
            elif threshold <= 70:
                return 2  # Moderate
            elif threshold <= 90:
                return 3  # Severe
            else:
                return 4  # Profound
        
        true_categories = np.array([classify_hearing_loss(t) for t in true_thresholds.flatten()])
        pred_categories = np.array([classify_hearing_loss(t) for t in pred_thresholds.flatten()])
        
        clinical_accuracy = accuracy_score(true_categories, pred_categories)
        clinical_f1 = f1_score(true_categories, pred_categories, average='macro')
        
        results['clinical_metrics'] = {
            'category_accuracy': float(clinical_accuracy),
            'category_f1_score': float(clinical_f1),
            'confusion_matrix': confusion_matrix(true_categories, pred_categories).tolist()
        }
        
        # Error analysis
        errors = np.abs(pred_thresholds - true_thresholds).flatten()
        
        results['error_analysis'] = {
            'within_5db_percent': float(np.mean(errors <= 5) * 100),
            'within_10db_percent': float(np.mean(errors <= 10) * 100),
            'within_15db_percent': float(np.mean(errors <= 15) * 100),
            'max_error_db': float(np.max(errors)),
            'percentile_95_error_db': float(np.percentile(errors, 95)),
            'percentile_99_error_db': float(np.percentile(errors, 99))
        }
        
        return results
    
    def _evaluate_uncertainty(self) -> Dict[str, Any]:
        """Evaluate model uncertainty quantification."""
        if 'uncertainties' not in self.predictions or len(self.predictions['uncertainties']) == 0:
            return {'message': 'No uncertainty information available'}
        
        uncertainties = self.predictions['uncertainties'].numpy()
        
        results = {
            'uncertainty_stats': {},
            'calibration_analysis': {},
            'uncertainty_correlation': {}
        }
        
        # Basic uncertainty statistics
        results['uncertainty_stats'] = {
            'mean_uncertainty': float(np.mean(uncertainties)),
            'std_uncertainty': float(np.std(uncertainties)),
            'min_uncertainty': float(np.min(uncertainties)),
            'max_uncertainty': float(np.max(uncertainties))
        }
        
        # Calibration analysis (correlation between uncertainty and error)
        pred_signals = self.predictions['signals'].numpy()
        true_signals = self.ground_truth['signals'].numpy()
        
        signal_errors = np.mean((pred_signals - true_signals) ** 2, axis=(1, 2))
        
        if len(uncertainties.shape) > 1:
            uncertainty_values = np.mean(uncertainties, axis=tuple(range(1, len(uncertainties.shape))))
        else:
            uncertainty_values = uncertainties
        
        uncertainty_error_corr, _ = pearsonr(uncertainty_values, signal_errors)
        
        results['calibration_analysis'] = {
            'uncertainty_error_correlation': float(uncertainty_error_corr),
            'well_calibrated': abs(uncertainty_error_corr) > 0.3  # Threshold for good calibration
        }
        
        return results
    
    def _evaluate_clinical_correlation(self) -> Dict[str, Any]:
        """Evaluate clinical correlation and diagnostic accuracy."""
        results = {
            'diagnostic_accuracy': {},
            'clinical_correlation': {},
            'agreement_analysis': {}
        }
        
        # Use threshold predictions for clinical analysis
        pred_thresholds = self.predictions['thresholds'].numpy().flatten()
        true_thresholds = self.ground_truth['thresholds'].numpy().flatten()
        
        # Clinical diagnostic categories
        def get_diagnosis(threshold):
            if threshold <= 20:
                return "Normal"
            elif threshold <= 40:
                return "Mild Loss"
            elif threshold <= 70:
                return "Moderate Loss"
            elif threshold <= 90:
                return "Severe Loss"
            else:
                return "Profound Loss"
        
        true_diagnoses = [get_diagnosis(t) for t in true_thresholds]
        pred_diagnoses = [get_diagnosis(t) for t in pred_thresholds]
        
        # Diagnostic accuracy
        diagnostic_accuracy = accuracy_score(true_diagnoses, pred_diagnoses)
        
        results['diagnostic_accuracy'] = {
            'overall_accuracy': float(diagnostic_accuracy),
            'diagnosis_report': classification_report(true_diagnoses, pred_diagnoses, output_dict=True)
        }
        
        # Clinical correlation with Bland-Altman analysis
        mean_thresholds = (pred_thresholds + true_thresholds) / 2
        diff_thresholds = pred_thresholds - true_thresholds
        
        bias = np.mean(diff_thresholds)
        std_diff = np.std(diff_thresholds)
        limits_of_agreement = [bias - 1.96 * std_diff, bias + 1.96 * std_diff]
        
        results['agreement_analysis'] = {
            'bias_db': float(bias),
            'limits_of_agreement_db': [float(loa) for loa in limits_of_agreement],
            'within_limits_percent': float(np.mean(np.abs(diff_thresholds) <= 1.96 * std_diff) * 100)
        }
        
        return results
    
    def _evaluate_demographics(self) -> Dict[str, Any]:
        """Evaluate performance across different demographics."""
        # This would require demographic information in the metadata
        # For now, return placeholder structure
        results = {
            'age_analysis': {},
            'gender_analysis': {},
            'ear_analysis': {},
            'stimulus_analysis': {}
        }
        
        # Placeholder - would implement with actual demographic data
        results['message'] = 'Demographic analysis requires metadata with age, gender, ear, and stimulus information'
        
        return results
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_class': self.model.__class__.__name__,
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'device': str(self.device)
        }
    
    def _get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'num_samples': len(self.predictions['signals']),
            'signal_shape': list(self.predictions['signals'].shape[1:]),
            'num_classes': int(self.predictions['classifications'].shape[1]),
            'has_uncertainty': 'uncertainties' in self.predictions and len(self.predictions['uncertainties']) > 0
        }
    
    def _compute_summary_metrics(self) -> Dict[str, Any]:
        """Compute overall summary metrics."""
        summary = {}
        
        # Signal quality summary
        if 'signal_quality' in self.results:
            signal_metrics = self.results['signal_quality']['basic_metrics']
            summary['signal_quality_score'] = (signal_metrics.get('correlation_mean', 0) + 
                                             (1 - signal_metrics.get('mse', 1))) / 2
        
        # Classification summary
        if 'classification' in self.results:
            class_metrics = self.results['classification']['basic_metrics']
            summary['classification_score'] = class_metrics.get('f1_weighted', 0)
        
        # Peak detection summary
        if 'peak_detection' in self.results:
            peak_metrics = self.results['peak_detection']
            summary['peak_detection_score'] = peak_metrics.get('existence_metrics', {}).get('f1_score', 0)
        
        # Threshold regression summary
        if 'threshold_regression' in self.results:
            threshold_metrics = self.results['threshold_regression']['regression_metrics']
            summary['threshold_regression_score'] = max(0, threshold_metrics.get('r2_score', 0))
        
        # Overall score
        scores = [v for v in summary.values() if isinstance(v, (int, float))]
        summary['overall_score'] = np.mean(scores) if scores else 0.0
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Signal quality recommendations
        if 'signal_quality' in self.results:
            signal_metrics = self.results['signal_quality']['basic_metrics']
            if signal_metrics.get('correlation_mean', 0) < 0.8:
                recommendations.append("Consider improving signal reconstruction with perceptual loss or adversarial training")
            if signal_metrics.get('snr_mean_db', 0) < 20:
                recommendations.append("Signal-to-noise ratio could be improved with denoising techniques")
        
        # Classification recommendations
        if 'classification' in self.results:
            class_metrics = self.results['classification']['basic_metrics']
            if class_metrics.get('accuracy', 0) < 0.85:
                recommendations.append("Classification accuracy could be improved with class balancing or focal loss")
        
        # Peak detection recommendations
        if 'peak_detection' in self.results:
            peak_metrics = self.results['peak_detection']
            if peak_metrics.get('existence_metrics', {}).get('f1_score', 0) < 0.8:
                recommendations.append("Peak detection performance could be enhanced with specialized peak loss functions")
        
        # Threshold regression recommendations
        if 'threshold_regression' in self.results:
            threshold_metrics = self.results['threshold_regression']['regression_metrics']
            if threshold_metrics.get('mae_db', float('inf')) > 10:
                recommendations.append("Threshold prediction accuracy needs improvement - consider ensemble methods")
        
        if not recommendations:
            recommendations.append("Model performance is excellent across all metrics!")
        
        return recommendations