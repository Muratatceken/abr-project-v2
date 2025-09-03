"""
Signal Analysis Tools

This module provides specialized analysis tools for ABR signals,
including physiological analysis, statistical tests, and quality assessments.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats, signal
from scipy.signal import find_peaks
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.stats import bootstrap
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import warnings


class SignalAnalyzer:
    """Comprehensive signal analysis tools for ABR evaluation."""
    
    def __init__(self):
        """Initialize the signal analyzer."""
        self.abr_wave_latencies = {
            'wave_I': (1.0, 2.0),    # ms
            'wave_III': (3.3, 4.3),  # ms  
            'wave_V': (5.1, 6.1)     # ms
        }
    
    def analyze_signal_properties(self, signal: torch.Tensor, sr: int = 1000) -> Dict[str, float]:
        """
        Analyze basic signal properties.
        
        Args:
            signal: Input signal tensor
            sr: Sampling rate
            
        Returns:
            Dictionary of signal properties
        """
        signal_np = signal.squeeze().cpu().numpy()
        
        properties = {
            # Amplitude properties
            'rms': float(np.sqrt(np.mean(signal_np**2))),
            'peak_amplitude': float(np.max(np.abs(signal_np))),
            'dynamic_range': float(np.max(signal_np) - np.min(signal_np)),
            'mean_amplitude': float(np.mean(signal_np)),
            'std_amplitude': float(np.std(signal_np)),
            
            # Shape properties
            'skewness': float(stats.skew(signal_np)),
            'kurtosis': float(stats.kurtosis(signal_np)),
            'zero_crossings': int(len(np.where(np.diff(np.signbit(signal_np)))[0])),
            
            # Energy properties
            'total_energy': float(np.sum(signal_np**2)),
            'signal_power': float(np.mean(signal_np**2)),
            
            # Frequency properties
            'dominant_frequency': self._estimate_dominant_frequency(signal_np, sr),
            'bandwidth': self._estimate_bandwidth(signal_np, sr),
        }
        
        return properties
    
    def analyze_abr_waves(self, signal: torch.Tensor, sr: int = 1000) -> Dict[str, Any]:
        """
        Analyze ABR-specific wave components.
        
        Args:
            signal: ABR signal tensor
            sr: Sampling rate
            
        Returns:
            Dictionary of ABR wave analysis results
        """
        signal_np = signal.squeeze().cpu().numpy()
        
        analysis = {
            'detected_waves': {},
            'wave_intervals': {},
            'interpeak_latencies': {},
            'amplitude_ratios': {}
        }
        
        # Detect peaks in typical ABR wave regions
        for wave_name, (start_ms, end_ms) in self.abr_wave_latencies.items():
            start_idx = int(start_ms * sr / 1000)
            end_idx = int(end_ms * sr / 1000)
            
            if end_idx < len(signal_np):
                wave_segment = signal_np[start_idx:end_idx]
                
                # Find peaks in this segment
                min_distance = max(1, int(0.5 * sr / 1000))  # Ensure minimum distance >= 1
                peaks, properties = find_peaks(np.abs(wave_segment), 
                                             height=np.std(signal_np) * 0.5,
                                             distance=min_distance)
                
                if len(peaks) > 0:
                    # Get the most prominent peak
                    peak_heights = np.abs(wave_segment[peaks])
                    main_peak_idx = peaks[np.argmax(peak_heights)]
                    peak_latency_ms = (start_idx + main_peak_idx) / sr * 1000
                    peak_amplitude = wave_segment[main_peak_idx]
                    
                    analysis['detected_waves'][wave_name] = {
                        'latency_ms': float(peak_latency_ms),
                        'amplitude': float(peak_amplitude),
                        'relative_position': float(main_peak_idx / len(wave_segment))
                    }
        
        # Calculate interpeak latencies
        waves = list(analysis['detected_waves'].keys())
        for i in range(len(waves)):
            for j in range(i+1, len(waves)):
                wave1, wave2 = waves[i], waves[j]
                if wave1 in analysis['detected_waves'] and wave2 in analysis['detected_waves']:
                    latency_diff = (analysis['detected_waves'][wave2]['latency_ms'] - 
                                  analysis['detected_waves'][wave1]['latency_ms'])
                    analysis['interpeak_latencies'][f'{wave1}_to_{wave2}'] = float(latency_diff)
        
        # Calculate amplitude ratios
        for i in range(len(waves)):
            for j in range(i+1, len(waves)):
                wave1, wave2 = waves[i], waves[j]
                if wave1 in analysis['detected_waves'] and wave2 in analysis['detected_waves']:
                    amp1 = analysis['detected_waves'][wave1]['amplitude']
                    amp2 = analysis['detected_waves'][wave2]['amplitude']
                    ratio = amp1 / max(abs(amp2), 1e-6)
                    analysis['amplitude_ratios'][f'{wave1}_to_{wave2}'] = float(ratio)
        
        return analysis
    
    def analyze_conditional_response(self, 
                                   generated_signal: torch.Tensor,
                                   conditions: torch.Tensor,
                                   condition_variation: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze how the generated signal responds to conditional inputs.
        
        Args:
            generated_signal: Generated signal tensor
            conditions: Condition parameters tensor
            condition_variation: Dictionary of condition variations applied
            
        Returns:
            Dictionary of conditional response analysis
        """
        signal_props = self.analyze_signal_properties(generated_signal)
        
        response_analysis = {
            'condition_params': conditions.squeeze().cpu().numpy().tolist(),
            'applied_variation': condition_variation,
            'signal_properties': signal_props,
            'response_strength': 0.0,
            'expected_response': {}
        }
        
        # Analyze expected responses based on condition variations
        for param_name, param_value in condition_variation.items():
            if param_name == 'hearing_loss':
                # Higher hearing loss should correlate with reduced amplitude
                expected_amplitude_reduction = param_value * 0.1  # Simple heuristic
                actual_amplitude = signal_props['peak_amplitude']
                response_analysis['expected_response'][param_name] = {
                    'expected_effect': 'amplitude_reduction',
                    'expected_value': expected_amplitude_reduction,
                    'actual_amplitude': actual_amplitude
                }
            elif param_name == 'age':
                # Older age might correlate with increased latency
                expected_latency_increase = (param_value - 30) * 0.01  # Simple heuristic
                response_analysis['expected_response'][param_name] = {
                    'expected_effect': 'latency_increase',
                    'expected_value': expected_latency_increase
                }
        
        return response_analysis
    
    def analyze_generation_consistency(self, generated_samples: List[torch.Tensor]) -> Dict[str, float]:
        """
        Analyze consistency across multiple generations with same conditions.
        
        Args:
            generated_samples: List of generated signal tensors
            
        Returns:
            Dictionary of consistency metrics
        """
        if len(generated_samples) < 2:
            return {'error': 'Need at least 2 samples for consistency analysis'}
        
        # Convert to numpy arrays
        samples_np = [s.squeeze().cpu().numpy() for s in generated_samples]
        sample_matrix = np.array(samples_np)
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(samples_np)):
            for j in range(i+1, len(samples_np)):
                corr, _ = stats.pearsonr(samples_np[i], samples_np[j])
                correlations.append(corr)
        
        # Calculate signal property consistency
        properties_list = [self.analyze_signal_properties(torch.tensor(s)) for s in samples_np]
        
        property_consistency = {}
        for prop_name in properties_list[0].keys():
            values = [props[prop_name] for props in properties_list]
            property_consistency[f'{prop_name}_mean'] = float(np.mean(values))
            property_consistency[f'{prop_name}_std'] = float(np.std(values))
            property_consistency[f'{prop_name}_cv'] = float(np.std(values) / (np.mean(values) + 1e-6))
        
        consistency_analysis = {
            'num_samples': len(generated_samples),
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'min_correlation': float(np.min(correlations)),
            'max_correlation': float(np.max(correlations)),
            'property_consistency': property_consistency,
            'overall_consistency_score': float(np.mean(correlations))
        }
        
        return consistency_analysis
    
    def statistical_comparison(self, 
                             generated_samples: List[torch.Tensor],
                             real_samples: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Perform statistical tests comparing generated and real samples.
        
        Args:
            generated_samples: List of generated signal tensors
            real_samples: List of real signal tensors
            
        Returns:
            Dictionary of statistical test results
        """
        # Convert to numpy
        gen_props = [self.analyze_signal_properties(s) for s in generated_samples]
        real_props = [self.analyze_signal_properties(s) for s in real_samples]
        
        statistical_tests = {}
        
        # Test each property
        for prop_name in gen_props[0].keys():
            gen_values = [props[prop_name] for props in gen_props]
            real_values = [props[prop_name] for props in real_props]
            
            # Two-sample t-test
            t_stat, t_p = stats.ttest_ind(gen_values, real_values)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p = stats.mannwhitneyu(gen_values, real_values, alternative='two-sided')
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(gen_values, real_values)
            
            statistical_tests[prop_name] = {
                't_test': {'statistic': float(t_stat), 'p_value': float(t_p)},
                'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_p)},
                'ks_test': {'statistic': float(ks_stat), 'p_value': float(ks_p)},
                'generated_mean': float(np.mean(gen_values)),
                'real_mean': float(np.mean(real_values)),
                'effect_size': float(abs(np.mean(gen_values) - np.mean(real_values)) / 
                                   np.sqrt((np.var(gen_values) + np.var(real_values)) / 2))
            }
        
        return statistical_tests
    
    def dimensionality_analysis(self, 
                              generated_samples: List[torch.Tensor],
                              real_samples: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze dimensionality and distribution of generated vs real samples.
        
        Args:
            generated_samples: List of generated signal tensors
            real_samples: List of real signal tensors
            
        Returns:
            Dictionary of dimensionality analysis results
        """
        # Combine and flatten samples
        gen_matrix = torch.stack(generated_samples).cpu().numpy()
        real_matrix = torch.stack(real_samples).cpu().numpy()
        
        gen_flat = gen_matrix.reshape(gen_matrix.shape[0], -1)
        real_flat = real_matrix.reshape(real_matrix.shape[0], -1)
        
        # PCA analysis
        combined_data = np.vstack([gen_flat, real_flat])
        pca = PCA(n_components=min(10, combined_data.shape[1]))
        pca_transformed = pca.fit_transform(combined_data)
        
        gen_pca = pca_transformed[:len(gen_flat)]
        real_pca = pca_transformed[len(gen_flat):]
        
        # Clustering analysis
        labels_true = np.concatenate([np.zeros(len(gen_flat)), np.ones(len(real_flat))])
        
        # K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels_pred = kmeans.fit_predict(pca_transformed[:, :5])  # Use first 5 PCs
        
        # Silhouette score
        silhouette_avg = silhouette_score(pca_transformed[:, :5], labels_true)
        silhouette_pred = silhouette_score(pca_transformed[:, :5], labels_pred)
        
        analysis = {
            'pca_explained_variance': pca.explained_variance_ratio_[:5].tolist(),
            'pca_cumulative_variance': np.cumsum(pca.explained_variance_ratio_[:5]).tolist(),
            'silhouette_score_true': float(silhouette_avg),
            'silhouette_score_predicted': float(silhouette_pred),
            'cluster_separation': float(np.mean([
                np.linalg.norm(np.mean(gen_pca, axis=0) - np.mean(real_pca, axis=0)),
                np.mean(np.linalg.norm(gen_pca - np.mean(gen_pca, axis=0), axis=1)),
                np.mean(np.linalg.norm(real_pca - np.mean(real_pca, axis=0), axis=1))
            ])),
        }
        
        return analysis
    
    def _estimate_dominant_frequency(self, signal: np.ndarray, sr: int) -> float:
        """Estimate dominant frequency using FFT."""
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sr)
        magnitude = np.abs(fft_result)
        
        # Find dominant frequency (excluding DC component)
        dominant_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        return float(freqs[dominant_idx])
    
    def _estimate_bandwidth(self, signal: np.ndarray, sr: int) -> float:
        """Estimate signal bandwidth."""
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sr)
        magnitude = np.abs(fft_result[:len(fft_result)//2])
        
        # Find -3dB bandwidth
        max_mag = np.max(magnitude)
        threshold = max_mag / np.sqrt(2)  # -3dB
        
        above_threshold = magnitude > threshold
        if np.any(above_threshold):
            freq_indices = np.where(above_threshold)[0]
            bandwidth = freqs[freq_indices[-1]] - freqs[freq_indices[0]]
            return float(abs(bandwidth))
        else:
            return 0.0

def bootstrap_classification_metrics(logits: np.ndarray, targets: np.ndarray, 
                                   n_bootstrap: int = 1000, 
                                   confidence_level: float = 0.95,
                                   method: str = 'percentile') -> Dict:
    """
    Calculate bootstrap confidence intervals for classification metrics.
    
    Args:
        logits: Model output logits [N]
        targets: Ground truth labels [N]
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        method: Bootstrap method ('percentile', 'bca', 'basic')
    
    Returns:
        Dictionary with mean, std, and confidence intervals for each metric
    """
    from utils.metrics import compute_classification_metrics
    
    def compute_metrics_bootstrap(data):
        indices, = data
        sample_logits = logits[indices]
        sample_targets = targets[indices]
        return compute_classification_metrics(sample_logits, sample_targets)
    
    # Generate bootstrap samples
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(logits), size=len(logits), replace=True)
        metrics = compute_metrics_bootstrap((indices,))
        bootstrap_samples.append(metrics)
    
    # Calculate statistics
    results = {}
    metric_names = bootstrap_samples[0].keys()
    
    for metric in metric_names:
        values = [sample[metric] for sample in bootstrap_samples if metric in sample]
        if not values:
            continue
            
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Calculate confidence intervals
        if method == 'percentile':
            alpha = 1 - confidence_level
            lower = np.percentile(values, alpha/2 * 100)
            upper = np.percentile(values, (1-alpha/2) * 100)
        elif method == 'basic':
            alpha = 1 - confidence_level
            lower = 2 * mean_val - np.percentile(values, (1-alpha/2) * 100)
            upper = 2 * mean_val - np.percentile(values, alpha/2 * 100)
        else:  # BCa method
            try:
                bootstrap_result = bootstrap((np.arange(len(logits)),), 
                                          compute_metrics_bootstrap, 
                                          n_resamples=n_bootstrap,
                                          confidence_level=confidence_level)
                lower, upper = bootstrap_result.confidence_interval
                # Extract specific metric from bootstrap result
                lower = lower[metric] if hasattr(lower, '__getitem__') else lower
                upper = upper[metric] if hasattr(upper, '__getitem__') else upper
            except:
                # Fallback to percentile method
                alpha = 1 - confidence_level
                lower = np.percentile(values, alpha/2 * 100)
                upper = np.percentile(values, (1-alpha/2) * 100)
        
        results[metric] = {
            'mean': mean_val,
            'std': std_val,
            'ci_lower': lower,
            'ci_upper': upper,
            'confidence_level': confidence_level
        }
    
    return results

def statistical_significance_tests(logits: np.ndarray, targets: np.ndarray,
                                 prevalence: Optional[float] = None,
                                 multiple_testing_correction: str = 'bonferroni') -> Dict:
    """
    Perform statistical significance tests for classification performance.
    
    Args:
        logits: Model output logits [N]
        targets: Ground truth labels [N]
        prevalence: Dataset prevalence (if None, calculated from targets)
        multiple_testing_correction: Correction method ('bonferroni', 'holm', 'fdr')
    
    Returns:
        Dictionary with test results, p-values, and effect sizes
    """
    from utils.metrics import compute_classification_metrics
    
    if prevalence is None:
        prevalence = np.mean(targets)
    
    # Calculate metrics
    metrics = compute_classification_metrics(logits, targets)
    
    # Convert logits to predictions
    predictions = (logits > 0).astype(int)
    
    results = {}
    
    # Test accuracy against chance
    chance_accuracy = max(prevalence, 1 - prevalence)  # Best random performance
    if len(targets) > 30:  # Use t-test for large samples
        t_stat, p_value = stats.ttest_1samp(predictions == targets, chance_accuracy)
        test_type = 'one_sample_t_test'
    else:  # Use exact binomial test for small samples
        correct_predictions = np.sum(predictions == targets)
        p_value = stats.binomtest(correct_predictions, len(targets), chance_accuracy).proportions_ci()[1]
        t_stat = None
        test_type = 'exact_binomial_test'
    
    results['accuracy_test'] = {
        'test_type': test_type,
        'null_hypothesis': f'accuracy = {chance_accuracy:.3f}',
        'statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # Calculate effect sizes
    if 'accuracy' in metrics:
        cohens_d = calculate_cohens_d(metrics['accuracy'], chance_accuracy, len(targets))
        cliff_delta = calculate_cliff_delta(predictions == targets, 
                                         np.random.binomial(1, chance_accuracy, len(targets)))
        
        results['effect_sizes'] = {
            'cohens_d': cohens_d,
            'cliff_delta': cliff_delta,
            'interpretation': interpret_effect_size(cohens_d)
        }
    
    # Test other metrics if available
    test_metrics = ['precision', 'recall', 'f1', 'auroc']
    p_values = []
    
    for metric in test_metrics:
        if metric in metrics:
            metric_value = metrics[metric]
            if metric == 'auroc':
                # Test AUROC against 0.5 (random)
                null_value = 0.5
            else:
                # Test other metrics against prevalence-based baseline
                null_value = prevalence if metric in ['precision', 'recall'] else 2 * prevalence * (1 - prevalence) / (prevalence + (1 - prevalence))
            
            if len(targets) > 30:
                # Use t-test for large samples
                t_stat, p_value = stats.ttest_1samp([metric_value], null_value)
            else:
                # Use Wilcoxon signed-rank test for small samples
                _, p_value = stats.wilcoxon([metric_value], [null_value])
            
            p_values.append(p_value)
            results[f'{metric}_test'] = {
                'test_type': 'one_sample_test',
                'null_hypothesis': f'{metric} = {null_value:.3f}',
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    # Apply multiple testing correction
    if len(p_values) > 1:
        corrected_p_values = apply_multiple_testing_correction(p_values, method=multiple_testing_correction)
        
        # Update results with corrected p-values
        for i, metric in enumerate(test_metrics):
            if f'{metric}_test' in results:
                results[f'{metric}_test']['corrected_p_value'] = corrected_p_values[i]
                results[f'{metric}_test']['significant_corrected'] = corrected_p_values[i] < 0.05
    
    return results

def roc_analysis(logits: np.ndarray, targets: np.ndarray,
                 specificity_targets: List[float] = [0.8, 0.9, 0.95]) -> Dict:
    """
    Perform comprehensive ROC analysis.
    
    Args:
        logits: Model output logits [N]
        targets: Ground truth labels [N]
        specificity_targets: Specificity levels for sensitivity analysis
    
    Returns:
        Dictionary with ROC curve data, optimal threshold, and sensitivity analysis
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(targets, logits)
    auroc = auc(fpr, tpr)
    
    # Find optimal threshold using Youden's index
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    
    # Calculate sensitivity at specific specificity levels
    sensitivity_at_specificity = {}
    for target_spec in specificity_targets:
        # Find threshold that gives closest specificity
        spec_diff = np.abs(1 - fpr - target_spec)
        closest_idx = np.argmin(spec_diff)
        sensitivity_at_specificity[f'sensitivity_at_specificity_{target_spec:.2f}'] = {
            'sensitivity': tpr[closest_idx],
            'specificity': 1 - fpr[closest_idx],
            'threshold': thresholds[closest_idx]
        }
    
    # Bootstrap confidence interval for AUROC
    try:
        auroc_ci = bootstrap_auroc_confidence_interval(logits, targets)
    except:
        auroc_ci = {'lower': auroc, 'upper': auroc}
    
    return {
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        },
        'auroc': auroc,
        'auroc_ci': auroc_ci,
        'optimal_threshold': {
            'threshold': optimal_threshold,
            'sensitivity': optimal_sensitivity,
            'specificity': optimal_specificity,
            'youden_index': youden_index[optimal_idx]
        },
        'sensitivity_at_specificity': sensitivity_at_specificity
    }

def precision_recall_analysis(logits: np.ndarray, targets: np.ndarray) -> Dict:
    """
    Perform precision-recall curve analysis.
    
    Args:
        logits: Model output logits [N]
        targets: Ground truth labels [N]
    
    Returns:
        Dictionary with PR curve data and optimal threshold information
    """
    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(targets, logits)
    average_precision = auc(recall, precision)
    
    # Find optimal threshold for F1 score maximization
    f1_scores = []
    valid_thresholds = []
    
    for threshold in thresholds:
        predictions = (logits >= threshold).astype(int)
        if np.sum(predictions) > 0:  # Avoid division by zero
            precision_val = np.sum((predictions == 1) & (targets == 1)) / np.sum(predictions == 1)
            recall_val = np.sum((predictions == 1) & (targets == 1)) / np.sum(targets == 1)
            
            if precision_val + recall_val > 0:
                f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
                f1_scores.append(f1)
                valid_thresholds.append(threshold)
    
    if f1_scores:
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = valid_thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
    else:
        optimal_threshold = 0.0
        optimal_f1 = 0.0
    
    # Bootstrap confidence interval for average precision
    try:
        ap_ci = bootstrap_average_precision_confidence_interval(logits, targets)
    except:
        ap_ci = {'lower': average_precision, 'upper': average_precision}
    
    return {
        'pr_curve': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist()
        },
        'average_precision': average_precision,
        'average_precision_ci': ap_ci,
        'optimal_threshold': {
            'threshold': optimal_threshold,
            'f1_score': optimal_f1
        }
    }

def clinical_validation_analysis(logits: np.ndarray, targets: np.ndarray,
                                prevalence: Optional[float] = None) -> Dict:
    """
    Perform clinical validation analysis for ABR peak detection.
    
    Args:
        logits: Model output logits [N]
        targets: Ground truth labels [N]
        prevalence: Dataset prevalence (if None, calculated from targets)
    
    Returns:
        Dictionary with clinical validation metrics
    """
    if prevalence is None:
        prevalence = np.mean(targets)
    
    predictions = (logits > 0).astype(int)
    
    # Calculate confusion matrix components
    tp = np.sum((predictions == 1) & (targets == 1))
    tn = np.sum((predictions == 0) & (targets == 0))
    fp = np.sum((predictions == 1) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))
    
    # Basic clinical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Predictive values
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    
    # Diagnostic odds ratio
    if fp > 0 and fn > 0:
        diagnostic_odds_ratio = (tp * tn) / (fp * fn)
        dor_ci = calculate_dor_confidence_interval(tp, tn, fp, fn)
    else:
        diagnostic_odds_ratio = np.inf
        dor_ci = {'lower': np.inf, 'upper': np.inf}
    
    # Likelihood ratios
    lr_positive = sensitivity / (1 - specificity) if specificity < 1 else np.inf
    lr_negative = (1 - sensitivity) / specificity if specificity > 0 else np.inf
    
    # Number needed to diagnose (NND)
    nnd = 1 / (sensitivity - (1 - specificity)) if (sensitivity + specificity) > 1 else np.inf
    
    # Prevalence-adjusted metrics
    adjusted_ppv = (sensitivity * prevalence) / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
    adjusted_npv = (specificity * (1 - prevalence)) / (specificity * (1 - prevalence) + (1 - sensitivity) * prevalence)
    
    return {
        'basic_metrics': {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv
        },
        'diagnostic_odds_ratio': {
            'value': diagnostic_odds_ratio,
            'confidence_interval': dor_ci
        },
        'likelihood_ratios': {
            'positive_likelihood_ratio': lr_positive,
            'negative_likelihood_ratio': lr_negative
        },
        'clinical_utility': {
            'number_needed_to_diagnose': nnd,
            'prevalence_adjusted_ppv': adjusted_ppv,
            'prevalence_adjusted_npv': adjusted_npv
        },
        'prevalence': prevalence
    }

def comparative_statistical_analysis(results1: Dict, results2: Dict,
                                   paired: bool = True,
                                   significance_level: float = 0.05) -> Dict:
    """
    Perform comparative statistical analysis between two evaluation results.
    
    Args:
        results1: First evaluation results
        results2: Second evaluation results
        paired: Whether the results are from paired samples
        significance_level: Statistical significance threshold
    
    Returns:
        Dictionary with comparative analysis results
    """
    comparison_results = {}
    
    # Extract metrics for comparison
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
    
    for metric in metrics_to_compare:
        if metric in results1 and metric in results2:
            val1 = results1[metric]
            val2 = results2[metric]
            
            # Calculate difference
            diff = val2 - val1
            
            if paired:
                # Paired t-test
                t_stat, p_value = stats.ttest_rel([val1], [val2])
                test_type = 'paired_t_test'
            else:
                # Independent t-test
                t_stat, p_value = stats.ttest_ind([val1], [val2])
                test_type = 'independent_t_test'
            
            # Calculate effect size
            cohens_d = calculate_cohens_d(val2, val1, 2)  # Sample size = 2 for paired comparison
            
            comparison_results[metric] = {
                'value1': val1,
                'value2': val2,
                'difference': diff,
                'test_type': test_type,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < significance_level,
                'effect_size': cohens_d,
                'interpretation': interpret_effect_size(cohens_d)
            }
    
    # McNemar's test for binary classification comparison
    if 'predictions' in results1 and 'predictions' in results2 and 'targets' in results1:
        mcnemar_result = perform_mcnemar_test(results1, results2)
        comparison_results['mcnemar_test'] = mcnemar_result
    
    return comparison_results

def calculate_cohens_d(value1: float, value2: float, n: int) -> float:
    """Calculate Cohen's d effect size."""
    pooled_std = np.sqrt(((n-1) * (value1**2 + value2**2)) / (2*n - 2))
    if pooled_std == 0:
        return 0.0
    return (value1 - value2) / pooled_std

def calculate_cliff_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cliff's delta effect size."""
    n1, n2 = len(group1), len(group2)
    wins = 0
    ties = 0
    
    for x in group1:
        for y in group2:
            if x > y:
                wins += 1
            elif x == y:
                ties += 1
    
    total = n1 * n2
    return (wins - (total - wins - ties)) / total

def interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    else:
        return 'large'

def apply_multiple_testing_correction(p_values: List[float], 
                                    method: str = 'bonferroni') -> List[float]:
    """Apply multiple testing correction to p-values."""
    p_values = np.array(p_values)
    
    if method == 'bonferroni':
        return np.minimum(p_values * len(p_values), 1.0)
    elif method == 'holm':
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        corrected_p_values = np.zeros_like(sorted_p_values)
        
        for i, p_val in enumerate(sorted_p_values):
            corrected_p_values[i] = min(p_val * (len(sorted_p_values) - i), 1.0)
        
        # Restore original order
        original_indices = np.argsort(sorted_indices)
        return corrected_p_values[original_indices]
    elif method == 'fdr':
        from statsmodels.stats.multitest import multipletests
        _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
        return corrected_p_values
    else:
        return p_values

def bootstrap_auroc_confidence_interval(logits: np.ndarray, targets: np.ndarray,
                                      n_bootstrap: int = 1000,
                                      confidence_level: float = 0.95) -> Dict:
    """Calculate bootstrap confidence interval for AUROC."""
    def compute_auroc(data):
        indices, = data
        sample_logits = logits[indices]
        sample_targets = targets[indices]
        fpr, tpr, _ = roc_curve(sample_targets, sample_logits)
        return auc(fpr, tpr)
    
    try:
        bootstrap_result = bootstrap((np.arange(len(logits)),), 
                                  compute_auroc, 
                                  n_resamples=n_bootstrap,
                                  confidence_level=confidence_level)
        return {'lower': bootstrap_result.confidence_interval[0],
                'upper': bootstrap_result.confidence_interval[1]}
    except:
        # Fallback to simple percentile method
        auroc_values = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(logits), size=len(logits), replace=True)
            sample_logits = logits[indices]
            sample_targets = targets[indices]
            fpr, tpr, _ = roc_curve(sample_targets, sample_logits)
            auroc_values.append(auc(fpr, tpr))
        
        alpha = 1 - confidence_level
        return {'lower': np.percentile(auroc_values, alpha/2 * 100),
                'upper': np.percentile(auroc_values, (1-alpha/2) * 100)}

def bootstrap_average_precision_confidence_interval(logits: np.ndarray, targets: np.ndarray,
                                                  n_bootstrap: int = 1000,
                                                  confidence_level: float = 0.95) -> Dict:
    """Calculate bootstrap confidence interval for average precision."""
    def compute_ap(data):
        indices, = data
        sample_logits = logits[indices]
        sample_targets = targets[indices]
        precision, recall, _ = precision_recall_curve(sample_targets, sample_logits)
        return auc(recall, precision)
    
    try:
        bootstrap_result = bootstrap((np.arange(len(logits)),), 
                                  compute_ap, 
                                  n_resamples=n_bootstrap,
                                  confidence_level=confidence_level)
        return {'lower': bootstrap_result.confidence_interval[0],
                'upper': bootstrap_result.confidence_interval[1]}
    except:
        # Fallback to simple percentile method
        ap_values = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(logits), size=len(logits), replace=True)
            sample_logits = logits[indices]
            sample_targets = targets[indices]
            precision, recall, _ = precision_recall_curve(sample_targets, sample_logits)
            ap_values.append(auc(recall, precision))
        
        alpha = 1 - confidence_level
        return {'lower': np.percentile(ap_values, alpha/2 * 100),
                'upper': np.percentile(ap_values, (1-alpha/2) * 100)}

def calculate_dor_confidence_interval(tp: int, tn: int, fp: int, fn: int,
                                    confidence_level: float = 0.95) -> Dict:
    """Calculate confidence interval for diagnostic odds ratio."""
    if tp == 0 or tn == 0 or fp == 0 or fn == 0:
        return {'lower': np.inf, 'upper': np.inf}
    
    # Calculate log odds ratio and its standard error
    log_or = np.log((tp * tn) / (fp * fn))
    se_log_or = np.sqrt(1/tp + 1/tn + 1/fp + 1/fn)
    
    # Calculate confidence interval
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = np.exp(log_or - z_score * se_log_or)
    ci_upper = np.exp(log_or + z_score * se_log_or)
    
    return {'lower': ci_lower, 'upper': ci_upper}

def perform_mcnemar_test(results1: Dict, results2: Dict) -> Dict:
    """Perform McNemar's test for comparing binary classification performance."""
    if 'predictions' not in results1 or 'predictions' not in results2 or 'targets' not in results1:
        return {'error': 'Missing required data for McNemar test'}
    
    pred1 = results1['predictions']
    pred2 = results2['predictions']
    targets = results1['targets']
    
    # Create contingency table
    a = np.sum((pred1 == targets) & (pred2 == targets))  # Both correct
    b = np.sum((pred1 != targets) & (pred2 == targets))  # Only pred2 correct
    c = np.sum((pred1 == targets) & (pred2 != targets))  # Only pred1 correct
    d = np.sum((pred1 != targets) & (pred2 != targets))  # Both wrong
    
    # Perform McNemar test
    try:
        statistic, p_value = stats.mcnemar([[a, b], [c, d]], exact=True)
        test_type = 'exact_mcnemar'
    except:
        # Fallback to chi-square approximation
        statistic, p_value = stats.mcnemar([[a, b], [c, d]], exact=False)
        test_type = 'chi_square_mcnemar'
    
    return {
        'test_type': test_type,
        'statistic': statistic,
        'p_value': p_value,
        'contingency_table': {
            'both_correct': a,
            'only_pred2_correct': b,
            'only_pred1_correct': c,
            'both_wrong': d
        }
    }