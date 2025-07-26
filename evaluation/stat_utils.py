#!/usr/bin/env python3
"""
Statistical Utilities for ABR Model Evaluation

This module provides statistical analysis functions including:
- Bootstrap confidence intervals
- Significance testing
- Distribution analysis
- Clinical statistical metrics

Author: AI Assistant
Date: January 2025
"""

import numpy as np
import scipy.stats
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import warnings
from collections import defaultdict


class StatisticalAnalyzer:
    """Advanced statistical analysis for model evaluation."""
    
    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 1000):
        """Initialize statistical analyzer.
        
        Args:
            confidence_level: Default confidence level for intervals
            n_bootstrap: Default number of bootstrap samples
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
    
    # ==================== BOOTSTRAP CONFIDENCE INTERVALS ====================
    
    def bootstrap_ci(
        self, 
        data: np.ndarray, 
        statistic: Union[str, Callable] = 'mean',
        n_bootstrap: Optional[int] = None,
        confidence_level: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence intervals for a statistic.
        
        Args:
            data: Input data array
            statistic: Statistic to compute ('mean', 'median', 'std', or callable)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (statistic_value, lower_ci, upper_ci)
        """
        if len(data) == 0:
            return 0.0, 0.0, 0.0
        
        if seed is not None:
            np.random.seed(seed)
        
        n_bootstrap = n_bootstrap or self.n_bootstrap
        confidence_level = confidence_level or self.confidence_level
        
        # Choose statistic function
        if isinstance(statistic, str):
            if statistic == 'mean':
                stat_func = np.mean
            elif statistic == 'median':
                stat_func = np.median
            elif statistic == 'std':
                stat_func = np.std
            elif statistic == 'var':
                stat_func = np.var
            elif statistic == 'min':
                stat_func = np.min
            elif statistic == 'max':
                stat_func = np.max
            else:
                stat_func = np.mean
        else:
            stat_func = statistic
        
        # Original statistic
        try:
            original_stat = stat_func(data)
        except:
            return 0.0, 0.0, 0.0
        
        # Bootstrap sampling
        bootstrap_stats = []
        data_flat = data.flatten()
        
        for _ in range(n_bootstrap):
            try:
                bootstrap_sample = np.random.choice(data_flat, size=len(data_flat), replace=True)
                bootstrap_stat = stat_func(bootstrap_sample)
                if not np.isnan(bootstrap_stat) and not np.isinf(bootstrap_stat):
                    bootstrap_stats.append(bootstrap_stat)
            except:
                continue
        
        if len(bootstrap_stats) == 0:
            return original_stat, original_stat, original_stat
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_ci = np.percentile(bootstrap_stats, lower_percentile)
        upper_ci = np.percentile(bootstrap_stats, upper_percentile)
        
        return original_stat, lower_ci, upper_ci
    
    def bootstrap_ci_difference(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        statistic: Union[str, Callable] = 'mean',
        n_bootstrap: Optional[int] = None,
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap CI for the difference between two statistics.
        
        Args:
            data1: First dataset
            data2: Second dataset
            statistic: Statistic to compute
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            
        Returns:
            Tuple of (difference, lower_ci, upper_ci)
        """
        if len(data1) == 0 or len(data2) == 0:
            return 0.0, 0.0, 0.0
        
        n_bootstrap = n_bootstrap or self.n_bootstrap
        confidence_level = confidence_level or self.confidence_level
        
        # Choose statistic function
        if isinstance(statistic, str):
            if statistic == 'mean':
                stat_func = np.mean
            elif statistic == 'median':
                stat_func = np.median
            elif statistic == 'std':
                stat_func = np.std
            else:
                stat_func = np.mean
        else:
            stat_func = statistic
        
        # Original difference
        try:
            stat1 = stat_func(data1)
            stat2 = stat_func(data2)
            original_diff = stat1 - stat2
        except:
            return 0.0, 0.0, 0.0
        
        # Bootstrap sampling
        bootstrap_diffs = []
        data1_flat = data1.flatten()
        data2_flat = data2.flatten()
        
        for _ in range(n_bootstrap):
            try:
                bootstrap_sample1 = np.random.choice(data1_flat, size=len(data1_flat), replace=True)
                bootstrap_sample2 = np.random.choice(data2_flat, size=len(data2_flat), replace=True)
                
                stat1_boot = stat_func(bootstrap_sample1)
                stat2_boot = stat_func(bootstrap_sample2)
                diff_boot = stat1_boot - stat2_boot
                
                if not np.isnan(diff_boot) and not np.isinf(diff_boot):
                    bootstrap_diffs.append(diff_boot)
            except:
                continue
        
        if len(bootstrap_diffs) == 0:
            return original_diff, original_diff, original_diff
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_ci = np.percentile(bootstrap_diffs, lower_percentile)
        upper_ci = np.percentile(bootstrap_diffs, upper_percentile)
        
        return original_diff, lower_ci, upper_ci
    
    # ==================== SIGNIFICANCE TESTING ====================
    
    def permutation_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        statistic: Union[str, Callable] = 'mean',
        n_permutations: int = 10000,
        alternative: str = 'two-sided'
    ) -> Tuple[float, float]:
        """
        Perform permutation test for difference in statistics.
        
        Args:
            data1: First dataset
            data2: Second dataset
            statistic: Statistic to test
            n_permutations: Number of permutations
            alternative: Alternative hypothesis ('two-sided', 'greater', 'less')
            
        Returns:
            Tuple of (test_statistic, p_value)
        """
        if len(data1) == 0 or len(data2) == 0:
            return 0.0, 1.0
        
        # Choose statistic function
        if isinstance(statistic, str):
            if statistic == 'mean':
                stat_func = np.mean
            elif statistic == 'median':
                stat_func = np.median
            elif statistic == 'std':
                stat_func = np.std
            else:
                stat_func = np.mean
        else:
            stat_func = statistic
        
        # Observed test statistic
        try:
            observed_stat = stat_func(data1) - stat_func(data2)
        except:
            return 0.0, 1.0
        
        # Combined data
        combined_data = np.concatenate([data1.flatten(), data2.flatten()])
        n1, n2 = len(data1.flatten()), len(data2.flatten())
        
        # Permutation distribution
        perm_stats = []
        for _ in range(n_permutations):
            try:
                permuted_data = np.random.permutation(combined_data)
                perm_data1 = permuted_data[:n1]
                perm_data2 = permuted_data[n1:n1+n2]
                
                perm_stat = stat_func(perm_data1) - stat_func(perm_data2)
                if not np.isnan(perm_stat) and not np.isinf(perm_stat):
                    perm_stats.append(perm_stat)
            except:
                continue
        
        if len(perm_stats) == 0:
            return observed_stat, 1.0
        
        perm_stats = np.array(perm_stats)
        
        # Compute p-value
        if alternative == 'two-sided':
            p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
        elif alternative == 'greater':
            p_value = np.mean(perm_stats >= observed_stat)
        elif alternative == 'less':
            p_value = np.mean(perm_stats <= observed_stat)
        else:
            p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
        
        return observed_stat, p_value
    
    def multiple_comparisons_correction(
        self,
        p_values: List[float],
        method: str = 'bonferroni',
        alpha: float = 0.05
    ) -> Tuple[List[bool], List[float]]:
        """
        Apply multiple comparisons correction.
        
        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni', 'holm', 'fdr')
            alpha: Significance level
            
        Returns:
            Tuple of (rejected_hypotheses, corrected_p_values)
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            corrected_p = p_values * n_tests
            corrected_p = np.minimum(corrected_p, 1.0)
            rejected = corrected_p <= alpha
            
        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected_p = np.zeros_like(p_values)
            rejected = np.zeros(n_tests, dtype=bool)
            
            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = sorted_p[i] * (n_tests - i)
                if corrected_p[idx] <= alpha:
                    rejected[idx] = True
                else:
                    break  # Stop at first non-significant result
            
            corrected_p = np.minimum(corrected_p, 1.0)
            
        elif method == 'fdr':
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected_p = np.zeros_like(p_values)
            rejected = np.zeros(n_tests, dtype=bool)
            
            for i in range(n_tests-1, -1, -1):
                idx = sorted_indices[i]
                corrected_p[idx] = sorted_p[i] * n_tests / (i + 1)
                if corrected_p[idx] <= alpha:
                    rejected[sorted_indices[:i+1]] = True
                    break
            
            corrected_p = np.minimum(corrected_p, 1.0)
            
        else:
            # Default to Bonferroni
            corrected_p = p_values * n_tests
            corrected_p = np.minimum(corrected_p, 1.0)
            rejected = corrected_p <= alpha
        
        return rejected.tolist(), corrected_p.tolist()
    
    # ==================== DISTRIBUTION ANALYSIS ====================
    
    def test_normality(
        self,
        data: np.ndarray,
        method: str = 'shapiro'
    ) -> Tuple[float, float]:
        """
        Test for normality of data distribution.
        
        Args:
            data: Input data
            method: Test method ('shapiro', 'kstest', 'anderson')
            
        Returns:
            Tuple of (test_statistic, p_value)
        """
        data_flat = data.flatten()
        data_clean = data_flat[~np.isnan(data_flat) & ~np.isinf(data_flat)]
        
        if len(data_clean) < 3:
            return 0.0, 1.0
        
        try:
            if method == 'shapiro':
                if len(data_clean) <= 5000:  # Shapiro-Wilk has sample size limit
                    statistic, p_value = scipy.stats.shapiro(data_clean)
                else:
                    # Use Kolmogorov-Smirnov for large samples
                    statistic, p_value = scipy.stats.kstest(data_clean, 'norm')
            elif method == 'kstest':
                # Standardize data for KS test
                data_std = (data_clean - np.mean(data_clean)) / np.std(data_clean)
                statistic, p_value = scipy.stats.kstest(data_std, 'norm')
            elif method == 'anderson':
                result = scipy.stats.anderson(data_clean, dist='norm')
                statistic = result.statistic
                # Convert critical value to approximate p-value
                p_value = 0.05 if statistic > result.critical_values[2] else 0.15
            else:
                statistic, p_value = scipy.stats.shapiro(data_clean)
                
            return statistic, p_value
            
        except Exception as e:
            warnings.warn(f"Normality test failed: {e}")
            return 0.0, 1.0
    
    def compute_effect_size(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        method: str = 'cohen_d'
    ) -> float:
        """
        Compute effect size between two groups.
        
        Args:
            data1: First group data
            data2: Second group data
            method: Effect size method ('cohen_d', 'glass_delta', 'hedges_g')
            
        Returns:
            Effect size value
        """
        data1_flat = data1.flatten()
        data2_flat = data2.flatten()
        
        # Remove invalid values
        data1_clean = data1_flat[~np.isnan(data1_flat) & ~np.isinf(data1_flat)]
        data2_clean = data2_flat[~np.isnan(data2_flat) & ~np.isinf(data2_flat)]
        
        if len(data1_clean) == 0 or len(data2_clean) == 0:
            return 0.0
        
        mean1, mean2 = np.mean(data1_clean), np.mean(data2_clean)
        std1, std2 = np.std(data1_clean, ddof=1), np.std(data2_clean, ddof=1)
        
        try:
            if method == 'cohen_d':
                # Cohen's d
                pooled_std = np.sqrt(((len(data1_clean) - 1) * std1**2 + 
                                    (len(data2_clean) - 1) * std2**2) / 
                                   (len(data1_clean) + len(data2_clean) - 2))
                effect_size = (mean1 - mean2) / pooled_std
                
            elif method == 'glass_delta':
                # Glass's delta (uses control group std)
                effect_size = (mean1 - mean2) / std2
                
            elif method == 'hedges_g':
                # Hedges' g (bias-corrected Cohen's d)
                pooled_std = np.sqrt(((len(data1_clean) - 1) * std1**2 + 
                                    (len(data2_clean) - 1) * std2**2) / 
                                   (len(data1_clean) + len(data2_clean) - 2))
                cohens_d = (mean1 - mean2) / pooled_std
                correction_factor = 1 - (3 / (4 * (len(data1_clean) + len(data2_clean)) - 9))
                effect_size = cohens_d * correction_factor
                
            else:
                # Default to Cohen's d
                pooled_std = np.sqrt(((len(data1_clean) - 1) * std1**2 + 
                                    (len(data2_clean) - 1) * std2**2) / 
                                   (len(data1_clean) + len(data2_clean) - 2))
                effect_size = (mean1 - mean2) / pooled_std
            
            return effect_size if not np.isnan(effect_size) else 0.0
            
        except:
            return 0.0
    
    # ==================== CLINICAL STATISTICS ====================
    
    def compute_clinical_agreement(
        self,
        true_values: np.ndarray,
        pred_values: np.ndarray,
        tolerance: float = 5.0
    ) -> Dict[str, float]:
        """
        Compute clinical agreement metrics.
        
        Args:
            true_values: True values
            pred_values: Predicted values
            tolerance: Agreement tolerance
            
        Returns:
            Dictionary of agreement metrics
        """
        true_flat = true_values.flatten()
        pred_flat = pred_values.flatten()
        
        # Ensure same length
        min_len = min(len(true_flat), len(pred_flat))
        true_flat = true_flat[:min_len]
        pred_flat = pred_flat[:min_len]
        
        if len(true_flat) == 0:
            return {}
        
        # Remove invalid values
        valid_mask = ~(np.isnan(true_flat) | np.isnan(pred_flat) | 
                      np.isinf(true_flat) | np.isinf(pred_flat))
        true_clean = true_flat[valid_mask]
        pred_clean = pred_flat[valid_mask]
        
        if len(true_clean) == 0:
            return {}
        
        metrics = {}
        
        # Agreement within tolerance
        within_tolerance = np.abs(true_clean - pred_clean) <= tolerance
        metrics['agreement_rate'] = np.mean(within_tolerance)
        metrics['agreement_count'] = int(np.sum(within_tolerance))
        
        # Bland-Altman analysis
        mean_values = (true_clean + pred_clean) / 2
        differences = pred_clean - true_clean
        
        metrics['mean_difference'] = np.mean(differences)
        metrics['std_difference'] = np.std(differences)
        metrics['limits_of_agreement'] = (
            np.mean(differences) - 1.96 * np.std(differences),
            np.mean(differences) + 1.96 * np.std(differences)
        )
        
        # Concordance correlation coefficient
        try:
            mean_true = np.mean(true_clean)
            mean_pred = np.mean(pred_clean)
            var_true = np.var(true_clean)
            var_pred = np.var(pred_clean)
            covariance = np.cov(true_clean, pred_clean)[0, 1]
            
            ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)
            metrics['concordance_correlation'] = ccc
        except:
            metrics['concordance_correlation'] = 0.0
        
        return metrics
    
    def compute_reliability_metrics(
        self,
        data: np.ndarray,
        groups: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute reliability metrics for repeated measurements.
        
        Args:
            data: Measurement data [subjects, measurements]
            groups: Group labels for each subject
            
        Returns:
            Dictionary of reliability metrics
        """
        if len(data.shape) != 2:
            return {}
        
        n_subjects, n_measurements = data.shape
        
        if n_measurements < 2:
            return {}
        
        metrics = {}
        
        # Intraclass correlation coefficient (ICC)
        try:
            # Between-subject variance
            subject_means = np.mean(data, axis=1)
            grand_mean = np.mean(data)
            between_subject_var = np.sum((subject_means - grand_mean)**2) / (n_subjects - 1)
            
            # Within-subject variance
            within_subject_var = np.mean(np.var(data, axis=1, ddof=1))
            
            # ICC(2,1) - two-way random effects, single measurement
            icc = (between_subject_var - within_subject_var) / (
                between_subject_var + (n_measurements - 1) * within_subject_var
            )
            metrics['icc'] = max(0, icc)  # ICC should be non-negative
            
        except:
            metrics['icc'] = 0.0
        
        # Coefficient of variation
        try:
            cv_values = []
            for i in range(n_subjects):
                subject_data = data[i, :]
                if np.mean(subject_data) != 0:
                    cv = np.std(subject_data) / np.abs(np.mean(subject_data))
                    cv_values.append(cv)
            
            if cv_values:
                metrics['coefficient_of_variation'] = np.mean(cv_values)
        except:
            metrics['coefficient_of_variation'] = 0.0
        
        return metrics 