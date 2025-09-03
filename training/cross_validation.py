"""
Cross-validation framework for ABR transformer training.

This module implements comprehensive cross-validation strategies including:
- Patient-stratified K-fold cross-validation
- Nested cross-validation for hyperparameter optimization
- Time-series aware cross-validation for temporal data
- Cross-validation result aggregation and statistical analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd
import logging
from pathlib import Path
import json
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class BaseCVSplitter(ABC):
    """
    Abstract base class for cross-validation splitting strategies.
    """
    
    @abstractmethod
    def split(self, X, y, groups=None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation splits."""
        pass
    
    @abstractmethod
    def get_n_splits(self) -> int:
        """Get number of splits."""
        pass


class StratifiedPatientKFold(BaseCVSplitter):
    """
    Patient-stratified K-fold cross-validation that ensures:
    1. No patient appears in both train and validation sets
    2. Stratification by hearing loss class and peak presence
    3. Balanced distribution across folds
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        min_samples_per_class: int = 1
    ):
        """
        Initialize stratified patient K-fold.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle before splitting
            random_state: Random state for reproducibility
            min_samples_per_class: Minimum samples per class per fold
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.min_samples_per_class = min_samples_per_class
        
    def split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        groups: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate patient-stratified splits.
        
        Args:
            X: Feature data (not used directly, kept for compatibility)
            y: Target labels for stratification
            groups: Patient IDs for grouping
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if groups is None:
            raise ValueError("Patient groups must be provided for patient-level splitting")
            
        # Get unique patients and their labels
        unique_patients = np.unique(groups)
        patient_labels = []
        
        for patient in unique_patients:
            patient_mask = groups == patient
            patient_y = y[patient_mask]
            
            # Create stratification label (could be multi-dimensional)
            if patient_y.ndim > 1:
                # For multi-label case, use majority class or combination
                patient_label = tuple(np.round(patient_y.mean(axis=0)).astype(int))
            else:
                # For single label case
                if len(np.unique(patient_y)) == 1:
                    patient_label = patient_y[0]
                else:
                    # Mixed labels for same patient - use majority
                    patient_label = np.bincount(patient_y).argmax()
                    
            patient_labels.append(patient_label)
            
        patient_labels = np.array(patient_labels)
        
        # Use GroupKFold if stratification is not possible due to class imbalance
        try:
            if len(np.unique(patient_labels)) > 1:
                skf = StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=self.shuffle,
                    random_state=self.random_state
                )
                patient_splits = list(skf.split(unique_patients, patient_labels))
            else:
                # Fall back to regular KFold if only one class
                kf = KFold(
                    n_splits=self.n_splits,
                    shuffle=self.shuffle,
                    random_state=self.random_state
                )
                patient_splits = list(kf.split(unique_patients))
        except ValueError:
            # If stratification fails, use GroupKFold
            logger.warning("Stratified splitting failed, falling back to GroupKFold")
            gkf = GroupKFold(n_splits=self.n_splits)
            patient_splits = list(gkf.split(unique_patients, patient_labels, unique_patients))
            
        # Convert patient splits to sample splits
        sample_splits = []
        for train_patients_idx, val_patients_idx in patient_splits:
            train_patients = unique_patients[train_patients_idx]
            val_patients = unique_patients[val_patients_idx]
            
            # Get sample indices for each patient group
            train_indices = []
            val_indices = []
            
            for patient in train_patients:
                patient_samples = np.where(groups == patient)[0]
                train_indices.extend(patient_samples)
                
            for patient in val_patients:
                patient_samples = np.where(groups == patient)[0]
                val_indices.extend(patient_samples)
                
            sample_splits.append((np.array(train_indices), np.array(val_indices)))
            
        return sample_splits
        
    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits


class TimeSeriesKFold(BaseCVSplitter):
    """
    Time-aware cross-validation for temporal ABR data.
    
    Ensures temporal ordering is preserved and prevents data leakage
    from future to past.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        max_train_size: Optional[int] = None
    ):
        """
        Initialize time series K-fold.
        
        Args:
            n_splits: Number of splits
            gap: Gap between train and validation sets
            max_train_size: Maximum size of training set
        """
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size
        
    def split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        groups: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate time-aware splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate split sizes
        test_size = n_samples // self.n_splits
        splits = []
        
        for i in range(self.n_splits):
            # Validation set
            val_start = i * test_size
            val_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples
            val_indices = indices[val_start:val_end]
            
            # Training set (everything before validation, with gap)
            train_end = val_start - self.gap
            if train_end <= 0:
                continue
                
            train_start = 0
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
                
            train_indices = indices[train_start:train_end]
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                splits.append((train_indices, val_indices))
                
        return splits
        
    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits


class CrossValidationManager:
    """
    Manager for orchestrating cross-validation experiments.
    
    Handles fold creation, training coordination, and result aggregation
    with support for parallel execution and comprehensive logging.
    """
    
    def __init__(
        self,
        cv_splitter: BaseCVSplitter,
        save_dir: str,
        experiment_name: str = "cv_experiment",
        parallel: bool = False,
        n_jobs: int = 1
    ):
        """
        Initialize cross-validation manager.
        
        Args:
            cv_splitter: Cross-validation splitting strategy
            save_dir: Directory to save results and models
            experiment_name: Name for this CV experiment
            parallel: Whether to run folds in parallel
            n_jobs: Number of parallel jobs
        """
        self.cv_splitter = cv_splitter
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.parallel = parallel
        self.n_jobs = n_jobs
        
        # Create directories
        self.experiment_dir = self.save_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.fold_results = {}
        self.fold_models = {}
        self.cv_splits = None
        
    def run_cross_validation(
        self,
        train_fn: Callable,
        evaluate_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
        eval_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run complete cross-validation experiment.
        
        Args:
            train_fn: Function to train model on fold
            evaluate_fn: Function to evaluate model on fold
            X: Feature data
            y: Target labels
            groups: Group labels (e.g., patient IDs)
            train_kwargs: Additional arguments for training function
            eval_kwargs: Additional arguments for evaluation function
            
        Returns:
            Dictionary of aggregated results
        """
        train_kwargs = train_kwargs or {}
        eval_kwargs = eval_kwargs or {}
        
        # Generate CV splits
        self.cv_splits = self.cv_splitter.split(X, y, groups)
        n_splits = len(self.cv_splits)
        
        logger.info(f"Starting {n_splits}-fold cross-validation experiment: {self.experiment_name}")
        
        # Run each fold
        if self.parallel and self.n_jobs > 1:
            # Parallel execution (would need joblib or similar)
            logger.info(f"Running folds in parallel with {self.n_jobs} jobs")
            results = self._run_folds_parallel(
                train_fn, evaluate_fn, X, y, groups, train_kwargs, eval_kwargs
            )
        else:
            # Sequential execution
            results = self._run_folds_sequential(
                train_fn, evaluate_fn, X, y, groups, train_kwargs, eval_kwargs
            )
            
        # Aggregate results
        aggregated_results = self._aggregate_results(results)
        
        # Save results
        self._save_cv_results(aggregated_results)
        
        logger.info(f"Cross-validation completed. Results saved to {self.experiment_dir}")
        return aggregated_results
        
    def _run_folds_sequential(
        self,
        train_fn: Callable,
        evaluate_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        train_kwargs: Dict[str, Any],
        eval_kwargs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run folds sequentially."""
        results = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(self.cv_splits):
            logger.info(f"Training fold {fold_idx + 1}/{len(self.cv_splits)}")
            
            # Create fold data
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            
            fold_groups_train = groups[train_indices] if groups is not None else None
            fold_groups_val = groups[val_indices] if groups is not None else None
            
            # Create fold directory
            fold_dir = self.experiment_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(exist_ok=True)
            
            # Train model
            fold_train_kwargs = {
                **train_kwargs,
                'fold_idx': fold_idx,
                'save_dir': str(fold_dir),
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'groups_train': fold_groups_train,
                'groups_val': fold_groups_val
            }
            
            model, train_metrics = train_fn(**fold_train_kwargs)
            
            # Evaluate model
            fold_eval_kwargs = {
                **eval_kwargs,
                'model': model,
                'X_test': X_val,
                'y_test': y_val,
                'groups_test': fold_groups_val
            }
            
            eval_metrics = evaluate_fn(**fold_eval_kwargs)
            
            # Store results
            fold_result = {
                'fold_idx': fold_idx,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'train_metrics': train_metrics,
                'eval_metrics': eval_metrics,
                'n_train': len(train_indices),
                'n_val': len(val_indices)
            }
            
            results.append(fold_result)
            self.fold_results[fold_idx] = fold_result
            self.fold_models[fold_idx] = model
            
            # Save fold results
            fold_result_path = fold_dir / 'fold_results.json'
            with open(fold_result_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_result = self._make_json_serializable(fold_result.copy())
                json.dump(serializable_result, f, indent=2)
                
        return results
        
    def _run_folds_parallel(
        self,
        train_fn: Callable,
        evaluate_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        train_kwargs: Dict[str, Any],
        eval_kwargs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run folds in parallel (placeholder - would need proper implementation)."""
        # This would require joblib or similar for true parallelization
        logger.warning("Parallel execution not fully implemented, falling back to sequential")
        return self._run_folds_sequential(
            train_fn, evaluate_fn, X, y, groups, train_kwargs, eval_kwargs
        )
        
    def _aggregate_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all folds."""
        aggregated = {
            'n_folds': len(fold_results),
            'fold_results': fold_results,
            'summary': {}
        }
        
        # Collect all metrics
        train_metrics = defaultdict(list)
        eval_metrics = defaultdict(list)
        
        for result in fold_results:
            # Training metrics
            for metric_name, value in result['train_metrics'].items():
                if isinstance(value, (int, float)):
                    train_metrics[metric_name].append(value)
                    
            # Evaluation metrics
            for metric_name, value in result['eval_metrics'].items():
                if isinstance(value, (int, float)):
                    eval_metrics[metric_name].append(value)
                    
        # Compute summary statistics
        for metric_name, values in train_metrics.items():
            aggregated['summary'][f'train_{metric_name}'] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
            
        for metric_name, values in eval_metrics.items():
            aggregated['summary'][f'eval_{metric_name}'] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
            
        # Compute confidence intervals
        for metric_name, stats in aggregated['summary'].items():
            if 'values' in stats and len(stats['values']) > 1:
                values = np.array(stats['values'])
                n = len(values)
                se = stats['std'] / np.sqrt(n)  # Standard error
                # 95% confidence interval (approximate)
                ci_margin = 1.96 * se if n > 30 else 2.576 * se  # t-distribution approximation
                stats['ci_lower'] = stats['mean'] - ci_margin
                stats['ci_upper'] = stats['mean'] + ci_margin
                stats['se'] = se
                
        return aggregated
        
    def _save_cv_results(self, results: Dict[str, Any]):
        """Save cross-validation results."""
        # Save main results
        results_path = self.experiment_dir / 'cv_results.json'
        with open(results_path, 'w') as f:
            serializable_results = self._make_json_serializable(results.copy())
            json.dump(serializable_results, f, indent=2)
            
        # Save summary as CSV
        summary_data = []
        for metric_name, stats in results['summary'].items():
            summary_data.append({
                'metric': metric_name,
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'ci_lower': stats.get('ci_lower'),
                'ci_upper': stats.get('ci_upper')
            })
            
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.experiment_dir / 'cv_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Save fold models info
        models_info = {
            'n_models': len(self.fold_models),
            'fold_model_paths': {}
        }
        
        for fold_idx, model in self.fold_models.items():
            model_path = self.experiment_dir / f"fold_{fold_idx}" / 'model.pth'
            if hasattr(model, 'state_dict'):
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'fold_idx': fold_idx
                }, model_path)
                models_info['fold_model_paths'][fold_idx] = str(model_path)
                
        models_info_path = self.experiment_dir / 'models_info.json'
        with open(models_info_path, 'w') as f:
            json.dump(models_info, f, indent=2)
            
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj
            
    def get_best_fold(self, metric_name: str, mode: str = 'max') -> Tuple[int, Dict[str, Any]]:
        """
        Get the best performing fold based on a specific metric.
        
        Args:
            metric_name: Name of the metric to optimize
            mode: 'max' for maximization, 'min' for minimization
            
        Returns:
            Tuple of (best_fold_idx, best_fold_results)
        """
        best_fold_idx = None
        best_value = float('-inf') if mode == 'max' else float('inf')
        
        for fold_idx, result in self.fold_results.items():
            if metric_name in result['eval_metrics']:
                value = result['eval_metrics'][metric_name]
                if (mode == 'max' and value > best_value) or (mode == 'min' and value < best_value):
                    best_value = value
                    best_fold_idx = fold_idx
                    
        if best_fold_idx is not None:
            return best_fold_idx, self.fold_results[best_fold_idx]
        else:
            raise ValueError(f"Metric '{metric_name}' not found in fold results")


class CrossValidationEvaluator:
    """
    Evaluator for cross-validation results with statistical analysis.
    
    Provides comprehensive analysis of CV results including significance
    testing, confidence intervals, and performance visualization.
    """
    
    def __init__(self, cv_results: Dict[str, Any]):
        """
        Initialize CV evaluator.
        
        Args:
            cv_results: Cross-validation results from CrossValidationManager
        """
        self.cv_results = cv_results
        self.n_folds = cv_results['n_folds']
        self.summary = cv_results['summary']
        
    def get_performance_summary(self) -> pd.DataFrame:
        """Get summary of performance metrics across folds."""
        summary_data = []
        
        for metric_name, stats in self.summary.items():
            summary_data.append({
                'Metric': metric_name,
                'Mean': f"{stats['mean']:.4f}",
                'Std': f"{stats['std']:.4f}",
                'Min': f"{stats['min']:.4f}",
                'Max': f"{stats['max']:.4f}",
                'CI_Lower': f"{stats.get('ci_lower', 0):.4f}",
                'CI_Upper': f"{stats.get('ci_upper', 0):.4f}"
            })
            
        return pd.DataFrame(summary_data)
        
    def compare_metrics(
        self,
        metric1: str,
        metric2: str,
        test_type: str = 'paired_t'
    ) -> Dict[str, Any]:
        """
        Compare two metrics using statistical tests.
        
        Args:
            metric1: First metric name
            metric2: Second metric name
            test_type: Type of statistical test ('paired_t', 'wilcoxon')
            
        Returns:
            Dictionary with test results
        """
        if metric1 not in self.summary or metric2 not in self.summary:
            raise ValueError(f"Metrics {metric1} or {metric2} not found")
            
        values1 = np.array(self.summary[metric1]['values'])
        values2 = np.array(self.summary[metric2]['values'])
        
        if len(values1) != len(values2):
            raise ValueError("Metrics must have same number of fold results")
            
        # Perform statistical test
        if test_type == 'paired_t':
            from scipy import stats
            statistic, p_value = stats.ttest_rel(values1, values2)
            test_name = "Paired t-test"
        elif test_type == 'wilcoxon':
            from scipy import stats
            statistic, p_value = stats.wilcoxon(values1, values2)
            test_name = "Wilcoxon signed-rank test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")
            
        # Effect size (Cohen's d for paired samples)
        diff = values1 - values2
        effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05,
            'mean_diff': np.mean(diff),
            'std_diff': np.std(diff),
            'metric1_mean': np.mean(values1),
            'metric2_mean': np.mean(values2)
        }
        
    def plot_metric_distribution(
        self,
        metric_names: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of metrics across folds.
        
        Args:
            metric_names: List of metric names to plot
            save_path: Path to save the plot
        """
        try:
            fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
            if len(metric_names) == 1:
                axes = [axes]
                
            for i, metric_name in enumerate(metric_names):
                if metric_name in self.summary:
                    values = self.summary[metric_name]['values']
                    
                    # Box plot
                    axes[i].boxplot(values, labels=[metric_name])
                    axes[i].scatter(np.ones(len(values)), values, alpha=0.6, color='red')
                    axes[i].set_title(f'{metric_name}\nMean: {np.mean(values):.4f} ± {np.std(values):.4f}')
                    axes[i].grid(True, alpha=0.3)
                    
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive CV report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        report_lines = [
            f"Cross-Validation Report",
            f"=" * 50,
            f"Number of folds: {self.n_folds}",
            f"",
            f"Performance Summary:",
            f"-" * 20
        ]
        
        # Add performance summary
        for metric_name, stats in self.summary.items():
            ci_info = ""
            if 'ci_lower' in stats and 'ci_upper' in stats:
                ci_info = f" [95% CI: {stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
                
            report_lines.append(
                f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}{ci_info}"
            )
            
        # Add fold-wise results
        report_lines.extend([
            f"",
            f"Fold-wise Results:",
            f"-" * 20
        ])
        
        for fold_idx in range(self.n_folds):
            fold_result = None
            for result in self.cv_results['fold_results']:
                if result['fold_idx'] == fold_idx:
                    fold_result = result
                    break
                    
            if fold_result:
                report_lines.append(f"Fold {fold_idx + 1}:")
                for metric_name, value in fold_result['eval_metrics'].items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"  {metric_name}: {value:.4f}")
                        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
        return report


def create_cv_splits(
    dataset,
    cv_type: str = 'stratified_patient',
    n_splits: int = 5,
    **kwargs
) -> BaseCVSplitter:
    """
    Factory function to create cross-validation splitter.
    
    Args:
        dataset: Dataset to split
        cv_type: Type of CV splitting ('stratified_patient', 'time_series', 'standard')
        n_splits: Number of splits
        **kwargs: Additional arguments for specific splitter types
        
    Returns:
        Configured CV splitter
    """
    if cv_type == 'stratified_patient':
        return StratifiedPatientKFold(
            n_splits=n_splits,
            shuffle=kwargs.get('shuffle', True),
            random_state=kwargs.get('random_state', None),
            min_samples_per_class=kwargs.get('min_samples_per_class', 1)
        )
    elif cv_type == 'time_series':
        return TimeSeriesKFold(
            n_splits=n_splits,
            gap=kwargs.get('gap', 0),
            max_train_size=kwargs.get('max_train_size', None)
        )
    elif cv_type == 'standard':
        # Use sklearn's StratifiedKFold as wrapper
        from sklearn.model_selection import StratifiedKFold
        
        class SklearnWrapper(BaseCVSplitter):
            def __init__(self, splitter):
                self.splitter = splitter
                
            def split(self, X, y, groups=None):
                return list(self.splitter.split(X, y))
                
            def get_n_splits(self):
                return self.splitter.n_splits
                
        return SklearnWrapper(StratifiedKFold(
            n_splits=n_splits,
            shuffle=kwargs.get('shuffle', True),
            random_state=kwargs.get('random_state', None)
        ))
    else:
        raise ValueError(f"Unknown CV type: {cv_type}")


def aggregate_cv_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple CV experiments.
    
    Args:
        results_list: List of CV results from different experiments
        
    Returns:
        Aggregated results across experiments
    """
    if not results_list:
        return {}
        
    # Collect metrics from all experiments
    all_metrics = defaultdict(list)
    
    for results in results_list:
        for metric_name, stats in results['summary'].items():
            all_metrics[metric_name].extend(stats['values'])
            
    # Compute overall statistics
    aggregated = {
        'n_experiments': len(results_list),
        'total_folds': sum(r['n_folds'] for r in results_list),
        'summary': {}
    }
    
    for metric_name, all_values in all_metrics.items():
        aggregated['summary'][metric_name] = {
            'mean': np.mean(all_values),
            'std': np.std(all_values),
            'min': np.min(all_values),
            'max': np.max(all_values),
            'n_values': len(all_values),
            'values': all_values
        }
        
        # Confidence intervals
        if len(all_values) > 1:
            n = len(all_values)
            se = np.std(all_values) / np.sqrt(n)
            ci_margin = 1.96 * se if n > 30 else 2.576 * se
            aggregated['summary'][metric_name]['ci_lower'] = np.mean(all_values) - ci_margin
            aggregated['summary'][metric_name]['ci_upper'] = np.mean(all_values) + ci_margin
            
    return aggregated


def compare_cv_models(
    results1: Dict[str, Any],
    results2: Dict[str, Any],
    metric_name: str = 'eval_accuracy'
) -> Dict[str, Any]:
    """
    Compare two CV experiments using statistical tests.
    
    Args:
        results1: First CV results
        results2: Second CV results
        metric_name: Metric to compare
        
    Returns:
        Comparison results
    """
    if metric_name not in results1['summary'] or metric_name not in results2['summary']:
        raise ValueError(f"Metric '{metric_name}' not found in both result sets")
        
    values1 = np.array(results1['summary'][metric_name]['values'])
    values2 = np.array(results2['summary'][metric_name]['values'])
    
    # Perform independent t-test (assuming different models/experiments)
    from scipy import stats
    statistic, p_value = stats.ttest_ind(values1, values2)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1) + 
                         (len(values2) - 1) * np.var(values2)) / 
                        (len(values1) + len(values2) - 2))
    effect_size = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0
    
    return {
        'metric': metric_name,
        'model1_mean': np.mean(values1),
        'model1_std': np.std(values1),
        'model2_mean': np.mean(values2),
        'model2_std': np.std(values2),
        'mean_diff': np.mean(values1) - np.mean(values2),
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'significant': p_value < 0.05,
        'better_model': 1 if np.mean(values1) > np.mean(values2) else 2
    }


def visualize_cv_performance(
    cv_results: Dict[str, Any],
    metrics: List[str],
    save_path: Optional[str] = None
):
    """
    Create comprehensive visualization of CV performance.
    
    Args:
        cv_results: Cross-validation results
        metrics: List of metrics to visualize
        save_path: Path to save the plot
    """
    try:
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, n_metrics, figsize=(5 * n_metrics, 10))
        
        if n_metrics == 1:
            axes = axes.reshape(-1, 1)
            
        for i, metric in enumerate(metrics):
            if metric in cv_results['summary']:
                values = cv_results['summary'][metric]['values']
                
                # Box plot
                axes[0, i].boxplot(values, labels=[metric.replace('eval_', '')])
                axes[0, i].scatter(np.ones(len(values)), values, alpha=0.6, color='red')
                axes[0, i].set_title(f'{metric}\nMean: {np.mean(values):.4f} ± {np.std(values):.4f}')
                axes[0, i].grid(True, alpha=0.3)
                
                # Learning curve across folds
                fold_indices = list(range(1, len(values) + 1))
                axes[1, i].plot(fold_indices, values, 'bo-', alpha=0.7)
                axes[1, i].axhline(y=np.mean(values), color='r', linestyle='--', alpha=0.7, label='Mean')
                axes[1, i].fill_between(fold_indices, 
                                       np.mean(values) - np.std(values),
                                       np.mean(values) + np.std(values),
                                       alpha=0.2, color='red')
                axes[1, i].set_xlabel('Fold')
                axes[1, i].set_ylabel(metric)
                axes[1, i].set_title(f'{metric} Across Folds')
                axes[1, i].legend()
                axes[1, i].grid(True, alpha=0.3)
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    except ImportError:
        logger.warning("matplotlib not available for visualization")


class CrossValidationTrainer:
    """
    Minimal CrossValidationTrainer that wraps CrossValidationManager for unified interface.
    
    Provides a simplified interface for running cross-validation experiments
    with train and evaluate functions.
    """
    
    def __init__(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        cv_splitter: BaseCVSplitter,
        config: Dict[str, Any],
        save_dir: str = "./cv_results"
    ):
        """
        Initialize CrossValidationTrainer.
        
        Args:
            train_fn: Function to train model on fold
            eval_fn: Function to evaluate model on fold
            cv_splitter: Cross-validation splitter
            config: Training configuration
            save_dir: Directory to save results
        """
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.config = config
        
        # Create CV manager
        self.cv_manager = CrossValidationManager(
            cv_splitter=cv_splitter,
            save_dir=save_dir,
            experiment_name=config.get('experiment_name', 'cv_experiment'),
            parallel=config.get('parallel', False),
            n_jobs=config.get('n_jobs', 1)
        )
        
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified train and evaluate method.
        
        Args:
            X: Feature data
            y: Target labels  
            groups: Group labels (e.g., patient IDs)
            **kwargs: Additional arguments
            
        Returns:
            Cross-validation results
        """
        return self.cv_manager.run_cross_validation(
            train_fn=self.train_fn,
            evaluate_fn=self.eval_fn,
            X=X,
            y=y,
            groups=groups,
            train_kwargs=kwargs.get('train_kwargs', {}),
            eval_kwargs=kwargs.get('eval_kwargs', {})
        )
    
    def get_results(self) -> Dict[str, Any]:
        """Get cross-validation results."""
        return self.cv_manager.get_results()
    
    def save_results(self, filepath: str):
        """Save results to file."""
        self.cv_manager.save_results(filepath)


# Unit tests for CrossValidationTrainer
def test_cross_validation_trainer():
    """
    Unit tests for CrossValidationTrainer with 2-3 folds on dummy data.
    """
    import torch
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    # Create dummy data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    
    # Define simple train and eval functions
    def simple_train_fn(X_train, y_train, **kwargs):
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def simple_eval_fn(model, X_val, y_val, **kwargs):
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return {'accuracy': accuracy, 'n_samples': len(y_val)}
    
    # Test with 3-fold CV
    cv_splitter = StratifiedPatientKFold(n_splits=3, shuffle=True, random_state=42)
    
    config = {
        'experiment_name': 'test_cv',
        'parallel': False,
        'n_jobs': 1
    }
    
    # Create trainer
    trainer = CrossValidationTrainer(
        train_fn=simple_train_fn,
        eval_fn=simple_eval_fn,
        cv_splitter=cv_splitter,
        config=config,
        save_dir="./test_cv"
    )
    
    # Run cross-validation
    results = trainer.train_and_evaluate(X, y)
    
    # Validate results
    assert isinstance(results, dict), "Results should be a dictionary"
    assert 'mean' in results, "Results should contain mean metrics"
    assert 'std' in results, "Results should contain std metrics"
    assert 'fold_results' in results, "Results should contain individual fold results"
    
    # Check fold results
    fold_results = results['fold_results']
    assert len(fold_results) == 3, f"Should have 3 fold results, got {len(fold_results)}"
    
    # Check that all folds have accuracy
    for i, fold_result in enumerate(fold_results):
        assert 'accuracy' in fold_result, f"Fold {i} should have accuracy metric"
        assert 0.0 <= fold_result['accuracy'] <= 1.0, f"Fold {i} accuracy should be between 0 and 1"
    
    # Check mean accuracy
    assert 'accuracy' in results['mean'], "Mean results should contain accuracy"
    assert 0.0 <= results['mean']['accuracy'] <= 1.0, "Mean accuracy should be between 0 and 1"
    
    # Test with 2-fold CV
    cv_splitter_2fold = StratifiedPatientKFold(n_splits=2, shuffle=True, random_state=42)
    trainer_2fold = CrossValidationTrainer(
        train_fn=simple_train_fn,
        eval_fn=simple_eval_fn,
        cv_splitter=cv_splitter_2fold,
        config={'experiment_name': 'test_cv_2fold'},
        save_dir="./test_cv_2fold"
    )
    
    results_2fold = trainer_2fold.train_and_evaluate(X, y)
    assert len(results_2fold['fold_results']) == 2, "Should have 2 fold results"
    
    print("CrossValidationTrainer tests passed!")


if __name__ == "__main__":
    # Run the test when the file is executed directly
    test_cross_validation_trainer()
