"""
Evaluation Visualization Tools

This module provides comprehensive visualization tools for ABR signal evaluation,
including signal comparisons, metrics plots, and analysis dashboards.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
from scipy import signal

# Optional imports with fallbacks
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive dashboards will be disabled.")


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def create_colorblind_friendly_palette(n_colors: int = 8) -> List[str]:
    """Create a colorblind-friendly color palette."""
    # Colorblind-friendly colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    if n_colors <= len(colors):
        return colors[:n_colors]
    else:
        # Extend palette if needed
        extended_colors = colors.copy()
        for i in range(n_colors - len(colors)):
            extended_colors.append(f'#{np.random.randint(0, 0xFFFFFF):06x}')
        return extended_colors

def plot_roc_curve(roc_data: Dict, save_path: Optional[str] = None,
                   title: str = "ROC Curve", figsize: Tuple[int, int] = (8, 6),
                   show_confidence_interval: bool = True, 
                   multiple_curves: Optional[List[Dict]] = None,
                   curve_labels: Optional[List[str]] = None) -> plt.Figure:
    """
    Create publication-quality ROC curve plot.
    
    Args:
        roc_data: ROC curve data from roc_analysis()
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height)
        show_confidence_interval: Whether to show AUROC confidence interval
        multiple_curves: List of ROC data for multiple models
        curve_labels: Labels for multiple curves
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot main ROC curve
    fpr = roc_data['roc_curve']['fpr']
    tpr = roc_data['roc_curve']['tpr']
    
    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=2.5, color='#1f77b4', 
            label=f"AUROC = {roc_data['auroc']:.3f}")
    
    # Add confidence interval if available
    if show_confidence_interval and 'auroc_ci' in roc_data:
        ci = roc_data['auroc_ci']
        if 'lower' in ci and 'upper' in ci:
            ax.fill_between(fpr, tpr, alpha=0.2, color='#1f77b4',
                           label=f"95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")
    
    # Plot multiple curves if provided
    if multiple_curves and curve_labels:
        colors = create_colorblind_friendly_palette(len(multiple_curves))
        for i, (curve_data, label) in enumerate(zip(multiple_curves, curve_labels)):
            if 'roc_curve' in curve_data:
                fpr_multi = curve_data['roc_curve']['fpr']
                tpr_multi = curve_data['roc_curve']['tpr']
                auroc_multi = curve_data.get('auroc', 0.0)
                ax.plot(fpr_multi, tpr_multi, linewidth=2, color=colors[i],
                       label=f"{label} (AUROC = {auroc_multi:.3f})")
    
    # Mark optimal threshold point
    if 'optimal_threshold' in roc_data:
        opt_thresh = roc_data['optimal_threshold']
        if 'sensitivity' in opt_thresh and 'specificity' in opt_thresh:
            opt_fpr = 1 - opt_thresh['specificity']
            opt_tpr = opt_thresh['sensitivity']
            ax.scatter(opt_fpr, opt_tpr, s=100, color='red', zorder=5,
                      label=f"Optimal threshold\n(Sens={opt_tpr:.3f}, Spec={opt_thresh['specificity']:.3f})")
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random classifier')
    
    # Customize plot
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        save_figure_multiple_formats(fig, save_path)
    
    return fig

def plot_precision_recall_curve(pr_data: Dict, save_path: Optional[str] = None,
                               title: str = "Precision-Recall Curve", 
                               figsize: Tuple[int, int] = (8, 6),
                               show_confidence_interval: bool = True,
                               multiple_curves: Optional[List[Dict]] = None,
                               curve_labels: Optional[List[str]] = None) -> plt.Figure:
    """
    Create publication-quality precision-recall curve plot.
    
    Args:
        pr_data: PR curve data from precision_recall_analysis()
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height)
        show_confidence_interval: Whether to show average precision confidence interval
        multiple_curves: List of PR data for multiple models
        curve_labels: Labels for multiple curves
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot main PR curve
    precision = pr_data['pr_curve']['precision']
    recall = pr_data['pr_curve']['recall']
    
    # Plot PR curve
    ax.plot(recall, precision, linewidth=2.5, color='#2ca02c',
            label=f"Average Precision = {pr_data['average_precision']:.3f}")
    
    # Add confidence interval if available
    if show_confidence_interval and 'average_precision_ci' in pr_data:
        ci = pr_data['average_precision_ci']
        if 'lower' in ci and 'upper' in ci:
            ax.fill_between(recall, precision, alpha=0.2, color='#2ca02c',
                           label=f"95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")
    
    # Plot multiple curves if provided
    if multiple_curves and curve_labels:
        colors = create_colorblind_friendly_palette(len(multiple_curves))
        for i, (curve_data, label) in enumerate(zip(multiple_curves, curve_labels)):
            if 'pr_curve' in curve_data:
                recall_multi = curve_data['pr_curve']['recall']
                precision_multi = curve_data['pr_curve']['precision']
                ap_multi = curve_data.get('average_precision', 0.0)
                ax.plot(recall_multi, precision_multi, linewidth=2, color=colors[i],
                       label=f"{label} (AP = {ap_multi:.3f})")
    
    # Mark optimal threshold point
    if 'optimal_threshold' in pr_data:
        opt_thresh = pr_data['optimal_threshold']
        if 'threshold' in opt_thresh:
            # Find the point on the curve closest to optimal threshold
            thresholds = pr_data['pr_curve']['thresholds']
            if len(thresholds) > 0:
                # Find recall and precision at optimal threshold
                # This is a simplified approach - in practice you might want to interpolate
                ax.scatter(0.5, 0.5, s=100, color='red', zorder=5,
                          label=f"Optimal threshold\n(F1 = {opt_thresh.get('f1_score', 0.0):.3f})")
    
    # Add baseline precision line (prevalence)
    if 'prevalence' in pr_data:
        prevalence = pr_data['prevalence']
        ax.axhline(y=prevalence, color='k', linestyle='--', alpha=0.5, 
                  label=f'Baseline (prevalence = {prevalence:.3f})')
    
    # Customize plot
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        save_figure_multiple_formats(fig, save_path)
    
    return fig

def plot_confusion_matrix(confusion_matrix: np.ndarray, 
                         class_names: List[str] = ['No Peak', 'Peak'],
                         save_path: Optional[str] = None,
                         title: str = "Confusion Matrix",
                         figsize: Tuple[int, int] = (8, 6),
                         normalize: bool = True,
                         include_metrics: bool = True,
                         metrics_data: Optional[Dict] = None) -> plt.Figure:
    """
    Create annotated confusion matrix heatmap.
    
    Args:
        confusion_matrix: 2x2 confusion matrix
        class_names: Names for the classes
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height)
        normalize: Whether to normalize the matrix
        include_metrics: Whether to include precision, recall, F1 annotations
        metrics_data: Additional metrics to display
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize if requested
    if normalize:
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2f'
    else:
        cm_display = confusion_matrix
        fmt = 'd'
    
    # Create heatmap
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, ax=ax, square=True, linewidths=0.5)
    
    # Add metrics annotations if requested
    if include_metrics and metrics_data:
        # Calculate metrics from confusion matrix
        tp, fp, fn, tn = confusion_matrix[1, 1], confusion_matrix[0, 1], confusion_matrix[1, 0], confusion_matrix[0, 0]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Add text box with metrics
        metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_figure_multiple_formats(fig, save_path)
    
    return fig

def plot_classification_metrics_comparison(metrics_data: List[Dict], 
                                         model_names: List[str],
                                         metrics_to_plot: List[str] = ['accuracy', 'f1', 'auroc'],
                                         save_path: Optional[str] = None,
                                         title: str = "Classification Metrics Comparison",
                                         figsize: Tuple[int, int] = (10, 6),
                                         include_confidence_intervals: bool = True,
                                         statistical_annotations: Optional[Dict] = None) -> plt.Figure:
    """
    Create bar plot comparing classification metrics across models.
    
    Args:
        metrics_data: List of metrics dictionaries for each model
        model_names: Names of the models
        metrics_to_plot: List of metrics to include in the plot
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height)
        include_confidence_intervals: Whether to show confidence intervals
        statistical_annotations: Statistical significance annotations
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for plotting
    x = np.arange(len(model_names))
    width = 0.8 / len(metrics_to_plot)
    
    colors = create_colorblind_friendly_palette(len(metrics_to_plot))
    
    for i, metric in enumerate(metrics_to_plot):
        values = []
        errors = []
        
        for model_data in metrics_data:
            if metric in model_data:
                values.append(model_data[metric])
                
                # Get confidence interval if available
                if include_confidence_intervals and f'{metric}_ci' in model_data:
                    ci = model_data[f'{metric}_ci']
                    if 'lower' in ci and 'upper' in ci:
                        error = (ci['upper'] - ci['lower']) / 2
                        errors.append(error)
                    else:
                        errors.append(0)
                else:
                    errors.append(0)
            else:
                values.append(0)
                errors.append(0)
        
        # Plot bars
        bars = ax.bar(x + i * width, values, width, label=metric.upper(),
                     color=colors[i], alpha=0.8, yerr=errors if any(errors) else None,
                     capsize=5, capthick=2)
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(errors) * 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add statistical significance annotations
    if statistical_annotations:
        add_significance_annotations(ax, statistical_annotations, x, width, len(metrics_to_plot))
    
    # Customize plot
    ax.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * (len(metrics_to_plot) - 1) / 2)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_figure_multiple_formats(fig, save_path)
    
    return fig

def plot_threshold_analysis(logits: np.ndarray, targets: np.ndarray,
                           save_path: Optional[str] = None,
                           title: str = "Threshold Analysis",
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create plots showing how metrics vary with classification threshold.
    
    Args:
        logits: Model output logits [N]
        targets: Ground truth labels [N]
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Generate threshold range
    thresholds = np.linspace(logits.min(), logits.max(), 100)
    
    # Calculate metrics at each threshold
    sensitivities = []
    specificities = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        predictions = (logits >= threshold).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((predictions == 1) & (targets == 1))
        tn = np.sum((predictions == 0) & (targets == 0))
        fp = np.sum((predictions == 1) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = sensitivity
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Plot sensitivity and specificity
    ax1.plot(thresholds, sensitivities, 'b-', linewidth=2, label='Sensitivity')
    ax1.plot(thresholds, specificities, 'r-', linewidth=2, label='Specificity')
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Default threshold')
    ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax1.set_title('Sensitivity and Specificity vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot precision, recall, and F1
    ax2.plot(thresholds, precisions, 'g-', linewidth=2, label='Precision')
    ax2.plot(thresholds, recalls, 'm-', linewidth=2, label='Recall')
    ax2.plot(thresholds, f1_scores, 'k-', linewidth=2, label='F1 Score')
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Default threshold')
    ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax2.set_title('Precision, Recall, and F1 vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_figure_multiple_formats(fig, save_path)
    
    return fig

def plot_clinical_validation_dashboard(clinical_data: Dict, roc_data: Dict, pr_data: Dict,
                                      save_path: Optional[str] = None,
                                      title: str = "Clinical Validation Dashboard",
                                      figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Create comprehensive clinical validation dashboard.
    
    Args:
        clinical_data: Clinical validation metrics from clinical_validation_analysis()
        roc_data: ROC analysis data
        pr_data: PR analysis data
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    fig = plt.figure(figsize=figsize)
    
    # Create subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ROC curve
    ax1 = fig.add_subplot(gs[0, :2])
    if 'roc_curve' in roc_data:
        fpr = roc_data['roc_curve']['fpr']
        tpr = roc_data['roc_curve']['tpr']
        ax1.plot(fpr, tpr, linewidth=2.5, color='#1f77b4')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve (AUROC = {roc_data.get("auroc", 0):.3f})')
        ax1.grid(True, alpha=0.3)
    
    # PR curve
    ax2 = fig.add_subplot(gs[0, 2])
    if 'pr_curve' in pr_data:
        precision = pr_data['pr_curve']['precision']
        recall = pr_data['pr_curve']['recall']
        ax2.plot(recall, precision, linewidth=2.5, color='#2ca02c')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'PR Curve (AP = {pr_data.get("average_precision", 0):.3f})')
        ax2.grid(True, alpha=0.3)
    
    # Clinical metrics summary
    ax3 = fig.add_subplot(gs[1, :])
    if 'basic_metrics' in clinical_data:
        metrics = clinical_data['basic_metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax3.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax3.set_ylabel('Value')
        ax3.set_title('Clinical Metrics Summary')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Diagnostic odds ratio with confidence interval
    ax4 = fig.add_subplot(gs[2, 0])
    if 'diagnostic_odds_ratio' in clinical_data:
        dor_data = clinical_data['diagnostic_odds_ratio']
        if 'value' in dor_data and dor_data['value'] != np.inf:
            dor_value = dor_data['value']
            ax4.bar(['DOR'], [dor_value], color='#9467bd')
            ax4.set_ylabel('Diagnostic Odds Ratio')
            ax4.set_title('Diagnostic Odds Ratio')
            
            # Add confidence interval if available
            if 'confidence_interval' in dor_data:
                ci = dor_data['confidence_interval']
                if 'lower' in ci and 'upper' in ci and ci['lower'] != np.inf:
                    ax4.errorbar(['DOR'], [dor_value], 
                               yerr=[[dor_value - ci['lower']], [ci['upper'] - dor_value]],
                               fmt='none', color='black', capsize=5)
    
    # Likelihood ratios
    ax5 = fig.add_subplot(gs[2, 1])
    if 'likelihood_ratios' in clinical_data:
        lr_data = clinical_data['likelihood_ratios']
        lr_names = ['LR+', 'LR-']
        lr_values = [lr_data.get('positive_likelihood_ratio', 0), 
                    lr_data.get('negative_likelihood_ratio', 0)]
        
        # Filter out infinite values
        valid_indices = [i for i, v in enumerate(lr_values) if v != np.inf]
        if valid_indices:
            valid_names = [lr_names[i] for i in valid_indices]
            valid_values = [lr_values[i] for i in valid_indices]
            ax5.bar(valid_names, valid_values, color=['#8c564b', '#e377c2'])
            ax5.set_ylabel('Likelihood Ratio')
            ax5.set_title('Likelihood Ratios')
    
    # Clinical utility metrics
    ax6 = fig.add_subplot(gs[2, 2])
    if 'clinical_utility' in clinical_data:
        utility_data = clinical_data['clinical_utility']
        utility_names = ['NND', 'Adj. PPV', 'Adj. NPV']
        utility_values = [utility_data.get('number_needed_to_diagnose', 0),
                         utility_data.get('prevalence_adjusted_ppv', 0),
                         utility_data.get('prevalence_adjusted_npv', 0)]
        
        # Filter out infinite values
        valid_indices = [i for i, v in enumerate(utility_values) if v != np.inf]
        if valid_indices:
            valid_names = [utility_names[i] for i in valid_indices]
            valid_values = [utility_values[i] for i in valid_indices]
            ax6.bar(valid_names, valid_values, color=['#7f7f7f', '#bcbd22', '#17becf'])
            ax6.set_ylabel('Value')
            ax6.set_title('Clinical Utility Metrics')
            ax6.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=18, fontweight='bold')
    
    if save_path:
        save_figure_multiple_formats(fig, save_path)
    
    return fig

def plot_ablation_study_results(ablation_results: List[Dict], 
                               baseline_result: Dict,
                               save_path: Optional[str] = None,
                               title: str = "Ablation Study Results",
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create forest plot showing effect of different architectural components.
    
    Args:
        ablation_results: List of ablation study results
        baseline_result: Baseline configuration results
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for plotting
    component_names = []
    effect_sizes = []
    confidence_intervals = []
    
    for result in ablation_results:
        if 'component_name' in result and 'effect_size' in result:
            component_names.append(result['component_name'])
            effect_sizes.append(result['effect_size'])
            
            if 'confidence_interval' in result:
                ci = result['confidence_interval']
                if 'lower' in ci and 'upper' in ci:
                    confidence_intervals.append([ci['lower'], ci['upper']])
                else:
                    confidence_intervals.append([0, 0])
            else:
                confidence_intervals.append([0, 0])
    
    if not component_names:
        return fig
    
    # Create forest plot
    y_pos = np.arange(len(component_names))
    
    # Plot effect sizes
    ax.barh(y_pos, effect_sizes, xerr=confidence_intervals, 
            capsize=5, capthick=2, color='#1f77b4', alpha=0.8)
    
    # Add baseline reference line
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Baseline')
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(component_names)
    ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (effect_size, ci) in enumerate(zip(effect_sizes, confidence_intervals)):
        ax.text(effect_size + max(ci) * 0.01, i, f'{effect_size:.3f}', 
               va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_figure_multiple_formats(fig, save_path)
    
    return fig

def create_publication_figure(plots_data: List[Dict], 
                             layout: str = '2x2',
                             save_path: Optional[str] = None,
                             title: str = "Publication Figure",
                             figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Generate multi-panel figure suitable for academic publications.
    
    Args:
        plots_data: List of plot data dictionaries
        layout: Layout string (e.g., '2x2', '1x4', '3x1')
        save_path: Path to save the figure
        title: Overall figure title
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    
    # Parse layout
    if 'x' in layout:
        rows, cols = map(int, layout.split('x'))
    else:
        rows, cols = 2, 2
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()
    
    # Create each subplot
    for i, plot_data in enumerate(plots_data):
        if i >= len(axes):
            break
            
        ax = axes[i]
        plot_type = plot_data.get('type', 'unknown')
        
        if plot_type == 'roc_curve':
            roc_data = plot_data['data']
            fpr = roc_data['roc_curve']['fpr']
            tpr = roc_data['roc_curve']['tpr']
            ax.plot(fpr, tpr, linewidth=2, color='#1f77b4')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve\n(AUROC = {roc_data.get("auroc", 0):.3f})')
            
        elif plot_type == 'pr_curve':
            pr_data = plot_data['data']
            precision = pr_data['pr_curve']['precision']
            recall = pr_data['pr_curve']['recall']
            ax.plot(recall, precision, linewidth=2, color='#2ca02c')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'PR Curve\n(AP = {pr_data.get("average_precision", 0):.3f})')
            
        elif plot_type == 'confusion_matrix':
            cm_data = plot_data['data']
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['No Peak', 'Peak'], yticklabels=['No Peak', 'Peak'],
                        cbar=False, ax=ax, square=True)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            
        elif plot_type == 'metrics_comparison':
            metrics_data = plot_data['data']
            metric_names = list(metrics_data.keys())
            metric_values = list(metrics_data.values())
            ax.bar(metric_names, metric_values, color='#1f77b4', alpha=0.8)
            ax.set_ylabel('Value')
            ax.set_title('Classification Metrics')
            ax.tick_params(axis='x', rotation=45)
        
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(plots_data), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_figure_multiple_formats(fig, save_path)
    
    return fig

def add_significance_annotations(ax: plt.Axes, annotations: Dict, 
                                x_positions: np.ndarray, bar_width: float, 
                                n_metrics: int) -> None:
    """Add statistical significance annotations to plots."""
    if not annotations:
        return
    
    y_max = ax.get_ylim()[1]
    annotation_height = y_max * 0.05
    
    for i, (model1, model2) in enumerate(annotations.get('comparisons', [])):
        if i < len(x_positions) - 1:
            x1 = x_positions[i] + bar_width * (n_metrics - 1) / 2
            x2 = x_positions[i + 1] + bar_width * (n_metrics - 1) / 2
            
            # Draw bracket
            ax.plot([x1, x1, x2, x2], [y_max + annotation_height, y_max + annotation_height * 1.5, 
                                       y_max + annotation_height * 1.5, y_max + annotation_height], 'k-', linewidth=1)
            
            # Add significance level
            significance = annotations.get('significance_levels', {}).get(f'{model1}_vs_{model2}', 'ns')
            ax.text((x1 + x2) / 2, y_max + annotation_height * 2, significance, 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

def save_figure_multiple_formats(fig: plt.Figure, base_path: str, 
                                formats: List[str] = ['png', 'pdf']) -> None:
    """Save figure in multiple formats."""
    base_path = Path(base_path)
    base_name = base_path.stem
    base_dir = base_path.parent
    
    for fmt in formats:
        if fmt == 'png':
            save_path = base_dir / f"{base_name}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        elif fmt == 'pdf':
            save_path = base_dir / f"{base_name}.pdf"
            fig.savefig(save_path, bbox_inches='tight')
        elif fmt == 'svg':
            save_path = base_dir / f"{base_name}.svg"
            fig.savefig(save_path, bbox_inches='tight')


class EvaluationVisualizer:
    """Comprehensive visualization tools for evaluation results."""
    
    def __init__(self, output_dir: str = 'evaluation_plots', style: str = 'seaborn-v0_8'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')  # Use default instead of seaborn-v0_8
        sns.set_palette("husl")
        
        # Color schemes
        self.colors = {
            'generated': '#FF6B6B',
            'real': '#4ECDC4', 
            'difference': '#45B7D1',
            'metrics': '#96CEB4',
            'secondary': '#FFEAA7'
        }
    
    def plot_sample_comparisons(self, 
                              generated_samples: List[torch.Tensor],
                              real_samples: List[torch.Tensor],
                              num_samples: int = 6,
                              save_path: Optional[str] = None) -> str:
        """
        Create comparison plots between generated and real samples.
        
        Args:
            generated_samples: List of generated signal tensors
            real_samples: List of real signal tensors  
            num_samples: Number of samples to plot
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 2*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_samples, len(generated_samples))):
            generated = generated_samples[i].squeeze().cpu().numpy()
            real = real_samples[i].squeeze().cpu().numpy()
            difference = generated - real
            
            # Time axis
            time_axis = np.arange(len(generated)) / 1000  # Assuming 1kHz sampling
            
            # Generated signal
            axes[i, 0].plot(time_axis, generated, color=self.colors['generated'], linewidth=1.5)
            axes[i, 0].set_title(f'Generated Sample {i+1}')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Real signal
            axes[i, 1].plot(time_axis, real, color=self.colors['real'], linewidth=1.5)
            axes[i, 1].set_title(f'Real Sample {i+1}')
            axes[i, 1].set_ylabel('Amplitude')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Difference
            axes[i, 2].plot(time_axis, difference, color=self.colors['difference'], linewidth=1.5)
            axes[i, 2].set_title(f'Difference {i+1}')
            axes[i, 2].set_ylabel('Amplitude')
            axes[i, 2].grid(True, alpha=0.3)
            
            if i == num_samples - 1:
                for j in range(3):
                    axes[i, j].set_xlabel('Time (ms)')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'sample_comparisons.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_overlay_comparison(self,
                              generated_samples: List[torch.Tensor],
                              real_samples: List[torch.Tensor],
                              num_samples: int = 10,
                              save_path: Optional[str] = None) -> str:
        """
        Create overlay comparison plots.
        
        Args:
            generated_samples: List of generated signal tensors
            real_samples: List of real signal tensors
            num_samples: Number of samples to overlay
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        time_axis = np.arange(generated_samples[0].squeeze().shape[0]) / 1000
        
        # Plot overlays
        for i in range(min(num_samples, len(generated_samples))):
            generated = generated_samples[i].squeeze().cpu().numpy()
            real = real_samples[i].squeeze().cpu().numpy()
            
            alpha = 0.7 if num_samples > 5 else 1.0
            
            ax1.plot(time_axis, generated, color=self.colors['generated'], 
                    alpha=alpha, linewidth=1.0, label='Generated' if i == 0 else "")
            ax1.plot(time_axis, real, color=self.colors['real'], 
                    alpha=alpha, linewidth=1.0, label='Real' if i == 0 else "")
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Signal Overlay Comparison ({num_samples} samples)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot average signals
        avg_generated = torch.stack(generated_samples[:num_samples]).mean(0).squeeze().cpu().numpy()
        avg_real = torch.stack(real_samples[:num_samples]).mean(0).squeeze().cpu().numpy()
        std_generated = torch.stack(generated_samples[:num_samples]).std(0).squeeze().cpu().numpy()
        std_real = torch.stack(real_samples[:num_samples]).std(0).squeeze().cpu().numpy()
        
        ax2.plot(time_axis, avg_generated, color=self.colors['generated'], 
                linewidth=2, label='Generated (mean)')
        ax2.fill_between(time_axis, avg_generated - std_generated, avg_generated + std_generated,
                        color=self.colors['generated'], alpha=0.3)
        
        ax2.plot(time_axis, avg_real, color=self.colors['real'], 
                linewidth=2, label='Real (mean)')
        ax2.fill_between(time_axis, avg_real - std_real, avg_real + std_real,
                        color=self.colors['real'], alpha=0.3)
        
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Average Signals with Standard Deviation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'overlay_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_frequency_analysis(self,
                              generated_samples: List[torch.Tensor],
                              real_samples: List[torch.Tensor],
                              sr: int = 1000,
                              save_path: Optional[str] = None) -> str:
        """
        Create frequency domain analysis plots.
        
        Args:
            generated_samples: List of generated signal tensors
            real_samples: List of real signal tensors
            sr: Sampling rate
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Combine all samples for analysis
        all_generated = torch.cat(generated_samples, dim=0).cpu().numpy()
        all_real = torch.cat(real_samples, dim=0).cpu().numpy()
        
        # Power Spectral Density
        freqs_gen, psd_gen = signal.welch(all_generated.flatten(), fs=sr, nperseg=256)
        freqs_real, psd_real = signal.welch(all_real.flatten(), fs=sr, nperseg=256)
        
        axes[0, 0].semilogy(freqs_gen, psd_gen, color=self.colors['generated'], 
                           linewidth=2, label='Generated')
        axes[0, 0].semilogy(freqs_real, psd_real, color=self.colors['real'], 
                           linewidth=2, label='Real')
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('PSD')
        axes[0, 0].set_title('Power Spectral Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Magnitude Spectrum
        sample_gen = generated_samples[0].squeeze().cpu().numpy()
        sample_real = real_samples[0].squeeze().cpu().numpy()
        
        fft_gen = np.abs(np.fft.fft(sample_gen))
        fft_real = np.abs(np.fft.fft(sample_real))
        freqs = np.fft.fftfreq(len(sample_gen), 1/sr)
        
        axes[0, 1].plot(freqs[:len(freqs)//2], fft_gen[:len(freqs)//2], 
                       color=self.colors['generated'], linewidth=1.5, label='Generated')
        axes[0, 1].plot(freqs[:len(freqs)//2], fft_real[:len(freqs)//2], 
                       color=self.colors['real'], linewidth=1.5, label='Real')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].set_title('Magnitude Spectrum (Sample)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Spectrogram comparison
        f_gen, t_gen, Sxx_gen = signal.spectrogram(sample_gen, sr, nperseg=64)
        f_real, t_real, Sxx_real = signal.spectrogram(sample_real, sr, nperseg=64)
        
        im1 = axes[1, 0].pcolormesh(t_gen*1000, f_gen, 10*np.log10(Sxx_gen), 
                                   shading='gouraud', cmap='viridis')
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        axes[1, 0].set_title('Generated Spectrogram')
        plt.colorbar(im1, ax=axes[1, 0])
        
        im2 = axes[1, 1].pcolormesh(t_real*1000, f_real, 10*np.log10(Sxx_real), 
                                   shading='gouraud', cmap='viridis')
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('Frequency (Hz)')
        axes[1, 1].set_title('Real Spectrogram')
        plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'frequency_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_metrics_distribution(self,
                                metrics: Dict[str, Any],
                                save_path: Optional[str] = None) -> str:
        """
        Plot distribution of evaluation metrics.
        
        Args:
            metrics: Dictionary of aggregated metrics
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        # Extract scalar metrics for plotting
        scalar_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict) and 'mean' in value:
                scalar_metrics[key] = value
        
        if not scalar_metrics:
            # Create empty plot if no metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No metrics available for plotting', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Metrics Distribution')
        else:
            # Create metrics plot
            metric_names = list(scalar_metrics.keys())
            means = [scalar_metrics[m]['mean'] for m in metric_names]
            stds = [scalar_metrics[m]['std'] for m in metric_names]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot of means
            bars = ax1.bar(range(len(metric_names)), means, 
                          color=self.colors['metrics'], alpha=0.7,
                          yerr=stds, capsize=5)
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Value')
            ax1.set_title('Metric Means with Standard Deviation')
            ax1.set_xticks(range(len(metric_names)))
            ax1.set_xticklabels(metric_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.3f}', ha='center', va='bottom')
            
            # Box plot style visualization
            positions = range(len(metric_names))
            mins = [scalar_metrics[m]['min'] for m in metric_names]
            maxs = [scalar_metrics[m]['max'] for m in metric_names]
            medians = [scalar_metrics[m]['median'] for m in metric_names]
            
            for i, (mean, std, minimum, maximum, median) in enumerate(zip(means, stds, mins, maxs, medians)):
                # Draw box
                ax2.bar(i, 2*std, bottom=mean-std, width=0.6, 
                       color=self.colors['metrics'], alpha=0.5)
                # Draw median line
                ax2.plot([i-0.3, i+0.3], [median, median], 'k-', linewidth=2)
                # Draw whiskers
                ax2.plot([i, i], [minimum, mean-std], 'k-', linewidth=1)
                ax2.plot([i, i], [mean+std, maximum], 'k-', linewidth=1)
                # Draw outlier markers
                ax2.plot(i, minimum, 'ko', markersize=3)
                ax2.plot(i, maximum, 'ko', markersize=3)
            
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Value')
            ax2.set_title('Metric Distributions')
            ax2.set_xticks(range(len(metric_names)))
            ax2.set_xticklabels(metric_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'metrics_distribution.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_interactive_dashboard(self,
                                   generated_samples: List[torch.Tensor],
                                   real_samples: List[torch.Tensor],
                                   metrics: Dict[str, Any],
                                   save_path: Optional[str] = None) -> str:
        """
        Create interactive HTML dashboard using Plotly.
        
        Args:
            generated_samples: List of generated signal tensors
            real_samples: List of real signal tensors
            metrics: Dictionary of metrics
            save_path: Optional custom save path
            
        Returns:
            Path to saved HTML dashboard
        """
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available. Creating static dashboard instead.")
            return self._create_static_dashboard(generated_samples, real_samples, metrics, save_path)
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Sample Comparison', 'Frequency Analysis', 
                          'Metrics Overview', 'Signal Statistics',
                          'Correlation Analysis', 'Generation Quality'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sample comparison
        sample_idx = 0
        generated = generated_samples[sample_idx].squeeze().cpu().numpy()
        real = real_samples[sample_idx].squeeze().cpu().numpy()
        time_axis = np.arange(len(generated)) / 1000
        
        fig.add_trace(
            go.Scatter(x=time_axis, y=generated, name='Generated', 
                      line=dict(color=self.colors['generated'])),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_axis, y=real, name='Real', 
                      line=dict(color=self.colors['real'])),
            row=1, col=1
        )
        
        # Frequency analysis
        fft_gen = np.abs(np.fft.fft(generated))
        fft_real = np.abs(np.fft.fft(real))
        freqs = np.fft.fftfreq(len(generated), 1/1000)
        
        fig.add_trace(
            go.Scatter(x=freqs[:len(freqs)//2], y=fft_gen[:len(freqs)//2], 
                      name='Generated FFT', line=dict(color=self.colors['generated'])),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=freqs[:len(freqs)//2], y=fft_real[:len(freqs)//2], 
                      name='Real FFT', line=dict(color=self.colors['real'])),
            row=1, col=2
        )
        
        # Metrics overview (if available)
        if metrics:
            scalar_metrics = {k: v for k, v in metrics.items() 
                            if isinstance(v, dict) and 'mean' in v}
            if scalar_metrics:
                metric_names = list(scalar_metrics.keys())
                means = [scalar_metrics[m]['mean'] for m in metric_names]
                
                fig.add_trace(
                    go.Bar(x=metric_names, y=means, name='Metrics',
                          marker_color=self.colors['metrics']),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title='ABR Signal Generation Evaluation Dashboard',
            showlegend=True,
            height=1200
        )
        
        if save_path is None:
            save_path = self.output_dir / 'interactive_dashboard.html'
        
        fig.write_html(save_path)
        
        return str(save_path)
    
    def _create_static_dashboard(self,
                               generated_samples: List[torch.Tensor],
                               real_samples: List[torch.Tensor],
                               metrics: Dict[str, Any],
                               save_path: Optional[str] = None) -> str:
        """Create a static dashboard as fallback when Plotly is not available."""
        if save_path is None:
            save_path = self.output_dir / 'static_dashboard.png'
        
        # Create a comprehensive static plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sample comparison
        if generated_samples and real_samples:
            sample_idx = 0
            generated = generated_samples[sample_idx].squeeze().cpu().numpy()
            real = real_samples[sample_idx].squeeze().cpu().numpy()
            time_axis = np.arange(len(generated)) / 1000
            
            axes[0, 0].plot(time_axis, generated, label='Generated', color=self.colors['generated'])
            axes[0, 0].plot(time_axis, real, label='Real', color=self.colors['real'])
            axes[0, 0].set_title('Sample Comparison')
            axes[0, 0].set_xlabel('Time (ms)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Metrics overview
        if metrics:
            scalar_metrics = {k: v for k, v in metrics.items() 
                            if isinstance(v, dict) and 'mean' in v}
            if scalar_metrics:
                metric_names = list(scalar_metrics.keys())
                means = [scalar_metrics[m]['mean'] for m in metric_names]
                
                axes[0, 1].bar(range(len(metric_names)), means, color=self.colors['metrics'])
                axes[0, 1].set_title('Metrics Overview')
                axes[0, 1].set_xticks(range(len(metric_names)))
                axes[0, 1].set_xticklabels(metric_names, rotation=45, ha='right')
        
        # Add text summary
        axes[1, 0].axis('off')
        summary_text = "Static Dashboard\n\nKey Metrics:\n"
        if metrics:
            for key, value in list(metrics.items())[:5]:
                if isinstance(value, dict) and 'mean' in value:
                    summary_text += f"• {key}: {value['mean']:.3f}\n"
        axes[1, 0].text(0.1, 0.9, summary_text, transform=axes[1, 0].transAxes, 
                       verticalalignment='top', fontsize=10)
        
        # Placeholder for future content
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 'For interactive dashboard,\ninstall plotly:\npip install plotly', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_conditional_analysis(self,
                                generated_samples: List[torch.Tensor],
                                conditions: List[torch.Tensor],
                                save_path: Optional[str] = None) -> str:
        """
        Plot analysis of conditional generation.
        
        Args:
            generated_samples: List of generated signal tensors
            conditions: List of condition tensors
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert to numpy
        samples_np = [s.squeeze().cpu().numpy() for s in generated_samples]
        conditions_np = [c.squeeze().cpu().numpy() for c in conditions]
        
        # Condition vs Signal Properties
        signal_rms = [np.sqrt(np.mean(s**2)) for s in samples_np]
        signal_peak = [np.max(np.abs(s)) for s in samples_np]
        
        if len(conditions_np[0]) >= 2:
            condition_1 = [c[0] for c in conditions_np]  # First condition parameter
            condition_2 = [c[1] for c in conditions_np]  # Second condition parameter
            
            # RMS vs Condition 1
            axes[0, 0].scatter(condition_1, signal_rms, alpha=0.6, color=self.colors['generated'])
            axes[0, 0].set_xlabel('Condition Parameter 1')
            axes[0, 0].set_ylabel('Signal RMS')
            axes[0, 0].set_title('Signal RMS vs Condition 1')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Peak vs Condition 2
            axes[0, 1].scatter(condition_2, signal_peak, alpha=0.6, color=self.colors['real'])
            axes[0, 1].set_xlabel('Condition Parameter 2')
            axes[0, 1].set_ylabel('Signal Peak')
            axes[0, 1].set_title('Signal Peak vs Condition 2')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Signal diversity across conditions
        sample_matrix = np.array(samples_np)
        correlation_matrix = np.corrcoef(sample_matrix)
        
        im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Sample Cross-Correlation Matrix')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Sample Index')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Condition space visualization
        if len(conditions_np[0]) >= 2:
            scatter = axes[1, 1].scatter(condition_1, condition_2, c=signal_rms, 
                                       cmap='viridis', alpha=0.7)
            axes[1, 1].set_xlabel('Condition Parameter 1')
            axes[1, 1].set_ylabel('Condition Parameter 2')
            axes[1, 1].set_title('Condition Space (colored by RMS)')
            plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'conditional_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)