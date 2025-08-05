#!/usr/bin/env python3
"""
Deep Analysis of ABR Model Evaluation Results
============================================

This script performs a comprehensive analysis of the evaluation results
to identify critical issues and provide actionable insights.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_evaluation_data(results_dir="evaluation_results_latest_model"):
    """Load all evaluation data from the results directory."""
    results_path = Path(results_dir)
    
    # Load JSON report
    with open(results_path / "reports" / "evaluation_report.json", 'r') as f:
        report = json.load(f)
    
    # Load predictions CSV
    predictions = pd.read_csv(results_path / "data" / "predictions.csv")
    
    return report, predictions

def analyze_critical_issues(report, predictions):
    """Identify and analyze critical issues in the model performance."""
    issues = []
    severity_scores = {}
    
    # 1. CRITICAL: Classification Catastrophic Failure
    cm = np.array(report['classification']['confusion_matrix']['matrix'])
    if np.sum(cm[1:, 1:]) == 0:  # No predictions for any minority class
        issues.append({
            'category': 'Classification',
            'severity': 'CRITICAL',
            'issue': 'Model only predicts class 0 (majority class)',
            'description': 'Complete failure to learn minority classes - model is essentially a majority class classifier',
            'impact': 'Clinically useless for detecting hearing loss',
            'evidence': f"Confusion matrix shows 0 predictions for classes 1-4, all {cm.sum()} samples predicted as class 0"
        })
        severity_scores['classification'] = 10
    
    # 2. CRITICAL: Signal Reconstruction Issues
    signal_metrics = report['signal_quality']['basic_metrics']
    if np.isnan(signal_metrics['correlation_mean']) or np.isinf(signal_metrics['snr_mean_db']):
        issues.append({
            'category': 'Signal Quality',
            'severity': 'CRITICAL', 
            'issue': 'Invalid signal correlation and SNR metrics',
            'description': 'NaN correlation and -Infinity SNR indicate fundamental signal reconstruction problems',
            'impact': 'Signal reconstruction is completely broken',
            'evidence': f"Correlation: {signal_metrics['correlation_mean']}, SNR: {signal_metrics['snr_mean_db']} dB"
        })
        severity_scores['signal_quality'] = 10
    
    # 3. HIGH: Peak Detection Correlation Issues
    peak_metrics = report['peak_detection']
    if np.isnan(peak_metrics['latency_metrics']['correlation']):
        issues.append({
            'category': 'Peak Detection',
            'severity': 'HIGH',
            'issue': 'Invalid correlation in peak latency predictions',
            'description': 'NaN correlation suggests no meaningful relationship between predicted and true peak latencies',
            'impact': 'Peak timing predictions are unreliable',
            'evidence': f"Latency correlation: {peak_metrics['latency_metrics']['correlation']}"
        })
        severity_scores['peak_detection'] = 8
    
    # 4. HIGH: Threshold Regression Poor Performance
    threshold_metrics = report['threshold_regression']['regression_metrics']
    if threshold_metrics['r2_score'] < 0:
        issues.append({
            'category': 'Threshold Regression',
            'severity': 'HIGH',
            'issue': 'Negative R² score indicates worse than baseline',
            'description': 'Model predictions are worse than simply predicting the mean threshold',
            'impact': 'Threshold predictions are clinically unreliable',
            'evidence': f"R² = {threshold_metrics['r2_score']:.4f}, MAE = {threshold_metrics['mae_db']:.2f} dB"
        })
        severity_scores['threshold_regression'] = 8
    
    # 5. MEDIUM: Poor Threshold Clinical Accuracy  
    clinical_metrics = report['threshold_regression']['clinical_metrics']
    if clinical_metrics['category_accuracy'] < 0.7:
        issues.append({
            'category': 'Clinical Performance',
            'severity': 'MEDIUM',
            'issue': 'Poor clinical category classification',
            'description': 'Low accuracy in classifying hearing loss categories',
            'impact': 'Limited clinical utility for diagnosis',
            'evidence': f"Clinical accuracy: {clinical_metrics['category_accuracy']:.3f}"
        })
        severity_scores['clinical'] = 6
    
    return issues, severity_scores

def analyze_threshold_predictions(predictions):
    """Analyze threshold prediction patterns."""
    analysis = {}
    
    # Basic statistics
    pred_thresh = predictions['predicted_threshold']
    true_thresh = predictions['true_threshold']
    
    analysis['prediction_statistics'] = {
        'predicted_mean': pred_thresh.mean(),
        'predicted_std': pred_thresh.std(),
        'predicted_min': pred_thresh.min(),
        'predicted_max': pred_thresh.max(),
        'predicted_range': pred_thresh.max() - pred_thresh.min(),
        'true_mean': true_thresh.mean(),
        'true_std': true_thresh.std(),
        'true_min': true_thresh.min(),
        'true_max': true_thresh.max(),
        'true_range': true_thresh.max() - true_thresh.min()
    }
    
    # Check for prediction collapse
    unique_predictions = pred_thresh.nunique()
    if unique_predictions < 10:
        analysis['prediction_collapse'] = {
            'severity': 'CRITICAL',
            'unique_values': unique_predictions,
            'most_common_prediction': pred_thresh.mode().iloc[0],
            'most_common_frequency': (pred_thresh == pred_thresh.mode().iloc[0]).mean()
        }
    
    # Error analysis
    errors = pred_thresh - true_thresh
    analysis['error_analysis'] = {
        'mean_error': errors.mean(),
        'abs_mean_error': errors.abs().mean(),
        'rmse': np.sqrt((errors**2).mean()),
        'error_std': errors.std(),
        'bias_direction': 'underestimating' if errors.mean() < 0 else 'overestimating'
    }
    
    return analysis

def analyze_classification_predictions(predictions):
    """Analyze classification prediction patterns."""
    analysis = {}
    
    pred_class = predictions['predicted_class']
    true_class = predictions['true_class']
    
    # Class distribution analysis
    analysis['prediction_distribution'] = pred_class.value_counts().to_dict()
    analysis['true_distribution'] = true_class.value_counts().to_dict()
    
    # Check for complete class collapse
    unique_predictions = pred_class.nunique()
    if unique_predictions == 1:
        analysis['class_collapse'] = {
            'severity': 'CRITICAL',
            'collapsed_to_class': pred_class.iloc[0],
            'total_samples': len(pred_class),
            'description': 'Model predicts only one class for all samples'
        }
    
    return analysis

def generate_insights_and_recommendations(issues, threshold_analysis, classification_analysis):
    """Generate actionable insights and recommendations."""
    insights = []
    recommendations = []
    
    # Classification issues
    if 'class_collapse' in classification_analysis:
        insights.append("CRITICAL: The model has completely collapsed to predicting only class 0. This indicates severe training issues.")
        recommendations.extend([
            "Re-examine loss function weights - current peak/threshold losses may be overwhelming classification loss",
            "Implement class balancing techniques (weighted loss, focal loss, or oversampling)",
            "Reduce learning rate and increase classification loss weight",
            "Check gradient flow to classification head during training",
            "Verify that classification targets are properly formatted and not corrupted"
        ])
    
    # Threshold prediction issues
    if 'prediction_collapse' in threshold_analysis:
        collapse = threshold_analysis['prediction_collapse']
        insights.append(f"CRITICAL: Threshold predictions have collapsed to essentially constant values (~{collapse['most_common_prediction']:.2f})")
        recommendations.extend([
            "The threshold regression head is not learning - check if gradients are flowing properly",
            "Reduce threshold loss weight relative to other losses",
            "Verify threshold target normalization and scaling",
            "Consider using different activation function for threshold head",
            "Check if threshold targets are properly preprocessed"
        ])
    
    # Signal quality issues  
    insights.append("CRITICAL: Signal reconstruction has fundamental issues with NaN correlations and infinite SNR values")
    recommendations.extend([
        "Debug signal preprocessing pipeline - check for NaN/Inf values in inputs",
        "Verify that signal targets are properly normalized",
        "Check diffusion noise schedule parameters",
        "Ensure proper handling of reconstruction loss computation",
        "Validate that generated signals are in valid range"
    ])
    
    # Training dynamics issues
    insights.append("SEVERE: Multiple task outputs suggest major training instability or improper loss balancing")
    recommendations.extend([
        "Dramatically reduce loss weights for peak and threshold tasks (try 0.01x current values)",
        "Implement gradient clipping more aggressively",
        "Use learning rate scheduling with warmup",
        "Consider training tasks sequentially rather than jointly",
        "Monitor individual loss components during training to identify divergence"
    ])
    
    return insights, recommendations

def create_diagnostic_plots(predictions, output_dir="evaluation_results_latest_model"):
    """Create diagnostic plots to visualize the issues."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ABR Model Diagnostic Analysis - Critical Issues Identified', fontsize=16, fontweight='bold')
    
    # 1. Threshold predictions scatter
    ax = axes[0, 0]
    ax.scatter(predictions['true_threshold'], predictions['predicted_threshold'], alpha=0.6, s=10)
    ax.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
    ax.set_xlabel('True Threshold (dB)')
    ax.set_ylabel('Predicted Threshold (dB)')
    ax.set_title('Threshold Predictions\n(Shows prediction collapse)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Threshold prediction distribution
    ax = axes[0, 1]
    predictions['predicted_threshold'].hist(bins=50, alpha=0.7, ax=ax, color='blue', label='Predicted')
    predictions['true_threshold'].hist(bins=50, alpha=0.7, ax=ax, color='red', label='True')
    ax.set_xlabel('Threshold (dB)')
    ax.set_ylabel('Frequency')
    ax.set_title('Threshold Distribution\n(Shows lack of variability)')
    ax.legend()
    
    # 3. Threshold error distribution
    ax = axes[0, 2]
    errors = predictions['predicted_threshold'] - predictions['true_threshold']
    ax.hist(errors, bins=50, alpha=0.7, color='orange')
    ax.axvline(errors.mean(), color='red', linestyle='--', label=f'Mean: {errors.mean():.2f}')
    ax.set_xlabel('Prediction Error (dB)')
    ax.set_ylabel('Frequency')
    ax.set_title('Threshold Error Distribution\n(Shows systematic bias)')
    ax.legend()
    
    # 4. Classification confusion matrix
    ax = axes[1, 0]
    true_class = predictions['true_class']
    pred_class = predictions['predicted_class']
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_class, pred_class)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Classification Confusion Matrix\n(Shows complete class collapse)')
    
    # 5. Class distribution comparison
    ax = axes[1, 1]
    true_dist = true_class.value_counts().sort_index()
    pred_dist = pred_class.value_counts().reindex(true_dist.index, fill_value=0)
    
    x = np.arange(len(true_dist))
    width = 0.35
    ax.bar(x - width/2, true_dist.values, width, label='True', alpha=0.7)
    ax.bar(x + width/2, pred_dist.values, width, label='Predicted', alpha=0.7)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution Comparison\n(Shows prediction bias)')
    ax.set_xticks(x)
    ax.set_xticklabels(true_dist.index)
    ax.legend()
    
    # 6. Prediction variance analysis
    ax = axes[1, 2]
    # Group by true class and show prediction variance
    threshold_by_class = []
    for class_id in sorted(true_class.unique()):
        class_predictions = predictions[predictions['true_class'] == class_id]['predicted_threshold']
        threshold_by_class.append(class_predictions.values)
    
    ax.boxplot(threshold_by_class, labels=sorted(true_class.unique()))
    ax.set_xlabel('True Class')
    ax.set_ylabel('Predicted Threshold (dB)')
    ax.set_title('Threshold Predictions by Class\n(Shows lack of class discrimination)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/diagnostic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function."""
    print("🔍 ABR Model Deep Evaluation Analysis")
    print("=" * 50)
    
    # Load data
    report, predictions = load_evaluation_data()
    
    # Analyze critical issues
    issues, severity_scores = analyze_critical_issues(report, predictions)
    
    # Analyze prediction patterns
    threshold_analysis = analyze_threshold_predictions(predictions)
    classification_analysis = analyze_classification_predictions(predictions)
    
    # Generate insights and recommendations
    insights, recommendations = generate_insights_and_recommendations(issues, threshold_analysis, classification_analysis)
    
    # Create diagnostic plots
    create_diagnostic_plots(predictions)
    
    # Print comprehensive analysis
    print("\n🚨 CRITICAL ISSUES IDENTIFIED:")
    print("-" * 40)
    for issue in issues:
        print(f"\n[{issue['severity']}] {issue['category']}: {issue['issue']}")
        print(f"Description: {issue['description']}")
        print(f"Impact: {issue['impact']}")
        print(f"Evidence: {issue['evidence']}")
    
    print(f"\n📊 THRESHOLD PREDICTION ANALYSIS:")
    print("-" * 40)
    stats = threshold_analysis['prediction_statistics']
    print(f"Predicted range: {stats['predicted_min']:.2f} - {stats['predicted_max']:.2f} dB (span: {stats['predicted_range']:.2f} dB)")
    print(f"True range: {stats['true_min']:.2f} - {stats['true_max']:.2f} dB (span: {stats['true_range']:.2f} dB)")
    print(f"Mean prediction error: {threshold_analysis['error_analysis']['mean_error']:.2f} dB")
    print(f"Bias direction: {threshold_analysis['error_analysis']['bias_direction']}")
    
    if 'prediction_collapse' in threshold_analysis:
        collapse = threshold_analysis['prediction_collapse']
        print(f"\n⚠️ PREDICTION COLLAPSE DETECTED:")
        print(f"Only {collapse['unique_values']} unique threshold values predicted")
        print(f"Most common prediction: {collapse['most_common_prediction']:.2f} dB ({collapse['most_common_frequency']:.1%} of samples)")
    
    print(f"\n🎯 CLASSIFICATION ANALYSIS:")
    print("-" * 40)
    if 'class_collapse' in classification_analysis:
        collapse = classification_analysis['class_collapse']
        print(f"⚠️ COMPLETE CLASS COLLAPSE: All {collapse['total_samples']} samples predicted as class {collapse['collapsed_to_class']}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 40)
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print(f"\n🔧 RECOMMENDED ACTIONS:")
    print("-" * 40)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n📈 DIAGNOSTIC PLOTS SAVED:")
    print("-" * 40)
    print("📁 diagnostic_analysis.png - Visual analysis of prediction patterns")
    
    # Overall severity assessment
    max_severity = max(severity_scores.values()) if severity_scores else 0
    if max_severity >= 9:
        status = "🚨 CRITICAL - MODEL REQUIRES COMPLETE RETRAINING"
    elif max_severity >= 7:
        status = "⚠️ SEVERE - MAJOR ISSUES NEED IMMEDIATE ATTENTION"
    elif max_severity >= 5:
        status = "🔶 MODERATE - ISSUES REQUIRE INVESTIGATION"
    else:
        status = "✅ GOOD - MINOR ISSUES TO ADDRESS"
    
    print(f"\n🎯 OVERALL STATUS: {status}")
    print("=" * 50)

if __name__ == "__main__":
    main()