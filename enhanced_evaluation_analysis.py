#!/usr/bin/env python3
"""
Enhanced ABR Model Evaluation Analysis
====================================

This script identifies and fixes the missing components in the evaluation pipeline
and creates comprehensive visualizations that were missing.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_evaluation_gaps():
    """Analyze what's missing from the current evaluation."""
    
    print("🔍 EVALUATION PIPELINE GAP ANALYSIS")
    print("=" * 50)
    
    # Check generated files
    results_path = Path("evaluation_results_latest_model")
    
    # Load report
    with open(results_path / "reports" / "evaluation_report.json", 'r') as f:
        report = json.load(f)
    
    gaps = []
    
    # 1. Check confusion matrix visualization
    if 'confusion_matrix' in report['classification']:
        cm = np.array(report['classification']['confusion_matrix']['matrix'])
        if cm.shape[0] == 5 and cm.shape[1] == 5:
            gaps.append("✅ Confusion matrix data exists but visualization missing")
        else:
            gaps.append("❌ Confusion matrix has wrong dimensions")
    else:
        gaps.append("❌ No confusion matrix data")
    
    # 2. Check peak detection results
    if 'peak_detection' in report:
        peak_data = report['peak_detection']
        if 'existence_metrics' in peak_data:
            gaps.append("✅ Peak existence data exists but visualization missing") 
        if 'latency_metrics' in peak_data:
            if np.isnan(peak_data['latency_metrics']['correlation']):
                gaps.append("❌ Peak latency correlation is NaN - evaluation broken")
            else:
                gaps.append("✅ Peak latency data exists but visualization missing")
        if 'amplitude_metrics' in peak_data:
            if np.isnan(peak_data['amplitude_metrics']['correlation']):
                gaps.append("❌ Peak amplitude correlation is NaN - evaluation broken")
            else:
                gaps.append("✅ Peak amplitude data exists but visualization missing")
    else:
        gaps.append("❌ No peak detection evaluation")
    
    # 3. Check signal quality metrics
    if 'signal_quality' in report:
        signal_data = report['signal_quality']['basic_metrics']
        if np.isnan(signal_data['correlation_mean']):
            gaps.append("❌ Signal correlation is NaN - evaluation broken")
        if np.isinf(signal_data['snr_mean_db']):
            gaps.append("❌ Signal SNR is infinite - evaluation broken")
    else:
        gaps.append("❌ No signal quality evaluation")
    
    # 4. Check for class-by-class analysis
    if 'per_class_metrics' in report['classification']:
        gaps.append("✅ Per-class data exists but class-by-class signal analysis missing")
    else:
        gaps.append("❌ No per-class analysis")
    
    # 5. Check summary metrics
    if report['summary_metrics']['overall_score'] == 0.0:
        gaps.append("❌ Summary metrics calculation broken")
    
    # 6. Check plot files
    plots_dir = results_path / "plots"
    expected_plots = [
        "classification_analysis.png",
        "peak_detection_analysis.png", 
        "threshold_regression_analysis.png",
        "signal_quality_analysis.png",
        "class_wise_signal_analysis.png"
    ]
    
    for plot in expected_plots:
        if (plots_dir / plot).exists():
            gaps.append(f"✅ {plot} exists")
        else:
            gaps.append(f"❌ {plot} missing")
    
    print("\n📊 IDENTIFIED GAPS:")
    print("-" * 30)
    for gap in gaps:
        print(gap)
    
    return gaps

def identify_broken_evaluations():
    """Identify specific evaluation methods that are broken."""
    
    print("\n🔧 BROKEN EVALUATION COMPONENTS:")
    print("-" * 40)
    
    broken_components = []
    
    # Load the evaluation report
    with open("evaluation_results_latest_model/reports/evaluation_report.json", 'r') as f:
        report = json.load(f)
    
    # Check for NaN/Inf values indicating broken evaluations
    def check_for_nans(data, path=""):
        issues = []
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                issues.extend(check_for_nans(value, current_path))
        elif isinstance(data, (int, float)):
            if np.isnan(data):
                issues.append(f"❌ NaN value at {path}")
            elif np.isinf(data):
                issues.append(f"❌ Infinite value at {path}")
        return issues
    
    nan_issues = check_for_nans(report)
    
    # Specific checks
    if np.isnan(report['signal_quality']['basic_metrics']['correlation_mean']):
        broken_components.append({
            'component': 'Signal Correlation Calculation',
            'issue': 'Returns NaN - likely due to invalid signal comparisons',
            'location': 'evaluation/comprehensive_evaluator.py _evaluate_signal_quality',
            'fix': 'Check for NaN/Inf in signals before correlation calculation'
        })
    
    if np.isinf(report['signal_quality']['basic_metrics']['snr_mean_db']):
        broken_components.append({
            'component': 'Signal SNR Calculation', 
            'issue': 'Returns -Infinity - likely division by zero in noise calculation',
            'location': 'evaluation/comprehensive_evaluator.py _evaluate_signal_quality',
            'fix': 'Add epsilon to denominator in SNR calculation'
        })
    
    if np.isnan(report['peak_detection']['latency_metrics']['correlation']):
        broken_components.append({
            'component': 'Peak Latency Correlation',
            'issue': 'Returns NaN - likely due to invalid peak data',
            'location': 'evaluation/comprehensive_evaluator.py _evaluate_peak_detection', 
            'fix': 'Check for valid peak data before correlation calculation'
        })
    
    if report['summary_metrics']['overall_score'] == 0.0:
        broken_components.append({
            'component': 'Summary Metrics Calculation',
            'issue': 'Overall score is 0.0 - calculation logic broken',
            'location': 'evaluate.py summary metrics calculation',
            'fix': 'Implement proper weighted average of all metrics'
        })
    
    # Check for empty sections
    if not report['clinical_analysis']['clinical_correlation']:
        broken_components.append({
            'component': 'Clinical Correlation Analysis',
            'issue': 'Empty results - evaluation not running',
            'location': 'evaluation/comprehensive_evaluator.py _evaluate_clinical_correlation',
            'fix': 'Check if clinical correlation methods are being called'
        })
    
    if not report['demographic_analysis']:
        broken_components.append({
            'component': 'Demographic Analysis',
            'issue': 'Empty results - evaluation not implemented or broken',
            'location': 'evaluate.py',
            'fix': 'Implement demographic breakdown analysis'
        })
    
    for component in broken_components:
        print(f"\n🔴 {component['component']}:")
        print(f"   Issue: {component['issue']}")
        print(f"   Location: {component['location']}")
        print(f"   Fix: {component['fix']}")
    
    return broken_components

def identify_missing_visualizations():
    """Identify missing visualization components."""
    
    print("\n📈 MISSING VISUALIZATIONS:")
    print("-" * 40)
    
    missing_viz = [
        {
            'name': 'Classification Confusion Matrix Heatmap',
            'description': 'Detailed confusion matrix with class names and percentages',
            'data_available': True,
            'fix': 'Create proper heatmap visualization in visualization_methods.py'
        },
        {
            'name': 'Peak Detection Analysis Plots',
            'description': 'Peak existence, latency, and amplitude distribution plots',
            'data_available': True,
            'fix': 'Implement _create_peak_detection_plots method'
        },
        {
            'name': 'Class-wise Signal Reconstruction Analysis',
            'description': 'Signal quality metrics broken down by hearing loss class',
            'data_available': False,
            'fix': 'Implement class-stratified signal evaluation'
        },
        {
            'name': 'Static Parameters vs Predictions Analysis',
            'description': 'How static params (age, gender, etc.) correlate with predictions',
            'data_available': False,
            'fix': 'Add static parameter analysis to evaluation pipeline'
        },
        {
            'name': 'ROC Curves for Each Class',
            'description': 'Individual ROC curves showing class discrimination',
            'data_available': True,
            'fix': 'Create ROC curve plots in visualization'
        },
        {
            'name': 'Detailed Error Analysis by Class',
            'description': 'Threshold and peak errors broken down by hearing loss severity',
            'data_available': False,
            'fix': 'Implement class-stratified error analysis'
        },
        {
            'name': 'Signal Spectral Analysis',
            'description': 'Frequency domain analysis of reconstructed vs true signals',
            'data_available': False,
            'fix': 'Add spectral analysis to signal quality evaluation'
        },
        {
            'name': 'Clinical Decision Support Visualization',
            'description': 'Diagnostic accuracy matrix and clinical correlation plots',
            'data_available': True,
            'fix': 'Create clinical analysis visualizations'
        }
    ]
    
    for viz in missing_viz:
        status = "✅ Data Available" if viz['data_available'] else "❌ Data Missing"
        print(f"\n📊 {viz['name']}: {status}")
        print(f"   Description: {viz['description']}")
        print(f"   Fix: {viz['fix']}")
    
    return missing_viz

def create_enhancement_plan():
    """Create a comprehensive plan to fix and enhance the evaluation pipeline."""
    
    print("\n🚀 EVALUATION ENHANCEMENT PLAN:")
    print("=" * 50)
    
    plan = {
        'Phase 1 - Critical Fixes': [
            'Fix NaN/Inf calculations in signal quality metrics',
            'Fix peak detection correlation calculations',
            'Fix summary metrics calculation logic',
            'Ensure all evaluation methods complete successfully'
        ],
        'Phase 2 - Missing Core Visualizations': [
            'Implement classification confusion matrix heatmap',
            'Create comprehensive peak detection analysis plots',
            'Add ROC curves for each class',
            'Implement threshold regression detailed plots'
        ],
        'Phase 3 - Advanced Analysis': [
            'Add class-wise signal reconstruction analysis',
            'Implement static parameters correlation analysis',
            'Create spectral analysis of signals',
            'Add demographic breakdown analysis'
        ],
        'Phase 4 - Clinical Enhancement': [
            'Create clinical decision support visualizations',
            'Add Bland-Altman plots for agreement analysis',
            'Implement error analysis by hearing loss severity',
            'Create interactive diagnostic dashboard'
        ]
    }
    
    for phase, tasks in plan.items():
        print(f"\n📋 {phase}:")
        for i, task in enumerate(tasks, 1):
            print(f"   {i}. {task}")
    
    return plan

def main():
    """Main analysis function."""
    gaps = analyze_evaluation_gaps()
    broken_components = identify_broken_evaluations()
    missing_viz = identify_missing_visualizations()
    plan = create_enhancement_plan()
    
    print(f"\n🎯 SUMMARY:")
    print("=" * 50)
    print(f"📊 Total gaps identified: {len(gaps)}")
    print(f"🔴 Broken components: {len(broken_components)}")
    print(f"📈 Missing visualizations: {len(missing_viz)}")
    print(f"\n💡 PRIORITY: Fix broken evaluation calculations first, then enhance visualizations")

if __name__ == "__main__":
    main()