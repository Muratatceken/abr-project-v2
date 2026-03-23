# Threshold Optimization Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Understanding Threshold Optimization](#understanding-threshold-optimization)
4. [Optimization Strategies](#optimization-strategies)
5. [Configuration Guide](#configuration-guide)
6. [Usage Examples](#usage-examples)
7. [Interpreting Results](#interpreting-results)
8. [Integration with Evaluation Pipeline](#integration-with-evaluation-pipeline)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)
12. [API Reference](#api-reference)

## Overview

The Threshold Optimization System provides advanced tools for optimizing classification thresholds in ABR (Auditory Brainstem Response) peak detection. This system goes beyond simple fixed thresholds by offering multiple optimization strategies, constraint-based optimization, and comprehensive analysis tools.

### Key Features

- **Multiple Optimization Strategies**: F1-optimal, Youden's J statistic, constrained optimization
- **Constraint-Based Optimization**: Target specific performance requirements (e.g., recall >80%, precision >90%)
- **Multi-Objective Optimization**: Pareto-optimal solutions balancing multiple objectives
- **Comprehensive Analysis**: Threshold sensitivity analysis, stability assessment, and performance curves
- **Clinical Context**: Clinical validation metrics and interpretation guidance
- **Seamless Integration**: Works with existing evaluation pipeline

### Importance in ABR Classification

ABR peak detection is critical for hearing assessment and diagnosis. The choice of classification threshold significantly impacts:

- **Sensitivity (Recall)**: Ability to detect true ABR peaks
- **Specificity**: Ability to avoid false positive detections  
- **Precision**: Proportion of detected peaks that are genuine
- **Clinical Utility**: Balance between missing true peaks vs. false alarms

Fixed thresholds (e.g., 0.5) may not be optimal for all scenarios. This system helps find the best threshold for your specific requirements.

## Quick Start

### 1. Analyze Existing ROC Data

If you have existing ROC classification results:

```bash
# Basic analysis with default constraints (recall >80%, precision >90%)
python analyze_roc_optimization.py \
  --roc_data evaluation_results/reconstruction/reconstruction_peak_classification.json \
  --constraints "recall:0.8,precision:0.9"

# Advanced analysis with custom output directory
python analyze_roc_optimization.py \
  --roc_data evaluation_results/reconstruction/reconstruction_peak_classification.json \
  --constraints "recall:0.85,precision:0.95,specificity:0.9" \
  --output_dir my_threshold_analysis \
  --log_level DEBUG
```

### 2. Integrate with Evaluation Pipeline

Enable threshold optimization in your evaluation by ensuring `configs/threshold_optimization.yaml` is configured:

```bash
# Run evaluation with threshold optimization enabled
python eval.py --config configs/eval.yaml
```

The system will automatically apply threshold optimization if enabled in the configuration.

### 3. Generate Comprehensive Reports

```bash
# Generate detailed reports from analysis results
python threshold_optimization_report.py \
  --input threshold_optimization_analysis/ \
  --output_dir reports/ \
  --formats "html,csv,plots,json"
```

## Understanding Threshold Optimization

### The Problem with Fixed Thresholds

Traditional binary classification uses a fixed threshold (often 0.5) to convert prediction scores to binary decisions. However:

- **Imbalanced Data**: When classes are imbalanced, 0.5 may not be optimal
- **Cost-Sensitive Decisions**: Different types of errors may have different costs
- **Performance Requirements**: Clinical applications may require minimum sensitivity or specificity
- **Domain Knowledge**: Medical applications often have specific performance constraints

### ROC Curves and Performance Metrics

The threshold optimization system analyzes the complete ROC (Receiver Operating Characteristic) curve to find optimal operating points:

- **ROC Curve**: Shows trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1-Specificity)
- **Precision-Recall Curve**: Shows trade-off between Precision and Recall
- **Threshold Sweep**: Analyzes performance across all possible thresholds

### Key Performance Metrics

- **Sensitivity (Recall)**: TP / (TP + FN) - Ability to detect true positives
- **Specificity**: TN / (TN + FP) - Ability to avoid false positives  
- **Precision (PPV)**: TP / (TP + FP) - Proportion of positive predictions that are correct
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall) - Harmonic mean
- **Youden's J**: Sensitivity + Specificity - 1 - Balance between sensitivity and specificity

## Optimization Strategies

### 1. F1-Optimal Threshold

Maximizes the F1-score, which balances precision and recall.

**When to Use**:
- Balanced importance of precision and recall
- General-purpose threshold optimization
- No specific performance constraints

**Mathematical Formula**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Optimal Threshold = argmax(F1(threshold))
```

### 2. Youden's J Statistic

Maximizes the sum of sensitivity and specificity minus 1.

**When to Use**:
- Equal importance of sensitivity and specificity
- Balanced classification problems
- Medical screening applications

**Mathematical Formula**:
```
J = Sensitivity + Specificity - 1
Optimal Threshold = argmax(J(threshold))
```

### 3. Constrained Optimization

Finds the best threshold that satisfies user-defined constraints.

**When to Use**:
- Specific performance requirements (e.g., minimum 80% recall)
- Regulatory or clinical guidelines
- Risk-sensitive applications

**Example Constraints**:
- Minimum Recall ≥ 0.80 (catch at least 80% of true peaks)
- Minimum Precision ≥ 0.90 (90% of detections should be genuine)
- Minimum Specificity ≥ 0.95 (avoid false positives)

### 4. Multi-Objective Optimization

Finds Pareto-optimal solutions that balance multiple objectives.

**When to Use**:
- Multiple competing objectives
- Need to explore trade-off space
- Uncertainty about relative importance of metrics

## Configuration Guide

### Main Configuration File: `configs/threshold_optimization.yaml`

#### Basic Configuration

```yaml
# Enable/disable optimization strategies
optimization:
  strategies:
    f1_optimal: true
    youden_j: true
    constrained: true
    multi_objective: true
  default_strategy: 'f1_optimal'

# Set performance constraints
constraints:
  min_recall: 0.80        # Require ≥80% recall
  min_precision: 0.90     # Require ≥90% precision
  min_specificity: null   # No specificity constraint
```

#### Advanced Configuration

```yaml
# Analysis parameters
analysis:
  threshold_search:
    resolution: 200       # Number of threshold points
    range: [0.0, 1.0]    # Threshold range
  
  performance_analysis:
    stability_window_size: 20    # For stability analysis
    sensitivity_analysis: true   # Enable sensitivity analysis

# Output settings
output:
  generate_reports: true
  create_plots: true
  include_results:
    threshold_analysis: true
    constraint_validation: true
    pareto_frontier: true

# Integration settings
integration:
  enable_in_evaluation: true
  save_optimization_results: true
```

### Configuration Options Reference

#### Optimization Strategies
- `f1_optimal`: Enable F1-score optimization
- `youden_j`: Enable Youden's J statistic optimization
- `constrained`: Enable constraint-based optimization
- `multi_objective`: Enable multi-objective Pareto optimization

#### Constraints
- `min_recall`: Minimum required recall/sensitivity (0.0-1.0)
- `min_precision`: Minimum required precision (0.0-1.0)
- `min_specificity`: Minimum required specificity (0.0-1.0)

#### Analysis Parameters
- `resolution`: Number of threshold points to analyze (default: 200)
- `range`: [min, max] threshold range (default: auto-detect)
- `confidence.level`: Confidence level for bootstrap intervals (default: 0.95)
- `bootstrap_samples`: Number of bootstrap samples (default: 1000)

## Usage Examples

### Example 1: Basic ROC Analysis

```python
from evaluation.metrics import ThresholdOptimizer

# Load ROC data
optimizer = ThresholdOptimizer()
roc_data = optimizer.analyze_roc_data('path/to/roc_data.json')

# Find F1-optimal threshold
f1_result = optimizer.find_optimal_threshold_f1()
print(f"F1-optimal threshold: {f1_result['optimal_threshold']:.4f}")
print(f"F1-score: {f1_result['performance']['f1_score']:.4f}")
```

### Example 2: Constrained Optimization

```python
# Find threshold satisfying specific constraints
constrained_result = optimizer.find_optimal_threshold_constrained(
    min_recall=0.80,      # At least 80% recall
    min_precision=0.90    # At least 90% precision
)

if constrained_result['success']:
    print(f"✓ Constraints satisfied!")
    print(f"Optimal threshold: {constrained_result['optimal_threshold']:.4f}")
    perf = constrained_result['performance']
    print(f"Recall: {perf['sensitivity']:.4f}")
    print(f"Precision: {perf['precision']:.4f}")
else:
    print("✗ Constraints cannot be satisfied")
    print(f"Reason: {constrained_result['message']}")
    for rec in constrained_result['recommendations']:
        print(f"Recommendation: {rec}")
```

### Example 3: Comprehensive Analysis

```python
# Threshold performance analysis
performance_analysis = optimizer.threshold_performance_analysis(
    threshold_range=(0.1, 0.9),
    num_points=100
)

# Plot results
import matplotlib.pyplot as plt
import numpy as np

thresholds = np.array(performance_analysis['thresholds'])
f1_scores = np.array(performance_analysis['f1_score'])
precision = np.array(performance_analysis['precision'])
recall = np.array(performance_analysis['sensitivity'])

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(thresholds, f1_scores)
plt.title('F1-Score vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('F1-Score')

plt.subplot(2, 2, 2)
plt.plot(thresholds, precision, label='Precision')
plt.plot(thresholds, recall, label='Recall')
plt.title('Precision/Recall vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()
```

### Example 4: Integration with Evaluation Pipeline

```python
from evaluation.analysis import roc_analysis, precision_recall_analysis

# Enhanced ROC analysis with constraints
constraint_params = {
    'min_recall': 0.80,
    'min_precision': 0.90
}

roc_results = roc_analysis(
    logits, targets,
    enable_constrained_optimization=True,
    constraint_params=constraint_params
)

# Check if constraints were satisfied
if 'constrained_optimization' in roc_results:
    const_opt = roc_results['constrained_optimization']
    if const_opt.get('success'):
        print(f"✓ Found optimal threshold: {const_opt['optimal_threshold']:.4f}")
    else:
        print("✗ Constraints cannot be satisfied")
```

## Interpreting Results

### Understanding Optimization Output

#### F1-Optimal Results
```json
{
  "optimal_threshold": 0.3456,
  "performance": {
    "f1_score": 0.8123,
    "sensitivity": 0.7891,
    "specificity": 0.9234,
    "precision": 0.8567
  },
  "optimization_method": "f1_optimal"
}
```

**Interpretation**:
- **Threshold 0.3456**: Use this value to make binary predictions
- **F1-Score 0.8123**: Balanced precision-recall performance
- **Sensitivity 0.7891**: Detects 78.9% of true ABR peaks
- **Specificity 0.9234**: Correctly rejects 92.3% of non-peaks
- **Precision 0.8567**: 85.7% of detections are genuine peaks

#### Constrained Optimization Results

**Success Case**:
```json
{
  "success": true,
  "optimal_threshold": 0.4123,
  "performance": {
    "sensitivity": 0.8045,
    "precision": 0.9012,
    "specificity": 0.8956,
    "f1_score": 0.8501
  },
  "constraints_satisfied": {
    "min_recall": true,
    "min_precision": true
  },
  "num_feasible_thresholds": 47
}
```

**Failure Case**:
```json
{
  "success": false,
  "message": "No thresholds satisfy the specified constraints",
  "recommendations": [
    "Consider recall ≥0.750 (80th percentile)",
    "Consider relaxing precision constraint by 5-10%"
  ]
}
```

### Clinical Interpretation Guidelines

#### High Sensitivity (Recall) Thresholds
- **Advantages**: Catches more true ABR peaks, fewer missed diagnoses
- **Disadvantages**: More false positives, increased follow-up testing
- **Use Case**: Screening applications, high-risk populations

#### High Specificity Thresholds  
- **Advantages**: Fewer false positives, higher diagnostic confidence
- **Disadvantages**: May miss some true peaks, potential missed diagnoses
- **Use Case**: Confirmatory testing, resource-constrained settings

#### High Precision Thresholds
- **Advantages**: Most detections are genuine, efficient use of resources
- **Disadvantages**: May miss some true peaks
- **Use Case**: Automated analysis, high-throughput screening

### Performance Benchmarks

#### Excellent Performance
- AUROC > 0.90
- F1-Score > 0.85
- Balanced sensitivity and specificity > 0.85

#### Good Performance  
- AUROC > 0.80
- F1-Score > 0.75
- Sensitivity and specificity > 0.75

#### Acceptable Performance
- AUROC > 0.70
- F1-Score > 0.65
- Sensitivity or specificity > 0.70

## Integration with Evaluation Pipeline

### Automatic Integration

The threshold optimization system automatically integrates with the evaluation pipeline when enabled in the configuration. 

**File**: `configs/threshold_optimization.yaml`
```yaml
integration:
  enable_in_evaluation: true
  save_optimization_results: true
```

### Manual Integration

You can manually integrate threshold optimization into custom evaluation workflows:

```python
# In your evaluation script
from eval import load_threshold_optimization_config
from evaluation.analysis import constrained_threshold_optimization

# Load configuration
threshold_config = load_threshold_optimization_config()

if threshold_config.get('integration', {}).get('enable_in_evaluation', True):
    # Apply threshold optimization
    constraint_params = threshold_config.get('constraints', {})
    optimization_results = constrained_threshold_optimization(
        logits, targets, constraint_params
    )
    
    # Use optimized threshold
    if optimization_results.get('success'):
        optimal_threshold = optimization_results['optimal_threshold']
        predictions = (logits >= optimal_threshold).astype(int)
```

### Output Files

When integrated with the evaluation pipeline, the system generates:

1. **Enhanced ROC Results**: `*_peak_classification.json`
   - Standard ROC analysis
   - Constrained optimization results
   - Multi-objective analysis (if enabled)

2. **Optimization Summary**: `*_threshold_optimization_summary.txt`
   - Executive summary of optimization results
   - Constraint satisfaction analysis
   - Performance recommendations

3. **CSV Metrics**: `*_peak_classification_metrics.csv`
   - Tabular performance metrics
   - Optimal thresholds for different strategies
   - Confidence intervals

## Advanced Features

### Multi-Objective Optimization

Find Pareto-optimal solutions that balance multiple objectives:

```python
from evaluation.analysis import multi_objective_threshold_optimization

# Find Pareto-optimal thresholds
pareto_results = multi_objective_threshold_optimization(
    logits, targets,
    objectives=['precision', 'recall', 'specificity']
)

# Explore trade-off solutions
for i, solution in enumerate(pareto_results['pareto_optimal_solutions'][:5]):
    print(f"Solution {i+1}:")
    print(f"  Threshold: {solution['threshold']:.4f}")
    print(f"  Precision: {solution['precision']:.4f}")
    print(f"  Recall: {solution['recall']:.4f}")
    print(f"  Specificity: {solution['specificity']:.4f}")
```

### Threshold Sensitivity Analysis

Analyze threshold stability and robustness:

```python
from evaluation.analysis import threshold_sensitivity_analysis

# Analyze threshold sensitivity
sensitivity_results = threshold_sensitivity_analysis(
    logits, targets,
    threshold_range=(0.2, 0.8),
    num_points=200
)

# Check stability analysis
if 'stability_analysis' in sensitivity_results:
    stability = sensitivity_results['stability_analysis']
    print(f"Most stable threshold range: {stability['most_stable_range']}")
    print(f"Stability score: {stability['stability_score']:.4f}")
```

### Custom Objective Functions

Define custom optimization objectives:

```python
def custom_objective(tpr, fpr, precision, recall):
    """Custom objective balancing sensitivity and positive predictive value."""
    # Weight sensitivity higher than PPV
    return 0.7 * tpr + 0.3 * precision

# Apply custom optimization (requires extending ThresholdOptimizer class)
```

### Bootstrap Confidence Intervals

All optimization results include bootstrap confidence intervals:

```python
# Confidence intervals are automatically calculated
print(f"AUROC: {roc_results['auroc']:.4f}")
print(f"95% CI: [{roc_results['auroc_ci']['lower']:.4f}, "
      f"{roc_results['auroc_ci']['upper']:.4f}]")
```

## Troubleshooting

### Common Issues

#### 1. "Constraints cannot be satisfied"

**Cause**: The requested performance constraints are too stringent for the current model/data.

**Solutions**:
- Relax constraints (e.g., reduce minimum recall from 0.90 to 0.85)
- Improve model performance through training or architecture changes
- Check constraint validation suggestions in the output

**Example**:
```bash
# Check what constraints are feasible
python analyze_roc_optimization.py \
  --roc_data your_data.json \
  --constraints "recall:0.70,precision:0.85"  # More relaxed
```

#### 2. "No ROC data found"

**Cause**: Input file doesn't contain expected ROC curve data structure.

**Solutions**:
- Ensure input JSON has `roc_analysis` and `precision_recall_analysis` sections
- Verify file is from peak classification evaluation
- Check file path is correct

**Expected Structure**:
```json
{
  "roc_analysis": {
    "roc_curve": {
      "fpr": [...],
      "tpr": [...],
      "thresholds": [...]
    },
    "auroc": 0.85
  },
  "precision_recall_analysis": {
    "pr_curve": {
      "precision": [...],
      "recall": [...],
      "thresholds": [...]
    }
  }
}
```

#### 3. "Threshold optimization disabled"

**Cause**: Threshold optimization is disabled in configuration.

**Solution**:
```yaml
# In configs/threshold_optimization.yaml
integration:
  enable_in_evaluation: true  # Make sure this is true
```

#### 4. "Memory error during bootstrap"

**Cause**: Too many bootstrap samples for large datasets.

**Solution**:
```yaml
# In configs/threshold_optimization.yaml
analysis:
  confidence:
    bootstrap_samples: 500  # Reduce from default 1000

performance:
  limits:
    memory_limit_mb: 1024  # Increase memory limit
```

#### 5. "Interpolation failed"

**Cause**: Issues with precision-recall threshold mapping.

**Solutions**:
- Check for NaN values in ROC data
- Ensure sufficient threshold points
- Try different interpolation method:

```yaml
analysis:
  threshold_search:
    interpolation_method: 'cubic'  # Try 'cubic' instead of 'linear'
```

### Debugging Tips

#### Enable Debug Logging
```bash
python analyze_roc_optimization.py \
  --roc_data your_data.json \
  --log_level DEBUG
```

#### Validate Input Data
```python
import json
import numpy as np

# Check ROC data structure
with open('your_roc_data.json', 'r') as f:
    data = json.load(f)

# Validate arrays
fpr = np.array(data['roc_analysis']['roc_curve']['fpr'])
tpr = np.array(data['roc_analysis']['roc_curve']['tpr'])
thresholds = np.array(data['roc_analysis']['roc_curve']['thresholds'])

print(f"FPR shape: {fpr.shape}")
print(f"TPR shape: {tpr.shape}")
print(f"Thresholds shape: {thresholds.shape}")
print(f"Any NaN values: {np.any(np.isnan(fpr)) or np.any(np.isnan(tpr))}")
```

#### Check Configuration Loading
```python
from eval import load_threshold_optimization_config

config = load_threshold_optimization_config()
print(json.dumps(config, indent=2))
```

## Best Practices

### 1. Validation Strategy

**Independent Test Set**:
- Always validate optimal thresholds on independent test data
- Never use the same data for optimization and validation
- Consider temporal validation for longitudinal studies

**Cross-Validation**:
```python
from sklearn.model_selection import StratifiedKFold

# Cross-validate threshold optimization
cv_thresholds = []
for train_idx, val_idx in StratifiedKFold(n_splits=5).split(data, labels):
    # Optimize on train split
    # Validate on val split
    pass
```

### 2. Clinical Context

**Domain Expertise**:
- Involve domain experts in constraint specification
- Consider clinical workflow and resource implications
- Validate results against clinical expectations

**Cost-Benefit Analysis**:
- Quantify costs of false positives vs false negatives
- Consider downstream testing and treatment costs
- Evaluate patient outcome implications

### 3. Monitoring and Maintenance

**Performance Monitoring**:
- Track threshold performance over time
- Monitor for dataset drift
- Set up alerts for performance degradation

**Regular Reoptimization**:
- Reoptimize thresholds as new data becomes available
- Consider seasonal or demographic variations
- Update constraints based on clinical feedback

### 4. Documentation and Reproducibility

**Document Decisions**:
```yaml
# Include rationale in configuration
constraints:
  min_recall: 0.80  # Clinical requirement: catch 80% of true positives
  min_precision: 0.90  # Resource constraint: limit false positive workup
  
# Document in commit messages, reports
analysis_rationale: "Optimized for emergency department use case"
```

**Version Control**:
- Version control configuration files
- Track optimization results over time
- Document threshold changes and rationales

### 5. Error Handling

**Robust Implementation**:
```python
def safe_threshold_optimization(logits, targets, constraints):
    try:
        result = constrained_threshold_optimization(logits, targets, constraints)
        if result['success']:
            return result['optimal_threshold']
        else:
            logging.warning(f"Optimization failed: {result['message']}")
            # Fallback to F1-optimal
            f1_result = find_optimal_threshold_f1(roc_data)
            return f1_result['optimal_threshold']
    except Exception as e:
        logging.error(f"Threshold optimization error: {e}")
        # Ultimate fallback
        return 0.5
```

## API Reference

### Core Classes

#### `ThresholdOptimizer`

Main class for threshold optimization operations.

```python
class ThresholdOptimizer:
    def __init__(self, roc_data: Optional[Dict[str, Any]] = None)
    def analyze_roc_data(self, roc_file_path: str) -> Dict[str, Any]
    def find_optimal_threshold_constrained(self, 
                                         min_recall: float = None,
                                         min_precision: float = None,
                                         min_specificity: float = None) -> Dict[str, Any]
    def find_optimal_threshold_f1(self) -> Dict[str, Any]
    def find_optimal_threshold_youden(self) -> Dict[str, Any]
    def threshold_performance_analysis(self, 
                                     threshold_range: Tuple[float, float] = None,
                                     num_points: int = 100) -> Dict[str, Any]
    def validate_threshold_constraints(self, constraints: Dict[str, float]) -> Dict[str, Any]
```

### Utility Functions

#### `analyze_roc_data(roc_file_path: str) -> Dict[str, Any]`

Load and analyze ROC classification data from JSON files.

**Parameters**:
- `roc_file_path`: Path to ROC classification JSON file

**Returns**: Dictionary containing analyzed ROC data

#### `find_optimal_threshold_constrained(roc_data, min_recall, min_precision, min_specificity) -> Dict[str, Any]`

Find optimal threshold with user-defined constraints.

**Parameters**:
- `roc_data`: ROC curve data dictionary  
- `min_recall`: Minimum required recall (0.0-1.0)
- `min_precision`: Minimum required precision (0.0-1.0)
- `min_specificity`: Minimum required specificity (0.0-1.0)

**Returns**: Dictionary with optimization results

#### `threshold_performance_analysis(roc_data, threshold_range, num_points) -> Dict[str, Any]`

Analyze threshold performance across operating points.

**Parameters**:
- `roc_data`: ROC curve data dictionary
- `threshold_range`: Range of thresholds to analyze
- `num_points`: Number of analysis points

**Returns**: Dictionary with performance analysis

### Analysis Functions

#### `constrained_threshold_optimization(logits, targets, constraint_params) -> Dict[str, Any]`

Implement optimization with user-defined constraints.

#### `threshold_sensitivity_analysis(logits, targets, threshold_range, num_points) -> Dict[str, Any]`

Analyze threshold sensitivity and robustness.

#### `multi_objective_threshold_optimization(logits, targets, objectives) -> Dict[str, Any]`

Implement Pareto-optimal threshold selection for multiple objectives.

### Configuration Functions

#### `load_threshold_optimization_config() -> Dict[str, Any]`

Load threshold optimization configuration from YAML file.

**Returns**: Configuration dictionary with defaults if file not found

---

## Support and Contributing

### Getting Help

1. **Documentation**: Check this guide and inline code documentation
2. **Issues**: Report bugs and feature requests via GitHub issues
3. **Examples**: See `examples/` directory for usage examples
4. **Community**: Join discussions in project forums

### Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines on:
- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

### License

This project is licensed under [LICENSE]. See LICENSE file for details.

---

*Last updated: 2024-01-15*
*Version: 1.0.0*
