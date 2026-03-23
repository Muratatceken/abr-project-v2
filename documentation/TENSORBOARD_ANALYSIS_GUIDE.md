# TensorBoard Analysis Guide

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Components Overview](#components-overview)
4. [Detailed Usage](#detailed-usage)
5. [Analysis Scripts](#analysis-scripts)
6. [Output Interpretation](#output-interpretation)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)
9. [Customization](#customization)
10. [API Reference](#api-reference)

## Overview

This guide provides comprehensive instructions for using the TensorBoard analysis tools to export, analyze, and generate insights from training dynamics. The system includes automated export capabilities, advanced analysis algorithms, and comprehensive reporting functionality.

### Key Features

- **Automated Export**: Export TensorBoard scalar metrics to CSV with category grouping
- **Advanced Analysis**: Convergence detection, instability analysis, multi-task learning insights
- **Loss Curve Analysis**: Specialized analysis with smoothing and statistical tests
- **Comprehensive Reporting**: HTML reports with interactive visualizations
- **Complete Pipeline**: End-to-end automation with error handling

### System Requirements

- Python 3.7+
- Required packages: pandas, numpy, matplotlib, seaborn, scipy, sklearn, jinja2
- TensorBoard log files with scalar metrics
- Minimum 1GB free disk space for exports and reports

## Quick Start

### 1. Complete Analysis Pipeline

To run the entire analysis pipeline with default settings:

```bash
# Make scripts executable
chmod +x run_tensorboard_export.sh
chmod +x run_analysis.sh

# Run complete pipeline
./run_analysis.sh
```

This will:

1. Export TensorBoard data to CSV files
2. Generate training progress visualizations
3. Perform advanced training dynamics analysis
4. Analyze loss curves with smoothing techniques
5. Generate comprehensive HTML reports

### 2. Individual Components

If you need to run individual components:

```bash
# Export TensorBoard data only
./run_tensorboard_export.sh

# Run specific analysis scripts
python example_tensorboard_analysis.py
python advanced_training_analysis.py
python loss_curve_analyzer.py
python training_dynamics_report.py
```

### 3. Expected Output

After running the complete pipeline, you'll find:

```
tensorboard_exports/           # CSV exports
├── tensorboard_train_metrics.csv
├── tensorboard_val_metrics.csv
├── tensorboard_curriculum_metrics.csv
└── tensorboard_summary_statistics.csv

training_analysis_results/     # Analysis results
├── advanced_training_analysis.json
├── loss_curve_analysis.json
├── convergence_analysis.png
├── instability_analysis.png
├── training_dynamics_report_YYYYMMDD_HHMMSS.html
└── comprehensive_training_report_YYYYMMDD_HHMMSS.json
```

## Components Overview

### Core Scripts

| Script                              | Purpose                     | Output                                |
| ----------------------------------- | --------------------------- | ------------------------------------- |
| `run_tensorboard_export.sh`       | Automate TensorBoard export | CSV files in `tensorboard_exports/` |
| `run_analysis.sh`                 | Complete analysis pipeline  | All analysis outputs                  |
| `export_tensorboard_to_csv.py`    | Export TensorBoard data     | Categorized CSV files                 |
| `example_tensorboard_analysis.py` | Basic visualizations        | Training progress plots               |
| `advanced_training_analysis.py`   | Training dynamics analysis  | JSON + convergence plots              |
| `loss_curve_analyzer.py`          | Specialized loss analysis   | JSON + loss curve plots               |
| `training_dynamics_report.py`     | Comprehensive reporting     | HTML reports + summaries              |

### Analysis Categories

1. **Convergence Analysis**: Detect when metrics stabilize and calculate convergence rates
2. **Instability Detection**: Identify oscillations, plateaus, and sudden jumps
3. **Multi-task Learning**: Analyze balance between different learning objectives
4. **Curriculum Learning**: Track progression through difficulty levels
5. **Overfitting Detection**: Compare training vs validation performance
6. **Statistical Testing**: Perform significance tests on improvements

## Detailed Usage

### TensorBoard Export

The export process extracts scalar metrics from TensorBoard event files and organizes them by category.

#### Basic Export

```bash
python export_tensorboard_to_csv.py
```

#### Advanced Export with Grouping

```bash
python export_tensorboard_to_csv.py --group_by_category --include_summary
```

#### Custom Log Directory

```bash
python export_tensorboard_to_csv.py --logdir custom/path/to/logs --output_dir custom_exports
```

#### Export Options

| Flag                    | Description                 | Default                                           |
| ----------------------- | --------------------------- | ------------------------------------------------- |
| `--logdir`            | TensorBoard log directory   | `runs/ablation/full_enhanced/abr_full_enhanced` |
| `--output_dir`        | CSV output directory        | `tensorboard_exports`                           |
| `--group_by_category` | Group metrics by category   | `False`                                         |
| `--include_summary`   | Generate summary statistics | `False`                                         |

### Advanced Training Analysis

Performs comprehensive analysis of training dynamics including convergence patterns and instability detection.

#### Basic Usage

```python
from advanced_training_analysis import AdvancedTrainingAnalyzer

analyzer = AdvancedTrainingAnalyzer()
analyzer.run_complete_analysis()
```

#### Custom Configuration

```python
analyzer = AdvancedTrainingAnalyzer(
    csv_dir="custom_exports",
    output_dir="custom_analysis"
)

# Load and analyze specific data
data_files = analyzer.load_data()
analyzer.analyze_convergence_patterns(data_files)
analyzer.detect_training_instabilities(data_files)
analyzer.export_analysis_results()
```

#### Analysis Components

1. **Convergence Detection**

   - Identifies convergence epochs using statistical criteria
   - Calculates convergence rates and improvement ratios
   - Detects optimal stopping points
2. **Instability Analysis**

   - Detects oscillatory behavior in loss curves
   - Identifies plateau regions with minimal progress
   - Finds sudden jumps or spikes in metrics
3. **Multi-task Dynamics**

   - Analyzes balance between different learning objectives
   - Calculates task correlation and conflict metrics
   - Provides recommendations for loss weighting

### Loss Curve Analysis

Specialized analysis focusing on loss curve characteristics and smoothing techniques.

#### Smoothing Techniques

The analyzer applies multiple smoothing methods:

- **Moving Average**: Simple and effective for trend identification
- **Exponential Smoothing**: Gives more weight to recent observations
- **Gaussian Smoothing**: Provides smooth curves with configurable sigma
- **Savitzky-Golay**: Preserves peak characteristics while smoothing

```python
from loss_curve_analyzer import LossCurveAnalyzer

analyzer = LossCurveAnalyzer()

# Load data and apply smoothing
loss_data = analyzer.load_loss_data()

# Analyze characteristics
analyzer.analyze_loss_characteristics(loss_data)
analyzer.analyze_overfitting_patterns(loss_data)
analyzer.calculate_derivatives(loss_data)
```

#### Overfitting Detection

The system detects overfitting by:

- Comparing training vs validation loss divergence
- Identifying optimal stopping points
- Calculating generalization gap metrics
- Recommending early stopping epochs

### Comprehensive Reporting

Generates detailed HTML reports with interactive visualizations and actionable recommendations.

#### Report Generation

```python
from training_dynamics_report import TrainingDynamicsReporter

reporter = TrainingDynamicsReporter()
reporter.run_complete_reporting()
```

#### Report Components

1. **Executive Summary**: High-level overview of training performance
2. **Key Findings**: Important insights and patterns discovered
3. **Visualizations**: Interactive plots and charts
4. **Recommendations**: Actionable optimization suggestions
5. **Statistical Analysis**: Significance tests and effect sizes

## Output Interpretation

### Convergence Metrics

| Metric                | Interpretation       | Good Range                   |
| --------------------- | -------------------- | ---------------------------- |
| `convergence_epoch` | When loss stabilized | < 80% of total epochs        |
| `improvement_ratio` | Total loss reduction | > 0.5 for good learning      |
| `convergence_rate`  | Speed of improvement | Higher is better             |
| `final_loss_ratio`  | Final/initial loss   | < 0.3 for effective training |

### Instability Scores

| Score | Interpretation       | Action Needed              |
| ----- | -------------------- | -------------------------- |
| 0-2   | Very stable training | None                       |
| 3-5   | Moderate instability | Monitor closely            |
| 6-10  | High instability     | Adjust hyperparameters     |
| >10   | Severe instability   | Redesign training approach |

### Overfitting Indicators

| Indicator     | Healthy Range        | Overfitting Signs       |
| ------------- | -------------------- | ----------------------- |
| Train-Val Gap | < 10% of loss value  | > 20% of loss value     |
| Gap Trend     | Stable or decreasing | Consistently increasing |
| Optimal Stop  | > 80% of epochs      | < 50% of epochs         |

### Statistical Significance

| P-Value | Interpretation         | Confidence |
| ------- | ---------------------- | ---------- |
| < 0.001 | Highly significant     | Very high  |
| < 0.01  | Significant            | High       |
| < 0.05  | Moderately significant | Moderate   |
| > 0.05  | Not significant        | Low        |

## Troubleshooting

### Common Issues

#### 1. No CSV Files Generated

**Problem**: Export script runs but no files appear in output directory.

**Solutions**:

- Check TensorBoard log directory path
- Ensure scalar metrics exist in logs
- Verify write permissions on output directory
- Check for error messages in export script output

```bash
# Debug export issues
python export_tensorboard_to_csv.py --logdir your/log/path --debug
```

#### 2. Analysis Scripts Fail

**Problem**: Python analysis scripts crash with errors.

**Solutions**:

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check CSV file format and completeness
- Verify sufficient system memory (>1GB free)
- Review error messages for specific issues

#### 3. Empty or Incomplete Reports

**Problem**: Generated reports lack content or visualizations.

**Solutions**:

- Ensure analysis scripts completed successfully
- Check for missing JSON files in analysis directory
- Verify CSV files contain actual data
- Review script output for warnings

#### 4. Permission Errors

**Problem**: Scripts fail with permission denied errors.

**Solutions**:

```bash
# Fix script permissions
chmod +x *.sh

# Fix directory permissions
chmod 755 tensorboard_exports/
chmod 755 training_analysis_results/
```

### Performance Issues

#### Large Dataset Handling

For very large datasets (>1M data points):

1. **Increase Memory**: Ensure at least 4GB RAM available
2. **Chunked Processing**: Process data in smaller batches
3. **Reduced Smoothing**: Use fewer smoothing techniques
4. **Selective Analysis**: Focus on specific metrics of interest

```python
# Memory-efficient analysis
analyzer = AdvancedTrainingAnalyzer()
analyzer.batch_size = 10000  # Process in smaller chunks
```

#### Slow Processing

To speed up analysis:

1. **Parallel Processing**: Use multiprocessing for independent metrics
2. **Reduced Resolution**: Downsample high-frequency data
3. **Focused Analysis**: Disable unused analysis components

## Best Practices

### Data Preparation

1. **Consistent Logging**: Ensure all metrics are logged at regular intervals
2. **Meaningful Names**: Use descriptive metric names for better analysis
3. **Complete Runs**: Avoid interrupted training runs for accurate analysis
4. **Version Control**: Keep track of different experimental configurations

### Analysis Workflow

1. **Start Simple**: Begin with basic visualizations before advanced analysis
2. **Iterative Analysis**: Run analysis multiple times as training progresses
3. **Compare Experiments**: Use consistent analysis across different runs
4. **Document Findings**: Save insights and recommendations for future reference

### Interpretation Guidelines

1. **Context Matters**: Consider experimental setup when interpreting results
2. **Multiple Metrics**: Don't rely on single metrics for conclusions
3. **Statistical Significance**: Verify findings with statistical tests
4. **Domain Knowledge**: Apply domain expertise to validate automated insights

## Customization

### Custom Analysis Scripts

To create custom analysis scripts:

```python
import pandas as pd
import numpy as np
from pathlib import Path

class CustomAnalyzer:
    def __init__(self, csv_dir="tensorboard_exports"):
        self.csv_dir = Path(csv_dir)
        self.results = {}
  
    def load_data(self):
        """Load CSV files for analysis."""
        data_files = {}
        for csv_file in self.csv_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            category = csv_file.stem.replace('tensorboard_', '')
            data_files[category] = df
        return data_files
  
    def custom_analysis(self, data_files):
        """Implement your custom analysis logic."""
        # Your analysis code here
        pass
  
    def export_results(self):
        """Export analysis results."""
        # Export logic here
        pass
```

### Custom Visualization

To create custom plots:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def custom_plot(data, output_path):
    """Create custom visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))
  
    # Your plotting code here
    ax.plot(data['x'], data['y'])
    ax.set_title('Custom Analysis Plot')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
  
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### Configuration Files

Create custom configuration for analysis parameters:

```yaml
# analysis_config.yaml
convergence:
  patience: 10
  min_improvement: 0.001
  
smoothing:
  methods: ['moving_avg', 'exponential', 'gaussian']
  window_sizes: [5, 10, 20]
  
visualization:
  figure_size: [12, 8]
  dpi: 300
  style: 'seaborn'
```

## API Reference

### Core Classes

#### AdvancedTrainingAnalyzer

```python
class AdvancedTrainingAnalyzer:
    def __init__(self, csv_dir: str, output_dir: str)
    def load_data(self) -> Dict[str, pd.DataFrame]
    def analyze_convergence_patterns(self, data_files: Dict)
    def detect_training_instabilities(self, data_files: Dict)
    def analyze_multitask_dynamics(self, data_files: Dict)
    def calculate_efficiency_metrics(self, data_files: Dict)
    def export_analysis_results(self)
    def run_complete_analysis(self)
```

#### LossCurveAnalyzer

```python
class LossCurveAnalyzer:
    def __init__(self, csv_dir: str, output_dir: str)
    def load_loss_data(self) -> Dict[str, pd.DataFrame]
    def apply_smoothing(self, data: np.array, method: str) -> Dict[str, np.array]
    def analyze_loss_characteristics(self, loss_data: Dict)
    def analyze_overfitting_patterns(self, loss_data: Dict)
    def calculate_derivatives(self, loss_data: Dict)
    def generate_publication_quality_plots(self, loss_data: Dict)
    def run_complete_analysis(self)
```

#### TrainingDynamicsReporter

```python
class TrainingDynamicsReporter:
    def __init__(self, analysis_dir: str, csv_dir: str, output_dir: str)
    def load_analysis_results(self)
    def generate_executive_summary(self)
    def generate_recommendations(self)
    def generate_html_report(self) -> str
    def export_results(self) -> Dict[str, Path]
    def run_complete_reporting(self)
```

### Utility Functions

#### Data Processing

```python
def smooth_series(data: np.array, window_size: int, method: str) -> np.array
def detect_convergence_epoch(loss_data: np.array, patience: int) -> Tuple[int, float]
def calculate_improvement_ratio(initial: float, final: float) -> float
```

#### Statistical Analysis

```python
def perform_significance_test(early_data: np.array, late_data: np.array) -> Dict
def calculate_effect_size(group1: np.array, group2: np.array) -> float
def detect_outliers(data: np.array, method: str) -> List[int]
```

### Command Line Interface

#### Export Script

```bash
python export_tensorboard_to_csv.py [OPTIONS]

Options:
  --logdir TEXT          TensorBoard log directory
  --output_dir TEXT      Output directory for CSV files
  --group_by_category    Group metrics by category
  --include_summary      Generate summary statistics
  --help                 Show help message
```

#### Analysis Scripts

```bash
python advanced_training_analysis.py [OPTIONS]
python loss_curve_analyzer.py [OPTIONS]
python training_dynamics_report.py [OPTIONS]

Common Options:
  --csv_dir TEXT         Directory with exported CSV files
  --output_dir TEXT      Directory for analysis results
  --config TEXT          Configuration file path
  --help                 Show help message
```

---

## Support and Contributing

### Getting Help

1. **Check Documentation**: Review this guide thoroughly
2. **Error Messages**: Pay attention to specific error messages
3. **Log Files**: Check generated log files for detailed information
4. **Examples**: Use provided examples as reference

### Common File Locations

```
project_root/
├── tensorboard_exports/           # CSV exports
├── training_analysis_results/     # Analysis outputs
├── logs/                         # TensorBoard logs
├── documentation/                # This guide
└── runs/                        # Training checkpoints
```

### Best Practices Summary

1. **Always backup** your original TensorBoard logs
2. **Run complete pipeline** first, then customize as needed
3. **Monitor disk space** during analysis (can generate large files)
4. **Review generated reports** thoroughly before making training decisions
5. **Keep analysis results** for future comparisons
6. **Document your findings** and optimization decisions

This guide provides comprehensive coverage of the TensorBoard analysis system. For additional questions or custom requirements, refer to the individual script documentation and inline comments.


