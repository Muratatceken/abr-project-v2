# Comprehensive ABR Model Evaluation Pipeline

A complete evaluation framework for multi-task ABR signal generation models, providing detailed assessment across all model outputs with visual diagnostics and clinical performance analysis.

## 🎯 **Overview**

This evaluation pipeline provides comprehensive assessment of ABR models across four main tasks:
- **Signal Reconstruction Quality** - MSE, MAE, SNR, correlation, DTW, spectral analysis
- **Peak Detection & Estimation** - Existence detection, latency/amplitude regression
- **Hearing Loss Classification** - Multi-class accuracy, F1-scores, confusion matrices
- **Hearing Threshold Estimation** - Regression metrics with clinical significance

## 🚀 **Quick Start**

### Basic Evaluation
```bash
# Evaluate a trained model on test set
python evaluate.py --checkpoint model.pth --split test

# Evaluate with DDIM sampling for realistic assessment
python evaluate.py --checkpoint model.pth --split test --use_ddim

# Evaluate with custom configuration
python evaluate.py --checkpoint model.pth --config evaluation/eval_config.yaml
```

### Advanced Usage
```bash
# Full evaluation with all logging
python evaluate.py \
    --checkpoint best_model.pth \
    --data_path data/processed/ultimate_dataset.pkl \
    --split test \
    --use_ddim \
    --use_tensorboard \
    --use_wandb \
    --wandb_project "abr-model-evaluation" \
    --output_dir "outputs/comprehensive_eval"
```

## 📊 **Evaluation Tasks**

### **Task 1: Signal Reconstruction Quality**

Comprehensive assessment of generated signal quality:

```python
def evaluate_reconstruction(y_true, y_pred):
    return {
        "mse": ...,           # Mean Squared Error
        "mae": ...,           # Mean Absolute Error  
        "rmse": ...,          # Root Mean Squared Error
        "snr": ...,           # Signal-to-Noise Ratio (dB)
        "pearson_corr": ...,  # Pearson correlation coefficient
        "dtw": ...,           # Dynamic Time Warping distance
        "spectral_mse": ...,  # FFT magnitude spectrum MSE
        "phase_coherence": ...# Phase coherence analysis
    }
```

**Features:**
- Per-sample metrics with mean ± std aggregation
- Spectral analysis via FFT
- Dynamic Time Warping (if fastdtw available)
- Robust correlation handling for constant signals

### **Task 2: Peak Detection & Estimation**

Multi-faceted peak analysis:

```python
def evaluate_peak_estimation(
    peak_exists_pred, peak_exists_true,
    peak_latency_pred, peak_latency_true,
    peak_amplitude_pred, peak_amplitude_true,
    peak_mask
):
    return {
        # Existence detection (binary classification)
        "existence_accuracy": ...,
        "existence_f1": ...,
        "existence_auc": ...,
        
        # Value regression (masked for valid peaks only)
        "latency_mae": ...,
        "latency_rmse": ...,
        "latency_r2": ...,
        "amplitude_mae": ...,
        "amplitude_rmse": ...,
        "amplitude_r2": ...,
        
        # Error distributions
        "latency_error_mean": ...,
        "amplitude_error_std": ...
    }
```

**Features:**
- Proper masking for missing peak data
- Binary classification metrics for peak existence
- Regression metrics for peak values (latency, amplitude)
- Error distribution analysis

### **Task 3: Hearing Loss Classification**

Complete multi-class evaluation:

```python
def evaluate_classification(pred_class_logits, true_class):
    return {
        "accuracy": ...,
        "balanced_accuracy": ...,
        "macro_f1": ...,
        "micro_f1": ...,
        "weighted_f1": ...,
        "confusion_matrix": ...,
        "per_class_f1": {...},
        "classification_report": {...},
        "class_distributions": {...}
    }
```

**Features:**
- Standard and balanced accuracy metrics
- Per-class F1 scores for all hearing loss types
- Confusion matrix analysis
- Class distribution comparison (true vs predicted)

### **Task 4: Threshold Estimation**

Regression analysis with clinical context:

```python
def evaluate_threshold_estimation(threshold_pred, threshold_true):
    return {
        "mae": ...,           # Mean Absolute Error (dB)
        "mse": ...,           # Mean Squared Error
        "rmse": ...,          # Root Mean Squared Error
        "r2": ...,            # R-squared coefficient
        "log_mae": ...,       # Log-scale MAE for better threshold handling
        "pearson_corr": ...,  # Correlation coefficient
        "error_percentiles": {...}  # Error distribution analysis
    }
```

**Features:**
- Standard regression metrics
- Log-scale analysis for threshold values
- Percentile-based error analysis
- Clinical significance assessment

### **Task 5: Clinical Failure Mode Analysis**

Automated detection of clinically significant failures:

```python
def compute_failure_modes(...):
    return {
        "false_peak_detected": ...,      # Predicted peak when none exists
        "missed_peak_detected": ...,     # Failed to detect existing peak
        "threshold_overestimated": ...,  # >15 dB overestimation
        "threshold_underestimated": ..., # >15 dB underestimation
        "severe_class_mismatch": ...,    # Misclassification in severe cases
        "normal_as_severe": ...          # Normal classified as severe
    }
```

**Clinical Significance:**
- **False Peak Detection**: Can lead to incorrect treatment decisions
- **Threshold Errors >15 dB**: Clinically significant for hearing aid fitting
- **Severe Case Mismatches**: Critical for treatment planning
- **Normal as Severe**: May cause unnecessary interventions

## 📈 **Diagnostic Visualizations**

### **Signal Reconstruction Plots**
- True vs predicted waveforms
- Peak annotations (latency, amplitude)
- Classification and threshold predictions
- Per-sample correlation and MSE

### **Peak Prediction Scatter Plots**
- Latency predictions with R² analysis
- Amplitude predictions with perfect prediction line
- Error distribution visualization

### **Classification Analysis**
- Normalized confusion matrices
- Per-class performance heatmaps
- Class distribution comparisons

### **Threshold Analysis**
- Scatter plots with regression lines
- Error distribution histograms
- Clinical range analysis

### **Error Distribution Analysis**
- MSE and MAE error histograms
- Peak error distributions
- Threshold error analysis

## 🔧 **Configuration**

### **Basic Configuration**
```yaml
# evaluation/eval_config.yaml
batch_size: 32
save_dir: "outputs/evaluation"

reconstruction:
  dtw: true              # Enable DTW analysis
  fft_mse: true          # Enable spectral analysis
  
clinical:
  thresholds:
    threshold_overestimate: 15.0  # dB clinical significance
    peak_latency_tolerance: 0.5   # ms acceptable error
    
visualization:
  waveform_samples: 5    # Number of diagnostic plots
  figsize: [15, 10]      # Plot dimensions
  dpi: 150               # Plot resolution
```

### **Advanced Options**
- **DTW Analysis**: Dynamic Time Warping for temporal alignment assessment
- **Spectral Analysis**: FFT-based magnitude and phase analysis
- **Clinical Thresholds**: Configurable significance levels
- **Bootstrap Analysis**: Statistical confidence intervals
- **Multi-format Output**: JSON, CSV, PNG, PDF export options

## 📁 **Output Structure**

```
outputs/evaluation/
├── data/
│   ├── comprehensive_eval_metrics.json      # Aggregate metrics
│   ├── comprehensive_eval_batch_results.json # Per-batch results
│   └── comprehensive_eval_summary.csv       # Summary table
├── figures/
│   ├── batch_0_signal_reconstruction.png    # Diagnostic plots
│   ├── batch_0_peak_predictions.png
│   ├── batch_0_classification.png
│   ├── batch_0_threshold_predictions.png
│   └── batch_0_error_distributions.png
└── tensorboard/                             # TensorBoard logs (if enabled)
```

## 🔬 **Integration with Training**

### **During Training**
```python
from evaluation.comprehensive_eval import ABRComprehensiveEvaluator

# Create evaluator
evaluator = ABRComprehensiveEvaluator(
    config=eval_config,
    class_names=class_names,
    save_dir="outputs/training_eval"
)

# Evaluate during validation
for batch in val_loader:
    outputs = model(batch['signal'], batch['static_params'])
    evaluator.evaluate_batch(batch, outputs)

# Compute and log metrics
metrics = evaluator.compute_aggregate_metrics()
evaluator.log_to_tensorboard(writer, epoch)
```

### **Standalone Evaluation**
```python
# Load trained model
model = load_model_from_checkpoint("best_model.pth")

# Run comprehensive evaluation
python evaluate.py --checkpoint best_model.pth --split test --use_ddim
```

## 📊 **Expected Output**

### **Console Summary**
```
🔬 COMPREHENSIVE ABR MODEL EVALUATION SUMMARY
================================================================================
📊 Dataset: 1,247 samples, 39 batches

📈 SIGNAL RECONSTRUCTION:
   MSE: 0.002143 ± 0.001205
   MAE: 0.031547 ± 0.018234
   SNR: 18.42 ± 4.67 dB
   Correlation: 0.8734 ± 0.1123

🎯 PEAK ESTIMATION:
   Existence F1: 0.8456 ± 0.0234
   Latency MAE: 0.3421 ± 0.1876 ms
   Amplitude MAE: 0.0876 ± 0.0543 μV

🏷️  CLASSIFICATION:
   Accuracy: 0.7823 ± 0.0156
   Macro F1: 0.7234 ± 0.0234
   Balanced Accuracy: 0.7654 ± 0.0198

📏 THRESHOLD ESTIMATION:
   MAE: 8.43 ± 3.21 dB
   RMSE: 12.76 ± 4.87 dB
   R²: 0.6543 ± 0.0876

⚠️  CLINICAL FAILURE MODES:
   False peaks detected: 23 (1.84%)
   Missed peaks: 45 (3.61%)
   Threshold overestimated: 67 (5.37%)
   Severe class mismatches: 12 (0.96%)
```

### **Saved Files**
- **JSON**: Detailed metrics with statistical analysis
- **CSV**: Summary table for spreadsheet analysis
- **PNG**: High-resolution diagnostic visualizations
- **TensorBoard**: Interactive metric exploration
- **W&B**: Cloud-based experiment tracking

## 🛠️ **Requirements**

### **Core Dependencies**
```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn
```

### **Optional Dependencies**
```bash
# For DTW analysis
pip install fastdtw

# For logging
pip install tensorboard wandb

# For configuration
pip install pyyaml
```

### **System Requirements**
- **Memory**: 8GB+ RAM recommended for large datasets
- **GPU**: Optional, but recommended for faster evaluation
- **Storage**: ~100MB per evaluation run (with visualizations)

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Memory Errors**
```bash
# Reduce batch size
python evaluate.py --checkpoint model.pth --batch_size 16

# Skip visualizations
python evaluate.py --checkpoint model.pth --no_visualizations
```

#### **Missing Dependencies**
```bash
# DTW not available
# Will automatically skip DTW analysis and continue

# TensorBoard/W&B not available  
# Will show warning and continue without logging
```

#### **Model Compatibility**
```bash
# Ensure model checkpoint contains proper state dict
# Check model architecture matches checkpoint
```

### **Performance Optimization**
- Use smaller batch sizes for memory-constrained systems
- Disable visualizations for faster evaluation
- Use CPU for small datasets, GPU for large ones
- Enable mixed precision for faster GPU evaluation

## 📞 **Support**

For issues or questions:
1. Check the troubleshooting section above
2. Review the configuration options in `eval_config.yaml`
3. Use debug mode: `python evaluate.py --checkpoint model.pth --batch_size 4`
4. Check model compatibility and data format

The comprehensive evaluation pipeline provides detailed, clinically-relevant assessment of ABR models with rich diagnostic capabilities and flexible configuration options. 