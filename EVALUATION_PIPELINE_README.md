# ABR Transformer Evaluation Pipeline

Comprehensive evaluation pipeline for trained ABR Transformer models with detailed metrics, rich visualizations, and professional reporting.

## 🎯 Overview

The evaluation pipeline provides two core evaluation modes:

1. **Reconstruction**: Denoise from noisy `x_t` back to `x̂_0` (tests denoising capability)
2. **Conditional Generation**: Sample from noise using static conditioning, compare to real `x_0` (tests generation quality)

## 📊 Features

### ✅ **Comprehensive Metrics**
- **Time-domain**: MSE, L1, correlation, SNR
- **Frequency-domain**: Multi-resolution STFT L1 loss
- **Temporal alignment**: Dynamic Time Warping (DTW) distance  
- **Peak analysis**: ABR waves I, III, V latency/amplitude metrics (auto-enabled if labels available)

### ✅ **Rich Visualizations**
- **Overlay plots**: Reference vs generated waveforms
- **Error curves**: `|reference - generated|` analysis
- **Spectrograms**: Frequency-domain comparison
- **Scatter plots**: Metric correlation analysis
- **Best/worst samples**: Automatic selection by MSE

### ✅ **Professional Reporting**
- **TensorBoard**: Real-time scalars and figures
- **CSV exports**: Per-sample metrics for further analysis
- **JSON summaries**: Aggregated statistics with metadata
- **Console tables**: Immediate results overview

### ✅ **Flexible Configuration**
- **Multiple evaluation modes**: Mix and match reconstruction/generation
- **Configurable metrics**: Enable/disable specific computations
- **Batch processing**: Efficient evaluation of large datasets
- **EMA support**: Use exponential moving average weights

## 🚀 Quick Start

### 1. Basic Evaluation
```bash
# Evaluate both reconstruction and generation
python eval.py --config configs/eval.yaml
```

### 2. Mode-Specific Evaluation
```bash
# Only reconstruction evaluation
python eval.py --config configs/eval.yaml --override "modes.generation: false"

# Only generation evaluation  
python eval.py --config configs/eval.yaml --override "modes.reconstruction: false"
```

### 3. Custom Checkpoint
```bash
# Evaluate specific checkpoint
python eval.py --config configs/eval.yaml --override "
  checkpoint.path: 'checkpoints/abr_transformer/abr_vpred_base_e95.pt',
  exp_name: 'eval_epoch_95'
"
```

### 4. Advanced Sampling
```bash
# More DDIM steps for higher quality
python eval.py --config configs/eval.yaml --override "
  diffusion.sample_steps: 100,
  diffusion.ddim_eta: 0.1
"
```

## ⚙️ Configuration

Edit `configs/eval.yaml` to customize evaluation:

```yaml
# Evaluation modes
modes:
  reconstruction: true    # Denoise from x_t to x_0
  generation: true       # Sample from noise conditioned on static params

# Metrics to compute
metrics:
  use_stft: true         # Frequency-domain metrics
  use_dtw: true          # Temporal alignment metrics
  use_corr: true         # Correlation analysis
  use_snr: true          # Signal-to-noise ratio

# Checkpoint settings
checkpoint:
  path: "checkpoints/abr_transformer/abr_vpred_base_best.pt"
  use_ema: true          # Use EMA weights if available

# Output configuration
report:
  out_dir: "results/abr_eval"
  save_csv: true         # Per-sample metrics
  save_topk_examples: 12 # Best/worst samples to visualize
```

## 📁 Output Structure

After evaluation, results are saved to `results/abr_eval/`:

```
results/abr_eval/
├── eval_reconstruction_metrics.csv    # Per-sample reconstruction metrics
├── eval_reconstruction_summary.json   # Reconstruction summary statistics
├── eval_generation_metrics.csv        # Per-sample generation metrics
├── eval_generation_summary.json       # Generation summary statistics
└── figures/ (if enabled)
    ├── overlay_best_reconstruction.png
    ├── overlay_worst_reconstruction.png
    ├── error_curves_reconstruction.png
    └── spectrograms_*.png
```

## 📈 TensorBoard Monitoring

### Launch TensorBoard:
```bash
tensorboard --logdir runs/abr_transformer
```

### What You'll See:

#### **Scalars** (`eval/{mode}/`)
- `mse_mean`, `mse_std` - Mean squared error statistics
- `l1_mean`, `l1_std` - L1 error statistics  
- `corr_mean`, `corr_std` - Correlation statistics
- `snr_db_mean`, `snr_db_std` - Signal-to-noise ratio
- `stft_l1_mean`, `stft_l1_std` - Frequency domain errors
- `dtw_mean`, `dtw_std` - Dynamic time warping distances

#### **Images** (`eval/{mode}/`)
- `overlay_best` - Best samples (reference vs generated)
- `overlay_worst` - Worst samples (reference vs generated)
- `error_curves` - Error analysis plots
- `spectrogram_ref/gen` - Frequency domain visualization
- `scatter_corr_dtw` - Correlation vs DTW analysis

## 🔬 Peak Analysis (Automatic)

If your dataset contains ABR peak labels, the pipeline automatically enables peak analysis:

### **Required Labels:**
- `I_Latency`, `III_Latency`, `V_Latency` (in ms)
- `I_Amplitude`, `III_Amplitude`, `V_Amplitude` (optional)

### **Computed Metrics:**
- **Detection Rate**: Percentage of peaks found within expected windows
- **Latency MAE**: Mean absolute error in peak timing (ms)
- **Amplitude MAE**: Mean absolute error in peak amplitude
- **Per-wave Analysis**: Separate metrics for waves I, III, and V

### **Configuration:**
```yaml
metrics:
  peaks:
    enable_if_available: true
    find_peaks:
      height_sigma: 1.0      # Peak detection threshold
      min_distance: 6        # Minimum samples between peaks
    latency_windows:         # Search windows (sample indices)
      I:   [20, 50]          # Wave I window
      III: [70, 110]         # Wave III window  
      V:   [130, 170]        # Wave V window
```

## 📊 Understanding Results

### **Reconstruction Mode**
- **Purpose**: Tests the model's denoising capability
- **Process**: Add noise at timestep `t`, then reconstruct `x_0`
- **Interpretation**: Lower MSE/L1 = better denoising, higher correlation = better signal preservation

### **Generation Mode**  
- **Purpose**: Tests unconditional generation quality
- **Process**: Generate from pure noise conditioned only on static parameters
- **Interpretation**: Lower DTW = better temporal alignment, higher SNR = cleaner signals

### **Key Metrics**
- **MSE/L1**: Overall reconstruction fidelity
- **Correlation**: Signal shape preservation  
- **SNR**: Signal quality vs noise
- **STFT L1**: Frequency content accuracy
- **DTW**: Temporal pattern alignment

## 🔧 Advanced Usage

### **Multiple Seeds for Robustness**
```yaml
advanced:
  num_seeds: 3  # Generate 3 samples per condition, average metrics
```

### **Subset Evaluation (Quick Testing)**
```yaml
advanced:
  max_samples: 1000  # Limit to first 1000 samples
```

### **Custom Timestep Strategy (Reconstruction)**
```yaml
advanced:
  reconstruction:
    timestep_strategy: "fixed"  # "uniform" | "fixed" | "random_subset"
    fixed_timestep: 500         # If strategy="fixed"
```

### **Classifier-Free Guidance (Generation)**
```yaml
advanced:
  generation:
    cfg_scale: 1.5  # Stronger conditioning (>1.0)
```

## 🎯 Example Results

### **Good Model Performance:**
- MSE: ~0.001-0.01
- Correlation: >0.8
- SNR: >10 dB
- DTW: <50
- Peak detection: >80%

### **Poor Model Performance:**
- MSE: >0.1
- Correlation: <0.5
- SNR: <5 dB  
- DTW: >200
- Peak detection: <50%

## 🛠️ Troubleshooting

### **Common Issues:**

1. **"STFT computation failed"**
   - Reduce `n_fft` or increase `sequence_length`
   - Set `use_stft: false` to disable

2. **"DTW computation failed"**
   - DTW is O(T²), can be slow for long sequences
   - Set `use_dtw: false` for faster evaluation

3. **Memory errors**
   - Reduce `batch_size` in config
   - Disable visualizations temporarily

4. **No peak metrics**
   - Check if dataset has `I_Latency`, `III_Latency`, `V_Latency` columns
   - Peak analysis auto-enables only if labels are present

### **Performance Tips:**

1. **Faster evaluation**: Set `use_dtw: false`, reduce `batch_size`
2. **More thorough**: Increase `sample_steps`, enable all metrics
3. **Quick testing**: Set `max_samples: 100` for rapid iteration

## 📈 Interpreting TensorBoard

### **Look for:**
- **Consistent metrics**: Low variance across samples
- **Realistic correlation**: >0.6 for good ABR morphology
- **Proper spectrograms**: Clear frequency patterns
- **Good peak detection**: >70% for clinical relevance

### **Red flags:**
- **Very high DTW**: Poor temporal alignment
- **Low correlation**: Generated signals don't match ABR patterns
- **NaN values**: Numerical instability or configuration issues

## 🎉 Ready to Evaluate!

The evaluation pipeline is comprehensive and production-ready. Use it to:

- **Validate model performance** across different checkpoints
- **Compare different architectures** with standardized metrics  
- **Analyze failure modes** through worst-case visualizations
- **Generate publication-quality results** with rich documentation

Start evaluating with:
   ```bash
python eval.py --config configs/eval.yaml
```

Monitor progress in TensorBoard and analyze detailed results in the output directory! 📊⚡