# ABR Signal Generation Model - Evaluation Pipeline

This comprehensive evaluation pipeline provides detailed assessment of your trained ABR signal generation model across multiple dimensions of quality, consistency, and performance.

## 🎯 Overview

The evaluation pipeline assesses:
- **Signal Quality**: Time-domain and frequency-domain metrics
- **Generation Performance**: Speed, consistency, and reliability  
- **Conditional Control**: Response to different input conditions
- **Physiological Accuracy**: ABR-specific wave analysis
- **Statistical Validation**: Comparison with real signals

## 📦 Components

### Core Modules

1. **`evaluation/metrics.py`** - Comprehensive signal quality metrics
2. **`evaluation/evaluator.py`** - Main evaluation orchestrator
3. **`evaluation/visualization.py`** - Advanced plotting and dashboards
4. **`evaluation/analysis.py`** - Specialized signal analysis tools
5. **`evaluate_model.py`** - Command-line evaluation script

### Key Features

- **Time-Domain Metrics**: MSE, RMSE, SNR, PSNR, correlation
- **Frequency Analysis**: Power spectral density, spectral features
- **Perceptual Metrics**: Morphological similarity, phase coherence
- **ABR-Specific Analysis**: Wave component detection and analysis
- **Interactive Dashboards**: HTML reports with Plotly visualizations
- **Statistical Tests**: Comprehensive comparisons with real signals

## 🚀 Quick Start

### Basic Evaluation

```bash
python evaluate_model.py \
    --config configs/config_colab_a100.yaml \
    --checkpoint checkpoints/pro/best.pth \
    --output_dir evaluation_results
```

### Comprehensive Evaluation

```bash
python evaluate_model.py \
    --config configs/config_colab_a100.yaml \
    --checkpoint checkpoints/pro/best.pth \
    --output_dir evaluation_results \
    --detailed_analysis \
    --num_samples 500 \
    --num_ddim_steps 100
```

### Custom Configuration

```bash
python evaluate_model.py \
    --config configs/config_colab_a100.yaml \
    --checkpoint checkpoints/pro/best.pth \
    --output_dir my_evaluation \
    --device cuda \
    --batch_size 64 \
    --cfg_scale 1.5
```

## 📊 Evaluation Phases

### Phase 1: Dataset Evaluation
- Evaluates model on test dataset
- Computes comprehensive metrics for each sample
- Aggregates statistics across all samples
- Saves sample comparisons for visualization

### Phase 2: Generation Quality Assessment
- Tests generation with different sampling parameters
- Measures generation speed and consistency
- Analyzes signal properties across generations
- Evaluates sampling parameter effects

### Phase 3: Conditional Control Analysis *(Detailed mode)*
- Tests model response to different conditions
- Evaluates interpolation and extrapolation
- Measures conditional control accuracy
- Analyzes condition-signal relationships

### Phase 4: Consistency Evaluation *(Detailed mode)*
- Tests generation consistency with same inputs
- Measures intra-condition variance
- Evaluates model determinism/stochasticity
- Analyzes generation stability

### Phase 5: Report Generation
- Creates comprehensive JSON report
- Generates visualization plots
- Saves summary CSV files
- Creates interactive HTML dashboard

## 📈 Output Structure

```
evaluation_results/
├── evaluation_report.json          # Comprehensive results
├── summary_metrics.csv             # Key metrics summary
└── plots/
    ├── sample_comparisons.png       # Generated vs real samples
    ├── overlay_comparison.png       # Signal overlays
    ├── frequency_analysis.png       # Frequency domain analysis
    ├── metrics_distribution.png     # Metrics distributions
    ├── conditional_analysis.png     # Conditional control analysis
    └── interactive_dashboard.html   # Interactive Plotly dashboard
```

## 🔬 Metrics Details

### Time-Domain Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| **MSE** | Mean Squared Error | < 0.01 |
| **RMSE** | Root Mean Squared Error | < 0.1 |
| **SNR** | Signal-to-Noise Ratio (dB) | > 20 dB |
| **PSNR** | Peak Signal-to-Noise Ratio | > 30 dB |
| **Correlation** | Pearson correlation | > 0.8 |

### Frequency-Domain Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| **Spectral Centroid** | Center of spectral mass | Frequency content |
| **Spectral Bandwidth** | Spread of spectrum | Signal complexity |
| **PSD Comparison** | Power spectral density | Frequency accuracy |
| **Frequency Response** | Magnitude spectrum | Spectral fidelity |

### ABR-Specific Metrics

| Metric | Description | Clinical Relevance |
|--------|-------------|-------------------|
| **Wave Detection** | I, III, V wave identification | Physiological accuracy |
| **Interpeak Latencies** | Time between waves | Neural conduction |
| **Amplitude Ratios** | Relative wave amplitudes | Hearing assessment |
| **Morphological Similarity** | Waveform shape matching | Clinical utility |

## 🎛️ Configuration Options

### Command Line Arguments

```bash
--config PATH              # Configuration file path
--checkpoint PATH           # Model checkpoint path  
--output_dir DIR           # Output directory
--device DEVICE            # cuda/cpu/auto
--num_samples N            # Number of samples to evaluate
--batch_size N             # Evaluation batch size
--num_ddim_steps N         # DDIM sampling steps
--cfg_scale FLOAT          # Classifier-free guidance scale
--detailed_analysis        # Enable comprehensive analysis
```

### Configuration File Settings

```yaml
data:
  path: "data/processed/ultimate_dataset_with_clinical_thresholds.pkl"
  batch_size: 32
  sampling_rate: 1000

model:
  signal_length: 200
  static_dim: 4
  # ... other model parameters

diffusion:
  schedule_type: "cosine"
  num_timesteps: 1000
```

## 📊 Understanding Results

### Signal Quality Assessment

- **SNR > 20 dB**: Excellent signal quality
- **Correlation > 0.8**: High fidelity reconstruction
- **RMSE < 0.1**: Low reconstruction error

### Generation Performance

- **Generation Time < 1s**: Real-time capable
- **Consistency Score > 0.7**: Stable generation
- **Condition Response**: Appropriate to clinical expectations

### Recommendations Interpretation

The pipeline provides automatic recommendations based on results:

- **Low SNR**: Increase training epochs or adjust loss weights
- **Poor Correlation**: Improve model architecture or data quality
- **Slow Generation**: Reduce DDIM steps or optimize model
- **Inconsistency**: Adjust noise schedule or model capacity

## 🔧 Advanced Usage

### Custom Metrics

Add custom metrics by extending the `SignalMetrics` class:

```python
from evaluation.metrics import SignalMetrics

class CustomMetrics(SignalMetrics):
    @staticmethod
    def my_custom_metric(predicted, target):
        # Your custom metric implementation
        return result
```

### Custom Analysis

Extend the `SignalAnalyzer` for specialized analysis:

```python
from evaluation.analysis import SignalAnalyzer

class CustomAnalyzer(SignalAnalyzer):
    def custom_analysis(self, signal):
        # Your custom analysis
        return results
```

### Batch Evaluation

Evaluate multiple checkpoints:

```bash
for checkpoint in checkpoints/*.pth; do
    python evaluate_model.py \
        --config configs/config_colab_a100.yaml \
        --checkpoint "$checkpoint" \
        --output_dir "results/$(basename $checkpoint .pth)"
done
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Memory Error**
   ```bash
   # Reduce batch size
   --batch_size 16
   ```

2. **Missing Dependencies**
   ```bash
   pip install librosa plotly scikit-learn
   ```

3. **Slow Evaluation**
   ```bash
   # Reduce samples or disable detailed analysis
   --num_samples 100
   # Remove --detailed_analysis flag
   ```

4. **Checkpoint Loading Error**
   ```
   # Ensure checkpoint path is correct and model architecture matches
   ```

### Performance Tips

- Use `--num_samples` to limit evaluation size during development
- Remove `--detailed_analysis` for faster basic evaluation
- Reduce `--num_ddim_steps` for faster generation
- Use smaller `--batch_size` if memory limited

## 📚 References

- **DDIM Sampling**: Denoising Diffusion Implicit Models
- **ABR Analysis**: Auditory Brainstem Response clinical standards
- **Signal Metrics**: Standard signal processing evaluation metrics
- **Perceptual Metrics**: Human auditory perception research

## 💡 Tips for Best Results

1. **Baseline Comparison**: Always compare with baseline models
2. **Multiple Runs**: Average results across multiple evaluation runs
3. **Condition Testing**: Test diverse conditional inputs
4. **Clinical Validation**: Validate with domain experts
5. **Iterative Improvement**: Use results to guide model improvements

---

*This evaluation pipeline provides comprehensive assessment of your ABR signal generation model. Use the results to understand model performance, identify improvement areas, and validate clinical utility.*