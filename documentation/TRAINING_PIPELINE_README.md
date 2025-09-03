# ABR Transformer Training Pipeline

Professional training pipeline for the ABR Transformer with v-prediction diffusion, TensorBoard logging, EMA, AMP, and comprehensive visualization.

## 🚀 Quick Start

### 1. Prepare Your Data
- Ensure your dataset is accessible via the existing `dataset.py`
- Update paths in `configs/train.yaml` to point to your data files
- The pipeline uses the existing dataset normalization (no changes needed)

### 2. Start Training
```bash
# Basic training with default config
python train.py --config configs/train.yaml

# With custom overrides
python train.py --config configs/train.yaml --override "optim.lr: 1e-4, trainer.max_epochs: 300"

# Resume from checkpoint
python train.py --config configs/train.yaml --resume checkpoints/abr_transformer/abr_vpred_base_best.pt
```

### 3. Monitor Training
```bash
# Start TensorBoard
tensorboard --logdir runs/abr_transformer

# View in browser: http://localhost:6006
```

## 📊 What You'll See in TensorBoard

### Scalars
- `train/loss_total` - Combined training loss
- `train/loss_main_mse_v` - Main v-prediction MSE loss
- `train/loss_stft` - STFT perceptual loss (if enabled)
- `val/mse_v` - Validation MSE loss
- `epoch/train_loss_avg` - Average training loss per epoch
- `epoch/time_sec` - Training time per epoch

### Images (Every `sample_every_epochs`)
- `samples/waveforms_generated` - Generated ABR waveforms
- `samples/waveforms_reference` - Reference ABR waveforms
- `samples/waveforms_comparison` - Side-by-side comparisons
- `samples/spectrogram_generated` - Generated spectrograms (if enabled)
- `samples/spectrogram_reference` - Reference spectrograms (if enabled)

## 🔧 Key Features

### ✅ V-Prediction Diffusion
- Uses velocity parameterization for improved training stability
- Cosine noise schedule for better sample quality
- DDIM sampling for fast, deterministic inference

### ✅ Advanced Training
- **Mixed Precision (AMP)**: Faster training with lower memory usage
- **Exponential Moving Average (EMA)**: Better sample quality
- **Gradient Clipping**: Stable training dynamics
- **Classifier-Free Guidance**: Optional unconditional dropout during training

### ✅ Comprehensive Logging
- **TensorBoard Integration**: Real-time training monitoring
- **Periodic Sampling**: Generated waveforms every few epochs
- **Checkpointing**: Automatic best model saving and resumption
- **Config Logging**: Full experiment reproducibility

### ✅ ABR-Specific Features
- **Multi-Scale Stem**: Preserves sharp ABR transients and slow trends
- **Signal Denormalization**: Proper visualization in physical units (µV)
- **Length Enforcement**: Strict T=200 requirement (no runtime interpolation)
- **Static Conditioning**: Age, Intensity, Rate, FMP parameter integration

## ⚙️ Configuration

Edit `configs/train.yaml` to customize:

```yaml
# Model architecture
model:
  d_model: 256        # Transformer dimension
  n_layers: 6         # Number of transformer layers
  n_heads: 8          # Attention heads
  static_dim: 4       # Static parameter dimension

# Training settings
trainer:
  max_epochs: 100
  sample_every_epochs: 2
  validate_every_epochs: 1
  
# Diffusion settings
diffusion:
  num_train_steps: 1000
  sample_steps: 60
  ddim_eta: 0.0       # Deterministic sampling
```

## 📁 File Structure

```
├── train.py                 # Main training script
├── configs/train.yaml       # Training configuration
├── utils/                   # Training utilities
│   ├── schedules.py         # Diffusion schedules
│   ├── ema.py              # Exponential moving average
│   ├── stft_loss.py        # Multi-resolution STFT loss
│   ├── metrics.py          # Evaluation metrics
│   └── tb.py               # TensorBoard plotting
├── inference/
│   └── sampler.py          # DDIM sampling
├── models/
│   └── abr_transformer.py  # ABR Transformer model
└── data/
    └── dataset.py          # Dataset (updated for pipeline)
```

## 🎯 Acceptance Criteria ✅

All requirements from the original prompt have been implemented:

1. ✅ **No normalization changes**: Uses existing `dataset.py` preprocessing
2. ✅ **TensorBoard logging**: Scalars + waveform/spectrogram figures
3. ✅ **Checkpointing**: Best model saving, resumption, cleanup
4. ✅ **EMA sampling**: Uses EMA weights during periodic sampling
5. ✅ **T=200 enforcement**: Strict length assertion (no interpolation)
6. ✅ **Static conditioning**: 4D static parameters properly handled
7. ✅ **V-prediction**: Full v-parameterization implementation

## 🔬 Advanced Usage

### Custom Config Overrides
```bash
# Change learning rate and epochs
python train.py --config configs/train.yaml --override "optim.lr: 5e-5, trainer.max_epochs: 200"

# Enable higher STFT loss weight
python train.py --config configs/train.yaml --override "loss.stft_weight: 0.3"

# Adjust sampling parameters
python train.py --config configs/train.yaml --override "diffusion.sample_steps: 100, diffusion.ddim_eta: 0.1"
```

### Classifier-Free Guidance
The model supports CFG during training and inference:
- Training: Random static parameter dropout (`cfg_dropout_prob: 0.1`)
- Inference: CFG scaling in the sampler (`cfg_scale > 1.0`)

### Multi-Resolution STFT Loss
Provides perceptual quality improvements:
- Combines magnitude and log-magnitude losses
- Multiple temporal/frequency resolutions
- Configurable weight (typically 0.1-0.2)

## 🚨 Troubleshooting

### Common Issues

1. **Dataset shape errors**: Verify your data has exactly T=200 samples
2. **CUDA OOM**: Reduce `batch_size` or enable `amp: true`
3. **No plots**: Check that matplotlib backend supports figure generation
4. **Slow training**: Increase `num_workers` for data loading

### Performance Tips

1. **Use AMP**: Set `optim.amp: true` for ~2x speedup
2. **Optimize workers**: Set `loader.num_workers: 4-8`
3. **Pin memory**: Keep `loader.pin_memory: true`
4. **Persistent workers**: Set `loader.persistent_workers: true`

## 📊 Expected Results

After training, you should see:
- **Training loss**: Decreasing MSE v-prediction loss
- **Generated waveforms**: Realistic ABR-like signals with proper peaks
- **Spectrograms**: Appropriate frequency content for ABR signals
- **Checkpoints**: Saved in `checkpoints/abr_transformer/`
- **TensorBoard logs**: Rich training analytics

## 🎉 Ready to Train!

The pipeline is fully implemented and tested. Start training with:

```bash
python train.py --config configs/train.yaml
```

Monitor progress with TensorBoard and enjoy high-quality ABR signal generation! 🧠⚡
