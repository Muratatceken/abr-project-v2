# Enhanced ABR Training Pipeline

A comprehensive, modular training pipeline for the **ProfessionalHierarchicalUNet** model with multi-task learning, classifier-free guidance (CFG), FiLM-based conditioning, and advanced optimization techniques.

## 🚀 Quick Start

### Basic Training
```bash
# Train with default configuration
python run_training.py

# Train with custom configuration
python run_training.py --config training/config.yaml

# Train with command line overrides
python run_training.py --config training/config.yaml --batch_size 64 --learning_rate 2e-4
```

### Quick Test Modes
```bash
# Debug mode (minimal settings for testing)
python run_training.py --debug

# Quick test mode (reduced epochs for validation)
python run_training.py --quick_test

# Valid peaks only (train on samples with V peak data)
python run_training.py --valid_peaks_only
```

## 📁 Pipeline Structure

```
training/
├── enhanced_train.py      # Main training script
├── config.yaml           # Default configuration
├── config_loader.py      # Configuration management
├── evaluation.py         # Comprehensive evaluation
└── README.md             # This file

run_training.py           # Simple training runner
```

## 🎯 Key Features

### ✅ **Model Architecture**
- **ProfessionalHierarchicalUNet** with S4 encoder and Transformer decoder
- **Cross-attention** between encoder and decoder
- **FiLM conditioning** with dropout for robustness
- **Multi-task heads**: signal reconstruction, peak prediction, classification, threshold
- **Positional encoding** throughout the architecture

### ✅ **Training Enhancements**
- **Mixed precision training** with automatic loss scaling
- **Gradient clipping** to prevent instability
- **Advanced schedulers**: Cosine warm restarts, ReduceLROnPlateau
- **Early stopping** with configurable patience
- **Class imbalance handling** with weights and focal loss

### ✅ **Data Processing**
- **Stratified train/validation splits** maintaining class distribution
- **Data augmentation**: noise injection, time shifts, amplitude scaling
- **CFG dropout**: random unconditional training for classifier-free guidance
- **Balanced sampling** for severely imbalanced classes
- **Proper masking** for missing V peak data

### ✅ **Loss Functions**
- **Multi-task loss** with configurable weights
- **Masked loss** for peak prediction (only valid peaks)
- **Huber loss** for robust signal reconstruction
- **Focal loss** option for severe class imbalance
- **Automatic class weighting** for imbalanced data

### ✅ **Monitoring & Logging**
- **TensorBoard** integration for training visualization
- **Weights & Biases** support for experiment tracking
- **Comprehensive metrics**: F1, balanced accuracy, correlation, R²
- **Automatic checkpointing** with best model saving
- **Detailed logging** with configurable verbosity

## ⚙️ Configuration

### YAML Configuration
The pipeline uses YAML-based configuration for maximum flexibility:

```yaml
# training/config.yaml
experiment:
  name: "enhanced_abr_v1"
  random_seed: 42

model:
  base_channels: 64
  n_levels: 4
  num_transformer_layers: 3
  use_cross_attention: true
  film_dropout: 0.15

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  use_amp: true
  patience: 15

loss:
  loss_weights:
    signal: 1.0
    classification: 1.5
    peak_exist: 0.5
    peak_latency: 1.0
    peak_amplitude: 1.0
    threshold: 0.8
```

### Command Line Arguments
All configuration options can be overridden via command line:

```bash
python run_training.py \
  --batch_size 64 \
  --learning_rate 2e-4 \
  --num_epochs 50 \
  --use_focal_loss \
  --use_wandb \
  --experiment_name "abr_focal_loss_test"
```

## 📊 Dataset Requirements

### Expected Dataset Structure
The pipeline expects the `ultimate_dataset.pkl` file with the following structure:

```python
{
    'data': [
        {
            'patient_id': str,
            'static_params': np.array([age, intensity, rate, fmp]),  # [4]
            'signal': np.array([...]),                               # [200]
            'v_peak': np.array([latency, amplitude]),                # [2]
            'v_peak_mask': np.array([lat_valid, amp_valid]),         # [2] bool
            'target': int  # 0-4: NORMAL, NÖROPATİ, SNİK, TOTAL, İTİK
        },
        ...
    ],
    'scaler': StandardScaler,
    'label_encoder': LabelEncoder,
    'metadata': {...}
}
```

### Class Distribution
The dataset handles imbalanced classes:
- **NORMAL**: 79.7% (41,391 samples)
- **SNİK**: 10.6% (5,513 samples)
- **İTİK**: 5.6% (2,925 samples)
- **TOTAL**: 3.3% (1,715 samples)
- **NÖROPATİ**: 0.8% (417 samples)

## 🏋️ Training Process

### 1. **Data Loading & Preprocessing**
```python
# Load dataset
data, scaler, label_encoder = load_dataset('data/processed/ultimate_dataset.pkl')

# Create stratified splits
train_loader, val_loader = create_data_loaders(data, config)
```

### 2. **Model Creation**
```python
# Initialize enhanced model
model = ProfessionalHierarchicalUNet(
    input_channels=1,
    static_dim=4,
    base_channels=64,
    n_levels=4,
    use_cross_attention=True,
    film_dropout=0.15,
    use_cfg=True
)
```

### 3. **Training Loop**
```python
for epoch in range(num_epochs):
    # Training phase
    train_metrics = trainer.train_epoch()
    
    # Validation phase
    val_metrics = trainer.validate_epoch()
    
    # Learning rate scheduling
    scheduler.step()
    
    # Save best model
    if val_metrics['f1_macro'] > best_f1:
        save_checkpoint('best_model.pth')
```

## 📈 Evaluation & Metrics

### Multi-Task Metrics
The pipeline computes comprehensive metrics for all tasks:

#### **Classification Metrics**
- F1-score (macro, weighted, per-class)
- Balanced accuracy
- Confusion matrix
- Classification report

#### **Peak Prediction Metrics**
- Peak existence F1 and accuracy
- Latency MAE, MSE, R² (masked)
- Amplitude MAE, MSE, R² (masked)
- Valid peaks ratio

#### **Signal Reconstruction Metrics**
- Signal MSE, MAE
- Signal correlation
- Signal-to-noise ratio (SNR)

#### **Threshold Prediction Metrics**
- Threshold MAE, MSE, R²

### Visualization
Automatic generation of evaluation plots:
- Confusion matrices (raw counts and normalized)
- Peak prediction scatter plots with R² scores
- Signal reconstruction examples
- Error distribution histograms

## 🔧 Advanced Features

### Classifier-Free Guidance (CFG)
```python
# During training: random unconditional samples
if torch.rand(1) < cfg_dropout_prob:
    force_uncond = True

# During inference: enhanced generation
outputs = model(x, static_params, cfg_guidance_scale=1.5)
```

### FiLM Dropout for Robustness
```python
# Randomly zero out conditioning during training
if training and film_dropout > 0:
    condition = torch.where(
        dropout_mask,
        condition,
        torch.zeros_like(condition)
    )
```

### Masked Loss for Peak Prediction
```python
# Only compute loss where peaks are valid
latency_loss = F.mse_loss(
    pred_latency[valid_mask], 
    true_latency[valid_mask]
)
```

### Mixed Precision Training
```python
# Automatic mixed precision for faster training
with autocast(enabled=use_amp):
    outputs = model(x, static_params)
    loss = loss_fn(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
```

## 🚨 Troubleshooting

### Common Issues

#### **CUDA Out of Memory**
```bash
# Reduce batch size
python run_training.py --batch_size 16

# Reduce model size
python run_training.py --base_channels 48 --n_levels 3
```

#### **Class Imbalance**
```bash
# Use focal loss
python run_training.py --use_focal_loss

# Use balanced sampling
python run_training.py --use_balanced_sampler

# Adjust loss weights
python run_training.py --classification_loss_weight 2.0
```

#### **Slow Convergence**
```bash
# Increase learning rate
python run_training.py --learning_rate 2e-4

# Use different scheduler
python run_training.py --scheduler reduce_on_plateau

# Reduce model complexity
python run_training.py --num_transformer_layers 2
```

#### **Overfitting**
```bash
# Increase regularization
python run_training.py --film_dropout 0.2 --weight_decay 0.02

# Use data augmentation
python run_training.py --augment

# Early stopping
python run_training.py --patience 10
```

### Debug Mode
```bash
# Minimal settings for debugging
python run_training.py --debug
```

This runs with:
- 2 epochs
- Batch size 4
- 0 workers
- Reduced model complexity

## 📝 Example Training Commands

### Basic Training
```bash
# Default settings
python run_training.py

# With Weights & Biases logging
python run_training.py --use_wandb --experiment_name "baseline_experiment"
```

### Advanced Training
```bash
# High-performance training
python run_training.py \
  --batch_size 64 \
  --learning_rate 2e-4 \
  --num_epochs 150 \
  --base_channels 96 \
  --num_transformer_layers 4 \
  --use_focal_loss \
  --use_wandb \
  --experiment_name "enhanced_large_model"
```

### Imbalanced Data Training
```bash
# Focus on class imbalance
python run_training.py \
  --use_focal_loss \
  --use_balanced_sampler \
  --classification_loss_weight 2.0 \
  --experiment_name "imbalanced_focus"
```

### Peak-Focused Training
```bash
# Train only on samples with valid peaks
python run_training.py \
  --valid_peaks_only \
  --peak_latency_weight 1.5 \
  --peak_amplitude_weight 1.5 \
  --experiment_name "peak_focused"
```

## 🎯 Expected Results

### Performance Targets
Based on the enhanced architecture, expect:

- **Classification F1 (Macro)**: 0.75-0.85
- **Peak Existence F1**: 0.80-0.90
- **Peak Latency R²**: 0.70-0.85
- **Signal Correlation**: 0.85-0.95
- **Training Time**: ~2-4 hours on RTX 3080

### Convergence
- **Initial loss**: ~2.5-3.0
- **Final loss**: ~0.8-1.2
- **Convergence**: 30-50 epochs typically
- **Best validation**: Usually around epoch 40-70

## 🔗 Integration

### With Existing Codebase
```python
# Import trained model
from models.hierarchical_unet import ProfessionalHierarchicalUNet

# Load checkpoint
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Use for inference
outputs = model(signal, static_params)
```

### With Legacy Code
The enhanced pipeline is compatible with existing ABR codebase and can be integrated gradually.

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the configuration options
3. Use debug mode to isolate problems
4. Check the evaluation metrics for training quality

The enhanced ABR training pipeline provides a robust, scalable foundation for training state-of-the-art ABR signal generation models with comprehensive multi-task learning capabilities. 