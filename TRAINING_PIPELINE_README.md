# ABR Training Pipeline Documentation

A comprehensive training, evaluation, and inference pipeline for the ABR Hierarchical U-Net model with multi-task learning and diffusion-based signal generation.

## 🏗️ **Pipeline Overview**

This training pipeline provides a complete, production-ready solution for training ABR models with:

- **Multi-task learning** (signal, peaks, classification, threshold)
- **Diffusion-based training** with noise scheduling
- **Professional monitoring** (TensorBoard, Weights & Biases)
- **Comprehensive evaluation** with clinical metrics
- **Flexible inference** for clinical deployment

## 📁 **Project Structure**

```
abr-project-v2/
├── configs/
│   └── config.yaml              # Main configuration file
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Core trainer class
│   ├── config_loader.py         # Configuration management
│   └── lr_scheduler.py          # Learning rate scheduling
├── evaluation/
│   ├── __init__.py
│   └── metrics.py               # Comprehensive metrics
├── inference/
│   └── __init__.py
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation script
├── inference.py                 # Inference script
├── run_training_demo.py         # Demo script
└── data/
    └── processed/
        └── ultimate_dataset_with_clinical_thresholds.pkl
```

## 🚀 **Quick Start**

### 1. **Demo Training**
Run a fast training demo to test the pipeline:

```bash
python run_training_demo.py
```

This will:
- Run 3 epochs of training with small batch size
- Evaluate the trained model
- Demonstrate inference capabilities
- Generate outputs in `outputs/` directory

### 2. **Full Training**
For production training:

```bash
python train.py --config configs/config.yaml --experiment production_run
```

### 3. **Custom Training**
With command-line overrides:

```bash
python train.py \
    --config configs/config.yaml \
    --experiment my_experiment \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 1e-4
```

## 📝 **Training Scripts**

### `train.py` - Main Training Script

**Features:**
- Full configuration management with OmegaConf
- Command-line overrides for key parameters
- Automatic device detection (CUDA/CPU)
- Mixed precision training support
- Professional logging and monitoring
- Experiment management with organized outputs

**Usage Examples:**

```bash
# Basic training
python train.py --config configs/config.yaml

# Custom experiment
python train.py --config configs/config.yaml --experiment ablation_study

# Resume from checkpoint
python train.py --config configs/config.yaml --resume checkpoints/latest_model.pt

# Override parameters
python train.py --config configs/config.yaml --batch_size 64 --epochs 100

# Fast development run (2 epochs, small batch)
python train.py --config configs/config.yaml --fast_dev_run

# Overfit single batch for debugging
python train.py --config configs/config.yaml --overfit_batch
```

**Key Arguments:**
- `--config`: Configuration file path
- `--experiment`: Experiment name for organized outputs
- `--resume`: Resume from checkpoint
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size override
- `--learning_rate`: Learning rate override
- `--device`: Force device (cuda/cpu)
- `--mixed_precision`: Enable mixed precision
- `--fast_dev_run`: Quick test run
- `--overfit_batch`: Debug mode

### `evaluate.py` - Model Evaluation

**Features:**
- Comprehensive metric computation
- Test set evaluation
- Sample generation capability
- Detailed reporting (JSON + text)
- Clinical relevance metrics

**Usage Examples:**

```bash
# Basic evaluation
python evaluate.py --checkpoint checkpoints/best_model.pt

# Custom config and output
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/config.yaml \
    --output_dir results/final_evaluation

# Generate samples during evaluation
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --generate_samples 100
```

### `inference.py` - Clinical Inference

**Features:**
- Single sample prediction
- Batch processing
- Sample generation
- Confidence intervals
- Multiple output formats (JSON, NPZ)

**Usage Examples:**

```bash
# Single patient prediction
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --age 35 --intensity 80 --rate 30 --fmp 0.8 \
    --confidence

# Batch processing
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input_file data/test_batch.npz

# Generate synthetic data
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --generate 100 \
    --output_dir results/synthetic_data
```

## ⚙️ **Configuration Management**

### Main Configuration (`configs/config.yaml`)

The configuration is organized into logical sections:

```yaml
# Project settings
project:
  name: "ABR_HierarchicalUNet_Diffusion"
  version: "1.0.0"

# Data configuration
data:
  dataset_path: "data/processed/ultimate_dataset_with_clinical_thresholds.pkl"
  signal_length: 200
  static_dim: 4
  n_classes: 5
  splits:
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15
  dataloader:
    batch_size: 32
    num_workers: 4

# Model architecture
model:
  architecture:
    base_channels: 64
    n_levels: 4
    dropout: 0.1
    encoder:
      n_s4_layers: 2
      d_state: 64
    decoder:
      n_transformer_layers: 2
      n_heads: 8

# Training parameters
training:
  epochs: 200
  optimizer:
    type: "adamw"
    learning_rate: 1e-4
    weight_decay: 1e-5
  scheduler:
    type: "cosine_annealing_warm_restarts"
    T_0: 50
    warmup_epochs: 10
  gradient_clip: 1.0

# Loss configuration
loss:
  weights:
    diffusion: 1.0
    peak_exist: 0.5
    peak_latency: 1.0
    classification: 1.0
    threshold: 0.8

# Hardware settings
hardware:
  device: "auto"
  mixed_precision: true

# Logging
logging:
  use_tensorboard: true
  use_wandb: false
```

### Environment Variable Overrides

You can override configuration using environment variables:

```bash
export ABR_BATCH_SIZE=64
export ABR_LEARNING_RATE=5e-5
export ABR_EPOCHS=300
python train.py --config configs/config.yaml
```

## 🎯 **Core Components**

### ABRTrainer Class (`training/trainer.py`)

**Key Features:**
- Multi-task loss handling with proper weighting
- Diffusion training with noise scheduling
- Mixed precision training (AMP)
- Gradient clipping and accumulation
- Advanced learning rate scheduling
- Comprehensive monitoring and logging
- Automatic checkpointing with best model saving
- Early stopping with patience
- Professional error handling and recovery

**Training Loop:**
```python
# Initialize trainer
trainer = ABRTrainer(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device
)

# Start training
trainer.train()
```

### Configuration Loader (`training/config_loader.py`)

**Features:**
- OmegaConf-based configuration management
- Environment variable integration
- Configuration validation and post-processing
- Experiment-specific configurations
- Configuration versioning and hashing

### Learning Rate Scheduling (`training/lr_scheduler.py`)

**Available Schedulers:**
- Cosine Annealing with Warm Restarts
- OneCycle Learning Rate
- Reduce on Plateau
- Custom ABR-optimized scheduler
- Warmup support for all schedulers

## 📊 **Evaluation Metrics**

### Signal Quality Metrics
- **MSE/MAE**: Basic reconstruction error
- **Correlation**: Signal similarity
- **SNR**: Signal-to-noise ratio
- **Spectral Similarity**: Frequency domain comparison
- **Morphological Similarity**: Envelope correlation

### Peak Prediction Metrics
- **Existence Accuracy**: Peak detection accuracy
- **Existence F1**: F1 score for peak existence
- **Latency MAE/RMSE**: Peak timing accuracy
- **Amplitude MAE/RMSE**: Peak magnitude accuracy

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Scores**: Macro and weighted F1
- **Precision/Recall**: Per-class performance
- **AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed class performance

### Clinical Metrics
- **Clinical Concordance**: Severity level agreement
- **Diagnostic Agreement**: Normal vs. hearing loss
- **Threshold Accuracy**: Within ±5dB, ±10dB, ±15dB
- **Age-stratified Performance**: Performance by age group

## 🔧 **Advanced Features**

### Mixed Precision Training
Automatic mixed precision (AMP) support for faster training:

```yaml
hardware:
  mixed_precision: true
```

### Gradient Accumulation
For effective large batch training on limited memory:

```yaml
training:
  accumulation_steps: 4  # Effective batch size = batch_size * 4
```

### Multi-GPU Support
Planned for future implementation with DDP support.

### Experiment Management
Organized experiment tracking:

```bash
python train.py --experiment exp_name
# Creates: 
#   checkpoints/exp_name/
#   logs/exp_name/
#   outputs/exp_name/
```

### Monitoring Integration
- **TensorBoard**: Real-time training monitoring
- **Weights & Biases**: Advanced experiment tracking (optional)
- **File Logging**: Persistent logs for debugging

## 📈 **Training Best Practices**

### 1. **Start with Fast Development Run**
```bash
python train.py --config configs/config.yaml --fast_dev_run
```

### 2. **Use Overfitting for Debugging**
```bash
python train.py --config configs/config.yaml --overfit_batch
```

### 3. **Monitor Training Progress**
```bash
tensorboard --logdir logs/
```

### 4. **Regular Checkpointing**
The trainer automatically saves:
- `best_model.pt`: Best validation performance
- `latest_model.pt`: Most recent checkpoint
- `checkpoint_epoch_N.pt`: Periodic checkpoints

### 5. **Resume Training**
```bash
python train.py --config configs/config.yaml --resume checkpoints/latest_model.pt
```

## 🏥 **Clinical Deployment**

### Single Patient Inference
```python
from inference import ABRInference

# Initialize inference
inference = ABRInference("checkpoints/best_model.pt")

# Predict for patient
results = inference.predict_single(
    age=45, intensity=80, stimulus_rate=30, fmp=0.75,
    return_confidence=True
)

print(f"Predicted hearing loss: {results['classification']['predicted_category']}")
print(f"Threshold: {results['threshold']['value']:.1f} dB nHL")
```

### Batch Processing
```python
# Process multiple patients
import numpy as np

patients = np.array([
    [45, 80, 30, 0.75],  # age, intensity, rate, fmp
    [65, 90, 20, 0.60],
    [30, 70, 40, 0.85]
])

results = inference.predict_batch(patients)
```

## 🚨 **Troubleshooting**

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Enable gradient accumulation
   - Use mixed precision training

2. **Slow Training**
   - Increase `num_workers` for data loading
   - Enable mixed precision
   - Use appropriate batch size

3. **Poor Convergence**
   - Check learning rate scheduling
   - Verify loss weights
   - Ensure proper data normalization

4. **Evaluation Errors**
   - Verify checkpoint compatibility
   - Check data preprocessing consistency
   - Ensure proper device handling

### Debug Mode
```bash
python train.py --config configs/config.yaml --log_level DEBUG
```

## 📋 **TODO / Future Enhancements**

- [ ] Multi-GPU training with DDP
- [ ] Model quantization for deployment
- [ ] Real-time inference optimization
- [ ] Advanced visualization tools
- [ ] Automated hyperparameter optimization
- [ ] Clinical validation pipeline
- [ ] Model interpretation tools
- [ ] Export to ONNX/TensorRT

## 🎉 **Success Criteria**

The training pipeline is working correctly when:

✅ **Training completes without errors**  
✅ **Validation loss decreases over time**  
✅ **All metrics are computed successfully**  
✅ **Checkpoints are saved properly**  
✅ **Evaluation produces reasonable results**  
✅ **Inference works for single samples and batches**  
✅ **Generated samples look realistic**  

## 📞 **Support**

For issues or questions:
1. Check the troubleshooting section
2. Review configuration settings
3. Enable debug logging
4. Check training logs in `logs/` directory

---

**This training pipeline provides a solid foundation for ABR model development with professional-grade features for research and clinical deployment.**