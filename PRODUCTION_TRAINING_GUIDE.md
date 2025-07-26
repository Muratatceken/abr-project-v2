# ABR Production Training Guide

This guide covers the complete production training system for ABR (Auditory Brainstem Response) signal analysis with 150-epoch training, comprehensive model saving, and visualization.

## 🚀 Quick Start

### Run Production Training (150 epochs)

```bash
python run_production_training.py
```

### Run Test Training (100 epochs)

```bash
python test_production_training.py
```

### Custom Training

```bash
python run_production_training.py \
    --epochs 150 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --save-every 10 \
    --plot-every 5 \
    --experiment-name "my_experiment"
```

## 📁 Directory Structure

After training, you'll have the following structure:

```
abr-project-v2/
├── checkpoints/production/          # Model checkpoints
│   ├── best_f1.pth                 # Best F1 score model
│   ├── best_loss.pth               # Best validation loss model
│   ├── final_model.pth             # Final trained model
│   ├── milestone_epoch_50.pth      # Milestone checkpoints
│   ├── milestone_epoch_100.pth
│   ├── milestone_epoch_150.pth
│   ├── checkpoint_epoch_10.pth     # Regular checkpoints
│   ├── checkpoint_epoch_20.pth
│   └── ...
├── plots/production/               # Training visualizations
│   ├── {experiment}_training_curves.png
│   ├── {experiment}_loss_components.png
│   ├── {experiment}_validation_metrics.png
│   ├── {experiment}_training_summary.png
│   └── {experiment}_metrics.csv
├── logs/production/                # Training logs
│   └── {experiment}.log
├── runs/production/                # TensorBoard logs
└── outputs/production/             # Other outputs
```

## 🎛️ Configuration

### Main Configuration File: `training/config_production.yaml`

Key settings for production training:

```yaml
# Core training parameters
num_epochs: 150
learning_rate: 0.0001
batch_size: 32
gradient_accumulation_steps: 2

# Model saving
save_every: 10  # Save checkpoint every 10 epochs
save_criteria:
  best_loss: true
  best_f1: true
  milestone_epochs: [50, 100, 150]

# Visualization
visualization:
  enabled: true
  plot_every: 5  # Create plots every 5 epochs
  
# Fast validation (speeds up training)
validation:
  fast_mode: true
  skip_generation: true
  full_validation_every: 10
```

## 📊 Model Saving

The system saves models in multiple formats and criteria:

### Automatic Saves

- **Best F1 Model**: `best_f1.pth` - Saved when validation F1 improves
- **Best Loss Model**: `best_loss.pth` - Saved when validation loss improves
- **Final Model**: `final_model.pth` - Saved at training completion
- **Regular Checkpoints**: `checkpoint_epoch_N.pth` - Saved every N epochs
- **Milestone Checkpoints**: `milestone_epoch_N.pth` - Saved at specific epochs

### Checkpoint Contents

Each checkpoint contains:

```python
{
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),  # For mixed precision
    'best_val_loss': best_validation_loss,
    'best_val_f1': best_validation_f1,
    'config': training_configuration,
    'metrics': current_validation_metrics,
    'training_time': training_start_time
}
```

### Model Summary

Each model save includes a human-readable summary (`*_summary.txt`):

```
================================================================================
ABR MODEL SUMMARY
================================================================================

Epoch: 150
Model Parameters: 46,371,139
Trainable Parameters: 46,371,139

Validation Metrics:
  direct_total_loss: 2.345678
  direct_f1_macro: 0.876543

Best Metrics:
  Best Val Loss: 2.123456
  Best Val F1: 0.892345

Training Configuration:
  Learning Rate: 0.0001
  Batch Size: 32
  Optimizer: adamw
  Scheduler: cosine_annealing_warm_restarts
```

## 📈 Visualization System

### Automatic Plots Created

1. **Training Curves** (`*_training_curves.png`)

   - Training and validation loss over time
   - Validation F1 score progression
   - Learning rate schedule
   - Recent performance (last 20 epochs)
2. **Loss Components** (`*_loss_components.png`)

   - Individual loss components (signal, classification, peak, etc.)
   - Training vs validation comparison
3. **Validation Metrics** (`*_validation_metrics.png`)

   - F1 scores and other validation metrics over time
4. **Training Summary** (`*_training_summary.png`)

   - Comprehensive dashboard with all key metrics
   - Training statistics and progress
5. **Metrics CSV** (`*_metrics.csv`)

   - All training metrics in CSV format for analysis

### Visualization Features

- **High-resolution plots** (300 DPI) for publication quality
- **Automatic styling** with seaborn themes
- **Progress tracking** with epoch-by-epoch metrics
- **Statistical summaries** with best performance tracking

## 🔧 Advanced Features

### Fast Validation

- **Skip expensive DDIM sampling** during training for speed
- **Direct forward pass only** for quick validation
- **Full validation every N epochs** for comprehensive evaluation

### Curriculum Learning

- **Gradual weight adjustment** over first 30 epochs
- **Signal reconstruction** emphasis early in training
- **Classification** emphasis later in training

### Mixed Precision Training

- **Automatic Mixed Precision (AMP)** for faster training
- **Memory optimization** with periodic cache clearing
- **Gradient scaling** for numerical stability

### Learning Rate Scheduling

- **Cosine Annealing with Warm Restarts**
- **Restarts every 25 epochs** with T_mult=2
- **Minimum learning rate**: 1e-7

### Early Stopping

- **Patience**: 25 epochs without improvement
- **Monitors**: Validation F1 score
- **Automatic termination** when no progress

## 🏃‍♂️ Performance Optimizations

### Speed Optimizations

```yaml
# In config file
use_amp: true                    # Mixed precision training
use_memory_efficient_attention: true
clear_cache_every: 100          # Clear CUDA cache periodically
pin_memory: true                # Faster data loading
persistent_workers: true        # Keep data workers alive
prefetch_factor: 4              # Prefetch batches
```

### Memory Optimizations

- **Gradient accumulation** for larger effective batch sizes
- **Memory-efficient attention** mechanisms
- **Periodic CUDA cache clearing**
- **Optimized data loading** with persistent workers

## 📊 Monitoring and Logging

### TensorBoard Integration

```bash
# View training progress in real-time
tensorboard --logdir runs/production
```

### Log Files

- **Comprehensive logging** to `logs/production/{experiment}.log`
- **Console output** with progress bars and metrics
- **Error handling** with detailed stack traces

### Metrics Tracking

- **Training loss** (total and components)
- **Validation loss** (with fast/full validation modes)
- **F1 scores** (macro, weighted, per-class)
- **Learning rate** progression
- **Training time** and performance statistics

## 🔄 Resuming Training

### Resume from Checkpoint

```bash
python run_production_training.py --resume checkpoints/production/checkpoint_epoch_50.pth
```

### Validation Only

```bash
python run_production_training.py --validate-only --resume checkpoints/production/best_f1.pth
```

## 🎯 Model Evaluation

### Load Trained Model

```python
import torch
from models.hierarchical_unet import ProfessionalHierarchicalUNet

# Load checkpoint
checkpoint = torch.load('checkpoints/production/best_f1.pth')
model = ProfessionalHierarchicalUNet(config=checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluation mode
model.eval()
```

### Best Model Selection

- **best_f1.pth**: Best for classification tasks
- **best_loss.pth**: Best overall performance
- **final_model.pth**: Latest trained model

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory**

   ```bash
   # Reduce batch size
   python run_production_training.py --batch-size 16
   ```
2. **Slow Training**

   ```bash
   # Reduce workers if I/O bound
   python run_production_training.py --num-workers 4
   ```
3. **Visualization Errors**

   ```bash
   # Disable plots if issues
   python run_production_training.py --no-plots
   ```

### Performance Tips

- **Use CUDA** if available for 10-20x speedup
- **Monitor GPU memory** usage during training
- **Adjust batch size** based on available memory
- **Use fast validation** for quicker iterations

## 📋 Training Checklist

Before starting production training:

- [ ] **Data prepared**: `data/processed/ultimate_dataset.pkl` exists
- [ ] **Configuration set**: Review `training/config_production.yaml`
- [ ] **Directories created**: Ensure write permissions
- [ ] **Resources available**: Sufficient disk space and memory
- [ ] **Monitoring ready**: TensorBoard or logging setup

## 🎉 Expected Results

After 150 epochs of training, you should expect:

- **Training time**: 4-8 hours (depending on hardware)
- **Model size**: ~180MB (46M parameters)
- **Best F1 score**: 0.85-0.95 (depending on data quality)
- **Convergence**: Usually by epoch 100-120
- **Checkpoints**: ~15-20 saved models
- **Visualizations**: 4-5 plot types with progress tracking

## 📞 Support

For issues or questions:

1. Check the training logs in `logs/production/`
2. Review TensorBoard plots for training progress
3. Examine the configuration in `training/config_production.yaml`
4. Verify data loading with the dataset examples

---

**Happy Training!** 🚀
