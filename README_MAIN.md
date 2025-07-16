# ABR CVAE Main Pipeline

A unified entry point for training, evaluating, and running inference with Conditional Variational Autoencoder (CVAE) models on ABR (Auditory Brainstem Response) signals.

## Overview

The `main.py` script provides a comprehensive pipeline that integrates all components of the ABR CVAE project:

- **Data preprocessing and loading**
- **Model training with advanced features**
- **Comprehensive evaluation and visualization**
- **Inference and sample generation**
- **Checkpoint management and logging**

## Quick Start

### Basic Usage

```bash
# Train a new model
python main.py --mode train

# Evaluate a trained model
python main.py --mode evaluate --checkpoint_path checkpoints/best_model.pth

# Generate new samples
python main.py --mode inference --checkpoint_path checkpoints/best_model.pth
```

### Command Line Arguments

- `--config_path`: Path to configuration file (default: `configs/default_config.json`)
- `--mode`: Operation mode (`train`, `evaluate`, `inference`)
- `--checkpoint_path`: Path to model checkpoint (for resuming training or evaluation)
- `--verbose`: Enable verbose logging

## Configuration System

The pipeline uses a comprehensive JSON configuration system. Key sections include:

### Project Configuration
```json
{
  "project": {
    "name": "ABR_CVAE",
    "description": "Conditional Variational Autoencoder for ABR Signal Generation",
    "version": "1.0.0"
  }
}
```

### Data Configuration
```json
{
  "data": {
    "raw_data_path": "../abr_dataset.xlsx",
    "processed_data_path": "data/processed/processed_data.pkl",
    "preprocessing": {
      "save_transformers": true,
      "verbose": true
    },
    "dataloader": {
      "batch_size": 32,
      "val_split": 0.2,
      "test_split": 0.1,
      "num_workers": 4
    }
  }
}
```

### Model Configuration
```json
{
  "model": {
    "architecture": {
      "latent_dim": 32,
      "predict_peaks": true,
      "num_peaks": 6
    }
  }
}
```

### Training Configuration
```json
{
  "training": {
    "optimizer": {
      "type": "adam",
      "learning_rate": 1e-3,
      "weight_decay": 1e-5
    },
    "epochs": 100,
    "early_stopping": {
      "patience": 15,
      "min_delta": 1e-4
    }
  }
}
```

## Pipeline Modes

### 1. Training Mode (`--mode train`)

**Features:**
- Automatic data preprocessing if needed
- Flexible model architecture configuration
- Advanced training features:
  - Early stopping
  - Learning rate scheduling
  - Beta annealing for KL divergence
  - Gradient clipping
  - Checkpoint saving
- TensorBoard integration
- Comprehensive logging

**Workflow:**
1. Load configuration
2. Setup device and reproducibility
3. Check/preprocess data
4. Create dataloaders
5. Initialize model and optimizer
6. Load checkpoint (if resuming)
7. Create trainer with advanced features
8. Start training loop
9. Save checkpoints and logs

**Example:**
```bash
# Train new model
python main.py --mode train --config_path configs/my_config.json

# Resume training
python main.py --mode train --checkpoint_path checkpoints/epoch_050.pth
```

### 2. Evaluation Mode (`--mode evaluate`)

**Features:**
- Comprehensive quantitative metrics
- Professional visualizations
- Latent space analysis
- Sample generation evaluation
- Detailed reporting

**Metrics:**
- **Reconstruction**: MSE, MAE, RMSE, correlation, R-squared
- **Peak Prediction**: Peak MAE, correlation, presence accuracy
- **Advanced**: DTW distance, SSIM, latent space quality

**Visualizations:**
- Signal reconstruction comparisons
- Peak prediction scatter plots
- Latent space t-SNE/PCA plots
- Generated sample variability

**Example:**
```bash
python main.py --mode evaluate --checkpoint_path checkpoints/best_model.pth
```

### 3. Inference Mode (`--mode inference`)

**Features:**
- Generate new ABR signals
- Configurable static parameters
- Batch generation for efficiency
- Multiple output formats

**Workflow:**
1. Load trained model
2. Generate random static parameters
3. Generate ABR signals in batches
4. Save generated samples and parameters
5. Create generation summary

**Example:**
```bash
python main.py --mode inference --checkpoint_path checkpoints/best_model.pth
```

## Key Features

### 🔧 **Automatic Data Preprocessing**
- Detects if processed data exists
- Automatically preprocesses raw data if needed
- Saves preprocessed data and transformers
- Handles both CSV and Excel input formats

### 🎯 **Advanced Training Features**
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Beta Annealing**: Gradual KL divergence weighting
- **Gradient Clipping**: Stable training
- **Checkpoint Management**: Automatic saving and loading

### 📊 **Comprehensive Evaluation**
- **Quantitative Metrics**: Multiple reconstruction and peak metrics
- **Advanced Metrics**: DTW distance, SSIM
- **Visualizations**: Professional plots and charts
- **Reporting**: JSON summaries and visual reports

### 🔬 **Flexible Model Architecture**
- Configurable latent dimensions
- Optional peak prediction
- Customizable encoder/decoder architectures
- Multiple initialization strategies

### 🛠 **Robust Infrastructure**
- **Device Management**: Auto-detection of GPU/CPU
- **Reproducibility**: Seed management for consistent results
- **Logging**: Comprehensive logging with multiple levels
- **Error Handling**: Graceful error handling and recovery

### 📁 **Output Management**
- Organized output directories
- Automatic directory creation
- Configurable save paths
- Multiple output formats

## File Structure

```
abr_cvae_project/
├── main.py                     # Main entry point
├── configs/
│   └── default_config.json     # Default configuration
├── models/                     # Model definitions
├── training/                   # Training utilities
├── evaluation/                 # Evaluation pipeline
├── utils/                      # Utility functions
├── data/                       # Data storage
├── checkpoints/                # Model checkpoints
├── outputs/                    # Generated outputs
└── logs/                       # Log files
```

## Configuration Examples

### Minimal Training Configuration
```json
{
  "data": {
    "raw_data_path": "data/abr_dataset.xlsx"
  },
  "model": {
    "architecture": {
      "latent_dim": 16
    }
  },
  "training": {
    "epochs": 50
  }
}
```

### Advanced Training Configuration
```json
{
  "training": {
    "optimizer": {
      "type": "adamw",
      "learning_rate": 1e-3,
      "weight_decay": 1e-4
    },
    "scheduler": {
      "type": "reduce_on_plateau",
      "patience": 10,
      "factor": 0.5
    },
    "loss": {
      "beta_scheduler": {
        "type": "cosine",
        "max_beta": 1.0,
        "warmup_epochs": 20
      }
    },
    "early_stopping": {
      "patience": 20,
      "min_delta": 1e-5
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **Missing Data File**
   ```
   Error: Raw data file not found
   Solution: Ensure the raw data path in config is correct
   ```

2. **CUDA Out of Memory**
   ```
   Error: CUDA out of memory
   Solution: Reduce batch_size in configuration
   ```

3. **Import Errors**
   ```
   Error: ModuleNotFoundError
   Solution: Install dependencies with pip install -r requirements.txt
   ```

### Performance Tips

- **GPU Usage**: Set `device.type` to "cuda" for GPU acceleration
- **Memory Optimization**: Reduce `batch_size` and `num_workers`
- **Faster Training**: Increase `batch_size` if memory allows
- **Reproducibility**: Set `reproducibility.deterministic` to true

## Advanced Usage

### Custom Configuration
```python
# Create custom configuration
config = {
    "model": {"architecture": {"latent_dim": 64}},
    "training": {"epochs": 200}
}

# Save to file
with open('custom_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Use with main.py
# python main.py --config_path custom_config.json
```

### Programmatic Usage
```python
from main import train_mode, evaluate_mode, inference_mode, load_config

# Load configuration
config = load_config('configs/default_config.json')

# Run training
train_mode(config)

# Run evaluation
evaluate_mode(config, checkpoint_path='checkpoints/best_model.pth')

# Run inference
inference_mode(config, checkpoint_path='checkpoints/best_model.pth')
```

## Integration with Other Tools

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir=runs

# View training progress at http://localhost:6006
```

### Jupyter Notebooks
```python
# Use in Jupyter notebooks
import sys
sys.path.append('path/to/abr_cvae_project')

from main import *
config = load_config('configs/default_config.json')
# ... use pipeline functions
```

## Best Practices

1. **Configuration Management**: Use version control for configuration files
2. **Experiment Tracking**: Use descriptive checkpoint names
3. **Data Backup**: Keep backups of processed data
4. **Resource Monitoring**: Monitor GPU/CPU usage during training
5. **Reproducibility**: Always set random seeds for consistent results

## Contributing

To extend the pipeline:

1. **Add New Modes**: Extend the mode handling in `main()`
2. **Custom Metrics**: Add to evaluation utilities
3. **New Optimizers**: Extend `create_optimizer()`
4. **Configuration Options**: Update default configuration schema

## License

This pipeline is part of the ABR CVAE project and follows the same license terms. 