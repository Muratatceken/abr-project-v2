# Enhanced Loss Function Implementation

## Overview

The CVAE loss function system has been significantly enhanced with configurable peak loss types, time alignment capabilities, hyperparameter sweeps, and comprehensive TensorBoard logging. The enhanced system provides more flexibility for training ABR signal generation models while maintaining full backward compatibility.

## Key Features

### 1. **Configurable Peak Loss Types**
- **MAE (Mean Absolute Error)** - Default, more sensitive to small amplitude errors
- **MSE (Mean Squared Error)** - Traditional choice for regression
- **Huber Loss** - Robust to outliers with configurable delta parameter
- **Smooth L1 Loss** - Similar to Huber with delta=1.0

### 2. **Time Alignment Loss**
- **Warped MSE** - Allows small temporal shifts for best alignment
- **Soft DTW** - Differentiable dynamic time warping approximation
- **Correlation** - Cross-correlation based alignment

### 3. **Hyperparameter Sweeps**
- **Linear scheduling** - Gradual linear increase/decrease
- **Exponential scheduling** - Exponential growth/decay patterns
- **Configurable parameters** - Any loss weight can be scheduled

### 4. **Enhanced TensorBoard Logging**
- **Individual loss components** - Separate tracking for each component
- **Hyperparameter scheduling** - Real-time weight visualization
- **Organized metrics** - Hierarchical organization for easy analysis

## Implementation Details

### Enhanced Loss Function

```python
def enhanced_cvae_loss(
    recon_signal: torch.Tensor,
    target_signal: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    predicted_peaks: Optional[torch.Tensor] = None,
    target_peaks: Optional[torch.Tensor] = None,
    peak_mask: Optional[torch.Tensor] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    loss_config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
```

**Components:**
1. **Reconstruction Loss** - MSE between predicted and target signals
2. **KL Divergence** - Latent space regularization
3. **Peak Loss** - Configurable loss for peak values
4. **Alignment Loss** - Optional temporal alignment loss

### Configurable Peak Loss

```python
def peak_loss(
    predicted_peaks: torch.Tensor, 
    target_peaks: torch.Tensor, 
    peak_mask: torch.Tensor,
    loss_type: str = 'mae',  # 'mae', 'mse', 'huber', 'smooth_l1'
    huber_delta: float = 1.0
) -> torch.Tensor
```

**Features:**
- Handles missing peaks (NaN values) with masking
- Supports partial supervision
- Configurable loss function types

### Time Alignment Loss

```python
def time_alignment_loss(
    predicted_signal: torch.Tensor,
    target_signal: torch.Tensor,
    alignment_type: str = 'warped_mse',  # 'warped_mse', 'soft_dtw', 'correlation'
    max_warp: int = 10,
    temperature: float = 1.0
) -> torch.Tensor
```

**Alignment Types:**
- **Warped MSE**: Finds best alignment within max_warp window
- **Soft DTW**: Differentiable approximation of dynamic time warping
- **Correlation**: Cross-correlation based alignment

## Configuration

### Enhanced Loss Configuration

```json
{
    "training": {
        "use_enhanced_loss": true,
        "loss_weights": {
            "reconstruction": 1.0,
            "kl": 0.01,
            "peak": 1.0,
            "alignment": 0.1
        },
        "loss_config": {
            "peak_loss_type": "mae",
            "huber_delta": 1.0,
            "alignment_type": "warped_mse",
            "max_warp": 10,
            "temperature": 1.0,
            "use_alignment_loss": true
        },
        "hyperparameter_sweep": {
            "peak_loss_weight": {
                "enabled": true,
                "type": "linear",
                "start": 0.5,
                "end": 2.0
            }
        }
    }
}
```

### Peak Loss Types Comparison

| Loss Type | Formula | Characteristics | Best For |
|-----------|---------|-----------------|----------|
| **MAE** | `|pred - target|` | More sensitive to small errors | Peak amplitude precision |
| **MSE** | `(pred - target)²` | Emphasizes large errors | Traditional regression |
| **Huber** | Smooth transition between L1/L2 | Robust to outliers | Noisy peak data |
| **Smooth L1** | PyTorch's smooth L1 | Similar to Huber δ=1 | Stable training |

### Hyperparameter Sweep Types

#### Linear Scheduling
```python
weight(epoch) = start + (end - start) * (epoch / total_epochs)
```

#### Exponential Scheduling
```python
weight(epoch) = start * ((end / start) ** (epoch / total_epochs))
```

## Usage Examples

### 1. Basic Enhanced Loss Training

```python
# Configuration
config = {
    "use_enhanced_loss": True,
    "loss_weights": {"reconstruction": 1.0, "kl": 0.01, "peak": 1.5},
    "loss_config": {"peak_loss_type": "mae"}
}

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(signals, static_params)
        
        # Enhanced loss computation
        total_loss, loss_components = enhanced_cvae_loss(
            recon_signal=outputs[0],
            target_signal=signals,
            mu=outputs[2],
            logvar=outputs[3],
            predicted_peaks=outputs[4] if len(outputs) > 4 else None,
            target_peaks=batch.get('peaks'),
            peak_mask=batch.get('peak_mask'),
            loss_weights=config['loss_weights'],
            loss_config=config['loss_config']
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
```

### 2. Hyperparameter Sweep Configuration

```python
# Linear sweep from 0.5 to 2.0 over training
sweep_config = {
    "peak_loss_weight": {
        "enabled": True,
        "type": "linear",
        "start": 0.5,
        "end": 2.0
    }
}

# In trainer initialization
if 'peak_loss_weight' in sweep_config:
    config = sweep_config['peak_loss_weight']
    start_val = config['start']
    end_val = config['end']
    total_epochs = num_epochs
    
    self.peak_loss_weight_schedule = lambda epoch: (
        start_val + (end_val - start_val) * (epoch / total_epochs)
    )
```

### 3. TensorBoard Logging Integration

```python
def _log_metrics_to_tensorboard(self, metrics, epoch, phase):
    """Log enhanced metrics to TensorBoard."""
    for metric_name, metric_value in metrics.items():
        if 'enhanced_' in metric_name:
            # Enhanced loss components
            component = metric_name.replace('enhanced_', '').replace('_loss', '')
            self.writer.add_scalar(f'Enhanced_Loss/{component}_{phase}', 
                                 metric_value, epoch)
        elif metric_name in ['beta', 'peak_loss_weight']:
            # Hyperparameters
            self.writer.add_scalar(f'Hyperparameters/{metric_name}', 
                                 metric_value, epoch)
        # ... other metrics
```

## Performance Analysis

### Computational Overhead

| Component | Overhead | Memory Impact | Training Impact |
|-----------|----------|---------------|-----------------|
| **Peak Loss Types** | Minimal | None | Negligible |
| **Time Alignment** | Low-Medium | Low | Small |
| **Enhanced Logging** | Minimal | None | Negligible |
| **Hyperparameter Sweep** | None | None | None |

### Training Improvements

1. **Better Peak Prediction**: MAE default improves peak amplitude accuracy
2. **Temporal Alignment**: Reduces timing errors in generated signals
3. **Adaptive Weighting**: Dynamic loss balancing throughout training
4. **Better Monitoring**: Detailed component tracking for optimization

## Migration Guide

### From Traditional to Enhanced Loss

1. **Enable Enhanced Loss**:
   ```json
   "use_enhanced_loss": true
   ```

2. **Configure Loss Weights**:
   ```json
   "loss_weights": {
       "reconstruction": 1.0,
       "kl": 0.01,  // Use your current beta value
       "peak": 1.0,  // Use your current peak_loss_weight
       "alignment": 0.1
   }
   ```

3. **Set Peak Loss Type**:
   ```json
   "loss_config": {
       "peak_loss_type": "mae"  // Switch from MSE default
   }
   ```

4. **Optional Time Alignment**:
   ```json
   "loss_config": {
       "use_alignment_loss": true,
       "alignment_type": "warped_mse",
       "max_warp": 10
   }
   ```

## TensorBoard Visualization

### Loss Components View
- `Loss/Total_train` - Total training loss
- `Loss/reconstruction_train` - Signal reconstruction loss
- `Loss/kl_train` - KL divergence loss  
- `Loss/peak_train` - Peak prediction loss
- `Loss/alignment_train` - Time alignment loss

### Enhanced Components View
- `Enhanced_Loss/reconstruction_train` - Enhanced reconstruction component
- `Enhanced_Loss/kl_train` - Enhanced KL component
- `Enhanced_Loss/peak_train` - Enhanced peak component
- `Enhanced_Loss/alignment_train` - Enhanced alignment component

### Hyperparameters View
- `Hyperparameters/beta` - KL weight (β) over time
- `Hyperparameters/peak_loss_weight` - Peak loss weight over time

## Best Practices

### 1. **Peak Loss Type Selection**
- Use **MAE** for precise peak amplitude prediction
- Use **Huber** for noisy datasets with outliers
- Use **MSE** for backward compatibility

### 2. **Time Alignment Configuration**
- Start with `max_warp=5-10` for ABR signals
- Use **warped_mse** for simple temporal shifts
- Use **correlation** for complex phase alignment

### 3. **Hyperparameter Sweeps**
- Start conservatively with small ranges
- Use linear scheduling for initial experiments
- Monitor individual components to guide tuning

### 4. **Training Monitoring**
- Watch individual loss components for convergence
- Monitor peak loss weight schedule effectiveness
- Use TensorBoard's scalar comparison features

## Troubleshooting

### Common Issues

1. **Loss Divergence**
   - Reduce alignment loss weight
   - Check peak loss weight scheduling
   - Verify loss component balance

2. **Poor Peak Prediction**
   - Try different peak loss types
   - Increase peak loss weight
   - Check peak mask validity

3. **Temporal Misalignment**
   - Enable time alignment loss
   - Adjust max_warp parameter
   - Try different alignment types

### Debug Commands

```python
# Check loss components
for name, value in loss_components.items():
    print(f"{name}: {value:.6f}")

# Verify peak mask
print(f"Valid peaks: {peak_mask.sum().item()}/{peak_mask.numel()}")

# Monitor alignment loss
if 'alignment' in loss_components:
    print(f"Alignment loss: {loss_components['alignment']:.6f}")
```

## References

1. **Peak Loss Types**:
   - Huber, P. J. "Robust Estimation of a Location Parameter." 1964
   - Girshick, R. "Fast R-CNN." ICCV 2015 (Smooth L1 Loss)

2. **Time Alignment**:
   - Cuturi, M. & Blondel, M. "Soft-DTW: a Differentiable Loss Function for Time-Series." ICML 2017

3. **Loss Function Design**:
   - Kingma, D. P. & Welling, M. "Auto-Encoding Variational Bayes." ICLR 2014

---

**Implementation Status:** ✅ Complete and Tested  
**Backward Compatibility:** ✅ Maintained  
**TensorBoard Integration:** ✅ Full Support  
**Ready for Production:** ✅ Yes 