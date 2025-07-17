# Enhanced Beta Scheduler Implementation

## Overview

The CVAE training system has been enhanced with sophisticated β (beta) annealing schedules that provide better control over KL divergence regularization during training. This implementation includes cyclical scheduling, warm restarts, and hierarchical scheduling specifically designed for advanced VAE training dynamics.

## New Scheduler Types

### 1. **Cyclical Beta Scheduler** (`CyclicalBetaScheduler`)

Implements periodic cycling of β values between minimum and maximum thresholds, allowing for exploration of different regularization strengths.

#### Features:
- **Wave Types**: Cosine and Triangle wave patterns
- **Configurable Cycles**: Adjustable cycle length and amplitude
- **Periodic Exploration**: Prevents getting stuck in local minima
- **Balanced Training**: Alternates between reconstruction focus and regularization

#### Configuration:
```json
{
    "beta_scheduler_type": "cyclical",
    "beta_scheduler_params": {
        "cycle_length": 20,
        "min_beta": 0.0,
        "max_beta": 1.0,
        "wave_type": "cosine"
    }
}
```

#### Wave Type Comparison:
- **Cosine Wave**: Smooth transitions, gradual changes
- **Triangle Wave**: Linear transitions, more abrupt changes

### 2. **Warm Restart Beta Scheduler** (`WarmRestartBetaScheduler`)

Implements periodic "warm restarts" by resetting β to 0 at specified intervals, allowing the model to periodically refocus on reconstruction quality.

#### Features:
- **Periodic Resets**: β returns to 0 at restart intervals
- **Adaptive Intervals**: Restart intervals can increase with multiplier
- **Base Scheduler**: Wraps any base scheduler (linear, cosine, exponential)
- **Escape Mechanism**: Helps escape local minima through regularization cycling

#### Configuration:
```json
{
    "beta_scheduler_type": "warm_restart",
    "beta_scheduler_params": {
        "base_scheduler_type": "cosine",
        "base_scheduler_kwargs": {
            "max_beta": 1.0,
            "warmup_epochs": 20
        },
        "restart_interval": 40,
        "restart_multiplier": 1.2
    }
}
```

#### Restart Schedule Example:
- First restart: Epoch 40
- Second restart: Epoch 88 (40 + 40×1.2)
- Third restart: Epoch 146 (88 + 48×1.2)

### 3. **Hierarchical Beta Scheduler** (`HierarchicalBetaScheduler`)

Designed for hierarchical CVAE models, providing independent scheduling for global and local KL weights.

#### Features:
- **Separate Scheduling**: Independent control of global and local KL weights
- **Mixed Strategies**: Different scheduler types for each latent space
- **Fine-Grained Control**: Optimal regularization for different feature levels
- **Hierarchical Optimization**: Better training dynamics for multi-level models

#### Configuration:
```json
{
    "beta_scheduler_type": "hierarchical",
    "beta_scheduler_params": {
        "global_scheduler": {
            "type": "cosine",
            "max_beta": 0.01,
            "warmup_epochs": 10
        },
        "local_scheduler": {
            "type": "cyclical",
            "cycle_length": 25,
            "min_beta": 0.0,
            "max_beta": 0.02,
            "wave_type": "cosine"
        }
    }
}
```

## Enhanced Training Integration

### 1. **Configuration Support**

The training system now supports the `beta_scheduler_type` parameter with comprehensive configuration options:

```python
# In training config
config = {
    "beta_scheduler_type": "cyclical",  # or "warm_restart", "hierarchical"
    "beta_scheduler_params": {...},     # Type-specific parameters
    "use_hierarchical_model": False     # Enable for hierarchical CVAE
}
```

### 2. **Enhanced Logging**

#### Batch-Level Logging:
- **KL Loss per Batch**: `Batch/KL_Loss`
- **Beta Values per Batch**: `Batch/Beta`, `Batch/Global_KL_Weight`, `Batch/Local_KL_Weight`
- **Total Loss per Batch**: `Batch/Total_Loss`

#### Epoch-Level Logging:
- **Beta Schedule Tracking**: `Beta_Schedule/beta`, `Beta_Schedule/global_kl_weight`, `Beta_Schedule/local_kl_weight`
- **Restart Information**: `Beta_Schedule/Restart/*` (for warm restart schedulers)
- **Hierarchical Comparison**: `Beta_Schedule/Hierarchical_Comparison`

#### TensorBoard Organization:
```
Beta_Schedule/
├── beta                          # Main beta value
├── global_kl_weight             # Global KL weight (hierarchical)
├── local_kl_weight              # Local KL weight (hierarchical)
├── Restart/
│   ├── last_restart_epoch       # Most recent restart
│   ├── epochs_since_restart     # Time since last restart
│   └── is_restart_epoch         # Boolean restart marker
└── Hierarchical_Comparison      # Side-by-side comparison
```

### 3. **Training Loop Integration**

The enhanced training loop automatically handles different scheduler types:

```python
# Automatic scheduler detection and beta computation
if self.use_hierarchical_model:
    beta_values = self.beta_scheduler(epoch)
    global_kl_weight = beta_values['global_kl_weight']
    local_kl_weight = beta_values['local_kl_weight']
else:
    beta = self.beta_scheduler(epoch)

# Enhanced logging with restart information
if hasattr(self.beta_scheduler, 'get_restart_info'):
    restart_info = self.beta_scheduler.get_restart_info(epoch)
    # Log restart status and timing
```

## Usage Examples

### 1. **Cyclical Training for Better Exploration**

```python
# Configuration for cyclical beta scheduling
config = {
    "beta_scheduler_type": "cyclical",
    "beta_scheduler_params": {
        "cycle_length": 25,      # 25-epoch cycles
        "min_beta": 0.0,         # Full reconstruction focus
        "max_beta": 1.0,         # Full regularization
        "wave_type": "cosine"    # Smooth transitions
    }
}

# Benefits:
# - Periodic exploration of reconstruction vs regularization trade-off
# - Prevents optimization from getting stuck in local minima
# - Better final model quality through diverse training phases
```

### 2. **Warm Restart for Robust Training**

```python
# Configuration for warm restart scheduling
config = {
    "beta_scheduler_type": "warm_restart",
    "beta_scheduler_params": {
        "base_scheduler_type": "cosine",
        "base_scheduler_kwargs": {
            "max_beta": 1.0,
            "warmup_epochs": 15
        },
        "restart_interval": 50,     # Restart every 50 epochs initially
        "restart_multiplier": 1.3   # Increase interval by 30% each restart
    }
}

# Benefits:
# - Periodic refocus on reconstruction quality
# - Escape from local minima through regularization resets
# - Adaptive restart intervals for different training phases
```

### 3. **Hierarchical Scheduling for Multi-Level Models**

```python
# Configuration for hierarchical CVAE training
config = {
    "use_hierarchical_model": True,
    "beta_scheduler_type": "hierarchical",
    "beta_scheduler_params": {
        "global_scheduler": {
            "type": "linear",
            "max_beta": 0.005,      # Lower regularization for global features
            "warmup_epochs": 10
        },
        "local_scheduler": {
            "type": "cyclical",
            "cycle_length": 20,
            "min_beta": 0.0,
            "max_beta": 0.015,      # Higher regularization for local details
            "wave_type": "triangle"
        }
    }
}

# Benefits:
# - Independent optimization of global and local latent spaces
# - Different regularization strategies for different feature levels
# - Better disentanglement and representation learning
```

## Performance Analysis

### Training Dynamics Comparison

| Scheduler Type | Exploration | Stability | Convergence | Best Use Case |
|---------------|-------------|-----------|-------------|---------------|
| **Linear** | Low | High | Fast | Baseline, stable training |
| **Cosine** | Medium | High | Smooth | General purpose, smooth annealing |
| **Cyclical** | High | Medium | Variable | Complex datasets, exploration |
| **Warm Restart** | Very High | Medium | Robust | Difficult optimization landscapes |
| **Hierarchical** | Customizable | High | Optimal | Multi-level models, hierarchical CVAE |

### Convergence Characteristics

#### Cyclical Scheduling:
- **Oscillating Loss**: Loss oscillates with β cycles
- **Better Minima**: Often finds better final solutions
- **Longer Training**: May require more epochs for convergence

#### Warm Restart Scheduling:
- **Sawtooth Pattern**: Sharp drops at restart points
- **Robust Convergence**: Less likely to get stuck
- **Adaptive Dynamics**: Self-adjusting restart intervals

#### Hierarchical Scheduling:
- **Independent Curves**: Separate convergence for global/local
- **Balanced Regularization**: Optimal for each latent space
- **Superior Disentanglement**: Better feature separation

## Advanced Features

### 1. **Restart Information Tracking**

```python
# Access detailed restart information
restart_info = scheduler.get_restart_info(epoch)
print(f"Last restart: {restart_info['last_restart_epoch']}")
print(f"Epochs since restart: {restart_info['epochs_since_restart']}")
print(f"Next restart: {restart_info['next_restart_epoch']}")
print(f"Is restart epoch: {restart_info['is_restart_epoch']}")
```

### 2. **Mixed Scheduler Strategies**

```python
# Combine different strategies for optimal training
hierarchical_config = {
    "global_scheduler": {
        "type": "warm_restart",
        "base_scheduler_type": "cosine",
        "base_scheduler_kwargs": {"max_beta": 0.01, "warmup_epochs": 10},
        "restart_interval": 30,
        "restart_multiplier": 1.0
    },
    "local_scheduler": {
        "type": "cyclical",
        "cycle_length": 15,
        "min_beta": 0.0,
        "max_beta": 0.02,
        "wave_type": "triangle"
    }
}
```

### 3. **Factory Function with Error Handling**

```python
# Robust scheduler creation with validation
try:
    scheduler = get_beta_scheduler("cyclical", **params)
except ValueError as e:
    print(f"Invalid configuration: {e}")
    # Fall back to default scheduler
    scheduler = get_beta_scheduler("cosine", max_beta=1.0, warmup_epochs=10)
```

## Best Practices

### 1. **Scheduler Selection Guidelines**

- **Simple Models**: Use `linear` or `cosine` for straightforward training
- **Complex Datasets**: Use `cyclical` for better exploration
- **Difficult Optimization**: Use `warm_restart` for robustness
- **Hierarchical Models**: Use `hierarchical` with appropriate sub-schedulers

### 2. **Parameter Tuning**

#### Cyclical Schedulers:
```python
# Short cycles for rapid exploration
{"cycle_length": 15, "min_beta": 0.0, "max_beta": 1.0}

# Long cycles for stable training
{"cycle_length": 40, "min_beta": 0.1, "max_beta": 0.8}
```

#### Warm Restart Schedulers:
```python
# Frequent restarts for active exploration
{"restart_interval": 25, "restart_multiplier": 1.0}

# Adaptive restarts for efficiency
{"restart_interval": 50, "restart_multiplier": 1.5}
```

### 3. **Monitoring and Analysis**

- **Watch TensorBoard**: Monitor `Beta_Schedule/` for scheduling behavior
- **Track Convergence**: Compare different scheduler types on your dataset
- **Analyze Restarts**: Check if restarts improve final performance
- **Validate Exploration**: Ensure cyclical scheduling helps optimization

## Configuration Examples

### Complete Training Configurations

#### 1. **Cyclical Configuration** (`configs/cyclical_beta_config.json`):
```json
{
    "training": {
        "num_epochs": 100,
        "beta_scheduler_type": "cyclical",
        "beta_scheduler_params": {
            "cycle_length": 20,
            "min_beta": 0.0,
            "max_beta": 1.0,
            "wave_type": "cosine"
        }
    }
}
```

#### 2. **Warm Restart Configuration** (`configs/warm_restart_config.json`):
```json
{
    "training": {
        "num_epochs": 150,
        "beta_scheduler_type": "warm_restart",
        "beta_scheduler_params": {
            "base_scheduler_type": "cosine",
            "base_scheduler_kwargs": {
                "max_beta": 1.0,
                "warmup_epochs": 20
            },
            "restart_interval": 40,
            "restart_multiplier": 1.2
        }
    }
}
```

#### 3. **Hierarchical Configuration** (integrated with `configs/hierarchical_config.json`):
```json
{
    "training": {
        "use_hierarchical_model": true,
        "beta_scheduler_type": "hierarchical",
        "hierarchical_beta_config": {
            "global_scheduler": {
                "type": "cosine",
                "max_beta": 0.01,
                "warmup_epochs": 10
            },
            "local_scheduler": {
                "type": "cyclical",
                "cycle_length": 25,
                "min_beta": 0.0,
                "max_beta": 0.02,
                "wave_type": "cosine"
            }
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Oscillating Validation Loss**
   - **Cause**: Cyclical scheduling with too short cycles
   - **Solution**: Increase `cycle_length` or use `cosine` wave type

2. **Training Instability**
   - **Cause**: Too frequent warm restarts
   - **Solution**: Increase `restart_interval` or reduce `max_beta`

3. **Poor Convergence**
   - **Cause**: Inappropriate scheduler for dataset complexity
   - **Solution**: Start with `cosine`, then experiment with advanced schedulers

### Debug Commands

```python
# Check scheduler configuration
print(f"Scheduler type: {type(scheduler).__name__}")
print(f"Scheduler config: {scheduler}")

# Monitor beta values
for epoch in range(10):
    beta = scheduler(epoch)
    print(f"Epoch {epoch}: Beta = {beta:.6f}")

# Check restart information
if hasattr(scheduler, 'get_restart_info'):
    info = scheduler.get_restart_info(epoch)
    print(f"Restart info: {info}")
```

## Implementation Benefits

### 1. **Training Quality**
- **Better Exploration**: Cyclical and restart scheduling explore more of the loss landscape
- **Robust Convergence**: Less likely to get stuck in poor local minima
- **Adaptive Dynamics**: Self-adjusting training dynamics

### 2. **Model Performance**
- **Improved Disentanglement**: Better latent space organization
- **Higher Quality Samples**: Better balance between reconstruction and regularization
- **Stable Training**: More predictable training outcomes

### 3. **Experimental Flexibility**
- **Easy Comparison**: Simple configuration changes for different strategies
- **Comprehensive Logging**: Detailed monitoring of training dynamics
- **Extensible Design**: Easy to add new scheduler types

## Future Enhancements

1. **Adaptive Scheduling**: Learn optimal β schedules during training
2. **Multi-Objective Optimization**: Balance multiple loss components dynamically
3. **Curriculum Learning**: Progressive difficulty in regularization
4. **Population-Based Training**: Evolutionary approach to scheduler optimization

---

**Implementation Status:** ✅ Complete and Tested  
**Training Integration:** ✅ Fully Integrated  
**TensorBoard Logging:** ✅ Comprehensive  
**Ready for Production:** ✅ Yes

The enhanced beta scheduler implementation provides sophisticated tools for VAE training optimization, enabling better exploration, more robust convergence, and superior model performance through advanced regularization scheduling strategies. 