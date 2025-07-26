# ABR Model Joint Generation Guide

## Overview

The enhanced ABR Hierarchical U-Net now supports **joint generation** of both ABR signals and their corresponding static parameters (age, intensity, stimulus_rate, fmp). This enables the model to generate realistic, correlated multi-modal data where static parameters are consistent with the generated signals.

## Generation Modes

The model supports three distinct generation modes:

### 1. Conditional Generation (Default)

- **Input**: Static parameters (age, intensity, stimulus_rate, fmp)
- **Output**: ABR signal, peaks, classification, threshold
- **Use case**: Generate signals for specific patient parameters

### 2. Joint Generation (New)

- **Input**: Random noise only
- **Output**: ABR signal + generated static parameters + peaks + classification + threshold
- **Use case**: Generate complete synthetic ABR datasets

### 3. Unconditional Generation

- **Input**: Random noise only
- **Output**: ABR signal, peaks, classification, threshold (no parameter generation)
- **Use case**: Exploration of model capabilities

## Key Features

### Multi-Scale Parameter Generation

- Uses multi-scale convolutions to capture parameter relationships at different scales
- Separate encoders for each parameter type (age, intensity, stimulus_rate, fmp)

### Cross-Parameter Dependencies

- Models realistic correlations between parameters (e.g., age-intensity relationships)
- Dependency adjustment layer ensures clinically plausible combinations

### Clinical Constraints

- Enforces realistic parameter ranges based on dataset statistics
- Applies clinical knowledge constraints (e.g., young patients → lower intensities)
- Outlier detection and correction

### Uncertainty Estimation

- Provides confidence bounds for generated parameters
- Temperature-controlled sampling for diversity control
- Negative log-likelihood training for robust generation

## Architecture Components

### Static Parameter Generation Head

```python
class StaticParameterGenerationHead(nn.Module):
    """
    Generates realistic static parameters with:
    - Parameter-specific encoders
    - Cross-parameter dependency modeling  
    - Clinical constraint enforcement
    - Uncertainty quantification
    """
```

**Key Components:**

- **Multi-scale feature extraction** for better parameter generation
- **Parameter-specific encoders** (one per parameter type)
- **Dependency encoder** for cross-parameter relationships
- **Clinical constraint enforcement** for realistic outputs

### Enhanced Model Architecture

```python
ProfessionalHierarchicalUNet(
    enable_joint_generation=True,  # Enable joint generation
    static_param_ranges={           # Parameter ranges
        'age': (-0.36, 11.37),
        'intensity': (-2.61, 1.99), 
        'stimulus_rate': (-6.79, 5.10),
        'fmp': (-0.20, 129.11)
    }
)
```

## Usage Examples

### Basic Joint Generation

```python
# Initialize model with joint generation
model = ProfessionalHierarchicalUNet(enable_joint_generation=True)

# Generate both signals and parameters jointly
outputs = model.generate_joint(
    batch_size=10,
    device=device,
    temperature=1.0,        # Control randomness
    use_constraints=True    # Apply clinical constraints
)

# Outputs contain:
# - 'recon': Generated ABR signals [batch, 200]
# - 'static_params': Generated parameters [batch, 4] 
# - 'peak': Peak predictions
# - 'class': Classification logits
# - 'threshold': Threshold predictions
```

### Conditional Generation

```python
# Create static parameters
static_params = torch.tensor([
    [0.5, -0.2, 1.0, 25.0],  # age, intensity, stimulus_rate, fmp
    [2.0, 0.8, -1.5, 45.0]
])

# Generate signals conditioned on parameters
outputs = model.generate_conditional(
    static_params=static_params,
    noise_level=1.0
)
```

### Advanced Sampling

```python
# Generate with custom temperature and constraints
outputs = model.generate_joint(
    batch_size=100,
    device=device,
    temperature=1.2,        # Higher temperature = more diversity
    use_constraints=True    # Apply clinical constraints
)

# Sample from uncertainty distributions
if model.static_param_head.use_uncertainty:
    sampled_params = model.static_param_head.sample_parameters(
        x_features,
        temperature=0.8,
        use_constraints=True
    )
```

## Training Configuration

### Enable Joint Generation in Config

```yaml
model:
  enable_joint_generation: true
  static_param_ranges:
    age: [-0.36, 11.37]
    intensity: [-2.61, 1.99]
    stimulus_rate: [-6.79, 5.10]
    fmp: [-0.20, 129.11]

loss_weights:
  static_params: 1.5  # Weight for parameter generation loss

curriculum:
  joint_generation:
    enabled: true
    start_conditional_epochs: 30    # Train conditional first
    introduce_joint_epoch: 50       # Then introduce joint
    joint_probability: 0.3          # 30% joint, 70% conditional
```

### Loss Function Updates

The loss function now includes static parameter generation loss:

```python
# Individual parameter losses
losses = {
    'static_age': age_loss,
    'static_intensity': intensity_loss, 
    'static_stimulus_rate': rate_loss,
    'static_fmp': fmp_loss,
    'static_total': total_param_loss
}

# Total loss includes parameter generation
total_loss = (
    signal_loss + peak_losses + classification_loss + 
    threshold_loss + static_param_loss
)
```

## Clinical Constraints

The model enforces realistic parameter combinations:

### Age-Related Constraints

- Young patients (normalized age < -0.2) get reduced intensities
- Prevents unrealistic age-intensity combinations

### Stimulus Rate Constraints

- High stimulus rates (> 2.0) get moderate intensities
- Ensures clinically feasible rate-intensity pairs

### FMP Constraints

- High FMP values (> 50.0) get adjusted intensity ranges
- Maintains realistic FMP-intensity relationships

### Custom Constraints

```python
def _apply_clinical_constraints(self, params: torch.Tensor) -> torch.Tensor:
    """Apply custom clinical constraints."""
    # Example: Ensure age-intensity correlation
    young_mask = params[:, 0] < -0.2  # Young patients
    if young_mask.any():
        params[young_mask, 1] *= 0.8  # Reduce intensity
  
    return params
```

## Evaluation Metrics

Joint generation adds new evaluation metrics:

### Parameter Generation Quality

- **Age R²**: Coefficient of determination for age prediction
- **Intensity R²**: Quality of intensity generation
- **Stimulus Rate R²**: Rate parameter accuracy
- **FMP R²**: FMP generation quality

### Cross-Modal Consistency

- **Parameter-Signal Correlation**: How well parameters match signals
- **Clinical Validity**: Adherence to clinical constraints
- **Distribution Matching**: How well generated parameters match real data

### Evaluation Configuration

```yaml
evaluation:
  metrics:
    static_params: [
      "age_r2", "intensity_r2", "stimulus_rate_r2", "fmp_r2",
      "param_correlation", "param_mse"
    ]
  
  joint_generation_eval:
    enabled: true
    num_samples: 100
    eval_modes: ["conditional", "joint", "unconditional"]
```

## Demo Script

Run the comprehensive demonstration:

```bash
python joint_generation_example.py
```

This generates:

- **conditional_generation.png**: Signals from given parameters
- **joint_generation.png**: Jointly generated signals and parameters
- **unconditional_generation.png**: Unconditional generation
- **mode_comparison.png**: Side-by-side comparison

## Advanced Features

### Uncertainty Quantification

```python
# Generate with uncertainty bounds
outputs = model.generate_joint(batch_size=10, device=device)

if 'static_params' in outputs:
    params = outputs['static_params']  # [batch, static_dim, 2]
    means = params[:, :, 0]  # Parameter means
    stds = params[:, :, 1]   # Parameter uncertainties
```

### Temperature Scaling

```python
# Control generation diversity
low_temp_outputs = model.generate_joint(temperature=0.5)   # Conservative
high_temp_outputs = model.generate_joint(temperature=1.5)  # Diverse
```

### Batch Generation

```python
# Generate large batches efficiently
large_batch = model.generate_joint(
    batch_size=1000, 
    device=device,
    use_constraints=True
)
```

## Benefits of Joint Generation

### 1. Complete Synthetic Datasets

- Generate both signals and corresponding metadata
- No need for separate parameter generation models
- Consistent parameter-signal relationships

### 2. Data Augmentation

- Create additional training data with realistic parameters
- Explore rare parameter combinations
- Balance dataset distributions

### 3. Clinical Applications

- Generate data for specific patient demographics
- Simulate various clinical scenarios
- Test model robustness across parameter ranges

### 4. Research Applications

- Study parameter-signal relationships
- Generate controlled experimental data
- Validate clinical hypotheses

## Performance Expectations

With the enhanced architecture, expect:

### Joint Generation Quality

- **Parameter MSE**: < 0.1for normalized parameters
- **Clinical Validity**: > 95% of generated parameters within clinical ranges
- **Cross-Modal Consistency**: > 0.8 correlation between parameters and signals

### Training Convergence

- **Joint Loss Convergence**: 20-30 epochs after introduction
- **Parameter R²**: > 0.7 for all parameters
- **Overall Performance**: Maintained signal quality with added parameter generation

## Best Practices

### 1. Curriculum Learning

- Start with conditional generation (epochs 1-30)
- Introduce joint generation gradually (epoch 50+)
- Use mixed training (70% conditional, 30% joint)

### 2. Loss Weighting

- Static parameter loss weight: 1.0-2.0
- Balance with existing task weights
- Monitor relative loss magnitudes

### 3. Clinical Validation

- Always enable clinical constraints in production
- Validate generated parameters against real data
- Test edge cases and parameter combinations

### 4. Evaluation Strategy

- Evaluate all three generation modes
- Use multiple temperature settings
- Validate clinical consistency

## Troubleshooting

### Common Issues

**Issue**: Generated parameters are unrealistic
**Solution**: Enable clinical constraints, adjust parameter ranges

**Issue**: Poor parameter-signal consistency
**Solution**: Increase static parameter loss weight, check dependency modeling

**Issue**: Training instability with joint generation
**Solution**: Use curriculum learning, start with conditional generation only

**Issue**: High parameter generation loss
**Solution**: Check parameter ranges, reduce complexity, increase regularization

This joint generation capability transforms your ABR model from a conditional generator to a complete synthetic data generator, enabling new applications in research, clinical validation, and data augmentation.
