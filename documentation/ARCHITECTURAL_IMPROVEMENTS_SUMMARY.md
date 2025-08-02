# ABR Model Architecture Improvements and Recommendations

## Executive Summary

Based on the performance analysis showing critical issues with peak prediction (NaN R² values), threshold estimation (negative R²), and classification (low macro F1), this document outlines comprehensive architectural improvements to address these problems.

## Current Performance Issues

### Critical Problems Identified

1. **Peak Prediction Failure**
   - Latency R²: NaN (complete failure)
   - Amplitude R²: NaN (complete failure)
   - High false peak detection rate: 21.1%

2. **Threshold Estimation Problems**
   - R²: -1.1989 (worse than trivial baseline)
   - Negative correlation: -0.2908
   - High over/underestimation rates (61.4%/36.6%)

3. **Classification Issues**
   - Macro F1: 0.2148 (poor minority class performance)
   - Severe class imbalance handling problems

4. **Signal Reconstruction**
   - Generally good (correlation: 0.8973)
   - But could be improved with better architecture

## Architectural Improvements Implemented

### 1. Enhanced Peak Prediction Head (`RobustPeakHead`)

**Problems Addressed:**
- NaN R² values due to poor gradient flow
- Inadequate masking strategy
- Single-scale feature processing

**Improvements:**
- **Multi-scale Feature Extraction**: Captures peak characteristics at different temporal scales
- **Separate Task-Specific Encoders**: Individual encoders for existence, latency, and amplitude prediction
- **Proper Masking**: Only computes regression losses for samples with existing peaks
- **Uncertainty Estimation**: Provides confidence bounds for predictions
- **Range Normalization**: Uses tanh activation to constrain outputs to valid ranges

```python
# Key improvements:
- Multi-scale convolutions (kernels: 1, 3, 5, 7)
- Task-specific feature encoders with residual connections
- Uncertainty-aware loss (negative log-likelihood)
- Proper gradient flow through masking
```

### 2. Robust Threshold Head (`RobustThresholdHead`)

**Problems Addressed:**
- Negative R² indicating poor regression
- Outlier sensitivity
- Poor range handling

**Improvements:**
- **Dual-Path Encoding**: Global and local feature paths for comprehensive representation
- **Robust Loss Functions**: Huber loss instead of MSE for outlier resistance
- **Multi-Objective Learning**: Main and auxiliary predictors for regularization
- **Better Range Scaling**: Sigmoid-based scaling to valid threshold ranges
- **Outlier Detection**: Built-in outlier probability estimation

### 3. Enhanced Classification Head (`RobustClassificationHead`)

**Problems Addressed:**
- Low macro F1 score
- Poor minority class handling
- Class imbalance

**Improvements:**
- **Hierarchical Feature Learning**: Residual blocks for better feature extraction
- **Class-Specific Extractors**: Individual feature extractors per class
- **Focal Loss Support**: Built-in focal loss for class imbalance
- **Dynamic Class Weighting**: Adaptive class weight updates
- **Temperature Scaling**: Better probability calibration

### 4. Improved Loss Function

**Key Changes:**
- **Uncertainty-Aware Losses**: Negative log-likelihood for predictions with uncertainty
- **Robust Regression**: Huber loss instead of MSE
- **Adaptive Loss Weighting**: Dynamic weight adjustment based on loss magnitudes
- **Better Masking**: Proper gradient flow for masked losses

### 5. Enhanced Training Configuration

**Major Improvements:**
- **Rebalanced Loss Weights**: 
  - Peak latency/amplitude: 2.5× (up from 1.0)
  - Threshold: 3.0× (up from 0.8)
  - Classification: 2.0× (up from 1.0)
- **Curriculum Learning**: Progressive weight changes over training
- **Enhanced Regularization**: Dropout scheduling, weight decay scheduling
- **Better Optimization**: Parameter group-specific learning rates

## Implementation Details

### Key Files Modified

1. **`models/blocks/heads.py`**
   - Added `RobustPeakHead` with multi-scale processing
   - Added `RobustThresholdHead` with dual-path encoding
   - Added `RobustClassificationHead` with hierarchical features

2. **`diffusion/loss.py`**
   - Updated peak loss computation with proper masking
   - Added uncertainty-aware losses
   - Implemented adaptive loss weighting

3. **`training/config_production_improved.yaml`**
   - Rebalanced loss weights based on performance analysis
   - Added curriculum learning schedule
   - Enhanced regularization strategies

### New Architecture Features

#### Multi-Scale Feature Extraction
```python
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, scales: List[int] = [1, 3, 5, 7]):
        # Captures features at different temporal scales
        # Critical for peak detection and signal analysis
```

#### Uncertainty Estimation
```python
# Peak predictions with uncertainty
latency_mean, latency_std = self.latency_head(features)
# Enables confidence-aware predictions and robust training
```

#### Robust Loss Functions
```python
# Huber loss for outlier resistance
loss = F.huber_loss(predictions, targets, delta=1.0)
# Better than MSE for medical data with outliers
```

## Expected Performance Improvements

### Peak Prediction
- **Latency R²**: Expected improvement from NaN to >0.7
- **Amplitude R²**: Expected improvement from NaN to >0.6
- **False Peak Rate**: Expected reduction from 21.1% to <10%

### Threshold Estimation
- **R²**: Expected improvement from -1.2 to >0.5
- **Correlation**: Expected improvement from -0.29 to >0.7
- **Over/Underestimation**: More balanced prediction distribution

### Classification
- **Macro F1**: Expected improvement from 0.21 to >0.6
- **Minority Classes**: Better handling of TOTAL, İTİK, NÖROPATİ classes
- **Class Balance**: Improved performance across all classes

## Training Recommendations

### 1. Loss Weight Strategy
```yaml
# Recommended loss weights (final)
loss_weights:
  signal: 0.8          # Slightly reduced (performing well)
  peak_exist: 1.2      # Increased for better detection
  peak_latency: 2.5    # Major increase to address NaN
  peak_amplitude: 2.5  # Major increase to address NaN  
  classification: 2.0  # Increased for minority classes
  threshold: 3.0       # Highest priority for negative R²
```

### 2. Curriculum Learning
- **Phase 1 (Epochs 1-20)**: Focus on peak existence and classification
- **Phase 2 (Epochs 21-50)**: Gradually introduce regression tasks
- **Phase 3 (Epochs 51+)**: Full multi-task learning with final weights

### 3. Regularization Strategy
- **Dropout Scheduling**: Start 0.1 → Peak 0.2 → End 0.15
- **Weight Decay**: Progressive increase 1e-4 → 5e-4
- **Gradient Clipping**: Aggressive clipping (0.5) for stability

### 4. Monitoring Strategy
- **Critical Metrics**: Track R² values for early NaN detection
- **Alerts**: Automatic alerts for performance degradation
- **Interventions**: Learning rate reduction on anomalies

## Architecture Validation

### Key Validation Points
1. **Gradient Flow**: Verify no NaN gradients in peak heads
2. **Loss Scaling**: Monitor relative loss magnitudes
3. **Convergence**: Ensure all tasks converge simultaneously
4. **Generalization**: Validate on held-out test set

### Performance Benchmarks
- **Peak Existence F1**: Target >0.9 (currently 0.87)
- **Peak Latency R²**: Target >0.7 (currently NaN)
- **Threshold R²**: Target >0.5 (currently -1.2)
- **Classification Macro F1**: Target >0.6 (currently 0.21)

## Next Steps

### Immediate Actions
1. **Test New Architecture**: Run training with improved configuration
2. **Monitor Critical Metrics**: Watch for NaN resolution
3. **Validate Improvements**: Compare against baseline performance

### Future Enhancements
1. **Attention Mechanisms**: Add cross-attention between tasks
2. **Meta-Learning**: Adaptive loss weighting algorithms
3. **Domain Adaptation**: Techniques for clinical data variations
4. **Model Ensemble**: Combine multiple architectures for robustness

## Risk Mitigation

### Potential Issues
1. **Overfitting**: Monitor validation metrics carefully
2. **Training Instability**: Use gradient clipping and monitoring
3. **Memory Usage**: Larger model may require batch size adjustment

### Mitigation Strategies
1. **Early Stopping**: Patience increased to 40 epochs
2. **Checkpoint Frequency**: Save every 5 epochs
3. **Multiple Metrics**: Monitor combined performance metrics

## Conclusion

The proposed architectural improvements address the root causes of the current performance issues:

- **Multi-scale processing** for better feature extraction
- **Task-specific architectures** for specialized learning
- **Robust loss functions** for outlier resistance
- **Better regularization** for generalization
- **Curriculum learning** for stable training

These changes should significantly improve performance across all tasks, particularly addressing the critical NaN R² issues in peak prediction and negative R² in threshold estimation.

## Performance Tracking

Use the following command to monitor training with the new architecture:

```bash
python run_production_training.py --config training/config_production_improved.yaml
```

Key metrics to monitor:
- Peak latency/amplitude R² (should become positive)
- Threshold R² (should become positive)
- Classification macro F1 (should increase significantly)
- Overall loss stability and convergence 