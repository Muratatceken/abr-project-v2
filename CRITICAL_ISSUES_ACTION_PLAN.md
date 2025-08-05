# 🚨 ABR Model Critical Issues - Action Plan

## Executive Summary
The comprehensive evaluation has revealed **CRITICAL FAILURES** across all model components. The model requires **COMPLETE RETRAINING** with fundamental changes to the training approach.

## 🔥 Critical Issues Identified

### 1. **CATASTROPHIC CLASSIFICATION FAILURE** 
- **Issue**: Model predicts ONLY class 0 for all 5,358 test samples
- **Impact**: Clinically useless - cannot detect any hearing loss
- **Root Cause**: Severe loss imbalance causing gradient dominance

### 2. **COMPLETE THRESHOLD PREDICTION COLLAPSE**
- **Issue**: All threshold predictions collapsed to ~17.60 dB (range: 0.00 dB)
- **True Range**: 0.00 - 111.14 dB (111.14 dB span)
- **Impact**: No clinical discrimination capability
- **Root Cause**: Threshold regression head not learning

### 3. **SIGNAL RECONSTRUCTION BREAKDOWN**
- **Issue**: NaN correlations, -Infinity SNR values
- **Impact**: Fundamental signal processing failure
- **Root Cause**: Numerical instabilities in diffusion process

### 4. **PEAK DETECTION FAILURE**
- **Issue**: NaN correlations in latency predictions
- **Impact**: Unreliable peak timing analysis
- **Root Cause**: Peak head not learning meaningful patterns

## 🎯 Immediate Actions Required

### Phase 1: Emergency Training Fixes (Priority 1)

#### 1.1 **Fix Loss Function Imbalance**
```yaml
# Update configs/config.yaml
training:
  loss:
    weights:
      recon: 1.0
      class: 10.0        # INCREASE from current ~1.0
      peak: 0.01         # DECREASE from current ~1.0  
      threshold: 0.01    # DECREASE from current ~1.0
```

#### 1.2 **Implement Class Balancing**
```python
# Add to training/trainer.py
class_weights = torch.tensor([0.2, 5.0, 2.0, 3.0, 3.0])  # Inverse frequency weighting
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

#### 1.3 **Reduce Learning Rate & Add Warmup**
```yaml
training:
  optimizer:
    learning_rate: 0.0001  # REDUCE from current 0.001
  scheduler:
    warmup_epochs: 5
    eta_min: 0.00001
```

### Phase 2: Architecture Validation (Priority 2)

#### 2.1 **Verify Gradient Flow**
- Add gradient monitoring to training loop
- Check if classification head receives gradients
- Ensure multi-task heads don't interfere

#### 2.2 **Debug Signal Processing**
- Add NaN/Inf checks in diffusion pipeline
- Validate input/output ranges
- Check noise schedule parameters

#### 2.3 **Simplify Training Initially**
```python
# Train classification ONLY first, then add other tasks
loss = classification_loss  # Remove other tasks temporarily
```

### Phase 3: Training Strategy Overhaul (Priority 3)

#### 3.1 **Sequential Task Training**
1. **Epoch 0-20**: Classification only
2. **Epoch 21-40**: Classification + Signal reconstruction
3. **Epoch 41-60**: Add peak detection
4. **Epoch 61+**: Add threshold regression

#### 3.2 **Enhanced Monitoring**
- Log individual loss components every step
- Monitor class prediction distributions
- Track gradient norms for each head

#### 3.3 **Data Augmentation & Sampling**
- Implement balanced sampling
- Add noise augmentation for minority classes
- Use focal loss for classification

## 🔧 Specific Configuration Changes

### configs/config.yaml - Emergency Fixes
```yaml
# CRITICAL FIXES
training:
  optimizer:
    learning_rate: 0.0001  # Reduce by 10x
    weight_decay: 0.01     # Increase regularization
  
  loss:
    weights:
      recon: 1.0
      class: 10.0          # Increase classification importance
      peak: 0.01           # Reduce peak loss weight
      threshold: 0.01      # Reduce threshold loss weight
    
    # Add focal loss for classification
    focal_loss:
      alpha: 0.25
      gamma: 2.0
  
  scheduler:
    type: "cosine_with_warmup"
    warmup_epochs: 5
    eta_min: 0.00001
  
  # Enhanced monitoring
  log_frequency: 10       # Log more frequently
  
  # Gradient management
  gradient_clip: 1.0      # More aggressive clipping
  accumulation_steps: 8   # Larger effective batch size

# Data handling improvements
data:
  dataloader:
    batch_size: 8         # Smaller batches for stability
    balanced_sampling: true  # Enable balanced sampling
```

### training/trainer.py - Add Class Balancing
```python
# Calculate class weights from training data
class_counts = torch.bincount(train_targets)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * len(class_weights)

# Use weighted loss
self.classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
```

## 📊 Success Metrics for Re-training

### Phase 1 Success Criteria (Classification Focus)
- [ ] Classification accuracy > 60% (vs current 80% on single class)
- [ ] All classes receive some predictions (vs current 0)
- [ ] F1-macro > 0.4 (vs current 0.18)
- [ ] Classification loss decreases consistently

### Phase 2 Success Criteria (Multi-task)
- [ ] Signal correlation > 0.5 (vs current NaN)
- [ ] SNR > 10 dB (vs current -Infinity)
- [ ] Threshold R² > 0.3 (vs current -0.08)
- [ ] Peak latency correlation > 0.5 (vs current NaN)

### Phase 3 Success Criteria (Clinical)
- [ ] Clinical diagnostic accuracy > 70%
- [ ] Threshold MAE < 10 dB (vs current 14.9 dB)
- [ ] Peak timing accuracy within 1ms > 60% (vs current 41%)

## ⏰ Timeline

### Week 1: Emergency Fixes
- [x] Implement loss rebalancing
- [x] Add class weighting
- [x] Reduce learning rate
- [ ] Start retraining with classification focus

### Week 2: Architecture Debug
- [ ] Add gradient monitoring
- [ ] Debug signal processing issues
- [ ] Validate multi-task integration

### Week 3: Full Training
- [ ] Sequential task introduction
- [ ] Enhanced monitoring implementation
- [ ] Performance validation

## 🚨 Red Flags to Monitor

During retraining, immediately stop if:
1. **Loss becomes NaN/Inf**
2. **All predictions collapse to single class again**
3. **Gradient norms explode (>10.0)**
4. **Individual loss components don't decrease**

## 📝 Next Steps

1. **IMMEDIATELY**: Update config with emergency fixes
2. **TODAY**: Start classification-only retraining
3. **Monitor**: Individual loss components hourly during first day
4. **Validate**: Check class predictions after 1 epoch

---

**Status**: 🚨 CRITICAL - IMMEDIATE ACTION REQUIRED
**Next Review**: After 24 hours of retraining with fixes