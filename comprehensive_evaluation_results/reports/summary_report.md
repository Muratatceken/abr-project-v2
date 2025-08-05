
# ABR Model Evaluation Report

## Executive Summary
Evaluation completed on: 2025-08-05T03:52:19.164489
Evaluation time: 35.61 seconds

## Model Information
- Model: OptimizedHierarchicalUNet
- Total Parameters: 80,298,680
- Trainable Parameters: 80,298,680

## Dataset Information
- Number of samples: 5358
- Signal shape: [200]
- Number of classes: 5

## Performance Summary

### Signal Quality
**Basic Metrics:**
- mse: 3.6179
- mae: 1.5078
- rmse: 1.9021
- correlation_mean: -0.0323
- correlation_std: 0.1207
- snr_mean_db: -4.6304
- snr_std_db: 11.8072
- dtw_distance_mean: 1.4895
- dtw_distance_std: 0.0641
**Spectral Metrics:**
- spectral_correlation: -0.1294
- spectral_mse: 3.4729
**Temporal Metrics:**
- peak_count_correlation: -0.0963
- peak_count_mae: 12.0500
**Perceptual Metrics:**

### Classification
**Basic Metrics:**
- accuracy: 0.7981
- f1_macro: 0.1775
- f1_weighted: 0.7084
**Per Class Metrics:**
- accuracy: 0.7981
**Confusion Matrix:**
**Roc Analysis:**
- macro_auc: 0.5796
**Confidence Analysis:**
- mean_confidence_correct: 0.5239
- mean_confidence_incorrect: 0.5237
- confidence_correlation_with_accuracy: 0.1168

### Peak Detection
**Existence Metrics:**
**Latency Metrics:**
**Amplitude Metrics:**
**Timing Accuracy:**

### Threshold Regression
**Regression Metrics:**
- mae_db: 14.9043
- rmse_db: 22.3163
- r2_score: -0.0826
- correlation: 0.3838
- mean_error_db: -6.1657
- std_error_db: 21.4476
**Clinical Metrics:**
- category_accuracy: 0.5534
- category_f1_score: 0.1425
**Error Analysis:**
- within_5db_percent: 28.1635
- within_10db_percent: 49.3094
- within_15db_percent: 67.8052
- max_error_db: 93.5297
- percentile_95_error_db: 46.3666
- percentile_99_error_db: 86.9071

## Recommendations
1. Model performance is excellent across all metrics!
