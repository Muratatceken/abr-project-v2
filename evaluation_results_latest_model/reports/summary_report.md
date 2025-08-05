
# ABR Model Evaluation Report

## Executive Summary
Evaluation completed on: 2025-08-05T03:09:23.415995
Evaluation time: 55.05 seconds

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
- correlation_mean: nan
- correlation_std: nan
- snr_mean_db: -inf
- snr_std_db: nan
- dtw_distance_mean: 1.4895
- dtw_distance_std: 0.0641
**Spectral Metrics:**
- spectral_correlation: -0.1290
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
- macro_auc: 0.5789
**Confidence Analysis:**
- mean_confidence_correct: 0.5239
- mean_confidence_incorrect: 0.5237
- confidence_correlation_with_accuracy: 0.1178

### Peak Detection
**Existence Metrics:**
- accuracy: 0.7602
- f1_score: 0.4319
**Latency Metrics:**
- mae_ms: 1.3497
- rmse_ms: 1.7290
- correlation: nan
- mean_error_ms: 0.1548
**Amplitude Metrics:**
- mae_uv: 0.3388
- rmse_uv: 0.3662
- correlation: nan
- mean_error_uv: 0.3309
**Timing Accuracy:**
- within_0_5ms_percent: 21.5320
- within_1_0ms_percent: 40.9281
- mean_timing_error_ms: 1.3497
- std_timing_error_ms: 1.0807

### Threshold Regression
**Regression Metrics:**
- mae_db: 14.9043
- rmse_db: 22.3163
- r2_score: -0.0826
- correlation: 0.3837
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
