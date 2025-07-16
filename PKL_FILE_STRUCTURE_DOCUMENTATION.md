# PKL File Structure Documentation - Categorical Encoding

## Overview

This document provides a comprehensive analysis of the `processed_data_categorical_200ts.pkl` file structure, which contains the ABR dataset with **categorical encoding** for hearing loss types.

## File Information

- **File**: `data/processed/processed_data_categorical_200ts.pkl`
- **Format**: Python pickle file containing a list of dictionaries
- **Size**: 42.35 MB (reduced from ~43.14 MB with one-hot encoding)
- **Samples**: 51,999 ABR recordings
- **Patients**: 2,038 unique patients
- **Encoding**: Categorical (1-5) for hearing loss instead of one-hot

## Top-Level Structure

```python
# Data structure
data = [
    {
        'patient_id': str,
        'static_params': numpy.ndarray,  # shape: (6,) - REDUCED FROM 10
        'signal': numpy.ndarray,         # shape: (200,)
        'peaks': numpy.ndarray,          # shape: (6,)
        'peak_mask': numpy.ndarray       # shape: (6,)
    },
    # ... 51,998 more samples
]
```

## Detailed Structure Analysis

### 1. Sample Keys
Each sample contains exactly **5 keys**:
- `patient_id`: String identifier
- `static_params`: 6-dimensional parameter array (**reduced from 10**)
- `signal`: 200-timestamp ABR signal
- `peaks`: Peak latencies and amplitudes
- `peak_mask`: Boolean mask for peak validity

### 2. Static Parameters (6 dimensions)

**Dimension Breakdown:**
- **Dimensions 0-4**: Continuous parameters (normalized using StandardScaler)
- **Dimension 5**: Categorical hearing loss (integer 1-5)

#### Continuous Parameters (Dimensions 0-4)
| Dimension | Parameter | Min | Max | Mean | Std |
|-----------|-----------|-----|-----|------|-----|
| 0 | Age | -0.3569 | 9.7816 | 0.0000 | 0.9843 |
| 1 | Intensity | -1.4774 | 1.8821 | 0.0000 | 1.0000 |
| 2 | Stimulus Rate | -6.2278 | 4.2521 | -0.0000 | 1.0000 |
| 3 | FMP | -0.2048 | 129.1079 | 0.0000 | 0.9998 |
| 4 | ResNo | -1.7107 | 27.2283 | 0.0000 | 0.9998 |

#### Categorical Parameter (Dimension 5)
| Category | Label | Encoding | Sample Count | Percentage |
|----------|-------|----------|--------------|------------|
| 1 | Normal | 1.0 | 41,417 | 79.6% |
| 2 | Nöropati̇ | 2.0 | 417 | 0.8% |
| 3 | Sni̇k | 3.0 | 5,517 | 10.6% |
| 4 | Total | 4.0 | 1,720 | 3.3% |
| 5 | İti̇k | 5.0 | 2,928 | 5.6% |

**Statistical Properties:**
- Min: 1.0, Max: 5.0
- Mean: 1.5447, Std: 1.1551
- Data type: float32
- All values are integers (1, 2, 3, 4, 5)

### 3. Signal Data (200 timestamps)

**Properties:**
- **Shape**: (200,) per sample
- **Data type**: float32
- **Normalization**: Z-score normalized per sample
- **Range**: Global min: -5.036, Global max: 7.449
- **Statistics**: Mean: 0.0007, Std: 0.719

**Quality Assurance:**
- ✅ No NaN values
- ✅ No Inf values
- ✅ Consistent length across all samples

### 4. Peak Data (6 peak types)

**Peak Types:**
1. **I Latency**: Valid in 11.1% of samples (5,754/51,999)
2. **III Latency**: Valid in 12.0% of samples (6,254/51,999)
3. **V Latency**: Valid in 74.6% of samples (38,780/51,999)
4. **I Amplitude**: Valid in 11.1% of samples (5,760/51,999)
5. **III Amplitude**: Valid in 12.0% of samples (6,254/51,999)
6. **V Amplitude**: Valid in 74.6% of samples (38,780/51,999)

**Statistical Summary:**
| Peak Type | Valid % | Min | Max | Mean | Std |
|-----------|---------|-----|-----|------|-----|
| I Latency | 11.1% | 0.0000 | 20.9300 | 1.5720 | 0.4711 |
| III Latency | 12.0% | 1.6700 | 8.6000 | 4.1363 | 0.4919 |
| V Latency | 74.6% | 4.8000 | 18.4700 | 7.8341 | 1.7100 |
| I Amplitude | 11.1% | -0.4750 | 1.1440 | 0.3048 | 0.1622 |
| III Amplitude | 12.0% | -0.6010 | 1.4220 | 0.4018 | 0.2149 |
| V Amplitude | 74.6% | -0.6880 | 1.2490 | 0.1740 | 0.1566 |

### 5. Peak Masks (6 boolean values)

**Purpose**: Indicates which peaks are valid (True) or missing (False)
- **Data type**: bool
- **Shape**: (6,) per sample
- **Usage**: Used in loss functions to ignore missing peaks

### 6. Patient Distribution

**Statistics:**
- **Total samples**: 51,999
- **Unique patients**: 2,038
- **Samples per patient**: 1-125 (mean: 25.51, median: 24.00)

**Top 10 patients by sample count:**
1. Patient 902373: 125 samples
2. Patient 797329: 120 samples
3. Patient 766595: 114 samples
4. Patient 191423: 114 samples
5. Patient 729868: 98 samples
6. Patient 805538: 92 samples
7. Patient 436892: 85 samples
8. Patient 576356: 82 samples
9. Patient 910165: 81 samples
10. Patient 770104: 79 samples

## Memory Usage Analysis

### Total Memory Breakdown
- **Total size**: 42.35 MB
- **Signals**: 39.67 MB (93.7%)
- **Static params**: 1.19 MB (2.8%)
- **Peaks**: 1.19 MB (2.8%)
- **Peak masks**: 0.30 MB (0.7%)

### Comparison with One-Hot Encoding
- **One-hot static params**: 1.98 MB
- **Categorical static params**: 1.19 MB
- **Memory saved**: 0.79 MB (40.0% reduction)

## Data Quality Assurance

### ✅ All Quality Checks Passed
- **Signal length consistency**: All 200 timestamps
- **Static parameter dimensions**: All 6 dimensions
- **Peak dimensions**: All 6 peak types
- **Data integrity**: No NaN/Inf values in signals or static params
- **Categorical values**: All within valid range (1-5)

### Data Types
- **patient_id**: Python string
- **static_params**: numpy.float32 array
- **signal**: numpy.float32 array
- **peaks**: numpy.float32 array (may contain NaN for missing peaks)
- **peak_mask**: numpy.bool array

## Usage Examples

### Loading the Data
```python
import joblib
import numpy as np

# Load the data
data = joblib.load('data/processed/processed_data_categorical_200ts.pkl')
print(f"Loaded {len(data)} samples")

# Access a sample
sample = data[0]
print(f"Patient ID: {sample['patient_id']}")
print(f"Static params shape: {sample['static_params'].shape}")
print(f"Signal shape: {sample['signal'].shape}")
```

### Working with Categorical Encoding
```python
# Extract hearing loss categories
hearing_loss_categories = [sample['static_params'][5] for sample in data]

# Convert to category names
label_encoder = joblib.load('data/processed/label_encoder.pkl')
category_names = label_encoder.classes_

# Map categories to names
for sample in data[:5]:
    category_id = int(sample['static_params'][5])
    category_name = category_names[category_id - 1]  # Convert 1-based to 0-based
    print(f"Patient {sample['patient_id']}: {category_name}")
```

### Generating Random Parameters for Inference
```python
import numpy as np

# Generate random continuous parameters (normalized)
continuous_params = np.random.randn(num_samples, 5)

# Generate random categorical parameters (1-5)
categorical_params = np.random.randint(1, 6, size=(num_samples, 1))

# Combine into static parameters
static_params = np.concatenate([continuous_params, categorical_params], axis=1)
```

## Key Differences from One-Hot Encoding

### Before (One-Hot)
```python
# 10-dimensional static parameters
static_params = [
    # Continuous (5 dims)
    age_norm, intensity_norm, rate_norm, fmp_norm, resno_norm,
    # One-hot hearing loss (5 dims)
    1.0, 0.0, 0.0, 0.0, 0.0  # Normal hearing
]
```

### After (Categorical)
```python
# 6-dimensional static parameters
static_params = [
    # Continuous (5 dims)
    age_norm, intensity_norm, rate_norm, fmp_norm, resno_norm,
    # Categorical hearing loss (1 dim)
    1.0  # Normal hearing
]
```

## Benefits of Categorical Encoding

1. **Reduced Dimensionality**: 10 → 6 dimensions (40% reduction)
2. **Memory Efficiency**: 0.79 MB saved (40% reduction in static params)
3. **Simpler Interpretation**: Single integer vs 5 binary values
4. **Faster Processing**: Fewer neural network parameters
5. **Easier Inference**: Simpler random parameter generation

## Model Compatibility

The categorical encoding is **fully compatible** with existing CVAE models:
- Input layer automatically adjusts to 6 dimensions
- No changes needed in encoder/decoder architecture
- All loss functions work identically
- Peak prediction remains unchanged

## Next Steps

1. **Train models** with categorical encoding
2. **Compare performance** with one-hot encoding models
3. **Benchmark speed** improvements
4. **Consider embedding layers** for categorical variables if needed

---

**Status**: ✅ **Analysis Complete**  
**Format**: Categorical encoding (1-5) for hearing loss  
**Dimensions**: 6 static parameters (5 continuous + 1 categorical)  
**Quality**: All integrity checks passed  
**Memory**: 40% reduction in static parameter storage 