# PKL File Structure Documentation - Normal Hearing Loss Only Dataset

## Overview

This document provides a comprehensive analysis of the `processed_data_normal_only_200ts.pkl` file structure, which contains the ABR dataset with **Normal hearing loss samples only** and **reduced dimensionality** for optimized processing.

## File Information

- **File**: `data/processed/processed_data_normal_only_200ts.pkl`
- **Format**: Python pickle file containing a list of dictionaries
- **Size**: Approximately 35 MB (reduced from 54 MB original dataset)
- **Samples**: 9,683 ABR recordings (filtered from 51,999 total)
- **Patients**: 1,386 unique patients (filtered from 2,038 total)
- **Hearing Loss Type**: Normal only
- **Encoding**: Simplified structure with reduced dimensions

## Filtering Criteria Applied

The dataset has been filtered using the following criteria:
1. **Stimulus Polarity**: `Alternate` only
2. **Sweeps Rejected**: ≤ 100
3. **FMP (Fast-forward Masking Paradigm)**: > 3.0
4. **Hearing Loss Type**: `Normal` only

**Filter Impact:**
- Original samples: 55,237
- After Alternate polarity: 53,280 (96.5%)
- After Sweep Rejection ≤ 100: 51,999 (94.1%)
- After FMP > 3.0: 10,625 (19.2%)
- After Normal hearing loss: 9,683 (17.5% of original)

## Top-Level Structure

```python
# Data structure
data = [
    {
        'patient_id': str,
        'static_params': numpy.ndarray,  # shape: (3,) - REDUCED FROM 6
        'signal': numpy.ndarray,         # shape: (200,)
        'peaks': numpy.ndarray,          # shape: (2,) - REDUCED FROM 6
        'peak_mask': numpy.ndarray       # shape: (2,) - REDUCED FROM 6
    },
    # ... 9,682 more samples
]
```

## Detailed Structure Analysis

### 1. Sample Keys
Each sample contains exactly **5 keys**:
- `patient_id`: String identifier
- `static_params`: 3-dimensional parameter array (**reduced from 6**)
- `signal`: 200-timestamp ABR signal
- `peaks`: V peak latency and amplitude only (**reduced from 6 peak types**)
- `peak_mask`: Boolean mask for V peak validity

### 2. Static Parameters (3 dimensions)

**All 3 dimensions are continuous parameters** (normalized using StandardScaler):

| Dimension | Parameter | Min | Max | Mean | Std |
|-----------|-----------|-----|-----|------|-----|
| 0 | Age | -0.3406 | 11.3658 | 0.0000 | 0.9951 |
| 1 | Intensity | -2.6096 | 1.9878 | -0.0000 | 1.0000 |
| 2 | Stimulus Rate | -6.7905 | 5.1031 | -0.0000 | 1.0000 |

**Key Changes:**
- **No categorical encoding**: Since only Normal hearing loss is included
- **Removed parameters**: FMP and ResNo are not included
- **Dimensionality**: Reduced from 6 to 3 dimensions (50% reduction)

### 3. Signal Data (200 timestamps)

**Properties remain the same:**
- **Shape**: (200,) per sample
- **Data type**: float32
- **Normalization**: Z-score normalized per sample
- **Quality**: All samples passed data cleaning

### 4. Peak Data (2 peak values only)

**Peak Types included:**
1. **V Latency**: Valid in 99.3% of samples (9,619/9,683)
2. **V Amplitude**: Valid in 99.3% of samples (9,619/9,683)

**Statistical Summary:**
| Peak Type | Valid % | Min | Max | Mean | Std |
|-----------|---------|-----|-----|------|-----|
| V Latency | 99.3% | 4.8000 | 14.1300 | 6.7591 | 1.0214 |
| V Amplitude | 99.3% | -0.5840 | 1.2490 | 0.3049 | 0.1637 |

**Key Changes:**
- **Only V peaks**: I and III peak data removed
- **Higher validity**: 99.3% vs ~75% in original dataset
- **Dimensionality**: Reduced from 6 to 2 peak values (67% reduction)

### 5. Peak Masks (2 boolean values)

**Purpose**: Indicates which V peaks are valid (True) or missing (False)
- **Data type**: bool
- **Shape**: (2,) per sample
- **Usage**: Used in loss functions to ignore missing peaks

### 6. Patient Distribution

**Statistics:**
- **Total samples**: 9,683
- **Unique patients**: 1,386
- **Samples per patient**: 1-36 (mean: 6.99, median: 6.00)
- **Distribution**: More evenly distributed compared to original dataset

## FMP Filter Analysis

**FMP Statistics (after > 3.0 filter):**
- **Min**: 3.01
- **Max**: 1,515.17
- **Mean**: 8.69
- **Standard Deviation**: 21.99

The FMP > 3.0 filter was the most restrictive, reducing the dataset from 51,999 to 10,625 samples (79.6% reduction).

## Memory Usage Analysis

### Total Memory Breakdown
- **Total size**: ~35 MB (35% reduction from original)
- **Signals**: ~30 MB (85.7%)
- **Static params**: ~0.11 MB (0.3%)
- **Peaks**: ~0.07 MB (0.2%)
- **Peak masks**: ~0.02 MB (0.1%)
- **Other metadata**: ~4.8 MB (13.7%)

### Comparison with Original Dataset
| Component | Original | Normal-Only | Reduction |
|-----------|----------|-------------|-----------|
| **Samples** | 51,999 | 9,683 | 81.4% |
| **Static dims** | 6 | 3 | 50.0% |
| **Peak dims** | 6 | 2 | 66.7% |
| **Total size** | 54 MB | 35 MB | 35.2% |

## Data Quality Assurance

### ✅ All Quality Checks Passed
- **Signal length consistency**: All 200 timestamps
- **Static parameter dimensions**: All 3 dimensions
- **Peak dimensions**: All 2 peak types
- **Data integrity**: 95 samples had NaN/Inf issues (fixed automatically)
- **V peak validity**: 99.3% valid peak data

### Data Types
- **patient_id**: Python string
- **static_params**: numpy.float32 array (3,)
- **signal**: numpy.float32 array (200,)
- **peaks**: numpy.float32 array (2,) - may contain NaN for missing peaks
- **peak_mask**: numpy.bool array (2,)

## Usage Examples

### Loading the Data
```python
import joblib
import numpy as np

# Load the data
data = joblib.load('data/processed/processed_data_normal_only_200ts.pkl')
print(f"Loaded {len(data)} Normal hearing loss samples")

# Access a sample
sample = data[0]
print(f"Patient ID: {sample['patient_id']}")
print(f"Static params (Age, Intensity, Rate): {sample['static_params']}")
print(f"Signal shape: {sample['signal'].shape}")
print(f"V peaks (Latency, Amplitude): {sample['peaks']}")
print(f"V peak validity: {sample['peak_mask']}")
```

### Working with the Simplified Structure
```python
# Extract static parameters
ages = [sample['static_params'][0] for sample in data]
intensities = [sample['static_params'][1] for sample in data]
rates = [sample['static_params'][2] for sample in data]

# Extract V peak data
v_latencies = []
v_amplitudes = []
for sample in data:
    if sample['peak_mask'][0]:  # V Latency valid
        v_latencies.append(sample['peaks'][0])
    if sample['peak_mask'][1]:  # V Amplitude valid
        v_amplitudes.append(sample['peaks'][1])

print(f"Valid V Latencies: {len(v_latencies)}")
print(f"Valid V Amplitudes: {len(v_amplitudes)}")
```

### Loading the Scaler
```python
# Load the scaler for denormalization if needed
scaler = joblib.load('data/processed/scaler_normal_only.pkl')

# Denormalize static parameters
sample_static = data[0]['static_params'].reshape(1, -1)
original_values = scaler.inverse_transform(sample_static)[0]
print(f"Original values - Age: {original_values[0]:.1f}, Intensity: {original_values[1]:.1f}, Rate: {original_values[2]:.1f}")
```

## Model Architecture Implications

### Recommended Model Configuration
```json
{
  "model": {
    "architecture": {
      "static_dim": 3,
      "signal_length": 200,
      "predict_peaks": true,
      "num_peaks": 2,
      "peak_types": ["V_Latency", "V_Amplitude"]
    }
  }
}
```

### Benefits of Simplified Structure

1. **Reduced Model Complexity**: 
   - Input dimensions: 3 static + 200 signal = 203 (vs 206 original)
   - Output dimensions: 200 signal + 2 peaks = 202 (vs 206 original)

2. **Faster Training**:
   - 81% fewer samples to process
   - Simpler parameter space
   - Higher peak validity reduces loss computation complexity

3. **Better Quality Data**:
   - Only high-quality samples (FMP > 3.0)
   - Homogeneous hearing loss type (Normal only)
   - 99.3% valid V peak data

4. **Memory Efficiency**:
   - 35% reduction in file size
   - Simpler data structures
   - Faster loading and processing

## Comparison with Previous Formats

### vs. Original Categorical Dataset
| Aspect | Original | Normal-Only | Change |
|--------|----------|-------------|---------|
| **Samples** | 51,999 | 9,683 | -81.4% |
| **Hearing Loss Types** | 5 categories | Normal only | -80% |
| **Static Dimensions** | 6 | 3 | -50% |
| **Peak Dimensions** | 6 | 2 | -67% |
| **Peak Validity** | ~75% (V peaks) | 99.3% | +24% |
| **File Size** | 54 MB | 35 MB | -35% |

### vs. One-Hot Encoded Dataset
| Aspect | One-Hot | Normal-Only | Change |
|--------|---------|-------------|---------|
| **Static Dimensions** | 10 | 3 | -70% |
| **Memory per Sample** | Higher | Lower | Significant |
| **Complexity** | Higher | Lower | Simplified |

## Preprocessing Function

The dataset was created using the new preprocessing function:

```python
from utils.preprocessing import preprocess_and_save_normal_only

preprocess_and_save_normal_only(
    file_path="data/abr_dataset.xlsx",
    output_dir="data/processed",
    verbose=True,
    signal_length=200
)
```

## Use Cases

This simplified dataset is ideal for:

1. **Proof of Concept Models**: Faster iteration with reduced complexity
2. **Normal Hearing Baseline**: Understanding normal ABR patterns
3. **High-Quality Training**: Clean data with minimal missing values
4. **Resource-Constrained Environments**: Smaller memory footprint
5. **V Peak Analysis**: Focus on the most reliable peak measurements

## Next Steps

1. **Train baseline models** on this simplified dataset
2. **Compare performance** with full categorical dataset
3. **Benchmark training speed** improvements
4. **Evaluate model quality** on Normal hearing samples
5. **Consider expansion** to other hearing loss types if needed

---

**Status**: ✅ **Analysis Complete**  
**Format**: Normal hearing loss only with simplified structure  
**Dimensions**: 3 static + 200 signal + 2 peaks  
**Quality**: High-quality filtered data with 99.3% V peak validity  
**Efficiency**: 35% smaller file size with 81% fewer samples  
**Focus**: Optimized for V peak analysis and normal hearing patterns 