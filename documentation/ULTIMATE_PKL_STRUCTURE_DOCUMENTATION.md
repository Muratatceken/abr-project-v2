# Ultimate ABR Dataset - PKL File Structure Documentation

## 📋 Overview

This document provides comprehensive documentation for the **Ultimate ABR Dataset** (`ultimate_dataset.pkl`), which represents the most streamlined and production-ready version of the ABR (Auditory Brainstem Response) dataset for machine learning applications.

## 📁 File Information

- **File Path**: `data/processed/ultimate_dataset.pkl`
- **File Size**: 93.5 MB
- **Total Samples**: 51,961
- **Version**: ultimate_v1
- **Creation Date**: January 2025
- **Description**: Ultimate ABR dataset with simplified structure and optimized filtering

## 🔍 Filtering Criteria Applied

The dataset includes only samples that meet the following strict criteria:

1. **Stimulus Polarity**: `'Alternate'` only

   - Original samples: 55,237
   - After polarity filter: 53,280 samples (96.5%)
2. **Sweeps Rejected**: `< 100`

   - After sweep rejection filter: 51,961 samples (94.1% of original)

## 📊 Dataset Structure

### Sample Keys

Each sample in the dataset contains the following keys:

- `patient_id`: Unique patient identifier
- `static_params`: Static parameters (4 features)
- `signal`: ABR time series (200 timestamps)
- `v_peak`: V peak data (2 features: latency + amplitude)
- `v_peak_mask`: V peak validity mask (2 boolean values)
- `target`: Hearing loss type (integer encoded)

### Data Shapes

- **Static Parameters**: `(4,)` - Age, Intensity, Stimulus Rate, FMP
- **Signal**: `(200,)` - ABR waveform timestamps
- **V Peak**: `(2,)` - V Latency and V Amplitude
- **V Peak Mask**: `(2,)` - Boolean validity indicators
- **Target**: `scalar` - Hearing loss classification (0-4)

## 🎯 Feature Details

### Static Parameters (4 features)

| Index | Parameter     | Description                   | Preprocessing             |
| ----- | ------------- | ----------------------------- | ------------------------- |
| 0     | Age           | Patient age in years          | StandardScaler normalized |
| 1     | Intensity     | Stimulus intensity in dB      | StandardScaler normalized |
| 2     | Stimulus Rate | Rate of stimulus presentation | StandardScaler normalized |
| 3     | FMP           | Figure of Merit Parameter     | StandardScaler normalized |

### Signal Data (200 timestamps)

- **Length**: 200 time points
- **Preprocessing**: Z-score normalization per sample
- **Original columns**: Excel columns "1" through "200"
- **No denoising applied** (as requested)

### V Peak Data (5th Peak)

| Index | Feature     | Description             | Masking        |
| ----- | ----------- | ----------------------- | -------------- |
| 0     | V Latency   | Peak V latency in ms    | v_peak_mask[0] |
| 1     | V Amplitude | Peak V amplitude in μV | v_peak_mask[1] |

### Target Variable (Hearing Loss Classification)

| Index | Class Name | Sample Count | Percentage |
| ----- | ---------- | ------------ | ---------- |
| 0     | NORMAL     | 41,391       | 79.7%      |
| 1     | NÖROPATİ | 417          | 0.8%       |
| 2     | SNİK      | 5,513        | 10.6%      |
| 3     | TOTAL      | 1,715        | 3.3%       |
| 4     | İTİK     | 2,925        | 5.6%       |

## 📈 Data Quality Statistics

### V Peak Validity

- **V Latency valid**: 38,754 / 51,961 samples (74.6%)
- **V Amplitude valid**: 38,754 / 51,961 samples (74.6%)
- **Both V peaks valid**: 38,754 / 51,961 samples (74.6%)

### Missing Data Handling

- **Static Parameters**: NaN values replaced with 0.0 and normalized
- **Signal Data**: Infinite values replaced with 0.0
- **V Peak Data**: NaN values replaced with 0.0, validity tracked via masking
- **Patient IDs**: NaN values replaced with sequential IDs

### Data Cleaning Summary

- **Static parameter fixes**: 1,627 invalid values corrected
- **Time series fixes**: 0 invalid values (clean data)

## 💾 Storage and Memory Usage

- **File Size**: 93.5 MB
- **Size per Sample**: 1.84 KB
- **Memory Efficiency**: Optimized for fast loading and training
- **Format**: Joblib pickle format for Python compatibility

## 🔧 Usage Examples

### Loading the Dataset

```python
import joblib
import numpy as np

# Load the complete dataset
data = joblib.load('data/processed/ultimate_dataset.pkl')

# Extract components
processed_data = data['data']
scaler = data['scaler']
label_encoder = data['label_encoder']
metadata = data['metadata']

print(f"Loaded {len(processed_data)} samples")
print(f"Classes: {label_encoder.classes_}")
```

### Accessing Sample Data

```python
# Get first sample
sample = processed_data[0]

print(f"Patient ID: {sample['patient_id']}")
print(f"Static params shape: {sample['static_params'].shape}")
print(f"Signal shape: {sample['signal'].shape}")
print(f"V peak: {sample['v_peak']}")
print(f"V peak mask: {sample['v_peak_mask']}")
print(f"Target: {sample['target']}")
```

### Preparing Data for PyTorch

```python
import torch
from torch.utils.data import Dataset, DataLoader

class ABRDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        sample = self.data[idx]
      
        # Extract features
        static_params = torch.FloatTensor(sample['static_params'])
        signal = torch.FloatTensor(sample['signal'])
        v_peak = torch.FloatTensor(sample['v_peak'])
        v_peak_mask = torch.BoolTensor(sample['v_peak_mask'])
        target = torch.LongTensor([sample['target']])
      
        return {
            'static_params': static_params,
            'signal': signal,
            'v_peak': v_peak,
            'v_peak_mask': v_peak_mask,
            'target': target
        }

# Create dataset and dataloader
dataset = ABRDataset(processed_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Filtering Samples with Valid V Peaks

```python
# Get samples with both V peaks valid
valid_v_peak_samples = [
    sample for sample in processed_data 
    if sample['v_peak_mask'][0] and sample['v_peak_mask'][1]
]

print(f"Samples with valid V peaks: {len(valid_v_peak_samples)}")
```

### Working with Specific Classes

```python
# Get samples by hearing loss type
normal_samples = [
    sample for sample in processed_data 
    if sample['target'] == 0  # NORMAL class
]

print(f"Normal hearing samples: {len(normal_samples)}")
```

## 🚨 Important Notes

### Class Imbalance

The dataset shows significant class imbalance:

- **NORMAL**: 79.7% (dominant class)
- **SNİK**: 10.6%
- **İTİK**: 5.6%
- **TOTAL**: 3.3%
- **NÖROPATİ**: 0.8% (minority class)

**Recommendations**:

- Use stratified sampling for train/validation splits
- Consider class weighting in loss functions
- Apply appropriate evaluation metrics (F1-score, balanced accuracy)
- Consider oversampling/undersampling techniques

### V Peak Masking

- 25.4% of samples have missing V peak data
- Always check `v_peak_mask` before using V peak values
- Consider imputation strategies or mask-aware model architectures

### Preprocessing Applied

- **Static parameters**: StandardScaler normalization
- **Signals**: Z-score normalization per sample
- **No denoising**: Raw signal data preserved
- **No FMP filtering**: All quality levels included

## 🔄 Preprocessing Pipeline

The dataset was created using the following pipeline:

1. **Data Loading**: Load from `data/abr_dataset.xlsx`
2. **Filtering**: Apply stimulus polarity and sweep rejection filters
3. **Feature Extraction**: Extract 4 static params + 200 timestamps + V peak
4. **Data Cleaning**: Handle NaN/infinite values
5. **Normalization**: StandardScaler for static, Z-score for signals
6. **Peak Masking**: Create validity masks for V peak data
7. **Target Encoding**: LabelEncoder for hearing loss classes
8. **Packaging**: Save with metadata and preprocessing objects

## 📚 Metadata Information

```python
metadata = {
    'version': 'ultimate_v1',
    'description': 'Ultimate ABR dataset with simplified structure',
    'filtering_criteria': {
        'stimulus_polarity': 'Alternate',
        'sweeps_rejected': '< 100'
    },
    'static_parameters': ['Age', 'Intensity', 'Stimulus Rate', 'FMP'],
    'peak_data': ['V Latency', 'V Amplitude'],
    'target': 'Hearing Loss Type',
    'total_samples': 51961,
    'preprocessing_steps': [
        'Alternate polarity filtering',
        'Sweeps rejected < 100 filtering',
        'Static parameter normalization (StandardScaler)',
        'Time series Z-score normalization',
        'V peak masking for missing values',
        'Target label encoding'
    ]
}
```

## 🎯 Model Development Recommendations

### Architecture Considerations

1. **Multi-modal Approach**: Combine static parameters, signals, and V peaks
2. **Attention Mechanisms**: For temporal signal processing
3. **Masking Support**: Handle missing V peak data gracefully
4. **Class Weighting**: Address severe class imbalance

### Training Strategies

1. **Stratified Splits**: Maintain class distribution across sets
2. **Regularization**: High dropout/weight decay due to class imbalance
3. **Evaluation**: Use F1-score, precision-recall curves
4. **Cross-validation**: Stratified k-fold recommended

### Data Augmentation

1. **Signal Augmentation**: Time warping, noise injection
2. **Static Parameter Perturbation**: Small random variations
3. **Balanced Sampling**: Oversample minority classes during training

## ✅ Quality Assurance

- **Data Integrity**: All samples verified for completeness
- **Preprocessing Validation**: Normalization ranges checked
- **Class Distribution**: Verified against original Excel data
- **Memory Efficiency**: Optimized data types used
- **Reproducibility**: Fixed random seeds for consistent results

---

**Generated**: January 2025
**Dataset Version**: ultimate_v1
**Total Samples**: 51,961
**File Size**: 93.5 MB
