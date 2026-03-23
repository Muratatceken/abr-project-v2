#!/usr/bin/env python3
"""
ABR Data Preprocessing Module - Ultimate Version

This module handles preprocessing of ABR (Auditory Brainstem Response) data
for machine learning models with the ultimate simplified structure.

Key Features:
- Direct Excel file processing
- Alternate stimulus polarity filtering
- Sweep rejection filtering (< 100)
- 4 static parameters: Age, Intensity, Stimulus Rate, FMP
- 5th peak extraction: V Latency and V Amplitude with masking
- Hearing loss type as target
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Any, Tuple, Optional
import warnings
import os
import logging

warnings.filterwarnings('ignore')

# Configure logging for preprocessing diagnostics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_signal_quality(signal: np.ndarray, signal_id: str = "unknown") -> dict:
    """
    Validate signal quality and detect degenerate cases that could cause SNR calculation issues.
    
    Args:
        signal: Input signal array
        signal_id: Identifier for logging purposes
        
    Returns:
        Dictionary with quality metrics and validation flags
    """
    quality_info = {
        'signal_id': signal_id,
        'is_all_zero': np.all(signal == 0),
        'is_constant': np.std(signal) < 1e-8,
        'variance': np.var(signal),
        'dynamic_range': np.max(signal) - np.min(signal),
        'has_nan': np.any(np.isnan(signal)),
        'has_inf': np.any(np.isinf(signal)),
        'mean': np.mean(signal),
        'std': np.std(signal)
    }
    
    # Flag degenerate cases
    quality_info['is_degenerate'] = (
        quality_info['is_all_zero'] or 
        quality_info['is_constant'] or
        quality_info['variance'] < 1e-10 or
        quality_info['has_nan'] or
        quality_info['has_inf']
    )
    
    return quality_info


def clean_numerical_data(data: np.ndarray, data_type: str) -> Tuple[np.ndarray, int]:
    """
    Clean numerical data by replacing NaN/Inf values with appropriate substitutes.
    Enhanced with signal quality validation.
    
    Args:
        data: Input data array
        data_type: Type of data for logging ("signal" or "static_params")
        
    Returns:
        Tuple of (cleaned_data, fixes_count)
    """
    fixes = 0
    
    # Replace NaN values with 0
    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        data[nan_mask] = 0.0
        fixes += np.sum(nan_mask)
    
    # Replace infinite values with 0
    inf_mask = np.isinf(data)
    if np.any(inf_mask):
        data[inf_mask] = 0.0
        fixes += np.sum(inf_mask)
    
    # Validate signal quality after cleaning for signal data
    if data_type == "signal":
        quality_info = validate_signal_quality(data, f"{data_type}_sample")
        if quality_info['is_degenerate']:
            logger.warning(f"Degenerate signal detected after cleaning: {quality_info}")
    
    return data, fixes

def load_and_preprocess_ultimate_dataset(
    excel_file: str = "data/abr_dataset.xlsx",
    verbose: bool = True
) -> Tuple[List[Dict], StandardScaler, LabelEncoder]:
    """
    Load and preprocess ABR data with ultimate simplified structure.
    
    Filtering criteria:
    - Stimulus Polarity: 'Alternate' only
    - Sweeps Rejected: < 100
    
    Static parameters (4):
    - Age, Intensity, Stimulus Rate, FMP
    
    Peak data:
    - V Latency and V Amplitude with masking
    
    Target:
    - Hearing Loss Type (categorical)
    
    Args:
        excel_file: Path to Excel file
        verbose: Whether to print progress
        
    Returns:
        Tuple of (processed_data_list, scaler, label_encoder)
    """
    if verbose:
        print("🔄 Loading and preprocessing ABR data (Ultimate Version)")
        print("=" * 60)
    
    # Load Excel file
    if verbose:
        print("📂 Loading Excel file...")
    df = pd.read_excel(excel_file)
    print(f"    Original dataset: {len(df)} samples")
    
    # Apply filtering criteria
    if verbose:
        print("🔍 Applying filtering criteria...")
    
    # Filter 1: Alternate stimulus polarity only
    df_filtered = df[df['Stimulus Polarity'] == 'Alternate'].copy()
    print(f"    After Alternate polarity filter: {len(df_filtered)} samples")
    
    # Filter 2: Sweeps Rejected < 100
    df_filtered = df_filtered[df_filtered['Sweeps Rejected'] < 100].copy()
    print(f"    After Sweeps Rejected < 100 filter: {len(df_filtered)} samples")
    
    if len(df_filtered) == 0:
        raise ValueError("No samples remaining after filtering!")
    
    # Define static parameter columns
    static_columns = ['Age', 'Intensity', 'Stimulus Rate', 'FMP']
    
    # Define peak columns (V peak only - 5th peak)
    v_latency_col = 'V Latancy'  # Note: typo in original data
    v_amplitude_col = 'V Amplitude'
    
    # Define target column
    target_column = 'Hear_Loss'
    
    # Check required columns exist
    required_cols = static_columns + [v_latency_col, v_amplitude_col, target_column]
    missing_cols = [col for col in required_cols if col not in df_filtered.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Get time series columns (200 timestamps)
    time_series_cols = [str(i) for i in range(1, 201)]
    missing_ts_cols = [col for col in time_series_cols if col not in df_filtered.columns]
    if missing_ts_cols:
        raise ValueError(f"Missing time series columns: {missing_ts_cols[:5]}...")
    
    if verbose:
        print("📊 Extracting and processing features...")
    
    # Initialize scalers and encoders
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    # Extract static parameters
    static_data = df_filtered[static_columns].values.astype(float)
    
    # Extract time series data
    time_series_data = df_filtered[time_series_cols].values.astype(float)
    
    # Extract V peak data
    v_latency_data = df_filtered[v_latency_col].values
    v_amplitude_data = df_filtered[v_amplitude_col].values
    
    # Extract target data
    target_data = df_filtered[target_column].values
    
    # Extract patient IDs
    patient_ids = df_filtered['Patient_ID'].values
    
    if verbose:
        print("🧹 Cleaning data...")
    
    # Clean static parameters
    static_data_clean = []
    static_fixes = 0
    for i in range(static_data.shape[0]):
        cleaned_static, fixes = clean_numerical_data(static_data[i].copy(), "static_params")
        static_data_clean.append(cleaned_static)
        static_fixes += fixes
    static_data_clean = np.array(static_data_clean)
    
    # Clean time series data
    time_series_clean = []
    ts_fixes = 0
    for i in range(time_series_data.shape[0]):
        cleaned_ts, fixes = clean_numerical_data(time_series_data[i].copy(), "signal")
        time_series_clean.append(cleaned_ts)
        ts_fixes += fixes
    time_series_clean = np.array(time_series_clean)
    
    if verbose:
        print(f"    Fixed {static_fixes} issues in static parameters")
        print(f"    Fixed {ts_fixes} issues in time series data")
        
    # Additional signal quality statistics
    degenerate_signal_count = 0
    for i in range(time_series_clean.shape[0]):
        quality_info = validate_signal_quality(time_series_clean[i], f"sample_{i}")
        if quality_info['is_degenerate']:
            degenerate_signal_count += 1
    
    if verbose and degenerate_signal_count > 0:
        logger.warning(f"Found {degenerate_signal_count}/{time_series_clean.shape[0]} signals with quality issues")
    
    # Normalize static parameters and time series
    if verbose:
        print("📏 Normalizing data with enhanced validation...")
    
    static_data_normalized = scaler.fit_transform(static_data_clean)
    
    # Enhanced Z-score normalization with comprehensive validation
    time_series_normalized = []
    normalization_stats = {
        'low_variance_count': 0,
        'fallback_normalization_count': 0,
        'successful_normalization_count': 0
    }
    
    for i in range(time_series_clean.shape[0]):
        ts = time_series_clean[i]
        
        # Validate signal quality before normalization
        quality_info = validate_signal_quality(ts, f"sample_{i}")
        
        # Enhanced normalization logic with multiple fallback strategies
        ts_std = np.std(ts)
        ts_mean = np.mean(ts)
        
        if ts_std > 1e-6:  # Increased threshold from 1e-8 to 1e-6 for better stability
            # Standard Z-score normalization
            ts_norm = (ts - ts_mean) / ts_std
            normalization_stats['successful_normalization_count'] += 1
        elif ts_std > 1e-10:  # Very low but non-zero variance
            # Use min-max scaling as fallback
            ts_min, ts_max = np.min(ts), np.max(ts)
            if ts_max - ts_min > 1e-10:
                ts_norm = (ts - ts_min) / (ts_max - ts_min) - 0.5  # Center around 0
                normalization_stats['fallback_normalization_count'] += 1
                logger.warning(f"Sample {i}: Used min-max normalization due to low variance ({ts_std:.2e})")
            else:
                # Signal is effectively constant
                ts_norm = np.zeros_like(ts)
                normalization_stats['low_variance_count'] += 1
                logger.warning(f"Sample {i}: Signal is constant, using zeros")
        else:
            # Signal has zero or near-zero variance
            ts_norm = np.zeros_like(ts)
            normalization_stats['low_variance_count'] += 1
            logger.warning(f"Sample {i}: Zero variance signal ({ts_std:.2e}), using zeros")
        
        # Final validation after normalization
        if np.any(np.isnan(ts_norm)) or np.any(np.isinf(ts_norm)):
            logger.error(f"Sample {i}: Normalization produced NaN/Inf values, using zeros")
            ts_norm = np.zeros_like(ts)
        
        time_series_normalized.append(ts_norm)
    
    time_series_normalized = np.array(time_series_normalized)

    # Compute and store global normalization statistics for denormalization
    global_signal_mean = float(np.mean(time_series_clean))
    global_signal_std = float(np.std(time_series_clean))

    # Log normalization statistics
    if verbose:
        logger.info(f"Normalization statistics: {normalization_stats}")
        logger.info(f"    Successful Z-score: {normalization_stats['successful_normalization_count']}/{time_series_clean.shape[0]}")
        logger.info(f"    Fallback min-max: {normalization_stats['fallback_normalization_count']}/{time_series_clean.shape[0]}")
        logger.info(f"    Zero variance: {normalization_stats['low_variance_count']}/{time_series_clean.shape[0]}")
    
    # Encode target labels
    target_encoded = label_encoder.fit_transform(target_data)
    
    if verbose:
        print("🎯 Processing V peak data...")
    
    # Process V peak data with masking
    v_peak_data = []
    v_peak_mask = []
    
    for i in range(len(df_filtered)):
        v_lat = v_latency_data[i]
        v_amp = v_amplitude_data[i]
        
        # Create V peak array [latency, amplitude]
        v_peak = np.array([v_lat, v_amp], dtype=float)
        
        # Create mask (True if valid, False if NaN)
        v_mask = np.array([not pd.isna(v_lat), not pd.isna(v_amp)], dtype=bool)
        
        # Replace NaN with 0 in peak data
        v_peak = np.nan_to_num(v_peak, nan=0.0)
        
        v_peak_data.append(v_peak)
        v_peak_mask.append(v_mask)
    
    v_peak_data = np.array(v_peak_data)
    v_peak_mask = np.array(v_peak_mask)
    
    if verbose:
        print("📦 Creating dataset structure...")
    
    # Create final dataset
    processed_data = []
    for i in range(len(df_filtered)):
        # Handle patient ID - convert NaN to a placeholder value
        patient_id = patient_ids[i]
        if pd.isna(patient_id):
            patient_id = i + 1  # Use sequential ID if ResNo is NaN
        else:
            patient_id = int(patient_id)
        
        sample = {
            'patient_id': patient_id,
            'static_params': static_data_normalized[i],
            'signal': time_series_normalized[i],
            'v_peak': v_peak_data[i],
            'v_peak_mask': v_peak_mask[i],
            'target': target_encoded[i]
        }
        processed_data.append(sample)
    
    if verbose:
        print("✅ Preprocessing completed successfully!")
        print(f"    Final dataset: {len(processed_data)} samples")
        print(f"    Static parameters: {len(static_columns)} features")
        print(f"    Time series length: {len(time_series_cols)} timestamps")
        print(f"    V peak features: 2 (latency + amplitude)")
        print(f"    Target classes: {len(label_encoder.classes_)}")
        print(f"    Target distribution:")
        for i, class_name in enumerate(label_encoder.classes_):
            count = np.sum(target_encoded == i)
            print(f"      {class_name}: {count} samples ({count/len(target_encoded)*100:.1f}%)")
    
    return processed_data, scaler, label_encoder

def preprocess_and_save_ultimate(
    excel_file: str = "data/abr_dataset.xlsx",
    output_file: str = "data/processed/ultimate_dataset.pkl",
    verbose: bool = True
) -> None:
    """
    Process and save the ultimate ABR dataset.
    
    Args:
        excel_file: Path to input Excel file
        output_file: Path to output PKL file
        verbose: Whether to print progress
    """
    if verbose:
        print("🚀 Creating Ultimate ABR Dataset")
        print("=" * 60)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process the data
    processed_data, scaler, label_encoder = load_and_preprocess_ultimate_dataset(
        excel_file=excel_file,
        verbose=verbose
    )
    
    # Prepare data to save
    data_to_save = {
        'data': processed_data,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'signal_stats': {
            'global_mean': global_signal_mean,
            'global_std': global_signal_std,
        },
        'metadata': {
            'version': 'ultimate_v2',
            'description': 'Ultimate ABR dataset with simplified structure',
            'filtering_criteria': {
                'stimulus_polarity': 'Alternate',
                'sweeps_rejected': '< 100'
            },
            'static_parameters': ['Age', 'Intensity', 'Stimulus Rate', 'FMP'],
            'peak_data': ['V Latency', 'V Amplitude'],
            'target': 'Hearing Loss Type',
            'total_samples': len(processed_data),
            'preprocessing_steps': [
                'Alternate polarity filtering',
                'Sweeps rejected < 100 filtering',
                'Static parameter normalization (StandardScaler)',
                'Time series per-sample Z-score normalization',
                'V peak masking for missing values',
                'Target label encoding'
            ]
        }
    }
    
    # Save the dataset
    if verbose:
        print(f"💾 Saving dataset to {output_file}...")
    
    joblib.dump(data_to_save, output_file)
    
    # Get file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    
    if verbose:
        print("✅ Dataset saved successfully!")
        print(f"    File size: {file_size:.1f} MB")
        print(f"    Location: {output_file}")

def analyze_ultimate_dataset(
    input_file: str = "data/processed/ultimate_dataset.pkl",
    verbose: bool = True
) -> dict:
    """
    Analyze the ultimate ABR dataset structure and statistics.
    
    Args:
        input_file: Path to the PKL file
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary with analysis results
    """
    if verbose:
        print("🔍 Analyzing Ultimate ABR Dataset")
        print("=" * 60)
    
    # Load the dataset
    data = joblib.load(input_file)
    processed_data = data['data']
    scaler = data['scaler']
    label_encoder = data['label_encoder']
    metadata = data['metadata']
    
    if verbose:
        print(f"📋 Dataset Overview:")
        print(f"    Version: {metadata['version']}")
        print(f"    Total samples: {metadata['total_samples']}")
        print(f"    Description: {metadata['description']}")
    
    # Analyze structure
    sample = processed_data[0]
    
    if verbose:
        print(f"\n📊 Data Structure:")
        print(f"    Sample keys: {list(sample.keys())}")
        print(f"    Static params shape: {sample['static_params'].shape}")
        print(f"    Signal shape: {sample['signal'].shape}")
        print(f"    V peak shape: {sample['v_peak'].shape}")
        print(f"    V peak mask shape: {sample['v_peak_mask'].shape}")
        print(f"    Target type: {type(sample['target'])}")
    
    # Analyze V peak statistics
    v_peak_latency_valid = 0
    v_peak_amplitude_valid = 0
    both_valid = 0
    
    for sample in processed_data:
        mask = sample['v_peak_mask']
        if mask[0]:  # V latency valid
            v_peak_latency_valid += 1
        if mask[1]:  # V amplitude valid
            v_peak_amplitude_valid += 1
        if mask[0] and mask[1]:  # Both valid
            both_valid += 1
    
    total_samples = len(processed_data)
    
    if verbose:
        print(f"\n🎯 V Peak Statistics:")
        print(f"    V Latency valid: {v_peak_latency_valid}/{total_samples} ({v_peak_latency_valid/total_samples*100:.1f}%)")
        print(f"    V Amplitude valid: {v_peak_amplitude_valid}/{total_samples} ({v_peak_amplitude_valid/total_samples*100:.1f}%)")
        print(f"    Both V peaks valid: {both_valid}/{total_samples} ({both_valid/total_samples*100:.1f}%)")
    
    # Analyze target distribution
    target_counts = {}
    for i, class_name in enumerate(label_encoder.classes_):
        count = sum(1 for sample in processed_data if sample['target'] == i)
        target_counts[class_name] = count
        if verbose:
            print(f"    {class_name}: {count} samples ({count/total_samples*100:.1f}%)")
    
    # File size analysis
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    if verbose:
        print(f"\n💾 Storage Information:")
        print(f"    File size: {file_size:.1f} MB")
        print(f"    Size per sample: {file_size*1024/total_samples:.2f} KB")
    
    # Return analysis results
    analysis_results = {
        'total_samples': total_samples,
        'static_params_shape': sample['static_params'].shape,
        'signal_shape': sample['signal'].shape,
        'v_peak_shape': sample['v_peak'].shape,
        'v_peak_mask_shape': sample['v_peak_mask'].shape,
        'v_latency_valid_count': v_peak_latency_valid,
        'v_amplitude_valid_count': v_peak_amplitude_valid,
        'both_v_peaks_valid_count': both_valid,
        'target_distribution': target_counts,
        'file_size_mb': file_size,
        'metadata': metadata
    }
    
    if verbose:
        print("\n✅ Analysis completed!")
    
    return analysis_results

if __name__ == "__main__":
    # Create the ultimate dataset
    preprocess_and_save_ultimate(
        excel_file="data/abr_dataset.xlsx",
        output_file="data/processed/ultimate_dataset.pkl",
        verbose=True
    )
    
    # Analyze the created dataset
    analyze_ultimate_dataset(
        input_file="data/processed/ultimate_dataset.pkl",
        verbose=True
    )
