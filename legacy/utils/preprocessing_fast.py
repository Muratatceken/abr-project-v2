#!/usr/bin/env python3
"""
ABR Data Preprocessing Module - Fast Version

Optimized preprocessing for faster execution with large datasets.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Any, Tuple, Optional
import warnings
import os

warnings.filterwarnings('ignore')

def clean_data_vectorized(data: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Vectorized data cleaning for better performance.
    
    Args:
        data: Input data array (can be 2D)
        
    Returns:
        Tuple of (cleaned_data, total_fixes)
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
    
    return data, fixes

def load_and_preprocess_ultimate_dataset_fast(
    excel_file: str = "data/abr_dataset.xlsx",
    verbose: bool = True
) -> Tuple[List[Dict], StandardScaler, LabelEncoder]:
    """
    Fast version of ultimate dataset preprocessing.
    """
    if verbose:
        print("🚀 Fast Loading and preprocessing ABR data")
        print("=" * 60)
    
    # Load Excel file
    if verbose:
        print("📂 Loading Excel file...")
    df = pd.read_excel(excel_file)
    if verbose:
        print(f"    Original dataset: {len(df)} samples")
    
    # Apply filtering criteria
    if verbose:
        print("🔍 Applying filtering criteria...")
    
    # Vectorized filtering
    df_filtered = df[
        (df['Stimulus Polarity'] == 'Alternate') & 
        (df['Sweeps Rejected'] < 100)
    ].copy()
    
    if verbose:
        print(f"    After filtering: {len(df_filtered)} samples")
    
    if len(df_filtered) == 0:
        raise ValueError("No samples remaining after filtering!")
    
    # Define columns
    static_columns = ['Age', 'Intensity', 'Stimulus Rate', 'FMP']
    v_latency_col = 'V Latancy'
    v_amplitude_col = 'V Amplitude'
    target_column = 'Hear_Loss'
    time_series_cols = [str(i) for i in range(1, 201)]
    
    if verbose:
        print("📊 Extracting features (vectorized)...")
    
    # Extract all data at once
    static_data = df_filtered[static_columns].values.astype(float)
    time_series_data = df_filtered[time_series_cols].values.astype(float)
    v_latency_data = df_filtered[v_latency_col].values
    v_amplitude_data = df_filtered[v_amplitude_col].values
    target_data = df_filtered[target_column].values
    patient_ids = df_filtered['Patient_ID'].values
    
    if verbose:
        print("🧹 Cleaning data (vectorized)...")
    
    # Vectorized data cleaning
    static_data_clean, static_fixes = clean_data_vectorized(static_data)
    time_series_clean, ts_fixes = clean_data_vectorized(time_series_data)
    
    if verbose:
        print(f"    Fixed {static_fixes} issues in static parameters")
        print(f"    Fixed {ts_fixes} issues in time series data")
    
    # Normalization
    if verbose:
        print("📏 Normalizing data...")
    
    scaler = StandardScaler()
    static_data_normalized = scaler.fit_transform(static_data_clean)
    
    # Vectorized Z-score normalization for time series
    means = np.mean(time_series_clean, axis=1, keepdims=True)
    stds = np.std(time_series_clean, axis=1, keepdims=True)
    stds = np.where(stds > 1e-8, stds, 1.0)  # Avoid division by zero
    time_series_normalized = (time_series_clean - means) / stds
    
    # Target encoding
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target_data)
    
    if verbose:
        print("🎯 Processing V peak data...")
    
    # Vectorized V peak processing
    v_peak_data = np.column_stack([v_latency_data, v_amplitude_data])
    v_peak_mask = ~np.isnan(v_peak_data)
    v_peak_data = np.nan_to_num(v_peak_data, nan=0.0)
    
    if verbose:
        print("📦 Creating dataset structure...")
    
    # Create final dataset
    processed_data = []
    for i in range(len(df_filtered)):
        # Handle patient ID
        patient_id = patient_ids[i]
        if pd.isna(patient_id):
            patient_id = i + 1
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
        print("✅ Fast preprocessing completed!")
        print(f"    Final dataset: {len(processed_data)} samples")
        print(f"    Target distribution:")
        for i, class_name in enumerate(label_encoder.classes_):
            count = np.sum(target_encoded == i)
            print(f"      {class_name}: {count} samples ({count/len(target_encoded)*100:.1f}%)")
    
    return processed_data, scaler, label_encoder

def preprocess_and_save_ultimate_fast(
    excel_file: str = "data/abr_dataset.xlsx",
    output_file: str = "data/processed/ultimate_dataset_fast.pkl",
    verbose: bool = True
) -> None:
    """
    Fast preprocessing and saving.
    """
    if verbose:
        print("⚡ Creating Ultimate ABR Dataset (Fast Version)")
        print("=" * 60)
    
    import time
    start_time = time.time()
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process the data
    processed_data, scaler, label_encoder = load_and_preprocess_ultimate_dataset_fast(
        excel_file=excel_file,
        verbose=verbose
    )
    
    # Prepare data to save
    data_to_save = {
        'data': processed_data,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'metadata': {
            'version': 'ultimate_fast_v1',
            'description': 'Ultimate ABR dataset with optimized processing',
            'filtering_criteria': {
                'stimulus_polarity': 'Alternate',
                'sweeps_rejected': '< 100'
            },
            'static_parameters': ['Age', 'Intensity', 'Stimulus Rate', 'FMP'],
            'peak_data': ['V Latency', 'V Amplitude'],
            'target': 'Hearing Loss Type',
            'total_samples': len(processed_data),
            'preprocessing_steps': [
                'Vectorized filtering',
                'Vectorized data cleaning',
                'Vectorized normalization',
                'Optimized V peak processing'
            ]
        }
    }
    
    # Save the dataset
    if verbose:
        print(f"💾 Saving dataset to {output_file}...")
    
    joblib.dump(data_to_save, output_file)
    
    # Performance metrics
    end_time = time.time()
    processing_time = end_time - start_time
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    
    if verbose:
        print("✅ Fast processing completed!")
        print(f"    Processing time: {processing_time:.2f} seconds")
        print(f"    File size: {file_size:.1f} MB")
        print(f"    Samples per second: {len(processed_data)/processing_time:.0f}")
        print(f"    Location: {output_file}")

if __name__ == "__main__":
    # Demonstrate the performance difference
    preprocess_and_save_ultimate_fast(
        excel_file="data/abr_dataset.xlsx",
        output_file="data/processed/ultimate_dataset_fast.pkl",
        verbose=True
    ) 