import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Any, Tuple
import joblib
from collections import Counter
import os


def clean_numerical_data(data: np.ndarray, data_type: str = "signal") -> Tuple[np.ndarray, int]:
    """
    Clean NaN and Inf values from numerical data.
    
    Args:
        data (np.ndarray): Input data array
        data_type (str): Type of data for logging ("signal" or "static_params")
        
    Returns:
        Tuple[np.ndarray, int]: Cleaned data and number of fixes applied
    """
    fixes_applied = 0
    
    # Check for NaN values
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=0.0)
        fixes_applied += 1
    
    # Check for Inf values
    if np.isinf(data).any():
        data = np.nan_to_num(data, posinf=1.0, neginf=-1.0)
        fixes_applied += 1
    
    return data, fixes_applied


def load_and_preprocess_dataset(
    file_path: str, 
    verbose: bool = True, 
    save_transformers: bool = False,
    signal_length: int = 200
) -> Tuple[List[Dict[str, Any]], StandardScaler, LabelEncoder]:
    """
    Load and preprocess ABR dataset with comprehensive data preparation and cleaning.
    Uses categorical encoding (1-5) for hearing loss instead of one-hot encoding.
    
    Args:
        file_path (str): Path to the CSV or Excel file containing ABR data
        verbose (bool): If True, print detailed information about the dataset
        save_transformers (bool): If True, save fitted transformers to disk
        signal_length (int): Number of timestamps to use from the signal (default: 200)
        
    Returns:
        Tuple containing:
        - data: List of dictionaries with preprocessed samples
        - scaler: Fitted StandardScaler for continuous parameters
        - label_encoder: Fitted LabelEncoder for categorical parameters
    """
    # Load the dataset (support both CSV and Excel files)
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.csv':
        df: pd.DataFrame = pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        df: pd.DataFrame = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Please use CSV or Excel files.")
    
    # Apply filters
    filtered_df: pd.DataFrame = df[
        (df['Stimulus Polarity'] == 'Alternate') & 
        (df['Sweeps Rejected'] <= 100)
    ].copy()
    
    # Define column groups
    continuous_params: List[str] = ['Age', 'Intensity', 'Stimulus Rate', 'FMP', 'ResNo']
    categorical_params: List[str] = ['Hear_Loss']
    peak_columns: List[str] = ['I Latancy', 'III Latancy', 'V Latancy', 'I Amplitude', 'III Amplitude', 'V Amplitude']
    
    # Update signal columns to use only first signal_length timestamps
    signal_columns: List[str] = [str(i) for i in range(1, signal_length + 1)]  # "1" through "signal_length"
    
    # Validate that we have enough signal columns
    available_columns = [str(i) for i in range(1, 468)]
    if signal_length > len(available_columns):
        raise ValueError(f"Requested signal_length ({signal_length}) exceeds available columns ({len(available_columns)})")
    
    # Robust category handling for Hear_Loss
    filtered_df['Hear_Loss'] = filtered_df['Hear_Loss'].astype(str).str.strip().str.capitalize()
    
    # Extract patient IDs
    patient_ids: np.ndarray = filtered_df['Patient_ID'].values
    unique_patients: np.ndarray = np.unique(patient_ids)
    
    # Verbose logging
    if verbose:
        print(f"📊 Dataset Statistics:")
        print(f"   Total samples: {len(filtered_df)}")
        print(f"   Unique patients: {len(unique_patients)}")
        print(f"   Signal length: {signal_length} timestamps (reduced from 467)")
        
        # Distribution of Hear_Loss
        hear_loss_dist = Counter(filtered_df['Hear_Loss'].values)
        print(f"   Hear_Loss distribution: {dict(hear_loss_dist)}")
        
        # Check for missing peak values
        peak_data_temp = filtered_df[peak_columns].copy()
        for col in peak_columns:
            peak_data_temp[col] = peak_data_temp[col].replace(['', -1], np.nan)
            peak_data_temp[col] = pd.to_numeric(peak_data_temp[col], errors='coerce')
        
        missing_peaks = peak_data_temp.isna().any(axis=1).sum()
        print(f"   Samples with missing peak values: {missing_peaks}")
        print()
    
    # Process continuous parameters
    continuous_data: np.ndarray = filtered_df[continuous_params].values.astype(np.float32)
    scaler: StandardScaler = StandardScaler()
    normalized_continuous: np.ndarray = scaler.fit_transform(continuous_data)
    
    # Process categorical parameters using LabelEncoder (1-based indexing)
    categorical_data: np.ndarray = filtered_df[categorical_params].values.flatten()
    label_encoder: LabelEncoder = LabelEncoder()
    encoded_categorical: np.ndarray = label_encoder.fit_transform(categorical_data)
    
    # Convert to 1-based indexing (1, 2, 3, 4, 5 instead of 0, 1, 2, 3, 4)
    encoded_categorical = encoded_categorical + 1
    
    # Reshape to column vector and convert to float32
    encoded_categorical = encoded_categorical.reshape(-1, 1).astype(np.float32)
    
    # Combine static parameters (5 continuous + 1 categorical = 6 dimensions)
    static_params: np.ndarray = np.concatenate([normalized_continuous, encoded_categorical], axis=1).astype(np.float32)
    
    # Process peak values
    peak_data: pd.DataFrame = filtered_df[peak_columns].copy()
    
    # Handle missing values in peaks (convert empty strings, NaNs, or -1 to np.nan)
    for col in peak_columns:
        peak_data[col] = peak_data[col].replace(['', -1], np.nan)
        peak_data[col] = pd.to_numeric(peak_data[col], errors='coerce')
    
    peak_values: np.ndarray = peak_data.values.astype(np.float32)
    peak_masks: np.ndarray = ~np.isnan(peak_values)  # True where values are present
    
    # Process ABR signals (using only first signal_length timestamps)
    signal_data: np.ndarray = filtered_df[signal_columns].values.astype(np.float32)
    
    # Vectorized signal normalization
    signal_means: np.ndarray = np.mean(signal_data, axis=1, keepdims=True)
    signal_stds: np.ndarray = np.std(signal_data, axis=1, keepdims=True)
    # Add small epsilon to prevent division by zero
    epsilon: float = 1e-8
    signal_stds = signal_stds + epsilon
    normalized_signals: np.ndarray = (signal_data - signal_means) / signal_stds
    
    # Data cleaning counters
    total_signal_fixes = 0
    total_static_fixes = 0
    samples_with_issues = 0
    
    # Create sample dictionaries with data cleaning
    samples: List[Dict[str, Any]] = []
    for i in range(len(filtered_df)):
        # Get original data
        signal = normalized_signals[i]
        static_params_sample = static_params[i]
        
        # Clean signal data
        signal_cleaned, signal_fixes = clean_numerical_data(signal, "signal")
        
        # Clean static parameters
        static_cleaned, static_fixes = clean_numerical_data(static_params_sample, "static_params")
        
        # Update counters
        if signal_fixes > 0 or static_fixes > 0:
            samples_with_issues += 1
        total_signal_fixes += signal_fixes
        total_static_fixes += static_fixes
        
        sample: Dict[str, Any] = {
            'patient_id': str(patient_ids[i]),
            'static_params': static_cleaned,
            'signal': signal_cleaned,
            'peaks': peak_values[i],
            'peak_mask': peak_masks[i]
        }
        samples.append(sample)
    
    # Return all preprocessed samples
    data: List[Dict[str, Any]] = samples
    
    if verbose:
        print(f"🧹 Data Cleaning Summary:")
        print(f"   Samples with NaN/Inf issues: {samples_with_issues}")
        print(f"   Signal fixes applied: {total_signal_fixes}")
        print(f"   Static parameter fixes applied: {total_static_fixes}")
        print()
        
        print(f"✅ Preprocessing Complete:")
        print(f"   Total preprocessed samples: {len(data)}")
        print(f"   Signal length: {signal_length} timestamps")
        print(f"   Static parameters: 6 dimensions (5 continuous + 1 categorical)")
        print(f"   Hearing loss categories: {len(label_encoder.classes_)} (encoded as 1-{len(label_encoder.classes_)})")
        print(f"   Category mapping: {dict(zip(label_encoder.classes_, range(1, len(label_encoder.classes_) + 1)))}")
        print()
    
    # Save transformers if requested
    if save_transformers:
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoder, 'label_encoder.pkl')
        if verbose:
            print("💾 Saved transformers:")
            print("   - scaler.pkl")
            print("   - label_encoder.pkl")
            print()
    
    return data, scaler, label_encoder


def preprocess_and_save(
    file_path: str, 
    output_dir: str = "data/processed", 
    verbose: bool = True,
    signal_length: int = 200
) -> None:
    """
    Preprocess ABR dataset and save all outputs to disk.
    
    Args:
        file_path (str): Path to the raw CSV or Excel file
        output_dir (str): Directory to save processed files (default: "data/processed")
        verbose (bool): If True, print detailed information about the process
        signal_length (int): Number of timestamps to use from the signal (default: 200)
    """
    # Call the preprocessing function
    data, scaler, label_encoder = load_and_preprocess_dataset(
        file_path=file_path, 
        verbose=verbose, 
        save_transformers=False,  # We'll save them ourselves to the output_dir
        signal_length=signal_length
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file paths with signal length in filename
    data_path = os.path.join(output_dir, f"processed_data_categorical_{signal_length}ts.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    label_encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    
    # Save all files
    joblib.dump(data, data_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, label_encoder_path)
    
    # Verbose logging
    if verbose:
        print(f"💾 Saved processed files to '{output_dir}':")
        print(f"   - processed_data_categorical_{signal_length}ts.pkl ({len(data)} samples)")
        print(f"   - scaler.pkl")
        print(f"   - label_encoder.pkl")
        print(f"✅ Preprocessing and saving complete!")
        print()


def reprocess_existing_data_categorical(
    input_file: str = "data/processed/processed_data_clean_200ts.pkl",
    output_file: str = "data/processed/processed_data_categorical_200ts.pkl",
    verbose: bool = True
) -> None:
    """
    Reprocess existing data to convert from one-hot encoding to categorical encoding.
    
    Args:
        input_file (str): Path to existing processed data file with one-hot encoding
        output_file (str): Path to save data with categorical encoding
        verbose (bool): If True, print detailed information
    """
    if verbose:
        print(f"🔄 Converting one-hot to categorical encoding...")
        print(f"   Input: {input_file}")
        print(f"   Output: {output_file}")
        print()
    
    # Load existing data
    try:
        data = joblib.load(input_file)
        if verbose:
            print(f"✅ Loaded {len(data)} samples from {input_file}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load the original label encoder to get category mapping
    try:
        onehot_encoder = joblib.load("data/processed/onehot_encoder.pkl")
        categories = onehot_encoder.categories_[0]
        if verbose:
            print(f"✅ Loaded original categories: {categories}")
    except FileNotFoundError:
        # If no original encoder, create mapping from the data
        categories = ['Normal', 'Nöropati̇', 'Sni̇k', 'Total', 'İti̇k']
        if verbose:
            print(f"⚠️  Using default categories: {categories}")
    
    # Create new label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(categories)
    
    # Process each sample
    converted_data = []
    for i, sample in enumerate(data):
        # Get original static parameters (10 dimensions: 5 continuous + 5 one-hot)
        static_params = sample['static_params']
        
        # Extract continuous parameters (dimensions 0-4)
        continuous_params = static_params[:5]
        
        # Extract one-hot encoded part (dimensions 5-9)
        onehot_part = static_params[5:10]
        
        # Convert one-hot to categorical (find which category is active)
        category_index = np.where(onehot_part == 1)[0]
        if len(category_index) > 0:
            # Convert to 1-based indexing
            categorical_value = category_index[0] + 1
        else:
            # Default to category 1 if no category is active
            categorical_value = 1
            if verbose and i < 5:  # Only warn for first few samples
                print(f"⚠️  Sample {i}: No active category found, defaulting to 1")
        
        # Create new static parameters (6 dimensions: 5 continuous + 1 categorical)
        new_static_params = np.concatenate([
            continuous_params, 
            np.array([categorical_value], dtype=np.float32)
        ])
        
        # Create new sample
        new_sample = {
            'patient_id': sample['patient_id'],
            'static_params': new_static_params,
            'signal': sample['signal'],
            'peaks': sample['peaks'],
            'peak_mask': sample['peak_mask']
        }
        converted_data.append(new_sample)
    
    # Save converted data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    joblib.dump(converted_data, output_file)
    
    # Save the new label encoder
    label_encoder_path = os.path.join(os.path.dirname(output_file), "label_encoder.pkl")
    joblib.dump(label_encoder, label_encoder_path)
    
    if verbose:
        print(f"✅ Conversion Complete:")
        print(f"   Total processed samples: {len(converted_data)}")
        print(f"   Static parameters reduced: 10 → 6 dimensions")
        print(f"   Categorical encoding: 1-{len(categories)} (1-based)")
        print(f"   Category mapping: {dict(zip(categories, range(1, len(categories) + 1)))}")
        print(f"   Saved to: {output_file}")
        print(f"   Label encoder saved to: {label_encoder_path}")
        print()
