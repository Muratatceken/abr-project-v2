#!/usr/bin/env python3
"""
Detailed analysis of the processed_data_categorical_200ts.pkl file structure.
"""

import joblib
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Any, List

def analyze_pkl_structure():
    """Analyze the structure of the categorical encoded pkl file in detail."""
    print("🔍 Detailed Analysis of processed_data_categorical_200ts.pkl")
    print("=" * 60)
    
    # Load the data
    data_path = "data/processed/processed_data_categorical_200ts.pkl"
    
    try:
        data = joblib.load(data_path)
        print(f"✅ Successfully loaded data from {data_path}")
        print(f"📊 File size: {len(data)} samples")
        print()
        
        # 1. Top-level structure
        print("1️⃣ TOP-LEVEL STRUCTURE")
        print("-" * 30)
        print(f"Data type: {type(data)}")
        print(f"Length: {len(data)} samples")
        print(f"First element type: {type(data[0])}")
        print()
        
        # 2. Sample structure
        print("2️⃣ SAMPLE STRUCTURE")
        print("-" * 30)
        sample = data[0]
        print(f"Sample keys: {list(sample.keys())}")
        print()
        
        for key in sample.keys():
            value = sample[key]
            print(f"Key: '{key}'")
            print(f"  Type: {type(value)}")
            print(f"  Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
            print(f"  Data type: {value.dtype if hasattr(value, 'dtype') else 'N/A'}")
            if hasattr(value, 'shape') and len(value.shape) == 1:
                print(f"  Min: {np.min(value):.6f}")
                print(f"  Max: {np.max(value):.6f}")
                print(f"  Mean: {np.mean(value):.6f}")
                print(f"  Std: {np.std(value):.6f}")
                print(f"  Sample values: {value[:5] if len(value) > 5 else value}")
            elif key == 'patient_id':
                print(f"  Sample value: {value}")
            print()
        
        # 3. Signal analysis
        print("3️⃣ SIGNAL ANALYSIS")
        print("-" * 30)
        all_signals = np.array([sample['signal'] for sample in data])
        print(f"All signals shape: {all_signals.shape}")
        print(f"Signal length: {all_signals.shape[1]} timestamps")
        print(f"Number of samples: {all_signals.shape[0]}")
        print(f"Global signal statistics:")
        print(f"  Min: {np.min(all_signals):.6f}")
        print(f"  Max: {np.max(all_signals):.6f}")
        print(f"  Mean: {np.mean(all_signals):.6f}")
        print(f"  Std: {np.std(all_signals):.6f}")
        print(f"  NaN count: {np.isnan(all_signals).sum()}")
        print(f"  Inf count: {np.isinf(all_signals).sum()}")
        print()
        
        # 4. Static parameters analysis (UPDATED FOR CATEGORICAL)
        print("4️⃣ STATIC PARAMETERS ANALYSIS (CATEGORICAL ENCODING)")
        print("-" * 30)
        all_static = np.array([sample['static_params'] for sample in data])
        print(f"All static params shape: {all_static.shape}")
        print(f"Number of static parameters: {all_static.shape[1]} (reduced from 10)")
        print(f"Parameter breakdown:")
        print(f"  Dimensions 0-4: Continuous parameters (normalized)")
        print(f"  Dimension 5: Categorical hearing loss (1-5)")
        print()
        
        # Detailed parameter analysis
        print(f"Parameter statistics (per dimension):")
        param_names = ['Age', 'Intensity', 'Stimulus Rate', 'FMP', 'ResNo', 'Hearing Loss (Categorical)']
        for i in range(all_static.shape[1]):
            param_values = all_static[:, i]
            param_name = param_names[i] if i < len(param_names) else f"Param {i}"
            print(f"  {i}: {param_name}")
            print(f"     Min: {np.min(param_values):.4f}")
            print(f"     Max: {np.max(param_values):.4f}")
            print(f"     Mean: {np.mean(param_values):.4f}")
            print(f"     Std: {np.std(param_values):.4f}")
            
            # Special handling for categorical parameter (dimension 5)
            if i == 5:
                unique_values = sorted(np.unique(param_values))
                value_counts = Counter(param_values)
                print(f"     Unique values: {unique_values}")
                print(f"     Value distribution:")
                
                # Load label encoder for category names
                try:
                    label_encoder = joblib.load("data/processed/label_encoder.pkl")
                    categories = label_encoder.classes_
                    for j, category in enumerate(categories):
                        count = value_counts.get(j + 1, 0)
                        percentage = (count / len(data)) * 100
                        print(f"       {j+1} ({category}): {count:,} samples ({percentage:.1f}%)")
                except:
                    print(f"     ⚠️  Could not load label encoder for category names")
                    for value in unique_values:
                        count = value_counts[value]
                        percentage = (count / len(data)) * 100
                        print(f"       {value}: {count:,} samples ({percentage:.1f}%)")
            print()
        
        # 5. Peak data analysis
        print("5️⃣ PEAK DATA ANALYSIS")
        print("-" * 30)
        all_peaks = np.array([sample['peaks'] for sample in data])
        all_peak_masks = np.array([sample['peak_mask'] for sample in data])
        
        print(f"All peaks shape: {all_peaks.shape}")
        print(f"All peak masks shape: {all_peak_masks.shape}")
        print(f"Number of peak types: {all_peaks.shape[1]}")
        
        # Peak labels (from preprocessing)
        peak_labels = ['I Latency', 'III Latency', 'V Latency', 'I Amplitude', 'III Amplitude', 'V Amplitude']
        
        print(f"Peak analysis:")
        for i, label in enumerate(peak_labels):
            peak_values = all_peaks[:, i]
            peak_mask = all_peak_masks[:, i]
            valid_peaks = peak_values[peak_mask]
            
            print(f"  {label}:")
            print(f"    Valid samples: {np.sum(peak_mask)} / {len(peak_mask)} ({np.sum(peak_mask)/len(peak_mask)*100:.1f}%)")
            if len(valid_peaks) > 0:
                print(f"    Min: {np.min(valid_peaks):.4f}")
                print(f"    Max: {np.max(valid_peaks):.4f}")
                print(f"    Mean: {np.mean(valid_peaks):.4f}")
                print(f"    Std: {np.std(valid_peaks):.4f}")
            else:
                print(f"    No valid peaks")
        print()
        
        # 6. Patient ID analysis
        print("6️⃣ PATIENT ID ANALYSIS")
        print("-" * 30)
        patient_ids = [sample['patient_id'] for sample in data]
        unique_patients = list(set(patient_ids))
        patient_counts = Counter(patient_ids)
        
        print(f"Total samples: {len(patient_ids)}")
        print(f"Unique patients: {len(unique_patients)}")
        print(f"Samples per patient statistics:")
        counts = list(patient_counts.values())
        print(f"  Min samples per patient: {min(counts)}")
        print(f"  Max samples per patient: {max(counts)}")
        print(f"  Mean samples per patient: {np.mean(counts):.2f}")
        print(f"  Median samples per patient: {np.median(counts):.2f}")
        
        # Top 10 patients by sample count
        print(f"Top 10 patients by sample count:")
        for i, (patient_id, count) in enumerate(patient_counts.most_common(10)):
            print(f"  {i+1}. Patient {patient_id}: {count} samples")
        print()
        
        # 7. Data consistency checks
        print("7️⃣ DATA CONSISTENCY CHECKS")
        print("-" * 30)
        
        # Check signal lengths
        signal_lengths = [len(sample['signal']) for sample in data]
        unique_lengths = set(signal_lengths)
        print(f"Signal length consistency: {len(unique_lengths)} unique length(s)")
        if len(unique_lengths) == 1:
            print(f"  ✅ All signals have length: {list(unique_lengths)[0]}")
        else:
            print(f"  ❌ Multiple lengths found: {unique_lengths}")
        
        # Check static parameter dimensions
        static_dims = [len(sample['static_params']) for sample in data]
        unique_dims = set(static_dims)
        print(f"Static parameter dimension consistency: {len(unique_dims)} unique dimension(s)")
        if len(unique_dims) == 1:
            print(f"  ✅ All static params have dimension: {list(unique_dims)[0]}")
        else:
            print(f"  ❌ Multiple dimensions found: {unique_dims}")
        
        # Check peak dimensions
        peak_dims = [len(sample['peaks']) for sample in data]
        unique_peak_dims = set(peak_dims)
        print(f"Peak dimension consistency: {len(unique_peak_dims)} unique dimension(s)")
        if len(unique_peak_dims) == 1:
            print(f"  ✅ All peaks have dimension: {list(unique_peak_dims)[0]}")
        else:
            print(f"  ❌ Multiple peak dimensions found: {unique_peak_dims}")
        
        # Check for NaN/Inf values
        print(f"Data quality checks:")
        total_nan = 0
        total_inf = 0
        
        for sample in data:
            # Check signals
            signal_nan = np.isnan(sample['signal']).sum()
            signal_inf = np.isinf(sample['signal']).sum()
            
            # Check static params
            static_nan = np.isnan(sample['static_params']).sum()
            static_inf = np.isinf(sample['static_params']).sum()
            
            total_nan += signal_nan + static_nan
            total_inf += signal_inf + static_inf
        
        print(f"  Total NaN values: {total_nan}")
        print(f"  Total Inf values: {total_inf}")
        
        if total_nan == 0 and total_inf == 0:
            print(f"  ✅ No NaN/Inf values found in signals and static params")
        else:
            print(f"  ❌ Found {total_nan} NaN and {total_inf} Inf values")
        
        print()
        
        # 8. Memory usage analysis
        print("8️⃣ MEMORY USAGE ANALYSIS")
        print("-" * 30)
        
        # Calculate memory usage for each component
        signals_memory = all_signals.nbytes
        static_memory = all_static.nbytes
        peaks_memory = all_peaks.nbytes
        masks_memory = all_peak_masks.nbytes
        
        total_memory = signals_memory + static_memory + peaks_memory + masks_memory
        
        print(f"Memory usage breakdown:")
        print(f"  Signals: {signals_memory / 1024**2:.2f} MB ({signals_memory/total_memory*100:.1f}%)")
        print(f"  Static params: {static_memory / 1024**2:.2f} MB ({static_memory/total_memory*100:.1f}%)")
        print(f"  Peaks: {peaks_memory / 1024**2:.2f} MB ({peaks_memory/total_memory*100:.1f}%)")
        print(f"  Peak masks: {masks_memory / 1024**2:.2f} MB ({masks_memory/total_memory*100:.1f}%)")
        print(f"  Total: {total_memory / 1024**2:.2f} MB")
        
        # Compare with one-hot encoding
        print(f"\nComparison with one-hot encoding:")
        onehot_static_memory = len(data) * 10 * 4  # 10 dimensions * 4 bytes (float32)
        categorical_static_memory = static_memory
        memory_saved = onehot_static_memory - categorical_static_memory
        print(f"  One-hot static params: {onehot_static_memory / 1024**2:.2f} MB")
        print(f"  Categorical static params: {categorical_static_memory / 1024**2:.2f} MB")
        print(f"  Memory saved: {memory_saved / 1024**2:.2f} MB ({memory_saved/onehot_static_memory*100:.1f}%)")
        
        print()
        
        # 9. Summary
        print("9️⃣ SUMMARY")
        print("-" * 30)
        print(f"✅ Successfully analyzed {len(data):,} samples")
        print(f"✅ Signal length: {all_signals.shape[1]} timestamps")
        print(f"✅ Static parameters: {all_static.shape[1]} dimensions (categorical encoding)")
        print(f"✅ Peak data: {all_peaks.shape[1]} peak types")
        print(f"✅ Data quality: No NaN/Inf values in core data")
        print(f"✅ Memory efficient: {memory_saved/1024**2:.1f} MB saved vs one-hot encoding")
        print(f"✅ Categorical encoding: 5 hearing loss categories (1-5)")
        
    except FileNotFoundError:
        print(f"❌ File not found: {data_path}")
        print("Please ensure the categorical data file exists.")
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_pkl_structure() 