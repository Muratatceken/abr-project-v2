#!/usr/bin/env python3
"""
Debug script to investigate data loading issues causing NaN/Inf values.
"""

import torch
import numpy as np
import json
import joblib
from pathlib import Path
from torch.utils.data import DataLoader

def debug_dataset_loading():
    """Debug the dataset loading process."""
    print("=== Dataset Loading Debug ===")
    
    # Load config
    with open('configs/fixed_config.json', 'r') as f:
        config = json.load(f)
    
    # Import required modules
    from utils.data_utils import create_dataloaders
    from training.dataset import ABRDataset
    
    try:
        # Create dataloaders
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(config, return_peaks=True)
        
        print(f"✅ Dataloaders created successfully")
        print(f"  Train batches: {len(train_dataloader)}")
        print(f"  Val batches: {len(val_dataloader)}")
        print(f"  Test batches: {len(test_dataloader)}")
        
        # Check first few batches
        print("\n=== Checking Training Batches ===")
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= 5:  # Check first 5 batches
                break
                
            print(f"\nBatch {batch_idx}:")
            signal = batch['signal']
            static_params = batch['static_params']
            peaks = batch['peaks']
            peak_mask = batch['peak_mask']
            
            print(f"  Signal shape: {signal.shape}")
            print(f"  Signal dtype: {signal.dtype}")
            print(f"  Signal stats: min={signal.min():.6f}, max={signal.max():.6f}, mean={signal.mean():.6f}")
            print(f"  Signal has NaN: {torch.isnan(signal).any()}")
            print(f"  Signal has Inf: {torch.isinf(signal).any()}")
            print(f"  Signal NaN count: {torch.isnan(signal).sum()}")
            print(f"  Signal Inf count: {torch.isinf(signal).sum()}")
            
            print(f"  Static params shape: {static_params.shape}")
            print(f"  Static params has NaN: {torch.isnan(static_params).any()}")
            print(f"  Static params has Inf: {torch.isinf(static_params).any()}")
            
            print(f"  Peaks shape: {peaks.shape}")
            print(f"  Peaks has NaN: {torch.isnan(peaks).any()}")
            print(f"  Peak mask shape: {peak_mask.shape}")
            print(f"  Peak mask sum: {peak_mask.sum()}")
            
            # If we find NaN/Inf, investigate further
            if torch.isnan(signal).any() or torch.isinf(signal).any():
                print("  🔍 Found NaN/Inf in signal! Investigating...")
                
                # Find which samples have NaN/Inf
                nan_mask = torch.isnan(signal).any(dim=1)
                inf_mask = torch.isinf(signal).any(dim=1)
                
                print(f"  Samples with NaN: {nan_mask.sum()}")
                print(f"  Samples with Inf: {inf_mask.sum()}")
                
                # Show details for problematic samples
                for i in range(signal.shape[0]):
                    if nan_mask[i] or inf_mask[i]:
                        sample_signal = signal[i]
                        print(f"    Sample {i}: NaN={torch.isnan(sample_signal).sum()}, Inf={torch.isinf(sample_signal).sum()}")
                        print(f"      Signal range: [{sample_signal.min():.6f}, {sample_signal.max():.6f}]")
                        
                        # Find positions of NaN/Inf
                        nan_positions = torch.where(torch.isnan(sample_signal))[0]
                        inf_positions = torch.where(torch.isinf(sample_signal))[0]
                        
                        if len(nan_positions) > 0:
                            print(f"      NaN at positions: {nan_positions[:10].tolist()}")
                        if len(inf_positions) > 0:
                            print(f"      Inf at positions: {inf_positions[:10].tolist()}")
                
                return False
        
        print("✅ All checked batches are clean (no NaN/Inf)")
        return True
        
    except Exception as e:
        print(f"❌ Error in dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_raw_data():
    """Debug the raw processed data file."""
    print("\n=== Raw Data Debug ===")
    
    try:
        # Load raw processed data
        data = joblib.load("data/processed/processed_data.pkl")
        print(f"✅ Loaded {len(data)} samples")
        
        # Check for NaN/Inf in raw data
        nan_signals = 0
        inf_signals = 0
        nan_static = 0
        inf_static = 0
        
        for i, sample in enumerate(data):
            signal = sample['signal']
            static_params = sample['static_params']
            
            if np.isnan(signal).any():
                nan_signals += 1
                if nan_signals <= 3:  # Show first few examples
                    print(f"  Sample {i} has NaN in signal: {np.isnan(signal).sum()} NaN values")
            
            if np.isinf(signal).any():
                inf_signals += 1
                if inf_signals <= 3:  # Show first few examples
                    print(f"  Sample {i} has Inf in signal: {np.isinf(signal).sum()} Inf values")
            
            if np.isnan(static_params).any():
                nan_static += 1
                if nan_static <= 3:
                    print(f"  Sample {i} has NaN in static_params: {np.isnan(static_params).sum()} NaN values")
            
            if np.isinf(static_params).any():
                inf_static += 1
                if inf_static <= 3:
                    print(f"  Sample {i} has Inf in static_params: {np.isinf(static_params).sum()} Inf values")
        
        print(f"\nSummary:")
        print(f"  Samples with NaN signals: {nan_signals}")
        print(f"  Samples with Inf signals: {inf_signals}")
        print(f"  Samples with NaN static params: {nan_static}")
        print(f"  Samples with Inf static params: {inf_static}")
        
        if nan_signals > 0 or inf_signals > 0 or nan_static > 0 or inf_static > 0:
            print("❌ Found NaN/Inf in raw data!")
            return False
        else:
            print("✅ Raw data is clean")
            return True
            
    except Exception as e:
        print(f"❌ Error loading raw data: {e}")
        return False

def debug_preprocessing():
    """Debug the preprocessing step."""
    print("\n=== Preprocessing Debug ===")
    
    try:
        # Re-run preprocessing with debug info
        from utils.preprocessing import load_and_preprocess_dataset
        
        print("Re-running preprocessing...")
        data, scaler, onehot = load_and_preprocess_dataset(
            "data/abr_dataset.xlsx",
            verbose=True,
            save_transformers=False
        )
        
        print(f"✅ Preprocessing completed, got {len(data)} samples")
        
        # Check for issues in preprocessing
        nan_count = 0
        inf_count = 0
        
        for i, sample in enumerate(data[:100]):  # Check first 100 samples
            signal = sample['signal']
            static_params = sample['static_params']
            
            if np.isnan(signal).any():
                nan_count += 1
                print(f"  Sample {i}: NaN in signal")
            
            if np.isinf(signal).any():
                inf_count += 1
                print(f"  Sample {i}: Inf in signal")
            
            if np.isnan(static_params).any():
                nan_count += 1
                print(f"  Sample {i}: NaN in static_params")
            
            if np.isinf(static_params).any():
                inf_count += 1
                print(f"  Sample {i}: Inf in static_params")
        
        if nan_count > 0 or inf_count > 0:
            print(f"❌ Found {nan_count} NaN and {inf_count} Inf issues in preprocessing")
            return False
        else:
            print("✅ Preprocessing output is clean")
            return True
            
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_data_issues():
    """Fix data issues by cleaning NaN/Inf values."""
    print("\n=== Fixing Data Issues ===")
    
    try:
        # Load data
        data = joblib.load("data/processed/processed_data.pkl")
        print(f"Loaded {len(data)} samples")
        
        fixed_count = 0
        removed_count = 0
        
        clean_data = []
        
        for i, sample in enumerate(data):
            signal = sample['signal']
            static_params = sample['static_params']
            
            # Check for issues
            signal_has_nan = np.isnan(signal).any()
            signal_has_inf = np.isinf(signal).any()
            static_has_nan = np.isnan(static_params).any()
            static_has_inf = np.isinf(static_params).any()
            
            if signal_has_nan or signal_has_inf or static_has_nan or static_has_inf:
                # Try to fix by replacing NaN/Inf with reasonable values
                if signal_has_nan:
                    signal = np.nan_to_num(signal, nan=0.0)
                    fixed_count += 1
                
                if signal_has_inf:
                    signal = np.nan_to_num(signal, posinf=1.0, neginf=-1.0)
                    fixed_count += 1
                
                if static_has_nan:
                    static_params = np.nan_to_num(static_params, nan=0.0)
                    fixed_count += 1
                
                if static_has_inf:
                    static_params = np.nan_to_num(static_params, posinf=1.0, neginf=-1.0)
                    fixed_count += 1
                
                # Update sample
                sample['signal'] = signal
                sample['static_params'] = static_params
            
            clean_data.append(sample)
        
        print(f"Fixed {fixed_count} issues")
        print(f"Kept {len(clean_data)} samples")
        
        # Save cleaned data
        joblib.dump(clean_data, "data/processed/processed_data_clean.pkl")
        print("✅ Saved cleaned data to data/processed/processed_data_clean.pkl")
        
        # Update config to use cleaned data
        with open('configs/fixed_config.json', 'r') as f:
            config = json.load(f)
        
        config['data']['processed_data_path'] = "data/processed/processed_data_clean.pkl"
        
        with open('configs/fixed_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Updated config to use cleaned data")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing data: {e}")
        return False

def main():
    """Run all debug steps."""
    print("🔍 Data Loading Debug")
    print("=" * 50)
    
    # First check raw data
    raw_ok = debug_raw_data()
    
    if not raw_ok:
        print("\n🔧 Attempting to fix data issues...")
        fix_ok = fix_data_issues()
        if fix_ok:
            print("✅ Data issues fixed!")
        else:
            print("❌ Could not fix data issues")
            return
    
    # Then check preprocessing
    preprocess_ok = debug_preprocessing()
    
    # Finally check dataset loading
    dataset_ok = debug_dataset_loading()
    
    if raw_ok and preprocess_ok and dataset_ok:
        print("\n✅ All data checks passed!")
        print("🚀 Try running training again:")
        print("python main.py --mode train --config_path configs/fixed_config.json --verbose")
    else:
        print("\n❌ Some data issues remain. Please review the output above.")

if __name__ == "__main__":
    main() 