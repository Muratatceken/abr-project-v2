#!/usr/bin/env python3
"""
Script to verify the 200-timestamp processed data.
"""

import joblib
import numpy as np
import json

def verify_data():
    """Verify the processed data with 200 timestamps."""
    print("🔍 Verifying 200-timestamp processed data...")
    print("=" * 50)
    
    # Load the new data
    data_path = "data/processed/processed_data_clean_200ts.pkl"
    
    try:
        data = joblib.load(data_path)
        print(f"✅ Successfully loaded data from {data_path}")
        print(f"   Total samples: {len(data)}")
        print()
        
        # Check first few samples
        print("📊 Sample Analysis:")
        for i in range(min(3, len(data))):
            sample = data[i]
            signal = sample['signal']
            static_params = sample['static_params']
            peaks = sample['peaks']
            peak_mask = sample['peak_mask']
            
            print(f"Sample {i}:")
            print(f"  Patient ID: {sample['patient_id']}")
            print(f"  Signal shape: {signal.shape}")
            print(f"  Signal length: {len(signal)} timestamps")
            print(f"  Signal stats: min={np.min(signal):.4f}, max={np.max(signal):.4f}, mean={np.mean(signal):.4f}")
            print(f"  Signal has NaN: {np.isnan(signal).any()}")
            print(f"  Signal has Inf: {np.isinf(signal).any()}")
            print(f"  Static params shape: {static_params.shape}")
            print(f"  Static params has NaN: {np.isnan(static_params).any()}")
            print(f"  Static params has Inf: {np.isinf(static_params).any()}")
            print(f"  Peaks shape: {peaks.shape}")
            print(f"  Valid peaks: {np.sum(peak_mask)}")
            print()
        
        # Overall statistics
        print("📈 Overall Statistics:")
        all_signals = np.array([sample['signal'] for sample in data])
        all_static_params = np.array([sample['static_params'] for sample in data])
        
        print(f"  Signal dimensions: {all_signals.shape}")
        print(f"  Static params dimensions: {all_static_params.shape}")
        print(f"  Total NaN in signals: {np.isnan(all_signals).sum()}")
        print(f"  Total Inf in signals: {np.isinf(all_signals).sum()}")
        print(f"  Total NaN in static params: {np.isnan(all_static_params).sum()}")
        print(f"  Total Inf in static params: {np.isinf(all_static_params).sum()}")
        print()
        
        # Check compatibility with config
        print("🔧 Configuration Compatibility:")
        config_path = "configs/config_200ts.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        expected_signal_length = 200
        expected_static_dim = all_static_params.shape[1]
        
        print(f"  Expected signal length: {expected_signal_length}")
        print(f"  Actual signal length: {all_signals.shape[1]}")
        print(f"  Signal length match: {all_signals.shape[1] == expected_signal_length}")
        print(f"  Static params dimension: {expected_static_dim}")
        print()
        
        if all_signals.shape[1] == expected_signal_length:
            print("✅ Data verification successful!")
            print("   Ready for training with 200 timestamps")
        else:
            print("❌ Data verification failed!")
            print("   Signal length mismatch")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_data() 