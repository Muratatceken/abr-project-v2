#!/usr/bin/env python3
"""
Script to reprocess ABR data with cleaning and reduced signal length.

This script will:
1. Load existing processed data
2. Apply data cleaning (NaN/Inf fixes)
3. Reduce signal length to 200 timestamps
4. Save cleaned data with new filename
"""

import os
import sys
from utils.preprocessing import reprocess_existing_data

def main():
    """Main function to reprocess data."""
    print("🔄 ABR Data Reprocessing Script")
    print("=" * 50)
    
    # Configuration
    input_file = "data/processed/processed_data_clean.pkl"  # Use the existing clean data
    output_file = "data/processed/processed_data_clean_200ts.pkl"
    signal_length = 200
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("   Please ensure you have processed data available.")
        print("   You can create it by running the preprocessing pipeline.")
        sys.exit(1)
    
    # Check if output file already exists
    if os.path.exists(output_file):
        response = input(f"⚠️  Output file already exists: {output_file}\n   Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("   Aborted.")
            sys.exit(0)
    
    try:
        # Reprocess the data
        reprocess_existing_data(
            input_file=input_file,
            output_file=output_file,
            signal_length=signal_length,
            verbose=True
        )
        
        print("🎉 Data reprocessing completed successfully!")
        print(f"   New data file: {output_file}")
        print(f"   Signal length: {signal_length} timestamps")
        
    except Exception as e:
        print(f"❌ Error during reprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 