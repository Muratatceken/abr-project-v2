#!/usr/bin/env python3
"""
Usage example for the updated ABRDataset class.
Shows how to use the dataset with S4 + Transformer-based Hierarchical U-Net model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import (
    ABRDataset,
    load_ultimate_dataset,
    create_dataloaders_compatible
)

def main():
    """Demonstrate usage of the updated ABRDataset."""
    
    print("🔍 ABRDataset Usage Example for S4 + Transformer Model")
    print("=" * 60)
    
    # Method 1: Load dataset directly
    print("\n📊 Method 1: Direct dataset loading")
    dataset = ABRDataset(
        data_path="data/processed/ultimate_dataset.pkl",
        normalize_signal=True,
        normalize_static=True
    )
    
    print(f"Dataset loaded: {len(dataset)} samples from {len(dataset.unique_patients)} patients")
    
    # Get a sample to see the format
    sample = dataset[0]
    print(f"\nSample format:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)} = {value}")
    
    # Method 2: Load with stratified splits
    print("\n📊 Method 2: Stratified patient splitting")
    full_dataset, train_dataset, val_dataset, test_dataset = load_ultimate_dataset(
        data_path="data/processed/ultimate_dataset.pkl",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples") 
    print(f"Test: {len(test_dataset)} samples")
    
    # Method 3: Create DataLoaders directly
    print("\n📊 Method 3: Direct DataLoader creation")
    train_loader, val_loader, test_loader = create_dataloaders_compatible(
        data_path="data/processed/ultimate_dataset.pkl",
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        num_workers=2,
        random_state=42
    )
    
    # Demonstrate batch processing
    print("\n🔄 Processing a training batch:")
    batch = next(iter(train_loader))
    
    print(f"Batch contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {len(value)} patient IDs")
    
    # Show how to handle class imbalance
    print("\n⚖️ Class imbalance handling:")
    class_weights = full_dataset.get_class_weights(method='balanced')
    print(f"Class weights: {class_weights}")
    
    # Example loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Weighted loss function created")
    
    # Demonstrate model input format
    print("\n🤖 Model input format for S4 + Transformer:")
    signal = batch['signal']      # [batch_size, 200] - time series for S4
    static = batch['static']      # [batch_size, 4] - for FiLM conditioning
    target = batch['target']      # [batch_size] - classification target
    v_peak = batch['v_peak']      # [batch_size, 2] - peak prediction target
    v_peak_mask = batch['v_peak_mask']  # [batch_size, 2] - peak validity mask
    
    print(f"Input signal shape: {signal.shape}")
    print(f"Static params shape: {static.shape}")
    print(f"Target shape: {target.shape}")
    print(f"V peak shape: {v_peak.shape}")
    print(f"V peak mask shape: {v_peak_mask.shape}")
    
    # Example usage in a training loop
    print("\n🔄 Example training loop structure:")
    print("""
    for epoch in range(num_epochs):
        for batch in train_loader:
            # Extract data
            signals = batch['signal']          # [B, 200]
            static_params = batch['static']    # [B, 4]
            targets = batch['target']          # [B]
            v_peaks = batch['v_peak']          # [B, 2]
            v_peak_masks = batch['v_peak_mask'] # [B, 2]
            
            # Forward pass through S4 + Transformer model
            outputs = model(
                x=signals,                     # Time series input
                film_params=static_params      # FiLM conditioning
            )
            
            # Multi-task outputs
            denoised_signal = outputs['signal']      # [B, 200]
            hearing_pred = outputs['classification'] # [B, 5]
            peak_pred = outputs['peaks']             # [B, 2]
            
            # Multi-task loss
            signal_loss = F.mse_loss(denoised_signal, signals)
            class_loss = F.cross_entropy(hearing_pred, targets, weight=class_weights)
            peak_loss = F.mse_loss(peak_pred[v_peak_masks], v_peaks[v_peak_masks])
            
            total_loss = signal_loss + class_loss + peak_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    """)
    
    print("\n✅ Dataset is ready for S4 + Transformer model training!")

if __name__ == "__main__":
    main() 