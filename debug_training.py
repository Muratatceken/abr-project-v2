#!/usr/bin/env python3
"""
Diagnostic script to debug NaN losses in ABR CVAE training.
"""

import torch
import numpy as np
import json
import joblib
from pathlib import Path

def check_data_quality():
    """Check for data quality issues that could cause NaN losses."""
    print("=== Data Quality Check ===")
    
    # Load processed data
    data_path = "data/processed/processed_data.pkl"
    if not Path(data_path).exists():
        print(f"❌ Processed data not found: {data_path}")
        return False
    
    try:
        data = joblib.load(data_path)
        print(f"✅ Loaded {len(data)} samples")
        
        # Check first few samples
        for i in range(min(5, len(data))):
            sample = data[i]
            signal = sample['signal']
            static_params = sample['static_params']
            peaks = sample['peaks']
            peak_mask = sample['peak_mask']
            
            print(f"\nSample {i}:")
            print(f"  Signal shape: {signal.shape}")
            print(f"  Signal stats: min={np.min(signal):.6f}, max={np.max(signal):.6f}, mean={np.mean(signal):.6f}, std={np.std(signal):.6f}")
            print(f"  Signal has NaN: {np.isnan(signal).any()}")
            print(f"  Signal has Inf: {np.isinf(signal).any()}")
            
            print(f"  Static params shape: {static_params.shape}")
            print(f"  Static params stats: min={np.min(static_params):.6f}, max={np.max(static_params):.6f}")
            print(f"  Static params has NaN: {np.isnan(static_params).any()}")
            print(f"  Static params has Inf: {np.isinf(static_params).any()}")
            
            print(f"  Peaks shape: {peaks.shape}")
            print(f"  Peaks has NaN: {np.isnan(peaks).any()}")
            print(f"  Peak mask sum: {np.sum(peak_mask)}")
            
            # Check for extreme values
            if np.any(np.abs(signal) > 1000):
                print(f"  ⚠️  Signal has extreme values")
            if np.any(np.abs(static_params) > 100):
                print(f"  ⚠️  Static params have extreme values")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def check_model_initialization():
    """Check if model initialization is causing issues."""
    print("\n=== Model Initialization Check ===")
    
    try:
        # Load config
        with open('configs/default_config.json', 'r') as f:
            config = json.load(f)
        
        # Load a sample to get dimensions
        data = joblib.load("data/processed/processed_data.pkl")
        sample = data[0]
        
        signal_length = len(sample['signal'])
        static_dim = len(sample['static_params'])
        latent_dim = config['model']['architecture']['latent_dim']
        
        print(f"Model dimensions:")
        print(f"  Signal length: {signal_length}")
        print(f"  Static dim: {static_dim}")
        print(f"  Latent dim: {latent_dim}")
        
        # Import model
        from models.cvae import CVAE
        
        # Create model
        model = CVAE(
            signal_length=signal_length,
            static_dim=static_dim,
            latent_dim=latent_dim,
            predict_peaks=True,
            num_peaks=6
        )
        
        print(f"✅ Model created successfully")
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Check for NaN in initial weights
        has_nan = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"❌ NaN found in {name}")
                has_nan = True
        
        if not has_nan:
            print("✅ No NaN in initial model weights")
        
        # Test forward pass with dummy data
        device = torch.device('cpu')
        model.to(device)
        
        # Create dummy batch
        batch_size = 4
        dummy_signal = torch.randn(batch_size, signal_length) * 0.1  # Small values
        dummy_static = torch.randn(batch_size, static_dim) * 0.1
        
        print(f"\nTesting forward pass with dummy data:")
        print(f"  Dummy signal stats: min={dummy_signal.min():.6f}, max={dummy_signal.max():.6f}")
        print(f"  Dummy static stats: min={dummy_static.min():.6f}, max={dummy_static.max():.6f}")
        
        try:
            model.eval()
            with torch.no_grad():
                output = model(dummy_signal, dummy_static)
                recon_signal, mu, logvar, predicted_peaks = output
                
                print(f"  Recon signal stats: min={recon_signal.min():.6f}, max={recon_signal.max():.6f}")
                print(f"  Mu stats: min={mu.min():.6f}, max={mu.max():.6f}")
                print(f"  Logvar stats: min={logvar.min():.6f}, max={logvar.max():.6f}")
                print(f"  Predicted peaks stats: min={predicted_peaks.min():.6f}, max={predicted_peaks.max():.6f}")
                
                # Check for NaN in outputs
                if torch.isnan(recon_signal).any():
                    print("❌ NaN in reconstructed signal")
                if torch.isnan(mu).any():
                    print("❌ NaN in mu")
                if torch.isnan(logvar).any():
                    print("❌ NaN in logvar")
                if torch.isnan(predicted_peaks).any():
                    print("❌ NaN in predicted peaks")
                
                print("✅ Forward pass completed")
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

def check_loss_computation():
    """Check if loss computation is causing NaN values."""
    print("\n=== Loss Computation Check ===")
    
    try:
        from utils.losses import cvae_loss, peak_loss
        
        # Create dummy data
        batch_size = 4
        signal_length = 467
        latent_dim = 32
        num_peaks = 6
        
        # Create realistic dummy data
        target_signal = torch.randn(batch_size, signal_length) * 0.1
        recon_signal = torch.randn(batch_size, signal_length) * 0.1
        mu = torch.randn(batch_size, latent_dim) * 0.1
        logvar = torch.randn(batch_size, latent_dim) * 0.1
        
        # Test CVAE loss
        total_loss, recon_loss, kl_loss = cvae_loss(recon_signal, target_signal, mu, logvar, beta=0.1)
        
        print(f"CVAE Loss test:")
        print(f"  Total loss: {total_loss.item():.6f}")
        print(f"  Recon loss: {recon_loss.item():.6f}")
        print(f"  KL loss: {kl_loss.item():.6f}")
        
        if torch.isnan(total_loss):
            print("❌ NaN in total loss")
        if torch.isnan(recon_loss):
            print("❌ NaN in reconstruction loss")
        if torch.isnan(kl_loss):
            print("❌ NaN in KL loss")
        
        # Test peak loss
        predicted_peaks = torch.randn(batch_size, num_peaks) * 0.1
        target_peaks = torch.randn(batch_size, num_peaks) * 0.1
        peak_mask = torch.ones(batch_size, num_peaks).bool()
        
        p_loss = peak_loss(predicted_peaks, target_peaks, peak_mask)
        print(f"  Peak loss: {p_loss.item():.6f}")
        
        if torch.isnan(p_loss):
            print("❌ NaN in peak loss")
        else:
            print("✅ Loss computation working")
            
        return True
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False

def create_fixed_config():
    """Create a fixed configuration to address NaN issues."""
    print("\n=== Creating Fixed Configuration ===")
    
    try:
        # Load original config
        with open('configs/default_config.json', 'r') as f:
            config = json.load(f)
        
        # Apply fixes
        config['training']['optimizer']['learning_rate'] = 1e-4  # Lower learning rate
        config['training']['loss']['gradient_clip'] = 0.5  # Lower gradient clipping
        config['training']['loss']['beta_scheduler']['warmup_epochs'] = 20  # Longer warmup
        config['model']['initialization']['type'] = 'xavier_normal'  # Different initialization
        config['model']['initialization']['gain'] = 0.1  # Lower gain
        
        # Add numerical stability settings
        config['training']['optimizer']['eps'] = 1e-7  # Smaller epsilon
        config['training']['numerical_stability'] = {
            'min_logvar': -20.0,
            'max_logvar': 2.0,
            'gradient_clip_value': 0.5,
            'loss_scale': 1.0
        }
        
        # Save fixed config
        with open('configs/fixed_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Fixed configuration saved to configs/fixed_config.json")
        print("Key changes made:")
        print("  - Reduced learning rate to 1e-4")
        print("  - Reduced gradient clipping to 0.5")
        print("  - Extended beta warmup to 20 epochs")
        print("  - Changed to xavier_normal initialization with gain=0.1")
        print("  - Added numerical stability settings")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create fixed config: {e}")
        return False

def main():
    """Run all diagnostic checks."""
    print("🔍 ABR CVAE Training Diagnostics")
    print("=" * 50)
    
    # Run checks
    data_ok = check_data_quality()
    model_ok = check_model_initialization()
    loss_ok = check_loss_computation()
    
    if data_ok and model_ok and loss_ok:
        print("\n✅ All checks passed! Creating fixed configuration...")
        create_fixed_config()
        print("\n🚀 Try running training with the fixed configuration:")
        print("python main.py --mode train --config_path configs/fixed_config.json --verbose")
    else:
        print("\n❌ Some checks failed. Please review the issues above.")

if __name__ == "__main__":
    main() 