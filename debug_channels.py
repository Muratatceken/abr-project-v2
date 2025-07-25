#!/usr/bin/env python3
"""
Debug script to trace channel dimensions through the model
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.hierarchical_unet import ProfessionalHierarchicalUNet

def debug_model_channels():
    """Debug the channel flow through the model."""
    
    # Model parameters (debug mode)
    model_params = {
        'input_channels': 1,
        'static_dim': 4,
        'base_channels': 32,
        'n_levels': 2,
        'sequence_length': 200,
        'num_transformer_layers': 1,
        'num_heads': 8,
        'use_cross_attention': True,
        'film_dropout': 0.15,
        'dropout': 0.1,
        'use_cfg': False,  # Disable CFG to avoid circular reference
        'num_classes': 5
    }
    
    print("Creating model with parameters:")
    for k, v in model_params.items():
        print(f"  {k}: {v}")
    
    # Create model
    model = ProfessionalHierarchicalUNet(**model_params)
    
    print(f"\nEncoder channels: {model.encoder_channels}")
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 1, 200)  # [batch, channels, seq_len]
    static_params = torch.randn(batch_size, 4)  # [batch, static_dim]
    
    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  static_params: {static_params.shape}")
    
    try:
        # Forward pass with detailed tracing
        print(f"\n=== FORWARD PASS ===")
        
        # Step 1: Encoder
        print("Step 1: Encoder forward...")
        encoded_features, skip_connections, static_emb, encoder_outputs = model.encoder_stack(
            x=x,
            static_params=static_params,
            cfg_guidance_scale=1.0,
            force_uncond=False
        )
        
        print(f"Encoded features shape: {encoded_features.shape}")
        print(f"Skip connections shapes: {[skip.shape for skip in skip_connections]}")
        print(f"Encoder outputs shapes: {[out.shape for out in encoder_outputs]}")
        print(f"Static embedding shape: {static_emb.shape}")
        
        # Step 2: Bottleneck
        print("\nStep 2: Bottleneck forward...")
        bottleneck_output = model.bottleneck(encoded_features, static_emb)
        print(f"Bottleneck output shape: {bottleneck_output.shape}")
        
        # Step 3: Decoder - this is where the error occurs
        print("\nStep 3: Decoder forward...")
        print("Decoder input shape:", bottleneck_output.shape)
        print("Skip connection shapes:", [skip.shape for skip in skip_connections])
        print("Encoder output shapes:", [out.shape for out in encoder_outputs])
        
        # Try to trace through each decoder level
        x_dec = bottleneck_output
        for i, decoder_level in enumerate(model.decoder_stack.decoder_levels):
            print(f"\nDecoder level {i}:")
            print(f"  Input shape: {x_dec.shape}")
            print(f"  Decoder out_channels: {decoder_level.out_channels}")
            
            # Get skip connection
            skip_idx = len(skip_connections) - 1 - i
            skip = skip_connections[skip_idx]
            print(f"  Skip connection shape: {skip.shape}")
            
            # Get encoder output
            encoder_output = encoder_outputs[skip_idx] if encoder_outputs else None
            if encoder_output is not None:
                print(f"  Encoder output shape: {encoder_output.shape}")
                print(f"  Encoder output channels: {encoder_output.shape[1]}")
            
            # Check the upsampling block configuration
            print(f"  Upsample block:")
            print(f"    in_channels: {decoder_level.upsample.in_channels}")
            print(f"    out_channels: {decoder_level.upsample.out_channels}")
            print(f"    skip_channels: {decoder_level.upsample.skip_channels}")
            print(f"    total_in_channels: {decoder_level.upsample.in_channels + decoder_level.upsample.skip_channels}")
            
            # Try the forward pass for this level
            try:
                x_dec = decoder_level(x_dec, skip, static_emb, encoder_output)
                print(f"  Output shape: {x_dec.shape}")
            except Exception as e:
                print(f"  ERROR at decoder level {i}: {e}")
                print(f"  Expected input channels: {decoder_level.upsample.in_channels + decoder_level.upsample.skip_channels}")
                print(f"  Actual concatenated channels: {x_dec.shape[1] + skip.shape[1]}")
                
                # Additional debugging for cross-attention
                if encoder_output is not None:
                    print(f"  Cross-attention debugging:")
                    print(f"    Encoder output shape: {encoder_output.shape}")
                    if hasattr(decoder_level, 'encoder_projection') and decoder_level.encoder_projection:
                        print(f"    Encoder projection: {decoder_level.encoder_projection}")
                        print(f"    Encoder projection weight shape: {decoder_level.encoder_projection.weight.shape}")
                
                break
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_channels() 