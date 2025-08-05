#!/usr/bin/env python3
"""
Simplified ABR Diffusion Model - Optimized Architecture

This is a simplified version of the hierarchical U-Net designed for:
- Better convergence and stability
- Easier debugging and training
- Reduced complexity while maintaining capabilities
- Focus on core ABR generation and classification tasks

Key simplifications:
- Fewer layers and parameters
- Simpler attention mechanisms
- More straightforward skip connections
- Cleaner separation of diffusion and multi-task components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional, Any


class SimplifiedConvBlock(nn.Module):
    """Simplified convolutional block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SimplifiedAttentionBlock(nn.Module):
    """Simplified attention mechanism for sequence modeling."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Residual connection
        residual = x
        x = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Attention
        scale = self.head_dim ** -0.5
        attention = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        # Output projection
        out = self.to_out(out)
        
        return residual + out


class SimplifiedResidualBlock(nn.Module):
    """Simplified residual block with convolution and attention."""
    
    def __init__(
        self,
        channels: int,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.conv1 = SimplifiedConvBlock(channels, channels, dropout=dropout)
        self.conv2 = SimplifiedConvBlock(channels, channels, dropout=dropout)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = SimplifiedAttentionBlock(channels, num_heads=4, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        """
        residual = x
        
        # Convolution layers
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Attention (if enabled)
        if self.use_attention:
            # Transpose for attention: [batch, channels, seq_len] -> [batch, seq_len, channels]
            x = x.transpose(1, 2)
            x = self.attention(x)
            x = x.transpose(1, 2)  # Back to [batch, channels, seq_len]
        
        return residual + x


class SimplifiedEncoder(nn.Module):
    """Simplified encoder with downsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Initial projection
        self.input_proj = SimplifiedConvBlock(in_channels, out_channels, dropout=dropout)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            SimplifiedResidualBlock(out_channels, dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        # Downsampling
        self.downsample = nn.Conv1d(out_channels, out_channels, 4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (downsampled_output, skip_connection)
        """
        x = self.input_proj(x)
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)
        
        skip = x  # Store for skip connection
        
        # Downsample
        x = self.downsample(x)
        
        return x, skip


class SimplifiedDecoder(nn.Module):
    """Simplified decoder with upsampling and skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, 4, stride=2, padding=1)
        
        # Skip connection fusion
        self.skip_fusion = SimplifiedConvBlock(
            in_channels + skip_channels, 
            out_channels, 
            dropout=dropout
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            SimplifiedResidualBlock(out_channels, dropout=dropout)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)
        
        # Fuse with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.skip_fusion(x)
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)
        
        return x


class SimplifiedConditioningBlock(nn.Module):
    """Simplified conditioning block for static parameters."""
    
    def __init__(self, static_dim: int, channels: int):
        super().__init__()
        
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels * 2)  # Scale and shift
        )
    
    def forward(self, x: torch.Tensor, static_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
            static_params: [batch, static_dim]
        """
        # Project static parameters
        conditioning = self.static_proj(static_params)  # [batch, channels * 2]
        scale, shift = conditioning.chunk(2, dim=1)  # [batch, channels] each
        
        # Apply FiLM conditioning
        scale = scale.unsqueeze(-1)  # [batch, channels, 1]
        shift = shift.unsqueeze(-1)  # [batch, channels, 1]
        
        return x * (1 + scale) + shift


class SimplifiedOutputHead(nn.Module):
    """Simplified output head for various tasks."""
    
    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = in_channels
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # MLP head
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            [batch, output_dim]
        """
        # Global pooling
        x = self.global_pool(x)  # [batch, channels, 1]
        x = x.squeeze(-1)  # [batch, channels]
        
        # MLP head
        return self.head(x)


class SimplifiedSignalHead(nn.Module):
    """Simplified signal reconstruction head."""
    
    def __init__(
        self,
        in_channels: int,
        signal_length: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.signal_length = signal_length
        
        # Final processing
        self.final_conv = nn.Sequential(
            SimplifiedConvBlock(in_channels, in_channels // 2, dropout=dropout),
            SimplifiedConvBlock(in_channels // 2, in_channels // 4, dropout=dropout),
            nn.Conv1d(in_channels // 4, 1, 1)  # To single channel
        )
        
        # Adaptive pooling to ensure correct length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(signal_length)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            [batch, signal_length]
        """
        x = self.final_conv(x)  # [batch, 1, seq_len]
        x = self.adaptive_pool(x)  # [batch, 1, signal_length]
        return x.squeeze(1)  # [batch, signal_length]


class SimplifiedABRModel(nn.Module):
    """
    Simplified ABR Diffusion Model for stable training and generation.
    
    Architecture:
    - 3-level encoder-decoder with skip connections
    - Simplified attention and residual blocks
    - Clean separation of diffusion and multi-task components
    - FiLM conditioning for static parameters
    """
    
    def __init__(
        self,
        # Basic parameters
        input_channels: int = 1,
        static_dim: int = 4,
        signal_length: int = 200,
        num_classes: int = 5,
        
        # Architecture parameters
        base_channels: int = 64,
        n_levels: int = 3,
        blocks_per_level: int = 2,
        dropout: float = 0.15,
        
        # Task parameters
        predict_uncertainty: bool = False
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.static_dim = static_dim
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.n_levels = n_levels
        self.predict_uncertainty = predict_uncertainty
        
        # Calculate channel dimensions
        self.channels = [base_channels * (2 ** i) for i in range(n_levels)]
        
        # Input projection
        self.input_proj = SimplifiedConvBlock(input_channels, base_channels, dropout=dropout)
        
        # Encoder levels
        self.encoders = nn.ModuleList()
        for i in range(n_levels):
            in_ch = self.channels[i-1] if i > 0 else base_channels
            out_ch = self.channels[i]
            
            encoder = SimplifiedEncoder(
                in_channels=in_ch,
                out_channels=out_ch,
                num_blocks=blocks_per_level,
                dropout=dropout
            )
            self.encoders.append(encoder)
        
        # Bottleneck
        bottleneck_channels = self.channels[-1]
        self.bottleneck = nn.Sequential(
            SimplifiedResidualBlock(bottleneck_channels, dropout=dropout),
            SimplifiedResidualBlock(bottleneck_channels, dropout=dropout)
        )
        
        # Decoder levels
        self.decoders = nn.ModuleList()
        for i in range(n_levels):
            level_idx = n_levels - 1 - i
            in_ch = self.channels[level_idx]
            skip_ch = self.channels[level_idx]
            out_ch = self.channels[level_idx-1] if level_idx > 0 else base_channels
            
            decoder = SimplifiedDecoder(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                num_blocks=blocks_per_level,
                dropout=dropout
            )
            self.decoders.append(decoder)
        
        # Conditioning blocks
        self.conditioning_blocks = nn.ModuleList([
            SimplifiedConditioningBlock(static_dim, ch) for ch in self.channels
        ])
        
        # Output heads
        final_channels = base_channels
        
        # Signal reconstruction/noise prediction head
        self.signal_head = SimplifiedSignalHead(
            in_channels=final_channels,
            signal_length=signal_length,
            dropout=dropout
        )
        
        # Classification head
        self.classification_head = SimplifiedOutputHead(
            in_channels=final_channels,
            output_dim=num_classes,
            dropout=dropout
        )
        
        # Peak detection head
        peak_dim = 3  # existence, latency, amplitude
        if predict_uncertainty:
            peak_dim = 5  # + latency_std, amplitude_std
        
        self.peak_head = SimplifiedOutputHead(
            in_channels=final_channels,
            output_dim=peak_dim,
            dropout=dropout
        )
        
        # Threshold regression head
        threshold_dim = 2 if predict_uncertainty else 1
        self.threshold_head = SimplifiedOutputHead(
            in_channels=final_channels,
            output_dim=threshold_dim,
            dropout=dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        static_params: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through simplified ABR model.
        
        Args:
            x: Input signal [batch, input_channels, signal_length]
            static_params: Static parameters [batch, static_dim]
            timesteps: Diffusion timesteps [batch] (optional)
        
        Returns:
            Dictionary with model outputs
        """
        batch_size = x.size(0)
        
        # Handle timesteps
        if timesteps is None:
            timesteps = torch.zeros(batch_size, device=x.device, dtype=torch.long)
        
        is_diffusion_mode = timesteps.max() > 0
        
        # Input projection
        x = self.input_proj(x)
        
        # Encoder pass with conditioning
        skip_connections = []
        
        for i, (encoder, conditioning) in enumerate(zip(self.encoders, self.conditioning_blocks)):
            # Apply conditioning
            x = conditioning(x, static_params)
            
            # Encode
            x, skip = encoder(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder pass
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(skip_connections) - 1 - i
            skip = skip_connections[skip_idx]
            x = decoder(x, skip)
        
        # Generate outputs
        outputs = {}
        
        # Signal reconstruction or noise prediction
        signal_output = self.signal_head(x)
        
        if is_diffusion_mode:
            # During training: predict noise
            outputs['noise'] = signal_output
            outputs['recon'] = signal_output  # For compatibility
        else:
            # During inference: predict clean signal
            outputs['recon'] = signal_output
        
        # Multi-task outputs (always from clean features)
        outputs['class'] = self.classification_head(x)
        
        # Peak detection
        peak_output = self.peak_head(x)
        if self.predict_uncertainty:
            # Split into components
            peak_exists = peak_output[:, 0:1]  # [batch, 1]
            peak_latency = peak_output[:, 1:2]  # [batch, 1] 
            peak_amplitude = peak_output[:, 2:3]  # [batch, 1]
            latency_std = torch.exp(peak_output[:, 3:4])  # [batch, 1], ensure positive
            amplitude_std = torch.exp(peak_output[:, 4:5])  # [batch, 1], ensure positive
            
            outputs['peak'] = (
                peak_exists.squeeze(-1),
                peak_latency.squeeze(-1),
                peak_amplitude.squeeze(-1),
                latency_std.squeeze(-1),
                amplitude_std.squeeze(-1)
            )
        else:
            # Simple peak outputs
            peak_exists = peak_output[:, 0]  # [batch]
            peak_latency = peak_output[:, 1]  # [batch]
            peak_amplitude = peak_output[:, 2]  # [batch]
            
            outputs['peak'] = (peak_exists, peak_latency, peak_amplitude)
        
        # Threshold regression
        outputs['threshold'] = self.threshold_head(x)
        
        # Add metadata
        outputs['is_diffusion_mode'] = is_diffusion_mode
        outputs['timesteps'] = timesteps
        
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'SimplifiedABRModel',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': {
                'type': 'Simplified U-Net with Multi-Task Learning',
                'levels': self.n_levels,
                'base_channels': self.base_channels,
                'signal_length': self.signal_length,
                'num_classes': self.num_classes,
                'predict_uncertainty': self.predict_uncertainty
            },
            'features': [
                'Simplified encoder-decoder architecture',
                'Clean skip connections',
                'FiLM conditioning with static parameters',
                'Multi-task learning (classification, peaks, threshold)',
                'Diffusion-compatible (noise prediction)',
                'Reduced complexity for stable training'
            ]
        }