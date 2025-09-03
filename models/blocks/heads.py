"""
Output Heads for ABR Hierarchical U-Net

Professional implementation of task-specific output heads
for multi-task learning in ABR signal processing.

Includes:
- Signal reconstruction head
- Peak prediction head (existence, latency, amplitude)  
- Classification head (hearing loss type)
- Threshold regression head

Updated with architectural improvements for better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
import numpy as np


class BaseHead(nn.Module):
    """Base class for output heads with common functionality."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        layer_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim // 2
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
    
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction for better peak and signal analysis."""
    
    def __init__(self, input_dim: int, scales: List[int] = [1, 3, 5, 7]):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        
        # Compute output channels per branch to sum to input_dim
        base_channels = input_dim // self.num_scales
        remainder = input_dim % self.num_scales
        
        # Allocate channels: first (num_scales - remainder) branches get base_channels
        # last remainder branches get (base_channels + 1)
        self.out_channels_per_branch = []
        for i in range(self.num_scales):
            if i < self.num_scales - remainder:
                self.out_channels_per_branch.append(base_channels)
            else:
                self.out_channels_per_branch.append(base_channels + 1)
        
        # Verify total channels sum to input_dim
        total_channels = sum(self.out_channels_per_branch)
        assert total_channels == input_dim, f"Total channels {total_channels} != input_dim {input_dim}"
        
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, out_channels, 
                     kernel_size=scale, padding=scale//2)
            for scale, out_channels in zip(scales, self.out_channels_per_branch)
        ])
        self.fusion = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale convolutions and fuse features."""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, channels, seq], got {x.shape}")
        
        multi_scale_features = []
        for conv in self.convs:
            feature = conv(x)
            multi_scale_features.append(feature)
        
        # Concatenate and fuse
        fused = torch.cat(multi_scale_features, dim=1)
        return self.fusion(fused)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling mechanism for sequence-to-vector tasks.
    """
    
    def __init__(self, input_dim: int, attention_dim: int = None, dropout: float = 0.1):
        super().__init__()
        
        if attention_dim is None:
            attention_dim = input_dim // 2
        
        self.attention_dim = attention_dim
        self.input_dim = input_dim
        
        # Layer normalization for feature dimension
        self.feature_norm = nn.LayerNorm(input_dim)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention pooling with optional masking.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim] or [batch, input_dim, seq_len]
            mask: Optional mask tensor [batch, seq_len] where 1 = valid, 0 = masked
            
        Returns:
            Pooled tensor [batch, input_dim]
        """
        # Ensure [batch, seq_len, input_dim] format
        if x.dim() == 3 and x.size(1) == self.input_dim:
            x = x.transpose(1, 2)  # [batch, seq_len, input_dim]
        
        # Apply feature normalization
        x = self.feature_norm(x)
        
        # Compute attention weights
        attention_weights = self.attention(x)  # [batch, seq_len, 1]
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention weights shape
            mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            # Apply -inf masking before softmax (already applied in attention mechanism)
            attention_weights = attention_weights * mask
        
        # Apply attention pooling
        pooled = torch.sum(x * attention_weights, dim=1)  # [batch, input_dim]
        
        # Apply dropout for regularization
        pooled = self.dropout(pooled)
        
        return pooled





class EnhancedSignalHead(nn.Module):
    """
    Locality-preserving signal head (TCN-style) for high-fidelity ABR reconstruction.
    - Operates on [batch, channels, seq_len] without global pooling
    - Dilated residual Conv1d blocks capture short-duration transients
    - Optional timestep conditioning via AdaGN (scale/shift)
    """

    def __init__(
        self,
        input_dim: int,
        signal_length: int,
        hidden_channels: int = 128,
        n_blocks: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.05,
        use_timestep_conditioning: bool = True,
        t_embed_dim: int = 128,
    ):
        super().__init__()

        self.signal_length = signal_length
        self.use_timestep_conditioning = use_timestep_conditioning

        # Project features to conv space
        self.input_proj = nn.Conv1d(input_dim, hidden_channels, kernel_size=1)

        # Timestep embedding for diffusion (sinusoidal + MLP)
        if use_timestep_conditioning:
            self.t_embed = nn.Sequential(
                nn.Linear(t_embed_dim, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels * 2),  # scale, shift
            )
        else:
            self.t_embed = None

        # Dilated residual blocks
        blocks = []
        for i in range(n_blocks):
            dilation = 2 ** i
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=dilation*(kernel_size//2), dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=dilation*(kernel_size//2), dilation=dilation),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.GroupNorm(1, hidden_channels)

        # Output projection
        self.out_proj = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    @staticmethod
    def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        device = timesteps.device
        half = dim // 2
        emb = np.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

    def forward(self, x: torch.Tensor, timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len] or [batch, seq_len, channels]
            timesteps: [batch] diffusion steps for conditioning
        Returns:
            [batch, signal_length]
        """
        if x.dim() == 3 and x.size(1) != self.input_proj.in_channels and x.size(2) == self.input_proj.in_channels:
            x = x.transpose(1, 2)

        x = self.input_proj(x)

        cond_scale = cond_shift = None
        if self.use_timestep_conditioning and timesteps is not None:
            t_emb = self.sinusoidal_embedding(timesteps, self.t_embed[0].in_features)
            cond = self.t_embed(t_emb)
            cond_scale, cond_shift = cond.chunk(2, dim=1)  # [B, C]

        for block in self.blocks:
            residual = x
            h = block(x)
            if cond_scale is not None and cond_shift is not None:
                # AdaGN-style mod
                h = self.norm(h)
                h = h * (1 + cond_scale.unsqueeze(-1)) + cond_shift.unsqueeze(-1)
            x = residual + h

        x = self.out_proj(x)
        if x.size(-1) != self.signal_length:
            x = F.interpolate(x, size=self.signal_length, mode='linear', align_corners=False)
        return x.squeeze(1)





class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.net(x))





