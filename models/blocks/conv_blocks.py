"""
Convolution Blocks for ABR Hierarchical U-Net

Professional implementation of 1D convolution blocks
for encoder/decoder paths in the hierarchical U-Net.

Includes:
- Basic Conv1D blocks with normalization and activation
- Residual blocks for deeper networks
- Downsampling and upsampling blocks
- Skip connection handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math


class Conv1dBlock(nn.Module):
    """
    Basic 1D convolution block with normalization and activation.
    
    Provides a standard building block for convolutional layers
    with configurable normalization, activation, and dropout.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        normalization: str = 'layer',
        activation: str = 'gelu',
        dropout: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.causal = causal
        
        # Calculate padding
        if padding is None:
            if causal:
                # Causal padding: pad only on the left
                padding = (kernel_size - 1) * dilation
            else:
                # Standard padding: pad symmetrically
                padding = ((kernel_size - 1) * dilation) // 2
        
        self.padding = padding
        
        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding if not causal else 0,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Normalization
        if normalization == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        elif normalization == 'layer':
            self.norm = nn.GroupNorm(1, out_channels)  # Layer norm for 1D
        elif normalization == 'group':
            self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        elif normalization == 'instance':
            self.norm = nn.InstanceNorm1d(out_channels)
        else:
            self.norm = nn.Identity()
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu' or activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize convolution weights."""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through conv block.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            
        Returns:
            Output tensor [batch, out_channels, new_seq_len]
        """
        # Apply causal padding if needed
        if self.causal and self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        
        # Convolution
        x = self.conv(x)
        
        # Normalization
        x = self.norm(x)
        
        # Activation
        x = self.activation(x)
        
        # Dropout
        x = self.dropout(x)
        
        return x


class EnhancedConvBlock(nn.Module):
    """
    Enhanced convolution block with Conv1d → ReLU → Conv1d → LayerNorm stack.
    Designed to be used before S4 layers for deeper feature extraction.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        groups: int = 1,
        depth_wise: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Use depth-wise separable convolutions if requested
        if depth_wise and in_channels == out_channels:
            # Depth-wise convolution
            self.conv1 = nn.Conv1d(
                in_channels, in_channels, 
                kernel_size=kernel_size,
                padding=kernel_size//2,
                dilation=dilation,
                groups=in_channels
            )
            # Point-wise convolution
            self.conv2 = nn.Conv1d(in_channels, out_channels, 1)
        else:
            # Standard convolutions
            mid_channels = max(in_channels, out_channels)
            self.conv1 = nn.Conv1d(
                in_channels, mid_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2,
                dilation=dilation,
                groups=groups
            )
            self.conv2 = nn.Conv1d(
                mid_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2,
                groups=groups
            )
        
        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.ReLU(inplace=True)
        
        # Layer normalization (will be applied to [batch, seq_len, channels])
        self.layer_norm = nn.LayerNorm(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Residual projection if needed
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_proj = nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for conv in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        
        if hasattr(self.residual_proj, 'weight'):
            nn.init.kaiming_normal_(self.residual_proj.weight, mode='fan_out', nonlinearity='linear')
            if self.residual_proj.bias is not None:
                nn.init.zeros_(self.residual_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through enhanced conv block.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            
        Returns:
            Output tensor [batch, out_channels, seq_len]
        """
        residual = self.residual_proj(x)
        
        # First convolution + activation
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.dropout(out)
        
        # Second convolution + activation
        out = self.conv2(out)
        out = self.activation2(out)
        
        # Add residual connection
        out = out + residual
        
        # Apply layer normalization (convert to [batch, seq_len, channels])
        out = out.transpose(1, 2)  # [batch, seq_len, channels]
        out = self.layer_norm(out)
        out = out.transpose(1, 2)  # [batch, channels, seq_len]
        
        return out


class ResidualS4Block(nn.Module):
    """
    Residual S4 block that wraps S4 with normalization and feedforward network.
    Inspired by Transformer residual blocks but adapted for S4.
    """
    
    def __init__(
        self,
        d_model: int,
        s4_kwargs: dict = None,
        ff_mult: int = 4,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pre_norm = pre_norm
        
        # Import S4Layer here to avoid circular imports
        from .s4_layer import S4Layer
        
        # S4 layer
        s4_kwargs = s4_kwargs or {}
        self.s4 = S4Layer(features=d_model, dropout=dropout, **s4_kwargs)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual S4 block.
        
        Args:
            x: Input tensor [batch, channels, seq_len]
            
        Returns:
            Output tensor [batch, channels, seq_len]
        """
        # Convert to [batch, seq_len, channels] for norm
        x_norm_format = x.transpose(1, 2)
        
        if self.pre_norm:
            # Pre-norm: norm → S4 → residual
            normed = self.norm1(x_norm_format).transpose(1, 2)  # Back to [batch, channels, seq_len]
            s4_out = self.s4(normed)
            x = x + self.dropout(s4_out)
            
            # Feedforward with residual
            x_ff_input = x.transpose(1, 2)  # [batch, seq_len, channels]
            normed_ff = self.norm2(x_ff_input)
            ff_out = self.ff(normed_ff)
            x_ff_output = x_ff_input + self.dropout(ff_out)
            x = x_ff_output.transpose(1, 2)  # Back to [batch, channels, seq_len]
        else:
            # Post-norm: S4 → residual → norm
            s4_out = self.s4(x)
            x = x + self.dropout(s4_out)
            x_norm_format = x.transpose(1, 2)
            x = self.norm1(x_norm_format).transpose(1, 2)
            
            # Feedforward
            x_ff_input = x.transpose(1, 2)
            ff_out = self.ff(x_ff_input)
            x_ff_output = x_ff_input + self.dropout(ff_out)
            x = self.norm2(x_ff_output).transpose(1, 2)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for deeper networks.
    
    Implements identity mapping with learnable residual function,
    supporting different normalization and activation schemes.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        normalization: str = 'layer',
        activation: str = 'gelu',
        dropout: float = 0.0,
        causal: bool = False,
        bottleneck: bool = False,
        expansion: int = 4
    ):
        super().__init__()
        
        self.channels = channels
        self.bottleneck = bottleneck
        
        if bottleneck:
            # Bottleneck design: 1x1 -> 3x3 -> 1x1
            hidden_channels = channels // expansion
            
            self.conv1 = Conv1dBlock(
                in_channels=channels,
                out_channels=hidden_channels,
                kernel_size=1,
                normalization=normalization,
                activation=activation,
                dropout=dropout,
                causal=causal
            )
            
            self.conv2 = Conv1dBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                normalization=normalization,
                activation=activation,
                dropout=dropout,
                causal=causal
            )
            
            self.conv3 = Conv1dBlock(
                in_channels=hidden_channels,
                out_channels=channels,
                kernel_size=1,
                normalization=normalization,
                activation='identity',  # No activation on final layer
                dropout=0.0,
                causal=causal
            )
            
            self.residual_path = nn.Sequential(self.conv1, self.conv2, self.conv3)
        else:
            # Standard design: 3x3 -> 3x3
            self.conv1 = Conv1dBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                normalization=normalization,
                activation=activation,
                dropout=dropout,
                causal=causal
            )
            
            self.conv2 = Conv1dBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                normalization=normalization,
                activation='identity',  # No activation on final layer
                dropout=0.0,
                causal=causal
            )
            
            self.residual_path = nn.Sequential(self.conv1, self.conv2)
        
        # Final activation
        if activation == 'relu':
            self.final_activation = nn.ReLU()
        elif activation == 'gelu':
            self.final_activation = nn.GELU()
        elif activation == 'silu':
            self.final_activation = nn.SiLU()
        else:
            self.final_activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor [batch, channels, seq_len]
            
        Returns:
            Output tensor [batch, channels, seq_len]
        """
        identity = x
        residual = self.residual_path(x)
        
        # Add residual connection
        out = identity + residual
        
        # Final activation
        out = self.final_activation(out)
        
        return out


class DownsampleBlock(nn.Module):
    """
    Downsampling block for encoder path.
    
    Reduces temporal resolution while increasing channel capacity,
    with options for different downsampling strategies.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_factor: int = 2,
        method: str = 'conv',
        kernel_size: int = 3,
        normalization: str = 'layer',
        activation: str = 'gelu',
        dropout: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_factor = downsample_factor
        self.method = method
        
        if method == 'conv':
            # Strided convolution
            self.downsample = Conv1dBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=downsample_factor,
                normalization=normalization,
                activation=activation,
                dropout=dropout,
                causal=causal
            )
        elif method == 'pool':
            # Pooling followed by 1x1 conv for channel adjustment
            if downsample_factor == 2:
                self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            else:
                self.pool = nn.AdaptiveMaxPool1d(None)  # Will be set dynamically
            
            self.channel_adjust = Conv1dBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                normalization=normalization,
                activation=activation,
                dropout=dropout,
                causal=causal
            ) if in_channels != out_channels else nn.Identity()
        
        elif method == 'avgpool':
            # Average pooling for smoother downsampling
            if downsample_factor == 2:
                self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
            else:
                self.pool = nn.AdaptiveAvgPool1d(None)
            
            self.channel_adjust = Conv1dBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                normalization=normalization,
                activation=activation,
                dropout=dropout,
                causal=causal
            ) if in_channels != out_channels else nn.Identity()
        
        else:
            raise ValueError(f"Unknown downsampling method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through downsampling block.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            
        Returns:
            Downsampled tensor [batch, out_channels, seq_len // downsample_factor]
        """
        if self.method == 'conv':
            return self.downsample(x)
        else:
            # For pooling methods
            if self.downsample_factor != 2:
                # Dynamic pooling for arbitrary factors
                target_length = x.size(-1) // self.downsample_factor
                if hasattr(self.pool, 'output_size'):
                    self.pool.output_size = target_length
                else:
                    # Create new adaptive pool
                    if isinstance(self.pool, nn.MaxPool1d):
                        self.pool = nn.AdaptiveMaxPool1d(target_length)
                    else:
                        self.pool = nn.AdaptiveAvgPool1d(target_length)
            
            x = self.pool(x)
            x = self.channel_adjust(x)
            return x


class UpsampleBlock(nn.Module):
    """
    Upsampling block for decoder path.
    
    Increases temporal resolution while adjusting channel capacity,
    with support for skip connections from encoder.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: Optional[int] = None,
        upsample_factor: int = 2,
        method: str = 'conv_transpose',
        kernel_size: int = 3,
        normalization: str = 'layer',
        activation: str = 'gelu',
        dropout: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels or 0
        self.upsample_factor = upsample_factor
        self.method = method
        
        # Calculate total input channels after skip concatenation
        total_in_channels = in_channels + self.skip_channels
        
        if method == 'conv_transpose':
            # Transposed convolution
            self.upsample = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=upsample_factor,
                padding=(kernel_size - 1) // 2,
                output_padding=upsample_factor - 1
            )
            
            # Process skip connections if present
            if self.skip_channels > 0:
                self.skip_conv = Conv1dBlock(
                    in_channels=total_in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    normalization=normalization,
                    activation=activation,
                    dropout=dropout,
                    causal=causal
                )
            else:
                self.skip_conv = nn.Identity()
                
        elif method == 'interpolate':
            # Interpolation followed by convolution
            self.upsample_conv = Conv1dBlock(
                in_channels=total_in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                normalization=normalization,
                activation=activation,
                dropout=dropout,
                causal=causal
            )
        
        else:
            raise ValueError(f"Unknown upsampling method: {method}")
        
        # Normalization and activation for conv_transpose method
        if method == 'conv_transpose':
            if normalization == 'batch':
                self.norm = nn.BatchNorm1d(out_channels)
            elif normalization == 'layer':
                self.norm = nn.GroupNorm(1, out_channels)
            elif normalization == 'group':
                self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
            else:
                self.norm = nn.Identity()
            
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'gelu':
                self.activation = nn.GELU()
            elif activation == 'silu':
                self.activation = nn.SiLU()
            else:
                self.activation = nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        if hasattr(self, 'upsample') and isinstance(self.upsample, nn.ConvTranspose1d):
            nn.init.kaiming_normal_(self.upsample.weight, mode='fan_out', nonlinearity='relu')
            if self.upsample.bias is not None:
                nn.init.zeros_(self.upsample.bias)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through upsampling block.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            skip: Skip connection [batch, skip_channels, target_seq_len]
            
        Returns:
            Upsampled tensor [batch, out_channels, seq_len * upsample_factor]
        """
        if self.method == 'conv_transpose':
            # Transposed convolution upsampling
            x = self.upsample(x)
            x = self.norm(x)
            x = self.activation(x)
            
            # Handle skip connections
            if skip is not None:
                # Ensure sizes match
                if x.size(-1) != skip.size(-1):
                    # Interpolate skip to match upsampled size
                    skip = F.interpolate(skip, size=x.size(-1), mode='linear', align_corners=False)
                
                # Concatenate skip connection
                x = torch.cat([x, skip], dim=1)
                x = self.skip_conv(x)
        
        elif self.method == 'interpolate':
            # Interpolation upsampling first
            target_length = x.size(-1) * self.upsample_factor
            x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
            
            # Handle skip connections after upsampling
            if skip is not None:
                # Resize skip to match upsampled length
                if skip.size(-1) != x.size(-1):
                    skip = F.interpolate(skip, size=x.size(-1), mode='linear', align_corners=False)
                # Concatenate with upsampled input
                x = torch.cat([x, skip], dim=1)
            
            # Apply convolution
            x = self.upsample_conv(x)
        
        return x


class SkipConnection(nn.Module):
    """
    Skip connection handler for U-Net architectures.
    
    Manages skip connections between encoder and decoder,
    with optional feature transformation and attention.
    """
    
    def __init__(
        self,
        skip_channels: int,
        decoder_channels: int,
        output_channels: Optional[int] = None,
        transform: bool = True,
        use_attention: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.skip_channels = skip_channels
        self.decoder_channels = decoder_channels
        self.output_channels = output_channels or decoder_channels
        self.transform = transform
        self.use_attention = use_attention
        
        if transform:
            # Transform skip features to match decoder
            self.skip_transform = Conv1dBlock(
                in_channels=skip_channels,
                out_channels=decoder_channels,
                kernel_size=1,
                normalization='layer',
                activation='gelu',
                dropout=dropout
            )
        else:
            self.skip_transform = nn.Identity()
        
        # Attention mechanism for skip connections
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=decoder_channels,
                num_heads=4,
                dropout=dropout,
                batch_first=False  # Expects [seq, batch, dim]
            )
        
        # Fusion layer
        self.fusion = Conv1dBlock(
            in_channels=decoder_channels * 2,  # skip + decoder
            out_channels=self.output_channels,
            kernel_size=1,
            normalization='layer',
            activation='gelu',
            dropout=dropout
        )
    
    def forward(self, decoder_features: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through skip connection.
        
        Args:
            decoder_features: Decoder tensor [batch, decoder_channels, seq_len]
            skip_features: Skip tensor [batch, skip_channels, skip_seq_len]
            
        Returns:
            Fused features [batch, output_channels, seq_len]
        """
        # Transform skip features
        skip = self.skip_transform(skip_features)
        
        # Resize skip to match decoder if needed
        if skip.size(-1) != decoder_features.size(-1):
            skip = F.interpolate(skip, size=decoder_features.size(-1), mode='linear', align_corners=False)
        
        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention: [seq, batch, channels]
            skip_attn = skip.transpose(1, 2).transpose(0, 1)
            decoder_attn = decoder_features.transpose(1, 2).transpose(0, 1)
            
            # Apply cross-attention: skip attends to decoder
            attended_skip, _ = self.attention(skip_attn, decoder_attn, decoder_attn)
            
            # Reshape back: [batch, channels, seq]
            skip = attended_skip.transpose(0, 1).transpose(1, 2)
        
        # Concatenate and fuse
        fused = torch.cat([decoder_features, skip], dim=1)
        output = self.fusion(fused)
        
        return output


class MultiScaleBlock(nn.Module):
    """
    Multi-scale convolution block with different kernel sizes.
    
    Captures features at multiple temporal scales simultaneously,
    useful for handling diverse signal characteristics.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list = [3, 5, 7],
        normalization: str = 'layer',
        activation: str = 'gelu',
        dropout: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        
        # Multi-scale convolutions
        self.conv_blocks = nn.ModuleList([
            Conv1dBlock(
                in_channels=in_channels,
                out_channels=out_channels // len(kernel_sizes),
                kernel_size=k,
                normalization=normalization,
                activation=activation,
                dropout=dropout,
                causal=causal
            )
            for k in kernel_sizes
        ])
        
        # 1x1 convolution for channel adjustment if needed
        total_output_channels = (out_channels // len(kernel_sizes)) * len(kernel_sizes)
        if total_output_channels != out_channels:
            self.channel_adjust = Conv1dBlock(
                in_channels=total_output_channels,
                out_channels=out_channels,
                kernel_size=1,
                normalization=normalization,
                activation='identity'
            )
        else:
            self.channel_adjust = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale block.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            
        Returns:
            Multi-scale features [batch, out_channels, seq_len]
        """
        # Apply different kernel sizes
        scale_features = []
        for conv_block in self.conv_blocks:
            scale_features.append(conv_block(x))
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_features, dim=1)
        
        # Adjust channels if needed
        output = self.channel_adjust(multi_scale)
        
        return output 