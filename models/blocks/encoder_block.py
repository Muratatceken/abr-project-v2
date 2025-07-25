"""
Enhanced Encoder Blocks for ABR Hierarchical U-Net

Professional implementation with S4 layers, enhanced convolution stacks,
and robust architectural patterns inspired by SSSD-ECG.

Author: AI Assistant
Based on: SSSD-ECG implementation patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

from .s4_layer import S4Layer, EnhancedS4Layer
from .conv_blocks import EnhancedConvBlock, ResidualS4Block, DownsampleBlock
from .film import AdaptiveFiLMWithDropout
from .positional import PositionalEmbedding


class EnhancedEncoderBlock(nn.Module):
    """
    Enhanced encoder block with Conv → ReLU → Conv → LayerNorm → S4 → FiLM pipeline.
    
    Features:
    - Enhanced convolution stack before S4
    - Multiple S4 layers with residual connections
    - Advanced FiLM conditioning with dropout
    - Professional weight initialization
    - Skip connection handling
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        static_dim: int,
        sequence_length: int,
        downsample_factor: int = 2,
        s4_state_size: int = 64,
        num_s4_layers: int = 2,  # Increased from 1
        num_conv_layers: int = 2,
        dropout: float = 0.1,
        film_dropout: float = 0.15,
        use_enhanced_s4: bool = True,
        use_depth_wise_conv: bool = False,
        use_positional_encoding: bool = True,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequence_length = sequence_length
        self.downsample_factor = downsample_factor
        self.num_s4_layers = num_s4_layers
        
        # ============== ENHANCED CONVOLUTION STACK ==============
        # Conv1d → ReLU → Conv1d → LayerNorm stack before S4
        
        # Input projection if needed
        if in_channels != out_channels:
            self.input_projection = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.input_projection = nn.Identity()
        
        # Enhanced convolution blocks
        self.conv_blocks = nn.ModuleList()
        for i in range(num_conv_layers):
            if i == 0:
                conv_in = out_channels
            else:
                conv_in = out_channels
            
            self.conv_blocks.append(
                EnhancedConvBlock(
                    in_channels=conv_in,
                    out_channels=out_channels,
                    kernel_size=3,
                    dropout=dropout,
                    depth_wise=use_depth_wise_conv
                )
            )
        
        # Downsampling after convolution processing
        self.downsample = DownsampleBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            downsample_factor=downsample_factor,
            method='conv',
            kernel_size=3,
            normalization='layer',
            activation=activation,
            dropout=dropout
        )
        
        # ============== POSITIONAL ENCODING ==============
        if use_positional_encoding:
            downsampled_length = sequence_length // downsample_factor
            self.pos_encoding = PositionalEmbedding(
                d_model=out_channels,
                max_len=downsampled_length,
                embedding_type='sinusoidal'
            )
        else:
            self.pos_encoding = None
        
        # ============== S4 PROCESSING LAYERS ==============
        downsampled_length = sequence_length // downsample_factor
        
        if use_enhanced_s4:
            # Use enhanced S4 layers with learnable parameters
            self.s4_layers = nn.ModuleList([
                EnhancedS4Layer(
                    features=out_channels,
                    lmax=downsampled_length,
                    N=s4_state_size,
                    dropout=dropout,
                    bidirectional=True,
                    layer_norm=True,
                    learnable_timescales=True,
                    kernel_mixing=True,
                    activation=activation
                )
                for _ in range(num_s4_layers)
            ])
        else:
            # Use residual S4 blocks for more complex processing
            self.s4_layers = nn.ModuleList([
                ResidualS4Block(
                    d_model=out_channels,
                    s4_kwargs={
                        'lmax': downsampled_length,
                        'N': s4_state_size,
                        'enhanced': True
                    },
                    ff_mult=4,
                    dropout=dropout
                )
                for _ in range(num_s4_layers)
            ])
        
        # ============== FILM CONDITIONING ==============
        # FiLM applied AFTER S4 processing (not before)
        self.film_layer = AdaptiveFiLMWithDropout(
            input_dim=static_dim,
            feature_dim=out_channels,
            num_layers=2,
            dropout=dropout,
            film_dropout=film_dropout,
            use_cfg=True,
            use_layer_scale=True,
            activation=activation
        )
        
        # ============== RESIDUAL CONNECTIONS ==============
        # Global residual connection from input to output
        if in_channels == out_channels:
            self.global_residual = True
            self.residual_downsample = DownsampleBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                downsample_factor=downsample_factor,
                method='conv',
                kernel_size=1,
                normalization='none',
                activation='identity'
            )
        else:
            self.global_residual = False
            self.residual_downsample = None
        
        # Final normalization
        self.final_norm = nn.LayerNorm(out_channels)
        
        # Initialize weights (handled by parent)
        # self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following best practices."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        static_params: torch.Tensor,
        cfg_guidance_scale: float = 1.0,
        force_uncond: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through enhanced encoder block.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            static_params: Static conditioning parameters [batch, static_dim]
            cfg_guidance_scale: CFG guidance scale
            force_uncond: Force unconditional processing
            
        Returns:
            Processed and downsampled tensor [batch, out_channels, seq_len // downsample_factor]
        """
        # Store input for global residual
        if self.global_residual:
            residual_input = x
        
        # ============== CONVOLUTION PROCESSING ==============
        # Input projection
        x = self.input_projection(x)
        
        # Apply enhanced convolution blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Downsampling
        x = self.downsample(x)
        
        # ============== POSITIONAL ENCODING ==============
        if self.pos_encoding is not None:
            # Convert to [batch, seq_len, channels] for positional encoding
            x = x.transpose(1, 2)  # [batch, seq_len, channels]
            x = self.pos_encoding(x)
            x = x.transpose(1, 2)  # [batch, channels, seq_len]
        
        # ============== S4 PROCESSING ==============
        # Apply S4 layers with residual connections
        for s4_layer in self.s4_layers:
            x_s4 = s4_layer(x)
            x = x + x_s4  # Residual connection
        
        # ============== FILM CONDITIONING ==============
        # Apply FiLM conditioning AFTER S4 processing
        x = self.film_layer(
            features=x,
            condition=static_params,
            cfg_guidance_scale=cfg_guidance_scale,
            force_uncond=force_uncond
        )
        
        # ============== GLOBAL RESIDUAL CONNECTION ==============
        if self.global_residual:
            # Downsample input and add as residual
            residual = self.residual_downsample(residual_input)
            if residual.size(-1) == x.size(-1):
                x = x + residual
        
        # ============== FINAL NORMALIZATION ==============
        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        x = self.final_norm(x)
        x = x.transpose(1, 2)  # [batch, channels, seq_len]
        
        return x


class MultiScaleEncoderStack(nn.Module):
    """
    Multi-scale encoder stack with hierarchical processing.
    
    Implements the complete encoder pathway with multiple levels,
    proper skip connection management, and advanced conditioning.
    """
    
    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        static_dim: int,
        sequence_length: int,
        n_levels: int = 4,
        channel_multiplier: float = 2.0,
        s4_state_size: int = 64,
        num_s4_layers: int = 2,
        dropout: float = 0.1,
        film_dropout: float = 0.15,
        use_enhanced_s4: bool = True,
        use_depth_wise_conv: bool = False
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.static_dim = static_dim
        self.sequence_length = sequence_length
        self.n_levels = n_levels
        
        # Calculate channel dimensions for each level
        self.encoder_channels = [
            int(base_channels * (channel_multiplier ** i)) for i in range(n_levels)
        ]
        
        # Input projection
        self.input_projection = EnhancedConvBlock(
            in_channels=input_channels,
            out_channels=base_channels,
            kernel_size=7,
            dropout=dropout
        )
        
        # Static parameter embedding
        self.static_embedding = nn.Sequential(
            nn.Linear(static_dim, static_dim * 2),
            nn.GELU(),
            nn.LayerNorm(static_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(static_dim * 2, static_dim * 2)
        )
        
        # Encoder levels
        self.encoder_levels = nn.ModuleList()
        
        for i in range(n_levels):
            in_ch = base_channels if i == 0 else self.encoder_channels[i - 1]
            out_ch = self.encoder_channels[i]
            seq_len = sequence_length // (2 ** i)
            
            encoder_level = EnhancedEncoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                static_dim=static_dim * 2,
                sequence_length=seq_len,
                downsample_factor=2,
                s4_state_size=s4_state_size,
                num_s4_layers=num_s4_layers,
                dropout=dropout,
                film_dropout=film_dropout,
                use_enhanced_s4=use_enhanced_s4,
                use_depth_wise_conv=use_depth_wise_conv
            )
            self.encoder_levels.append(encoder_level)
        
        # Initialize weights (handled by parent)
        # self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following best practices."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        static_params: torch.Tensor,
        cfg_guidance_scale: float = 1.0,
        force_uncond: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through multi-scale encoder stack.
        
        Args:
            x: Input signal [batch, input_channels, sequence_length]
            static_params: Static parameters [batch, static_dim]
            cfg_guidance_scale: CFG guidance scale
            force_uncond: Force unconditional processing
            
        Returns:
            Tuple of (final_encoding, skip_connections_list, embedded_static_params, encoder_outputs_list)
        """
        # Embed static parameters
        static_emb = self.static_embedding(static_params)
        
        # Input projection
        x = self.input_projection(x)  # [batch, base_channels, seq_len]
        
        # Encoder forward pass with skip collection and encoder outputs
        skip_connections = []
        encoder_outputs = []
        
        for i, encoder_level in enumerate(self.encoder_levels):
            # Save skip connection before processing
            skip_connections.append(x)
            
            # Process through encoder level
            x = encoder_level(
                x=x, 
                static_params=static_emb,
                cfg_guidance_scale=cfg_guidance_scale,
                force_uncond=force_uncond
            )
            
            # Save encoder output after processing (for cross-attention)
            encoder_outputs.append(x)
        
        return x, skip_connections, static_emb, encoder_outputs 