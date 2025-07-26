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
            input_dim=static_dim * 2,  # Static params are embedded to 2x size
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


class OptimizedEncoderBlock(nn.Module):
    """
    Optimized encoder block with proper transformer placement for long sequences.
    
    Architectural flow:
    Conv → Transformer (long sequences) → S4 → Downsample → FiLM
    
    This addresses the critical issue of transformer placement by using attention
    on long sequences where it's most effective.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        static_dim: int,
        sequence_length: int,
        downsample_factor: int = 2,
        s4_state_size: int = 64,
        num_s4_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        film_dropout: float = 0.15,
        use_enhanced_s4: bool = True,
        use_multi_scale_attention: bool = True,
        use_attention_skip: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequence_length = sequence_length
        self.downsample_factor = downsample_factor
        self.use_attention_skip = use_attention_skip
        
        # ============== CONVOLUTION PROCESSING ==============
        # Enhanced convolution blocks
        self.conv_blocks = nn.ModuleList([
            EnhancedConvBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=3,
                dropout=dropout
            )
            for i in range(2)
        ])
        
        # ============== TRANSFORMER PROCESSING (LONG SEQUENCES) ==============
        # Use transformer BEFORE downsampling when sequences are long
        if sequence_length >= 50:  # Only use transformer for long sequences
            if use_multi_scale_attention:
                from .transformer_block import EnhancedTransformerBlock
                self.transformer_layers = nn.ModuleList([
                    EnhancedTransformerBlock(
                        d_model=out_channels,
                        n_heads=num_heads,
                        dropout=dropout,
                        use_multi_scale=True,
                        use_cross_attention=False  # No cross attention in encoder
                    )
                    for _ in range(num_transformer_layers)
                ])
            else:
                from .transformer_block import TransformerBlock
                self.transformer_layers = nn.ModuleList([
                    TransformerBlock(
                        d_model=out_channels,
                        n_heads=num_heads,
                        dropout=dropout
                    )
                    for _ in range(num_transformer_layers)
                ])
            self.use_transformer = True
        else:
            self.transformer_layers = None
            self.use_transformer = False
        
        # ============== DOWNSAMPLING ==============
        self.downsample = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=downsample_factor * 2,
            stride=downsample_factor,
            padding=downsample_factor // 2
        )
        
        # ============== S4 PROCESSING (AFTER DOWNSAMPLING) ==============
        # S4 works better on shorter sequences after downsampling
        if use_enhanced_s4:
            from .s4_layer import EnhancedS4Layer
            self.s4_layers = nn.ModuleList([
                EnhancedS4Layer(
                    features=out_channels,
                    lmax=sequence_length,
                    N=s4_state_size,
                    dropout=dropout
                )
                for _ in range(num_s4_layers)
            ])
        else:
            from .s4_layer import S4Layer
            self.s4_layers = nn.ModuleList([
                S4Layer(
                    d_model=out_channels,
                    d_state=s4_state_size,
                    dropout=dropout
                )
                for _ in range(num_s4_layers)
            ])
        
        # ============== FILM CONDITIONING ==============
        from .film import AdaptiveFiLMWithDropout
        self.film_layer = AdaptiveFiLMWithDropout(
            input_dim=static_dim * 2,  # Static params are embedded to 2x size
            feature_dim=out_channels,
            dropout=film_dropout
        )
        
        # ============== ATTENTION-BASED SKIP CONNECTIONS ==============
        if use_attention_skip:
            self.skip_attention = AttentionSkipConnection(
                in_channels=in_channels,
                out_channels=out_channels,
                downsample_factor=downsample_factor
            )
        
        # ============== NORMALIZATION ==============
        self.final_norm = nn.LayerNorm(out_channels)
        
        # ============== POSITIONAL ENCODING ==============
        # Only for transformer layers
        if self.use_transformer:
            from .positional import SinusoidalEmbedding
            self.pos_encoding = SinusoidalEmbedding(
                d_model=out_channels,
                max_len=sequence_length
            )
    
    def forward(
        self,
        x: torch.Tensor,
        static_params: torch.Tensor,
        cfg_guidance_scale: float = 1.0,
        force_uncond: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass with proper transformer placement.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            static_params: Static conditioning parameters [batch, static_dim]
            cfg_guidance_scale: CFG guidance scale
            force_uncond: Force unconditional processing
            
        Returns:
            Tuple of (processed_tensor, skip_connection)
        """
        # Store input for skip connection
        skip_input = x
        
        # ============== CONVOLUTION PROCESSING ==============
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # ============== TRANSFORMER PROCESSING (LONG SEQUENCES) ==============
        if self.use_transformer and x.size(-1) >= 50:
            # Convert to transformer format [batch, seq_len, channels]
            x = x.transpose(1, 2)
            
            # Add positional encoding
            x = self.pos_encoding(x)
            
            # Apply transformer layers
            for transformer_layer in self.transformer_layers:
                x = transformer_layer(x)
            
            # Convert back to conv format [batch, channels, seq_len]
            x = x.transpose(1, 2)
        
        # ============== DOWNSAMPLING ==============
        x = self.downsample(x)
        
        # ============== S4 PROCESSING (SHORT SEQUENCES) ==============
        # S4 is more effective on shorter sequences after downsampling
        for s4_layer in self.s4_layers:
            x_s4 = s4_layer(x)
            x = x + x_s4  # Residual connection
        
        # ============== FILM CONDITIONING ==============
        x = self.film_layer(
            features=x,
            condition=static_params,
            cfg_guidance_scale=cfg_guidance_scale,
            force_uncond=force_uncond
        )
        
        # ============== FINAL NORMALIZATION ==============
        x = x.transpose(1, 2)
        x = self.final_norm(x)
        x = x.transpose(1, 2)
        
        # ============== ATTENTION-BASED SKIP CONNECTION ==============
        if self.use_attention_skip:
            skip_connection = self.skip_attention(skip_input, x)
        else:
            # Simple downsampling for skip connection
            skip_connection = F.avg_pool1d(skip_input, kernel_size=self.downsample_factor)
            if skip_connection.size(1) != self.out_channels:
                skip_connection = F.conv1d(
                    skip_connection,
                    weight=torch.eye(min(skip_connection.size(1), self.out_channels))
                    .unsqueeze(-1).to(skip_connection.device),
                    padding=0
                )
        
        return x, skip_connection


class OptimizedBottleneckProcessor(nn.Module):
    """
    Optimized bottleneck processor using S4-only for short sequences.
    
    This addresses the critical issue of using transformers on sequences
    that are too short (~12 tokens). Uses S4 layers which are more
    effective for short sequences at the bottleneck.
    """
    
    def __init__(
        self,
        channels: int,
        static_dim: int,
        sequence_length: int,
        s4_state_size: int = 64,
        num_s4_layers: int = 3,  # More S4 layers since no transformer
        dropout: float = 0.1,
        film_dropout: float = 0.15,
        use_enhanced_s4: bool = True,
        use_residual_s4: bool = True
    ):
        super().__init__()
        
        self.channels = channels
        self.sequence_length = sequence_length
        self.use_residual_s4 = use_residual_s4
        
        # Input processing
        self.input_norm = nn.LayerNorm(channels)
        
        # ============== S4 PROCESSING ONLY ==============
        # No transformer here - sequences are too short (~12 tokens)
        # S4 is more effective for short sequences
        if use_enhanced_s4:
            from .s4_layer import EnhancedS4Layer
            self.s4_layers = nn.ModuleList([
                EnhancedS4Layer(
                    features=channels,
                    lmax=sequence_length,
                    N=s4_state_size,
                    dropout=dropout,
                    bidirectional=True  # Use bidirectional for bottleneck
                )
                for _ in range(num_s4_layers)
            ])
        else:
            from .s4_layer import S4Layer
            self.s4_layers = nn.ModuleList([
                S4Layer(
                    d_model=channels,
                    d_state=s4_state_size,
                    dropout=dropout
                )
                for _ in range(num_s4_layers)
            ])
        
        # ============== ENHANCED BOTTLENECK PROCESSING ==============
        # Deep feature processing at the bottleneck
        self.bottleneck_processing = nn.Sequential(
            nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, channels * 2),  # Use GroupNorm for Conv1d output
            nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, channels)  # Use GroupNorm for Conv1d output
        )
        
        # ============== FILM CONDITIONING ==============
        from .film import AdaptiveFiLMWithDropout
        self.film_layer = AdaptiveFiLMWithDropout(
            input_dim=static_dim * 2,  # Static params are embedded to 2x size
            feature_dim=channels,
            dropout=film_dropout
        )
        
        # ============== GLOBAL CONTEXT MODELING ==============
        # Since we can't use transformer, use alternative global modeling
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Conv1d(channels, channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()  # Attention weights
        )
        
        # ============== OUTPUT PROCESSING ==============
        self.output_norm = nn.LayerNorm(channels)
        
    def forward(
        self,
        x: torch.Tensor,
        static_params: torch.Tensor,
        cfg_guidance_scale: float = 1.0,
        force_uncond: bool = False
    ) -> torch.Tensor:
        """
        Process features through optimized bottleneck.
        
        Args:
            x: Input features [batch, channels, short_seq_len]
            static_params: Static conditioning [batch, static_dim]
            cfg_guidance_scale: CFG guidance scale
            force_uncond: Force unconditional processing
            
        Returns:
            Processed features [batch, channels, short_seq_len]
        """
        # Store input for residual connection
        residual = x
        
        # Input normalization
        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        x = self.input_norm(x)
        x = x.transpose(1, 2)  # [batch, channels, seq_len]
        
        # ============== S4 PROCESSING ==============
        # Apply multiple S4 layers with residual connections
        for i, s4_layer in enumerate(self.s4_layers):
            x_s4 = s4_layer(x)
            if self.use_residual_s4:
                x = x + x_s4  # Residual connection
            else:
                x = x_s4
        
        # ============== ENHANCED BOTTLENECK PROCESSING ==============
        x = self.bottleneck_processing(x)
        
        # ============== GLOBAL CONTEXT MODELING ==============
        # Apply global context attention
        global_weights = self.global_context(x)
        x = x * global_weights  # Apply attention
        
        # ============== FILM CONDITIONING ==============
        x = self.film_layer(
            features=x,
            condition=static_params,
            cfg_guidance_scale=cfg_guidance_scale,
            force_uncond=force_uncond
        )
        
        # ============== FINAL PROCESSING ==============
        # Output normalization
        x = x.transpose(1, 2)
        x = self.output_norm(x)
        x = x.transpose(1, 2)
        
        # Global residual connection
        x = x + residual
        
        return x


class AttentionSkipConnection(nn.Module):
    """
    Attention-based skip connection that learns which features to preserve
    across encoder-decoder levels.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_factor: int = 2,
        attention_heads: int = 4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_factor = downsample_factor
        
        # Channel alignment
        if in_channels != out_channels:
            self.channel_align = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.channel_align = nn.Identity()
        
        # Downsampling
        self.downsample = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=downsample_factor * 2,
            stride=downsample_factor,
            padding=downsample_factor // 2
        )
        
        # Attention mechanism for feature selection
        self.attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(1, out_channels),  # Use GroupNorm for Conv1d output
        )
    
    def forward(self, skip_input: torch.Tensor, current_features: torch.Tensor) -> torch.Tensor:
        """
        Create attention-weighted skip connection.
        
        Args:
            skip_input: Original input features [batch, in_channels, seq_len]
            current_features: Current processed features [batch, out_channels, short_seq_len]
            
        Returns:
            Enhanced skip connection [batch, out_channels, short_seq_len]
        """
        # Align channels
        skip_aligned = self.channel_align(skip_input)
        
        # Downsample to match current features
        skip_downsampled = self.downsample(skip_aligned)
        
        # Ensure matching sequence lengths
        if skip_downsampled.size(-1) != current_features.size(-1):
            skip_downsampled = F.interpolate(
                skip_downsampled,
                size=current_features.size(-1),
                mode='linear',
                align_corners=False
            )
        
        # Apply attention for feature selection
        # Convert to [batch, seq_len, channels] for attention
        skip_attn_input = skip_downsampled.transpose(1, 2)
        current_attn_input = current_features.transpose(1, 2)
        
        # Use current features as query, skip features as key/value
        attended_skip, _ = self.attention(
            query=current_attn_input,
            key=skip_attn_input,
            value=skip_attn_input
        )
        
        # Convert back to [batch, channels, seq_len]
        attended_skip = attended_skip.transpose(1, 2)
        
        # Fuse attended skip with current features
        fused_features = torch.cat([attended_skip, current_features], dim=1)
        enhanced_skip = self.fusion(fused_features)
        
        return enhanced_skip 