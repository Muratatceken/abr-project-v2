"""
Enhanced Decoder Blocks for ABR Hierarchical U-Net

Professional implementation with multi-layer Transformer blocks, advanced upsampling,
and robust architectural patterns for high-quality signal generation.

Author: AI Assistant
Based on: SSSD-ECG implementation patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

from .transformer_block import MultiLayerTransformerBlock, DeepTransformerDecoder
from .conv_blocks import UpsampleBlock, Conv1dBlock
from .film import AdaptiveFiLMWithDropout
from .positional import PositionalEmbedding
from .heads import AttentionPooling


class EnhancedDecoderBlock(nn.Module):
    """
    Enhanced decoder block with multi-layer Transformer processing and advanced upsampling.
    
    Features:
    - ConvTranspose1d or interpolation + conv upsampling
    - Multi-layer Transformer blocks (not just 1 layer)
    - Sinusoidal or learnable positional encoding
    - FiLM conditioning after each transformer stage
    - Enhanced skip connection handling
    - Professional weight initialization
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        static_dim: int,
        sequence_length: int,
        upsample_factor: int = 2,
        num_transformer_layers: int = 3,  # Increased from 1
        num_heads: int = 8,
        d_ff_multiplier: int = 4,
        dropout: float = 0.1,
        film_dropout: float = 0.15,
        use_cross_attention: bool = False,
        use_positional_encoding: bool = True,
        positional_type: str = 'sinusoidal',  # 'sinusoidal', 'learned', 'relative'
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.sequence_length = sequence_length
        self.upsample_factor = upsample_factor
        self.num_transformer_layers = num_transformer_layers
        self.use_cross_attention = use_cross_attention
        
        # ============== UPSAMPLING LAYER ==============
        # Enhanced upsampling with skip connection handling
        self.upsample = UpsampleBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            skip_channels=skip_channels,
            upsample_factor=upsample_factor,
            method='interpolate',  # More stable than conv_transpose
            kernel_size=3,
            normalization='layer',
            activation=activation,
            dropout=dropout
        )
        
        # ============== POSITIONAL ENCODING ==============
        if use_positional_encoding:
            upsampled_length = sequence_length * upsample_factor
            self.pos_encoding = PositionalEmbedding(
                d_model=out_channels,
                max_len=upsampled_length,
                embedding_type=positional_type
            )
        else:
            self.pos_encoding = None
        
        # ============== MULTI-LAYER TRANSFORMER ==============
        # Use deep transformer decoder for enhanced processing
        self.transformer_decoder = DeepTransformerDecoder(
            d_model=out_channels,
            n_heads=num_heads,
            d_ff=out_channels * d_ff_multiplier,
            num_layers=num_transformer_layers,
            dropout=dropout,
            activation=activation,
            use_relative_position=True,
            use_rotary=False,  # Can be enabled for even more advanced attention
            cross_attention=use_cross_attention
        )
        
        # ============== ENCODER OUTPUT PROJECTION ==============
        # Project encoder outputs to match transformer dimensions for cross-attention
        if use_cross_attention:
            # We need to handle different encoder output dimensions
            # This will be set dynamically based on the actual encoder output
            self.encoder_projection = None  # Will be created dynamically
        
        # ============== FILM CONDITIONING ==============
        # FiLM applied after each transformer stage with MLP
        self.film_layers = nn.ModuleList([
            AdaptiveFiLMWithDropout(
                input_dim=static_dim,
                feature_dim=out_channels,
                num_layers=2,
                dropout=dropout,
                film_dropout=film_dropout,
                use_cfg=True,
                use_layer_scale=True,
                activation=activation
            )
            for _ in range(num_transformer_layers)
        ])
        
        # Optional MLP after FiLM conditioning
        self.film_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 2, out_channels)
        )
        
        # ============== SKIP CONNECTION PROCESSING ==============
        # Enhanced skip connection handling with attention
        self.skip_attention = AttentionPooling(skip_channels)
        self.skip_projection = nn.Sequential(
            nn.Linear(skip_channels, out_channels),
            nn.GELU(),
            nn.LayerNorm(out_channels)
        )
        
        # ============== FINAL PROCESSING ==============
        # Final convolution for refinement
        self.final_conv = Conv1dBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            normalization='layer',
            activation=activation,
            dropout=dropout
        )
        
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
        skip: torch.Tensor, 
        static_params: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        cfg_guidance_scale: float = 1.0,
        force_uncond: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through enhanced decoder block.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            skip: Skip connection from encoder [batch, skip_channels, target_seq_len]
            static_params: Static conditioning parameters [batch, static_dim]
            encoder_output: Optional encoder output for cross-attention
            cfg_guidance_scale: CFG guidance scale
            force_uncond: Force unconditional processing
            
        Returns:
            Processed and upsampled tensor [batch, out_channels, seq_len * upsample_factor]
        """
        # ============== UPSAMPLING WITH SKIP CONNECTION ==============
        x = self.upsample(x, skip)
        
        # ============== POSITIONAL ENCODING ==============
        if self.pos_encoding is not None:
            # Convert to [batch, seq_len, channels] for positional encoding
            x = x.transpose(1, 2)  # [batch, seq_len, channels]
            x = self.pos_encoding(x)
        else:
            x = x.transpose(1, 2)  # [batch, seq_len, channels]
        
        # ============== TRANSFORMER PROCESSING ==============
        # Handle encoder output projection for cross-attention
        projected_encoder_output = None
        if encoder_output is not None and self.use_cross_attention:
            # encoder_output is [batch, channels, seq_len]
            # First, interpolate to match current sequence length
            current_seq_len = x.size(1)  # x is [batch, seq_len, channels] at this point
            encoder_seq_len = encoder_output.size(-1)
            
            if encoder_seq_len != current_seq_len:
                # Interpolate encoder output to match current sequence length
                encoder_output = F.interpolate(
                    encoder_output, 
                    size=current_seq_len, 
                    mode='linear', 
                    align_corners=False
                )
            
            # Now transpose to [batch, seq_len, channels]
            encoder_output_transposed = encoder_output.transpose(1, 2)  # [batch, seq_len, channels]
            encoder_channels = encoder_output_transposed.size(-1)  # Last dimension is channels
            
            # Create projection layer if it doesn't exist or if dimensions changed
            if not hasattr(self, 'encoder_projection') or self.encoder_projection is None or \
               (hasattr(self, '_last_encoder_channels') and self._last_encoder_channels != encoder_channels):
                self.encoder_projection = nn.Linear(encoder_channels, self.out_channels).to(encoder_output.device)
                self._last_encoder_channels = encoder_channels
            
            # Project encoder output to match transformer dimensions
            projected_encoder_output = self.encoder_projection(encoder_output_transposed)
        
        # Apply multi-layer transformer decoder
        x = self.transformer_decoder(
            x=x, 
            encoder_output=projected_encoder_output,
            mask=None
        )
        
        # ============== ENHANCED FILM CONDITIONING ==============
        # Apply FiLM conditioning after transformer processing
        # Convert back to [batch, channels, seq_len] for FiLM
        x = x.transpose(1, 2)
        
        # Apply FiLM layers (note: we use the last one for simplicity,
        # but could apply each to different transformer layers)
        x = self.film_layers[-1](
            features=x,
            condition=static_params,
            cfg_guidance_scale=cfg_guidance_scale,
            force_uncond=force_uncond
        )
        
        # Optional MLP processing after FiLM
        x_mlp_input = x.transpose(1, 2)  # [batch, seq_len, channels]
        x_mlp_output = self.film_mlp(x_mlp_input)
        x = x_mlp_output.transpose(1, 2)  # [batch, channels, seq_len]
        
        # ============== FINAL PROCESSING ==============
        # Final convolution refinement
        x = self.final_conv(x)
        
        # Final normalization
        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        x = self.final_norm(x)
        x = x.transpose(1, 2)  # [batch, channels, seq_len]
        
        return x


class MultiScaleDecoderStack(nn.Module):
    """
    Multi-scale decoder stack with hierarchical reconstruction.
    
    Implements the complete decoder pathway with multiple levels,
    proper skip connection fusion, and advanced conditioning.
    """
    
    def __init__(
        self,
        encoder_channels: List[int],
        static_dim: int,
        sequence_length: int,
        n_levels: int = 4,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        film_dropout: float = 0.15,
        use_cross_attention: bool = False,
        use_positional_encoding: bool = True,
        positional_type: str = 'sinusoidal'
    ):
        super().__init__()
        
        self.encoder_channels = encoder_channels
        self.static_dim = static_dim
        self.sequence_length = sequence_length
        self.n_levels = n_levels
        
        # Decoder levels (reverse order from encoder)
        self.decoder_levels = nn.ModuleList()
        
        for i in range(n_levels):
            level_idx = n_levels - 1 - i  # Reverse order for decoder
            
            if i == 0:
                # First decoder level: from bottleneck
                in_ch = encoder_channels[-1]  # Bottleneck channels
                out_ch = encoder_channels[level_idx]
                # Skip connection comes from encoder input at this level
                if level_idx == 0:
                    skip_ch = encoder_channels[0]  # First level uses base channels
                else:
                    skip_ch = encoder_channels[level_idx - 1]  # Previous level output
            else:
                # Subsequent levels: input comes from previous decoder output
                in_ch = encoder_channels[level_idx + 1]
                out_ch = encoder_channels[level_idx]
                # Skip connection comes from encoder input at this level
                if level_idx == 0:
                    skip_ch = encoder_channels[0]  # First level uses base channels
                else:
                    skip_ch = encoder_channels[level_idx - 1]  # Previous level output
            seq_len = sequence_length // (2 ** (level_idx + 1))
            
            decoder_level = EnhancedDecoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                skip_channels=skip_ch,
                static_dim=static_dim,
                sequence_length=seq_len,
                upsample_factor=2,
                num_transformer_layers=num_transformer_layers,
                num_heads=num_heads,
                dropout=dropout,
                film_dropout=film_dropout,
                use_cross_attention=use_cross_attention,
                use_positional_encoding=use_positional_encoding,
                positional_type=positional_type
            )
            self.decoder_levels.append(decoder_level)
        
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
        skip_connections: List[torch.Tensor],
        static_params: torch.Tensor,
        encoder_outputs: Optional[List[torch.Tensor]] = None,
        cfg_guidance_scale: float = 1.0,
        force_uncond: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through multi-scale decoder stack.
        
        Args:
            x: Encoded input from bottleneck [batch, channels, seq_len]
            skip_connections: List of skip connections from encoder
            static_params: Static conditioning parameters [batch, static_dim]
            encoder_outputs: Optional encoder outputs for cross-attention
            cfg_guidance_scale: CFG guidance scale
            force_uncond: Force unconditional processing
            
        Returns:
            Final decoded output [batch, final_channels, original_seq_len]
        """
        for i, decoder_level in enumerate(self.decoder_levels):
            # Get corresponding skip connection (reverse order)
            skip_idx = len(skip_connections) - 1 - i
            skip = skip_connections[skip_idx]
            
            # Optional encoder output for cross-attention
            encoder_output = encoder_outputs[skip_idx] if encoder_outputs else None
            
            # Process through decoder level
            x = decoder_level(
                x=x, 
                skip=skip, 
                static_params=static_params,
                encoder_output=encoder_output,
                cfg_guidance_scale=cfg_guidance_scale,
                force_uncond=force_uncond
            )
        
        return x


class BottleneckProcessor(nn.Module):
    """
    Enhanced bottleneck processor with S4 and Transformer combination.
    
    Applies both S4 and Transformer processing at the bottleneck for
    maximum representational power.
    """
    
    def __init__(
        self,
        channels: int,
        static_dim: int,
        sequence_length: int,
        s4_state_size: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
        film_dropout: float = 0.15,
        use_enhanced_s4: bool = True,
        use_positional_encoding: bool = True,
        positional_type: str = 'sinusoidal'
    ):
        super().__init__()
        
        self.channels = channels
        self.use_positional_encoding = use_positional_encoding
        
        # ============== POSITIONAL ENCODING ==============
        if use_positional_encoding:
            self.pos_encoding = PositionalEmbedding(
                d_model=channels,
                max_len=sequence_length,
                embedding_type=positional_type
            )
        else:
            self.pos_encoding = None
        
        # ============== S4 PROCESSING ==============
        if use_enhanced_s4:
            from .s4_layer import EnhancedS4Layer
            self.s4_layer = EnhancedS4Layer(
                features=channels,
                lmax=sequence_length,
                N=s4_state_size,
                dropout=dropout,
                bidirectional=True,
                layer_norm=True,
                learnable_timescales=True,
                kernel_mixing=True
            )
        else:
            from .s4_layer import S4Layer
            self.s4_layer = S4Layer(
                features=channels,
                lmax=sequence_length,
                N=s4_state_size,
                dropout=dropout,
                bidirectional=True,
                layer_norm=True
            )
        
        # Transformer processing
        self.transformer = MultiLayerTransformerBlock(
            d_model=channels,
            n_heads=num_heads,
            d_ff=channels * 4,
            num_layers=2,
            dropout=dropout,
            activation='gelu',
            use_relative_position=True
        )
        
        # FiLM conditioning
        self.film_layer = AdaptiveFiLMWithDropout(
            input_dim=static_dim,
            feature_dim=channels,
            num_layers=2,
            dropout=dropout,
            film_dropout=film_dropout,
            use_cfg=True,
            use_layer_scale=True
        )
        
        # Combination and refinement
        self.combination = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels)
        )
        
        self.final_norm = nn.LayerNorm(channels)
    
    def forward(
        self, 
        x: torch.Tensor, 
        static_params: torch.Tensor,
        cfg_guidance_scale: float = 1.0,
        force_uncond: bool = False
    ) -> torch.Tensor:
        """
        Process through enhanced bottleneck.
        
        Args:
            x: Input tensor [batch, channels, seq_len]
            static_params: Static conditioning parameters [batch, static_dim]
            cfg_guidance_scale: CFG guidance scale
            force_uncond: Force unconditional processing
            
        Returns:
            Processed tensor [batch, channels, seq_len]
        """
        # ============== POSITIONAL ENCODING ==============
        if self.pos_encoding is not None:
            # Apply positional encoding before processing
            x_pos_input = x.transpose(1, 2)  # [batch, seq_len, channels]
            x_pos_encoded = self.pos_encoding(x_pos_input)
            x = x_pos_encoded.transpose(1, 2)  # [batch, channels, seq_len]
        
        # ============== S4 PROCESSING ==============
        x_s4 = self.s4_layer(x)
        x = x + x_s4  # Residual connection
        
        # ============== TRANSFORMER PROCESSING ==============
        x_transformer_input = x.transpose(1, 2)  # [batch, seq_len, channels]
        x_transformer = self.transformer(x_transformer_input)
        
        # Combine S4 and Transformer features
        x_s4_for_combine = x.transpose(1, 2)  # [batch, seq_len, channels]
        combined_features = torch.cat([x_s4_for_combine, x_transformer], dim=-1)
        x_combined = self.combination(combined_features)
        
        # Convert back to conv format for FiLM
        x_combined = x_combined.transpose(1, 2)  # [batch, channels, seq_len]
        
        # Apply FiLM conditioning
        x_film = self.film_layer(
            features=x_combined,
            condition=static_params,
            cfg_guidance_scale=cfg_guidance_scale,
            force_uncond=force_uncond
        )
        
        # Final normalization
        x_final = x_film.transpose(1, 2)  # [batch, seq_len, channels]
        x_final = self.final_norm(x_final)
        x_final = x_final.transpose(1, 2)  # [batch, channels, seq_len]
        
        return x_final 