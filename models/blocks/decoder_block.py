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

from .transformer_block import MultiLayerTransformerBlock, DeepTransformerDecoder, CrossAttentionTransformerBlock
from .conv_blocks import UpsampleBlock, Conv1dBlock, EnhancedConvBlock
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
            method='conv_transpose',  # Sharper than interpolate for transients
            kernel_size=upsample_factor * 2 + 1,
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
                input_dim=static_dim * 2,  # Static params are embedded to 2x size
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
            input_dim=static_dim * 2,  # Static params are embedded to 2x size
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


class OptimizedDecoderBlock(nn.Module):
    """
    Optimized decoder block with proper transformer placement.
    
    Architectural flow:
    S4 (short sequences) → Upsample → Transformer (long sequences) → Skip Fusion → FiLM
    
    This addresses the architectural issue by using S4 on short sequences
    and transformers after upsampling when sequences are longer.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        static_dim: int,
        sequence_length: int,
        upsample_factor: int = 2,
        s4_state_size: int = 64,
        num_s4_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        film_dropout: float = 0.15,
        use_enhanced_s4: bool = True,
        use_cross_attention: bool = True,
        use_multi_scale_attention: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.sequence_length = sequence_length
        self.upsample_factor = upsample_factor
        self.use_cross_attention = use_cross_attention
        
        # ============== S4 PROCESSING (SHORT SEQUENCES FIRST) ==============
        # S4 works well on short sequences at decoder input
        if use_enhanced_s4:
            from .s4_layer import EnhancedS4Layer
            self.s4_layers = nn.ModuleList([
                EnhancedS4Layer(
                    features=in_channels,
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
                    d_model=in_channels,
                    d_state=s4_state_size,
                    dropout=dropout
                )
                for _ in range(num_s4_layers)
            ])
        
        # ============== UPSAMPLING ==============
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=upsample_factor * 2,
            stride=upsample_factor,
            padding=upsample_factor // 2
        )
        
        # ============== SKIP CONNECTION FUSION ==============
        self.skip_fusion = EnhancedSkipFusion(
            main_channels=out_channels,
            skip_channels=skip_channels,
            output_channels=out_channels,
            use_attention=True
        )
        
        # ============== TRANSFORMER PROCESSING (LONG SEQUENCES AFTER UPSAMPLING) ==============
        # Expected sequence length after upsampling
        upsampled_seq_len = sequence_length * upsample_factor
        
        # Use transformer AFTER upsampling when sequences are long
        if upsampled_seq_len >= 50:  # Only use transformer for long sequences
            if use_cross_attention:
                from .transformer_block import CrossAttentionTransformerBlock
                self.transformer_layers = nn.ModuleList([
                    CrossAttentionTransformerBlock(
                        d_model=out_channels,
                        n_heads=num_heads,
                        dropout=dropout,
                        use_pre_norm=True
                    )
                    for _ in range(num_transformer_layers)
                ])
            elif use_multi_scale_attention:
                from .transformer_block import EnhancedTransformerBlock
                self.transformer_layers = nn.ModuleList([
                    EnhancedTransformerBlock(
                        d_model=out_channels,
                        n_heads=num_heads,
                        dropout=dropout,
                        use_multi_scale=True,
                        use_cross_attention=False
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
        
        # ============== FILM CONDITIONING ==============
        from .film import AdaptiveFiLMWithDropout
        self.film_layer = AdaptiveFiLMWithDropout(
            input_dim=static_dim * 2,  # Static params are embedded to 2x size
            feature_dim=out_channels,
            dropout=film_dropout
        )
        
        # ============== FINAL PROCESSING ==============
        self.final_conv = EnhancedConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            dropout=dropout
        )
        
        # ============== NORMALIZATION ==============
        self.final_norm = nn.LayerNorm(out_channels)
        
        # ============== POSITIONAL ENCODING ==============
        if self.use_transformer:
            from .positional import SinusoidalEmbedding
            self.pos_encoding = SinusoidalEmbedding(
                d_model=out_channels,
                max_len=upsampled_seq_len
            )
    
    def forward(
        self,
        x: torch.Tensor,
        skip_connection: torch.Tensor,
        static_params: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        cfg_guidance_scale: float = 1.0,
        force_uncond: bool = False
    ) -> torch.Tensor:
        """
        Optimized forward pass with proper architectural flow.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            skip_connection: Skip connection [batch, skip_channels, seq_len]
            static_params: Static conditioning [batch, static_dim]
            encoder_output: Encoder output for cross-attention [batch, channels, enc_seq_len]
            cfg_guidance_scale: CFG guidance scale
            force_uncond: Force unconditional processing
            
        Returns:
            Output tensor [batch, out_channels, upsampled_seq_len]
        """
        # ============== S4 PROCESSING (SHORT SEQUENCES) ==============
        # Process short sequences with S4 first
        for s4_layer in self.s4_layers:
            x_s4 = s4_layer(x)
            x = x + x_s4  # Residual connection
        
        # ============== UPSAMPLING ==============
        x = self.upsample(x)
        
        # ============== SKIP CONNECTION FUSION ==============
        # Fuse with skip connection after upsampling
        x = self.skip_fusion(x, skip_connection)
        
        # ============== TRANSFORMER PROCESSING (LONG SEQUENCES) ==============
        if self.use_transformer and x.size(-1) >= 50:
            # Convert to transformer format [batch, seq_len, channels]
            x = x.transpose(1, 2)
            
            # Add positional encoding
            x = self.pos_encoding(x)
            
            # Apply transformer layers
            for transformer_layer in self.transformer_layers:
                if self.use_cross_attention and encoder_output is not None:
                    # Convert encoder output to transformer format
                    encoder_transformer = encoder_output.transpose(1, 2)
                    
                    # Apply cross-attention transformer
                    if isinstance(transformer_layer, CrossAttentionTransformerBlock):
                        x, self_attn_weights, cross_attn_weights = transformer_layer(
                            decoder_input=x,
                            encoder_output=encoder_transformer
                        )
                    else:
                        x = transformer_layer(x, encoder_output=encoder_transformer)
                else:
                    # Apply regular transformer
                    x = transformer_layer(x)
            
            # Convert back to conv format [batch, channels, seq_len]
            x = x.transpose(1, 2)
        
        # ============== FILM CONDITIONING ==============
        x = self.film_layer(
            features=x,
            condition=static_params,
            cfg_guidance_scale=cfg_guidance_scale,
            force_uncond=force_uncond
        )
        
        # ============== FINAL PROCESSING ==============
        x = self.final_conv(x)
        
        # ============== FINAL NORMALIZATION ==============
        x = x.transpose(1, 2)
        x = self.final_norm(x)
        x = x.transpose(1, 2)
        
        return x


class EnhancedSkipFusion(nn.Module):
    """
    Enhanced skip connection fusion with attention-based feature selection.
    
    Learns to selectively combine main features with skip connections
    based on content and relevance.
    """
    
    def __init__(
        self,
        main_channels: int,
        skip_channels: int,
        output_channels: int,
        use_attention: bool = True,
        attention_heads: int = 4
    ):
        super().__init__()
        
        self.main_channels = main_channels
        self.skip_channels = skip_channels
        self.output_channels = output_channels
        self.use_attention = use_attention
        
        # Channel alignment for skip connection
        if skip_channels != main_channels:
            self.skip_align = nn.Conv1d(skip_channels, main_channels, kernel_size=1)
        else:
            self.skip_align = nn.Identity()
        
        # Attention-based fusion
        if use_attention:
            self.attention_fusion = nn.MultiheadAttention(
                embed_dim=main_channels,
                num_heads=attention_heads,
                dropout=0.1,
                batch_first=True
            )
            
            # Gating mechanism for fusion control
            self.fusion_gate = nn.Sequential(
                nn.Conv1d(main_channels * 2, main_channels, kernel_size=1),
                nn.Sigmoid()
            )
        
        # Final fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Conv1d(main_channels * 2, output_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, output_channels),
            nn.Dropout(0.1),
            nn.Conv1d(output_channels, output_channels, kernel_size=1)
        )
    
    def forward(self, main_features: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse main features with skip connection using attention.
        
        Args:
            main_features: Main pathway features [batch, main_channels, seq_len]
            skip_features: Skip connection features [batch, skip_channels, skip_seq_len]
            
        Returns:
            Fused features [batch, output_channels, seq_len]
        """
        # Align skip connection channels
        skip_aligned = self.skip_align(skip_features)
        
        # Handle sequence length mismatch
        if skip_aligned.size(-1) != main_features.size(-1):
            skip_aligned = F.interpolate(
                skip_aligned,
                size=main_features.size(-1),
                mode='linear',
                align_corners=False
            )
        
        if self.use_attention:
            # Attention-based fusion
            # Convert to [batch, seq_len, channels] for attention
            main_attn = main_features.transpose(1, 2)
            skip_attn = skip_aligned.transpose(1, 2)
            
            # Use main features as query, skip features as key/value
            attended_skip, _ = self.attention_fusion(
                query=main_attn,
                key=skip_attn,
                value=skip_attn
            )
            
            # Convert back to [batch, channels, seq_len]
            attended_skip = attended_skip.transpose(1, 2)
            
            # Gated fusion
            combined = torch.cat([main_features, attended_skip], dim=1)
            gate = self.fusion_gate(combined)
            gated_skip = attended_skip * gate
            
            # Final combination
            final_combined = torch.cat([main_features, gated_skip], dim=1)
        else:
            # Simple concatenation fusion
            final_combined = torch.cat([main_features, skip_aligned], dim=1)
        
        # Apply fusion layers
        fused_output = self.fusion_layers(final_combined)
        
        return fused_output


class TaskSpecificFeatureExtractor(nn.Module):
    """
    Task-specific feature extractor that learns specialized representations
    for different prediction tasks (peaks, classification, threshold).
    
    This addresses the multi-task learning issues by creating dedicated
    feature extraction pathways for each task.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        num_tasks: int = 4,  # signal, peaks, classification, threshold
        dropout: float = 0.1,
        use_cross_task_attention: bool = True
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.use_cross_task_attention = use_cross_task_attention
        
        # Task names for reference
        self.task_names = ['signal', 'peaks', 'classification', 'threshold']
        
        # Shared feature encoder
        self.shared_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),  # Use GroupNorm instead of LayerNorm for Conv1d
            nn.Dropout(dropout)
        )
        
        # Task-specific feature extractors
        self.task_extractors = nn.ModuleDict({
            'signal': self._create_signal_extractor(hidden_dim, dropout),
            'peaks': self._create_peak_extractor(hidden_dim, dropout),
            'classification': self._create_class_extractor(hidden_dim, dropout),
            'threshold': self._create_threshold_extractor(hidden_dim, dropout)
        })
        
        # Cross-task attention for feature sharing
        if use_cross_task_attention:
            self.cross_task_attention = nn.ModuleDict({
                task: nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=max(1, hidden_dim // 64),
                    dropout=dropout,
                    batch_first=True
                )
                for task in self.task_names
            })
        
        # Task-specific output projections
        self.task_projections = nn.ModuleDict({
            task: nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            for task in self.task_names
        })
    
    def _create_signal_extractor(self, hidden_dim: int, dropout: float) -> nn.Module:
        """Create specialized extractor for signal reconstruction."""
        return nn.Sequential(
            # Focus on temporal continuity and smoothness
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim)
        )
    
    def _create_peak_extractor(self, hidden_dim: int, dropout: float) -> nn.Module:
        """Create specialized extractor for peak detection."""
        return nn.Sequential(
            # Focus on local features and sharp changes
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def _create_class_extractor(self, hidden_dim: int, dropout: float) -> nn.Module:
        """Create specialized extractor for classification."""
        return nn.Sequential(
            # Focus on global patterns and morphology
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=11, padding=5),
            nn.GELU(),
            # Use dilated convolutions instead of adaptive pooling to preserve sequence length
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def _create_threshold_extractor(self, hidden_dim: int, dropout: float) -> nn.Module:
        """Create specialized extractor for threshold regression."""
        return nn.Sequential(
            # Focus on amplitude ranges and overall signal strength
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract task-specific features.
        
        Args:
            x: Input features [batch, input_dim, seq_len]
            
        Returns:
            Dictionary of task-specific features
        """
        # Shared feature encoding
        shared_features = self.shared_encoder(x)
        
        # Extract task-specific features
        task_features = {}
        for task_name, extractor in self.task_extractors.items():
            task_features[task_name] = extractor(shared_features)
        
        # Apply cross-task attention if enabled
        if self.use_cross_task_attention:
            enhanced_features = {}
            for task_name in self.task_names:
                # Current task features as query
                query = task_features[task_name].transpose(1, 2)  # [batch, seq, dim]
                
                # Other task features as context
                context_features = []
                for other_task in self.task_names:
                    if other_task != task_name:
                        context_features.append(task_features[other_task].transpose(1, 2))
                
                if context_features:
                    # Concatenate context from other tasks
                    context = torch.cat(context_features, dim=1)  # [batch, seq*3, dim]
                    
                    # Apply cross-attention
                    attended_features, _ = self.cross_task_attention[task_name](
                        query=query,
                        key=context,
                        value=context
                    )
                    
                    # Combine with original features
                    enhanced = query + attended_features
                    enhanced_features[task_name] = enhanced.transpose(1, 2)
                else:
                    enhanced_features[task_name] = task_features[task_name]
            
            task_features = enhanced_features
        
        # Apply final projections
        output_features = {}
        for task_name, features in task_features.items():
            output_features[task_name] = self.task_projections[task_name](features)
        
        return output_features 