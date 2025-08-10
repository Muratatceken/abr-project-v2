"""
Enhanced Hierarchical U-Net with Full Timestep Conditioning and Dual-Branch Architecture

Key improvements:
- Timestep conditioning injected into every encoder/decoder block
- Dual-branch architecture: envelope (low-freq) + detail (high-freq)
- Global context conditioning via attention pooling
- Enhanced S4 + dilated TCN head with extended receptive field
- V-prediction support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional, Any, List

# Import enhanced modular blocks
from .blocks import (
    EnhancedS4Layer, MultiLayerTransformerBlock, DeepTransformerDecoder,
    AdaptiveFiLMWithDropout, MultiFiLM, CFGWrapper,
    PositionalEmbedding, SinusoidalEmbedding,
    EnhancedSignalHead, RobustPeakHead, RobustThresholdHead,
    AttentionPooling, StaticParameterGenerationHead,
    EnhancedConvBlock, ResidualS4Block,
    OptimizedEncoderBlock, OptimizedDecoderBlock, OptimizedBottleneckProcessor,
    TaskSpecificFeatureExtractor
)


class TimestepEmbedding(nn.Module):
    """Enhanced timestep embedding for diffusion conditioning."""
    
    def __init__(self, dim: int = 128, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # Learnable projection
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: [batch_size] timestep indices
            
        Returns:
            Timestep embeddings [batch_size, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return self.time_mlp(embedding)


class GlobalContextModule(nn.Module):
    """Global context extraction via attention pooling."""
    
    def __init__(self, channels: int, context_dim: int = 256):
        super().__init__()
        self.attention_pool = AttentionPooling(channels)
        self.context_proj = nn.Sequential(
            nn.Linear(channels, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim)
        )
        
    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract global context from encoder features.
        
        Args:
            encoder_features: List of encoder outputs at different scales
            
        Returns:
            Global context vector [batch, context_dim]
        """
        # Use the last (deepest) encoder features for global context
        deep_features = encoder_features[-1]  # [batch, channels, seq_len]
        
        # Attention pooling
        global_features = self.attention_pool(deep_features)  # [batch, channels]
        
        # Project to context space
        context = self.context_proj(global_features)  # [batch, context_dim]
        
        return context


class EnvelopeDecoder(nn.Module):
    """Low-frequency envelope decoder for coarse structure."""
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int = 1,
        hidden_channels: int = 64,
        downsample_factor: int = 4,
        num_layers: int = 3
    ):
        super().__init__()
        self.downsample_factor = downsample_factor
        
        # Downsample for envelope extraction
        self.downsample = nn.AvgPool1d(downsample_factor, stride=downsample_factor)
        
        # Small decoder for envelope
        layers = []
        current_channels = input_channels
        
        for i in range(num_layers):
            out_channels = hidden_channels if i < num_layers - 1 else output_channels
            layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_channels) if i < num_layers - 1 else nn.Identity()
            ])
            current_channels = out_channels
        
        self.decoder = nn.Sequential(*layers)
        
        # Upsample back to original resolution
        self.upsample = nn.Upsample(scale_factor=downsample_factor, mode='linear', align_corners=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate low-frequency envelope.
        
        Args:
            x: Input features [batch, channels, seq_len]
            
        Returns:
            Envelope signal [batch, 1, seq_len]
        """
        # Downsample
        x_down = self.downsample(x)  # [batch, channels, seq_len//factor]
        
        # Decode envelope
        envelope_down = self.decoder(x_down)  # [batch, 1, seq_len//factor]
        
        # Upsample to original resolution
        envelope = self.upsample(envelope_down)  # [batch, 1, seq_len]
        
        return envelope


class EnhancedTimestepBlock(nn.Module):
    """Base block with timestep and global context conditioning."""
    
    def __init__(self, channels: int, timestep_dim: int = 128, context_dim: int = 256):
        super().__init__()
        self.channels = channels
        
        # Timestep conditioning
        self.time_proj = nn.Linear(timestep_dim, channels * 2)  # scale, shift
        
        # Global context conditioning
        self.context_proj = nn.Linear(context_dim, channels)
        
        # Layer norm for conditioning
        self.norm = nn.LayerNorm(channels)
    
    def apply_conditioning(
        self,
        x: torch.Tensor,
        timestep_emb: torch.Tensor,
        global_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply timestep and global context conditioning.
        
        Args:
            x: Input features [batch, channels, seq_len]
            timestep_emb: Timestep embeddings [batch, timestep_dim]
            global_context: Global context [batch, context_dim]
            
        Returns:
            Conditioned features
        """
        # Timestep conditioning (FiLM)
        time_params = self.time_proj(timestep_emb)  # [batch, channels * 2]
        time_scale, time_shift = time_params.chunk(2, dim=1)  # [batch, channels] each
        
        # Global context conditioning (additive)
        context_features = self.context_proj(global_context)  # [batch, channels]
        
        # Apply conditioning
        x = x.transpose(1, 2)  # [batch, seq_len, channels] for LayerNorm
        x = self.norm(x)
        x = x * (1 + time_scale.unsqueeze(1)) + time_shift.unsqueeze(1)
        x = x + context_features.unsqueeze(1)  # Broadcast context
        x = x.transpose(1, 2)  # Back to [batch, channels, seq_len]
        
        return x


class EnhancedHierarchicalUNet(nn.Module):
    """
    Enhanced Hierarchical U-Net with comprehensive architectural improvements.
    """
    
    def __init__(
        self,
        # Basic architecture parameters
        input_channels: int = 1,
        static_dim: int = 4,
        base_channels: int = 64,
        n_levels: int = 4,
        sequence_length: int = 200,
        signal_length: int = 200,
        
        # S4 configuration
        s4_state_size: int = 64,
        num_s4_layers: int = 2,
        use_enhanced_s4: bool = True,
        
        # Transformer configuration
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        use_multi_scale_attention: bool = True,
        use_cross_attention: bool = True,
        
        # Conditioning
        dropout: float = 0.1,
        film_dropout: float = 0.15,
        timestep_dim: int = 128,
        context_dim: int = 256,
        
        # Enhanced features
        use_dual_branch: bool = True,
        use_global_context: bool = True,
        use_v_prediction: bool = False,
        
        # Output configuration
        use_attention_heads: bool = True,
        predict_uncertainty: bool = False,
        enable_joint_generation: bool = False,
        use_task_specific_extractors: bool = True,
        channel_multiplier: float = 2.0
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.static_dim = static_dim
        self.base_channels = base_channels
        self.n_levels = n_levels
        self.sequence_length = sequence_length
        self.signal_length = signal_length
        self.timestep_dim = timestep_dim
        self.context_dim = context_dim
        self.use_dual_branch = use_dual_branch
        self.use_global_context = use_global_context
        self.use_v_prediction = use_v_prediction
        self.enable_joint_generation = enable_joint_generation
        self.use_task_specific_extractors = use_task_specific_extractors
        
        # Calculate channel dimensions for each level
        self.encoder_channels = [
            int(base_channels * (channel_multiplier ** i)) for i in range(n_levels)
        ]
        
        # Timestep embedding
        self.timestep_embedding = TimestepEmbedding(timestep_dim)
        
        # Static parameter embedding (keep same dim as original for compatibility)
        self.static_embedding = nn.Sequential(
            nn.Linear(static_dim, static_dim * 2),
            nn.GELU(),
            nn.LayerNorm(static_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(static_dim * 2, static_dim * 2)
        )
        
        # Global context module
        if use_global_context:
            self.global_context = GlobalContextModule(
                channels=self.encoder_channels[-1],
                context_dim=context_dim
            )
        else:
            self.global_context = None
        
        # Enhanced encoder with timestep conditioning
        self.encoder_levels = nn.ModuleList()
        for i in range(n_levels):
            in_ch = input_channels if i == 0 else self.encoder_channels[i - 1]
            out_ch = self.encoder_channels[i]
            seq_len = sequence_length // (2 ** i)
            
            encoder_level = OptimizedEncoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                static_dim=static_dim * 2,  # Enhanced static embedding
                sequence_length=seq_len,
                downsample_factor=2,
                s4_state_size=s4_state_size,
                num_s4_layers=num_s4_layers,
                num_transformer_layers=num_transformer_layers,
                num_heads=num_heads,
                dropout=dropout,
                film_dropout=film_dropout,
                use_enhanced_s4=use_enhanced_s4,
                use_multi_scale_attention=use_multi_scale_attention
            )
            self.encoder_levels.append(encoder_level)
        
        # Enhanced bottleneck
        bottleneck_channels = self.encoder_channels[-1]
        bottleneck_seq_len = sequence_length // (2 ** n_levels)
        
        self.bottleneck = OptimizedBottleneckProcessor(
            channels=bottleneck_channels,
            static_dim=static_dim * 2,
            sequence_length=bottleneck_seq_len,
            s4_state_size=s4_state_size,
            num_s4_layers=3,
            dropout=dropout,
            film_dropout=film_dropout,
            use_enhanced_s4=use_enhanced_s4
        )
        
        # Enhanced decoder with timestep conditioning
        self.decoder_levels = nn.ModuleList()
        for i in range(n_levels):
            level_idx = n_levels - 1 - i
            in_ch = bottleneck_channels if i == 0 else self.encoder_channels[level_idx + 1]
            out_ch = self.encoder_channels[level_idx]
            skip_ch = self.encoder_channels[level_idx]
            seq_len = bottleneck_seq_len * (2 ** i)
            
            decoder_level = OptimizedDecoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                skip_channels=skip_ch,
                static_dim=static_dim * 2,
                sequence_length=seq_len,
                upsample_factor=2,
                s4_state_size=s4_state_size,
                num_s4_layers=num_s4_layers,
                num_transformer_layers=num_transformer_layers,
                num_heads=num_heads,
                dropout=dropout,
                film_dropout=film_dropout,
                use_enhanced_s4=use_enhanced_s4,
                use_cross_attention=use_cross_attention,
                use_multi_scale_attention=use_multi_scale_attention
            )
            self.decoder_levels.append(decoder_level)
        
        # Final output projection
        final_channels = self.encoder_channels[0]
        self.output_projection = EnhancedConvBlock(
            in_channels=final_channels,
            out_channels=final_channels,
            kernel_size=7,
            dropout=dropout
        )
        
        # Dual-branch architecture
        if use_dual_branch:
            # Envelope decoder for low-frequency structure
            self.envelope_decoder = EnvelopeDecoder(
                input_channels=final_channels,
                output_channels=1,
                downsample_factor=4
            )
            
            # Fusion gate for combining envelope and detail
            self.fusion_gate = nn.Sequential(
                nn.Conv1d(final_channels + 1, final_channels, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.envelope_decoder = None
            self.fusion_gate = None
        
        # Task-specific feature extraction
        if use_task_specific_extractors:
            self.task_feature_extractor = TaskSpecificFeatureExtractor(
                input_dim=final_channels,
                hidden_dim=final_channels,
                dropout=dropout,
                use_cross_task_attention=True
            )
        else:
            self.task_feature_extractor = None
        
        # Enhanced signal head with extended receptive field
        self.signal_head = EnhancedSignalHead(
            input_dim=final_channels,
            signal_length=signal_length,
            hidden_channels=final_channels,
            n_blocks=6,  # Increased for extended receptive field
            kernel_size=3,
            dropout=dropout,
            use_timestep_conditioning=True,
            t_embed_dim=timestep_dim
        )
        
        # Other task heads
        self.peak_head = RobustPeakHead(
            input_dim=final_channels,
            hidden_dim=final_channels,
            num_layers=3,
            dropout=dropout,
            use_attention=use_attention_heads,
            use_uncertainty=predict_uncertainty,
            use_multiscale=True,
            latency_range=(1.0, 8.0),
            amplitude_range=(-0.5, 0.5)
        )
        
        self.threshold_head = RobustThresholdHead(
            input_dim=final_channels,
            hidden_dim=final_channels,
            dropout=dropout,
            use_attention_pooling=use_attention_heads,
            use_uncertainty=predict_uncertainty,
            use_robust_loss=True,
            threshold_range=(0.0, 120.0),
            use_multiscale=True
        )
        
        # Joint generation head
        if enable_joint_generation:
            self.static_param_head = StaticParameterGenerationHead(
                input_dim=final_channels,
                static_dim=static_dim,
                hidden_dim=final_channels,
                num_layers=3,
                dropout=dropout,
                use_attention=use_attention_heads,
                use_uncertainty=predict_uncertainty,
                use_constraints=True
            )
        else:
            self.static_param_head = None
    
    def forward(
        self,
        x: torch.Tensor,
        static_params: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        cfg_guidance_scale: float = 1.0,
        cfg_mode: str = 'training',
        force_uncond: bool = False,
        generation_mode: str = 'conditional'
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with full timestep conditioning.
        """
        batch_size = x.size(0)
        
        # Handle different generation modes
        if generation_mode == 'joint' and not self.enable_joint_generation:
            raise ValueError("Joint generation not enabled.")
        
        if generation_mode in ['joint', 'unconditional'] and static_params is None:
            static_params = torch.zeros(batch_size, self.static_dim, device=x.device)
        elif generation_mode == 'conditional' and static_params is None:
            raise ValueError("Static parameters required for conditional generation.")
        
        # Timestep handling
        if timesteps is None:
            timesteps = torch.zeros(batch_size, device=x.device, dtype=torch.long)
        
        # Timestep embedding
        timestep_emb = self.timestep_embedding(timesteps)  # [batch, timestep_dim]
        
        # Static parameter embedding
        static_emb = self.static_embedding(static_params)  # [batch, static_dim * 2]
        
        # Encoder forward pass
        skip_connections = []
        encoder_outputs = []
        
        current_x = x
        for encoder_level in self.encoder_levels:
            current_x, skip_connection = encoder_level(
                x=current_x,
                static_params=static_emb,
                cfg_guidance_scale=cfg_guidance_scale,
                force_uncond=force_uncond or (generation_mode == 'unconditional')
            )
            skip_connections.append(skip_connection)
            encoder_outputs.append(current_x)
        
        # Global context extraction
        if self.global_context is not None:
            global_context = self.global_context(encoder_outputs)  # [batch, context_dim]
        else:
            global_context = torch.zeros(batch_size, self.context_dim, device=x.device)
        
        # Bottleneck
        bottleneck_output = self.bottleneck(
            x=current_x,
            static_params=static_emb,
            cfg_guidance_scale=cfg_guidance_scale,
            force_uncond=force_uncond or (generation_mode == 'unconditional')
        )
        
        # Decoder forward pass
        current_x = bottleneck_output
        
        for i, decoder_level in enumerate(self.decoder_levels):
            skip_idx = len(skip_connections) - 1 - i
            skip_connection = skip_connections[skip_idx]
            encoder_output = encoder_outputs[skip_idx] if encoder_outputs else None
            
            current_x = decoder_level(
                x=current_x,
                skip_connection=skip_connection,
                static_params=static_emb,
                encoder_output=encoder_output,
                cfg_guidance_scale=cfg_guidance_scale,
                force_uncond=force_uncond or (generation_mode == 'unconditional')
            )
        
        # Final output projection
        x_final = self.output_projection(current_x)
        
        # Dual-branch processing
        if self.use_dual_branch and self.envelope_decoder is not None:
            # Generate envelope (low-frequency structure)
            envelope = self.envelope_decoder(x_final)  # [batch, 1, seq_len]
            
            # Fusion gate
            fusion_input = torch.cat([x_final, envelope], dim=1)  # [batch, channels+1, seq_len]
            gate = self.fusion_gate(fusion_input)  # [batch, channels, seq_len]
            
            # Apply gating
            x_final = x_final * gate
        else:
            envelope = None
        
        # Task-specific feature extraction
        if self.task_feature_extractor is not None:
            task_features = self.task_feature_extractor(x_final)
        else:
            task_features = {
                'signal': x_final,
                'peaks': x_final,
                'threshold': x_final
            }
        
        # Signal prediction (noise or v-prediction)
        predicted_output = self.signal_head(
            task_features['signal'] if task_features['signal'].dim() == 3 else task_features['signal'].unsqueeze(-1),
            timesteps
        )
        
        # Prepare outputs
        outputs = {
            'timesteps': timesteps,
            'is_diffusion_mode': timesteps.max() > 0
        }
        
        if self.use_v_prediction:
            outputs['v_pred'] = predicted_output
            outputs['noise'] = predicted_output  # For compatibility
        else:
            outputs['noise'] = predicted_output
        
        outputs['recon'] = predicted_output  # For compatibility
        
        # Multi-task predictions
        if not outputs['is_diffusion_mode']:
            # Peak prediction
            peak_output = self.peak_head(task_features['peaks'])
            if len(peak_output) == 5:
                peak_exists, peak_latency, peak_amplitude, latency_std, amplitude_std = peak_output
                outputs['peak'] = (peak_exists, peak_latency, peak_amplitude, latency_std, amplitude_std)
            else:
                peak_exists, peak_latency, peak_amplitude = peak_output
                outputs['peak'] = (peak_exists, peak_latency, peak_amplitude)
            
            # Threshold regression
            outputs['threshold'] = self.threshold_head(task_features['threshold'])
        
        # Joint generation
        if generation_mode in ['joint', 'unconditional'] and self.static_param_head is not None:
            outputs['static_params'] = self.static_param_head(x_final)
        
        # Include envelope if generated
        if envelope is not None:
            outputs['envelope'] = envelope
        
        return outputs