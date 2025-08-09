"""
Enhanced Hierarchical U-Net with S4 Encoder and Transformer Decoder for ABR Signal Generation

PROFESSIONAL IMPLEMENTATION - Version 2.0
- Enhanced S4 encoder with learnable state-space matrices
- Multi-layer Transformer decoder with advanced attention
- FiLM conditioning with dropout and classifier-free guidance
- Enhanced prediction heads with attention mechanisms
- Modular, scalable architecture inspired by SSSD-ECG

Key Features:
- Conv → ReLU → Conv → LayerNorm → S4 → FiLM pipeline in encoder
- Multi-layer Transformer with positional encoding in decoder
- Attention-based prediction heads
- CFG support for enhanced generation quality
- Professional weight initialization and architectural patterns

Author: AI Assistant
Based on: SSSD-ECG implementation patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional, Any, List

# Import enhanced modular blocks
from .blocks import (
    # Enhanced S4 and Transformer components  
    EnhancedS4Layer, MultiLayerTransformerBlock, DeepTransformerDecoder,
    
    # Enhanced conditioning and positional encoding
    AdaptiveFiLMWithDropout, MultiFiLM, CFGWrapper,
    PositionalEmbedding, SinusoidalEmbedding,
    
    # Enhanced prediction heads
    EnhancedSignalHead, EnhancedPeakHead, EnhancedClassificationHead, 
    EnhancedThresholdHead, AttentionPooling, StaticParameterGenerationHead,
    
    # Enhanced convolution and architectural blocks
    EnhancedConvBlock, ResidualS4Block,
    
    # OPTIMIZED encoder and decoder stacks
    OptimizedEncoderBlock, OptimizedDecoderBlock, OptimizedBottleneckProcessor,
    TaskSpecificFeatureExtractor,
    
    # Original blocks for backward compatibility
    EnhancedEncoderBlock, MultiScaleEncoderStack,
    EnhancedDecoderBlock, MultiScaleDecoderStack, BottleneckProcessor
)



class OptimizedHierarchicalUNet(nn.Module):
    """
    Optimized Hierarchical U-Net with fixed transformer placement and enhanced architecture.
    
    ARCHITECTURAL IMPROVEMENTS:
    - Encoder: Conv → Transformer (long sequences) → S4 → Downsample → FiLM
    - Bottleneck: S4-only (no transformer on short sequences)
    - Decoder: S4 → Upsample → Transformer (long sequences) → Skip Fusion → FiLM
    - Task-specific feature extractors for multi-task learning
    - Attention-based skip connections
    
    This addresses the critical architectural issues identified in the original model.
    """
    
    def __init__(
        self,
        # Basic architecture parameters (same as original)
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
        
        # Transformer configuration (optimized placement)
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        use_multi_scale_attention: bool = True,
        use_cross_attention: bool = True,
        
        # FiLM and conditioning
        dropout: float = 0.1,
        film_dropout: float = 0.15,
        use_cfg: bool = True,
        
        # Output configuration
        use_attention_heads: bool = True,
        predict_uncertainty: bool = True,
        
        # Joint generation parameters
        enable_joint_generation: bool = True,
        static_param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        
        # Optimization parameters
        use_task_specific_extractors: bool = True,
        use_attention_skip_connections: bool = True,
        channel_multiplier: float = 2.0
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.static_dim = static_dim
        self.base_channels = base_channels
        self.n_levels = n_levels
        self.sequence_length = sequence_length
        self.signal_length = signal_length
        self.use_cfg = use_cfg
        self.enable_joint_generation = enable_joint_generation
        self.use_task_specific_extractors = use_task_specific_extractors
        
        # Calculate channel dimensions for each level
        self.encoder_channels = [
            int(base_channels * (channel_multiplier ** i)) for i in range(n_levels)
        ]
        
        # ============== OPTIMIZED ENCODER STACK ==============
        # Uses transformers on LONG sequences before downsampling
        self.encoder_levels = nn.ModuleList()
        
        for i in range(n_levels):
            in_ch = input_channels if i == 0 else self.encoder_channels[i - 1]
            out_ch = self.encoder_channels[i]
            seq_len = sequence_length // (2 ** i)
            
            encoder_level = OptimizedEncoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                static_dim=static_dim,
                sequence_length=seq_len,
                downsample_factor=2,
                s4_state_size=s4_state_size,
                num_s4_layers=num_s4_layers,
                num_transformer_layers=num_transformer_layers,
                num_heads=num_heads,
                dropout=dropout,
                film_dropout=film_dropout,
                use_enhanced_s4=use_enhanced_s4,
                use_multi_scale_attention=use_multi_scale_attention,
                use_attention_skip=use_attention_skip_connections
            )
            self.encoder_levels.append(encoder_level)
        
        # ============== OPTIMIZED BOTTLENECK ==============
        # Uses S4-ONLY (no transformer on short sequences ~12 tokens)
        bottleneck_channels = self.encoder_channels[-1]
        bottleneck_seq_len = sequence_length // (2 ** n_levels)
        
        self.bottleneck = OptimizedBottleneckProcessor(
            channels=bottleneck_channels,
            static_dim=static_dim,
            sequence_length=bottleneck_seq_len,
            s4_state_size=s4_state_size,
            num_s4_layers=3,  # More S4 layers since no transformer
            dropout=dropout,
            film_dropout=film_dropout,
            use_enhanced_s4=use_enhanced_s4
        )
        
        # ============== OPTIMIZED DECODER STACK ==============
        # Uses S4 first, then transformers after upsampling (long sequences)
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
                static_dim=static_dim,
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
        
        # ============== TASK-SPECIFIC FEATURE EXTRACTION ==============
        final_channels = self.encoder_channels[0]
        
        if use_task_specific_extractors:
            self.task_feature_extractor = TaskSpecificFeatureExtractor(
                input_dim=final_channels,
                hidden_dim=final_channels,
                dropout=dropout,
                use_cross_task_attention=True
            )
        else:
            self.task_feature_extractor = None
            
        # Final output projection
        self.output_projection = EnhancedConvBlock(
            in_channels=final_channels,
            out_channels=final_channels,
            kernel_size=7,
            dropout=dropout
        )
        
        # ============== ENHANCED OUTPUT HEADS ==============
        
        # Signal reconstruction / noise prediction head (locality-preserving TCN)
        self.signal_head = EnhancedSignalHead(
            input_dim=final_channels,
            signal_length=signal_length,
            hidden_channels=final_channels,
            n_blocks=5,
            kernel_size=3,
            dropout=dropout,
            use_timestep_conditioning=True,
            t_embed_dim=128
        )
        
        # Peak prediction head (robust version)
        from .blocks.heads import RobustPeakHead
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
        
        # Classification head removed - signal generation only
        
        # Threshold regression head (robust version)
        from .blocks.heads import RobustThresholdHead
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
        
        # ============== STATIC PARAMETER GENERATION HEAD ==============
        if enable_joint_generation:
            self.static_param_head = StaticParameterGenerationHead(
                input_dim=final_channels,
                static_dim=static_dim,
                hidden_dim=final_channels,
                num_layers=3,
                dropout=dropout,
                use_attention=use_attention_heads,
                use_uncertainty=predict_uncertainty,
                use_constraints=True,
                parameter_ranges=static_param_ranges
            )
        else:
            self.static_param_head = None
        
        # ============== STATIC PARAMETER EMBEDDING ==============
        self.static_embedding = nn.Sequential(
            nn.Linear(static_dim, static_dim * 2),
            nn.GELU(),
            nn.LayerNorm(static_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(static_dim * 2, static_dim * 2)
        )
    
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
        Optimized forward pass through the hierarchical U-Net with diffusion support.
        
        Args:
            x: Input signal [batch, input_channels, sequence_length]
            static_params: Static parameters [batch, static_dim] (optional for joint generation)
            timesteps: Diffusion timesteps [batch] (optional, defaults to 0 for inference)
            cfg_guidance_scale: CFG guidance scale
            cfg_mode: CFG mode ('training', 'inference', 'unconditional')
            force_uncond: Force unconditional generation
            generation_mode: Generation mode ('conditional', 'joint', 'unconditional')
            
        Returns:
            Dictionary containing all model outputs
        """
        # Handle different generation modes
        if generation_mode == 'joint' and not self.enable_joint_generation:
            raise ValueError("Joint generation not enabled.")
        
        if generation_mode in ['joint', 'unconditional'] and static_params is None:
            batch_size = x.size(0)
            static_params = torch.zeros(batch_size, self.static_dim, device=x.device)
        elif generation_mode == 'conditional' and static_params is None:
            raise ValueError("Static parameters required for conditional generation.")
        
        batch_size = x.size(0)
        
        # ============== TIMESTEP HANDLING FOR DIFFUSION ==============
        if timesteps is None:
            # Default to t=0 for inference (clean signal)
            timesteps = torch.zeros(batch_size, device=x.device, dtype=torch.long)
        
        # Determine if we're in diffusion mode (t > 0) or inference mode (t = 0)
        is_diffusion_mode = timesteps.max() > 0
        is_inference_mode = timesteps.max() == 0
        
        # ============== STATIC PARAMETER EMBEDDING ==============
        static_emb = self.static_embedding(static_params)
        
        # ============== OPTIMIZED ENCODER FORWARD ==============
        # Uses transformers on LONG sequences before downsampling
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
        
        # ============== OPTIMIZED BOTTLENECK ==============
        # Uses S4-ONLY (no transformer on short sequences)
        bottleneck_output = self.bottleneck(
            x=current_x,
            static_params=static_emb,
            cfg_guidance_scale=cfg_guidance_scale,
            force_uncond=force_uncond or (generation_mode == 'unconditional')
        )
        
        # ============== OPTIMIZED DECODER FORWARD ==============
        # Uses S4 first, then transformers after upsampling
        current_x = bottleneck_output
        
        for i, decoder_level in enumerate(self.decoder_levels):
            skip_idx = len(skip_connections) - 1 - i
            skip_connection = skip_connections[skip_idx]
            
            # Use encoder output for cross-attention if available
            encoder_output = encoder_outputs[skip_idx] if encoder_outputs else None
            
            current_x = decoder_level(
                x=current_x,
                skip_connection=skip_connection,
                static_params=static_emb,
                encoder_output=encoder_output,
                cfg_guidance_scale=cfg_guidance_scale,
                force_uncond=force_uncond or (generation_mode == 'unconditional')
            )
        
        # ============== FINAL OUTPUT PROJECTION ==============
        x_final = self.output_projection(current_x)
        
        # ============== TASK-SPECIFIC FEATURE EXTRACTION ==============
        if self.task_feature_extractor is not None:
            task_features = self.task_feature_extractor(x_final)
        else:
            # Use same features for all tasks
            task_features = {
                'signal': x_final,
                'peaks': x_final,
                'classification': x_final,
                'threshold': x_final
            }
        
        # ============== ENHANCED OUTPUT HEADS WITH DIFFUSION SUPPORT ==============
        
        # ============== SIGNAL NOISE PREDICTION (DIFFUSION PARADIGM) ==============
        # ALWAYS predict noise for proper diffusion sampling
        # The signal head should predict noise at ALL timesteps for consistent generation
        predicted_noise = self.signal_head(
            task_features['signal'] if task_features['signal'].dim() == 3 else task_features['signal'].unsqueeze(-1),
            timesteps
        )
        
        # For compatibility, set signal_recon = predicted_noise
        # The sampling process will handle the noise-to-signal conversion
        signal_recon = predicted_noise
        
        # ============== MULTI-TASK PREDICTIONS (ALWAYS FROM CLEAN FEATURES) ==============
        # These predictions should be consistent regardless of diffusion mode
        # Extract clean features for multi-task learning
        if is_diffusion_mode:
            # For diffusion mode, we need to extract clean features
            # Use the final features as they contain semantic information
            clean_features = {
                'peaks': x_final,
                'threshold': x_final
            }
        else:
            # For inference mode, features are already clean
            clean_features = {
                'peaks': task_features['peaks'],
                'threshold': task_features['threshold']
            }
        
        # Peak prediction (with proper handling of outputs)
        peak_output = self.peak_head(clean_features['peaks'])
        if len(peak_output) == 5:  # With uncertainty
            peak_exists, peak_latency, peak_amplitude, latency_std, amplitude_std = peak_output
            peak_result = (peak_exists, peak_latency, peak_amplitude, latency_std, amplitude_std)
        else:  # Without uncertainty
            peak_exists, peak_latency, peak_amplitude = peak_output
            peak_result = (peak_exists, peak_latency, peak_amplitude)
        
        # Classification removed - signal generation only
        
        # Threshold regression
        threshold = self.threshold_head(clean_features['threshold'])
        
        # ============== PREPARE OUTPUT DICTIONARY ==============
        outputs = {
            'recon': signal_recon,
            'peak': peak_result,
            'threshold': threshold
        }
        
        # Always include noise output for consistent diffusion sampling
        outputs['noise'] = predicted_noise
        
        # Add diffusion mode information for loss computation
        outputs['is_diffusion_mode'] = is_diffusion_mode
        outputs['timesteps'] = timesteps
        
        # ============== JOINT GENERATION: STATIC PARAMETER GENERATION ==============
        if generation_mode in ['joint', 'unconditional'] and self.static_param_head is not None:
            generated_static_params = self.static_param_head(x_final)
            outputs['static_params'] = generated_static_params
        
        return outputs
    
    def generate_joint(
        self, 
        batch_size: int, 
        device: torch.device,
        sequence_length: Optional[int] = None,
        temperature: float = 1.0,
        use_constraints: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Generate joint samples with the optimized architecture."""
        if not self.enable_joint_generation:
            raise ValueError("Joint generation not enabled.")
        
        seq_len = sequence_length or self.sequence_length
        x = torch.randn(batch_size, self.input_channels, seq_len, device=device)
        
        with torch.no_grad():
            outputs = self.forward(
                x=x,
                static_params=None,
                generation_mode='joint'
            )
            
        # The static parameters are already generated in the forward pass
        # No need for additional sampling in joint generation mode
            
            return outputs
    
    def generate_conditional(
        self,
        static_params: torch.Tensor,
        sequence_length: Optional[int] = None,
        noise_level: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Generate ABR signals conditioned on given static parameters."""
        batch_size = static_params.size(0)
        device = static_params.device
        seq_len = sequence_length or self.sequence_length
        
        x = torch.randn(batch_size, self.input_channels, seq_len, device=device) * noise_level
        
        with torch.no_grad():
            return self.forward(
                x=x,
                static_params=static_params,
                generation_mode='conditional'
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        features = [
            'Optimized transformer placement (long sequences only)',
            'S4-only bottleneck (no transformer on short sequences)',
            'Multi-scale attention for peak detection',
            'Task-specific feature extractors',
            'Attention-based skip connections',
            'Robust prediction heads with uncertainty',
            'Cross-attention encoder-decoder interaction'
        ]
        
        if self.enable_joint_generation:
            features.append('Joint generation of signals and static parameters')
        
        return {
            'model_name': 'OptimizedHierarchicalUNet',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': {
                'encoder_flow': 'Conv → Transformer (long) → S4 → Downsample → FiLM',
                'bottleneck_flow': 'S4-only (no transformer on short sequences)',
                'decoder_flow': 'S4 → Upsample → Transformer (long) → Skip Fusion → FiLM',
                'encoder_levels': self.n_levels,
                'encoder_channels': self.encoder_channels,
                'sequence_length': self.sequence_length,
                'signal_length': self.signal_length,
                'num_classes': self.num_classes,
                'joint_generation': self.enable_joint_generation,
                'task_specific_extractors': self.use_task_specific_extractors
            },
            'features': features,
            'architectural_improvements': [
                'Fixed transformer placement issues',
                'Proper sequence length consideration',
                'Enhanced multi-task learning',
                'Attention-based feature fusion',
                'Robust loss functions'
            ]
        } 