"""
ABR Transformer Generator for T=200 ABR Signal Generation

A flat Transformer-based approach replacing the hierarchical U-Net+S4 design.
This implementation focuses on simplicity and effectiveness for short ABR windows.

Features:
- Single-path Transformer architecture (no U-Net hierarchy)
- FiLM conditioning for static parameters
- Timestep conditioning for diffusion training
- Single signal generation head
- T=200 sequence length optimized
"""

import math
from typing import Optional, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.transformer_block import MultiLayerTransformerBlock, MultiHeadAttention, CrossAttentionTransformerBlock
from .blocks.film import ConditionalEmbedding, TokenFiLM
from .blocks.heads import AttentionPooling
from .blocks.positional import LearnedPositionalEmbedding
from .blocks.heads import MultiScaleFeatureExtractor


class MultiScaleStem(nn.Module):
    """
    Multi-branch Conv1D stem to preserve sharp transients (k=3),
    peak shapes (k=7), and slow trends (k=15). Fused to d_model.
    """
    def __init__(self, d_model: int):
        super().__init__()
        c = max(1, d_model // 3)
        # Three branches; if d_model not divisible by 3, the 1x1 fuse maps back to d_model.
        self.b3  = nn.Sequential(nn.Conv1d(1, c,  kernel_size=3,  padding=1),
                                 nn.GroupNorm(1, c), nn.GELU())
        self.b7  = nn.Sequential(nn.Conv1d(1, c,  kernel_size=7,  padding=3),
                                 nn.GroupNorm(1, c), nn.GELU())
        self.b15 = nn.Sequential(nn.Conv1d(1, c,  kernel_size=15, padding=7),
                                 nn.GroupNorm(1, c), nn.GELU())
        self.fuse = nn.Conv1d(3 * c, d_model, kernel_size=1)

    def forward(self, x):  # x: [B, 1, T]
        h = torch.cat([self.b3(x), self.b7(x), self.b15(x)], dim=1)
        return self.fuse(h)





class TimestepAdapter(nn.Module):
    """
    Additive timestep embedding for diffusion, broadcast across tokens.
    If timesteps is None, it's a no-op.
    """
    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden), 
            nn.SiLU(),
            nn.Linear(hidden, d_model)
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        device = t.device
        half = dim // 2
        t = t.float().unsqueeze(1)  # [B, 1]
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device).float() / float(half)
        )  # [half]
        args = t * freqs  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, dim or dim-1]
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

    def forward(self, x: torch.Tensor, timesteps: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Add timestep conditioning to token embeddings.
        
        Args:
            x: Token embeddings [B, T, D]
            timesteps: Timestep indices [B] or None
            
        Returns:
            Conditioned embeddings [B, T, D]
        """
        if timesteps is None:
            return x
        
        B, T, D = x.shape
        t_emb = self.sinusoidal_embedding(timesteps, D)  # [B, D]
        t_emb = self.proj(t_emb)                         # [B, D]
        return x + t_emb.unsqueeze(1)                    # broadcast over T


class ABRTransformerGenerator(nn.Module):
    """
    Flat Transformer-based ABR generator for short windows (e.g., T=200).
    No U-Net, no S4, single output head for signal generation.

    Forward:
        x: [B, 1, T]
        static_params: [B, S] or None
        timesteps: [B] or None
        returns {"signal": [B, 1, T]} (or [B, 1, T] if return_dict=False)
    """

    def __init__(
        self,
        input_channels: int = 1,
        static_dim: int = 0,
        sequence_length: int = 200,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.10,
        use_timestep_cond: bool = True,
        use_static_film: bool = True,
        use_cross_attention: bool = False,
        use_learned_pos_emb: bool = False,
        film_residual: bool = False,
        use_multi_scale_fusion: bool = False,
        joint_static_generation: bool = False,
        ablation_mode: str = 'none',
        use_advanced_blocks: bool = False,
        use_multi_scale_attention: bool = False,
        use_gated_ffn: bool = False,
        attention_dropout: float = 0.1,
        cross_attention_heads: int = 4,
        cross_attention_dropout: float = 0.1,
        num_static_tokens: int = 1,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.static_dim = static_dim
        self.use_timestep_cond = use_timestep_cond
        self.use_static_film = use_static_film
        self.ablation_mode = ablation_mode
        
        # Apply ablation configuration to compute effective flags
        effective_flags = self._compute_effective_flags(
            use_cross_attention, use_learned_pos_emb, film_residual,
            use_multi_scale_fusion, joint_static_generation,
            use_advanced_blocks, use_multi_scale_attention, use_gated_ffn
        )
        
        # Store effective flags
        self.use_cross_attention = effective_flags['use_cross_attention']
        self.use_learned_pos_emb = effective_flags['use_learned_pos_emb']
        self.film_residual = effective_flags['film_residual']
        self.use_multi_scale_fusion = effective_flags['use_multi_scale_fusion']
        self.joint_static_generation = effective_flags['joint_static_generation']
        self.use_advanced_blocks = effective_flags['use_advanced_blocks']
        self.use_multi_scale_attention = effective_flags['use_multi_scale_attention']
        self.use_gated_ffn = effective_flags['use_gated_ffn']
        self.attention_dropout = attention_dropout
        self.cross_attention_heads = cross_attention_heads
        self.cross_attention_dropout = cross_attention_dropout
        self.num_static_tokens = num_static_tokens

        # Multi-scale stem: [B, 1, T] -> [B, D, T]
        self.stem = MultiScaleStem(d_model)

        # Token space & positions (conditional positional encoding)
        if self.use_learned_pos_emb:
            # Use max(sequence_length, 2048) to prevent index overflow
            pos_emb_max_len = max(sequence_length, 2048)
            self.pos = LearnedPositionalEmbedding(d_model, max_len=pos_emb_max_len, dropout=dropout)
        else:
            self.pos = nn.Identity()

        # Optional conditioning adapters
        if use_static_film and static_dim > 0:
            self.static_film_pre = TokenFiLM(static_dim, d_model, hidden=256, dropout=dropout, use_residual=self.film_residual)
        else:
            self.static_film_pre = None
            
        if use_timestep_cond:
            self.t_adapter = TimestepAdapter(d_model)
        else:
            self.t_adapter = None

        # Core Transformer
        self.transformer = MultiLayerTransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * ff_mult,
            num_layers=n_layers,
            dropout=dropout,
            activation='gelu',
            pre_norm=True,
            use_relative_position=True,
            use_conv_module=True,
            conv_kernel_size=7,
            use_advanced_blocks=self.use_advanced_blocks,
            use_multi_scale_attention=self.use_multi_scale_attention,
            use_gated_ffn=self.use_gated_ffn,
            attention_dropout=self.attention_dropout
        )

        # Optional post-FiLM
        if use_static_film and static_dim > 0:
            self.static_film_post = TokenFiLM(static_dim, d_model, hidden=256, dropout=dropout, use_residual=self.film_residual)
        else:
            self.static_film_post = None

        # Cross-attention mechanism between static params and signal features
        if self.use_cross_attention and static_dim > 0:
            self.static_encoder = nn.Linear(static_dim, d_model)
            self.cross_attention = CrossAttentionTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                static_attention_heads=self.cross_attention_heads,
                static_attention_dropout=self.cross_attention_dropout
            )
            # Learnable static tokens for cross-attention: provide additional context
            # beyond encoded static parameters. These tokens can learn to represent
            # global patterns or additional conditioning information.
            self.static_tokens = nn.Parameter(torch.randn(1, self.num_static_tokens, d_model))
            # Add normalization and dropout for cross-attention stability
            self.cross_attn_norm = nn.LayerNorm(d_model)
            self.cross_attn_dropout = nn.Dropout(dropout)
        else:
            self.static_encoder = None
            self.cross_attention = None
            self.static_tokens = None
            self.cross_attn_norm = None
            self.cross_attn_dropout = None

        # Output: per-timestep linear projection -> 1 channel
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

        # Peak classification components
        self.attn_pool = AttentionPooling(d_model, attention_dim=max(1, d_model // 2), dropout=dropout)
        self.peak5_head = nn.Linear(d_model, 1)

        # Multi-scale feature fusion
        if self.use_multi_scale_fusion:
            self.multi_scale_fusion = MultiScaleFeatureExtractor(d_model)
        else:
            self.multi_scale_fusion = None

        # Joint static parameter generation
        if self.joint_static_generation and static_dim > 0:
            self.static_recon_head = nn.Linear(d_model, static_dim)
        else:
            self.static_recon_head = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Use 'relu' as nonlinearity since 'gelu' is not supported by kaiming_normal_
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_effective_flags(self, use_cross_attention, use_learned_pos_emb, film_residual, use_multi_scale_fusion, joint_static_generation, use_advanced_blocks, use_multi_scale_attention, use_gated_ffn):
        """Compute effective flags based on ablation mode."""
        if self.ablation_mode == 'minimal':
            return {
                'use_cross_attention': False,
                'use_learned_pos_emb': False,
                'film_residual': False,
                'use_multi_scale_fusion': False,
                'joint_static_generation': False,
                'use_advanced_blocks': False,
                'use_multi_scale_attention': False,
                'use_gated_ffn': False
            }
        elif self.ablation_mode == 'cross_attn_only':
            return {
                'use_cross_attention': True,
                'use_learned_pos_emb': False,
                'film_residual': False,
                'use_multi_scale_fusion': False,
                'joint_static_generation': False,
                'use_advanced_blocks': False,
                'use_multi_scale_attention': False,
                'use_gated_ffn': False
            }
        elif self.ablation_mode == 'film_residual_only':
            return {
                'use_cross_attention': False,
                'use_learned_pos_emb': False,
                'film_residual': True,
                'use_multi_scale_fusion': False,
                'joint_static_generation': False,
                'use_advanced_blocks': False,
                'use_multi_scale_attention': False,
                'use_gated_ffn': False
            }
        elif self.ablation_mode == 'pos_emb_only':
            return {
                'use_cross_attention': False,
                'use_learned_pos_emb': True,
                'film_residual': False,
                'use_multi_scale_fusion': False,
                'joint_static_generation': False,
                'use_advanced_blocks': False,
                'use_multi_scale_attention': False,
                'use_gated_ffn': False
            }
        elif self.ablation_mode == 'multi_scale_only':
            return {
                'use_cross_attention': False,
                'use_learned_pos_emb': False,
                'film_residual': False,
                'use_multi_scale_fusion': True,
                'joint_static_generation': False,
                'use_advanced_blocks': False,
                'use_multi_scale_attention': False,
                'use_gated_ffn': False
            }
        elif self.ablation_mode == 'joint_gen_only':
            return {
                'use_cross_attention': False,
                'use_learned_pos_emb': False,
                'film_residual': False,
                'use_multi_scale_fusion': False,
                'joint_static_generation': True,
                'use_advanced_blocks': False,
                'use_multi_scale_attention': False,
                'use_gated_ffn': False
            }
        elif self.ablation_mode == 'full':
            return {
                'use_cross_attention': True,
                'use_learned_pos_emb': True,
                'film_residual': True,
                'use_multi_scale_fusion': True,
                'joint_static_generation': True,
                'use_advanced_blocks': True,
                'use_multi_scale_attention': True,
                'use_gated_ffn': True
            }
        else:
            # Use provided arguments when ablation_mode is 'none'
            return {
                'use_cross_attention': use_cross_attention,
                'use_learned_pos_emb': use_learned_pos_emb,
                'film_residual': film_residual,
                'use_multi_scale_fusion': use_multi_scale_fusion,
                'joint_static_generation': joint_static_generation,
                'use_advanced_blocks': use_advanced_blocks,
                'use_multi_scale_attention': use_multi_scale_attention,
                'use_gated_ffn': use_gated_ffn
            }

    def _apply_ablation_config(self):
        """Apply ablation study configuration to modify architecture.
        
        Note: This method is now a no-op since ablation configuration
        is handled during __init__ via _compute_effective_flags.
        Kept for backward compatibility.
        """
        pass

    def forward(
        self,
        x: torch.Tensor,                     # [B, 1, T]
        static_params: Optional[torch.Tensor] = None,  # [B, S]
        timesteps: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through the ABR Transformer generator.
        
        Args:
            x: Input signal [B, 1, T]
            static_params: Static parameters [B, S] (optional)
            timesteps: Diffusion timesteps [B] (optional)
            return_dict: Whether to return dict or tensor
            
        Returns:
            Dictionary containing:
            - "signal": [B, 1, T] - Generated ABR signal
            - "peak_5th_exists": [B] - Binary classification logits for 5th peak detection
            - "static_recon": [B, S] - Static parameter reconstruction (when joint_static_generation=True)
            
            Or tensor [B, 1, T] if return_dict=False
        """
        assert x.dim() == 3 and x.size(1) == 1, f"x must be [B, 1, T], got {x.shape}"
        B, _, T = x.shape
        
        # Enforce fixed length at input
        assert T == self.sequence_length, (
            f"Expected input length {self.sequence_length}, got {T}. "
            "Pad/crop or resample outside the model with a high-quality method."
        )

        # Stem to token space
        h = self.stem(x)                     # [B, D, T]
        h = h.transpose(1, 2)                # [B, T, D]
        h = self.pos(h)                      # add positional encodings

        # Conditioning
        if self.static_film_pre is not None:
            h = self.static_film_pre(h, static_params)
        if self.t_adapter is not None:
            h = self.t_adapter(h, timesteps)

        # Transformer stack
        h = self.transformer(h, mask=attn_mask)              # [B, T, D]

        if self.static_film_post is not None:
            h = self.static_film_post(h, static_params)

        # Cross-attention between static params and signal features
        if self.use_cross_attention and self.cross_attention is not None and static_params is not None:
            static_encoded = self.static_encoder(static_params).unsqueeze(1)  # [B, 1, D]
            # Concatenate static tokens to K/V for richer cross-attention
            # static_encoded: [B, 1, D], static_tokens: [B, num_static_tokens, D]
            kv = torch.cat([static_encoded, self.static_tokens.expand(B, -1, -1)], dim=1)  # [B, 1+num_static_tokens, D]
            # Apply normalization and dropout for stability
            h_norm = self.cross_attn_norm(h)
            # Use CrossAttentionTransformerBlock interface: (decoder_input, encoder_output)
            cross_attn_output, _, _ = self.cross_attention(h_norm, kv)
            h = h + self.cross_attn_dropout(cross_attn_output)

        # Multi-scale feature fusion
        if self.use_multi_scale_fusion and self.multi_scale_fusion is not None:
            h = self.multi_scale_fusion(h.transpose(1, 2)).transpose(1, 2)

        # Project back to signal
        h_norm = self.out_norm(h)
        y = self.out_proj(h_norm)                 # [B, T, 1]
        y = y.transpose(1, 2).contiguous()   # [B, 1, T]

        if return_dict:
            # Compute peak classification only when needed
            pooled_features = self.attn_pool(h_norm, mask=None)  # Keep API compatible for now
            peak_logits = self.peak5_head(pooled_features).squeeze(-1)
            
            # Joint static parameter generation
            if self.joint_static_generation and self.static_recon_head is not None:
                static_recon = self.static_recon_head(pooled_features)
                return {"signal": y, "peak_5th_exists": peak_logits, "static_recon": static_recon}
            else:
                return {"signal": y, "peak_5th_exists": peak_logits}
        return y


# For testing and validation
def create_abr_transformer(
    static_dim: int = 4,
    sequence_length: int = 200,
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 8,
    ff_mult: int = 4,
    dropout: float = 0.1,
    use_advanced_blocks: bool = False,
    use_multi_scale_attention: bool = False,
    use_gated_ffn: bool = False,
    attention_dropout: float = 0.1,
    cross_attention_heads: int = 4,
    cross_attention_dropout: float = 0.1,
    num_static_tokens: int = 1,
    **kwargs
) -> ABRTransformerGenerator:
    """
    Factory function to create ABR Transformer with sensible defaults.
    
    Args:
        static_dim: Dimension of static parameters (age, intensity, etc.)
        sequence_length: ABR signal length (default 200)
        d_model: Transformer model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        ff_mult: Feed-forward dimension multiplier
        dropout: Dropout rate
        **kwargs: Additional arguments for advanced features
        
    Returns:
        ABRTransformerGenerator instance with enhanced architecture:
        - Signal generation head for ABR signal reconstruction
        - Peak classification head for 5th peak detection
        - Optional cross-attention mechanism for static parameter integration
        - Optional joint static parameter generation
        - Optional learnable positional embeddings
        - Optional residual FiLM connections
        - Optional multi-scale feature fusion
        - Comprehensive ablation study support
    """
    return ABRTransformerGenerator(
        input_channels=1,
        static_dim=static_dim,
        sequence_length=sequence_length,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_mult=ff_mult,
        dropout=dropout,
        use_timestep_cond=True,
        use_static_film=True,
        use_advanced_blocks=use_advanced_blocks,
        use_multi_scale_attention=use_multi_scale_attention,
        use_gated_ffn=use_gated_ffn,
        attention_dropout=attention_dropout,
        cross_attention_heads=cross_attention_heads,
        cross_attention_dropout=cross_attention_dropout,
        num_static_tokens=num_static_tokens,
        **kwargs
    )
