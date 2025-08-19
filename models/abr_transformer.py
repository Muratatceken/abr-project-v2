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

from .blocks.transformer_block import MultiLayerTransformerBlock
from .blocks.film import ConditionalEmbedding


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


class TokenFiLM(nn.Module):
    """
    FiLM conditioning for token embeddings [B, T, D].
    Produces per-feature (D) gamma/beta from static params [B, S] and
    broadcasts over T.
    """
    def __init__(
        self, 
        static_dim: int, 
        d_model: int, 
        hidden: int = 256, 
        dropout: float = 0.1,
        init_gamma: float = 0.0, 
        init_beta: float = 0.0
    ):
        super().__init__()
        self.static_dim = static_dim
        self.d_model = d_model
        
        if static_dim > 0:
            self.embed = nn.Sequential(
                nn.Linear(static_dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2 * d_model)  # gamma and beta
            )
        else:
            self.embed = None
            
        self.layer_norm = nn.LayerNorm(d_model)
        self.init_gamma = init_gamma
        self.init_beta = init_beta
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        if self.embed is not None:
            # Initialize the final layer specially
            with torch.no_grad():
                # Initialize gamma to init_gamma and beta to init_beta
                final_layer = self.embed[-1]
                nn.init.zeros_(final_layer.weight)
                if final_layer.bias is not None:
                    final_layer.bias[:self.d_model].fill_(self.init_gamma)  # gamma
                    final_layer.bias[self.d_model:].fill_(self.init_beta)   # beta

    def forward(self, x: torch.Tensor, static_params: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply FiLM conditioning to token embeddings.
        
        Args:
            x: Token embeddings [B, T, D]
            static_params: Static parameters [B, S] or None
            
        Returns:
            Modulated embeddings [B, T, D]
        """
        if static_params is None or self.embed is None:
            return x
        
        B, T, D = x.shape
        
        # Generate gamma and beta from static params
        gam_beta = self.embed(static_params)  # [B, 2D]
        gamma, beta = gam_beta.chunk(2, dim=-1)  # [B, D], [B, D]
        
        # Apply layer norm first
        x = self.layer_norm(x)
        
        # Apply FiLM: x * (1 + gamma) + beta
        x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        
        return x


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
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.static_dim = static_dim
        self.use_timestep_cond = use_timestep_cond
        self.use_static_film = use_static_film

        # Multi-scale stem: [B, 1, T] -> [B, D, T]
        self.stem = MultiScaleStem(d_model)

        # Token space & positions (using relative attention only)
        self.pos = nn.Identity()

        # Optional conditioning adapters
        if use_static_film and static_dim > 0:
            self.static_film_pre = TokenFiLM(static_dim, d_model, hidden=256, dropout=dropout)
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
            conv_kernel_size=7
        )

        # Optional post-FiLM
        if use_static_film and static_dim > 0:
            self.static_film_post = TokenFiLM(static_dim, d_model, hidden=256, dropout=dropout)
        else:
            self.static_film_post = None

        # Output: per-timestep linear projection -> 1 channel
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

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

    def forward(
        self,
        x: torch.Tensor,                     # [B, 1, T]
        static_params: Optional[torch.Tensor] = None,  # [B, S]
        timesteps: Optional[torch.Tensor] = None,
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
            Dictionary containing {"signal": [B, 1, T]} or tensor [B, 1, T]
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
        h = self.transformer(h)              # [B, T, D]

        if self.static_film_post is not None:
            h = self.static_film_post(h, static_params)

        # Project back to signal
        h = self.out_norm(h)
        y = self.out_proj(h)                 # [B, T, 1]
        y = y.transpose(1, 2).contiguous()   # [B, 1, T]

        if return_dict:
            return {"signal": y}
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
        **kwargs: Additional arguments
        
    Returns:
        ABRTransformerGenerator instance
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
        **kwargs
    )
