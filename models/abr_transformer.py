"""
ABR Transformer Generator V2 — Intensity-Conditioned Diffusion

Simplified, focused architecture for synthetic ABR signal generation.
Key design choices:
- Intensity gets a dedicated sinusoidal embedding pathway (primary ABR parameter)
- Class-conditional generation for 5 hearing loss types
- Lightweight backbone (~2.8M params) appropriate for T=200 signals
- Dual classifier-free guidance (static + class)
- V-prediction diffusion parameterization

Forward:
    x: [B, 1, T]
    intensity: [B] or None          # 0–1 normalized intensity
    aux_static: [B, 3] or None      # [Age, StimRate, FMP]
    class_label: [B] (long) or None # hearing loss class 0–4
    timesteps: [B] or None          # diffusion timestep index
    returns {"signal": [B, 1, T]}
"""

import math
from typing import Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.transformer_block import MultiLayerTransformerBlock
from .blocks.film import TokenFiLM


# ═══════════════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════════════

class MultiScaleStem(nn.Module):
    """Multi-branch Conv1D stem: sharp transients (k=3), peaks (k=7), trends (k=15)."""

    def __init__(self, d_model: int):
        super().__init__()
        c = max(1, d_model // 3)
        self.b3  = nn.Sequential(nn.Conv1d(1, c, kernel_size=3,  padding=1),
                                 nn.GroupNorm(1, c), nn.GELU())
        self.b7  = nn.Sequential(nn.Conv1d(1, c, kernel_size=7,  padding=3),
                                 nn.GroupNorm(1, c), nn.GELU())
        self.b15 = nn.Sequential(nn.Conv1d(1, c, kernel_size=15, padding=7),
                                 nn.GroupNorm(1, c), nn.GELU())
        self.fuse = nn.Conv1d(3 * c, d_model, kernel_size=1)

    def forward(self, x):  # x: [B, 1, T]
        return self.fuse(torch.cat([self.b3(x), self.b7(x), self.b15(x)], dim=1))


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for scalar values [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=values.device).float() / float(half)
    )
    args = values.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepAdapter(nn.Module):
    """Additive timestep embedding for diffusion, broadcast across tokens."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor, timesteps: Optional[torch.Tensor]) -> torch.Tensor:
        if timesteps is None:
            return x
        t_emb = sinusoidal_embedding(timesteps, self.d_model)
        return x + self.proj(t_emb).unsqueeze(1)


class IntensityEmbedding(nn.Module):
    """Dedicated sinusoidal + MLP embedding for stimulus intensity, with learned unconditional."""

    def __init__(self, d_model: int, emb_dim: int = 64):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Learned unconditional embedding for CFG
        self.uncond_emb = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.uncond_emb, std=0.02)

    def forward(self, intensity: Optional[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        """intensity: [B] scalar in [0, 1] or None -> [B, d_model]"""
        if intensity is None:
            return self.uncond_emb.unsqueeze(0).expand(batch_size, -1)
        emb = sinusoidal_embedding(intensity * 1000.0, self.emb_dim)
        return self.mlp(emb)


class AuxStaticEmbedding(nn.Module):
    """MLP embedding for auxiliary static parameters, with learned unconditional."""

    def __init__(self, input_dim: int, d_model: int, hidden: int = 64):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, d_model),
        )
        # Learned unconditional embedding for CFG
        self.uncond_emb = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.uncond_emb, std=0.02)

    def forward(self, aux: Optional[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        """aux: [B, input_dim] or None -> [B, d_model]"""
        if aux is None:
            return self.uncond_emb.unsqueeze(0).expand(batch_size, -1)
        return self.mlp(aux)


class ClassEmbedding(nn.Module):
    """Learnable embedding for hearing loss class labels with unconditional support."""

    def __init__(self, num_classes: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(num_classes, d_model)
        self.uncond_emb = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.uncond_emb, std=0.02)

    def forward(self, class_label: Optional[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        """class_label: [B] long or None -> [B, d_model]"""
        if class_label is None:
            return self.uncond_emb.unsqueeze(0).expand(batch_size, -1)
        return self.emb(class_label)


class ConditioningFiLM(nn.Module):
    """
    FiLM conditioning that accepts pre-embedded conditioning vectors.
    Applies: x_norm * (1 + gamma) + beta, where gamma/beta are projected from conditioning.
    """

    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.08):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * d_model),
        )
        # Unconditional embedding for CFG
        self.uncond_emb = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.uncond_emb, std=0.02)
        # Small non-zero init for gradient flow
        nn.init.normal_(self.proj[-1].weight, std=1e-3)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: [B, T, D], cond: [B, D] or None -> [B, T, D]
        """
        if cond is None:
            cond = self.uncond_emb.unsqueeze(0).expand(x.shape[0], -1)
        gam_beta = self.proj(cond)            # [B, 2*D]
        gamma, beta = gam_beta.chunk(2, dim=-1)
        x_norm = self.norm(x)
        return x_norm * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main model
# ═══════════════════════════════════════════════════════════════════════════════

class ABRTransformerGenerator(nn.Module):
    """
    Intensity-conditioned ABR Transformer generator for v-prediction diffusion.

    ~2.8M params. Designed for T=200 ABR signals.
    """

    def __init__(
        self,
        sequence_length: int = 200,
        d_model: int = 192,
        n_layers: int = 4,
        n_heads: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.08,
        num_classes: int = 5,
        intensity_emb_dim: int = 64,
        aux_static_dim: int = 3,
        # Legacy compat — ignored but accepted so old configs don't crash
        input_channels: int = 1,
        static_dim: int = 4,
        use_timestep_cond: bool = True,
        use_static_film: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_classes = num_classes
        # Legacy attributes for pipeline compat
        self.static_dim = static_dim

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem = MultiScaleStem(d_model)

        # ── Conditioning pathways ─────────────────────────────────────────
        self.intensity_emb = IntensityEmbedding(d_model, emb_dim=intensity_emb_dim)
        self.aux_static_emb = AuxStaticEmbedding(aux_static_dim, d_model)
        self.class_emb = ClassEmbedding(num_classes, d_model)
        self.aux_scale = nn.Parameter(torch.tensor(0.5))  # learnable aux blending

        # ── Timestep adapter ──────────────────────────────────────────────
        self.t_adapter = TimestepAdapter(d_model)

        # ── FiLM layers ──────────────────────────────────────────────────
        self.film_pre = ConditioningFiLM(d_model, hidden=256, dropout=dropout)
        self.film_post = ConditioningFiLM(d_model, hidden=256, dropout=dropout)

        # ── Transformer backbone ─────────────────────────────────────────
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
        )

        # ── Output projection ────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Conditioning helpers ──────────────────────────────────────────────

    def _embed_conditioning(
        self,
        intensity: Optional[torch.Tensor],
        aux_static: Optional[torch.Tensor],
        class_label: Optional[torch.Tensor],
        B: int,
    ):
        """Compute conditioning embeddings using learned unconditional where needed.

        Returns:
            pre_cond: [B, D] or None for pre-transformer FiLM (intensity + aux)
            post_cond: [B, D] or None for post-transformer FiLM (intensity + class)
            class_emb: [B, D] for additive class injection
        """
        # Each embedding module handles None internally with learned uncond
        i_emb = self.intensity_emb(intensity, batch_size=B)   # [B, D]
        a_emb = self.aux_static_emb(aux_static, batch_size=B) # [B, D]
        c_emb = self.class_emb(class_label, batch_size=B)     # [B, D]

        # Check if all conditioning is dropped (full unconditional)
        all_dropped = (intensity is None and aux_static is None and class_label is None)

        # Pre-FiLM: intensity + scaled aux
        pre_cond = None if all_dropped else (i_emb + self.aux_scale * a_emb)
        # Post-FiLM: intensity + class
        post_cond = None if all_dropped else (i_emb + c_emb)

        return pre_cond, post_cond, c_emb

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        intensity: Optional[torch.Tensor] = None,
        aux_static: Optional[torch.Tensor] = None,
        class_label: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        # Legacy kwargs (ignored)
        static_params: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Noisy signal [B, 1, T]
            intensity: Normalized intensity [B] in [0, 1] or None
            aux_static: Auxiliary params [B, 3] (Age, StimRate, FMP) or None
            class_label: Hearing loss class [B] (long, 0–4) or None
            timesteps: Diffusion timestep [B] or None
            static_params: LEGACY — [B, 4], auto-decomposed if intensity is None

        Returns:
            {"signal": [B, 1, T]}
        """
        B, C, T = x.shape
        assert C == 1 and T == self.sequence_length, f"Expected [B, 1, {self.sequence_length}], got {x.shape}"

        # Legacy compatibility: decompose static_params if new-style args not provided
        if static_params is not None and intensity is None:
            intensity = static_params[:, 1]  # Intensity is index 1
            aux_static = torch.cat([static_params[:, 0:1], static_params[:, 2:4]], dim=1)  # Age, StimRate, FMP

        # Compute conditioning embeddings
        pre_cond, post_cond, c_emb = self._embed_conditioning(intensity, aux_static, class_label, B)

        # Stem → token space
        h = self.stem(x).transpose(1, 2)  # [B, T, D]

        # Pre-transformer FiLM (intensity + aux conditioning)
        h = self.film_pre(h, pre_cond)

        # Timestep conditioning (additive)
        h = self.t_adapter(h, timesteps)

        # Class conditioning (additive, broadcast over T)
        h = h + c_emb.unsqueeze(1)

        # Transformer backbone
        h = self.transformer(h)  # [B, T, D]

        # Post-transformer FiLM (intensity + class conditioning)
        h = self.film_post(h, post_cond)

        # Output projection
        y = self.out_proj(self.out_norm(h))  # [B, T, 1]
        y = y.transpose(1, 2).contiguous()   # [B, 1, T]

        return {"signal": y}
