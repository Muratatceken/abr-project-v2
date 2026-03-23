"""
DDIM sampler for v-prediction diffusion models.

Supports:
- Deterministic (eta=0) and stochastic (eta>0) sampling
- Dual classifier-free guidance (static + class conditioning)
- Intensity-conditioned generation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import numpy as np
from tqdm import tqdm

from utils.schedules import prepare_noise_schedule, predict_x0_from_v


# ═══════════════════════════════════════════════════════════════════════════════
# Core DDIM sampling
# ═══════════════════════════════════════════════════════════════════════════════

def ddim_sample_vpred(
    model: nn.Module,
    sched: Dict[str, torch.Tensor],
    shape: Tuple[int, ...],
    intensity: Optional[torch.Tensor] = None,
    aux_static: Optional[torch.Tensor] = None,
    class_label: Optional[torch.Tensor] = None,
    steps: int = 50,
    eta: float = 0.0,
    device: torch.device = torch.device('cpu'),
    progress: bool = True,
    cfg_scale: float = 1.0,
    class_cfg_scale: float = 1.0,
    # Legacy support
    static_params: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    DDIM sampling with v-prediction and dual classifier-free guidance.

    Args:
        model: Trained diffusion model
        sched: Noise schedule dict from prepare_noise_schedule()
        shape: (batch_size, 1, sequence_length)
        intensity: [B] normalized intensity or None
        aux_static: [B, 3] auxiliary static params or None
        class_label: [B] long class labels or None
        steps: DDIM denoising steps
        eta: 0.0=deterministic, 1.0=DDPM
        cfg_scale: Guidance scale for static conditioning
        class_cfg_scale: Guidance scale for class conditioning
        static_params: LEGACY [B, 4] — auto-decomposed
    """
    model.eval()

    # Legacy compat
    if static_params is not None and intensity is None:
        intensity = static_params[:, 1]
        aux_static = torch.cat([static_params[:, 0:1], static_params[:, 2:4]], dim=1)

    # Timestep schedule
    total_timesteps = len(sched['betas'])
    timestep_indices = np.linspace(0, total_timesteps - 1, steps).astype(int)
    timesteps = torch.from_numpy(timestep_indices).long().to(device)

    # Start from pure noise
    x_t = torch.randn(shape, device=device)

    use_cfg = (cfg_scale > 1.0 or class_cfg_scale > 1.0) and (intensity is not None or class_label is not None)

    for i, t in enumerate(tqdm(timesteps.flip(0), desc="DDIM Sampling", disable=not progress)):
        t_batch = t.repeat(shape[0])

        with torch.no_grad():
            if use_cfg:
                # Fully conditioned
                v_cond = model(x_t, intensity=intensity, aux_static=aux_static,
                              class_label=class_label, timesteps=t_batch)["signal"]
                # Fully unconditional
                v_uncond = model(x_t, intensity=None, aux_static=None,
                                class_label=None, timesteps=t_batch)["signal"]

                if class_label is not None and class_cfg_scale != cfg_scale:
                    # Class-only conditioned (for separate class guidance)
                    v_class = model(x_t, intensity=None, aux_static=None,
                                   class_label=class_label, timesteps=t_batch)["signal"]
                    # Dual CFG: static guidance + class guidance
                    v_pred = (v_uncond
                              + cfg_scale * (v_cond - v_class)
                              + class_cfg_scale * (v_class - v_uncond))
                else:
                    # Single CFG scale
                    v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = model(x_t, intensity=intensity, aux_static=aux_static,
                              class_label=class_label, timesteps=t_batch)["signal"]

        # Predict x_0 and clamp
        x_0_pred = predict_x0_from_v(x_t, v_pred, t_batch, sched)
        x_0_pred = x_0_pred.clamp(-3.0, 3.0)

        if i < len(timesteps) - 1:
            t_prev = timesteps.flip(0)[i + 1]
            t_prev_batch = t_prev.repeat(shape[0])
            x_t = ddim_step(x_t, x_0_pred, t_batch, t_prev_batch, sched, eta)
        else:
            x_t = x_0_pred

    return x_t


def ddim_step(
    x_t: torch.Tensor,
    x_0_pred: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    sched: Dict[str, torch.Tensor],
    eta: float = 0.0,
) -> torch.Tensor:
    """Single DDIM denoising step."""
    alpha_bar_t = sched['alpha_bars'][t]
    alpha_bar_t_prev = sched['alpha_bars'][t_prev]

    shape = [t.shape[0]] + [1] * (x_t.ndim - 1)
    alpha_bar_t = alpha_bar_t.view(shape)
    alpha_bar_t_prev = alpha_bar_t_prev.view(shape)

    # Predicted noise
    sqrt_ab = torch.sqrt(alpha_bar_t)
    sqrt_1mab = torch.sqrt(1 - alpha_bar_t)
    eps_pred = (x_t - sqrt_ab * x_0_pred) / sqrt_1mab

    # DDIM variance (with NaN protection)
    sigma_t = eta * torch.sqrt(
        torch.clamp((1 - alpha_bar_t_prev) / (1 - alpha_bar_t), min=0.0)
    ) * torch.sqrt(
        torch.clamp(1 - alpha_bar_t / alpha_bar_t_prev, min=0.0)
    )

    # Predicted mean
    sqrt_ab_prev = torch.sqrt(alpha_bar_t_prev)
    pred_mean = sqrt_ab_prev * x_0_pred + torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * eps_pred

    if eta > 0:
        return pred_mean + sigma_t * torch.randn_like(x_t)
    return pred_mean


# ═══════════════════════════════════════════════════════════════════════════════
# DDIMSampler class
# ═══════════════════════════════════════════════════════════════════════════════

class DDIMSampler:
    """High-level DDIM sampler with dual CFG support."""

    def __init__(self, model: nn.Module, schedule_steps: int = 1000,
                 device: Optional[torch.device] = None):
        self.model = model
        self.schedule_steps = schedule_steps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sched = prepare_noise_schedule(schedule_steps, self.device)

    def sample(
        self,
        batch_size: int,
        sequence_length: int = 200,
        intensity: Optional[torch.Tensor] = None,
        aux_static: Optional[torch.Tensor] = None,
        class_label: Optional[torch.Tensor] = None,
        steps: int = 50,
        eta: float = 0.0,
        cfg_scale: float = 1.0,
        class_cfg_scale: float = 1.0,
        progress: bool = True,
        # Legacy
        static_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples. Returns [B, 1, T]."""
        shape = (batch_size, 1, sequence_length)
        return ddim_sample_vpred(
            self.model, self.sched, shape,
            intensity=intensity, aux_static=aux_static, class_label=class_label,
            steps=steps, eta=eta, device=self.device, progress=progress,
            cfg_scale=cfg_scale, class_cfg_scale=class_cfg_scale,
            static_params=static_params,
        )

    def sample_conditioned(
        self,
        intensity: Optional[torch.Tensor] = None,
        aux_static: Optional[torch.Tensor] = None,
        class_label: Optional[torch.Tensor] = None,
        steps: int = 50,
        eta: float = 0.0,
        cfg_scale: float = 2.0,
        class_cfg_scale: float = 1.5,
        progress: bool = True,
        # Legacy
        static_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate conditioned samples. Batch size inferred from inputs."""
        # Determine batch size from any provided conditioning
        B = 1
        for t in [intensity, aux_static, class_label, static_params]:
            if t is not None:
                B = t.shape[0]
                break

        return self.sample(
            batch_size=B, intensity=intensity, aux_static=aux_static,
            class_label=class_label, steps=steps, eta=eta,
            cfg_scale=cfg_scale, class_cfg_scale=class_cfg_scale,
            progress=progress, static_params=static_params,
        )

    def sample_class_conditioned(
        self,
        class_label: torch.Tensor,
        intensity: Optional[torch.Tensor] = None,
        aux_static: Optional[torch.Tensor] = None,
        steps: int = 50,
        eta: float = 0.0,
        cfg_scale: float = 2.0,
        class_cfg_scale: float = 1.5,
        progress: bool = True,
    ) -> torch.Tensor:
        """Generate samples conditioned on hearing loss class."""
        return self.sample_conditioned(
            intensity=intensity, aux_static=aux_static, class_label=class_label,
            steps=steps, eta=eta, cfg_scale=cfg_scale,
            class_cfg_scale=class_cfg_scale, progress=progress,
        )
