"""
Diffusion noise schedules for v-prediction training.

Implements cosine beta schedule and related utilities for DDPM/DDIM.
"""

import math
import torch
from typing import Dict


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule as proposed in Improved DDPM (Nichol & Dhariwal).
    
    Args:
        T: Number of diffusion timesteps
        s: Small offset to prevent beta from being too small near t=0
        
    Returns:
        Beta schedule tensor of shape [T]
    """
    def alpha_bar(t):
        return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    
    betas = []
    for i in range(T):
        t1 = i / T
        t2 = (i + 1) / T
        beta = min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999)
        betas.append(beta)
    
    return torch.tensor(betas, dtype=torch.float32)


def prepare_noise_schedule(T: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Prepare all noise schedule terms for diffusion.
    
    Args:
        T: Number of diffusion timesteps
        device: Device to place tensors on
        
    Returns:
        Dictionary containing:
            - betas: Beta schedule [T]
            - alphas: Alpha schedule [T] 
            - alpha_bars: Cumulative alpha product [T]
            - sqrt_alpha_bars: sqrt(alpha_bar) [T]
            - sqrt_one_minus_alpha_bars: sqrt(1 - alpha_bar) [T]
    """
    betas = cosine_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
    
    return {
        'betas': betas,
        'alphas': alphas, 
        'alpha_bars': alpha_bars,
        'sqrt_alpha_bars': sqrt_alpha_bars,
        'sqrt_one_minus_alpha_bars': sqrt_one_minus_alpha_bars
    }


def q_sample_vpred(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, 
                   sched: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample q(x_t | x_0) and compute v-prediction target.
    
    V-prediction parameterization:
        x_t = alpha * x_0 + sigma * epsilon
        v = alpha * epsilon - sigma * x_0
    where alpha = sqrt(alpha_bar_t), sigma = sqrt(1 - alpha_bar_t)
    
    Args:
        x0: Clean data [B, ...]
        t: Timesteps [B] 
        noise: Noise tensor [B, ...] ~ N(0, I)
        sched: Noise schedule dictionary
        
    Returns:
        Tuple of (x_t, v_target) where both have shape [B, ...]
    """
    # Get schedule terms for timesteps t
    sqrt_alpha_bar_t = sched['sqrt_alpha_bars'][t]         # [B]
    sqrt_one_minus_alpha_bar_t = sched['sqrt_one_minus_alpha_bars'][t]  # [B]
    
    # Reshape for broadcasting with [B, C, T] tensors
    shape = [t.shape[0]] + [1] * (x0.ndim - 1)
    alpha = sqrt_alpha_bar_t.view(shape)
    sigma = sqrt_one_minus_alpha_bar_t.view(shape)
    
    # Forward diffusion: x_t = alpha * x_0 + sigma * epsilon
    x_t = alpha * x0 + sigma * noise
    
    # V-prediction target: v = alpha * epsilon - sigma * x_0
    v_target = alpha * noise - sigma * x0
    
    return x_t, v_target


def predict_x0_from_v(x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor,
                      sched: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Predict x_0 from v-prediction and x_t.
    
    From v-prediction formulation:
        v = alpha * epsilon - sigma * x_0
        epsilon = (v + sigma * x_0) / alpha  
        x_0 = (alpha * x_t - sigma * v) / (alpha^2 + sigma^2)
        
    Since alpha^2 + sigma^2 = alpha_bar + (1 - alpha_bar) = 1:
        x_0 = alpha * x_t - sigma * v
    
    Args:
        x_t: Noisy samples [B, ...]
        v_pred: Predicted v [B, ...]
        t: Timesteps [B]
        sched: Noise schedule dictionary
        
    Returns:
        Predicted x_0 [B, ...]
    """
    sqrt_alpha_bar_t = sched['sqrt_alpha_bars'][t]
    sqrt_one_minus_alpha_bar_t = sched['sqrt_one_minus_alpha_bars'][t]
    
    # Reshape for broadcasting
    shape = [t.shape[0]] + [1] * (x_t.ndim - 1)
    alpha = sqrt_alpha_bar_t.view(shape)
    sigma = sqrt_one_minus_alpha_bar_t.view(shape)
    
    x0_pred = alpha * x_t - sigma * v_pred
    return x0_pred
