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


# ============================================================================
# Progressive Training Utilities for Multi-Task Learning
# ============================================================================

def linear_weight_schedule(epoch: int, start_epoch: int, end_epoch: int, 
                          start_weight: float, end_weight: float) -> float:
    """
    Create linear interpolation between start and end weights over specified epoch range.
    
    Args:
        epoch: Current training epoch
        start_epoch: Epoch to start weight ramping
        end_epoch: Epoch to finish weight ramping
        start_weight: Initial weight value
        end_weight: Final weight value
        
    Returns:
        Current weight value based on linear interpolation
    """
    if epoch < start_epoch:
        return start_weight
    elif epoch >= end_epoch:
        return end_weight
    else:
        # Linear interpolation between start and end epochs
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        return start_weight + progress * (end_weight - start_weight)


def cosine_weight_schedule(epoch: int, start_epoch: int, end_epoch: int,
                          start_weight: float, end_weight: float) -> float:
    """
    Create cosine interpolation between start and end weights over specified epoch range.
    
    Args:
        epoch: Current training epoch
        start_epoch: Epoch to start weight ramping
        end_epoch: Epoch to finish weight ramping
        start_weight: Initial weight value
        end_weight: Final weight value
        
    Returns:
        Current weight value based on cosine interpolation
    """
    if epoch < start_epoch:
        return start_weight
    elif epoch >= end_epoch:
        return end_weight
    else:
        # Cosine interpolation for smoother transitions
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        # Use cosine function for smooth interpolation
        cos_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        return start_weight + cos_progress * (end_weight - start_weight)


def create_param_groups(model, base_lr: float, task_lr_multipliers: dict) -> list:
    """
    Separate model parameters into task-specific groups for different learning rates.
    
    Args:
        model: PyTorch model with named parameters
        base_lr: Base learning rate
        task_lr_multipliers: Dictionary of task-specific learning rate multipliers
        
    Returns:
        List of parameter groups for optimizer
    """
    param_groups = []
    
    # Signal generation parameters (stem, transformer, output projection)
    signal_params = []
    peak_params = []
    static_params = []
    cross_attn_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(x in name for x in ['stem', 'transformer', 'out_proj', 'out_norm']):
                signal_params.append(param)
            elif any(x in name for x in ['attn_pool', 'peak5_head']):
                peak_params.append(param)
            elif any(x in name for x in ['static_recon_head']):
                static_params.append(param)
            elif any(x in name for x in ['static_encoder', 'cross_attention']):
                cross_attn_params.append(param)
            else:
                other_params.append(param)
    
    # Create parameter groups with appropriate learning rates
    if signal_params:
        param_groups.append({
            'params': signal_params,
            'lr': base_lr * task_lr_multipliers.get('signal', 1.0),
            'name': 'signal_generation'
        })
    
    if peak_params:
        param_groups.append({
            'params': peak_params,
            'lr': base_lr * task_lr_multipliers.get('peak_classification', 1.0),
            'name': 'peak_classification'
        })
    
    if static_params:
        param_groups.append({
            'params': static_params,
            'lr': base_lr * task_lr_multipliers.get('static_reconstruction', 1.0),
            'name': 'static_reconstruction'
        })
    
    if cross_attn_params:
        param_groups.append({
            'params': cross_attn_params,
            'lr': base_lr * task_lr_multipliers.get('cross_attention', 1.0),
            'name': 'cross_attention'
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'other'
        })
    
    return param_groups
