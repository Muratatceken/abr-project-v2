"""
DDIM sampler for v-prediction diffusion models.

Implements deterministic and stochastic sampling from trained diffusion models.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import numpy as np
from tqdm import tqdm

from utils.schedules import prepare_noise_schedule, predict_x0_from_v


def ddim_sample_vpred(
    model: nn.Module,
    sched: Dict[str, torch.Tensor], 
    shape: Tuple[int, ...],
    static_params: Optional[torch.Tensor] = None,
    steps: int = 50,
    eta: float = 0.0,
    device: torch.device = torch.device('cpu'),
    progress: bool = True,
    cfg_scale: float = 1.0
) -> torch.Tensor:
    """
    DDIM sampling with v-prediction parameterization.
    
    Args:
        model: Trained diffusion model
        sched: Noise schedule dictionary from prepare_noise_schedule()
        shape: Output shape tuple, e.g., (batch_size, 1, sequence_length)
        static_params: Static conditioning parameters [B, S] or None
        steps: Number of DDIM steps
        eta: DDIM eta parameter (0.0 = deterministic, 1.0 = DDPM)
        device: Device to run sampling on
        progress: Whether to show progress bar
        cfg_scale: Classifier-free guidance scale (>1.0 enables CFG)
        
    Returns:
        Generated samples [B, C, T]
    """
    model.eval()
    
    # Create timestep schedule for DDIM
    total_timesteps = len(sched['betas'])
    timestep_indices = np.linspace(0, total_timesteps - 1, steps).astype(int)
    timesteps = torch.from_numpy(timestep_indices).long().to(device)
    
    # Start from pure noise
    x_t = torch.randn(shape, device=device)
    
    # Reverse process
    for i, t in enumerate(tqdm(timesteps.flip(0), desc="DDIM Sampling", disable=not progress)):
        # Current timestep
        t_batch = t.repeat(shape[0])
        
        # Get model prediction
        with torch.no_grad():
            if cfg_scale > 1.0 and static_params is not None:
                # Classifier-free guidance
                # Conditional prediction
                v_pred_cond = model(x_t, static_params=static_params, timesteps=t_batch)["signal"]
                
                # Unconditional prediction
                v_pred_uncond = model(x_t, static_params=None, timesteps=t_batch)["signal"]
                
                # Apply CFG
                v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
            else:
                # Standard conditional or unconditional prediction
                v_pred = model(x_t, static_params=static_params, timesteps=t_batch)["signal"]
        
        # Predict x_0 from v-prediction
        x_0_pred = predict_x0_from_v(x_t, v_pred, t_batch, sched)
        
        if i < len(timesteps) - 1:
            # Get next timestep
            t_prev = timesteps.flip(0)[i + 1]
            t_prev_batch = t_prev.repeat(shape[0])
            
            # DDIM update
            x_t = ddim_step(x_t, x_0_pred, t_batch, t_prev_batch, sched, eta)
        else:
            # Final step
            x_t = x_0_pred
    
    return x_t


def ddim_step(
    x_t: torch.Tensor,
    x_0_pred: torch.Tensor, 
    t: torch.Tensor,
    t_prev: torch.Tensor,
    sched: Dict[str, torch.Tensor],
    eta: float = 0.0
) -> torch.Tensor:
    """
    Single DDIM step.
    
    Args:
        x_t: Current noisy sample [B, ...]
        x_0_pred: Predicted clean sample [B, ...]
        t: Current timestep [B]
        t_prev: Previous timestep [B] 
        sched: Noise schedule dictionary
        eta: DDIM eta parameter
        
    Returns:
        Next sample x_{t-1} [B, ...]
    """
    # Get alpha_bar values
    alpha_bar_t = sched['alpha_bars'][t]
    alpha_bar_t_prev = sched['alpha_bars'][t_prev]
    
    # Reshape for broadcasting
    shape = [t.shape[0]] + [1] * (x_t.ndim - 1)
    alpha_bar_t = alpha_bar_t.view(shape)
    alpha_bar_t_prev = alpha_bar_t_prev.view(shape)
    
    # Compute predicted noise
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    eps_pred = (x_t - sqrt_alpha_bar_t * x_0_pred) / sqrt_one_minus_alpha_bar_t
    
    # DDIM variance
    sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
    
    # Predicted mean
    sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
    sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)
    
    pred_mean = sqrt_alpha_bar_t_prev * x_0_pred + torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * eps_pred
    
    # Add noise if eta > 0
    if eta > 0:
        noise = torch.randn_like(x_t)
        x_t_prev = pred_mean + sigma_t * noise
    else:
        x_t_prev = pred_mean
    
    return x_t_prev


class DDIMSampler:
    """
    DDIM sampler class for easier usage and configuration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        schedule_steps: int = 1000,
        device: Optional[torch.device] = None
    ):
        """
        Initialize DDIM sampler.
        
        Args:
            model: Trained diffusion model
            schedule_steps: Number of steps in training noise schedule
            device: Device to run on
        """
        self.model = model
        self.schedule_steps = schedule_steps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare noise schedule
        self.sched = prepare_noise_schedule(schedule_steps, self.device)
    
    def sample(
        self,
        batch_size: int,
        sequence_length: int = 200,
        static_params: Optional[torch.Tensor] = None,
        steps: int = 50,
        eta: float = 0.0,
        cfg_scale: float = 1.0,
        progress: bool = True
    ) -> torch.Tensor:
        """
        Generate samples using DDIM.
        
        Args:
            batch_size: Number of samples to generate
            sequence_length: Length of each sequence
            static_params: Static conditioning [B, S] or None
            steps: Number of DDIM steps
            eta: DDIM eta parameter
            cfg_scale: Classifier-free guidance scale
            progress: Show progress bar
            
        Returns:
            Generated samples [B, 1, T]
        """
        shape = (batch_size, 1, sequence_length)
        
        return ddim_sample_vpred(
            model=self.model,
            sched=self.sched,
            shape=shape,
            static_params=static_params,
            steps=steps,
            eta=eta,
            device=self.device,
            progress=progress,
            cfg_scale=cfg_scale
        )
    
    def sample_conditioned(
        self,
        static_params: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0,
        cfg_scale: float = 1.0,
        progress: bool = True
    ) -> torch.Tensor:
        """
        Generate samples conditioned on given static parameters.
        
        Args:
            static_params: Static conditioning parameters [B, S]
            steps: Number of DDIM steps
            eta: DDIM eta parameter
            cfg_scale: Classifier-free guidance scale
            progress: Show progress bar
            
        Returns:
            Generated samples [B, 1, T]
        """
        batch_size = static_params.shape[0]
        static_params = static_params.to(self.device)
        
        return self.sample(
            batch_size=batch_size,
            static_params=static_params,
            steps=steps,
            eta=eta,
            cfg_scale=cfg_scale,
            progress=progress
        )
    
    def interpolate(
        self,
        static_params_a: torch.Tensor,
        static_params_b: torch.Tensor,
        num_interpolations: int = 8,
        steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Generate interpolations between two sets of static parameters.
        
        Args:
            static_params_a: Starting static parameters [1, S]
            static_params_b: Ending static parameters [1, S]
            num_interpolations: Number of interpolation steps
            steps: Number of DDIM steps
            eta: DDIM eta parameter
            
        Returns:
            Interpolated samples [num_interpolations, 1, T]
        """
        # Create interpolation weights
        weights = torch.linspace(0, 1, num_interpolations, device=self.device)
        
        # Interpolate static parameters
        interp_static = []
        for w in weights:
            interp_param = (1 - w) * static_params_a + w * static_params_b
            interp_static.append(interp_param)
        
        interp_static = torch.cat(interp_static, dim=0)  # [num_interpolations, S]
        
        return self.sample_conditioned(
            static_params=interp_static,
            steps=steps,
            eta=eta,
            progress=True
        )
