"""
DDIM Sampling for ABR Signal Generation

Implements Denoising Diffusion Implicit Models (DDIM) sampling for conditional
ABR signal generation with classifier-free guidance support.

Author: AI Assistant
Date: January 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, Union, Dict, Any
from tqdm import tqdm


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) sampler for ABR signal generation.
    
    Supports:
    - Conditional generation with static parameters
    - Classifier-free guidance (CFG)
    - Deterministic and stochastic sampling
    - Flexible noise schedules
    """
    
    def __init__(
        self,
        noise_schedule: Dict[str, torch.Tensor],
        eta: float = 0.0,
        clip_denoised: bool = True
    ):
        """
        Initialize DDIM sampler.
        
        Args:
            noise_schedule: Dictionary containing noise schedule parameters
            eta: Stochasticity parameter (0.0 = deterministic, 1.0 = DDPM)
            clip_denoised: Whether to clip denoised predictions
        """
        self.noise_schedule = noise_schedule
        self.eta = eta
        self.clip_denoised = clip_denoised
        
        # Extract schedule parameters
        self.betas = noise_schedule['betas']
        self.alphas = noise_schedule['alphas']
        self.alphas_cumprod = noise_schedule['alphas_cumprod']
        self.alphas_cumprod_prev = noise_schedule['alphas_cumprod_prev']
        self.sqrt_alphas_cumprod = noise_schedule['sqrt_alphas_cumprod']
        self.sqrt_one_minus_alphas_cumprod = noise_schedule['sqrt_one_minus_alphas_cumprod']
        
        self.num_timesteps = len(self.betas)
    
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        static_params: torch.Tensor,
        device: torch.device,
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        cfg_mask: Optional[torch.Tensor] = None,
        progress: bool = True,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Generate samples using DDIM sampling.
        
        Args:
            model: Trained diffusion model
            shape: Shape of samples to generate (batch, channels, length)
            static_params: Static conditioning parameters [batch, static_dim]
            device: Device for computation
            num_steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            cfg_mask: Mask for which samples to apply CFG [batch]
            progress: Show progress bar
            return_intermediates: Return intermediate states
            
        Returns:
            Generated samples [batch, channels, length]
        """
        batch_size = shape[0]
        
        # Create timestep schedule
        timesteps = self._get_timestep_schedule(num_steps)
        
        # Initialize with noise
        x = torch.randn(shape, device=device)
        
        # Storage for intermediates
        intermediates = [] if return_intermediates else None
        
        # Setup progress bar
        iterator = tqdm(timesteps, desc="DDIM Sampling") if progress else timesteps
        
        # Sampling loop
        for i, timestep in enumerate(iterator):
            # Current and previous timesteps
            t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            
            # Predict noise
            if cfg_scale > 1.0 and cfg_mask is not None:
                # Classifier-free guidance
                noise_pred = self._apply_cfg(
                    model, x, t, static_params, cfg_scale, cfg_mask
                )
            else:
                # Standard conditional prediction
                noise_pred = model(x, static_params, t)
                if isinstance(noise_pred, dict):
                    # Prefer explicit noise when in diffusion mode, else recon
                    if 'noise' in noise_pred:
                        noise_pred = noise_pred['noise']
                    elif 'recon' in noise_pred:
                        noise_pred = noise_pred['recon']
                    else:
                        noise_pred = noise_pred.get('noise_pred', noise_pred.get('pred', x))
                
                # Ensure noise_pred has the same shape as x
                if noise_pred.dim() == 2 and x.dim() == 3:
                    noise_pred = noise_pred.unsqueeze(1)  # Add channel dimension
                elif noise_pred.shape != x.shape:
                    # Handle other shape mismatches
                    if noise_pred.size(0) == x.size(0) and noise_pred.size(-1) == x.size(-1):
                        # Reshape to match input channels
                        noise_pred = noise_pred.view(x.shape)
            
            # DDIM step
            x = self._ddim_step(x, noise_pred, timestep, i, timesteps)
            
            # Store intermediate if requested
            if return_intermediates:
                intermediates.append(x.clone())
        
        # Final denoising step
        x = self._final_denoise(model, x, static_params, cfg_scale, cfg_mask)
        
        if return_intermediates:
            return x, intermediates
        else:
            return x
    
    def _get_timestep_schedule(self, num_steps: int) -> torch.Tensor:
        """Create timestep schedule for sampling."""
        # We must iterate from high noise to low noise (T-1 -> 0)
        # Returning in descending order ensures the DDIM update uses the
        # previous (less noisy) timestep correctly.
        timesteps = torch.linspace(self.num_timesteps - 1, 0, steps=num_steps, dtype=torch.long)
        return timesteps  # Already descending: [T-1, ..., 0]
    
    def _apply_cfg(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        static_params: torch.Tensor,
        cfg_scale: float,
        cfg_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply classifier-free guidance."""
        batch_size = x.size(0)
        
        # Create unconditional static parameters (zeros or special tokens)
        uncond_static = torch.zeros_like(static_params)
        
        # Combine conditional and unconditional inputs
        x_combined = torch.cat([x, x], dim=0)
        t_combined = torch.cat([t, t], dim=0)
        static_combined = torch.cat([static_params, uncond_static], dim=0)
        
        # Get model predictions
        noise_pred_combined = model(x_combined, static_combined, t_combined)
        
        # Handle different output formats
        if isinstance(noise_pred_combined, dict):
            if 'noise' in noise_pred_combined:
                noise_pred_combined = noise_pred_combined['noise']
            elif 'recon' in noise_pred_combined:
                noise_pred_combined = noise_pred_combined['recon']
            else:
                noise_pred_combined = noise_pred_combined.get('noise_pred', noise_pred_combined.get('pred', x_combined))
        
        # Ensure noise_pred_combined has the same shape as x_combined
        if noise_pred_combined.dim() == 2 and x_combined.dim() == 3:
            noise_pred_combined = noise_pred_combined.unsqueeze(1)  # Add channel dimension
        elif noise_pred_combined.shape != x_combined.shape:
            # Handle other shape mismatches
            if noise_pred_combined.size(0) == x_combined.size(0) and noise_pred_combined.size(-1) == x_combined.size(-1):
                # Reshape to match input channels
                noise_pred_combined = noise_pred_combined.view(x_combined.shape)
        
        # Split conditional and unconditional predictions
        noise_pred_cond, noise_pred_uncond = noise_pred_combined.chunk(2, dim=0)
        
        # Apply CFG only to masked samples
        noise_pred = noise_pred_uncond.clone()
        if cfg_mask.any():
            # CFG formula: uncond + scale * (cond - uncond)
            cfg_correction = cfg_scale * (noise_pred_cond - noise_pred_uncond)
            noise_pred[cfg_mask] = noise_pred_uncond[cfg_mask] + cfg_correction[cfg_mask]
        
        return noise_pred
    
    def _ddim_step(
        self,
        x: torch.Tensor,
        noise_pred: torch.Tensor,
        timestep: int,
        step_idx: int,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Perform single DDIM denoising step."""
        # Get noise schedule values
        alpha_cumprod = self.alphas_cumprod[timestep]
        
        # Get previous timestep
        if step_idx < len(timesteps) - 1:
            prev_timestep = timesteps[step_idx + 1]
            alpha_cumprod_prev = self.alphas_cumprod[prev_timestep]
        else:
            alpha_cumprod_prev = torch.tensor(1.0, device=x.device)
        
        # Predict x0 (clean signal)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
        
        pred_x0 = (x - sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod
        
        # Clip if requested
        if self.clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Compute direction to x_t
        sqrt_alpha_cumprod_prev = torch.sqrt(alpha_cumprod_prev)
        sqrt_one_minus_alpha_cumprod_prev = torch.sqrt(1 - alpha_cumprod_prev)
        
        # DDIM formula
        dir_xt = sqrt_one_minus_alpha_cumprod_prev * noise_pred
        
        # Add stochasticity if eta > 0
        if self.eta > 0:
            sigma = self.eta * torch.sqrt(
                (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * 
                (1 - alpha_cumprod / alpha_cumprod_prev)
            )
            noise = torch.randn_like(x)
            dir_xt = dir_xt + sigma * noise
        
        # Compute x_{t-1}
        x_prev = sqrt_alpha_cumprod_prev * pred_x0 + dir_xt
        
        return x_prev
    
    def _final_denoise(
        self,
        model: nn.Module,
        x: torch.Tensor,
        static_params: torch.Tensor,
        cfg_scale: float,
        cfg_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Final denoising step to get clean signal."""
        batch_size = x.size(0)
        t = torch.zeros((batch_size,), device=x.device, dtype=torch.long)
        
        # Get final noise prediction
        if cfg_scale > 1.0 and cfg_mask is not None:
            noise_pred = self._apply_cfg(model, x, t, static_params, cfg_scale, cfg_mask)
        else:
            noise_pred = model(x, static_params, t)
            if isinstance(noise_pred, dict):
                if 'noise' in noise_pred:
                    noise_pred = noise_pred['noise']
                elif 'recon' in noise_pred:
                    noise_pred = noise_pred['recon']
                else:
                    noise_pred = noise_pred.get('noise_pred', noise_pred.get('pred', x))
            
            # Ensure noise_pred has the same shape as x
            if noise_pred.dim() == 2 and x.dim() == 3:
                noise_pred = noise_pred.unsqueeze(1)  # Add channel dimension
            elif noise_pred.shape != x.shape:
                # Handle other shape mismatches
                if noise_pred.size(0) == x.size(0) and noise_pred.size(-1) == x.size(-1):
                    # Reshape to match input channels
                    noise_pred = noise_pred.view(x.shape)
        
        # Final denoising
        alpha_cumprod = self.alphas_cumprod[0]
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
        
        x_clean = (x - sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod
        
        if self.clip_denoised:
            x_clean = torch.clamp(x_clean, -1.0, 1.0)
        
        return x_clean
    
    def ddim_reverse_loop(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        static_params: torch.Tensor,
        num_steps: int = 50,
        progress: bool = False
    ) -> torch.Tensor:
        """
        DDIM reverse process (encoding) - useful for evaluation.
        
        Args:
            model: Trained diffusion model
            x_start: Clean signals to encode [batch, channels, length]
            static_params: Static conditioning parameters
            num_steps: Number of reverse steps
            progress: Show progress bar
            
        Returns:
            Encoded latent representations
        """
        batch_size = x_start.size(0)
        device = x_start.device
        
        # Create reverse timestep schedule
        timesteps = self._get_timestep_schedule(num_steps)
        timesteps = timesteps.flip(0)  # Reverse for encoding
        
        x = x_start.clone()
        
        iterator = tqdm(timesteps, desc="DDIM Reverse") if progress else timesteps
        
        for i, timestep in enumerate(iterator):
            t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model(x, static_params, t)
            if isinstance(noise_pred, dict):
                if 'noise' in noise_pred:
                    noise_pred = noise_pred['noise']
                elif 'recon' in noise_pred:
                    noise_pred = noise_pred['recon']
                else:
                    noise_pred = noise_pred.get('noise_pred', noise_pred.get('pred', x))
            
            # Reverse DDIM step
            x = self._ddim_reverse_step(x, noise_pred, timestep, i, timesteps)
        
        return x
    
    def _ddim_reverse_step(
        self,
        x: torch.Tensor,
        noise_pred: torch.Tensor,
        timestep: int,
        step_idx: int,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Perform single DDIM reverse (encoding) step."""
        # Get noise schedule values
        alpha_cumprod = self.alphas_cumprod[timestep]
        
        # Get next timestep
        if step_idx < len(timesteps) - 1:
            next_timestep = timesteps[step_idx + 1]
            alpha_cumprod_next = self.alphas_cumprod[next_timestep]
        else:
            alpha_cumprod_next = torch.tensor(0.0, device=x.device)
        
        # Predict x0
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
        
        pred_x0 = (x - sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod
        
        # Compute x_{t+1}
        sqrt_alpha_cumprod_next = torch.sqrt(alpha_cumprod_next)
        sqrt_one_minus_alpha_cumprod_next = torch.sqrt(1 - alpha_cumprod_next)
        
        x_next = sqrt_alpha_cumprod_next * pred_x0 + sqrt_one_minus_alpha_cumprod_next * noise_pred
        
        return x_next 

def create_ddim_sampler(
    noise_schedule_type: str = 'cosine',
    num_timesteps: int = 1000,
    eta: float = 0.0,
    clip_denoised: bool = True
) -> DDIMSampler:
    """
    Create DDIM sampler with specified noise schedule.
    
    Args:
        noise_schedule_type: Type of noise schedule ('cosine', 'linear')
        num_timesteps: Number of diffusion timesteps
        eta: Stochasticity parameter
        clip_denoised: Whether to clip denoised predictions
        
    Returns:
        Configured DDIM sampler
    """
    from .schedule import get_noise_schedule
    
    noise_schedule = get_noise_schedule(noise_schedule_type, num_timesteps)
    
    return DDIMSampler(
        noise_schedule=noise_schedule,
        eta=eta,
        clip_denoised=clip_denoised
    ) 