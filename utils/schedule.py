"""
Noise Schedules for Diffusion Models

Implements various noise schedules for denoising diffusion probabilistic models,
including cosine and linear schedules with proper parameterization.

Author: AI Assistant
Date: January 2025
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
import math


def get_noise_schedule(
    schedule_type: str = 'cosine',
    num_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    s: float = 0.008
) -> Dict[str, torch.Tensor]:
    """
    Create noise schedule for diffusion models.
    
    Args:
        schedule_type: Type of schedule ('cosine', 'linear', 'quadratic')
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value for linear schedule
        beta_end: Ending beta value for linear schedule
        s: Small offset for cosine schedule
        
    Returns:
        Dictionary containing all schedule parameters
    """
    if schedule_type == 'cosine':
        return cosine_beta_schedule(num_timesteps, s)
    elif schedule_type == 'linear':
        return linear_beta_schedule(num_timesteps, beta_start, beta_end)
    elif schedule_type == 'quadratic':
        return quadratic_beta_schedule(num_timesteps, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> Dict[str, torch.Tensor]:
    """
    Cosine beta schedule as proposed in "Improved Denoising Diffusion Probabilistic Models".
    
    Args:
        num_timesteps: Number of diffusion timesteps
        s: Small offset to prevent beta from being too small near t=0
        
    Returns:
        Dictionary containing schedule parameters
    """
    def alpha_bar(t):
        return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    
    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        beta = min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999)
        betas.append(beta)
    
    betas = torch.tensor(betas, dtype=torch.float32)
    return _compute_schedule_parameters(betas)


def linear_beta_schedule(
    num_timesteps: int, 
    beta_start: float = 0.0001, 
    beta_end: float = 0.02
) -> Dict[str, torch.Tensor]:
    """
    Linear beta schedule.
    
    Args:
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        
    Returns:
        Dictionary containing schedule parameters
    """
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    return _compute_schedule_parameters(betas)


def quadratic_beta_schedule(
    num_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02
) -> Dict[str, torch.Tensor]:
    """
    Quadratic beta schedule.
    
    Args:
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        
    Returns:
        Dictionary containing schedule parameters
    """
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2
    return _compute_schedule_parameters(betas)


def _compute_schedule_parameters(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute all derived parameters from beta schedule.
    
    Args:
        betas: Beta values [num_timesteps]
        
    Returns:
        Dictionary containing all schedule parameters
    """
    num_timesteps = len(betas)
    
    # Basic parameters
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    
    # Square roots for sampling
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
    
    # Posterior variance
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    # Log variance (clipped for numerical stability)
    posterior_log_variance_clipped = torch.log(
        torch.cat([posterior_variance[1:2], posterior_variance[1:]])
    )
    
    # Mean coefficients
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas_cumprod': sqrt_recip_alphas_cumprod,
        'sqrt_recipm1_alphas_cumprod': sqrt_recipm1_alphas_cumprod,
        'posterior_variance': posterior_variance,
        'posterior_log_variance_clipped': posterior_log_variance_clipped,
        'posterior_mean_coef1': posterior_mean_coef1,
        'posterior_mean_coef2': posterior_mean_coef2,
        'num_timesteps': num_timesteps
    }


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    Extract values from tensor a at indices t and reshape to match x_shape.
    
    Args:
        a: Tensor to extract from [num_timesteps, ...]
        t: Timestep indices [batch_size]
        x_shape: Shape to broadcast to
        
    Returns:
        Extracted and reshaped tensor
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class NoiseSchedule:
    """
    Wrapper class for noise schedules with convenient methods.
    """
    
    def __init__(self, schedule_dict: Dict[str, torch.Tensor]):
        """
        Initialize noise schedule.
        
        Args:
            schedule_dict: Dictionary containing schedule parameters
        """
        for key, value in schedule_dict.items():
            setattr(self, key, value)
        
        self.num_timesteps = len(self.betas)
    
    def q_mean_variance(self, x_start: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Get the distribution q(x_t | x_0).
        
        Args:
            x_start: Clean data [batch, ...]
            t: Timesteps [batch]
            
        Returns:
            Tuple of (mean, variance, log_variance)
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(torch.log(1.0 - self.alphas_cumprod), t, x_start.shape)
        
        return mean, variance, log_variance
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        
        Args:
            x_start: Clean data [batch, ...]
            t: Timesteps [batch]
            noise: Optional noise tensor
            
        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        
        Args:
            x_start: Clean data [batch, ...]
            x_t: Noisy data at timestep t [batch, ...]
            t: Timesteps [batch]
            
        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        
        Args:
            x_t: Noisy data at timestep t [batch, ...]
            t: Timesteps [batch]
            noise: Predicted noise [batch, ...]
            
        Returns:
            Predicted clean data x_0
        """
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        """
        Predict noise from x_t and x_0.
        
        Args:
            x_t: Noisy data at timestep t [batch, ...]
            t: Timesteps [batch]
            x_start: Clean data [batch, ...]
            
        Returns:
            Predicted noise
        """
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return (sqrt_recip_alphas_cumprod_t * x_t - x_start) / sqrt_recipm1_alphas_cumprod_t 