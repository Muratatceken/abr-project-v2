import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np


def cvae_loss(
    recon_signal: torch.Tensor, 
    target_signal: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: torch.Tensor, 
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the full loss for a Conditional Variational Autoencoder (CVAE).
    
    The CVAE loss consists of two components:
    1. Reconstruction loss: How well the model reconstructs the input signal
    2. KL divergence: Regularization term that encourages the latent distribution 
       to be close to the prior N(0, I)
    
    Args:
        recon_signal (torch.Tensor): Reconstructed signal from decoder [batch, signal_length]
        target_signal (torch.Tensor): Original input signal [batch, signal_length]
        mu (torch.Tensor): Mean of latent distribution from encoder [batch, latent_dim]
        logvar (torch.Tensor): Log variance of latent distribution from encoder [batch, latent_dim]
        beta (float): Weight for KL divergence term (beta-VAE parameter, default=1.0)
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_loss: Combined reconstruction + KL loss
            - recon_loss: Reconstruction loss component
            - kl_loss: KL divergence loss component
    """
    # Reconstruction loss (Mean Squared Error)
    # We use MSE since ABR signals are continuous and we want to preserve signal shape
    recon_loss = F.mse_loss(recon_signal, target_signal, reduction='mean')
    
    # KL divergence loss
    # KL(q(z|x) || p(z)) where q(z|x) ~ N(mu, exp(logvar)) and p(z) ~ N(0, I)
    # Analytical solution: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Since we have logvar = log(sigma^2), we can substitute:
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)  # Average over batch
    
    # Total loss with beta weighting
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def static_params_loss(
    recon_static_params: torch.Tensor,
    target_static_params: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the loss for static parameters reconstruction.
    
    Args:
        recon_static_params (torch.Tensor): Reconstructed static parameters [batch, static_dim]
        target_static_params (torch.Tensor): Original static parameters [batch, static_dim]
    
    Returns:
        torch.Tensor: Static parameters reconstruction loss
    """
    # Use MSE loss for static parameters
    return F.mse_loss(recon_static_params, target_static_params, reduction='mean')


def joint_cvae_loss(
    recon_signal: torch.Tensor,
    target_signal: torch.Tensor,
    recon_static_params: torch.Tensor,
    target_static_params: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    static_loss_weight: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the full loss for joint signal and static parameters generation.
    
    Args:
        recon_signal (torch.Tensor): Reconstructed signal [batch, signal_length]
        target_signal (torch.Tensor): Original signal [batch, signal_length]
        recon_static_params (torch.Tensor): Reconstructed static parameters [batch, static_dim]
        target_static_params (torch.Tensor): Original static parameters [batch, static_dim]
        mu (torch.Tensor): Mean of latent distribution [batch, latent_dim]
        logvar (torch.Tensor): Log variance of latent distribution [batch, latent_dim]
        beta (float): Weight for KL divergence term (default=1.0)
        static_loss_weight (float): Weight for static parameters loss (default=1.0)
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_loss: Combined signal + static + KL loss
            - signal_recon_loss: Signal reconstruction loss component
            - static_recon_loss: Static parameters reconstruction loss component
            - kl_loss: KL divergence loss component
    """
    # Signal reconstruction loss
    signal_recon_loss = F.mse_loss(recon_signal, target_signal, reduction='mean')
    
    # Static parameters reconstruction loss
    static_recon_loss = static_params_loss(recon_static_params, target_static_params)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)
    
    # Total loss with weights
    total_loss = signal_recon_loss + static_loss_weight * static_recon_loss + beta * kl_loss
    
    return total_loss, signal_recon_loss, static_recon_loss, kl_loss


def reconstruction_loss(recon_signal: torch.Tensor, target_signal: torch.Tensor) -> torch.Tensor:
    """
    Calculate reconstruction loss (MSE) between reconstructed and target signals.
    
    Args:
        recon_signal (torch.Tensor): Reconstructed signal [batch, signal_length]
        target_signal (torch.Tensor): Target signal [batch, signal_length]
    
    Returns:
        torch.Tensor: Mean squared error loss
    """
    return F.mse_loss(recon_signal, target_signal, reduction='mean')


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Calculate KL divergence loss for VAE.
    
    Computes KL(q(z|x) || p(z)) where:
    - q(z|x) ~ N(mu, exp(logvar)) (approximate posterior)
    - p(z) ~ N(0, I) (prior)
    
    Args:
        mu (torch.Tensor): Mean of latent distribution [batch, latent_dim]
        logvar (torch.Tensor): Log variance of latent distribution [batch, latent_dim]
    
    Returns:
        torch.Tensor: KL divergence loss
    """
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.mean(kl_loss)


def beta_vae_loss(
    recon_signal: torch.Tensor, 
    target_signal: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: torch.Tensor, 
    beta: float = 4.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate beta-VAE loss with higher beta for better disentanglement.
    
    This is a wrapper around cvae_loss with a default beta > 1.0 for 
    encouraging disentangled representations.
    
    Args:
        recon_signal (torch.Tensor): Reconstructed signal [batch, signal_length]
        target_signal (torch.Tensor): Target signal [batch, signal_length]
        mu (torch.Tensor): Mean of latent distribution [batch, latent_dim]
        logvar (torch.Tensor): Log variance of latent distribution [batch, latent_dim]
        beta (float): Weight for KL divergence term (default=4.0 for beta-VAE)
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_loss: Combined reconstruction + beta * KL loss
            - recon_loss: Reconstruction loss component
            - kl_loss: KL divergence loss component
    """
    return cvae_loss(recon_signal, target_signal, mu, logvar, beta=beta)


def weighted_mse_loss(
    recon_signal: torch.Tensor, 
    target_signal: torch.Tensor, 
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Calculate weighted MSE loss for reconstruction.
    
    Useful for emphasizing certain parts of the signal (e.g., peak regions).
    
    Args:
        recon_signal (torch.Tensor): Reconstructed signal [batch, signal_length]
        target_signal (torch.Tensor): Target signal [batch, signal_length]
        weights (torch.Tensor): Weights for each time point [signal_length] or [batch, signal_length]
    
    Returns:
        torch.Tensor: Weighted mean squared error loss
    """
    mse = (recon_signal - target_signal).pow(2)
    weighted_mse = mse * weights
    return torch.mean(weighted_mse)


def peak_loss(
    predicted_peaks: torch.Tensor, 
    target_peaks: torch.Tensor, 
    peak_mask: torch.Tensor,
    loss_type: str = 'mae',
    huber_delta: float = 1.0
) -> torch.Tensor:
    """
    Compute masked loss over peak values for partial supervision with configurable loss types.
    
    This function enables training with partially labeled peak data, where some
    peak values may be missing (NaN) in the target. The peak_mask indicates
    which peak values are valid and should contribute to the loss calculation.
    
    Args:
        predicted_peaks (torch.Tensor): Predicted peak values [batch, num_peaks]
        target_peaks (torch.Tensor): Target peak values [batch, num_peaks] (may contain NaN)
        peak_mask (torch.Tensor): Boolean mask indicating valid peaks [batch, num_peaks]
        loss_type (str): Type of loss function ('mae', 'mse', 'huber', 'smooth_l1')
        huber_delta (float): Delta parameter for Huber loss (default=1.0)
    
    Returns:
        torch.Tensor: Masked loss averaged over valid peaks only
    """
    # Ensure peak_mask is boolean
    peak_mask = peak_mask.bool()
    
    # Create a mask for non-NaN values in target_peaks
    nan_mask = ~torch.isnan(target_peaks)
    
    # Combine with the provided peak_mask
    valid_mask = peak_mask & nan_mask
    
    # Check if there are any valid peaks
    num_valid_peaks = torch.sum(valid_mask.float())
    
    if num_valid_peaks == 0:
        # Return zero loss if no valid peaks
        return torch.tensor(0.0, device=predicted_peaks.device, requires_grad=True)
    
    # Replace NaN values with zeros (they will be masked out anyway)
    target_peaks_clean = torch.where(torch.isnan(target_peaks), torch.zeros_like(target_peaks), target_peaks)
    
    # Calculate loss based on type
    if loss_type == 'mae':
        # Mean Absolute Error - more sensitive to small amplitude errors
        loss = torch.abs(predicted_peaks - target_peaks_clean)
    elif loss_type == 'mse':
        # Mean Squared Error - traditional choice
        loss = (predicted_peaks - target_peaks_clean).pow(2)
    elif loss_type == 'huber':
        # Huber loss - robust to outliers
        diff = torch.abs(predicted_peaks - target_peaks_clean)
        loss = torch.where(diff < huber_delta, 
                          0.5 * diff.pow(2), 
                          huber_delta * (diff - 0.5 * huber_delta))
    elif loss_type == 'smooth_l1':
        # Smooth L1 loss (equivalent to Huber with delta=1.0)
        loss = F.smooth_l1_loss(predicted_peaks, target_peaks_clean, reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Supported: 'mae', 'mse', 'huber', 'smooth_l1'")
    
    # Apply mask to exclude invalid peaks
    masked_loss = loss * valid_mask.float()
    
    # Sum over valid peaks and average
    total_loss = torch.sum(masked_loss)
    
    return total_loss / num_valid_peaks


def time_alignment_loss(
    predicted_signal: torch.Tensor,
    target_signal: torch.Tensor,
    alignment_type: str = 'warped_mse',
    max_warp: int = 10,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute time alignment loss to encourage temporal peak alignment.
    
    This function helps align temporal features between predicted and target signals,
    which is crucial for ABR signals where peak timing is important but may have
    slight temporal variations.
    
    Args:
        predicted_signal (torch.Tensor): Predicted signal [batch, signal_length]
        target_signal (torch.Tensor): Target signal [batch, signal_length]
        alignment_type (str): Type of alignment ('warped_mse', 'soft_dtw', 'correlation')
        max_warp (int): Maximum warping distance for alignment (default=10)
        temperature (float): Temperature for soft DTW (default=1.0)
    
    Returns:
        torch.Tensor: Time alignment loss
    """
    batch_size, signal_length = predicted_signal.shape
    device = predicted_signal.device
    
    if alignment_type == 'warped_mse':
        # Warped MSE: allow small temporal shifts to find best alignment
        min_loss = torch.full((batch_size,), float('inf'), device=device)
        
        for shift in range(-max_warp, max_warp + 1):
            if shift == 0:
                # No shift
                pred_shifted = predicted_signal
                target_shifted = target_signal
            elif shift > 0:
                # Positive shift: pad beginning of predicted, crop end
                pred_shifted = F.pad(predicted_signal[:, :-shift], (shift, 0))
                target_shifted = target_signal
            else:
                # Negative shift: crop beginning of predicted, pad end
                pred_shifted = F.pad(predicted_signal[:, -shift:], (0, -shift))
                target_shifted = target_signal
            
            # Compute MSE for this shift
            mse = torch.mean((pred_shifted - target_shifted).pow(2), dim=1)
            min_loss = torch.minimum(min_loss, mse)
        
        return torch.mean(min_loss)
    
    elif alignment_type == 'soft_dtw':
        # Simplified soft DTW implementation
        # Note: This is a basic approximation - full soft DTW would require more complex implementation
        
        # Compute pairwise distances
        pred_expanded = predicted_signal.unsqueeze(2)  # [batch, signal_length, 1]
        target_expanded = target_signal.unsqueeze(1)   # [batch, 1, signal_length]
        distances = (pred_expanded - target_expanded).pow(2)  # [batch, signal_length, signal_length]
        
        # Apply temperature scaling
        distances = distances / temperature
        
        # Simple path-based alignment (not full DTW but captures alignment idea)
        # Use diagonal path with small deviations
        alignment_loss = 0.0
        for i in range(signal_length):
            # Consider alignment around diagonal
            start_j = max(0, i - max_warp)
            end_j = min(signal_length, i + max_warp + 1)
            
            # Softmin over possible alignments
            alignment_costs = distances[:, i, start_j:end_j]
            soft_cost = -temperature * torch.logsumexp(-alignment_costs / temperature, dim=1)
            alignment_loss += torch.mean(soft_cost)
        
        return alignment_loss / signal_length
    
    elif alignment_type == 'correlation':
        # Cross-correlation based alignment
        # Normalize signals to zero mean, unit variance
        pred_norm = (predicted_signal - predicted_signal.mean(dim=1, keepdim=True))
        pred_norm = pred_norm / (pred_norm.std(dim=1, keepdim=True) + 1e-8)
        
        target_norm = (target_signal - target_signal.mean(dim=1, keepdim=True))
        target_norm = target_norm / (target_norm.std(dim=1, keepdim=True) + 1e-8)
        
        # Compute cross-correlation using 1D convolution
        # Reshape for conv1d: target as input, pred as kernel
        target_reshaped = target_norm.unsqueeze(1)  # [batch, 1, signal_length]
        pred_kernel = pred_norm.flip(dims=[1]).unsqueeze(0).unsqueeze(0)  # [1, 1, signal_length]
        
        # For batch processing, we need to compute correlation for each sample separately
        correlations = []
        for i in range(batch_size):
            corr = F.conv1d(
                target_reshaped[i:i+1],  # [1, 1, signal_length]
                pred_kernel,  # [1, 1, signal_length]
                padding=signal_length - 1
            )
            correlations.append(corr)
        
        correlation = torch.cat(correlations, dim=0).squeeze(1)  # [batch, 2*signal_length - 1]
        
        # Find maximum correlation (best alignment)
        max_correlation = torch.max(correlation, dim=1)[0]
        
        # Convert to loss (negative correlation)
        return torch.mean(-max_correlation)
    
    else:
        raise ValueError(f"Unknown alignment type: {alignment_type}. Supported: 'warped_mse', 'soft_dtw', 'correlation'")


def kl_annealing(
    epoch: int, 
    max_beta: float = 1.0, 
    warmup_epochs: int = 10
) -> float:
    """
    Return a gradually increasing beta value for KL divergence annealing.
    
    KL annealing is a technique to prevent KL collapse early in training by
    gradually increasing the weight of the KL divergence term. This allows
    the model to first learn good reconstructions before enforcing the
    latent space regularization, leading to more stable training and better
    final performance.
    
    Args:
        epoch (int): Current training epoch (0-indexed)
        max_beta (float): Maximum beta value to reach after warmup (default=1.0)
        warmup_epochs (int): Number of epochs over which to linearly increase beta (default=10)
    
    Returns:
        float: Beta value for current epoch, linearly increasing from 0 to max_beta
    """
    if epoch >= warmup_epochs:
        return max_beta
    else:
        # Linear annealing from 0 to max_beta over warmup_epochs
        beta = (epoch / warmup_epochs) * max_beta
        return min(beta, max_beta)  # Clamp to not exceed max_beta


def enhanced_cvae_loss(
    recon_signal: torch.Tensor,
    target_signal: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    predicted_peaks: Optional[torch.Tensor] = None,
    target_peaks: Optional[torch.Tensor] = None,
    peak_mask: Optional[torch.Tensor] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    loss_config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Enhanced CVAE loss with configurable components and detailed logging.
    
    This function provides a flexible loss computation with multiple configurable
    components for comprehensive training and analysis.
    
    Args:
        recon_signal (torch.Tensor): Reconstructed signal [batch, signal_length]
        target_signal (torch.Tensor): Target signal [batch, signal_length]
        mu (torch.Tensor): Mean of latent distribution [batch, latent_dim]
        logvar (torch.Tensor): Log variance of latent distribution [batch, latent_dim]
        predicted_peaks (torch.Tensor, optional): Predicted peaks [batch, num_peaks]
        target_peaks (torch.Tensor, optional): Target peaks [batch, num_peaks]
        peak_mask (torch.Tensor, optional): Peak validity mask [batch, num_peaks]
        loss_weights (Dict[str, float], optional): Weights for different loss components
        loss_config (Dict[str, Any], optional): Configuration for loss computation
    
    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            - total_loss: Combined weighted loss
            - loss_components: Dictionary with individual loss components
    """
    # Default weights
    default_weights = {
        'reconstruction': 1.0,
        'kl': 1.0,
        'peak': 1.0,
        'alignment': 0.1
    }
    weights = {**default_weights, **(loss_weights or {})}
    
    # Default configuration
    default_config = {
        'peak_loss_type': 'mae',
        'huber_delta': 1.0,
        'alignment_type': 'warped_mse',
        'max_warp': 10,
        'temperature': 1.0,
        'use_alignment_loss': False
    }
    config = {**default_config, **(loss_config or {})}
    
    # Initialize loss components
    loss_components = {}
    
    # 1. Reconstruction loss
    recon_loss = F.mse_loss(recon_signal, target_signal, reduction='mean')
    loss_components['reconstruction'] = recon_loss
    
    # 2. KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)
    loss_components['kl'] = kl_loss
    
    # 3. Peak loss (if peaks are provided)
    if predicted_peaks is not None and target_peaks is not None:
        if peak_mask is None:
            # Create default mask (all peaks valid)
            peak_mask = torch.ones_like(target_peaks, dtype=torch.bool)
        
        peak_loss_val = peak_loss(
            predicted_peaks, 
            target_peaks, 
            peak_mask,
            loss_type=config['peak_loss_type'],
            huber_delta=config['huber_delta']
        )
        loss_components['peak'] = peak_loss_val
    else:
        loss_components['peak'] = torch.tensor(0.0, device=recon_signal.device)
    
    # 4. Time alignment loss (optional)
    if config['use_alignment_loss']:
        alignment_loss = time_alignment_loss(
            recon_signal,
            target_signal,
            alignment_type=config['alignment_type'],
            max_warp=config['max_warp'],
            temperature=config['temperature']
        )
        loss_components['alignment'] = alignment_loss
    else:
        loss_components['alignment'] = torch.tensor(0.0, device=recon_signal.device)
    
    # Compute total weighted loss
    total_loss = (
        weights['reconstruction'] * loss_components['reconstruction'] +
        weights['kl'] * loss_components['kl'] +
        weights['peak'] * loss_components['peak'] +
        weights['alignment'] * loss_components['alignment']
    )
    
    loss_components['total'] = total_loss
    
    return total_loss, loss_components


def hierarchical_kl_loss(
    mu_global: torch.Tensor,
    logvar_global: torch.Tensor,
    mu_local: torch.Tensor,
    logvar_local: torch.Tensor,
    global_kl_weight: float = 1.0,
    local_kl_weight: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute KL divergence loss for both global and local latent spaces.
    
    Args:
        mu_global (torch.Tensor): Global latent mean [batch, global_latent_dim]
        logvar_global (torch.Tensor): Global latent log variance [batch, global_latent_dim]
        mu_local (torch.Tensor): Local latent mean [batch, local_latent_dim]
        logvar_local (torch.Tensor): Local latent log variance [batch, local_latent_dim]
        global_kl_weight (float): Weight for global KL loss
        local_kl_weight (float): Weight for local KL loss
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_kl_loss: Combined weighted KL loss
            - global_kl_loss: Global KL divergence component
            - local_kl_loss: Local KL divergence component
    """
    # Global KL divergence: KL(q(z_global|x) || p(z_global))
    global_kl_loss = -0.5 * torch.sum(1 + logvar_global - mu_global.pow(2) - logvar_global.exp(), dim=1)
    global_kl_loss = torch.mean(global_kl_loss)
    
    # Local KL divergence: KL(q(z_local|x) || p(z_local))
    local_kl_loss = -0.5 * torch.sum(1 + logvar_local - mu_local.pow(2) - logvar_local.exp(), dim=1)
    local_kl_loss = torch.mean(local_kl_loss)
    
    # Total weighted KL loss
    total_kl_loss = global_kl_weight * global_kl_loss + local_kl_weight * local_kl_loss
    
    return total_kl_loss, global_kl_loss, local_kl_loss


def hierarchical_cvae_loss(
    recon_signal: torch.Tensor,
    target_signal: torch.Tensor,
    recon_static_params: torch.Tensor,
    target_static_params: torch.Tensor,
    mu_global: torch.Tensor,
    logvar_global: torch.Tensor,
    mu_local: torch.Tensor,
    logvar_local: torch.Tensor,
    global_kl_weight: float = 1.0,
    local_kl_weight: float = 1.0,
    static_loss_weight: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the full loss for hierarchical CVAE with global and local latents.
    
    Args:
        recon_signal (torch.Tensor): Reconstructed signal [batch, signal_length]
        target_signal (torch.Tensor): Original signal [batch, signal_length]
        recon_static_params (torch.Tensor): Reconstructed static parameters [batch, static_dim]
        target_static_params (torch.Tensor): Original static parameters [batch, static_dim]
        mu_global (torch.Tensor): Global latent mean [batch, global_latent_dim]
        logvar_global (torch.Tensor): Global latent log variance [batch, global_latent_dim]
        mu_local (torch.Tensor): Local latent mean [batch, local_latent_dim]
        logvar_local (torch.Tensor): Local latent log variance [batch, local_latent_dim]
        global_kl_weight (float): Weight for global KL divergence
        local_kl_weight (float): Weight for local KL divergence
        static_loss_weight (float): Weight for static parameters loss
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_loss: Combined signal + static + KL loss
            - signal_recon_loss: Signal reconstruction loss component
            - static_recon_loss: Static parameters reconstruction loss component
            - global_kl_loss: Global KL divergence loss component
            - local_kl_loss: Local KL divergence loss component
    """
    # Signal reconstruction loss
    signal_recon_loss = F.mse_loss(recon_signal, target_signal, reduction='mean')
    
    # Static parameters reconstruction loss
    static_recon_loss = static_params_loss(recon_static_params, target_static_params)
    
    # Hierarchical KL divergence loss
    total_kl_loss, global_kl_loss, local_kl_loss = hierarchical_kl_loss(
        mu_global, logvar_global, mu_local, logvar_local,
        global_kl_weight, local_kl_weight
    )
    
    # Total loss
    total_loss = signal_recon_loss + static_loss_weight * static_recon_loss + total_kl_loss
    
    return total_loss, signal_recon_loss, static_recon_loss, global_kl_loss, local_kl_loss


def enhanced_hierarchical_cvae_loss(
    recon_signal: torch.Tensor,
    target_signal: torch.Tensor,
    recon_static_params: torch.Tensor,
    target_static_params: torch.Tensor,
    mu_global: torch.Tensor,
    logvar_global: torch.Tensor,
    mu_local: torch.Tensor,
    logvar_local: torch.Tensor,
    predicted_peaks: Optional[torch.Tensor] = None,
    target_peaks: Optional[torch.Tensor] = None,
    peak_mask: Optional[torch.Tensor] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    loss_config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Enhanced hierarchical CVAE loss with configurable components and detailed logging.
    
    Args:
        recon_signal (torch.Tensor): Reconstructed signal [batch, signal_length]
        target_signal (torch.Tensor): Target signal [batch, signal_length]
        recon_static_params (torch.Tensor): Reconstructed static parameters [batch, static_dim]
        target_static_params (torch.Tensor): Target static parameters [batch, static_dim]
        mu_global (torch.Tensor): Global latent mean [batch, global_latent_dim]
        logvar_global (torch.Tensor): Global latent log variance [batch, global_latent_dim]
        mu_local (torch.Tensor): Local latent mean [batch, local_latent_dim]
        logvar_local (torch.Tensor): Local latent log variance [batch, local_latent_dim]
        predicted_peaks (torch.Tensor, optional): Predicted peaks [batch, num_peaks]
        target_peaks (torch.Tensor, optional): Target peaks [batch, num_peaks]
        peak_mask (torch.Tensor, optional): Peak validity mask [batch, num_peaks]
        loss_weights (Dict[str, float], optional): Weights for different loss components
        loss_config (Dict[str, Any], optional): Configuration for loss computation
    
    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            - total_loss: Combined weighted loss
            - loss_components: Dictionary with individual loss components
    """
    # Default weights for hierarchical model
    default_weights = {
        'reconstruction': 1.0,
        'static': 1.0,
        'global_kl': 1.0,
        'local_kl': 1.0,
        'peak': 1.0,
        'alignment': 0.1
    }
    weights = {**default_weights, **(loss_weights or {})}
    
    # Default configuration
    default_config = {
        'peak_loss_type': 'mae',
        'huber_delta': 1.0,
        'alignment_type': 'warped_mse',
        'max_warp': 10,
        'temperature': 1.0,
        'use_alignment_loss': False
    }
    config = {**default_config, **(loss_config or {})}
    
    # Initialize loss components
    loss_components = {}
    
    # 1. Reconstruction loss
    recon_loss = F.mse_loss(recon_signal, target_signal, reduction='mean')
    loss_components['reconstruction'] = recon_loss
    
    # 2. Static parameters loss
    static_loss = static_params_loss(recon_static_params, target_static_params)
    loss_components['static'] = static_loss
    
    # 3. Hierarchical KL divergence losses
    total_kl_loss, global_kl_loss, local_kl_loss = hierarchical_kl_loss(
        mu_global, logvar_global, mu_local, logvar_local,
        weights['global_kl'], weights['local_kl']
    )
    loss_components['global_kl'] = global_kl_loss
    loss_components['local_kl'] = local_kl_loss
    loss_components['total_kl'] = total_kl_loss
    
    # 4. Peak loss (if peaks are provided)
    if predicted_peaks is not None and target_peaks is not None:
        if peak_mask is None:
            peak_mask = torch.ones_like(target_peaks, dtype=torch.bool)
        
        peak_loss_val = peak_loss(
            predicted_peaks, 
            target_peaks, 
            peak_mask,
            loss_type=config['peak_loss_type'],
            huber_delta=config['huber_delta']
        )
        loss_components['peak'] = peak_loss_val
    else:
        loss_components['peak'] = torch.tensor(0.0, device=recon_signal.device)
    
    # 5. Time alignment loss (optional)
    if config['use_alignment_loss']:
        alignment_loss = time_alignment_loss(
            recon_signal,
            target_signal,
            alignment_type=config['alignment_type'],
            max_warp=config['max_warp'],
            temperature=config['temperature']
        )
        loss_components['alignment'] = alignment_loss
    else:
        loss_components['alignment'] = torch.tensor(0.0, device=recon_signal.device)
    
    # Compute total weighted loss
    total_loss = (
        weights['reconstruction'] * loss_components['reconstruction'] +
        weights['static'] * loss_components['static'] +
        total_kl_loss +  # Already weighted inside hierarchical_kl_loss
        weights['peak'] * loss_components['peak'] +
        weights['alignment'] * loss_components['alignment']
    )
    
    loss_components['total'] = total_loss
    
    return total_loss, loss_components


def static_reconstruction_loss(
    recon_static: torch.Tensor, 
    target_static: torch.Tensor
) -> torch.Tensor:
    """
    Calculate MSE loss between reconstructed static parameters and true static parameters.
    
    This loss encourages the latent space to capture information about static parameters,
    enabling causal regularization between latents and static variables.
    
    Args:
        recon_static (torch.Tensor): Reconstructed static parameters [batch, static_dim]
        target_static (torch.Tensor): True static parameters [batch, static_dim]
    
    Returns:
        torch.Tensor: MSE loss between reconstructed and true static parameters
    """
    return F.mse_loss(recon_static, target_static, reduction='mean')


def infonce_contrastive_loss(
    z: torch.Tensor,
    static_params: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE-style contrastive loss for latent-static alignment.
    
    For each latent vector z[i], the positive pair is static_params[i] (same sample),
    and negatives are static_params[j] where j != i (other samples in batch).
    
    This encourages the latent space to be predictive of static parameters while
    maintaining separation between different samples.
    
    Args:
        z (torch.Tensor): Latent vectors [batch, latent_dim]
        static_params (torch.Tensor): Static parameters [batch, static_dim]
        temperature (float): Temperature parameter for softmax (default=0.07)
    
    Returns:
        torch.Tensor: InfoNCE contrastive loss
    """
    batch_size = z.size(0)
    
    if batch_size < 2:
        # Need at least 2 samples for contrastive learning
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    
    # Normalize features
    z_norm = F.normalize(z, dim=1)
    static_norm = F.normalize(static_params, dim=1)
    
    # For InfoNCE, we need to compute cross-modal similarities
    # Project both to a common space or use a learned projection
    # For now, we'll use the smaller dimension and project accordingly
    z_dim = z.size(1)
    static_dim = static_params.size(1)
    
    if z_dim != static_dim:
        # Project static to latent space using a simple linear transformation
        # This should ideally be learned, but for the loss function we'll use identity mapping on common dims
        min_dim = min(z_dim, static_dim)
        z_proj = z_norm[:, :min_dim]
        static_proj = static_norm[:, :min_dim]
    else:
        z_proj = z_norm
        static_proj = static_norm
    
    # Compute similarity matrix: [batch, batch]
    # similarity[i,j] = cosine similarity between z[i] and static_params[j]
    similarity_matrix = torch.matmul(z_proj, static_proj.T) / temperature
    
    # Create labels for positive pairs (diagonal elements)
    labels = torch.arange(batch_size, device=z.device)
    
    # InfoNCE loss using cross-entropy
    # For each row i, we want the maximum similarity at column i (positive pair)
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss


def hierarchical_static_reconstruction_loss(
    recon_static_global: torch.Tensor,
    recon_static_local: torch.Tensor,
    recon_static_combined: torch.Tensor,
    target_static: torch.Tensor,
    global_weight: float = 0.3,
    local_weight: float = 0.3,
    combined_weight: float = 0.4
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Calculate hierarchical static reconstruction loss for HierarchicalCVAE.
    
    Computes separate losses for global, local, and combined static reconstructions,
    encouraging different latent levels to capture different aspects of static parameters.
    
    Args:
        recon_static_global (torch.Tensor): Global static reconstruction [batch, static_dim//2]
        recon_static_local (torch.Tensor): Local static reconstruction [batch, static_dim//2]
        recon_static_combined (torch.Tensor): Combined static reconstruction [batch, static_dim]
        target_static (torch.Tensor): True static parameters [batch, static_dim]
        global_weight (float): Weight for global reconstruction loss
        local_weight (float): Weight for local reconstruction loss
        combined_weight (float): Weight for combined reconstruction loss
    
    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and component losses
    """
    # Split target static into global and local parts for partial reconstructions
    static_dim = target_static.size(1)
    target_global = target_static[:, :static_dim//2]
    target_local = target_static[:, static_dim//2:]
    
    # Calculate component losses
    global_loss = F.mse_loss(recon_static_global, target_global, reduction='mean')
    local_loss = F.mse_loss(recon_static_local, target_local, reduction='mean')
    combined_loss = F.mse_loss(recon_static_combined, target_static, reduction='mean')
    
    # Weighted total loss
    total_loss = (global_weight * global_loss + 
                  local_weight * local_loss + 
                  combined_weight * combined_loss)
    
    # Component losses for logging
    components = {
        'static_global_loss': global_loss,
        'static_local_loss': local_loss,
        'static_combined_loss': combined_loss
    }
    
    return total_loss, components


def hierarchical_infonce_loss(
    z_global: torch.Tensor,
    z_local: torch.Tensor,
    static_params: torch.Tensor,
    temperature: float = 0.07,
    global_weight: float = 0.5,
    local_weight: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Hierarchical InfoNCE contrastive loss for global and local latents.
    
    Applies InfoNCE loss separately to global and local latents with static parameters,
    encouraging both latent levels to be predictive of static information.
    
    Args:
        z_global (torch.Tensor): Global latent vectors [batch, global_latent_dim]
        z_local (torch.Tensor): Local latent vectors [batch, local_latent_dim]
        static_params (torch.Tensor): Static parameters [batch, static_dim]
        temperature (float): Temperature parameter for softmax
        global_weight (float): Weight for global InfoNCE loss
        local_weight (float): Weight for local InfoNCE loss
    
    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and component losses
    """
    # Calculate InfoNCE for global and local latents
    global_infonce = infonce_contrastive_loss(z_global, static_params, temperature)
    local_infonce = infonce_contrastive_loss(z_local, static_params, temperature)
    
    # Weighted total loss
    total_loss = global_weight * global_infonce + local_weight * local_infonce
    
    # Component losses for logging
    components = {
        'infonce_global_loss': global_infonce,
        'infonce_local_loss': local_infonce
    }
    
    return total_loss, components


def enhanced_cvae_loss_with_static_regularization(
    recon_signal: torch.Tensor,
    target_signal: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    recon_static_from_z: torch.Tensor,
    target_static: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    predicted_peaks: Optional[torch.Tensor] = None,
    target_peaks: Optional[torch.Tensor] = None,
    peak_mask: Optional[torch.Tensor] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    loss_config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Enhanced CVAE loss with static regularization for encouraging latent-static alignment.
    
    Combines standard CVAE loss with static reconstruction and optionally InfoNCE contrastive loss.
    
    Args:
        recon_signal (torch.Tensor): Reconstructed signal [batch, signal_length]
        target_signal (torch.Tensor): Target signal [batch, signal_length]
        mu (torch.Tensor): Latent mean [batch, latent_dim]
        logvar (torch.Tensor): Latent log variance [batch, latent_dim]
        recon_static_from_z (torch.Tensor): Static reconstructed from z [batch, static_dim]
        target_static (torch.Tensor): Target static parameters [batch, static_dim]
        z (torch.Tensor, optional): Latent samples for InfoNCE [batch, latent_dim]
        predicted_peaks (torch.Tensor, optional): Predicted peaks [batch, num_peaks]
        target_peaks (torch.Tensor, optional): Target peaks [batch, num_peaks]
        peak_mask (torch.Tensor, optional): Peak validity mask [batch, num_peaks]
        loss_weights (Dict[str, float], optional): Weights for different loss components
        loss_config (Dict[str, Any], optional): Configuration for loss computation
    
    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and component losses
    """
    # Default loss weights
    default_weights = {
        'reconstruction': 1.0,
        'kl_divergence': 1.0,
        'peak_loss': 1.0,
        'alignment_loss': 0.1,
        'static_reconstruction': 0.5,
        'infonce_contrastive': 0.1
    }
    
    # Default loss configuration
    default_config = {
        'peak_loss_type': 'mae',
        'alignment_loss_type': 'warped_mse',
        'use_infonce': True,
        'infonce_temperature': 0.07
    }
    
    if loss_weights is None:
        loss_weights = default_weights
    else:
        loss_weights = {**default_weights, **loss_weights}
    
    if loss_config is None:
        loss_config = default_config
    else:
        loss_config = {**default_config, **loss_config}
    
    # Initialize loss components
    loss_components = {}
    
    # 1. Reconstruction loss
    recon_loss = F.mse_loss(recon_signal, target_signal, reduction='mean')
    loss_components['reconstruction_loss'] = recon_loss
    
    # 2. KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)
    loss_components['kl_loss'] = kl_loss
    
    # 3. Static reconstruction loss
    static_recon_loss = static_reconstruction_loss(recon_static_from_z, target_static)
    loss_components['static_reconstruction_loss'] = static_recon_loss
    
    # 4. InfoNCE contrastive loss (if enabled and z is provided)
    infonce_loss = torch.tensor(0.0, device=recon_signal.device)
    if loss_config['use_infonce'] and z is not None:
        infonce_loss = infonce_contrastive_loss(
            z, target_static, temperature=loss_config['infonce_temperature']
        )
    loss_components['infonce_loss'] = infonce_loss
    
    # 5. Peak loss (if peaks are provided)
    peak_loss_value = torch.tensor(0.0, device=recon_signal.device)
    if predicted_peaks is not None and target_peaks is not None:
        peak_loss_value = peak_loss(
            predicted_peaks, target_peaks, peak_mask, 
            loss_type=loss_config['peak_loss_type']
        )
    loss_components['peak_loss'] = peak_loss_value
    
    # 6. Time alignment loss (if enabled)
    alignment_loss_value = torch.tensor(0.0, device=recon_signal.device)
    if loss_weights['alignment_loss'] > 0:
        alignment_loss_value = time_alignment_loss(
            recon_signal, target_signal, alignment_type=loss_config['alignment_loss_type']
        )
    loss_components['alignment_loss'] = alignment_loss_value
    
    # Compute total loss
    total_loss = (
        loss_weights['reconstruction'] * recon_loss +
        loss_weights['kl_divergence'] * kl_loss +
        loss_weights['static_reconstruction'] * static_recon_loss +
        loss_weights['infonce_contrastive'] * infonce_loss +
        loss_weights['peak_loss'] * peak_loss_value +
        loss_weights['alignment_loss'] * alignment_loss_value
    )
    
    loss_components['total_loss'] = total_loss
    
    return total_loss, loss_components


def enhanced_hierarchical_cvae_loss_with_static_regularization(
    recon_signal: torch.Tensor,
    target_signal: torch.Tensor,
    recon_static_params: torch.Tensor,
    target_static_params: torch.Tensor,
    mu_global: torch.Tensor,
    logvar_global: torch.Tensor,
    mu_local: torch.Tensor,
    logvar_local: torch.Tensor,
    recon_static_from_z: torch.Tensor,
    z_global: Optional[torch.Tensor] = None,
    z_local: Optional[torch.Tensor] = None,
    recon_static_global: Optional[torch.Tensor] = None,
    recon_static_local: Optional[torch.Tensor] = None,
    predicted_peaks: Optional[torch.Tensor] = None,
    target_peaks: Optional[torch.Tensor] = None,
    peak_mask: Optional[torch.Tensor] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    loss_config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Enhanced hierarchical CVAE loss with comprehensive static regularization.
    
    Includes both hierarchical static reconstruction and hierarchical InfoNCE losses
    for better latent-static alignment at multiple levels.
    
    Args:
        recon_signal (torch.Tensor): Reconstructed signal [batch, signal_length]
        target_signal (torch.Tensor): Target signal [batch, signal_length]
        recon_static_params (torch.Tensor): Reconstructed static parameters [batch, static_dim]
        target_static_params (torch.Tensor): Target static parameters [batch, static_dim]
        mu_global (torch.Tensor): Global latent mean [batch, global_latent_dim]
        logvar_global (torch.Tensor): Global latent log variance [batch, global_latent_dim]
        mu_local (torch.Tensor): Local latent mean [batch, local_latent_dim]
        logvar_local (torch.Tensor): Local latent log variance [batch, local_latent_dim]
        recon_static_from_z (torch.Tensor): Combined static reconstruction [batch, static_dim]
        z_global (torch.Tensor, optional): Global latent samples [batch, global_latent_dim]
        z_local (torch.Tensor, optional): Local latent samples [batch, local_latent_dim]
        recon_static_global (torch.Tensor, optional): Global static reconstruction [batch, static_dim//2]
        recon_static_local (torch.Tensor, optional): Local static reconstruction [batch, static_dim//2]
        predicted_peaks (torch.Tensor, optional): Predicted peaks [batch, num_peaks]
        target_peaks (torch.Tensor, optional): Target peaks [batch, num_peaks]
        peak_mask (torch.Tensor, optional): Peak validity mask [batch, num_peaks]
        loss_weights (Dict[str, float], optional): Weights for different loss components
        loss_config (Dict[str, Any], optional): Configuration for loss computation
    
    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and component losses
    """
    # Default loss weights
    default_weights = {
        'reconstruction': 1.0,
        'static_params_reconstruction': 1.0,
        'global_kl': 1.0,
        'local_kl': 1.0,
        'peak_loss': 1.0,
        'alignment_loss': 0.1,
        'static_regularization': 0.5,
        'hierarchical_infonce': 0.1
    }
    
    # Default loss configuration
    default_config = {
        'peak_loss_type': 'mae',
        'alignment_loss_type': 'warped_mse',
        'use_hierarchical_static_loss': True,
        'use_hierarchical_infonce': True,
        'infonce_temperature': 0.07,
        'static_loss_weights': {'global': 0.3, 'local': 0.3, 'combined': 0.4},
        'infonce_weights': {'global': 0.5, 'local': 0.5}
    }
    
    if loss_weights is None:
        loss_weights = default_weights
    else:
        loss_weights = {**default_weights, **loss_weights}
    
    if loss_config is None:
        loss_config = default_config
    else:
        loss_config = {**default_config, **loss_config}
    
    # Initialize loss components
    loss_components = {}
    
    # 1. Signal reconstruction loss
    signal_recon_loss = F.mse_loss(recon_signal, target_signal, reduction='mean')
    loss_components['signal_reconstruction_loss'] = signal_recon_loss
    
    # 2. Static parameters reconstruction loss (decoder output)
    static_params_recon_loss = static_params_loss(recon_static_params, target_static_params)
    loss_components['static_params_reconstruction_loss'] = static_params_recon_loss
    
    # 3. Hierarchical KL divergence losses
    total_kl_loss, global_kl_loss, local_kl_loss = hierarchical_kl_loss(
        mu_global, logvar_global, mu_local, logvar_local,
        loss_weights['global_kl'], loss_weights['local_kl']
    )
    loss_components['global_kl_loss'] = global_kl_loss
    loss_components['local_kl_loss'] = local_kl_loss
    loss_components['total_kl_loss'] = total_kl_loss
    
    # 4. Static regularization losses
    static_reg_loss = torch.tensor(0.0, device=recon_signal.device)
    if loss_config['use_hierarchical_static_loss'] and recon_static_global is not None and recon_static_local is not None:
        # Hierarchical static reconstruction loss
        static_reg_loss, static_components = hierarchical_static_reconstruction_loss(
            recon_static_global, recon_static_local, recon_static_from_z, target_static_params,
            global_weight=loss_config['static_loss_weights']['global'],
            local_weight=loss_config['static_loss_weights']['local'],
            combined_weight=loss_config['static_loss_weights']['combined']
        )
        loss_components.update(static_components)
    else:
        # Simple static reconstruction loss
        static_reg_loss = static_reconstruction_loss(recon_static_from_z, target_static_params)
        loss_components['static_reconstruction_loss'] = static_reg_loss
    
    # 5. Hierarchical InfoNCE loss
    hierarchical_infonce_loss_value = torch.tensor(0.0, device=recon_signal.device)
    if (loss_config['use_hierarchical_infonce'] and 
        z_global is not None and z_local is not None):
        hierarchical_infonce_loss_value, infonce_components = hierarchical_infonce_loss(
            z_global, z_local, target_static_params,
            temperature=loss_config['infonce_temperature'],
            global_weight=loss_config['infonce_weights']['global'],
            local_weight=loss_config['infonce_weights']['local']
        )
        loss_components.update(infonce_components)
    loss_components['hierarchical_infonce_loss'] = hierarchical_infonce_loss_value
    
    # 6. Peak loss (if peaks are provided)
    peak_loss_value = torch.tensor(0.0, device=recon_signal.device)
    if predicted_peaks is not None and target_peaks is not None:
        peak_loss_value = peak_loss(
            predicted_peaks, target_peaks, peak_mask,
            loss_type=loss_config['peak_loss_type']
        )
    loss_components['peak_loss'] = peak_loss_value
    
    # 7. Time alignment loss (if enabled)
    alignment_loss_value = torch.tensor(0.0, device=recon_signal.device)
    if loss_weights['alignment_loss'] > 0:
        alignment_loss_value = time_alignment_loss(
            recon_signal, target_signal, alignment_type=loss_config['alignment_loss_type']
        )
    loss_components['alignment_loss'] = alignment_loss_value
    
    # Compute total loss
    total_loss = (
        loss_weights['reconstruction'] * signal_recon_loss +
        loss_weights['static_params_reconstruction'] * static_params_recon_loss +
        loss_weights['global_kl'] * global_kl_loss +
        loss_weights['local_kl'] * local_kl_loss +
        loss_weights['static_regularization'] * static_reg_loss +
        loss_weights['hierarchical_infonce'] * hierarchical_infonce_loss_value +
        loss_weights['peak_loss'] * peak_loss_value +
        loss_weights['alignment_loss'] * alignment_loss_value
    )
    
    loss_components['total_loss'] = total_loss
    
    return total_loss, loss_components
