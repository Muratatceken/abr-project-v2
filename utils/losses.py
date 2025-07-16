import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


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
    peak_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute masked MSE loss over peak values for partial supervision.
    
    This function enables training with partially labeled peak data, where some
    peak values may be missing (NaN) in the target. The peak_mask indicates
    which peak values are valid and should contribute to the loss calculation.
    This is particularly useful for ABR data where peak detection may fail
    for some samples or certain peaks may be absent due to hearing conditions.
    
    Args:
        predicted_peaks (torch.Tensor): Predicted peak values [batch, num_peaks]
        target_peaks (torch.Tensor): Target peak values [batch, num_peaks] (may contain NaN)
        peak_mask (torch.Tensor): Boolean mask indicating valid peaks [batch, num_peaks]
    
    Returns:
        torch.Tensor: Masked MSE loss averaged over valid peaks only
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
    
    # Calculate squared differences
    mse = (predicted_peaks - target_peaks_clean).pow(2)
    
    # Apply mask to exclude invalid peaks
    masked_mse = mse * valid_mask.float()
    
    # Sum over valid peaks and average
    total_loss = torch.sum(masked_mse)
    
    return total_loss / num_valid_peaks


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
