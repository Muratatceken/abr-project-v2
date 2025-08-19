"""
Simple metrics for training monitoring and evaluation.

Basic time-domain metrics for signal reconstruction quality.
"""

import torch
import torch.nn.functional as F


def l1_time(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Time-domain L1 (MAE) loss.
    
    Args:
        x_hat: Predicted signal [B, ...]
        x: Target signal [B, ...]
        
    Returns:
        L1 loss scalar
    """
    return F.l1_loss(x_hat, x)


def mse_time(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Time-domain MSE loss.
    
    Args:
        x_hat: Predicted signal [B, ...]
        x: Target signal [B, ...]
        
    Returns:
        MSE loss scalar
    """
    return F.mse_loss(x_hat, x)


def rmse_time(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Time-domain RMSE.
    
    Args:
        x_hat: Predicted signal [B, ...]
        x: Target signal [B, ...]
        
    Returns:
        RMSE scalar
    """
    return torch.sqrt(F.mse_loss(x_hat, x))


def snr_db(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Signal-to-Noise Ratio in dB.
    
    Args:
        x_hat: Predicted signal [B, ...]
        x: Target signal [B, ...]
        eps: Small epsilon to avoid log(0)
        
    Returns:
        SNR in dB
    """
    signal_power = torch.mean(x ** 2)
    noise_power = torch.mean((x_hat - x) ** 2)
    
    snr = 10 * torch.log10(signal_power / (noise_power + eps))
    return snr


def pearson_correlation(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson correlation coefficient between predicted and target signals.
    
    Args:
        x_hat: Predicted signal [B, ...]
        x: Target signal [B, ...]
        eps: Small epsilon for numerical stability
        
    Returns:
        Pearson correlation coefficient
    """
    # Flatten signals
    x_hat_flat = x_hat.view(-1)
    x_flat = x.view(-1)
    
    # Center the signals
    x_hat_mean = torch.mean(x_hat_flat)
    x_mean = torch.mean(x_flat)
    
    x_hat_centered = x_hat_flat - x_hat_mean
    x_centered = x_flat - x_mean
    
    # Compute correlation
    numerator = torch.sum(x_hat_centered * x_centered)
    denominator = torch.sqrt(torch.sum(x_hat_centered ** 2) * torch.sum(x_centered ** 2))
    
    correlation = numerator / (denominator + eps)
    return correlation


def dynamic_range(x: torch.Tensor) -> torch.Tensor:
    """
    Dynamic range of signal (max - min).
    
    Args:
        x: Signal tensor [B, ...]
        
    Returns:
        Dynamic range scalar
    """
    return torch.max(x) - torch.min(x)


def rms(x: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Square of signal.
    
    Args:
        x: Signal tensor [B, ...]
        
    Returns:
        RMS scalar
    """
    return torch.sqrt(torch.mean(x ** 2))


def peak_signal_amplitude(x: torch.Tensor) -> torch.Tensor:
    """
    Peak signal amplitude (maximum absolute value).
    
    Args:
        x: Signal tensor [B, ...]
        
    Returns:
        Peak amplitude scalar
    """
    return torch.max(torch.abs(x))


def compute_basic_metrics(x_hat: torch.Tensor, x: torch.Tensor) -> dict:
    """
    Compute a set of basic signal metrics.
    
    Args:
        x_hat: Predicted signal [B, ...]
        x: Target signal [B, ...]
        
    Returns:
        Dictionary of metric values
    """
    metrics = {
        'l1': l1_time(x_hat, x).item(),
        'mse': mse_time(x_hat, x).item(),
        'rmse': rmse_time(x_hat, x).item(),
        'snr_db': snr_db(x_hat, x).item(),
        'correlation': pearson_correlation(x_hat, x).item(),
        'pred_dynamic_range': dynamic_range(x_hat).item(),
        'target_dynamic_range': dynamic_range(x).item(),
        'pred_rms': rms(x_hat).item(),
        'target_rms': rms(x).item(),
        'pred_peak': peak_signal_amplitude(x_hat).item(),
        'target_peak': peak_signal_amplitude(x).item()
    }
    
    return metrics
