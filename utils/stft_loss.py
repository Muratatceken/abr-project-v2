"""
Multi-resolution STFT loss for improved frequency domain reconstruction.

Provides perceptual loss terms for audio/signal generation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def stft_magnitude(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
    """
    Compute STFT magnitude spectrogram.
    
    Args:
        x: Input signal [B, 1, T] or [B, T]
        n_fft: FFT size
        hop_length: Hop length for STFT
        win_length: Window length
        
    Returns:
        Magnitude spectrogram [B, F, T_frames]
    """
    # Ensure input is [B, T] format for torch.stft
    if x.dim() == 3:
        x = x.squeeze(1)  # [B, 1, T] -> [B, T]
    
    # Compute STFT
    stft = torch.stft(
        x, 
        n_fft=n_fft,
        hop_length=hop_length, 
        win_length=win_length,
        window=torch.hann_window(win_length, device=x.device),
        return_complex=True
    )
    
    # Return magnitude
    magnitude = torch.abs(stft)  # [B, F, T_frames]
    return magnitude


class MultiResSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss combining magnitude and log-magnitude losses.
    
    Computes STFT loss at multiple resolutions to capture both
    fine-grained and coarse-grained frequency content.
    """
    
    def __init__(
        self, 
        n_fft: int = 64,
        hop_length: int = 16, 
        win_length: int = 64,
        eps: float = 1e-8,
        weight: float = 0.15
    ):
        """
        Initialize multi-resolution STFT loss.
        
        Args:
            n_fft: FFT size for primary resolution
            hop_length: Hop length
            win_length: Window length  
            eps: Small epsilon for log stability
            weight: Loss weight factor
        """
        super().__init__()
        self.eps = eps
        self.weight = weight
        
        # Define multiple resolutions (primary + additional scales)
        self.stft_params = [
            (n_fft, hop_length, win_length),
            (n_fft // 2, hop_length // 2, win_length // 2),  # Higher temporal resolution
            (n_fft * 2, hop_length * 2, win_length * 2),    # Higher frequency resolution
        ]
        
        # Remove any invalid configurations (where hop > win)
        self.stft_params = [(n, h, w) for n, h, w in self.stft_params if h <= w and n >= w]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.
        
        Args:
            pred: Predicted signal [B, 1, T] or [B, T]
            target: Target signal [B, 1, T] or [B, T]
            
        Returns:
            Weighted STFT loss scalar
        """
        total_loss = 0.0
        
        for n_fft, hop_length, win_length in self.stft_params:
            # Skip if sequence too short for this configuration
            seq_len = pred.shape[-1]
            if seq_len < win_length:
                continue
                
            try:
                # Compute magnitude spectrograms
                pred_mag = stft_magnitude(pred, n_fft, hop_length, win_length)
                target_mag = stft_magnitude(target, n_fft, hop_length, win_length)
                
                # L1 magnitude loss
                mag_loss = F.l1_loss(pred_mag, target_mag)
                
                # Log-magnitude loss (adds perceptual weighting)
                pred_log_mag = torch.log(pred_mag + self.eps)
                target_log_mag = torch.log(target_mag + self.eps)
                log_mag_loss = F.l1_loss(pred_log_mag, target_log_mag)
                
                # Combine losses for this resolution
                resolution_loss = mag_loss + log_mag_loss
                total_loss += resolution_loss
                
            except RuntimeError:
                # Skip this resolution if STFT fails (e.g., sequence too short)
                continue
        
        # Average across resolutions and apply weight
        if len(self.stft_params) > 0:
            total_loss = total_loss / len(self.stft_params)
        
        return self.weight * total_loss


def spectral_convergence_loss(pred: torch.Tensor, target: torch.Tensor, 
                            n_fft: int = 64, hop_length: int = 16, 
                            win_length: int = 64) -> torch.Tensor:
    """
    Spectral convergence loss - measures relative error in magnitude spectrogram.
    
    Args:
        pred: Predicted signal [B, 1, T] or [B, T]
        target: Target signal [B, 1, T] or [B, T]
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        
    Returns:
        Spectral convergence loss
    """
    pred_mag = stft_magnitude(pred, n_fft, hop_length, win_length)
    target_mag = stft_magnitude(target, n_fft, hop_length, win_length)
    
    numerator = torch.norm(target_mag - pred_mag, p=2, dim=(1, 2))
    denominator = torch.norm(target_mag, p=2, dim=(1, 2))
    
    # Avoid division by zero
    sc_loss = numerator / (denominator + 1e-8)
    
    return sc_loss.mean()


def log_stft_magnitude_loss(pred: torch.Tensor, target: torch.Tensor,
                           n_fft: int = 64, hop_length: int = 16, 
                           win_length: int = 64, eps: float = 1e-8) -> torch.Tensor:
    """
    Log STFT magnitude loss for perceptual quality.
    
    Args:
        pred: Predicted signal [B, 1, T] or [B, T]
        target: Target signal [B, 1, T] or [B, T]
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        eps: Small epsilon for log stability
        
    Returns:
        Log magnitude loss
    """
    pred_mag = stft_magnitude(pred, n_fft, hop_length, win_length)
    target_mag = stft_magnitude(target, n_fft, hop_length, win_length)
    
    pred_log_mag = torch.log(pred_mag + eps)
    target_log_mag = torch.log(target_mag + eps)
    
    return F.l1_loss(pred_log_mag, target_log_mag)
