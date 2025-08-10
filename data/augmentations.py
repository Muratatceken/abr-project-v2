"""
Data augmentations for ABR signals.

Light augmentations that preserve ABR morphology while adding regularization.
"""

import torch
import numpy as np
from typing import Dict, Union, Optional


class ABRAugmentations:
    """
    Light augmentations for ABR signals that preserve physiological structure.
    """
    
    def __init__(
        self,
        time_shift_samples: int = 2,
        noise_std: float = 0.01,
        apply_prob: float = 0.5,
        preserve_peaks: bool = True
    ):
        """
        Initialize ABR augmentations.
        
        Args:
            time_shift_samples: Maximum samples to shift ±
            noise_std: Standard deviation of additive Gaussian noise
            apply_prob: Probability of applying each augmentation
            preserve_peaks: Whether to preserve peak structure
        """
        self.time_shift_samples = time_shift_samples
        self.noise_std = noise_std
        self.apply_prob = apply_prob
        self.preserve_peaks = preserve_peaks
    
    def __call__(self, sample: Dict[str, Union[torch.Tensor, str]]) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Apply augmentations to a sample.
        
        Args:
            sample: Dictionary containing 'signal', 'static', 'target', etc.
            
        Returns:
            Augmented sample
        """
        if 'signal' not in sample:
            return sample
            
        signal = sample['signal'].clone()
        
        # Time shift augmentation
        if torch.rand(1).item() < self.apply_prob and self.time_shift_samples > 0:
            signal = self._time_shift(signal)
        
        # Additive noise augmentation
        if torch.rand(1).item() < self.apply_prob and self.noise_std > 0:
            signal = self._add_noise(signal)
        
        # Update sample
        augmented_sample = sample.copy()
        augmented_sample['signal'] = signal
        
        return augmented_sample
    
    def _time_shift(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply small time shift to signal.
        
        Args:
            signal: Input signal [seq_len] or [1, seq_len]
            
        Returns:
            Time-shifted signal
        """
        if signal.dim() == 1:
            seq_len = signal.size(0)
            shift = torch.randint(-self.time_shift_samples, self.time_shift_samples + 1, (1,)).item()
            
            if shift == 0:
                return signal
            elif shift > 0:
                # Shift right: pad left, truncate right
                shifted = torch.cat([torch.zeros(shift), signal[:-shift]])
            else:
                # Shift left: pad right, truncate left
                shifted = torch.cat([signal[-shift:], torch.zeros(-shift)])
            
            return shifted
        else:
            # Handle multi-dimensional case
            return signal  # Skip for now to avoid complexity
    
    def _add_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Add small amount of Gaussian noise.
        
        Args:
            signal: Input signal
            
        Returns:
            Noisy signal
        """
        noise = torch.randn_like(signal) * self.noise_std
        return signal + noise


class NoAugmentation:
    """No-op augmentation for when augmentations are disabled."""
    
    def __call__(self, sample: Dict[str, Union[torch.Tensor, str]]) -> Dict[str, Union[torch.Tensor, str]]:
        return sample


def create_augmentation(
    enable: bool = True,
    time_shift_samples: int = 2,
    noise_std: float = 0.01,
    apply_prob: float = 0.5
) -> Union[ABRAugmentations, NoAugmentation]:
    """
    Create augmentation transform.
    
    Args:
        enable: Whether to enable augmentations
        time_shift_samples: Maximum samples to shift ±
        noise_std: Standard deviation of additive noise
        apply_prob: Probability of applying each augmentation
        
    Returns:
        Augmentation transform
    """
    if enable:
        return ABRAugmentations(
            time_shift_samples=time_shift_samples,
            noise_std=noise_std,
            apply_prob=apply_prob
        )
    else:
        return NoAugmentation()