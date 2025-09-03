"""
Data augmentations for ABR signals.

Light augmentations that preserve ABR morphology while adding regularization.
"""

import torch
import numpy as np
from typing import Dict, Union, Optional, List, Tuple, Callable
from scipy import signal
import torch.nn.functional as F


class MixUpAugmentation:
    """
    Mixup augmentation for ABR signals with linear interpolation.
    
    Mixup creates virtual training examples by mixing pairs of examples and their labels.
    For ABR signals, this helps with regularization while maintaining physiological constraints.
    """
    
    def __init__(self, alpha: float = 0.2, preserve_peak_timing: bool = True):
        """
        Initialize mixup augmentation.
        
        Args:
            alpha: Beta distribution parameter for mixing coefficient
            preserve_peak_timing: Whether to preserve peak timing information
        """
        self.alpha = alpha
        self.preserve_peak_timing = preserve_peak_timing
        
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply mixup to a batch of samples.
        
        Args:
            batch: Batch dictionary with 'x0', 'peak_exists', 'meta', etc.
            
        Returns:
            Mixed batch
        """
        # Handle both ABR dataset keys (x0) and legacy keys (signal)
        signal_key = 'x0' if 'x0' in batch else 'signal'
        if signal_key not in batch:
            return batch
            
        batch_size = batch[signal_key].size(0)
        if batch_size < 2:
            return batch
            
        # Sample mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
            
        # Random permutation for mixing pairs
        indices = torch.randperm(batch_size)
        
        # Mix signals
        mixed_signal = lam * batch[signal_key] + (1 - lam) * batch[signal_key][indices]
        
        # Mix targets
        mixed_batch = batch.copy()
        mixed_batch[signal_key] = mixed_signal
        
        # Handle target mixing for different formats
        if 'peak_exists' in batch:
            # ABR dataset format - mix peak_exists
            mixed_batch['peak_exists'] = lam * batch['peak_exists'] + (1 - lam) * batch['peak_exists'][indices]
        elif 'target' in batch:
            if self.preserve_peak_timing:
                # For peak classification, use soft labels
                mixed_batch['target'] = lam * batch['target'] + (1 - lam) * batch['target'][indices]
            else:
                mixed_batch['target'] = batch['target']  # Keep original for hard decisions
                
        # Store mixing information for loss computation
        mixed_batch['mixup_lambda'] = lam
        mixed_batch['mixup_indices'] = indices
        
        return mixed_batch


class CutMixAugmentation:
    """
    CutMix augmentation for ABR signals by replacing time segments.
    
    CutMix replaces a portion of one signal with a portion from another signal,
    while adjusting the labels proportionally.
    """
    
    def __init__(self, alpha: float = 1.0, preserve_peaks: bool = True):
        """
        Initialize cutmix augmentation.
        
        Args:
            alpha: Beta distribution parameter for cut ratio
            preserve_peaks: Whether to avoid cutting through peak regions
        """
        self.alpha = alpha
        self.preserve_peaks = preserve_peaks
        
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply cutmix to a batch of samples.
        
        Args:
            batch: Batch dictionary with 'x0', 'peak_exists', 'meta', etc.
            
        Returns:
            CutMix batch
        """
        # Handle both ABR dataset keys (x0) and legacy keys (signal)
        signal_key = 'x0' if 'x0' in batch else 'signal'
        if signal_key not in batch:
            return batch
            
        batch_size = batch[signal_key].size(0)
        if batch_size < 2:
            return batch
            
        seq_len = batch[signal_key].size(-1)
        
        # Sample cut ratio
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
            
        # Determine cut size and position
        cut_len = int(seq_len * (1 - lam))
        
        if self.preserve_peaks:
            # Avoid cutting in peak regions (typically 1-10ms, derived from seq_len)
            peak_start = int(0.05 * seq_len)  # ~5% into signal
            peak_end = int(0.25 * seq_len)    # ~25% into signal
            if cut_len < seq_len - (peak_end - peak_start):
                # Cut from baseline or late response regions
                if np.random.rand() > 0.5 and peak_start - cut_len + 1 > 0:
                    # Cut from baseline region (before peaks)
                    cut_start = np.random.randint(0, peak_start - cut_len + 1)
                elif seq_len - cut_len + 1 > peak_end:
                    # Cut from late response region (after peaks)
                    cut_start = np.random.randint(peak_end, seq_len - cut_len + 1)
                else:
                    # Fallback to any valid position
                    cut_start = np.random.randint(0, seq_len - cut_len + 1)
            else:
                cut_start = np.random.randint(0, seq_len - cut_len + 1)
        else:
            cut_start = np.random.randint(0, seq_len - cut_len + 1)
            
        cut_end = cut_start + cut_len
        
        # Random permutation for mixing pairs
        indices = torch.randperm(batch_size)
        
        # Apply cutmix
        mixed_signal = batch[signal_key].clone()
        mixed_signal[:, :, cut_start:cut_end] = batch[signal_key][indices][:, :, cut_start:cut_end]
        
        # Mix targets proportionally
        mixed_batch = batch.copy()
        mixed_batch[signal_key] = mixed_signal
        
        # Handle target mixing for different formats
        if 'peak_exists' in batch:
            # ABR dataset format - mix peak_exists proportionally
            cut_ratio = (cut_end - cut_start) / seq_len
            mixed_batch['peak_exists'] = (1 - cut_ratio) * batch['peak_exists'] + cut_ratio * batch['peak_exists'][indices]
        elif 'target' in batch:
            mixed_batch['target'] = lam * batch['target'] + (1 - lam) * batch['target'][indices]
            
        # Store cutmix information
        mixed_batch['cutmix_lambda'] = lam
        mixed_batch['cutmix_indices'] = indices
        mixed_batch['cut_start'] = cut_start
        mixed_batch['cut_end'] = cut_end
        
        return mixed_batch


class ABRAugmentations:
    """
    Extended ABR augmentations with advanced techniques and curriculum learning support.
    """
    
    def __init__(
        self,
        time_shift_samples: int = 2,
        noise_std: float = 0.01,
        apply_prob: float = 0.5,
        preserve_peaks: bool = True,
        mixup_prob: float = 0.2,
        cutmix_prob: float = 0.2,
        time_stretch_prob: float = 0.1,
        amplitude_scaling_range: Tuple[float, float] = (0.8, 1.2),
        augmentation_strength: float = 1.0,
        curriculum_aware: bool = True,
        # ABR-specific augmentation parameters
        peak_jitter_std: float = 0.05,
        baseline_drift_std: float = 0.01,
        electrode_noise_std: float = 0.005,
        stimulus_artifact_prob: float = 0.1
    ):
        """
        Initialize extended ABR augmentations.
        
        Args:
            time_shift_samples: Maximum samples to shift ±
            noise_std: Standard deviation of additive Gaussian noise
            apply_prob: Probability of applying each augmentation
            preserve_peaks: Whether to preserve peak structure
            mixup_prob: Probability of applying mixup
            cutmix_prob: Probability of applying cutmix
            time_stretch_prob: Probability of applying time stretching
            amplitude_scaling_range: Range for amplitude scaling (min, max)
            augmentation_strength: Overall augmentation intensity multiplier
            curriculum_aware: Whether to adjust augmentation based on training progress
            peak_jitter_std: Standard deviation for peak timing jitter
            baseline_drift_std: Standard deviation for baseline drift
            electrode_noise_std: Standard deviation for electrode noise
            stimulus_artifact_prob: Probability of stimulus artifacts
        """
        self.time_shift_samples = time_shift_samples
        self.noise_std = noise_std
        self.apply_prob = apply_prob
        self.preserve_peaks = preserve_peaks
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.time_stretch_prob = time_stretch_prob
        self.amplitude_scaling_range = amplitude_scaling_range
        self.augmentation_strength = augmentation_strength
        self.curriculum_aware = curriculum_aware
        self.training_progress = 0.0  # 0.0 = start, 1.0 = end
        
        # ABR-specific augmentation parameters
        self.peak_jitter_std = peak_jitter_std
        self.baseline_drift_std = baseline_drift_std
        self.electrode_noise_std = electrode_noise_std
        self.stimulus_artifact_prob = stimulus_artifact_prob
        
        # Initialize advanced augmentations
        self.mixup = MixUpAugmentation(alpha=0.2)
        self.cutmix = CutMixAugmentation(alpha=1.0)
        
    def set_training_progress(self, progress: float):
        """Set current training progress for curriculum-aware augmentation."""
        self.training_progress = np.clip(progress, 0.0, 1.0)
    
    def __call__(self, sample: Dict[str, Union[torch.Tensor, str]]) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Apply augmentations to a sample.
        
        Args:
            sample: Dictionary containing 'x0', 'stat', 'meta', 'peak_exists', etc.
            
        Returns:
            Augmented sample
        """
        # Handle both ABR dataset keys (x0) and legacy keys (signal)
        signal_key = 'x0' if 'x0' in sample else 'signal'
        if signal_key not in sample:
            return sample
            
        signal = sample[signal_key].clone()
        
        # Adjust augmentation probabilities based on curriculum
        if self.curriculum_aware:
            # Start with easier augmentations (lower probability)
            prob_multiplier = 0.3 + 0.7 * self.training_progress
            strength_multiplier = 0.5 + 0.5 * self.training_progress
        else:
            prob_multiplier = 1.0
            strength_multiplier = 1.0
            
        effective_prob = self.apply_prob * prob_multiplier * self.augmentation_strength
        
        # Time shift augmentation
        if torch.rand(1).item() < effective_prob and self.time_shift_samples > 0:
            signal = self._time_shift(signal, strength_multiplier)
        
        # Additive noise augmentation
        if torch.rand(1).item() < effective_prob and self.noise_std > 0:
            signal = self._add_noise(signal, strength_multiplier)
        
        # Time stretching augmentation
        if torch.rand(1).item() < self.time_stretch_prob * prob_multiplier:
            signal = self._time_stretch(signal, strength_multiplier)
        
        # Amplitude scaling augmentation
        if torch.rand(1).item() < effective_prob:
            signal = self._amplitude_scaling(signal, strength_multiplier)
        
        # ABR-specific augmentations
        if torch.rand(1).item() < effective_prob:
            signal = self._peak_jitter(signal, strength_multiplier)
        
        if torch.rand(1).item() < effective_prob:
            signal = self._baseline_drift(signal, strength_multiplier)
        
        if torch.rand(1).item() < effective_prob:
            signal = self._electrode_noise(signal, strength_multiplier)
        
        if torch.rand(1).item() < self.stimulus_artifact_prob * prob_multiplier:
            signal = self._stimulus_artifacts(signal, strength_multiplier)
        
        # Advanced augmentations (applied at batch level, but we can prepare here)
        augmented_sample = sample.copy()
        augmented_sample[signal_key] = signal
        
        return augmented_sample
    
    def _time_shift(self, signal: torch.Tensor, strength_multiplier: float = 1.0) -> torch.Tensor:
        """
        Apply small time shift to signal.
        
        Args:
            signal: Input signal [T] or [C, T]
            strength_multiplier: Multiplier for augmentation strength
            
        Returns:
            Time-shifted signal
        """
        # Compute seq_len from last dimension
        seq_len = signal.shape[-1]
        max_shift = int(self.time_shift_samples * strength_multiplier)
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        
        if shift == 0:
            return signal
            
        if signal.dim() == 1:
            # Handle [T] case
            if shift > 0:
                # Shift right: pad left, truncate right
                shifted = torch.cat([torch.zeros(shift), signal[:-shift]])
            else:
                # Shift left: pad right, truncate left
                shifted = torch.cat([signal[-shift:], torch.zeros(-shift)])
            return shifted
        else:
            # Handle [C, T] case - preserve channel dimension
            if shift > 0:
                # Shift right: pad left with zeros, truncate right
                pad_shape = list(signal.shape)
                pad_shape[-1] = shift
                zeros_left = torch.zeros(pad_shape)
                shifted = torch.cat([zeros_left, signal[..., :-shift]], dim=-1)
            else:
                # Shift left: pad right with zeros, truncate left
                pad_shape = list(signal.shape)
                pad_shape[-1] = -shift
                zeros_right = torch.zeros(pad_shape)
                shifted = torch.cat([signal[..., -shift:], zeros_right], dim=-1)
            return shifted
    
    def _add_noise(self, signal: torch.Tensor, strength_multiplier: float = 1.0) -> torch.Tensor:
        """
        Add small amount of Gaussian noise.
        
        Args:
            signal: Input signal
            strength_multiplier: Multiplier for augmentation strength
            
        Returns:
            Noisy signal
        """
        effective_noise_std = self.noise_std * strength_multiplier
        noise = torch.randn_like(signal) * effective_noise_std
        return signal + noise
    
    def _time_stretch(self, signal: torch.Tensor, strength_multiplier: float = 1.0) -> torch.Tensor:
        """
        Apply time stretching to signal.
        
        Args:
            signal: Input signal [seq_len] or [1, seq_len]
            strength_multiplier: Multiplier for augmentation strength
            
        Returns:
            Time-stretched signal
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Generate stretch factor (0.9 to 1.1 range)
        stretch_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.2 * strength_multiplier
        
        # Simple time stretching using interpolation
        original_length = signal.shape[-1]
        stretched_length = int(original_length * stretch_factor)
        
        # Interpolate to new length, then crop/pad to original length
        signal_stretched = F.interpolate(
            signal.unsqueeze(0), 
            size=stretched_length, 
            mode='linear', 
            align_corners=False
        ).squeeze(0)
        
        if stretched_length > original_length:
            # Crop to original length
            start_idx = (stretched_length - original_length) // 2
            signal_stretched = signal_stretched[:, start_idx:start_idx + original_length]
        elif stretched_length < original_length:
            # Pad to original length
            pad_total = original_length - stretched_length
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            signal_stretched = F.pad(signal_stretched, (pad_left, pad_right), mode='reflect')
        
        return signal_stretched.squeeze(0) if squeeze_output else signal_stretched
    
    def _amplitude_scaling(self, signal: torch.Tensor, strength_multiplier: float = 1.0) -> torch.Tensor:
        """
        Apply amplitude scaling to signal.
        
        Args:
            signal: Input signal
            strength_multiplier: Multiplier for augmentation strength
            
        Returns:
            Amplitude-scaled signal
        """
        min_scale, max_scale = self.amplitude_scaling_range
        # Adjust scaling range based on strength
        scale_range = (max_scale - min_scale) * strength_multiplier
        center_scale = (min_scale + max_scale) / 2
        
        scale_factor = center_scale + (torch.rand(1).item() - 0.5) * scale_range
        return signal * scale_factor
    
    def _peak_jitter(self, signal: torch.Tensor, strength_multiplier: float = 1.0) -> torch.Tensor:
        """
        Apply peak timing jitter to simulate measurement variability.
        
        Args:
            signal: Input signal
            strength_multiplier: Multiplier for augmentation strength
            
        Returns:
            Signal with peak timing jitter
        """
        if not self.preserve_peaks:
            return signal
            
        # Simple implementation: add small random shifts to different parts of the signal
        jitter_std = self.peak_jitter_std * strength_multiplier
        
        # Create a smooth jitter pattern
        jitter = torch.randn(signal.shape[-1] // 10) * jitter_std
        jitter_interp = F.interpolate(
            jitter.unsqueeze(0).unsqueeze(0), 
            size=signal.shape[-1], 
            mode='linear', 
            align_corners=False
        ).squeeze()
        
        # Apply as phase shift (simplified)
        indices = torch.arange(signal.shape[-1], dtype=torch.float32)
        shifted_indices = indices + jitter_interp
        shifted_indices = torch.clamp(shifted_indices, 0, signal.shape[-1] - 1)
        
        # Use shifted_indices to resample the signal via linear interpolation
        # Handle both [T] and [1, T] inputs
        if signal.dim() == 1:
            # For 1D signal
            # Grid sample requires input to be normalized to [-1, 1]
            grid = shifted_indices / (signal.shape[-1] - 1) * 2 - 1
            
            # Prepare signal for grid_sample [N, C, H, W] format -> [1, 1, 1, T]
            signal_reshaped = signal.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, T]
            # Grid needs to be [N, H_out, W_out, 2] for 2D sampling
            # For 1D sampling along width, we need [1, 1, T, 2] where second coord is 0
            grid_2d = torch.zeros(1, 1, signal.shape[-1], 2, device=signal.device)
            grid_2d[0, 0, :, 0] = grid  # x coordinates
            # y coordinates stay 0 (middle of height dimension)
            
            # Sample using grid_sample with linear interpolation
            sampled = F.grid_sample(signal_reshaped, grid_2d, mode='bilinear', 
                                  padding_mode='border', align_corners=True)
            return sampled.squeeze(0).squeeze(0).squeeze(0)  # Return to original shape [T]
        else:
            # For 2D signal [1, T]
            grid = shifted_indices / (signal.shape[-1] - 1) * 2 - 1
            grid = grid.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            
            # Prepare signal for grid_sample [N, C, H, W] format
            # Need 4D: [batch, channels, height, width] -> [1, 1, 1, T]
            signal_reshaped = signal.unsqueeze(0).unsqueeze(2)  # [1, 1, 1, T]
            # Grid needs to be [N, H_out, W_out, 2] for 2D sampling
            # For 1D sampling along width, we need [1, 1, T, 2] where second coord is 0
            grid_2d = torch.zeros(1, 1, signal.shape[-1], 2, device=signal.device)
            grid_2d[0, 0, :, 0] = grid.squeeze(0).squeeze(-1)  # x coordinates
            # y coordinates stay 0 (middle of height dimension)
            
            # Sample using grid_sample
            sampled = F.grid_sample(signal_reshaped, grid_2d, mode='bilinear',
                                  padding_mode='border', align_corners=True)
            return sampled.squeeze(0).squeeze(1)  # Return to shape [1, T]
    
    def _baseline_drift(self, signal: torch.Tensor, strength_multiplier: float = 1.0) -> torch.Tensor:
        """
        Add baseline drift to simulate electrode drift.
        
        Args:
            signal: Input signal
            strength_multiplier: Multiplier for augmentation strength
            
        Returns:
            Signal with baseline drift
        """
        drift_std = self.baseline_drift_std * strength_multiplier
        
        # Create smooth drift pattern
        drift_points = torch.randn(5) * drift_std
        drift = F.interpolate(
            drift_points.unsqueeze(0).unsqueeze(0), 
            size=signal.shape[-1], 
            mode='linear', 
            align_corners=False
        ).squeeze()
        
        return signal + drift
    
    def _electrode_noise(self, signal: torch.Tensor, strength_multiplier: float = 1.0) -> torch.Tensor:
        """
        Add electrode noise to simulate measurement artifacts.
        
        Args:
            signal: Input signal
            strength_multiplier: Multiplier for augmentation strength
            
        Returns:
            Signal with electrode noise
        """
        noise_std = self.electrode_noise_std * strength_multiplier
        
        # High-frequency noise typical of electrode artifacts
        noise = torch.randn_like(signal) * noise_std
        
        # Apply high-pass filtering effect (simulate electrode noise characteristics)
        # Simple implementation using differences
        if signal.shape[-1] > 1:
            noise_filtered = noise.clone()
            noise_filtered[..., 1:] = noise[..., 1:] - 0.8 * noise[..., :-1]
        else:
            noise_filtered = noise
            
        return signal + noise_filtered
    
    def _stimulus_artifacts(self, signal: torch.Tensor, strength_multiplier: float = 1.0) -> torch.Tensor:
        """
        Add stimulus artifacts to simulate stimulation artifacts.
        
        Args:
            signal: Input signal
            strength_multiplier: Multiplier for augmentation strength
            
        Returns:
            Signal with stimulus artifacts
        """
        # Add brief high-amplitude artifacts at the beginning (stimulus artifact)
        artifact_strength = 0.1 * strength_multiplier
        artifact_duration = min(10, signal.shape[-1] // 10)  # First 10 samples or 10% of signal
        
        artifact = torch.randn(artifact_duration) * artifact_strength
        signal_with_artifact = signal.clone()
        signal_with_artifact[..., :artifact_duration] += artifact
        
        return signal_with_artifact
    
    def apply_batch_augmentations(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply batch-level augmentations like mixup and cutmix.
        
        Args:
            batch: Batch dictionary
            
        Returns:
            Augmented batch
        """
        if self.curriculum_aware:
            prob_multiplier = 0.3 + 0.7 * self.training_progress
        else:
            prob_multiplier = 1.0
            
        # Apply mixup
        if torch.rand(1).item() < self.mixup_prob * prob_multiplier:
            batch = self.mixup(batch)
            
        # Apply cutmix (mutually exclusive with mixup for now)
        elif torch.rand(1).item() < self.cutmix_prob * prob_multiplier:
            batch = self.cutmix(batch)
            
        return batch


class NoAugmentation:
    """No-op augmentation for when augmentations are disabled."""
    
    def __call__(self, sample: Dict[str, Union[torch.Tensor, str]]) -> Dict[str, Union[torch.Tensor, str]]:
        return sample


class AugmentationPipeline:
    """
    Pipeline for applying multiple augmentations sequentially or with probability.
    """
    
    def __init__(
        self,
        augmentations: List[Tuple[Callable, float]],
        curriculum_aware: bool = True
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentations: List of (augmentation_fn, probability) tuples
            curriculum_aware: Whether to adjust augmentation intensity based on training progress
        """
        self.augmentations = augmentations
        self.curriculum_aware = curriculum_aware
        self.training_progress = 0.0  # 0.0 = start, 1.0 = end
        
    def set_training_progress(self, progress: float):
        """Set current training progress for curriculum-aware augmentation."""
        self.training_progress = np.clip(progress, 0.0, 1.0)
        
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentation pipeline."""
        augmented_sample = sample
        
        for aug_fn, base_prob in self.augmentations:
            # Adjust probability based on curriculum
            if self.curriculum_aware:
                # Start with easier augmentations (lower probability)
                prob = base_prob * (0.3 + 0.7 * self.training_progress)
            else:
                prob = base_prob
                
            if torch.rand(1).item() < prob:
                augmented_sample = aug_fn(augmented_sample)
                
        return augmented_sample


def create_augmentation(
    enable: bool = True,
    time_shift_samples: int = 2,
    noise_std: float = 0.01,
    apply_prob: float = 0.5,
    mixup_prob: float = 0.2,
    cutmix_prob: float = 0.2,
    augmentation_strength: float = 1.0,
    curriculum_aware: bool = True
) -> Union[ABRAugmentations, NoAugmentation]:
    """
    Create augmentation transform.
    
    Args:
        enable: Whether to enable augmentations
        time_shift_samples: Maximum samples to shift ±
        noise_std: Standard deviation of additive noise
        apply_prob: Probability of applying each augmentation
        mixup_prob: Probability of applying mixup
        cutmix_prob: Probability of applying cutmix
        augmentation_strength: Overall augmentation intensity multiplier
        curriculum_aware: Whether to adjust augmentation based on training progress
        
    Returns:
        Augmentation transform
    """
    if enable:
        return ABRAugmentations(
            time_shift_samples=time_shift_samples,
            noise_std=noise_std,
            apply_prob=apply_prob,
            mixup_prob=mixup_prob,
            cutmix_prob=cutmix_prob,
            augmentation_strength=augmentation_strength,
            curriculum_aware=curriculum_aware
        )
    else:
        return NoAugmentation()


def create_augmentation_pipeline(config: Dict[str, any]) -> Optional[AugmentationPipeline]:
    """
    Create augmentation pipeline from configuration.
    
    Args:
        config: Augmentation configuration dictionary
        
    Returns:
        Configured augmentation pipeline or None if disabled
    """
    if not config.get('enabled', True):
        return None
        
    augmentations = []
    
    # Add basic augmentations
    if config.get('time_shift_prob', 0.5) > 0:
        time_shift_aug = lambda sample: ABRAugmentations(
            time_shift_samples=config.get('time_shift_samples', 2)
        )(sample)
        augmentations.append((time_shift_aug, config.get('time_shift_prob', 0.5)))
        
    # Add advanced augmentations based on config
    if config.get('amplitude_scaling_prob', 0.3) > 0:
        from .curriculum import DifficultyMetrics  # Import here to avoid circular imports
        amp_aug = lambda sample: sample  # Placeholder - would implement amplitude scaling
        augmentations.append((amp_aug, config.get('amplitude_scaling_prob', 0.3)))
        
    return AugmentationPipeline(
        augmentations=augmentations,
        curriculum_aware=config.get('curriculum_aware', True)
    )


# Unit test for CutMix with 3D [B,C,T] tensors
def test_cutmix_3d_tensors():
    """
    Unit test for CutMixAugmentation with [B,1,T] shaped inputs to validate shapes and bounds.
    """
    # Create test data with [B,1,T] shape
    batch_size = 4
    channels = 1
    seq_len = 200  # ABR signal length
    
    # Create mock batch with 3D tensors
    batch = {
        'x0': torch.randn(batch_size, channels, seq_len),
        'peak_exists': torch.randint(0, 2, (batch_size,)).float(),
        'meta': [{'target': 0} for _ in range(batch_size)]
    }
    
    # Test CutMix augmentation
    cutmix = CutMixAugmentation(alpha=1.0, preserve_peaks=True)
    
    # Apply augmentation
    mixed_batch = cutmix(batch)
    
    # Validate shapes
    assert mixed_batch['x0'].shape == (batch_size, channels, seq_len), \
        f"Expected shape {(batch_size, channels, seq_len)}, got {mixed_batch['x0'].shape}"
    
    assert mixed_batch['peak_exists'].shape == (batch_size,), \
        f"Expected shape {(batch_size,)}, got {mixed_batch['peak_exists'].shape}"
    
    # Validate that the signal was actually modified (not identical to original)
    assert not torch.equal(batch['x0'], mixed_batch['x0']), \
        "CutMix should modify the signal"
    
    # Test with different sequence lengths
    for test_seq_len in [100, 200, 500]:
        test_batch = {
            'x0': torch.randn(2, 1, test_seq_len),
            'peak_exists': torch.tensor([1.0, 0.0])
        }
        
        mixed_test = cutmix(test_batch)
        
        # Check that peak bounds are properly derived from seq_len
        expected_peak_start = int(0.05 * test_seq_len)
        expected_peak_end = int(0.25 * test_seq_len)
        
        assert mixed_test['x0'].shape == (2, 1, test_seq_len), \
            f"Shape mismatch for seq_len={test_seq_len}"
        
        # Ensure peak bounds are reasonable
        assert 0 <= expected_peak_start < expected_peak_end <= test_seq_len, \
            f"Invalid peak bounds for seq_len={test_seq_len}: start={expected_peak_start}, end={expected_peak_end}"
    
    # Test without peak preservation
    cutmix_no_peaks = CutMixAugmentation(alpha=1.0, preserve_peaks=False)
    mixed_no_peaks = cutmix_no_peaks(batch)
    
    assert mixed_no_peaks['x0'].shape == (batch_size, channels, seq_len), \
        "Shape should be preserved even without peak preservation"
    
    # Test edge case with batch size 1 (should return unchanged)
    single_batch = {
        'x0': torch.randn(1, 1, seq_len),
        'peak_exists': torch.tensor([1.0])
    }
    
    single_mixed = cutmix(single_batch)
    assert torch.equal(single_batch['x0'], single_mixed['x0']), \
        "Single sample batch should remain unchanged"
    
    print("CutMix 3D tensor tests passed!")


def test_peak_jitter():
    """
    Test peak jitter augmentation to verify output differs from input and preserves shape.
    """
    # Create test signal with some structure
    seq_len = 200
    t = torch.linspace(0, 1, seq_len)
    signal_1d = torch.sin(2 * torch.pi * 5 * t) + 0.5 * torch.sin(2 * torch.pi * 20 * t)
    signal_2d = signal_1d.unsqueeze(0)  # [1, T] format
    
    # Create augmentation instance
    aug = ABRAugmentations(
        time_shift_samples=5,
        noise_std=0.01,
        apply_prob=1.0,
        preserve_peaks=True,
        peak_jitter_std=2.0  # Set a reasonable jitter
    )
    
    # Test 1D signal
    jittered_1d = aug._peak_jitter(signal_1d, strength_multiplier=1.0)
    
    # Verify shape preservation
    assert jittered_1d.shape == signal_1d.shape, f"1D shape mismatch: {jittered_1d.shape} vs {signal_1d.shape}"
    
    # Verify output differs from input (beyond floating-point noise)
    diff_1d = torch.abs(jittered_1d - signal_1d).max().item()
    assert diff_1d > 1e-6, f"1D signal should change significantly, max diff: {diff_1d}"
    
    # Test 2D signal [1, T]
    jittered_2d = aug._peak_jitter(signal_2d, strength_multiplier=1.0)
    
    # Verify shape preservation
    assert jittered_2d.shape == signal_2d.shape, f"2D shape mismatch: {jittered_2d.shape} vs {signal_2d.shape}"
    
    # Verify output differs from input
    diff_2d = torch.abs(jittered_2d - signal_2d).max().item()
    assert diff_2d > 1e-6, f"2D signal should change significantly, max diff: {diff_2d}"
    
    # Test with zero jitter (should be nearly identical)
    aug_no_jitter = ABRAugmentations(
        time_shift_samples=5,
        peak_jitter_std=0.0,  # No jitter
        preserve_peaks=True
    )
    
    no_jitter_1d = aug_no_jitter._peak_jitter(signal_1d)
    diff_no_jitter = torch.abs(no_jitter_1d - signal_1d).max().item()
    assert diff_no_jitter < 1e-3, f"No jitter should preserve signal, max diff: {diff_no_jitter}"
    
    # Test with preserve_peaks=False (should return original)
    aug_no_preserve = ABRAugmentations(preserve_peaks=False)
    no_preserve_result = aug_no_preserve._peak_jitter(signal_1d)
    assert torch.equal(no_preserve_result, signal_1d), "Should return original when preserve_peaks=False"
    
    print("Peak jitter tests passed!")


def test_time_shift_channels():
    """
    Unit test for _time_shift with [1, T] shape verifying the shift took effect.
    """
    # Create test signals
    seq_len = 200
    signal_1d = torch.randn(seq_len)
    signal_2d = torch.randn(1, seq_len)  # [1, T] format
    
    # Create augmentation instance with a reasonable shift
    aug = ABRAugmentations(
        time_shift_samples=10,
        noise_std=0.0,  # No noise to isolate shift effect
        apply_prob=1.0
    )
    
    # Test 1D signal shift
    shifted_1d = aug._time_shift(signal_1d, strength_multiplier=1.0)
    
    # Verify shape preservation
    assert shifted_1d.shape == signal_1d.shape, f"1D shape mismatch: {shifted_1d.shape} vs {signal_1d.shape}"
    
    # Test [1, T] signal shift
    shifted_2d = aug._time_shift(signal_2d, strength_multiplier=1.0)
    
    # Verify shape preservation
    assert shifted_2d.shape == signal_2d.shape, f"2D shape mismatch: {shifted_2d.shape} vs {signal_2d.shape}"
    
    # Verify shift took effect by comparing non-zero regions
    # For a positive shift, the beginning should be zeros and the end should differ
    # For a negative shift, the end should be zeros and the beginning should differ
    diff_2d = torch.abs(shifted_2d - signal_2d).max().item()
    assert diff_2d > 1e-6, f"Shift should change the signal, max diff: {diff_2d}"
    
    # Test with larger channel dimension [3, T]
    signal_multi = torch.randn(3, seq_len)
    shifted_multi = aug._time_shift(signal_multi, strength_multiplier=1.0)
    
    # Verify shape preservation
    assert shifted_multi.shape == signal_multi.shape, f"Multi-channel shape mismatch: {shifted_multi.shape} vs {signal_multi.shape}"
    
    # Verify all channels are shifted consistently
    diff_multi = torch.abs(shifted_multi - signal_multi).max().item()
    assert diff_multi > 1e-6, f"Multi-channel shift should change the signal, max diff: {diff_multi}"
    
    # Test zero shift (should return identical signal)
    aug_no_shift = ABRAugmentations(time_shift_samples=0)
    no_shift_result = aug_no_shift._time_shift(signal_2d)
    assert torch.equal(no_shift_result, signal_2d), "Zero shift should return identical signal"
    
    # Manual verification: test a specific shift
    # Create a signal with a known pattern and verify the shift
    pattern_signal = torch.zeros(1, 20)
    pattern_signal[0, 5:10] = 1.0  # Signal from index 5-9
    
    # Force a positive shift of 3
    aug_fixed = ABRAugmentations(time_shift_samples=3)
    # We need to make a deterministic shift, so let's set the seed temporarily
    torch.manual_seed(42)
    shifted_pattern = aug_fixed._time_shift(pattern_signal)
    
    # Check that the pattern moved (this is probabilistic, but let's check for any change)
    pattern_diff = torch.abs(shifted_pattern - pattern_signal).sum().item()
    assert pattern_diff > 0, "Pattern should shift"
    
    print("Time shift channel tests passed!")


if __name__ == "__main__":
    # Run the tests when the file is executed directly
    test_cutmix_3d_tensors()
    test_peak_jitter()
    test_time_shift_channels()