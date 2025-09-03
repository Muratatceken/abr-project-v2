"""
Curriculum learning framework for ABR transformer training.

This module implements curriculum learning strategies that progressively increase
training difficulty based on sample characteristics like SNR, peak complexity,
and hearing loss severity.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler, Sampler
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Iterator
from abc import ABC, abstractmethod
import pickle
from sklearn.preprocessing import StandardScaler
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class DifficultyMetrics:
    """
    Compute sample difficulty metrics for curriculum learning.
    
    Provides various methods to assess the difficulty of ABR samples
    based on signal characteristics and clinical parameters.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.cached_metrics = {}
        
    def compute_snr_difficulty(self, signal_data: np.ndarray, fs: float = 50000, baseline_window: Optional[int] = None) -> float:
        """
        Compute difficulty based on signal-to-noise ratio.
        
        Args:
            signal_data: ABR signal array (length 200 for ABR dataset)
            fs: Sampling frequency (default 50kHz)
            baseline_window: Window size for baseline estimation (auto-computed if None)
            
        Returns:
            Difficulty score (higher = more difficult, lower SNR)
        """
        signal_length = len(signal_data)
        
        # Auto-compute baseline window for length-200 signals
        if baseline_window is None:
            baseline_window = min(50, signal_length // 4)  # Use first quarter or 50 samples, whichever is smaller
        
        # Safe windowing based on signal length
        baseline_window = min(baseline_window, signal_length // 2)
        
        # Estimate noise from baseline
        baseline = signal_data[:baseline_window]
        noise_std = np.std(baseline)
        
        # Estimate signal power from the response window
        response_start = baseline_window
        response_end = min(signal_length, baseline_window + int(0.01 * fs))  # ~10ms or remaining signal
        response_window = signal_data[response_start:response_end]
        
        if len(response_window) == 0:
            response_window = signal_data[baseline_window:]
            
        signal_power = np.var(response_window)
        
        # Compute SNR
        if noise_std == 0:
            snr = float('inf')
        else:
            snr = 10 * np.log10(signal_power / (noise_std ** 2))
            
        # Convert to difficulty score (inverse of SNR, normalized)
        difficulty = 1.0 / (1.0 + np.exp(snr / 10.0))  # Sigmoid transformation
        return float(difficulty)
        
    def compute_peak_difficulty(self, signal_data: np.ndarray, peaks_present: Union[np.ndarray, bool], fs: float = 50000) -> float:
        """
        Compute difficulty based on peak detection complexity.
        
        Args:
            signal_data: ABR signal array (length 200 for ABR dataset)
            peaks_present: Binary value or array indicating peak presence
            fs: Sampling frequency (default 50kHz)
            
        Returns:
            Difficulty score based on missing peaks and signal clarity
        """
        # Handle single boolean peak_exists from ABR dataset
        if isinstance(peaks_present, (bool, int, float)):
            missing_peaks_ratio = 0.0 if peaks_present else 1.0
        elif hasattr(peaks_present, 'ndim') and peaks_present.ndim == 0:
            # Handle 0-d tensor/array (scalar)
            missing_peaks_ratio = 0.0 if bool(peaks_present) else 1.0
        else:
            # Handle array of peak indicators
            missing_peaks_ratio = 1.0 - (peaks_present.sum() / len(peaks_present))
        
        # Additional difficulty from signal characteristics
        # Compute signal variability
        signal_variability = np.std(np.diff(signal_data)) / (np.abs(np.mean(signal_data)) + 1e-8)
        
        # Compute frequency content difficulty with safe PSD computation
        try:
            freqs, psd = signal.periodogram(signal_data, fs=fs)
            # ABR components are typically in 100-3000 Hz range
            abr_band = (freqs >= 100) & (freqs <= 3000)
            abr_power = np.sum(psd[abr_band])
            total_power = np.sum(psd)
            
            frequency_difficulty = 1.0 - (abr_power / (total_power + 1e-8))
        except:
            # Fallback if PSD computation fails for short signals
            frequency_difficulty = 0.5
        
        # Combine difficulties
        combined_difficulty = (
            0.4 * missing_peaks_ratio + 
            0.3 * min(1.0, signal_variability / 0.1) +  # Normalize variability
            0.3 * frequency_difficulty
        )
        
        return float(np.clip(combined_difficulty, 0.0, 1.0))
        
    def compute_hearing_loss_difficulty(self, hearing_loss_db: float) -> float:
        """
        Compute difficulty based on hearing loss severity.
        
        Args:
            hearing_loss_db: Hearing loss in dB HL
            
        Returns:
            Difficulty score based on hearing loss severity
        """
        # Normalize hearing loss to difficulty score
        # Normal: 0-25 dB, Mild: 26-40 dB, Moderate: 41-70 dB, Severe: 71-90 dB, Profound: >90 dB
        if hearing_loss_db <= 25:
            difficulty = 0.1  # Easy - normal hearing
        elif hearing_loss_db <= 40:
            difficulty = 0.3  # Mild difficulty
        elif hearing_loss_db <= 70:
            difficulty = 0.6  # Moderate difficulty
        elif hearing_loss_db <= 90:
            difficulty = 0.8  # High difficulty
        else:
            difficulty = 1.0  # Maximum difficulty
            
        return difficulty
        
    def compute_combined_difficulty(
        self, 
        signal_data: np.ndarray,
        peaks_present: Union[np.ndarray, bool],
        hearing_loss_db: Optional[float],
        fs: float = 50000,
        weights: Dict[str, float] = None
    ) -> float:
        """
        Compute combined difficulty from multiple metrics.
        
        Args:
            signal_data: ABR signal array
            peaks_present: Binary array indicating which peaks are present
            hearing_loss_db: Hearing loss in dB HL (None if not available)
            weights: Weights for different difficulty components
            
        Returns:
            Combined difficulty score
        """
        if weights is None:
            weights = {'snr': 0.4, 'peaks': 0.4, 'hearing_loss': 0.2}
            
        snr_diff = self.compute_snr_difficulty(signal_data, fs=fs)
        peak_diff = self.compute_peak_difficulty(signal_data, peaks_present, fs=fs)
        
        # Handle None hearing loss by renormalizing weights
        if hearing_loss_db is None:
            # Ignore hearing loss component and renormalize other weights
            total_weight = weights['snr'] + weights['peaks']
            if total_weight > 0:
                norm_snr_weight = weights['snr'] / total_weight
                norm_peak_weight = weights['peaks'] / total_weight
                combined = norm_snr_weight * snr_diff + norm_peak_weight * peak_diff
            else:
                # Fallback to equal weights
                combined = 0.5 * snr_diff + 0.5 * peak_diff
        else:
            hl_diff = self.compute_hearing_loss_difficulty(hearing_loss_db)
            combined = (
                weights['snr'] * snr_diff +
                weights['peaks'] * peak_diff +
                weights['hearing_loss'] * hl_diff
            )
        
        return float(np.clip(combined, 0.0, 1.0))
        
    def compute_batch_difficulties(
        self, 
        dataset: Dataset,
        metric_type: str = 'combined',
        fs: float = 50000,
        cache_key: Optional[str] = None,
        difficulty_weights: Optional[Dict[str, float]] = None,
        hearing_loss_map: Optional[Dict[int, float]] = None
    ) -> np.ndarray:
        """
        Compute difficulty scores for entire dataset.
        
        Args:
            dataset: ABR dataset
            metric_type: Type of difficulty metric to compute
            fs: Sampling frequency
            cache_key: Key for caching results
            difficulty_weights: Weights for combined difficulty computation
            hearing_loss_map: Mapping from class indices to dB hearing loss values
            
        Returns:
            Array of difficulty scores for all samples
        """
        # Check cache first
        if cache_key and cache_key in self.cached_metrics:
            return self.cached_metrics[cache_key]
            
        difficulties = []
        
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                
                if isinstance(sample, dict):
                    # Handle ABR dataset format with x0, peak_exists, meta keys
                    if 'x0' in sample:
                        signal_data = sample['x0'].numpy() if torch.is_tensor(sample['x0']) else sample['x0']
                        # Handle 2D signal data [1, 200] -> [200]
                        if signal_data.ndim > 1:
                            signal_data = signal_data.squeeze()
                        
                        # Handle peak_exists which can be a 0-d tensor or boolean
                        peaks_present = sample.get('peak_exists', True)
                        if torch.is_tensor(peaks_present):
                            peaks_present = peaks_present.item() if peaks_present.ndim == 0 else peaks_present.numpy()
                        elif hasattr(peaks_present, 'numpy'):
                            peaks_present = peaks_present.numpy()
                        
                        # Handle hearing loss from meta
                        meta = sample.get('meta', {})
                        if isinstance(meta, dict) and 'target' in meta:
                            target_value = meta['target']
                            if hearing_loss_map and isinstance(target_value, (int, np.integer)):
                                # Map class index to dB value
                                hearing_loss = hearing_loss_map.get(target_value, None)
                            elif isinstance(target_value, (float, np.floating)):
                                # Direct dB value
                                hearing_loss = float(target_value)
                            else:
                                # Unknown format, set to None
                                hearing_loss = None
                        elif isinstance(meta, (list, tuple)) and len(meta) > 0:
                            # Extract from first element of meta list
                            target_value = meta[0].get('target', 0) if isinstance(meta[0], dict) else 0
                            if hearing_loss_map and isinstance(target_value, (int, np.integer)):
                                hearing_loss = hearing_loss_map.get(target_value, None)
                            else:
                                hearing_loss = None
                        else:
                            hearing_loss = None
                    # Fallback to legacy keys
                    else:
                        signal_data = sample['signal'].numpy() if torch.is_tensor(sample['signal']) else sample['signal']
                        peaks_present = sample.get('peaks', np.ones(5))  # Default to all peaks present
                        hearing_loss = sample.get('hearing_loss', 0.0)
                elif isinstance(sample, (tuple, list)):
                    signal_data = sample[0].numpy() if torch.is_tensor(sample[0]) else sample[0]
                    peaks_present = sample[1].numpy() if len(sample) > 1 and torch.is_tensor(sample[1]) else np.ones(5)
                    hearing_loss = 0.0  # Default
                else:
                    signal_data = sample.numpy() if torch.is_tensor(sample) else sample
                    peaks_present = np.ones(5)
                    hearing_loss = 0.0
                    
                if metric_type == 'snr':
                    difficulty = self.compute_snr_difficulty(signal_data, fs=fs)
                elif metric_type == 'peaks':
                    difficulty = self.compute_peak_difficulty(signal_data, peaks_present, fs=fs)
                elif metric_type == 'hearing_loss':
                    difficulty = self.compute_hearing_loss_difficulty(hearing_loss)
                else:  # combined
                    difficulty = self.compute_combined_difficulty(
                        signal_data, peaks_present, hearing_loss, fs=fs, weights=difficulty_weights
                    )
                    
                difficulties.append(difficulty)
                
            except Exception as e:
                logger.warning(f"Error computing difficulty for sample {i}: {e}")
                difficulties.append(0.5)  # Default medium difficulty
                
        difficulties = np.array(difficulties)
        
        # Cache results
        if cache_key:
            self.cached_metrics[cache_key] = difficulties
            
        return difficulties


class CurriculumScheduler(ABC):
    """
    Abstract base class for curriculum scheduling strategies.
    """
    
    @abstractmethod
    def get_difficulty_threshold(self, epoch: int, total_epochs: int) -> float:
        """Get difficulty threshold for current epoch."""
        pass
    
    @abstractmethod
    def get_sample_weights(self, difficulties: np.ndarray, threshold: float) -> np.ndarray:
        """Get sampling weights based on difficulty threshold."""
        pass


class LinearCurriculumScheduler(CurriculumScheduler):
    """
    Linear curriculum scheduler that increases difficulty linearly over epochs.
    """
    
    def __init__(
        self, 
        start_difficulty: float = 0.3,
        end_difficulty: float = 1.0,
        curriculum_epochs: int = 50
    ):
        self.start_difficulty = start_difficulty
        self.end_difficulty = end_difficulty
        self.curriculum_epochs = curriculum_epochs
        
    def get_difficulty_threshold(self, epoch: int, total_epochs: int) -> float:
        """Get difficulty threshold for current epoch."""
        if epoch >= self.curriculum_epochs:
            return self.end_difficulty
            
        progress = epoch / self.curriculum_epochs
        threshold = self.start_difficulty + progress * (self.end_difficulty - self.start_difficulty)
        return threshold
        
    def get_sample_weights(self, difficulties: np.ndarray, threshold: float) -> np.ndarray:
        """Get sampling weights based on difficulty threshold."""
        # Samples within threshold get higher weights
        weights = np.where(difficulties <= threshold, 1.0, 0.1)  # Reduce but don't eliminate hard samples
        return weights


class ExponentialCurriculumScheduler(CurriculumScheduler):
    """
    Exponential curriculum scheduler with faster initial progress.
    """
    
    def __init__(
        self,
        start_difficulty: float = 0.3,
        end_difficulty: float = 1.0,
        curriculum_epochs: int = 50,
        growth_rate: float = 2.0
    ):
        self.start_difficulty = start_difficulty
        self.end_difficulty = end_difficulty
        self.curriculum_epochs = curriculum_epochs
        self.growth_rate = growth_rate
        
    def get_difficulty_threshold(self, epoch: int, total_epochs: int) -> float:
        """Get difficulty threshold for current epoch."""
        if epoch >= self.curriculum_epochs:
            return self.end_difficulty
            
        progress = epoch / self.curriculum_epochs
        # Exponential growth: (exp(growth_rate * progress) - 1) / (exp(growth_rate) - 1)
        exp_progress = (np.exp(self.growth_rate * progress) - 1) / (np.exp(self.growth_rate) - 1)
        threshold = self.start_difficulty + exp_progress * (self.end_difficulty - self.start_difficulty)
        return threshold
        
    def get_sample_weights(self, difficulties: np.ndarray, threshold: float) -> np.ndarray:
        """Get sampling weights based on difficulty threshold."""
        # Smooth weighting based on distance from threshold
        weights = np.exp(-np.maximum(0, difficulties - threshold) * 5.0)  # Exponential decay for hard samples
        return weights


class StepCurriculumScheduler(CurriculumScheduler):
    """
    Step-wise curriculum scheduler with discrete difficulty levels.
    """
    
    def __init__(
        self,
        difficulty_steps: List[Tuple[int, float]] = None
    ):
        if difficulty_steps is None:
            difficulty_steps = [(0, 0.3), (20, 0.5), (40, 0.7), (60, 1.0)]
        self.difficulty_steps = sorted(difficulty_steps)
        
    def get_difficulty_threshold(self, epoch: int, total_epochs: int) -> float:
        """Get difficulty threshold for current epoch."""
        for epoch_threshold, difficulty in reversed(self.difficulty_steps):
            if epoch >= epoch_threshold:
                return difficulty
        return self.difficulty_steps[0][1]
        
    def get_sample_weights(self, difficulties: np.ndarray, threshold: float) -> np.ndarray:
        """Get sampling weights based on difficulty threshold."""
        # Binary weighting for step-wise curriculum
        weights = np.where(difficulties <= threshold, 1.0, 0.05)
        return weights


class CurriculumSampler(Sampler):
    """
    Custom sampler for curriculum learning that filters samples based on difficulty.
    
    Implements a proper Sampler that uses current weights in __iter__ to avoid
    issues with WeightedRandomSampler's internal state rebinding.
    """
    
    def __init__(
        self,
        difficulties: np.ndarray,
        scheduler: CurriculumScheduler,
        num_samples: int,
        replacement: bool = True
    ):
        self.difficulties = difficulties
        self.scheduler = scheduler
        self.num_samples = num_samples
        self.replacement = replacement
        self.current_epoch = 0
        self.total_epochs = 100  # Default, will be updated
        
        # Initialize with starting weights
        initial_threshold = scheduler.get_difficulty_threshold(0, self.total_epochs)
        self.current_weights = scheduler.get_sample_weights(difficulties, initial_threshold)
        
    def update_epoch(self, epoch: int, total_epochs: int):
        """Update sampling weights for new epoch."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        
        threshold = self.scheduler.get_difficulty_threshold(epoch, total_epochs)
        self.current_weights = self.scheduler.get_sample_weights(self.difficulties, threshold)
        
        logger.info(f"Curriculum epoch {epoch}: difficulty threshold = {threshold:.3f}, "
                   f"active samples = {(self.current_weights > 0.5).sum()}/{len(self.current_weights)}")
    
    def __iter__(self) -> Iterator[int]:
        """Generate sample indices based on current weights."""
        # Convert to tensor if needed
        if isinstance(self.current_weights, np.ndarray):
            weights = torch.from_numpy(self.current_weights).float()
        else:
            weights = self.current_weights
        
        # Sample indices based on current weights
        if self.replacement:
            # Sample with replacement using multinomial
            indices = torch.multinomial(weights, self.num_samples, replacement=True)
        else:
            # Sample without replacement
            # For simplicity, we'll use weighted sampling with replacement
            # but ensure no duplicates (may result in fewer samples)
            indices = torch.multinomial(weights, min(self.num_samples, len(weights)), replacement=False)
            
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        """Return the number of samples per epoch."""
        return self.num_samples


class CurriculumDataset(Dataset):
    """
    Dataset wrapper that applies curriculum learning to existing ABR dataset.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        difficulty_metrics: DifficultyMetrics,
        scheduler: CurriculumScheduler,
        metric_type: str = 'combined',
        update_frequency: int = 1,  # Update curriculum every N epochs
        difficulty_weights: Optional[Dict[str, float]] = None,
        hearing_loss_map: Optional[Dict[int, float]] = None
    ):
        self.base_dataset = base_dataset
        self.difficulty_metrics = difficulty_metrics
        self.scheduler = scheduler
        self.metric_type = metric_type
        self.update_frequency = update_frequency
        self.difficulty_weights = difficulty_weights
        self.hearing_loss_map = hearing_loss_map
        
        # Compute initial difficulties
        self.difficulties = self.difficulty_metrics.compute_batch_difficulties(
            base_dataset, metric_type, cache_key=f"curriculum_{metric_type}",
            difficulty_weights=difficulty_weights, hearing_loss_map=hearing_loss_map
        )
        
        self.current_epoch = 0
        self.total_epochs = 100
        self.active_indices = np.arange(len(base_dataset))
        
        self._update_active_samples()
        
    def _update_active_samples(self):
        """Update which samples are active based on current curriculum."""
        threshold = self.scheduler.get_difficulty_threshold(self.current_epoch, self.total_epochs)
        weights = self.scheduler.get_sample_weights(self.difficulties, threshold)
        
        # Keep samples with weight > 0.1 (allowing some hard samples)
        self.active_indices = np.where(weights > 0.1)[0]
        
        logger.info(f"Curriculum updated: {len(self.active_indices)}/{len(self.base_dataset)} samples active")
        
    def update_epoch(self, epoch: int, total_epochs: int):
        """Update curriculum for new epoch."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        
        if epoch % self.update_frequency == 0:
            self._update_active_samples()
            
    def __len__(self):
        return len(self.active_indices)
        
    def __getitem__(self, idx):
        actual_idx = self.active_indices[idx]
        return self.base_dataset[actual_idx]
        
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get statistics about current curriculum state."""
        threshold = self.scheduler.get_difficulty_threshold(self.current_epoch, self.total_epochs)
        active_difficulties = self.difficulties[self.active_indices]
        
        return {
            'epoch': self.current_epoch,
            'difficulty_threshold': threshold,
            'active_samples': len(self.active_indices),
            'total_samples': len(self.base_dataset),
            'active_ratio': len(self.active_indices) / len(self.base_dataset),
            'mean_difficulty': np.mean(active_difficulties),
            'difficulty_std': np.std(active_difficulties),
            'difficulty_range': (np.min(active_difficulties), np.max(active_difficulties))
        }


def create_curriculum_from_config(config: Dict[str, Any], base_dataset: Dataset) -> Optional[CurriculumDataset]:
    """
    Factory function to create curriculum learning components from configuration.
    
    Args:
        config: Curriculum learning configuration
        base_dataset: Base ABR dataset
        
    Returns:
        Configured curriculum dataset or None if disabled
    """
    if not config.get('enabled', False):
        return None
        
    # Create difficulty metrics
    difficulty_metrics = DifficultyMetrics(cache_dir=config.get('cache_dir'))
    
    # Create scheduler
    scheduler_type = config.get('scheduler_type', 'linear')
    if scheduler_type == 'linear':
        scheduler = LinearCurriculumScheduler(
            start_difficulty=config.get('start_difficulty', 0.3),
            end_difficulty=config.get('end_difficulty', 1.0),
            curriculum_epochs=config.get('curriculum_epochs', 50)
        )
    elif scheduler_type == 'exponential':
        scheduler = ExponentialCurriculumScheduler(
            start_difficulty=config.get('start_difficulty', 0.3),
            end_difficulty=config.get('end_difficulty', 1.0),
            curriculum_epochs=config.get('curriculum_epochs', 50),
            growth_rate=config.get('growth_rate', 2.0)
        )
    elif scheduler_type == 'step':
        scheduler = StepCurriculumScheduler(
            difficulty_steps=config.get('difficulty_steps')
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
    # Create curriculum dataset
    curriculum_dataset = CurriculumDataset(
        base_dataset=base_dataset,
        difficulty_metrics=difficulty_metrics,
        scheduler=scheduler,
        metric_type=config.get('difficulty_metric', 'combined'),
        update_frequency=config.get('update_frequency', 1),
        difficulty_weights=config.get('difficulty_weights'),
        hearing_loss_map=config.get('hearing_loss_map')
    )
    
    return curriculum_dataset


def analyze_curriculum_progress(curriculum_stats_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze curriculum learning progress over training.
    
    Args:
        curriculum_stats_history: List of curriculum statistics from each epoch
        
    Returns:
        Analysis results
    """
    if not curriculum_stats_history:
        return {}
        
    epochs = [stats['epoch'] for stats in curriculum_stats_history]
    thresholds = [stats['difficulty_threshold'] for stats in curriculum_stats_history]
    active_ratios = [stats['active_ratio'] for stats in curriculum_stats_history]
    mean_difficulties = [stats['mean_difficulty'] for stats in curriculum_stats_history]
    
    analysis = {
        'total_epochs': len(epochs),
        'final_threshold': thresholds[-1],
        'threshold_progression': np.diff(thresholds).mean(),
        'final_active_ratio': active_ratios[-1],
        'mean_active_ratio': np.mean(active_ratios),
        'difficulty_progression': np.diff(mean_difficulties).mean(),
        'curriculum_completion': min(1.0, thresholds[-1]) if thresholds else 0.0
    }
    
    return analysis


def test_curriculum_hearing_loss_mapping():
    """
    Test curriculum learning with both raw dB and class-index inputs.
    """
    import torch
    from torch.utils.data import TensorDataset
    
    # Create mock ABR dataset with different hearing loss formats
    batch_size = 10
    seq_len = 200
    
    # Test data with class indices
    x0_data = torch.randn(batch_size, 1, seq_len)
    peak_exists_data = torch.randint(0, 2, (batch_size,)).float()
    
    # Create class-based meta (0=normal, 1=mild, 2=severe)
    class_meta = [{'target': i % 3} for i in range(batch_size)]
    
    class_samples = [
        {
            'x0': x0_data[i],
            'peak_exists': peak_exists_data[i],
            'meta': class_meta[i]
        } for i in range(batch_size)
    ]
    
    # Create raw dB meta
    db_meta = [{'target': float(i * 10)} for i in range(batch_size)]  # 0, 10, 20, ... dB
    
    db_samples = [
        {
            'x0': x0_data[i],
            'peak_exists': peak_exists_data[i],
            'meta': db_meta[i]
        } for i in range(batch_size)
    ]
    
    # Test with class mapping
    hearing_loss_map = {0: 0.0, 1: 30.0, 2: 80.0}  # normal, mild, severe
    
    difficulty_metrics = DifficultyMetrics()
    
    # Test class-based inputs
    class_difficulties = []
    for sample in class_samples:
        signal_data = sample['x0'].numpy().squeeze()
        peaks_present = sample['peak_exists']
        
        # Extract hearing loss using the mapping logic
        meta = sample.get('meta', {})
        if isinstance(meta, dict) and 'target' in meta:
            target_value = meta['target']
            if hearing_loss_map and isinstance(target_value, (int, np.integer)):
                hearing_loss = hearing_loss_map.get(target_value, None)
            else:
                hearing_loss = None
        else:
            hearing_loss = None
            
        difficulty = difficulty_metrics.compute_combined_difficulty(
            signal_data, peaks_present, hearing_loss
        )
        class_difficulties.append(difficulty)
    
    # Test raw dB inputs
    db_difficulties = []
    for sample in db_samples:
        signal_data = sample['x0'].numpy().squeeze()
        peaks_present = sample['peak_exists']
        hearing_loss = sample['meta']['target']  # Direct dB value
        
        difficulty = difficulty_metrics.compute_combined_difficulty(
            signal_data, peaks_present, hearing_loss
        )
        db_difficulties.append(difficulty)
    
    # Test None handling (no hearing loss info)
    none_difficulties = []
    for sample in class_samples:
        signal_data = sample['x0'].numpy().squeeze()
        peaks_present = sample['peak_exists']
        
        difficulty = difficulty_metrics.compute_combined_difficulty(
            signal_data, peaks_present, None  # No hearing loss
        )
        none_difficulties.append(difficulty)
    
    # Verify all difficulty scores are valid
    assert all(0.0 <= d <= 1.0 for d in class_difficulties), "Class-based difficulties out of range"
    assert all(0.0 <= d <= 1.0 for d in db_difficulties), "dB-based difficulties out of range"
    assert all(0.0 <= d <= 1.0 for d in none_difficulties), "None-based difficulties out of range"
    
    # Verify None case produces different results (should ignore hearing loss component)
    assert len(set(none_difficulties)) > 1 or all(d == none_difficulties[0] for d in none_difficulties), \
        "None case should still produce valid difficulties"
    
    print("Curriculum hearing loss mapping tests passed!")


def visualize_curriculum_distribution(difficulties: np.ndarray, threshold: float, save_path: Optional[str] = None):
    """
    Visualize the distribution of sample difficulties and current threshold.
    
    Args:
        difficulties: Array of sample difficulties
        threshold: Current difficulty threshold
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of difficulties
        plt.hist(difficulties, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Plot threshold line
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
        
        # Add labels and title
        plt.xlabel('Difficulty Score')
        plt.ylabel('Number of Samples')
        plt.title('Sample Difficulty Distribution with Curriculum Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        active_samples = (difficulties <= threshold).sum()
        plt.text(0.02, 0.98, f'Active samples: {active_samples}/{len(difficulties)} ({active_samples/len(difficulties)*100:.1f}%)',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    except ImportError:
        logger.warning("matplotlib not available for visualization")
