"""
Comprehensive metrics for training monitoring and evaluation.

Basic time-domain metrics plus advanced evaluation metrics for signal reconstruction quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Optional

# Configure logging for SNR calculation diagnostics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def validate_snr_inputs(x_hat: torch.Tensor, x: torch.Tensor) -> dict:
    """
    Validate input signals for SNR calculation and detect degenerate cases.
    
    Args:
        x_hat: Predicted signal [B, ...]
        x: Target signal [B, ...]
        
    Returns:
        Dictionary with validation results and flags
    """
    x_flat = x.view(-1)
    x_hat_flat = x_hat.view(-1)
    
    validation_info = {
        'target_all_zero': torch.all(x_flat == 0),
        'pred_all_zero': torch.all(x_hat_flat == 0),
        'target_constant': torch.std(x_flat) < 1e-8,
        'pred_constant': torch.std(x_hat_flat) < 1e-8,
        'target_variance': torch.var(x_flat).item(),
        'pred_variance': torch.var(x_hat_flat).item(),
        'signal_power': torch.mean(x_flat ** 2).item(),
        'noise_power': torch.mean((x_hat_flat - x_flat) ** 2).item()
    }
    
    validation_info['is_degenerate'] = (
        validation_info['target_all_zero'] or 
        validation_info['pred_all_zero'] or
        validation_info['target_constant'] or
        validation_info['signal_power'] < 1e-10
    )
    
    return validation_info


def compute_robust_snr(x_hat: torch.Tensor, x: torch.Tensor, 
                       eps: float = 1e-6, min_snr: float = -60.0, 
                       max_snr: float = 60.0, log_edge_cases: bool = True) -> torch.Tensor:
    """
    Robust Signal-to-Noise Ratio calculation with comprehensive edge case handling.
    
    Args:
        x_hat: Predicted signal [B, ...]
        x: Target signal [B, ...]
        eps: Enhanced epsilon for numerical stability (increased from 1e-8 to 1e-6)
        min_snr: Minimum SNR bound in dB
        max_snr: Maximum SNR bound in dB
        log_edge_cases: Whether to log edge case handling
        
    Returns:
        SNR in dB, bounded and validated
    """
    # Validate inputs
    validation_info = validate_snr_inputs(x_hat, x)
    
    if validation_info['is_degenerate']:
        if log_edge_cases:
            logger.warning(f"Degenerate SNR calculation detected: {validation_info}")
        
        # Handle degenerate cases
        if validation_info['target_all_zero'] and validation_info['pred_all_zero']:
            # Both signals are zero - perfect but meaningless match
            return torch.tensor(0.0, device=x.device)
        elif validation_info['target_all_zero']:
            # Target is zero - return minimum SNR
            return torch.tensor(min_snr, device=x.device)
        elif validation_info['signal_power'] < 1e-10:
            # Signal power too small - return minimum SNR
            return torch.tensor(min_snr, device=x.device)
    
    # Compute signal and noise power
    signal_power = torch.mean(x ** 2)
    noise_power = torch.mean((x_hat - x) ** 2)
    
    # Enhanced epsilon protection
    noise_power_protected = torch.clamp(noise_power, min=eps)
    signal_power_protected = torch.clamp(signal_power, min=eps)
    
    # Compute SNR with bounds checking
    snr_ratio = signal_power_protected / noise_power_protected
    snr_db = 10 * torch.log10(snr_ratio)
    
    # Apply bounds
    snr_bounded = torch.clamp(snr_db, min=min_snr, max=max_snr)
    
    # Log extreme cases
    if log_edge_cases and (snr_db < min_snr or snr_db > max_snr):
        logger.warning(f"SNR bounded: original={snr_db.item():.2f}, bounded={snr_bounded.item():.2f}")
    
    return snr_bounded


def snr_db(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Signal-to-Noise Ratio in dB with enhanced robustness.
    
    Args:
        x_hat: Predicted signal [B, ...]
        x: Target signal [B, ...]
        eps: Enhanced epsilon to avoid log(0) (increased from 1e-8 to 1e-6)
        
    Returns:
        SNR in dB, robustly calculated
    """
    return compute_robust_snr(x_hat, x, eps=eps, log_edge_cases=False)


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


# ===== ADVANCED EVALUATION METRICS =====

def stft_l1(x_hat: torch.Tensor, x: torch.Tensor, n_fft: int = 64, 
           hop_length: int = 16, win_length: int = 64) -> torch.Tensor:
    """
    STFT-based L1 loss for frequency domain comparison.
    
    Args:
        x_hat: Predicted signal [B, 1, T] or [B, T]
        x: Target signal [B, 1, T] or [B, T]
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        
    Returns:
        STFT L1 loss scalar
    """
    # Ensure input is [B, T] format for torch.stft
    if x_hat.dim() == 3:
        x_hat = x_hat.squeeze(1)
    if x.dim() == 3:
        x = x.squeeze(1)
    
    # Compute STFT
    device = x_hat.device
    window = torch.hann_window(win_length, device=device)
    
    stft_hat = torch.stft(x_hat, n_fft=n_fft, hop_length=hop_length, 
                         win_length=win_length, window=window, return_complex=True)
    stft_target = torch.stft(x, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, window=window, return_complex=True)
    
    # L1 on magnitude spectrograms
    mag_hat = torch.abs(stft_hat)
    mag_target = torch.abs(stft_target)
    
    return F.l1_loss(mag_hat, mag_target)


def pearson_r_batch(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson correlation per batch item, then averaged.
    
    Args:
        x_hat: Predicted signal [B, ...]
        x: Target signal [B, ...]
        eps: Small epsilon for numerical stability
        
    Returns:
        Average Pearson correlation across batch
    """
    batch_size = x_hat.shape[0]
    correlations = []
    
    for i in range(batch_size):
        x_hat_i = x_hat[i].flatten()
        x_i = x[i].flatten()
        
        # Center the signals
        x_hat_mean = torch.mean(x_hat_i)
        x_mean = torch.mean(x_i)
        
        x_hat_centered = x_hat_i - x_hat_mean
        x_centered = x_i - x_mean
        
        # Compute correlation
        numerator = torch.sum(x_hat_centered * x_centered)
        denominator = torch.sqrt(torch.sum(x_hat_centered ** 2) * torch.sum(x_centered ** 2))
        
        correlation = numerator / (denominator + eps)
        correlations.append(correlation)
    
    return torch.stack(correlations).mean()


def dtw_distance(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Dynamic Time Warping distance for 1D signals.
    
    Lightweight O(T²) implementation suitable for T=200.
    
    Args:
        x_hat: Predicted signal [B, T] or [B, 1, T]
        x: Target signal [B, T] or [B, 1, T]
        
    Returns:
        Average DTW distance across batch
    """
    # Ensure signals are [B, T]
    if x_hat.dim() == 3:
        x_hat = x_hat.squeeze(1)
    if x.dim() == 3:
        x = x.squeeze(1)
    
    batch_size, seq_len = x_hat.shape
    distances = []
    
    for b in range(batch_size):
        s1 = x_hat[b].cpu().numpy()
        s2 = x[b].cpu().numpy()
        
        # DTW dynamic programming
        dtw_matrix = np.full((seq_len + 1, seq_len + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, seq_len + 1):
            for j in range(1, seq_len + 1):
                cost = abs(s1[i-1] - s2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],     # insertion
                    dtw_matrix[i, j-1],     # deletion
                    dtw_matrix[i-1, j-1]    # match
                )
        
        distances.append(dtw_matrix[seq_len, seq_len])
    
    return torch.tensor(np.mean(distances), device=x_hat.device)


def snr_db_batch(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Signal-to-Noise Ratio computed per batch item with robust handling, then averaged.
    
    Args:
        x_hat: Predicted signal [B, ...]
        x: Target signal [B, ...]
        eps: Enhanced epsilon to avoid log(0) (increased from 1e-8 to 1e-6)
        
    Returns:
        Average SNR in dB across batch, robustly calculated
    """
    batch_size = x_hat.shape[0]
    snrs = []
    
    for i in range(batch_size):
        # Use robust SNR calculation for each batch item
        sample_snr = compute_robust_snr(
            x_hat[i:i+1], x[i:i+1], 
            eps=eps, log_edge_cases=False
        )
        snrs.append(sample_snr)
    
    return torch.stack(snrs).mean()


def compute_evaluation_metrics(x_hat: torch.Tensor, x: torch.Tensor, 
                             use_stft: bool = True, use_dtw: bool = True,
                             stft_params: Optional[dict] = None) -> dict:
    """
    Compute comprehensive evaluation metrics for ABR signal comparison.
    
    Args:
        x_hat: Predicted signals [B, 1, T] or [B, T]
        x: Target signals [B, 1, T] or [B, T]
        use_stft: Whether to compute STFT-based metrics
        use_dtw: Whether to compute DTW distance (can be slow)
        stft_params: Parameters for STFT computation
        
    Returns:
        Dictionary of metric values
    """
    if stft_params is None:
        stft_params = {'n_fft': 64, 'hop_length': 16, 'win_length': 64}
    
    metrics = {
        'mse': mse_time(x_hat, x).item(),
        'l1': l1_time(x_hat, x).item(),
        'corr': pearson_r_batch(x_hat, x).item(),
        'snr_db': snr_db_batch(x_hat, x).item(),
    }
    
    if use_stft:
        try:
            metrics['stft_l1'] = stft_l1(x_hat, x, **stft_params).item()
        except Exception as e:
            print(f"Warning: STFT computation failed: {e}")
            metrics['stft_l1'] = float('nan')
    
    if use_dtw:
        try:
            metrics['dtw'] = dtw_distance(x_hat, x).item()
        except Exception as e:
            print(f"Warning: DTW computation failed: {e}")
            metrics['dtw'] = float('nan')
    
    return metrics


def compute_per_sample_metrics(x_hat: torch.Tensor, x: torch.Tensor,
                              use_stft: bool = True, use_dtw: bool = True,
                              stft_params: Optional[dict] = None) -> list:
    """
    Compute metrics per sample in the batch with enhanced error handling.
    
    Args:
        x_hat: Predicted signals [B, 1, T] or [B, T]
        x: Target signals [B, 1, T] or [B, T]
        use_stft: Whether to compute STFT-based metrics
        use_dtw: Whether to compute DTW distance
        stft_params: Parameters for STFT computation
        
    Returns:
        List of dictionaries with per-sample metrics
    """
    if stft_params is None:
        stft_params = {'n_fft': 64, 'hop_length': 16, 'win_length': 64}
    
    batch_size = x_hat.shape[0]
    per_sample_metrics = []
    
    for i in range(batch_size):
        x_hat_i = x_hat[i:i+1]  # Keep batch dimension
        x_i = x[i:i+1]
        
        sample_metrics = {
            'mse': mse_time(x_hat_i, x_i).item(),
            'l1': l1_time(x_hat_i, x_i).item(),
            'corr': pearson_correlation(x_hat_i, x_i).item(),
        }
        
        # Robust SNR calculation with error handling
        try:
            snr_value = compute_robust_snr(x_hat_i, x_i, log_edge_cases=True).item()
            # Check for extreme values and log warnings
            if snr_value <= -50 or snr_value >= 50:
                logger.warning(f"Extreme SNR value {snr_value:.2f} dB in sample {i}")
            sample_metrics['snr_db'] = snr_value
        except Exception as e:
            logger.error(f"SNR calculation failed for sample {i}: {e}")
            sample_metrics['snr_db'] = float('nan')
        
        if use_stft:
            try:
                sample_metrics['stft_l1'] = stft_l1(x_hat_i, x_i, **stft_params).item()
            except Exception as e:
                logger.warning(f"STFT calculation failed for sample {i}: {e}")
                sample_metrics['stft_l1'] = float('nan')
        
        if use_dtw:
            try:
                sample_metrics['dtw'] = dtw_distance(x_hat_i, x_i).item()
            except Exception as e:
                logger.warning(f"DTW calculation failed for sample {i}: {e}")
                sample_metrics['dtw'] = float('nan')
        
        per_sample_metrics.append(sample_metrics)
    
    return per_sample_metrics


# ============================================================================
# Classification Metrics for Peak Detection
# ============================================================================

def binary_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Calculate binary classification accuracy.
    
    Args:
        logits: Model output logits [B] or [B, 1]
        targets: Binary targets [B] (0 or 1)
        threshold: Classification threshold (default: 0.0 for sigmoid)
        
    Returns:
        Accuracy as scalar tensor
    """
    # Ensure logits are 1D
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    
    # Convert logits to predictions
    predictions = (logits > threshold).float()
    
    # Calculate accuracy
    correct = (predictions == targets).float()
    accuracy = correct.mean()
    
    return accuracy


def binary_precision_recall_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0) -> dict:
    """
    Calculate binary classification precision, recall, and F1 score.
    
    Args:
        logits: Model output logits [B] or [B, 1]
        targets: Binary targets [B] (0 or 1)
        threshold: Classification threshold (default: 0.0 for sigmoid)
        
    Returns:
        Dictionary with precision, recall, f1 values
    """
    # Ensure logits are 1D
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    
    # Convert logits to predictions
    predictions = (logits > threshold).float()
    
    # Calculate confusion matrix components
    true_positives = ((predictions == 1) & (targets == 1)).float().sum()
    false_positives = ((predictions == 1) & (targets == 0)).float().sum()
    false_negatives = ((predictions == 0) & (targets == 1)).float().sum()
    true_negatives = ((predictions == 0) & (targets == 0)).float().sum()
    
    # Calculate metrics with small epsilon to avoid division by zero
    eps = 1e-8
    
    precision = true_positives / (true_positives + false_positives + eps)
    recall = true_positives / (true_positives + false_negatives + eps)
    
    # Calculate F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall + eps)
    else:
        f1 = torch.tensor(0.0, device=logits.device)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }


def auroc_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate Area Under ROC Curve (AUROC) score.
    
    Args:
        logits: Model output logits [B] or [B, 1]
        targets: Binary targets [B] (0 or 1)
        
    Returns:
        AUROC score as scalar tensor, or NaN if not computable
    """
    # Ensure logits are 1D
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    
    # Check if we have both classes
    unique_targets = torch.unique(targets)
    if len(unique_targets) < 2:
        return torch.tensor(float('nan'), device=logits.device)
    
    # Sort logits and targets by logit values (descending)
    sorted_indices = torch.argsort(logits, descending=True)
    sorted_logits = logits[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    # Calculate TPR and FPR at each threshold
    total_positives = targets.sum()
    total_negatives = (targets == 0).sum()
    
    if total_positives == 0 or total_negatives == 0:
        return torch.tensor(float('nan'), device=logits.device)
    
    # Initialize arrays
    tpr = torch.zeros(len(logits) + 1, device=logits.device)  # True Positive Rate
    fpr = torch.zeros(len(logits) + 1, device=logits.device)  # False Positive Rate
    
    # Calculate TPR and FPR at each threshold
    tp = 0
    fp = 0
    
    for i in range(len(sorted_logits)):
        if sorted_targets[i] == 1:
            tp += 1
        else:
            fp += 1
        
        tpr[i + 1] = tp / total_positives
        fpr[i + 1] = fp / total_negatives
    
    # Calculate AUROC using trapezoidal rule
    auroc = 0.0
    for i in range(len(tpr) - 1):
        auroc += (fpr[i + 1] - fpr[i]) * (tpr[i] + tpr[i + 1]) / 2
    
    return torch.tensor(auroc, device=logits.device)


def compute_classification_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0) -> dict:
    """
    Compute comprehensive classification metrics for binary classification.
    
    Args:
        logits: Model output logits [B] or [B, 1]
        targets: Binary targets [B] (0 or 1)
        threshold: Classification threshold (default: 0.0 for sigmoid)
        
    Returns:
        Dictionary with all classification metrics
    """
    # Basic metrics
    accuracy = binary_accuracy(logits, targets, threshold)
    precision_recall_f1 = binary_precision_recall_f1(logits, targets, threshold)
    auroc = auroc_score(logits, targets)
    
    # Combine all metrics
    metrics = {
        'accuracy': accuracy.item(),
        'precision': precision_recall_f1['precision'].item(),
        'recall': precision_recall_f1['recall'].item(),
        'f1': precision_recall_f1['f1'].item(),
        'auroc': auroc.item() if not torch.isnan(auroc) else float('nan'),
        'true_positives': precision_recall_f1['true_positives'].item(),
        'false_positives': precision_recall_f1['false_positives'].item(),
        'false_negatives': precision_recall_f1['false_negatives'].item(),
        'true_negatives': precision_recall_f1['true_negatives'].item()
    }
    
    return metrics
