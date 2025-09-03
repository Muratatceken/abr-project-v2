"""
Comprehensive metrics for training monitoring and evaluation.

Basic time-domain metrics plus advanced evaluation metrics for signal reconstruction quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


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


def snr_db_batch(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Signal-to-Noise Ratio computed per batch item, then averaged.
    
    Args:
        x_hat: Predicted signal [B, ...]
        x: Target signal [B, ...]
        eps: Small epsilon to avoid log(0)
        
    Returns:
        Average SNR in dB across batch
    """
    batch_size = x_hat.shape[0]
    snrs = []
    
    for i in range(batch_size):
        signal_power = torch.mean(x[i] ** 2)
        noise_power = torch.mean((x_hat[i] - x[i]) ** 2)
        
        snr = 10 * torch.log10(signal_power / (noise_power + eps))
        snrs.append(snr)
    
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
    Compute metrics per sample in the batch.
    
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
            'snr_db': snr_db(x_hat_i, x_i).item(),
        }
        
        if use_stft:
            try:
                sample_metrics['stft_l1'] = stft_l1(x_hat_i, x_i, **stft_params).item()
            except:
                sample_metrics['stft_l1'] = float('nan')
        
        if use_dtw:
            try:
                sample_metrics['dtw'] = dtw_distance(x_hat_i, x_i).item()
            except:
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
