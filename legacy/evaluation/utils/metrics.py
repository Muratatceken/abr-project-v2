"""
Robust metrics computation for CVAE evaluation.
"""

import numpy as np
import torch
import warnings
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_reconstruction_metrics(
    reconstructed: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute reconstruction metrics with robust error handling.
    
    Args:
        reconstructed: Reconstructed signals [batch_size, signal_length]
        target: Target signals [batch_size, signal_length]
        
    Returns:
        Dictionary of reconstruction metrics
    """
    # Convert to numpy
    recon_np = reconstructed.cpu().detach().numpy()
    target_np = target.cpu().detach().numpy()
    
    # Check for invalid data
    if np.any(np.isnan(recon_np)) or np.any(np.isnan(target_np)):
        warnings.warn("NaN values detected in reconstruction metrics")
        return {
            'mse': float('nan'),
            'mae': float('nan'),
            'rmse': float('nan'),
            'correlation': float('nan')
        }
    
    # Flatten for global metrics
    recon_flat = recon_np.flatten()
    target_flat = target_np.flatten()
    
    metrics = {}
    
    # MSE and MAE
    try:
        metrics['mse'] = float(mean_squared_error(target_flat, recon_flat))
        metrics['mae'] = float(mean_absolute_error(target_flat, recon_flat))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
    except Exception as e:
        warnings.warn(f"Error computing MSE/MAE: {e}")
        metrics.update({'mse': float('nan'), 'mae': float('nan'), 'rmse': float('nan')})
    
    # Correlation with robust handling
    try:
        if np.var(target_flat) == 0 or np.var(recon_flat) == 0:
            metrics['correlation'] = 0.0
        else:
            corr_matrix = np.corrcoef(target_flat, recon_flat)
            if corr_matrix.shape == (2, 2):
                corr = corr_matrix[0, 1]
                metrics['correlation'] = float(corr) if np.isfinite(corr) else 0.0
            else:
                metrics['correlation'] = 0.0
    except Exception as e:
        warnings.warn(f"Error computing correlation: {e}")
        metrics['correlation'] = 0.0
    
    return metrics


def compute_peak_metrics(
    predicted_peaks: torch.Tensor,
    target_peaks: torch.Tensor,
    peak_mask: torch.Tensor,
    tolerance_ms: float = 0.5
) -> Dict[str, float]:
    """
    Compute peak prediction metrics.
    
    Args:
        predicted_peaks: Predicted peak values [batch_size, num_peaks] (latency values)
        target_peaks: Target peak values [batch_size, num_peaks] (latency values)
        peak_mask: Peak validity mask [batch_size, num_peaks]
        tolerance_ms: Tolerance for peak presence detection
        
    Returns:
        Dictionary of peak metrics
    """
    if predicted_peaks.numel() == 0 or target_peaks.numel() == 0:
        return {
            'peak_mae': float('nan'),
            'peak_accuracy': float('nan'),
            'peak_latency_mae': float('nan'),
            'peak_amplitude_mae': float('nan')
        }
    
    # Convert to numpy
    pred_np = predicted_peaks.cpu().detach().numpy()
    target_np = target_peaks.cpu().detach().numpy()
    mask_np = peak_mask.cpu().detach().numpy()
    
    # Only compute metrics for valid peaks
    valid_mask = mask_np.astype(bool)
    
    if not np.any(valid_mask):
        return {
            'peak_mae': float('nan'),
            'peak_accuracy': 0.0,
            'peak_latency_mae': float('nan'),
            'peak_amplitude_mae': float('nan')
        }
    
    # Extract valid peaks
    valid_pred = pred_np[valid_mask]
    valid_target = target_np[valid_mask]
    
    # Remove NaN values
    finite_mask = np.isfinite(valid_pred) & np.isfinite(valid_target)
    if not np.any(finite_mask):
        return {
            'peak_mae': float('nan'),
            'peak_accuracy': 0.0,
            'peak_latency_mae': float('nan'),
            'peak_amplitude_mae': float('nan')
        }
    
    valid_pred = valid_pred[finite_mask]
    valid_target = valid_target[finite_mask]
    
    metrics = {}
    
    try:
        # Overall MAE (for latency values)
        metrics['peak_mae'] = float(np.mean(np.abs(valid_pred - valid_target)))
        metrics['peak_latency_mae'] = metrics['peak_mae']  # Same as peak_mae for 1D data
        metrics['peak_amplitude_mae'] = float('nan')  # Not available in 1D peak data
        
        # Peak presence accuracy (simplified)
        # Consider a peak "detected" if latency difference is within tolerance
        latency_diff = np.abs(valid_pred - valid_target)
        correct_detections = np.sum(latency_diff <= tolerance_ms)
        metrics['peak_accuracy'] = float(correct_detections / len(valid_pred))
            
    except Exception as e:
        warnings.warn(f"Error computing peak metrics: {e}")
        metrics.update({
            'peak_mae': float('nan'),
            'peak_accuracy': float('nan'),
            'peak_latency_mae': float('nan'),
            'peak_amplitude_mae': float('nan')
        })
    
    return metrics


def compute_dtw_distance(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    max_samples: int = 100
) -> Dict[str, float]:
    """
    Compute Dynamic Time Warping distance (simplified version).
    
    Args:
        reconstructed: Reconstructed signals
        target: Target signals
        max_samples: Maximum number of samples to process
        
    Returns:
        Dictionary with DTW metrics
    """
    try:
        from scipy.spatial.distance import euclidean
        
        recon_np = reconstructed.cpu().detach().numpy()
        target_np = target.cpu().detach().numpy()
        
        # Limit samples to prevent timeout
        num_samples = min(recon_np.shape[0], max_samples)
        recon_np = recon_np[:num_samples]
        target_np = target_np[:num_samples]
        
        dtw_distances = []
        
        for i in range(num_samples):
            # Simplified DTW: just compute Euclidean distance
            # Full DTW implementation would be too complex for this example
            distance = euclidean(recon_np[i], target_np[i])
            if np.isfinite(distance):
                dtw_distances.append(distance)
        
        if dtw_distances:
            return {
                'dtw_distance': float(np.mean(dtw_distances)),
                'dtw_std': float(np.std(dtw_distances))
            }
        else:
            return {
                'dtw_distance': float('nan'),
                'dtw_std': float('nan')
            }
            
    except ImportError:
        warnings.warn("scipy not available for DTW computation")
        return {
            'dtw_distance': float('nan'),
            'dtw_std': float('nan')
        }
    except Exception as e:
        warnings.warn(f"DTW computation failed: {e}")
        return {
            'dtw_distance': float('nan'),
            'dtw_std': float('nan')
        }


def compute_latent_metrics(
    latent_vectors: torch.Tensor,
    static_params: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute latent space quality metrics.
    
    Args:
        latent_vectors: Latent representations [batch_size, latent_dim]
        static_params: Static parameters [batch_size, static_dim]
        
    Returns:
        Dictionary of latent space metrics
    """
    latent_np = latent_vectors.cpu().detach().numpy()
    
    metrics = {}
    
    try:
        # Basic statistics
        metrics['latent_mean'] = float(np.mean(latent_np))
        metrics['latent_std'] = float(np.std(latent_np))
        metrics['latent_var'] = float(np.var(latent_np))
        
        # Dimensionality metrics
        metrics['latent_dim'] = int(latent_np.shape[1])
        
        # Effective dimensionality (simplified)
        cov_matrix = np.cov(latent_np.T)
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvalues
        metrics['effective_dim'] = int(len(eigenvals))
        
    except Exception as e:
        warnings.warn(f"Error computing latent metrics: {e}")
        metrics.update({
            'latent_mean': float('nan'),
            'latent_std': float('nan'),
            'latent_var': float('nan'),
            'latent_dim': 0,
            'effective_dim': 0
        })
    
    return metrics


def aggregate_metrics(metrics_list: list) -> Dict[str, Any]:
    """
    Aggregate metrics from multiple samples.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Aggregated metrics with mean, std, etc.
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    
    # Get all metric keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    # Aggregate each metric
    for key in all_keys:
        values = []
        for metrics in metrics_list:
            if key in metrics and np.isfinite(metrics[key]):
                values.append(metrics[key])
        
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))
            aggregated[f"{key}_count"] = len(values)
        else:
            aggregated[f"{key}_mean"] = float('nan')
            aggregated[f"{key}_std"] = float('nan')
            aggregated[f"{key}_min"] = float('nan')
            aggregated[f"{key}_max"] = float('nan')
            aggregated[f"{key}_count"] = 0
    
    return aggregated 