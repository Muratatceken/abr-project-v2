"""
Plotting utilities for CVAE evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
import torch
import os
import warnings

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    # Fallback to available seaborn style
    plt.style.use('seaborn')
sns.set_palette("husl")


def plot_reconstruction_comparison(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_path: str,
    num_samples: int = 8,
    predicted_peaks: Optional[torch.Tensor] = None,
    target_peaks: Optional[torch.Tensor] = None,
    peak_mask: Optional[torch.Tensor] = None
) -> None:
    """
    Plot original vs reconstructed signals with optional peak overlays.
    
    Args:
        original: Original signals [batch_size, signal_length]
        reconstructed: Reconstructed signals [batch_size, signal_length]
        save_path: Path to save the plot
        num_samples: Number of samples to plot
        predicted_peaks: Predicted peaks [batch_size, num_peaks, 2]
        target_peaks: Target peaks [batch_size, num_peaks, 2]
        peak_mask: Peak validity mask [batch_size, num_peaks]
    """
    # Convert to numpy
    orig_np = original.cpu().detach().numpy()
    recon_np = reconstructed.cpu().detach().numpy()
    
    # Limit number of samples
    num_samples = min(num_samples, orig_np.shape[0])
    orig_np = orig_np[:num_samples]
    recon_np = recon_np[:num_samples]
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Plot signals
        time_axis = np.arange(orig_np.shape[1])
        ax.plot(time_axis, orig_np[i], 'b-', label='Original', alpha=0.7)
        ax.plot(time_axis, recon_np[i], 'r--', label='Reconstructed', alpha=0.7)
        
        # Plot peaks if available
        if (predicted_peaks is not None and target_peaks is not None and 
            peak_mask is not None and i < predicted_peaks.shape[0]):
            
            pred_peaks_np = predicted_peaks[i].cpu().detach().numpy()
            target_peaks_np = target_peaks[i].cpu().detach().numpy()
            mask_np = peak_mask[i].cpu().detach().numpy().astype(bool)
            
            # Plot valid peaks
            for j in range(len(mask_np)):
                if mask_np[j] and j < len(pred_peaks_np) and j < len(target_peaks_np):
                    # Target peak (convert from normalized to sample index)
                    if np.isfinite(target_peaks_np[j]):
                        latency_idx = int(target_peaks_np[j] * orig_np.shape[1] / 100)  # Assuming peaks are in ms, convert to samples
                        if 0 <= latency_idx < orig_np.shape[1]:
                            ax.axvline(latency_idx, color='blue', linestyle=':', alpha=0.5, label='Target Peak' if j == 0 else "")
                    
                    # Predicted peak
                    if np.isfinite(pred_peaks_np[j]):
                        latency_idx = int(pred_peaks_np[j] * orig_np.shape[1] / 100)  # Assuming peaks are in ms, convert to samples
                        if 0 <= latency_idx < orig_np.shape[1]:
                            ax.axvline(latency_idx, color='red', linestyle=':', alpha=0.5, label='Predicted Peak' if j == 0 else "")
        
        ax.set_title(f'Sample {i+1}')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_latent_space_2d(
    latent_vectors: torch.Tensor,
    static_params: torch.Tensor,
    save_path: str,
    method: str = 'pca',
    color_by: str = 'intensity',
    static_param_names: Optional[List[str]] = None
) -> None:
    """
    Plot 2D latent space visualization.
    
    Args:
        latent_vectors: Latent representations [batch_size, latent_dim]
        static_params: Static parameters [batch_size, static_dim]
        save_path: Path to save the plot
        method: Dimensionality reduction method ('pca' or 'tsne')
        color_by: Parameter to color by
        static_param_names: Names of static parameters
    """
    try:
        latent_np = latent_vectors.cpu().detach().numpy()
        static_np = static_params.cpu().detach().numpy()
        
        # Perform dimensionality reduction
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            latent_2d = reducer.fit_transform(latent_np)
            title_suffix = f"PCA (explained variance: {reducer.explained_variance_ratio_.sum():.2f})"
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, latent_np.shape[0]-1))
            latent_2d = reducer.fit_transform(latent_np)
            title_suffix = "t-SNE"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Determine color values
        if static_param_names and color_by in static_param_names:
            color_idx = static_param_names.index(color_by)
            color_values = static_np[:, color_idx]
        elif color_by == 'intensity' and static_np.shape[1] > 0:
            color_values = static_np[:, 0]  # Assume first parameter is intensity
        else:
            color_values = np.arange(len(latent_2d))  # Default to sample index
        
        # Create plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                            c=color_values, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label=color_by)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'Latent Space Visualization ({title_suffix})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError as e:
        warnings.warn(f"Required library not available for {method}: {e}")
    except Exception as e:
        warnings.warn(f"Error creating latent space plot: {e}")


def plot_generation_samples(
    generated_samples: torch.Tensor,
    static_params: torch.Tensor,
    save_path: str,
    samples_per_condition: int = 3
) -> None:
    """
    Plot generated samples for different conditions.
    
    Args:
        generated_samples: Generated signals [batch_size, signal_length]
        static_params: Static parameters [batch_size, static_dim]
        save_path: Path to save the plot
        samples_per_condition: Number of samples per condition
    """
    gen_np = generated_samples.cpu().detach().numpy()
    static_np = static_params.cpu().detach().numpy()
    
    # Group by conditions (simplified - group by first static parameter)
    if static_np.shape[1] > 0:
        unique_conditions = np.unique(np.round(static_np[:, 0], 1))
        num_conditions = min(len(unique_conditions), 5)  # Limit to 5 conditions
        
        fig, axes = plt.subplots(num_conditions, 1, figsize=(12, 3*num_conditions))
        if num_conditions == 1:
            axes = [axes]
        
        for i, condition in enumerate(unique_conditions[:num_conditions]):
            # Find samples for this condition
            condition_mask = np.abs(static_np[:, 0] - condition) < 0.1
            condition_samples = gen_np[condition_mask]
            
            if len(condition_samples) > 0:
                # Plot multiple samples for this condition
                time_axis = np.arange(condition_samples.shape[1])
                num_plot = min(samples_per_condition, len(condition_samples))
                
                for j in range(num_plot):
                    axes[i].plot(time_axis, condition_samples[j], alpha=0.7)
                
                # Plot mean
                mean_signal = np.mean(condition_samples[:num_plot], axis=0)
                axes[i].plot(time_axis, mean_signal, 'k-', linewidth=2, label='Mean')
                
                axes[i].set_title(f'Condition {condition:.1f} ({num_plot} samples)')
                axes[i].set_xlabel('Time (samples)')
                axes[i].set_ylabel('Amplitude')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_peak_analysis(
    predicted_peaks: torch.Tensor,
    target_peaks: torch.Tensor,
    peak_mask: torch.Tensor,
    save_path: str
) -> None:
    """
    Plot peak prediction analysis.
    
    Args:
        predicted_peaks: Predicted peaks [batch_size, num_peaks] (latency values)
        target_peaks: Target peaks [batch_size, num_peaks] (latency values)
        peak_mask: Peak validity mask [batch_size, num_peaks]
        save_path: Path to save the plot
    """
    pred_np = predicted_peaks.cpu().detach().numpy()
    target_np = target_peaks.cpu().detach().numpy()
    mask_np = peak_mask.cpu().detach().numpy().astype(bool)
    
    # Extract valid peaks
    valid_pred = pred_np[mask_np]
    valid_target = target_np[mask_np]
    
    # Remove NaN values
    finite_mask = np.isfinite(valid_pred) & np.isfinite(valid_target)
    valid_pred = valid_pred[finite_mask]
    valid_target = valid_target[finite_mask]
    
    if len(valid_pred) == 0:
        # Create empty plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No valid peaks to plot', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Peak Analysis - No Valid Peaks')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Create plot for latency analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Latency scatter plot
    axes[0].scatter(valid_target, valid_pred, alpha=0.6)
    axes[0].plot([valid_target.min(), valid_target.max()], 
                [valid_target.min(), valid_target.max()], 'r--', alpha=0.8)
    axes[0].set_xlabel('Target Latency (ms)')
    axes[0].set_ylabel('Predicted Latency (ms)')
    axes[0].set_title('Peak Latency Prediction')
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = valid_pred - valid_target
    axes[1].scatter(valid_target, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1].set_xlabel('Target Latency (ms)')
    axes[1].set_ylabel('Residuals (Predicted - Target)')
    axes[1].set_title('Peak Prediction Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_summary(
    metrics: Dict[str, Any],
    save_path: str
) -> None:
    """
    Plot summary of evaluation metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the plot
    """
    # Extract reconstruction metrics
    recon_metrics = {}
    peak_metrics = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and np.isfinite(value):
            if 'mse' in key or 'mae' in key or 'rmse' in key or 'correlation' in key:
                recon_metrics[key] = value
            elif 'peak' in key:
                peak_metrics[key] = value
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reconstruction metrics
    if recon_metrics:
        keys = list(recon_metrics.keys())
        values = list(recon_metrics.values())
        
        axes[0].bar(keys, values)
        axes[0].set_title('Reconstruction Metrics')
        axes[0].set_ylabel('Value')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[0].text(i, v + max(values) * 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Peak metrics
    if peak_metrics:
        keys = list(peak_metrics.keys())
        values = list(peak_metrics.values())
        
        axes[1].bar(keys, values)
        axes[1].set_title('Peak Prediction Metrics')
        axes[1].set_ylabel('Value')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[1].text(i, v + max(values) * 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 