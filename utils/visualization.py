"""
Visualization utilities for ABR signals and training monitoring.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from typing import Optional, Tuple, List


def plot_signals_with_peaks(
    ground_truth: torch.Tensor,
    reconstructed: torch.Tensor,
    predicted_peaks: Optional[torch.Tensor] = None,
    target_peaks: Optional[torch.Tensor] = None,
    peak_mask: Optional[torch.Tensor] = None,
    sample_rate: float = 1000.0,
    title: str = "ABR Signal Reconstruction"
) -> plt.Figure:
    """
    Plot ground truth vs reconstructed ABR signals with optional peak overlays.
    
    Args:
        ground_truth (torch.Tensor): Ground truth signal [signal_length]
        reconstructed (torch.Tensor): Reconstructed signal [signal_length]
        predicted_peaks (torch.Tensor, optional): Predicted peak values [num_peaks]
        target_peaks (torch.Tensor, optional): Target peak values [num_peaks]
        peak_mask (torch.Tensor, optional): Mask for valid peaks [num_peaks]
        sample_rate (float): Sample rate for time axis
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Convert tensors to numpy
    gt_signal = ground_truth.cpu().detach().numpy()
    recon_signal = reconstructed.cpu().detach().numpy()
    
    # Create time axis
    time_axis = np.arange(len(gt_signal)) / sample_rate
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot signals
    axes[0].plot(time_axis, gt_signal, 'b-', label='Ground Truth', linewidth=1.5)
    axes[0].plot(time_axis, recon_signal, 'r--', label='Reconstructed', linewidth=1.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'{title} - Signal Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot difference
    difference = gt_signal - recon_signal
    axes[1].plot(time_axis, difference, 'g-', label='Difference (GT - Recon)', linewidth=1.0)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude Difference')
    axes[1].set_title('Reconstruction Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add peak information if available
    if predicted_peaks is not None and target_peaks is not None:
        pred_peaks = predicted_peaks.cpu().detach().numpy()
        tgt_peaks = target_peaks.cpu().detach().numpy()
        
        if peak_mask is not None:
            mask = peak_mask.cpu().detach().numpy()
        else:
            mask = np.ones_like(pred_peaks, dtype=bool)
        
        # Peak labels
        peak_labels = ['I Latency', 'III Latency', 'V Latency', 
                      'I Amplitude', 'III Amplitude', 'V Amplitude']
        
        # Add text box with peak information
        peak_text = "Peak Comparison:\n"
        for i, (pred, tgt, valid, label) in enumerate(zip(pred_peaks, tgt_peaks, mask, peak_labels)):
            if valid and not np.isnan(tgt):
                peak_text += f"{label}: Pred={pred:.3f}, GT={tgt:.3f}, Diff={abs(pred-tgt):.3f}\n"
            elif valid:
                peak_text += f"{label}: Pred={pred:.3f}, GT=NaN\n"
        
        # Add text box to the first subplot
        axes[0].text(0.02, 0.98, peak_text, transform=axes[0].transAxes, 
                    verticalalignment='top', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_training_progress(
    train_losses: List[dict],
    val_losses: List[dict],
    metrics: List[str] = ['total_loss', 'recon_loss', 'kl_loss', 'peak_loss']
) -> plt.Figure:
    """
    Plot training progress for multiple metrics.
    
    Args:
        train_losses (List[dict]): List of training loss dictionaries
        val_losses (List[dict]): List of validation loss dictionaries
        metrics (List[str]): List of metric names to plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    epochs = range(len(train_losses))
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            
            # Extract metric values
            train_values = [loss_dict.get(metric, 0) for loss_dict in train_losses]
            val_values = [loss_dict.get(metric, 0) for loss_dict in val_losses]
            
            ax.plot(epochs, train_values, 'b-', label=f'Train {metric}', linewidth=1.5)
            ax.plot(epochs, val_values, 'r--', label=f'Val {metric}', linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_latent_space_2d(
    latent_samples: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    title: str = "Latent Space Visualization"
) -> plt.Figure:
    """
    Plot 2D visualization of latent space (for latent_dim >= 2).
    
    Args:
        latent_samples (torch.Tensor): Latent samples [batch_size, latent_dim]
        labels (torch.Tensor, optional): Labels for coloring points
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    latent_np = latent_samples.cpu().detach().numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    if labels is not None:
        labels_np = labels.cpu().detach().numpy()
        scatter = ax.scatter(latent_np[:, 0], latent_np[:, 1], c=labels_np, 
                           cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Labels')
    else:
        ax.scatter(latent_np[:, 0], latent_np[:, 1], alpha=0.6, s=20)
    
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_beta_schedule(
    beta_values: List[float],
    title: str = "Beta Schedule"
) -> plt.Figure:
    """
    Plot beta annealing schedule.
    
    Args:
        beta_values (List[float]): List of beta values over epochs
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    epochs = range(len(beta_values))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(epochs, beta_values, 'b-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Beta Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(beta_values) * 1.1)
    
    plt.tight_layout()
    return fig


def plot_peak_predictions(
    predicted_peaks: torch.Tensor,
    target_peaks: torch.Tensor,
    peak_mask: torch.Tensor,
    peak_names: List[str] = None
) -> plt.Figure:
    """
    Plot predicted vs target peaks as scatter plot.
    
    Args:
        predicted_peaks (torch.Tensor): Predicted peak values [batch_size, num_peaks]
        target_peaks (torch.Tensor): Target peak values [batch_size, num_peaks]
        peak_mask (torch.Tensor): Mask for valid peaks [batch_size, num_peaks]
        peak_names (List[str], optional): Names for each peak type
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    pred_np = predicted_peaks.cpu().detach().numpy()
    tgt_np = target_peaks.cpu().detach().numpy()
    mask_np = peak_mask.cpu().detach().numpy()
    
    if peak_names is None:
        peak_names = ['I Latency', 'III Latency', 'V Latency', 
                     'I Amplitude', 'III Amplitude', 'V Amplitude']
    
    num_peaks = pred_np.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(num_peaks, len(axes))):
        ax = axes[i]
        
        # Get valid predictions for this peak
        valid_mask = mask_np[:, i] & ~np.isnan(tgt_np[:, i])
        if np.sum(valid_mask) > 0:
            pred_valid = pred_np[valid_mask, i]
            tgt_valid = tgt_np[valid_mask, i]
            
            # Scatter plot
            ax.scatter(tgt_valid, pred_valid, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(np.min(tgt_valid), np.min(pred_valid))
            max_val = max(np.max(tgt_valid), np.max(pred_valid))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            # Calculate correlation
            correlation = np.corrcoef(tgt_valid, pred_valid)[0, 1]
            ax.set_title(f'{peak_names[i]}\nCorr: {correlation:.3f}')
            
            ax.set_xlabel('Target')
            ax.set_ylabel('Predicted')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.set_title(f'{peak_names[i]}\nNo valid data')
            ax.set_xlabel('Target')
            ax.set_ylabel('Predicted')
    
    # Hide unused subplots
    for i in range(num_peaks, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig 