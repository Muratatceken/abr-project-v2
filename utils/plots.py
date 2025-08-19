"""
Advanced plotting utilities for ABR signal evaluation.

Creates matplotlib figures for detailed evaluation visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import List, Optional, Tuple


def overlay_waveforms(ref: torch.Tensor, gen: torch.Tensor, 
                     titles: Optional[List[str]] = None,
                     max_plots: int = 8, figsize: Tuple[int, int] = (14, 10)) -> matplotlib.figure.Figure:
    """
    Create overlay plots of reference and generated waveforms.
    
    Args:
        ref: Reference waveforms [N, 1, T] or [N, T]
        gen: Generated waveforms [N, 1, T] or [N, T]
        titles: Optional titles for each subplot
        max_plots: Maximum number of overlay plots
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy and handle dimensions
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()
    if isinstance(gen, torch.Tensor):
        gen = gen.detach().cpu().numpy()
    
    if ref.ndim == 3:
        ref = ref.squeeze(1)  # [N, 1, T] -> [N, T]
    if gen.ndim == 3:
        gen = gen.squeeze(1)
    
    n_signals = min(ref.shape[0], gen.shape[0], max_plots)
    
    # Create subplot grid
    ncols = min(3, n_signals)
    nrows = (n_signals + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle('Reference vs Generated ABR Waveforms', fontsize=16, fontweight='bold')
    
    # Time axis (assuming 10ms duration, 200 samples)
    time_ms = np.linspace(0, 10, ref.shape[1])
    
    for i in range(n_signals):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        # Plot overlaid waveforms
        ax.plot(time_ms, ref[i], label='Reference', linewidth=1.8, color='navy', alpha=0.8)
        ax.plot(time_ms, gen[i], label='Generated', linewidth=1.8, color='red', alpha=0.8, linestyle='--')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set title
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=11)
        else:
            ax.set_title(f'Sample {i+1}', fontsize=11)
        
        # Set consistent y-axis limits
        y_max = max(
            abs(np.min(ref[i])), abs(np.max(ref[i])),
            abs(np.min(gen[i])), abs(np.max(gen[i]))
        )
        if y_max > 0:
            ax.set_ylim(-y_max * 1.1, y_max * 1.1)
    
    # Hide unused subplots
    for i in range(n_signals, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig


def error_curve(ref: torch.Tensor, gen: torch.Tensor,
               titles: Optional[List[str]] = None,
               max_plots: int = 8, figsize: Tuple[int, int] = (14, 8)) -> matplotlib.figure.Figure:
    """
    Plot error curves |ref - gen| for analysis.
    
    Args:
        ref: Reference waveforms [N, 1, T] or [N, T]
        gen: Generated waveforms [N, 1, T] or [N, T]
        titles: Optional titles for each subplot
        max_plots: Maximum number of error plots
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy and handle dimensions
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()
    if isinstance(gen, torch.Tensor):
        gen = gen.detach().cpu().numpy()
    
    if ref.ndim == 3:
        ref = ref.squeeze(1)
    if gen.ndim == 3:
        gen = gen.squeeze(1)
    
    n_signals = min(ref.shape[0], gen.shape[0], max_plots)
    
    # Compute error
    error = np.abs(ref - gen)
    
    # Create subplot grid
    ncols = min(4, n_signals)
    nrows = (n_signals + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle('Reconstruction Error |Reference - Generated|', fontsize=16, fontweight='bold')
    
    # Time axis
    time_ms = np.linspace(0, 10, ref.shape[1])
    
    for i in range(n_signals):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        # Plot error curve
        ax.plot(time_ms, error[i], linewidth=1.5, color='darkred')
        ax.fill_between(time_ms, 0, error[i], alpha=0.3, color='red')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('|Error| (µV)')
        ax.grid(True, alpha=0.3)
        
        # Set title with error statistics
        mse = np.mean(error[i] ** 2)
        mae = np.mean(error[i])
        if titles and i < len(titles):
            title = f'{titles[i]}\nMSE: {mse:.4f}, MAE: {mae:.4f}'
        else:
            title = f'Error {i+1}\nMSE: {mse:.4f}, MAE: {mae:.4f}'
        ax.set_title(title, fontsize=10)
    
    # Hide unused subplots
    for i in range(n_signals, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig


def spectrograms(batch: torch.Tensor, n_fft: int = 64, hop_length: int = 16, 
                win_length: int = 64, max_plots: int = 6,
                figsize: Tuple[int, int] = (12, 8)) -> matplotlib.figure.Figure:
    """
    Plot spectrograms of ABR signals.
    
    Args:
        batch: Waveform tensor [N, 1, T] or [N, T]
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        max_plots: Maximum number of spectrogram plots
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy and handle dimensions
    if isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu().numpy()
    
    if batch.ndim == 3:
        batch = batch.squeeze(1)
    
    n_signals = min(batch.shape[0], max_plots)
    
    # Create subplot grid
    ncols = min(3, n_signals)
    nrows = (n_signals + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle('ABR Signal Spectrograms', fontsize=16, fontweight='bold')
    
    for i in range(n_signals):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        try:
            # Compute spectrogram using matplotlib
            Pxx, freqs, bins, im = ax.specgram(
                batch[i],
                NFFT=n_fft,
                Fs=200/0.01,  # 20kHz sampling rate for 10ms, 200 samples
                noverlap=win_length - hop_length,
                window=np.hanning(win_length),
                cmap='viridis'
            )
            
            # Convert frequency to kHz and time to ms
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Frequency (kHz)')
            ax.set_title(f'Spectrogram {i+1}', fontsize=11)
            
            # Scale axes
            ax.set_xlim(0, 10)  # 10ms duration
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Power (dB)')
            
        except Exception as e:
            # If spectrogram computation fails, show text
            ax.text(0.5, 0.5, f'Spectrogram\nfailed:\n{str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'Spectrogram {i+1} (failed)', fontsize=11)
    
    # Hide unused subplots
    for i in range(n_signals, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig


def scatter_xy(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str,
              figsize: Tuple[int, int] = (8, 6)) -> matplotlib.figure.Figure:
    """
    Create scatter plot for metric analysis.
    
    Args:
        x: X-axis values
        y: Y-axis values
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    ax.scatter(x, y, alpha=0.6, s=30, color='steelblue', edgecolors='navy', linewidth=0.5)
    
    # Add regression line if enough points
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.3f}')
        ax.legend()
    
    # Compute correlation
    if len(x) > 1:
        correlation = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def metrics_summary_plot(metrics_dict: dict, mode: str,
                        figsize: Tuple[int, int] = (12, 8)) -> matplotlib.figure.Figure:
    """
    Create summary plot of evaluation metrics.
    
    Args:
        metrics_dict: Dictionary with metric names as keys and (mean, std) tuples as values
        mode: Evaluation mode ('reconstruction' or 'generation')
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    metric_names = list(metrics_dict.keys())
    means = [metrics_dict[name][0] for name in metric_names]
    stds = [metrics_dict[name][1] for name in metric_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot with error bars
    x_pos = np.arange(len(metric_names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                  color='steelblue', edgecolor='navy', linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Metric Values', fontsize=12)
    ax.set_title(f'Evaluation Metrics Summary - {mode.title()} Mode', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.4f}±{std:.4f}',
               ha='center', va='bottom', fontsize=10, rotation=0)
    
    plt.tight_layout()
    return fig


def close_figure(fig: matplotlib.figure.Figure):
    """
    Safely close a matplotlib figure to free memory.
    
    Args:
        fig: Matplotlib figure to close
    """
    plt.close(fig)
