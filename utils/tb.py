"""
TensorBoard plotting utilities for ABR signal visualization.

Creates matplotlib figures for logging waveforms and spectrograms.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import List, Optional


def plot_waveforms(batch: torch.Tensor, titles: Optional[List[str]] = None, 
                   max_plots: int = 8, figsize: tuple = (12, 8)) -> matplotlib.figure.Figure:
    """
    Plot ABR waveforms in a grid layout.
    
    Args:
        batch: Waveform tensor [N, 1, T] or [N, T]
        titles: Optional list of titles for each subplot
        max_plots: Maximum number of plots to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy and handle batch dimension
    if isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu().numpy()
    
    if batch.ndim == 3:
        batch = batch.squeeze(1)  # [N, 1, T] -> [N, T]
    
    n_signals = min(batch.shape[0], max_plots)
    
    # Create subplot grid
    ncols = min(4, n_signals)
    nrows = (n_signals + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle('ABR Waveforms', fontsize=14, fontweight='bold')
    
    # Time axis (assuming 10ms duration, 200 samples)
    time_ms = np.linspace(0, 10, batch.shape[1])
    
    for i in range(n_signals):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        # Plot waveform
        ax.plot(time_ms, batch[i], linewidth=1.5, color='navy')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.grid(True, alpha=0.3)
        
        # Set title
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
        else:
            ax.set_title(f'Signal {i+1}', fontsize=10)
        
        # Set y-axis limits for better visualization
        y_max = max(abs(np.min(batch[i])), abs(np.max(batch[i])))
        if y_max > 0:
            ax.set_ylim(-y_max * 1.1, y_max * 1.1)
    
    # Hide unused subplots
    for i in range(n_signals, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_spectrogram(batch: torch.Tensor, n_fft: int = 64, hop_length: int = 16, 
                    win_length: int = 64, max_plots: int = 8, 
                    figsize: tuple = (12, 8)) -> matplotlib.figure.Figure:
    """
    Plot spectrograms of ABR signals.
    
    Args:
        batch: Waveform tensor [N, 1, T] or [N, T]
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length  
        max_plots: Maximum number of plots to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy and handle batch dimension
    if isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu().numpy()
    
    if batch.ndim == 3:
        batch = batch.squeeze(1)  # [N, 1, T] -> [N, T]
    
    n_signals = min(batch.shape[0], max_plots)
    
    # Create subplot grid
    ncols = min(4, n_signals)
    nrows = (n_signals + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle('ABR Spectrograms', fontsize=14, fontweight='bold')
    
    for i in range(n_signals):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        # Compute spectrogram
        try:
            f, t, Sxx = plt.mlab.specgram(
                batch[i], 
                NFFT=n_fft,
                Fs=200/0.01,  # Assuming 10ms duration, 200 samples -> 20kHz
                noverlap=win_length - hop_length,
                window=np.hanning(win_length)
            )[:3]  # Get only f, t, Sxx
            
            # Convert to log scale
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            # Plot spectrogram
            im = ax.imshow(
                Sxx_db, 
                aspect='auto', 
                origin='lower',
                extent=[t[0]*1000, t[-1]*1000, f[0]/1000, f[-1]/1000],  # Convert to ms and kHz
                cmap='viridis'
            )
            
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Frequency (kHz)')
            ax.set_title(f'Spectrogram {i+1}', fontsize=10)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Power (dB)')
            
        except Exception:
            # If spectrogram computation fails, show empty plot
            ax.text(0.5, 0.5, 'Spectrogram\ncomputation failed', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Spectrogram {i+1} (failed)', fontsize=10)
    
    # Hide unused subplots
    for i in range(n_signals, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_comparison(pred_batch: torch.Tensor, target_batch: torch.Tensor, 
                   max_plots: int = 4, figsize: tuple = (14, 10)) -> matplotlib.figure.Figure:
    """
    Plot side-by-side comparison of predicted and target waveforms.
    
    Args:
        pred_batch: Predicted waveforms [N, 1, T] or [N, T]
        target_batch: Target waveforms [N, 1, T] or [N, T]
        max_plots: Maximum number of comparisons to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy and handle batch dimension
    if isinstance(pred_batch, torch.Tensor):
        pred_batch = pred_batch.detach().cpu().numpy()
    if isinstance(target_batch, torch.Tensor):
        target_batch = target_batch.detach().cpu().numpy()
    
    if pred_batch.ndim == 3:
        pred_batch = pred_batch.squeeze(1)
    if target_batch.ndim == 3:
        target_batch = target_batch.squeeze(1)
    
    n_signals = min(pred_batch.shape[0], target_batch.shape[0], max_plots)
    
    fig, axes = plt.subplots(n_signals, 1, figsize=figsize)
    if n_signals == 1:
        axes = [axes]
    
    fig.suptitle('Predicted vs Target ABR Waveforms', fontsize=14, fontweight='bold')
    
    # Time axis
    time_ms = np.linspace(0, 10, pred_batch.shape[1])
    
    for i in range(n_signals):
        ax = axes[i]
        
        # Plot both waveforms
        ax.plot(time_ms, target_batch[i], label='Target', linewidth=1.5, color='navy', alpha=0.8)
        ax.plot(time_ms, pred_batch[i], label='Predicted', linewidth=1.5, color='red', alpha=0.8)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Comparison {i+1}', fontsize=10)
        
        # Set consistent y-axis limits
        y_max = max(
            abs(np.min(target_batch[i])), abs(np.max(target_batch[i])),
            abs(np.min(pred_batch[i])), abs(np.max(pred_batch[i]))
        )
        if y_max > 0:
            ax.set_ylim(-y_max * 1.1, y_max * 1.1)
    
    plt.tight_layout()
    return fig


def close_figure(fig: matplotlib.figure.Figure):
    """
    Safely close a matplotlib figure to free memory.
    
    Args:
        fig: Matplotlib figure to close
    """
    plt.close(fig)
