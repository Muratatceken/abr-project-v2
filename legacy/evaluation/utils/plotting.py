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
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to numpy
    orig_np = original.cpu().detach().numpy()
    recon_np = reconstructed.cpu().detach().numpy()
    
    # Limit number of samples to what's actually available
    available_samples = orig_np.shape[0]
    num_samples = min(num_samples, available_samples)
    
    # Also check peak data if available
    if predicted_peaks is not None:
        num_samples = min(num_samples, predicted_peaks.shape[0])
    if target_peaks is not None:
        num_samples = min(num_samples, target_peaks.shape[0])
    if peak_mask is not None:
        num_samples = min(num_samples, peak_mask.shape[0])
    
    orig_np = orig_np[:num_samples]
    recon_np = recon_np[:num_samples]
    
    # Create subplots - adjust grid based on number of samples
    if num_samples <= 4:
        rows, cols = 1, num_samples
        figsize = (4 * num_samples, 4)
    elif num_samples <= 8:
        rows, cols = 2, 4
        figsize = (16, 8)
    else:
        rows, cols = 3, 4
        figsize = (16, 12)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_samples == 1:
        axes = [axes]  # Make it iterable
    else:
        axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Plot signals
        time_axis = np.arange(orig_np.shape[1])
        ax.plot(time_axis, orig_np[i], 'b-', label='Original', alpha=0.7)
        ax.plot(time_axis, recon_np[i], 'r--', label='Reconstructed', alpha=0.7)
        
        # Plot peaks if available
        if (predicted_peaks is not None and target_peaks is not None and 
            peak_mask is not None and i < predicted_peaks.shape[0] and 
            i < target_peaks.shape[0] and i < peak_mask.shape[0]):
            
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
    static_param_names: Optional[List[str]] = None,
    max_samples: int = 5000
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
        max_samples: Maximum number of samples to use for visualization
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        latent_np = latent_vectors.cpu().detach().numpy()
        static_np = static_params.cpu().detach().numpy()
        
        # Limit samples for visualization efficiency and memory management
        if len(latent_np) > max_samples:
            print(f"Limiting visualization to {max_samples} samples (from {len(latent_np)})")
            indices = np.random.choice(len(latent_np), max_samples, replace=False)
            latent_np = latent_np[indices]
            static_np = static_np[indices]
        
        # Perform dimensionality reduction
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            latent_2d = reducer.fit_transform(latent_np)
            title_suffix = f"PCA (explained variance: {reducer.explained_variance_ratio_.sum():.2f})"
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            # Limit perplexity for t-SNE stability
            perplexity = min(30, max(5, latent_np.shape[0] // 4))
            print(f"Using t-SNE with perplexity={perplexity} for {len(latent_np)} samples")
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                          n_iter=1000, init='pca', learning_rate='auto')
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
                            c=color_values, cmap='viridis', alpha=0.6, s=20)
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
        # Create a fallback plot
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, f'Required library not available for {method}:\n{str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title(f'Latent Space Visualization - {method} Not Available')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating latent space plot: {e}")
        warnings.warn(f"Error creating latent space plot: {e}")
        # Create a fallback plot
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, f'Error creating latent space visualization:\n{str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Latent Space Visualization - Error')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


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
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
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
        else:
            # Fallback if no static parameters
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            time_axis = np.arange(gen_np.shape[1])
            for i in range(min(samples_per_condition, len(gen_np))):
                ax.plot(time_axis, gen_np[i], alpha=0.7)
            ax.set_title('Generated Samples')
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating generation samples plot: {e}")
        # Create a fallback plot
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Error creating generation samples:\n{str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Generation Samples - Error')
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
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Extract reconstruction metrics
        recon_metrics = {}
        peak_metrics = {}
        latent_metrics = {}
        
        # Handle nested structure
        for section_key, section_value in metrics.items():
            if isinstance(section_value, dict):
                for key, value in section_value.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        if section_key == 'reconstruction':
                            if 'mean' in key:  # Focus on mean values
                                clean_key = key.replace('_mean', '')
                                recon_metrics[clean_key] = value
                        elif section_key == 'peaks':
                            if 'accuracy' in key or 'mae' in key:
                                peak_metrics[key] = value
                        elif section_key == 'latent':
                            if 'effective_dim' in key:
                                latent_metrics[key] = value
            elif isinstance(section_value, (int, float)) and np.isfinite(section_value):
                # Handle flat structure
                if 'mse' in section_key or 'mae' in section_key or 'rmse' in section_key or 'correlation' in section_key:
                    recon_metrics[section_key] = section_value
                elif 'peak' in section_key:
                    peak_metrics[section_key] = section_value
        
        # Create subplots
        num_plots = sum([bool(recon_metrics), bool(peak_metrics), bool(latent_metrics)])
        if num_plots == 0:
            # Create empty plot with message
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No metrics available for plotting', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Metrics Summary')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Reconstruction metrics
        if recon_metrics:
            keys = list(recon_metrics.keys())
            values = list(recon_metrics.values())
            
            axes[plot_idx].bar(keys, values, color='skyblue')
            axes[plot_idx].set_title('Reconstruction Metrics')
            axes[plot_idx].set_ylabel('Value')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[plot_idx].text(i, v + max(values) * 0.01, f'{v:.3f}', 
                                   ha='center', va='bottom', fontsize=10)
            plot_idx += 1
        
        # Peak metrics
        if peak_metrics:
            keys = list(peak_metrics.keys())
            values = list(peak_metrics.values())
            
            axes[plot_idx].bar(keys, values, color='lightcoral')
            axes[plot_idx].set_title('Peak Prediction Metrics')
            axes[plot_idx].set_ylabel('Value')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[plot_idx].text(i, v + max(values) * 0.01, f'{v:.3f}', 
                                   ha='center', va='bottom', fontsize=10)
            plot_idx += 1
        
        # Latent metrics
        if latent_metrics:
            keys = list(latent_metrics.keys())
            values = list(latent_metrics.values())
            
            axes[plot_idx].bar(keys, values, color='lightgreen')
            axes[plot_idx].set_title('Latent Space Metrics')
            axes[plot_idx].set_ylabel('Value')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[plot_idx].text(i, v + max(values) * 0.01, f'{v:.0f}', 
                                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating metrics summary plot: {e}")
        # Create fallback plot
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Error creating metrics summary:\n{str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Metrics Summary - Error')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_latent_static_correlations(
    z: np.ndarray,
    static_params: np.ndarray,
    save_path: str = None,
    title: str = "Latent-Static Correlations"
) -> None:
    """
    Plot correlation matrix between latent dimensions and static parameters.
    
    Args:
        z (np.ndarray): Latent vectors [n_samples, latent_dim]
        static_params (np.ndarray): Static parameters [n_samples, static_dim]
        save_path (str, optional): Path to save the plot
        title (str): Plot title
    """
    latent_dim = z.shape[1]
    static_dim = static_params.shape[1]
    
    # Compute correlation matrix
    correlations = np.zeros((latent_dim, static_dim))
    for i in range(latent_dim):
        for j in range(static_dim):
            corr_coef = np.corrcoef(z[:, i], static_params[:, j])[0, 1]
            correlations[i, j] = corr_coef if not np.isnan(corr_coef) else 0.0
    
    # Create plot
    plt.figure(figsize=(max(8, static_dim), max(6, latent_dim // 2)))
    
    # Plot heatmap
    im = plt.imshow(np.abs(correlations), cmap='viridis', aspect='auto')
    plt.colorbar(im, label='|Correlation Coefficient|')
    
    # Add correlation values as text
    for i in range(min(latent_dim, 20)):  # Limit text for readability
        for j in range(static_dim):
            plt.text(j, i, f'{correlations[i, j]:.2f}', 
                    ha='center', va='center', color='white', fontsize=8)
    
    plt.title(title)
    plt.xlabel('Static Parameter Index')
    plt.ylabel('Latent Dimension Index')
    plt.xticks(range(static_dim), [f'S{i}' for i in range(static_dim)])
    
    # Limit y-axis labels for readability
    y_ticks = range(0, latent_dim, max(1, latent_dim // 20))
    plt.yticks(y_ticks, [f'Z{i}' for i in y_ticks])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_static_reconstruction_accuracy(
    recon_static: np.ndarray,
    target_static: np.ndarray,
    save_path: str = None,
    title: str = "Static Parameter Reconstruction Accuracy"
) -> None:
    """
    Plot static parameter reconstruction accuracy.
    
    Args:
        recon_static (np.ndarray): Reconstructed static parameters [n_samples, static_dim]
        target_static (np.ndarray): Target static parameters [n_samples, static_dim]
        save_path (str, optional): Path to save the plot
        title (str): Plot title
    """
    static_dim = recon_static.shape[1]
    errors = np.abs(recon_static - target_static)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
    
    # 1. Box plot of errors per parameter
    axes[0, 0].boxplot([errors[:, i] for i in range(static_dim)], 
                       labels=[f'S{i}' for i in range(static_dim)])
    axes[0, 0].set_title('Reconstruction Errors by Parameter')
    axes[0, 0].set_ylabel('Absolute Error')
    axes[0, 0].set_xlabel('Static Parameter')
    
    # 2. Scatter plot: target vs reconstructed
    for i in range(min(static_dim, 4)):  # Show first 4 parameters
        row = i // 2
        col = 1 if i < 2 else 0
        if i >= 2:
            row = 1
            col = i - 2
        
        if i < static_dim:
            axes[row, col].scatter(target_static[:, i], recon_static[:, i], alpha=0.6)
            axes[row, col].plot([target_static[:, i].min(), target_static[:, i].max()], 
                               [target_static[:, i].min(), target_static[:, i].max()], 
                               'r--', label='Perfect reconstruction')
            axes[row, col].set_xlabel(f'Target Static {i}')
            axes[row, col].set_ylabel(f'Reconstructed Static {i}')
            axes[row, col].set_title(f'Parameter {i}')
            axes[row, col].legend()
            
            # Add R² score
            r2 = np.corrcoef(target_static[:, i], recon_static[:, i])[0, 1] ** 2
            axes[row, col].text(0.05, 0.95, f'R² = {r2:.3f}', 
                               transform=axes[row, col].transAxes, 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_infonce_similarity_matrix(
    z: np.ndarray,
    static_params: np.ndarray,
    save_path: str = None,
    title: str = "InfoNCE Similarity Matrix"
) -> None:
    """
    Plot InfoNCE similarity matrix between latent vectors and static parameters.
    
    Args:
        z (np.ndarray): Latent vectors [n_samples, latent_dim]
        static_params (np.ndarray): Static parameters [n_samples, static_dim]
        save_path (str, optional): Path to save the plot
        title (str): Plot title
    """
    # Normalize vectors
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    static_norm = static_params / (np.linalg.norm(static_params, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix
    similarities = np.dot(z_norm, static_norm.T)
    
    plt.figure(figsize=(10, 8))
    
    # Plot similarity matrix
    im = plt.imshow(similarities, cmap='RdBu_r', aspect='equal', vmin=-1, vmax=1)
    plt.colorbar(im, label='Cosine Similarity')
    
    # Highlight diagonal (positive pairs)
    n_samples = min(similarities.shape)
    for i in range(n_samples):
        plt.plot(i, i, 'ko', markersize=3)
    
    plt.title(title)
    plt.xlabel('Static Parameter Sample Index')
    plt.ylabel('Latent Vector Sample Index')
    
    # Add text annotations for diagonal elements
    diagonal_similarities = np.diag(similarities)
    mean_diag = np.mean(diagonal_similarities)
    plt.text(0.02, 0.98, f'Mean diagonal similarity: {mean_diag:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_static_regularization_analysis(
    z: np.ndarray,
    static_params: np.ndarray,
    recon_static: np.ndarray,
    save_path: str = None,
    title: str = "Static Regularization Analysis"
) -> None:
    """
    Comprehensive plot of static regularization effects.
    
    Args:
        z (np.ndarray): Latent vectors [n_samples, latent_dim]
        static_params (np.ndarray): Static parameters [n_samples, static_dim]
        recon_static (np.ndarray): Reconstructed static parameters [n_samples, static_dim]
        save_path (str, optional): Path to save the plot
        title (str): Overall plot title
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=16)
    
    # 1. Latent-static correlations
    plt.subplot(2, 3, 1)
    latent_dim = z.shape[1]
    static_dim = static_params.shape[1]
    correlations = np.zeros((min(latent_dim, 20), static_dim))
    
    for i in range(min(latent_dim, 20)):
        for j in range(static_dim):
            corr_coef = np.corrcoef(z[:, i], static_params[:, j])[0, 1]
            correlations[i, j] = corr_coef if not np.isnan(corr_coef) else 0.0
    
    im1 = plt.imshow(np.abs(correlations), cmap='viridis', aspect='auto')
    plt.colorbar(im1, label='|Correlation|')
    plt.title('Latent-Static Correlations')
    plt.xlabel('Static Parameter')
    plt.ylabel('Latent Dimension')
    
    # 2. Static reconstruction accuracy
    plt.subplot(2, 3, 2)
    errors = np.abs(recon_static - static_params)
    plt.boxplot([errors[:, i] for i in range(static_dim)], 
               labels=[f'S{i}' for i in range(static_dim)])
    plt.title('Reconstruction Errors')
    plt.ylabel('Absolute Error')
    plt.xlabel('Static Parameter')
    
    # 3. InfoNCE similarity matrix (subset)
    plt.subplot(2, 3, 3)
    n_show = min(50, z.shape[0])  # Show subset for clarity
    z_subset = z[:n_show]
    static_subset = static_params[:n_show]
    
    z_norm = z_subset / (np.linalg.norm(z_subset, axis=1, keepdims=True) + 1e-8)
    static_norm = static_subset / (np.linalg.norm(static_subset, axis=1, keepdims=True) + 1e-8)
    similarities = np.dot(z_norm, static_norm.T)
    
    im3 = plt.imshow(similarities, cmap='RdBu_r', aspect='equal', vmin=-1, vmax=1)
    plt.colorbar(im3, label='Similarity')
    plt.title(f'InfoNCE Similarities (first {n_show})')
    plt.xlabel('Static Index')
    plt.ylabel('Latent Index')
    
    # 4. Reconstruction scatter plot for first static parameter
    plt.subplot(2, 3, 4)
    plt.scatter(static_params[:, 0], recon_static[:, 0], alpha=0.6)
    min_val, max_val = static_params[:, 0].min(), static_params[:, 0].max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    plt.xlabel('Target Static 0')
    plt.ylabel('Reconstructed Static 0')
    plt.title('Static Parameter 0 Reconstruction')
    plt.legend()
    
    # Add R² score
    r2 = np.corrcoef(static_params[:, 0], recon_static[:, 0])[0, 1] ** 2
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Loss landscape approximation
    plt.subplot(2, 3, 5)
    # Compute static reconstruction loss for each sample
    sample_losses = np.mean((recon_static - static_params) ** 2, axis=1)
    plt.hist(sample_losses, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Static Reconstruction Loss')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Static Losses')
    plt.axvline(np.mean(sample_losses), color='red', linestyle='--', 
                label=f'Mean: {np.mean(sample_losses):.3f}')
    plt.legend()
    
    # 6. Correlation strength distribution
    plt.subplot(2, 3, 6)
    all_correlations = []
    for i in range(min(latent_dim, 50)):  # Limit for efficiency
        for j in range(static_dim):
            corr = np.corrcoef(z[:, i], static_params[:, j])[0, 1]
            if not np.isnan(corr):
                all_correlations.append(abs(corr))
    
    plt.hist(all_correlations, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('|Correlation Coefficient|')
    plt.ylabel('Number of Pairs')
    plt.title('Distribution of Correlation Strengths')
    plt.axvline(np.mean(all_correlations), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_correlations):.3f}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_hierarchical_static_analysis(
    z_global: np.ndarray,
    z_local: np.ndarray,
    static_params: np.ndarray,
    recon_static_global: np.ndarray,
    recon_static_local: np.ndarray,
    recon_static_combined: np.ndarray,
    save_path: str = None,
    title: str = "Hierarchical Static Regularization Analysis"
) -> None:
    """
    Analyze hierarchical static regularization effects.
    
    Args:
        z_global (np.ndarray): Global latent vectors
        z_local (np.ndarray): Local latent vectors  
        static_params (np.ndarray): Target static parameters
        recon_static_global (np.ndarray): Global static reconstruction
        recon_static_local (np.ndarray): Local static reconstruction
        recon_static_combined (np.ndarray): Combined static reconstruction
        save_path (str, optional): Path to save the plot
        title (str): Plot title
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(title, fontsize=16)
    
    static_dim = static_params.shape[1]
    static_dim_half = static_dim // 2
    
    # 1. Global latent-static correlations
    plt.subplot(3, 4, 1)
    global_correlations = np.zeros((min(z_global.shape[1], 16), static_dim))
    for i in range(min(z_global.shape[1], 16)):
        for j in range(static_dim):
            corr = np.corrcoef(z_global[:, i], static_params[:, j])[0, 1]
            global_correlations[i, j] = corr if not np.isnan(corr) else 0.0
    
    plt.imshow(np.abs(global_correlations), cmap='viridis', aspect='auto')
    plt.colorbar(label='|Correlation|')
    plt.title('Global Latent-Static Correlations')
    plt.xlabel('Static Parameter')
    plt.ylabel('Global Latent Dim')
    
    # 2. Local latent-static correlations
    plt.subplot(3, 4, 2)
    local_correlations = np.zeros((min(z_local.shape[1], 16), static_dim))
    for i in range(min(z_local.shape[1], 16)):
        for j in range(static_dim):
            corr = np.corrcoef(z_local[:, i], static_params[:, j])[0, 1]
            local_correlations[i, j] = corr if not np.isnan(corr) else 0.0
    
    plt.imshow(np.abs(local_correlations), cmap='viridis', aspect='auto')
    plt.colorbar(label='|Correlation|')
    plt.title('Local Latent-Static Correlations')
    plt.xlabel('Static Parameter')
    plt.ylabel('Local Latent Dim')
    
    # 3. Global reconstruction accuracy
    plt.subplot(3, 4, 3)
    target_global = static_params[:, :static_dim_half]
    global_errors = np.abs(recon_static_global - target_global)
    plt.boxplot([global_errors[:, i] for i in range(static_dim_half)],
               labels=[f'S{i}' for i in range(static_dim_half)])
    plt.title('Global Reconstruction Errors')
    plt.ylabel('Absolute Error')
    
    # 4. Local reconstruction accuracy
    plt.subplot(3, 4, 4)
    target_local = static_params[:, static_dim_half:]
    local_errors = np.abs(recon_static_local - target_local)
    plt.boxplot([local_errors[:, i] for i in range(static_dim_half)],
               labels=[f'S{i+static_dim_half}' for i in range(static_dim_half)])
    plt.title('Local Reconstruction Errors')
    plt.ylabel('Absolute Error')
    
    # 5. Combined reconstruction accuracy
    plt.subplot(3, 4, 5)
    combined_errors = np.abs(recon_static_combined - static_params)
    plt.boxplot([combined_errors[:, i] for i in range(static_dim)],
               labels=[f'S{i}' for i in range(static_dim)])
    plt.title('Combined Reconstruction Errors')
    plt.ylabel('Absolute Error')
    
    # 6. Comparison of reconstruction methods
    plt.subplot(3, 4, 6)
    global_mse = np.mean(global_errors ** 2, axis=0)
    local_mse = np.mean(local_errors ** 2, axis=0) 
    combined_mse = np.mean(combined_errors ** 2, axis=0)
    
    x = np.arange(static_dim)
    width = 0.25
    
    plt.bar(x[:static_dim_half] - width, global_mse, width, label='Global', alpha=0.8)
    plt.bar(x[static_dim_half:] - width, local_mse, width, label='Local', alpha=0.8)
    plt.bar(x, combined_mse, width, label='Combined', alpha=0.8)
    
    plt.xlabel('Static Parameter')
    plt.ylabel('MSE')
    plt.title('Reconstruction MSE Comparison')
    plt.legend()
    plt.xticks(x, [f'S{i}' for i in range(static_dim)])
    
    # 7. Global InfoNCE similarities
    plt.subplot(3, 4, 7)
    n_show = min(30, z_global.shape[0])
    z_global_norm = z_global[:n_show] / (np.linalg.norm(z_global[:n_show], axis=1, keepdims=True) + 1e-8)
    static_norm = static_params[:n_show] / (np.linalg.norm(static_params[:n_show], axis=1, keepdims=True) + 1e-8)
    global_similarities = np.dot(z_global_norm, static_norm.T)
    
    plt.imshow(global_similarities, cmap='RdBu_r', aspect='equal', vmin=-1, vmax=1)
    plt.colorbar(label='Similarity')
    plt.title(f'Global InfoNCE (first {n_show})')
    
    # 8. Local InfoNCE similarities  
    plt.subplot(3, 4, 8)
    z_local_norm = z_local[:n_show] / (np.linalg.norm(z_local[:n_show], axis=1, keepdims=True) + 1e-8)
    local_similarities = np.dot(z_local_norm, static_norm.T)
    
    plt.imshow(local_similarities, cmap='RdBu_r', aspect='equal', vmin=-1, vmax=1)
    plt.colorbar(label='Similarity')
    plt.title(f'Local InfoNCE (first {n_show})')
    
    # 9. Reconstruction component contributions
    plt.subplot(3, 4, 9)
    # Compute how much each component contributes to final reconstruction
    global_contrib = np.mean(np.abs(recon_static_global), axis=0)
    local_contrib = np.mean(np.abs(recon_static_local), axis=0)
    
    plt.bar(range(static_dim_half), global_contrib, alpha=0.7, label='Global Component')
    plt.bar(range(static_dim_half, static_dim), local_contrib, alpha=0.7, label='Local Component')
    plt.xlabel('Static Parameter')
    plt.ylabel('Mean |Reconstruction|')
    plt.title('Component Contributions')
    plt.legend()
    
    # 10. Latent space dimensionality comparison
    plt.subplot(3, 4, 10)
    # Effective dimensionality via PCA
    from sklearn.decomposition import PCA
    
    pca_global = PCA()
    pca_local = PCA()
    pca_global.fit(z_global)
    pca_local.fit(z_local)
    
    # Find 95% variance cutoff
    global_cumvar = np.cumsum(pca_global.explained_variance_ratio_)
    local_cumvar = np.cumsum(pca_local.explained_variance_ratio_)
    global_95 = np.argmax(global_cumvar >= 0.95) + 1
    local_95 = np.argmax(local_cumvar >= 0.95) + 1
    
    plt.plot(global_cumvar[:20], 'o-', label=f'Global (95%: {global_95})')
    plt.plot(local_cumvar[:20], 's-', label=f'Local (95%: {local_95})')
    plt.axhline(0.95, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Latent Space Dimensionality')
    plt.legend()
    
    # 11. Cross-correlation between global and local
    plt.subplot(3, 4, 11)
    cross_corr = np.corrcoef(z_global.T, z_local.T)
    global_dim = z_global.shape[1]
    cross_section = cross_corr[:global_dim, global_dim:]
    
    plt.imshow(np.abs(cross_section), cmap='viridis', aspect='auto')
    plt.colorbar(label='|Cross-Correlation|')
    plt.title('Global-Local Cross-Correlation')
    plt.xlabel('Local Dimension')
    plt.ylabel('Global Dimension')
    
    # 12. Summary statistics
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # Compute summary stats
    stats_text = f"""
    Summary Statistics:
    
    Global Latent Dim: {z_global.shape[1]}
    Local Latent Dim: {z_local.shape[1]}
    Static Parameters: {static_dim}
    
    Reconstruction MSE:
    - Global: {np.mean(global_errors**2):.4f}
    - Local: {np.mean(local_errors**2):.4f}
    - Combined: {np.mean(combined_errors**2):.4f}
    
    Mean |Correlation|:
    - Global-Static: {np.mean(np.abs(global_correlations)):.3f}
    - Local-Static: {np.mean(np.abs(local_correlations)):.3f}
    
    InfoNCE Diagonal Similarity:
    - Global: {np.mean(np.diag(global_similarities)):.3f}
    - Local: {np.mean(np.diag(local_similarities)):.3f}
    
    Effective Dimensionality (95%):
    - Global: {global_95}
    - Local: {local_95}
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 