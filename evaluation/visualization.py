"""
Evaluation Visualization Tools

This module provides comprehensive visualization tools for ABR signal evaluation,
including signal comparisons, metrics plots, and analysis dashboards.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from scipy import signal

# Optional imports with fallbacks
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive dashboards will be disabled.")


class EvaluationVisualizer:
    """Comprehensive visualization tools for evaluation results."""
    
    def __init__(self, output_dir: str = 'evaluation_plots', style: str = 'seaborn-v0_8'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')  # Use default instead of seaborn-v0_8
        sns.set_palette("husl")
        
        # Color schemes
        self.colors = {
            'generated': '#FF6B6B',
            'real': '#4ECDC4', 
            'difference': '#45B7D1',
            'metrics': '#96CEB4',
            'secondary': '#FFEAA7'
        }
    
    def plot_sample_comparisons(self, 
                              generated_samples: List[torch.Tensor],
                              real_samples: List[torch.Tensor],
                              num_samples: int = 6,
                              save_path: Optional[str] = None) -> str:
        """
        Create comparison plots between generated and real samples.
        
        Args:
            generated_samples: List of generated signal tensors
            real_samples: List of real signal tensors  
            num_samples: Number of samples to plot
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 2*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_samples, len(generated_samples))):
            generated = generated_samples[i].squeeze().cpu().numpy()
            real = real_samples[i].squeeze().cpu().numpy()
            difference = generated - real
            
            # Time axis
            time_axis = np.arange(len(generated)) / 1000  # Assuming 1kHz sampling
            
            # Generated signal
            axes[i, 0].plot(time_axis, generated, color=self.colors['generated'], linewidth=1.5)
            axes[i, 0].set_title(f'Generated Sample {i+1}')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Real signal
            axes[i, 1].plot(time_axis, real, color=self.colors['real'], linewidth=1.5)
            axes[i, 1].set_title(f'Real Sample {i+1}')
            axes[i, 1].set_ylabel('Amplitude')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Difference
            axes[i, 2].plot(time_axis, difference, color=self.colors['difference'], linewidth=1.5)
            axes[i, 2].set_title(f'Difference {i+1}')
            axes[i, 2].set_ylabel('Amplitude')
            axes[i, 2].grid(True, alpha=0.3)
            
            if i == num_samples - 1:
                for j in range(3):
                    axes[i, j].set_xlabel('Time (ms)')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'sample_comparisons.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_overlay_comparison(self,
                              generated_samples: List[torch.Tensor],
                              real_samples: List[torch.Tensor],
                              num_samples: int = 10,
                              save_path: Optional[str] = None) -> str:
        """
        Create overlay comparison plots.
        
        Args:
            generated_samples: List of generated signal tensors
            real_samples: List of real signal tensors
            num_samples: Number of samples to overlay
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        time_axis = np.arange(generated_samples[0].squeeze().shape[0]) / 1000
        
        # Plot overlays
        for i in range(min(num_samples, len(generated_samples))):
            generated = generated_samples[i].squeeze().cpu().numpy()
            real = real_samples[i].squeeze().cpu().numpy()
            
            alpha = 0.7 if num_samples > 5 else 1.0
            
            ax1.plot(time_axis, generated, color=self.colors['generated'], 
                    alpha=alpha, linewidth=1.0, label='Generated' if i == 0 else "")
            ax1.plot(time_axis, real, color=self.colors['real'], 
                    alpha=alpha, linewidth=1.0, label='Real' if i == 0 else "")
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Signal Overlay Comparison ({num_samples} samples)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot average signals
        avg_generated = torch.stack(generated_samples[:num_samples]).mean(0).squeeze().cpu().numpy()
        avg_real = torch.stack(real_samples[:num_samples]).mean(0).squeeze().cpu().numpy()
        std_generated = torch.stack(generated_samples[:num_samples]).std(0).squeeze().cpu().numpy()
        std_real = torch.stack(real_samples[:num_samples]).std(0).squeeze().cpu().numpy()
        
        ax2.plot(time_axis, avg_generated, color=self.colors['generated'], 
                linewidth=2, label='Generated (mean)')
        ax2.fill_between(time_axis, avg_generated - std_generated, avg_generated + std_generated,
                        color=self.colors['generated'], alpha=0.3)
        
        ax2.plot(time_axis, avg_real, color=self.colors['real'], 
                linewidth=2, label='Real (mean)')
        ax2.fill_between(time_axis, avg_real - std_real, avg_real + std_real,
                        color=self.colors['real'], alpha=0.3)
        
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Average Signals with Standard Deviation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'overlay_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_frequency_analysis(self,
                              generated_samples: List[torch.Tensor],
                              real_samples: List[torch.Tensor],
                              sr: int = 1000,
                              save_path: Optional[str] = None) -> str:
        """
        Create frequency domain analysis plots.
        
        Args:
            generated_samples: List of generated signal tensors
            real_samples: List of real signal tensors
            sr: Sampling rate
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Combine all samples for analysis
        all_generated = torch.cat(generated_samples, dim=0).cpu().numpy()
        all_real = torch.cat(real_samples, dim=0).cpu().numpy()
        
        # Power Spectral Density
        freqs_gen, psd_gen = signal.welch(all_generated.flatten(), fs=sr, nperseg=256)
        freqs_real, psd_real = signal.welch(all_real.flatten(), fs=sr, nperseg=256)
        
        axes[0, 0].semilogy(freqs_gen, psd_gen, color=self.colors['generated'], 
                           linewidth=2, label='Generated')
        axes[0, 0].semilogy(freqs_real, psd_real, color=self.colors['real'], 
                           linewidth=2, label='Real')
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('PSD')
        axes[0, 0].set_title('Power Spectral Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Magnitude Spectrum
        sample_gen = generated_samples[0].squeeze().cpu().numpy()
        sample_real = real_samples[0].squeeze().cpu().numpy()
        
        fft_gen = np.abs(np.fft.fft(sample_gen))
        fft_real = np.abs(np.fft.fft(sample_real))
        freqs = np.fft.fftfreq(len(sample_gen), 1/sr)
        
        axes[0, 1].plot(freqs[:len(freqs)//2], fft_gen[:len(freqs)//2], 
                       color=self.colors['generated'], linewidth=1.5, label='Generated')
        axes[0, 1].plot(freqs[:len(freqs)//2], fft_real[:len(freqs)//2], 
                       color=self.colors['real'], linewidth=1.5, label='Real')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].set_title('Magnitude Spectrum (Sample)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Spectrogram comparison
        f_gen, t_gen, Sxx_gen = signal.spectrogram(sample_gen, sr, nperseg=64)
        f_real, t_real, Sxx_real = signal.spectrogram(sample_real, sr, nperseg=64)
        
        im1 = axes[1, 0].pcolormesh(t_gen*1000, f_gen, 10*np.log10(Sxx_gen), 
                                   shading='gouraud', cmap='viridis')
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        axes[1, 0].set_title('Generated Spectrogram')
        plt.colorbar(im1, ax=axes[1, 0])
        
        im2 = axes[1, 1].pcolormesh(t_real*1000, f_real, 10*np.log10(Sxx_real), 
                                   shading='gouraud', cmap='viridis')
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('Frequency (Hz)')
        axes[1, 1].set_title('Real Spectrogram')
        plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'frequency_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_metrics_distribution(self,
                                metrics: Dict[str, Any],
                                save_path: Optional[str] = None) -> str:
        """
        Plot distribution of evaluation metrics.
        
        Args:
            metrics: Dictionary of aggregated metrics
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        # Extract scalar metrics for plotting
        scalar_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict) and 'mean' in value:
                scalar_metrics[key] = value
        
        if not scalar_metrics:
            # Create empty plot if no metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No metrics available for plotting', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Metrics Distribution')
        else:
            # Create metrics plot
            metric_names = list(scalar_metrics.keys())
            means = [scalar_metrics[m]['mean'] for m in metric_names]
            stds = [scalar_metrics[m]['std'] for m in metric_names]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot of means
            bars = ax1.bar(range(len(metric_names)), means, 
                          color=self.colors['metrics'], alpha=0.7,
                          yerr=stds, capsize=5)
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Value')
            ax1.set_title('Metric Means with Standard Deviation')
            ax1.set_xticks(range(len(metric_names)))
            ax1.set_xticklabels(metric_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.3f}', ha='center', va='bottom')
            
            # Box plot style visualization
            positions = range(len(metric_names))
            mins = [scalar_metrics[m]['min'] for m in metric_names]
            maxs = [scalar_metrics[m]['max'] for m in metric_names]
            medians = [scalar_metrics[m]['median'] for m in metric_names]
            
            for i, (mean, std, minimum, maximum, median) in enumerate(zip(means, stds, mins, maxs, medians)):
                # Draw box
                ax2.bar(i, 2*std, bottom=mean-std, width=0.6, 
                       color=self.colors['metrics'], alpha=0.5)
                # Draw median line
                ax2.plot([i-0.3, i+0.3], [median, median], 'k-', linewidth=2)
                # Draw whiskers
                ax2.plot([i, i], [minimum, mean-std], 'k-', linewidth=1)
                ax2.plot([i, i], [mean+std, maximum], 'k-', linewidth=1)
                # Draw outlier markers
                ax2.plot(i, minimum, 'ko', markersize=3)
                ax2.plot(i, maximum, 'ko', markersize=3)
            
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Value')
            ax2.set_title('Metric Distributions')
            ax2.set_xticks(range(len(metric_names)))
            ax2.set_xticklabels(metric_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'metrics_distribution.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_interactive_dashboard(self,
                                   generated_samples: List[torch.Tensor],
                                   real_samples: List[torch.Tensor],
                                   metrics: Dict[str, Any],
                                   save_path: Optional[str] = None) -> str:
        """
        Create interactive HTML dashboard using Plotly.
        
        Args:
            generated_samples: List of generated signal tensors
            real_samples: List of real signal tensors
            metrics: Dictionary of metrics
            save_path: Optional custom save path
            
        Returns:
            Path to saved HTML dashboard
        """
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available. Creating static dashboard instead.")
            return self._create_static_dashboard(generated_samples, real_samples, metrics, save_path)
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Sample Comparison', 'Frequency Analysis', 
                          'Metrics Overview', 'Signal Statistics',
                          'Correlation Analysis', 'Generation Quality'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sample comparison
        sample_idx = 0
        generated = generated_samples[sample_idx].squeeze().cpu().numpy()
        real = real_samples[sample_idx].squeeze().cpu().numpy()
        time_axis = np.arange(len(generated)) / 1000
        
        fig.add_trace(
            go.Scatter(x=time_axis, y=generated, name='Generated', 
                      line=dict(color=self.colors['generated'])),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_axis, y=real, name='Real', 
                      line=dict(color=self.colors['real'])),
            row=1, col=1
        )
        
        # Frequency analysis
        fft_gen = np.abs(np.fft.fft(generated))
        fft_real = np.abs(np.fft.fft(real))
        freqs = np.fft.fftfreq(len(generated), 1/1000)
        
        fig.add_trace(
            go.Scatter(x=freqs[:len(freqs)//2], y=fft_gen[:len(freqs)//2], 
                      name='Generated FFT', line=dict(color=self.colors['generated'])),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=freqs[:len(freqs)//2], y=fft_real[:len(freqs)//2], 
                      name='Real FFT', line=dict(color=self.colors['real'])),
            row=1, col=2
        )
        
        # Metrics overview (if available)
        if metrics:
            scalar_metrics = {k: v for k, v in metrics.items() 
                            if isinstance(v, dict) and 'mean' in v}
            if scalar_metrics:
                metric_names = list(scalar_metrics.keys())
                means = [scalar_metrics[m]['mean'] for m in metric_names]
                
                fig.add_trace(
                    go.Bar(x=metric_names, y=means, name='Metrics',
                          marker_color=self.colors['metrics']),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title='ABR Signal Generation Evaluation Dashboard',
            showlegend=True,
            height=1200
        )
        
        if save_path is None:
            save_path = self.output_dir / 'interactive_dashboard.html'
        
        fig.write_html(save_path)
        
        return str(save_path)
    
    def _create_static_dashboard(self,
                               generated_samples: List[torch.Tensor],
                               real_samples: List[torch.Tensor],
                               metrics: Dict[str, Any],
                               save_path: Optional[str] = None) -> str:
        """Create a static dashboard as fallback when Plotly is not available."""
        if save_path is None:
            save_path = self.output_dir / 'static_dashboard.png'
        
        # Create a comprehensive static plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sample comparison
        if generated_samples and real_samples:
            sample_idx = 0
            generated = generated_samples[sample_idx].squeeze().cpu().numpy()
            real = real_samples[sample_idx].squeeze().cpu().numpy()
            time_axis = np.arange(len(generated)) / 1000
            
            axes[0, 0].plot(time_axis, generated, label='Generated', color=self.colors['generated'])
            axes[0, 0].plot(time_axis, real, label='Real', color=self.colors['real'])
            axes[0, 0].set_title('Sample Comparison')
            axes[0, 0].set_xlabel('Time (ms)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Metrics overview
        if metrics:
            scalar_metrics = {k: v for k, v in metrics.items() 
                            if isinstance(v, dict) and 'mean' in v}
            if scalar_metrics:
                metric_names = list(scalar_metrics.keys())
                means = [scalar_metrics[m]['mean'] for m in metric_names]
                
                axes[0, 1].bar(range(len(metric_names)), means, color=self.colors['metrics'])
                axes[0, 1].set_title('Metrics Overview')
                axes[0, 1].set_xticks(range(len(metric_names)))
                axes[0, 1].set_xticklabels(metric_names, rotation=45, ha='right')
        
        # Add text summary
        axes[1, 0].axis('off')
        summary_text = "Static Dashboard\n\nKey Metrics:\n"
        if metrics:
            for key, value in list(metrics.items())[:5]:
                if isinstance(value, dict) and 'mean' in value:
                    summary_text += f"• {key}: {value['mean']:.3f}\n"
        axes[1, 0].text(0.1, 0.9, summary_text, transform=axes[1, 0].transAxes, 
                       verticalalignment='top', fontsize=10)
        
        # Placeholder for future content
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 'For interactive dashboard,\ninstall plotly:\npip install plotly', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_conditional_analysis(self,
                                generated_samples: List[torch.Tensor],
                                conditions: List[torch.Tensor],
                                save_path: Optional[str] = None) -> str:
        """
        Plot analysis of conditional generation.
        
        Args:
            generated_samples: List of generated signal tensors
            conditions: List of condition tensors
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert to numpy
        samples_np = [s.squeeze().cpu().numpy() for s in generated_samples]
        conditions_np = [c.squeeze().cpu().numpy() for c in conditions]
        
        # Condition vs Signal Properties
        signal_rms = [np.sqrt(np.mean(s**2)) for s in samples_np]
        signal_peak = [np.max(np.abs(s)) for s in samples_np]
        
        if len(conditions_np[0]) >= 2:
            condition_1 = [c[0] for c in conditions_np]  # First condition parameter
            condition_2 = [c[1] for c in conditions_np]  # Second condition parameter
            
            # RMS vs Condition 1
            axes[0, 0].scatter(condition_1, signal_rms, alpha=0.6, color=self.colors['generated'])
            axes[0, 0].set_xlabel('Condition Parameter 1')
            axes[0, 0].set_ylabel('Signal RMS')
            axes[0, 0].set_title('Signal RMS vs Condition 1')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Peak vs Condition 2
            axes[0, 1].scatter(condition_2, signal_peak, alpha=0.6, color=self.colors['real'])
            axes[0, 1].set_xlabel('Condition Parameter 2')
            axes[0, 1].set_ylabel('Signal Peak')
            axes[0, 1].set_title('Signal Peak vs Condition 2')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Signal diversity across conditions
        sample_matrix = np.array(samples_np)
        correlation_matrix = np.corrcoef(sample_matrix)
        
        im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Sample Cross-Correlation Matrix')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Sample Index')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Condition space visualization
        if len(conditions_np[0]) >= 2:
            scatter = axes[1, 1].scatter(condition_1, condition_2, c=signal_rms, 
                                       cmap='viridis', alpha=0.7)
            axes[1, 1].set_xlabel('Condition Parameter 1')
            axes[1, 1].set_ylabel('Condition Parameter 2')
            axes[1, 1].set_title('Condition Space (colored by RMS)')
            plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'conditional_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)