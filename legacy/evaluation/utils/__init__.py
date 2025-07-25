"""
Evaluation utilities package.
"""

from .metrics import (
    compute_reconstruction_metrics,
    compute_peak_metrics,
    compute_dtw_distance,
    compute_latent_metrics,
    aggregate_metrics
)

from .plotting import (
    plot_reconstruction_comparison,
    plot_latent_space_2d,
    plot_generation_samples,
    plot_peak_analysis,
    plot_metrics_summary
)

__all__ = [
    'compute_reconstruction_metrics',
    'compute_peak_metrics',
    'compute_dtw_distance',
    'compute_latent_metrics',
    'aggregate_metrics',
    'plot_reconstruction_comparison',
    'plot_latent_space_2d',
    'plot_generation_samples',
    'plot_peak_analysis',
    'plot_metrics_summary'
] 