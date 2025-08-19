"""
Utility modules for ABR Transformer training pipeline.
"""

from .schedules import cosine_beta_schedule, prepare_noise_schedule, q_sample_vpred, predict_x0_from_v
from .ema import EMA, EMAContextManager, create_ema
from .stft_loss import MultiResSTFTLoss, spectral_convergence_loss, log_stft_magnitude_loss
from .metrics import l1_time, mse_time, rmse_time, snr_db, pearson_correlation, compute_basic_metrics
from .tb import plot_waveforms, plot_spectrogram, plot_comparison, close_figure

__all__ = [
    # Diffusion schedules
    'cosine_beta_schedule', 'prepare_noise_schedule', 'q_sample_vpred', 'predict_x0_from_v',
    
    # EMA
    'EMA', 'EMAContextManager', 'create_ema',
    
    # STFT loss
    'MultiResSTFTLoss', 'spectral_convergence_loss', 'log_stft_magnitude_loss',
    
    # Metrics
    'l1_time', 'mse_time', 'rmse_time', 'snr_db', 'pearson_correlation', 'compute_basic_metrics',
    
    # TensorBoard plotting
    'plot_waveforms', 'plot_spectrogram', 'plot_comparison', 'close_figure'
]
