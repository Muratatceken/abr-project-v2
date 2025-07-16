import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union

from .encoder import Encoder
from .decoder import Decoder


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for ABR signal generation.
    
    Combines an encoder and decoder to learn a latent representation
    of ABR signals conditioned on static parameters.
    """
    
    def __init__(self, signal_length: int, static_dim: int, latent_dim: int, predict_peaks: bool = False, num_peaks: int = 6):
        """
        Initialize the CVAE.
        
        Args:
            signal_length (int): Length of the ABR signal
            static_dim (int): Dimension of static parameters
            latent_dim (int): Dimension of the latent space
            predict_peaks (bool): Whether to predict peak values
            num_peaks (int): Number of peaks to predict (default=6 for ABR)
        """
        super(CVAE, self).__init__()
        
        self.signal_length = signal_length
        self.static_dim = static_dim
        self.latent_dim = latent_dim
        self.predict_peaks = predict_peaks
        self.num_peaks = num_peaks
        
        # Numerical stability parameters
        self.min_logvar = -20.0
        self.max_logvar = 2.0
        self.eps = 1e-8
        
        # Initialize encoder and decoder
        self.encoder = Encoder(signal_length, static_dim, latent_dim)
        self.decoder = Decoder(signal_length, static_dim, latent_dim, predict_peaks, num_peaks)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, signal: torch.Tensor, static_params: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the CVAE.
        
        Args:
            signal (torch.Tensor): Input signal of shape [batch, signal_length]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
                If predict_peaks=False: (recon_signal, mu, logvar)
                If predict_peaks=True: (recon_signal, mu, logvar, predicted_peaks)
        """
        # Check for NaN/Inf in inputs
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            raise ValueError("NaN or Inf detected in input signal")
        if torch.isnan(static_params).any() or torch.isinf(static_params).any():
            raise ValueError("NaN or Inf detected in static parameters")
        
        # Encode to get latent distribution parameters
        mu, logvar = self.encoder(signal, static_params)
        
        # Clamp mu and logvar for numerical stability
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        
        # Check for NaN/Inf in encoder outputs
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            raise ValueError("NaN or Inf detected in mu")
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            raise ValueError("NaN or Inf detected in logvar")
        
        # Sample from latent distribution using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decode to reconstruct signal (and optionally predict peaks)
        decoder_output = self.decoder(z, static_params)
        
        if self.predict_peaks:
            recon_signal, predicted_peaks = decoder_output
            
            # Check for NaN/Inf in decoder outputs
            if torch.isnan(recon_signal).any() or torch.isinf(recon_signal).any():
                raise ValueError("NaN or Inf detected in reconstructed signal")
            if torch.isnan(predicted_peaks).any() or torch.isinf(predicted_peaks).any():
                raise ValueError("NaN or Inf detected in predicted peaks")
            
            return recon_signal, mu, logvar, predicted_peaks
        else:
            recon_signal = decoder_output
            
            # Check for NaN/Inf in decoder output
            if torch.isnan(recon_signal).any() or torch.isinf(recon_signal).any():
                raise ValueError("NaN or Inf detected in reconstructed signal")
            
            return recon_signal, mu, logvar
    
    def sample(self, static_params: torch.Tensor, n_samples: int = 1) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate samples from the CVAE given static parameters.
        
        Args:
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            n_samples (int): Number of samples to generate for each set of static parameters
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                If predict_peaks=False: Generated signals of shape [batch * n_samples, signal_length]
                If predict_peaks=True: Tuple of (generated signals, generated peaks)
        """
        batch_size = static_params.size(0)
        device = static_params.device
        
        # Expand static parameters for multiple samples
        if n_samples > 1:
            static_params_expanded = static_params.repeat_interleave(n_samples, dim=0)
        else:
            static_params_expanded = static_params
        
        # Sample from prior distribution N(0, I)
        z = torch.randn(batch_size * n_samples, self.latent_dim, device=device)
        
        # Decode to generate signals (and optionally peaks)
        with torch.no_grad():
            decoder_output = self.decoder(z, static_params_expanded)
        
        return decoder_output
    
    def encode(self, signal: torch.Tensor, static_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode signal and static parameters to latent space.
        
        Args:
            signal (torch.Tensor): Input signal of shape [batch, signal_length]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mu and logvar of latent distribution
        """
        mu, logvar = self.encoder(signal, static_params)
        
        # Clamp for numerical stability
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        
        return mu, logvar
    
    def decode(self, z: torch.Tensor, static_params: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode latent vector and static parameters to signal.
        
        Args:
            z (torch.Tensor): Latent vector of shape [batch, latent_dim]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                If predict_peaks=False: Reconstructed signal of shape [batch, signal_length]
                If predict_peaks=True: Tuple of (reconstructed signal, predicted peaks)
        """
        return self.decoder(z, static_params)
    
    def get_latent_sample(self, signal: torch.Tensor, static_params: torch.Tensor) -> torch.Tensor:
        """
        Get a sample from the latent distribution given input signal and static parameters.
        
        Args:
            signal (torch.Tensor): Input signal of shape [batch, signal_length]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            torch.Tensor: Sampled latent vector of shape [batch, latent_dim]
        """
        mu, logvar = self.encoder(signal, static_params)
        
        # Clamp for numerical stability
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        
        return self.reparameterize(mu, logvar)
