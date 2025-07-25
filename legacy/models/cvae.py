import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict

from .encoder import Encoder, HierarchicalEncoder
from .decoder import Decoder, HierarchicalDecoder


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for ABR signal generation.
    
    Combines an encoder and decoder to learn a latent representation
    of ABR signals conditioned on static parameters. Supports both
    legacy mode (conditioning decoder on static params) and joint
    generation mode (generating both signals and static params).
    """
    
    def __init__(self, signal_length: int, static_dim: int, latent_dim: int, predict_peaks: bool = False, num_peaks: int = 6, joint_generation: bool = False, use_film: bool = False):
        """
        Initialize the CVAE.
        
        Args:
            signal_length (int): Length of the ABR signal
            static_dim (int): Dimension of static parameters
            latent_dim (int): Dimension of the latent space
            predict_peaks (bool): Whether to predict peak values
            num_peaks (int): Number of peaks to predict (default=6 for ABR)
            joint_generation (bool): Whether to enable joint generation of signals and static params
            use_film (bool): Whether to enable FiLM conditioning in decoder
        """
        super(CVAE, self).__init__()
        
        self.signal_length = signal_length
        self.static_dim = static_dim
        self.latent_dim = latent_dim
        self.predict_peaks = predict_peaks
        self.num_peaks = num_peaks
        self.joint_generation = joint_generation
        self.use_film = use_film
        
        # Numerical stability parameters
        self.min_logvar = -20.0
        self.max_logvar = 2.0
        self.eps = 1e-8
        
        # Initialize encoder and decoder
        self.encoder = Encoder(signal_length, static_dim, latent_dim)
        self.decoder = Decoder(signal_length, static_dim, latent_dim, predict_peaks, num_peaks, use_film)
        
        # Static decoder for latent-static causal regularization
        # Maps from latent z to reconstructed static parameters
        self.static_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, static_dim)
        )
        
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
    
    def forward(self, signal: torch.Tensor, static_params: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the CVAE.
        
        Args:
            signal (torch.Tensor): Input signal of shape [batch, signal_length]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            Union[...]: Depending on joint_generation and predict_peaks flags:
                Legacy mode (joint_generation=False):
                    If predict_peaks=False: (recon_signal, mu, logvar)
                    If predict_peaks=True: (recon_signal, mu, logvar, predicted_peaks)
                Joint generation mode (joint_generation=True):
                    If predict_peaks=False: (recon_signal, recon_static_params, mu, logvar)
                    If predict_peaks=True: (recon_signal, recon_static_params, mu, logvar, predicted_peaks)
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
        
        # Reconstruct static parameters from latent z for causal regularization
        recon_static_from_z = self.static_decoder(z)
        
        # Check for NaN/Inf in static reconstruction
        if torch.isnan(recon_static_from_z).any() or torch.isinf(recon_static_from_z).any():
            raise ValueError("NaN or Inf detected in reconstructed static from z")
        
        # Decode based on generation mode
        if self.joint_generation:
            # Joint generation mode: decoder outputs both signal and static params
            # Pass static params for FiLM conditioning if enabled
            if self.use_film:
                decoder_output = self.decoder(z, static_params)
            else:
                decoder_output = self.decoder(z)
            
            if self.predict_peaks:
                recon_signal, recon_static_params, predicted_peaks = decoder_output
                
                # Check for NaN/Inf in decoder outputs
                if torch.isnan(recon_signal).any() or torch.isinf(recon_signal).any():
                    raise ValueError("NaN or Inf detected in reconstructed signal")
                if torch.isnan(recon_static_params).any() or torch.isinf(recon_static_params).any():
                    raise ValueError("NaN or Inf detected in reconstructed static parameters")
                if torch.isnan(predicted_peaks).any() or torch.isinf(predicted_peaks).any():
                    raise ValueError("NaN or Inf detected in predicted peaks")
                
                return recon_signal, recon_static_params, mu, logvar, predicted_peaks, recon_static_from_z
            else:
                recon_signal, recon_static_params = decoder_output
                
                # Check for NaN/Inf in decoder outputs
                if torch.isnan(recon_signal).any() or torch.isinf(recon_signal).any():
                    raise ValueError("NaN or Inf detected in reconstructed signal")
                if torch.isnan(recon_static_params).any() or torch.isinf(recon_static_params).any():
                    raise ValueError("NaN or Inf detected in reconstructed static parameters")
                
                return recon_signal, recon_static_params, mu, logvar, recon_static_from_z
        else:
            # Legacy mode: decoder conditions on static params
            decoder_output = self.decoder.forward_legacy(z, static_params)
            
            if self.predict_peaks:
                recon_signal, predicted_peaks = decoder_output
                
                # Check for NaN/Inf in decoder outputs
                if torch.isnan(recon_signal).any() or torch.isinf(recon_signal).any():
                    raise ValueError("NaN or Inf detected in reconstructed signal")
                if torch.isnan(predicted_peaks).any() or torch.isinf(predicted_peaks).any():
                    raise ValueError("NaN or Inf detected in predicted peaks")
                
                return recon_signal, mu, logvar, predicted_peaks, recon_static_from_z
            else:
                recon_signal = decoder_output
                
                # Check for NaN/Inf in decoder output
                if torch.isnan(recon_signal).any() or torch.isinf(recon_signal).any():
                    raise ValueError("NaN or Inf detected in reconstructed signal")
                
                return recon_signal, mu, logvar, recon_static_from_z
    
    def sample(self, static_params: torch.Tensor = None, n_samples: int = 1, generate_static_from_z: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate samples from the CVAE.
        
        Args:
            static_params (torch.Tensor, optional): Static parameters for legacy mode [batch, static_dim]
            n_samples (int): Number of samples to generate
            generate_static_from_z (bool): Whether to generate static parameters from latent z
            
        Returns:
            Union[...]: Depending on joint_generation and predict_peaks flags:
                Joint generation mode (joint_generation=True):
                    If predict_peaks=False: (generated_signals, generated_static_params)
                    If predict_peaks=True: (generated_signals, generated_static_params, generated_peaks)
                Legacy mode (joint_generation=False):
                    If predict_peaks=False: generated_signals
                    If predict_peaks=True: (generated_signals, generated_peaks)
        """
        if self.joint_generation:
            # Joint generation mode: generate both signals and static params
            device = next(self.parameters()).device
            
            # Sample from prior distribution N(0, I)
            z = torch.randn(n_samples, self.latent_dim, device=device)
            
            # Decode to generate both signals and static params
            with torch.no_grad():
                # Generate static parameters from z if requested
                if generate_static_from_z:
                    generated_static_from_z = self.static_decoder(z)
                
                # Pass static params for FiLM conditioning if available and enabled
                if self.use_film and static_params is not None:
                    # Expand static parameters for multiple samples if needed
                    if n_samples > 1 and static_params.size(0) == 1:
                        static_params_expanded = static_params.repeat(n_samples, 1)
                    elif n_samples > 1 and static_params.size(0) != n_samples:
                        static_params_expanded = static_params.repeat_interleave(n_samples, dim=0)
                    else:
                        static_params_expanded = static_params
                    decoder_output = self.decoder(z, static_params_expanded)
                else:
                    decoder_output = self.decoder(z)
                
                # Return decoder output with optional static generation
                if generate_static_from_z:
                    if isinstance(decoder_output, tuple):
                        return decoder_output + (generated_static_from_z,)
                    else:
                        return (decoder_output, generated_static_from_z)
                else:
                    return decoder_output
        else:
            # Legacy mode: requires static parameters
            if static_params is None:
                raise ValueError("static_params must be provided for legacy mode")
            
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
                # Generate static parameters from z if requested
                if generate_static_from_z:
                    generated_static_from_z = self.static_decoder(z)
                
                decoder_output = self.decoder.forward_legacy(z, static_params_expanded)
                
                # Return decoder output with optional static generation
                if generate_static_from_z:
                    if isinstance(decoder_output, tuple):
                        return decoder_output + (generated_static_from_z,)
                    else:
                        return (decoder_output, generated_static_from_z)
                else:
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
    
    def decode(self, z: torch.Tensor, static_params: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Decode latent vector to signal (and optionally static parameters).
        
        Args:
            z (torch.Tensor): Latent vector of shape [batch, latent_dim]
            static_params (torch.Tensor, optional): Static parameters for legacy mode [batch, static_dim]
            
        Returns:
            Union[...]: Depending on joint_generation and predict_peaks flags:
                Joint generation mode: (signal, static_params) or (signal, static_params, peaks)
                Legacy mode: signal or (signal, peaks)
        """
        if self.joint_generation:
            return self.decoder(z)
        else:
            if static_params is None:
                raise ValueError("static_params must be provided for legacy mode")
            return self.decoder.forward_legacy(z, static_params)
    
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
    
    def set_joint_generation(self, joint_generation: bool):
        """
        Set the joint generation mode.
        
        Args:
            joint_generation (bool): Whether to enable joint generation
        """
        self.joint_generation = joint_generation


class HierarchicalCVAE(nn.Module):
    """
    Hierarchical Conditional Variational Autoencoder (HCVAE) for ABR signal generation.
    
    Uses separate global and local encoders to capture different aspects of the signal:
    - Global encoder: Captures overall signal characteristics and patterns
    - Local encoder: Captures fine-grained waveform details
    
    The decoder uses z_global for FiLM modulation and z_local + static_params for detail generation.
    """
    
    def __init__(self, signal_length: int, static_dim: int, 
                 global_latent_dim: int = 32, local_latent_dim: int = 32,
                 predict_peaks: bool = False, num_peaks: int = 6,
                 use_film: bool = True, early_signal_ratio: float = 0.3):
        """
        Initialize the Hierarchical CVAE.
        
        Args:
            signal_length (int): Length of the ABR signal
            static_dim (int): Dimension of static parameters
            global_latent_dim (int): Dimension of global latent space
            local_latent_dim (int): Dimension of local latent space
            predict_peaks (bool): Whether to predict peak values
            num_peaks (int): Number of peaks to predict
            use_film (bool): Whether to use FiLM conditioning
            early_signal_ratio (float): Ratio of signal for local encoder conditioning
        """
        super(HierarchicalCVAE, self).__init__()
        
        self.signal_length = signal_length
        self.static_dim = static_dim
        self.global_latent_dim = global_latent_dim
        self.local_latent_dim = local_latent_dim
        self.total_latent_dim = global_latent_dim + local_latent_dim
        self.predict_peaks = predict_peaks
        self.num_peaks = num_peaks
        self.use_film = use_film
        self.early_signal_ratio = early_signal_ratio
        
        # Numerical stability parameters
        self.min_logvar = -20.0
        self.max_logvar = 2.0
        self.eps = 1e-8
        
        # Initialize hierarchical encoder and decoder
        self.encoder = HierarchicalEncoder(
            signal_length, static_dim, global_latent_dim, local_latent_dim, early_signal_ratio
        )
        self.decoder = HierarchicalDecoder(
            signal_length, static_dim, global_latent_dim, local_latent_dim,
            predict_peaks, num_peaks, use_film
        )
        
        # Static decoders for latent-static causal regularization
        # Global static decoder: maps global latent to global static characteristics
        self.global_static_decoder = nn.Sequential(
            nn.Linear(global_latent_dim, global_latent_dim // 2),
            nn.ReLU(),
            nn.Linear(global_latent_dim // 2, static_dim // 2)
        )
        
        # Local static decoder: maps local latent to local static characteristics  
        self.local_static_decoder = nn.Sequential(
            nn.Linear(local_latent_dim, local_latent_dim // 2),
            nn.ReLU(),
            nn.Linear(local_latent_dim // 2, static_dim // 2)
        )
        
        # Combined static decoder: combines global and local reconstructions
        self.static_combiner = nn.Sequential(
            nn.Linear(static_dim, static_dim // 2),
            nn.ReLU(),
            nn.Linear(static_dim // 2, static_dim)
        )
    
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
    
    def forward(self, signal: torch.Tensor, static_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hierarchical CVAE.
        
        Args:
            signal (torch.Tensor): Input signal of shape [batch, signal_length]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - recon_signal: Reconstructed signal [batch, signal_length]
                - recon_static_params: Reconstructed static parameters [batch, static_dim]
                - mu_global: Global latent mean [batch, global_latent_dim]
                - logvar_global: Global latent log variance [batch, global_latent_dim]
                - mu_local: Local latent mean [batch, local_latent_dim]
                - logvar_local: Local latent log variance [batch, local_latent_dim]
                - z_global: Sampled global latent [batch, global_latent_dim]
                - z_local: Sampled local latent [batch, local_latent_dim]
                - predicted_peaks: (optional) Predicted peaks [batch, num_peaks]
        """
        # Check for NaN/Inf in inputs
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            raise ValueError("NaN or Inf detected in input signal")
        if torch.isnan(static_params).any() or torch.isinf(static_params).any():
            raise ValueError("NaN or Inf detected in static parameters")
        
        # Encode to get hierarchical latent distribution parameters
        encoder_output = self.encoder(signal, static_params)
        mu_global = encoder_output['mu_global']
        logvar_global = encoder_output['logvar_global']
        mu_local = encoder_output['mu_local']
        logvar_local = encoder_output['logvar_local']
        
        # Clamp mu and logvar for numerical stability
        mu_global = torch.clamp(mu_global, min=-10.0, max=10.0)
        logvar_global = torch.clamp(logvar_global, min=self.min_logvar, max=self.max_logvar)
        mu_local = torch.clamp(mu_local, min=-10.0, max=10.0)
        logvar_local = torch.clamp(logvar_local, min=self.min_logvar, max=self.max_logvar)
        
        # Sample from latent distributions using reparameterization trick
        z_global = self.reparameterize(mu_global, logvar_global)
        z_local = self.reparameterize(mu_local, logvar_local)
        
        # Reconstruct static parameters from hierarchical latents
        global_static_recon = self.global_static_decoder(z_global)
        local_static_recon = self.local_static_decoder(z_local)
        
        # Combine global and local static reconstructions
        combined_static_recon = torch.cat([global_static_recon, local_static_recon], dim=1)
        recon_static_from_z = self.static_combiner(combined_static_recon)
        
        # Check for NaN/Inf in static reconstruction
        if torch.isnan(recon_static_from_z).any() or torch.isinf(recon_static_from_z).any():
            raise ValueError("NaN or Inf detected in hierarchical static reconstruction")
        
        # Decode using hierarchical decoder
        decoder_output = self.decoder(z_global, z_local, static_params)
        
        # Prepare return dictionary
        result = {
            'mu_global': mu_global,
            'logvar_global': logvar_global,
            'mu_local': mu_local,
            'logvar_local': logvar_local,
            'z_global': z_global,
            'z_local': z_local,
        }
        
        if self.predict_peaks:
            recon_signal, recon_static_params, predicted_peaks = decoder_output
            result.update({
                'recon_signal': recon_signal,
                'recon_static_params': recon_static_params,
                'predicted_peaks': predicted_peaks,
                'recon_static_from_z': recon_static_from_z
            })
        else:
            recon_signal, recon_static_params = decoder_output
            result.update({
                'recon_signal': recon_signal,
                'recon_static_params': recon_static_params,
                'recon_static_from_z': recon_static_from_z
            })
        
        return result
    
    def sample(self, static_params: Optional[torch.Tensor] = None, 
               n_samples: int = 1, 
               z_global: Optional[torch.Tensor] = None,
               z_local: Optional[torch.Tensor] = None,
               generate_static_from_z: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate samples from the hierarchical CVAE.
        
        Args:
            static_params (torch.Tensor, optional): Static parameters [batch, static_dim]
            n_samples (int): Number of samples to generate
            z_global (torch.Tensor, optional): Pre-specified global latent [batch, global_latent_dim]
            z_local (torch.Tensor, optional): Pre-specified local latent [batch, local_latent_dim]
            generate_static_from_z (bool): Whether to generate static parameters from latent z
            
        Returns:
            Dict[str, torch.Tensor]: Generated samples dictionary containing:
                - generated_signals: Generated signals [n_samples, signal_length]
                - generated_static_params: Generated static parameters [n_samples, static_dim]
                - z_global: Global latent vectors used [n_samples, global_latent_dim]
                - z_local: Local latent vectors used [n_samples, local_latent_dim]
                - generated_peaks: (optional) Generated peaks [n_samples, num_peaks]
        """
        device = next(self.parameters()).device
        
        # Generate or use provided latent vectors
        if z_global is None:
            z_global = torch.randn(n_samples, self.global_latent_dim, device=device)
        else:
            if z_global.size(0) != n_samples:
                z_global = z_global.repeat(n_samples, 1)
        
        if z_local is None:
            z_local = torch.randn(n_samples, self.local_latent_dim, device=device)
        else:
            if z_local.size(0) != n_samples:
                z_local = z_local.repeat(n_samples, 1)
        
        # Handle static parameters
        if static_params is None:
            # Generate random static parameters if not provided
            static_params = torch.randn(n_samples, self.static_dim, device=device)
        else:
            # Expand static parameters for multiple samples if needed
            if static_params.size(0) == 1 and n_samples > 1:
                static_params = static_params.repeat(n_samples, 1)
            elif static_params.size(0) != n_samples:
                static_params = static_params[:n_samples]
        
        # Generate samples using decoder
        with torch.no_grad():
            # Generate static parameters from z if requested
            if generate_static_from_z:
                global_static_recon = self.global_static_decoder(z_global)
                local_static_recon = self.local_static_decoder(z_local)
                combined_static_recon = torch.cat([global_static_recon, local_static_recon], dim=1)
                generated_static_from_z = self.static_combiner(combined_static_recon)
            
            decoder_output = self.decoder(z_global, z_local, static_params)
        
        # Prepare return dictionary
        result = {
            'z_global': z_global,
            'z_local': z_local,
        }
        
        # Add generated static from z if requested
        if generate_static_from_z:
            result['generated_static_from_z'] = generated_static_from_z
        
        if self.predict_peaks:
            generated_signals, generated_static_params, generated_peaks = decoder_output
            result.update({
                'generated_signals': generated_signals,
                'generated_static_params': generated_static_params,
                'generated_peaks': generated_peaks
            })
        else:
            generated_signals, generated_static_params = decoder_output
            result.update({
                'generated_signals': generated_signals,
                'generated_static_params': generated_static_params
            })
        
        return result
    
    def encode(self, signal: torch.Tensor, static_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode signal and static parameters to hierarchical latent spaces.
        
        Args:
            signal (torch.Tensor): Input signal of shape [batch, signal_length]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            Dict[str, torch.Tensor]: Latent distribution parameters
        """
        encoder_output = self.encoder(signal, static_params)
        
        # Clamp for numerical stability
        encoder_output['mu_global'] = torch.clamp(encoder_output['mu_global'], min=-10.0, max=10.0)
        encoder_output['logvar_global'] = torch.clamp(encoder_output['logvar_global'], 
                                                     min=self.min_logvar, max=self.max_logvar)
        encoder_output['mu_local'] = torch.clamp(encoder_output['mu_local'], min=-10.0, max=10.0)
        encoder_output['logvar_local'] = torch.clamp(encoder_output['logvar_local'], 
                                                    min=self.min_logvar, max=self.max_logvar)
        
        return encoder_output
    
    def decode(self, z_global: torch.Tensor, z_local: torch.Tensor, 
               static_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode from hierarchical latent vectors.
        
        Args:
            z_global (torch.Tensor): Global latent vector [batch, global_latent_dim]
            z_local (torch.Tensor): Local latent vector [batch, local_latent_dim]
            static_params (torch.Tensor): Static parameters [batch, static_dim]
            
        Returns:
            Dict[str, torch.Tensor]: Decoded outputs
        """
        decoder_output = self.decoder(z_global, z_local, static_params)
        
        result = {}
        if self.predict_peaks:
            recon_signal, recon_static_params, predicted_peaks = decoder_output
            result.update({
                'recon_signal': recon_signal,
                'recon_static_params': recon_static_params,
                'predicted_peaks': predicted_peaks
            })
        else:
            recon_signal, recon_static_params = decoder_output
            result.update({
                'recon_signal': recon_signal,
                'recon_static_params': recon_static_params
            })
        
        return result
    
    def get_latent_dims(self) -> Dict[str, int]:
        """Get the dimensions of different latent spaces."""
        return self.encoder.get_latent_dims()
