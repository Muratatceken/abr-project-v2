import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """
    Encoder network for Conditional Variational Autoencoder (CVAE).
    
    Takes signal and static parameters as input and outputs parameters
    for the latent distribution (mu and logvar).
    """
    
    def __init__(self, signal_length: int, static_dim: int, latent_dim: int):
        """
        Initialize the Encoder.
        
        Args:
            signal_length (int): Length of the input signal
            static_dim (int): Dimension of static parameters
            latent_dim (int): Dimension of the latent space
        """
        super(Encoder, self).__init__()
        
        self.signal_length = signal_length
        self.static_dim = static_dim
        self.latent_dim = latent_dim
        
        # Input dimension is signal + static parameters
        input_dim = signal_length + static_dim
        
        # MLP layers for encoding
        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Output layers for mu and logvar
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)
        
        # Initialize weights for better numerical stability
        self._init_weights()
        
    def forward(self, signal: torch.Tensor, static_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            signal (torch.Tensor): Input signal of shape [batch, signal_length]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mu and logvar tensors, each of shape [batch, latent_dim]
        """
        # Concatenate signal and static parameters
        x = torch.cat([signal, static_params], dim=1)
        
        # Pass through encoder layers
        encoded = self.encoder_layers(x)
        
        # Compute mu and logvar
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-20, max=20)
        
        # Check for NaN/Inf and handle them
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print(f"Warning: NaN/Inf detected in mu. Replacing with zeros.")
            mu = torch.where(torch.isnan(mu) | torch.isinf(mu), torch.zeros_like(mu), mu)
        
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            print(f"Warning: NaN/Inf detected in logvar. Replacing with zeros.")
            logvar = torch.where(torch.isnan(logvar) | torch.isinf(logvar), torch.zeros_like(logvar), logvar)
        
        return mu, logvar
    
    def _init_weights(self):
        """Initialize weights for better numerical stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
