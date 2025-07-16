import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


class Decoder(nn.Module):
    """
    Decoder network for Conditional Variational Autoencoder (CVAE).
    
    Takes latent vector and static parameters as input and reconstructs
    the original signal.
    """
    
    def __init__(self, signal_length: int, static_dim: int, latent_dim: int, predict_peaks: bool = False, num_peaks: int = 6):
        """
        Initialize the Decoder.
        
        Args:
            signal_length (int): Length of the output signal
            static_dim (int): Dimension of static parameters
            latent_dim (int): Dimension of the latent space
            predict_peaks (bool): Whether to predict peak values
            num_peaks (int): Number of peaks to predict (default=6 for ABR)
        """
        super(Decoder, self).__init__()
        
        self.signal_length = signal_length
        self.static_dim = static_dim
        self.latent_dim = latent_dim
        self.predict_peaks = predict_peaks
        self.num_peaks = num_peaks
        
        # Input dimension is latent + static parameters
        input_dim = latent_dim + static_dim
        
        # MLP layers for decoding
        self.decoder_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Output layer for reconstructed signal
        self.output_layer = nn.Linear(1024, signal_length)
        
        # Optional peak prediction head
        if self.predict_peaks:
            self.peak_head = nn.Linear(1024, num_peaks)
        
    def forward(self, z: torch.Tensor, static_params: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the decoder.
        
        Args:
            z (torch.Tensor): Latent vector of shape [batch, latent_dim]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                If predict_peaks=False: Reconstructed signal of shape [batch, signal_length]
                If predict_peaks=True: Tuple of (reconstructed signal, predicted peaks)
        """
        # Concatenate latent vector and static parameters
        x = torch.cat([z, static_params], dim=1)
        
        # Pass through decoder layers
        decoded = self.decoder_layers(x)
        
        # Generate reconstructed signal
        recon_signal = self.output_layer(decoded)
        
        if self.predict_peaks:
            # Generate peak predictions
            predicted_peaks = self.peak_head(decoded)
            return recon_signal, predicted_peaks
        else:
            return recon_signal
