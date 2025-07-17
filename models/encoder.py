import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


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


class GlobalEncoder(nn.Module):
    """
    Global Encoder for hierarchical CVAE.
    
    Captures global, coarse-grained features from the entire signal and static parameters.
    This encoder focuses on overall signal characteristics and global patterns.
    """
    
    def __init__(self, signal_length: int, static_dim: int, global_latent_dim: int):
        """
        Initialize the Global Encoder.
        
        Args:
            signal_length (int): Length of the input signal
            static_dim (int): Dimension of static parameters
            global_latent_dim (int): Dimension of the global latent space
        """
        super(GlobalEncoder, self).__init__()
        
        self.signal_length = signal_length
        self.static_dim = static_dim
        self.global_latent_dim = global_latent_dim
        
        # Input dimension is signal + static parameters
        input_dim = signal_length + static_dim
        
        # Global encoder focuses on overall patterns with larger receptive fields
        self.global_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
        )
        
        # Output layers for global latent distribution
        self.mu_global = nn.Linear(64, global_latent_dim)
        self.logvar_global = nn.Linear(64, global_latent_dim)
        
        self._init_weights()
        
    def forward(self, signal: torch.Tensor, static_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the global encoder.
        
        Args:
            signal (torch.Tensor): Input signal of shape [batch, signal_length]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mu_global and logvar_global of shape [batch, global_latent_dim]
        """
        # Concatenate signal and static parameters
        x = torch.cat([signal, static_params], dim=1)
        
        # Pass through global encoder layers
        global_features = self.global_layers(x)
        
        # Compute global latent distribution parameters
        mu_global = self.mu_global(global_features)
        logvar_global = self.logvar_global(global_features)
        
        # Clamp for numerical stability
        logvar_global = torch.clamp(logvar_global, min=-20, max=20)
        
        # Handle NaN/Inf values
        if torch.isnan(mu_global).any() or torch.isinf(mu_global).any():
            mu_global = torch.where(torch.isnan(mu_global) | torch.isinf(mu_global), 
                                   torch.zeros_like(mu_global), mu_global)
        
        if torch.isnan(logvar_global).any() or torch.isinf(logvar_global).any():
            logvar_global = torch.where(torch.isnan(logvar_global) | torch.isinf(logvar_global), 
                                       torch.zeros_like(logvar_global), logvar_global)
        
        return mu_global, logvar_global
    
    def _init_weights(self):
        """Initialize weights for better numerical stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class LocalEncoder(nn.Module):
    """
    Local Encoder for hierarchical CVAE.
    
    Captures local, fine-grained features from the signal, potentially conditioned
    on early signal segments and static parameters. This encoder focuses on 
    detailed waveform characteristics and local patterns.
    """
    
    def __init__(self, signal_length: int, static_dim: int, local_latent_dim: int, 
                 early_signal_ratio: float = 0.3):
        """
        Initialize the Local Encoder.
        
        Args:
            signal_length (int): Length of the input signal
            static_dim (int): Dimension of static parameters
            local_latent_dim (int): Dimension of the local latent space
            early_signal_ratio (float): Ratio of signal to use for early conditioning (default: 0.3)
        """
        super(LocalEncoder, self).__init__()
        
        self.signal_length = signal_length
        self.static_dim = static_dim
        self.local_latent_dim = local_latent_dim
        self.early_signal_length = int(signal_length * early_signal_ratio)
        
        # Local encoder can condition on early signal + static params + full signal
        # This allows it to capture local patterns while being aware of global context
        input_dim = signal_length + self.early_signal_length + static_dim
        
        # Local encoder focuses on fine details with more complex architecture
        self.local_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.15),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.15),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
        )
        
        # Output layers for local latent distribution
        self.mu_local = nn.Linear(64, local_latent_dim)
        self.logvar_local = nn.Linear(64, local_latent_dim)
        
        self._init_weights()
        
    def forward(self, signal: torch.Tensor, static_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the local encoder.
        
        Args:
            signal (torch.Tensor): Input signal of shape [batch, signal_length]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mu_local and logvar_local of shape [batch, local_latent_dim]
        """
        # Extract early signal for conditioning
        early_signal = signal[:, :self.early_signal_length]
        
        # Concatenate full signal, early signal, and static parameters
        # This gives the local encoder access to both global and local information
        x = torch.cat([signal, early_signal, static_params], dim=1)
        
        # Pass through local encoder layers
        local_features = self.local_layers(x)
        
        # Compute local latent distribution parameters
        mu_local = self.mu_local(local_features)
        logvar_local = self.logvar_local(local_features)
        
        # Clamp for numerical stability
        logvar_local = torch.clamp(logvar_local, min=-20, max=20)
        
        # Handle NaN/Inf values
        if torch.isnan(mu_local).any() or torch.isinf(mu_local).any():
            mu_local = torch.where(torch.isnan(mu_local) | torch.isinf(mu_local), 
                                  torch.zeros_like(mu_local), mu_local)
        
        if torch.isnan(logvar_local).any() or torch.isinf(logvar_local).any():
            logvar_local = torch.where(torch.isnan(logvar_local) | torch.isinf(logvar_local), 
                                      torch.zeros_like(logvar_local), logvar_local)
        
        return mu_local, logvar_local
    
    def _init_weights(self):
        """Initialize weights for better numerical stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical Encoder combining Global and Local encoders.
    
    This encoder produces two separate latent representations:
    - Global latent (z_global): Captures overall signal characteristics
    - Local latent (z_local): Captures fine-grained waveform details
    """
    
    def __init__(self, signal_length: int, static_dim: int, 
                 global_latent_dim: int, local_latent_dim: int,
                 early_signal_ratio: float = 0.3):
        """
        Initialize the Hierarchical Encoder.
        
        Args:
            signal_length (int): Length of the input signal
            static_dim (int): Dimension of static parameters
            global_latent_dim (int): Dimension of the global latent space
            local_latent_dim (int): Dimension of the local latent space
            early_signal_ratio (float): Ratio of signal for local encoder conditioning
        """
        super(HierarchicalEncoder, self).__init__()
        
        self.signal_length = signal_length
        self.static_dim = static_dim
        self.global_latent_dim = global_latent_dim
        self.local_latent_dim = local_latent_dim
        self.total_latent_dim = global_latent_dim + local_latent_dim
        
        # Initialize global and local encoders
        self.global_encoder = GlobalEncoder(signal_length, static_dim, global_latent_dim)
        self.local_encoder = LocalEncoder(signal_length, static_dim, local_latent_dim, early_signal_ratio)
        
    def forward(self, signal: torch.Tensor, static_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hierarchical encoder.
        
        Args:
            signal (torch.Tensor): Input signal of shape [batch, signal_length]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - mu_global: Global latent mean [batch, global_latent_dim]
                - logvar_global: Global latent log variance [batch, global_latent_dim]
                - mu_local: Local latent mean [batch, local_latent_dim]
                - logvar_local: Local latent log variance [batch, local_latent_dim]
        """
        # Encode global features
        mu_global, logvar_global = self.global_encoder(signal, static_params)
        
        # Encode local features
        mu_local, logvar_local = self.local_encoder(signal, static_params)
        
        return {
            'mu_global': mu_global,
            'logvar_global': logvar_global,
            'mu_local': mu_local,
            'logvar_local': logvar_local
        }
    
    def get_latent_dims(self) -> Dict[str, int]:
        """Get the dimensions of different latent spaces."""
        return {
            'global_latent_dim': self.global_latent_dim,
            'local_latent_dim': self.local_latent_dim,
            'total_latent_dim': self.total_latent_dim
        }
