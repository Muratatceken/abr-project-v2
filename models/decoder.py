import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) block for conditioning decoder features
    on static parameters.
    
    FiLM applies affine transformations: output = gamma * input + beta
    where gamma and beta are learned from static parameters.
    """
    
    def __init__(self, feature_dim: int, static_dim: int, hidden_dim: int = 64):
        """
        Initialize FiLM block.
        
        Args:
            feature_dim (int): Dimension of input features to modulate
            static_dim (int): Dimension of static parameters
            hidden_dim (int): Hidden dimension for gamma/beta MLPs
        """
        super(FiLMBlock, self).__init__()
        
        self.feature_dim = feature_dim
        self.static_dim = static_dim
        
        # MLP for learning gamma (scale) parameters
        self.gamma_mlp = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # MLP for learning beta (shift) parameters
        self.beta_mlp = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Initialize gamma to 1 and beta to 0 for identity transformation initially
        with torch.no_grad():
            self.gamma_mlp[-1].weight.fill_(0.0)
            self.gamma_mlp[-1].bias.fill_(1.0)
            self.beta_mlp[-1].weight.fill_(0.0)
            self.beta_mlp[-1].bias.fill_(0.0)
    
    def forward(self, features: torch.Tensor, static_params: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning to input features.
        
        Args:
            features (torch.Tensor): Input features of shape [batch, feature_dim]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            torch.Tensor: Modulated features of shape [batch, feature_dim]
        """
        gamma = self.gamma_mlp(static_params)  # [batch, feature_dim]
        beta = self.beta_mlp(static_params)    # [batch, feature_dim]
        
        # Apply feature-wise linear modulation
        modulated = gamma * features + beta
        
        return modulated


class Decoder(nn.Module):
    """
    Decoder network for Conditional Variational Autoencoder (CVAE).
    
    Takes latent vector as input and reconstructs both the original signal
    and static parameters for joint generation. Supports FiLM conditioning
    for enhanced control over generation.
    """
    
    def __init__(self, signal_length: int, static_dim: int, latent_dim: int, 
                 predict_peaks: bool = False, num_peaks: int = 6, use_film: bool = False):
        """
        Initialize the Decoder.
        
        Args:
            signal_length (int): Length of the output signal
            static_dim (int): Dimension of static parameters
            latent_dim (int): Dimension of the latent space
            predict_peaks (bool): Whether to predict peak values
            num_peaks (int): Number of peaks to predict (default=6 for ABR)
            use_film (bool): Whether to use FiLM conditioning
        """
        super(Decoder, self).__init__()
        
        self.signal_length = signal_length
        self.static_dim = static_dim
        self.latent_dim = latent_dim
        self.predict_peaks = predict_peaks
        self.num_peaks = num_peaks
        self.use_film = use_film
        
        # Shared decoder layers (condition only on latent vector)
        self.layer1 = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.15)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.15)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.15)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.15)
        )
        
        # FiLM blocks for conditioning (if enabled)
        if self.use_film:
            self.film1 = FiLMBlock(128, static_dim)
            self.film2 = FiLMBlock(256, static_dim)
            self.film3 = FiLMBlock(512, static_dim)
            self.film4 = FiLMBlock(1024, static_dim)
        
        # Signal reconstruction head
        self.signal_head = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.1),
            nn.Linear(2048, signal_length)
        )
        
        # Static parameters reconstruction head
        self.static_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, static_dim)
        )
        
        # Optional peak prediction head
        if self.predict_peaks:
            self.peak_head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(512),
                nn.Dropout(0.1),
                nn.Linear(512, num_peaks)
            )
    
    def forward(self, z: torch.Tensor, static_params: torch.Tensor = None) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the decoder.
        
        Args:
            z (torch.Tensor): Latent vector of shape [batch, latent_dim]
            static_params (torch.Tensor, optional): Static parameters for FiLM conditioning
            
        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                If predict_peaks=False: (reconstructed_signal, reconstructed_static_params)
                If predict_peaks=True: (reconstructed_signal, reconstructed_static_params, predicted_peaks)
        """
        # Pass through shared layers with optional FiLM conditioning
        h1 = self.layer1(z)
        if self.use_film and static_params is not None:
            h1 = self.film1(h1, static_params)
        
        h2 = self.layer2(h1)
        if self.use_film and static_params is not None:
            h2 = self.film2(h2, static_params)
        
        h3 = self.layer3(h2)
        if self.use_film and static_params is not None:
            h3 = self.film3(h3, static_params)
        
        h4 = self.layer4(h3)
        if self.use_film and static_params is not None:
            shared_features = self.film4(h4, static_params)
        else:
            shared_features = h4
        
        # Generate reconstructed signal
        recon_signal = self.signal_head(shared_features)
        
        # Generate reconstructed static parameters
        recon_static_params = self.static_head(shared_features)
        
        if self.predict_peaks:
            # Generate peak predictions
            predicted_peaks = self.peak_head(shared_features)
            return recon_signal, recon_static_params, predicted_peaks
        else:
            return recon_signal, recon_static_params
    
    def forward_legacy(self, z: torch.Tensor, static_params: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Legacy forward pass for backward compatibility.
        This method conditions on both latent vector and static parameters.
        
        Args:
            z (torch.Tensor): Latent vector of shape [batch, latent_dim]
            static_params (torch.Tensor): Static parameters of shape [batch, static_dim]
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                If predict_peaks=False: Reconstructed signal of shape [batch, signal_length]
                If predict_peaks=True: Tuple of (reconstructed signal, predicted peaks)
        """
        # Concatenate latent vector and static parameters (old behavior)
        x = torch.cat([z, static_params], dim=1)
        
        # Create legacy decoder layers if not already created
        if not hasattr(self, 'legacy_layers'):
            input_dim = self.latent_dim + self.static_dim
            self.legacy_layers = nn.Sequential(
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
            ).to(x.device)
            
            self.legacy_output_layer = nn.Linear(1024, self.signal_length).to(x.device)
            
            if self.predict_peaks:
                self.legacy_peak_head = nn.Linear(1024, self.num_peaks).to(x.device)
        
        # Pass through legacy decoder layers
        decoded = self.legacy_layers(x)
        
        # Generate reconstructed signal
        recon_signal = self.legacy_output_layer(decoded)
        
        if self.predict_peaks:
            # Generate peak predictions
            predicted_peaks = self.legacy_peak_head(decoded)
            return recon_signal, predicted_peaks
        else:
            return recon_signal


class HierarchicalDecoder(nn.Module):
    """
    Hierarchical Decoder for CVAE with global and local latent spaces.
    
    This decoder accepts both z_global and z_local:
    - z_global controls FiLM modulation for coarse-grained features
    - z_local concatenated with static params decodes fine waveform details
    """
    
    def __init__(self, signal_length: int, static_dim: int, 
                 global_latent_dim: int, local_latent_dim: int,
                 predict_peaks: bool = False, num_peaks: int = 6, 
                 use_film: bool = True):
        """
        Initialize the Hierarchical Decoder.
        
        Args:
            signal_length (int): Length of the output signal
            static_dim (int): Dimension of static parameters
            global_latent_dim (int): Dimension of global latent space
            local_latent_dim (int): Dimension of local latent space
            predict_peaks (bool): Whether to predict peak values
            num_peaks (int): Number of peaks to predict
            use_film (bool): Whether to use FiLM conditioning with z_global
        """
        super(HierarchicalDecoder, self).__init__()
        
        self.signal_length = signal_length
        self.static_dim = static_dim
        self.global_latent_dim = global_latent_dim
        self.local_latent_dim = local_latent_dim
        self.predict_peaks = predict_peaks
        self.num_peaks = num_peaks
        self.use_film = use_film
        
        # Input for main decoder is z_local + static_params
        main_input_dim = local_latent_dim + static_dim
        
        # Main decoder layers that process z_local + static_params
        self.layer1 = nn.Sequential(
            nn.Linear(main_input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.15)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.15)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.15)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.15)
        )
        
        # FiLM blocks for global conditioning (if enabled)
        if self.use_film:
            self.film1 = FiLMBlock(256, global_latent_dim)
            self.film2 = FiLMBlock(512, global_latent_dim)
            self.film3 = FiLMBlock(1024, global_latent_dim)
            self.film4 = FiLMBlock(2048, global_latent_dim)
        
        # Signal reconstruction head
        self.signal_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.1),
            nn.Linear(2048, signal_length)
        )
        
        # Static parameters reconstruction head
        self.static_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, static_dim)
        )
        
        # Optional peak prediction head
        if self.predict_peaks:
            self.peak_head = nn.Sequential(
                nn.Linear(2048, 512),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(512),
                nn.Dropout(0.1),
                nn.Linear(512, num_peaks)
            )
    
    def forward(self, z_global: torch.Tensor, z_local: torch.Tensor, 
                static_params: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                                     Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the hierarchical decoder.
        
        Args:
            z_global (torch.Tensor): Global latent vector [batch, global_latent_dim]
            z_local (torch.Tensor): Local latent vector [batch, local_latent_dim]
            static_params (torch.Tensor): Static parameters [batch, static_dim]
            
        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                If predict_peaks=False: (reconstructed_signal, reconstructed_static_params)
                If predict_peaks=True: (reconstructed_signal, reconstructed_static_params, predicted_peaks)
        """
        # Concatenate z_local with static parameters for main processing
        x = torch.cat([z_local, static_params], dim=1)
        
        # Pass through decoder layers with FiLM conditioning from z_global
        h1 = self.layer1(x)
        if self.use_film:
            h1 = self.film1(h1, z_global)
        
        h2 = self.layer2(h1)
        if self.use_film:
            h2 = self.film2(h2, z_global)
        
        h3 = self.layer3(h2)
        if self.use_film:
            h3 = self.film3(h3, z_global)
        
        h4 = self.layer4(h3)
        if self.use_film:
            shared_features = self.film4(h4, z_global)
        else:
            shared_features = h4
        
        # Generate reconstructed signal
        recon_signal = self.signal_head(shared_features)
        
        # Generate reconstructed static parameters
        recon_static_params = self.static_head(shared_features)
        
        if self.predict_peaks:
            # Generate peak predictions
            predicted_peaks = self.peak_head(shared_features)
            return recon_signal, recon_static_params, predicted_peaks
        else:
            return recon_signal, recon_static_params
    
    def forward_from_combined_z(self, z_combined: torch.Tensor, 
                               static_params: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                                                   Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass with combined latent vector (for backward compatibility).
        
        Args:
            z_combined (torch.Tensor): Combined latent vector [batch, global_latent_dim + local_latent_dim]
            static_params (torch.Tensor): Static parameters [batch, static_dim]
            
        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                Same as forward method
        """
        # Split combined latent into global and local parts
        z_global = z_combined[:, :self.global_latent_dim]
        z_local = z_combined[:, self.global_latent_dim:]
        
        return self.forward(z_global, z_local, static_params)
