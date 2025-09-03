"""
FiLM (Feature-wise Linear Modulation) Implementation for ABR Signal Processing

Professional implementation of FiLM conditioning for incorporating
static parameters (age, intensity, stimulus rate, FMP) into the model.

Reference: "FiLM: Visual Reasoning with a General Conditioning Layer"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Dict
import math


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Applies element-wise affine transformation to input features
    based on conditioning information:
    
    FiLM(x, c) = γ(c) ⊙ x + β(c)
    
    where γ and β are learned functions of the conditioning input c.
    """
    
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = 'relu',
        dropout: float = 0.0,
        bias: bool = True,
        init_gamma: float = 1.0,
        init_beta: float = 0.0
    ):
        """
        Initialize FiLM layer.
        
        Args:
            input_dim: Dimension of conditioning input
            feature_dim: Dimension of features to be modulated
            hidden_dim: Hidden dimension for MLP (default: max(input_dim, feature_dim))
            num_layers: Number of layers in the conditioning MLP
            activation: Activation function ('relu', 'gelu', 'silu')
            dropout: Dropout rate
            bias: Whether to use bias in linear layers
            init_gamma: Initial value for gamma (scale parameter)
            init_beta: Initial value for beta (shift parameter)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim or max(input_dim, feature_dim)
        self.init_gamma = init_gamma
        self.init_beta = init_beta
        
        # Build conditioning MLP
        layers = []
        in_dim = input_dim
        
        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, self.hidden_dim, bias=bias))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_dim = self.hidden_dim
        
        # Output layer: projects to 2 * feature_dim (gamma and beta)
        layers.append(nn.Linear(in_dim, 2 * feature_dim, bias=bias))
        
        self.conditioning_mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Initialize conditioning MLP with Xavier uniform
        for module in self.conditioning_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Special initialization for output layer
        output_layer = self.conditioning_mlp[-1]
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            # Initialize gamma to init_gamma and beta to init_beta
            nn.init.constant_(output_layer.bias[:self.feature_dim], self.init_gamma)
            nn.init.constant_(output_layer.bias[self.feature_dim:], self.init_beta)
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning to input features.
        
        Args:
            x: Features to modulate [batch, feature_dim, ...] or [batch, ..., feature_dim]
            conditioning: Conditioning input [batch, input_dim]
            
        Returns:
            Modulated features with same shape as x
        """
        # Generate gamma and beta from conditioning
        film_params = self.conditioning_mlp(conditioning)  # [batch, 2 * feature_dim]
        gamma, beta = torch.chunk(film_params, 2, dim=1)   # Each [batch, feature_dim]
        
        # Handle different tensor layouts
        if x.dim() == 3:  # [batch, feature_dim, seq_len]
            gamma = gamma.unsqueeze(-1)  # [batch, feature_dim, 1]
            beta = beta.unsqueeze(-1)    # [batch, feature_dim, 1]
        elif x.dim() == 4:  # [batch, feature_dim, height, width]
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [batch, feature_dim, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)    # [batch, feature_dim, 1, 1]
        elif x.dim() == 2:  # [batch, feature_dim]
            pass  # gamma and beta are already [batch, feature_dim]
        else:
            # For other dimensions, assume feature_dim is the last dimension
            while gamma.dim() < x.dim():
                gamma = gamma.unsqueeze(-1)
                beta = beta.unsqueeze(-1)
        
        # Apply FiLM: x' = gamma * x + beta
        return gamma * x + beta


class TokenFiLM(nn.Module):
    """
    FiLM conditioning for token embeddings [B, T, D].
    Produces per-feature (D) gamma/beta from static params [B, S] and
    broadcasts over T.
    
    Supports residual connections for improved gradient flow and training stability.
    """
    def __init__(
        self, 
        static_dim: int, 
        d_model: int, 
        hidden: int = 256, 
        dropout: float = 0.1,
        init_gamma: float = 0.0, 
        init_beta: float = 0.0,
        use_residual: bool = False
    ):
        super().__init__()
        self.static_dim = static_dim
        self.d_model = d_model
        self.use_residual = use_residual
        
        if static_dim > 0:
            self.embed = nn.Sequential(
                nn.Linear(static_dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2 * d_model)  # gamma and beta
            )
        else:
            self.embed = None
            
        self.layer_norm = nn.LayerNorm(d_model)
        self.init_gamma = init_gamma
        self.init_beta = init_beta
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        if self.embed is not None:
            # Initialize the final layer specially
            with torch.no_grad():
                # Initialize gamma to init_gamma and beta to init_beta
                final_layer = self.embed[-1]
                nn.init.zeros_(final_layer.weight)
                if final_layer.bias is not None:
                    final_layer.bias[:self.d_model].fill_(self.init_gamma)  # gamma
                    final_layer.bias[self.d_model:].fill_(self.init_beta)   # beta

    def forward(self, x: torch.Tensor, static_params: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply FiLM conditioning to token embeddings.
        
        Args:
            x: Token embeddings [B, T, D]
            static_params: Static parameters [B, S] or None
            
        Returns:
            Modulated embeddings [B, T, D]
        """
        if static_params is None or self.embed is None:
            return x
        
        B, T, D = x.shape
        
        # Generate gamma and beta from static params
        gam_beta = self.embed(static_params)  # [B, 2D]
        gamma, beta = gam_beta.chunk(2, dim=-1)  # [B, D], [B, D]
        
        # Store original input for residual connection before normalization
        if self.use_residual:
            x_pre = x
        
        # Apply layer norm first
        x_norm = self.layer_norm(x)
        
        # Apply FiLM: x_norm * (1 + gamma) + beta
        x_film = x_norm * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        
        # Apply residual connection if enabled
        if self.use_residual:
            x = x_pre + x_film
        else:
            x = x_film
        
        return x

    def enable_residual(self):
        """Enable residual connections for better gradient flow."""
        self.use_residual = True

    def disable_residual(self):
        """Disable residual connections."""
        self.use_residual = False


class ConditionalEmbedding(nn.Module):
    """
    Embedding layer for conditional information.
    
    Converts categorical variables to dense embeddings and
    optionally combines with continuous variables.
    """
    
    def __init__(
        self,
        categorical_dims: Optional[list] = None,
        continuous_dim: int = 0,
        embedding_dim: int = 128,
        output_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        """
        Initialize conditional embedding.
        
        Args:
            categorical_dims: List of vocabulary sizes for categorical variables
            continuous_dim: Dimension of continuous variables
            embedding_dim: Dimension of categorical embeddings
            output_dim: Output dimension (default: sum of embedding dimensions)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.categorical_dims = categorical_dims or []
        self.continuous_dim = continuous_dim
        self.embedding_dim = embedding_dim
        
        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for vocab_size in self.categorical_dims
        ])
        
        # Calculate total input dimension
        total_categorical_dim = len(self.categorical_dims) * embedding_dim
        total_input_dim = total_categorical_dim + continuous_dim
        
        # Output projection
        self.output_dim = output_dim or total_input_dim
        if self.output_dim != total_input_dim:
            self.output_proj = nn.Linear(total_input_dim, self.output_dim)
        else:
            self.output_proj = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        
        if isinstance(self.output_proj, nn.Linear):
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self, 
        categorical_inputs: Optional[torch.Tensor] = None,
        continuous_inputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through conditional embedding.
        
        Args:
            categorical_inputs: Categorical variables [batch, num_categorical]
            continuous_inputs: Continuous variables [batch, continuous_dim]
            
        Returns:
            Embedded conditioning vector [batch, output_dim]
        """
        embeddings = []
        
        # Process categorical inputs
        if categorical_inputs is not None and len(self.embeddings) > 0:
            assert categorical_inputs.shape[1] == len(self.embeddings), \
                f"Expected {len(self.embeddings)} categorical inputs, got {categorical_inputs.shape[1]}"
            
            for i, embedding_layer in enumerate(self.embeddings):
                cat_emb = embedding_layer(categorical_inputs[:, i])
                embeddings.append(cat_emb)
        
        # Process continuous inputs
        if continuous_inputs is not None:
            embeddings.append(continuous_inputs)
        
        # Concatenate all embeddings
        if embeddings:
            x = torch.cat(embeddings, dim=1)
        else:
            # Return zero tensor if no inputs
            batch_size = (categorical_inputs.shape[0] if categorical_inputs is not None
                         else continuous_inputs.shape[0])
            # Get device from model parameters
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
            x = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Apply output projection and dropout
        x = self.output_proj(x)
        x = self.dropout(x)
        
        return x
    
    @property
    def device(self):
        """Get device of the module."""
        return next(self.parameters()).device


class AdaptiveFiLM(nn.Module):
    """
    Adaptive FiLM layer that learns to modulate different channels differently.
    
    Uses channel-wise attention to adaptively weight the conditioning
    based on the current feature representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = 'relu',
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_heads: int = 4
    ):
        """
        Initialize adaptive FiLM layer.
        
        Args:
            input_dim: Dimension of conditioning input
            feature_dim: Dimension of features to be modulated
            hidden_dim: Hidden dimension for MLP
            num_layers: Number of layers in conditioning MLP
            activation: Activation function
            dropout: Dropout rate
            use_attention: Whether to use channel attention
            attention_heads: Number of attention heads
        """
        super().__init__()
        
        self.use_attention = use_attention
        
        # Base FiLM layer
        self.film_layer = FiLMLayer(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout
        )
        
        # Channel attention mechanism
        if use_attention:
            self.channel_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive FiLM conditioning.
        
        Args:
            x: Features to modulate [batch, feature_dim, seq_len]
            conditioning: Conditioning input [batch, input_dim]
            
        Returns:
            Modulated features [batch, feature_dim, seq_len]
        """
        # Apply base FiLM conditioning
        x_film = self.film_layer(x, conditioning)
        
        if self.use_attention and x.dim() == 3:
            # Apply channel attention for sequence data
            # Transpose for attention: [batch, seq_len, feature_dim]
            x_transposed = x_film.transpose(1, 2)
            
            # Self-attention over features
            attn_output, _ = self.channel_attention(x_transposed, x_transposed, x_transposed)
            attn_output = self.attention_norm(attn_output + x_transposed)
            
            # Transpose back: [batch, feature_dim, seq_len]
            x_film = attn_output.transpose(1, 2)
        
        return x_film


class MultiScaleFiLM(nn.Module):
    """
    Multi-scale FiLM conditioning for hierarchical models.
    
    Applies different conditioning at different scales,
    useful for U-Net style architectures.
    """
    
    def __init__(
        self,
        input_dim: int,
        feature_dims: list,
        hidden_dim: Optional[int] = None,
        shared_conditioning: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize multi-scale FiLM.
        
        Args:
            input_dim: Dimension of conditioning input
            feature_dims: List of feature dimensions for different scales
            hidden_dim: Hidden dimension for conditioning MLPs
            shared_conditioning: Whether to share conditioning MLP across scales
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.feature_dims = feature_dims
        self.shared_conditioning = shared_conditioning
        
        if shared_conditioning:
            # Single conditioning MLP for all scales
            self.conditioning_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim or input_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim or input_dim * 2, hidden_dim or input_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Scale-specific projection layers
            self.scale_projections = nn.ModuleList([
                nn.Linear(hidden_dim or input_dim * 2, 2 * feature_dim)
                for feature_dim in feature_dims
            ])
        else:
            # Separate FiLM layers for each scale
            self.film_layers = nn.ModuleList([
                FiLMLayer(input_dim, feature_dim, hidden_dim, dropout=dropout)
                for feature_dim in feature_dims
            ])
    
    def forward(self, x_list: list, conditioning: torch.Tensor) -> list:
        """
        Apply multi-scale FiLM conditioning.
        
        Args:
            x_list: List of feature tensors at different scales
            conditioning: Conditioning input [batch, input_dim]
            
        Returns:
            List of modulated feature tensors
        """
        assert len(x_list) == len(self.feature_dims), \
            f"Expected {len(self.feature_dims)} feature tensors, got {len(x_list)}"
        
        if self.shared_conditioning:
            # Shared conditioning
            shared_features = self.conditioning_mlp(conditioning)
            
            modulated_features = []
            for i, (x, projection) in enumerate(zip(x_list, self.scale_projections)):
                # Get scale-specific gamma and beta
                film_params = projection(shared_features)
                gamma, beta = torch.chunk(film_params, 2, dim=1)
                
                # Apply conditioning
                if x.dim() == 3:
                    gamma = gamma.unsqueeze(-1)
                    beta = beta.unsqueeze(-1)
                
                x_modulated = gamma * x + beta
                modulated_features.append(x_modulated)
            
            return modulated_features
        else:
            # Separate conditioning for each scale
            return [film_layer(x, conditioning) 
                   for x, film_layer in zip(x_list, self.film_layers)] 


class AdaptiveFiLMWithDropout(nn.Module):
    """
    Enhanced FiLM layer with dropout and classifier-free guidance support.
    
    Features:
    - FiLM dropout during training for robustness
    - Classifier-free guidance preparation
    - Multi-scale modulation support
    - Advanced conditioning strategies
    """
    
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        film_dropout: float = 0.15,  # Probability of zeroing out conditioning
        use_cfg: bool = True,        # Support classifier-free guidance
        use_layer_scale: bool = True,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.film_dropout = film_dropout
        self.use_cfg = use_cfg
        self.use_layer_scale = use_layer_scale
        
        # Condition embedding network
        layers = []
        for i in range(num_layers):
            if i == 0:
                in_features = input_dim
            else:
                in_features = feature_dim
            
            layers.extend([
                nn.Linear(in_features, feature_dim),
                get_activation(activation),
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout)
            ])
        
        self.condition_net = nn.Sequential(*layers)
        
        # FiLM parameter generators
        self.gamma_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            get_activation(activation),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.beta_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            get_activation(activation),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Layer-wise scaling factors
        if use_layer_scale:
            self.layer_scale_gamma = nn.Parameter(torch.ones(feature_dim) * 0.1)
            self.layer_scale_beta = nn.Parameter(torch.ones(feature_dim) * 0.1)
        
        # Unconditional embedding for classifier-free guidance
        if use_cfg:
            self.uncond_embedding = nn.Parameter(torch.randn(input_dim) * 0.02)
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize gamma to near 1, beta to near 0
        with torch.no_grad():
            self.gamma_net[-1].weight.data *= 0.1
            self.beta_net[-1].weight.data *= 0.1
            if hasattr(self, 'layer_scale_gamma'):
                self.layer_scale_gamma.data.fill_(0.1)
                self.layer_scale_beta.data.fill_(0.1)
    
    def forward(
        self, 
        features: torch.Tensor, 
        condition: torch.Tensor,
        cfg_guidance_scale: float = 1.0,
        force_uncond: bool = False
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning with optional dropout and CFG.
        
        Enhanced implementation with explicit FiLM dropout pattern:
        - During training: with p=dropout_rate, replace condition vector with zeros
        - Supports classifier-free guidance with unconditional embeddings
        
        Args:
            features: Input features [batch, channels, ...] or [batch, ..., channels]
            condition: Conditioning vector [batch, input_dim]
            cfg_guidance_scale: Scale factor for classifier-free guidance
            force_uncond: Force unconditional generation
            
        Returns:
            Modulated features with same shape as input
        """
        batch_size = condition.size(0)
        original_shape = features.shape
        
        # Handle different feature tensor formats
        if features.dim() == 3:
            if features.size(1) == self.feature_dim:
                # [batch, channels, seq] -> [batch, seq, channels]
                features = features.transpose(1, 2)
                channel_last = True
            else:
                # [batch, seq, channels]
                channel_last = False
        else:
            channel_last = False
        
        # ===== ENHANCED FILM DROPOUT FOR ROBUSTNESS =====
        # During training: with p=film_dropout, randomly zero out the condition vector
        if self.training and self.film_dropout > 0:
            # Generate dropout mask: True = keep condition, False = drop condition
            dropout_mask = torch.rand(batch_size, device=condition.device) > self.film_dropout
            
            if self.use_cfg:
                # Use unconditional embedding for dropped samples (better than zeros)
                condition = torch.where(
                    dropout_mask.unsqueeze(1),
                    condition,
                    self.uncond_embedding.unsqueeze(0).expand(batch_size, -1)
                )
            else:
                # Explicit zeroing out of condition vector (as per specification)
                condition = torch.where(
                    dropout_mask.unsqueeze(1),
                    condition,
                    torch.zeros_like(condition)
                )
        
        # Handle classifier-free guidance (inference time)
        if force_uncond and self.use_cfg:
            condition = self.uncond_embedding.unsqueeze(0).expand(batch_size, -1)
        
        # Process conditioning through embedding network
        cond_features = self.condition_net(condition)  # [batch, feature_dim]
        
        # Generate FiLM parameters
        gamma = self.gamma_net(cond_features)  # [batch, feature_dim]
        beta = self.beta_net(cond_features)    # [batch, feature_dim]
        
        # Apply layer-wise scaling
        if self.use_layer_scale:
            gamma = gamma * self.layer_scale_gamma
            beta = beta * self.layer_scale_beta
        
        # Initialize gamma near 1 for stability
        gamma = gamma + 1.0
        
        # Apply FiLM: features * gamma + beta
        # Broadcast gamma and beta to match features shape
        if features.dim() == 3:
            gamma = gamma.unsqueeze(1)  # [batch, 1, feature_dim]
            beta = beta.unsqueeze(1)    # [batch, 1, feature_dim]
        
        modulated = features * gamma + beta
        
        # Restore original format
        if channel_last and features.dim() == 3:
            modulated = modulated.transpose(1, 2)
        
        return modulated


class MultiFiLM(nn.Module):
    """
    Multi-scale FiLM conditioning for different network depths.
    
    Applies different conditioning strategies at different network levels,
    allowing for hierarchical control over generation.
    """
    
    def __init__(
        self,
        input_dim: int,
        feature_dims: List[int],
        depth_weights: Optional[List[float]] = None,
        shared_conditioning: bool = True,
        individual_dropouts: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.feature_dims = feature_dims
        self.num_levels = len(feature_dims)
        self.shared_conditioning = shared_conditioning
        
        # Depth-specific weights
        if depth_weights is None:
            depth_weights = [1.0] * self.num_levels
        self.register_buffer('depth_weights', torch.tensor(depth_weights))
        
        # Shared conditioning network (if enabled)
        if shared_conditioning:
            self.shared_conditioner = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.GELU(),
                nn.LayerNorm(input_dim * 2),
                nn.Dropout(0.1),
                nn.Linear(input_dim * 2, input_dim * 2)
            )
            cond_input_dim = input_dim * 2
        else:
            self.shared_conditioner = None
            cond_input_dim = input_dim
        
        # Level-specific FiLM layers
        self.film_layers = nn.ModuleList([
            AdaptiveFiLMWithDropout(
                input_dim=cond_input_dim,
                feature_dim=dim,
                film_dropout=0.1 + 0.05 * i if individual_dropouts else 0.15,
                use_cfg=True
            )
            for i, dim in enumerate(feature_dims)
        ])
    
    def forward(
        self,
        features_list: List[torch.Tensor],
        condition: torch.Tensor,
        level_idx: Optional[int] = None,
        cfg_guidance_scale: float = 1.0
    ) -> List[torch.Tensor]:
        """
        Apply multi-level FiLM conditioning.
        
        Args:
            features_list: List of feature tensors at different scales
            condition: Conditioning vector [batch, input_dim]
            level_idx: If specified, only process this level
            cfg_guidance_scale: CFG scale factor
            
        Returns:
            List of modulated feature tensors
        """
        # Process shared conditioning
        if self.shared_conditioner is not None:
            processed_condition = self.shared_conditioner(condition)
        else:
            processed_condition = condition
        
        modulated_features = []
        
        for i, features in enumerate(features_list):
            if level_idx is not None and i != level_idx:
                # Skip this level
                modulated_features.append(features)
                continue
            
            # Apply depth-specific weighting
            level_condition = processed_condition * self.depth_weights[i]
            
            # Apply FiLM conditioning
            modulated = self.film_layers[i](
                features=features,
                condition=level_condition,
                cfg_guidance_scale=cfg_guidance_scale
            )
            
            modulated_features.append(modulated)
        
        return modulated_features


class CFGWrapper(nn.Module):
    """
    Enhanced Classifier-Free Guidance wrapper for any model.
    
    Features:
    - Controllable conditioning strength at generation time
    - Dynamic guidance scaling for different output types
    - Optimized inference with optional caching
    - Support for task-specific guidance scales
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        uncond_scale: float = 0.1,
        task_specific_scales: Optional[Dict[str, float]] = None,
        enable_dynamic_scaling: bool = True
    ):
        super().__init__()
        
        self.model = model
        self.uncond_scale = uncond_scale
        self.enable_dynamic_scaling = enable_dynamic_scaling
        
        # Task-specific guidance scales (e.g., stronger for classification, weaker for regression)
        self.task_specific_scales = task_specific_scales or {
            'recon': 1.0,      # Signal reconstruction
            'peak': 0.8,       # Peak prediction (regression)
            'class': 1.2,      # Classification (stronger guidance)
            'threshold': 0.9   # Threshold regression
        }
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        guidance_scale: float = 1.0,
        cfg_mode: str = 'training',  # 'training', 'inference', 'unconditional'
        task_scales: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with classifier-free guidance support.
        
        Args:
            x: Input tensor
            condition: Conditioning information
            guidance_scale: Global CFG guidance scale (> 1.0 for stronger conditioning)
            cfg_mode: CFG mode ('training', 'inference', 'unconditional')
            task_scales: Optional task-specific guidance scale overrides
            
        Returns:
            Model outputs (conditional, unconditional, or combined with enhanced CFG)
        """
        if cfg_mode == 'unconditional':
            # Pure unconditional generation
            return self.model(x, condition, force_uncond=True)
        
        elif cfg_mode == 'training':
            # Regular training with FiLM dropout
            return self.model(x, condition)
        
        elif cfg_mode == 'inference' and guidance_scale > 1.0:
            # Enhanced classifier-free guidance inference
            # Conditional pass
            cond_output = self.model(x, condition)
            
            # Unconditional pass
            uncond_output = self.model(x, condition, force_uncond=True)
            
            # Enhanced CFG combination with task-specific scaling
            guided_output = {}
            
            # Merge task-specific scales with defaults
            effective_scales = self.task_specific_scales.copy()
            if task_scales:
                effective_scales.update(task_scales)
            
            for key in cond_output.keys():
                # Get task-specific guidance scale
                task_scale = effective_scales.get(key, 1.0) * guidance_scale
                
                if isinstance(cond_output[key], torch.Tensor):
                    # Standard CFG: uncond + scale * (cond - uncond)
                    guided_output[key] = (
                        uncond_output[key] + 
                        task_scale * (cond_output[key] - uncond_output[key])
                    )
                else:
                    # Handle tuple outputs (e.g., peak predictions)
                    if isinstance(cond_output[key], tuple):
                        guided_output[key] = tuple(
                            uncond_val + task_scale * (cond_val - uncond_val)
                            for cond_val, uncond_val in zip(cond_output[key], uncond_output[key])
                        )
                    else:
                        guided_output[key] = cond_output[key]
            
            return guided_output
        
        else:
            # Regular conditional inference
            return self.model(x, condition)
    
    def set_guidance_scales(self, task_scales: Dict[str, float]):
        """
        Update task-specific guidance scales.
        
        Args:
            task_scales: Dictionary mapping task names to guidance scales
        """
        self.task_specific_scales.update(task_scales)
    
    def get_guidance_scales(self) -> Dict[str, float]:
        """Get current task-specific guidance scales."""
        return self.task_specific_scales.copy()
    
    def sample_with_cfg(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        guidance_scale: float = 2.0,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        apply_constraints: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced sampling with CFG and additional controls.
        
        Args:
            x: Input tensor
            condition: Conditioning information
            guidance_scale: CFG guidance scale
            temperature: Temperature for sampling (affects randomness)
            top_k: Top-k sampling for discrete outputs
            apply_constraints: Whether to apply clinical constraints
            
        Returns:
            Sampled outputs with CFG applied
        """
        # Get CFG outputs
        outputs = self.forward(
            x=x,
            condition=condition,
            guidance_scale=guidance_scale,
            cfg_mode='inference'
        )
        
        # Apply temperature scaling
        if temperature != 1.0:
            for key, value in outputs.items():
                if key == 'class' and isinstance(value, torch.Tensor):
                    # Apply temperature to classification logits
                    outputs[key] = value / temperature
        
        # Apply clinical constraints if requested
        if apply_constraints:
            outputs = self._apply_clinical_constraints(outputs)
        
        return outputs
    
    def _apply_clinical_constraints(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply clinical constraints to model outputs.
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            Constrained outputs
        """
        constrained = outputs.copy()
        
        # Constrain peak latency to physiologically plausible range (1-8 ms)
        if 'peak' in constrained and isinstance(constrained['peak'], tuple):
            peak_exists, peak_latency, peak_amplitude = constrained['peak'][:3]
            # Clamp latency to [1.0, 8.0] ms range
            peak_latency = torch.clamp(peak_latency, min=1.0, max=8.0)
            constrained['peak'] = (peak_exists, peak_latency, peak_amplitude) + constrained['peak'][3:]
        
        # Constrain threshold to [0, 120] dB range
        if 'threshold' in constrained and isinstance(constrained['threshold'], torch.Tensor):
            constrained['threshold'] = torch.clamp(constrained['threshold'], min=0.0, max=120.0)
        
        # Ensure signal amplitude is within reasonable bounds (disabled for better dynamics)
        if 'recon' in constrained and isinstance(constrained['recon'], torch.Tensor):
            # Skip clamping to preserve dynamics - let the model learn proper ranges
            pass  # constrained['recon'] = torch.clamp(constrained['recon'], min=-1.0, max=1.0)
        
        return constrained


def get_activation(name: str):
    """Get activation function by name."""
    if name == 'gelu':
        return nn.GELU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'silu' or name == 'swish':
        return nn.SiLU()
    elif name == 'mish':
        return nn.Mish()
    else:
        return nn.GELU() 