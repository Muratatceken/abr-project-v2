"""
Output Heads for ABR Hierarchical U-Net

Professional implementation of task-specific output heads
for multi-task learning in ABR signal processing.

Includes:
- Signal reconstruction head
- Peak prediction head (existence, latency, amplitude)  
- Classification head (hearing loss type)
- Threshold regression head

Updated with architectural improvements for better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
import numpy as np


class BaseHead(nn.Module):
    """Base class for output heads with common functionality."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        layer_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim // 2
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
    
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction for better peak and signal analysis."""
    
    def __init__(self, input_dim: int, scales: List[int] = [1, 3, 5, 7]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim // len(scales), 
                     kernel_size=scale, padding=scale//2)
            for scale in scales
        ])
        self.fusion = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale convolutions and fuse features."""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, channels, seq], got {x.shape}")
        
        multi_scale_features = []
        for conv in self.convs:
            feature = conv(x)
            multi_scale_features.append(feature)
        
        # Concatenate and fuse
        fused = torch.cat(multi_scale_features, dim=1)
        return self.fusion(fused)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling mechanism for sequence-to-vector tasks.
    """
    
    def __init__(self, input_dim: int, attention_dim: int = None):
        super().__init__()
        
        if attention_dim is None:
            attention_dim = input_dim // 2
        
        self.attention_dim = attention_dim
        self.input_dim = input_dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim] or [batch, input_dim, seq_len]
            
        Returns:
            Pooled tensor [batch, input_dim]
        """
        # Ensure [batch, seq_len, input_dim] format
        if x.dim() == 3 and x.size(1) == self.input_dim:
            x = x.transpose(1, 2)  # [batch, seq_len, input_dim]
        
        # Compute attention weights
        attention_weights = self.attention(x)  # [batch, seq_len, 1]
        
        # Apply attention pooling
        pooled = torch.sum(x * attention_weights, dim=1)  # [batch, input_dim]
        
        return pooled


class RobustPeakHead(nn.Module):
    """
    Improved peak prediction head with better masking and multi-scale processing.
    
    Addresses the NaN R² issues by:
    1. Proper gradient flow through masking
    2. Multi-scale feature extraction
    3. Uncertainty estimation
    4. Separate decoders for each prediction type
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_uncertainty: bool = True,
        use_multiscale: bool = True,
        latency_range: Tuple[float, float] = (1.0, 8.0),  # ms
        amplitude_range: Tuple[float, float] = (-0.5, 0.5)  # μV
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.use_uncertainty = use_uncertainty
        self.use_multiscale = use_multiscale
        self.latency_range = latency_range
        self.amplitude_range = amplitude_range
        
        # Multi-scale feature extraction
        if use_multiscale:
            self.multiscale_extractor = MultiScaleFeatureExtractor(input_dim)
        else:
            self.multiscale_extractor = None
        
        # Attention pooling for global features
        if use_attention:
            self.attention_pool = AttentionPooling(input_dim)
        
        # Separate feature encoders for each task
        self.existence_encoder = self._make_encoder(input_dim, hidden_dim, num_layers, dropout)
        self.latency_encoder = self._make_encoder(input_dim, hidden_dim, num_layers, dropout)
        self.amplitude_encoder = self._make_encoder(input_dim, hidden_dim, num_layers, dropout)
        
        # Output heads
        self.existence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Logits for BCE
        )
        
        # Latency head with range normalization
        if use_uncertainty:
            self.latency_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)  # [mean, log_std]
            )
        else:
            self.latency_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Amplitude head with range normalization
        if use_uncertainty:
            self.amplitude_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)  # [mean, log_std]
            )
        else:
            self.amplitude_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        self.apply(self._init_weights)
    
    def _make_encoder(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
        """Create a task-specific feature encoder."""
        layers = []
        
        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """Initialize weights properly."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through peak prediction head.
        
        Args:
            x: Input features [batch, channels, seq_len]
            
        Returns:
            Tuple of (existence_logits, latency_pred, amplitude_pred[, uncertainties])
        """
        batch_size = x.size(0)
        
        # Multi-scale feature extraction
        if self.multiscale_extractor is not None:
            x = self.multiscale_extractor(x)
        
        # Global pooling using attention
        if self.use_attention:
            global_features = self.attention_pool(x)
        else:
            # Simple average pooling as fallback
            global_features = x.mean(dim=-1)  # [batch, channels]
        
        # Task-specific feature encoding
        existence_features = self.existence_encoder(global_features)
        latency_features = self.latency_encoder(global_features)
        amplitude_features = self.amplitude_encoder(global_features)
        
        # Predictions
        existence_logits = self.existence_head(existence_features)  # [batch, 1]
        
        if self.use_uncertainty:
            # Latency with uncertainty
            latency_params = self.latency_head(latency_features)  # [batch, 2]
            latency_mean = latency_params[:, 0:1]  # [batch, 1]
            latency_log_std = latency_params[:, 1:2]  # [batch, 1]
            
            # Scale to proper range
            latency_mean = self._scale_to_range(latency_mean, self.latency_range)
            latency_std = torch.exp(latency_log_std.clamp(-10, 2))  # Prevent numerical issues
            
            # Amplitude with uncertainty
            amplitude_params = self.amplitude_head(amplitude_features)  # [batch, 2]
            amplitude_mean = amplitude_params[:, 0:1]  # [batch, 1]
            amplitude_log_std = amplitude_params[:, 1:2]  # [batch, 1]
            
            # Scale to proper range
            amplitude_mean = self._scale_to_range(amplitude_mean, self.amplitude_range)
            amplitude_std = torch.exp(amplitude_log_std.clamp(-10, 2))  # Prevent numerical issues
            
            return (
                existence_logits.squeeze(-1),  # [batch]
                latency_mean.squeeze(-1),      # [batch]
                amplitude_mean.squeeze(-1),    # [batch]
                latency_std.squeeze(-1),       # [batch]
                amplitude_std.squeeze(-1)      # [batch]
            )
        else:
            # Simple predictions without uncertainty
            latency_pred = self.latency_head(latency_features)  # [batch, 1]
            amplitude_pred = self.amplitude_head(amplitude_features)  # [batch, 1]
            
            # Scale to proper ranges
            latency_pred = self._scale_to_range(latency_pred, self.latency_range)
            amplitude_pred = self._scale_to_range(amplitude_pred, self.amplitude_range)
            
            return (
                existence_logits.squeeze(-1),  # [batch]
                latency_pred.squeeze(-1),      # [batch]
                amplitude_pred.squeeze(-1)     # [batch]
            )
    
    def _scale_to_range(self, x: torch.Tensor, target_range: Tuple[float, float]) -> torch.Tensor:
        """Scale predictions to target range using tanh activation."""
        min_val, max_val = target_range
        # Use tanh to map to [-1, 1], then scale to target range
        scaled = torch.tanh(x)
        return min_val + (max_val - min_val) * (scaled + 1) / 2
    
    def compute_masked_loss(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute masked losses for peak predictions.
        
        Args:
            predictions: Tuple of (existence, latency, amplitude[, uncertainties])
            targets: Dict with 'existence', 'latency', 'amplitude' keys
            masks: Boolean mask [batch] indicating valid peaks
            reduction: Loss reduction method
            
        Returns:
            Dictionary of individual losses
        """
        losses = {}
        
        # Unpack predictions
        if len(predictions) == 5:  # With uncertainty
            pred_exist, pred_latency, pred_amplitude, latency_std, amplitude_std = predictions
            use_uncertainty = True
        else:  # Without uncertainty
            pred_exist, pred_latency, pred_amplitude = predictions
            use_uncertainty = False
        
        # Existence loss (no masking needed for this)
        existence_loss = F.binary_cross_entropy_with_logits(
            pred_exist, targets['existence'].float(), reduction=reduction
        )
        losses['existence'] = existence_loss
        
        # Only compute latency/amplitude losses for samples with existing peaks
        if masks.sum() > 0:  # Ensure we have valid samples
            valid_indices = masks.bool()
            
            if use_uncertainty:
                # Uncertainty-aware losses (negative log-likelihood)
                valid_latency_pred = pred_latency[valid_indices]
                valid_latency_std = latency_std[valid_indices]
                valid_latency_target = targets['latency'][valid_indices]
                
                latency_nll = ((valid_latency_pred - valid_latency_target) ** 2 / 
                              (2 * valid_latency_std ** 2) + 
                              torch.log(valid_latency_std * np.sqrt(2 * np.pi)))
                losses['latency'] = latency_nll.mean() if reduction == 'mean' else latency_nll.sum()
                
                valid_amplitude_pred = pred_amplitude[valid_indices]
                valid_amplitude_std = amplitude_std[valid_indices]
                valid_amplitude_target = targets['amplitude'][valid_indices]
                
                amplitude_nll = ((valid_amplitude_pred - valid_amplitude_target) ** 2 / 
                                (2 * valid_amplitude_std ** 2) + 
                                torch.log(valid_amplitude_std * np.sqrt(2 * np.pi)))
                losses['amplitude'] = amplitude_nll.mean() if reduction == 'mean' else amplitude_nll.sum()
            else:
                # Standard MSE losses for valid samples only
                valid_latency_pred = pred_latency[valid_indices]
                valid_latency_target = targets['latency'][valid_indices]
                losses['latency'] = F.mse_loss(valid_latency_pred, valid_latency_target, reduction=reduction)
                
                valid_amplitude_pred = pred_amplitude[valid_indices]
                valid_amplitude_target = targets['amplitude'][valid_indices]
                losses['amplitude'] = F.mse_loss(valid_amplitude_pred, valid_amplitude_target, reduction=reduction)
        else:
            # No valid peaks - set losses to zero but keep gradients
            device = pred_exist.device
            losses['latency'] = torch.tensor(0.0, device=device, requires_grad=True)
            losses['amplitude'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        return losses


# Keep the original EnhancedPeakHead as an alias for backward compatibility
EnhancedPeakHead = RobustPeakHead


class EnhancedSignalHead(nn.Module):
    """
    Enhanced signal reconstruction head with attention over decoder output.
    """
    
    def __init__(
        self,
        input_dim: int,
        signal_length: int,
        hidden_dim: int = None,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_sequence_modeling: bool = True
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim * 2
        
        self.input_dim = input_dim
        self.signal_length = signal_length
        self.use_attention = use_attention
        self.use_sequence_modeling = use_sequence_modeling
        
        # Attention pooling for global features
        if use_attention:
            self.attention_pool = AttentionPooling(input_dim)
        else:
            self.attention_pool = None
        
        # Sequence modeling for fine-grained reconstruction
        if use_sequence_modeling:
            self.sequence_processor = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=max(1, input_dim // 64),
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=2
            )
        else:
            self.sequence_processor = None
        
        # Main MLP layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                in_features = input_dim
            else:
                in_features = hidden_dim
            
            if i == num_layers - 1:
                out_features = signal_length
                activation = nn.Identity()
            else:
                out_features = hidden_dim
                activation = nn.GELU()
            
            layers.extend([
                nn.Linear(in_features, out_features),
                activation,
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity()
            ])
        
        self.mlp = nn.Sequential(*layers)
        
        # Optional output refinement
        self.output_refinement = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate signal reconstruction.
        
        Args:
            x: Input features [batch, input_dim] or [batch, seq_len, input_dim] or [batch, input_dim, seq_len]
            
        Returns:
            Reconstructed signal [batch, signal_length]
        """
        if x.dim() == 3:
            # Process sequence input
            if self.sequence_processor is not None:
                # Ensure [batch, seq_len, input_dim] format
                if x.size(1) == self.input_dim:
                    x = x.transpose(1, 2)
                x = self.sequence_processor(x)
            
            # Pool to vector
            if self.attention_pool is not None:
                x = self.attention_pool(x)
            else:
                x = x.mean(dim=1)  # Simple average pooling
        
        # Generate signal through MLP
        signal = self.mlp(x)  # [batch, signal_length]
        
        # Optional refinement with 1D convolution
        signal_refined = signal.unsqueeze(1)  # [batch, 1, signal_length]
        signal_refined = self.output_refinement(signal_refined)
        signal_refined = signal_refined.squeeze(1)  # [batch, signal_length]
        
        # Residual connection
        signal = signal + signal_refined
        
        return signal


class RobustClassificationHead(nn.Module):
    """
    Improved classification head with focal loss support and better minority class handling.
    
    Addresses classification issues by:
    1. Hierarchical feature learning
    2. Class-aware attention mechanisms
    3. Built-in focal loss support
    4. Better handling of class imbalance
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = None,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_focal_loss_prep: bool = True,
        use_class_weights: bool = True,
        temperature: float = 1.0
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_focal_loss_prep = use_focal_loss_prep
        self.temperature = temperature
        
        # Multi-scale feature extractor for classification
        self.multiscale_extractor = MultiScaleFeatureExtractor(input_dim)
        
        # Attention pooling
        if use_attention:
            self.attention_pool = AttentionPooling(input_dim)
        
        # Hierarchical feature learning
        self.feature_encoder = self._make_hierarchical_encoder(input_dim, hidden_dim, num_layers, dropout)
        
        # Class-specific feature extractors (for better minority class handling)
        self.class_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4)
            )
            for _ in range(num_classes)
        ])
        
        # Final classification head
        total_feature_dim = hidden_dim + (hidden_dim // 4) * num_classes
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Class weights for balanced learning (will be set dynamically)
        if use_class_weights:
            self.register_buffer('class_weights', torch.ones(num_classes))
        else:
            self.class_weights = None
        
        self.apply(self._init_weights)
    
    def _make_hierarchical_encoder(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
        """Create hierarchical feature encoder with residual connections."""
        layers = []
        
        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
        
        # Residual blocks
        for i in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim, dropout))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """Initialize weights properly."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def update_class_weights(self, class_counts: torch.Tensor):
        """Update class weights based on class distribution."""
        if self.class_weights is not None:
            total_samples = class_counts.sum()
            weights = total_samples / (len(class_counts) * class_counts.clamp(min=1))
            self.class_weights.data = weights / weights.sum() * len(class_counts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            x: Input features [batch, channels, seq_len]
            
        Returns:
            Classification logits [batch, num_classes]
        """
        # Multi-scale feature extraction
        x = self.multiscale_extractor(x)
        
        # Global pooling using attention
        if self.use_attention:
            global_features = self.attention_pool(x)
        else:
            global_features = x.mean(dim=-1)
        
        # Hierarchical feature encoding
        encoded_features = self.feature_encoder(global_features)
        
        # Class-specific feature extraction
        class_features = []
        for extractor in self.class_extractors:
            class_feature = extractor(encoded_features)
            class_features.append(class_feature)
        
        # Combine all features
        all_features = torch.cat([encoded_features] + class_features, dim=1)
        
        # Final classification
        logits = self.classifier(all_features)
        
        # Apply temperature scaling if requested
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        return logits
    
    def compute_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Compute focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.net(x))


class RobustThresholdHead(nn.Module):
    """
    Improved threshold regression head with robust loss and better normalization.
    
    Addresses threshold estimation issues by:
    1. Robust regression with multiple objectives
    2. Better normalization and scaling
    3. Uncertainty estimation
    4. Outlier-resistant loss functions
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
        use_attention_pooling: bool = True,
        use_uncertainty: bool = True,
        use_robust_loss: bool = True,
        threshold_range: Tuple[float, float] = (0.0, 120.0),
        use_multiscale: bool = True
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_uncertainty = use_uncertainty
        self.use_robust_loss = use_robust_loss
        self.threshold_range = threshold_range
        self.use_multiscale = use_multiscale
        
        # Multi-scale feature extraction
        if use_multiscale:
            self.multiscale_extractor = MultiScaleFeatureExtractor(input_dim)
        
        # Attention pooling
        if use_attention_pooling:
            self.attention_pool = AttentionPooling(input_dim)
        else:
            self.attention_pool = None
        
        # Feature encoder with multiple paths
        self.global_encoder = self._make_encoder(input_dim, hidden_dim, 3, dropout)
        self.local_encoder = self._make_encoder(input_dim, hidden_dim, 2, dropout)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multiple regression heads for robustness
        if use_uncertainty:
            # Main predictor (mean and log-std)
            self.main_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)  # [mean, log_std]
            )
            
            # Auxiliary predictor for regularization
            self.aux_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1)
            )
        else:
            # Simple regression head
            self.main_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.aux_head = None
        
        # Outlier detection head (for robust loss)
        if use_robust_loss:
            self.outlier_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()  # Probability of being an outlier
            )
        else:
            self.outlier_head = None
        
        self.apply(self._init_weights)
    
    def _make_encoder(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
        """Create feature encoder."""
        layers = []
        
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
        
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim, dropout))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """Initialize weights properly."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through threshold head.
        
        Args:
            x: Input features [batch, channels, seq_len]
            
        Returns:
            Threshold predictions [batch, 1] or [batch, 2] if uncertainty
        """
        # Multi-scale feature extraction
        if self.use_multiscale:
            x = self.multiscale_extractor(x)
        
        # Global pooling
        if self.attention_pool is not None:
            global_features = self.attention_pool(x)
        else:
            global_features = x.mean(dim=-1)
        
        # Dual-path encoding
        global_encoded = self.global_encoder(global_features)
        local_encoded = self.local_encoder(global_features)
        
        # Fusion
        fused_features = self.fusion(torch.cat([global_encoded, local_encoded], dim=1))
        
        # Main prediction
        main_pred = self.main_head(fused_features)
        
        if self.use_uncertainty:
            # Split into mean and log_std
            pred_mean = main_pred[:, 0:1]
            pred_log_std = main_pred[:, 1:2]
            
            # Scale mean to proper range
            pred_mean = self._scale_to_range(pred_mean, self.threshold_range)
            pred_std = torch.exp(pred_log_std.clamp(-10, 2))
            
            return torch.cat([pred_mean, pred_std], dim=1)
        else:
            # Simple prediction
            pred = self._scale_to_range(main_pred, self.threshold_range)
            return pred
    
    def _scale_to_range(self, x: torch.Tensor, target_range: Tuple[float, float]) -> torch.Tensor:
        """Scale predictions to target range."""
        min_val, max_val = target_range
        # Use sigmoid to map to [0, 1], then scale to target range
        scaled = torch.sigmoid(x)
        return min_val + (max_val - min_val) * scaled
    
    def compute_robust_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute robust threshold loss with outlier handling.
        
        Args:
            predictions: Model predictions [batch, 1] or [batch, 2]
            targets: Target threshold values [batch]
            reduction: Loss reduction method
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        if self.use_uncertainty and predictions.size(-1) == 2:
            # Uncertainty-aware loss
            pred_mean = predictions[:, 0]
            pred_std = predictions[:, 1]
            
            # Negative log-likelihood
            nll = ((pred_mean - targets) ** 2 / (2 * pred_std ** 2) + 
                   torch.log(pred_std * np.sqrt(2 * np.pi)))
            
            losses['main_loss'] = nll.mean() if reduction == 'mean' else nll.sum()
        else:
            # Standard regression loss
            pred_threshold = predictions.squeeze(-1)
            
            # Huber loss (more robust to outliers than MSE)
            if self.use_robust_loss:
                losses['main_loss'] = F.huber_loss(pred_threshold, targets, delta=1.0, reduction=reduction)
            else:
                losses['main_loss'] = F.mse_loss(pred_threshold, targets, reduction=reduction)
        
        # Auxiliary loss for regularization
        if self.aux_head is not None:
            aux_pred = self.aux_head(self.last_features).squeeze(-1)
            aux_pred_scaled = self._scale_to_range(aux_pred.unsqueeze(-1), self.threshold_range).squeeze(-1)
            losses['aux_loss'] = F.mse_loss(aux_pred_scaled, targets, reduction=reduction)
        
        return losses


# Keep original class names as aliases for backward compatibility
EnhancedClassificationHead = RobustClassificationHead
EnhancedThresholdHead = RobustThresholdHead


class StaticParameterGenerationHead(nn.Module):
    """
    Static parameter generation head for joint generation of ABR signals and static parameters.
    
    Generates realistic values for:
    - Age (continuous, normalized)
    - Intensity (continuous, normalized) 
    - Stimulus Rate (continuous, normalized)
    - FMP (continuous, normalized)
    
    Supports both unconditional generation and conditional generation with constraints.
    """
    
    def __init__(
        self,
        input_dim: int,
        static_dim: int = 4,  # age, intensity, stimulus_rate, fmp
        hidden_dim: int = None,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_uncertainty: bool = True,
        use_constraints: bool = True,
        parameter_ranges: Dict[str, Tuple[float, float]] = None
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.use_uncertainty = use_uncertainty
        self.use_constraints = use_constraints
        
        # Default parameter ranges (normalized values from dataset analysis)
        if parameter_ranges is None:
            self.parameter_ranges = {
                'age': (-0.36, 11.37),           # Age range from dataset
                'intensity': (-2.61, 1.99),     # Intensity range
                'stimulus_rate': (-6.79, 5.10), # Stimulus rate range
                'fmp': (-0.20, 129.11)          # FMP range
            }
        else:
            self.parameter_ranges = parameter_ranges
        
        # Multi-scale feature extraction for better parameter generation
        self.multiscale_extractor = MultiScaleFeatureExtractor(input_dim)
        
        # Attention pooling for global context
        if use_attention:
            self.attention_pool = AttentionPooling(input_dim)
        
        # Parameter-specific encoders for better generation
        self.param_encoders = nn.ModuleList([
            self._make_param_encoder(input_dim, hidden_dim, num_layers, dropout)
            for _ in range(static_dim)
        ])
        
        # Parameter names for easier debugging
        self.param_names = ['age', 'intensity', 'stimulus_rate', 'fmp']
        
        # Output heads for each parameter
        if use_uncertainty:
            # Generate mean and log_std for each parameter
            self.param_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 2)  # [mean, log_std]
                )
                for _ in range(static_dim)
            ])
        else:
            # Simple point estimates
            self.param_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1)
                )
                for _ in range(static_dim)
            ])
        
        # Cross-parameter dependency modeling (important for realistic generation)
        self.dependency_encoder = nn.Sequential(
            nn.Linear(hidden_dim * static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, static_dim)  # Dependency adjustment
        )
        
        self.apply(self._init_weights)
    
    def _make_param_encoder(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
        """Create parameter-specific encoder."""
        layers = []
        
        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
        
        # Residual blocks
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim, dropout))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """Initialize weights properly."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate static parameters from input features.
        
        Args:
            x: Input features [batch, channels, seq_len]
            
        Returns:
            Generated static parameters [batch, static_dim] or [batch, static_dim, 2] if uncertainty
        """
        batch_size = x.size(0)
        
        # Multi-scale feature extraction
        x = self.multiscale_extractor(x)
        
        # Global pooling using attention
        if self.use_attention:
            global_features = self.attention_pool(x)
        else:
            global_features = x.mean(dim=-1)
        
        # Parameter-specific encoding
        param_features = []
        for encoder in self.param_encoders:
            param_feature = encoder(global_features)
            param_features.append(param_feature)
        
        # Model cross-parameter dependencies
        combined_features = torch.cat(param_features, dim=1)  # [batch, hidden_dim * static_dim]
        dependency_adjustment = self.dependency_encoder(combined_features)  # [batch, static_dim]
        
        # Generate parameters
        generated_params = []
        
        for i, (param_head, param_feature) in enumerate(zip(self.param_heads, param_features)):
            if self.use_uncertainty:
                # Generate mean and log_std
                param_output = param_head(param_feature)  # [batch, 2]
                param_mean = param_output[:, 0:1]  # [batch, 1]
                param_log_std = param_output[:, 1:2]  # [batch, 1]
                
                # Apply dependency adjustment to mean
                param_mean = param_mean + dependency_adjustment[:, i:i+1] * 0.1
                
                # Scale to parameter range
                param_name = self.param_names[i]
                param_range = self.parameter_ranges[param_name]
                scaled_mean = self._scale_to_range(param_mean, param_range)
                scaled_std = torch.exp(param_log_std.clamp(-10, 2))
                
                generated_params.append(torch.cat([scaled_mean, scaled_std], dim=1))
            else:
                # Simple point estimate
                param_output = param_head(param_feature)  # [batch, 1]
                
                # Apply dependency adjustment
                param_output = param_output + dependency_adjustment[:, i:i+1] * 0.1
                
                # Scale to parameter range
                param_name = self.param_names[i]
                param_range = self.parameter_ranges[param_name]
                scaled_param = self._scale_to_range(param_output, param_range)
                
                generated_params.append(scaled_param)
        
        if self.use_uncertainty:
            # Stack and return [batch, static_dim, 2]
            return torch.stack(generated_params, dim=1)  # [batch, static_dim, 2]
        else:
            # Concatenate and return [batch, static_dim]
            return torch.cat(generated_params, dim=1)  # [batch, static_dim]
    
    def _scale_to_range(self, x: torch.Tensor, target_range: Tuple[float, float]) -> torch.Tensor:
        """Scale parameters to target range using tanh activation."""
        min_val, max_val = target_range
        # Use tanh to map to [-1, 1], then scale to target range
        scaled = torch.tanh(x)
        return min_val + (max_val - min_val) * (scaled + 1) / 2
    
    def sample_parameters(
        self, 
        x: torch.Tensor, 
        temperature: float = 1.0,
        use_constraints: bool = True
    ) -> torch.Tensor:
        """
        Sample static parameters with optional temperature scaling and constraints.
        
        Args:
            x: Input features [batch, channels, seq_len]
            temperature: Temperature for sampling (higher = more random)
            use_constraints: Whether to apply clinical constraints
            
        Returns:
            Sampled static parameters [batch, static_dim]
        """
        with torch.no_grad():
            param_output = self.forward(x)
            
            if self.use_uncertainty:
                # Sample from distributions
                means = param_output[:, :, 0]  # [batch, static_dim]
                stds = param_output[:, :, 1]   # [batch, static_dim]
                
                # Apply temperature scaling
                scaled_stds = stds * temperature
                
                # Sample from normal distributions
                noise = torch.randn_like(means)
                sampled_params = means + scaled_stds * noise
            else:
                # Add noise for sampling
                sampled_params = param_output + torch.randn_like(param_output) * temperature * 0.1
            
            # Apply clinical constraints if requested
            if use_constraints and self.use_constraints:
                sampled_params = self._apply_clinical_constraints(sampled_params)
            
            return sampled_params
    
    def _apply_clinical_constraints(self, params: torch.Tensor) -> torch.Tensor:
        """Apply clinical constraints to ensure realistic parameter combinations."""
        # Clone to avoid in-place operations
        constrained_params = params.clone()
        
        # Example constraints (can be expanded based on clinical knowledge):
        # 1. Very young patients (age < -0.2 normalized) should have lower intensities
        young_mask = constrained_params[:, 0] < -0.2  # Young patients
        if young_mask.any():
            # Reduce intensity for young patients
            constrained_params[young_mask, 1] = torch.clamp(
                constrained_params[young_mask, 1], 
                max=constrained_params[young_mask, 1] * 0.8
            )
        
        # 2. High stimulus rates should typically have moderate intensities
        high_rate_mask = constrained_params[:, 2] > 2.0  # High stimulus rate
        if high_rate_mask.any():
            # Moderate intensity for high rates
            constrained_params[high_rate_mask, 1] = torch.clamp(
                constrained_params[high_rate_mask, 1],
                min=-1.0, max=1.0
            )
        
        # 3. Ensure FMP is reasonable relative to other parameters
        # High FMP should correlate with certain intensity ranges
        high_fmp_mask = constrained_params[:, 3] > 50.0  # High FMP
        if high_fmp_mask.any():
            # Adjust intensity for high FMP
            constrained_params[high_fmp_mask, 1] = torch.clamp(
                constrained_params[high_fmp_mask, 1],
                min=-0.5, max=1.5
            )
        
        return constrained_params
    
    def compute_generation_loss(
        self,
        generated_params: torch.Tensor,
        target_params: torch.Tensor,
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for static parameter generation.
        
        Args:
            generated_params: Generated parameters [batch, static_dim] or [batch, static_dim, 2]
            target_params: Target parameters [batch, static_dim]
            reduction: Loss reduction method
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        if self.use_uncertainty and generated_params.dim() == 3:
            # Uncertainty-aware loss
            pred_means = generated_params[:, :, 0]  # [batch, static_dim]
            pred_stds = generated_params[:, :, 1]   # [batch, static_dim]
            
            # Negative log-likelihood for each parameter
            param_losses = []
            for i in range(self.static_dim):
                pred_mean = pred_means[:, i]
                pred_std = pred_stds[:, i]
                target = target_params[:, i]
                
                nll = ((pred_mean - target) ** 2 / (2 * pred_std ** 2) + 
                       torch.log(pred_std * np.sqrt(2 * np.pi)))
                param_losses.append(nll)
            
            # Individual parameter losses
            for i, param_name in enumerate(self.param_names):
                losses[f'static_{param_name}'] = param_losses[i].mean() if reduction == 'mean' else param_losses[i].sum()
            
            # Total static parameter loss
            total_loss = torch.stack(param_losses).mean(dim=0)
            losses['static_total'] = total_loss.mean() if reduction == 'mean' else total_loss.sum()
        else:
            # Standard MSE loss for each parameter
            param_losses = []
            for i in range(self.static_dim):
                pred = generated_params[:, i]
                target = target_params[:, i]
                loss = F.mse_loss(pred, target, reduction='none')
                param_losses.append(loss)
            
            # Individual parameter losses
            for i, param_name in enumerate(self.param_names):
                losses[f'static_{param_name}'] = param_losses[i].mean() if reduction == 'mean' else param_losses[i].sum()
            
            # Total static parameter loss
            total_loss = torch.stack(param_losses).mean(dim=0)
            losses['static_total'] = total_loss.mean() if reduction == 'mean' else total_loss.sum()
        
        return losses 