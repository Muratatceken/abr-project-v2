"""
Output Heads for ABR Hierarchical U-Net

Professional implementation of task-specific output heads
for multi-task learning in ABR signal processing.

Includes:
- Signal reconstruction head
- Peak prediction head (existence, latency, amplitude)  
- Classification head (hearing loss type)
- Threshold regression head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
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


class EnhancedPeakHead(nn.Module):
    """
    Enhanced peak prediction head with separate specialized MLPs and masked loss support.
    
    Features:
    - Separate MLPs for existence, latency, and amplitude prediction
    - Built-in support for peak masking during loss computation
    - Uncertainty estimation (optional)
    - Attention-based pooling for sequence inputs
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_uncertainty: bool = False,
        apply_mask_in_forward: bool = False  # Whether to apply mask in forward pass
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.input_dim = input_dim
        self.use_attention = use_attention
        self.use_uncertainty = use_uncertainty
        self.apply_mask_in_forward = apply_mask_in_forward
        
        # Attention pooling
        if use_attention:
            self.attention_pool = AttentionPooling(input_dim)
        
        # Shared feature extractor
        shared_layers = []
        for i in range(num_layers):
            if i == 0:
                in_features = input_dim
            else:
                in_features = hidden_dim
            
            shared_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        self.shared_features = nn.Sequential(*shared_layers)
        
        # Binary existence prediction (sigmoid)
        self.existence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Latency regression with specialized processing
        self.latency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Amplitude regression with specialized processing
        self.amplitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Uncertainty estimation (if enabled)
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3),  # 3 uncertainties: existence, latency, amplitude
                nn.Softplus()
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        peak_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict peak information with optional masking support.
        
        Args:
            x: Input features [batch, input_dim] or [batch, seq_len, input_dim] or [batch, input_dim, seq_len]
            peak_mask: Optional peak mask [batch] - 1 where peak exists, 0 otherwise
            
        Returns:
            Tuple of (existence_prob, latency, amplitude) [batch] each
            
        Note:
            For masked loss computation, use the returned values as:
            amp_loss = (pred_amp - true_amp)**2 * peak_mask
            lat_loss = (pred_lat - true_lat)**2 * peak_mask
        """
        # Handle different input shapes
        if x.dim() == 3:
            if self.use_attention:
                x = self.attention_pool(x)
            else:
                x = x.mean(dim=1) if x.size(1) != self.input_dim else x.mean(dim=2)
        
        # Extract shared features
        features = self.shared_features(x)
        
        # Separate predictions
        existence = self.existence_head(features).squeeze(-1)  # [batch]
        latency = self.latency_head(features).squeeze(-1)      # [batch]
        amplitude = self.amplitude_head(features).squeeze(-1)  # [batch]
        
        # Apply masking if requested and mask is provided
        if self.apply_mask_in_forward and peak_mask is not None:
            # Apply mask to regression outputs (existence is always predicted)
            latency = latency * peak_mask
            amplitude = amplitude * peak_mask
        
        if self.use_uncertainty:
            uncertainties = self.uncertainty_head(features)  # [batch, 3]
            return existence, latency, amplitude, uncertainties
        
        return existence, latency, amplitude
    
    def compute_masked_loss(
        self,
        pred_existence: torch.Tensor,
        pred_latency: torch.Tensor,
        pred_amplitude: torch.Tensor,
        true_existence: torch.Tensor,
        true_latency: torch.Tensor,
        true_amplitude: torch.Tensor,
        peak_mask: torch.Tensor,
        existence_weight: float = 1.0,
        latency_weight: float = 1.0,
        amplitude_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute masked losses for peak prediction as specified in Fix 2.
        
        Args:
            pred_existence: Predicted existence probability [batch]
            pred_latency: Predicted latency [batch]
            pred_amplitude: Predicted amplitude [batch]
            true_existence: True existence (0 or 1) [batch]
            true_latency: True latency [batch]
            true_amplitude: True amplitude [batch]
            peak_mask: Peak mask (1 where peak exists, 0 otherwise) [batch]
            existence_weight: Weight for existence loss
            latency_weight: Weight for latency loss
            amplitude_weight: Weight for amplitude loss
            
        Returns:
            Dictionary with individual and total losses
        """
        # Existence loss (BCE) - always computed (no masking needed)
        existence_loss = F.binary_cross_entropy(
            pred_existence, 
            true_existence.float()
        )
        
        # Masked regression losses (only where peaks exist)
        # amp_loss = (pred_amp - true_amp)**2 * peak_mask
        # lat_loss = (pred_lat - true_lat)**2 * peak_mask
        latency_loss_unmasked = (pred_latency - true_latency) ** 2
        amplitude_loss_unmasked = (pred_amplitude - true_amplitude) ** 2
        
        # Apply peak mask
        latency_loss_masked = latency_loss_unmasked * peak_mask
        amplitude_loss_masked = amplitude_loss_unmasked * peak_mask
        
        # Take mean only over samples where peaks exist
        num_peaks = peak_mask.sum().clamp(min=1)  # Avoid division by zero
        latency_loss = latency_loss_masked.sum() / num_peaks
        amplitude_loss = amplitude_loss_masked.sum() / num_peaks
        
        # Total weighted loss
        total_loss = (
            existence_weight * existence_loss +
            latency_weight * latency_loss +
            amplitude_weight * amplitude_loss
        )
        
        return {
            'existence_loss': existence_loss,
            'latency_loss': latency_loss,
            'amplitude_loss': amplitude_loss,
            'total_loss': total_loss,
            'num_peaks': num_peaks
        }


class EnhancedClassificationHead(nn.Module):
    """
    Enhanced classification head with attention pooling and class balancing.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss_prep: bool = True
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_focal_loss_prep = use_focal_loss_prep
        
        # Register class weights
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        # Attention pooling
        if use_attention:
            self.attention_pool = AttentionPooling(input_dim)
        
        # Feature processing layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                in_features = input_dim
            else:
                in_features = hidden_dim
            
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Classification head with potential focal loss preparation
        if use_focal_loss_prep:
            # Add an intermediate layer for better gradient flow
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Optional temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input features.
        
        Args:
            x: Input features [batch, input_dim] or [batch, seq_len, input_dim] or [batch, input_dim, seq_len]
            
        Returns:
            Classification logits [batch, num_classes]
        """
        # Handle different input shapes
        if x.dim() == 3:
            if self.use_attention:
                x = self.attention_pool(x)
            else:
                x = x.mean(dim=1) if x.size(1) != self.input_dim else x.mean(dim=2)
        
        # Feature processing
        features = self.feature_layers(x)
        
        # Classification
        logits = self.classifier(features)
        
        # Temperature scaling
        logits = logits / self.temperature
        
        return logits


class EnhancedThresholdHead(nn.Module):
    """
    Enhanced threshold regression head with log-scale loss support and uncertainty estimation.
    
    Features:
    - Attention pooling or mean pooling
    - Log-scale regression for better threshold prediction
    - Optional uncertainty estimation (μ, σ outputs)
    - Clinical constraint enforcement
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        use_attention_pooling: bool = True,
        use_uncertainty: bool = False,
        use_log_scale: bool = True,
        dropout: float = 0.1,
        threshold_range: Tuple[float, float] = (0.0, 120.0)
    ):
        """
        Initialize enhanced threshold head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            use_attention_pooling: Whether to use attention pooling
            use_uncertainty: Whether to predict uncertainty (σ) along with mean (μ)
            use_log_scale: Whether to use log-scale for better regression
            dropout: Dropout rate
            threshold_range: Valid threshold range (dB SPL)
        """
        super().__init__()
        
        self.use_attention_pooling = use_attention_pooling
        self.use_uncertainty = use_uncertainty
        self.use_log_scale = use_log_scale
        self.threshold_range = threshold_range
        
        # Pooling layer
        if use_attention_pooling:
            self.pooling = AttentionPooling(input_dim)
        else:
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Feature processing
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Output layers
        output_dim = 2 if use_uncertainty else 1
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
        
        # Initialize weights for better threshold regression
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable threshold regression."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize output layer to predict reasonable threshold values
        if hasattr(self, 'output_layer'):
            # Initialize to predict mid-range thresholds
            mid_threshold = (self.threshold_range[0] + self.threshold_range[1]) / 2
            if self.use_log_scale:
                init_value = np.log1p(mid_threshold)
            else:
                init_value = mid_threshold
            
            nn.init.constant_(self.output_layer.bias[0], init_value)
            if self.use_uncertainty and self.output_layer.bias.size(0) > 1:
                nn.init.constant_(self.output_layer.bias[1], -2.0)  # Small initial uncertainty
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for threshold prediction.
        
        Args:
            x: Input features [batch, channels, sequence_length]
            
        Returns:
            Threshold predictions [batch, 1] or [batch, 2] if uncertainty enabled
        """
        # Apply pooling
        if self.use_attention_pooling:
            pooled = self.pooling(x)  # [batch, channels]
        else:
            pooled = self.pooling(x).squeeze(-1)  # [batch, channels]
        
        # Feature processing
        features = self.feature_layers(pooled)  # [batch, hidden_dim // 2]
        
        # Output prediction
        output = self.output_layer(features)  # [batch, output_dim]
        
        # Apply constraints and transformations
        if self.use_uncertainty:
            mu = output[:, 0:1]  # [batch, 1]
            log_sigma = output[:, 1:2]  # [batch, 1]
            
            # Apply log-scale transformation to mean
            if self.use_log_scale:
                # Transform from log space back to threshold space
                mu = torch.expm1(mu)  # Inverse of log1p
            
            # Ensure positive sigma and apply constraints
            sigma = F.softplus(log_sigma) + 1e-6
            
            # Apply threshold range constraints to mu
            mu = torch.clamp(mu, self.threshold_range[0], self.threshold_range[1])
            
            # Combine mu and sigma
            output = torch.cat([mu, sigma], dim=1)  # [batch, 2]
        else:
            # Single threshold prediction
            if self.use_log_scale:
                # Transform from log space back to threshold space
                output = torch.expm1(output)  # [batch, 1]
            
            # Apply threshold range constraints
            output = torch.clamp(output, self.threshold_range[0], self.threshold_range[1])
        
        return output
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute threshold regression loss with log-scale and uncertainty support.
        
        Args:
            predictions: Model predictions [batch, 1] or [batch, 2]
            targets: Target threshold values [batch]
            reduction: Loss reduction method
            
        Returns:
            Computed loss tensor
        """
        if self.use_uncertainty and predictions.size(-1) == 2:
            # Uncertainty-aware loss (negative log-likelihood)
            pred_mu = predictions[:, 0]  # [batch]
            pred_sigma = predictions[:, 1]  # [batch]
            
            # NLL loss: -log p(y|μ,σ) = 0.5 * ((y-μ)/σ)² + log(σ) + 0.5*log(2π)
            nll = ((pred_mu - targets) ** 2 / (2 * pred_sigma ** 2) + 
                   torch.log(pred_sigma * np.sqrt(2 * np.pi)))
            
            if reduction == 'mean':
                return nll.mean()
            elif reduction == 'sum':
                return nll.sum()
            else:
                return nll
        else:
            # Standard regression loss
            pred_threshold = predictions.squeeze(-1)  # [batch]
            
            if self.use_log_scale:
                # Log-scale MSE loss
                log_pred = torch.log1p(torch.clamp(pred_threshold, min=0))
                log_target = torch.log1p(torch.clamp(targets, min=0))
                return F.mse_loss(log_pred, log_target, reduction=reduction)
            else:
                # Standard MSE loss
                return F.mse_loss(pred_threshold, targets, reduction=reduction)


# Update existing heads to use enhanced versions by default
class SignalHead(EnhancedSignalHead):
    """Signal reconstruction head (enhanced version)."""
    pass

class PeakHead(EnhancedPeakHead):
    """Peak prediction head (enhanced version)."""
    pass

class ClassificationHead(EnhancedClassificationHead):
    """Classification head (enhanced version)."""
    pass

class ThresholdHead(EnhancedThresholdHead):
    """Threshold regression head (enhanced version)."""
    pass


class MultiTaskHead(nn.Module):
    """
    Combined multi-task head that outputs all predictions simultaneously.
    
    Provides a unified interface for all ABR prediction tasks
    with shared feature extraction and task-specific heads.
    """
    
    def __init__(
        self,
        input_dim: int,
        signal_length: int = 200,
        num_classes: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_shared_features: bool = True,
        predict_uncertainty: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.use_shared_features = use_shared_features
        
        # Shared feature extraction
        if use_shared_features:
            self.shared_features = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim or input_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            head_input_dim = hidden_dim or input_dim
        else:
            self.shared_features = nn.Identity()
            head_input_dim = input_dim
        
        # Individual task heads
        self.signal_head = SignalHead(
            input_dim=head_input_dim,
            signal_length=signal_length,
            dropout=dropout
        )
        
        self.peak_head = PeakHead(
            input_dim=head_input_dim,
            dropout=dropout
        )
        
        self.class_head = ClassificationHead(
            input_dim=head_input_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.threshold_head = ThresholdHead(
            input_dim=head_input_dim,
            dropout=dropout,
            predict_uncertainty=predict_uncertainty
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task head.
        
        Args:
            x: Input features [batch, seq_len, input_dim] or [batch, input_dim]
            
        Returns:
            Dictionary with task predictions:
            {
                'signal': [batch, signal_length],
                'peak_exists': [batch, 1],
                'peak_latency': [batch, 1], 
                'peak_amplitude': [batch, 1],
                'class': [batch, num_classes],
                'threshold': [batch, 1] or [batch, 2]
            }
        """
        # Extract shared features
        shared_features = self.shared_features(x)
        
        # Get predictions from each head
        signal = self.signal_head(shared_features)
        peak_exists, peak_latency, peak_amplitude = self.peak_head(shared_features)
        class_logits = self.class_head(shared_features)
        threshold = self.threshold_head(shared_features)
        
        return {
            'signal': signal,
            'peak_exists': peak_exists,
            'peak_latency': peak_latency,
            'peak_amplitude': peak_amplitude,
            'class': class_logits,
            'threshold': threshold
        } 