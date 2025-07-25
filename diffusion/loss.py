#!/usr/bin/env python3
"""
Enhanced Loss Functions for ABR Diffusion Model

Implements comprehensive loss functions for multi-task ABR signal generation including:
- Signal reconstruction loss
- Peak prediction with proper masking
- Classification loss with class weighting
- Threshold regression with log-scale loss
- Curriculum learning support

Author: AI Assistant
Date: January 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ABRDiffusionLoss(nn.Module):
    """
    Comprehensive loss function for ABR diffusion model with curriculum learning support.
    
    Combines:
    - Signal reconstruction loss (MSE/Huber)
    - Peak prediction with proper masking
    - Classification with class weighting/focal loss
    - Threshold regression with log-scale loss
    """
    
    def __init__(
        self,
        n_classes: int = 5,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        peak_loss_type: str = 'mse',
        huber_delta: float = 1.0,
        use_log_threshold: bool = True,
        use_uncertainty_threshold: bool = False,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.use_focal_loss = use_focal_loss
        self.peak_loss_type = peak_loss_type
        self.huber_delta = huber_delta
        self.use_log_threshold = use_log_threshold
        self.use_uncertainty_threshold = use_uncertainty_threshold
        self.device = device or torch.device('cpu')
        
        # Classification loss
        if use_focal_loss:
            self.classification_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.classification_loss = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device) if class_weights is not None else None
            )
        
        # Peak existence loss
        self.peak_exist_loss = nn.BCEWithLogitsLoss()
        
        # Initialize loss weights (will be updated by curriculum learning)
        self.loss_weights = {
            'signal': 1.0,
            'peak_exist': 0.5,
            'peak_latency': 1.0,
            'peak_amplitude': 1.0,
            'classification': 1.5,
            'threshold': 0.8
        }
    
    def update_loss_weights(self, weights: Dict[str, float]):
        """Update loss weights (used for curriculum learning)."""
        self.loss_weights.update(weights)
    
    def compute_signal_loss(self, pred_signal: torch.Tensor, true_signal: torch.Tensor) -> torch.Tensor:
        """Compute signal reconstruction loss."""
        # Ensure shape compatibility
        if pred_signal.shape != true_signal.shape:
            # Handle common shape mismatches
            if pred_signal.dim() == 2 and true_signal.dim() == 3:
                # pred_signal: [batch, seq_len], true_signal: [batch, 1, seq_len]
                if true_signal.size(1) == 1:
                    true_signal = true_signal.squeeze(1)  # [batch, seq_len]
            elif pred_signal.dim() == 3 and true_signal.dim() == 2:
                # pred_signal: [batch, 1, seq_len], true_signal: [batch, seq_len]
                if pred_signal.size(1) == 1:
                    pred_signal = pred_signal.squeeze(1)  # [batch, seq_len]
            elif pred_signal.dim() == 3 and true_signal.dim() == 3:
                # Both are 3D, ensure they have the same shape
                if pred_signal.size(1) == 1 and true_signal.size(1) != 1:
                    pred_signal = pred_signal.squeeze(1)
                elif true_signal.size(1) == 1 and pred_signal.size(1) != 1:
                    true_signal = true_signal.squeeze(1)
        
        if self.peak_loss_type == 'huber':
            return F.huber_loss(pred_signal, true_signal, delta=self.huber_delta)
        elif self.peak_loss_type == 'mae':
            return F.l1_loss(pred_signal, true_signal)
        else:  # mse
            return F.mse_loss(pred_signal, true_signal)
    
    def compute_peak_loss(
        self, 
        peak_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        true_peaks: torch.Tensor,
        peak_masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute peak prediction losses with proper masking.
        
        Args:
            peak_outputs: (existence_logits, latency, amplitude)
            true_peaks: [B, 2] true peak values (latency, amplitude)
            peak_masks: [B, 2] boolean masks for valid peaks
            
        Returns:
            Dictionary of peak loss components
        """
        exist_logits, pred_latency, pred_amplitude = peak_outputs
        
        # Peak existence loss (binary classification)
        # Convert masks to float for BCE loss
        exist_targets = peak_masks.any(dim=1).float()  # [B]
        exist_loss = self.peak_exist_loss(exist_logits.squeeze(-1), exist_targets)
        
        # Peak value losses with proper masking
        true_latency = true_peaks[:, 0]  # [B]
        true_amplitude = true_peaks[:, 1]  # [B]
        
        latency_mask = peak_masks[:, 0]  # [B]
        amplitude_mask = peak_masks[:, 1]  # [B]
        
        # Compute masked losses
        if latency_mask.sum() > 0:
            if self.peak_loss_type == 'huber':
                latency_loss = F.huber_loss(
                    pred_latency.squeeze(-1)[latency_mask],
                    true_latency[latency_mask],
                    delta=self.huber_delta,
                    reduction='mean'
                )
            elif self.peak_loss_type == 'mae':
                latency_loss = F.l1_loss(
                    pred_latency.squeeze(-1)[latency_mask],
                    true_latency[latency_mask]
                )
            else:  # mse
                latency_loss = F.mse_loss(
                    pred_latency.squeeze(-1)[latency_mask],
                    true_latency[latency_mask]
                )
        else:
            latency_loss = torch.tensor(0.0, device=self.device)
        
        if amplitude_mask.sum() > 0:
            if self.peak_loss_type == 'huber':
                amplitude_loss = F.huber_loss(
                    pred_amplitude.squeeze(-1)[amplitude_mask],
                    true_amplitude[amplitude_mask],
                    delta=self.huber_delta,
                    reduction='mean'
                )
            elif self.peak_loss_type == 'mae':
                amplitude_loss = F.l1_loss(
                    pred_amplitude.squeeze(-1)[amplitude_mask],
                    true_amplitude[amplitude_mask]
                )
            else:  # mse
                amplitude_loss = F.mse_loss(
                    pred_amplitude.squeeze(-1)[amplitude_mask],
                    true_amplitude[amplitude_mask]
                )
        else:
            amplitude_loss = torch.tensor(0.0, device=self.device)
        
        return {
            'exist': exist_loss,
            'latency': latency_loss,
            'amplitude': amplitude_loss
        }
    
    def compute_threshold_loss(
        self, 
        pred_threshold: torch.Tensor, 
        true_threshold: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute threshold regression loss with log-scale option.
        
        Args:
            pred_threshold: [B, 1] or [B, 2] predicted threshold (μ, σ if uncertainty)
            true_threshold: [B] true threshold values
            
        Returns:
            Threshold loss tensor
        """
        if self.use_uncertainty_threshold and pred_threshold.size(-1) == 2:
            # Uncertainty-aware threshold loss
            pred_mu = pred_threshold[:, 0]
            pred_sigma = F.softplus(pred_threshold[:, 1]) + 1e-6  # Ensure positive
            
            # Negative log-likelihood loss
            nll = ((pred_mu - true_threshold) ** 2 / (2 * pred_sigma ** 2) + 
                   torch.log(pred_sigma * np.sqrt(2 * np.pi)))
            return nll.mean()
        
        else:
            # Standard regression loss
            pred_threshold = pred_threshold.squeeze(-1)
            
            if self.use_log_threshold:
                # Log-scale loss for better threshold regression
                return F.mse_loss(
                    torch.log1p(torch.clamp(pred_threshold, min=0)),
                    torch.log1p(torch.clamp(true_threshold, min=0))
                )
            else:
                return F.mse_loss(pred_threshold, true_threshold)
    
    def forward(
        self, 
        outputs: Dict[str, Any], 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and individual components.
        
        Args:
            outputs: Model outputs dictionary
            batch: Batch data dictionary
            
        Returns:
            (total_loss, loss_components)
        """
        loss_components = {}
        
        # Signal reconstruction loss
        signal_loss = self.compute_signal_loss(
            outputs['recon'], 
            batch['signal']
        )
        loss_components['signal_loss'] = signal_loss
        
        # Peak prediction losses
        peak_losses = self.compute_peak_loss(
            outputs['peak'],
            batch['v_peak'],
            batch['v_peak_mask']
        )
        loss_components['peak_exist_loss'] = peak_losses['exist']
        loss_components['peak_latency_loss'] = peak_losses['latency']
        loss_components['peak_amplitude_loss'] = peak_losses['amplitude']
        
        # Classification loss
        classification_loss = self.classification_loss(
            outputs['class'], 
            batch['target']
        )
        loss_components['classification_loss'] = classification_loss
        
        # Threshold loss
        if 'threshold' in outputs and 'threshold' in batch:
            threshold_loss = self.compute_threshold_loss(
                outputs['threshold'],
                batch.get('threshold', torch.zeros_like(batch['target']).float())
            )
            loss_components['threshold_loss'] = threshold_loss
        else:
            threshold_loss = torch.tensor(0.0, device=self.device)
            loss_components['threshold_loss'] = threshold_loss
        
        # Combine losses with curriculum weights
        total_loss = (
            self.loss_weights['signal'] * signal_loss +
            self.loss_weights['peak_exist'] * peak_losses['exist'] +
            self.loss_weights['peak_latency'] * peak_losses['latency'] +
            self.loss_weights['peak_amplitude'] * peak_losses['amplitude'] +
            self.loss_weights['classification'] * classification_loss +
            self.loss_weights['threshold'] * threshold_loss
        )
        
        loss_components['total_loss'] = total_loss
        
        return total_loss, loss_components


def create_class_weights(targets: list, n_classes: int, device: torch.device) -> torch.Tensor:
    """Create class weights for imbalanced data."""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(n_classes),
        y=targets
    )
    return torch.tensor(class_weights, dtype=torch.float32, device=device) 