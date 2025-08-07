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
    Simplified loss for generator-focused training.
    - Primary: Signal reconstruction (MSE/Huber/MAE)
    - Optional: classification (if logits provided)
    """

    def __init__(
        self,
        n_classes: int = 5,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        signal_loss_type: str = 'mse',
        huber_delta: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.use_focal_loss = use_focal_loss
        self.signal_loss_type = signal_loss_type
        self.huber_delta = huber_delta
        self.device = device or torch.device('cpu')

        # Optional classification loss
        if use_focal_loss:
            self.classification_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.classification_loss = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device) if class_weights is not None else None
            )
    
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
        
        if self.signal_loss_type == 'huber':
            return F.huber_loss(pred_signal, true_signal, delta=self.huber_delta)
        elif self.signal_loss_type == 'mae':
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
            peak_outputs: Tuple of (existence, latency, amplitude[, uncertainties])
            true_peaks: True peak values [batch, 2] for [latency, amplitude]
            peak_masks: Valid peak mask [batch, 2] - 1 where peak exists, 0 otherwise
            
        Returns:
            Dictionary with individual peak losses
        """
        # Handle both with and without uncertainty outputs
        if len(peak_outputs) == 5:  # With uncertainty
            pred_exist, pred_latency, pred_amplitude, latency_std, amplitude_std = peak_outputs
            use_uncertainty = True
        else:  # Without uncertainty
            pred_exist, pred_latency, pred_amplitude = peak_outputs
            use_uncertainty = False
        
        # Extract true values - v_peak has shape [batch, 2] for [latency, amplitude]
        true_latency = true_peaks[:, 0]    # [batch] 
        true_amplitude = true_peaks[:, 1]  # [batch]
        
        # Create existence target from peak masks - peak exists if both latency and amplitude are valid
        true_existence = (peak_masks[:, 0] & peak_masks[:, 1]).float()  # [batch]
        
        # Existence loss (BCE) - always computed
        existence_loss = F.binary_cross_entropy_with_logits(
            pred_exist, true_existence
        )
        
        # Only compute regression losses for samples with existing peaks
        valid_mask = true_existence.bool()
        
        if valid_mask.sum() > 0:
            valid_indices = valid_mask
            
            if use_uncertainty:
                # Uncertainty-aware losses (negative log-likelihood)
                valid_latency_pred = pred_latency[valid_indices]
                valid_latency_std = latency_std[valid_indices]
                valid_latency_target = true_latency[valid_indices]
                
                latency_nll = ((valid_latency_pred - valid_latency_target) ** 2 / 
                              (2 * valid_latency_std ** 2) + 
                              torch.log(valid_latency_std * np.sqrt(2 * np.pi)))
                latency_loss = latency_nll.mean()
                
                valid_amplitude_pred = pred_amplitude[valid_indices]
                valid_amplitude_std = amplitude_std[valid_indices]
                valid_amplitude_target = true_amplitude[valid_indices]
                
                amplitude_nll = ((valid_amplitude_pred - valid_amplitude_target) ** 2 / 
                                (2 * valid_amplitude_std ** 2) + 
                                torch.log(valid_amplitude_std * np.sqrt(2 * np.pi)))
                amplitude_loss = amplitude_nll.mean()
            else:
                # Standard MSE losses for valid samples only
                valid_latency_pred = pred_latency[valid_indices]
                valid_latency_target = true_latency[valid_indices]
                
                valid_amplitude_pred = pred_amplitude[valid_indices]
                valid_amplitude_target = true_amplitude[valid_indices]
                
                if self.peak_loss_type == 'huber':
                    latency_loss = F.huber_loss(valid_latency_pred, valid_latency_target, delta=self.huber_delta)
                    amplitude_loss = F.huber_loss(valid_amplitude_pred, valid_amplitude_target, delta=self.huber_delta)
                elif self.peak_loss_type == 'mae':
                    latency_loss = F.l1_loss(valid_latency_pred, valid_latency_target)
                    amplitude_loss = F.l1_loss(valid_amplitude_pred, valid_amplitude_target)
                else:  # mse
                    latency_loss = F.mse_loss(valid_latency_pred, valid_latency_target)
                    amplitude_loss = F.mse_loss(valid_amplitude_pred, valid_amplitude_target)
        else:
            # No valid peaks - set losses to zero but maintain gradients
            device = pred_exist.device
            latency_loss = torch.tensor(0.0, device=device, requires_grad=True)
            amplitude_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return {
            'exist': existence_loss,
            'latency': latency_loss,
            'amplitude': amplitude_loss
        }
    
    def compute_threshold_loss(
        self,
        threshold_output: torch.Tensor,
        true_threshold: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute threshold regression loss with robust handling.
        
        Args:
            threshold_output: Predicted threshold [batch, 1] or [batch, 2] if uncertainty
            true_threshold: True threshold values [batch]
            
        Returns:
            Threshold loss tensor
        """
        # Ensure true_threshold has the correct shape and device
        if true_threshold.dim() == 1:
            true_threshold = true_threshold.float()
        else:
            true_threshold = true_threshold.squeeze().float()
            
        # Ensure both tensors are on the same device
        true_threshold = true_threshold.to(threshold_output.device)
            
        # Check if this is uncertainty prediction (2 outputs: mean, std)
        if threshold_output.size(-1) == 2:
            if self.use_uncertainty_threshold:
                # Uncertainty-aware loss
                pred_mean = threshold_output[:, 0]
                pred_std = torch.clamp(threshold_output[:, 1], min=1e-6)  # Prevent zero std
                
                # Negative log-likelihood with robust handling
                diff_sq = (pred_mean - true_threshold) ** 2
                nll = (diff_sq / (2 * pred_std ** 2) + 
                       torch.log(pred_std * np.sqrt(2 * np.pi)))
                
                # Add regularization to prevent very small std values
                std_reg = torch.mean(torch.clamp(1.0 / pred_std - 1.0, min=0.0))
                
                return nll.mean() + 0.01 * std_reg
            else:
                # Use only the mean prediction, ignore std
                pred_threshold = threshold_output[:, 0]
                return F.huber_loss(pred_threshold, true_threshold, delta=5.0)
        else:
            # Standard regression loss with improved robustness
            if threshold_output.dim() > 1:
                pred_threshold = threshold_output.squeeze(-1)
            else:
                pred_threshold = threshold_output
            
            # Use Huber loss for better outlier resistance
            if self.use_log_threshold:
                # Log-scale Huber loss
                log_pred = torch.log1p(torch.clamp(pred_threshold, min=0))
                log_target = torch.log1p(torch.clamp(true_threshold, min=0))
                return F.huber_loss(log_pred, log_target, delta=1.0)
            else:
                # Standard Huber loss
                return F.huber_loss(pred_threshold, true_threshold, delta=5.0)  # Larger delta for thresholds
    
    def compute_static_param_loss(
        self,
        generated_params: torch.Tensor,
        target_params: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute static parameter generation loss for joint generation.
        
        Args:
            generated_params: Generated parameters [batch, static_dim] or [batch, static_dim, 2]
            target_params: Target parameters [batch, static_dim]
            
        Returns:
            Dictionary of static parameter losses
        """
        if not self.enable_static_param_loss:
            return {'static_params': torch.tensor(0.0, device=self.device)}
        
        param_names = ['age', 'intensity', 'stimulus_rate', 'fmp']
        losses = {}
        
        if generated_params.dim() == 3 and generated_params.size(-1) == 2:
            # Uncertainty-aware loss
            pred_means = generated_params[:, :, 0]  # [batch, static_dim]
            pred_stds = generated_params[:, :, 1]   # [batch, static_dim]
            
            # Compute negative log-likelihood for each parameter
            param_losses = []
            for i in range(min(generated_params.size(1), len(param_names))):
                pred_mean = pred_means[:, i]
                pred_std = pred_stds[:, i]
                target = target_params[:, i]
                
                nll = ((pred_mean - target) ** 2 / (2 * pred_std ** 2) + 
                       torch.log(pred_std * np.sqrt(2 * np.pi)))
                param_losses.append(nll.mean())
                losses[f'static_{param_names[i]}'] = nll.mean()
            
            # Total static parameter loss
            losses['static_params'] = torch.stack(param_losses).mean()
        else:
            # Standard MSE loss for each parameter
            param_losses = []
            for i in range(min(generated_params.size(1), len(param_names))):
                pred = generated_params[:, i]
                target = target_params[:, i] 
                
                # Use Huber loss for robustness
                loss = F.huber_loss(pred, target, delta=1.0)
                param_losses.append(loss)
                losses[f'static_{param_names[i]}'] = loss
            
            # Total static parameter loss
            losses['static_params'] = torch.stack(param_losses).mean()
        
        return losses
    
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
        
        # Get device from the first available tensor
        device = next(iter(outputs.values())).device if outputs else next(iter(batch.values())).device
        
        # Signal reconstruction loss - handle different output keys
        signal_key = 'recon' if 'recon' in outputs else 'signal'
        if signal_key in outputs:
            signal_loss = self.compute_signal_loss(
                outputs[signal_key], 
                batch['signal']
            )
            loss_components['signal_loss'] = signal_loss
        else:
            signal_loss = torch.tensor(0.0, device=device)
            loss_components['signal_loss'] = signal_loss
        
        # No peak losses in simplified setting
        zero_loss = torch.tensor(0.0, device=device)
        loss_components['peak_exist_loss'] = zero_loss
        loss_components['peak_latency_loss'] = zero_loss
        loss_components['peak_amplitude_loss'] = zero_loss
        peak_losses = {'exist': zero_loss, 'latency': zero_loss, 'amplitude': zero_loss}
        
        # Classification loss - handle different output keys
        class_key = 'class' if 'class' in outputs else 'classification_logits'
        if class_key in outputs:
            if self.use_focal_loss:
                classification_loss = self.classification_loss(
                    outputs[class_key], 
                    batch['target']
                )
            else:
                # Use class weights if available
                classification_loss = F.cross_entropy(
                    outputs[class_key], 
                    batch['target'],
                    weight=self.classification_loss.weight if hasattr(self.classification_loss, 'weight') else None
                )
            loss_components['classification_loss'] = classification_loss
        else:
            classification_loss = torch.tensor(0.0, device=device)
            loss_components['classification_loss'] = classification_loss
        
        # No threshold loss in simplified setting
        threshold_loss = torch.tensor(0.0, device=device)
        loss_components['threshold_loss'] = threshold_loss
        
        # No static param loss in simplified setting
        static_param_total_loss = torch.tensor(0.0, device=self.device)
        loss_components['static_loss_static_params'] = static_param_total_loss
        
        # Adaptive loss weighting based on relative magnitudes
        # This helps prevent one loss from dominating during training
        with torch.no_grad():
            # Normalize loss weights based on current loss magnitudes
            losses_for_weighting = {
                'signal': signal_loss.detach(),
                'peak_exist': peak_losses['exist'].detach(),
                'peak_latency': peak_losses['latency'].detach(),
                'peak_amplitude': peak_losses['amplitude'].detach(), 
                'classification': classification_loss.detach(),
                'threshold': threshold_loss.detach(),
                'static_params': static_param_total_loss.detach()
            }
            
            # Compute adaptive weights (optional - can be disabled)
            adaptive_weights = {}
            for key, loss_val in losses_for_weighting.items():
                if loss_val.item() > 0:
                    adaptive_weights[key] = 1.0 / (1.0 + loss_val.item())
                else:
                    adaptive_weights[key] = 1.0
        
        # Compute total loss: signal + optional classification
        total_loss = signal_loss + classification_loss
        
        loss_components['total_loss'] = total_loss
        loss_components['adaptive_weights'] = adaptive_weights
        
        return total_loss, loss_components


def create_class_weights(targets: list, n_classes: int, device: torch.device) -> torch.Tensor:
    """Create class weights for imbalanced data."""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(n_classes),
        y=targets
    )
    return torch.tensor(class_weights, dtype=torch.float32, device=device) 