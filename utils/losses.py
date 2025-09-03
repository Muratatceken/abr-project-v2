"""
Comprehensive loss functions module for ABR transformer training.

This module implements advanced loss functions including:
- FocalLoss for handling severe class imbalance in peak detection
- DistillationLoss for knowledge distillation training
- CombinedLoss for multi-task learning with dynamic weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for handling severe class imbalance.
    
    Supports both binary and multiclass focal loss:
    - Binary mode: Uses BCE with sigmoid for binary targets (0/1 floats)
    - Multiclass mode: Uses CE with softmax for class indices (long tensors)
    
    Focal loss applies a modulating term to the cross entropy loss in order to
    focus learning on hard negative examples. It is designed to address class
    imbalance by down-weighting easy examples and focusing on hard examples.
    
    Args:
        alpha: Weighting factor for rare class(es). For binary: scalar. For multiclass: tensor of class weights (default: 1.0)
        gamma: Focusing parameter for hard examples (default: 2.0)
        reduction: Specifies the reduction to apply ('mean', 'sum', 'none')
        pos_weight: Weight for positive class in binary mode (default: None)
    """
    
    def __init__(
        self, 
        alpha: Union[float, torch.Tensor] = 1.0, 
        gamma: float = 2.0, 
        reduction: str = 'mean',
        pos_weight: Optional[torch.Tensor] = None
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.eps = 1e-8  # For numerical stability
        
        # Register alpha as buffer if it's a tensor
        if isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = alpha
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for focal loss computation.
        
        Args:
            inputs: Predicted logits of shape (N, C) for multiclass or (N, *) for binary
            targets: Ground truth labels of shape (N,) for multiclass (long) or (N, *) for binary (float)
            
        Returns:
            Computed focal loss
        """
        # Determine if this is binary or multiclass based on target dtype
        is_multiclass = targets.dtype == torch.long
        
        if is_multiclass:
            return self._multiclass_focal_loss(inputs, targets)
        else:
            return self._binary_focal_loss(inputs, targets)
    
    def _binary_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary focal loss implementation."""
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        # Compute p_t
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Compute alpha_t
        if isinstance(self.alpha, torch.Tensor):
            # Handle tensor alpha - if shape (2,), select alpha_pos = alpha[1]
            if self.alpha.shape == torch.Size([2]):
                alpha_pos = self.alpha[1]
                alpha_neg = self.alpha[0]
                alpha_t = alpha_pos * targets + alpha_neg * (1 - targets)
            else:
                # Broadcast tensor alpha to targets shape
                alpha_t = self.alpha.expand_as(targets)
        else:
            # Scalar alpha
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        # Apply reduction
        return self._apply_reduction(focal_loss)
    
    def _multiclass_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Multiclass focal loss implementation with softmax."""
        # Apply softmax to get probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Compute cross entropy loss
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        
        # Get the probability of the true class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute alpha_t
        if isinstance(self.alpha, torch.Tensor):
            # Use class-specific alphas
            alpha_t = self.alpha.gather(0, targets)
        else:
            # Use scalar alpha for all classes
            alpha_t = torch.full_like(p_t, self.alpha)
        
        # Compute focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        return self._apply_reduction(focal_loss)
    
    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss for teacher-student training.
    
    Combines student task loss with distillation loss from teacher model.
    Supports both output-level and feature-level distillation.
    
    Args:
        temperature: Temperature for softening distributions (default: 4.0)
        alpha: Balance between student loss and distillation loss (default: 0.7)
        feature_weight: Weight for feature-level distillation (default: 0.1)
    """
    
    def __init__(
        self, 
        temperature: float = 4.0, 
        alpha: float = 0.7,
        feature_weight: float = 0.1
    ):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
    def forward(
        self, 
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
        task_loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for distillation loss computation.
        
        Args:
            student_outputs: Student model predictions
            teacher_outputs: Teacher model predictions  
            targets: Ground truth labels
            student_features: Student intermediate features (optional)
            teacher_features: Teacher intermediate features (optional)
            task_loss_fn: Task-specific loss function
            
        Returns:
            Dictionary containing different loss components
        """
        losses = {}
        
        # Student task loss
        if task_loss_fn is not None:
            student_loss = task_loss_fn(student_outputs, targets)
            losses['student_loss'] = student_loss
        else:
            student_loss = F.binary_cross_entropy_with_logits(student_outputs, targets)
            losses['student_loss'] = student_loss
            
        # Output distillation loss (KL divergence)
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=-1)
        
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        losses['distillation_loss'] = distillation_loss
        
        # Feature distillation loss (if features provided)
        if student_features is not None and teacher_features is not None:
            # Ensure features have same dimensions
            if student_features.shape != teacher_features.shape:
                # Apply adaptive pooling or linear projection if needed
                if len(student_features.shape) > 2:
                    teacher_features = F.adaptive_avg_pool1d(
                        teacher_features.transpose(1, 2), 
                        student_features.shape[-1]
                    ).transpose(1, 2)
                    
            feature_loss = self.mse_loss(student_features, teacher_features.detach())
            losses['feature_loss'] = feature_loss
        else:
            losses['feature_loss'] = torch.tensor(0.0, device=student_outputs.device)
            
        # Combined loss
        combined_loss = (
            (1 - self.alpha) * student_loss + 
            self.alpha * distillation_loss + 
            self.feature_weight * losses['feature_loss']
        )
        losses['combined_loss'] = combined_loss
        
        return losses


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning with dynamic weighting.
    
    Supports automatic loss balancing using gradient magnitudes and
    uncertainty-based weighting for different tasks.
    
    Args:
        task_weights: Initial weights for different tasks
        adaptive_weighting: Whether to use adaptive loss weighting
        uncertainty_weighting: Whether to use uncertainty-based weighting
    """
    
    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        adaptive_weighting: bool = True,
        uncertainty_weighting: bool = False
    ):
        super(CombinedLoss, self).__init__()
        self.task_weights = task_weights or {}
        self.adaptive_weighting = adaptive_weighting
        self.uncertainty_weighting = uncertainty_weighting
        
        # For uncertainty weighting
        if uncertainty_weighting:
            self.log_vars = nn.ParameterDict()
            
        # For gradient-based adaptive weighting
        self.loss_history = {}
        self.gradient_history = {}
        
    def add_task(self, task_name: str, initial_weight: float = 1.0):
        """Add a new task to the multi-task loss."""
        self.task_weights[task_name] = initial_weight
        if self.uncertainty_weighting:
            self.log_vars[task_name] = nn.Parameter(torch.zeros(1))
            
    def forward(
        self, 
        task_losses: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for combined loss computation.
        
        Args:
            task_losses: Dictionary of task-specific losses
            model: Model for gradient-based weighting (optional)
            
        Returns:
            Dictionary containing weighted losses and total loss
        """
        weighted_losses = {}
        total_loss = 0
        
        # Update loss history
        for task_name, loss in task_losses.items():
            if task_name not in self.loss_history:
                self.loss_history[task_name] = []
            self.loss_history[task_name].append(loss.item())
            
        # Compute weights
        if self.uncertainty_weighting:
            weights = self._compute_uncertainty_weights()
        elif self.adaptive_weighting:
            weights = self._compute_adaptive_weights(task_losses, model)
        else:
            weights = self.task_weights
            
        # Apply weights and compute total loss
        for task_name, loss in task_losses.items():
            weight = weights.get(task_name, 1.0)
            
            if self.uncertainty_weighting and task_name in self.log_vars:
                # Uncertainty-based weighting
                precision = torch.exp(-self.log_vars[task_name])
                weighted_loss = precision * loss + self.log_vars[task_name]
            else:
                weighted_loss = weight * loss
                
            weighted_losses[f'{task_name}_weighted'] = weighted_loss
            total_loss += weighted_loss
            
        weighted_losses['total_loss'] = total_loss
        weighted_losses['weights'] = weights
        
        return weighted_losses
        
    def _compute_uncertainty_weights(self) -> Dict[str, float]:
        """Compute uncertainty-based weights."""
        weights = {}
        for task_name in self.task_weights.keys():
            if task_name in self.log_vars:
                weights[task_name] = torch.exp(-self.log_vars[task_name]).item()
            else:
                weights[task_name] = self.task_weights.get(task_name, 1.0)
        return weights
        
    def _compute_adaptive_weights(
        self, 
        task_losses: Dict[str, torch.Tensor],
        model: Optional[nn.Module]
    ) -> Dict[str, float]:
        """Compute gradient-based adaptive weights."""
        if model is None or len(self.loss_history) < 2:
            return self.task_weights
            
        weights = {}
        
        # Compute relative loss rates
        for task_name, loss in task_losses.items():
            if len(self.loss_history[task_name]) >= 2:
                recent_losses = self.loss_history[task_name][-10:]  # Last 10 losses
                if len(recent_losses) > 1:
                    loss_rate = np.mean(np.diff(recent_losses))
                    # Increase weight for tasks with slower improvement
                    weight = self.task_weights.get(task_name, 1.0)
                    if loss_rate > 0:  # Loss is increasing
                        weight *= 1.1
                    elif loss_rate < -0.01:  # Loss is decreasing fast
                        weight *= 0.95
                    weights[task_name] = max(0.1, min(10.0, weight))  # Clamp weights
                else:
                    weights[task_name] = self.task_weights.get(task_name, 1.0)
            else:
                weights[task_name] = self.task_weights.get(task_name, 1.0)
                
        return weights


def focal_loss_with_logits(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Functional interface for focal loss computation.
    
    Args:
        inputs: Predicted logits
        targets: Ground truth labels
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        reduction: Reduction method
        
    Returns:
        Computed focal loss
    """
    focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
    return focal_loss(inputs, targets)


def create_loss_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create loss function from configuration.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Configured loss function
    """
    loss_type = config.get('type', 'bce')
    
    if loss_type == 'focal':
        return FocalLoss(
            alpha=config.get('alpha', 0.25),
            gamma=config.get('gamma', 2.0),
            reduction=config.get('reduction', 'mean')
        )
    elif loss_type == 'distillation':
        return DistillationLoss(
            temperature=config.get('temperature', 4.0),
            alpha=config.get('alpha', 0.7),
            feature_weight=config.get('feature_weight', 0.1)
        )
    elif loss_type == 'combined':
        return CombinedLoss(
            task_weights=config.get('task_weights', {}),
            adaptive_weighting=config.get('adaptive_weighting', True),
            uncertainty_weighting=config.get('uncertainty_weighting', False)
        )
    else:
        return nn.BCEWithLogitsLoss()


def compute_loss_weights(targets: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        targets: Ground truth labels
        device: Optional device to place the result tensor on
        
    Returns:
        Computed class weights on the specified device
    """
    pos_count = targets.sum()
    neg_count = targets.numel() - pos_count
    
    if pos_count == 0:
        pos_weight = torch.tensor(1.0)
    else:
        pos_weight = neg_count / pos_count
    
    # Move to specified device if provided
    if device is not None:
        pos_weight = pos_weight.to(device)
    
    return pos_weight


def test_focal_loss_tensor_alpha():
    """Test FocalLoss with tensor alpha and device handling."""
    # Test tensor alpha with shape (2,)
    alpha_tensor = torch.tensor([0.25, 0.75])  # [negative_class, positive_class]
    focal_loss = FocalLoss(alpha=alpha_tensor, gamma=2.0)
    
    # Test binary classification
    batch_size = 4
    inputs = torch.randn(batch_size, 1)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    
    # Compute loss
    loss = focal_loss(inputs, targets)
    
    # Verify loss is computed without error
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert loss.item() >= 0
    
    # Test that alpha is properly registered as buffer
    assert hasattr(focal_loss, 'alpha')
    assert torch.equal(focal_loss.alpha, alpha_tensor)
    
    print("Tensor alpha test passed!")


def test_compute_loss_weights_device():
    """Test compute_loss_weights with device argument."""
    # Create test targets
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])  # 3 positive, 2 negative
    
    # Test without device
    pos_weight_cpu = compute_loss_weights(targets)
    assert isinstance(pos_weight_cpu, torch.Tensor)
    expected_weight = 2.0 / 3.0  # neg_count / pos_count
    assert abs(pos_weight_cpu.item() - expected_weight) < 1e-6
    
    # Test with CPU device
    pos_weight_cpu_explicit = compute_loss_weights(targets, device='cpu')
    assert pos_weight_cpu_explicit.device.type == 'cpu'
    assert torch.equal(pos_weight_cpu, pos_weight_cpu_explicit)
    
    # Test with CUDA device if available
    if torch.cuda.is_available():
        pos_weight_cuda = compute_loss_weights(targets, device='cuda')
        assert pos_weight_cuda.device.type == 'cuda'
        assert abs(pos_weight_cuda.item() - expected_weight) < 1e-6
    
    # Test edge case with no positive samples
    no_pos_targets = torch.zeros(5)
    pos_weight_no_pos = compute_loss_weights(no_pos_targets, device='cpu')
    assert pos_weight_no_pos.item() == 1.0
    
    print("Device handling test passed!")


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_focal_loss_tensor_alpha()
    test_compute_loss_weights_device()
