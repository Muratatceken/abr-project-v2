#!/usr/bin/env python3
"""
Enhanced Class Imbalance Handling for ABR Classification

Implements advanced techniques to handle severe class imbalance:
- Dynamic class weights
- Enhanced focal loss with class-specific parameters
- Oversampling strategies
- Class-aware data augmentation
- Balanced batch sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import logging

logger = logging.getLogger(__name__)


class EnhancedFocalLoss(nn.Module):
    """
    Enhanced Focal Loss with class-specific alpha and gamma parameters.
    
    Addresses severe class imbalance by:
    - Different alpha values per class
    - Adaptive gamma based on class frequency
    - Temperature scaling for confidence calibration
    """
    
    def __init__(
        self,
        n_classes: int = 5,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        class_specific_gamma: bool = True,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.gamma = gamma
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.class_specific_gamma = class_specific_gamma
        
        # Set default alpha values if not provided
        if alpha is None:
            # Default: higher alpha for minority classes
            alpha = torch.ones(n_classes)
            alpha[0] = 0.25  # Majority class (Normal)
            alpha[1:] = [4.0, 1.5, 3.0, 2.5]  # Minority classes weighted by rarity
        
        self.register_buffer('alpha', alpha)
        
        # Class-specific gamma values (higher for minority classes)
        if class_specific_gamma:
            gamma_values = torch.tensor([
                1.5,  # Class 0 (Normal) - lower gamma
                4.0,  # Class 1 (Neuropathy) - highest gamma (rarest)
                2.5,  # Class 2 (SNIK)
                3.5,  # Class 3 (Total)
                3.0   # Class 4 (ITIK)
            ])
            self.register_buffer('gamma_values', gamma_values)
        else:
            self.register_buffer('gamma_values', torch.full((n_classes,), gamma))
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Enhanced focal loss computation.
        
        Args:
            logits: Prediction logits [batch_size, n_classes]
            targets: True class labels [batch_size]
            
        Returns:
            Focal loss value
        """
        # Temperature scaling
        logits = logits / self.temperature
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_smooth = self._smooth_labels(targets, logits.size(1))
            ce_loss = F.cross_entropy(logits, targets_smooth, reduction='none')
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Compute pt (confidence for true class)
        pt = torch.exp(-ce_loss)
        
        # Get class-specific alpha and gamma
        alpha_t = self.alpha[targets]
        gamma_t = self.gamma_values[targets]
        
        # Compute focal loss with class-specific parameters
        focal_loss = alpha_t * (1 - pt) ** gamma_t * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _smooth_labels(self, targets: torch.Tensor, n_classes: int) -> torch.Tensor:
        """Apply label smoothing."""
        device = targets.device
        targets_smooth = torch.zeros(targets.size(0), n_classes, device=device)
        targets_smooth.fill_(self.label_smoothing / (n_classes - 1))
        targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        return targets_smooth


class BalancedBatchSampler:
    """
    Balanced batch sampler ensuring each batch has representation from minority classes.
    """
    
    def __init__(
        self,
        targets: List[int],
        batch_size: int,
        min_samples_per_class: int = 1,
        oversample_factor: float = 2.0
    ):
        self.targets = np.array(targets)
        self.batch_size = batch_size
        self.min_samples_per_class = min_samples_per_class
        self.oversample_factor = oversample_factor
        
        # Group indices by class
        self.class_indices = {}
        for class_id in np.unique(self.targets):
            self.class_indices[class_id] = np.where(self.targets == class_id)[0]
        
        # Compute class weights for sampling
        class_counts = Counter(targets)
        total_samples = len(targets)
        self.class_weights = {
            class_id: total_samples / (len(np.unique(self.targets)) * count)
            for class_id, count in class_counts.items()
        }
        
        logger.info(f"Balanced sampler class weights: {self.class_weights}")
    
    def __iter__(self):
        """Generate balanced batches."""
        n_classes = len(self.class_indices)
        
        # Calculate samples per class per batch
        base_samples_per_class = max(1, self.batch_size // n_classes)
        
        while True:
            batch_indices = []
            
            # Sample from each class
            for class_id, indices in self.class_indices.items():
                # Determine number of samples for this class
                if class_id == 0:  # Majority class
                    n_samples = max(1, self.batch_size - (n_classes - 1) * base_samples_per_class)
                else:  # Minority classes
                    n_samples = min(base_samples_per_class, len(indices))
                
                # Sample with replacement for minority classes
                if len(indices) < n_samples:
                    sampled = np.random.choice(indices, size=n_samples, replace=True)
                else:
                    sampled = np.random.choice(indices, size=n_samples, replace=False)
                
                batch_indices.extend(sampled)
            
            # Shuffle batch
            np.random.shuffle(batch_indices)
            
            # Trim to exact batch size
            batch_indices = batch_indices[:self.batch_size]
            
            yield batch_indices
    
    def __len__(self):
        # Estimate number of batches per epoch
        return len(self.targets) // self.batch_size


class DynamicClassWeights:
    """
    Dynamic class weights that adapt during training based on performance.
    """
    
    def __init__(
        self,
        n_classes: int = 5,
        initial_weights: Optional[torch.Tensor] = None,
        adaptation_rate: float = 0.1,
        min_weight: float = 0.1,
        max_weight: float = 10.0
    ):
        self.n_classes = n_classes
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        if initial_weights is None:
            # Start with balanced weights
            initial_weights = torch.ones(n_classes)
        
        self.weights = initial_weights.clone()
        self.class_errors = torch.zeros(n_classes)
        self.update_count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update class weights based on current performance.
        
        Args:
            predictions: Model predictions [batch_size, n_classes]
            targets: True labels [batch_size]
        """
        with torch.no_grad():
            # Compute per-class error rates
            pred_classes = predictions.argmax(dim=1)
            
            for class_id in range(self.n_classes):
                class_mask = targets == class_id
                if class_mask.sum() > 0:
                    class_acc = (pred_classes[class_mask] == targets[class_mask]).float().mean()
                    class_error = 1.0 - class_acc
                    
                    # Update exponential moving average of error
                    if self.update_count == 0:
                        self.class_errors[class_id] = class_error
                    else:
                        self.class_errors[class_id] = (
                            (1 - self.adaptation_rate) * self.class_errors[class_id] +
                            self.adaptation_rate * class_error
                        )
            
            # Update weights based on error rates
            # Higher error -> higher weight
            max_error = self.class_errors.max()
            if max_error > 0:
                self.weights = 1.0 + (self.class_errors / max_error) * 4.0
                
                # Clamp weights
                self.weights = torch.clamp(self.weights, self.min_weight, self.max_weight)
            
            self.update_count += 1
    
    def get_weights(self, device: torch.device) -> torch.Tensor:
        """Get current class weights."""
        return self.weights.to(device)


def compute_enhanced_class_weights(
    targets: List[int],
    n_classes: int = 5,
    method: str = "balanced_sqrt",
    min_weight: float = 0.1,
    max_weight: float = 10.0
) -> torch.Tensor:
    """
    Compute enhanced class weights using various strategies.
    
    Args:
        targets: List of target class labels
        n_classes: Number of classes
        method: Weighting method ('balanced', 'balanced_sqrt', 'inverse_freq')
        min_weight: Minimum weight value
        max_weight: Maximum weight value
        
    Returns:
        Class weights tensor
    """
    class_counts = Counter(targets)
    total_samples = len(targets)
    
    weights = torch.zeros(n_classes)
    
    for class_id in range(n_classes):
        count = class_counts.get(class_id, 1)  # Avoid division by zero
        
        if method == "balanced":
            weight = total_samples / (n_classes * count)
        elif method == "balanced_sqrt":
            weight = np.sqrt(total_samples / (n_classes * count))
        elif method == "inverse_freq":
            weight = 1.0 / count
        else:
            weight = 1.0
        
        weights[class_id] = weight
    
    # Normalize weights
    weights = weights / weights.sum() * n_classes
    
    # Clamp weights
    weights = torch.clamp(weights, min_weight, max_weight)
    
    logger.info(f"Enhanced class weights ({method}): {weights}")
    logger.info(f"Class distribution: {dict(class_counts)}")
    
    return weights


def create_balanced_loss_function(
    targets: List[int],
    n_classes: int = 5,
    use_focal: bool = True,
    use_dynamic: bool = False,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """
    Create a balanced loss function for severe class imbalance.
    
    Args:
        targets: Training targets for computing class weights
        n_classes: Number of classes
        use_focal: Whether to use enhanced focal loss
        use_dynamic: Whether to use dynamic class weights
        device: Device for computation
        
    Returns:
        Configured loss function
    """
    if use_focal:
        # Compute enhanced class weights
        class_weights = compute_enhanced_class_weights(
            targets, n_classes, method="balanced_sqrt"
        )
        
        loss_fn = EnhancedFocalLoss(
            n_classes=n_classes,
            alpha=class_weights,
            gamma=2.0,
            class_specific_gamma=True,
            temperature=1.0,
            label_smoothing=0.1  # Small amount of label smoothing
        ).to(device)
        
        logger.info("Created Enhanced Focal Loss with class-specific parameters")
        
    else:
        # Standard cross-entropy with class weights
        class_weights = compute_enhanced_class_weights(
            targets, n_classes, method="balanced"
        )
        
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        logger.info("Created Weighted Cross-Entropy Loss")
    
    return loss_fn


# Example usage function
def setup_class_imbalance_handling(
    train_dataset,
    config,
    device: torch.device
) -> Dict:
    """
    Setup comprehensive class imbalance handling.
    
    Returns:
        Dictionary with loss function, sampler, and other components
    """
    # Extract targets from dataset
    if hasattr(train_dataset, 'targets'):
        targets = train_dataset.targets
    else:
        # Extract from dataset manually
        targets = []
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            if isinstance(sample, dict):
                targets.append(sample['target'].item())
            else:
                targets.append(sample[1].item())  # Assuming (x, y) format
    
    # Create balanced loss function
    loss_fn = create_balanced_loss_function(
        targets=targets,
        n_classes=config.data.n_classes,
        use_focal=config.loss.focal_loss.get('use_focal', True),
        use_dynamic=False,  # Can be enabled later
        device=device
    )
    
    # Create balanced batch sampler
    batch_sampler = BalancedBatchSampler(
        targets=targets,
        batch_size=config.data.dataloader.batch_size,
        min_samples_per_class=1,
        oversample_factor=2.0
    )
    
    # Create dynamic weights (optional)
    dynamic_weights = DynamicClassWeights(
        n_classes=config.data.n_classes,
        adaptation_rate=0.1
    )
    
    return {
        'loss_function': loss_fn,
        'batch_sampler': batch_sampler,
        'dynamic_weights': dynamic_weights,
        'class_distribution': Counter(targets)
    }