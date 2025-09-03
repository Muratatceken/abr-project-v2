"""
Knowledge distillation framework for ABR transformer models.

This module implements various knowledge distillation strategies including:
- Teacher-student training with output distillation
- Feature-level distillation between intermediate layers
- Multi-task distillation for signal generation and classification
- Self-distillation and temporal ensembling approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseDistillation(ABC):
    """
    Abstract base class for knowledge distillation methods.
    """
    
    @abstractmethod
    def compute_distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss."""
        pass
    
    @abstractmethod
    def prepare_teacher_targets(
        self,
        teacher_outputs: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Prepare soft targets from teacher outputs."""
        pass


class KnowledgeDistillation(BaseDistillation, nn.Module):
    """
    Standard knowledge distillation with temperature-scaled softmax.
    
    Combines task loss from ground truth labels with distillation loss
    from teacher model soft targets.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        task_loss_fn: Optional[nn.Module] = None,
        feature_matching: bool = False,
        feature_weight: float = 0.1,
        max_feature_dim: int = 1024
    ):
        """
        Initialize knowledge distillation.
        
        Args:
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss vs task loss
            task_loss_fn: Task-specific loss function
            feature_matching: Whether to include feature matching
            feature_weight: Weight for feature matching loss
            max_feature_dim: Maximum feature dimension for projection layers
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.task_loss_fn = task_loss_fn or nn.CrossEntropyLoss()
        self.feature_matching = feature_matching
        self.feature_weight = feature_weight
        
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
        # Initialize projection layers for feature alignment
        # We'll create them dynamically when needed but register them properly
        self.projection_layers = nn.ModuleDict()
        self.max_feature_dim = max_feature_dim
    
    def _get_projection_layer(self, input_dim: int, output_dim: int) -> nn.Module:
        """Get or create a projection layer for feature alignment."""
        layer_key = f"proj_{input_dim}_{output_dim}"
        
        if layer_key not in self.projection_layers:
            # Create and register new projection layer
            self.projection_layers[layer_key] = nn.Linear(input_dim, output_dim)
            
        return self.projection_layers[layer_key]
        
    def prepare_teacher_targets(
        self,
        teacher_outputs: torch.Tensor,
        temperature: float = None
    ) -> torch.Tensor:
        """Prepare soft targets from teacher outputs."""
        temp = temperature or self.temperature
        return F.softmax(teacher_outputs / temp, dim=-1)
        
    def compute_distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_outputs: Student model predictions
            teacher_outputs: Teacher model predictions
            targets: Ground truth labels
            student_features: Student intermediate features (optional)
            teacher_features: Teacher intermediate features (optional)
            
        Returns:
            Dictionary containing loss components
        """
        losses = {}
        
        # Task loss (student predictions vs ground truth)
        if targets is not None:
            if hasattr(self.task_loss_fn, '__call__'):
                task_loss = self.task_loss_fn(student_outputs, targets)
            else:
                task_loss = F.cross_entropy(student_outputs, targets)
            losses['task_loss'] = task_loss
        else:
            losses['task_loss'] = torch.tensor(0.0, device=student_outputs.device)
            
        # Distillation loss (KL divergence between student and teacher)
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=-1)
        teacher_soft = self.prepare_teacher_targets(teacher_outputs, self.temperature)
        
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        losses['distillation_loss'] = distillation_loss
        
        # Feature matching loss (if features provided)
        if self.feature_matching and student_features is not None and teacher_features is not None:
            feature_loss = self._compute_feature_matching_loss(student_features, teacher_features)
            losses['feature_loss'] = feature_loss
        else:
            losses['feature_loss'] = torch.tensor(0.0, device=student_outputs.device)
            
        # Combined loss
        combined_loss = (
            (1 - self.alpha) * losses['task_loss'] +
            self.alpha * losses['distillation_loss'] +
            self.feature_weight * losses['feature_loss']
        )
        losses['combined_loss'] = combined_loss
        
        return losses
        
    def _compute_feature_matching_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature matching loss between student and teacher."""
        # Handle different feature dimensions
        if student_features.shape != teacher_features.shape:
            # Apply adaptive pooling or projection
            if len(student_features.shape) == 3:  # [batch, seq, dim]
                if student_features.shape[-1] != teacher_features.shape[-1]:
                    # Use registered projection layer
                    projection = self._get_projection_layer(
                        student_features.shape[-1], 
                        teacher_features.shape[-1]
                    )
                    student_features = projection(student_features)
                    
                if student_features.shape[1] != teacher_features.shape[1]:
                    # Adaptive pooling for sequence length
                    student_features = F.adaptive_avg_pool1d(
                        student_features.transpose(1, 2),
                        teacher_features.shape[1]
                    ).transpose(1, 2)
                    
        return self.mse_loss(student_features, teacher_features.detach())


class MultiTaskDistillation(BaseDistillation):
    """
    Knowledge distillation for multi-task learning.
    
    Handles distillation for multiple tasks (e.g., signal generation
    and peak classification) with task-specific weights.
    """
    
    def __init__(
        self,
        task_configs: Dict[str, Dict[str, Any]],
        task_types: Optional[Dict[str, str]] = None,
        global_temperature: float = 4.0,
        cross_task_distillation: bool = False
    ):
        """
        Initialize multi-task distillation.
        
        Args:
            task_configs: Configuration for each task
            task_types: Type of each task ('classification' or 'regression')
            global_temperature: Default temperature for all tasks
            cross_task_distillation: Whether to enable cross-task knowledge transfer
        """
        self.task_configs = task_configs
        self.task_types = task_types or {}
        self.global_temperature = global_temperature
        self.cross_task_distillation = cross_task_distillation
        
        # Validate configuration
        self._validate_configs()
        
        # Create individual distillation instances only for classification tasks
        self.task_distillers = {}
        for task_name, config in task_configs.items():
            task_type = self.task_types.get(task_name, 'classification')
            
            if task_type == 'regression':
                logging.warning(
                    f"Skipping distillation for task '{task_name}' "
                    f"(regression tasks do not support KL divergence distillation)"
                )
                continue
            elif task_type == 'classification':
                self.task_distillers[task_name] = KnowledgeDistillation(
                    temperature=config.get('temperature', global_temperature),
                    alpha=config.get('alpha', 0.7),
                    task_loss_fn=config.get('loss_fn'),
                    feature_matching=config.get('feature_matching', False),
                    feature_weight=config.get('feature_weight', 0.1)
                )
            else:
                raise ValueError(
                    f"Invalid task type '{task_type}' for task '{task_name}'. "
                    f"Must be 'classification' or 'regression'."
                )
    
    def _validate_configs(self):
        """Validate distillation configs for regression/classification compatibility."""
        for task_name, config in self.task_configs.items():
            task_type = self.task_types.get(task_name, 'classification')
            loss_fn = config.get('loss_fn', 'ce')
            
            # Check for incompatible loss functions with regression
            if task_type == 'regression':
                incompatible_losses = ['ce', 'cross_entropy', 'focal', 'bce']
                if loss_fn in incompatible_losses:
                    raise ValueError(
                        f"Task '{task_name}' is marked as regression but uses "
                        f"classification loss function '{loss_fn}'. "
                        f"Regression tasks should use 'mse', 'l1', or similar losses."
                    )
            
            # Check for KL divergence on regression (shouldn't happen due to filtering above)
            if task_type == 'regression' and config.get('temperature') is not None:
                logging.warning(
                    f"Task '{task_name}' is regression but has temperature setting. "
                    f"Temperature-based distillation only applies to classification."
                )
            
    def prepare_teacher_targets(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        temperature: float = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare soft targets for all tasks."""
        soft_targets = {}
        temp = temperature or self.global_temperature
        
        for task_name, outputs in teacher_outputs.items():
            # Only process classification tasks
            task_type = self.task_types.get(task_name, 'classification')
            if task_type == 'regression':
                # Skip regression tasks - no soft targets needed
                continue
                
            if task_name in self.task_distillers:
                soft_targets[task_name] = self.task_distillers[task_name].prepare_teacher_targets(
                    outputs, temp
                )
            else:
                # Default softmax for classification tasks not in distillers
                soft_targets[task_name] = F.softmax(outputs / temp, dim=-1)
                
        return soft_targets
        
    def compute_distillation_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        student_features: Optional[Dict[str, torch.Tensor]] = None,
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task distillation loss.
        
        Args:
            student_outputs: Dictionary of student predictions for each task
            teacher_outputs: Dictionary of teacher predictions for each task
            targets: Dictionary of ground truth labels for each task
            student_features: Student features for each task (optional)
            teacher_features: Teacher features for each task (optional)
            
        Returns:
            Dictionary containing loss components for all tasks
        """
        all_losses = {}
        total_loss = 0
        
        # Compute distillation loss for each task (only classification tasks)
        for task_name in student_outputs.keys():
            # Skip regression tasks
            task_type = self.task_types.get(task_name, 'classification')
            if task_type == 'regression':
                logging.debug(f"Skipping distillation for regression task: {task_name}")
                continue
                
            if task_name in teacher_outputs and task_name in self.task_distillers:
                task_losses = self.task_distillers[task_name].compute_distillation_loss(
                    student_outputs[task_name],
                    teacher_outputs[task_name],
                    targets.get(task_name),
                    student_features.get(task_name) if student_features else None,
                    teacher_features.get(task_name) if teacher_features else None
                )
                
                # Add task prefix to loss names
                for loss_name, loss_value in task_losses.items():
                    all_losses[f'{task_name}_{loss_name}'] = loss_value
                    
                total_loss += task_losses['combined_loss']
                
        # Cross-task distillation (experimental)
        if self.cross_task_distillation:
            cross_task_loss = self._compute_cross_task_loss(student_outputs, teacher_outputs)
            all_losses['cross_task_loss'] = cross_task_loss
            total_loss += cross_task_loss
            
        all_losses['total_loss'] = total_loss
        return all_losses
        
    def _compute_cross_task_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute cross-task knowledge transfer loss."""
        # This is a placeholder for more sophisticated cross-task distillation
        # Could involve attention mechanisms or shared representations
        return torch.tensor(0.0, device=next(iter(student_outputs.values())).device)


class SelfDistillation(BaseDistillation):
    """
    Self-distillation using ensemble of previous model states.
    
    Uses exponential moving average of model parameters or
    ensemble of snapshots as the teacher.
    """
    
    def __init__(
        self,
        momentum: float = 0.99,
        temperature: float = 4.0,
        alpha: float = 0.5,
        update_frequency: int = 1
    ):
        """
        Initialize self-distillation.
        
        Args:
            momentum: Momentum for exponential moving average
            temperature: Temperature for soft targets
            alpha: Weight for self-distillation loss
            update_frequency: Frequency of teacher model updates
        """
        self.momentum = momentum
        self.temperature = temperature
        self.alpha = alpha
        self.update_frequency = update_frequency
        
        self.teacher_model = None
        self.update_counter = 0
        
    def update_teacher_model(self, student_model: nn.Module):
        """Update teacher model using exponential moving average."""
        self.update_counter += 1
        
        if self.update_counter % self.update_frequency != 0:
            return
            
        if self.teacher_model is None:
            # Initialize teacher as deep copy of student
            self.teacher_model = copy.deepcopy(student_model).eval()
            # Move to the correct device
            if next(student_model.parameters()).is_cuda:
                self.teacher_model = self.teacher_model.cuda()
        else:
            # Update teacher with EMA
            with torch.no_grad():
                for teacher_param, student_param in zip(
                    self.teacher_model.parameters(),
                    student_model.parameters()
                ):
                    teacher_param.data = (
                        self.momentum * teacher_param.data +
                        (1 - self.momentum) * student_param.data
                    )
                    
    def prepare_teacher_targets(
        self,
        teacher_outputs: torch.Tensor,
        temperature: float = None
    ) -> torch.Tensor:
        """Prepare soft targets from teacher outputs."""
        temp = temperature or self.temperature
        return F.softmax(teacher_outputs / temp, dim=-1)
        
    def compute_distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute self-distillation loss."""
        losses = {}
        
        # Task loss
        if targets is not None:
            task_loss = F.cross_entropy(student_outputs, targets)
            losses['task_loss'] = task_loss
        else:
            losses['task_loss'] = torch.tensor(0.0, device=student_outputs.device)
            
        # Self-distillation loss
        if self.teacher_model is not None:
            student_soft = F.log_softmax(student_outputs / self.temperature, dim=-1)
            teacher_soft = self.prepare_teacher_targets(teacher_outputs)
            
            kl_div = nn.KLDivLoss(reduction='batchmean')
            distillation_loss = kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
            losses['distillation_loss'] = distillation_loss
        else:
            losses['distillation_loss'] = torch.tensor(0.0, device=student_outputs.device)
            
        # Combined loss
        combined_loss = (
            (1 - self.alpha) * losses['task_loss'] +
            self.alpha * losses['distillation_loss']
        )
        losses['combined_loss'] = combined_loss
        
        return losses


class FeatureMatching:
    """
    Feature matching for intermediate layer distillation.
    
    Matches feature representations between teacher and student
    at specified intermediate layers.
    """
    
    def __init__(
        self,
        layer_mapping: Dict[str, str],
        matching_loss: str = 'mse',
        feature_transform: Optional[str] = None
    ):
        """
        Initialize feature matching.
        
        Args:
            layer_mapping: Mapping from student layers to teacher layers
            matching_loss: Type of matching loss ('mse', 'cosine', 'attention')
            feature_transform: Feature transformation method ('linear', 'conv', None)
        """
        self.layer_mapping = layer_mapping
        self.matching_loss = matching_loss
        self.feature_transform = feature_transform
        
        # Initialize loss functions
        if matching_loss == 'mse':
            self.loss_fn = nn.MSELoss()
        elif matching_loss == 'cosine':
            self.loss_fn = nn.CosineEmbeddingLoss()
        else:
            self.loss_fn = nn.MSELoss()
            
        # Feature transformation layers
        self.transform_layers = nn.ModuleDict()
        
    def add_transform_layer(self, layer_name: str, input_dim: int, output_dim: int):
        """Add a transformation layer for feature matching."""
        if self.feature_transform == 'linear':
            self.transform_layers[layer_name] = nn.Linear(input_dim, output_dim)
        elif self.feature_transform == 'conv':
            self.transform_layers[layer_name] = nn.Conv1d(input_dim, output_dim, 1)
        else:
            self.transform_layers[layer_name] = nn.Identity()
            
    def compute_matching_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            student_features: Dictionary of student layer features
            teacher_features: Dictionary of teacher layer features
            
        Returns:
            Feature matching loss
        """
        total_loss = 0
        num_layers = 0
        
        for student_layer, teacher_layer in self.layer_mapping.items():
            if student_layer in student_features and teacher_layer in teacher_features:
                student_feat = student_features[student_layer]
                teacher_feat = teacher_features[teacher_layer]
                
                # Apply transformation if needed
                if student_layer in self.transform_layers:
                    if student_feat.dim() == 3:  # [batch, seq, dim]
                        student_feat = student_feat.transpose(1, 2)  # [batch, dim, seq]
                        student_feat = self.transform_layers[student_layer](student_feat)
                        student_feat = student_feat.transpose(1, 2)  # [batch, seq, dim]
                    else:
                        student_feat = self.transform_layers[student_layer](student_feat)
                        
                # Compute matching loss
                if self.matching_loss == 'cosine':
                    # Flatten features for cosine similarity
                    student_flat = student_feat.view(student_feat.size(0), -1)
                    teacher_flat = teacher_feat.view(teacher_feat.size(0), -1)
                    target = torch.ones(student_flat.size(0)).to(student_feat.device)
                    loss = self.loss_fn(student_flat, teacher_flat.detach(), target)
                elif self.matching_loss == 'attention':
                    loss = self._compute_attention_matching_loss(student_feat, teacher_feat)
                else:  # MSE
                    loss = self.loss_fn(student_feat, teacher_feat.detach())
                    
                total_loss += loss
                num_layers += 1
                
        return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)
        
    def _compute_attention_matching_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention-based feature matching loss."""
        # Compute attention maps
        student_attention = torch.softmax(
            torch.sum(student_features ** 2, dim=-1, keepdim=True), dim=1
        )
        teacher_attention = torch.softmax(
            torch.sum(teacher_features ** 2, dim=-1, keepdim=True), dim=1
        )
        
        # Match attention distributions
        kl_div = nn.KLDivLoss(reduction='batchmean')
        return kl_div(
            torch.log(student_attention + 1e-8),
            teacher_attention.detach()
        )


class DistillationTrainer:
    """
    Training wrapper that integrates knowledge distillation into training loop.
    
    Handles teacher model loading, distillation loss computation,
    and training progress tracking.
    """
    
    def __init__(
        self,
        distillation_method: BaseDistillation,
        teacher_model: Optional[nn.Module] = None,
        teacher_checkpoint: Optional[str] = None,
        freeze_teacher: bool = True,
        distillation_schedule: Optional[Callable] = None,
        model_class: Optional[type] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize distillation trainer.
        
        Args:
            distillation_method: Distillation method to use
            teacher_model: Pre-loaded teacher model
            teacher_checkpoint: Path to teacher model checkpoint
            freeze_teacher: Whether to freeze teacher model parameters
            distillation_schedule: Schedule for distillation weight
            model_class: Model class for loading teacher from state_dict
            model_kwargs: Model keyword arguments for teacher instantiation
        """
        self.distillation_method = distillation_method
        self.teacher_model = teacher_model
        self.freeze_teacher = freeze_teacher
        self.distillation_schedule = distillation_schedule
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        
        # Load teacher model if checkpoint provided
        if teacher_checkpoint and teacher_model is None:
            self.teacher_model = self._load_teacher_model(teacher_checkpoint)
            
        # Freeze teacher model
        if self.teacher_model and freeze_teacher:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            
    def _load_teacher_model(self, checkpoint_path: str) -> nn.Module:
        """Load teacher model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Try to load full model first
        if 'model' in checkpoint:
            return checkpoint['model']
        elif 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
            # Load from state dict using provided model class and kwargs
            if self.model_class is None:
                raise ValueError("Loading from state dict requires model_class to be provided")
            
            return load_teacher_model(
                checkpoint_path, 
                self.model_class, 
                self.model_kwargs
            )
        else:
            raise ValueError("Unknown checkpoint format")
            
    def compute_distillation_loss(
        self,
        student_model: nn.Module,
        batch: Dict[str, torch.Tensor],
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss for a batch.
        
        Args:
            student_model: Student model
            batch: Training batch
            epoch: Current epoch (for scheduling)
            
        Returns:
            Dictionary of loss components
        """
        if self.teacher_model is None:
            raise ValueError("No teacher model available for distillation")
            
        # Extract inputs and targets from batch using ABR dataset keys
        if 'x0' in batch:
            # ABR dataset format - construct proper model inputs
            model_inputs = {
                'x0': batch['x0'],
                'stat': batch.get('stat')
            }
            
            # Extract targets based on task type
            if 'peak_exists' in batch:
                # Peak classification task
                targets = batch['peak_exists']
            elif 'meta' in batch and isinstance(batch['meta'], (list, tuple)):
                # Hearing loss classification task - extract from meta
                targets = torch.tensor([m['target'] for m in batch['meta']], 
                                     dtype=torch.long, device=batch['x0'].device)
            else:
                targets = None
        elif 'signal' in batch:
            # Legacy format
            model_inputs = batch['signal']  # Assume legacy models use tensor input
            targets = batch.get('target', batch.get('peaks'))
        else:
            # Tuple format
            model_inputs = batch[0]
            targets = batch[1] if len(batch) > 1 else None
            
        # Get student predictions
        if isinstance(model_inputs, dict):
            student_outputs = student_model(**model_inputs)
        else:
            student_outputs = student_model(model_inputs)
        
        # Get teacher predictions
        with torch.no_grad():
            if isinstance(model_inputs, dict):
                teacher_outputs = self.teacher_model(**model_inputs)
            else:
                teacher_outputs = self.teacher_model(model_inputs)
            
        # Apply distillation schedule if provided
        if self.distillation_schedule:
            alpha = self.distillation_schedule(epoch)
            if hasattr(self.distillation_method, 'alpha'):
                original_alpha = self.distillation_method.alpha
                self.distillation_method.alpha = alpha
            else:
                original_alpha = None
        else:
            original_alpha = None
            
        # Determine task loss function based on targets dtype/shape
        task_loss_fn = self._get_task_loss_fn(targets)
        
        # Compute distillation loss
        if isinstance(self.distillation_method, MultiTaskDistillation):
            # Handle multi-task case
            if not isinstance(student_outputs, dict):
                student_outputs = {'main': student_outputs}
            if not isinstance(teacher_outputs, dict):
                teacher_outputs = {'main': teacher_outputs}
            if not isinstance(targets, dict):
                targets = {'main': targets}
                
            losses = self.distillation_method.compute_distillation_loss(
                student_outputs, teacher_outputs, targets, task_loss_fn=task_loss_fn
            )
        else:
            # Handle single-task case
            if isinstance(student_outputs, dict):
                student_outputs = student_outputs[list(student_outputs.keys())[0]]
            if isinstance(teacher_outputs, dict):
                teacher_outputs = teacher_outputs[list(teacher_outputs.keys())[0]]
                
            losses = self.distillation_method.compute_distillation_loss(
                student_outputs, teacher_outputs, targets, task_loss_fn=task_loss_fn
            )
            
        # Restore original alpha if modified
        if original_alpha is not None and hasattr(self.distillation_method, 'alpha'):
            self.distillation_method.alpha = original_alpha
            
        return losses
    
    def _get_task_loss_fn(self, targets: torch.Tensor) -> str:
        """
        Determine appropriate task loss function based on targets dtype/shape.
        
        Args:
            targets: Target tensor
            
        Returns:
            Loss function type: 'bce' for binary classification, 'ce' for multiclass
        """
        if targets is None:
            return 'bce'  # Default for reconstruction tasks
            
        if targets.dtype == torch.long:
            # Long dtype typically indicates multiclass classification
            return 'ce'
        elif targets.dtype == torch.float:
            # Float dtype with values in [0,1] typically indicates binary classification
            if torch.all((targets >= 0) & (targets <= 1)):
                return 'bce'
            else:
                return 'ce'  # Regression or other tasks
        else:
            return 'bce'  # Default fallback
        
    def update_teacher(self, student_model: nn.Module, epoch: int):
        """Update teacher model (for self-distillation)."""
        if isinstance(self.distillation_method, SelfDistillation):
            self.distillation_method.update_teacher_model(student_model)


def create_teacher_student_pair(
    teacher_config: Dict[str, Any],
    student_config: Dict[str, Any],
    device: str = 'cpu'
) -> Tuple[nn.Module, nn.Module]:
    """
    Create teacher-student model pair from configurations.
    
    Args:
        teacher_config: Teacher model configuration
        student_config: Student model configuration
        device: Device to load models on
        
    Returns:
        Tuple of (teacher_model, student_model)
    """
    # This would need to be adapted based on the specific model classes
    # For now, return placeholder
    raise NotImplementedError("Model creation from config not implemented")


def load_teacher_model(
    checkpoint_path: str,
    model_class: type,
    model_kwargs: Dict[str, Any],
    device: str = 'cpu'
) -> nn.Module:
    """
    Load teacher model from checkpoint.
    
    Args:
        checkpoint_path: Path to teacher model checkpoint
        model_class: Class of the teacher model
        model_kwargs: Keyword arguments for model instantiation
        device: Device to load model on
        
    Returns:
        Loaded teacher model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model instance
    model = model_class(**model_kwargs)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    return model


def evaluate_distillation_effectiveness(
    teacher_model: nn.Module,
    student_model: nn.Module,
    test_loader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate the effectiveness of knowledge distillation.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        test_loader: Test data loader
        device: Device for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    teacher_model.eval()
    student_model.eval()
    
    teacher_correct = 0
    student_correct = 0
    total_samples = 0
    
    teacher_losses = []
    student_losses = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                inputs = batch['signal'].to(device)
                targets = batch.get('target', batch.get('peaks')).to(device)
            else:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
            # Get predictions
            teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)
            
            # Handle multi-output case
            if isinstance(teacher_outputs, dict):
                teacher_outputs = teacher_outputs[list(teacher_outputs.keys())[0]]
            if isinstance(student_outputs, dict):
                student_outputs = student_outputs[list(student_outputs.keys())[0]]
                
            # Compute accuracy
            if teacher_outputs.dim() > 1 and teacher_outputs.size(-1) > 1:
                teacher_pred = torch.argmax(teacher_outputs, dim=-1)
                student_pred = torch.argmax(student_outputs, dim=-1)
                
                if targets.dim() > 1:
                    targets = torch.argmax(targets, dim=-1)
                    
                teacher_correct += (teacher_pred == targets).sum().item()
                student_correct += (student_pred == targets).sum().item()
                
            # Compute losses
            teacher_loss = criterion(teacher_outputs, targets)
            student_loss = criterion(student_outputs, targets)
            
            teacher_losses.append(teacher_loss.item())
            student_losses.append(student_loss.item())
            
            total_samples += targets.size(0)
            
    metrics = {
        'teacher_accuracy': teacher_correct / total_samples if total_samples > 0 else 0.0,
        'student_accuracy': student_correct / total_samples if total_samples > 0 else 0.0,
        'teacher_loss': np.mean(teacher_losses),
        'student_loss': np.mean(student_losses),
        'accuracy_gap': (teacher_correct - student_correct) / total_samples if total_samples > 0 else 0.0,
        'loss_gap': np.mean(teacher_losses) - np.mean(student_losses)
    }
    
    # Compute knowledge transfer efficiency
    if metrics['teacher_accuracy'] > 0:
        metrics['knowledge_transfer_ratio'] = metrics['student_accuracy'] / metrics['teacher_accuracy']
    else:
        metrics['knowledge_transfer_ratio'] = 0.0
        
    return metrics


# Distillation scheduling functions
def linear_distillation_schedule(epoch: int, max_epochs: int, start_alpha: float = 0.1, end_alpha: float = 0.9) -> float:
    """Linear schedule for distillation weight."""
    progress = min(epoch / max_epochs, 1.0)
    return start_alpha + progress * (end_alpha - start_alpha)


def cosine_distillation_schedule(epoch: int, max_epochs: int, start_alpha: float = 0.1, end_alpha: float = 0.9) -> float:
    """Cosine schedule for distillation weight."""
    progress = min(epoch / max_epochs, 1.0)
    cosine_progress = 0.5 * (1 + np.cos(np.pi * (1 - progress)))
    return end_alpha + (start_alpha - end_alpha) * cosine_progress


def step_distillation_schedule(epoch: int, step_epochs: List[int], alphas: List[float]) -> float:
    """Step schedule for distillation weight."""
    for i, step_epoch in enumerate(step_epochs):
        if epoch < step_epoch:
            return alphas[i]
    return alphas[-1]


# Smoke test for DistillationTrainer with dict input
def test_distillation_trainer_dict_input():
    """
    Smoke test that runs a forward pass with dict input for DistillationTrainer.
    """
    # Create mock models that accept dict inputs
    class MockABRModel(nn.Module):
        def __init__(self, output_dim=1):
            super().__init__()
            self.output_dim = output_dim
            self.linear = nn.Linear(200, output_dim)  # ABR signal length 200
            
        def forward(self, x0, stat=None):
            # Handle [B, C, T] input by flattening C*T
            if x0.dim() == 3:
                batch_size = x0.size(0)
                x = x0.view(batch_size, -1)  # Flatten [B, C*T]
            else:
                x = x0
            return self.linear(x)
    
    # Test setup
    batch_size = 4
    seq_len = 200
    
    # Create teacher and student models
    teacher_model = MockABRModel(output_dim=1)
    student_model = MockABRModel(output_dim=1)
    
    # Create distillation method
    distillation_method = KnowledgeDistillation(
        temperature=4.0,
        alpha=0.7,
        feature_matching=False
    )
    
    # Create trainer
    trainer = DistillationTrainer(
        distillation_method=distillation_method,
        teacher_model=teacher_model,
        model_class=MockABRModel,
        model_kwargs={'output_dim': 1}
    )
    
    # Create test batch with dict format
    batch = {
        'x0': torch.randn(batch_size, 1, seq_len),
        'stat': None,  # Optional field
        'peak_exists': torch.randint(0, 2, (batch_size,)).float()
    }
    
    # Test forward pass
    losses = trainer.compute_distillation_loss(
        student_model=student_model,
        batch=batch,
        epoch=0
    )
    
    # Validate outputs
    assert isinstance(losses, dict), "Losses should be a dictionary"
    assert 'distillation_loss' in losses, "Should contain distillation loss"
    assert isinstance(losses['distillation_loss'], torch.Tensor), "Loss should be a tensor"
    assert not torch.isnan(losses['distillation_loss']), "Loss should not be NaN"
    
    # Test task loss function selection
    assert trainer._get_task_loss_fn(batch['peak_exists']) == 'bce', "Should select BCE for float targets"
    
    # Test with multiclass targets (long dtype)
    multiclass_targets = torch.randint(0, 3, (batch_size,)).long()
    assert trainer._get_task_loss_fn(multiclass_targets) == 'ce', "Should select CE for long targets"
    
    # Test with different batch format (legacy)
    legacy_batch = {
        'signal': torch.randn(batch_size, seq_len),
        'target': torch.randint(0, 2, (batch_size,)).float()
    }
    
    legacy_losses = trainer.compute_distillation_loss(
        student_model=student_model,
        batch=legacy_batch,
        epoch=0
    )
    
    assert isinstance(legacy_losses, dict), "Legacy format should also work"
    
    print("DistillationTrainer dict input smoke test passed!")


if __name__ == "__main__":
    # Run the test when the file is executed directly
    test_distillation_trainer_dict_input()
