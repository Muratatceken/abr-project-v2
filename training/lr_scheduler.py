#!/usr/bin/env python3
"""
Learning Rate Schedulers for ABR Training

Implements various learning rate scheduling strategies including
cosine annealing, warm restarts, and custom ABR-optimized schedules.

Author: AI Assistant
Date: January 2025
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, 
    CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
)
from typing import Dict, Any, Optional, Union
import math
import warnings


class WarmupScheduler:
    """
    Learning rate scheduler with warmup period.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        main_scheduler: optim.lr_scheduler._LRScheduler,
        warmup_epochs: int = 10,
        warmup_factor: float = 0.1
    ):
        self.optimizer = optimizer
        self.main_scheduler = main_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.current_epoch = 0
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_lr_factor = (
                self.warmup_factor + 
                (1.0 - self.warmup_factor) * self.current_epoch / self.warmup_epochs
            )
            
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_lr_factor
        else:
            # Main scheduler phase
            if hasattr(self.main_scheduler, 'step'):
                if isinstance(self.main_scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau needs metric, handled separately
                    pass
                else:
                    self.main_scheduler.step(self.current_epoch - self.warmup_epochs)
    
    def step_with_metric(self, metric: float):
        """Step scheduler with metric (for ReduceLROnPlateau)."""
        if self.current_epoch >= self.warmup_epochs:
            if isinstance(self.main_scheduler, ReduceLROnPlateau):
                self.main_scheduler.step(metric)
    
    def get_last_lr(self):
        """Get last learning rate."""
        if self.current_epoch < self.warmup_epochs:
            warmup_lr_factor = (
                self.warmup_factor + 
                (1.0 - self.warmup_factor) * self.current_epoch / self.warmup_epochs
            )
            return [base_lr * warmup_lr_factor for base_lr in self.base_lrs]
        else:
            return self.main_scheduler.get_last_lr()


class CosineAnnealingWarmRestartsWithDecay(CosineAnnealingWarmRestarts):
    """
    Enhanced CosineAnnealingWarmRestarts with decay factor for restart amplitude.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        decay_factor: float = 1.0
    ):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.decay_factor = decay_factor
        self.restart_count = 0
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch=None):
        """Step with decay on restarts."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.last_epoch = epoch
        else:
            self.last_epoch = epoch
        
        # Check if we're at a restart
        current_cycle_length = self.T_0 * (self.T_mult ** self.restart_count)
        if epoch >= current_cycle_length:
            self.restart_count += 1
            # Apply decay to base learning rates
            for param_group, initial_lr in zip(self.optimizer.param_groups, self.initial_lrs):
                param_group['initial_lr'] = initial_lr * (self.decay_factor ** self.restart_count)
        
        super().step(epoch)


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    config: Dict[str, Any]
) -> Union[optim.lr_scheduler._LRScheduler, WarmupScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Scheduler configuration
        
    Returns:
        Configured learning rate scheduler
    """
    scheduler_type = config.get('type', 'cosine_annealing_warm_restarts')
    warmup_epochs = config.get('warmup_epochs', 0)
    
    # Create main scheduler
    if scheduler_type == 'step':
        main_scheduler = StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'multistep':
        main_scheduler = MultiStepLR(
            optimizer,
            milestones=config.get('milestones', [60, 120, 160]),
            gamma=config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'exponential':
        main_scheduler = ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95)
        )
    
    elif scheduler_type == 'cosine_annealing':
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 200),
            eta_min=config.get('eta_min', 1e-7)
        )
    
    elif scheduler_type == 'cosine_annealing_warm_restarts':
        if config.get('use_decay', False):
            main_scheduler = CosineAnnealingWarmRestartsWithDecay(
                optimizer,
                T_0=config.get('T_0', 50),
                T_mult=config.get('T_mult', 2),
                eta_min=config.get('eta_min', 1e-7),
                decay_factor=config.get('decay_factor', 0.9)
            )
        else:
            main_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.get('T_0', 50),
                T_mult=config.get('T_mult', 2),
                eta_min=config.get('eta_min', 1e-7)
            )
    
    elif scheduler_type == 'one_cycle':
        total_steps = config.get('total_steps')
        if total_steps is None:
            raise ValueError("total_steps must be specified for OneCycleLR")
        
        main_scheduler = OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', 1e-3),
            total_steps=total_steps,
            pct_start=config.get('pct_start', 0.3),
            anneal_strategy=config.get('anneal_strategy', 'cos'),
            div_factor=config.get('div_factor', 25.0),
            final_div_factor=config.get('final_div_factor', 1e4)
        )
    
    elif scheduler_type == 'reduce_on_plateau':
        main_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'min'),
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 10),
            verbose=config.get('verbose', True),
            threshold=config.get('threshold', 1e-4),
            threshold_mode=config.get('threshold_mode', 'rel'),
            cooldown=config.get('cooldown', 0),
            min_lr=config.get('min_lr', 1e-7),
            eps=config.get('eps', 1e-8)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Add warmup if specified
    if warmup_epochs > 0:
        return WarmupScheduler(
            optimizer,
            main_scheduler,
            warmup_epochs=warmup_epochs,
            warmup_factor=config.get('warmup_factor', 0.1)
        )
    else:
        return main_scheduler


class ABRCustomScheduler:
    """
    Custom learning rate scheduler optimized for ABR training.
    
    Features:
    - Warmup period for stable initial training
    - Cosine annealing with warm restarts
    - Automatic decay on validation plateau
    - Task-specific learning rate scaling
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int = 10,
        cosine_T0: int = 50,
        cosine_Tmult: int = 2,
        eta_min: float = 1e-7,
        plateau_patience: int = 15,
        plateau_factor: float = 0.5,
        task_lr_factors: Optional[Dict[str, float]] = None
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.plateau_count = 0
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Create cosine scheduler
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cosine_T0, T_mult=cosine_Tmult, eta_min=eta_min
        )
        
        # Task-specific scaling factors
        self.task_lr_factors = task_lr_factors or {}
    
    def step(self, val_metric: Optional[float] = None):
        """Step the scheduler."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase
            warmup_factor = self.current_epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        else:
            # Cosine annealing phase
            self.cosine_scheduler.step(self.current_epoch - self.warmup_epochs)
        
        # Plateau detection
        if val_metric is not None:
            if val_metric < self.best_metric - 1e-6:
                self.best_metric = val_metric
                self.plateau_count = 0
            else:
                self.plateau_count += 1
                
                # Apply plateau reduction
                if self.plateau_count >= self.plateau_patience:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= self.plateau_factor
                    self.plateau_count = 0
                    print(f"Reduced learning rate by factor {self.plateau_factor}")
    
    def get_lr(self):
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def apply_task_scaling(self, task_name: str):
        """Apply task-specific learning rate scaling."""
        if task_name in self.task_lr_factors:
            factor = self.task_lr_factors[task_name]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= factor


def create_abr_scheduler(
    optimizer: optim.Optimizer,
    config: Dict[str, Any]
) -> ABRCustomScheduler:
    """
    Create ABR-optimized custom scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        config: Scheduler configuration
        
    Returns:
        ABR custom scheduler
    """
    return ABRCustomScheduler(
        optimizer,
        warmup_epochs=config.get('warmup_epochs', 10),
        cosine_T0=config.get('T_0', 50),
        cosine_Tmult=config.get('T_mult', 2),
        eta_min=config.get('eta_min', 1e-7),
        plateau_patience=config.get('plateau_patience', 15),
        plateau_factor=config.get('plateau_factor', 0.5),
        task_lr_factors=config.get('task_lr_factors', {})
    )


def get_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> tuple:
    """
    Create optimizer and scheduler from configuration.
    
    Args:
        model: PyTorch model
        config: Training configuration
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Create optimizer
    optimizer_config = config['training']['optimizer']
    optimizer_type = optimizer_config.get('type', 'adamw').lower()
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config.get('weight_decay', 0),
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8)
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config.get('weight_decay', 1e-2),
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8),
            amsgrad=optimizer_config.get('amsgrad', False)
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 0),
            nesterov=optimizer_config.get('nesterov', False)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Create scheduler
    scheduler_config = config['training']['scheduler']
    
    if scheduler_config.get('type') == 'abr_custom':
        scheduler = create_abr_scheduler(optimizer, scheduler_config)
    else:
        scheduler = get_lr_scheduler(optimizer, scheduler_config)
    
    return optimizer, scheduler