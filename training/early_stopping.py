"""
Comprehensive early stopping framework for ABR transformer training.

This module implements various early stopping strategies including:
- Basic early stopping with patience and minimum delta
- Multi-task early stopping for different validation metrics
- Adaptive early stopping that adjusts based on training progress
- Early stopping callbacks with checkpoint management
"""

import torch
import numpy as np
import copy
from typing import Dict, List, Optional, Any, Union, Callable
import logging
from pathlib import Path
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Basic early stopping implementation with patience and minimum improvement threshold.
    
    Monitors a validation metric and stops training when no improvement is observed
    for a specified number of epochs (patience).
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        warmup_epochs: int = 0,
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for minimization, 'max' for maximization
            restore_best_weights: Whether to restore best weights when stopping
            warmup_epochs: Number of epochs to wait before starting early stopping
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose
        
        # Internal state
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_state_dict = None
        self.history = []
        
        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda current, best: current < best - self.min_delta
        elif mode == 'max':
            self.is_better = lambda current, best: current > best + self.min_delta
        else:
            raise ValueError(f"Mode {mode} is unknown!")
            
    def __call__(
        self, 
        score: float, 
        model: torch.nn.Module = None, 
        epoch: int = 0
    ) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            score: Current validation score
            model: Model to save best weights (optional)
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        # Skip early stopping during warmup
        if epoch < self.warmup_epochs:
            self.history.append({
                'epoch': epoch,
                'score': score,
                'best_score': self.best_score,
                'counter': 0,
                'action': 'warmup'
            })
            return False
            
        # Initialize best score on first call
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if model is not None and self.restore_best_weights:
                self.best_state_dict = copy.deepcopy(model.state_dict())
            self.history.append({
                'epoch': epoch,
                'score': score,
                'best_score': self.best_score,
                'counter': self.counter,
                'action': 'initialize'
            })
            return False
            
        # Check for improvement
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_state_dict = copy.deepcopy(model.state_dict())
            
            if self.verbose:
                logger.info(f"EarlyStopping: New best score {score:.6f} at epoch {epoch}")
                
            self.history.append({
                'epoch': epoch,
                'score': score,
                'best_score': self.best_score,
                'counter': self.counter,
                'action': 'improvement'
            })
        else:
            self.counter += 1
            self.history.append({
                'epoch': epoch,
                'score': score,
                'best_score': self.best_score,
                'counter': self.counter,
                'action': 'no_improvement'
            })
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"EarlyStopping: Stopping at epoch {epoch}. "
                              f"Best score {self.best_score:.6f} at epoch {self.best_epoch}")
                    
        return self.early_stop
        
    def restore_best_model(self, model: torch.nn.Module):
        """Restore the best model weights."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            if self.verbose:
                logger.info(f"Restored best model weights from epoch {self.best_epoch}")
        else:
            logger.warning("No best state dict available for restoration")
            
    def get_best_score(self) -> float:
        """Get the best score achieved."""
        return self.best_score
        
    def get_best_epoch(self) -> int:
        """Get the epoch where best score was achieved."""
        return self.best_epoch
        
    def reset(self):
        """Reset early stopping state."""
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_state_dict = None
        self.history = []
        
    def save_state(self, filepath: str):
        """Save early stopping state to file."""
        state = {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
            'early_stop': self.early_stop,
            'history': self.history,
            'config': {
                'patience': self.patience,
                'min_delta': self.min_delta,
                'mode': self.mode,
                'warmup_epochs': self.warmup_epochs
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_state(self, filepath: str):
        """Load early stopping state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        self.best_score = state['best_score']
        self.best_epoch = state['best_epoch']
        self.counter = state['counter']
        self.early_stop = state['early_stop']
        self.history = state['history']


class MultiTaskEarlyStopping:
    """
    Early stopping for multi-task learning with multiple validation metrics.
    
    Monitors multiple metrics and applies different early stopping criteria
    for different tasks or uses a combined metric approach.
    """
    
    def __init__(
        self,
        task_configs: Dict[str, Dict[str, Any]],
        combination_strategy: str = 'weighted_average',
        task_weights: Optional[Dict[str, float]] = None,
        global_patience: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize multi-task early stopping.
        
        Args:
            task_configs: Dictionary of task-specific early stopping configs
            combination_strategy: How to combine multiple metrics ('weighted_average', 'min', 'max', 'any')
            task_weights: Weights for combining metrics (if using weighted_average)
            global_patience: Global patience that overrides individual task patience
            verbose: Whether to print early stopping messages
        """
        self.task_configs = task_configs
        self.combination_strategy = combination_strategy
        self.task_weights = task_weights or {}
        self.global_patience = global_patience
        self.verbose = verbose
        
        # Create individual early stopping instances for each task
        self.task_early_stoppers = {}
        for task_name, config in task_configs.items():
            self.task_early_stoppers[task_name] = EarlyStopping(
                patience=config.get('patience', 10),
                min_delta=config.get('min_delta', 0.0),
                mode=config.get('mode', 'min'),
                restore_best_weights=False,  # Handle globally
                warmup_epochs=config.get('warmup_epochs', 0),
                verbose=False  # Handle globally
            )
            
        # Global state
        self.best_combined_score = None
        self.best_epoch = 0
        self.best_state_dict = None
        self.global_counter = 0
        self.early_stop = False
        
    def __call__(
        self,
        scores: Dict[str, float],
        model: torch.nn.Module = None,
        epoch: int = 0
    ) -> bool:
        """
        Check if early stopping should be triggered based on multiple metrics.
        
        Args:
            scores: Dictionary of task-specific validation scores
            model: Model to save best weights (optional)
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        # Update individual task early stoppers
        task_should_stop = {}
        for task_name, score in scores.items():
            if task_name in self.task_early_stoppers:
                should_stop = self.task_early_stoppers[task_name](score, None, epoch)
                task_should_stop[task_name] = should_stop
                
        # Compute combined score
        combined_score = self._compute_combined_score(scores)
        
        # Check for global improvement
        improved = False
        if self.best_combined_score is None or self._is_combined_better(combined_score, self.best_combined_score):
            self.best_combined_score = combined_score
            self.best_epoch = epoch
            self.global_counter = 0
            improved = True
            
            if model is not None:
                self.best_state_dict = copy.deepcopy(model.state_dict())
                
            if self.verbose:
                logger.info(f"MultiTaskEarlyStopping: New best combined score {combined_score:.6f} at epoch {epoch}")
        else:
            self.global_counter += 1
            
        # Determine if should stop based on combination strategy
        if self.combination_strategy == 'any':
            # Stop if any task should stop
            self.early_stop = any(task_should_stop.values())
        elif self.combination_strategy == 'all':
            # Stop only if all tasks should stop
            self.early_stop = all(task_should_stop.values()) and len(task_should_stop) > 0
        else:
            # Use global patience with combined metric
            if self.global_patience is not None:
                self.early_stop = self.global_counter >= self.global_patience
            else:
                # Use maximum patience from individual tasks
                max_patience = max([config.get('patience', 10) for config in self.task_configs.values()])
                self.early_stop = self.global_counter >= max_patience
                
        if self.early_stop and self.verbose:
            logger.info(f"MultiTaskEarlyStopping: Stopping at epoch {epoch}. "
                       f"Best combined score {self.best_combined_score:.6f} at epoch {self.best_epoch}")
            logger.info(f"Task stopping status: {task_should_stop}")
            
        return self.early_stop
        
    def _compute_combined_score(self, scores: Dict[str, float]) -> float:
        """Compute combined score from multiple task scores."""
        if self.combination_strategy == 'weighted_average':
            total_weight = 0
            weighted_sum = 0
            for task_name, score in scores.items():
                weight = self.task_weights.get(task_name, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        elif self.combination_strategy == 'min':
            return min(scores.values())
        elif self.combination_strategy == 'max':
            return max(scores.values())
        else:
            # Default to average
            return np.mean(list(scores.values()))
            
    def _is_combined_better(self, current: float, best: float) -> bool:
        """Check if current combined score is better than best."""
        # This assumes 'min' mode for combined score - could be made configurable
        return current < best
        
    def restore_best_model(self, model: torch.nn.Module):
        """Restore the best model weights."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            if self.verbose:
                logger.info(f"Restored best model weights from epoch {self.best_epoch}")
                
    def get_task_best_scores(self) -> Dict[str, float]:
        """Get best scores for each task."""
        return {task_name: stopper.get_best_score() 
                for task_name, stopper in self.task_early_stoppers.items()}


class AdaptiveEarlyStopping(EarlyStopping):
    """
    Adaptive early stopping that adjusts patience based on training progress.
    
    Can increase patience when learning rate is reduced, or adjust thresholds
    based on training dynamics.
    """
    
    def __init__(
        self,
        initial_patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        patience_factor: float = 1.5,
        lr_patience_multiplier: float = 2.0,
        plateau_threshold: int = 5,
        **kwargs
    ):
        """
        Initialize adaptive early stopping.
        
        Args:
            initial_patience: Initial patience value
            patience_factor: Factor to multiply patience when adapting
            lr_patience_multiplier: Multiply patience by this when LR is reduced
            plateau_threshold: Number of epochs to detect plateau
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(patience=initial_patience, min_delta=min_delta, mode=mode, **kwargs)
        
        self.initial_patience = initial_patience
        self.patience_factor = patience_factor
        self.lr_patience_multiplier = lr_patience_multiplier
        self.plateau_threshold = plateau_threshold
        
        self.lr_reductions = 0
        self.score_history = []
        
    def adapt_to_lr_reduction(self):
        """Adapt patience when learning rate is reduced."""
        old_patience = self.patience
        self.patience = int(self.patience * self.lr_patience_multiplier)
        self.lr_reductions += 1
        
        if self.verbose:
            logger.info(f"AdaptiveEarlyStopping: LR reduced, patience adapted from {old_patience} to {self.patience}")
            
    def __call__(self, score: float, model: torch.nn.Module = None, epoch: int = 0) -> bool:
        """Enhanced call with adaptive behavior."""
        self.score_history.append(score)
        
        # Detect plateau and adapt patience
        if len(self.score_history) >= self.plateau_threshold:
            recent_scores = self.score_history[-self.plateau_threshold:]
            score_std = np.std(recent_scores)
            
            # If scores are very stable (low std), increase patience
            if score_std < self.min_delta * 0.1:
                old_patience = self.patience
                self.patience = int(self.patience * self.patience_factor)
                if self.verbose:
                    logger.info(f"AdaptiveEarlyStopping: Plateau detected, patience adapted from {old_patience} to {self.patience}")
                    
        return super().__call__(score, model, epoch)


class EarlyStoppingCallback:
    """
    Callback interface for early stopping integration with training loops.
    
    Provides hooks for training loop integration and handles checkpoint
    saving and restoration automatically.
    """
    
    def __init__(
        self,
        early_stopper: Union[EarlyStopping, MultiTaskEarlyStopping, AdaptiveEarlyStopping],
        checkpoint_dir: Optional[str] = None,
        save_best_only: bool = True,
        metric_name: str = 'val_loss'
    ):
        """
        Initialize early stopping callback.
        
        Args:
            early_stopper: Early stopping instance
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Whether to save only the best model
            metric_name: Name of the metric being monitored
        """
        self.early_stopper = early_stopper
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
    def on_epoch_end(
        self,
        epoch: int,
        logs: Dict[str, float],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None
    ) -> bool:
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
            model: Model being trained
            optimizer: Optimizer (optional)
            
        Returns:
            True if training should stop
        """
        # Extract relevant metric(s)
        if isinstance(self.early_stopper, MultiTaskEarlyStopping):
            # Multi-task case - extract all relevant metrics
            scores = {key: value for key, value in logs.items() 
                     if key in self.early_stopper.task_configs}
            should_stop = self.early_stopper(scores, model, epoch)
        else:
            # Single metric case
            if self.metric_name not in logs:
                logger.warning(f"Metric '{self.metric_name}' not found in logs. Available: {list(logs.keys())}")
                return False
                
            score = logs[self.metric_name]
            should_stop = self.early_stopper(score, model, epoch)
            
        # Save checkpoint if improved or if not save_best_only
        if self.checkpoint_dir:
            if hasattr(self.early_stopper, 'counter'):
                improved = self.early_stopper.counter == 0
            else:
                improved = True  # For multi-task, assume improved if no early stop
                
            if improved or not self.save_best_only:
                self._save_checkpoint(epoch, model, optimizer, logs)
                
        return should_stop
        
    def on_training_end(self, model: torch.nn.Module):
        """Called when training ends (either early stopping or completion)."""
        if hasattr(self.early_stopper, 'restore_best_model'):
            self.early_stopper.restore_best_model(model)
            
        # Save final state
        if self.checkpoint_dir:
            state_file = self.checkpoint_dir / 'early_stopping_state.json'
            if hasattr(self.early_stopper, 'save_state'):
                self.early_stopper.save_state(str(state_file))
                
    def _save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        logs: Dict[str, float] = None
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'logs': logs or {}
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        checkpoint_file = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_file)
        
        # Also save as 'best.pth' if this is the best model
        if hasattr(self.early_stopper, 'counter') and self.early_stopper.counter == 0:
            best_file = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_file)


def create_early_stopping_from_config(config: Dict[str, Any]) -> Optional[Union[EarlyStopping, MultiTaskEarlyStopping]]:
    """
    Factory function to create early stopping from configuration.
    
    Args:
        config: Early stopping configuration dictionary
        
    Returns:
        Configured early stopping instance or None if disabled
    """
    if not config.get('enabled', True):
        return None
        
    stopping_type = config.get('type', 'basic')
    
    if stopping_type == 'basic':
        return EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.0),
            mode=config.get('mode', 'min'),
            restore_best_weights=config.get('restore_best_weights', True),
            warmup_epochs=config.get('warmup_epochs', 0),
            verbose=config.get('verbose', True)
        )
    elif stopping_type == 'multi_task':
        return MultiTaskEarlyStopping(
            task_configs=config.get('task_configs', {}),
            combination_strategy=config.get('combination_strategy', 'weighted_average'),
            task_weights=config.get('task_weights', {}),
            global_patience=config.get('global_patience'),
            verbose=config.get('verbose', True)
        )
    elif stopping_type == 'adaptive':
        return AdaptiveEarlyStopping(
            initial_patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.0),
            mode=config.get('mode', 'min'),
            patience_factor=config.get('patience_factor', 1.5),
            lr_patience_multiplier=config.get('lr_patience_multiplier', 2.0),
            plateau_threshold=config.get('plateau_threshold', 5),
            restore_best_weights=config.get('restore_best_weights', True),
            warmup_epochs=config.get('warmup_epochs', 0),
            verbose=config.get('verbose', True)
        )
    else:
        raise ValueError(f"Unknown early stopping type: {stopping_type}")


class EarlyStoppingCallback:
    """
    Callback wrapper for early stopping that integrates with training loops.
    
    Provides a convenient interface for using early stopping in training loops
    with automatic checkpointing and metric tracking.
    """
    
    def __init__(
        self,
        early_stopper,
        checkpoint_dir: str,
        save_best_only: bool = True,
        metric_name: str = 'val_loss'
    ):
        """
        Initialize early stopping callback.
        
        Args:
            early_stopper: Early stopping instance (EarlyStopping, etc.)
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Whether to save only the best checkpoint
            metric_name: Name of the metric to monitor
        """
        self.early_stopper = early_stopper
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model=None, **kwargs) -> bool:
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of validation metrics
            model: Model instance for checkpointing
            **kwargs: Additional arguments (optimizer, etc.)
            
        Returns:
            True if training should stop, False otherwise
        """
        # Get the metric value
        metric_value = metrics.get(self.metric_name, float('inf'))
        
        # Check if we should stop
        should_stop = self.early_stopper(metric_value, model=model)
        
        # Save checkpoint if this is the best model so far
        if self._is_best_score(metric_value):
            if model is not None:
                self._save_best_checkpoint(model, epoch, metrics, **kwargs)
                
        return should_stop
    
    def _is_best_score(self, metric_value: float) -> bool:
        """
        Check if the current metric value is the best seen so far.
        
        Args:
            metric_value: Current metric value
            
        Returns:
            True if this is the best score so far
        """
        if not hasattr(self.early_stopper, 'best_score') or self.early_stopper.best_score is None:
            return True
            
        # Check based on early stopper's mode
        if hasattr(self.early_stopper, 'mode'):
            if self.early_stopper.mode == 'min':
                return metric_value < self.early_stopper.best_score
            elif self.early_stopper.mode == 'max':
                return metric_value > self.early_stopper.best_score
        
        # Default to minimization if mode is not available
        return metric_value < self.early_stopper.best_score
    
    def on_training_end(self, model=None):
        """Called when training ends."""
        if hasattr(self.early_stopper, 'restore_best_weights') and self.early_stopper.restore_best_weights:
            if model is not None and hasattr(self.early_stopper, 'best_state_dict'):
                if self.early_stopper.best_state_dict is not None:
                    model.load_state_dict(self.early_stopper.best_state_dict)
                    logger.info("Restored best model weights")
    
    def _save_best_checkpoint(self, model, epoch: int, metrics: Dict[str, float], **kwargs):
        """Save the best checkpoint."""
        if not self.save_best_only:
            return
            
        checkpoint_path = self.checkpoint_dir / "best_model.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'early_stopping_state': {
                'best_score': getattr(self.early_stopper, 'best_score', None),
                'counter': getattr(self.early_stopper, 'counter', 0),
            }
        }
        
        # Add optimizer state if provided
        if 'optimizer' in kwargs:
            checkpoint['optimizer_state_dict'] = kwargs['optimizer'].state_dict()
            
        # Add other states
        for key, value in kwargs.items():
            if hasattr(value, 'state_dict'):
                checkpoint[f'{key}_state_dict'] = value.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved best checkpoint to {checkpoint_path}")


def analyze_early_stopping_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze early stopping history for insights.
    
    Args:
        history: List of early stopping history records
        
    Returns:
        Analysis results
    """
    if not history:
        return {}
        
    epochs = [record['epoch'] for record in history]
    scores = [record['score'] for record in history]
    best_scores = [record['best_score'] for record in history if record['best_score'] is not None]
    counters = [record['counter'] for record in history]
    
    improvements = sum(1 for record in history if record['action'] == 'improvement')
    no_improvements = sum(1 for record in history if record['action'] == 'no_improvement')
    
    analysis = {
        'total_epochs': len(history),
        'improvements': improvements,
        'no_improvements': no_improvements,
        'improvement_rate': improvements / len(history) if len(history) > 0 else 0,
        'final_score': scores[-1] if scores else None,
        'best_score': min(best_scores) if best_scores else None,
        'best_epoch': history[np.argmin([record.get('best_score', float('inf')) for record in history])]['epoch'] if best_scores else None,
        'max_counter': max(counters) if counters else 0,
        'score_improvement': scores[0] - scores[-1] if len(scores) > 1 else 0
    }
    
    return analysis


# Minimal unit test for EarlyStoppingCallback
def test_early_stopping_callback():
    """
    Minimal unit test for the callback's improved/not-improved branches.
    """
    import tempfile
    import shutil
    
    # Create a temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create a simple mock model
        class MockModel:
            def state_dict(self):
                return {'weight': torch.tensor([1.0])}
            
            def load_state_dict(self, state_dict):
                pass
        
        # Test with EarlyStopping
        early_stopper = EarlyStopping(patience=2, mode='min', restore_best_weights=True)
        callback = EarlyStoppingCallback(
            early_stopper=early_stopper,
            checkpoint_dir=temp_dir,
            metric_name='val_loss'
        )
        
        model = MockModel()
        
        # Test improvement case (should save checkpoint)
        metrics1 = {'val_loss': 0.5}
        should_stop1 = callback.on_epoch_end(0, metrics1, model=model)
        assert not should_stop1, "Should not stop on first improvement"
        
        # Test no improvement case
        metrics2 = {'val_loss': 0.6}  # Worse score
        should_stop2 = callback.on_epoch_end(1, metrics2, model=model)
        assert not should_stop2, "Should not stop after one bad epoch"
        
        # Test another no improvement (should trigger stop)
        metrics3 = {'val_loss': 0.7}  # Even worse
        should_stop3 = callback.on_epoch_end(2, metrics3, model=model)
        assert should_stop3, "Should stop after patience exceeded"
        
        # Test training end (should restore best weights)
        callback.on_training_end(model=model)
        
        # Test with MultiTaskEarlyStopping
        multi_early_stopper = MultiTaskEarlyStopping(
            task_configs={'task1': {'patience': 2, 'mode': 'min'}},
            combination_strategy='min'
        )
        multi_callback = EarlyStoppingCallback(
            early_stopper=multi_early_stopper,
            checkpoint_dir=temp_dir,
            metric_name='val_combined_score'
        )
        
        # Test with multi-task metrics
        multi_metrics = {'val_combined_score': 0.3, 'task1': 0.3}
        should_stop_multi = multi_callback.on_epoch_end(0, multi_metrics, model=model)
        assert not should_stop_multi, "Multi-task should not stop on first epoch"
        
        logger.info("EarlyStoppingCallback tests passed!")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run the test when the file is executed directly
    test_early_stopping_callback()
