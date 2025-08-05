#!/usr/bin/env python3
"""
Sequential Training Strategy for ABR Diffusion Model

Implements curriculum learning approach to address multi-task learning conflicts:

Phase 1: Signal reconstruction only (diffusion training)
Phase 2: Signal + Classification
Phase 3: Signal + Classification + Threshold
Phase 4: All tasks (Signal + Classification + Threshold + Peaks)

This approach prevents the task conflicts that were causing training instability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from omegaconf import DictConfig

from .trainer import ABRTrainer, TrainingMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingPhase:
    """Configuration for a training phase."""
    name: str
    epochs: int
    loss_weights: Dict[str, float]
    enabled_tasks: List[str]
    description: str
    learning_rate_multiplier: float = 1.0
    early_stopping_patience: Optional[int] = None


class SequentialTrainer(ABRTrainer):
    """
    Sequential trainer that implements curriculum learning for ABR model.
    
    Gradually introduces tasks to prevent multi-task learning conflicts.
    """
    
    def __init__(
        self,
        config: DictConfig,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader: Optional = None,
        device: Optional[torch.device] = None
    ):
        super().__init__(config, model, train_loader, val_loader, test_loader, device)
        
        # Setup sequential training phases
        self.setup_training_phases()
        
        # Current phase tracking
        self.current_phase = 0
        self.phase_epoch = 0
        self.total_epochs_completed = 0
        
        # Initialize loss weights with the first phase
        self.loss_weights = self.phases[0].loss_weights.copy() if self.phases else {}
        
        logger.info(f"Sequential trainer initialized with {len(self.phases)} phases")
        for i, phase in enumerate(self.phases):
            logger.info(f"Phase {i+1}: {phase.name} - {phase.epochs} epochs - {phase.enabled_tasks}")
    
    def setup_training_phases(self):
        """Setup the sequential training phases."""
        
        # Check if phases are defined in config
        if hasattr(self.config, 'sequential_training') and self.config.sequential_training.get('enabled', False):
            self.phases = self._load_phases_from_config()
        else:
            self.phases = self._get_default_phases()
        
        # Validate phases
        self._validate_phases()
    
    def _load_phases_from_config(self) -> List[TrainingPhase]:
        """Load training phases from configuration."""
        phases = []
        
        for phase_config in self.config.sequential_training.phases:
            phase = TrainingPhase(
                name=phase_config.get('phase', f'phase_{len(phases)+1}'),
                epochs=phase_config.get('epochs', 20),
                loss_weights=dict(phase_config.get('loss_weights', {})),
                enabled_tasks=phase_config.get('enabled_tasks', ['diffusion']),
                description=phase_config.get('description', ''),
                learning_rate_multiplier=phase_config.get('lr_multiplier', 1.0),
                early_stopping_patience=phase_config.get('early_stopping_patience')
            )
            phases.append(phase)
        
        return phases
    
    def _get_default_phases(self) -> List[TrainingPhase]:
        """Get default sequential training phases."""
        
        phases = [
            TrainingPhase(
                name="signal_only",
                epochs=30,
                loss_weights={
                    'diffusion': 1.0,
                    'peak_exist': 0.0,
                    'peak_latency': 0.0,
                    'peak_amplitude': 0.0,
                    'classification': 0.0,
                    'threshold': 0.0
                },
                enabled_tasks=['diffusion'],
                description="Learn basic signal reconstruction and diffusion process",
                learning_rate_multiplier=1.0,
                early_stopping_patience=15
            ),
            
            TrainingPhase(
                name="signal_classification",
                epochs=40,
                loss_weights={
                    'diffusion': 1.0,
                    'peak_exist': 0.0,
                    'peak_latency': 0.0,
                    'peak_amplitude': 0.0,
                    'classification': 2.0,  # High weight for classification
                    'threshold': 0.0
                },
                enabled_tasks=['diffusion', 'classification'],
                description="Add classification while maintaining signal quality",
                learning_rate_multiplier=0.5,  # Lower LR for stability
                early_stopping_patience=20
            ),
            
            TrainingPhase(
                name="signal_classification_threshold",
                epochs=30,
                loss_weights={
                    'diffusion': 1.0,
                    'peak_exist': 0.0,
                    'peak_latency': 0.0,
                    'peak_amplitude': 0.0,
                    'classification': 1.5,
                    'threshold': 0.8
                },
                enabled_tasks=['diffusion', 'classification', 'threshold'],
                description="Add threshold regression",
                learning_rate_multiplier=0.3,
                early_stopping_patience=15
            ),
            
            TrainingPhase(
                name="all_tasks",
                epochs=50,
                loss_weights={
                    'diffusion': 1.0,
                    'peak_exist': 0.2,
                    'peak_latency': 0.1,
                    'peak_amplitude': 0.05,  # Keep very low
                    'classification': 1.2,
                    'threshold': 0.6
                },
                enabled_tasks=['diffusion', 'classification', 'threshold', 'peaks'],
                description="Train all tasks with balanced weights",
                learning_rate_multiplier=0.2,
                early_stopping_patience=25
            )
        ]
        
        return phases
    
    def _validate_phases(self):
        """Validate training phases configuration."""
        if not self.phases:
            raise ValueError("No training phases defined")
        
        # Check that at least one phase exists
        total_epochs = sum(phase.epochs for phase in self.phases)
        if total_epochs == 0:
            raise ValueError("Total training epochs is zero")
        
        # Validate loss weights
        for phase in self.phases:
            if not phase.loss_weights:
                logger.warning(f"Phase {phase.name} has no loss weights defined")
        
        logger.info(f"Sequential training validated: {len(self.phases)} phases, {total_epochs} total epochs")
    
    def train(self, resume_from: Optional[str] = None):
        """
        Main sequential training loop.
        """
        try:
            # Resume from checkpoint if specified
            if resume_from:
                self.load_checkpoint(resume_from)
            
            logger.info("Starting sequential training...")
            logger.info(f"Total phases: {len(self.phases)}")
            
            # Train each phase sequentially
            for phase_idx, phase in enumerate(self.phases):
                self.current_phase = phase_idx
                self.phase_epoch = 0
                
                logger.info(f"\n{'='*60}")
                logger.info(f"PHASE {phase_idx + 1}: {phase.name.upper()}")
                logger.info(f"{'='*60}")
                logger.info(f"Description: {phase.description}")
                logger.info(f"Epochs: {phase.epochs}")
                logger.info(f"Enabled tasks: {phase.enabled_tasks}")
                logger.info(f"Loss weights: {phase.loss_weights}")
                
                # Setup phase-specific configuration
                self._setup_phase(phase)
                
                # Train this phase
                phase_success = self._train_phase(phase)
                
                if not phase_success:
                    logger.warning(f"Phase {phase.name} ended early")
                
                # Save phase checkpoint
                self.save_checkpoint(
                    is_best=False,
                    checkpoint_name=f"phase_{phase_idx + 1}_{phase.name}.pt"
                )
                
                logger.info(f"Phase {phase.name} completed")
            
            # Final checkpoint
            self.save_checkpoint(is_best=False, checkpoint_name="sequential_training_complete.pt")
            logger.info("Sequential training completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Sequential training interrupted by user")
            self.save_checkpoint(
                is_best=False,
                checkpoint_name=f"interrupted_phase_{self.current_phase + 1}.pt"
            )
        except Exception as e:
            logger.error(f"Sequential training failed: {e}")
            raise
        finally:
            # Cleanup
            if self.writer:
                self.writer.close()
            if self.use_wandb:
                import wandb
                wandb.finish()
    
    def _setup_phase(self, phase: TrainingPhase):
        """Setup configuration for a specific training phase."""
        
        # Update loss weights
        self.loss_weights = phase.loss_weights.copy()
        
        # Adjust learning rate if specified
        if phase.learning_rate_multiplier != 1.0:
            base_lr = self.config.training.optimizer.learning_rate
            new_lr = base_lr * phase.learning_rate_multiplier
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            logger.info(f"Adjusted learning rate: {base_lr:.2e} -> {new_lr:.2e}")
        
        # Update early stopping patience if specified
        if phase.early_stopping_patience:
            self.early_stopping_patience = phase.early_stopping_patience
        
        # Log phase setup
        logger.info(f"Phase setup complete: {phase.name}")
    
    def _train_phase(self, phase: TrainingPhase) -> bool:
        """
        Train a single phase.
        
        Returns:
            True if phase completed successfully, False if stopped early
        """
        phase_best_loss = float('inf')
        phase_patience_counter = 0
        
        for epoch_in_phase in range(phase.epochs):
            self.phase_epoch = epoch_in_phase
            self.current_epoch = self.total_epochs_completed + epoch_in_phase
            
            # Training epoch
            train_metrics = self.train_epoch()
            
            # Validation epoch
            val_metrics = self.validate_epoch()
            
            # Learning rate scheduling
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step()
            
            # Log phase progress
            logger.info(
                f"Phase {self.current_phase + 1} - Epoch {epoch_in_phase + 1}/{phase.epochs} "
                f"(Global: {self.current_epoch + 1}) - "
                f"Train Loss: {train_metrics.total_loss:.4f}, "
                f"Val Loss: {val_metrics.total_loss:.4f}"
            )
            
            # Enhanced monitoring
            if self.training_monitor is not None:
                epoch_time = 0  # Simple placeholder
                self.training_monitor.log_epoch(
                    epoch=self.current_epoch,
                    train_metrics=train_metrics.to_dict(),
                    val_metrics=val_metrics.to_dict(),
                    epoch_time=epoch_time
                )
            
            # Phase-specific early stopping
            if val_metrics.total_loss < phase_best_loss:
                phase_best_loss = val_metrics.total_loss
                phase_patience_counter = 0
                
                # Save best model for this phase
                self.save_checkpoint(
                    is_best=True,
                    checkpoint_name=f"phase_{self.current_phase + 1}_best.pt"
                )
            else:
                phase_patience_counter += 1
            
            # Check early stopping for this phase
            if (phase.early_stopping_patience and 
                phase_patience_counter >= phase.early_stopping_patience):
                logger.info(f"Phase {phase.name} early stopping triggered")
                break
        
        # Update total epochs
        self.total_epochs_completed += (epoch_in_phase + 1)
        
        return epoch_in_phase + 1 == phase.epochs  # True if completed all epochs
    
    def _compute_enhanced_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced loss computation with phase-aware task weighting.
        """
        loss_components = {}
        current_phase = self.phases[self.current_phase]
        
        # 1. DIFFUSION LOSS
        if 'diffusion' in current_phase.enabled_tasks:
            if 'noise' in outputs and outputs['noise'] is not None:
                diffusion_loss = F.mse_loss(outputs['noise'], noise)
                loss_components['diffusion_loss'] = diffusion_loss
            else:
                # Fallback: signal reconstruction loss
                signal_loss = F.mse_loss(outputs['recon'], batch['signal'])
                loss_components['signal_loss'] = signal_loss
        
        # 2. CLASSIFICATION LOSS
        if 'classification' in current_phase.enabled_tasks and 'class' in outputs:
            try:
                if self.enhanced_class_loss is not None:
                    class_loss = self.enhanced_class_loss(outputs['class'], batch['target'])
                    if self.dynamic_weights is not None:
                        self.dynamic_weights.update(outputs['class'], batch['target'])
                else:
                    class_loss = F.cross_entropy(outputs['class'], batch['target'])
                loss_components['classification_loss'] = class_loss
            except Exception as e:
                logging.getLogger(__name__).warning(f"Classification loss failed: {e}")
                loss_components['classification_loss'] = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 3. THRESHOLD LOSS
        if 'threshold' in current_phase.enabled_tasks and 'threshold' in outputs:
            try:
                # Handle threshold predictions with uncertainty (mean, std)
                threshold_pred = outputs['threshold']
                threshold_true = batch['threshold']
                
                # If prediction includes uncertainty, take only the mean (first dimension)
                if threshold_pred.shape[-1] == 2:
                    threshold_pred = threshold_pred[..., 0]  # Take mean, ignore std
                
                # Ensure both tensors have compatible shapes
                if threshold_true.dim() > threshold_pred.dim():
                    threshold_true = threshold_true.squeeze(-1)
                elif threshold_pred.dim() > threshold_true.dim():
                    threshold_pred = threshold_pred.squeeze(-1)
                
                threshold_loss = F.mse_loss(threshold_pred, threshold_true)
                loss_components['threshold_loss'] = threshold_loss
            except Exception as e:
                logging.getLogger(__name__).warning(f"Threshold loss failed: {e}")
                loss_components['threshold_loss'] = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 4. PEAK LOSSES
        if 'peaks' in current_phase.enabled_tasks and 'peak' in outputs:
            try:
                peak_losses = self.loss_fn.compute_peak_loss(
                    outputs['peak'], batch['v_peak'], batch['v_peak_mask']
                )
                loss_components['peak_exist_loss'] = peak_losses['exist']
                loss_components['peak_latency_loss'] = peak_losses['latency']
                loss_components['peak_amplitude_loss'] = peak_losses['amplitude']
            except Exception as e:
                logging.getLogger(__name__).warning(f"Peak loss failed: {e}")
                device = outputs['peak'][0].device if isinstance(outputs['peak'], (list, tuple)) else outputs['peak'].device
                loss_components['peak_exist_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                loss_components['peak_latency_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                loss_components['peak_amplitude_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 5. COMPUTE WEIGHTED TOTAL LOSS using phase-specific weights
        total_loss = 0.0
        
        for loss_name, weight in self.loss_weights.items():
            if weight > 0:
                # Map loss names to components
                component_map = {
                    'diffusion': 'diffusion_loss',
                    'signal': 'signal_loss',
                    'classification': 'classification_loss',
                    'threshold': 'threshold_loss',
                    'peak_exist': 'peak_exist_loss',
                    'peak_latency': 'peak_latency_loss',
                    'peak_amplitude': 'peak_amplitude_loss'
                }
                
                component_name = component_map.get(loss_name, loss_name)
                if component_name in loss_components:
                    total_loss += weight * loss_components[component_name]
        
        loss_components['total_loss'] = total_loss
        
        return total_loss, loss_components
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get detailed training status."""
        status = {
            'current_phase': self.current_phase + 1,
            'total_phases': len(self.phases),
            'phase_name': self.phases[self.current_phase].name,
            'phase_epoch': self.phase_epoch + 1,
            'phase_total_epochs': self.phases[self.current_phase].epochs,
            'total_epochs_completed': self.total_epochs_completed,
            'enabled_tasks': self.phases[self.current_phase].enabled_tasks,
            'current_loss_weights': self.loss_weights
        }
        
        return status


def create_sequential_trainer(
    config: DictConfig,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader: Optional = None,
    device: Optional[torch.device] = None
) -> SequentialTrainer:
    """Create sequential trainer instance."""
    
    trainer = SequentialTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device
    )
    
    logger.info("Sequential trainer created successfully")
    return trainer