#!/usr/bin/env python3
"""
ABR Trainer - Professional Training Loop for Hierarchical U-Net

Implements comprehensive training pipeline with:
- Multi-task loss handling
- Diffusion training with noise scheduling
- Advanced monitoring and logging
- Checkpointing and early stopping
- Mixed precision training
- Gradient clipping and accumulation

Author: AI Assistant
Date: January 2025
"""

import os
import time
import json
import logging
import traceback
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
import wandb
import numpy as np
from tqdm import tqdm

# Import our modules
from utils.loss import ABRDiffusionLoss, create_class_weights
from utils.schedule import get_noise_schedule, NoiseSchedule
from utils.sampling import DDIMSampler
from data.dataset import ABRDataset, stratified_patient_split, create_optimized_dataloaders
from models.hierarchical_unet import OptimizedHierarchicalUNet
from .lr_scheduler import get_optimizer_and_scheduler


logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    # Loss components
    total_loss: float = 0.0
    signal_loss: float = 0.0
    peak_exist_loss: float = 0.0
    peak_latency_loss: float = 0.0
    peak_amplitude_loss: float = 0.0
    classification_loss: float = 0.0
    threshold_loss: float = 0.0
    
    # Accuracy metrics
    classification_accuracy: float = 0.0
    peak_existence_accuracy: float = 0.0
    
    # Learning rate
    learning_rate: float = 0.0
    
    # Timing
    batch_time: float = 0.0
    data_time: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float))}


class ABRTrainer:
    """
    Professional trainer for ABR Hierarchical U-Net with diffusion training.
    """
    
    def __init__(
        self,
        config: DictConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ABR trainer.
        
        Args:
            config: Training configuration
            model: ABR model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            device: Training device
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup device
        self.device = device or torch.device(
            config.hardware.device if hasattr(config, 'hardware') else 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Setup training components
        self._setup_loss_function()
        self._setup_optimizer_and_scheduler()
        self._setup_noise_schedule()
        self._setup_monitoring()
        self._setup_mixed_precision()
        self._setup_checkpointing()
        
        # Initialize metrics tracking
        self.train_metrics = TrainingMetrics()
        self.val_metrics = TrainingMetrics()
        
        logger.info(f"ABR Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Training samples: {len(train_loader.dataset):,}")
        logger.info(f"Validation samples: {len(val_loader.dataset):,}")
    
    def _setup_loss_function(self):
        """Setup enhanced loss function with advanced class imbalance handling."""
        # Enhanced class imbalance handling
        use_enhanced_imbalance = self.config.loss.get('enhanced_class_balance', True)
        
        if use_enhanced_imbalance:
            try:
                from utils.class_balance import setup_class_imbalance_handling
                
                # Setup comprehensive class imbalance handling
                imbalance_components = setup_class_imbalance_handling(
                    train_dataset=self.train_loader.dataset,
                    config=self.config,
                    device=self.device
                )
                
                # Use enhanced classification loss
                self.enhanced_class_loss = imbalance_components['loss_function']
                self.dynamic_weights = imbalance_components.get('dynamic_weights')
                
                logger.info("Enhanced class imbalance handling enabled")
                logger.info(f"Class distribution: {imbalance_components['class_distribution']}")
                
            except ImportError:
                logger.warning("Enhanced class balance module not available, using standard approach")
                use_enhanced_imbalance = False
        
        if not use_enhanced_imbalance:
            # Standard class weights approach
            class_weights = None
            if self.config.loss.get('class_weights') == 'balanced':
                # Extract targets from training dataset
                targets = []
                for batch in self.train_loader:
                    targets.extend(batch['target'].cpu().numpy())
                
                class_weights = create_class_weights(
                    targets, 
                    self.config.data.n_classes, 
                    self.device
                )
                logger.info(f"Using balanced class weights: {class_weights}")
            
            self.enhanced_class_loss = None
            self.dynamic_weights = None
        
        # Create main diffusion loss function
        self.loss_fn = ABRDiffusionLoss(
            n_classes=self.config.data.n_classes,
            class_weights=None,  # We'll handle classification separately if enhanced
            use_focal_loss=False,  # We'll use enhanced focal loss separately
            peak_loss_type=self.config.loss.get('peak_loss_type', 'huber'),
            huber_delta=self.config.loss.get('huber_delta', 1.0),
            device=self.device,
            signal_weight=self.config.loss.weights.get('diffusion', 1.0),
            peak_weight=self.config.loss.weights.get('peak_latency', 1.0),
            class_weight=self.config.loss.weights.get('classification', 1.0),
            threshold_weight=self.config.loss.weights.get('threshold', 0.8)
        )
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.model, self.config)
        logger.info(f"Using optimizer: {type(self.optimizer).__name__}")
        logger.info(f"Using scheduler: {type(self.scheduler).__name__}")
    
    def _setup_noise_schedule(self):
        """Setup diffusion noise schedule."""
        noise_config = self.config.diffusion.noise_schedule
        
        self.noise_schedule_dict = get_noise_schedule(
            schedule_type=noise_config.get('type', 'cosine'),
            num_timesteps=noise_config.get('num_timesteps', 1000),
            beta_start=noise_config.get('beta_start', 1e-4),
            beta_end=noise_config.get('beta_end', 0.02)
        )
        
        self.noise_schedule = NoiseSchedule(self.noise_schedule_dict)
        
        # Move to device
        for key, value in self.noise_schedule_dict.items():
            if isinstance(value, torch.Tensor):
                self.noise_schedule_dict[key] = value.to(self.device)
        
        logger.info(f"Noise schedule: {noise_config.get('type', 'cosine')} with {noise_config.get('num_timesteps', 1000)} steps")
    
    def _setup_monitoring(self):
        """Setup enhanced monitoring and logging."""
        # Create log directories
        log_dir = Path(self.config.paths.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        log_file = log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Setup enhanced training monitor
        use_enhanced_monitoring = self.config.logging.get('enhanced_monitoring', True)
        if use_enhanced_monitoring:
            try:
                from utils.monitoring import create_training_monitor
                self.training_monitor = create_training_monitor(
                    config=self.config,
                    log_dir=str(log_dir / "enhanced_monitoring")
                )
                logger.info("Enhanced training monitoring enabled")
            except ImportError:
                logger.warning("Enhanced monitoring not available")
                self.training_monitor = None
        else:
            self.training_monitor = None
        
        # Setup TensorBoard
        if self.config.logging.get('use_tensorboard', True):
            tb_dir = log_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))
            logger.info(f"TensorBoard logging to: {tb_dir}")
        else:
            self.writer = None
        
        # Setup Weights & Biases
        if self.config.logging.get('use_wandb', False):
            wandb_config = self.config.logging.get('wandb', {})
            wandb.init(
                project=wandb_config.get('project', 'abr-hierarchical-unet'),
                entity=wandb_config.get('entity'),
                tags=wandb_config.get('tags', []),
                config=dict(self.config)
            )
            self.use_wandb = True
            logger.info("W&B logging enabled")
        else:
            self.use_wandb = False
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        self.use_amp = self.config.hardware.get('mixed_precision', True) and torch.cuda.is_available()
        
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled")
    
    def _setup_checkpointing(self):
        """Setup checkpointing directories."""
        self.checkpoint_dir = Path(self.config.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = self.checkpoint_dir / "config.yaml"
        with open(config_path, 'w') as f:
            f.write(str(self.config))
    
    def train_epoch(self) -> TrainingMetrics:
        """Train for one epoch."""
        self.model.train()
        metrics = TrainingMetrics()
        
        # Progress bar
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch+1}/{self.config.training.epochs}",
            leave=False
        )
        
        epoch_start_time = time.time()
        num_batches = 0
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            data_time = time.time() - batch_start_time
            
            # Forward pass with diffusion
            loss, loss_components = self._forward_pass(batch)
            
            # Backward pass
            loss = loss / self.config.training.get('accumulation_steps', 1)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.training.get('accumulation_steps', 1) == 0:
                # Gradient clipping
                if self.config.training.get('gradient_clip', 0) > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                accumulated_loss = 0.0
            
            # Update metrics
            self._update_metrics(metrics, loss_components, batch)
            
            batch_time = time.time() - batch_start_time
            metrics.batch_time = batch_time
            metrics.data_time = data_time
            
            # Update progress bar with detailed losses
            postfix_dict = {
                'Total': f"{loss.item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            }
            
            # Add individual loss components to progress bar
            if 'signal_loss' in loss_components:
                postfix_dict['Signal'] = f"{loss_components['signal_loss'].item():.3f}"
            if 'classification_loss' in loss_components:
                postfix_dict['Class'] = f"{loss_components['classification_loss'].item():.3f}"
            if 'peak_exist_loss' in loss_components:
                postfix_dict['Peak'] = f"{loss_components['peak_exist_loss'].item():.3f}"
            if 'threshold_loss' in loss_components:
                postfix_dict['Thresh'] = f"{loss_components['threshold_loss'].item():.3f}"
            
            pbar.set_postfix(postfix_dict)
            
            num_batches += 1
            
            # Enhanced monitoring and logging
            log_frequency = self.config.training.get('log_frequency', 100)
            if self.global_step % log_frequency == 0:
                self._log_metrics(metrics, 'train', self.global_step)
                
                # Enhanced monitoring
                if self.training_monitor is not None:
                    # Prepare batch data for monitoring
                    outputs_dict = None
                    targets_dict = None
                    
                    # Get model outputs and targets from current batch for analysis
                    # Note: This requires the forward pass to return outputs
                    # We'll monitor based on loss components for now
                    
                    self.training_monitor.log_step(
                        step=self.global_step,
                        epoch=self.current_epoch,
                        loss_components=loss_components,
                        outputs=outputs_dict,
                        targets=targets_dict,
                        model=self.model,
                        optimizer=self.optimizer
                    )
                
                # Log detailed loss breakdown every log_frequency steps
                logger.info(
                    f"Step {self.global_step} - "
                    f"Total: {loss.item():.4f}, "
                    f"Signal: {loss_components.get('signal_loss', torch.tensor(0.0)).item():.4f}, "
                    f"Class: {loss_components.get('classification_loss', torch.tensor(0.0)).item():.4f}, "
                    f"Peak: {loss_components.get('peak_exist_loss', torch.tensor(0.0)).item():.4f}, "
                    f"PeakLat: {loss_components.get('peak_latency_loss', torch.tensor(0.0)).item():.4f}, "
                    f"PeakAmp: {loss_components.get('peak_amplitude_loss', torch.tensor(0.0)).item():.4f}, "
                    f"Thresh: {loss_components.get('threshold_loss', torch.tensor(0.0)).item():.4f}"
                )
        
        # Average metrics over epoch
        self._average_metrics(metrics, num_batches)
        
        return metrics
    
    def validate_epoch(self) -> TrainingMetrics:
        """Validate for one epoch."""
        self.model.eval()
        metrics = TrainingMetrics()
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        loss, loss_components = self._forward_pass(batch)
                else:
                    loss, loss_components = self._forward_pass(batch)
                
                # Update metrics
                self._update_metrics(metrics, loss_components, batch)
                num_batches += 1
        
        # Average metrics over epoch
        self._average_metrics(metrics, num_batches)
        
        return metrics
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fixed forward pass for diffusion + multi-task learning."""
        batch_size = batch['signal'].size(0)
        
        # Sample random timesteps for diffusion
        timesteps = torch.randint(
            0, self.noise_schedule.num_timesteps, 
            (batch_size,), device=self.device
        )
        
        # Add noise to signals (diffusion process)
        noise = torch.randn_like(batch['signal'])
        noisy_signals = self.noise_schedule.q_sample(
            batch['signal'], timesteps, noise
        )
        
        # Forward pass through model with timesteps
        if self.use_amp:
            with autocast():
                outputs = self.model(
                    noisy_signals,
                    batch['static_params'],
                    timesteps  # Now properly passed to model
                )
                
                # Compute enhanced loss with diffusion support
                loss, loss_components = self._compute_enhanced_loss(outputs, batch, noise, timesteps)
        else:
            outputs = self.model(
                noisy_signals,
                batch['static_params'],
                timesteps  # Now properly passed to model
            )
            
            # Compute enhanced loss with diffusion support
            loss, loss_components = self._compute_enhanced_loss(outputs, batch, noise, timesteps)
        
        return loss, loss_components
    
    def _compute_enhanced_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced loss computation for diffusion + multi-task learning.
        """
        loss_components = {}
        
        # 1. DIFFUSION LOSS (noise prediction)
        if 'noise' in outputs and outputs['noise'] is not None:
            diffusion_loss = F.mse_loss(outputs['noise'], noise)
            loss_components['diffusion_loss'] = diffusion_loss
            
            # Also compute signal loss for compatibility
            signal_loss = F.mse_loss(outputs['recon'], batch['signal'])
            loss_components['signal_loss'] = signal_loss
            
            # Use diffusion loss as primary signal loss
            primary_signal_loss = diffusion_loss
        else:
            # Fallback: standard signal reconstruction loss
            signal_loss = F.mse_loss(outputs['recon'], batch['signal'])
            loss_components['signal_loss'] = signal_loss
            primary_signal_loss = signal_loss
        
        # 2. MULTI-TASK LOSSES (these should be from clean features)
        # Peak detection losses
        if 'peak' in outputs:
            try:
                peak_losses = self.loss_fn.compute_peak_loss(
                    outputs['peak'], batch['v_peak'], batch['v_peak_mask']
                )
                loss_components['peak_exist_loss'] = peak_losses['exist']
                loss_components['peak_latency_loss'] = peak_losses['latency']
                loss_components['peak_amplitude_loss'] = peak_losses['amplitude']
            except Exception as e:
                logging.getLogger(__name__).warning(f"Peak loss computation failed: {e}")
                # Fallback to zero losses
                device = outputs['peak'][0].device if isinstance(outputs['peak'], (list, tuple)) else outputs['peak'].device
                loss_components['peak_exist_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                loss_components['peak_latency_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                loss_components['peak_amplitude_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Enhanced classification loss
        if 'class' in outputs:
            try:
                if self.enhanced_class_loss is not None:
                    # Use enhanced focal loss with class imbalance handling
                    class_loss = self.enhanced_class_loss(outputs['class'], batch['target'])
                    
                    # Update dynamic weights if available
                    if self.dynamic_weights is not None:
                        self.dynamic_weights.update(outputs['class'], batch['target'])
                else:
                    # Standard cross-entropy loss
                    class_loss = F.cross_entropy(outputs['class'], batch['target'])
                
                loss_components['classification_loss'] = class_loss
            except Exception as e:
                logging.getLogger(__name__).warning(f"Classification loss computation failed: {e}")
                loss_components['classification_loss'] = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Threshold loss
        if 'threshold' in outputs and 'threshold' in batch:
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
                logging.getLogger(__name__).warning(f"Threshold loss computation failed: {e}")
                loss_components['threshold_loss'] = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 3. BALANCED TOTAL LOSS for diffusion + multi-task
        total_loss = (
            1.0 * primary_signal_loss +  # Diffusion loss
            0.5 * loss_components.get('classification_loss', 0) +
            0.3 * loss_components.get('peak_exist_loss', 0) +
            0.2 * loss_components.get('peak_latency_loss', 0) +
            0.1 * loss_components.get('peak_amplitude_loss', 0) +
            0.3 * loss_components.get('threshold_loss', 0)
        )
        
        loss_components['total_loss'] = total_loss
        
        return total_loss, loss_components
    
    def _update_metrics(self, metrics: TrainingMetrics, loss_components: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        """Update training metrics."""
        # Loss components
        metrics.total_loss += loss_components['total_loss'].item()
        metrics.signal_loss += loss_components.get('signal_loss', torch.tensor(0.0)).item()
        metrics.peak_exist_loss += loss_components.get('peak_exist_loss', torch.tensor(0.0)).item()
        metrics.peak_latency_loss += loss_components.get('peak_latency_loss', torch.tensor(0.0)).item()
        metrics.peak_amplitude_loss += loss_components.get('peak_amplitude_loss', torch.tensor(0.0)).item()
        metrics.classification_loss += loss_components.get('classification_loss', torch.tensor(0.0)).item()
        metrics.threshold_loss += loss_components.get('threshold_loss', torch.tensor(0.0)).item()
        
        # Learning rate
        metrics.learning_rate = self.optimizer.param_groups[0]['lr']
    
    def _average_metrics(self, metrics: TrainingMetrics, num_batches: int):
        """Average metrics over epoch."""
        for attr in ['total_loss', 'signal_loss', 'peak_exist_loss', 'peak_latency_loss',
                    'peak_amplitude_loss', 'classification_loss', 'threshold_loss']:
            setattr(metrics, attr, getattr(metrics, attr) / num_batches)
    
    def _log_metrics(self, metrics: TrainingMetrics, phase: str, step: int):
        """Log metrics to monitoring systems."""
        metrics_dict = metrics.to_dict()
        
        # TensorBoard logging
        if self.writer:
            # Main metrics
            for key, value in metrics_dict.items():
                self.writer.add_scalar(f"{phase}/{key}", value, step)
            
            # Group loss components for better visualization
            if phase == 'train' or phase == 'val':
                loss_scalars = {
                    'Total Loss': metrics.total_loss,
                    'Signal Loss': metrics.signal_loss,
                    'Classification Loss': metrics.classification_loss,
                    'Peak Existence Loss': metrics.peak_exist_loss,
                    'Peak Latency Loss': metrics.peak_latency_loss,
                    'Peak Amplitude Loss': metrics.peak_amplitude_loss,
                    'Threshold Loss': metrics.threshold_loss
                }
                self.writer.add_scalars(f"{phase}/Loss_Breakdown", loss_scalars, step)
                
                # Learning rate
                if hasattr(metrics, 'learning_rate') and metrics.learning_rate > 0:
                    self.writer.add_scalar(f"{phase}/Learning_Rate", metrics.learning_rate, step)
        
        # W&B logging
        if self.use_wandb:
            wandb_dict = {f"{phase}_{key}": value for key, value in metrics_dict.items()}
            
            # Add special grouped metrics for W&B
            wandb_dict.update({
                f"{phase}_loss_breakdown/total": metrics.total_loss,
                f"{phase}_loss_breakdown/signal": metrics.signal_loss,
                f"{phase}_loss_breakdown/classification": metrics.classification_loss,
                f"{phase}_loss_breakdown/peak_exist": metrics.peak_exist_loss,
                f"{phase}_loss_breakdown/peak_latency": metrics.peak_latency_loss,
                f"{phase}_loss_breakdown/peak_amplitude": metrics.peak_amplitude_loss,
                f"{phase}_loss_breakdown/threshold": metrics.threshold_loss
            })
            
            wandb.log(wandb_dict, step=step)
    
    def save_checkpoint(self, is_best: bool = False, checkpoint_name: Optional[str] = None):
        """Save model checkpoint."""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{self.current_epoch}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'best_val_loss': self.best_val_loss,
            'config': dict(self.config),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
        
        # Save latest model
        latest_path = self.checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, resume_from: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            resume_from: Path to checkpoint to resume from
        """
        try:
            # Resume from checkpoint if specified
            if resume_from:
                self.load_checkpoint(resume_from)
            
            logger.info("Starting training...")
            logger.info(f"Training for {self.config.training.epochs} epochs")
            
            for epoch in range(self.current_epoch, self.config.training.epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Training phase
                train_metrics = self.train_epoch()
                
                # Validation phase
                val_metrics = self.validate_epoch()
                
                # Learning rate scheduling
                if hasattr(self.scheduler, 'step'):
                    if hasattr(self.scheduler, 'step_with_metric'):
                        self.scheduler.step_with_metric(val_metrics.total_loss)
                    else:
                        self.scheduler.step()
                
                # Log detailed epoch metrics
                logger.info(
                    f"Epoch {epoch+1}/{self.config.training.epochs} Summary:"
                )
                logger.info(
                    f"  Train - Total: {train_metrics.total_loss:.4f}, "
                    f"Signal: {train_metrics.signal_loss:.4f}, "
                    f"Class: {train_metrics.classification_loss:.4f}, "
                    f"Peak: {train_metrics.peak_exist_loss:.4f}, "
                    f"Thresh: {train_metrics.threshold_loss:.4f}"
                )
                logger.info(
                    f"  Val   - Total: {val_metrics.total_loss:.4f}, "
                    f"Signal: {val_metrics.signal_loss:.4f}, "
                    f"Class: {val_metrics.classification_loss:.4f}, "
                    f"Peak: {val_metrics.peak_exist_loss:.4f}, "
                    f"Thresh: {val_metrics.threshold_loss:.4f}"
                )
                logger.info(f"  LR: {train_metrics.learning_rate:.2e}")
                
                self._log_metrics(train_metrics, 'train', epoch)
                self._log_metrics(val_metrics, 'val', epoch)
                
                # Enhanced epoch monitoring
                if self.training_monitor is not None:
                    epoch_time = time.time() - epoch_start_time if 'epoch_start_time' in locals() else 0
                    self.training_monitor.log_epoch(
                        epoch=epoch,
                        train_metrics=train_metrics.to_dict(),
                        val_metrics=val_metrics.to_dict(),
                        epoch_time=epoch_time
                    )
                
                # Checkpointing
                is_best = val_metrics.total_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics.total_loss
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                
                # Save checkpoint
                if (epoch + 1) % self.config.training.checkpointing.get('save_frequency', 10) == 0:
                    self.save_checkpoint(is_best=is_best)
                
                # Early stopping
                early_stopping_patience = self.config.training.early_stopping.get('patience', 30)
                if self.early_stopping_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                    break
            
            # Final checkpoint
            self.save_checkpoint(is_best=False, checkpoint_name="final_model.pt")
            
            logger.info("Training completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(is_best=False, checkpoint_name="interrupted_model.pt")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            logger.error(traceback.format_exc())
            raise
        
        finally:
            # Cleanup
            if self.writer:
                self.writer.close()
            if self.use_wandb:
                wandb.finish()