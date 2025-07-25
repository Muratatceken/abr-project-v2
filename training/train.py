"""
Training Script for ABR Hierarchical U-Net Diffusion Model

This script implements the training pipeline for the Hierarchical U-Net with S4 Encoder
and Transformer Decoder for ABR signal generation using denoising diffusion.

Author: AI Assistant  
Date: July 2024
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import logging
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.hierarchical_unet import HierarchicalUNet
from diffusion.schedule import get_noise_schedule
from diffusion.loss import ABRDiffusionLoss, get_loss_function
from training.dataset import load_ultimate_dataset
from diffusion.sampling import create_sampler

warnings.filterwarnings('ignore')


class ABRTrainer:
    """
    Trainer class for ABR Hierarchical U-Net diffusion model.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_reproducibility()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.noise_schedule = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Logging
        self.writer = None
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Create logs directory
        log_dir = log_config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # TensorBoard
        if log_config.get('use_tensorboard', True):
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
    
    def setup_device(self):
        """Setup computation device."""
        hardware_config = self.config.get('hardware', {})
        device_type = hardware_config.get('device', 'auto')
        
        if device_type == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_type == 'cuda':
            gpu_id = hardware_config.get('gpu_id', 0)
            self.device = torch.device(f'cuda:{gpu_id}')
        else:
            self.device = torch.device('cpu')
        
        self.logger.info(f"Using device: {self.device}")
        
        # Mixed precision
        self.use_amp = hardware_config.get('mixed_precision', True) and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Mixed precision training enabled")
    
    def setup_reproducibility(self):
        """Setup reproducibility settings."""
        repro_config = self.config.get('reproducibility', {})
        seed = repro_config.get('seed', 42)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if repro_config.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif repro_config.get('benchmark', True):
            torch.backends.cudnn.benchmark = True
        
        self.logger.info(f"Reproducibility configured with seed: {seed}")
    
    def setup_data(self):
        """Setup data loaders."""
        data_config = self.config['data']
        
        # Load dataset with stratified patient splits
        full_dataset, train_dataset, val_dataset, test_dataset = load_ultimate_dataset(
            data_path=data_config['dataset_path'],
            train_ratio=data_config['splits']['train_ratio'],
            val_ratio=data_config['splits']['val_ratio'],
            test_ratio=data_config['splits']['test_ratio'],
            random_state=data_config['splits']['random_seed']
        )
        
        # Create data loaders
        dataloader_config = data_config['dataloader']
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=dataloader_config['batch_size'],
            shuffle=dataloader_config['shuffle_train'],
            num_workers=dataloader_config['num_workers'],
            pin_memory=dataloader_config['pin_memory'],
            drop_last=dataloader_config['drop_last']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=dataloader_config['batch_size'],
            shuffle=False,
            num_workers=dataloader_config['num_workers'],
            pin_memory=dataloader_config['pin_memory'],
            drop_last=False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=dataloader_config['batch_size'],
            shuffle=False,
            num_workers=dataloader_config['num_workers'],
            pin_memory=dataloader_config['pin_memory'],
            drop_last=False
        )
        
        # Get class weights for imbalanced data
        if self.config['loss'].get('class_weights') == 'balanced':
            class_weights = full_dataset.get_class_weights('balanced')
            self.class_weights = class_weights.to(self.device)
        else:
            self.class_weights = None
        
        self.logger.info(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
    
    def setup_model(self):
        """Setup model architecture."""
        model_config = self.config['model']['architecture']
        
        self.model = HierarchicalUNet(
            signal_length=model_config['signal_length'],
            static_dim=model_config['static_dim'],
            base_channels=model_config['base_channels'],
            n_levels=model_config['n_levels'],
            n_classes=model_config['n_classes'],
            dropout=model_config['dropout']
        ).to(self.device)
        
        # Model compilation (PyTorch 2.0)
        if self.config['hardware'].get('compile_model', False):
            self.model = torch.compile(self.model)
            self.logger.info("Model compiled with PyTorch 2.0")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    def setup_optimization(self):
        """Setup optimizer and scheduler."""
        train_config = self.config['training']
        opt_config = train_config['optimizer']
        
        # Optimizer
        if opt_config['type'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas'],
                eps=opt_config['eps'],
                amsgrad=opt_config.get('amsgrad', False)
            )
        elif opt_config['type'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas'],
                eps=opt_config['eps']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")
        
        # Scheduler
        sched_config = train_config['scheduler']
        if sched_config['type'] == 'cosine_annealing_warm_restarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config['T_0'],
                T_mult=sched_config['T_mult'],
                eta_min=sched_config['eta_min']
            )
        elif sched_config['type'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=sched_config.get('patience', 10),
                factor=sched_config.get('factor', 0.5)
            )
        
        self.logger.info(f"Optimizer: {opt_config['type']}, Scheduler: {sched_config['type']}")
    
    def setup_diffusion(self):
        """Setup diffusion components."""
        diffusion_config = self.config['diffusion']
        
        # Noise schedule
        schedule_config = diffusion_config['noise_schedule']
        self.noise_schedule = get_noise_schedule(
            schedule_type=schedule_config['type'],
            num_timesteps=schedule_config['num_timesteps'],
            **{k: v for k, v in schedule_config.items() if k not in ['type', 'num_timesteps']}
        )
        
        # Move schedule to device
        self.noise_schedule.betas = self.noise_schedule.betas.to(self.device)
        self.noise_schedule.alphas = self.noise_schedule.alphas.to(self.device)
        self.noise_schedule.alpha_cumprod = self.noise_schedule.alpha_cumprod.to(self.device)
        self.noise_schedule.alpha_cumprod_prev = self.noise_schedule.alpha_cumprod_prev.to(self.device)
        self.noise_schedule.sqrt_alpha_cumprod = self.noise_schedule.sqrt_alpha_cumprod.to(self.device)
        self.noise_schedule.sqrt_one_minus_alpha_cumprod = self.noise_schedule.sqrt_one_minus_alpha_cumprod.to(self.device)
        self.noise_schedule.sqrt_recip_alpha_cumprod = self.noise_schedule.sqrt_recip_alpha_cumprod.to(self.device)
        self.noise_schedule.sqrt_recipm1_alpha_cumprod = self.noise_schedule.sqrt_recipm1_alpha_cumprod.to(self.device)
        self.noise_schedule.posterior_variance = self.noise_schedule.posterior_variance.to(self.device)
        
        self.logger.info(f"Diffusion setup: {schedule_config['type']} schedule with {schedule_config['num_timesteps']} timesteps")
    
    def setup_loss(self):
        """Setup loss function."""
        loss_config = self.config['loss']
        
        self.loss_fn = get_loss_function(loss_config, self.class_weights)
        self.logger.info(f"Loss function: {loss_config['type']}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        
        # Move data to device
        signal = batch['signal'].to(self.device)  # [batch, 200]
        static_params = batch['static'].to(self.device)  # [batch, 4]
        v_peak = batch['v_peak'].to(self.device)  # [batch, 2]
        v_peak_mask = batch['v_peak_mask'].to(self.device)  # [batch, 2]
        target_class = batch['target'].to(self.device)  # [batch]
        
        # Ensure signal has channel dimension
        if signal.dim() == 2:
            signal = signal.unsqueeze(1)  # [batch, 1, 200]
        
        batch_size = signal.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.noise_schedule.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise to signal
        noise = torch.randn_like(signal)
        noisy_signal = self.noise_schedule.q_sample(signal, t, noise)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Model forward pass
            model_output = self.model(noisy_signal, static_params, t)
            
            # Prepare targets
            targets = {
                'signal': signal,
                'noise': noise,
                'v_peak': v_peak,
                'v_peak_mask': v_peak_mask,
                'target': target_class,
                'static_params': static_params
            }
            
            # Compute loss
            total_loss, loss_components = self.loss_fn(model_output, targets, t)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
        
        # Convert losses to float
        losses = {k: v.item() if isinstance(v, torch.Tensor) else v 
                 for k, v in loss_components.items()}
        
        return losses
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single validation step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of losses
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            signal = batch['signal'].to(self.device)
            static_params = batch['static'].to(self.device)
            v_peak = batch['v_peak'].to(self.device)
            v_peak_mask = batch['v_peak_mask'].to(self.device)
            target_class = batch['target'].to(self.device)
            
            # Ensure signal has channel dimension
            if signal.dim() == 2:
                signal = signal.unsqueeze(1)
            
            batch_size = signal.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, self.noise_schedule.num_timesteps, (batch_size,), device=self.device)
            
            # Add noise to signal
            noise = torch.randn_like(signal)
            noisy_signal = self.noise_schedule.q_sample(signal, t, noise)
            
            # Forward pass
            model_output = self.model(noisy_signal, static_params, t)
            
            # Prepare targets
            targets = {
                'signal': signal,
                'noise': noise,
                'v_peak': v_peak,
                'v_peak_mask': v_peak_mask,
                'target': target_class,
                'static_params': static_params
            }
            
            # Compute loss
            total_loss, loss_components = self.loss_fn(model_output, targets, t)
        
        # Convert losses to float
        losses = {k: v.item() if isinstance(v, torch.Tensor) else v 
                 for k, v in loss_components.items()}
        
        return losses
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training losses
        """
        self.model.train()
        
        total_losses = {}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Training step
            losses = self.train_step(batch)
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value
            
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'diff': f"{losses['diffusion_loss']:.4f}",
                'class': f"{losses['classification_loss']:.4f}"
            })
            
            # Log to tensorboard
            if self.writer and batch_idx % 100 == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f'train_batch/{key}', value, self.global_step)
        
        # Calculate average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        return avg_losses
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average validation losses
        """
        self.model.eval()
        
        total_losses = {}
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}")
        
        for batch in pbar:
            # Validation step
            losses = self.validate_step(batch)
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'diff': f"{losses['diffusion_loss']:.4f}",
                'class': f"{losses['classification_loss']:.4f}"
            })
        
        # Calculate average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        return avg_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.config['training']['checkpointing']['save_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        if self.config['training']['checkpointing']['save_last']:
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'last_checkpoint.pth'))
        
        # Save periodic checkpoint
        if epoch % self.config['training']['checkpointing']['save_frequency'] == 0:
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth'))
        
        # Save best checkpoint
        if is_best and self.config['training']['checkpointing']['save_best']:
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_checkpoint.pth'))
            self.logger.info(f"New best checkpoint saved at epoch {epoch}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Setup all components
        self.setup_data()
        self.setup_model()
        self.setup_optimization()
        self.setup_diffusion()
        self.setup_loss()
        
        num_epochs = self.config['training']['epochs']
        early_stopping_patience = self.config['training']['early_stopping']['patience']
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_losses = self.train_epoch(epoch)
            
            # Validation phase
            val_losses = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_losses['total_loss'])
            else:
                self.scheduler.step()
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.logger.info(
                f"Epoch {epoch:03d}: "
                f"Train Loss: {train_losses['total_loss']:.6f}, "
                f"Val Loss: {val_losses['total_loss']:.6f}, "
                f"LR: {current_lr:.2e}"
            )
            
            # TensorBoard logging
            if self.writer:
                for key, value in train_losses.items():
                    self.writer.add_scalar(f'train_epoch/{key}', value, epoch)
                for key, value in val_losses.items():
                    self.writer.add_scalar(f'val_epoch/{key}', value, epoch)
                self.writer.add_scalar('learning_rate', current_lr, epoch)
            
            # Check for best model
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        if self.writer:
            self.writer.close()
        
        self.logger.info("Training completed!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ABR Hierarchical U-Net")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and start training
    trainer = ABRTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main() 