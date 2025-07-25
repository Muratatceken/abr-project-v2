#!/usr/bin/env python3
"""
Enhanced ABR Training Pipeline

Comprehensive training script for the ProfessionalHierarchicalUNet model with:
- Multi-task learning (signal reconstruction, peak prediction, classification)
- Classifier-free guidance (CFG) support
- FiLM-based conditioning with dropout
- Class imbalance handling
- Mixed precision training
- Advanced scheduling and optimization

Author: AI Assistant
Date: January 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import joblib
import argparse
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import model and components
from models.hierarchical_unet import ProfessionalHierarchicalUNet
from models.blocks.film import CFGWrapper
from diffusion.loss import ABRDiffusionLoss, create_class_weights
from training.evaluation import ABREvaluator


class ABRDataset(Dataset):
    """
    Enhanced ABR Dataset with proper collation and masking support.
    """
    
    def __init__(
        self, 
        data_list: List[Dict], 
        mode: str = 'train',
        augment: bool = False,
        cfg_dropout_prob: float = 0.1
    ):
        """
        Initialize ABR Dataset.
        
        Args:
            data_list: List of data samples
            mode: Dataset mode ('train', 'val', 'test')
            augment: Whether to apply data augmentation
            cfg_dropout_prob: Probability of dropping conditioning for CFG
        """
        self.data = data_list
        self.mode = mode
        self.augment = augment
        self.cfg_dropout_prob = cfg_dropout_prob
        
        # Compute class statistics
        self.class_counts = defaultdict(int)
        for sample in self.data:
            self.class_counts[sample['target']] += 1
        
        print(f"{mode.upper()} Dataset: {len(self.data)} samples")
        print(f"Class distribution: {dict(self.class_counts)}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        # Extract features
        static_params = torch.FloatTensor(sample['static_params'])  # [4]
        signal = torch.FloatTensor(sample['signal']).unsqueeze(0)   # [1, 200]
        v_peak = torch.FloatTensor(sample['v_peak'])                # [2] - latency, amplitude
        v_peak_mask = torch.BoolTensor(sample['v_peak_mask'])       # [2] - validity mask
        target = torch.LongTensor([sample['target']])[0]            # scalar
        
        # Data augmentation for training
        if self.augment and self.mode == 'train':
            signal = self._augment_signal(signal)
            static_params = self._augment_static_params(static_params)
        
        # CFG conditioning dropout during training
        force_uncond = False
        if self.mode == 'train' and torch.rand(1).item() < self.cfg_dropout_prob:
            force_uncond = True
        
        return {
            'patient_id': sample.get('patient_id', idx),
            'static_params': static_params,
            'signal': signal,
            'v_peak': v_peak,
            'v_peak_mask': v_peak_mask,
            'target': target,
            'force_uncond': force_uncond
        }
    
    def _augment_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply signal augmentation."""
        # Add small amount of noise
        if torch.rand(1).item() < 0.3:
            noise_scale = 0.02
            signal = signal + torch.randn_like(signal) * noise_scale
        
        # Time shift (small)
        if torch.rand(1).item() < 0.2:
            shift = torch.randint(-3, 4, (1,)).item()
            if shift != 0:
                signal = torch.roll(signal, shift, dims=-1)
        
        # Amplitude scaling
        if torch.rand(1).item() < 0.2:
            scale = 0.95 + torch.rand(1).item() * (1.05 - 0.95)
            signal = signal * scale
        
        return signal
    
    def _augment_static_params(self, static_params: torch.Tensor) -> torch.Tensor:
        """Apply static parameter augmentation."""
        # Small random perturbations (data is already normalized)
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(static_params) * 0.05
            static_params = static_params + noise
        
        return static_params


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for ABR data.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data dictionary
    """
    # Stack all tensors
    patient_ids = [sample['patient_id'] for sample in batch]
    static_params = torch.stack([sample['static_params'] for sample in batch])      # [B, 4]
    signals = torch.stack([sample['signal'] for sample in batch])                   # [B, 1, 200]
    v_peaks = torch.stack([sample['v_peak'] for sample in batch])                   # [B, 2]
    v_peak_masks = torch.stack([sample['v_peak_mask'] for sample in batch])         # [B, 2]
    targets = torch.stack([sample['target'] for sample in batch])                   # [B]
    force_uncond = torch.tensor([sample['force_uncond'] for sample in batch])       # [B]
    
    return {
        'patient_ids': patient_ids,
        'static_params': static_params,
        'signal': signals,
        'v_peak': v_peaks,
        'v_peak_mask': v_peak_masks,
        'target': targets,
        'force_uncond': force_uncond
    }


class EnhancedABRLoss(nn.Module):
    """Legacy wrapper for backward compatibility - redirects to ABRDiffusionLoss"""
    
    def __init__(self, n_classes: int = 5, use_focal_loss: bool = False, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss_fn = ABRDiffusionLoss(
            n_classes=n_classes,
            use_focal_loss=use_focal_loss,
            class_weights=class_weights
        )
    
    def forward(self, outputs: Dict[str, Any], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.loss_fn(outputs, batch)


class ABRTrainer:
    """
    Enhanced trainer for ABR model with comprehensive features.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup loss function
        self.setup_loss_function()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Setup mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.epoch = 0
        self.best_val_f1 = 0.0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        
        # Early stopping
        self.patience = config.get('patience', 15)
        self.patience_counter = 0
        
        # CFG wrapper
        if hasattr(model, 'cfg_wrapper') and model.cfg_wrapper is not None:
            self.cfg_wrapper = model.cfg_wrapper
        else:
            self.cfg_wrapper = None
    
    def setup_loss_function(self):
        """Setup loss function with class weights."""
        # Create loss function with class weights
        if self.config.get('use_class_weights', True):
            targets = [sample['target'] for sample in self.train_loader.dataset.data]
            class_weights = create_class_weights(targets, self.config.get('num_classes', 5), self.device)
        else:
            class_weights = None
        
        self.loss_fn = ABRDiffusionLoss(
            n_classes=self.config.get('num_classes', 5),
            class_weights=class_weights,
            use_focal_loss=self.config.get('use_focal_loss', False),
            peak_loss_type=self.config.get('loss', {}).get('peak_loss_type', 'mse'),
            use_log_threshold=self.config.get('loss', {}).get('use_log_threshold', True),
            device=self.device
        )
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler_type = self.config.get('scheduler', 'cosine_warm_restarts')
        
        if scheduler_type == 'cosine_warm_restarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('T_0', 10),
                T_mult=self.config.get('T_mult', 2),
                eta_min=self.config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=True
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 20),
                gamma=self.config.get('gamma', 0.5)
            )
        else:
            self.scheduler = None
    
    def setup_logging(self):
        """Setup logging and monitoring."""
        # Create output directory
        self.output_dir = self.config.get('output_dir', f'runs/abr_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir)
        
        # Weights & Biases (optional)
        if self.config.get('use_wandb', False):
            if WANDB_AVAILABLE:
                wandb.init(
                    project=self.config.get('wandb_project', 'abr-training'),
                    config=self.config,
                    name=self.config.get('experiment_name', f'abr_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                )
            else:
                self.logger.warning("Weights & Biases requested but not installed. Install with: pip install wandb")
        
        # Save config
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        # Apply curriculum learning weights
        curriculum_weights = self.get_curriculum_weights(self.epoch)
        self.loss_fn.update_loss_weights(curriculum_weights)
        
        # Log curriculum weights
        if self.epoch % 5 == 0:  # Log every 5 epochs
            self.logger.info(f"Epoch {self.epoch} curriculum weights: {curriculum_weights}")
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for key in ['static_params', 'signal', 'v_peak', 'v_peak_mask', 'target']:
                if key in batch:
                    batch[key] = batch[key].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.get('use_amp', False)):
                outputs = self.model(batch['signal'], batch['static_params'])
                loss, loss_dict = self.loss_fn(outputs, batch)
            
            # Backward pass
            if self.config.get('use_amp', False):
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip_norm', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('gradient_clip_norm', 1.0)
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('gradient_clip_norm', 1.0)
                    )
                
                self.optimizer.step()
            
            # Update metrics
            for key, value in loss_dict.items():
                epoch_metrics[key] += value.item()
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.config.get('log_every', 10) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'signal_w': f'{curriculum_weights["signal"]:.2f}',
                    'class_w': f'{curriculum_weights["classification"]:.2f}'
                })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch with DDIM sampling."""
        self.model.eval()
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        # Initialize DDIM sampler for validation
        from diffusion.sampling import DDIMSampler
        from diffusion.schedule import get_noise_schedule
        
        # Create noise schedule for DDIM sampling
        noise_schedule = get_noise_schedule('cosine', num_timesteps=1000)
        ddim_sampler = DDIMSampler(noise_schedule, eta=0.0)  # Deterministic sampling
        
        # Set deterministic seed for reproducible validation
        torch.manual_seed(42)
        
        progress_bar = tqdm(self.val_loader, desc=f'Validation {self.epoch}')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                for key in ['static_params', 'signal', 'v_peak', 'v_peak_mask', 'target']:
                    if key in batch:
                        batch[key] = batch[key].to(self.device)
                
                batch_size = batch['signal'].size(0)
                signal_length = batch['signal'].size(-1)
                
                # Generate samples using DDIM for realistic evaluation
                generated_signals = ddim_sampler.sample(
                    model=self.model,
                    shape=(batch_size, 1, signal_length),
                    static_params=batch['static_params'],
                    device=self.device,
                    num_steps=50,  # Faster sampling with fewer steps
                    progress=False
                )
                
                # Get predictions from generated signals
                outputs = self.model(generated_signals, batch['static_params'])
                
                # Also evaluate direct forward pass for comparison
                direct_outputs = self.model(batch['signal'], batch['static_params'])
                
                # Compute loss on both generated and direct outputs
                gen_loss, gen_loss_dict = self.loss_fn(outputs, batch)
                direct_loss, direct_loss_dict = self.loss_fn(direct_outputs, batch)
                
                # Update metrics with both types
                for key, value in gen_loss_dict.items():
                    epoch_metrics[f'gen_{key}'] += value.item()
                
                for key, value in direct_loss_dict.items():
                    epoch_metrics[f'direct_{key}'] += value.item()
                
                # Store generated signals for evaluation
                if hasattr(self, 'evaluator'):
                    # Create batch with generated signals for evaluation
                    eval_batch = batch.copy()
                    eval_batch['signal'] = generated_signals
                    self.evaluator.evaluate_batch(eval_batch, compute_loss=False)
                
                num_batches += 1
                
                # Update progress bar with key metrics
                progress_bar.set_postfix({
                    'gen_loss': f'{gen_loss.item():.4f}',
                    'direct_loss': f'{direct_loss.item():.4f}',
                    'signal_diff': f'{torch.mean((generated_signals - batch["signal"])**2).item():.4f}'
                })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Reset random seed
        torch.manual_seed(torch.initial_seed())
        
        return dict(epoch_metrics)
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        num_epochs = self.config.get('num_epochs', 100)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - start_time
            self.log_metrics(train_metrics, val_metrics, epoch_time)
            
            # Save best model
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.save_checkpoint('best_f1.pth', val_metrics)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_checkpoint('best_loss.pth', val_metrics)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', val_metrics)
        
        self.logger.info("Training completed!")
        self.writer.close()
        
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.finish()
    
    def log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """Log training metrics."""
        # Console logging
        self.logger.info(
            f"Epoch {self.epoch + 1:3d} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} | "
            f"Val Loss: {val_metrics['total_loss']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # TensorBoard logging with enhanced diagnostics
        if self.writer:
            # Log curriculum weights
            for weight_name, weight_value in curriculum_weights.items():
                self.writer.add_scalar(f'curriculum/{weight_name}', weight_value, self.epoch)
            
            # Log training metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, self.epoch)
            
            # Log validation metrics
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, self.epoch)
            
            # Log learning rate
            self.writer.add_scalar('optimizer/learning_rate', self.optimizer.param_groups[0]['lr'], self.epoch)
            
            # Enhanced diagnostic logging every few epochs
            if hasattr(self, 'evaluator') and self.epoch % 5 == 0:
                try:
                    self.evaluator.log_to_tensorboard(self.writer, self.epoch)
                except Exception as e:
                    self.logger.warning(f"Failed to log diagnostics to TensorBoard: {str(e)}")
        
        # Weights & Biases logging
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            log_dict = {
                'epoch': self.epoch,
                'train_loss': train_metrics.get('total_loss', 0),
                'val_loss': val_metrics.get('gen_total_loss', val_metrics.get('total_loss', 0)),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
                **{f'curriculum_{k}': v for k, v in curriculum_weights.items()}
            }
            
            # Add diagnostic visualizations to wandb
            if hasattr(self, 'evaluator') and self.epoch % 10 == 0:
                try:
                    visualizations = self.evaluator.create_diagnostic_visualizations(self.epoch)
                    for viz_name, viz_data in visualizations.items():
                        if viz_data:
                            import wandb
                            from PIL import Image
                            from io import BytesIO
                            
                            image = Image.open(BytesIO(viz_data))
                            log_dict[f'diagnostics/{viz_name}'] = wandb.Image(image)
                except Exception as e:
                    self.logger.warning(f"Failed to log diagnostics to W&B: {str(e)}")
            
            wandb.log(log_dict)
    
    def save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_f1': self.best_val_f1,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'metrics': metrics
        }
        
        torch.save(checkpoint, os.path.join(self.output_dir, filename))
        self.logger.info(f"Checkpoint saved: {filename}")

    def get_curriculum_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get curriculum learning weights based on current epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Dictionary of loss component weights
        """
        base_weights = self.config.get('loss', {}).get('loss_weights', {
            'signal': 1.0,
            'peak_exist': 0.5,
            'peak_latency': 1.0,
            'peak_amplitude': 1.0,
            'classification': 1.5,
            'threshold': 0.8
        })
        
        # Check if curriculum learning is enabled
        curriculum_config = self.config.get('advanced', {}).get('curriculum', {})
        if not curriculum_config.get('enabled', False):
            return base_weights
        
        def ramp(epoch: int, start_epoch: int, ramp_epochs: int = 5) -> float:
            """Compute ramp-up weight for curriculum learning."""
            if epoch < start_epoch:
                return 0.0
            elif epoch >= start_epoch + ramp_epochs:
                return 1.0
            else:
                return (epoch - start_epoch + 1) / ramp_epochs
        
        ramp_epochs = curriculum_config.get('ramp_epochs', 5)
        
        # Apply curriculum schedule
        curriculum_weights = {
            'signal': base_weights.get('signal', 1.0),  # Always active
            'peak_exist': base_weights.get('peak_exist', 0.5) * ramp(
                epoch, curriculum_config.get('peak_start', 5), ramp_epochs
            ),
            'peak_latency': base_weights.get('peak_latency', 1.0) * ramp(
                epoch, curriculum_config.get('peak_start', 5), ramp_epochs
            ),
            'peak_amplitude': base_weights.get('peak_amplitude', 1.0) * ramp(
                epoch, curriculum_config.get('peak_start', 5), ramp_epochs
            ),
            'classification': base_weights.get('classification', 1.5) * ramp(
                epoch, curriculum_config.get('class_start', 3), ramp_epochs
            ),
            'threshold': base_weights.get('threshold', 0.8) * ramp(
                epoch, curriculum_config.get('threshold_start', 10), ramp_epochs
            )
        }
        
        return curriculum_weights


def load_dataset(data_path: str, valid_peaks_only: bool = False) -> Tuple[List[Dict], Any, Any]:
    """Load and prepare dataset."""
    print(f"Loading dataset from {data_path}...")
    
    # Load the ultimate dataset
    data = joblib.load(data_path)
    processed_data = data['data']
    scaler = data['scaler']
    label_encoder = data['label_encoder']
    
    print(f"Loaded {len(processed_data)} samples")
    print(f"Classes: {label_encoder.classes_}")
    
    # Filter for valid peaks if requested
    if valid_peaks_only:
        valid_samples = [
            sample for sample in processed_data
            if sample['v_peak_mask'][0] and sample['v_peak_mask'][1]
        ]
        print(f"Filtered to {len(valid_samples)} samples with valid V peaks")
        processed_data = valid_samples
    
    return processed_data, scaler, label_encoder


def create_data_loaders(
    data: List[Dict], 
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    
    # Stratified train-val split
    targets = [sample['target'] for sample in data]
    train_indices, val_indices = train_test_split(
        range(len(data)),
        test_size=config.get('val_split', 0.2),
        stratify=targets,
        random_state=config.get('random_seed', 42)
    )
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    
    # Create datasets
    train_dataset = ABRDataset(
        train_data, 
        mode='train', 
        augment=config.get('augment', True),
        cfg_dropout_prob=config.get('cfg_dropout_prob', 0.1)
    )
    val_dataset = ABRDataset(val_data, mode='val', augment=False)
    
    # Create samplers for balanced training (optional)
    train_sampler = None
    if config.get('use_balanced_sampler', False):
        train_targets = [sample['target'] for sample in train_data]
        class_counts = np.bincount(train_targets)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[target] for target in train_targets]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create the enhanced ABR model."""
    
    model = ProfessionalHierarchicalUNet(
        input_channels=config.get('input_channels', 1),
        static_dim=config.get('static_dim', 4),
        base_channels=config.get('base_channels', 64),
        n_levels=config.get('n_levels', 4),
        sequence_length=config.get('sequence_length', 200),
        signal_length=config.get('signal_length', 200),
        num_classes=config.get('num_classes', 5),
        
        # Enhanced features
        num_transformer_layers=config.get('num_transformer_layers', 3),
        use_cross_attention=config.get('use_cross_attention', True),
        use_positional_encoding=config.get('use_positional_encoding', True),
        film_dropout=config.get('film_dropout', 0.15),
        use_cfg=config.get('use_cfg', True),
        
        # Additional parameters
        dropout=config.get('dropout', 0.1),
        use_attention_heads=config.get('use_attention_heads', True),
        predict_uncertainty=config.get('predict_uncertainty', False)
    )
    
    return model


def main(args=None):
    """Main training function."""
    if args is None:
        parser = argparse.ArgumentParser(description='Enhanced ABR Model Training')
    
        # Data arguments
        parser.add_argument('--data_path', type=str, 
                           default='data/processed/ultimate_dataset.pkl',
                           help='Path to dataset file')
        parser.add_argument('--valid_peaks_only', action='store_true',
                           help='Train only on samples with valid V peaks')
        
        # Model arguments
        parser.add_argument('--base_channels', type=int, default=64,
                           help='Base number of channels')
        parser.add_argument('--n_levels', type=int, default=4,
                           help='Number of U-Net levels')
        parser.add_argument('--num_transformer_layers', type=int, default=3,
                           help='Number of transformer layers')
        parser.add_argument('--use_cross_attention', action='store_true', default=True,
                           help='Use cross-attention')
        parser.add_argument('--film_dropout', type=float, default=0.15,
                           help='FiLM dropout rate')
        
        # Training arguments
        parser.add_argument('--batch_size', type=int, default=32,
                           help='Batch size')
        parser.add_argument('--learning_rate', type=float, default=1e-4,
                           help='Learning rate')
        parser.add_argument('--num_epochs', type=int, default=100,
                           help='Number of epochs')
        parser.add_argument('--weight_decay', type=float, default=0.01,
                           help='Weight decay')
        parser.add_argument('--use_amp', action='store_true', default=True,
                           help='Use mixed precision training')
        parser.add_argument('--patience', type=int, default=15,
                           help='Early stopping patience')
        
        # Loss arguments
        parser.add_argument('--use_focal_loss', action='store_true',
                           help='Use focal loss for classification')
        parser.add_argument('--use_class_weights', action='store_true', default=True,
                           help='Use class weights for imbalanced data')
        
        # Augmentation and sampling
        parser.add_argument('--augment', action='store_true', default=True,
                           help='Use data augmentation')
        parser.add_argument('--use_balanced_sampler', action='store_true',
                           help='Use balanced sampling')
        parser.add_argument('--cfg_dropout_prob', type=float, default=0.1,
                           help='CFG dropout probability')
        
        # Output and logging
        parser.add_argument('--output_dir', type=str, default=None,
                           help='Output directory')
        parser.add_argument('--use_wandb', action='store_true',
                           help='Use Weights & Biases logging')
        parser.add_argument('--wandb_project', type=str, default='abr-training',
                           help='Weights & Biases project name')
        parser.add_argument('--experiment_name', type=str, default=None,
                           help='Experiment name')
        
        # System arguments
        parser.add_argument('--num_workers', type=int, default=4,
                           help='Number of data loader workers')
        parser.add_argument('--random_seed', type=int, default=42,
                           help='Random seed')
        
        args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert args to config dict
    config = vars(args)
    
    # Load dataset
    data, scaler, label_encoder = load_dataset(
        config['data_path'], 
        config['valid_peaks_only']
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(data, config)
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = ABRTrainer(model, train_loader, val_loader, config, device)
    
    # Start training
    trainer.train()


def run_cross_validation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run cross-validation training with StratifiedGroupKFold.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Dictionary containing cross-validation results
    """
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold
    
    # Load dataset
    data, scaler, label_encoder = load_dataset(
        config['data_path'], 
        config.get('valid_peaks_only', False)
    )
    
    # Extract targets and groups
    targets = [sample['target'] for sample in data]
    groups = [sample.get('patient_id', f'patient_{i}') for i, sample in enumerate(data)]
    
    # Setup cross-validation strategy
    cv_strategy = config.get('validation', {}).get('cv_strategy', 'StratifiedGroupKFold')
    n_folds = config.get('validation', {}).get('cv_folds', 5)
    
    if cv_strategy == 'StratifiedGroupKFold':
        cv_splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_args = (range(len(data)), targets, groups)
    elif cv_strategy == 'StratifiedKFold':
        cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_args = (range(len(data)), targets)
    elif cv_strategy == 'GroupKFold':
        cv_splitter = GroupKFold(n_splits=n_folds)
        split_args = (range(len(data)), targets, groups)
    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}")
    
    # Cross-validation results storage
    cv_results = {
        'fold_results': [],
        'fold_models': [],
        'mean_metrics': {},
        'std_metrics': {}
    }
    
    print(f"Starting {n_folds}-fold cross-validation using {cv_strategy}")
    print(f"Total samples: {len(data)}")
    print(f"Class distribution: {dict(zip(*np.unique(targets, return_counts=True)))}")
    
    # Run cross-validation
    for fold_idx, (train_indices, val_indices) in enumerate(cv_splitter.split(*split_args)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Create fold-specific data splits
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Create data loaders for this fold
        train_loader, val_loader = create_data_loaders_from_data(train_data, val_data, config)
        
        # Create model for this fold
        model = create_model(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create fold-specific config
        fold_config = config.copy()
        fold_config['output_dir'] = os.path.join(
            config.get('output_dir', 'outputs'), 
            f'fold_{fold_idx + 1}'
        )
        fold_config['experiment_name'] = f"{config.get('experiment_name', 'cv_experiment')}_fold_{fold_idx + 1}"
        
        # Create trainer for this fold
        trainer = ABRTrainer(model, train_loader, val_loader, fold_config, device)
        
        # Train this fold
        try:
            trainer.train()
            
            # Get best metrics for this fold
            fold_results = {
                'fold': fold_idx + 1,
                'best_epoch': trainer.best_epoch,
                'best_f1_macro': trainer.best_f1_macro,
                'final_metrics': trainer.best_metrics
            }
            
            cv_results['fold_results'].append(fold_results)
            
            # Save model if requested
            if config.get('validation', {}).get('cv_save_all_folds', False):
                cv_results['fold_models'].append(trainer.model.state_dict())
            
            print(f"Fold {fold_idx + 1} completed - Best F1 Macro: {trainer.best_f1_macro:.4f}")
            
        except Exception as e:
            print(f"Error in fold {fold_idx + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute cross-validation statistics
    if cv_results['fold_results']:
        # Extract metrics from all folds
        all_metrics = {}
        for fold_result in cv_results['fold_results']:
            for metric_name, metric_value in fold_result['final_metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)
        
        # Compute mean and std for each metric
        for metric_name, values in all_metrics.items():
            cv_results['mean_metrics'][metric_name] = np.mean(values)
            cv_results['std_metrics'][metric_name] = np.std(values)
        
        # Print cross-validation summary
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Strategy: {cv_strategy}")
        print(f"Folds completed: {len(cv_results['fold_results'])}/{n_folds}")
        print(f"\nKey Metrics (Mean ± Std):")
        for metric in ['f1_macro', 'f1_weighted', 'balanced_accuracy', 'total_loss']:
            if metric in cv_results['mean_metrics']:
                mean_val = cv_results['mean_metrics'][metric]
                std_val = cv_results['std_metrics'][metric]
                print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save cross-validation results
        cv_output_dir = config.get('output_dir', 'outputs')
        os.makedirs(cv_output_dir, exist_ok=True)
        
        cv_results_file = os.path.join(cv_output_dir, 'cv_results.json')
        with open(cv_results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {
                'fold_results': cv_results['fold_results'],
                'mean_metrics': {k: float(v) for k, v in cv_results['mean_metrics'].items()},
                'std_metrics': {k: float(v) for k, v in cv_results['std_metrics'].items()},
                'config': config
            }
            json.dump(json_results, f, indent=2)
        
        print(f"\nCross-validation results saved to: {cv_results_file}")
    
    return cv_results


def create_data_loaders_from_data(
    train_data: List[Dict], 
    val_data: List[Dict], 
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders from pre-split data (used in cross-validation).
    
    Args:
        train_data: Training data samples
        val_data: Validation data samples
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    # Create datasets
    train_dataset = ABRDataset(
        train_data, 
        mode='train', 
        augment=config.get('augment', True),
        cfg_dropout_prob=config.get('cfg_dropout_prob', 0.1)
    )
    val_dataset = ABRDataset(val_data, mode='val', augment=False)
    
    # Create samplers for balanced training (optional)
    train_sampler = None
    if config.get('use_balanced_sampler', False):
        train_targets = [sample['target'] for sample in train_data]
        class_counts = np.bincount(train_targets)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[target] for target in train_targets]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    main() 