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
from torch.utils.data import Dataset, DataLoader
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

try:
    from .visualization import TrainingVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    try:
        from training.visualization import TrainingVisualizer
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False

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


# Removed: collate_fn is now in data.dataset.py


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
        
        # Setup memory optimization
        self.setup_memory_optimization()
        
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
            
        # Setup visualization
        self.setup_visualization()
    
    def setup_loss_function(self):
        """Setup loss function with class weights."""
        # Create loss function with class weights
        if self.config.get('use_class_weights', True):
            # Handle both Subset and regular dataset objects
            dataset = self.train_loader.dataset
            if hasattr(dataset, 'dataset'):
                # This is a Subset, get the underlying dataset
                base_dataset = dataset.dataset
                indices = dataset.indices
                targets = [base_dataset[i]['target'].item() if torch.is_tensor(base_dataset[i]['target']) 
                          else base_dataset[i]['target'] for i in indices]
            elif hasattr(dataset, 'data'):
                # This is a regular dataset with data attribute
                targets = [sample['target'] for sample in dataset.data]
            else:
                # Fallback: iterate through the dataset
                targets = []
                for i in range(len(dataset)):
                    sample = dataset[i]
                    target = sample['target'].item() if torch.is_tensor(sample['target']) else sample['target']
                    targets.append(target)
            
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
        # Optimizer with configurable parameters
        optimizer_config = self.config.get('optimizer', {})
        
        # Ensure parameters are correct types
        learning_rate = float(self.config.get('learning_rate', 1e-4))
        weight_decay = float(self.config.get('weight_decay', 0.01))
        betas = optimizer_config.get('betas', [0.9, 0.999])
        eps = float(optimizer_config.get('eps', 1e-8))
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
        
        # Learning rate scheduler with configurable parameters
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine_warm_restarts')
        
        if scheduler_type == 'cosine_warm_restarts' or scheduler_type == 'cosine_annealing_warm_restarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(scheduler_config.get('T_0', 10)),
                T_mult=int(scheduler_config.get('T_mult', 2)),
                eta_min=float(scheduler_config.get('eta_min', 1e-6))
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
    
    def setup_memory_optimization(self):
        """Setup memory optimization features."""
        memory_config = self.config.get('memory', {})
        
        # Enable memory efficient attention if available
        if memory_config.get('use_memory_efficient_attention', False):
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                print("✓ Enabled Flash Attention for memory efficiency")
            except:
                print("⚠️ Flash Attention not available")
        
        # Set memory management options
        if torch.cuda.is_available():
            # Enable memory pool for faster allocation
            torch.cuda.empty_cache()
            # Set memory fraction if specified
            if 'memory_fraction' in memory_config:
                torch.cuda.set_per_process_memory_fraction(memory_config['memory_fraction'])
        
        # Store memory optimization settings
        self.clear_cache_every = memory_config.get('clear_cache_every', 100)
        self.memory_efficient = memory_config.get('use_memory_efficient_attention', False)
    
    def setup_logging(self):
        """Setup logging and monitoring."""
        # Create output directory
        output_dir = self.config.get('output_dir')
        if output_dir is None:
            output_dir = f'runs/abr_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.output_dir = output_dir
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
    
    def setup_visualization(self):
        """Setup training visualization."""
        visualization_config = self.config.get('visualization', {})
        
        if VISUALIZATION_AVAILABLE and visualization_config.get('enabled', False):
            plot_dir = visualization_config.get('plot_dir', 'plots')
            experiment_name = self.config.get('experiment_name', 'abr_training')
            
            self.visualizer = TrainingVisualizer(
                plot_dir=plot_dir,
                experiment_name=experiment_name
            )
            
            self.plot_every = visualization_config.get('plot_every', 5)
            self.logger.info(f"Visualization enabled: plots will be saved to {plot_dir}")
        else:
            self.visualizer = None
            if not VISUALIZATION_AVAILABLE:
                self.logger.warning("Visualization not available - missing dependencies")
            else:
                self.logger.info("Visualization disabled")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        # Apply curriculum learning weights
        curriculum_weights = self.get_curriculum_weights(self.epoch)
        self.curriculum_weights = curriculum_weights  # Store for logging
        self.loss_fn.update_loss_weights(curriculum_weights)
        
        # Log curriculum weights
        if self.epoch % 5 == 0:  # Log every 5 epochs
            self.logger.info(f"Epoch {self.epoch} curriculum weights: {curriculum_weights}")
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        # Gradient accumulation setup
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for key in ['static_params', 'signal', 'v_peak', 'v_peak_mask', 'target']:
                if key in batch:
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.get('use_amp', False)):
                outputs = self.model(batch['signal'], batch['static_params'])
                loss, loss_dict = self.loss_fn(outputs, batch)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass
            if self.config.get('use_amp', False):
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step only after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.config.get('use_amp', False):
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
                    # Gradient clipping
                    if self.config.get('gradient_clip_norm', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.get('gradient_clip_norm', 1.0)
                        )
                    
                    self.optimizer.step()
                
                # Zero gradients after optimizer step
                self.optimizer.zero_grad()
            
            # Update metrics
            for key, value in loss_dict.items():
                epoch_metrics[key] += value.item()
            num_batches += 1
            
            # Memory optimization: clear cache periodically
            if batch_idx % self.clear_cache_every == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
        """Validate for one epoch with optional DDIM sampling."""
        self.model.eval()
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        # Check if we should do fast validation
        validation_config = self.config.get('validation', {})
        fast_mode = validation_config.get('fast_mode', False)
        skip_generation = validation_config.get('skip_generation', False)
        full_validation_every = validation_config.get('full_validation_every', 10)
        ddim_steps = validation_config.get('ddim_steps', 50)
        
        # Decide whether to do full validation with sampling
        do_full_validation = not fast_mode or (self.epoch % full_validation_every == 0)
        do_generation = do_full_validation and not skip_generation
        
        ddim_sampler = None
        if do_generation:
            # Initialize DDIM sampler for validation
            from diffusion.sampling import DDIMSampler
            from diffusion.schedule import get_noise_schedule
            
            # Create noise schedule for DDIM sampling
            noise_schedule = get_noise_schedule('cosine', num_timesteps=1000)
            ddim_sampler = DDIMSampler(noise_schedule, eta=0.0)  # Deterministic sampling
        
        # Set deterministic seed for reproducible validation
        torch.manual_seed(42)
        
        val_type = "Full" if do_generation else "Fast"
        progress_bar = tqdm(self.val_loader, desc=f'{val_type} Validation {self.epoch}')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                for key in ['static_params', 'signal', 'v_peak', 'v_peak_mask', 'target']:
                    if key in batch:
                        batch[key] = batch[key].to(self.device)
                
                batch_size = batch['signal'].size(0)
                signal_length = batch['signal'].size(-1)
                
                # Always do direct forward pass
                direct_outputs = self.model(batch['signal'], batch['static_params'])
                direct_loss, direct_loss_dict = self.loss_fn(direct_outputs, batch)
                
                # Update metrics with direct outputs
                for key, value in direct_loss_dict.items():
                    epoch_metrics[f'direct_{key}'] += value.item()
                
                # Conditionally do generation-based evaluation
                if do_generation and ddim_sampler is not None:
                    # Generate samples using DDIM for realistic evaluation
                    generated_signals = ddim_sampler.sample(
                        model=self.model,
                        shape=(batch_size, 1, signal_length),
                        static_params=batch['static_params'],
                        device=self.device,
                        num_steps=ddim_steps,  # Use configurable steps
                        progress=False
                    )
                    
                    # Get predictions from generated signals
                    outputs = self.model(generated_signals, batch['static_params'])
                    gen_loss, gen_loss_dict = self.loss_fn(outputs, batch)
                    
                    # Update metrics with generated outputs
                    for key, value in gen_loss_dict.items():
                        epoch_metrics[f'gen_{key}'] += value.item()
                    
                    # Update progress bar with both metrics
                    progress_bar.set_postfix({
                        'gen_loss': f'{gen_loss.item():.4f}',
                        'direct_loss': f'{direct_loss.item():.4f}',
                        'signal_diff': f'{torch.mean((generated_signals - batch["signal"])**2).item():.4f}'
                    })
                else:
                    # Update progress bar with only direct metrics
                    progress_bar.set_postfix({
                        'direct_loss': f'{direct_loss.item():.4f}',
                        'mode': 'fast'
                    })
                
                # Store generated signals for evaluation (only if we generated them)
                if hasattr(self, 'evaluator') and do_generation and 'generated_signals' in locals():
                    # Create batch with generated signals for evaluation
                    eval_batch = batch.copy()
                    eval_batch['signal'] = generated_signals
                    self.evaluator.evaluate_batch(eval_batch, compute_loss=False)
                
                num_batches += 1
        
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
            
            # Extract key metrics with fallbacks
            val_loss = val_metrics.get('gen_total_loss', val_metrics.get('direct_total_loss', float('inf')))
            val_f1 = val_metrics.get('gen_f1_macro', val_metrics.get('direct_f1_macro', 0.0))
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - start_time
            self.log_metrics(train_metrics, val_metrics, epoch_time)
            
            # Create visualization plots
            if hasattr(self, 'visualizer') and self.visualizer is not None:
                # Update metrics in visualizer
                current_lr = self.optimizer.param_groups[0]['lr']
                self.visualizer.update_metrics(self.epoch, train_metrics, val_metrics, current_lr)
                
                # Create plots periodically
                if (self.epoch + 1) % getattr(self, 'plot_every', 5) == 0:
                    try:
                        created_plots = self.visualizer.create_all_plots(save=True)
                        if created_plots:
                            self.logger.info(f"Created {len(created_plots)} visualization plots")
                    except Exception as e:
                        self.logger.warning(f"Failed to create visualization plots: {str(e)}")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.save_checkpoint('best_f1.pth', val_metrics)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_loss.pth', val_metrics)
            
            # Save milestone checkpoints
            milestone_epochs = self.config.get('save_criteria', {}).get('milestone_epochs', [])
            if (epoch + 1) in milestone_epochs:
                milestone_filename = f'milestone_epoch_{epoch + 1}.pth'
                self.save_checkpoint(milestone_filename, val_metrics)
                self.logger.info(f"Saved milestone checkpoint at epoch {epoch + 1}")
            
            # Save regular checkpoints
            save_every = self.config.get('save_every', 10)
            if (epoch + 1) % save_every == 0:
                regular_filename = f'checkpoint_epoch_{epoch + 1}.pth'
                self.save_checkpoint(regular_filename, val_metrics)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Training completed - final saves and visualization
        self.logger.info("Training completed!")
        
        # Save final model
        final_metrics = {'final_val_loss': val_loss, 'final_val_f1': val_f1}
        self.save_checkpoint('final_model.pth', final_metrics)
        
        # Create final visualization
        if hasattr(self, 'visualizer') and self.visualizer is not None:
            try:
                created_plots = self.visualizer.create_all_plots(save=True)
                self.logger.info(f"Created final visualization with {len(created_plots)} plots")
            except Exception as e:
                self.logger.warning(f"Failed to create final visualization: {str(e)}")
        
        # Log final statistics
        self.logger.info(f"Final Statistics:")
        self.logger.info(f"  Best Validation Loss: {self.best_val_loss:.6f}")
        self.logger.info(f"  Best Validation F1: {self.best_val_f1:.6f}")
        self.logger.info(f"  Total Epochs: {epoch + 1}")
        self.logger.info(f"  Output Directory: {self.output_dir}")
        
        self.writer.close()
        
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.finish()
    
    def log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """Log training metrics."""
        # Console logging
        val_loss = val_metrics.get('gen_total_loss', val_metrics.get('direct_total_loss', 0.0))
        val_f1 = val_metrics.get('gen_f1_macro', val_metrics.get('direct_f1_macro', 0.0))
        
        self.logger.info(
            f"Epoch {self.epoch + 1:3d} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # TensorBoard logging with enhanced diagnostics
        if self.writer:
            # Log curriculum weights
            if hasattr(self, 'curriculum_weights'):
                for weight_name, weight_value in self.curriculum_weights.items():
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
            curriculum_dict = {f'curriculum_{k}': v for k, v in self.curriculum_weights.items()} if hasattr(self, 'curriculum_weights') else {}
            
            log_dict = {
                'epoch': self.epoch,
                'train_loss': train_metrics.get('total_loss', 0),
                'val_loss': val_metrics.get('gen_total_loss', val_metrics.get('total_loss', 0)),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
                **curriculum_dict
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
    
    def save_checkpoint(self, filename: str, metrics: Dict, save_formats: list = None):
        """Save model checkpoint in multiple formats."""
        if save_formats is None:
            save_formats = self.config.get('save_formats', ['pytorch'])
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'config': self.config,
            'metrics': metrics,
            'training_time': getattr(self, 'training_start_time', None)
        }
        
        # Create checkpoint directory
        checkpoint_dir = self.config.get('checkpoint_dir', self.output_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save in PyTorch format
        if 'pytorch' in save_formats:
            pytorch_path = os.path.join(checkpoint_dir, filename)
            torch.save(checkpoint, pytorch_path)
            self.logger.info(f"Saved PyTorch checkpoint: {pytorch_path}")
        
        # Save in ONNX format (model only)
        if 'onnx' in save_formats:
            try:
                onnx_filename = filename.replace('.pth', '.onnx')
                onnx_path = os.path.join(checkpoint_dir, onnx_filename)
                
                # Create dummy input for ONNX export
                dummy_signal = torch.randn(1, 1, 200).to(self.device)
                dummy_static = torch.randn(1, 4).to(self.device)
                
                # Set model to eval mode for export
                self.model.eval()
                
                with torch.no_grad():
                    torch.onnx.export(
                        self.model,
                        (dummy_signal, dummy_static),
                        onnx_path,
                        export_params=True,
                        opset_version=16,  # Updated to version 16 for better compatibility
                        do_constant_folding=True,
                        input_names=['signal', 'static_params'],
                        output_names=['output'],
                        dynamic_axes={
                            'signal': {0: 'batch_size'},
                            'static_params': {0: 'batch_size'},
                            'output': {0: 'batch_size'}
                        },
                        verbose=False  # Reduce verbose output
                    )
                
                # Restore training mode
                self.model.train()
                self.logger.info(f"Saved ONNX model: {onnx_path}")
                
            except Exception as e:
                # Restore training mode even if export fails
                self.model.train()
                self.logger.warning(f"Failed to save ONNX model: {str(e)}")
                self.logger.info("Continuing training without ONNX export...")
        
        # Save model summary
        summary_filename = filename.replace('.pth', '_summary.txt')
        summary_path = os.path.join(checkpoint_dir, summary_filename)
        self.save_model_summary(summary_path, metrics)
    
    def save_model_summary(self, filepath: str, metrics: Dict = None):
        """Save a human-readable model summary."""
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ABR MODEL SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Model info
            f.write(f"Epoch: {self.epoch + 1}\n")
            f.write(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")
            f.write(f"Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n")
            
            # Training metrics
            if metrics:
                f.write(f"\nValidation Metrics:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value:.6f}\n")
            
            f.write(f"\nBest Metrics:\n")
            f.write(f"  Best Val Loss: {self.best_val_loss:.6f}\n")
            f.write(f"  Best Val F1: {self.best_val_f1:.6f}\n")
            
            # Configuration
            f.write(f"\nTraining Configuration:\n")
            f.write(f"  Learning Rate: {self.config.get('learning_rate', 'N/A')}\n")
            f.write(f"  Batch Size: {self.config.get('batch_size', 'N/A')}\n")
            f.write(f"  Optimizer: {self.config.get('optimizer', {}).get('type', 'N/A')}\n")
            f.write(f"  Scheduler: {self.config.get('scheduler', {}).get('type', 'N/A')}\n")
            
        self.logger.info(f"Saved model summary: {filepath}")

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


# Removed: Use create_optimized_dataloaders from data.dataset instead


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
    
    # Import optimized dataloader from dataset.py
    from data.dataset import create_optimized_dataloaders
    
    # Create optimized data loaders directly from dataset.py
    train_loader, val_loader, _, _ = create_optimized_dataloaders(
        data_path=config['data_path'],
        config=config
    )
    
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
        
        # For cross-validation, we need to create temporary datasets and use the optimized loader
        # TODO: Update this to work with the new optimized dataloader when CV is needed
        from torch.utils.data import DataLoader
        from data.dataset import ABRDataset
        
        # Create datasets for this fold
        train_dataset = ABRDataset(train_data, mode='train', augment=config.get('augment', True))
        val_dataset = ABRDataset(val_data, mode='val', augment=False)
        
        # Create simple data loaders (can be optimized later)
        train_loader = DataLoader(
            train_dataset, batch_size=config.get('batch_size', 32), 
            shuffle=True, num_workers=config.get('num_workers', 4), drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.get('batch_size', 32), 
            shuffle=False, num_workers=config.get('num_workers', 4)
        )
        
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


# Removed: Use create_optimized_dataloaders from data.dataset instead


if __name__ == '__main__':
    main() 