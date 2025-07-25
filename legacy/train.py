import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import os
import json
import time
from datetime import datetime

# Import custom modules
from models.cvae import CVAE
from data.dataset import ABRDataset
from utils.losses import cvae_loss, peak_loss
from utils.preprocessing import load_and_preprocess_dataset
from utils.schedulers import get_beta_scheduler
from utils.early_stopping import EarlyStopping, ReduceLROnPlateau
from utils.visualization import plot_signals_with_peaks


class CVAETrainer:
    """
    Trainer class for CVAE model with support for peak prediction and KL annealing.
    """
    
    def __init__(
        self,
        model: CVAE,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize the trainer.
        
        Args:
            model: CVAE model instance
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer instance
            device: Device to run training on
            config: Training configuration dictionary
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Training parameters
        self.max_beta = config.get('max_beta', 1.0)
        self.warmup_epochs = config.get('warmup_epochs', 10)
        self.peak_loss_weight = config.get('peak_loss_weight', 1.0)
        self.use_peak_loss = config.get('use_peak_loss', True)
        
        # Initialize beta scheduler using factory function
        beta_scheduler_type = self.config.get('beta_scheduler_type', 'linear')
        beta_scheduler_params = self.config.get('beta_scheduler_params', {
            'max_beta': self.max_beta,
            'warmup_epochs': self.warmup_epochs
        })
        self.beta_scheduler = get_beta_scheduler(beta_scheduler_type, **beta_scheduler_params)
        
        # Initialize early stopping
        self.early_stopper = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 15),
            min_delta=self.config.get('early_stopping_delta', 1e-4)
        )
        
        # Initialize learning rate scheduler
        self.use_lr_scheduler = self.config.get('lr_scheduler', False)
        if self.use_lr_scheduler:
            self.lr_scheduler = ReduceLROnPlateau(
                patience=self.config.get('lr_scheduler_patience', 5),
                factor=self.config.get('lr_scheduler_factor', 0.5),
                min_lr=self.config.get('min_lr', 1e-6),
                verbose=True
            )
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.save_dir = config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.save_dir)
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        # Get current beta value using beta scheduler
        beta_value = self.beta_scheduler(epoch)
        if isinstance(beta_value, dict):  # Hierarchical scheduler
            # For standard loss compatibility, use average of global and local
            beta = (beta_value['global_kl_weight'] + beta_value['local_kl_weight']) / 2.0
            global_beta = beta_value['global_kl_weight']
            local_beta = beta_value['local_kl_weight']
        else:  # Standard scheduler
            beta = beta_value
            global_beta = beta_value
            local_beta = beta_value
        
        total_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        peak_loss_sum = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            start_time = time.time()
            
            # Move data to device
            signal = batch['signal'].to(self.device)
            static_params = batch['static_params'].to(self.device)
            
            # Forward pass
            model_outputs = self.model(signal, static_params)
            
            # Handle different model architectures
            if isinstance(model_outputs, dict):  # Hierarchical CVAE
                recon_signal = model_outputs['recon_signal']
                mu_global = model_outputs['mu_global']
                logvar_global = model_outputs['logvar_global'] 
                mu_local = model_outputs['mu_local']
                logvar_local = model_outputs['logvar_local']
                predicted_peaks = model_outputs.get('predicted_peaks', None)
                recon_static_from_z = model_outputs.get('recon_static_from_z', None)
                # For compatibility with loss functions, we'll use combined mu/logvar
                mu = torch.cat([mu_global, mu_local], dim=1)
                logvar = torch.cat([logvar_global, logvar_local], dim=1)
            else:  # Standard CVAE
                if self.model.predict_peaks:
                    if len(model_outputs) == 5:  # includes recon_static_from_z
                        recon_signal, mu, logvar, predicted_peaks, recon_static_from_z = model_outputs
                    else:  # standard 4 outputs
                        recon_signal, mu, logvar, predicted_peaks = model_outputs
                        recon_static_from_z = None
                else:
                    if len(model_outputs) == 4:  # includes recon_static_from_z
                        recon_signal, mu, logvar, recon_static_from_z = model_outputs
                    else:  # standard 3 outputs
                        recon_signal, mu, logvar = model_outputs
                        recon_static_from_z = None
                    predicted_peaks = None
            
            # Calculate CVAE loss with annealed beta
            cvae_total_loss, recon_loss, kl_loss = cvae_loss(
                recon_signal, signal, mu, logvar, beta=beta
            )
            
            # Initialize total loss
            batch_total_loss = cvae_total_loss
            batch_peak_loss = torch.tensor(0.0, device=self.device)
            
            # Add peak loss if enabled and model predicts peaks
            if self.use_peak_loss and self.model.predict_peaks:
                target_peaks = batch['peaks'].to(self.device)
                peak_mask = batch['peak_mask'].to(self.device)
                
                batch_peak_loss = peak_loss(predicted_peaks, target_peaks, peak_mask)
                batch_total_loss = batch_total_loss + self.peak_loss_weight * batch_peak_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_total_loss.backward()
            
            # Gradient clipping (optional)
            if self.config.get('gradient_clip', None):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += batch_total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            peak_loss_sum += batch_peak_loss.item()
            num_batches += 1
            
            # Calculate batch time
            duration = time.time() - start_time
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 200) == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_dataloader)}, '
                      f'Beta: {beta:.3f}, Total Loss: {batch_total_loss.item():.4f}, '
                      f'Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}, '
                      f'Peak: {batch_peak_loss.item():.4f}, Time per batch: {duration:.3f} sec')
        
        # Calculate average losses
        avg_metrics = {
            'total_loss': total_loss / num_batches,
            'recon_loss': recon_loss_sum / num_batches,
            'kl_loss': kl_loss_sum / num_batches,
            'peak_loss': peak_loss_sum / num_batches,
            'beta': beta
        }
        
        return avg_metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        peak_loss_sum = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move data to device
                signal = batch['signal'].to(self.device)
                static_params = batch['static_params'].to(self.device)
                
                # Forward pass
                model_outputs = self.model(signal, static_params)
                
                # Handle different model architectures
                if isinstance(model_outputs, dict):  # Hierarchical CVAE
                    recon_signal = model_outputs['recon_signal']
                    mu_global = model_outputs['mu_global']
                    logvar_global = model_outputs['logvar_global'] 
                    mu_local = model_outputs['mu_local']
                    logvar_local = model_outputs['logvar_local']
                    predicted_peaks = model_outputs.get('predicted_peaks', None)
                    recon_static_from_z = model_outputs.get('recon_static_from_z', None)
                    # For compatibility with loss functions, we'll use combined mu/logvar
                    mu = torch.cat([mu_global, mu_local], dim=1)
                    logvar = torch.cat([logvar_global, logvar_local], dim=1)
                else:  # Standard CVAE
                    if self.model.predict_peaks:
                        if len(model_outputs) == 5:  # includes recon_static_from_z
                            recon_signal, mu, logvar, predicted_peaks, recon_static_from_z = model_outputs
                        else:  # standard 4 outputs
                            recon_signal, mu, logvar, predicted_peaks = model_outputs
                            recon_static_from_z = None
                    else:
                        if len(model_outputs) == 4:  # includes recon_static_from_z
                            recon_signal, mu, logvar, recon_static_from_z = model_outputs
                        else:  # standard 3 outputs
                            recon_signal, mu, logvar = model_outputs
                            recon_static_from_z = None
                        predicted_peaks = None
                
                # Calculate CVAE loss (use max_beta for validation)
                cvae_total_loss, recon_loss, kl_loss = cvae_loss(
                    recon_signal, signal, mu, logvar, beta=self.max_beta
                )
                
                # Initialize total loss
                batch_total_loss = cvae_total_loss
                batch_peak_loss = torch.tensor(0.0, device=self.device)
                
                # Add peak loss if enabled and model predicts peaks
                if self.use_peak_loss and self.model.predict_peaks:
                    target_peaks = batch['peaks'].to(self.device)
                    peak_mask = batch['peak_mask'].to(self.device)
                    
                    batch_peak_loss = peak_loss(predicted_peaks, target_peaks, peak_mask)
                    batch_total_loss = batch_total_loss + self.peak_loss_weight * batch_peak_loss
                
                # Accumulate losses
                total_loss += batch_total_loss.item()
                recon_loss_sum += recon_loss.item()
                kl_loss_sum += kl_loss.item()
                peak_loss_sum += batch_peak_loss.item()
                num_batches += 1
        
        # Calculate average losses
        avg_metrics = {
            'total_loss': total_loss / num_batches,
            'recon_loss': recon_loss_sum / num_batches,
            'kl_loss': kl_loss_sum / num_batches,
            'peak_loss': peak_loss_sum / num_batches
        }
        
        return avg_metrics
    
    def train(self, num_epochs: int):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model predicts peaks: {self.model.predict_peaks}")
        print(f"Using peak loss: {self.use_peak_loss}")
        print(f"Peak loss weight: {self.peak_loss_weight}")
        print(f"Beta scheduler: {self.beta_scheduler}")
        print(f"Using LR scheduler: {self.use_lr_scheduler}")
        if self.use_lr_scheduler:
            print(f"LR scheduler: {self.lr_scheduler}")
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics)
            
            # Validation
            val_metrics = self.validate(epoch)
            self.val_losses.append(val_metrics)
            
            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/Train_Total', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Loss/Train_Recon', train_metrics['recon_loss'], epoch)
            self.writer.add_scalar('Loss/Train_KL', train_metrics['kl_loss'], epoch)
            self.writer.add_scalar('Loss/Train_Peak', train_metrics['peak_loss'], epoch)
            self.writer.add_scalar('Beta', train_metrics['beta'], epoch)
            
            self.writer.add_scalar('Loss/Val_Total', val_metrics['total_loss'], epoch)
            self.writer.add_scalar('Loss/Val_Recon', val_metrics['recon_loss'], epoch)
            self.writer.add_scalar('Loss/Val_KL', val_metrics['kl_loss'], epoch)
            self.writer.add_scalar('Loss/Val_Peak', val_metrics['peak_loss'], epoch)
            
            # Log learning rate
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('LearningRate', lr, epoch)
            
            # Log sample reconstructions every 10 epochs
            if epoch % 10 == 0:
                self.log_sample_reconstructions(epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - Total: {train_metrics['total_loss']:.4f}, "
                  f"Recon: {train_metrics['recon_loss']:.4f}, "
                  f"KL: {train_metrics['kl_loss']:.4f}, "
                  f"Peak: {train_metrics['peak_loss']:.4f}, "
                  f"Beta: {train_metrics['beta']:.3f}")
            print(f"  Val   - Total: {val_metrics['total_loss']:.4f}, "
                  f"Recon: {val_metrics['recon_loss']:.4f}, "
                  f"KL: {val_metrics['kl_loss']:.4f}, "
                  f"Peak: {val_metrics['peak_loss']:.4f}")
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(epoch, is_best=True)
                print(f"  New best validation loss: {self.best_val_loss:.4f}")
            
            # Update learning rate scheduler
            if self.use_lr_scheduler:
                self.lr_scheduler.step(val_metrics['total_loss'], self.optimizer, epoch)
            
            # Check early stopping
            self.early_stopper.step(val_metrics['total_loss'], epoch)
            if self.early_stopper.should_stop:
                print(f"Early stopping triggered at epoch {epoch}. Training terminated.")
                print(f"Best validation loss: {self.early_stopper.best_loss:.4f} at epoch {self.early_stopper.best_epoch}")
                break
            
            # Save regular checkpoint
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            print("-" * 60)
        
        print("Training completed!")
        self.writer.close()
        self.plot_training_curves()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'early_stopper_state': {
                'counter': self.early_stopper.counter,
                'best_loss': self.early_stopper.best_loss,
                'best_epoch': self.early_stopper.best_epoch,
                'should_stop': self.early_stopper.should_stop
            }
        }
        
        # Add LR scheduler state if used
        if self.use_lr_scheduler:
            checkpoint['lr_scheduler_state'] = {
                'counter': self.lr_scheduler.counter,
                'best_loss': self.lr_scheduler.best_loss,
                'lr_reduced': self.lr_scheduler.lr_reduced
            }
        
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"  Saved best model to {filepath}")
    
    def log_sample_reconstructions(self, epoch: int):
        """
        Log sample reconstructions to TensorBoard.
        
        Args:
            epoch: Current epoch number
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch from validation data
            val_batch = next(iter(self.val_dataloader))
            signal = val_batch['signal'][:4].to(self.device)  # Take first 4 samples
            static_params = val_batch['static_params'][:4].to(self.device)
            
            # Forward pass
            model_outputs = self.model(signal, static_params)
            
            # Handle different model architectures
            if isinstance(model_outputs, dict):  # Hierarchical CVAE
                recon_signal = model_outputs['recon_signal']
                mu_global = model_outputs['mu_global']
                logvar_global = model_outputs['logvar_global'] 
                mu_local = model_outputs['mu_local']
                logvar_local = model_outputs['logvar_local']
                predicted_peaks = model_outputs.get('predicted_peaks', None)
                recon_static_from_z = model_outputs.get('recon_static_from_z', None)
                # For compatibility with existing code
                mu = torch.cat([mu_global, mu_local], dim=1)
                logvar = torch.cat([logvar_global, logvar_local], dim=1)
            else:  # Standard CVAE
                if self.model.predict_peaks:
                    if len(model_outputs) == 5:  # includes recon_static_from_z
                        recon_signal, mu, logvar, predicted_peaks, recon_static_from_z = model_outputs
                    else:  # standard 4 outputs
                        recon_signal, mu, logvar, predicted_peaks = model_outputs
                        recon_static_from_z = None
                else:
                    if len(model_outputs) == 4:  # includes recon_static_from_z
                        recon_signal, mu, logvar, recon_static_from_z = model_outputs
                    else:  # standard 3 outputs
                        recon_signal, mu, logvar = model_outputs
                        recon_static_from_z = None
                    predicted_peaks = None
                
            # Set target data based on model configuration
            if self.model.predict_peaks and predicted_peaks is not None:
                target_peaks = val_batch['peaks'][:4].to(self.device)
                peak_mask = val_batch['peak_mask'][:4].to(self.device)
            else:
                target_peaks = None
                peak_mask = None
            
            # Create plots for each sample
            for i in range(min(4, signal.shape[0])):
                pred_peaks_i = predicted_peaks[i] if predicted_peaks is not None else None
                tgt_peaks_i = target_peaks[i] if target_peaks is not None else None
                peak_mask_i = peak_mask[i] if peak_mask is not None else None
                
                fig = plot_signals_with_peaks(
                    ground_truth=signal[i],
                    reconstructed=recon_signal[i],
                    predicted_peaks=pred_peaks_i,
                    target_peaks=tgt_peaks_i,
                    peak_mask=peak_mask_i,
                    title=f"Sample {i+1} Reconstruction"
                )
                
                self.writer.add_figure(f"Reconstructions/Sample_{i+1}", fig, epoch)
                plt.close(fig)
        
        self.model.train()
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Load early stopping state
        if 'early_stopper_state' in checkpoint:
            es_state = checkpoint['early_stopper_state']
            self.early_stopper.counter = es_state['counter']
            self.early_stopper.best_loss = es_state['best_loss']
            self.early_stopper.best_epoch = es_state['best_epoch']
            self.early_stopper.should_stop = es_state['should_stop']
        
        # Load LR scheduler state
        if self.use_lr_scheduler and 'lr_scheduler_state' in checkpoint:
            lr_state = checkpoint['lr_scheduler_state']
            self.lr_scheduler.counter = lr_state['counter']
            self.lr_scheduler.best_loss = lr_state['best_loss']
            self.lr_scheduler.lr_reduced = lr_state['lr_reduced']
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return start_epoch
    
    def plot_training_curves(self):
        """
        Plot training and validation curves.
        """
        epochs = range(len(self.train_losses))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(epochs, [m['total_loss'] for m in self.train_losses], label='Train')
        axes[0, 0].plot(epochs, [m['total_loss'] for m in self.val_losses], label='Val')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        axes[0, 1].plot(epochs, [m['recon_loss'] for m in self.train_losses], label='Train')
        axes[0, 1].plot(epochs, [m['recon_loss'] for m in self.val_losses], label='Val')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # KL loss
        axes[1, 0].plot(epochs, [m['kl_loss'] for m in self.train_losses], label='Train')
        axes[1, 0].plot(epochs, [m['kl_loss'] for m in self.val_losses], label='Val')
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Peak loss and Beta
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        
        ax1.plot(epochs, [m['peak_loss'] for m in self.train_losses], 'b-', label='Peak Loss (Train)')
        ax1.plot(epochs, [m['peak_loss'] for m in self.val_losses], 'b--', label='Peak Loss (Val)')
        ax2.plot(epochs, [m['beta'] for m in self.train_losses], 'r-', label='Beta')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Peak Loss', color='b')
        ax2.set_ylabel('Beta', color='r')
        ax1.set_title('Peak Loss and Beta Schedule')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax1.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()


def create_dataloaders(data_path: str, batch_size: int = 32, val_split: float = 0.2, return_peaks: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        data_path: Path to preprocessed data
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        return_peaks: Whether to return peak data
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Load full dataset
    dataset = ABRDataset(data_path, return_peaks=return_peaks)
    
    # Split dataset
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_dataloader, val_dataloader


def main():
    """
    Main training function.
    """
    # Configuration
    config = {
        'data_path': 'data/processed/processed_data.pkl',
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'latent_dim': 32,
        'predict_peaks': True,
        'use_peak_loss': True,
        'peak_loss_weight': 1.0,
        'max_beta': 1.0,
        'warmup_epochs': 15,
        'gradient_clip': 1.0,
        'log_interval': 50,
        'save_interval': 10,
        'save_dir': 'checkpoints',
        'val_split': 0.2,
        'early_stopping_patience': 15,
        'early_stopping_delta': 1e-4,
        
        # Beta scheduler configuration
        'beta_scheduler_type': 'linear',  # 'linear', 'cosine', 'exponential'
        'beta_scheduler_params': {
            'max_beta': 1.0,
            'warmup_epochs': 15,
            'decay_rate': 0.1  # Only used for exponential scheduler
        },
        
        # Learning rate scheduler configuration
        'lr_scheduler': True,
        'lr_scheduler_patience': 5,
        'lr_scheduler_factor': 0.5,
        'min_lr': 1e-6
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        config['data_path'], 
        config['batch_size'], 
        config['val_split'],
        config['predict_peaks']
    )
    
    # Get dataset info
    sample_batch = next(iter(train_dataloader))
    signal_length = sample_batch['signal'].shape[1]
    static_dim = sample_batch['static_params'].shape[1]
    
    print(f"Dataset info:")
    print(f"  Signal length: {signal_length}")
    print(f"  Static params dimension: {static_dim}")
    print(f"  Training batches: {len(train_dataloader)}")
    print(f"  Validation batches: {len(val_dataloader)}")
    
    # Create model
    model = CVAE(
        signal_length=signal_length,
        static_dim=static_dim,
        latent_dim=config['latent_dim'],
        predict_peaks=config['predict_peaks'],
        num_peaks=6
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create trainer
    trainer = CVAETrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        config=config
    )
    
    # Save configuration
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Start training
    trainer.train(config['num_epochs'])


if __name__ == "__main__":
    main()
