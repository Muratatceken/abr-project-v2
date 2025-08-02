#!/usr/bin/env python3
"""
ABR Convergence-Focused Training Runner

Implements specific fixes for training convergence issues:
- Proper loss weight balancing
- Task-specific learning rates
- Enhanced monitoring
- Gradient flow diagnostics

Usage:
    python run_convergence_training.py
    python run_convergence_training.py --config training/config_convergence_fix.yaml
    python run_convergence_training.py --diagnose_first
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
from tqdm import tqdm

# Import project modules
from training.enhanced_train import ABRTrainer, create_model
from data.dataset import create_optimized_dataloaders
from diffusion.loss import ABRDiffusionLoss


class ConvergenceTrainer(ABRTrainer):
    """Enhanced trainer with specific convergence fixes."""
    
    def __init__(self, model, train_loader, val_loader, config, device):
        super().__init__(model, train_loader, val_loader, config, device)
        
        # Override loss function with convergence-focused settings
        self.setup_convergence_loss()
        
        # Setup task-specific learning rates
        self.setup_task_specific_optimizer()
        
        # Enhanced monitoring
        self.loss_history = {
            'total': [], 'signal': [], 'classification': [], 
            'peak_exist': [], 'peak_latency': [], 'peak_amplitude': [], 'threshold': []
        }
        
        self.gradient_norms = []
        
    def setup_convergence_loss(self):
        """Setup loss function with convergence-focused weights."""
        loss_config = self.config.get('loss', {})
        
        # Use fixed, balanced loss weights
        loss_weights = loss_config.get('loss_weights', {
            'signal': 0.5,
            'peak_exist': 1.5,
            'peak_latency': 3.0,
            'peak_amplitude': 3.0,
            'classification': 4.0,
            'threshold': 2.5
        })
        
        self.loss_fn = ABRDiffusionLoss(
            n_classes=self.config['model']['num_classes'],
            use_focal_loss=loss_config.get('use_focal_loss', True),
            focal_alpha=loss_config.get('focal_alpha', 1.0),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            peak_loss_type=loss_config.get('peak_loss_type', 'huber'),
            huber_delta=loss_config.get('huber_delta', 1.0),
            device=self.device,
            signal_weight=loss_weights.get('signal', 0.5),
            peak_weight=max(loss_weights.get('peak_latency', 3.0), loss_weights.get('peak_amplitude', 3.0)),
            class_weight=loss_weights.get('classification', 4.0),
            threshold_weight=loss_weights.get('threshold', 2.5),
            joint_generation_weight=loss_weights.get('static_params', 0.5)
        )
        
        # Disable adaptive loss weighting for stability
        if hasattr(self.loss_fn, 'use_adaptive_weighting'):
            self.loss_fn.use_adaptive_weighting = False
        
        self.logger.info(f"🎯 Convergence loss weights: {loss_weights}")
    
    def setup_task_specific_optimizer(self):
        """Setup optimizer with task-specific learning rates."""
        base_lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        # Group parameters by task
        param_groups = []
        
        # Signal reconstruction head (lower LR - converges easily)
        signal_params = []
        if hasattr(self.model, 'signal_head'):
            signal_params.extend(list(self.model.signal_head.parameters()))
        
        # Classification head (higher LR - needs more attention)
        class_params = []
        if hasattr(self.model, 'class_head'):
            class_params.extend(list(self.model.class_head.parameters()))
        
        # Peak heads (higher LR - regression tasks)
        peak_params = []
        if hasattr(self.model, 'peak_head'):
            peak_params.extend(list(self.model.peak_head.parameters()))
        
        # Threshold head (higher LR - regression task)
        threshold_params = []
        if hasattr(self.model, 'threshold_head'):
            threshold_params.extend(list(self.model.threshold_head.parameters()))
        
        # Encoder/decoder (standard LR)
        backbone_params = []
        for name, param in self.model.named_parameters():
            if not any(param is p for group_params in [signal_params, class_params, peak_params, threshold_params] 
                      for p in group_params):
                backbone_params.append(param)
        
        # Create parameter groups with different learning rates
        if signal_params:
            param_groups.append({
                'params': signal_params,
                'lr': base_lr * 0.5,  # Lower LR for signal head
                'weight_decay': weight_decay,
                'name': 'signal_head'
            })
        
        if class_params:
            param_groups.append({
                'params': class_params,
                'lr': base_lr * 2.0,  # Higher LR for classification
                'weight_decay': weight_decay,
                'name': 'class_head'
            })
        
        if peak_params:
            param_groups.append({
                'params': peak_params,
                'lr': base_lr * 1.5,  # Higher LR for peak prediction
                'weight_decay': weight_decay,
                'name': 'peak_head'
            })
        
        if threshold_params:
            param_groups.append({
                'params': threshold_params,
                'lr': base_lr * 1.5,  # Higher LR for threshold
                'weight_decay': weight_decay,
                'name': 'threshold_head'
            })
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': base_lr,  # Standard LR for backbone
                'weight_decay': weight_decay,
                'name': 'backbone'
            })
        
        # Create optimizer with parameter groups
        self.optimizer = optim.AdamW(
            param_groups,
            betas=self.config['optimizer'].get('betas', [0.9, 0.95]),
            eps=self.config['optimizer'].get('eps', 1e-8)
        )
        
        # Log parameter group info
        for i, group in enumerate(param_groups):
            self.logger.info(f"📊 Parameter group {i} ({group['name']}): {len(group['params'])} params, LR={group['lr']:.2e}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Enhanced training epoch with convergence monitoring."""
        self.model.train()
        epoch_metrics = {}
        loss_components = {key: [] for key in self.loss_history.keys()}
        gradient_norms_epoch = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Training Epoch {self.epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for key in ['static_params', 'signal', 'v_peak', 'v_peak_mask', 'target']:
                if key in batch:
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(batch['signal'], batch['static_params'])
            total_loss, loss_dict = self.loss_fn(outputs, batch)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip_norm'):
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip_norm']
                    )
                    gradient_norms_epoch.append(grad_norm.item())
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip_norm'):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip_norm']
                    )
                    gradient_norms_epoch.append(grad_norm.item())
                
                self.optimizer.step()
            
            # Store loss components
            loss_components['total'].append(total_loss.item())
            loss_components['signal'].append(loss_dict.get('signal_loss', torch.tensor(0.0)).item())
            loss_components['classification'].append(loss_dict.get('classification_loss', torch.tensor(0.0)).item())
            loss_components['peak_exist'].append(loss_dict.get('peak_exist_loss', torch.tensor(0.0)).item())
            loss_components['peak_latency'].append(loss_dict.get('peak_latency_loss', torch.tensor(0.0)).item())
            loss_components['peak_amplitude'].append(loss_dict.get('peak_amplitude_loss', torch.tensor(0.0)).item())
            loss_components['threshold'].append(loss_dict.get('threshold_loss', torch.tensor(0.0)).item())
            
            # Enhanced progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            avg_grad_norm = np.mean(gradient_norms_epoch) if gradient_norms_epoch else 0.0
            
            progress_bar.set_postfix({
                'total': f'{total_loss.item():.3f}',
                'signal': f'{loss_components["signal"][-1]:.3f}',
                'class': f'{loss_components["classification"][-1]:.3f}',
                'peak': f'{loss_components["peak_latency"][-1]:.3f}',
                'lr': f'{current_lr:.1e}',
                'grad': f'{avg_grad_norm:.2f}'
            })
        
        # Compute epoch statistics
        for key, values in loss_components.items():
            if values:
                epoch_metrics[f'{key}_loss'] = np.mean(values)
                epoch_metrics[f'{key}_loss_std'] = np.std(values)
        
        # Store gradient norm statistics
        if gradient_norms_epoch:
            epoch_metrics['gradient_norm_mean'] = np.mean(gradient_norms_epoch)
            epoch_metrics['gradient_norm_std'] = np.std(gradient_norms_epoch)
            self.gradient_norms.extend(gradient_norms_epoch)
        
        # Update loss history for convergence analysis
        for key, values in loss_components.items():
            if values:
                self.loss_history[key].extend(values)
        
        return epoch_metrics
    
    def log_convergence_analysis(self):
        """Log convergence analysis every few epochs."""
        if self.epoch % 5 == 0 and len(self.loss_history['total']) > 50:
            # Analyze loss trends
            recent_losses = self.loss_history['total'][-50:]  # Last 50 batches
            early_losses = self.loss_history['total'][:50]    # First 50 batches
            
            if len(early_losses) >= 50:
                improvement = np.mean(early_losses) - np.mean(recent_losses)
                improvement_pct = improvement / np.mean(early_losses) * 100
                
                self.logger.info(f"📈 Convergence Analysis (Epoch {self.epoch + 1}):")
                self.logger.info(f"   Total loss improvement: {improvement:.4f} ({improvement_pct:.1f}%)")
                
                # Check for convergence plateaus
                if len(recent_losses) >= 20:
                    recent_std = np.std(recent_losses)
                    if recent_std < 0.01:
                        self.logger.warning("⚠️  Loss has plateaued - consider reducing learning rate")
                
                # Analyze gradient norms
                if len(self.gradient_norms) >= 100:
                    recent_grads = self.gradient_norms[-100:]
                    avg_grad = np.mean(recent_grads)
                    if avg_grad < 1e-4:
                        self.logger.warning(f"⚠️  Very small gradients ({avg_grad:.2e}) - may need higher learning rate")
                    elif avg_grad > 5.0:
                        self.logger.warning(f"⚠️  Large gradients ({avg_grad:.2f}) - may need lower learning rate")
    
    def train(self):
        """Main training loop with enhanced convergence monitoring."""
        self.logger.info("🚀 Starting convergence-focused training")
        self.logger.info(f"🎯 Using task-specific learning rates")
        self.logger.info(f"📊 Loss weights: {self.loss_fn.loss_weights}")
        
        # Call parent training method but with our enhancements
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # Training epoch
            train_metrics = self.train_epoch()
            
            # Validation epoch
            val_metrics = self.validate_epoch()
            
            # Enhanced logging
            self.log_metrics(train_metrics, val_metrics, epoch_time=0.0)  # TODO: Add timing
            
            # Convergence analysis
            self.log_convergence_analysis()
            
            # Learning rate scheduling
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # Check for early stopping
            val_loss = val_metrics.get('direct_total_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.save_checkpoint('best_model.pth', val_metrics)
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config['training'].get('patience', 25):
                self.logger.info(f"🛑 Early stopping at epoch {epoch + 1}")
                break
        
        # Final convergence report
        self.log_final_convergence_report()
        
        self.logger.info("✅ Training completed!")
    
    def log_final_convergence_report(self):
        """Generate final convergence analysis report."""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 FINAL CONVERGENCE REPORT")
        self.logger.info("="*60)
        
        if len(self.loss_history['total']) > 100:
            initial_loss = np.mean(self.loss_history['total'][:50])
            final_loss = np.mean(self.loss_history['total'][-50:])
            total_improvement = initial_loss - final_loss
            improvement_pct = (total_improvement / initial_loss) * 100
            
            self.logger.info(f"🎯 Total Loss Improvement:")
            self.logger.info(f"   Initial: {initial_loss:.4f}")
            self.logger.info(f"   Final: {final_loss:.4f}")
            self.logger.info(f"   Improvement: {total_improvement:.4f} ({improvement_pct:.1f}%)")
            
            # Component-wise analysis
            self.logger.info(f"\n📈 Component Analysis:")
            for component in ['signal', 'classification', 'peak_latency', 'threshold']:
                if len(self.loss_history[component]) > 100:
                    initial = np.mean(self.loss_history[component][:50])
                    final = np.mean(self.loss_history[component][-50:])
                    improvement = initial - final
                    self.logger.info(f"   {component:15s}: {initial:.4f} → {final:.4f} ({improvement:+.4f})")
        
        if self.gradient_norms:
            avg_grad_norm = np.mean(self.gradient_norms)
            self.logger.info(f"\n🔄 Gradient Flow:")
            self.logger.info(f"   Average gradient norm: {avg_grad_norm:.4f}")
            
        self.logger.info("="*60)


def run_diagnostics():
    """Run pre-training diagnostics."""
    print("🔬 Running pre-training diagnostics...")
    import subprocess
    result = subprocess.run([
        'python', 'diagnose_training.py', 
        '--config', 'training/config_convergence_fix.yaml'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Diagnostics completed successfully")
        return True
    else:
        print("❌ Diagnostics failed:")
        print(result.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Run convergence-focused ABR training')
    parser.add_argument('--config', type=str, default='training/config_convergence_fix.yaml',
                       help='Training configuration file')
    parser.add_argument('--diagnose_first', action='store_true',
                       help='Run diagnostics before training')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU training (for debugging)')
    
    args = parser.parse_args()
    
    # Run diagnostics if requested
    if args.diagnose_first:
        if not run_diagnostics():
            print("❌ Diagnostics failed. Please fix issues before training.")
            return
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Starting convergence-focused training")
    print(f"📄 Config: {args.config}")
    print(f"💻 Device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🏗️  Model: {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Create data loaders
    train_loader, val_loader, _, _ = create_optimized_dataloaders(
        data_path=config['data']['data_path'],
        config=config
    )
    
    print(f"📊 Data: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create convergence trainer
    trainer = ConvergenceTrainer(model, train_loader, val_loader, config, device)
    
    # Start training
    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        trainer.save_checkpoint('interrupted_model.pth', {})
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint('failed_model.pth', {})


if __name__ == "__main__":
    main()