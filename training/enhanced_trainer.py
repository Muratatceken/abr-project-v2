#!/usr/bin/env python3
"""
Enhanced Professional Diffusion Trainer for ABR Generator

Key improvements:
- Enhanced model support with full timestep conditioning
- V-prediction support
- P2 loss weighting
- Config-driven clipping and augmentations
- Dual-branch architecture support
"""

from __future__ import annotations

import os
import math
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import create_optimized_dataloaders
from data.augmentations import create_augmentation
from models.hierarchical_unet import OptimizedHierarchicalUNet
from models.improved_hierarchical_unet import ImprovedHierarchicalUNet
from utils.schedule import get_noise_schedule, NoiseSchedule
from utils.sampling import create_ddim_sampler
from utils.monitoring import TrainingMonitor


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register(model)

    def register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class EnhancedABRTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        set_seed(config.get('training', {}).get('random_seed', 42))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Configuration sections
        model_cfg = config.get('model', {})
        data_cfg = config.get('data', {})
        train_cfg = config.get('training', {})
        opt_cfg = config.get('optimization', {})
        diffusion_cfg = config.get('diffusion', {})

        # Data loaders (augmentation temporarily disabled for stability)
        self.train_loader, self.val_loader, self.test_loader, self.full_dataset = create_optimized_dataloaders(
            data_path=data_cfg.get('path', 'data/processed/ultimate_dataset_with_clinical_thresholds.pkl'),
            batch_size=data_cfg.get('batch_size', 128),
            train_ratio=data_cfg.get('train_ratio', 0.7),
            val_ratio=data_cfg.get('val_ratio', 0.15),
            test_ratio=data_cfg.get('test_ratio', 0.15),
            num_workers=data_cfg.get('num_workers', 4),
            pin_memory=data_cfg.get('pin_memory', True),
            random_state=train_cfg.get('random_seed', 42),
        )

        # Enhanced model selection
        if model_cfg.get('use_enhanced_model', False):
            print("🚀 Using Improved Hierarchical U-Net (Safe Enhanced Version)")
            self.model = ImprovedHierarchicalUNet(
                input_channels=model_cfg.get('input_channels', 1),
                signal_length=model_cfg.get('signal_length', 200),
                static_dim=model_cfg.get('static_dim', 4),
                base_channels=model_cfg.get('base_channels', 64),
                n_levels=model_cfg.get('n_levels', 4),
                dropout=model_cfg.get('dropout', 0.1),
                s4_state_size=model_cfg.get('s4_state_size', 64),
                num_s4_layers=model_cfg.get('num_s4_layers', 2),
                num_transformer_layers=model_cfg.get('num_transformer_layers', 2),
                num_heads=model_cfg.get('num_heads', 8),
                use_enhanced_s4=model_cfg.get('use_enhanced_s4', True),
                use_multi_scale_attention=model_cfg.get('use_multi_scale_attention', True),
                use_cross_attention=model_cfg.get('use_cross_attention', True),
                film_dropout=model_cfg.get('film_dropout', 0.15),
                use_v_prediction=diffusion_cfg.get('prediction_type', 'noise') == 'v_prediction',
                use_attention_heads=model_cfg.get('use_attention_heads', True),
                predict_uncertainty=model_cfg.get('predict_uncertainty', False),
                enable_joint_generation=model_cfg.get('enable_joint_generation', False),
                use_task_specific_extractors=model_cfg.get('use_task_specific_extractors', True),
                channel_multiplier=model_cfg.get('channel_multiplier', 2.0)
            ).to(self.device)
        else:
            print("📦 Using Standard Optimized Hierarchical U-Net")
            self.model = OptimizedHierarchicalUNet(
                input_channels=model_cfg.get('input_channels', 1),
                signal_length=model_cfg.get('signal_length', 200),
                static_dim=model_cfg.get('static_dim', 4),
                base_channels=model_cfg.get('base_channels', 64),
                n_levels=model_cfg.get('n_levels', 4),
                dropout=model_cfg.get('dropout', 0.1),
                s4_state_size=model_cfg.get('s4_state_size', 64),
                num_s4_layers=model_cfg.get('num_s4_layers', 2),
                num_transformer_layers=model_cfg.get('num_transformer_layers', 2),
                num_heads=model_cfg.get('num_heads', 8),
            ).to(self.device)

        # Diffusion schedule
        schedule = get_noise_schedule(
            diffusion_cfg.get('schedule_type', 'cosine'),
            diffusion_cfg.get('num_timesteps', 1000)
        )
        self.noise_schedule = NoiseSchedule(schedule)
        
        # Move noise schedule tensors to device
        for key, tensor in self.noise_schedule.__dict__.items():
            if isinstance(tensor, torch.Tensor):
                setattr(self.noise_schedule, key, tensor.to(self.device))

        # V-prediction support
        self.use_v_prediction = diffusion_cfg.get('prediction_type', 'noise') == 'v_prediction'
        if self.use_v_prediction:
            print("🎯 Using V-prediction for better convergence")

        # P2 weighting support
        self.use_p2_weighting = diffusion_cfg.get('use_p2_weighting', False)
        if self.use_p2_weighting:
            print("⚖️ Using P2 loss weighting for stability")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg.get('learning_rate', 1e-4),
            weight_decay=opt_cfg.get('weight_decay', 1e-4)
        )

        # Training parameters
        self.total_epochs = opt_cfg.get('epochs', 150)
        self.warmup_epochs = opt_cfg.get('warmup_epochs', 10)
        self.base_lr = opt_cfg.get('learning_rate', 1e-4)
        self.accumulation_steps = opt_cfg.get('accumulation_steps', 1)
        self.grad_clip_norm = opt_cfg.get('grad_clip_norm', 1.0)
        self.mixed_precision = train_cfg.get('mixed_precision', True)

        # Logging
        log_cfg = config.get('logging', {})
        self.log_dir = log_cfg.get('log_dir', 'logs/enhanced')
        self.checkpoint_dir = log_cfg.get('checkpoint_dir', 'checkpoints/enhanced')
        
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        if log_cfg.get('use_tensorboard', True):
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None

        # W&B logging
        self.use_wandb = log_cfg.get('use_wandb', False)
        if self.use_wandb:
            import wandb
            wandb.init(
                project=log_cfg.get('wandb_project', 'abr-diffusion-enhanced'),
                name=log_cfg.get('wandb_run_name', 'enhanced_training'),
                config=config
            )
            self.wandb = wandb
        else:
            self.wandb = None

        # Monitoring
        try:
            self.monitor = TrainingMonitor(
                log_dir=self.log_dir,
                model=self.model,
                config=config
            )
        except Exception:
            self.monitor = None

        # State
        self.best_val = float('inf')
        self.epochs_no_improve = 0
        self.early_stop_patience = train_cfg.get('early_stopping_patience', 30)
        self.global_step = 0
        self.start_epoch = 0

        # EMA
        self.ema = EMA(self.model, decay=opt_cfg.get('ema_decay', 0.999))

        # AMP scaler
        self.scaler = GradScaler(enabled=self.mixed_precision)

    def _lr_at_epoch(self, epoch_idx: int) -> float:
        if epoch_idx < self.warmup_epochs:
            return self.base_lr * (epoch_idx + 1) / max(1, self.warmup_epochs)
        progress = (epoch_idx - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
        return 0.5 * self.base_lr * (1 + math.cos(math.pi * progress))

    def _apply_lr(self, epoch_idx: int) -> float:
        lr = self._lr_at_epoch(epoch_idx)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def _sample_timesteps(self, batch_size: int, low: int = 1) -> torch.Tensor:
        high = self.noise_schedule.num_timesteps - 1
        return torch.randint(low, high, (batch_size,), device=self.device, dtype=torch.long)

    def _compute_v_prediction_target(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute v-prediction target: v = sqrt(alpha_cumprod) * noise - sqrt(1 - alpha_cumprod) * x_start"""
        from utils.schedule import extract
        sqrt_alpha_cumprod = extract(self.noise_schedule.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha_cumprod = extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        
        v_target = sqrt_alpha_cumprod * noise - sqrt_one_minus_alpha_cumprod * x_start
        return v_target

    def _compute_losses(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Enhanced loss computation with v-prediction and p2 weighting support."""
        
        if self.use_v_prediction:
            pred = outputs.get('v_pred', outputs.get('noise', outputs.get('recon', None)))
        else:
            pred = outputs.get('noise', outputs.get('recon', None))
            
        if pred is None:
            raise RuntimeError("Model outputs must contain appropriate prediction for diffusion training.")
            
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)

        # Apply P2 weighting if enabled
        if self.use_p2_weighting and timesteps is not None:
            from utils.schedule import extract
            alpha_cumprod_t = extract(self.noise_schedule.alphas_cumprod, timesteps, target.shape)
            p2_k = self.cfg.get('diffusion', {}).get('p2_k', 1.0)
            p2_gamma = self.cfg.get('diffusion', {}).get('p2_gamma', 1.0)
            
            weights = (alpha_cumprod_t ** p2_k) / ((1 - alpha_cumprod_t) ** p2_gamma)
            
            loss_main = F.mse_loss(pred, target, reduction='none')
            loss_main = (loss_main * weights).mean()
        else:
            loss_main = F.mse_loss(pred, target)

        # Envelope loss if dual-branch is used
        total_loss = loss_main
        envelope_loss = torch.tensor(0.0, device=self.device)
        
        if 'envelope' in outputs:
            # Low-pass filter target for envelope supervision
            target_envelope = F.avg_pool1d(target, kernel_size=4, stride=1, padding=2)
            envelope_loss = F.mse_loss(outputs['envelope'], target_envelope)
            total_loss = total_loss + 0.1 * envelope_loss  # Weight envelope loss lower

        metrics = {
            'loss_total': float(total_loss.detach().item()),
            'loss_main': float(loss_main.detach().item()),
            'loss_envelope': float(envelope_loss.detach().item()) if 'envelope' in outputs else 0.0,
        }
        
        return total_loss, metrics

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        accum_steps = self.accumulation_steps
        running = {'total': 0.0, 'main': 0.0, 'envelope': 0.0}
        count = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch+1}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            signal = batch['signal'].to(self.device)
            static = batch['static_params'].to(self.device)

            t = self._sample_timesteps(signal.size(0))
            noise = torch.randn_like(signal)
            x_noisy = self.noise_schedule.q_sample(signal, t, noise)

            # Prepare target based on prediction type
            if self.use_v_prediction:
                target = self._compute_v_prediction_target(signal, noise, t)
            else:
                target = noise

            with autocast(enabled=self.mixed_precision):
                outputs = self.model(x_noisy, static, t)
                total_loss, metrics = self._compute_losses(outputs, target, t)
                loss = total_loss / max(1, accum_steps)

                self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.model)

                self.global_step += 1

            running['total'] += metrics['loss_total']
            running['main'] += metrics['loss_main']
            running['envelope'] += metrics.get('loss_envelope', 0.0)
            count += 1

            if self.writer and (self.global_step % 50 == 0):
                self.writer.add_scalar('train/loss_total', metrics['loss_total'], self.global_step)
                self.writer.add_scalar('train/loss_main', metrics['loss_main'], self.global_step)
                if metrics.get('loss_envelope', 0.0) > 0:
                    self.writer.add_scalar('train/loss_envelope', metrics['loss_envelope'], self.global_step)

            if self.use_wandb:
                log_dict = {
                    'train/loss_total': metrics['loss_total'],
                    'train/loss_main': metrics['loss_main'],
                    'step': self.global_step,
                }
                if metrics.get('loss_envelope', 0.0) > 0:
                    log_dict['train/loss_envelope'] = metrics['loss_envelope']
                self.wandb.log(log_dict)

            pbar.set_postfix({
                'total': f"{metrics['loss_total']:.4f}",
                'main': f"{metrics['loss_main']:.4f}",
                'envelope': f"{metrics.get('loss_envelope', 0.0):.4f}" if metrics.get('loss_envelope', 0.0) > 0 else '0.0000'
            })

        return {k: v / max(1, count) for k, v in running.items()}

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        totals = []
        mains = []
        envelopes = []
        
        for batch in self.val_loader:
            signal = batch['signal'].to(self.device)
            static = batch['static_params'].to(self.device)

            t = self._sample_timesteps(signal.size(0))
            noise = torch.randn_like(signal)
            x_noisy = self.noise_schedule.q_sample(signal, t, noise)

            if self.use_v_prediction:
                target = self._compute_v_prediction_target(signal, noise, t)
            else:
                target = noise

            outputs = self.model(x_noisy, static, t)
            total_loss, metrics = self._compute_losses(outputs, target, t)
            
            totals.append(total_loss.item())
            mains.append(metrics['loss_main'])
            envelopes.append(metrics.get('loss_envelope', 0.0))

        val_metrics = {
            'val_total': float(sum(totals) / max(1, len(totals))),
            'val_main': float(sum(mains) / max(1, len(mains))),
            'val_envelope': float(sum(envelopes) / max(1, len(envelopes)))
        }
        
        if self.writer:
            self.writer.add_scalar('val/total', val_metrics['val_total'], epoch)
            self.writer.add_scalar('val/main', val_metrics['val_main'], epoch)
            if val_metrics['val_envelope'] > 0:
                self.writer.add_scalar('val/envelope', val_metrics['val_envelope'], epoch)
                
        if self.use_wandb:
            log_dict = {
                'val/total': val_metrics['val_total'],
                'val/main': val_metrics['val_main'],
                'epoch': epoch
            }
            if val_metrics['val_envelope'] > 0:
                log_dict['val/envelope'] = val_metrics['val_envelope']
            self.wandb.log(log_dict)
            
        return val_metrics

    @torch.no_grad()
    def log_preview_samples(self, epoch: int, num_samples: int = 8):
        """Generate preview samples with enhanced settings."""
        self.ema.apply_shadow(self.model)
        try:
            sampler = create_ddim_sampler(
                noise_schedule_type=self.cfg.get('diffusion', {}).get('schedule_type', 'cosine'),
                num_timesteps=self.cfg.get('diffusion', {}).get('num_timesteps', 1000),
                eta=self.cfg.get('diffusion', {}).get('eta', 0.0),
                clip_denoised=self.cfg.get('diffusion', {}).get('clip_denoised', False),
            )
            
            batch = next(iter(self.val_loader))
            static = batch['static_params'][:num_samples].to(self.device)
            signal_length = self.cfg.get('model', {}).get('signal_length', 200)
            in_channels = getattr(self.model, 'input_channels', 1)
            shape = (static.size(0), in_channels, signal_length)
            
            # Use more DDIM steps for better preview quality
            steps = self.cfg.get('evaluation', {}).get('ddim_steps', 100)
            
            samples = sampler.sample(
                self.model,
                shape=shape,
                static_params=static,
                device=self.device,
                num_steps=steps,
                cfg_scale=1.0,
                progress=False,
            )
            
            if self.writer:
                import matplotlib.pyplot as plt
                import numpy as np
                fig, axes = plt.subplots(min(num_samples, 4), 1, figsize=(12, 8))
                if not isinstance(axes, (list, tuple, np.ndarray)):
                    axes = [axes]
                    
                for i in range(min(num_samples, 4)):
                    sample_data = samples[i].detach().cpu().squeeze().numpy()
                    time_axis = np.arange(len(sample_data)) / 1000  # Assuming 1kHz sampling
                    axes[i].plot(time_axis, sample_data, 'b-', linewidth=1.0)
                    axes[i].set_title(f'Generated ABR Sample {i+1} (Epoch {epoch})')
                    axes[i].set_ylabel('Amplitude')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].set_ylim([-3, 3])  # Allow wider range without clipping
                    
                axes[-1].set_xlabel('Time (s)')
                plt.tight_layout()
                self.writer.add_figure('preview/samples', fig, epoch)
                plt.close(fig)
                
        except Exception as e:
            if self.writer:
                self.writer.add_text('preview/error', str(e), epoch)
        finally:
            self.ema.restore(self.model)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Enhanced checkpoint saving with EMA weights."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_shadow': self.ema.shadow,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val,
            'global_step': self.global_step,
            'config': self.cfg
        }

        # Save latest
        torch.save(checkpoint, Path(self.checkpoint_dir) / 'last.pth')

        # Save best
        if is_best:
            torch.save(checkpoint, Path(self.checkpoint_dir) / 'best.pth')
            print(f"💾 Saved best checkpoint at epoch {epoch}")

    def train(self, resume: bool = False):
        """Enhanced training loop with all improvements."""
        print("🚀 Starting Enhanced ABR Diffusion Training")
        print(f"📱 Device: {self.device}")
        print(f"🔢 Total epochs: {self.total_epochs}")
        print(f"📊 Batch size: {self.cfg.get('data', {}).get('batch_size', 128)}")
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if self.use_v_prediction:
            print("🎯 V-prediction enabled")
        if self.use_p2_weighting:
            print("⚖️ P2 loss weighting enabled")

        for epoch in range(self.start_epoch, self.total_epochs):
            # Update learning rate
            current_lr = self._apply_lr(epoch)

            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            val_metrics = self.validate_epoch(epoch)

            # Preview samples
            if epoch % 5 == 0:  # Generate previews every 5 epochs
                self.log_preview_samples(epoch)

            # Logging
            print(f"Epoch {epoch+1}/{self.total_epochs}")
            print(f"  Train - Total: {train_metrics['total']:.4f}, Main: {train_metrics['main']:.4f}")
            if train_metrics.get('envelope', 0.0) > 0:
                print(f"          Envelope: {train_metrics['envelope']:.4f}")
            print(f"  Val   - Total: {val_metrics['val_total']:.4f}, Main: {val_metrics['val_main']:.4f}")
            if val_metrics.get('val_envelope', 0.0) > 0:
                print(f"          Envelope: {val_metrics['val_envelope']:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Checkpointing
            is_best = val_metrics['val_total'] < self.best_val
            if is_best:
                self.best_val = val_metrics['val_total']
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.save_checkpoint(epoch, is_best)

            # Early stopping
            if self.epochs_no_improve >= self.early_stop_patience:
                print(f"🛑 Early stopping after {self.epochs_no_improve} epochs without improvement")
                break

        print("✅ Training completed!")
        if self.writer:
            self.writer.close()
        if self.use_wandb:
            self.wandb.finish()