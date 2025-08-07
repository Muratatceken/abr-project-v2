#!/usr/bin/env python3
"""
Professional Diffusion Trainer for ABR Generator

- Config-driven (YAML) training
- Patient-stratified data splits
- Diffusion noise-prediction objective with cosine schedule
- Mixed precision, gradient clipping, accumulation
- Cosine LR with warmup
- EMA weights for stable validation
- Checkpointing (last/best) and resume
- TensorBoard and optional W&B logging
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
from models.hierarchical_unet import OptimizedHierarchicalUNet
from utils.schedule import get_noise_schedule, NoiseSchedule
from utils.sampling import create_ddim_sampler


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
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class ABRTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

        train_cfg = config.get('training', {})
        logging_cfg = config.get('logging', {})
        data_cfg = config.get('data', {})
        opt_cfg = config.get('optimization', {})
        model_cfg = config.get('model', {})
        diffusion_cfg = config.get('diffusion', {})

        # Seed and device
        set_seed(train_cfg.get('random_seed', 42))
        if train_cfg.get('device', 'auto') == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(train_cfg.get('device'))

        # Dataloaders
        self.train_loader, self.val_loader, self.test_loader, self.full_dataset = create_optimized_dataloaders(
            data_path=data_cfg.get('path', 'data/processed/ultimate_dataset_with_clinical_thresholds.pkl'),
            batch_size=data_cfg.get('batch_size', 64),
            train_ratio=data_cfg.get('train_ratio', 0.7),
            val_ratio=data_cfg.get('val_ratio', 0.15),
            test_ratio=data_cfg.get('test_ratio', 0.15),
            num_workers=data_cfg.get('num_workers', 4),
            pin_memory=data_cfg.get('pin_memory', True),
            random_state=train_cfg.get('random_seed', 42),
        )

        # Model
        num_classes = model_cfg.get('num_classes')
        if num_classes is None:
            num_classes = len(set(self.full_dataset.targets)) if hasattr(self.full_dataset, 'targets') else 5
        self.model = OptimizedHierarchicalUNet(
            signal_length=model_cfg.get('signal_length', 200),
            static_dim=model_cfg.get('static_dim', 4),
            base_channels=model_cfg.get('base_channels', 64),
            n_levels=model_cfg.get('n_levels', 4),
            num_classes=num_classes,
            dropout=model_cfg.get('dropout', 0.1),
            s4_state_size=model_cfg.get('s4_state_size', 64),
            num_s4_layers=model_cfg.get('num_s4_layers', 2),
            num_transformer_layers=model_cfg.get('num_transformer_layers', 2),
            num_heads=model_cfg.get('num_heads', 8),
        ).to(self.device)

        # Noise schedule
        schedule = get_noise_schedule(
            diffusion_cfg.get('schedule_type', 'cosine'),
            diffusion_cfg.get('num_timesteps', 1000)
        )
        self.noise_schedule = NoiseSchedule(schedule)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg.get('learning_rate', 2e-4),
            weight_decay=opt_cfg.get('weight_decay', 1e-4)
        )

        # LR schedule (warmup + cosine per-epoch)
        self.base_lr = opt_cfg.get('learning_rate', 2e-4)
        self.warmup_epochs = opt_cfg.get('warmup_epochs', 5)
        self.total_epochs = opt_cfg.get('epochs', 200)

        # Training settings
        self.grad_clip_norm = opt_cfg.get('grad_clip_norm', 1.0)
        self.accumulation_steps = opt_cfg.get('accumulation_steps', 1)
        self.mixed_precision = train_cfg.get('mixed_precision', True)
        self.classification_weight = opt_cfg.get('classification_weight', 0.0)

        # EMA
        self.ema = EMA(self.model, decay=opt_cfg.get('ema_decay', 0.999))

        # Logging paths
        self.checkpoint_dir = Path(logging_cfg.get('checkpoint_dir', 'checkpoints/pro'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = Path(logging_cfg.get('log_dir', 'logs/pro')) / 'tensorboard'
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(tb_dir)) if logging_cfg.get('use_tensorboard', True) else None

        # W&B (optional)
        self.use_wandb = logging_cfg.get('use_wandb', False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=logging_cfg.get('wandb_project', 'abr-diffusion'),
                    name=logging_cfg.get('wandb_run_name'),
                    config=config
                )
                self.wandb = wandb
            except Exception:
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

        # State
        self.best_val = float('inf')
        self.epochs_no_improve = 0
        self.early_stop_patience = train_cfg.get('early_stopping_patience', 20)
        self.global_step = 0
        self.start_epoch = 0

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

    def _compute_losses(self, outputs: Dict[str, torch.Tensor], noise_target: torch.Tensor, targets: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred_noise = outputs.get('noise', outputs.get('recon', None))
        if pred_noise is None:
            raise RuntimeError("Model outputs must contain 'noise' or 'recon' for diffusion training.")
        if pred_noise.dim() == 2:
            pred_noise = pred_noise.unsqueeze(1)
        loss_noise = F.mse_loss(pred_noise, noise_target)

        loss_cls = torch.tensor(0.0, device=self.device)
        if self.classification_weight and self.classification_weight > 0 and 'class' in outputs and targets is not None:
            loss_cls = F.cross_entropy(outputs['class'], targets)

        total = loss_noise + self.classification_weight * loss_cls
        metrics = {
            'loss_total': float(total.detach().item()),
            'loss_noise': float(loss_noise.detach().item()),
            'loss_cls': float(loss_cls.detach().item()),
        }
        return total, metrics

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        accum_steps = self.accumulation_steps
        running = {'total': 0.0, 'noise': 0.0, 'cls': 0.0}
        count = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch+1}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            signal = batch['signal'].to(self.device)
            static = batch['static_params'].to(self.device)
            targets = batch['target'].to(self.device)

            t = self._sample_timesteps(signal.size(0))
            noise = torch.randn_like(signal)
            x_noisy = self.noise_schedule.q_sample(signal, t, noise)

            with autocast(enabled=self.mixed_precision):
                outputs = self.model(x_noisy, static, t)
                total_loss, metrics = self._compute_losses(outputs, noise, targets)
                loss = total_loss / accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.model)

            # Logging
            self.global_step += 1
            running['total'] += metrics['loss_total']
            running['noise'] += metrics['loss_noise']
            running['cls'] += metrics['loss_cls']
            count += 1
            if self.writer and (self.global_step % self.cfg.get('logging', {}).get('log_interval', 50) == 0):
                self.writer.add_scalar('train/loss_total', metrics['loss_total'], self.global_step)
                self.writer.add_scalar('train/loss_noise', metrics['loss_noise'], self.global_step)
                self.writer.add_scalar('train/loss_cls', metrics['loss_cls'], self.global_step)
            if self.use_wandb:
                self.wandb.log({'train/loss_total': metrics['loss_total'], 'train/loss_noise': metrics['loss_noise'], 'train/loss_cls': metrics['loss_cls'], 'step': self.global_step})

            pbar.set_postfix({
                'total': f"{metrics['loss_total']:.4f}",
                'noise': f"{metrics['loss_noise']:.4f}",
                'cls': f"{metrics['loss_cls']:.4f}",
            })

        return {k: v / max(1, count) for k, v in running.items()}

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        totals = []
        noises = []
        clss = []
        for batch in self.val_loader:
            signal = batch['signal'].to(self.device)
            static = batch['static_params'].to(self.device)
            targets = batch['target'].to(self.device)

            t = self._sample_timesteps(signal.size(0))
            noise = torch.randn_like(signal)
            x_noisy = self.noise_schedule.q_sample(signal, t, noise)
            outputs = self.model(x_noisy, static, t)
            pred_noise = outputs.get('noise', outputs.get('recon', None))
            if pred_noise is None:
                raise RuntimeError("Model outputs must contain 'noise' or 'recon' for diffusion training.")
            if pred_noise.dim() == 2:
                pred_noise = pred_noise.unsqueeze(1)
            loss_noise = F.mse_loss(pred_noise, noise)
            loss_cls = torch.tensor(0.0, device=self.device)
            if self.classification_weight and 'class' in outputs:
                loss_cls = F.cross_entropy(outputs['class'], targets)
            totals.append((loss_noise + self.classification_weight * loss_cls).item())
            noises.append(loss_noise.item())
            clss.append(loss_cls.item())

        metrics = {
            'val_total': float(sum(totals) / max(1, len(totals))),
            'val_noise': float(sum(noises) / max(1, len(noises))),
            'val_cls': float(sum(clss) / max(1, len(clss))),
        }
        if self.writer:
            self.writer.add_scalar('val/total', metrics['val_total'], epoch)
            self.writer.add_scalar('val/noise', metrics['val_noise'], epoch)
            self.writer.add_scalar('val/cls', metrics['val_cls'], epoch)
        if self.use_wandb:
            self.wandb.log({'val/total': metrics['val_total'], 'val/noise': metrics['val_noise'], 'val/cls': metrics['val_cls'], 'epoch': epoch})
        return metrics

    @torch.no_grad()
    def log_preview_samples(self, epoch: int, num_samples: int = 8, steps: int = 50):
        # Use EMA for preview
        self.ema.apply_shadow(self.model)
        try:
            sampler = create_ddim_sampler(
                noise_schedule_type=self.cfg.get('diffusion', {}).get('schedule_type', 'cosine'),
                num_timesteps=self.cfg.get('diffusion', {}).get('num_timesteps', 1000),
                eta=0.0,
                clip_denoised=True
            )
            # Take a batch of static params from val loader
            batch = next(iter(self.val_loader))
            static = batch['static_params'][:num_samples].to(self.device)
            shape = (static.size(0), 1, self.cfg.get('model', {}).get('signal_length', 200))
            samples = sampler.sample(self.model, shape=shape, static_params=static, device=self.device, num_steps=steps, cfg_scale=1.0, progress=False)
            # Log to TensorBoard as images
            if self.writer:
                import matplotlib.pyplot as plt
                import numpy as np
                fig, ax = plt.subplots(min(num_samples, 4), 1, figsize=(8, 8))
                if not isinstance(ax, (list, tuple, np.ndarray)):
                    ax = [ax]
                for i in range(min(num_samples, 4)):
                    ax[i].plot(samples[i].detach().cpu().squeeze().numpy())
                    ax[i].set_ylim([-2, 2])
                plt.tight_layout()
                self.writer.add_figure('preview/samples', fig, epoch)
                plt.close(fig)
        finally:
            self.ema.restore(self.model)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'ema_shadow': {k: v.cpu() for k, v in self.ema.shadow.items()},
            'config': self.cfg,
            'global_step': self.global_step,
        }
        last_path = self.checkpoint_dir / 'last.pth'
        torch.save(ckpt, last_path)
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(ckpt, best_path)

    def load_checkpoint(self, path: Optional[str] = None) -> int:
        ckpt_path = path or (self.checkpoint_dir / 'last.pth')
        if not Path(ckpt_path).exists():
            return 0
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        for name, tensor in ckpt.get('ema_shadow', {}).items():
            self.ema.shadow[name] = tensor.to(self.device)
        self.global_step = ckpt.get('global_step', 0)
        return int(ckpt.get('epoch', 0))

    def train(self, resume: bool = False):
        start_epoch = self.load_checkpoint() if resume else 0
        for epoch in range(start_epoch, self.total_epochs):
            lr = self._apply_lr(epoch)
            if self.writer:
                self.writer.add_scalar('lr', lr, epoch)
            if self.use_wandb:
                self.wandb.log({'lr': lr, 'epoch': epoch})

            train_metrics = self.train_epoch(epoch)

            # Validate with EMA
            self.ema.apply_shadow(self.model)
            val_metrics = self.validate_epoch(epoch)
            self.ema.restore(self.model)

            # Preview
            if (epoch + 1) % 5 == 0:
                self.log_preview_samples(epoch, num_samples=self.cfg.get('logging', {}).get('val_preview_samples', 8), steps=50)

            # Checkpointing / early stopping
            is_best = val_metrics['val_total'] < self.best_val
            if is_best:
                self.best_val = val_metrics['val_total']
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            self.save_checkpoint(epoch, is_best=is_best)

            if self.epochs_no_improve >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print("Training completed. Best val loss:", self.best_val)

