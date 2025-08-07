#!/usr/bin/env python3
"""
Simple professional diffusion training pipeline for ABR generator

Features:
- Data loading with patient-stratified splits
- Diffusion training (noise prediction) with cosine schedule
- Optional classification auxiliary loss
- AMP mixed precision, grad clipping
- Cosine LR schedule with warmup
- EMA of model weights
- Checkpointing (best + last) and resume support
- TensorBoard logging
"""

from __future__ import annotations

import os
import math
import time
import json
import yaml
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Project imports
from data.dataset import create_optimized_dataloaders
from models.hierarchical_unet import OptimizedHierarchicalUNet
from utils.schedule import get_noise_schedule, NoiseSchedule


@dataclass
class TrainConfig:
    # Paths
    data_path: str = "data/processed/ultimate_dataset_with_clinical_thresholds.pkl"
    output_dir: str = "outputs/simple_training"
    checkpoint_dir: str = "checkpoints/simple"
    log_dir: str = "logs/simple"

    # Data
    batch_size: int = 64
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 4
    pin_memory: bool = True
    random_seed: int = 42

    # Model
    base_channels: int = 64
    n_levels: int = 4
    dropout: float = 0.1
    s4_state_size: int = 64
    num_s4_layers: int = 2
    num_transformer_layers: int = 2
    num_heads: int = 8

    # Diffusion
    schedule_type: str = "cosine"
    num_timesteps: int = 1000

    # Optimization
    epochs: int = 200
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 5
    ema_decay: float = 0.999
    classification_weight: float = 0.5  # auxiliary

    # Training
    mixed_precision: bool = True
    early_stopping_patience: int = 20
    resume: bool = False


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
                new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()
                self.shadow[name] = new_avg

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


class SimpleDiffusionTrainer:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        set_seed(self.cfg.random_seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Data
        self.train_loader, self.val_loader, self.test_loader, self.full_dataset = create_optimized_dataloaders(
            data_path=self.cfg.data_path,
            batch_size=self.cfg.batch_size,
            train_ratio=self.cfg.train_ratio,
            val_ratio=self.cfg.val_ratio,
            test_ratio=self.cfg.test_ratio,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            random_state=self.cfg.random_seed,
        )

        # Model
        num_classes = len(set(self.full_dataset.targets)) if hasattr(self.full_dataset, 'targets') else 5
        self.model = OptimizedHierarchicalUNet(
            signal_length=200,
            static_dim=4,
            base_channels=self.cfg.base_channels,
            n_levels=self.cfg.n_levels,
            num_classes=num_classes,
            dropout=self.cfg.dropout,
            s4_state_size=self.cfg.s4_state_size,
            num_s4_layers=self.cfg.num_s4_layers,
            num_transformer_layers=self.cfg.num_transformer_layers,
            num_heads=self.cfg.num_heads,
        ).to(self.device)

        # Diffusion schedule
        schedule = get_noise_schedule(self.cfg.schedule_type, self.cfg.num_timesteps)
        self.noise_schedule = NoiseSchedule(schedule)

        # Optimizer & LR scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        self.scaler = GradScaler(enabled=self.cfg.mixed_precision)
        self.global_step = 0

        # Cosine LR with warmup implemented manually per-epoch
        self.base_lr = self.cfg.learning_rate

        # EMA
        self.ema = EMA(self.model, decay=self.cfg.ema_decay)

        # Logging
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        tb_dir = Path(self.cfg.log_dir) / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(tb_dir))

        # State
        self.best_val = float('inf')
        self.epochs_no_improve = 0

    def _sample_timesteps(self, batch_size: int, low: int = 1) -> torch.Tensor:
        # sample from [1, T-1]; avoid 0 in training
        high = self.noise_schedule.num_timesteps - 1
        t = torch.randint(low, high, (batch_size,), device=self.device, dtype=torch.long)
        return t

    def _lr_at_epoch(self, epoch_idx: int) -> float:
        # Warmup + cosine schedule
        if epoch_idx < self.cfg.warmup_epochs:
            return self.base_lr * (epoch_idx + 1) / max(1, self.cfg.warmup_epochs)
        # Cosine decay over remaining epochs
        progress = (epoch_idx - self.cfg.warmup_epochs) / max(1, (self.cfg.epochs - self.cfg.warmup_epochs))
        return 0.5 * self.base_lr * (1 + math.cos(math.pi * progress))

    def _apply_lr(self, epoch_idx: int):
        lr = self._lr_at_epoch(epoch_idx)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def _compute_losses(self, outputs: Dict[str, torch.Tensor], noise_target: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Predicted noise
        pred_noise = outputs.get('noise', None)
        if pred_noise is None:
            pred_noise = outputs.get('recon', None)
        if pred_noise is None:
            raise RuntimeError("Model outputs must contain 'noise' or 'recon' for diffusion training.")

        # Ensure shape [B,1,L]
        if pred_noise.dim() == 2:
            pred_noise = pred_noise.unsqueeze(1)

        loss_noise = F.mse_loss(pred_noise, noise_target)

        # Optional classification loss if available
        loss_cls = torch.tensor(0.0, device=self.device)
        if 'class' in outputs and 'target' in batch:
            loss_cls = F.cross_entropy(outputs['class'], batch['target'])

        total = loss_noise + self.cfg.classification_weight * loss_cls

        metrics = {
            'loss_total': float(total.detach().item()),
            'loss_noise': float(loss_noise.detach().item()),
            'loss_cls': float(loss_cls.detach().item()),
        }
        return total, metrics

    def _step(self, batch: Dict[str, torch.Tensor], train: bool = True) -> Tuple[float, Dict[str, float]]:
        self.model.train(train)
        signal = batch['signal'].to(self.device)  # [B,1,200]
        static = batch['static_params'].to(self.device)  # [B,4]
        targets = batch['target'].to(self.device)

        # Sample timesteps and noise
        t = self._sample_timesteps(signal.size(0))
        noise = torch.randn_like(signal)
        x_noisy = self.noise_schedule.q_sample(signal, t, noise)

        # Forward
        with autocast(enabled=self.cfg.mixed_precision):
            outputs = self.model(x_noisy, static, t)
            total_loss, metrics = self._compute_losses(outputs, noise, {'target': targets})

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            if self.cfg.grad_clip_norm is not None and self.cfg.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)
            self.global_step += 1

        return float(total_loss.detach().item()), metrics

    @torch.no_grad()
    def _evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        losses = []
        noise_losses = []
        cls_losses = []
        for batch in loader:
            signal = batch['signal'].to(self.device)
            static = batch['static_params'].to(self.device)
            targets = batch['target'].to(self.device)

            t = self._sample_timesteps(signal.size(0))
            noise = torch.randn_like(signal)
            x_noisy = self.noise_schedule.q_sample(signal, t, noise)

            outputs = self.model(x_noisy, static, t)
            # Pred noise
            pred_noise = outputs.get('noise', outputs.get('recon', None))
            if pred_noise is None:
                raise RuntimeError("Model outputs must contain 'noise' or 'recon' for diffusion training.")
            if pred_noise.dim() == 2:
                pred_noise = pred_noise.unsqueeze(1)
            loss_noise = F.mse_loss(pred_noise, noise)
            loss_cls = torch.tensor(0.0, device=self.device)
            if 'class' in outputs:
                loss_cls = F.cross_entropy(outputs['class'], targets)
            total = loss_noise + self.cfg.classification_weight * loss_cls
            losses.append(total.item())
            noise_losses.append(loss_noise.item())
            cls_losses.append(loss_cls.item())
        return {
            'val_total': float(sum(losses)/max(1,len(losses))),
            'val_noise': float(sum(noise_losses)/max(1,len(noise_losses))),
            'val_cls': float(sum(cls_losses)/max(1,len(cls_losses))),
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_shadow': {k: v.cpu() for k, v in self.ema.shadow.items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': asdict(self.cfg),
            'global_step': self.global_step,
        }
        last_path = Path(self.cfg.checkpoint_dir) / 'last.pth'
        torch.save(ckpt, last_path)
        if is_best:
            best_path = Path(self.cfg.checkpoint_dir) / 'best.pth'
            torch.save(ckpt, best_path)

    def load_checkpoint(self, path: Optional[str] = None) -> int:
        ckpt_path = path or (Path(self.cfg.checkpoint_dir) / 'last.pth')
        if not Path(ckpt_path).exists():
            return 0
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        # Restore EMA
        for name, tensor in ckpt.get('ema_shadow', {}).items():
            self.ema.shadow[name] = tensor.to(self.device)
        self.global_step = ckpt.get('global_step', 0)
        return int(ckpt.get('epoch', 0))

    def train(self):
        start_epoch = 0
        if self.cfg.resume:
            start_epoch = self.load_checkpoint()

        # Save config snapshot
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(self.cfg.output_dir) / 'training_config.yaml', 'w') as f:
            yaml.safe_dump(asdict(self.cfg), f)

        for epoch in range(start_epoch, self.cfg.epochs):
            lr = self._apply_lr(epoch)
            self.writer.add_scalar('lr', lr, epoch)

            # Train epoch
            train_losses = []
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs}", leave=False)
            for batch in pbar:
                total_loss, metrics = self._step(batch, train=True)
                train_losses.append(total_loss)
                if self.global_step % 50 == 0:
                    self.writer.add_scalar('train/loss_total', metrics['loss_total'], self.global_step)
                    self.writer.add_scalar('train/loss_noise', metrics['loss_noise'], self.global_step)
                    self.writer.add_scalar('train/loss_cls', metrics['loss_cls'], self.global_step)
                pbar.set_postfix({
                    'total': f"{metrics['loss_total']:.4f}",
                    'noise': f"{metrics['loss_noise']:.4f}",
                    'cls': f"{metrics['loss_cls']:.4f}",
                })

            # Validate with EMA weights
            self.ema.apply_shadow(self.model)
            val_metrics = self._evaluate(self.val_loader)
            self.ema.restore(self.model)

            self.writer.add_scalar('val/total', val_metrics['val_total'], epoch)
            self.writer.add_scalar('val/noise', val_metrics['val_noise'], epoch)
            self.writer.add_scalar('val/cls', val_metrics['val_cls'], epoch)

            # Early stopping & checkpoint
            is_best = val_metrics['val_total'] < self.best_val
            if is_best:
                self.best_val = val_metrics['val_total']
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.save_checkpoint(epoch, is_best=is_best)

            if self.epochs_no_improve >= self.cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print("Training completed. Best val loss:", self.best_val)

