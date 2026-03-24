#!/usr/bin/env python3
"""
Professional training pipeline for ABR Transformer with v-prediction diffusion.

Supports TensorBoard logging, EMA, AMP, checkpointing, and periodic sampling.
"""

import os
import sys
import time
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Local imports
from models import ABRTransformerGenerator
from data.dataset import ABRDataset, abr_collate_fn, create_stratified_datasets
from utils import (
    prepare_noise_schedule, q_sample_vpred, predict_x0_from_v,
    create_ema, EMAContextManager,
    MultiResSTFTLoss, mse_time, 
    plot_waveforms, plot_spectrogram, plot_comparison, close_figure
)
from inference import DDIMSampler

# Advanced training imports
from utils.losses import create_loss_from_config, FocalLoss
from data.curriculum import create_curriculum_from_config
from data.augmentations import create_augmentation, create_augmentation_pipeline
from training.early_stopping import create_early_stopping_from_config, EarlyStoppingCallback
from training.ensemble import SnapshotEnsemble
from training.distillation import DistillationTrainer, load_teacher_model
from training.monitoring import TrainingMonitor
from training.cross_validation import CrossValidationManager


def setup_logging(log_level: str = "INFO"):
    """Setup console logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str, overrides: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML config with optional dot-notation overrides and placeholder resolution.
    
    Args:
        config_path: Path to YAML config file
        overrides: String like "optim.lr: 1e-4, trainer.max_epochs: 300"
        
    Returns:
        Configuration dictionary with resolved placeholders
    """
    import re
    import os
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve placeholders
    config = _resolve_placeholders(config)

    # Convert string booleans/numbers from placeholder resolution
    config = _coerce_resolved_types(config)

    # Validate config for unresolved placeholders
    _validate_config_placeholders(config)
    
    # Apply overrides
    if overrides:
        for override in overrides.split(','):
            key, value = override.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Parse value
            try:
                value = yaml.safe_load(value)
            except:
                pass  # Keep as string
            
            # Set nested key
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    
    return config


def _resolve_placeholders(config: Any, env_vars: Optional[Dict[str, str]] = None) -> Any:
    """
    Resolve ${...} placeholders in config values.
    
    Supports:
    - Environment variables: ${HOME}, ${USER}
    - Config references: ${exp_name}, ${data.train_csv}
    - Default values: ${VAR:default_value}
    """
    import re
    import os
    
    if env_vars is None:
        env_vars = dict(os.environ)
    
    def resolve_value(value, config_root):
        if not isinstance(value, str):
            return value
        
        # Pattern to match ${...} placeholders
        pattern = r'\$\{([^}]+)\}'
        
        def replace_placeholder(match):
            placeholder = match.group(1)
            
            # Handle default values: VAR:default
            if ':' in placeholder:
                var_name, default_value = placeholder.split(':', 1)
            else:
                var_name = placeholder
                default_value = None
            
            # Try environment variable first
            if var_name in env_vars:
                return env_vars[var_name]
            
            # Try config reference
            try:
                keys = var_name.split('.')
                result = config_root
                for key in keys:
                    result = result[key]
                return str(result)
            except (KeyError, TypeError):
                pass
            
            # Use default if provided
            if default_value is not None:
                return default_value
            
            # Return original placeholder if can't resolve
            logging.warning(f"Could not resolve placeholder: ${{{placeholder}}}")
            return match.group(0)
        
        return re.sub(pattern, replace_placeholder, value)
    
    if isinstance(config, dict):
        return {k: _resolve_placeholders(resolve_value(v, config), env_vars) for k, v in config.items()}
    elif isinstance(config, list):
        return [_resolve_placeholders(resolve_value(item, config), env_vars) for item in config]
    else:
        return resolve_value(config, config)


def _coerce_resolved_types(config: Any) -> Any:
    """Convert string 'true'/'false'/numbers from placeholder resolution to native types."""
    if isinstance(config, dict):
        return {k: _coerce_resolved_types(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_coerce_resolved_types(item) for item in config]
    elif isinstance(config, str):
        lower = config.lower().strip()
        if lower == 'true':
            return True
        elif lower == 'false':
            return False
        elif lower == 'none' or lower == 'null':
            return None
        # Try numeric conversion
        try:
            if '.' in config or 'e' in lower:
                return float(config)
            return int(config)
        except ValueError:
            pass
    return config


def _validate_config_placeholders(config: Any, path: str = ""):
    """
    Validate that all placeholders in config have been resolved.
    Warns about any remaining ${...} patterns.
    """
    import re
    
    def check_value(value, current_path):
        if isinstance(value, str):
            # Check for unresolved placeholders
            unresolved = re.findall(r'\$\{[^}]+\}', value)
            if unresolved:
                for placeholder in unresolved:
                    logging.warning(
                        f"Unresolved placeholder '{placeholder}' at config path: {current_path}"
                    )
        elif isinstance(value, dict):
            for k, v in value.items():
                check_value(v, f"{current_path}.{k}" if current_path else k)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                check_value(item, f"{current_path}[{i}]" if current_path else f"[{i}]")
    
    check_value(config, path)


def create_datasets_and_loaders(cfg: Dict[str, Any]) -> tuple:
    """
    Create datasets and data loaders with class-balanced sampling.

    Returns:
        Tuple of (train_loader, val_loader, dataset, augmentation_pipeline)
    """
    from torch.utils.data import WeightedRandomSampler
    from collections import Counter

    # Augmentation pipeline
    augmentation_pipeline = None
    if cfg.get('advanced_augmentation', {}).get('enabled', True):
        augmentation_pipeline = create_augmentation(
            enable=True,
            time_shift_samples=cfg.get('advanced_augmentation', {}).get('time_shift_samples', 2),
            noise_std=cfg.get('advanced_augmentation', {}).get('noise_std', 0.01),
            apply_prob=cfg.get('advanced_augmentation', {}).get('apply_prob', 0.5),
            mixup_prob=cfg.get('advanced_augmentation', {}).get('mixup_prob', 0.2),
            cutmix_prob=cfg.get('advanced_augmentation', {}).get('cutmix_prob', 0.2),
            augmentation_strength=cfg.get('advanced_augmentation', {}).get('augmentation_strength', 1.0),
            curriculum_aware=cfg.get('advanced_augmentation', {}).get('curriculum_aware', True)
        )
        logging.info("✓ Created augmentation pipeline")

    # Dataset (no augmentation at dataset level)
    dataset = ABRDataset(
        data_path=cfg['data']['train_csv'],
        normalize_signal=False,
        normalize_static=True,
        return_peak_labels=False,
        transform=None,
    )

    assert dataset.sequence_length == cfg['data']['sequence_length']

    # Stratified patient-level split
    train_dataset, val_dataset, _ = create_stratified_datasets(
        dataset,
        train_ratio=cfg['data']['train_ratio'],
        val_ratio=cfg['data']['val_ratio'],
        test_ratio=cfg['data']['test_ratio'],
        random_state=cfg['seed']
    )

    # Augmentation only on train
    if augmentation_pipeline is not None and train_dataset is not None:
        from data.dataset import _AugmentedSubset
        train_dataset = _AugmentedSubset(train_dataset, augmentation_pipeline)
        logging.info("✓ Augmentation applied ONLY to training split")

    # Class-balanced sampler for hearing loss classes (handles 99x imbalance)
    train_sampler = None
    if train_dataset is not None:
        try:
            # Get class labels from the underlying subset
            subset = train_dataset.subset if hasattr(train_dataset, 'subset') else train_dataset
            idxs = subset.indices if hasattr(subset, 'indices') else list(range(len(subset)))
            labels = [int(dataset.targets[i]) for i in idxs]
            class_counts = Counter(labels)
            class_weights = {c: 1.0 / cnt for c, cnt in class_counts.items()}
            sample_weights = [class_weights[l] for l in labels]
            train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            logging.info(f"✓ Class-balanced sampler: {dict(class_counts)}")
        except Exception as e:
            logging.warning(f"Class-balanced sampler failed ({e}), using shuffle")

    # Data loaders
    loader_kwargs = {
        'batch_size': cfg['loader']['batch_size'],
        'num_workers': cfg['loader']['num_workers'],
        'pin_memory': cfg['loader']['pin_memory'],
        'collate_fn': abr_collate_fn,
        'prefetch_factor': cfg['loader'].get('prefetch_factor', 2) if cfg['loader']['num_workers'] > 0 else None,
        'persistent_workers': cfg['loader'].get('persistent_workers', False) and cfg['loader']['num_workers'] > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=cfg['loader']['drop_last'],
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_kwargs,
    )

    logging.info(f"✓ Datasets: train={len(train_dataset)}, val={len(val_dataset)}")
    return train_loader, val_loader, dataset, augmentation_pipeline


def create_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and initialize model."""
    model = ABRTransformerGenerator(**cfg['model'])
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"✓ Created ABRTransformerGenerator")
    logging.info(f"  - Total parameters: {total_params:,}")
    logging.info(f"  - Trainable parameters: {trainable_params:,}")
    logging.info(f"  - Model size: ~{total_params * 4 / 1024**2:.1f}MB (fp32)")
    
    # Log conditioning info
    if hasattr(model, 'intensity_emb'):
        logging.info(f"  - Intensity embedding: ✓")
    if hasattr(model, 'class_emb'):
        logging.info(f"  - Class embedding ({model.num_classes} classes): ✓")
    
    return model


def setup_training(model: nn.Module, cfg: Dict[str, Any], device: torch.device, dataset=None) -> tuple:
    """Setup optimizer, scheduler, losses, and training components."""
    advanced_components = {
        'early_stopping': None,
        'ensemble': None,
        'monitoring': None
    }

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['optim']['lr'],
        betas=cfg['optim']['betas'],
        weight_decay=cfg['optim']['weight_decay']
    )
    
    # LR Scheduler (warmup + cosine decay)
    scheduler = None
    lr_scheduler_type = cfg.get('advanced', {}).get('lr_scheduler', 'constant')
    warmup_steps = cfg.get('advanced', {}).get('lr_warmup_steps', 0)
    if lr_scheduler_type == 'cosine' and warmup_steps > 0:
        from torch.optim.lr_scheduler import LambdaLR
        import math

        total_steps = cfg['trainer']['max_epochs'] * cfg.get('_estimated_steps_per_epoch', 100)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        logging.info(f"✓ Cosine LR scheduler with {warmup_steps} warmup steps")
    elif lr_scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg['trainer']['max_epochs'],
            eta_min=cfg.get('advanced_optimizer', {}).get('lr_scheduling', {}).get('eta_min', 1e-6)
        )
        logging.info("✓ Cosine annealing LR scheduler (no warmup)")

    # AMP scaler - Handle CPU fallback
    if cfg['optim']['amp'] and not torch.cuda.is_available():
        logging.warning("AMP requested but CUDA not available, disabling AMP for CPU compatibility")
        cfg['optim']['amp'] = False

    scaler = GradScaler() if cfg['optim']['amp'] else None
    
    # EMA
    ema = create_ema(model, decay=cfg['optim']['ema_decay'], device=device)
    
    # Diffusion schedule
    noise_schedule = prepare_noise_schedule(cfg['diffusion']['num_train_steps'], device)
    
    # STFT loss
    stft_loss = MultiResSTFTLoss(**cfg['loss']['stft']) if cfg['loss']['stft_weight'] > 0 else None

    # Early Stopping
    if cfg.get('early_stopping', {}).get('enabled', True):
        advanced_components['early_stopping'] = create_early_stopping_from_config(cfg.get('early_stopping', {}))
        if advanced_components['early_stopping']:
            logging.info(f"✓ Early stopping enabled (patience={cfg.get('early_stopping', {}).get('patience', 20)})")
    
    # Ensemble Training (Snapshot Ensemble)
    if cfg.get('ensemble_training', {}).get('enabled', False):
        snapshot_epochs = cfg['ensemble_training'].get('snapshot_epochs', [])
        if snapshot_epochs:
            advanced_components['ensemble'] = SnapshotEnsemble(
                base_model=model,
                snapshot_epochs=snapshot_epochs,
                device=device,
                max_snapshots=cfg['ensemble_training'].get('ensemble_size', 5)
            )
            logging.info(f"✓ Snapshot ensemble enabled (epochs: {snapshot_epochs})")
    
    # Training Monitoring
    if cfg.get('monitoring', {}).get('enabled', True):
        monitor_save_dir = cfg.get('monitoring', {}).get('save_dir', 'monitoring')
        advanced_components['monitoring'] = TrainingMonitor(
            save_dir=monitor_save_dir,
            enable_resource_monitoring=cfg.get('monitoring', {}).get('track_resources', True),
            enable_model_health=cfg.get('monitoring', {}).get('model_health', {}).get('enabled', True),
            enable_alerts=cfg.get('monitoring', {}).get('alerts', {}).get('enabled', True),
            track_activations=cfg.get('monitoring', {}).get('track_activations', False),
            dashboard_port=cfg.get('monitoring', {}).get('dashboard', {}).get('port') if cfg.get('monitoring', {}).get('dashboard', {}).get('enabled', False) else None
        )
        
        # Setup alerts
        alert_configs = cfg.get('monitoring', {}).get('alerts', {}).get('alert_configs', [])
        for alert_config in alert_configs:
            advanced_components['monitoring'].add_alert(
                metric_name=alert_config['metric_name'],
                threshold=alert_config['threshold'],
                condition=alert_config.get('condition', 'greater'),
                consecutive_violations=alert_config.get('consecutive_violations', 3)
            )
        
        logging.info(f"✓ Comprehensive monitoring enabled")
    
    # Sampler for periodic sampling
    sampler = DDIMSampler(model, cfg['diffusion']['num_train_steps'], device)
    
    logging.info(f"✓ Setup training components")
    logging.info(f"  - Optimizer: AdamW (lr={cfg['optim']['lr']})")
    logging.info(f"  - AMP: {cfg['optim']['amp']}")
    logging.info(f"  - EMA decay: {cfg['optim']['ema_decay']}")
    logging.info(f"  - STFT loss weight: {cfg['loss']['stft_weight']}")
    
    return optimizer, scaler, ema, noise_schedule, stft_loss, sampler, advanced_components, scheduler


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    ema: Any,
    noise_schedule: Dict[str, torch.Tensor],
    stft_loss: Optional[nn.Module],
    cfg: Dict[str, Any],
    device: torch.device,
    scheduler: Optional[Any] = None,
    augmentation_pipeline: Optional[Any] = None,
    **kwargs,
) -> Dict[str, float]:
    """Single training step — simplified for signal-only generation."""
    model.train()

    # Move batch to device
    x0 = batch['x0'].to(device)           # [B, 1, T]
    intensity = batch['intensity'].to(device)   # [B]
    aux_static = batch['aux_static'].to(device) # [B, 3]
    class_label = batch['class_label'].to(device) # [B]

    B, C, T = x0.shape

    # Apply batch-level augmentations (mixup, cutmix)
    if augmentation_pipeline is not None and hasattr(augmentation_pipeline, 'apply_batch_augmentations'):
        batch_dict = {'signal': x0, 'static': batch['stat'].to(device)}
        augmented_batch = augmentation_pipeline.apply_batch_augmentations(batch_dict)
        x0 = augmented_batch.get('signal', x0)

    # Sample timesteps and noise
    t = torch.randint(0, cfg['diffusion']['num_train_steps'], (B,), device=device)
    noise = torch.randn_like(x0)
    x_t, v_target = q_sample_vpred(x0, t, noise, noise_schedule)

    # ── Structured CFG dropout ────────────────────────────────────────────
    cond_cfg = cfg.get('conditioning', {})
    p_both = cond_cfg.get('both_cfg_dropout_prob', 0.05)
    p_class = cond_cfg.get('class_cfg_dropout_prob', 0.15)
    p_static = cond_cfg.get('cfg_dropout_prob', 0.10)

    r = random.random()
    int_input, aux_input, cls_input = intensity, aux_static, class_label
    if r < p_both:
        int_input = aux_input = cls_input = None
    elif r < p_both + p_class:
        cls_input = None
    elif r < p_both + p_class + p_static:
        int_input = aux_input = None

    # ── Forward pass ──────────────────────────────────────────────────────
    with autocast(enabled=cfg['optim']['amp']):
        v_pred = model(x_t, intensity=int_input, aux_static=aux_input,
                       class_label=cls_input, timesteps=t)["signal"]

        # V-prediction MSE loss
        loss_main = mse_time(v_pred, v_target)

        # Optional STFT loss (float32 — torch.stft doesn't support fp16)
        loss_stft = torch.tensor(0.0, device=device)
        stft_weight = cfg['loss']['stft_weight']
        if stft_loss is not None and stft_weight > 0:
            x0_pred = predict_x0_from_v(x_t, v_pred, t, noise_schedule)
            with torch.cuda.amp.autocast(enabled=False):
                loss_stft = stft_loss(x0_pred.float(), x0.float())

        loss_total = loss_main + stft_weight * loss_stft

    # ── Backward ──────────────────────────────────────────────────────────
    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(loss_total).backward()
        if cfg['optim']['grad_clip'] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optim']['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_total.backward()
        if cfg['optim']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optim']['grad_clip'])
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    ema.update(model)

    return {
        'loss_total': loss_total.item(),
        'loss_main_mse_v': loss_main.item(),
        'loss_stft': loss_stft.item(),
        'lr': optimizer.param_groups[0]['lr'],
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    noise_schedule: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    device: torch.device
) -> Dict[str, float]:
    """Validation — x0 reconstruction MSE for model selection."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x0 = batch['x0'].to(device)
            intensity = batch['intensity'].to(device)
            aux_static = batch['aux_static'].to(device)
            class_label = batch['class_label'].to(device)
            B = x0.shape[0]

            t = torch.randint(0, cfg['diffusion']['num_train_steps'], (B,), device=device)
            noise = torch.randn_like(x0)
            x_t, v_target = q_sample_vpred(x0, t, noise, noise_schedule)

            v_pred = model(x_t, intensity=intensity, aux_static=aux_static,
                          class_label=class_label, timesteps=t)["signal"]

            x0_pred = predict_x0_from_v(x_t, v_pred, t, noise_schedule)
            total_loss += mse_time(x0_pred, x0).item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    return {'val_signal_mse': avg_loss}


def periodic_sampling(
    model: nn.Module,
    ema: Any,
    sampler: DDIMSampler,
    val_loader: DataLoader,
    dataset: ABRDataset,
    writer: SummaryWriter,
    epoch: int,
    cfg: Dict[str, Any],
    device: torch.device
):
    """Periodic sampling and visualization."""
    logging.info(f"Generating samples for epoch {epoch}...")
    
    # Switch to EMA weights
    with EMAContextManager(ema, model):
        # Get validation samples for conditioning
        val_batch = next(iter(val_loader))
        n = cfg['trainer']['num_sample_plots']
        x0_val = val_batch['x0'][:n].to(device)
        intensity_val = val_batch['intensity'][:n].to(device)
        aux_val = val_batch['aux_static'][:n].to(device)
        cls_val = val_batch['class_label'][:n].to(device)

        # Generate samples with same conditioning
        samples = sampler.sample_conditioned(
            intensity=intensity_val,
            aux_static=aux_val,
            class_label=cls_val,
            steps=cfg['diffusion']['sample_steps'],
            eta=cfg['diffusion']['ddim_eta'],
            cfg_scale=cfg['diffusion'].get('cfg_scale', 2.0),
            class_cfg_scale=cfg['diffusion'].get('class_cfg_scale', 1.5),
            progress=False,
        )
        
        # Denormalize for visualization
        x0_val_denorm = dataset.denormalize_signal(x0_val)
        samples_denorm = dataset.denormalize_signal(samples)
        
        # Create plots
        try:
            # Waveform comparison
            fig_comparison = plot_comparison(
                pred_batch=samples_denorm.cpu(),
                target_batch=x0_val_denorm.cpu(),
                max_plots=cfg['trainer']['num_sample_plots']
            )
            writer.add_figure('samples/waveforms_comparison', fig_comparison, epoch)
            close_figure(fig_comparison)
            
            # Individual generated waveforms
            fig_generated = plot_waveforms(
                batch=samples_denorm.cpu(),
                titles=[f"Generated {i+1}" for i in range(samples_denorm.shape[0])],
                max_plots=cfg['trainer']['num_sample_plots']
            )
            writer.add_figure('samples/waveforms_generated', fig_generated, epoch)
            close_figure(fig_generated)
            
            # Reference waveforms
            fig_reference = plot_waveforms(
                batch=x0_val_denorm.cpu(),
                titles=[f"Reference {i+1}" for i in range(x0_val_denorm.shape[0])],
                max_plots=cfg['trainer']['num_sample_plots']
            )
            writer.add_figure('samples/waveforms_reference', fig_reference, epoch)
            close_figure(fig_reference)
            
            # Spectrograms if enabled
            if cfg['viz']['plot_spectrogram']:
                fig_spec_gen = plot_spectrogram(
                    batch=samples_denorm.cpu(),
                    max_plots=cfg['viz']['max_spectrogram_plots']
                )
                writer.add_figure('samples/spectrogram_generated', fig_spec_gen, epoch)
                close_figure(fig_spec_gen)
                
                fig_spec_ref = plot_spectrogram(
                    batch=x0_val_denorm.cpu(),
                    max_plots=cfg['viz']['max_spectrogram_plots']
                )
                writer.add_figure('samples/spectrogram_reference', fig_spec_ref, epoch)
                close_figure(fig_spec_ref)
            
            logging.info(f"✓ Generated and logged {samples.shape[0]} samples")
            
        except Exception as e:
            logging.warning(f"Failed to create plots: {e}")


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    ema: Any,
    scaler: Optional[GradScaler],
    epoch: int,
    global_step: int,
    val_loss: float,
    cfg: Dict[str, Any],
    is_best: bool = False
) -> str:
    """Save checkpoint."""
    ckpt_dir = Path(cfg['trainer']['ckpt_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'val_loss': val_loss,
        'config': cfg
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Save best checkpoint
    if is_best:
        best_path = ckpt_dir / f"{cfg['exp_name']}_best.pt"
        torch.save(checkpoint, best_path)
        logging.info(f"✓ Saved best checkpoint: {best_path}")
        return str(best_path)
    
    # Save epoch checkpoint
    epoch_path = ckpt_dir / f"{cfg['exp_name']}_e{epoch:03d}.pt"
    torch.save(checkpoint, epoch_path)
    
    # Clean up old checkpoints
    if cfg['trainer']['keep_last_k'] > 0:
        checkpoints = sorted(ckpt_dir.glob(f"{cfg['exp_name']}_e*.pt"))
        if len(checkpoints) > cfg['trainer']['keep_last_k']:
            for old_ckpt in checkpoints[:-cfg['trainer']['keep_last_k']]:
                old_ckpt.unlink()
    
    logging.info(f"✓ Saved epoch checkpoint: {epoch_path}")
    return str(epoch_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    ema: Any,
    scaler: Optional[GradScaler],
    resume_path: str
) -> tuple:
    """Load checkpoint and return (start_epoch, global_step, best_val_loss)."""
    logging.info(f"Loading checkpoint from {resume_path}")
    
    checkpoint = torch.load(resume_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ema.load_state_dict(checkpoint['ema_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint['global_step']
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    logging.info(f"✓ Resumed from epoch {start_epoch}, step {global_step}")
    return start_epoch, global_step, best_val_loss


def _log_generation_quality(model, ema, sampler, val_loader, writer, epoch, cfg, device):
    """Measure generation quality by comparing generated vs real signals."""
    try:
        from utils import mse_time, pearson_correlation

        # Apply EMA weights temporarily (apply_to returns original params for restore)
        original_params = ema.apply_to(model)
        model.eval()

        val_batch = next(iter(val_loader))
        n = min(16, val_batch['x0'].shape[0])
        x0 = val_batch['x0'][:n].to(device)
        intensity = val_batch['intensity'][:n].to(device)
        aux = val_batch['aux_static'][:n].to(device)
        cls = val_batch['class_label'][:n].to(device)

        with torch.no_grad():
            gen = sampler.sample_conditioned(
                intensity=intensity, aux_static=aux, class_label=cls,
                steps=cfg['diffusion']['sample_steps'],
                cfg_scale=cfg['diffusion'].get('cfg_scale', 3.0),
                class_cfg_scale=cfg['diffusion'].get('class_cfg_scale', 2.0),
                progress=False,
            )
            gen_mse = mse_time(gen, x0).item()
            gen_corr = pearson_correlation(gen, x0).item()

        writer.add_scalar('gen_quality/mse', gen_mse, epoch)
        writer.add_scalar('gen_quality/correlation', gen_corr, epoch)
        logging.info(f"  Gen quality: MSE={gen_mse:.4f}, Corr={gen_corr:.4f}")

        # Restore original training weights
        ema.restore(model, original_params)
    except Exception as e:
        logging.warning(f"Gen quality check failed: {e}")


def _log_uncond_health(model, writer, epoch):
    """Log unconditional embedding norms for CFG health monitoring.

    Note: Gradients are zeroed after optimizer.step(), so we track param norms.
    Growing norms indicate the uncond embeddings are learning.
    """
    try:
        for name, param in model.named_parameters():
            if 'uncond_emb' in name:
                norm = param.data.norm().item()
                short_name = name.replace('module.', '')
                writer.add_scalar(f'uncond/{short_name}_norm', norm, epoch)
    except Exception as e:
        logging.warning(f"Uncond health check failed: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ABR Transformer")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--override", type=str, default="", help="Config overrides (dotted notation)")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config, args.override)
    if args.resume:
        cfg['trainer']['resume'] = args.resume
    
    # Setup
    setup_logging(cfg.get('logging', {}).get('console_log_level', 'INFO'))
    set_seed(cfg['seed'])
    
    # Device
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create datasets and loaders
    train_loader, val_loader, dataset, augmentation_pipeline = create_datasets_and_loaders(cfg)
    
    # Create model
    model = create_model(cfg, device)
    
    # Setup training components with advanced features
    optimizer, scaler, ema, noise_schedule, stft_loss, sampler, advanced_components, scheduler = setup_training(model, cfg, device, dataset)
    
    # TensorBoard
    log_dir = Path(cfg['log_dir']) / cfg['exp_name']
    writer = SummaryWriter(log_dir)
    
    # Log config
    config_text = yaml.dump(cfg, default_flow_style=False)
    writer.add_text('config', f"```yaml\n{config_text}\n```", 0)
    
    # Advanced training components
    monitoring = advanced_components.get('monitoring')
    early_stopping = advanced_components.get('early_stopping')
    ensemble = advanced_components.get('ensemble')
    curriculum_dataset = None  # removed, kept variable for compat
    
    if monitoring:
        monitoring.start_training()
    
    # Create early stopping callback if enabled
    early_stopping_callback = None
    if early_stopping:
        # Get metric and mode from config
        early_stopping_config = cfg.get('early_stopping', {})
        metric_name = early_stopping_config.get('metric', 'val_loss')
        mode = early_stopping_config.get('mode', 'min')
        
        early_stopping_callback = EarlyStoppingCallback(
            early_stopper=early_stopping,
            checkpoint_dir=cfg['trainer']['ckpt_dir'],
            save_best_only=True,
            metric_name=metric_name
        )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if cfg['trainer']['resume']:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            model, optimizer, ema, scaler, cfg['trainer']['resume']
        )
    
    # Training loop with advanced features
    logging.info("="*60)
    logging.info("STARTING TRAINING")
    logging.info("="*60)
    
    for epoch in range(start_epoch, cfg['trainer']['max_epochs']):
        epoch_start_time = time.time()
        epoch_losses = []
        
        # Update curriculum learning
        if curriculum_dataset is not None:
            curriculum_dataset.update_epoch(epoch, cfg['trainer']['max_epochs'])
            
            # Log curriculum stats
            curriculum_stats = curriculum_dataset.get_curriculum_stats()
            if curriculum_stats:
                for stat_name, stat_value in curriculum_stats.items():
                    # Handle tuple values (e.g., difficulty_range)
                    if isinstance(stat_value, tuple):
                        if len(stat_value) == 2:
                            writer.add_scalar(f'curriculum/{stat_name}_min', stat_value[0], epoch)
                            writer.add_scalar(f'curriculum/{stat_name}_max', stat_value[1], epoch)
                        else:
                            # Skip complex tuples that can't be easily logged
                            continue
                    else:
                        writer.add_scalar(f'curriculum/{stat_name}', stat_value, epoch)
                logging.info(f"Curriculum stats: {curriculum_stats}")
            
        # Update augmentation progress for curriculum-aware augmentation
        if augmentation_pipeline is not None and hasattr(augmentation_pipeline, 'set_training_progress'):
            progress = epoch / cfg['trainer']['max_epochs']
            augmentation_pipeline.set_training_progress(progress)
        
        # Training
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        
        for batch_idx, batch in enumerate(train_pbar):
            # Training step with advanced features
            losses = train_step(
                model, batch, optimizer, scaler, ema,
                noise_schedule, stft_loss, cfg, device,
                scheduler=scheduler, augmentation_pipeline=augmentation_pipeline
            )
            
            epoch_losses.append(losses)
            global_step += 1
            
            # Logging
            if global_step % cfg['trainer']['log_every_steps'] == 0:
                # Log all training losses
                for key, value in losses.items():
                    writer.add_scalar(f'train/{key}', value, global_step)
                
                # Log LR
                if 'lr' in losses:
                    writer.add_scalar('train/lr', losses['lr'], global_step)

                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{losses['loss_total']:.4e}",
                    'v_mse': f"{losses['loss_main_mse_v']:.4e}",
                })
        
        # Validation
        if epoch % cfg['trainer']['validate_every_epochs'] == 0:
            val_metrics = validate(model, val_loader, noise_schedule, cfg, device)
            
            # Validation metric for best model selection
            val_loss = val_metrics['val_signal_mse']
            writer.add_scalar('val/mse_x0', val_loss, epoch)
            
            # Update monitoring with validation metrics
            if monitoring:
                combined_metrics = {**val_metrics}
                # Add training metrics from last batch
                if epoch_losses:
                    latest_train_losses = epoch_losses[-1]
                    for key, value in latest_train_losses.items():
                        combined_metrics[f'train_{key}'] = value
                
                monitoring.update(epoch, global_step, combined_metrics, model)
            
            # Check if best
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                
            # Early stopping check
            should_stop = False
            if early_stopping_callback:
                should_stop = early_stopping_callback.on_epoch_end(epoch, val_metrics, model, optimizer)
            elif early_stopping:
                should_stop = early_stopping(val_loss, model, epoch)
                
            if should_stop:
                logging.info(f"Early stopping triggered at epoch {epoch}")
                if early_stopping and hasattr(early_stopping, 'restore_best_model'):
                    early_stopping.restore_best_model(model)
                break
        else:
            val_loss = best_val_loss
            is_best = False
            
        # Ensemble snapshot collection
        if ensemble and ensemble.should_take_snapshot(epoch):
            ensemble.take_snapshot(model, epoch, {'val_loss': val_loss} if 'val_loss' in locals() else None)
        
        # Periodic sampling + generation quality metrics
        if epoch % cfg['trainer']['sample_every_epochs'] == 0:
            periodic_sampling(
                model, ema, sampler, val_loader, dataset,
                writer, epoch, cfg, device
            )
            # Log generation quality metrics
            _log_generation_quality(model, ema, sampler, val_loader, writer, epoch, cfg, device)
            # Log unconditional embedding norms (CFG health check)
            _log_uncond_health(model, writer, epoch)
        
        # Checkpointing
        if epoch % cfg['trainer']['save_every_epochs'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, ema, scaler, epoch, global_step,
                val_loss, cfg, is_best
            )
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = np.mean([losses['loss_total'] for losses in epoch_losses])
        
        writer.add_scalar('epoch/train_loss_avg', avg_train_loss, epoch)
        writer.add_scalar('epoch/time_sec', epoch_time, epoch)
        
        logging.info(
            f"Epoch {epoch:03d} | train {avg_train_loss:.4e} | "
            f"val(best) {best_val_loss:.4e} | {epoch_time:.1f}s"
        )
    
    # Final checkpoint — use last completed epoch (max_epochs - 1), not max_epochs
    save_checkpoint(
        model, optimizer, ema, scaler, cfg['trainer']['max_epochs'] - 1,
        global_step, best_val_loss, cfg, is_best=False
    )
    
    # Cleanup advanced training components
    if early_stopping_callback:
        early_stopping_callback.on_training_end(model)
        
    if monitoring:
        monitoring.end_training()
        
    # Generate final reports
    if monitoring:
        try:
            monitoring.plot_training_progress(save_path=f"{cfg['log_dir']}/{cfg['exp_name']}/training_progress.png")
            from training.monitoring import generate_training_report
            generate_training_report(monitoring, f"{cfg['log_dir']}/{cfg['exp_name']}/training_report.html")
        except Exception as e:
            logging.warning(f"Failed to generate monitoring reports: {e}")
    
    writer.close()
    logging.info("✅ Training completed with advanced features!")


if __name__ == "__main__":
    main()
