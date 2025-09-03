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
    Create datasets and data loaders from config with advanced features.
    
    Returns:
        Tuple of (train_loader, val_loader, dataset, curriculum_dataset, augmentation_pipeline)
    """
    # Load dataset with multi-task support
    return_peak_labels = cfg.get('multi_task', {}).get('enabled', False)
    use_peak_balanced_sampler = cfg.get('loader', {}).get('use_peak_balanced_sampler', False)
    
    # Create augmentation pipeline first
    augmentation_pipeline = None
    if cfg.get('advanced_augmentation', {}).get('enabled', True):
        # Use the imported create_augmentation function
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
        logging.info("✓ Created advanced augmentation pipeline")
    else:
        # Use basic augmentation for backward compatibility
        augmentation_pipeline = create_augmentation(
            enable=True,
            time_shift_samples=2,
            noise_std=0.01,
            apply_prob=0.5
        )
    
    dataset = ABRDataset(
        data_path=cfg['data']['train_csv'],  # Using same file for train/val
        normalize_signal=True,
        normalize_static=True,
        return_peak_labels=return_peak_labels,
        transform=augmentation_pipeline
    )
    
    # Verify dataset properties match config
    assert dataset.sequence_length == cfg['data']['sequence_length'], \
        f"Dataset sequence_length {dataset.sequence_length} != config {cfg['data']['sequence_length']}"
    assert dataset.static_dim == cfg['model']['static_dim'], \
        f"Dataset static_dim {dataset.static_dim} != config {cfg['model']['static_dim']}"
    
    # Create stratified splits
    train_dataset, val_dataset, _ = create_stratified_datasets(
        dataset,
        train_ratio=cfg['data']['train_ratio'],
        val_ratio=cfg['data']['val_ratio'],
        test_ratio=cfg['data']['test_ratio'],
        random_state=cfg['seed']
    )
    
    # Create curriculum learning wrapper if enabled
    curriculum_dataset = None
    curriculum_sampler = None
    if cfg.get('curriculum_learning', {}).get('enabled', False):
        curriculum_dataset = create_curriculum_from_config(
            cfg['curriculum_learning'], 
            train_dataset
        )
        if curriculum_dataset is not None:
            # Use CurriculumDataset which handles filtering internally
            # No need for CurriculumSampler as the dataset already filters samples
            train_dataset = curriculum_dataset
            logging.info("✓ Created curriculum learning dataset wrapper and sampler")
            logging.info("✓ Using curriculum sampler for training")
        else:
            logging.warning("Failed to create curriculum dataset, using original dataset")
    
    # Create data loaders
    loader_kwargs = {
        'batch_size': cfg['loader']['batch_size'],
        'num_workers': cfg['loader']['num_workers'],
        'pin_memory': cfg['loader']['pin_memory'],
        'collate_fn': abr_collate_fn,
        'prefetch_factor': cfg['loader'].get('prefetch_factor', 2) if cfg['loader']['num_workers'] > 0 else None,
        'persistent_workers': cfg['loader'].get('persistent_workers', False) and cfg['loader']['num_workers'] > 0
    }
    
    # Determine which sampler to use for training
    train_sampler = None
    
    # Priority 1: Curriculum sampler if curriculum learning is enabled
    if curriculum_sampler is not None:
        train_sampler = curriculum_sampler
        logging.info("✓ Using curriculum sampler for training")
    # Priority 2: Peak-balanced sampling for multi-task training
    elif cfg.get('loader', {}).get('use_peak_balanced_sampler', False) and return_peak_labels:
        from torch.utils.data import WeightedRandomSampler
        
        # Calculate peak class weights
        peak_class_weights = dataset.get_peak_class_weights()
        logging.info(f"✓ Peak class weights: {peak_class_weights.tolist()}")
        
        # Create weighted sampler for training
        # Get indices from Subset to properly align weights with training samples
        if hasattr(train_dataset, 'indices'):  # If it's a Subset
            idxs = train_dataset.indices
        else:  # If it's the full dataset or CurriculumDataset
            idxs = list(range(len(train_dataset)))
        
        weights = [peak_class_weights[int(dataset.peak_labels[i])] for i in idxs]
        
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(idxs),
            replacement=True
        )
        logging.info("✓ Created peak-balanced sampler for training")
        logging.info(f"  - Training subset size: {len(idxs)}")
        logging.info(f"  - Weight alignment: Using Subset indices for proper sample-weight mapping")
    
    train_loader = DataLoader(
        train_dataset, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        drop_last=cfg['loader']['drop_last'],
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )
    
    logging.info(f"✓ Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")
    logging.info(f"✓ Created loaders: batch_size={cfg['loader']['batch_size']}, workers={cfg['loader']['num_workers']}")
    
    # Log multi-task setup information
    if return_peak_labels:
        logging.info(f"✓ Multi-task training enabled with peak labels")
        if use_peak_balanced_sampler:
            logging.info(f"✓ Peak-balanced sampling enabled")
        else:
            logging.info(f"⚠️  Peak-balanced sampling disabled (use loader.use_peak_balanced_sampler: true)")
    
    return train_loader, val_loader, dataset, curriculum_dataset, augmentation_pipeline


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
    
    # Log multi-task capabilities
    if hasattr(model, 'peak5_head'):
        logging.info(f"  - Peak classification head: ✓")
    if hasattr(model, 'static_recon_head'):
        logging.info(f"  - Static reconstruction head: ✓")
    if hasattr(model, 'cross_attention'):
        logging.info(f"  - Cross-attention: ✓")
    
    return model


def setup_training(model: nn.Module, cfg: Dict[str, Any], device: torch.device, dataset=None) -> tuple:
    """Setup optimizer, scheduler, losses, and other advanced training components."""
    # Multi-task configuration
    multi_task_enabled = cfg.get('multi_task', {}).get('enabled', False)
    
    # Advanced training components
    advanced_components = {
        'focal_loss': None,
        'distillation_trainer': None,
        'early_stopping': None,
        'ensemble': None,
        'monitoring': None
    }
    
    # Optimizer with multi-task support
    if multi_task_enabled and cfg.get('multi_task', {}).get('advanced_optimization', {}).get('per_task_lr', False):
        # Use parameter groups for different learning rates per task
        from utils.schedules import create_param_groups
        task_lr_multipliers = cfg['multi_task']['task_lr_multipliers']
        param_groups = create_param_groups(model, cfg['optim']['lr'], task_lr_multipliers)
        
        optimizer = optim.AdamW(
            param_groups,
            betas=cfg['optim']['betas'],
            weight_decay=cfg['optim']['weight_decay']
        )
        
        # Log per-task learning rates
        for group in param_groups:
            logging.info(f"  - {group['name']}: lr={group['lr']:.2e}")
    else:
        # Standard optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg['optim']['lr'],
            betas=cfg['optim']['betas'],
            weight_decay=cfg['optim']['weight_decay']
        )
    
    # AMP scaler
    scaler = GradScaler() if cfg['optim']['amp'] else None
    
    # EMA
    ema = create_ema(model, decay=cfg['optim']['ema_decay'], device=device)
    
    # Diffusion schedule
    noise_schedule = prepare_noise_schedule(cfg['diffusion']['num_train_steps'], device)
    
    # Multi-task losses
    stft_loss = MultiResSTFTLoss(**cfg['loss']['stft']) if cfg['loss']['stft_weight'] > 0 else None
    
    # Peak classification loss for multi-task training
    peak_bce_loss = None
    if multi_task_enabled:
        import torch.nn.functional as F
        
        # Get peak class weights from dataset if available
        peak_class_weights = None
        if dataset is not None and hasattr(dataset, 'get_peak_class_weights'):
            try:
                class_weighting_method = cfg.get('multi_task', {}).get('class_weighting_method', 'inverse_freq')
                peak_class_weights = dataset.get_peak_class_weights(class_weighting_method)
                logging.info(f"✓ Peak classification loss with class weights: {peak_class_weights.tolist()}")
            except Exception as e:
                logging.warning(f"⚠️  Could not get peak class weights: {e}, using default")
                peak_class_weights = None
        
        # Create loss for peak classification (BCE or Focal)
        if cfg.get('focal_loss', {}).get('enabled', False):
            # Use Focal Loss for handling class imbalance
            pos_weight = None
            if peak_class_weights is not None:
                pos_weight = torch.tensor(peak_class_weights[1] / peak_class_weights[0], device=device)
            
            advanced_components['focal_loss'] = FocalLoss(
                alpha=cfg['focal_loss'].get('alpha', 0.25),
                gamma=cfg['focal_loss'].get('gamma', 2.0),
                reduction=cfg['focal_loss'].get('reduction', 'mean'),
                pos_weight=pos_weight
            )
            peak_bce_loss = advanced_components['focal_loss']
            logging.info(f"✓ Using Focal Loss for peak classification (alpha={cfg['focal_loss'].get('alpha', 0.25)}, gamma={cfg['focal_loss'].get('gamma', 2.0)})")
        else:
            # Standard BCE loss
            if peak_class_weights is not None:
                pos_weight = peak_class_weights[1] / peak_class_weights[0]  # positive class weight
                peak_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
            else:
                peak_bce_loss = nn.BCEWithLogitsLoss()
            logging.info(f"✓ Using BCE Loss for peak classification")
        
        logging.info(f"✓ Multi-task training enabled")
        logging.info(f"  - Peak classification loss: ✓")
        logging.info(f"  - Loss weights: {cfg['multi_task']['loss_weights']}")
    
    # Progressive training schedule
    progressive_schedule = None
    if multi_task_enabled and cfg.get('multi_task', {}).get('progressive_weighting', {}).get('enabled', False):
        from utils.schedules import linear_weight_schedule, cosine_weight_schedule
        
        progressive_config = cfg['multi_task']['progressive_weighting']
        schedule_type = progressive_config.get('schedule_type', 'linear')
        
        if schedule_type == 'linear':
            progressive_schedule = linear_weight_schedule
        elif schedule_type == 'cosine':
            progressive_schedule = cosine_weight_schedule
        else:
            logging.warning(f"⚠️  Unknown progressive schedule type: {schedule_type}")
            progressive_schedule = linear_weight_schedule
        
        logging.info(f"✓ Progressive training enabled: {schedule_type}")
    
    # Knowledge Distillation
    if cfg.get('knowledge_distillation', {}).get('enabled', False):
        teacher_checkpoint = cfg['knowledge_distillation'].get('teacher_checkpoint')
        if teacher_checkpoint:
            try:
                teacher_model = load_teacher_model(
                    teacher_checkpoint, 
                    ABRTransformerGenerator,
                    cfg['model'],
                    device
                )
                
                from training.distillation import KnowledgeDistillation
                distillation_method = KnowledgeDistillation(
                    temperature=cfg['knowledge_distillation'].get('temperature', 4.0),
                    alpha=cfg['knowledge_distillation'].get('alpha', 0.7),
                    feature_matching=cfg['knowledge_distillation'].get('feature_distillation', True),
                    feature_weight=cfg['knowledge_distillation'].get('feature_weight', 0.1)
                )
                
                advanced_components['distillation_trainer'] = DistillationTrainer(
                    distillation_method=distillation_method,
                    teacher_model=teacher_model,
                    freeze_teacher=True
                )
                logging.info(f"✓ Knowledge distillation enabled with teacher model")
            except Exception as e:
                logging.error(f"Failed to load teacher model: {e}")
                advanced_components['distillation_trainer'] = None
    
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
    
    return optimizer, scaler, ema, noise_schedule, stft_loss, peak_bce_loss, progressive_schedule, sampler, advanced_components


def train_step(
    model: nn.Module, 
    batch: Dict[str, torch.Tensor], 
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    ema: Any,
    noise_schedule: Dict[str, torch.Tensor],
    stft_loss: Optional[nn.Module],
    peak_bce_loss: Optional[nn.Module],
    progressive_schedule: Optional[callable],
    cfg: Dict[str, Any],
    device: torch.device,
    epoch: int = 0,
    advanced_components: Optional[Dict[str, Any]] = None,
    augmentation_pipeline: Optional[Any] = None
) -> Dict[str, float]:
    """Single training step with advanced features."""
    model.train()
    advanced_components = advanced_components or {}
    
    # Move batch to device
    x0 = batch['x0'].to(device)  # [B, 1, T]
    stat = batch['stat'].to(device)  # [B, S]
    
    # Apply batch-level augmentations (mixup, cutmix)
    if augmentation_pipeline is not None and hasattr(augmentation_pipeline, 'apply_batch_augmentations'):
        batch_dict = {'signal': x0, 'static': stat}
        if 'peak_exists' in batch:
            batch_dict['target'] = batch['peak_exists'].to(device)
            
        augmented_batch = augmentation_pipeline.apply_batch_augmentations(batch_dict)
        x0 = augmented_batch.get('signal', x0)
        if 'target' in augmented_batch:
            batch['peak_exists'] = augmented_batch['target']
    
    # Extract peak labels for multi-task training
    peak_targets = None
    if 'peak_exists' in batch:
        peak_targets = batch['peak_exists'].to(device)  # [B]
    
    # Assert correct shapes
    B, C, T = x0.shape
    assert C == 1 and T == cfg['data']['sequence_length'], f"x0 shape {x0.shape} invalid"
    assert stat.shape == (B, cfg['model']['static_dim']), f"stat shape {stat.shape} invalid"
    
    # Sample timesteps and noise
    t = torch.randint(0, cfg['diffusion']['num_train_steps'], (B,), device=device)
    noise = torch.randn_like(x0)
    
    # Forward diffusion: get x_t and v_target
    x_t, v_target = q_sample_vpred(x0, t, noise, noise_schedule)
    
    # CFG dropout: randomly set static params to None
    if random.random() < cfg.get('advanced', {}).get('cfg_dropout_prob', 0.1):
        stat_input = None
    else:
        stat_input = stat
    
    # Model forward pass
    with autocast(enabled=cfg['optim']['amp']):
        model_output = model(x_t, static_params=stat_input, timesteps=t)
        v_pred = model_output["signal"]
        peak_logits = model_output.get("peak_5th_exists", None)
        static_recon = model_output.get("static_recon", None)
        
        # Main v-prediction loss
        loss_main = mse_time(v_pred, v_target)
        
        # Multi-task losses
        loss_peak = torch.tensor(0.0, device=device)
        loss_static = torch.tensor(0.0, device=device)
        current_peak_weight = 0.0
        current_static_weight = 0.0
        
        # Peak classification loss
        if peak_bce_loss is not None and peak_targets is not None and peak_logits is not None:
            # Apply progressive weight scheduling if enabled
            if progressive_schedule is not None:
                peak_config = cfg['multi_task']['progressive_weighting']['peak_classification']
                current_peak_weight = progressive_schedule(
                    epoch, 
                    peak_config['start_epoch'], 
                    peak_config['end_epoch'],
                    peak_config['start_weight'], 
                    peak_config['end_weight']
                )
            else:
                current_peak_weight = cfg['multi_task']['loss_weights']['peak_classification']
            
            loss_peak = peak_bce_loss(peak_logits, peak_targets)
        
        # Static reconstruction loss
        if static_recon is not None and stat_input is not None:
            if progressive_schedule is not None:
                static_config = cfg['multi_task']['progressive_weighting']['static_reconstruction']
                current_static_weight = progressive_schedule(
                    epoch,
                    static_config['start_epoch'],
                    static_config['end_epoch'],
                    static_config['start_weight'],
                    static_config['end_weight']
                )
            else:
                current_static_weight = cfg['multi_task']['loss_weights']['static_reconstruction']
            
            loss_static = mse_time(static_recon, stat_input)
        
        # Optional STFT loss
        loss_stft = torch.tensor(0.0, device=device)
        if stft_loss is not None and cfg['loss']['stft_weight'] > 0:
            # Reconstruct x0 from v_pred for STFT loss
            x0_pred = predict_x0_from_v(x_t, v_pred, t, noise_schedule)
            loss_stft = stft_loss(x0_pred, x0)
        
        # Knowledge Distillation Loss
        loss_distillation = torch.tensor(0.0, device=device)
        if advanced_components.get('distillation_trainer') is not None:
            try:
                # Create batch dict for distillation
                distill_batch = {
                    'signal': x_t,
                    'static': stat_input,
                    'timesteps': t,
                    'target': peak_targets if peak_targets is not None else None
                }
                
                distillation_losses = advanced_components['distillation_trainer'].compute_distillation_loss(
                    model, distill_batch, epoch
                )
                
                loss_distillation = distillation_losses.get('combined_loss', torch.tensor(0.0, device=device))
            except Exception as e:
                logging.warning(f"Distillation loss computation failed: {e}")
        
        # Total loss with multi-task weights - safely derive weights for backward compatibility
        w_signal = cfg.get('multi_task', {}).get('loss_weights', {}).get('signal', 1.0)
        w_peak = current_peak_weight if peak_bce_loss is not None and peak_targets is not None and peak_logits is not None else 0.0
        w_static = current_static_weight if static_recon is not None and stat_input is not None else 0.0
        w_distill = cfg.get('knowledge_distillation', {}).get('alpha', 0.7) if loss_distillation.item() > 0 else 0.0
        
        loss_total = (
            w_signal * loss_main +
            w_peak * loss_peak +
            w_static * loss_static +
            cfg['loss']['stft_weight'] * loss_stft +
            w_distill * loss_distillation
        )
    
    # Backward pass
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
    
    # Update EMA
    ema.update(model)
    
    result = {
        'loss_total': loss_total.item(),
        'loss_main_mse_v': loss_main.item(),
        'loss_stft': loss_stft.item(),
        'loss_distillation': loss_distillation.item()
    }
    
    # Add multi-task loss components when available
    if peak_bce_loss is not None and peak_targets is not None and peak_logits is not None:
        result['loss_peak_bce'] = loss_peak.item()
        result['peak_weight_current'] = current_peak_weight
    
    if static_recon is not None and stat_input is not None:
        result['loss_static_mse'] = loss_static.item()
        result['static_weight_current'] = current_static_weight
    
    return result


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    noise_schedule: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    device: torch.device
) -> Dict[str, float]:
    """Validation step with multi-task support."""
    model.eval()
    total_signal_loss = 0.0
    total_peak_loss = 0.0
    total_batches = 0
    
    # Multi-task validation metrics
    multi_task_enabled = cfg.get('multi_task', {}).get('enabled', False)
    all_peak_logits = []
    all_peak_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            x0 = batch['x0'].to(device)
            stat = batch['stat'].to(device)
            B = x0.shape[0]
            
            # Extract peak labels for multi-task validation
            peak_targets = None
            if 'peak_exists' in batch:
                peak_targets = batch['peak_exists'].to(device)
            
            # Random timesteps for validation
            t = torch.randint(0, cfg['diffusion']['num_train_steps'], (B,), device=device)
            noise = torch.randn_like(x0)
            
            # Forward diffusion
            x_t, v_target = q_sample_vpred(x0, t, noise, noise_schedule)
            
            # Model prediction
            model_output = model(x_t, static_params=stat, timesteps=t)
            v_pred = model_output["signal"]
            peak_logits = model_output.get("peak_5th_exists", None)
            
            # Signal generation loss
            signal_loss = mse_time(v_pred, v_target)
            total_signal_loss += signal_loss.item()
            
            # Peak classification loss and metrics
            if multi_task_enabled and peak_targets is not None and peak_logits is not None:
                from utils.metrics import compute_classification_metrics
                
                # Store for overall metrics
                all_peak_logits.append(peak_logits.cpu())
                all_peak_targets.append(peak_targets.cpu())
                
                # Compute BCE loss
                peak_bce = nn.BCEWithLogitsLoss()(peak_logits, peak_targets)
                total_peak_loss += peak_bce.item()
            
            total_batches += 1
    
    # Calculate average losses
    avg_signal_loss = total_signal_loss / max(total_batches, 1)
    avg_peak_loss = total_peak_loss / max(total_batches, 1) if multi_task_enabled else 0.0
    
    # Compute comprehensive classification metrics
    classification_metrics = {}
    if multi_task_enabled and all_peak_logits and all_peak_targets:
        from utils.metrics import compute_classification_metrics
        
        # Concatenate all predictions and targets
        all_peak_logits = torch.cat(all_peak_logits, dim=0)
        all_peak_targets = torch.cat(all_peak_targets, dim=0)
        
        # Compute classification metrics
        classification_metrics = compute_classification_metrics(all_peak_logits, all_peak_targets)
    
    # Return metrics
    if multi_task_enabled:
        # Combined validation score for best model selection
        validation_weights = cfg.get('multi_task', {}).get('validation_weights', {'signal': 0.7, 'peak_classification': 0.3})
        combined_score = (
            validation_weights['signal'] * avg_signal_loss +
            validation_weights['peak_classification'] * avg_peak_loss
        )
        
        # Choose validation metric based on configuration
        metric = cfg.get('multi_task', {}).get('validation_metric', 'combined')
        val_selection = {
            'combined': combined_score,
            'signal': avg_signal_loss,
            'peak': avg_peak_loss
        }.get(metric, combined_score)
        
        return {
            'val_signal_mse': avg_signal_loss,
            'val_peak_bce': avg_peak_loss,
            'val_combined_score': combined_score,
            'val_selection': val_selection,  # The metric to use for best model selection
            **classification_metrics
        }
    else:
        # Backward compatibility: return single loss
        return {'val_signal_mse': avg_signal_loss}


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
        x0_val = val_batch['x0'][:cfg['trainer']['num_sample_plots']].to(device)
        stat_val = val_batch['stat'][:cfg['trainer']['num_sample_plots']].to(device)
        
        # Generate samples with same conditioning
        samples = sampler.sample_conditioned(
            static_params=stat_val,
            steps=cfg['diffusion']['sample_steps'],
            eta=cfg['diffusion']['ddim_eta'],
            progress=False
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
    
    # Create datasets and loaders with advanced features
    train_loader, val_loader, dataset, curriculum_dataset, augmentation_pipeline = create_datasets_and_loaders(cfg)
    
    # Create model
    model = create_model(cfg, device)
    
    # Setup training components with advanced features
    optimizer, scaler, ema, noise_schedule, stft_loss, peak_bce_loss, progressive_schedule, sampler, advanced_components = setup_training(model, cfg, device, dataset)
    
    # TensorBoard
    log_dir = Path(cfg['log_dir']) / cfg['exp_name']
    writer = SummaryWriter(log_dir)
    
    # Log config
    config_text = yaml.dump(cfg, default_flow_style=False)
    writer.add_text('config', f"```yaml\n{config_text}\n```", 0)
    
    # Initialize advanced training components
    monitoring = advanced_components.get('monitoring')
    early_stopping = advanced_components.get('early_stopping')
    ensemble = advanced_components.get('ensemble')
    
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
                noise_schedule, stft_loss, peak_bce_loss, progressive_schedule, 
                cfg, device, epoch, advanced_components, augmentation_pipeline
            )
            
            epoch_losses.append(losses)
            global_step += 1
            
            # Logging
            if global_step % cfg['trainer']['log_every_steps'] == 0:
                # Log all training losses
                for key, value in losses.items():
                    writer.add_scalar(f'train/{key}', value, global_step)
                
                # Multi-task specific logging
                if cfg.get('multi_task', {}).get('enabled', False):
                    # Log loss ratios for analysis
                    if 'loss_peak_bce' in losses and 'loss_main_mse_v' in losses:
                        peak_to_signal_ratio = losses['loss_peak_bce'] / (losses['loss_main_mse_v'] + 1e-8)
                        writer.add_scalar('train/peak_to_signal_loss_ratio', peak_to_signal_ratio, global_step)
                    
                    # Log current weights for progressive training
                    if 'peak_weight_current' in losses:
                        writer.add_scalar('train/peak_weight_current', losses['peak_weight_current'], global_step)
                    if 'static_weight_current' in losses:
                        writer.add_scalar('train/static_weight_current', losses['static_weight_current'], global_step)
                
                # Update progress bar
                postfix = {
                    'loss': f"{losses['loss_total']:.4e}",
                    'v_mse': f"{losses['loss_main_mse_v']:.4e}"
                }
                
                # Add multi-task information
                if 'loss_peak_bce' in losses:
                    postfix['peak'] = f"{losses['loss_peak_bce']:.4e}"
                if 'loss_static_mse' in losses:
                    postfix['static'] = f"{losses['loss_static_mse']:.4e}"
                
                train_pbar.set_postfix(postfix)
        
        # Validation
        if epoch % cfg['trainer']['validate_every_epochs'] == 0:
            val_metrics = validate(model, val_loader, noise_schedule, cfg, device)
            
            # Handle multi-task validation metrics
            if cfg.get('multi_task', {}).get('enabled', False):
                # Log all validation metrics
                for key, value in val_metrics.items():
                    writer.add_scalar(f'val/{key}', value, epoch)
                
                # Use configured validation metric for best model selection
                val_loss = val_metrics['val_selection']
                logging.info(f"  Using validation metric: {cfg.get('multi_task', {}).get('validation_metric', 'combined')}")
            else:
                # Backward compatibility: use signal MSE
                val_loss = val_metrics['val_signal_mse']
                writer.add_scalar('val/mse_v', val_loss, epoch)
            
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
        
        # Periodic sampling
        if epoch % cfg['trainer']['sample_every_epochs'] == 0:
            periodic_sampling(
                model, ema, sampler, val_loader, dataset, 
                writer, epoch, cfg, device
            )
        
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
        
        # Multi-task epoch summaries
        if cfg.get('multi_task', {}).get('enabled', False):
            # Log epoch averages for all metrics
            epoch_metrics = {}
            for key in ['loss_peak_bce', 'loss_static_mse']:
                values = [losses.get(key, 0) for losses in epoch_losses if key in losses]
                if values:
                    epoch_metrics[key] = np.mean(values)
                    writer.add_scalar(f'epoch/{key}_avg', epoch_metrics[key], epoch)
            
            # Log epoch summary with multi-task info
            logging.info(
                f"Epoch {epoch:03d} | train {avg_train_loss:.4e} | "
                f"val(best) {best_val_loss:.4e} | {epoch_time:.1f}s"
            )
            if epoch_metrics:
                logging.info(f"  Multi-task: {epoch_metrics}")
        else:
            logging.info(
                f"Epoch {epoch:03d} | train {avg_train_loss:.4e} | "
                f"val(best) {best_val_loss:.4e} | {epoch_time:.1f}s"
            )
    
    # Final checkpoint
    save_checkpoint(
        model, optimizer, ema, scaler, cfg['trainer']['max_epochs'], 
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
