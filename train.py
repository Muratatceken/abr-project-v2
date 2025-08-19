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
    Load YAML config with optional dot-notation overrides.
    
    Args:
        config_path: Path to YAML config file
        overrides: String like "optim.lr: 1e-4, trainer.max_epochs: 300"
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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


def create_datasets_and_loaders(cfg: Dict[str, Any]) -> tuple:
    """
    Create datasets and data loaders from config.
    
    Returns:
        Tuple of (train_loader, val_loader, dataset)
    """
    # Load dataset
    dataset = ABRDataset(
        data_path=cfg['data']['train_csv'],  # Using same file for train/val
        normalize_signal=True,
        normalize_static=True
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
    
    # Create data loaders
    loader_kwargs = {
        'batch_size': cfg['loader']['batch_size'],
        'num_workers': cfg['loader']['num_workers'],
        'pin_memory': cfg['loader']['pin_memory'],
        'collate_fn': abr_collate_fn,
        'prefetch_factor': cfg['loader'].get('prefetch_factor', 2) if cfg['loader']['num_workers'] > 0 else None,
        'persistent_workers': cfg['loader'].get('persistent_workers', False) and cfg['loader']['num_workers'] > 0
    }
    
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
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
    
    return train_loader, val_loader, dataset


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
    
    return model


def setup_training(model: nn.Module, cfg: Dict[str, Any], device: torch.device) -> tuple:
    """Setup optimizer, scheduler, losses, and other training components."""
    # Optimizer
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
    
    # Losses
    stft_loss = MultiResSTFTLoss(**cfg['loss']['stft']) if cfg['loss']['stft_weight'] > 0 else None
    
    # Sampler for periodic sampling
    sampler = DDIMSampler(model, cfg['diffusion']['num_train_steps'], device)
    
    logging.info(f"✓ Setup training components")
    logging.info(f"  - Optimizer: AdamW (lr={cfg['optim']['lr']})")
    logging.info(f"  - AMP: {cfg['optim']['amp']}")
    logging.info(f"  - EMA decay: {cfg['optim']['ema_decay']}")
    logging.info(f"  - STFT loss weight: {cfg['loss']['stft_weight']}")
    
    return optimizer, scaler, ema, noise_schedule, stft_loss, sampler


def train_step(
    model: nn.Module, 
    batch: Dict[str, torch.Tensor], 
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    ema: Any,
    noise_schedule: Dict[str, torch.Tensor],
    stft_loss: Optional[nn.Module],
    cfg: Dict[str, Any],
    device: torch.device
) -> Dict[str, float]:
    """Single training step."""
    model.train()
    
    # Move batch to device
    x0 = batch['x0'].to(device)  # [B, 1, T]
    stat = batch['stat'].to(device)  # [B, S]
    
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
        v_pred = model(x_t, static_params=stat_input, timesteps=t)["signal"]
        
        # Main v-prediction loss
        loss_main = mse_time(v_pred, v_target)
        
        # Optional STFT loss
        loss_stft = torch.tensor(0.0, device=device)
        if stft_loss is not None and cfg['loss']['stft_weight'] > 0:
            # Reconstruct x0 from v_pred for STFT loss
            x0_pred = predict_x0_from_v(x_t, v_pred, t, noise_schedule)
            loss_stft = stft_loss(x0_pred, x0)
        
        # Total loss
        loss_total = loss_main + cfg['loss']['stft_weight'] * loss_stft
    
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
    
    return {
        'loss_total': loss_total.item(),
        'loss_main_mse_v': loss_main.item(),
        'loss_stft': loss_stft.item()
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    noise_schedule: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    device: torch.device
) -> float:
    """Validation step."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x0 = batch['x0'].to(device)
            stat = batch['stat'].to(device)
            B = x0.shape[0]
            
            # Random timesteps for validation
            t = torch.randint(0, cfg['diffusion']['num_train_steps'], (B,), device=device)
            noise = torch.randn_like(x0)
            
            # Forward diffusion
            x_t, v_target = q_sample_vpred(x0, t, noise, noise_schedule)
            
            # Model prediction
            v_pred = model(x_t, static_params=stat, timesteps=t)["signal"]
            
            # MSE loss
            loss = mse_time(v_pred, v_target)
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss


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
    
    # Create datasets and loaders
    train_loader, val_loader, dataset = create_datasets_and_loaders(cfg)
    
    # Create model
    model = create_model(cfg, device)
    
    # Setup training components
    optimizer, scaler, ema, noise_schedule, stft_loss, sampler = setup_training(model, cfg, device)
    
    # TensorBoard
    log_dir = Path(cfg['log_dir']) / cfg['exp_name']
    writer = SummaryWriter(log_dir)
    
    # Log config
    config_text = yaml.dump(cfg, default_flow_style=False)
    writer.add_text('config', f"```yaml\n{config_text}\n```", 0)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if cfg['trainer']['resume']:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            model, optimizer, ema, scaler, cfg['trainer']['resume']
        )
    
    # Training loop
    logging.info("="*60)
    logging.info("STARTING TRAINING")
    logging.info("="*60)
    
    for epoch in range(start_epoch, cfg['trainer']['max_epochs']):
        epoch_start_time = time.time()
        epoch_losses = []
        
        # Training
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        
        for batch_idx, batch in enumerate(train_pbar):
            # Training step
            losses = train_step(
                model, batch, optimizer, scaler, ema, 
                noise_schedule, stft_loss, cfg, device
            )
            
            epoch_losses.append(losses['loss_total'])
            global_step += 1
            
            # Logging
            if global_step % cfg['trainer']['log_every_steps'] == 0:
                for key, value in losses.items():
                    writer.add_scalar(f'train/{key}', value, global_step)
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{losses['loss_total']:.4e}",
                    'v_mse': f"{losses['loss_main_mse_v']:.4e}"
                })
        
        # Validation
        if epoch % cfg['trainer']['validate_every_epochs'] == 0:
            val_loss = validate(model, val_loader, noise_schedule, cfg, device)
            writer.add_scalar('val/mse_v', val_loss, epoch)
            
            # Check if best
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            val_loss = best_val_loss
            is_best = False
        
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
        avg_train_loss = np.mean(epoch_losses)
        
        writer.add_scalar('epoch/train_loss_avg', avg_train_loss, epoch)
        writer.add_scalar('epoch/time_sec', epoch_time, epoch)
        
        logging.info(
            f"Epoch {epoch:03d} | train {avg_train_loss:.4e} | "
            f"val(best) {best_val_loss:.4e} | {epoch_time:.1f}s"
        )
    
    # Final checkpoint
    save_checkpoint(
        model, optimizer, ema, scaler, cfg['trainer']['max_epochs'], 
        global_step, best_val_loss, cfg, is_best=False
    )
    
    writer.close()
    logging.info("✅ Training completed!")


if __name__ == "__main__":
    main()
