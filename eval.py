#!/usr/bin/env python3
"""
Professional evaluation pipeline for ABR Transformer.

Supports reconstruction and conditional generation evaluation with comprehensive
metrics, visualizations, and TensorBoard logging.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
from models import ABRTransformerGenerator
from data.dataset import ABRDataset, abr_collate_fn, create_stratified_datasets
from utils import (
    prepare_noise_schedule, q_sample_vpred, compute_per_sample_metrics,
    overlay_waveforms, error_curve, spectrograms, scatter_xy, metrics_summary_plot,
    close_figure
)
from inference import DDIMSampler
from evaluation import check_peak_labels_available, extract_peak_labels, peak_metrics


def setup_logging(log_level: str = "INFO"):
    """Setup console logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_config(config_path: str, overrides: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML config with optional dot-notation overrides.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if overrides:
        for override in overrides.split(','):
            if ':' not in override:
                continue
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


def create_evaluation_dataset(cfg: Dict[str, Any]) -> Tuple[DataLoader, ABRDataset]:
    """
    Create evaluation dataset and loader.
    """
    # Use test_csv if available, otherwise val_csv
    data_path = cfg['data'].get('test_csv', '')
    if not data_path or not os.path.exists(data_path):
        data_path = cfg['data']['val_csv']
        logging.info(f"Using validation set for evaluation: {data_path}")
    else:
        logging.info(f"Using test set for evaluation: {data_path}")
    
    # Load dataset
    dataset = ABRDataset(
        data_path=data_path,
        normalize_signal=True,
        normalize_static=True
    )
    
    # Verify dataset properties
    assert dataset.sequence_length == cfg['data']['sequence_length'], \
        f"Dataset sequence_length {dataset.sequence_length} != config {cfg['data']['sequence_length']}"
    assert dataset.static_dim == cfg['model']['static_dim'], \
        f"Dataset static_dim {dataset.static_dim} != config {cfg['model']['static_dim']}"
    
    # For evaluation, we can use the full dataset or create a validation split
    # If we're using the same file as training, create a split
    if data_path == cfg['data']['val_csv']:
        # Create splits to get validation portion
        _, val_dataset, _ = create_stratified_datasets(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15, 
            test_ratio=0.15,
            random_state=cfg.get('seed', 42)
        )
        eval_dataset = val_dataset
    else:
        eval_dataset = dataset
    
    # Apply max_samples limit if specified
    max_samples = cfg.get('advanced', {}).get('max_samples')
    if max_samples and len(eval_dataset) > max_samples:
        indices = torch.randperm(len(eval_dataset))[:max_samples]
        eval_dataset = torch.utils.data.Subset(eval_dataset, indices)
        logging.info(f"Limited evaluation to {max_samples} samples")
    
    # Create data loader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg['loader']['batch_size'],
        shuffle=False,
        num_workers=cfg['loader']['num_workers'],
        pin_memory=cfg['loader']['pin_memory'],
        collate_fn=abr_collate_fn,
        drop_last=False
    )
    
    logging.info(f"✓ Created evaluation dataset: {len(eval_dataset)} samples")
    
    return eval_loader, dataset


def load_model_and_checkpoint(cfg: Dict[str, Any], device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model and checkpoint with optional EMA weights.
    """
    # Create model
    model = ABRTransformerGenerator(**cfg['model'])
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = cfg['checkpoint']['path']
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Apply EMA weights if requested and available
    if cfg['checkpoint'].get('use_ema', False) and 'ema_state_dict' in checkpoint:
        ema_state = checkpoint['ema_state_dict']
        if 'shadow' in ema_state:
            logging.info("Applying EMA weights for evaluation")
            # Copy EMA shadow weights to model
            for name, param in model.named_parameters():
                if name in ema_state['shadow']:
                    param.data.copy_(ema_state['shadow'][name])
        else:
            logging.warning("EMA weights requested but not found in checkpoint")
    
    model.eval()
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"✓ Loaded model with {total_params:,} parameters")
    
    return model, checkpoint


def evaluate_reconstruction(
    model: nn.Module,
    eval_loader: DataLoader,
    noise_schedule: Dict[str, torch.Tensor],
    dataset: ABRDataset,
    cfg: Dict[str, Any],
    device: torch.device
) -> Tuple[List[Dict[str, Any]], List[torch.Tensor], List[torch.Tensor]]:
    """
    Evaluate reconstruction mode: denoise from x_t back to x_0.
    """
    logging.info("Starting reconstruction evaluation...")
    
    model.eval()
    all_metrics = []
    all_ref_signals = []
    all_gen_signals = []
    
    # Metrics configuration
    use_stft = cfg['metrics']['use_stft']
    use_dtw = cfg['metrics']['use_dtw']
    stft_params = cfg['metrics'].get('stft', {})
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Reconstruction")):
            x0 = batch['x0'].to(device)  # [B, 1, T]
            stat = batch['stat'].to(device)  # [B, S]
            meta = batch['meta']
            
            B, C, T = x0.shape
            
            # Sample timesteps based on strategy
            recon_cfg = cfg.get('advanced', {}).get('reconstruction', {})
            timestep_strategy = recon_cfg.get('timestep_strategy', 'uniform')
            
            if timestep_strategy == 'fixed':
                fixed_t = recon_cfg.get('fixed_timestep', 500)
                t = torch.full((B,), fixed_t, device=device, dtype=torch.long)
            elif timestep_strategy == 'uniform':
                t = torch.randint(0, cfg['diffusion']['num_train_steps'], (B,), device=device)
            else:  # random_subset or other
                t = torch.randint(0, cfg['diffusion']['num_train_steps'], (B,), device=device)
            
            # Add noise
            noise = torch.randn_like(x0)
            x_t, v_target = q_sample_vpred(x0, t, noise, noise_schedule)
            
            # Model prediction (reconstruction)
            v_pred = model(x_t, static_params=stat, timesteps=t)["signal"]
            
            # Reconstruct x0 from v-prediction
            from utils.schedules import predict_x0_from_v
            x0_recon = predict_x0_from_v(x_t, v_pred, t, noise_schedule)
            
            # Denormalize for metrics and visualization
            x0_denorm = dataset.denormalize_signal(x0)
            x0_recon_denorm = dataset.denormalize_signal(x0_recon)
            
            # Compute per-sample metrics
            per_sample_metrics = compute_per_sample_metrics(
                x0_recon_denorm, x0_denorm, use_stft, use_dtw, stft_params
            )
            
            # Add metadata to metrics
            for i, sample_metrics in enumerate(per_sample_metrics):
                sample_metrics.update({
                    'mode': 'reconstruction',
                    'sample_id': meta[i].get('sample_idx', batch_idx * cfg['loader']['batch_size'] + i),
                    'timestep': t[i].item()
                })
                # Add static parameters
                for j, param_name in enumerate(dataset.static_names):
                    sample_metrics[f'static_{param_name}'] = stat[i, j].item()
            
            all_metrics.extend(per_sample_metrics)
            all_ref_signals.append(x0_denorm.cpu())
            all_gen_signals.append(x0_recon_denorm.cpu())
    
    logging.info(f"✓ Reconstruction evaluation completed: {len(all_metrics)} samples")
    
    return all_metrics, all_ref_signals, all_gen_signals


def evaluate_generation(
    model: nn.Module,
    eval_loader: DataLoader,
    sampler: DDIMSampler,
    dataset: ABRDataset,
    cfg: Dict[str, Any],
    device: torch.device
) -> Tuple[List[Dict[str, Any]], List[torch.Tensor], List[torch.Tensor]]:
    """
    Evaluate conditional generation: sample from noise conditioned on static params.
    """
    logging.info("Starting generation evaluation...")
    
    model.eval()
    all_metrics = []
    all_ref_signals = []
    all_gen_signals = []
    
    # Metrics configuration
    use_stft = cfg['metrics']['use_stft']
    use_dtw = cfg['metrics']['use_dtw']
    stft_params = cfg['metrics'].get('stft', {})
    
    # Generation configuration
    gen_cfg = cfg.get('advanced', {}).get('generation', {})
    cfg_scale = gen_cfg.get('cfg_scale', 1.0)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Generation")):
            x0 = batch['x0'].to(device)  # [B, 1, T] - reference for comparison
            stat = batch['stat'].to(device)  # [B, S] - conditioning
            meta = batch['meta']
            
            B, C, T = x0.shape
            
            # Generate samples conditioned on static parameters
            x_gen = sampler.sample_conditioned(
                static_params=stat,
                steps=cfg['diffusion']['sample_steps'],
                eta=cfg['diffusion']['ddim_eta'],
                cfg_scale=cfg_scale,
                progress=False
            )
            
            # Denormalize for metrics and visualization
            x0_denorm = dataset.denormalize_signal(x0)
            x_gen_denorm = dataset.denormalize_signal(x_gen)
            
            # Compute per-sample metrics (comparing generated to reference)
            per_sample_metrics = compute_per_sample_metrics(
                x_gen_denorm, x0_denorm, use_stft, use_dtw, stft_params
            )
            
            # Add metadata to metrics
            for i, sample_metrics in enumerate(per_sample_metrics):
                sample_metrics.update({
                    'mode': 'generation',
                    'sample_id': meta[i].get('sample_idx', batch_idx * cfg['loader']['batch_size'] + i),
                    'cfg_scale': cfg_scale
                })
                # Add static parameters
                for j, param_name in enumerate(dataset.static_names):
                    sample_metrics[f'static_{param_name}'] = stat[i, j].item()
            
            all_metrics.extend(per_sample_metrics)
            all_ref_signals.append(x0_denorm.cpu())
            all_gen_signals.append(x_gen_denorm.cpu())
    
    logging.info(f"✓ Generation evaluation completed: {len(all_metrics)} samples")
    
    return all_metrics, all_ref_signals, all_gen_signals


def create_visualizations(
    ref_signals: List[torch.Tensor],
    gen_signals: List[torch.Tensor], 
    metrics: List[Dict[str, Any]],
    mode: str,
    cfg: Dict[str, Any],
    writer: SummaryWriter,
    epoch: int = 0
):
    """
    Create and log visualizations to TensorBoard.
    """
    if not cfg['report']['tensorboard']['write_figures']:
        return
    
    logging.info(f"Creating visualizations for {mode} mode...")
    
    # Concatenate all signals
    all_ref = torch.cat(ref_signals, dim=0)  # [N, 1, T]
    all_gen = torch.cat(gen_signals, dim=0)  # [N, 1, T]
    
    # Sort by MSE for best/worst selection
    mse_values = [m['mse'] for m in metrics]
    sorted_indices = np.argsort(mse_values)
    
    topk = cfg['report']['save_topk_examples']
    best_indices = sorted_indices[:topk//2]
    worst_indices = sorted_indices[-topk//2:]
    
    try:
        # Best samples overlay
        best_ref = all_ref[best_indices]
        best_gen = all_gen[best_indices]
        best_titles = [f"Best #{i+1} (MSE: {mse_values[idx]:.4f})" for i, idx in enumerate(best_indices)]
        
        fig_best = overlay_waveforms(best_ref, best_gen, best_titles)
        writer.add_figure(f'eval/{mode}/overlay_best', fig_best, epoch)
        close_figure(fig_best)
        
        # Worst samples overlay
        worst_ref = all_ref[worst_indices]
        worst_gen = all_gen[worst_indices]
        worst_titles = [f"Worst #{i+1} (MSE: {mse_values[idx]:.4f})" for i, idx in enumerate(worst_indices)]
        
        fig_worst = overlay_waveforms(worst_ref, worst_gen, worst_titles)
        writer.add_figure(f'eval/{mode}/overlay_worst', fig_worst, epoch)
        close_figure(fig_worst)
        
        # Error curves for best samples
        fig_error = error_curve(best_ref, best_gen, best_titles[:4])
        writer.add_figure(f'eval/{mode}/error_curves', fig_error, epoch)
        close_figure(fig_error)
        
        # Spectrograms if enabled
        if cfg['viz']['spectrogram_plots']:
            spec_params = cfg['viz']['spectrogram_params']
            
            fig_spec_best = spectrograms(best_ref[:4], **spec_params)
            writer.add_figure(f'eval/{mode}/spectrogram_ref', fig_spec_best, epoch)
            close_figure(fig_spec_best)
            
            fig_spec_gen = spectrograms(best_gen[:4], **spec_params)
            writer.add_figure(f'eval/{mode}/spectrogram_gen', fig_spec_gen, epoch)
            close_figure(fig_spec_gen)
        
        # Scatter plots
        if cfg['viz']['scatter_plots']:
            # Correlation vs DTW
            if 'dtw' in metrics[0] and not np.isnan(metrics[0]['dtw']):
                corr_values = [m['corr'] for m in metrics]
                dtw_values = [m['dtw'] for m in metrics]
                
                fig_scatter = scatter_xy(
                    np.array(corr_values), np.array(dtw_values),
                    'Correlation', 'DTW Distance', f'{mode.title()} - Correlation vs DTW'
                )
                writer.add_figure(f'eval/{mode}/scatter_corr_dtw', fig_scatter, epoch)
                close_figure(fig_scatter)
        
        logging.info(f"✓ Created visualizations for {mode}")
        
    except Exception as e:
        logging.warning(f"Failed to create some visualizations: {e}")


def save_results(
    metrics: List[Dict[str, Any]], 
    mode: str, 
    cfg: Dict[str, Any], 
    checkpoint_info: Dict[str, Any]
):
    """
    Save evaluation results to CSV and JSON.
    """
    output_dir = Path(cfg['report']['out_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-sample metrics as CSV
    if cfg['report']['save_csv']:
        csv_path = output_dir / f"{cfg['report']['csv_prefix']}_{mode}_metrics.csv"
        df = pd.DataFrame(metrics)
        df.to_csv(csv_path, index=False)
        logging.info(f"✓ Saved per-sample metrics: {csv_path}")
    
    # Compute summary statistics
    numeric_columns = ['mse', 'l1', 'corr', 'snr_db']
    if 'stft_l1' in metrics[0]:
        numeric_columns.append('stft_l1')
    if 'dtw' in metrics[0]:
        numeric_columns.append('dtw')
    
    summary = {}
    for col in numeric_columns:
        values = [m[col] for m in metrics if not np.isnan(m.get(col, np.nan))]
        if values:
            summary[f'{col}_mean'] = np.mean(values)
            summary[f'{col}_std'] = np.std(values)
            summary[f'{col}_median'] = np.median(values)
            summary[f'{col}_min'] = np.min(values)
            summary[f'{col}_max'] = np.max(values)
        else:
            summary[f'{col}_mean'] = float('nan')
            summary[f'{col}_std'] = float('nan')
            summary[f'{col}_median'] = float('nan')
            summary[f'{col}_min'] = float('nan')
            summary[f'{col}_max'] = float('nan')
    
    # Add metadata
    summary.update({
        'mode': mode,
        'num_samples': len(metrics),
        'checkpoint_path': cfg['checkpoint']['path'],
        'checkpoint_epoch': checkpoint_info.get('epoch', 'unknown'),
        'model_config': cfg['model'],
        'evaluation_config': cfg
    })
    
    # Save summary as JSON
    summary_path = output_dir / f"{cfg['report']['csv_prefix']}_{mode}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logging.info(f"✓ Saved summary: {summary_path}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY - {mode.upper()} MODE")
    print(f"{'='*60}")
    print(f"Samples: {len(metrics)}")
    print(f"Checkpoint: {cfg['checkpoint']['path']}")
    print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"{'-'*60}")
    
    for col in numeric_columns:
        if f'{col}_mean' in summary and not np.isnan(summary[f'{col}_mean']):
            print(f"{col:<15} {summary[f'{col}_mean']:<12.6f} {summary[f'{col}_std']:<12.6f} "
                  f"{summary[f'{col}_min']:<12.6f} {summary[f'{col}_max']:<12.6f}")
    
    return summary


def log_scalars_to_tensorboard(metrics: List[Dict[str, Any]], mode: str, 
                              writer: SummaryWriter, epoch: int = 0):
    """
    Log scalar metrics to TensorBoard.
    """
    # Compute means for scalar logging
    numeric_columns = ['mse', 'l1', 'corr', 'snr_db']
    if 'stft_l1' in metrics[0]:
        numeric_columns.append('stft_l1')
    if 'dtw' in metrics[0]:
        numeric_columns.append('dtw')
    
    for col in numeric_columns:
        values = [m[col] for m in metrics if not np.isnan(m.get(col, np.nan))]
        if values:
            writer.add_scalar(f'eval/{mode}/{col}_mean', np.mean(values), epoch)
            writer.add_scalar(f'eval/{mode}/{col}_std', np.std(values), epoch)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate ABR Transformer")
    parser.add_argument("--config", type=str, default="configs/eval.yaml", help="Config file path")
    parser.add_argument("--override", type=str, default="", help="Config overrides (dotted notation)")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config, args.override)
    
    # Setup
    setup_logging(cfg.get('logging', {}).get('console_level', 'INFO'))
    
    # Set seed
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    
    # Device
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create evaluation dataset
    eval_loader, dataset = create_evaluation_dataset(cfg)
    
    # Load model and checkpoint
    model, checkpoint = load_model_and_checkpoint(cfg, device)
    
    # Setup diffusion components
    noise_schedule = prepare_noise_schedule(cfg['diffusion']['num_train_steps'], device)
    sampler = DDIMSampler(model, cfg['diffusion']['num_train_steps'], device)
    
    # TensorBoard writer
    log_dir = Path(cfg['log_dir']) / cfg['exp_name']
    writer = SummaryWriter(log_dir)
    
    # Log configuration
    if cfg['report']['tensorboard']['write_scalars']:
        config_text = yaml.dump(cfg, default_flow_style=False)
        writer.add_text('config', f"```yaml\n{config_text}\n```", 0)
    
    # Check for peak labels
    has_peak_labels = check_peak_labels_available(dataset)
    if has_peak_labels:
        logging.info("Peak labels detected - will compute peak metrics")
    else:
        logging.info("Peak labels not available - skipping peak metrics")
    
    logging.info("="*60)
    logging.info("STARTING EVALUATION")
    logging.info("="*60)
    
    # Run evaluations
    all_results = {}
    
    # Reconstruction evaluation
    if cfg['modes']['reconstruction']:
        logging.info("Running reconstruction evaluation...")
        recon_metrics, recon_ref, recon_gen = evaluate_reconstruction(
            model, eval_loader, noise_schedule, dataset, cfg, device
        )
        
        # Save results
        recon_summary = save_results(recon_metrics, 'reconstruction', cfg, checkpoint)
        all_results['reconstruction'] = recon_summary
        
        # TensorBoard logging
        if cfg['report']['tensorboard']['write_scalars']:
            log_scalars_to_tensorboard(recon_metrics, 'reconstruction', writer)
        
        # Visualizations
        create_visualizations(recon_ref, recon_gen, recon_metrics, 'reconstruction', cfg, writer)
    
    # Generation evaluation
    if cfg['modes']['generation']:
        logging.info("Running generation evaluation...")
        gen_metrics, gen_ref, gen_gen = evaluate_generation(
            model, eval_loader, sampler, dataset, cfg, device
        )
        
        # Save results
        gen_summary = save_results(gen_metrics, 'generation', cfg, checkpoint)
        all_results['generation'] = gen_summary
        
        # TensorBoard logging
        if cfg['report']['tensorboard']['write_scalars']:
            log_scalars_to_tensorboard(gen_metrics, 'generation', writer)
        
        # Visualizations
        create_visualizations(gen_ref, gen_gen, gen_metrics, 'generation', cfg, writer)
    
    writer.close()
    
    logging.info("="*60)
    logging.info("EVALUATION COMPLETED")
    logging.info("="*60)
    logging.info(f"Results saved to: {cfg['report']['out_dir']}")
    logging.info(f"TensorBoard logs: {log_dir}")
    
    return all_results


if __name__ == "__main__":
    main()
