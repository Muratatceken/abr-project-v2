#!/usr/bin/env python3
"""
Evaluation pipeline for ABR Transformer V2.

Supports reconstruction and conditional generation evaluation with signal quality
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
import matplotlib.pyplot as plt
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
from utils.schedules import predict_x0_from_v
from inference import DDIMSampler


def setup_logging(log_level: str = "INFO"):
    """Setup console logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_config(config_path: str, overrides: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config with optional dot-notation overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if overrides:
        for override in overrides.split(','):
            if ':' not in override:
                continue
            key, value = override.split(':', 1)
            key = key.strip()
            value = value.strip()
            try:
                value = yaml.safe_load(value)
            except Exception:
                pass
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def create_evaluation_dataset(cfg: Dict[str, Any]) -> Tuple[DataLoader, ABRDataset]:
    """Create evaluation dataset and loader."""
    data_path = cfg['dataset']['data_path']
    batch_size = cfg['dataset']['batch_size']
    num_workers = cfg['dataset']['num_workers']
    pin_memory = cfg['dataset']['pin_memory']

    logging.info(f"Loading dataset from: {data_path}")

    dataset = ABRDataset(
        data_path=data_path,
        normalize_signal=False,   # preprocessing already normalised
        normalize_static=True,
        return_peak_labels=False,
        transform=None,
    )

    assert dataset.sequence_length == cfg['model']['sequence_length'], \
        f"Dataset seq_len {dataset.sequence_length} != config {cfg['model']['sequence_length']}"

    # Use validation split for evaluation
    _, val_dataset, _ = create_stratified_datasets(
        dataset,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_state=cfg['evaluation']['seed'],
    )

    # Optional sample limit
    max_samples = cfg['evaluation'].get('num_samples')
    if max_samples and len(val_dataset) > max_samples:
        indices = torch.randperm(len(val_dataset))[:max_samples]
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
        logging.info(f"Limited evaluation to {max_samples} samples")

    eval_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=abr_collate_fn,
        drop_last=False,
    )

    logging.info(f"✓ Created evaluation dataset: {len(val_dataset)} samples")
    return eval_loader, dataset


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model_and_checkpoint(
    cfg: Dict[str, Any], device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model and checkpoint, applying EMA weights when available."""
    checkpoint_path = cfg['model']['checkpoint_path']
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Prefer config stored in checkpoint, fall back to eval.yaml
    ckpt_cfg = checkpoint.get('config', {}).get('model', {})

    model_config = {
        'input_channels': ckpt_cfg.get('input_channels', cfg['model']['input_channels']),
        'static_dim': ckpt_cfg.get('static_dim', cfg['model']['static_dim']),
        'sequence_length': ckpt_cfg.get('sequence_length', cfg['model']['sequence_length']),
        'd_model': ckpt_cfg.get('d_model', cfg['model']['d_model']),
        'n_layers': ckpt_cfg.get('n_layers', cfg['model']['n_layers']),
        'n_heads': ckpt_cfg.get('n_heads', cfg['model']['n_heads']),
        'ff_mult': ckpt_cfg.get('ff_mult', cfg['model']['ff_mult']),
        'dropout': ckpt_cfg.get('dropout', cfg['model']['dropout']),
        'num_classes': ckpt_cfg.get('num_classes', cfg['model'].get('num_classes', 5)),
        'intensity_emb_dim': ckpt_cfg.get('intensity_emb_dim', cfg['model'].get('intensity_emb_dim', 64)),
        'aux_static_dim': ckpt_cfg.get('aux_static_dim', cfg['model'].get('aux_static_dim', 3)),
    }

    model = ABRTransformerGenerator(**model_config)
    model = model.to(device)

    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logging.info("✓ Loaded checkpoint (strict)")
    except RuntimeError:
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing:
            logging.warning(f"Missing keys: {len(missing)}")
        if unexpected:
            logging.warning(f"Unexpected keys: {len(unexpected)}")
        logging.info("✓ Loaded checkpoint (non-strict)")

    # EMA weights
    if cfg['model'].get('use_ema', True) and 'ema_state_dict' in checkpoint:
        ema_state = checkpoint['ema_state_dict']
        if 'shadow' in ema_state:
            logging.info("Applying EMA weights")
            for name, param in model.named_parameters():
                if name in ema_state['shadow']:
                    param.data.copy_(ema_state['shadow'][name])

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"✓ Model: {total_params:,} parameters")

    return model, checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack_conditioning(batch, device):
    """Extract conditioning tensors from a batch dict."""
    intensity = batch['intensity'].to(device)
    aux_static = batch['aux_static'].to(device)
    class_label = batch['class_label'].to(device)
    return intensity, aux_static, class_label


def _validate_signals(signals: torch.Tensor, label: str, batch_idx: int):
    """NaN/Inf check and repair."""
    if torch.any(torch.isnan(signals)) or torch.any(torch.isinf(signals)):
        logging.warning(f"{label} batch {batch_idx}: invalid values — replaced with 0")
        return torch.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    return signals


# ---------------------------------------------------------------------------
# Reconstruction Evaluation
# ---------------------------------------------------------------------------

def evaluate_reconstruction(
    model: nn.Module,
    eval_loader: DataLoader,
    noise_schedule: Dict[str, torch.Tensor],
    dataset: ABRDataset,
    cfg: Dict[str, Any],
    device: torch.device,
    training_T: int = 1000,
) -> Tuple[List[Dict[str, Any]], List[torch.Tensor], List[torch.Tensor]]:
    """Evaluate reconstruction: denoise x_t → x_0."""
    logging.info("Starting reconstruction evaluation...")

    model.eval()
    all_metrics: List[Dict[str, Any]] = []
    all_ref: List[torch.Tensor] = []
    all_gen: List[torch.Tensor] = []

    use_stft = cfg['metrics']['signal'].get('stft_loss', True)
    use_dtw = cfg['metrics']['signal'].get('dtw', False)
    stft_params = cfg['metrics'].get('stft', {})

    recon_cfg = cfg.get('evaluation', {}).get('reconstruction', {})
    timestep_strategy = recon_cfg.get('timestep_strategy', 'uniform')

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Reconstruction")):
            x0 = batch['x0'].to(device)
            intensity, aux_static, class_label = _unpack_conditioning(batch, device)
            meta = batch.get('meta', [{}] * x0.shape[0])
            B = x0.shape[0]

            # Timesteps — always sampled from [0, training_T)
            if timestep_strategy == 'fixed':
                fixed_t = recon_cfg.get('fixed_timestep', 500)
                t = torch.full((B,), fixed_t, device=device, dtype=torch.long)
            else:
                t = torch.randint(0, training_T, (B,), device=device)

            # Forward diffusion
            noise = torch.randn_like(x0)
            x_t, v_target = q_sample_vpred(x0, t, noise, noise_schedule)

            # Model prediction
            model_output = model(
                x_t,
                intensity=intensity,
                aux_static=aux_static,
                class_label=class_label,
                timesteps=t,
            )
            v_pred = model_output['signal']

            # Reconstruct x0
            x0_recon = predict_x0_from_v(x_t, v_pred, t, noise_schedule)

            # Denormalize
            x0_denorm = _validate_signals(
                dataset.denormalize_signal(x0), "target", batch_idx
            )
            x0_recon_denorm = _validate_signals(
                dataset.denormalize_signal(x0_recon), "recon", batch_idx
            )

            # Per-sample metrics
            try:
                per_sample = compute_per_sample_metrics(
                    x0_recon_denorm, x0_denorm, use_stft, use_dtw, stft_params
                )
            except Exception as e:
                logging.error(f"Batch {batch_idx} metrics failed: {e}")
                per_sample = [{'mse': float('nan'), 'l1': float('nan'),
                               'corr': float('nan'), 'snr_db': float('nan')}] * B

            for i, sm in enumerate(per_sample):
                sm.update({
                    'mode': 'reconstruction',
                    'sample_id': meta[i].get('sample_idx', batch_idx * cfg['dataset']['batch_size'] + i)
                        if isinstance(meta[i], dict) else batch_idx * cfg['dataset']['batch_size'] + i,
                    'timestep': t[i].item(),
                    'intensity': intensity[i].item(),
                    'class_label': class_label[i].item(),
                })

            all_metrics.extend(per_sample)
            all_ref.append(x0_denorm.cpu())
            all_gen.append(x0_recon_denorm.cpu())

    logging.info(f"✓ Reconstruction: {len(all_metrics)} samples")
    return all_metrics, all_ref, all_gen


# ---------------------------------------------------------------------------
# Generation Evaluation
# ---------------------------------------------------------------------------

def evaluate_generation(
    model: nn.Module,
    eval_loader: DataLoader,
    sampler: DDIMSampler,
    dataset: ABRDataset,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[torch.Tensor], List[torch.Tensor]]:
    """Evaluate conditional generation: noise → x_0."""
    logging.info("Starting generation evaluation...")

    model.eval()
    all_metrics: List[Dict[str, Any]] = []
    all_ref: List[torch.Tensor] = []
    all_gen: List[torch.Tensor] = []

    use_stft = cfg['metrics']['signal'].get('stft_loss', True)
    use_dtw = cfg['metrics']['signal'].get('dtw', False)
    stft_params = cfg['metrics'].get('stft', {})

    diff_cfg = cfg.get('diffusion', {})
    gen_cfg = cfg.get('evaluation', {}).get('generation', {})
    sample_steps = gen_cfg.get('num_steps', diff_cfg.get('sample_steps', 50))
    eta = diff_cfg.get('ddim_eta', 0.0)
    cfg_scale = diff_cfg.get('cfg_scale', 2.0)
    class_cfg_scale = diff_cfg.get('class_cfg_scale', 1.5)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Generation")):
            x0 = batch['x0'].to(device)
            intensity, aux_static, class_label = _unpack_conditioning(batch, device)
            meta = batch.get('meta', [{}] * x0.shape[0])
            B = x0.shape[0]

            # Generate with same conditioning as real data
            x_gen = sampler.sample_conditioned(
                intensity=intensity,
                aux_static=aux_static,
                class_label=class_label,
                steps=sample_steps,
                eta=eta,
                cfg_scale=cfg_scale,
                class_cfg_scale=class_cfg_scale,
                progress=False,
            )

            # Denormalize
            x0_denorm = _validate_signals(
                dataset.denormalize_signal(x0), "target", batch_idx
            )
            x_gen_denorm = _validate_signals(
                dataset.denormalize_signal(x_gen), "generated", batch_idx
            )

            # Metrics
            try:
                per_sample = compute_per_sample_metrics(
                    x_gen_denorm, x0_denorm, use_stft, use_dtw, stft_params
                )
            except Exception as e:
                logging.error(f"Batch {batch_idx} metrics failed: {e}")
                per_sample = [{'mse': float('nan'), 'l1': float('nan'),
                               'corr': float('nan'), 'snr_db': float('nan')}] * B

            for i, sm in enumerate(per_sample):
                sm.update({
                    'mode': 'generation',
                    'sample_id': meta[i].get('sample_idx', batch_idx * cfg['dataset']['batch_size'] + i)
                        if isinstance(meta[i], dict) else batch_idx * cfg['dataset']['batch_size'] + i,
                    'cfg_scale': cfg_scale,
                    'intensity': intensity[i].item(),
                    'class_label': class_label[i].item(),
                })

            all_metrics.extend(per_sample)
            all_ref.append(x0_denorm.cpu())
            all_gen.append(x_gen_denorm.cpu())

    logging.info(f"✓ Generation: {len(all_metrics)} samples")
    return all_metrics, all_ref, all_gen


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def create_visualizations(
    ref_signals: List[torch.Tensor],
    gen_signals: List[torch.Tensor],
    metrics: List[Dict[str, Any]],
    mode: str,
    cfg: Dict[str, Any],
    writer: SummaryWriter,
    epoch: int = 0,
):
    """Create and log visualisations to TensorBoard."""
    if not cfg['tensorboard']['log_plots']:
        return

    logging.info(f"Creating visualisations for {mode}...")

    all_ref = torch.cat(ref_signals, dim=0)
    all_gen = torch.cat(gen_signals, dim=0)

    mse_values = [m['mse'] for m in metrics]
    sorted_idx = np.argsort(mse_values)

    topk = cfg['report'].get('save_topk_examples', 10)
    best = sorted_idx[:topk // 2]
    worst = sorted_idx[-topk // 2:]

    try:
        # Best overlay
        fig = overlay_waveforms(
            all_ref[best], all_gen[best],
            [f"Best #{i+1} (MSE:{mse_values[idx]:.4f})" for i, idx in enumerate(best)],
        )
        writer.add_figure(f'eval/{mode}/overlay_best', fig, epoch)
        close_figure(fig)

        # Worst overlay
        fig = overlay_waveforms(
            all_ref[worst], all_gen[worst],
            [f"Worst #{i+1} (MSE:{mse_values[idx]:.4f})" for i, idx in enumerate(worst)],
        )
        writer.add_figure(f'eval/{mode}/overlay_worst', fig, epoch)
        close_figure(fig)

        # Error curves
        fig = error_curve(all_ref[best[:4]], all_gen[best[:4]])
        writer.add_figure(f'eval/{mode}/error_curves', fig, epoch)
        close_figure(fig)

        # Spectrograms
        if cfg['report'].get('save_spectrograms', False):
            spec_params = cfg.get('visualization', {}).get('spectrogram_params', {})
            fig = spectrograms(all_ref[best[:4]], **spec_params)
            writer.add_figure(f'eval/{mode}/spectrogram_ref', fig, epoch)
            close_figure(fig)
            fig = spectrograms(all_gen[best[:4]], **spec_params)
            writer.add_figure(f'eval/{mode}/spectrogram_gen', fig, epoch)
            close_figure(fig)

        # Scatter: MSE vs correlation
        if len(metrics) > 1:
            mse_arr = np.array([m.get('mse', 0) for m in metrics])
            corr_arr = np.array([m.get('corr', 0) for m in metrics])
            fig = scatter_xy(mse_arr, corr_arr, 'MSE', 'Correlation',
                             f'{mode.title()} MSE vs Correlation')
            writer.add_figure(f'eval/{mode}/metrics_correlation', fig, epoch)
            close_figure(fig)

        # Summary bar chart
        if metrics:
            summary = {}
            for key in ['mse', 'l1', 'corr', 'snr_db', 'stft_l1']:
                vals = [m.get(key, 0) for m in metrics if key in m]
                if vals:
                    summary[key] = (np.mean(vals), np.std(vals))
            if summary:
                fig = metrics_summary_plot(summary, mode)
                writer.add_figure(f'eval/{mode}/metrics_summary', fig, epoch)
                close_figure(fig)

    except Exception as e:
        logging.warning(f"Visualisation error: {e}")


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def log_scalars_to_tensorboard(
    metrics: List[Dict[str, Any]], mode: str, writer: SummaryWriter, epoch: int = 0
):
    """Log scalar metrics to TensorBoard."""
    for col in ['mse', 'l1', 'corr', 'snr_db', 'stft_l1', 'dtw']:
        vals = [m[col] for m in metrics if col in m and not np.isnan(m.get(col, np.nan))]
        if vals:
            writer.add_scalar(f'eval/{mode}/{col}_mean', np.mean(vals), epoch)
            writer.add_scalar(f'eval/{mode}/{col}_std', np.std(vals), epoch)


def save_results(
    metrics: List[Dict[str, Any]], mode: str, cfg: Dict[str, Any],
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    """Save evaluation results and compute summary statistics."""
    output_dir = Path(cfg['output']['save_dir']) / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics)

    if cfg['output'].get('save_samples', False):
        fmt = cfg['output'].get('save_format', 'json')
        path = output_dir / f"{mode}_detailed.{fmt}"
        if fmt == 'json':
            df.to_json(path, orient='records', indent=2)
        else:
            df.to_pickle(path)

    # Summary
    summary: Dict[str, Any] = {
        'mode': mode,
        'num_samples': len(metrics),
        'checkpoint_epoch': checkpoint.get('epoch', 'unknown'),
    }

    for col in ['mse', 'l1', 'corr', 'snr_db', 'stft_l1', 'dtw']:
        vals = [m[col] for m in metrics if col in m and not np.isnan(m.get(col, np.nan))]
        if vals:
            summary[f'{col}_mean'] = float(np.mean(vals))
            summary[f'{col}_std'] = float(np.std(vals))
            summary[f'{col}_min'] = float(np.min(vals))
            summary[f'{col}_max'] = float(np.max(vals))

    # Per-class breakdown
    class_ids = set(m.get('class_label') for m in metrics if 'class_label' in m)
    if class_ids:
        class_names = {0: 'NORMAL', 1: 'SNIK', 2: 'ITIK', 3: 'TOTAL', 4: 'NOROPATI'}
        for cid in sorted(class_ids):
            cls_metrics = [m for m in metrics if m.get('class_label') == cid]
            if cls_metrics:
                vals = [m['mse'] for m in cls_metrics if not np.isnan(m.get('mse', np.nan))]
                if vals:
                    name = class_names.get(int(cid), str(cid))
                    summary[f'mse_{name}_mean'] = float(np.mean(vals))
                    summary[f'mse_{name}_n'] = len(vals)

    summary_file = output_dir / f"{mode}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"✓ Saved {mode} results to {output_dir}")
    return summary


def save_evaluation_summary(all_results: Dict[str, Any], cfg: Dict[str, Any]):
    """Save unified evaluation summary."""
    output_dir = Path(cfg['output']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {},
    }

    for mode, results in all_results.items():
        if not isinstance(results, dict):
            continue
        for key, val in results.items():
            if isinstance(val, (int, float)):
                evaluation_summary['metrics'][f'{mode}.{key}'] = val

    path = output_dir / "evaluation_summary.json"
    with open(path, 'w') as f:
        json.dump(evaluation_summary, f, indent=2, default=str)
    logging.info(f"✓ Summary: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate ABR Transformer V2")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--override", default="")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    setup_logging(cfg.get('logging', {}).get('console_level', 'INFO'))

    torch.manual_seed(cfg['evaluation']['seed'])
    np.random.seed(cfg['evaluation']['seed'])

    device = torch.device(cfg['model']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    # Data
    eval_loader, dataset = create_evaluation_dataset(cfg)

    # Model
    model, checkpoint = load_model_and_checkpoint(cfg, device)

    # Diffusion — always use training T for noise schedule
    training_T = cfg.get('diffusion', {}).get('num_train_steps', 1000)
    noise_schedule = prepare_noise_schedule(training_T, device)
    sampler = DDIMSampler(model, training_T, device)

    # TensorBoard
    writer = None
    if cfg.get('tensorboard', {}).get('enabled', True):
        log_dir = Path(cfg['tensorboard']['log_dir'])
        writer = SummaryWriter(log_dir)
        config_text = yaml.dump(cfg, default_flow_style=False)
        writer.add_text('config', f"```yaml\n{config_text}\n```", 0)

    logging.info("=" * 60)
    logging.info("STARTING EVALUATION")
    logging.info("=" * 60)

    all_results: Dict[str, Any] = {}
    eval_mode = cfg['evaluation'].get('mode', 'both')

    # --- Reconstruction ---
    if eval_mode in ('reconstruction', 'both'):
        recon_metrics, recon_ref, recon_gen = evaluate_reconstruction(
            model, eval_loader, noise_schedule, dataset, cfg, device, training_T
        )
        summary = save_results(recon_metrics, 'reconstruction', cfg, checkpoint)
        all_results['reconstruction'] = summary

        if writer:
            log_scalars_to_tensorboard(recon_metrics, 'reconstruction', writer)
            create_visualizations(recon_ref, recon_gen, recon_metrics,
                                  'reconstruction', cfg, writer)

    # --- Generation ---
    if eval_mode in ('generation', 'both'):
        gen_metrics, gen_ref, gen_gen = evaluate_generation(
            model, eval_loader, sampler, dataset, cfg, device
        )
        summary = save_results(gen_metrics, 'generation', cfg, checkpoint)
        all_results['generation'] = summary

        if writer:
            log_scalars_to_tensorboard(gen_metrics, 'generation', writer)
            create_visualizations(gen_ref, gen_gen, gen_metrics,
                                  'generation', cfg, writer)

    if writer:
        writer.close()

    save_evaluation_summary(all_results, cfg)

    logging.info("=" * 60)
    logging.info("EVALUATION COMPLETED")
    logging.info("=" * 60)
    logging.info(f"Results: {cfg['output']['save_dir']}")

    return all_results


if __name__ == "__main__":
    main()
