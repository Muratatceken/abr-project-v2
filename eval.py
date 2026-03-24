#!/usr/bin/env python3
"""
Publication-grade evaluation pipeline for ABR Transformer V2.

Produces comprehensive metrics, per-class stratified analysis, intensity-binned
breakdowns, and publication-quality figures (300 DPI, serif fonts, colorblind-safe).
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy import stats as scipy_stats

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
from evaluation.visualization import set_publication_style, create_colorblind_friendly_palette

# ── Constants ─────────────────────────────────────────────────────────────
CLASS_NAMES = {0: 'Normal', 1: 'SNIK', 2: 'ITIK', 3: 'Total', 4: 'Neuropathy'}
CLASS_NAMES_TR = {0: 'NORMAL', 1: 'SNİK', 2: 'İTİK', 3: 'TOTAL', 4: 'NÖROPATİ'}
METRIC_LABELS = {
    'mse': 'MSE', 'l1': 'MAE', 'corr': 'Pearson r',
    'snr_db': 'SNR (dB)', 'stft_l1': 'STFT-L1',
}


def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_config(config_path: str, overrides: Optional[str] = None) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if overrides:
        for override in overrides.split(','):
            if ':' not in override:
                continue
            key, value = override.split(':', 1)
            key, value = key.strip(), value.strip()
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


# ══════════════════════════════════════════════════════════════════════════
# Dataset & Model Loading
# ══════════════════════════════════════════════════════════════════════════

def create_evaluation_dataset(cfg: Dict[str, Any]) -> Tuple[DataLoader, ABRDataset]:
    data_path = cfg['dataset']['data_path']
    logging.info(f"Loading dataset from: {data_path}")

    dataset = ABRDataset(
        data_path=data_path, normalize_signal=False,
        normalize_static=True, return_peak_labels=False, transform=None,
    )
    assert dataset.sequence_length == cfg['model']['sequence_length']

    _, val_dataset, _ = create_stratified_datasets(
        dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_state=cfg['evaluation']['seed'],
    )

    max_samples = cfg['evaluation'].get('num_samples')
    if max_samples and len(val_dataset) > max_samples:
        indices = torch.randperm(len(val_dataset))[:max_samples]
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
        logging.info(f"Limited to {max_samples} samples")

    loader = DataLoader(
        val_dataset, batch_size=cfg['dataset']['batch_size'], shuffle=False,
        num_workers=cfg['dataset']['num_workers'], pin_memory=cfg['dataset']['pin_memory'],
        collate_fn=abr_collate_fn, drop_last=False,
    )
    logging.info(f"Eval dataset: {len(val_dataset)} samples")
    return loader, dataset


def load_model_and_checkpoint(cfg, device):
    checkpoint_path = cfg['model']['checkpoint_path']
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_cfg = checkpoint.get('config', {}).get('model', {})
    model_config = {
        k: ckpt_cfg.get(k, cfg['model'].get(k))
        for k in ['input_channels', 'static_dim', 'sequence_length', 'd_model',
                   'n_layers', 'n_heads', 'ff_mult', 'dropout', 'num_classes',
                   'intensity_emb_dim', 'aux_static_dim']
    }
    model = ABRTransformerGenerator(**model_config).to(device)

    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logging.info("Checkpoint loaded (strict)")
    except RuntimeError:
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logging.warning(f"Non-strict load: {len(missing)} missing, {len(unexpected)} unexpected")

    # EMA
    if cfg['model'].get('use_ema', True) and 'ema_state_dict' in checkpoint:
        shadow = checkpoint['ema_state_dict'].get('shadow', {})
        if shadow:
            for name, param in model.named_parameters():
                if name in shadow:
                    param.data.copy_(shadow[name])
            logging.info("Applied EMA weights")

    model.eval()
    logging.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    return model, checkpoint


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _unpack(batch, device):
    return (batch['intensity'].to(device), batch['aux_static'].to(device),
            batch['class_label'].to(device))


def _sanitize(t: torch.Tensor) -> torch.Tensor:
    if torch.any(torch.isnan(t)) or torch.any(torch.isinf(t)):
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def _bootstrap_ci(values, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval for mean."""
    arr = np.array(values)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return float(np.mean(arr)), float(np.mean(arr)), float(np.mean(arr))
    boot_means = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
    return float(np.mean(arr)), float(lo), float(hi)


# ══════════════════════════════════════════════════════════════════════════
# Core Evaluation Loops
# ══════════════════════════════════════════════════════════════════════════

def evaluate_reconstruction(model, eval_loader, noise_schedule, dataset, cfg, device, training_T=1000):
    logging.info("Reconstruction evaluation...")
    model.eval()
    all_metrics, all_ref, all_gen = [], [], []

    use_stft = cfg['metrics']['signal'].get('stft_loss', True)
    use_dtw = cfg['metrics']['signal'].get('dtw', False)
    stft_params = cfg['metrics'].get('stft', {})
    recon_cfg = cfg.get('evaluation', {}).get('reconstruction', {})
    strategy = recon_cfg.get('timestep_strategy', 'uniform')

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(eval_loader, desc="Reconstruction")):
            x0 = batch['x0'].to(device)
            intensity, aux, cls = _unpack(batch, device)
            B = x0.shape[0]

            t = (torch.full((B,), recon_cfg.get('fixed_timestep', 500), device=device, dtype=torch.long)
                 if strategy == 'fixed'
                 else torch.randint(0, training_T, (B,), device=device))

            noise = torch.randn_like(x0)
            x_t, _ = q_sample_vpred(x0, t, noise, noise_schedule)
            v_pred = model(x_t, intensity=intensity, aux_static=aux, class_label=cls, timesteps=t)['signal']
            x0_rec = predict_x0_from_v(x_t, v_pred, t, noise_schedule)

            x0_d = _sanitize(dataset.denormalize_signal(x0))
            rec_d = _sanitize(dataset.denormalize_signal(x0_rec))

            try:
                per = compute_per_sample_metrics(rec_d, x0_d, use_stft, use_dtw, stft_params)
            except Exception as e:
                logging.error(f"Batch {bi}: {e}")
                per = [{'mse': np.nan, 'l1': np.nan, 'corr': np.nan, 'snr_db': np.nan}] * B

            for i, sm in enumerate(per):
                sm.update({'mode': 'reconstruction', 'timestep': t[i].item(),
                           'intensity': intensity[i].item(), 'class_label': int(cls[i].item())})
            all_metrics.extend(per)
            all_ref.append(x0_d.cpu())
            all_gen.append(rec_d.cpu())

    logging.info(f"Reconstruction: {len(all_metrics)} samples")
    return all_metrics, all_ref, all_gen


def evaluate_generation(model, eval_loader, sampler, dataset, cfg, device):
    logging.info("Generation evaluation...")
    model.eval()
    all_metrics, all_ref, all_gen = [], [], []

    use_stft = cfg['metrics']['signal'].get('stft_loss', True)
    use_dtw = cfg['metrics']['signal'].get('dtw', False)
    stft_params = cfg['metrics'].get('stft', {})
    diff = cfg.get('diffusion', {})
    gen_cfg = cfg.get('evaluation', {}).get('generation', {})
    steps = gen_cfg.get('num_steps', diff.get('sample_steps', 50))
    eta = diff.get('ddim_eta', 0.0)
    cfg_s = diff.get('cfg_scale', 2.0)
    cls_cfg_s = diff.get('class_cfg_scale', 1.5)

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(eval_loader, desc="Generation")):
            x0 = batch['x0'].to(device)
            intensity, aux, cls = _unpack(batch, device)
            B = x0.shape[0]

            x_gen = sampler.sample_conditioned(
                intensity=intensity, aux_static=aux, class_label=cls,
                steps=steps, eta=eta, cfg_scale=cfg_s, class_cfg_scale=cls_cfg_s, progress=False,
            )

            x0_d = _sanitize(dataset.denormalize_signal(x0))
            gen_d = _sanitize(dataset.denormalize_signal(x_gen))

            try:
                per = compute_per_sample_metrics(gen_d, x0_d, use_stft, use_dtw, stft_params)
            except Exception as e:
                logging.error(f"Batch {bi}: {e}")
                per = [{'mse': np.nan, 'l1': np.nan, 'corr': np.nan, 'snr_db': np.nan}] * B

            for i, sm in enumerate(per):
                sm.update({'mode': 'generation', 'cfg_scale': cfg_s,
                           'intensity': intensity[i].item(), 'class_label': int(cls[i].item())})
            all_metrics.extend(per)
            all_ref.append(x0_d.cpu())
            all_gen.append(gen_d.cpu())

    logging.info(f"Generation: {len(all_metrics)} samples")
    return all_metrics, all_ref, all_gen


# ══════════════════════════════════════════════════════════════════════════
# Publication-Quality Figures
# ══════════════════════════════════════════════════════════════════════════

def _pub_style():
    """Apply publication style globally."""
    set_publication_style()
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white'})


def _savefig(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logging.info(f"  Saved: {path}")


def fig_waveform_gallery(ref, gen, metrics_list, mode, out_dir):
    """Best/worst waveform comparison — Fig. 1 in paper."""
    _pub_style()
    colors = create_colorblind_friendly_palette()

    mse_vals = np.array([m.get('mse', np.nan) for m in metrics_list])
    valid = ~np.isnan(mse_vals)
    sorted_idx = np.argsort(mse_vals)
    n = min(5, len(sorted_idx) // 2)
    best_idx = sorted_idx[:n]
    worst_idx = sorted_idx[-n:]

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 6), sharey=True)
    if n == 1:
        axes = axes.reshape(2, 1)
    t_ms = np.linspace(0, 10, ref.shape[-1])

    for col, idx in enumerate(best_idx):
        ax = axes[0, col]
        ax.plot(t_ms, ref[idx, 0].numpy(), color=colors[0], lw=1.2, label='Reference')
        ax.plot(t_ms, gen[idx, 0].numpy(), color=colors[1], lw=1.2, ls='--', label='Generated')
        ax.set_title(f'Best #{col+1}\nMSE={mse_vals[idx]:.5f}', fontsize=9)
        if col == 0:
            ax.set_ylabel('Amplitude')
            ax.legend(fontsize=7, loc='upper right')

    for col, idx in enumerate(worst_idx):
        ax = axes[1, col]
        ax.plot(t_ms, ref[idx, 0].numpy(), color=colors[0], lw=1.2)
        ax.plot(t_ms, gen[idx, 0].numpy(), color=colors[1], lw=1.2, ls='--')
        ax.set_title(f'Worst #{col+1}\nMSE={mse_vals[idx]:.5f}', fontsize=9)
        ax.set_xlabel('Time (ms)')
        if col == 0:
            ax.set_ylabel('Amplitude')

    fig.suptitle(f'Waveform Comparison — {mode.title()}', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, out_dir / f'{mode}_waveform_gallery.png')


def fig_metrics_boxplot(df, mode, out_dir):
    """Per-class box plots for all metrics — Fig. 2."""
    _pub_style()
    import seaborn as sns

    metric_cols = [c for c in ['mse', 'l1', 'corr', 'snr_db', 'stft_l1'] if c in df.columns]
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5))
    if n == 1:
        axes = [axes]
    colors = create_colorblind_friendly_palette(5)

    df['class_name'] = df['class_label'].map(CLASS_NAMES)

    for i, col in enumerate(metric_cols):
        ax = axes[i]
        valid_df = df.dropna(subset=[col])
        # Filter extreme outliers for better visualization
        q99 = valid_df[col].quantile(0.99)
        q01 = valid_df[col].quantile(0.01)
        plot_df = valid_df[(valid_df[col] >= q01) & (valid_df[col] <= q99)]
        sns.boxplot(data=plot_df, x='class_name', y=col, ax=ax, palette=colors[:5],
                    width=0.6, fliersize=2, linewidth=1.2)
        ax.set_title(METRIC_LABELS.get(col, col), fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(METRIC_LABELS.get(col, col))
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle(f'Per-Class Metric Distributions — {mode.title()}', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, out_dir / f'{mode}_metrics_boxplot.png')


def fig_intensity_analysis(df, mode, out_dir):
    """Metrics vs intensity level — Fig. 3."""
    _pub_style()
    colors = create_colorblind_friendly_palette()

    # Bin intensity into groups
    df = df.copy()
    df['intensity_bin'] = pd.cut(df['intensity'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    metric_cols = [c for c in ['mse', 'corr', 'snr_db'] if c in df.columns]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 5))
    if len(metric_cols) == 1:
        axes = [axes]

    for i, col in enumerate(metric_cols):
        ax = axes[i]
        grouped = df.groupby('intensity_bin', observed=True)[col].agg(['mean', 'std', 'count'])
        grouped = grouped.dropna()
        x = range(len(grouped))
        ax.bar(x, grouped['mean'], yerr=grouped['std'], color=colors[i],
               alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, rotation=20, fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(col, col))
        ax.set_title(f'{METRIC_LABELS.get(col, col)} vs Intensity', fontsize=11, fontweight='bold')
        # Add count labels
        for j, (_, row) in enumerate(grouped.iterrows()):
            ax.text(j, row['mean'] + row['std'] * 0.1, f'n={int(row["count"])}',
                    ha='center', va='bottom', fontsize=7, color='gray')

    fig.suptitle(f'Intensity-Stratified Analysis — {mode.title()}', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, out_dir / f'{mode}_intensity_analysis.png')


def fig_error_heatmap(ref, gen, mode, out_dir, n_samples=50):
    """Error heatmap across time — Fig. 4."""
    _pub_style()

    N = min(n_samples, ref.shape[0])
    errors = (ref[:N, 0] - gen[:N, 0]).abs().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(errors, aspect='auto', cmap='hot', interpolation='nearest',
                   extent=[0, 10, N, 0])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Sample Index')
    ax.set_title(f'Absolute Error Heatmap — {mode.title()}', fontsize=13, fontweight='bold')
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('|Reference - Generated|')
    fig.tight_layout()
    _savefig(fig, out_dir / f'{mode}_error_heatmap.png')


def fig_mean_waveform_by_class(ref, gen, metrics_list, mode, out_dir):
    """Mean waveform ± std per class — Fig. 5."""
    _pub_style()
    colors = create_colorblind_friendly_palette(5)
    t_ms = np.linspace(0, 10, ref.shape[-1])

    class_labels = np.array([m['class_label'] for m in metrics_list])
    unique_classes = sorted(set(class_labels))
    n_cls = len(unique_classes)

    fig, axes = plt.subplots(2, n_cls, figsize=(4 * n_cls, 7), sharey='row')
    if n_cls == 1:
        axes = axes.reshape(2, 1)

    for col, cid in enumerate(unique_classes):
        mask = class_labels == cid
        ref_cls = ref[mask, 0].numpy()
        gen_cls = gen[mask, 0].numpy()
        name = CLASS_NAMES.get(cid, str(cid))
        n = mask.sum()

        for row, (data, label) in enumerate([(ref_cls, 'Reference'), (gen_cls, 'Generated')]):
            ax = axes[row, col]
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            ax.plot(t_ms, mean, color=colors[col], lw=1.5)
            ax.fill_between(t_ms, mean - std, mean + std, color=colors[col], alpha=0.2)
            ax.set_title(f'{name} — {label} (n={n})', fontsize=9, fontweight='bold')
            ax.set_xlabel('Time (ms)')
            if col == 0:
                ax.set_ylabel('Amplitude')

    fig.suptitle(f'Mean Waveform by Class — {mode.title()}', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, out_dir / f'{mode}_mean_waveform_by_class.png')


def fig_psd_comparison(ref, gen, mode, out_dir, n_samples=200):
    """Power Spectral Density comparison — Fig. 6."""
    _pub_style()
    colors = create_colorblind_friendly_palette()

    N = min(n_samples, ref.shape[0])
    fs = 20000  # 200 samples / 10ms = 20kHz

    ref_np = ref[:N, 0].numpy()
    gen_np = gen[:N, 0].numpy()

    from scipy.signal import welch
    freqs_r, psd_r = welch(ref_np, fs=fs, nperseg=min(64, ref_np.shape[-1]), axis=-1)
    freqs_g, psd_g = welch(gen_np, fs=fs, nperseg=min(64, gen_np.shape[-1]), axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mean PSD
    ax = axes[0]
    ax.semilogy(freqs_r, psd_r.mean(axis=0), color=colors[0], lw=1.5, label='Reference')
    ax.fill_between(freqs_r, psd_r.mean(0) - psd_r.std(0), psd_r.mean(0) + psd_r.std(0),
                    color=colors[0], alpha=0.15)
    ax.semilogy(freqs_g, psd_g.mean(axis=0), color=colors[1], lw=1.5, label='Generated')
    ax.fill_between(freqs_g, psd_g.mean(0) - psd_g.std(0), psd_g.mean(0) + psd_g.std(0),
                    color=colors[1], alpha=0.15)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('Power Spectral Density', fontweight='bold')
    ax.legend()

    # PSD ratio
    ax = axes[1]
    eps = 1e-10
    ratio = (psd_g.mean(0) + eps) / (psd_r.mean(0) + eps)
    ax.plot(freqs_r, ratio, color=colors[3], lw=1.5)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD Ratio (Gen / Ref)')
    ax.set_title('Spectral Fidelity Ratio', fontweight='bold')
    ax.set_ylim(0, 3)

    fig.suptitle(f'Spectral Analysis — {mode.title()}', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, out_dir / f'{mode}_psd_comparison.png')


def fig_scatter_matrix(df, mode, out_dir):
    """Pairwise metric scatter — Fig. 7."""
    _pub_style()
    cols = [c for c in ['mse', 'corr', 'snr_db', 'stft_l1'] if c in df.columns]
    if len(cols) < 2:
        return

    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))

    colors_by_class = create_colorblind_friendly_palette(5)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Histogram
                for cid in sorted(df['class_label'].unique()):
                    vals = df[df['class_label'] == cid][cols[i]].dropna()
                    ax.hist(vals, bins=30, alpha=0.5, color=colors_by_class[int(cid)],
                            label=CLASS_NAMES.get(int(cid), ''))
                if i == 0:
                    ax.legend(fontsize=5)
            else:
                for cid in sorted(df['class_label'].unique()):
                    sub = df[df['class_label'] == cid]
                    ax.scatter(sub[cols[j]], sub[cols[i]], s=5, alpha=0.3,
                               color=colors_by_class[int(cid)])

            if j == 0:
                ax.set_ylabel(METRIC_LABELS.get(cols[i], cols[i]), fontsize=8)
            if i == n - 1:
                ax.set_xlabel(METRIC_LABELS.get(cols[j], cols[j]), fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle(f'Metric Correlation Matrix — {mode.title()}', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, out_dir / f'{mode}_scatter_matrix.png')


def fig_summary_table(summary_recon, summary_gen, out_dir):
    """Summary comparison table as figure — Table 1."""
    _pub_style()

    metrics = ['mse', 'l1', 'corr', 'snr_db', 'stft_l1']
    rows = []
    for m in metrics:
        r_mean = summary_recon.get(f'{m}_mean', np.nan)
        r_std = summary_recon.get(f'{m}_std', np.nan)
        g_mean = summary_gen.get(f'{m}_mean', np.nan) if summary_gen else np.nan
        g_std = summary_gen.get(f'{m}_std', np.nan) if summary_gen else np.nan
        rows.append([METRIC_LABELS.get(m, m),
                      f'{r_mean:.4f} ± {r_std:.4f}',
                      f'{g_mean:.4f} ± {g_std:.4f}'])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=['Metric', 'Reconstruction', 'Generation'],
        cellLoc='center', loc='center',
        colColours=['#e6e6e6'] * 3,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Bold header
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight='bold')
        cell.set_edgecolor('#cccccc')

    fig.suptitle('Evaluation Summary (Mean ± Std)', fontsize=14, fontweight='bold', y=0.95)
    fig.tight_layout()
    _savefig(fig, out_dir / 'summary_table.png')


def fig_class_comparison_bars(summary, mode, out_dir):
    """Per-class MSE bar chart with CI — Fig. 8."""
    _pub_style()
    colors = create_colorblind_friendly_palette(5)

    classes = []
    means = []
    for cid, name in CLASS_NAMES.items():
        key = f'mse_{name}_mean'
        if key in summary:
            classes.append(f'{name}\n(n={summary.get(f"mse_{name}_n", "?")})')
            means.append(summary[key])

    if not classes:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(classes))
    bars = ax.bar(x, means, color=colors[:len(classes)], edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel('MSE')
    ax.set_title(f'Per-Class MSE — {mode.title()}', fontsize=13, fontweight='bold')

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    _savefig(fig, out_dir / f'{mode}_class_mse_bars.png')


# ══════════════════════════════════════════════════════════════════════════
# Results & LaTeX Table
# ══════════════════════════════════════════════════════════════════════════

def save_results(metrics, mode, cfg, checkpoint):
    out_dir = Path(cfg['output']['save_dir']) / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics)
    df.to_csv(out_dir / f'{mode}_all_samples.csv', index=False)

    summary = {'mode': mode, 'num_samples': len(metrics),
               'checkpoint_epoch': checkpoint.get('epoch', 'unknown')}

    for col in ['mse', 'l1', 'corr', 'snr_db', 'stft_l1', 'dtw']:
        vals = df[col].dropna().values if col in df.columns else []
        if len(vals):
            mean, lo, hi = _bootstrap_ci(vals)
            summary[f'{col}_mean'] = mean
            summary[f'{col}_std'] = float(np.std(vals))
            summary[f'{col}_ci_lo'] = lo
            summary[f'{col}_ci_hi'] = hi
            summary[f'{col}_min'] = float(np.min(vals))
            summary[f'{col}_max'] = float(np.max(vals))
            summary[f'{col}_median'] = float(np.median(vals))

    # Per-class
    if 'class_label' in df.columns:
        for cid, name in CLASS_NAMES.items():
            sub = df[df['class_label'] == cid]
            if len(sub):
                for col in ['mse', 'corr', 'snr_db']:
                    vals = sub[col].dropna().values if col in sub.columns else []
                    if len(vals):
                        summary[f'{col}_{name}_mean'] = float(np.mean(vals))
                        summary[f'{col}_{name}_std'] = float(np.std(vals))
                        summary[f'{col}_{name}_n'] = len(vals)

    # Save JSON
    with open(out_dir / f'{mode}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # LaTeX table
    _save_latex_table(summary, mode, out_dir)

    logging.info(f"Saved {mode} results to {out_dir}")
    return summary


def _save_latex_table(summary, mode, out_dir):
    """Generate a LaTeX-ready table."""
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        f'\\caption{{{mode.title()} Evaluation Results (N={summary["num_samples"]})}}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Metric & Mean & Std & 95\% CI & Median \\',
        r'\midrule',
    ]
    for col in ['mse', 'l1', 'corr', 'snr_db', 'stft_l1']:
        if f'{col}_mean' not in summary:
            continue
        name = METRIC_LABELS.get(col, col)
        mean = summary[f'{col}_mean']
        std = summary.get(f'{col}_std', 0)
        lo = summary.get(f'{col}_ci_lo', mean)
        hi = summary.get(f'{col}_ci_hi', mean)
        med = summary.get(f'{col}_median', mean)
        lines.append(f'  {name} & {mean:.4f} & {std:.4f} & [{lo:.4f}, {hi:.4f}] & {med:.4f} \\\\')

    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    with open(out_dir / f'{mode}_table.tex', 'w') as f:
        f.write('\n'.join(lines))


def save_evaluation_summary(all_results, cfg):
    out_dir = Path(cfg['output']['save_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'metrics': {}}
    for mode, results in all_results.items():
        if isinstance(results, dict):
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    summary['metrics'][f'{mode}.{k}'] = v

    with open(out_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logging.info(f"Summary: {out_dir / 'evaluation_summary.json'}")


# ══════════════════════════════════════════════════════════════════════════
# TensorBoard
# ══════════════════════════════════════════════════════════════════════════

def log_to_tensorboard(metrics, ref, gen, mode, cfg, writer):
    if not writer:
        return

    # Scalars
    for col in ['mse', 'l1', 'corr', 'snr_db', 'stft_l1']:
        vals = [m[col] for m in metrics if col in m and not np.isnan(m.get(col, np.nan))]
        if vals:
            writer.add_scalar(f'eval/{mode}/{col}_mean', np.mean(vals), 0)

    # Overlay figures
    try:
        all_ref = torch.cat(ref, dim=0)
        all_gen = torch.cat(gen, dim=0)
        mse_vals = [m.get('mse', np.nan) for m in metrics]
        sorted_idx = np.argsort(mse_vals)
        best = sorted_idx[:5]

        fig = overlay_waveforms(all_ref[best], all_gen[best],
                                [f"MSE={mse_vals[i]:.4f}" for i in best])
        writer.add_figure(f'eval/{mode}/best_overlay', fig, 0)
        close_figure(fig)
    except Exception as e:
        logging.warning(f"TensorBoard figure error: {e}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ABR Transformer V2 — Paper-Grade Evaluation")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--override", default="")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    setup_logging(cfg.get('logging', {}).get('console_level', 'INFO'))
    torch.manual_seed(cfg['evaluation']['seed'])
    np.random.seed(cfg['evaluation']['seed'])

    device = torch.device(cfg['model']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    eval_loader, dataset = create_evaluation_dataset(cfg)
    model, checkpoint = load_model_and_checkpoint(cfg, device)

    training_T = cfg.get('diffusion', {}).get('num_train_steps', 1000)
    noise_schedule = prepare_noise_schedule(training_T, device)
    sampler = DDIMSampler(model, training_T, device)

    writer = None
    if cfg.get('tensorboard', {}).get('enabled', True):
        writer = SummaryWriter(Path(cfg['tensorboard']['log_dir']))

    out_dir = Path(cfg['output']['save_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    logging.info("=" * 70)
    logging.info("  ABR TRANSFORMER V2 — PAPER-GRADE EVALUATION")
    logging.info("=" * 70)

    all_results = {}
    eval_mode = cfg['evaluation'].get('mode', 'both')

    summary_recon = None
    summary_gen = None

    # ── Reconstruction ──
    if eval_mode in ('reconstruction', 'both'):
        recon_m, recon_ref, recon_gen = evaluate_reconstruction(
            model, eval_loader, noise_schedule, dataset, cfg, device, training_T)

        summary_recon = save_results(recon_m, 'reconstruction', cfg, checkpoint)
        all_results['reconstruction'] = summary_recon

        all_ref = torch.cat(recon_ref, dim=0)
        all_gen = torch.cat(recon_gen, dim=0)
        df_recon = pd.DataFrame(recon_m)

        logging.info("Generating reconstruction figures...")
        fig_waveform_gallery(all_ref, all_gen, recon_m, 'reconstruction', figures_dir)
        fig_metrics_boxplot(df_recon, 'reconstruction', figures_dir)
        fig_intensity_analysis(df_recon, 'reconstruction', figures_dir)
        fig_error_heatmap(all_ref, all_gen, 'reconstruction', figures_dir)
        fig_mean_waveform_by_class(all_ref, all_gen, recon_m, 'reconstruction', figures_dir)
        fig_psd_comparison(all_ref, all_gen, 'reconstruction', figures_dir)
        fig_scatter_matrix(df_recon, 'reconstruction', figures_dir)
        fig_class_comparison_bars(summary_recon, 'reconstruction', figures_dir)

        log_to_tensorboard(recon_m, recon_ref, recon_gen, 'reconstruction', cfg, writer)

    # ── Generation ──
    if eval_mode in ('generation', 'both'):
        gen_m, gen_ref, gen_gen = evaluate_generation(
            model, eval_loader, sampler, dataset, cfg, device)

        summary_gen = save_results(gen_m, 'generation', cfg, checkpoint)
        all_results['generation'] = summary_gen

        all_ref = torch.cat(gen_ref, dim=0)
        all_gen = torch.cat(gen_gen, dim=0)
        df_gen = pd.DataFrame(gen_m)

        logging.info("Generating generation figures...")
        fig_waveform_gallery(all_ref, all_gen, gen_m, 'generation', figures_dir)
        fig_metrics_boxplot(df_gen, 'generation', figures_dir)
        fig_intensity_analysis(df_gen, 'generation', figures_dir)
        fig_error_heatmap(all_ref, all_gen, 'generation', figures_dir)
        fig_mean_waveform_by_class(all_ref, all_gen, gen_m, 'generation', figures_dir)
        fig_psd_comparison(all_ref, all_gen, 'generation', figures_dir)
        fig_scatter_matrix(df_gen, 'generation', figures_dir)
        fig_class_comparison_bars(summary_gen, 'generation', figures_dir)

        log_to_tensorboard(gen_m, gen_ref, gen_gen, 'generation', cfg, writer)

    # ── Cross-mode summary ──
    if summary_recon and summary_gen:
        fig_summary_table(summary_recon, summary_gen, figures_dir)

    if writer:
        writer.close()

    save_evaluation_summary(all_results, cfg)

    # Print final summary
    logging.info("")
    logging.info("=" * 70)
    logging.info("  RESULTS SUMMARY")
    logging.info("=" * 70)
    for mode, s in all_results.items():
        logging.info(f"\n  {mode.upper()} (n={s['num_samples']}):")
        for col in ['mse', 'corr', 'snr_db']:
            if f'{col}_mean' in s:
                logging.info(f"    {METRIC_LABELS.get(col, col):>12s}: "
                             f"{s[f'{col}_mean']:.4f} ± {s.get(f'{col}_std', 0):.4f}  "
                             f"[{s.get(f'{col}_ci_lo', 0):.4f}, {s.get(f'{col}_ci_hi', 0):.4f}]")

    logging.info(f"\nFigures:  {figures_dir}")
    logging.info(f"Results:  {out_dir}")
    logging.info("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
