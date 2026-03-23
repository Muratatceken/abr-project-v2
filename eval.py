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
from inference import DDIMSampler
from evaluation.analysis import roc_analysis, precision_recall_analysis, bootstrap_classification_metrics, statistical_significance_tests, clinical_validation_analysis, constrained_threshold_optimization, threshold_sensitivity_analysis, multi_objective_threshold_optimization


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


def load_threshold_optimization_config() -> Dict[str, Any]:
    """
    Load threshold optimization configuration.
    
    Returns:
        Dictionary with threshold optimization configuration
    """
    config_path = Path("configs/threshold_optimization.yaml")
    
    # Check if threshold optimization config exists
    if not config_path.exists():
        logging.warning(f"Threshold optimization config not found at {config_path}")
        # Return default configuration
        return {
            'optimization': {
                'strategies': {
                    'f1_optimal': True,
                    'youden_j': True,
                    'constrained': True,
                    'multi_objective': True
                },
                'default_strategy': 'f1_optimal'
            },
            'constraints': {
                'min_recall': 0.80,
                'min_precision': 0.90,
                'min_specificity': None
            },
            'output': {
                'generate_reports': True,
                'create_plots': True,
                'include_results': {
                    'threshold_analysis': True,
                    'constraint_validation': True
                }
            },
            'integration': {
                'enable_in_evaluation': True,
                'save_optimization_results': True
            }
        }
    
    try:
        with open(config_path, 'r') as f:
            threshold_config = yaml.safe_load(f)
        logging.info(f"✓ Loaded threshold optimization config from {config_path}")
        return threshold_config
    except Exception as e:
        logging.error(f"Failed to load threshold optimization config: {e}")
        return {}


def check_peak_labels_available(dataset: ABRDataset) -> bool:
    """Check if dataset has peak labels available."""
    return getattr(dataset, 'return_peak_labels', False) or hasattr(dataset, 'peak_exists')


def infer_peak_logits(model: nn.Module, x_gen: torch.Tensor, stat: torch.Tensor, cfg: Dict[str, Any]) -> Optional[torch.Tensor]:
    """
    Infer peak logits for generated signals using the model's classification head.
    
    Args:
        model: The ABR Transformer model
        x_gen: Generated signals [B, C, T]
        stat: Static parameters [B, S]  
        cfg: Configuration dictionary
        
    Returns:
        Peak logits tensor [B] or None if not available
    """
    if not cfg['metrics']['peak_classification']['enabled']:
        return None
    
    try:
        # Check if model has a classification head for peak detection
        if hasattr(model, 'classify_peak'):
            # Use dedicated classification method
            with torch.no_grad():
                peak_logits = model.classify_peak(x_gen, static_params=stat)
            return peak_logits
        elif hasattr(model, 'peak_head'):
            # Use classification head directly
            with torch.no_grad():
                # Extract features from generated signals
                features = model.extract_features(x_gen, static_params=stat)
                peak_logits = model.peak_head(features)
            return peak_logits
        else:
            # Alternative: Use the model in classification mode
            # This requires the model to support classification inference
            with torch.no_grad():
                # Create dummy timesteps (not used for classification)
                dummy_t = torch.zeros(x_gen.shape[0], device=x_gen.device, dtype=torch.long)
                model_output = model(x_gen, static_params=stat, timesteps=dummy_t)
                peak_logits = model_output.get('peak_5th_exists')
            return peak_logits
    except Exception as e:
        logging.warning(f"Could not infer peak logits for generated signals: {e}")
        return None


def create_evaluation_dataset(cfg: Dict[str, Any]) -> Tuple[DataLoader, ABRDataset]:
    """
    Create evaluation dataset and loader.
    """
    # Use dataset config from new schema
    data_path = cfg['dataset']['data_path']
    batch_size = cfg['dataset']['batch_size']
    num_workers = cfg['dataset']['num_workers']
    pin_memory = cfg['dataset']['pin_memory']
    return_peak_labels = cfg['dataset'].get('return_peak_labels', False)
    
    logging.info(f"Loading dataset from: {data_path}")
    
    # Load dataset
    dataset = ABRDataset(
        data_path=data_path,
        normalize_signal=True,
        normalize_static=True,
        return_peak_labels=return_peak_labels
    )
    
    # Verify dataset properties
    assert dataset.sequence_length == cfg['model']['sequence_length'], \
        f"Dataset sequence_length {dataset.sequence_length} != config {cfg['model']['sequence_length']}"
    assert dataset.static_dim == cfg['model']['static_dim'], \
        f"Dataset static_dim {dataset.static_dim} != config {cfg['model']['static_dim']}"
    
    # For evaluation, we can use the full dataset or create a validation split
    # If we're using the same file as training, create a split
    # Create splits to get validation portion
    _, val_dataset, _ = create_stratified_datasets(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15, 
        test_ratio=0.15,
        random_state=cfg['evaluation']['seed']
    )
    eval_dataset = val_dataset
    
    # Apply max_samples limit if specified
    max_samples = cfg['evaluation'].get('num_samples')
    if max_samples and len(eval_dataset) > max_samples:
        indices = torch.randperm(len(eval_dataset))[:max_samples]
        eval_dataset = torch.utils.data.Subset(eval_dataset, indices)
        logging.info(f"Limited evaluation to {max_samples} samples")
    
    # Create data loader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=abr_collate_fn,
        drop_last=False
    )
    
    # Log class balance if peak labels are available
    if return_peak_labels:
        # Get first batch to check peak label distribution
        first_batch = next(iter(eval_loader))
        if 'peak_exists' in first_batch:
            targets = first_batch['peak_exists']
            pos_rate = targets.float().mean().item()
            logging.info(f"Peak label class balance - Positive rate: {pos_rate:.3f} ({pos_rate*100:.1f}%)")
            logging.info(f"Total samples: {len(eval_dataset)}, Positive: {targets.sum().item()}, Negative: {(1-targets).sum().item()}")
    
    logging.info(f"✓ Created evaluation dataset: {len(eval_dataset)} samples")
    
    return eval_loader, dataset


def load_model_and_checkpoint(cfg: Dict[str, Any], device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model and checkpoint with optional EMA weights.
    """
    # Load checkpoint first to inspect what model architecture was used
    checkpoint_path = cfg['model']['checkpoint_path']
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model using model config with enhanced features detection
    model_config = {
        'input_channels': cfg['model']['input_channels'],
        'static_dim': cfg['model']['static_dim'],
        'sequence_length': cfg['model']['sequence_length'],
        'd_model': cfg['model']['d_model'],
        'n_layers': cfg['model']['n_layers'],
        'n_heads': cfg['model']['n_heads'],
        'ff_mult': cfg['model']['ff_mult'],
        'dropout': cfg['model']['dropout'],
        'use_timestep_cond': cfg['model']['use_timestep_cond'],
        'use_static_film': cfg['model']['use_static_film']
    }
    
    # Auto-detect enhanced features from checkpoint keys
    checkpoint_keys = set(checkpoint['model_state_dict'].keys())
    
    # Detect cross attention
    if any('cross_attention' in key for key in checkpoint_keys):
        model_config['use_cross_attention'] = True
        logging.info("✓ Detected cross attention in checkpoint")
    else:
        model_config['use_cross_attention'] = cfg['model'].get('use_cross_attention', False)
    
    # Detect joint static generation
    if any('static_recon_head' in key for key in checkpoint_keys):
        model_config['joint_static_generation'] = True
        logging.info("✓ Detected joint static generation in checkpoint")
    else:
        model_config['joint_static_generation'] = cfg['model'].get('joint_static_generation', False)
    
    # Detect multi-scale fusion
    if any('multi_scale_fusion' in key for key in checkpoint_keys):
        model_config['use_multi_scale_fusion'] = True
        logging.info("✓ Detected multi-scale fusion in checkpoint")
    else:
        model_config['use_multi_scale_fusion'] = cfg['model'].get('use_multi_scale_fusion', False)
    
    # Detect advanced transformer blocks
    if any('scale_attentions' in key for key in checkpoint_keys):
        model_config['use_advanced_blocks'] = True
        model_config['use_multi_scale_attention'] = True
        logging.info("✓ Detected advanced transformer blocks in checkpoint")
    else:
        model_config['use_advanced_blocks'] = cfg['model'].get('use_advanced_blocks', False)
        model_config['use_multi_scale_attention'] = cfg['model'].get('use_multi_scale_attention', False)
    
    # Detect gated feed forward
    if any('gate_proj' in key for key in checkpoint_keys):
        model_config['use_gated_ffn'] = True
        logging.info("✓ Detected gated feed forward in checkpoint")
    else:
        model_config['use_gated_ffn'] = cfg['model'].get('use_gated_ffn', False)
    
    # Detect learned positional embeddings
    if any('pos.position_embeddings' in key for key in checkpoint_keys):
        model_config['use_learned_pos_emb'] = True
        logging.info("✓ Detected learned positional embeddings in checkpoint")
    else:
        model_config['use_learned_pos_emb'] = cfg['model'].get('use_learned_pos_emb', False)
    
    # Additional enhanced features
    model_config['film_residual'] = cfg['model'].get('film_residual', True)
    model_config['ablation_mode'] = cfg['model'].get('ablation_mode', 'full')
    
    model = ABRTransformerGenerator(**model_config)
    model = model.to(device)
    
    # Load model weights with error handling
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logging.info("✓ Successfully loaded checkpoint with strict matching")
    except RuntimeError as e:
        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
            logging.warning("Checkpoint architecture mismatch detected, attempting flexible loading...")
            # Try loading with strict=False to allow partial loading
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                logging.warning(f"Missing keys in checkpoint: {len(missing_keys)} keys")
                logging.debug(f"Missing keys: {missing_keys[:5]}...")  # Show first 5 for brevity
            if unexpected_keys:
                logging.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                logging.debug(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5 for brevity
            logging.info("✓ Loaded checkpoint with partial matching (some weights may be randomly initialized)")
        else:
            raise e
    
    # Apply EMA weights if requested and available
    if cfg['model'].get('use_ema', False) and 'ema_state_dict' in checkpoint:
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
) -> Tuple[List[Dict[str, Any]], List[torch.Tensor], List[torch.Tensor], Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]]:
    """
    Evaluate reconstruction mode: denoise from x_t back to x_0.
    """
    logging.info("Starting reconstruction evaluation...")
    
    model.eval()
    all_metrics = []
    all_ref_signals = []
    all_gen_signals = []
    
    # Peak classification tracking
    peak_logits_list = []
    peak_targets_list = []
    
    # Metrics configuration
    use_stft = cfg['metrics']['signal']['stft_loss']
    use_dtw = cfg['metrics']['signal'].get('dtw', False)
    stft_params = cfg['metrics'].get('stft', {})
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Reconstruction")):
            x0 = batch['x0'].to(device)  # [B, 1, T]
            stat = batch['stat'].to(device)  # [B, S]
            meta = batch['meta']
            
            B, C, T = x0.shape
            
            # Sample timesteps based on strategy
            recon_cfg = cfg.get('evaluation', {}).get('advanced', {}).get('reconstruction', {})
            if not recon_cfg and 'advanced' in cfg:
                # Backward compatibility fallback
                recon_cfg = cfg.get('advanced', {}).get('reconstruction', {})
                logging.info("Using legacy 'advanced.reconstruction' config path for backward compatibility")
            timestep_strategy = recon_cfg.get('timestep_strategy', 'uniform')
            if not recon_cfg:
                logging.info("No 'evaluation.advanced.reconstruction' config found, defaulting timestep_strategy to 'uniform'")
            
            if timestep_strategy == 'fixed':
                fixed_t = recon_cfg.get('fixed_timestep', 500)
                t = torch.full((B,), fixed_t, device=device, dtype=torch.long)
            elif timestep_strategy == 'uniform':
                t = torch.randint(0, cfg['evaluation']['generation']['num_steps'], (B,), device=device)
            else:  # random_subset or other
                t = torch.randint(0, cfg['evaluation']['generation']['num_steps'], (B,), device=device)
            
            # Add noise
            noise = torch.randn_like(x0)
            x_t, v_target = q_sample_vpred(x0, t, noise, noise_schedule)
            
            # Model prediction (reconstruction)
            model_output = model(x_t, static_params=stat, timesteps=t)
            v_pred = model_output["signal"]
            
            # Reconstruct x0 from v-prediction
            from utils.schedules import predict_x0_from_v
            x0_recon = predict_x0_from_v(x_t, v_pred, t, noise_schedule)
            
            # Collect peak classification outputs if available
            if cfg['metrics']['peak_classification']['enabled']:
                peak_logits = model_output.get('peak_5th_exists')
                peak_targets = batch.get('peak_exists')
                if peak_logits is not None and peak_targets is not None:
                    peak_logits_list.append(peak_logits.cpu())
                    peak_targets_list.append(peak_targets.cpu())
            
            # Denormalize for metrics and visualization
            x0_denorm = dataset.denormalize_signal(x0)
            x0_recon_denorm = dataset.denormalize_signal(x0_recon)
            
            # Validate denormalized signals before metrics computation
            try:
                # Check for invalid values in denormalized signals
                if torch.any(torch.isnan(x0_denorm)) or torch.any(torch.isinf(x0_denorm)):
                    logging.warning(f"Reconstruction batch {batch_idx}: Invalid values in target signal after denormalization")
                    x0_denorm = torch.nan_to_num(x0_denorm, nan=0.0, posinf=0.0, neginf=0.0)
                    
                if torch.any(torch.isnan(x0_recon_denorm)) or torch.any(torch.isinf(x0_recon_denorm)):
                    logging.warning(f"Reconstruction batch {batch_idx}: Invalid values in reconstructed signal after denormalization")
                    x0_recon_denorm = torch.nan_to_num(x0_recon_denorm, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Validate signal properties for SNR computation
                for i in range(B):
                    target_var = torch.var(x0_denorm[i])
                    pred_var = torch.var(x0_recon_denorm[i])
                    if target_var < 1e-10:
                        logging.warning(f"Reconstruction sample {batch_idx}:{i}: Target signal has very low variance ({target_var:.2e})")
                    if pred_var < 1e-10:
                        logging.warning(f"Reconstruction sample {batch_idx}:{i}: Predicted signal has very low variance ({pred_var:.2e})")
                        
            except Exception as e:
                logging.error(f"Reconstruction batch {batch_idx}: Signal validation failed: {e}")
            
            # Compute per-sample metrics with enhanced error handling
            try:
                per_sample_metrics = compute_per_sample_metrics(
                    x0_recon_denorm, x0_denorm, use_stft, use_dtw, stft_params
                )
                
                # Validate computed metrics
                for i, metrics in enumerate(per_sample_metrics):
                    snr_value = metrics.get('snr_db', float('nan'))
                    if np.isnan(snr_value) or np.isinf(snr_value):
                        logging.warning(f"Reconstruction sample {batch_idx}:{i}: Invalid SNR value ({snr_value})")
                    elif snr_value < -50 or snr_value > 50:
                        logging.warning(f"Reconstruction sample {batch_idx}:{i}: Extreme SNR value ({snr_value:.2f} dB)")
                        
            except Exception as e:
                logging.error(f"Reconstruction batch {batch_idx}: Metrics computation failed: {e}")
                # Create fallback metrics
                per_sample_metrics = []
                for i in range(B):
                    fallback_metrics = {
                        'mse': float('nan'),
                        'l1': float('nan'),
                        'corr': float('nan'),
                        'snr_db': float('nan')
                    }
                    if use_stft:
                        fallback_metrics['stft_l1'] = float('nan')
                    if use_dtw:
                        fallback_metrics['dtw'] = float('nan')
                    per_sample_metrics.append(fallback_metrics)
            
            # Add metadata to metrics with additional diagnostic information
            for i, sample_metrics in enumerate(per_sample_metrics):
                sample_metrics.update({
                    'mode': 'reconstruction',
                    'sample_id': meta[i].get('sample_idx', batch_idx * cfg['dataset']['batch_size'] + i),
                    'timestep': t[i].item(),
                    'batch_idx': batch_idx
                })
                # Add static parameters
                try:
                    for j, param_name in enumerate(dataset.static_names):
                        sample_metrics[f'static_{param_name}'] = stat[i, j].item()
                except Exception as e:
                    logging.warning(f"Reconstruction sample {batch_idx}:{i}: Failed to add static parameters: {e}")
            
            all_metrics.extend(per_sample_metrics)
            all_ref_signals.append(x0_denorm.cpu())
            all_gen_signals.append(x0_recon_denorm.cpu())
    
    # Summary statistics on SNR calculation issues
    invalid_snr_count = sum(1 for metrics in all_metrics 
                          if np.isnan(metrics.get('snr_db', 0)) or np.isinf(metrics.get('snr_db', 0)))
    extreme_snr_count = sum(1 for metrics in all_metrics 
                          if not (np.isnan(metrics.get('snr_db', 0)) or np.isinf(metrics.get('snr_db', 0))) 
                          and (metrics.get('snr_db', 0) < -50 or metrics.get('snr_db', 0) > 50))
    
    if invalid_snr_count > 0:
        logging.warning(f"Reconstruction: {invalid_snr_count}/{len(all_metrics)} samples had invalid SNR values")
    if extreme_snr_count > 0:
        logging.warning(f"Reconstruction: {extreme_snr_count}/{len(all_metrics)} samples had extreme SNR values")
    
    logging.info(f"✓ Reconstruction evaluation completed: {len(all_metrics)} samples")
    
    # Return peak classification data if available
    peak_data = None
    if peak_logits_list and peak_targets_list:
        peak_data = (torch.cat(peak_logits_list, dim=0), torch.cat(peak_targets_list, dim=0))
    
    return all_metrics, all_ref_signals, all_gen_signals, peak_data


def evaluate_generation(
    model: nn.Module,
    eval_loader: DataLoader,
    sampler: DDIMSampler,
    dataset: ABRDataset,
    cfg: Dict[str, Any],
    device: torch.device
) -> Tuple[List[Dict[str, Any]], List[torch.Tensor], List[torch.Tensor], Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]]:
    """
    Evaluate conditional generation: sample from noise conditioned on static params.
    """
    logging.info("Starting generation evaluation...")
    
    model.eval()
    all_metrics = []
    all_ref_signals = []
    all_gen_signals = []
    
    # Peak classification tracking
    peak_logits_list = []
    peak_targets_list = []
    
    # Metrics configuration
    use_stft = cfg['metrics']['signal']['stft_loss']
    use_dtw = cfg['metrics']['signal'].get('dtw', False)
    stft_params = cfg['metrics'].get('stft', {})
    
    # Generation configuration - use new config fields
    gen_cfg = cfg['evaluation']['generation']
    cfg_scale = gen_cfg.get('guidance_scale', 1.0)  # Use guidance_scale from config
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Generation")):
            x0 = batch['x0'].to(device)  # [B, 1, T] - reference for comparison
            stat = batch['stat'].to(device)  # [B, S] - conditioning
            meta = batch['meta']
            
            B, C, T = x0.shape
            
            # Generate samples conditioned on static parameters
            x_gen = sampler.sample_conditioned(
                static_params=stat,
                steps=gen_cfg['num_steps'],
                eta=gen_cfg.get('ddim_eta', 0.0),
                cfg_scale=cfg_scale,
                progress=False
            )
            
            # Collect peak classification outputs if available
            if cfg['metrics']['peak_classification']['enabled']:
                # Infer peak logits for generated signals
                peak_logits = infer_peak_logits(model, x_gen, stat, cfg)
                peak_targets = batch.get('peak_exists')
                
                if peak_logits is not None and peak_targets is not None:
                    peak_logits_list.append(peak_logits.cpu())
                    peak_targets_list.append(peak_targets.cpu())
                elif peak_targets is not None:
                    # Still collect targets even if we can't get logits
                    peak_targets_list.append(peak_targets.cpu())
                    if len(peak_logits_list) == 0:  # First batch
                        logging.warning("Peak classification enabled but no logits available for generated signals. Classification metrics will be limited.")
            
            # Denormalize for metrics and visualization
            x0_denorm = dataset.denormalize_signal(x0)
            x_gen_denorm = dataset.denormalize_signal(x_gen)
            
            # Validate denormalized signals before metrics computation
            try:
                # Check for invalid values in denormalized signals
                if torch.any(torch.isnan(x0_denorm)) or torch.any(torch.isinf(x0_denorm)):
                    logging.warning(f"Generation batch {batch_idx}: Invalid values in target signal after denormalization")
                    x0_denorm = torch.nan_to_num(x0_denorm, nan=0.0, posinf=0.0, neginf=0.0)
                    
                if torch.any(torch.isnan(x_gen_denorm)) or torch.any(torch.isinf(x_gen_denorm)):
                    logging.warning(f"Generation batch {batch_idx}: Invalid values in generated signal after denormalization")
                    x_gen_denorm = torch.nan_to_num(x_gen_denorm, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Validate signal properties for SNR computation
                for i in range(B):
                    target_var = torch.var(x0_denorm[i])
                    gen_var = torch.var(x_gen_denorm[i])
                    if target_var < 1e-10:
                        logging.warning(f"Generation sample {batch_idx}:{i}: Target signal has very low variance ({target_var:.2e})")
                    if gen_var < 1e-10:
                        logging.warning(f"Generation sample {batch_idx}:{i}: Generated signal has very low variance ({gen_var:.2e})")
                        
            except Exception as e:
                logging.error(f"Generation batch {batch_idx}: Signal validation failed: {e}")
            
            # Compute per-sample metrics with enhanced error handling (comparing generated to reference)
            try:
                per_sample_metrics = compute_per_sample_metrics(
                    x_gen_denorm, x0_denorm, use_stft, use_dtw, stft_params
                )
                
                # Validate computed metrics
                for i, metrics in enumerate(per_sample_metrics):
                    snr_value = metrics.get('snr_db', float('nan'))
                    if np.isnan(snr_value) or np.isinf(snr_value):
                        logging.warning(f"Generation sample {batch_idx}:{i}: Invalid SNR value ({snr_value})")
                    elif snr_value < -50 or snr_value > 50:
                        logging.warning(f"Generation sample {batch_idx}:{i}: Extreme SNR value ({snr_value:.2f} dB)")
                        
            except Exception as e:
                logging.error(f"Generation batch {batch_idx}: Metrics computation failed: {e}")
                # Create fallback metrics
                per_sample_metrics = []
                for i in range(B):
                    fallback_metrics = {
                        'mse': float('nan'),
                        'l1': float('nan'),
                        'corr': float('nan'),
                        'snr_db': float('nan')
                    }
                    if use_stft:
                        fallback_metrics['stft_l1'] = float('nan')
                    if use_dtw:
                        fallback_metrics['dtw'] = float('nan')
                    per_sample_metrics.append(fallback_metrics)
            
            # Add metadata to metrics with additional diagnostic information
            for i, sample_metrics in enumerate(per_sample_metrics):
                sample_metrics.update({
                    'mode': 'generation',
                    'sample_id': meta[i].get('sample_idx', batch_idx * cfg['dataset']['batch_size'] + i),
                    'cfg_scale': cfg_scale,
                    'batch_idx': batch_idx
                })
                # Add static parameters
                try:
                    for j, param_name in enumerate(dataset.static_names):
                        sample_metrics[f'static_{param_name}'] = stat[i, j].item()
                except Exception as e:
                    logging.warning(f"Generation sample {batch_idx}:{i}: Failed to add static parameters: {e}")
            
            all_metrics.extend(per_sample_metrics)
            all_ref_signals.append(x0_denorm.cpu())
            all_gen_signals.append(x_gen_denorm.cpu())
    
    # Summary statistics on SNR calculation issues
    invalid_snr_count = sum(1 for metrics in all_metrics 
                          if np.isnan(metrics.get('snr_db', 0)) or np.isinf(metrics.get('snr_db', 0)))
    extreme_snr_count = sum(1 for metrics in all_metrics 
                          if not (np.isnan(metrics.get('snr_db', 0)) or np.isinf(metrics.get('snr_db', 0))) 
                          and (metrics.get('snr_db', 0) < -50 or metrics.get('snr_db', 0) > 50))
    
    if invalid_snr_count > 0:
        logging.warning(f"Generation: {invalid_snr_count}/{len(all_metrics)} samples had invalid SNR values")
    if extreme_snr_count > 0:
        logging.warning(f"Generation: {extreme_snr_count}/{len(all_metrics)} samples had extreme SNR values")
    
    logging.info(f"✓ Generation evaluation completed: {len(all_metrics)} samples")
    
    # Return peak classification data if available
    peak_data = None
    if peak_logits_list and peak_targets_list:
        peak_data = (torch.cat(peak_logits_list, dim=0), torch.cat(peak_targets_list, dim=0))
    elif peak_targets_list:
        # Only targets available, no logits
        peak_data = (None, torch.cat(peak_targets_list, dim=0))
    
    return all_metrics, all_ref_signals, all_gen_signals, peak_data


def create_classification_visualizations(
    peak_logits: torch.Tensor,
    peak_targets: torch.Tensor,
    mode: str,
    cfg: Dict[str, Any],
    writer: SummaryWriter,
    output_dir: Path,
    epoch: int = 0
):
    """
    Create publication-quality classification visualizations.
    """
    if not cfg['metrics']['peak_classification']['enabled']:
        return
    
    if peak_logits is None:
        logging.warning("No peak logits available for classification visualizations")
        return
    
    logging.info(f"Creating classification visualizations for {mode} mode...")
    
    # Convert to numpy
    logits = peak_logits.numpy()
    targets = peak_targets.numpy()
    
    # Edge-case handling
    unique_targets = np.unique(targets)
    if len(unique_targets) < 2:
        logging.warning(f"Only {len(unique_targets)} unique class found. Skipping classification plots.")
        return
    
    try:
        from evaluation.visualization import (
            plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix,
            plot_classification_metrics_comparison, plot_threshold_analysis
        )
        
        # Get ROC and PR data from previous analysis
        roc_results = roc_analysis(
            logits, targets,
            specificity_targets=cfg['metrics']['clinical_metrics']['specificity_targets']
        )
        pr_results = precision_recall_analysis(logits, targets)
        
        # Create ROC curve
        if cfg['report']['save_roc_curves']:
            fig_roc = plot_roc_curve(
                roc_results,
                title=f"ROC Curve - {mode.title()} Mode",
                show_confidence_interval=True
            )
            
            # Save figure
            roc_save_path = output_dir / f"{mode}_roc_curve.png"
            fig_roc.savefig(roc_save_path, dpi=300, bbox_inches='tight')
            
            # Log to TensorBoard
            if cfg['tensorboard']['log_plots']:
                writer.add_figure(f'eval/{mode}/classification/roc_curve', fig_roc, epoch)
            
            plt.close(fig_roc)
            logging.info(f"✓ Saved ROC curve: {roc_save_path}")
        
        # Create PR curve
        if cfg['report']['save_pr_curves']:
            fig_pr = plot_precision_recall_curve(
                pr_results,
                title=f"Precision-Recall Curve - {mode.title()} Mode"
            )
            
            # Save figure
            pr_save_path = output_dir / f"{mode}_pr_curve.png"
            fig_pr.savefig(pr_save_path, dpi=300, bbox_inches='tight')
            
            # Log to TensorBoard
            if cfg['tensorboard']['log_plots']:
                writer.add_figure(f'eval/{mode}/classification/pr_curve', fig_pr, epoch)
            
            plt.close(fig_pr)
            logging.info(f"✓ Saved PR curve: {pr_save_path}")
        
        # Create confusion matrix
        if cfg['report']['save_confusion_matrices']:
            # Get predictions at threshold
            threshold = cfg['metrics']['peak_classification']['threshold']
            predictions = (logits > threshold).astype(int)
            
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(targets, predictions)
            
            fig_cm = plot_confusion_matrix(
                cm,
                class_names=['No Peak', 'Peak Present'],
                title=f"Confusion Matrix - {mode.title()} Mode (Threshold: {threshold})"
            )
            
            # Save figure
            cm_save_path = output_dir / f"{mode}_confusion_matrix.png"
            fig_cm.savefig(cm_save_path, dpi=300, bbox_inches='tight')
            
            # Log to TensorBoard
            if cfg['tensorboard']['log_plots']:
                writer.add_figure(f'eval/{mode}/classification/confusion_matrix', fig_cm, epoch)
            
            plt.close(fig_cm)
            logging.info(f"✓ Saved confusion matrix: {cm_save_path}")
        
        # Create threshold analysis
        if cfg['report'].get('save_threshold_analysis', False):
            fig_thresh = plot_threshold_analysis(
                logits, targets,
                title=f"Threshold Analysis - {mode.title()} Mode"
            )
            
            # Save figure
            thresh_save_path = output_dir / f"{mode}_threshold_analysis.png"
            fig_thresh.savefig(thresh_save_path, dpi=300, bbox_inches='tight')
            
            # Log to TensorBoard
            if cfg['tensorboard']['log_plots']:
                writer.add_figure(f'eval/{mode}/classification/threshold_analysis', fig_thresh, epoch)
            
            plt.close(fig_thresh)
            logging.info(f"✓ Saved threshold analysis: {thresh_save_path}")
        
        logging.info(f"✓ Created classification visualizations for {mode} mode")
        
    except Exception as e:
        logging.warning(f"Error creating classification visualizations: {e}")


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
    if not cfg['tensorboard']['log_plots']:
        return
    
    logging.info(f"Creating visualizations for {mode} mode...")
    
    # Concatenate all signals
    all_ref = torch.cat(ref_signals, dim=0)  # [N, 1, T]
    all_gen = torch.cat(gen_signals, dim=0)  # [N, 1, T]
    
    # Sort by MSE for best/worst selection
    mse_values = [m['mse'] for m in metrics]
    sorted_indices = np.argsort(mse_values)
    
    topk = cfg['report'].get('save_topk_examples', 10)
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
        if cfg['report'].get('save_spectrograms', False):
            spec_params = cfg.get('visualization', {}).get('spectrogram_params', {})
            if not spec_params and 'spectrogram_params' in cfg:
                # Backward compatibility fallback
                spec_params = cfg.get('spectrogram_params', {})
                logging.info("Using legacy 'spectrogram_params' config path for backward compatibility")
            
            fig_spec_best = spectrograms(best_ref[:4], **spec_params)
            writer.add_figure(f'eval/{mode}/spectrogram_ref', fig_spec_best, epoch)
            close_figure(fig_spec_best)
            
            fig_spec_gen = spectrograms(best_gen[:4], **spec_params)
            writer.add_figure(f'eval/{mode}/spectrogram_gen', fig_spec_gen, epoch)
            close_figure(fig_spec_gen)
        
        # Scatter plots for metrics correlation
        if len(metrics) > 1:
            # Extract MSE and correlation values for scatter plot
            mse_values = np.array([m.get('mse', 0) for m in metrics])
            corr_values = np.array([m.get('corr', 0) for m in metrics])
            fig_scatter = scatter_xy(mse_values, corr_values, 'MSE', 'Correlation', f'{mode.title()} MSE vs Correlation')
            writer.add_figure(f'eval/{mode}/metrics_correlation', fig_scatter, epoch)
            close_figure(fig_scatter)
        
        # Metrics summary plot - convert list of metrics to summary dict
        if metrics:
            # Calculate mean and std for each metric
            metric_keys = ['mse', 'l1', 'corr', 'snr_db', 'stft_l1']
            metrics_summary = {}
            for key in metric_keys:
                values = [m.get(key, 0) for m in metrics if key in m]
                if values:
                    metrics_summary[key] = (np.mean(values), np.std(values))
            
            if metrics_summary:
                fig_summary = metrics_summary_plot(metrics_summary, mode)
                writer.add_figure(f'eval/{mode}/metrics_summary', fig_summary, epoch)
                close_figure(fig_summary)
        
    except Exception as e:
        logging.warning(f"Error creating visualizations: {e}")


def save_results(metrics: List[Dict[str, Any]], mode: str, cfg: Dict[str, Any], checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save evaluation results and compute summary statistics.
    """
    # Create output directory
    output_dir = Path(cfg['output']['save_dir']) / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics)
    
    # Save detailed results
    if cfg['output']['save_samples']:
        results_file = output_dir / f"{mode}_detailed_results.{cfg['output']['save_format']}"
        if cfg['output']['save_format'] == 'json':
            df.to_json(results_file, orient='records', indent=2)
        else:
            df.to_pickle(results_file)
    
    # Compute summary statistics
    summary = {
        'mode': mode,
        'num_samples': len(metrics),
        'checkpoint_info': {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'step': checkpoint.get('step', 'unknown'),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }
    }
    
    # Add metric summaries
    numeric_columns = ['mse', 'l1', 'corr', 'snr_db']
    if 'stft_l1' in metrics[0]:
        numeric_columns.append('stft_l1')
    if 'dtw' in metrics[0]:
        numeric_columns.append('dtw')
    
    for col in numeric_columns:
        values = [m[col] for m in metrics if not np.isnan(m.get(col, np.nan))]
        if values:
            summary[f'{col}_mean'] = float(np.mean(values))
            summary[f'{col}_std'] = float(np.std(values))
            summary[f'{col}_min'] = float(np.min(values))
            summary[f'{col}_max'] = float(np.max(values))
    
    # Save summary
    summary_file = output_dir / f"{mode}_summary.{cfg['output']['save_format']}"
    if cfg['output']['save_format'] == 'json':
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    else:
        with open(summary_file, 'wb') as f:
            import pickle
            pickle.dump(summary, f)
    
    logging.info(f"✓ Saved {mode} results to {output_dir}")
    return summary


def save_evaluation_summary(all_results: Dict[str, Any], cfg: Dict[str, Any]):
    """
    Save unified evaluation summary for comparative analysis.
    Option A (preferred): Build a root-level metrics dict with flattened keys and numeric values.
    """
    # Create root output directory
    output_dir = Path(cfg['output']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unified summary with flattened metrics structure for ComparativeAnalyzer
    evaluation_summary = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': cfg,
            'total_modes': len(all_results)
        },
        'metrics': {}
    }
    
    # Add metrics from each mode with flattened keys
    for mode, mode_results in all_results.items():
        if isinstance(mode_results, dict):
            # Extract signal metrics with flattened keys
            for key, value in mode_results.items():
                if key.endswith('_mean') and isinstance(value, (int, float)):
                    metric_name = key.replace('_mean', '')
                    # Flattened key format: mode.metric.stat
                    evaluation_summary['metrics'][f'{mode}.{metric_name}.mean'] = value
                    
                    # Add other statistics if available
                    if f'{metric_name}_std' in mode_results:
                        evaluation_summary['metrics'][f'{mode}.{metric_name}.std'] = mode_results[f'{metric_name}_std']
                    if f'{metric_name}_min' in mode_results:
                        evaluation_summary['metrics'][f'{mode}.{metric_name}.min'] = mode_results[f'{metric_name}_min']
                    if f'{metric_name}_max' in mode_results:
                        evaluation_summary['metrics'][f'{mode}.{metric_name}.max'] = mode_results[f'{metric_name}_max']
            
            # Add classification metrics with flattened keys if available
            if 'peak_classification' in mode_results:
                peak_results = mode_results['peak_classification']
                if 'bootstrap_metrics' in peak_results:
                    bootstrap_metrics = peak_results['bootstrap_metrics']
                    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                        if metric_name in bootstrap_metrics:
                            # Flattened key format: mode.classification.metric.stat
                            evaluation_summary['metrics'][f'{mode}.classification.{metric_name}'] = bootstrap_metrics[metric_name]['mean']
                            evaluation_summary['metrics'][f'{mode}.classification.{metric_name}.mean'] = bootstrap_metrics[metric_name]['mean']
                            evaluation_summary['metrics'][f'{mode}.classification.{metric_name}.ci_lower'] = bootstrap_metrics[metric_name]['ci_lower']
                            evaluation_summary['metrics'][f'{mode}.classification.{metric_name}.ci_upper'] = bootstrap_metrics[metric_name]['ci_upper']
                
                if 'roc_analysis' in peak_results:
                    evaluation_summary['metrics'][f'{mode}.classification.auroc'] = peak_results['roc_analysis']['auroc']
                
                # Add prevalence
                if 'prevalence' in peak_results:
                    evaluation_summary['metrics'][f'{mode}.classification.prevalence'] = peak_results['prevalence']
            
            # Add sample count
            evaluation_summary['metrics'][f'{mode}.num_samples'] = mode_results.get('num_samples', 0)
    
    # Save unified summary
    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(evaluation_summary, f, indent=2, default=str)
    
    logging.info(f"✓ Saved unified evaluation summary: {summary_file}")
    return evaluation_summary


def save_peak_classification_results(
    peak_logits: torch.Tensor,
    peak_targets: torch.Tensor,
    mode: str,
    cfg: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Save peak classification results and compute metrics with threshold optimization.
    """
    if not cfg['metrics']['peak_classification']['enabled']:
        return {}
    
    logging.info(f"Computing peak classification metrics for {mode} mode...")
    
    # Convert to numpy
    logits = peak_logits.numpy() if peak_logits is not None else None
    targets = peak_targets.numpy()
    
    if logits is None:
        logging.warning("No peak logits available for classification metrics")
        return {}
    
    # Edge-case handling for single-class or perfect predictions
    unique_targets = np.unique(targets)
    if len(unique_targets) < 2:
        logging.warning(f"Only {len(unique_targets)} unique class found in targets. Skipping classification metrics.")
        return {
            'warning': f"Only {len(unique_targets)} unique class found in targets",
            'targets_unique': unique_targets.tolist()
        }
    
    # Apply sigmoid to convert logits to probabilities before thresholding
    from scipy.special import expit
    probabilities = expit(logits)
    predictions = (probabilities > cfg['metrics']['peak_classification']['threshold']).astype(int)
    unique_predictions = np.unique(predictions)
    if len(unique_predictions) == 1:
        logging.warning(f"All predictions are the same ({unique_predictions[0]}). This may indicate model issues.")
    
    # Ensure bootstrap resampling stratifies classes when possible
    if len(unique_targets) >= 2:
        min_class_count = min(np.sum(targets == 0), np.sum(targets == 1))
        if min_class_count < 10:
            logging.warning(f"Small class size detected: {min_class_count}. Bootstrap results may be unreliable.")
    
    try:
        # Load threshold optimization configuration
        threshold_config = load_threshold_optimization_config()
        enable_threshold_optimization = threshold_config.get('integration', {}).get('enable_in_evaluation', True)
        
        # Compute bootstrap classification metrics
        bootstrap_results = bootstrap_classification_metrics(
            logits, targets,
            n_bootstrap=cfg['metrics']['peak_classification']['bootstrap_ci'],
            confidence_level=cfg['metrics']['statistical_analysis']['confidence_level']
        )
        
        # Compute ROC analysis with optional threshold optimization
        constraint_params = None
        if enable_threshold_optimization:
            constraint_params = threshold_config.get('constraints', {})
            constraint_params = {k: v for k, v in constraint_params.items() if v is not None}
        
        roc_results = roc_analysis(
            logits, targets,
            specificity_targets=cfg['metrics']['clinical_metrics']['specificity_targets'],
            enable_constrained_optimization=enable_threshold_optimization,
            constraint_params=constraint_params
        )
        
        # Compute precision-recall analysis with optimization
        pr_results = precision_recall_analysis(
            logits, targets,
            enable_constrained_optimization=enable_threshold_optimization,
            constraint_params=constraint_params
        )
        
        # Compute statistical significance tests
        prevalence = targets.mean()
        significance_results = statistical_significance_tests(
            logits, targets, 
            prevalence=prevalence,
            multiple_testing_correction=cfg['metrics']['statistical_analysis']['multiple_testing_correction']
        )
        
        # Compute clinical validation metrics
        clinical_results = clinical_validation_analysis(
            logits, targets,
            prevalence=prevalence
        )
        
        # Additional threshold optimization analyses if enabled
        threshold_optimization_results = {}
        if enable_threshold_optimization and threshold_config.get('output', {}).get('include_results', {}).get('threshold_analysis', True):
            logging.info("Running advanced threshold optimization analyses...")
            
            try:
                # Threshold sensitivity analysis
                if threshold_config.get('output', {}).get('include_results', {}).get('stability_analysis', True):
                    sensitivity_analysis = threshold_sensitivity_analysis(logits, targets, num_points=100)
                    threshold_optimization_results['sensitivity_analysis'] = sensitivity_analysis
                
                # Multi-objective optimization
                if threshold_config.get('optimization', {}).get('strategies', {}).get('multi_objective', True):
                    multi_obj_results = multi_objective_threshold_optimization(
                        logits, targets, 
                        objectives=['precision', 'recall', 'specificity']
                    )
                    threshold_optimization_results['multi_objective'] = multi_obj_results
                
                logging.info("✓ Completed advanced threshold optimization analyses")
                
            except Exception as e:
                logging.warning(f"Advanced threshold optimization failed: {e}")
                threshold_optimization_results['error'] = str(e)
        
        # Combine results
        results = {
            'bootstrap_metrics': bootstrap_results,
            'roc_analysis': roc_results,
            'precision_recall_analysis': pr_results,
            'statistical_significance': significance_results,
            'clinical_validation': clinical_results,
            'prevalence': float(prevalence)
        }
        
        # Add threshold optimization results if available
        if threshold_optimization_results:
            results['threshold_optimization'] = threshold_optimization_results
            
            # Add configuration used
            results['threshold_optimization_config'] = {
                'constraints': constraint_params,
                'strategies_enabled': threshold_config.get('optimization', {}).get('strategies', {})
            }
        
        # Save results with enhanced threshold optimization data
        results_file = output_dir / f"{mode}_peak_classification.{cfg['output']['save_format']}"
        if cfg['output']['save_format'] == 'json':
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        else:
            with open(results_file, 'wb') as f:
                import pickle
                pickle.dump(results, f)
        
        # Enhanced CSV export with threshold optimization metrics
        csv_data = [{
            'metric': 'accuracy',
            'value': bootstrap_results['accuracy']['mean'],
            'ci_lower': bootstrap_results['accuracy']['ci_lower'],
            'ci_upper': bootstrap_results['accuracy']['ci_upper'],
            'mode': mode
        }, {
            'metric': 'precision',
            'value': bootstrap_results['precision']['mean'],
            'ci_lower': bootstrap_results['precision']['ci_lower'],
            'ci_upper': bootstrap_results['precision']['ci_upper'],
            'mode': mode
        }, {
            'metric': 'recall',
            'value': bootstrap_results['recall']['mean'],
            'ci_lower': bootstrap_results['recall']['ci_lower'],
            'ci_upper': bootstrap_results['recall']['ci_upper'],
            'mode': mode
        }, {
            'metric': 'f1',
            'value': bootstrap_results['f1']['mean'],
            'ci_lower': bootstrap_results['f1']['ci_lower'],
            'ci_upper': bootstrap_results['f1']['ci_upper'],
            'mode': mode
        }, {
            'metric': 'auroc',
            'value': roc_results['auroc'],
            'ci_lower': roc_results.get('auroc_ci', {}).get('lower'),
            'ci_upper': roc_results.get('auroc_ci', {}).get('upper'),
            'mode': mode
        }]
        
        # Add optimal threshold information
        if 'constrained_optimization' in roc_results and roc_results['constrained_optimization'].get('success'):
            constrained_opt = roc_results['constrained_optimization']
            csv_data.append({
                'metric': 'optimal_threshold_constrained',
                'value': constrained_opt['optimal_threshold'],
                'ci_lower': None,
                'ci_upper': None,
                'mode': mode
            })
        
        csv_file = output_dir / f"{mode}_peak_classification_metrics.csv"
        metrics_df = pd.DataFrame(csv_data)
        metrics_df.to_csv(csv_file, index=False)
        
        # Save threshold optimization report if enabled
        if enable_threshold_optimization and threshold_config.get('output', {}).get('generate_reports', True):
            optimization_summary = generate_threshold_optimization_summary(results, threshold_config)
            summary_file = output_dir / f"{mode}_threshold_optimization_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(optimization_summary)
            logging.info(f"✓ Saved threshold optimization summary to {summary_file}")
        
        logging.info(f"✓ Saved peak classification results to {output_dir}")
        return results
        
    except Exception as e:
        import traceback
        error_msg = f"Error computing peak classification metrics: {str(e)}"
        logging.error(error_msg)
        logging.debug(f"Full traceback: {traceback.format_exc()}")
        return {'error': str(e), 'traceback': traceback.format_exc()}


def generate_threshold_optimization_summary(results: Dict[str, Any], 
                                          threshold_config: Dict[str, Any]) -> str:
    """
    Generate a summary report for threshold optimization results.
    
    Args:
        results: Peak classification results with optimization
        threshold_config: Threshold optimization configuration
        
    Returns:
        Summary text report
    """
    lines = [
        "=" * 80,
        "THRESHOLD OPTIMIZATION SUMMARY",
        "=" * 80,
        "",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    # Configuration summary
    constraints = threshold_config.get('constraints', {})
    lines.extend([
        "CONFIGURATION:",
        f"  Applied Constraints:",
        f"    - Min Recall: {constraints.get('min_recall', 'N/A')}",
        f"    - Min Precision: {constraints.get('min_precision', 'N/A')}",
        f"    - Min Specificity: {constraints.get('min_specificity', 'N/A')}",
        ""
    ])
    
    # ROC Analysis Results
    if 'roc_analysis' in results:
        roc_data = results['roc_analysis']
        lines.extend([
            "ROC ANALYSIS RESULTS:",
            f"  AUROC: {roc_data.get('auroc', 'N/A'):.4f}",
        ])
        
        if 'constrained_optimization' in roc_data:
            const_opt = roc_data['constrained_optimization']
            if const_opt.get('success'):
                # Use 'recall' key if available, fallback to 'sensitivity' for backward compatibility
                perf = const_opt['performance']
                recall_value = perf.get('recall', perf.get('sensitivity', 'N/A'))
                recall_str = f"{recall_value:.4f}" if isinstance(recall_value, (int, float)) else str(recall_value)
                
                lines.extend([
                    f"  ✓ Constrained Optimization Successful",
                    f"    - Optimal Threshold: {const_opt['optimal_threshold']:.4f}",
                    f"    - Sensitivity (Recall): {recall_str}",
                    f"    - Specificity: {const_opt['performance']['specificity']:.4f}",
                    f"    - Precision: {const_opt['performance']['precision']:.4f}",
                    f"    - F1-Score: {const_opt['performance']['f1_score']:.4f}",
                ])
            else:
                lines.extend([
                    f"  ✗ Constrained Optimization Failed",
                    f"    - Reason: {const_opt.get('message', 'Unknown')}",
                ])
                for rec in const_opt.get('recommendations', []):
                    lines.append(f"    - Recommendation: {rec}")
        lines.append("")
    
    # Precision-Recall Analysis
    if 'precision_recall_analysis' in results:
        pr_data = results['precision_recall_analysis']
        lines.extend([
            "PRECISION-RECALL ANALYSIS:",
            f"  Average Precision: {pr_data.get('average_precision', 'N/A'):.4f}",
        ])
        
        if 'optimal_threshold' in pr_data:
            opt_thresh = pr_data['optimal_threshold']
            lines.extend([
                f"  F1-Optimal Threshold: {opt_thresh.get('threshold', 'N/A'):.4f}",
                f"  F1-Score: {opt_thresh.get('f1_score', 'N/A'):.4f}",
            ])
        lines.append("")
    
    # Multi-objective optimization results
    if 'threshold_optimization' in results and 'multi_objective' in results['threshold_optimization']:
        multi_obj = results['threshold_optimization']['multi_objective']
        if 'pareto_optimal_solutions' in multi_obj:
            lines.extend([
                "MULTI-OBJECTIVE OPTIMIZATION:",
                f"  Pareto-optimal solutions found: {len(multi_obj['pareto_optimal_solutions'])}",
                f"  Top solution threshold: {multi_obj['pareto_optimal_solutions'][0]['threshold']:.4f}" if multi_obj['pareto_optimal_solutions'] else "  No solutions found",
                ""
            ])
    
    lines.extend([
        "RECOMMENDATIONS:",
        "  - Review constraint feasibility if optimization failed",
        "  - Consider multi-objective solutions for balanced performance", 
        "  - Validate optimal thresholds on independent test set",
        "",
        "=" * 80
    ])
    
    return "\n".join(lines)


def log_scalars_to_tensorboard(metrics: List[Dict[str, Any]], mode: str, writer: SummaryWriter, epoch: int = 0):
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


def log_classification_to_tensorboard(
    peak_results: Dict[str, Any],
    mode: str,
    writer: SummaryWriter,
    epoch: int = 0
):
    """
    Log classification metrics and curves to TensorBoard.
    """
    if not peak_results or 'error' in peak_results:
        return
    
    # Log classification scalars
    if 'bootstrap_metrics' in peak_results:
        bootstrap_metrics = peak_results['bootstrap_metrics']
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            if metric_name in bootstrap_metrics:
                writer.add_scalar(f'eval/{mode}/classification/{metric_name}_mean', 
                                bootstrap_metrics[metric_name]['mean'], epoch)
                writer.add_scalar(f'eval/{mode}/classification/{metric_name}_ci_lower', 
                                bootstrap_metrics[metric_name]['ci_lower'], epoch)
                writer.add_scalar(f'eval/{mode}/classification/{metric_name}_ci_upper', 
                                bootstrap_metrics[metric_name]['ci_upper'], epoch)
    
    # Log AUROC
    if 'roc_analysis' in peak_results:
        roc_data = peak_results['roc_analysis']
        writer.add_scalar(f'eval/{mode}/classification/auroc', roc_data['auroc'], epoch)
        if 'auroc_ci' in roc_data:
            writer.add_scalar(f'eval/{mode}/classification/auroc_ci_lower', 
                            roc_data['auroc_ci'][0], epoch)
            writer.add_scalar(f'eval/{mode}/classification/auroc_ci_upper', 
                            roc_data['auroc_ci'][1], epoch)
    
    # Log statistical significance results
    if 'statistical_significance' in peak_results:
        stats_data = peak_results['statistical_significance']
        for test_name, test_result in stats_data.items():
            if isinstance(test_result, dict) and 'p_value' in test_result:
                writer.add_scalar(f'eval/{mode}/stats/{test_name}_p_value', 
                                test_result['p_value'], epoch)
            if isinstance(test_result, dict) and 'effect_size' in test_result:
                writer.add_scalar(f'eval/{mode}/stats/{test_name}_effect_size', 
                                test_result['effect_size'], epoch)
    
    # Log clinical validation metrics
    if 'clinical_validation' in peak_results:
        clinical_data = peak_results['clinical_validation']
        for metric_name, metric_value in clinical_data.items():
            if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                writer.add_scalar(f'eval/{mode}/clinical/{metric_name}', metric_value, epoch)
    
    # Log prevalence
    if 'prevalence' in peak_results:
        writer.add_scalar(f'eval/{mode}/classification/prevalence', 
                         peak_results['prevalence'], epoch)


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
    torch.manual_seed(cfg['evaluation']['seed'])
    np.random.seed(cfg['evaluation']['seed'])
    
    # Device
    device = torch.device(cfg['model']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create evaluation dataset
    eval_loader, dataset = create_evaluation_dataset(cfg)
    
    # Load model and checkpoint
    model, checkpoint = load_model_and_checkpoint(cfg, device)
    
    # Setup diffusion components — always use training T for the noise schedule
    # num_steps in eval config controls DDIM sampling steps, NOT the schedule length
    training_T = cfg.get('diffusion', {}).get('num_train_steps', 1000)
    noise_schedule = prepare_noise_schedule(training_T, device)
    sampler = DDIMSampler(model, training_T, device)
    
    # TensorBoard writer
    log_dir = Path(cfg['tensorboard']['log_dir'])
    writer = SummaryWriter(log_dir) if cfg['tensorboard']['enabled'] else None
    
    # Log configuration
    if writer and cfg['tensorboard']['log_metrics']:
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
    if cfg['evaluation']['mode'] == 'reconstruction':
        logging.info("Running reconstruction evaluation...")
        recon_metrics, recon_ref, recon_gen, recon_peak_data = evaluate_reconstruction(
            model, eval_loader, noise_schedule, dataset, cfg, device
        )
        
        # Save results
        recon_summary = save_results(recon_metrics, 'reconstruction', cfg, checkpoint)
        all_results['reconstruction'] = recon_summary
        
        # Save peak classification results if available
        if recon_peak_data:
            output_dir = Path(cfg['output']['save_dir']) / 'reconstruction'
            output_dir.mkdir(parents=True, exist_ok=True)
            peak_results = save_peak_classification_results(
                recon_peak_data[0], recon_peak_data[1], 'reconstruction', cfg, output_dir
            )
            recon_summary['peak_classification'] = peak_results
            
            # Create classification visualizations
            if cfg['report']['save_roc_curves'] or cfg['report']['save_pr_curves'] or cfg['report']['save_confusion_matrices']:
                create_classification_visualizations(
                    recon_peak_data[0], recon_peak_data[1], 'reconstruction', cfg, writer, output_dir
                )
        
        # TensorBoard logging
        if writer and cfg['tensorboard']['log_metrics']:
            log_scalars_to_tensorboard(recon_metrics, 'reconstruction', writer)
            # Log classification metrics to TensorBoard
            if recon_peak_data and recon_peak_data[0] is not None:
                log_classification_to_tensorboard(peak_results, 'reconstruction', writer)
        
        # Visualizations
        if writer and cfg['tensorboard']['log_plots']:
            create_visualizations(recon_ref, recon_gen, recon_metrics, 'reconstruction', cfg, writer)
    
    # Generation evaluation
    if cfg['evaluation']['mode'] == 'generation':
        logging.info("Running generation evaluation...")
        gen_metrics, gen_ref, gen_gen, gen_peak_data = evaluate_generation(
            model, eval_loader, sampler, dataset, cfg, device
        )
        
        # Save results
        gen_summary = save_results(gen_metrics, 'generation', cfg, checkpoint)
        all_results['generation'] = gen_summary
        
        # Save peak classification results if available
        if gen_peak_data:
            output_dir = Path(cfg['output']['save_dir']) / 'generation'
            output_dir.mkdir(parents=True, exist_ok=True)
            peak_results = save_peak_classification_results(
                gen_peak_data[0], gen_peak_data[1], 'generation', cfg, output_dir
            )
            gen_summary['peak_classification'] = peak_results
            
            # Create classification visualizations
            if cfg['report']['save_roc_curves'] or cfg['report']['save_pr_curves'] or cfg['report']['save_confusion_matrices']:
                create_classification_visualizations(
                    gen_peak_data[0], gen_peak_data[1], 'generation', cfg, writer, output_dir
                )
        
        # TensorBoard logging
        if writer and cfg['tensorboard']['log_metrics']:
            log_scalars_to_tensorboard(gen_metrics, 'generation', writer)
            # Log classification metrics to TensorBoard
            if gen_peak_data and gen_peak_data[0] is not None:
                log_classification_to_tensorboard(peak_results, 'generation', writer)
        
        # Visualizations
        if writer and cfg['tensorboard']['log_plots']:
            create_visualizations(gen_ref, gen_gen, gen_metrics, 'generation', cfg, writer)
    
    if writer:
        writer.close()
    
    # Save unified evaluation summary
    save_evaluation_summary(all_results, cfg)
    
    logging.info("="*60)
    logging.info("EVALUATION COMPLETED")
    logging.info("="*60)
    logging.info(f"Results saved to: {cfg['output']['save_dir']}")
    if cfg['tensorboard']['enabled']:
        logging.info(f"TensorBoard logs: {log_dir}")
    
    return all_results


if __name__ == "__main__":
    main()
