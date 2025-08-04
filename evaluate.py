#!/usr/bin/env python3
"""
ABR Model Evaluation Script

Comprehensive evaluation script for the ABR Hierarchical U-Net model.
Evaluates model performance on test data and generates detailed reports.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --config configs/config.yaml
    python evaluate.py --checkpoint checkpoints/best_model.pt --output_dir results/evaluation
    python evaluate.py --checkpoint checkpoints/best_model.pt --generate_samples 100

Author: AI Assistant
Date: January 2025
"""

# Suppress CUDA warnings before any imports
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import sys
import argparse
import logging
from pathlib import Path
import json
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from omegaconf import DictConfig
from training.config_loader import load_config
from data.dataset import ABRDataset, stratified_patient_split, create_optimized_dataloaders
from models.hierarchical_unet import OptimizedHierarchicalUNet
from evaluation.metrics import compute_all_metrics, ABRMetrics
from utils.sampling import DDIMSampler, create_ddim_sampler


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def load_model_and_config(checkpoint_path: str, config_path: Optional[str] = None) -> tuple:
    """
    Load model and configuration from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file (uses checkpoint config if None)
        
    Returns:
        Tuple of (model, config, device)
    """
    logger = logging.getLogger(__name__)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        # Use config from checkpoint
        config_dict = checkpoint.get('config', {})
        if not config_dict:
            raise ValueError("No configuration found in checkpoint and no config path provided")
        config = DictConfig(config_dict)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model_config = config.model.architecture
    model = OptimizedHierarchicalUNet(
        signal_length=model_config.signal_length,
        static_dim=model_config.static_dim,
        base_channels=model_config.base_channels,
        n_levels=model_config.n_levels,
        num_classes=model_config.n_classes,  # Correct parameter name
        dropout=model_config.dropout,
        num_s4_layers=model_config.encoder.n_s4_layers,  # Correct parameter name
        s4_state_size=model_config.encoder.d_state,      # Correct parameter name
        num_transformer_layers=model_config.decoder.n_transformer_layers,  # Correct parameter name
        num_heads=model_config.decoder.n_heads  # Correct parameter name
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded with {total_params:,} parameters")
    
    return model, config, device


def create_test_loader(config: DictConfig) -> DataLoader:
    """Create test data loader."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading dataset from: {config.data.dataset_path}")
    
    # Use the optimized dataloaders function to get test loader
    _, _, test_loader, full_dataset = create_optimized_dataloaders(
        data_path=config.data.dataset_path,
        batch_size=config.data.dataloader.batch_size,
        train_ratio=config.data.splits.train_ratio,
        val_ratio=config.data.splits.val_ratio,
        test_ratio=config.data.splits.test_ratio,
        num_workers=config.data.dataloader.num_workers,
        pin_memory=config.data.dataloader.pin_memory,
        random_state=config.data.splits.random_seed
    )
    
    logger.info(f"Test batches: {len(test_loader)}")
    
    return test_loader


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: DictConfig
) -> Dict[str, Any]:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Computation device
        config: Configuration
        
    Returns:
        Dictionary containing evaluation results
    """
    logger = logging.getLogger(__name__)
    
    model.eval()
    all_metrics = []
    all_outputs = []
    all_targets = []
    
    logger.info("Evaluating model on test data...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass (inference mode - no noise)
            # For evaluation, we use the denoising capability
            batch_size = batch['signal'].size(0)
            
            # Use clean signal as input for direct evaluation
            outputs = model(
                batch['signal'],  # Clean signal
                batch['static_params'],
                torch.zeros(batch_size, device=device, dtype=torch.long)  # t=0 for clean
            )
            
            # Compute metrics for this batch
            batch_metrics = compute_all_metrics(
                model_outputs=outputs,
                batch_targets=batch,
                static_params=batch['static_params']
            )
            
            all_metrics.append(batch_metrics)
            
            # Store outputs and targets for analysis
            batch_outputs = {}
            batch_targets = {}
            
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    batch_outputs[key] = value.cpu().numpy()
                elif isinstance(value, (list, tuple)):
                    batch_outputs[key] = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in value]
                else:
                    batch_outputs[key] = value
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_targets[key] = value.cpu().numpy()
                else:
                    batch_targets[key] = value
            
            all_outputs.append(batch_outputs)
            all_targets.append(batch_targets)
    
    # Aggregate metrics
    logger.info("Aggregating metrics...")
    aggregated_metrics = _aggregate_metrics(all_metrics)
    
    # Create results dictionary
    results = {
        'metrics': aggregated_metrics.to_dict(),
        'detailed_metrics': [m.to_dict() for m in all_metrics],
        'sample_outputs': all_outputs[:10],  # Store first 10 for visualization
        'sample_targets': all_targets[:10],
        'num_samples': len(test_loader.dataset),
        'num_batches': len(test_loader)
    }
    
    return results


def _aggregate_metrics(metrics_list: List[ABRMetrics]) -> ABRMetrics:
    """Aggregate metrics across batches."""
    if not metrics_list:
        return ABRMetrics()
    
    # Average all metrics
    aggregated = ABRMetrics()
    
    for attr in aggregated.__dict__.keys():
        values = [getattr(m, attr) for m in metrics_list if hasattr(m, attr)]
        if values and all(isinstance(v, (int, float)) for v in values):
            setattr(aggregated, attr, np.mean(values))
    
    return aggregated


def generate_samples(
    model: torch.nn.Module,
    config: DictConfig,
    device: torch.device,
    num_samples: int = 100,
    output_dir: str = "outputs/generated_samples"
) -> Dict[str, Any]:
    """
    Generate samples using the trained model.
    
    Args:
        model: Trained model
        config: Configuration
        device: Computation device
        num_samples: Number of samples to generate
        output_dir: Output directory
        
    Returns:
        Dictionary containing generated samples
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating {num_samples} samples...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DDIM sampler
    sampler = create_ddim_sampler(
        noise_schedule_type=config.diffusion.noise_schedule.get('type', 'cosine'),
        num_timesteps=config.diffusion.noise_schedule.get('num_timesteps', 1000),
        eta=config.diffusion.sampling.get('ddim_eta', 0.0),
        clip_denoised=config.diffusion.sampling.get('clip_denoised', True)
    )
    
    # Create random static parameters
    batch_size = min(num_samples, 16)  # Generate in batches
    signal_length = config.data.signal_length
    static_dim = config.data.static_dim
    
    generated_samples = {
        'signals': [],
        'static_params': [],
        'peaks': [],
        'classifications': [],
        'thresholds': []
    }
    
    model.eval()
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Generate random static parameters
            static_params = torch.randn(current_batch_size, static_dim, device=device)
            
            # Generate signals
            shape = (current_batch_size, 1, signal_length)
            generated_signals = sampler.sample(
                model=model,
                shape=shape,
                static_params=static_params,
                device=device,
                num_steps=config.diffusion.sampling.get('num_sampling_steps', 50),
                cfg_scale=config.inference.generation.get('guidance_scale', 1.0),
                progress=False
            )
            
            # Get model predictions for generated signals
            outputs = model(
                generated_signals,
                static_params,
                torch.zeros(current_batch_size, device=device, dtype=torch.long)
            )
            
            # Store generated data
            generated_samples['signals'].append(generated_signals.cpu().numpy())
            generated_samples['static_params'].append(static_params.cpu().numpy())
            
            if 'peak' in outputs:
                peak_outputs = outputs['peak']
                if isinstance(peak_outputs, (list, tuple)):
                    peaks = {
                        'existence': torch.sigmoid(peak_outputs[0]).cpu().numpy(),
                        'latency': peak_outputs[1].cpu().numpy(),
                        'amplitude': peak_outputs[2].cpu().numpy()
                    }
                else:
                    peaks = peak_outputs.cpu().numpy()
                generated_samples['peaks'].append(peaks)
            
            if 'class' in outputs or 'classification_logits' in outputs:
                class_key = 'class' if 'class' in outputs else 'classification_logits'
                classifications = F.softmax(outputs[class_key], dim=1).cpu().numpy()
                generated_samples['classifications'].append(classifications)
            
            if 'threshold' in outputs:
                thresholds = outputs['threshold'].cpu().numpy()
                generated_samples['thresholds'].append(thresholds)
    
    # Concatenate all generated data
    for key in generated_samples:
        if generated_samples[key]:
            if key == 'peaks' and isinstance(generated_samples[key][0], dict):
                # Handle peak dictionary
                concatenated = {}
                for peak_key in generated_samples[key][0].keys():
                    concatenated[peak_key] = np.concatenate([batch[peak_key] for batch in generated_samples[key]])
                generated_samples[key] = concatenated
            else:
                generated_samples[key] = np.concatenate(generated_samples[key])
    
    # Save generated samples
    save_path = os.path.join(output_dir, "generated_samples.npz")
    np.savez(save_path, **generated_samples)
    logger.info(f"Generated samples saved to: {save_path}")
    
    return generated_samples


def create_evaluation_report(
    results: Dict[str, Any],
    output_dir: str,
    config: DictConfig
) -> None:
    """Create evaluation report."""
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    
    # Create text report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("ABR Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Number of test samples: {results['num_samples']}\n")
        f.write(f"Number of batches: {results['num_batches']}\n\n")
        
        # Signal quality metrics
        f.write("Signal Quality Metrics:\n")
        f.write("-" * 25 + "\n")
        f.write(f"MSE: {results['metrics']['signal_mse']:.6f}\n")
        f.write(f"MAE: {results['metrics']['signal_mae']:.6f}\n")
        f.write(f"Correlation: {results['metrics']['signal_correlation']:.4f}\n")
        f.write(f"SNR: {results['metrics']['signal_snr']:.2f} dB\n")
        f.write(f"Spectral Similarity: {results['metrics']['spectral_similarity']:.4f}\n")
        f.write(f"Morphological Similarity: {results['metrics']['morphological_similarity']:.4f}\n\n")
        
        # Peak prediction metrics
        f.write("Peak Prediction Metrics:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Existence Accuracy: {results['metrics']['peak_existence_accuracy']:.4f}\n")
        f.write(f"Existence F1: {results['metrics']['peak_existence_f1']:.4f}\n")
        f.write(f"Latency MAE: {results['metrics']['peak_latency_mae']:.4f} ms\n")
        f.write(f"Latency RMSE: {results['metrics']['peak_latency_rmse']:.4f} ms\n")
        f.write(f"Amplitude MAE: {results['metrics']['peak_amplitude_mae']:.4f} μV\n")
        f.write(f"Amplitude RMSE: {results['metrics']['peak_amplitude_rmse']:.4f} μV\n\n")
        
        # Classification metrics
        f.write("Classification Metrics:\n")
        f.write("-" * 23 + "\n")
        f.write(f"Accuracy: {results['metrics']['classification_accuracy']:.4f}\n")
        f.write(f"F1 (macro): {results['metrics']['classification_f1_macro']:.4f}\n")
        f.write(f"F1 (weighted): {results['metrics']['classification_f1_weighted']:.4f}\n")
        f.write(f"Precision (macro): {results['metrics']['classification_precision_macro']:.4f}\n")
        f.write(f"Recall (macro): {results['metrics']['classification_recall_macro']:.4f}\n")
        f.write(f"AUC: {results['metrics']['classification_auc']:.4f}\n\n")
        
        # Threshold metrics
        f.write("Threshold Metrics:\n")
        f.write("-" * 18 + "\n")
        f.write(f"MAE: {results['metrics']['threshold_mae']:.2f} dB nHL\n")
        f.write(f"RMSE: {results['metrics']['threshold_rmse']:.2f} dB nHL\n")
        f.write(f"R²: {results['metrics']['threshold_r2']:.4f}\n")
        f.write(f"Correlation: {results['metrics']['threshold_correlation']:.4f}\n\n")
        
        # Clinical metrics
        f.write("Clinical Metrics:\n")
        f.write("-" * 16 + "\n")
        f.write(f"Clinical Concordance: {results['metrics']['clinical_concordance']:.4f}\n")
        f.write(f"Diagnostic Agreement: {results['metrics']['diagnostic_agreement']:.4f}\n")
        f.write(f"Severity Correlation: {results['metrics']['severity_correlation']:.4f}\n")
    
    logger.info(f"Evaluation report saved to: {report_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="ABR Model Evaluation")
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (uses checkpoint config if not provided)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--generate_samples",
        type=int,
        default=0,
        help="Number of samples to generate (0 to skip generation)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("ABR Model Evaluation")
    logger.info("=" * 80)
    
    try:
        # Load model and configuration
        model, config, device = load_model_and_config(args.checkpoint, args.config)
        
        # Override batch size if specified
        if args.batch_size is not None:
            config.data.dataloader.batch_size = args.batch_size
        
        # Create test data loader
        test_loader = create_test_loader(config)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Evaluate model
        logger.info("Starting evaluation...")
        results = evaluate_model(model, test_loader, device, config)
        
        # Create evaluation report
        create_evaluation_report(results, args.output_dir, config)
        
        # Print summary
        logger.info("\nEvaluation Results Summary:")
        logger.info("-" * 40)
        logger.info(f"Signal MSE: {results['metrics']['signal_mse']:.6f}")
        logger.info(f"Signal Correlation: {results['metrics']['signal_correlation']:.4f}")
        logger.info(f"Peak Existence Accuracy: {results['metrics']['peak_existence_accuracy']:.4f}")
        logger.info(f"Classification Accuracy: {results['metrics']['classification_accuracy']:.4f}")
        logger.info(f"Threshold MAE: {results['metrics']['threshold_mae']:.2f} dB nHL")
        logger.info(f"Clinical Concordance: {results['metrics']['clinical_concordance']:.4f}")
        
        # Generate samples if requested
        if args.generate_samples > 0:
            sample_output_dir = os.path.join(args.output_dir, "generated_samples")
            generated_samples = generate_samples(
                model, config, device, args.generate_samples, sample_output_dir
            )
            logger.info(f"Generated {args.generate_samples} samples")
        
        logger.info(f"\nEvaluation completed! Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()