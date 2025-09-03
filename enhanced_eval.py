#!/usr/bin/env python3
"""
Enhanced ABR Transformer Evaluation Script

This script provides comprehensive evaluation capabilities including:
- Signal generation/reconstruction evaluation
- 5th peak classification evaluation with statistical analysis
- Clinical validation metrics
- Publication-quality visualization
- Enhanced reporting with confidence intervals
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.abr_transformer import ABRTransformerGenerator
from data.dataset import ABRDataset
from torch.utils.data import DataLoader
from evaluation.analysis import (
    bootstrap_classification_metrics,
    statistical_significance_tests,
    roc_analysis,
    precision_recall_analysis,
    clinical_validation_analysis
)
from evaluation.visualization import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_threshold_analysis,
    plot_clinical_validation_dashboard
)
from utils.metrics import compute_classification_metrics
from utils.tb import TensorBoardLogger
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_evaluation_dataset(config: Dict) -> Tuple[ABRDataset, DataLoader]:
    """Create evaluation dataset with peak labels if enabled."""
    dataset_path = config['dataset']['data_path']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    pin_memory = config['dataset']['pin_memory']
    
    # Check if peak classification evaluation is enabled
    peak_classification_enabled = config['metrics'].get('peak_classification', {}).get('enabled', False)
    
    # Create dataset with peak labels if enabled
    if peak_classification_enabled:
        print("Creating dataset with peak labels for classification evaluation...")
        dataset = ABRDataset(
            data_path=dataset_path,
            return_peak_labels=True,  # Enable peak labels
            sequence_length=config['model']['sequence_length'],
            static_order=config['model']['static_order']
        )
        
        # Log peak label distribution
        peak_labels = [sample['peak_exists'] for sample in dataset]
        positive_samples = sum(peak_labels)
        total_samples = len(peak_labels)
        prevalence = positive_samples / total_samples
        
        print(f"Peak label distribution:")
        print(f"  Total samples: {total_samples}")
        print(f"  Positive samples (peak exists): {positive_samples}")
        print(f"  Negative samples (no peak): {total_samples - positive_samples}")
        print(f"  Prevalence: {prevalence:.3f}")
        
        if prevalence < 0.1 or prevalence > 0.9:
            print(f"Warning: Highly imbalanced dataset (prevalence: {prevalence:.3f})")
    else:
        print("Creating dataset without peak labels...")
        dataset = ABRDataset(
            data_path=dataset_path,
            return_peak_labels=False,
            sequence_length=config['model']['sequence_length'],
            static_order=config['model']['static_order']
        )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for evaluation
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn
    )
    
    return dataset, dataloader


def load_model(config: Dict, device: torch.device) -> ABRTransformerGenerator:
    """Load trained ABR Transformer model."""
    checkpoint_path = config['model']['checkpoint_path']
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        # Fallback to config file
        model_config = config['model']
    
    # Create model
    model = ABRTransformerGenerator(
        input_channels=model_config['input_channels'],
        static_dim=model_config['static_dim'],
        sequence_length=model_config['sequence_length'],
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        ff_mult=model_config['ff_mult'],
        dropout=model_config['dropout'],
        use_timestep_cond=model_config.get('use_timestep_cond', True),
        use_static_film=model_config.get('use_static_film', True)
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")
    
    return model


def calculate_signal_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    """Calculate comprehensive signal quality metrics."""
    metrics = {}
    
    # MSE (already calculated elsewhere, but included for completeness)
    mse = torch.mean((original - reconstructed) ** 2).item()
    metrics['mse'] = mse
    
    # Correlation
    correlation = torch.corrcoef(torch.stack([original.flatten(), reconstructed.flatten()]))[0, 1].item()
    metrics['correlation'] = correlation
    
    # SNR (Signal-to-Noise Ratio)
    signal_power = torch.mean(original ** 2).item()
    noise_power = mse
    if noise_power > 0:
        snr = 10 * torch.log10(torch.tensor(signal_power / noise_power)).item()
        metrics['snr'] = snr
    else:
        metrics['snr'] = float('inf')
    
    # PSNR (Peak Signal-to-Noise Ratio)
    max_val = torch.max(original).item()
    if mse > 0:
        psnr = 20 * torch.log10(torch.tensor(max_val / torch.sqrt(torch.tensor(mse)))).item()
        metrics['psnr'] = psnr
    else:
        metrics['psnr'] = float('inf')
    
    # SSIM (Structural Similarity Index)
    try:
        # Simple SSIM approximation
        mu_x = torch.mean(original)
        mu_y = torch.mean(reconstructed)
        sigma_x = torch.std(original)
        sigma_y = torch.std(reconstructed)
        sigma_xy = torch.mean((original - mu_x) * (reconstructed - mu_y))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
        metrics['ssim'] = ssim.item()
    except:
        metrics['ssim'] = float('nan')
    
    return metrics


def evaluate_reconstruction(model: ABRTransformerGenerator, 
                          dataloader: DataLoader, 
                          device: torch.device,
                          config: Dict) -> Dict:
    """Evaluate model reconstruction performance with peak classification."""
    print("Evaluating reconstruction performance...")
    
    model.eval()
    
    # Initialize metrics collection
    signal_metrics = {
        'mse': [],
        'correlation': [],
        'snr': [],
        'psnr': [],
        'ssim': []
    }
    
    # Initialize peak classification data collection
    peak_classification_enabled = config['metrics'].get('peak_classification', {}).get('enabled', False)
    all_peak_logits = []
    all_peak_targets = []
    all_predictions = []
    all_targets = []
    
    # Get threshold from config (default 0.5)
    threshold = config['metrics'].get('peak_classification', {}).get('threshold', 0.5)
    
    # Get sample limit
    num_samples = config['evaluation'].get('num_samples', None)
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(dataloader)}")
            
            # Move batch to device
            x0 = batch['x0'].to(device)
            stat = batch['stat'].to(device)
            
            # Get peak labels if available
            peak_targets = None
            if peak_classification_enabled and 'peak_exists' in batch:
                peak_targets = batch['peak_exists'].to(device)
            
            # Forward pass
            model_output = model(x0, stat)
            
            # Extract signal and peak outputs
            reconstructed_signal = model_output['signal']
            peak_logits = model_output.get('peak_5th_exists', None)
            
            # Calculate signal metrics
            for i in range(x0.size(0)):
                # Check sample limit
                if num_samples is not None and sample_count >= num_samples:
                    break
                
                # Calculate comprehensive signal metrics
                batch_metrics = calculate_signal_metrics(x0[i], reconstructed_signal[i])
                
                for metric_name, value in batch_metrics.items():
                    if not np.isnan(value) and not np.isinf(value):
                        signal_metrics[metric_name].append(value)
                
                sample_count += 1
            
            if num_samples is not None and sample_count >= num_samples:
                break
            
            # Collect peak classification data
            if peak_classification_enabled and peak_logits is not None and peak_targets is not None:
                all_peak_logits.extend(peak_logits.cpu().numpy())
                all_peak_targets.extend(peak_targets.cpu().numpy())
                
                # Convert logits to probabilities and then to predictions using configurable threshold
                probabilities = torch.sigmoid(peak_logits).cpu().numpy()
                predictions = (probabilities > threshold).astype(int)
                all_predictions.extend(predictions)
                all_targets.extend(peak_targets.cpu().numpy())
    
    # Calculate average signal metrics
    avg_signal_metrics = {}
    for metric_name, values in signal_metrics.items():
        if values:
            avg_signal_metrics[metric_name] = np.mean(values)
            avg_signal_metrics[f'{metric_name}_std'] = np.std(values)
        else:
            # Gate metrics that weren't computed
            if metric_name in ['snr', 'psnr', 'ssim']:
                avg_signal_metrics[metric_name] = None
                avg_signal_metrics[f'{metric_name}_std'] = None
    
    results = {
        'signal_metrics': avg_signal_metrics,
        'mode': 'reconstruction',
        'samples_evaluated': sample_count
    }
    
    # Add peak classification results if enabled
    if peak_classification_enabled and all_peak_logits:
        print("Computing peak classification metrics...")
        peak_results = compute_peak_classification_metrics(
            all_peak_logits, all_peak_targets, config
        )
        results.update(peak_results)
        
        # Add confusion matrix data
        results['confusion_matrix'] = np.array([
            [sum((np.array(all_targets) == 0) & (np.array(all_predictions) == 0)),  # TN
             sum((np.array(all_targets) == 0) & (np.array(all_predictions) == 1))], # FP
            [sum((np.array(all_targets) == 1) & (np.array(all_predictions) == 0)),  # FN
             sum((np.array(all_targets) == 1) & (np.array(all_predictions) == 1))]  # TP
        ])
        
        # Store actual classification samples for visualization
        results['classification_samples'] = {
            'logits': all_peak_logits,
            'targets': all_peak_targets,
            'predictions': all_predictions,
            'probabilities': torch.sigmoid(torch.tensor(all_peak_logits)).numpy().tolist(),
            'threshold': threshold
        }
    
    return results


def evaluate_generation(model: ABRTransformerGenerator, 
                       dataloader: DataLoader, 
                       device: torch.device,
                       config: Dict) -> Dict:
    """Evaluate model generation performance with peak classification."""
    print("Evaluating generation performance...")
    
    model.eval()
    
    # Get generation parameters
    generation_config = config['evaluation']['generation']
    num_steps = generation_config['num_steps']
    guidance_scale = generation_config['guidance_scale']
    temperature = generation_config['temperature']
    
    # Initialize metrics collection
    signal_metrics = {
        'mse': [],
        'correlation': [],
        'snr': [],
        'psnr': [],
        'ssim': []
    }
    
    # Initialize peak classification data collection
    peak_classification_enabled = config['metrics'].get('peak_classification', {}).get('enabled', False)
    all_peak_logits = []
    all_peak_targets = []
    all_predictions = []
    all_targets = []
    
    # Get threshold from config (default 0.5)
    threshold = config['metrics'].get('peak_classification', {}).get('threshold', 0.5)
    
    # Get sample limit
    num_samples = config['evaluation'].get('num_samples', None)
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(dataloader)}")
            
            # Move batch to device
            stat = batch['stat'].to(device)
            
            # Get peak labels if available
            peak_targets = None
            if peak_classification_enabled and 'peak_exists' in batch:
                peak_targets = batch['peak_exists'].to(device)
            
            # Generate signals
            generated_signals = model.generate(
                stat, 
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                temperature=temperature
            )
            
            # Extract peak outputs if available
            peak_logits = None
            if hasattr(generated_signals, 'get'):
                peak_logits = generated_signals.get('peak_5th_exists', None)
                generated_signal = generated_signals['signal']
            else:
                # Assume generated_signals is just the signal tensor
                generated_signal = generated_signals
            
            # Calculate signal metrics (compare with original if available)
            if 'x0' in batch:
                x0 = batch['x0'].to(device)
                for i in range(x0.size(0)):
                    # Check sample limit
                    if num_samples is not None and sample_count >= num_samples:
                        break
                    
                    # Calculate comprehensive signal metrics
                    batch_metrics = calculate_signal_metrics(x0[i], generated_signal[i])
                    
                    for metric_name, value in batch_metrics.items():
                        if not np.isnan(value) and not np.isinf(value):
                            signal_metrics[metric_name].append(value)
                    
                    sample_count += 1
                
                if num_samples is not None and sample_count >= num_samples:
                    break
            
            # Collect peak classification data
            if peak_classification_enabled and peak_logits is not None and peak_targets is not None:
                all_peak_logits.extend(peak_logits.cpu().numpy())
                all_peak_targets.extend(peak_targets.cpu().numpy())
                
                # Convert logits to probabilities and then to predictions using configurable threshold
                probabilities = torch.sigmoid(peak_logits).cpu().numpy()
                predictions = (probabilities > threshold).astype(int)
                all_predictions.extend(predictions)
                all_targets.extend(peak_targets.cpu().numpy())
    
    # Calculate average signal metrics
    avg_signal_metrics = {}
    for metric_name, values in signal_metrics.items():
        if values:
            avg_signal_metrics[metric_name] = np.mean(values)
            avg_signal_metrics[f'{metric_name}_std'] = np.std(values)
        else:
            # Gate metrics that weren't computed
            if metric_name in ['snr', 'psnr', 'ssim']:
                avg_signal_metrics[metric_name] = None
                avg_signal_metrics[f'{metric_name}_std'] = None
    
    results = {
        'signal_metrics': avg_signal_metrics,
        'mode': 'generation',
        'samples_evaluated': sample_count,
        'generation_params': {
            'num_steps': num_steps,
            'guidance_scale': guidance_scale,
            'temperature': temperature
        }
    }
    
    # Add peak classification results if enabled
    if peak_classification_enabled and all_peak_logits:
        print("Computing peak classification metrics...")
        peak_results = compute_peak_classification_metrics(
            all_peak_logits, all_peak_targets, config
        )
        results.update(peak_results)
        
        # Add confusion matrix data
        results['confusion_matrix'] = np.array([
            [sum((np.array(all_targets) == 0) & (np.array(all_predictions) == 0)),  # TN
             sum((np.array(all_targets) == 0) & (np.array(all_predictions) == 1))], # FP
            [sum((np.array(all_targets) == 1) & (np.array(all_predictions) == 0)),  # FN
             sum((np.array(all_targets) == 1) & (np.array(all_predictions) == 1))]  # TP
        ])
        
        # Store actual classification samples for visualization
        results['classification_samples'] = {
            'logits': all_peak_logits,
            'targets': all_peak_targets,
            'predictions': all_predictions,
            'probabilities': torch.sigmoid(torch.tensor(all_peak_logits)).numpy().tolist(),
            'threshold': threshold
        }
    
    return results


def compute_peak_classification_metrics(logits: List[float], 
                                      targets: List[int], 
                                      config: Dict) -> Dict:
    """Compute comprehensive peak classification metrics with statistical analysis."""
    logits = np.array(logits)
    targets = np.array(targets)
    
    print(f"Computing classification metrics for {len(logits)} samples...")
    
    # Basic classification metrics
    basic_metrics = compute_classification_metrics(logits, targets)
    
    # Statistical significance testing
    statistical_config = config['metrics'].get('statistical_analysis', {})
    significance_results = statistical_significance_tests(
        logits, targets,
        prevalence=np.mean(targets),
        multiple_testing_correction=statistical_config.get('multiple_testing_correction', 'bonferroni')
    )
    
    # Bootstrap confidence intervals
    bootstrap_config = config['metrics'].get('peak_classification', {})
    n_bootstrap = bootstrap_config.get('bootstrap_ci', 1000)
    confidence_level = statistical_config.get('confidence_level', 0.95)
    bootstrap_method = statistical_config.get('bootstrap_method', 'percentile')
    
    bootstrap_results = bootstrap_classification_metrics(
        logits, targets,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=bootstrap_method
    )
    
    # ROC analysis
    roc_config = config['metrics'].get('clinical_metrics', {})
    if roc_config.get('sensitivity_analysis', True):
        roc_results = roc_analysis(
            logits, targets,
            specificity_targets=roc_config.get('specificity_targets', [0.8, 0.9, 0.95])
        )
    else:
        roc_results = {}
    
    # Precision-recall analysis
    pr_results = precision_recall_analysis(logits, targets)
    
    # Clinical validation analysis - use peak_classification section for gating
    if config['metrics']['peak_classification'].get('clinical_validation', True):
        clinical_results = clinical_validation_analysis(
            logits, targets,
            prevalence=np.mean(targets)
        )
    else:
        clinical_results = {}
    
    # Combine all results
    results = {
        'classification_metrics': basic_metrics,
        'statistical_significance': significance_results,
        'bootstrap_confidence_intervals': bootstrap_results,
        'roc_analysis': roc_results,
        'precision_recall_analysis': pr_results,
        'clinical_validation': clinical_results,
        'sample_info': {
            'total_samples': len(logits),
            'positive_samples': int(np.sum(targets)),
            'negative_samples': int(np.sum(1 - targets)),
            'prevalence': float(np.mean(targets))
        }
    }
    
    return results


def to_serializable(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(to_serializable(item) for item in obj)
    else:
        return obj


def save_results(results: Dict, config: Dict, output_dir: str) -> None:
    """Save comprehensive evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = output_path / f"evaluation_results_{timestamp}.json"
    
    # Convert all numpy types to Python types for JSON serialization
    serializable_results = to_serializable(results)
    
    # Add metadata
    serializable_results['metadata'] = {
        'timestamp': timestamp,
        'evaluation_config': to_serializable(config),
        'model_info': {
            'checkpoint_path': config['model']['checkpoint_path'],
            'sequence_length': config['model']['sequence_length'],
            'd_model': config['model']['d_model']
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Save summary CSV
    if 'classification_metrics' in results:
        summary_data = {
            'metric': [],
            'value': [],
            'confidence_interval_lower': [],
            'confidence_interval_upper': []
        }
        
        # Add basic classification metrics
        for metric_name, value in results['classification_metrics'].items():
            summary_data['metric'].append(metric_name)
            summary_data['value'].append(value)
            
            # Add confidence intervals if available
            if metric_name in results['bootstrap_confidence_intervals']:
                ci_data = results['bootstrap_confidence_intervals'][metric_name]
                summary_data['confidence_interval_lower'].append(ci_data['ci_lower'])
                summary_data['confidence_interval_upper'].append(ci_data['ci_upper'])
            else:
                summary_data['confidence_interval_lower'].append(None)
                summary_data['confidence_interval_upper'].append(None)
        
        # Add signal metrics
        if 'signal_metrics' in results:
            for metric_name, value in results['signal_metrics'].items():
                if not metric_name.endswith('_std'):
                    summary_data['metric'].append(metric_name)
                    summary_data['value'].append(value)
                    summary_data['confidence_interval_lower'].append(None)
                    summary_data['confidence_interval_upper'].append(None)
        
        # Create and save summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_path / f"evaluation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to: {summary_file}")
    
    # Save evaluation summary for ComparativeAnalyzer compatibility
    evaluation_summary = {
        'metrics': {},
        'timestamp': timestamp,
        'model_config': {
            'checkpoint_path': config['model']['checkpoint_path'],
            'sequence_length': config['model']['sequence_length'],
            'd_model': config['model']['d_model'],
            'n_layers': config['model']['n_layers'],
            'n_heads': config['model']['n_heads']
        },
        'dataset_info': {
            'data_path': config['dataset']['data_path'],
            'batch_size': config['dataset']['batch_size']
        },
        'evaluation_config': {
            'mode': config['evaluation']['mode'],
            'num_samples': config['evaluation'].get('num_samples', 1000),
            'seed': config['evaluation'].get('seed', 42)
        }
    }
    
    # Aggregate key scalar metrics
    if 'signal_metrics' in results:
        for metric_name, value in results['signal_metrics'].items():
            if not metric_name.endswith('_std'):
                evaluation_summary['metrics'][f'signal_{metric_name}'] = float(value)
    
    if 'classification_metrics' in results:
        for metric_name, value in results['classification_metrics'].items():
            evaluation_summary['metrics'][f'classification_{metric_name}'] = float(value)
    
    # Save evaluation summary
    summary_file = output_path / f"evaluation_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"Evaluation summary saved to: {summary_file}")


def create_visualizations(results: Dict, config: Dict, output_dir: str) -> None:
    """Create comprehensive visualizations for evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if peak classification evaluation was performed
    if 'classification_metrics' not in results:
        print("No classification metrics available for visualization")
        return
    
    print("Creating visualizations...")
    
    # Use stored classification samples instead of generating mock data
    if 'classification_samples' in results:
        logits = results['classification_samples']['logits']
        targets = results['classification_samples']['targets']
        
        # Create ROC curve
        if 'roc_analysis' in results and results['roc_analysis']:
            roc_fig = plot_roc_curve(
                results['roc_analysis'],
                save_path=str(output_path / f"roc_curve_{timestamp}"),
                title="ROC Curve - Peak Classification"
            )
            plt.close(roc_fig)
        
        # Create PR curve
        if 'precision_recall_analysis' in results and results['precision_recall_analysis']:
            pr_fig = plot_precision_recall_curve(
                results['precision_recall_analysis'],
                save_path=str(output_path / f"pr_curve_{timestamp}"),
                title="Precision-Recall Curve - Peak Classification"
            )
            plt.close(pr_fig)
        
        # Create confusion matrix
        if 'confusion_matrix' in results:
            cm_fig = plot_confusion_matrix(
                results['confusion_matrix'],
                save_path=str(output_path / f"confusion_matrix_{timestamp}"),
                title="Confusion Matrix - Peak Classification"
            )
            plt.close(cm_fig)
        
        # Create threshold analysis using actual data
        threshold_fig = plot_threshold_analysis(
            logits, targets,
            save_path=str(output_path / f"threshold_analysis_{timestamp}"),
            title="Threshold Analysis - Peak Classification"
        )
        plt.close(threshold_fig)
        
        # Create clinical validation dashboard
        if 'clinical_validation' in results and results['clinical_validation']:
            dashboard_fig = plot_clinical_validation_dashboard(
                results['clinical_validation'],
                results.get('roc_analysis', {}),
                results.get('precision_recall_analysis', {}),
                save_path=str(output_path / f"clinical_dashboard_{timestamp}"),
                title="Clinical Validation Dashboard"
            )
            plt.close(dashboard_fig)
    else:
        print("No classification samples available for visualization")
    
    print(f"Visualizations saved to: {output_path}")


def log_to_tensorboard(results: Dict, config: Dict, log_dir: str) -> None:
    """Log evaluation results to TensorBoard."""
    if not config.get('tensorboard', {}).get('enabled', False):
        return
    
    print("Logging results to TensorBoard...")
    
    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger(log_dir)
    
    # Log signal metrics
    if 'signal_metrics' in results:
        for metric_name, value in results['signal_metrics'].items():
            if not metric_name.endswith('_std'):
                tb_logger.log_scalar(f'signal/{metric_name}', value, 0)
    
    # Log classification metrics
    if 'classification_metrics' in results:
        for metric_name, value in results['classification_metrics'].items():
            tb_logger.log_scalar(f'classification/{metric_name}', value, 0)
    
    # Log confidence intervals
    if 'bootstrap_confidence_intervals' in results:
        for metric_name, ci_data in results['bootstrap_confidence_intervals'].items():
            tb_logger.log_scalar(f'confidence_intervals/{metric_name}_lower', ci_data['ci_lower'], 0)
            tb_logger.log_scalar(f'confidence_intervals/{metric_name}_upper', ci_data['ci_upper'], 0)
    
    # Log ROC curve
    if 'roc_analysis' in results and results['roc_analysis']:
        roc_data = results['roc_analysis']
        if 'roc_curve' in roc_data:
            # Log AUROC
            tb_logger.log_scalar('roc/auroc', roc_data['auroc'], 0)
            
            # Log ROC curve data
            fpr = roc_data['roc_curve']['fpr']
            tpr = roc_data['roc_curve']['tpr']
            
            # Create ROC curve plot for TensorBoard
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, 'b-', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve (AUROC = {roc_data["auroc"]:.3f})')
            ax.grid(True, alpha=0.3)
            
            tb_logger.log_figure('roc_curve', fig, 0)
            plt.close(fig)
    
    # Log confusion matrix
    if 'confusion_matrix' in results:
        cm = results['confusion_matrix']
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Create heatmap
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Peak', 'Peak'])
        ax.set_yticklabels(['No Peak', 'Peak'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
        
        tb_logger.log_figure('confusion_matrix', fig, 0)
        plt.close(fig)
    
    print(f"TensorBoard logs saved to: {log_dir}")


def main():
    """Main evaluation function."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced ABR Transformer Evaluation')
    parser.add_argument('--config', type=str, default='configs/eval.yaml',
                       help='Path to configuration file (default: configs/eval.yaml)')
    parser.add_argument('--mode', type=str, choices=['reconstruction', 'generation'],
                       help='Override evaluation mode from config')
    parser.add_argument('--checkpoint', type=str,
                       help='Override checkpoint path from config')
    parser.add_argument('--num-samples', type=int,
                       help='Override number of samples to evaluate from config')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       help='Override device from config')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output including prevalence computation')
    parser.add_argument('--output-dir', type=str,
                       help='Override output directory from config')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return
    
    config = load_config(config_path)
    
    # Apply command line overrides
    if args.mode:
        config['evaluation']['mode'] = args.mode
    if args.checkpoint:
        config['model']['checkpoint_path'] = args.checkpoint
    if args.num_samples:
        config['evaluation']['num_samples'] = args.num_samples
    if args.device:
        config['model']['device'] = args.device
    if args.verbose:
        config['verbose'] = True
    if args.output_dir:
        config['output']['save_dir'] = args.output_dir
    
    # Set device
    device = torch.device(config['model']['device'])
    
    # Create output directory
    output_dir = config['output']['save_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print("ABR Transformer Enhanced Evaluation")
    print("=" * 50)
    print(f"Configuration: {config_path}")
    print(f"Device: {device}")
    print(f"Evaluation mode: {config['evaluation']['mode']}")
    if args.verbose:
        print(f"Verbose mode: enabled")
    
    # Create dataset and dataloader
    dataset = create_evaluation_dataset(config)
    dataloader = DataLoader(dataset, batch_size=config['evaluation']['batch_size'], shuffle=False)
    
    # Load model
    model = load_model(config, device)
    
    # Run evaluation
    if config['evaluation']['mode'] == 'reconstruction':
        results = evaluate_reconstruction(model, dataloader, device, config)
    elif config['evaluation']['mode'] == 'generation':
        results = evaluate_generation(model, dataloader, device, config)
    else:
        print(f"Unknown evaluation mode: {config['evaluation']['mode']}")
        return
    
    # Create visualizations
    if config['output'].get('create_plots', True):
        print("Creating visualizations...")
        create_visualizations(results, output_dir, config)
    
    # Save results
    save_results(results, output_dir, config)
    
    # Print summary
    print("\nEvaluation Results Summary")
    print("-" * 30)
    print(f"Mode: {results['mode']}")
    
    if 'signal_metrics' in results:
        print("\nSignal Quality Metrics:")
        for metric_name, value in results['signal_metrics'].items():
            if not metric_name.endswith('_std'):
                if value is not None:
                    print(f"  {metric_name.upper()}: {value:.4f}")
                else:
                    print(f"  {metric_name.upper()}: Not computed")
    
    if 'classification_metrics' in results:
        print("\nPeak Classification Metrics:")
        for metric_name, value in results['classification_metrics'].items():
            if isinstance(value, dict) and 'value' in value:
                print(f"  {metric_name}: {value['value']:.4f}")
            else:
                print(f"  {metric_name}: {value:.4f}")
        
        if 'statistical_tests' in results:
            print("\nStatistical Significance Tests:")
            for test_name, test_data in results['statistical_tests'].items():
                significance = ""
                if test_data['p_value'] < 0.001:
                    significance = "***"
                elif test_data['p_value'] < 0.01:
                    significance = "**"
                elif test_data['p_value'] < 0.05:
                    significance = "*"
                print(f"  {test_name}: p = {test_data['p_value']:.4f} {significance}")
    
    if 'samples_evaluated' in results:
        print(f"\nSamples evaluated: {results['samples_evaluated']}")
    
    print(f"\nResults saved to: {output_dir}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
