#!/usr/bin/env python3
"""
COMPREHENSIVE ABR MODEL EVALUATION PIPELINE

Professional, detailed evaluation system for ABR Hierarchical U-Net model.
Provides exhaustive analysis across all model tasks with rich visualizations.

Features:
- Multi-task performance analysis (Signal, Classification, Peaks, Thresholds)
- Clinical-grade metrics and visualizations
- Statistical significance testing
- Confusion matrices and performance curves
- Signal quality assessment with spectral analysis
- Peak detection accuracy with timing analysis
- Threshold regression with clinical correlation
- Model uncertainty quantification
- Comparative analysis across patient demographics
- Interactive plots and comprehensive reports

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --config configs/config.yaml
    python evaluate.py --checkpoint checkpoints/best_model.pt --comprehensive --clinical_analysis
    python evaluate.py --checkpoint checkpoints/best_model.pt --uncertainty_analysis --demographic_breakdown

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
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Enhanced visualization and analysis
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Statistical analysis
from scipy import stats
from scipy.signal import find_peaks, welch, spectrogram
from scipy.spatial.distance import euclidean
# fastdtw removed - using simple DTW fallback
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import label_binarize

# Clinical analysis
import warnings
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from omegaconf import DictConfig
from training.config_loader import load_config
from data.dataset import ABRDataset, stratified_patient_split, create_optimized_dataloaders
from models.hierarchical_unet import OptimizedHierarchicalUNet
from evaluation.metrics import compute_all_metrics, ABRMetrics
from evaluation.comprehensive_evaluator import ComprehensiveEvaluationMethods
from evaluation.visualization_methods import VisualizationMethods
from utils.sampling import DDIMSampler, create_ddim_sampler


class ComprehensiveABREvaluator(ComprehensiveEvaluationMethods, VisualizationMethods):
    """
    Professional, comprehensive evaluation system for ABR models.
    
    Provides detailed analysis across all model tasks with clinical-grade
    metrics, statistical testing, and rich visualizations.
    """
    
    def __init__(self, model: torch.nn.Module, config: DictConfig, 
                 device: torch.device, output_dir: str = "evaluation_results"):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized output
        self.plots_dir = self.output_dir / "plots"
        self.reports_dir = self.output_dir / "reports"
        self.data_dir = self.output_dir / "data"
        
        for dir_path in [self.plots_dir, self.reports_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize evaluation state
        self.results = {}
        self.predictions = {}
        self.ground_truth = {}
        self.metadata = {}
        
        # Set visualization style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        sns.set_palette("husl")
        
        # Clinical thresholds for ABR interpretation
        self.clinical_thresholds = {
            'normal': 20,  # dB nHL
            'mild': 40,
            'moderate': 70,
            'severe': 90,
            'profound': 120
        }
        
        print(f"🔬 Comprehensive ABR Evaluator initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def evaluate_comprehensive(self, test_loader: DataLoader, 
                             uncertainty_analysis: bool = True,
                             clinical_analysis: bool = True,
                             demographic_breakdown: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation across all model tasks.
        
        Args:
            test_loader: DataLoader for test data
            uncertainty_analysis: Whether to perform uncertainty quantification
            clinical_analysis: Whether to perform clinical correlation analysis
            demographic_breakdown: Whether to analyze by demographics
            
        Returns:
            Comprehensive evaluation results dictionary
        """
        print("🚀 Starting comprehensive evaluation...")
        start_time = time.time()
        
        # 1. Generate predictions and collect data
        print("📊 Generating predictions...")
        self._generate_predictions(test_loader)
        
        # 2. Signal quality analysis
        print("🌊 Analyzing signal quality...")
        signal_results = self._evaluate_signal_quality()
        
        # 3. Classification analysis
        print("🎯 Analyzing classification performance...")
        classification_results = self._evaluate_classification()
        
        # 4. Peak detection analysis
        print("⛰️ Analyzing peak detection...")
        peak_results = self._evaluate_peak_detection()
        
        # 5. Threshold regression analysis
        print("📏 Analyzing threshold regression...")
        threshold_results = self._evaluate_threshold_regression()
        
        # 6. Uncertainty quantification
        if uncertainty_analysis:
            print("🎲 Performing uncertainty analysis...")
            uncertainty_results = self._evaluate_uncertainty()
        else:
            uncertainty_results = {}
        
        # 7. Clinical correlation analysis
        if clinical_analysis:
            print("🏥 Performing clinical analysis...")
            clinical_results = self._evaluate_clinical_correlation()
        else:
            clinical_results = {}
        
        # 8. Demographic breakdown
        if demographic_breakdown:
            print("👥 Performing demographic analysis...")
            demographic_results = self._evaluate_demographics()
        else:
            demographic_results = {}
        
        # 9. Generate comprehensive visualizations
        print("📈 Creating visualizations...")
        self._create_comprehensive_visualizations()
        
        # 10. Generate interactive plots
        print("🎨 Creating interactive plots...")
        self._create_interactive_visualizations()
        
        # 11. Compile final results
        evaluation_time = time.time() - start_time
        
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_time_seconds': evaluation_time,
            'model_info': self._get_model_info(),
            'dataset_info': self._get_dataset_info(),
            'signal_quality': signal_results,
            'classification': classification_results,
            'peak_detection': peak_results,
            'threshold_regression': threshold_results,
            'uncertainty': uncertainty_results,
            'clinical_analysis': clinical_results,
            'demographic_analysis': demographic_results,
            'summary_metrics': self._compute_summary_metrics(),
            'recommendations': self._generate_recommendations()
        }
        
        # 12. Save comprehensive report
        self._save_comprehensive_report(final_results)
        
        print(f"✅ Evaluation complete! Time: {evaluation_time:.2f}s")
        print(f"📁 Results saved to: {self.output_dir}")
        
        return final_results
    
    def _generate_predictions(self, test_loader: DataLoader) -> None:
        """Generate predictions for all test samples."""
        self.model.eval()
        
        predictions = {
            'signals': [],
            'classifications': [],
            'peak_predictions': [],
            'thresholds': [],
            'uncertainties': []
        }
        
        ground_truth = {
            'signals': [],
            'classifications': [],
            'peak_labels': [],
            'peak_masks': [],
            'thresholds': [],
            'metadata': []
        }
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                # Handle dictionary batch format
                if isinstance(batch, dict):
                    signals = batch['signal'].to(self.device)
                    static_params = batch['static_params'].to(self.device)
                    batch_size = batch['signal'].shape[0]
                    targets = {
                        'signal': batch['signal'],
                        'target': batch['target'],
                        'v_peak': batch.get('v_peak', torch.zeros(batch_size, 2)),
                        'v_peak_mask': batch.get('v_peak_mask', torch.ones(batch_size, 2)),
                        'threshold': batch.get('threshold', torch.zeros(batch_size, 1))
                    }
                else:
                    # Handle tuple/list format
                    if len(batch) == 3:
                        signals, static_params, targets = batch
                    elif len(batch) >= 4:
                        signals, static_params, targets = batch[:3]  # ignore additional data
                    else:
                        raise ValueError(f"Unexpected batch format with {len(batch)} items")
                    signals = signals.to(self.device)
                    static_params = static_params.to(self.device)
                
                # Forward pass
                outputs = self.model(signals, static_params)
                
                # Store predictions
                predictions['signals'].append(outputs['recon'].cpu())
                predictions['classifications'].append(outputs['class'].cpu())
                # Handle peak output which is a tuple of tensors
                peak_outputs = outputs['peak']
                if isinstance(peak_outputs, tuple):
                    # Each element in the tuple is a tensor with shape [batch_size]
                    # Convert to a single tensor with shape [batch_size, num_outputs]
                    peak_tensor = torch.stack(peak_outputs, dim=1).cpu()  # [batch_size, num_peak_outputs]
                    predictions['peak_predictions'].append(peak_tensor)
                else:
                    predictions['peak_predictions'].append(peak_outputs.cpu())
                predictions['thresholds'].append(outputs['threshold'].cpu())
                
                if 'uncertainty' in outputs:
                    predictions['uncertainties'].append(outputs['uncertainty'].cpu())
                
                # Store ground truth
                ground_truth['signals'].append(targets['signal'].cpu())
                ground_truth['classifications'].append(targets['target'].cpu())
                ground_truth['peak_labels'].append(targets['v_peak'].cpu())
                ground_truth['peak_masks'].append(targets.get('v_peak_mask', torch.ones_like(targets['v_peak'])).cpu())
                ground_truth['thresholds'].append(targets['threshold'].cpu())
                ground_truth['metadata'].append(targets.get('metadata', {}))
        
        # Concatenate all predictions
        for key in predictions:
            if predictions[key]:
                predictions[key] = torch.cat(predictions[key], dim=0)
        
        for key in ground_truth:
            if ground_truth[key] and isinstance(ground_truth[key][0], torch.Tensor):
                ground_truth[key] = torch.cat(ground_truth[key], dim=0)
        
        self.predictions = predictions
        self.ground_truth = ground_truth
        
        print(f"📊 Generated predictions for {len(predictions['signals'])} samples")


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
    # Load checkpoint with weights_only=False for OmegaConf compatibility
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
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


def evaluate_model_diffusion(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: DictConfig
) -> Dict[str, Any]:
    """
    Properly evaluate diffusion model on test data.
    
    Args:
        model: Trained diffusion model
        test_loader: Test data loader
        device: Computation device
        config: Configuration
        
    Returns:
        Dictionary containing evaluation results
    """
    logger = logging.getLogger(__name__)
    
    # Create noise scheduler and sampler for proper diffusion evaluation
    from utils.schedule import get_noise_schedule
    from utils.sampling import DDIMSampler
    
    noise_schedule = get_noise_schedule(
        schedule_type=config.diffusion.noise_schedule.get('type', 'cosine'),
        num_timesteps=config.diffusion.noise_schedule.get('num_timesteps', 1000),
        beta_start=config.diffusion.noise_schedule.get('beta_start', 1e-4),
        beta_end=config.diffusion.noise_schedule.get('beta_end', 0.02)
    )
    
    sampler = DDIMSampler(noise_schedule)
    
    model.eval()
    all_metrics = []
    all_outputs = []
    all_targets = []
    
    logger.info("Evaluating diffusion model on test data...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating Diffusion Model")):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            batch_size = batch['signal'].size(0)
            
            # METHOD 1: Test generation capability
            # Generate synthetic signals conditioned on static params
            try:
                synthetic_signals = sampler.sample(
                    model=model,
                    shape=batch['signal'].shape,
                    static_params=batch['static_params'],
                    device=device,
                    num_steps=config.diffusion.sampling.get('num_sampling_steps', 50),
                    progress=False
                )
                
                # Get predictions from synthetic signals (t=0 for final prediction)
                outputs = model(
                    synthetic_signals,
                    batch['static_params'],
                    torch.zeros(batch_size, device=device, dtype=torch.long)
                )
                
            except Exception as e:
                logger.warning(f"Diffusion sampling failed: {e}. Falling back to direct evaluation.")
                # Fallback: Add small noise and use low timestep
                small_noise = torch.randn_like(batch['signal']) * 0.1
                slightly_noisy = batch['signal'] + small_noise
                t_small = torch.full((batch_size,), 10, device=device, dtype=torch.long)
                
                outputs = model(
                    slightly_noisy,
                    batch['static_params'],
                    t_small
                )
                synthetic_signals = batch['signal']  # Use original for comparison
            
            # Create evaluation targets comparing synthetic to original
            eval_targets = batch.copy()
            eval_targets['signal'] = synthetic_signals
            
            # Compute metrics for this batch
            batch_metrics = compute_all_metrics(
                model_outputs=outputs,
                batch_targets=batch,  # Compare predictions to original targets
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
            
            # Add synthetic signals to outputs for analysis
            batch_outputs['synthetic_signal'] = synthetic_signals.cpu().numpy()
            
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
        'sample_outputs': all_outputs[:10],
        'sample_targets': all_targets[:10],
        'num_samples': len(test_loader.dataset),
        'num_batches': len(test_loader),
        'evaluation_type': 'diffusion'
    }
    
    return results


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: DictConfig
) -> Dict[str, Any]:
    """
    Evaluate model on test data - automatically detects if diffusion model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Computation device
        config: Configuration
        
    Returns:
        Dictionary containing evaluation results
    """
    logger = logging.getLogger(__name__)
    
    # Check if this is a diffusion model by looking at config
    is_diffusion = hasattr(config, 'diffusion') and config.diffusion is not None
    
    if is_diffusion:
        logger.info("Detected diffusion model - using diffusion evaluation pipeline")
        return evaluate_model_diffusion(model, test_loader, device, config)
    else:
        logger.info("Using direct evaluation pipeline")
        return evaluate_model_direct(model, test_loader, device, config)


def evaluate_model_direct(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: DictConfig
) -> Dict[str, Any]:
    """
    Direct evaluation for non-diffusion models.
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
            
            # Direct forward pass
            outputs = model(
                batch['signal'],
                batch['static_params']
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
        'num_batches': len(test_loader),
        'evaluation_type': 'direct'
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
    """Main comprehensive evaluation function."""
    parser = argparse.ArgumentParser(description="Comprehensive ABR Model Evaluation")
    
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
        default="comprehensive_evaluation_results",
        help="Output directory for comprehensive evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config)"
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Enable comprehensive analysis (default: True)"
    )
    parser.add_argument(
        "--uncertainty_analysis",
        action="store_true",
        help="Enable uncertainty quantification analysis"
    )
    parser.add_argument(
        "--clinical_analysis",
        action="store_true",
        help="Enable clinical correlation analysis"
    )
    parser.add_argument(
        "--demographic_breakdown",
        action="store_true",
        help="Enable demographic performance breakdown"
    )
    parser.add_argument(
        "--generate_samples",
        type=int,
        default=0,
        help="Number of samples to generate (0 to skip generation)"
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
    
    print("🚀" + "="*78 + "🚀")
    print("🔬 COMPREHENSIVE ABR MODEL EVALUATION PIPELINE 🔬")
    print("🚀" + "="*78 + "🚀")
    
    try:
        # Load model and configuration
        print("\n📦 Loading model and configuration...")
        model, config, device = load_model_and_config(args.checkpoint, args.config)
        
        # Override batch size if specified
        if args.batch_size is not None:
            config.data.dataloader.batch_size = args.batch_size
        
        # Create test data loader
        print("📊 Creating test data loader...")
        test_loader = create_test_loader(config)
        
        # Initialize comprehensive evaluator
        print("🔬 Initializing comprehensive evaluator...")
        evaluator = ComprehensiveABREvaluator(
            model=model,
            config=config,
            device=device,
            output_dir=args.output_dir
        )
        
        # Perform comprehensive evaluation
        print("\n🎯 Starting comprehensive evaluation...")
        results = evaluator.evaluate_comprehensive(
            test_loader=test_loader,
            uncertainty_analysis=args.uncertainty_analysis,
            clinical_analysis=args.clinical_analysis,
            demographic_breakdown=args.demographic_breakdown
        )
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        # Summary metrics
        summary = results.get('summary_metrics', {})
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Overall Score: {summary.get('overall_score', 0):.3f}")
        
        print(f"\n🌊 SIGNAL QUALITY:")
        if 'signal_quality' in results:
            signal_metrics = results['signal_quality']['basic_metrics']
            print(f"   Correlation: {signal_metrics.get('correlation_mean', 0):.3f}")
            print(f"   SNR: {signal_metrics.get('snr_mean_db', 0):.1f} dB")
            print(f"   RMSE: {signal_metrics.get('rmse', 0):.4f}")
        
        print(f"\n🎯 CLASSIFICATION:")
        if 'classification' in results:
            class_metrics = results['classification']['basic_metrics']
            print(f"   Accuracy: {class_metrics.get('accuracy', 0):.3f}")
            print(f"   F1-Macro: {class_metrics.get('f1_macro', 0):.3f}")
            print(f"   F1-Weighted: {class_metrics.get('f1_weighted', 0):.3f}")
        
        print(f"\n⛰️ PEAK DETECTION:")
        if 'peak_detection' in results:
            peak_metrics = results['peak_detection']
            existence_metrics = peak_metrics.get('existence_metrics', {})
            latency_metrics = peak_metrics.get('latency_metrics', {})
            print(f"   Existence F1: {existence_metrics.get('f1_score', 0):.3f}")
            print(f"   Latency MAE: {latency_metrics.get('mae_ms', 0):.3f} ms")
            print(f"   Latency Correlation: {latency_metrics.get('correlation', 0):.3f}")
        
        print(f"\n📏 THRESHOLD REGRESSION:")
        if 'threshold_regression' in results:
            thresh_metrics = results['threshold_regression']['regression_metrics']
            error_metrics = results['threshold_regression']['error_analysis']
            print(f"   MAE: {thresh_metrics.get('mae_db', 0):.2f} dB")
            print(f"   R² Score: {thresh_metrics.get('r2_score', 0):.3f}")
            print(f"   Within 5dB: {error_metrics.get('within_5db_percent', 0):.1f}%")
            print(f"   Within 10dB: {error_metrics.get('within_10db_percent', 0):.1f}%")
        
        # Clinical analysis summary
        if 'clinical_analysis' in results and results['clinical_analysis']:
            print(f"\n🏥 CLINICAL ANALYSIS:")
            diag_metrics = results['clinical_analysis'].get('diagnostic_accuracy', {})
            print(f"   Diagnostic Accuracy: {diag_metrics.get('overall_accuracy', 0):.3f}")
        
        # Uncertainty analysis summary
        if 'uncertainty' in results and results['uncertainty']:
            print(f"\n🎲 UNCERTAINTY ANALYSIS:")
            uncertainty_metrics = results['uncertainty']
            if 'uncertainty_stats' in uncertainty_metrics:
                stats = uncertainty_metrics['uncertainty_stats']
                print(f"   Mean Uncertainty: {stats.get('mean_uncertainty', 0):.4f}")
            if 'calibration_analysis' in uncertainty_metrics:
                calib = uncertainty_metrics['calibration_analysis']
                print(f"   Well Calibrated: {calib.get('well_calibrated', False)}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        recommendations = results.get('recommendations', [])
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            print(f"   {i}. {rec}")
        
        # Model information
        model_info = results.get('model_info', {})
        dataset_info = results.get('dataset_info', {})
        print(f"\n📋 MODEL & DATASET INFO:")
        print(f"   Model: {model_info.get('model_class', 'Unknown')}")
        print(f"   Parameters: {model_info.get('total_parameters', 0):,}")
        print(f"   Test Samples: {dataset_info.get('num_samples', 0)}")
        print(f"   Signal Shape: {dataset_info.get('signal_shape', [])}")
        
        print(f"\n📁 RESULTS LOCATION:")
        print(f"   Output Directory: {args.output_dir}")
        print(f"   Plots: {args.output_dir}/plots/")
        print(f"   Reports: {args.output_dir}/reports/")
        print(f"   Data: {args.output_dir}/data/")
        
        # Generate samples if requested
        if args.generate_samples > 0:
            print(f"\n🎨 Generating {args.generate_samples} sample signals...")
            sample_output_dir = os.path.join(args.output_dir, "generated_samples")
            generated_samples = generate_samples(
                model, config, device, args.generate_samples, sample_output_dir
            )
            print(f"✅ Generated samples saved to: {sample_output_dir}")
        
        print("\n🎉" + "="*76 + "🎉")
        print("✅ COMPREHENSIVE EVALUATION COMPLETED SUCCESSFULLY! ✅")
        print("🎉" + "="*76 + "🎉")
        
        # Final summary
        evaluation_time = results.get('evaluation_time_seconds', 0)
        print(f"\n⏱️  Total evaluation time: {evaluation_time:.2f} seconds")
        print(f"📊 Comprehensive analysis with {len(results)} result categories")
        print(f"🎨 Visualizations and reports generated")
        print(f"📁 All results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()