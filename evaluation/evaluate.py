#!/usr/bin/env python3
"""
Comprehensive CVAE Model Evaluation Script

This script provides a complete evaluation pipeline for trained CVAE models including:
- Quantitative metrics (reconstruction, peak prediction)
- Visual diagnostics (signal comparison, latent space, generation)
- Modular design with robust error handling

Usage:
    python evaluation/evaluate.py --config evaluation/configs/eval_config.json
    python evaluation/evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.cvae import CVAE
from training.dataset import ABRDataset
from evaluation.utils.metrics import (
    compute_reconstruction_metrics,
    compute_peak_metrics,
    compute_dtw_distance,
    compute_latent_metrics,
    aggregate_metrics
)
from evaluation.utils.plotting import (
    plot_reconstruction_comparison,
    plot_latent_space_2d,
    plot_generation_samples,
    plot_peak_analysis,
    plot_metrics_summary
)


class CVAEEvaluator:
    """
    Comprehensive CVAE model evaluator.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        # Create output directories
        self.output_dir = self._create_output_directories()
        
        # Initialize model and data
        self.model = None
        self.dataset = None
        self.dataloader = None
        
        # Results storage
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': config
            },
            'metrics': {},
            'visualizations': {}
        }
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        device_config = self.config['model'].get('device', 'auto')
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging_config = self.config.get('logging', {})
        log_level = getattr(logging, logging_config.get('level', 'INFO'))
        
        # Create logger
        logger = logging.getLogger('cvae_evaluator')
        logger.setLevel(log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_output_directories(self) -> str:
        """Create output directories."""
        base_dir = self.config['output']['base_dir']
        
        if self.config['output'].get('timestamp', True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(base_dir, f'eval_{timestamp}')
        else:
            output_dir = base_dir
        
        # Create directories
        subdirs = ['plots', 'latent_space', 'generated_samples', 'logs']
        for subdir in subdirs:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir
    
    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Load trained CVAE model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if checkpoint_path is None:
            checkpoint_path = self.config['model']['checkpoint_path']
        
        self.logger.info(f"Loading model from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get model configuration
        model_config = checkpoint.get('config', {})
        
        # Create temporary dataset to get dimensions
        temp_dataset = ABRDataset(
            self.config['data']['data_path'],
            return_peaks=model_config.get('predict_peaks', True)
        )
        sample_info = temp_dataset.get_sample_info()
        
        # Create model
        self.model = CVAE(
            signal_length=sample_info['signal_length'],
            static_dim=sample_info['static_params_dim'],
            latent_dim=model_config.get('latent_dim', 32),
            predict_peaks=model_config.get('predict_peaks', True),
            num_peaks=sample_info.get('num_peaks', 6)
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Store model info
        self.results['metadata']['model_info'] = {
            'signal_length': sample_info['signal_length'],
            'static_params_dim': sample_info['static_params_dim'],
            'latent_dim': model_config.get('latent_dim', 32),
            'predict_peaks': model_config.get('predict_peaks', True),
            'num_peaks': sample_info.get('num_peaks', 6),
            'checkpoint_path': checkpoint_path,
            'training_epoch': checkpoint.get('epoch', 'unknown')
        }
        
        self.logger.info(f"Model loaded successfully!")
        self.logger.info(f"  Signal length: {sample_info['signal_length']}")
        self.logger.info(f"  Static params dim: {sample_info['static_params_dim']}")
        self.logger.info(f"  Latent dim: {model_config.get('latent_dim', 32)}")
        self.logger.info(f"  Predict peaks: {model_config.get('predict_peaks', True)}")
    
    def load_data(self) -> None:
        """Load evaluation dataset."""
        self.logger.info("Loading evaluation dataset...")
        
        # Create dataset
        self.dataset = ABRDataset(
            self.config['data']['data_path'],
            return_peaks=self.model.predict_peaks
        )
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.logger.info(f"Dataset loaded: {len(self.dataset)} samples")
    
    def evaluate_reconstruction(self) -> Dict[str, Any]:
        """
        Evaluate reconstruction quality.
        
        Returns:
            Dictionary of reconstruction metrics
        """
        self.logger.info("Evaluating reconstruction quality...")
        
        all_metrics = []
        all_original = []
        all_reconstructed = []
        all_predicted_peaks = []
        all_target_peaks = []
        all_peak_masks = []
        all_latent = []
        all_static_params = []
        
        max_samples = self.config['data'].get('max_samples', float('inf'))
        processed_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
                if processed_samples >= max_samples:
                    break
                
                # Move to device
                signals = batch['signal'].to(self.device)
                static_params = batch['static_params'].to(self.device)
                
                # Handle peaks if available
                if self.model.predict_peaks and 'peaks' in batch:
                    peaks = batch['peaks'].to(self.device)
                    peak_mask = batch['peak_mask'].to(self.device)
                else:
                    peaks = None
                    peak_mask = None
                
                # Forward pass
                try:
                    # Encode
                    mu, logvar = self.model.encoder(signals, static_params)
                    
                    # Decode (use mean for reconstruction)
                    reconstructed, predicted_peaks = self.model.decoder(mu, static_params)
                    
                    # Compute metrics for this batch
                    batch_metrics = compute_reconstruction_metrics(reconstructed, signals)
                    all_metrics.append(batch_metrics)
                    
                    # Store data for visualization
                    all_original.append(signals.cpu())
                    all_reconstructed.append(reconstructed.cpu())
                    all_latent.append(mu.cpu())
                    all_static_params.append(static_params.cpu())
                    
                    if self.model.predict_peaks and peaks is not None:
                        all_predicted_peaks.append(predicted_peaks.cpu())
                        all_target_peaks.append(peaks.cpu())
                        all_peak_masks.append(peak_mask.cpu())
                    
                    processed_samples += signals.shape[0]
                    
                except Exception as e:
                    self.logger.warning(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        # Aggregate metrics
        aggregated_metrics = aggregate_metrics(all_metrics)
        self.results['metrics']['reconstruction'] = aggregated_metrics
        
        # Store data for visualization
        self.results['data'] = {
            'original': torch.cat(all_original, dim=0),
            'reconstructed': torch.cat(all_reconstructed, dim=0),
            'latent': torch.cat(all_latent, dim=0),
            'static_params': torch.cat(all_static_params, dim=0)
        }
        
        if all_predicted_peaks:
            self.results['data']['predicted_peaks'] = torch.cat(all_predicted_peaks, dim=0)
            self.results['data']['target_peaks'] = torch.cat(all_target_peaks, dim=0)
            self.results['data']['peak_masks'] = torch.cat(all_peak_masks, dim=0)
        
        self.logger.info(f"Reconstruction evaluation completed!")
        self.logger.info(f"  Processed {processed_samples} samples")
        self.logger.info(f"  MSE: {aggregated_metrics.get('mse_mean', 'N/A'):.6f}")
        self.logger.info(f"  MAE: {aggregated_metrics.get('mae_mean', 'N/A'):.6f}")
        self.logger.info(f"  Correlation: {aggregated_metrics.get('correlation_mean', 'N/A'):.6f}")
        
        return aggregated_metrics
    
    def evaluate_peaks(self) -> Dict[str, Any]:
        """
        Evaluate peak prediction quality.
        
        Returns:
            Dictionary of peak metrics
        """
        if not self.model.predict_peaks:
            self.logger.info("Peak prediction not enabled, skipping peak evaluation.")
            return {}
        
        if 'predicted_peaks' not in self.results['data']:
            self.logger.warning("No peak data available for evaluation.")
            return {}
        
        self.logger.info("Evaluating peak prediction quality...")
        
        predicted_peaks = self.results['data']['predicted_peaks']
        target_peaks = self.results['data']['target_peaks']
        peak_masks = self.results['data']['peak_masks']
        
        # Compute peak metrics
        peak_metrics = compute_peak_metrics(predicted_peaks, target_peaks, peak_masks)
        self.results['metrics']['peaks'] = peak_metrics
        
        self.logger.info(f"Peak evaluation completed!")
        self.logger.info(f"  Peak MAE: {peak_metrics.get('peak_mae', 'N/A'):.6f}")
        self.logger.info(f"  Peak Accuracy: {peak_metrics.get('peak_accuracy', 'N/A'):.6f}")
        
        return peak_metrics
    
    def evaluate_latent_space(self) -> Dict[str, Any]:
        """
        Evaluate latent space quality.
        
        Returns:
            Dictionary of latent space metrics
        """
        if 'latent' not in self.results['data']:
            self.logger.warning("No latent data available for evaluation.")
            return {}
        
        self.logger.info("Evaluating latent space quality...")
        
        latent_vectors = self.results['data']['latent']
        static_params = self.results['data']['static_params']
        
        # Compute latent metrics
        latent_metrics = compute_latent_metrics(latent_vectors, static_params)
        self.results['metrics']['latent'] = latent_metrics
        
        self.logger.info(f"Latent space evaluation completed!")
        self.logger.info(f"  Latent dim: {latent_metrics.get('latent_dim', 'N/A')}")
        self.logger.info(f"  Effective dim: {latent_metrics.get('effective_dim', 'N/A')}")
        
        return latent_metrics
    
    def generate_samples(self) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Returns:
            Generated samples
        """
        self.logger.info("Generating samples...")
        
        generation_config = self.config['visualization']['generation']
        num_conditions = generation_config['num_conditions']
        samples_per_condition = generation_config['samples_per_condition']
        
        # Get some static parameters from the dataset
        static_params = self.results['data']['static_params'][:num_conditions]
        
        generated_samples = []
        generation_static_params = []
        
        with torch.no_grad():
            for i in range(num_conditions):
                condition_static = static_params[i:i+1].to(self.device)
                
                for _ in range(samples_per_condition):
                    # Sample from prior
                    z = torch.randn(1, self.model.latent_dim).to(self.device)
                    
                    # Decode
                    generated, _ = self.model.decoder(z, condition_static)
                    
                    generated_samples.append(generated.cpu())
                    generation_static_params.append(condition_static.cpu())
        
        generated_samples = torch.cat(generated_samples, dim=0)
        generation_static_params = torch.cat(generation_static_params, dim=0)
        
        self.results['data']['generated_samples'] = generated_samples
        self.results['data']['generation_static_params'] = generation_static_params
        
        self.logger.info(f"Generated {len(generated_samples)} samples")
        
        return generated_samples
    
    def create_visualizations(self) -> None:
        """Create all visualizations."""
        if not self.config['output']['save_plots']:
            self.logger.info("Plot saving disabled, skipping visualizations.")
            return
        
        self.logger.info("Creating visualizations...")
        
        viz_config = self.config['visualization']
        
        # 1. Reconstruction comparison
        if viz_config['reconstruction']['enabled']:
            self.logger.info("  Creating reconstruction comparison plots...")
            
            original = self.results['data']['original']
            reconstructed = self.results['data']['reconstructed']
            num_samples = viz_config['reconstruction']['num_samples']
            
            # Prepare peak data if available
            predicted_peaks = self.results['data'].get('predicted_peaks')
            target_peaks = self.results['data'].get('target_peaks')
            peak_masks = self.results['data'].get('peak_masks')
            
            save_path = os.path.join(self.output_dir, 'plots', 'reconstruction_comparison.png')
            plot_reconstruction_comparison(
                original, reconstructed, save_path, num_samples,
                predicted_peaks, target_peaks, peak_masks
            )
            
            self.results['visualizations']['reconstruction_comparison'] = save_path
        
        # 2. Latent space visualization
        if viz_config['latent_space']['enabled']:
            self.logger.info("  Creating latent space visualization...")
            
            latent_vectors = self.results['data']['latent']
            static_params = self.results['data']['static_params']
            
            # Limit samples for visualization
            max_samples = viz_config['latent_space']['num_samples']
            if len(latent_vectors) > max_samples:
                indices = torch.randperm(len(latent_vectors))[:max_samples]
                latent_vectors = latent_vectors[indices]
                static_params = static_params[indices]
            
            save_path = os.path.join(self.output_dir, 'latent_space', 'latent_space_2d.png')
            plot_latent_space_2d(
                latent_vectors, static_params, save_path,
                method=viz_config['latent_space']['method'],
                color_by=viz_config['latent_space']['color_by']
            )
            
            self.results['visualizations']['latent_space'] = save_path
        
        # 3. Generation samples
        if viz_config['generation']['enabled']:
            self.logger.info("  Creating generation samples plot...")
            
            if 'generated_samples' not in self.results['data']:
                self.generate_samples()
            
            generated_samples = self.results['data']['generated_samples']
            generation_static_params = self.results['data']['generation_static_params']
            
            save_path = os.path.join(self.output_dir, 'generated_samples', 'generation_samples.png')
            plot_generation_samples(
                generated_samples, generation_static_params, save_path,
                viz_config['generation']['samples_per_condition']
            )
            
            self.results['visualizations']['generation_samples'] = save_path
        
        # 4. Peak analysis
        if (self.model.predict_peaks and 'predicted_peaks' in self.results['data']):
            self.logger.info("  Creating peak analysis plot...")
            
            predicted_peaks = self.results['data']['predicted_peaks']
            target_peaks = self.results['data']['target_peaks']
            peak_masks = self.results['data']['peak_masks']
            
            save_path = os.path.join(self.output_dir, 'plots', 'peak_analysis.png')
            plot_peak_analysis(predicted_peaks, target_peaks, peak_masks, save_path)
            
            self.results['visualizations']['peak_analysis'] = save_path
        
        # 5. Metrics summary
        self.logger.info("  Creating metrics summary plot...")
        
        save_path = os.path.join(self.output_dir, 'plots', 'metrics_summary.png')
        plot_metrics_summary(self.results['metrics'], save_path)
        
        self.results['visualizations']['metrics_summary'] = save_path
        
        self.logger.info("Visualizations completed!")
    
    def save_results(self) -> None:
        """Save evaluation results."""
        if not self.config['output']['save_summary']:
            return
        
        self.logger.info("Saving evaluation results...")
        
        # Save detailed results as JSON
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        
        # Convert tensors to lists for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if key == 'data':
                # Skip data tensors for JSON
                continue
            elif isinstance(value, dict):
                json_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        json_results[key][subkey] = subvalue.tolist()
                    else:
                        json_results[key][subkey] = subvalue
            else:
                json_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save summary text
        summary_path = os.path.join(self.output_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("CVAE Model Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Model info
            f.write("Model Information:\n")
            f.write("-" * 20 + "\n")
            model_info = self.results['metadata']['model_info']
            for key, value in model_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Reconstruction metrics
            f.write("Reconstruction Metrics:\n")
            f.write("-" * 20 + "\n")
            recon_metrics = self.results['metrics'].get('reconstruction', {})
            for key, value in recon_metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
            f.write("\n")
            
            # Peak metrics
            if 'peaks' in self.results['metrics']:
                f.write("Peak Prediction Metrics:\n")
                f.write("-" * 20 + "\n")
                peak_metrics = self.results['metrics']['peaks']
                for key, value in peak_metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.6f}\n")
                f.write("\n")
            
            # Latent metrics
            if 'latent' in self.results['metrics']:
                f.write("Latent Space Metrics:\n")
                f.write("-" * 20 + "\n")
                latent_metrics = self.results['metrics']['latent']
                for key, value in latent_metrics.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Visualization paths
            f.write("Generated Visualizations:\n")
            f.write("-" * 20 + "\n")
            for key, path in self.results['visualizations'].items():
                f.write(f"  {key}: {path}\n")
        
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Returns:
            Dictionary of evaluation results
        """
        self.logger.info("Starting CVAE model evaluation...")
        
        # Load model and data
        self.load_model()
        self.load_data()
        
        # Run evaluations
        self.evaluate_reconstruction()
        self.evaluate_peaks()
        self.evaluate_latent_space()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        self.logger.info("Evaluation completed successfully!")
        
        return self.results


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="CVAE Model Evaluation")
    parser.add_argument('--config', type=str, default='evaluation/configs/eval_config.json',
                       help='Path to evaluation configuration file')
    parser.add_argument('--checkpoint', type=str, 
                       help='Path to model checkpoint (overrides config)')
    parser.add_argument('--output', type=str,
                       help='Output directory (overrides config)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.checkpoint:
        config['model']['checkpoint_path'] = args.checkpoint
    
    if args.output:
        config['output']['base_dir'] = args.output
    
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    
    # Run evaluation
    try:
        evaluator = CVAEEvaluator(config)
        results = evaluator.run_evaluation()
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {evaluator.output_dir}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 