#!/usr/bin/env python3
"""
ABR Inference Script

Professional inference pipeline for ABR signal analysis and generation.
Supports both single sample and batch processing for clinical applications.

Usage:
    # Single sample inference
    python inference.py --checkpoint checkpoints/best_model.pt --age 35 --intensity 80 --rate 30 --fmp 0.8
    
    # Batch inference from file
    python inference.py --checkpoint checkpoints/best_model.pt --input_file data/test_samples.npz
    
    # Generate synthetic data
    python inference.py --checkpoint checkpoints/best_model.pt --generate 100 --output_dir results/generated

Author: AI Assistant
Date: January 2025
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
from typing import Optional, Dict, Any, List, Union

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from omegaconf import DictConfig
from training.config_loader import load_config
from models.hierarchical_unet import OptimizedHierarchicalUNet
from utils.sampling import DDIMSampler, create_ddim_sampler
from evaluation.metrics import compute_all_metrics


class ABRInference:
    """
    Professional ABR inference pipeline for clinical applications.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ABR inference pipeline.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Optional path to config file
            device: Computation device
        """
        self.logger = logging.getLogger(__name__)
        
        # Load model and configuration
        self.model, self.config, self.device = self._load_model_and_config(
            checkpoint_path, config_path, device
        )
        
        # Create sampler for generation
        self.sampler = self._create_sampler()
        
        # Model is ready for inference
        self.model.eval()
        
        self.logger.info("ABR Inference pipeline initialized successfully")
    
    def _load_model_and_config(
        self, 
        checkpoint_path: str, 
        config_path: Optional[str],
        device: Optional[torch.device]
    ) -> tuple:
        """Load model and configuration."""
        # Load checkpoint
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load configuration
        if config_path:
            config = load_config(config_path)
        else:
            config_dict = checkpoint.get('config', {})
            if not config_dict:
                raise ValueError("No configuration found in checkpoint")
            config = DictConfig(config_dict)
        
        # Setup device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger.info(f"Using device: {device}")
        
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
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model, config, device
    
    def _create_sampler(self) -> DDIMSampler:
        """Create DDIM sampler for generation."""
        sampling_config = self.config.diffusion.sampling
        
        return create_ddim_sampler(
            noise_schedule_type=self.config.diffusion.noise_schedule.get('type', 'cosine'),
            num_timesteps=self.config.diffusion.noise_schedule.get('num_timesteps', 1000),
            eta=sampling_config.get('ddim_eta', 0.0),
            clip_denoised=sampling_config.get('clip_denoised', True)
        )
    
    def predict_single(
        self,
        age: float,
        intensity: float,
        stimulus_rate: float,
        fmp: float,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Predict ABR characteristics for a single set of parameters.
        
        Args:
            age: Patient age in years
            intensity: Stimulus intensity in dB SPL
            stimulus_rate: Stimulus rate in Hz
            fmp: Functional middle pressure
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Dictionary with predictions and metadata
        """
        # Prepare static parameters
        static_params = torch.tensor(
            [[age, intensity, stimulus_rate, fmp]], 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Normalize parameters (simple normalization for demo)
        # In practice, you'd use the same normalization as training
        static_params[:, 0] = (static_params[:, 0] - 50) / 25  # Age: mean=50, std=25
        static_params[:, 1] = (static_params[:, 1] - 80) / 15  # Intensity: mean=80, std=15
        static_params[:, 2] = (static_params[:, 2] - 30) / 20  # Rate: mean=30, std=20
        static_params[:, 3] = (static_params[:, 3] - 0.75) / 0.25  # FMP: mean=0.75, std=0.25
        
        with torch.no_grad():
            # Generate ABR signal
            signal_shape = (1, 1, self.config.data.signal_length)
            generated_signal = self.sampler.sample(
                model=self.model,
                shape=signal_shape,
                static_params=static_params,
                device=self.device,
                num_steps=self.config.diffusion.sampling.get('num_sampling_steps', 50),
                progress=False
            )
            
            # Get model predictions
            outputs = self.model(
                generated_signal,
                static_params,
                torch.zeros(1, device=self.device, dtype=torch.long)
            )
            
            # Process outputs
            results = self._process_outputs(outputs, generated_signal, static_params)
            
            # Add input parameters
            results['input_parameters'] = {
                'age': age,
                'intensity': intensity,
                'stimulus_rate': stimulus_rate,
                'fmp': fmp
            }
            
            if return_confidence:
                results['confidence_intervals'] = self._compute_confidence_intervals(
                    static_params, num_samples=10
                )
        
        return results
    
    def predict_batch(
        self,
        static_params: np.ndarray,
        return_signals: bool = True
    ) -> Dict[str, Any]:
        """
        Predict ABR characteristics for a batch of parameters.
        
        Args:
            static_params: Static parameters [batch, 4] (age, intensity, rate, fmp)
            return_signals: Whether to return generated signals
            
        Returns:
            Dictionary with batch predictions
        """
        batch_size = static_params.shape[0]
        
        # Convert to tensor
        static_tensor = torch.tensor(static_params, dtype=torch.float32, device=self.device)
        
        # Normalize (same as single prediction)
        static_tensor[:, 0] = (static_tensor[:, 0] - 50) / 25
        static_tensor[:, 1] = (static_tensor[:, 1] - 80) / 15
        static_tensor[:, 2] = (static_tensor[:, 2] - 30) / 20
        static_tensor[:, 3] = (static_tensor[:, 3] - 0.75) / 0.25
        
        results = {
            'predictions': [],
            'signals': [] if return_signals else None,
            'batch_size': batch_size
        }
        
        # Process in chunks to avoid memory issues
        chunk_size = min(16, batch_size)
        
        with torch.no_grad():
            for i in tqdm(range(0, batch_size, chunk_size), desc="Processing batch"):
                end_idx = min(i + chunk_size, batch_size)
                chunk_params = static_tensor[i:end_idx]
                chunk_size_actual = end_idx - i
                
                # Generate signals for chunk
                signal_shape = (chunk_size_actual, 1, self.config.data.signal_length)
                generated_signals = self.sampler.sample(
                    model=self.model,
                    shape=signal_shape,
                    static_params=chunk_params,
                    device=self.device,
                    num_steps=self.config.diffusion.sampling.get('num_sampling_steps', 50),
                    progress=False
                )
                
                # Get predictions
                outputs = self.model(
                    generated_signals,
                    chunk_params,
                    torch.zeros(chunk_size_actual, device=self.device, dtype=torch.long)
                )
                
                # Process chunk outputs
                chunk_results = self._process_batch_outputs(
                    outputs, generated_signals, chunk_params
                )
                
                results['predictions'].extend(chunk_results['predictions'])
                
                if return_signals:
                    results['signals'].extend(chunk_results['signals'])
        
        return results
    
    def analyze_signal(
        self,
        signal: np.ndarray,
        static_params: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze an existing ABR signal.
        
        Args:
            signal: ABR signal [signal_length] or [batch, signal_length]
            static_params: Static parameters [4] or [batch, 4]
            
        Returns:
            Analysis results
        """
        # Ensure proper dimensions
        if signal.ndim == 1:
            signal = signal[np.newaxis, np.newaxis, :]  # [1, 1, signal_length]
            static_params = static_params[np.newaxis, :]  # [1, 4]
        elif signal.ndim == 2:
            signal = signal[:, np.newaxis, :]  # [batch, 1, signal_length]
        
        # Convert to tensors
        signal_tensor = torch.tensor(signal, dtype=torch.float32, device=self.device)
        static_tensor = torch.tensor(static_params, dtype=torch.float32, device=self.device)
        
        # Normalize static parameters
        static_tensor[:, 0] = (static_tensor[:, 0] - 50) / 25
        static_tensor[:, 1] = (static_tensor[:, 1] - 80) / 15
        static_tensor[:, 2] = (static_tensor[:, 2] - 30) / 20
        static_tensor[:, 3] = (static_tensor[:, 3] - 0.75) / 0.25
        
        with torch.no_grad():
            # Analyze signal (no diffusion, direct analysis)
            outputs = self.model(
                signal_tensor,
                static_tensor,
                torch.zeros(signal_tensor.size(0), device=self.device, dtype=torch.long)
            )
            
            # Process results
            if signal_tensor.size(0) == 1:
                results = self._process_outputs(outputs, signal_tensor, static_tensor)
            else:
                results = self._process_batch_outputs(outputs, signal_tensor, static_tensor)
        
        return results
    
    def _process_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        signal: torch.Tensor,
        static_params: torch.Tensor
    ) -> Dict[str, Any]:
        """Process model outputs for single sample."""
        results = {}
        
        # Signal
        results['signal'] = signal.squeeze().cpu().numpy()
        
        # Peak predictions
        if 'peak' in outputs:
            peak_outputs = outputs['peak']
            if isinstance(peak_outputs, (list, tuple)):
                results['peaks'] = {
                    'existence_probability': torch.sigmoid(peak_outputs[0]).item(),
                    'latency': peak_outputs[1].item(),
                    'amplitude': peak_outputs[2].item(),
                    'exists': torch.sigmoid(peak_outputs[0]).item() > 0.5
                }
            else:
                results['peaks'] = {
                    'existence_probability': torch.sigmoid(peak_outputs[0, 0]).item(),
                    'latency': peak_outputs[0, 1].item(),
                    'amplitude': peak_outputs[0, 2].item(),
                    'exists': torch.sigmoid(peak_outputs[0, 0]).item() > 0.5
                }
        
        # Classification
        if 'class' in outputs or 'classification_logits' in outputs:
            class_key = 'class' if 'class' in outputs else 'classification_logits'
            class_probs = F.softmax(outputs[class_key], dim=1).squeeze().cpu().numpy()
            predicted_class = int(np.argmax(class_probs))
            
            # Map to hearing loss categories
            class_names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Profound']
            
            results['classification'] = {
                'predicted_class': predicted_class,
                'predicted_category': class_names[min(predicted_class, len(class_names)-1)],
                'probabilities': {
                    class_names[i]: float(class_probs[i]) 
                    for i in range(min(len(class_probs), len(class_names)))
                },
                'confidence': float(np.max(class_probs))
            }
        
        # Threshold
        if 'threshold' in outputs:
            threshold_value = outputs['threshold'].item()
            results['threshold'] = {
                'value': threshold_value,
                'unit': 'dB nHL',
                'severity': self._categorize_threshold(threshold_value)
            }
        
        return results
    
    def _process_batch_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        signals: torch.Tensor,
        static_params: torch.Tensor
    ) -> Dict[str, Any]:
        """Process model outputs for batch."""
        batch_size = signals.size(0)
        
        results = {
            'predictions': [],
            'signals': []
        }
        
        for i in range(batch_size):
            # Extract single sample outputs
            sample_outputs = {}
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() > 1:
                        sample_outputs[key] = value[i:i+1]
                    else:
                        sample_outputs[key] = value[i:i+1]
                else:
                    sample_outputs[key] = value
            
            # Process single sample
            sample_signal = signals[i:i+1]
            sample_params = static_params[i:i+1]
            
            sample_result = self._process_outputs(sample_outputs, sample_signal, sample_params)
            
            results['predictions'].append(sample_result)
            results['signals'].append(sample_result['signal'])
        
        return results
    
    def _compute_confidence_intervals(
        self,
        static_params: torch.Tensor,
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """Compute confidence intervals through sampling."""
        samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                signal_shape = (1, 1, self.config.data.signal_length)
                generated_signal = self.sampler.sample(
                    model=self.model,
                    shape=signal_shape,
                    static_params=static_params,
                    device=self.device,
                    num_steps=self.config.diffusion.sampling.get('num_sampling_steps', 50),
                    progress=False
                )
                
                outputs = self.model(
                    generated_signal,
                    static_params,
                    torch.zeros(1, device=self.device, dtype=torch.long)
                )
                
                sample_result = self._process_outputs(outputs, generated_signal, static_params)
                samples.append(sample_result)
        
        # Compute statistics
        confidence_intervals = {}
        
        # Threshold confidence
        if all('threshold' in s for s in samples):
            threshold_values = [s['threshold']['value'] for s in samples]
            confidence_intervals['threshold'] = {
                'mean': float(np.mean(threshold_values)),
                'std': float(np.std(threshold_values)),
                'percentile_5': float(np.percentile(threshold_values, 5)),
                'percentile_95': float(np.percentile(threshold_values, 95))
            }
        
        # Peak confidence
        if all('peaks' in s for s in samples):
            latencies = [s['peaks']['latency'] for s in samples]
            amplitudes = [s['peaks']['amplitude'] for s in samples]
            
            confidence_intervals['peaks'] = {
                'latency': {
                    'mean': float(np.mean(latencies)),
                    'std': float(np.std(latencies)),
                    'percentile_5': float(np.percentile(latencies, 5)),
                    'percentile_95': float(np.percentile(latencies, 95))
                },
                'amplitude': {
                    'mean': float(np.mean(amplitudes)),
                    'std': float(np.std(amplitudes)),
                    'percentile_5': float(np.percentile(amplitudes, 5)),
                    'percentile_95': float(np.percentile(amplitudes, 95))
                }
            }
        
        return confidence_intervals
    
    def _categorize_threshold(self, threshold: float) -> str:
        """Categorize hearing threshold."""
        if threshold < 25:
            return "Normal hearing"
        elif threshold < 40:
            return "Mild hearing loss"
        elif threshold < 55:
            return "Moderate hearing loss"
        elif threshold < 70:
            return "Moderately severe hearing loss"
        elif threshold < 90:
            return "Severe hearing loss"
        else:
            return "Profound hearing loss"
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        format: str = 'json'
    ) -> None:
        """Save inference results."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format == 'npz':
            # Convert to numpy arrays for npz
            np_results = {}
            for key, value in results.items():
                if isinstance(value, (list, np.ndarray)):
                    np_results[key] = np.array(value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (list, np.ndarray)):
                            np_results[f"{key}_{sub_key}"] = np.array(sub_value)
            
            np.savez(output_path, **np_results)
        
        self.logger.info(f"Results saved to: {output_path}")


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="ABR Inference Pipeline")
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input_file",
        type=str,
        help="Input file with static parameters (npz format)"
    )
    input_group.add_argument(
        "--generate",
        type=int,
        help="Number of samples to generate with random parameters"
    )
    
    # Single sample parameters
    parser.add_argument("--age", type=float, help="Patient age in years")
    parser.add_argument("--intensity", type=float, help="Stimulus intensity in dB SPL")
    parser.add_argument("--rate", type=float, help="Stimulus rate in Hz")
    parser.add_argument("--fmp", type=float, help="Functional middle pressure")
    
    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference",
        help="Output directory"
    )
    parser.add_argument(
        "--confidence",
        action="store_true",
        help="Compute confidence intervals"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "npz"],
        default="json",
        help="Output format"
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
    logger.info("ABR Inference Pipeline")
    logger.info("=" * 80)
    
    try:
        # Initialize inference pipeline
        inference = ABRInference(args.checkpoint, args.config)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Single sample inference
        if all(param is not None for param in [args.age, args.intensity, args.rate, args.fmp]):
            logger.info("Running single sample inference...")
            
            results = inference.predict_single(
                age=args.age,
                intensity=args.intensity,
                stimulus_rate=args.rate,
                fmp=args.fmp,
                return_confidence=args.confidence
            )
            
            # Save results
            output_path = os.path.join(args.output_dir, f"single_prediction.{args.format}")
            inference.save_results(results, output_path, args.format)
            
            # Print summary
            logger.info("\nPrediction Results:")
            logger.info(f"Classification: {results['classification']['predicted_category']}")
            logger.info(f"Confidence: {results['classification']['confidence']:.3f}")
            logger.info(f"Threshold: {results['threshold']['value']:.1f} dB nHL ({results['threshold']['severity']})")
            
            if 'peaks' in results:
                logger.info(f"Peak exists: {results['peaks']['exists']}")
                if results['peaks']['exists']:
                    logger.info(f"Peak latency: {results['peaks']['latency']:.2f} ms")
                    logger.info(f"Peak amplitude: {results['peaks']['amplitude']:.3f} μV")
        
        # Batch inference
        elif args.input_file:
            logger.info(f"Running batch inference from: {args.input_file}")
            
            # Load input data
            data = np.load(args.input_file)
            static_params = data['static_params']
            
            logger.info(f"Processing {len(static_params)} samples...")
            
            results = inference.predict_batch(static_params, return_signals=True)
            
            # Save results
            output_path = os.path.join(args.output_dir, f"batch_predictions.{args.format}")
            inference.save_results(results, output_path, args.format)
            
            logger.info(f"Batch inference completed for {results['batch_size']} samples")
        
        # Generate samples
        elif args.generate:
            logger.info(f"Generating {args.generate} random samples...")
            
            # Create random parameters
            np.random.seed(42)  # For reproducibility
            random_params = np.random.rand(args.generate, 4)
            
            # Scale to realistic ranges
            random_params[:, 0] = random_params[:, 0] * 60 + 20   # Age: 20-80
            random_params[:, 1] = random_params[:, 1] * 40 + 60   # Intensity: 60-100 dB
            random_params[:, 2] = random_params[:, 2] * 70 + 10   # Rate: 10-80 Hz
            random_params[:, 3] = random_params[:, 3] * 0.5 + 0.5 # FMP: 0.5-1.0
            
            results = inference.predict_batch(random_params, return_signals=True)
            
            # Save results
            output_path = os.path.join(args.output_dir, f"generated_samples.{args.format}")
            inference.save_results(results, output_path, args.format)
            
            # Save parameters
            param_path = os.path.join(args.output_dir, "generation_parameters.npz")
            np.savez(param_path, static_params=random_params)
            
            logger.info(f"Generated {args.generate} samples")
        
        else:
            logger.error("Please specify input parameters, input file, or generation count")
            parser.print_help()
            sys.exit(1)
        
        logger.info(f"\nInference completed! Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()