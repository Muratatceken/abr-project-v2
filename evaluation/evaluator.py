"""
Comprehensive Signal Generation Evaluator

This module provides a complete evaluation pipeline for the ABR signal generation model,
including model loading, data processing, metrics computation, and result analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig

from .metrics import compute_all_metrics, SignalMetrics, SpectralMetrics, PerceptualMetrics, ABRSpecificMetrics
from .visualization import EvaluationVisualizer
from .analysis import SignalAnalyzer
from utils.sampling import create_ddim_sampler
from data.dataset import create_optimized_dataloaders


class SignalGenerationEvaluator:
    """Comprehensive evaluator for ABR signal generation models."""
    
    def __init__(self, 
                 model: nn.Module,
                 config: DictConfig,
                 device: str = 'cuda',
                 output_dir: str = 'evaluation_results'):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained ABR generation model
            config: Model configuration
            device: Device to run evaluation on
            output_dir: Directory to save evaluation results
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.visualizer = EvaluationVisualizer(output_dir=str(self.output_dir / 'plots'))
        self.analyzer = SignalAnalyzer()
        
        # Create sampler for generation
        self.sampler = create_ddim_sampler(
            noise_schedule_type=config.get('diffusion', {}).get('schedule_type', 'cosine'),
            num_timesteps=config.get('diffusion', {}).get('num_timesteps', 1000),
            eta=config.get('diffusion', {}).get('eta', 0.0),
            clip_denoised=config.get('diffusion', {}).get('clip_denoised', False)
        )
        
        # Results storage
        self.results = {
            'metrics': {},
            'generation_stats': {},
            'analysis': {},
            'samples': []
        }
    
    def evaluate_on_dataset(self, 
                           test_loader: DataLoader,
                           num_samples: Optional[int] = None,
                           save_samples: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_loader: DataLoader for test data
            num_samples: Number of samples to evaluate (None for all)
            save_samples: Whether to save generated samples
            
        Returns:
            Dictionary containing evaluation results
        """
        self.model.eval()
        
        all_metrics = []
        generated_samples = []
        real_samples = []
        static_params = []
        
        print("🔍 Evaluating model on test dataset...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                if num_samples and batch_idx * test_loader.batch_size >= num_samples:
                    break
                
                # Get batch data
                real_signals = batch['signal'].to(self.device)
                batch_static = batch['static_params'].to(self.device)
                
                # Generate signals
                generated_signals = self._generate_signals(batch_static, real_signals.shape[-1])
                
                # Compute metrics for each sample in batch
                for i in range(real_signals.size(0)):
                    real_sample = real_signals[i:i+1]
                    generated_sample = generated_signals[i:i+1]
                    
                    # Compute comprehensive metrics
                    sample_metrics = compute_all_metrics(
                        generated_sample, 
                        real_sample,
                        sr=self.config.get('data', {}).get('sampling_rate', 1000)
                    )
                    
                    # Add sample info
                    sample_metrics['batch_idx'] = batch_idx
                    sample_metrics['sample_idx'] = i
                    all_metrics.append(sample_metrics)
                    
                    # Store samples if requested
                    if save_samples:
                        generated_samples.append(generated_sample.cpu())
                        real_samples.append(real_sample.cpu())
                        static_params.append(batch_static[i:i+1].cpu())
        
        # Aggregate results
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        
        # Store results
        self.results['metrics'] = aggregated_metrics
        self.results['samples'] = {
            'generated': generated_samples[:100],  # Store first 100 samples
            'real': real_samples[:100],
            'static_params': static_params[:100]
        }
        
        return aggregated_metrics
    
    def evaluate_generation_quality(self, 
                                   static_conditions: torch.Tensor,
                                   num_steps: int = 50,
                                   cfg_scale: float = 1.0) -> Dict[str, Any]:
        """
        Evaluate generation quality with different sampling parameters.
        
        Args:
            static_conditions: Static parameters for generation
            num_steps: Number of DDIM sampling steps
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            Generation quality metrics
        """
        self.model.eval()
        
        generation_stats = {
            'sampling_steps': num_steps,
            'cfg_scale': cfg_scale,
            'generation_times': [],
            'signal_properties': []
        }
        
        print(f"🎯 Evaluating generation quality (steps={num_steps}, cfg={cfg_scale})...")
        
        with torch.no_grad():
            for i in tqdm(range(static_conditions.size(0)), desc="Generating"):
                static_sample = static_conditions[i:i+1]
                
                # Measure generation time
                start_time = time.time()
                generated_signal = self._generate_signals(static_sample, 
                                                        self.config.get('model', {}).get('signal_length', 200),
                                                        num_steps=num_steps,
                                                        cfg_scale=cfg_scale)
                generation_time = time.time() - start_time
                
                generation_stats['generation_times'].append(generation_time)
                
                # Analyze signal properties
                signal_props = self.analyzer.analyze_signal_properties(generated_signal)
                generation_stats['signal_properties'].append(signal_props)
        
        # Aggregate generation statistics
        generation_stats['avg_generation_time'] = np.mean(generation_stats['generation_times'])
        generation_stats['std_generation_time'] = np.std(generation_stats['generation_times'])
        
        self.results['generation_stats'] = generation_stats
        return generation_stats
    
    def evaluate_conditional_control(self, 
                                   test_loader: DataLoader,
                                   condition_variations: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Evaluate how well the model responds to different conditional inputs.
        
        Args:
            test_loader: Test data loader
            condition_variations: List of condition variations to test
            
        Returns:
            Conditional control evaluation results
        """
        self.model.eval()
        
        control_results = {
            'condition_response': [],
            'interpolation_quality': [],
            'extrapolation_robustness': []
        }
        
        print("🎛️ Evaluating conditional control...")
        
        # Get a reference batch
        reference_batch = next(iter(test_loader))
        base_static = reference_batch['static_params'][:1].to(self.device)
        
        with torch.no_grad():
            # Test condition variations
            for variation in tqdm(condition_variations, desc="Testing conditions"):
                # Modify static parameters
                modified_static = base_static.clone()
                for param_name, param_value in variation.items():
                    # Assuming static params are ordered as [age, gender, hearing_loss, intensity]
                    param_idx = {'age': 0, 'gender': 1, 'hearing_loss': 2, 'intensity': 3}.get(param_name, 0)
                    modified_static[0, param_idx] = param_value
                
                # Generate with modified conditions
                generated_signal = self._generate_signals(modified_static, 
                                                        self.config.get('model', {}).get('signal_length', 200))
                
                # Analyze response
                response_analysis = self.analyzer.analyze_conditional_response(
                    generated_signal, modified_static, variation
                )
                control_results['condition_response'].append(response_analysis)
        
        self.results['analysis']['conditional_control'] = control_results
        return control_results
    
    def evaluate_consistency(self, 
                           static_conditions: torch.Tensor,
                           num_generations: int = 10) -> Dict[str, Any]:
        """
        Evaluate generation consistency with same input conditions.
        
        Args:
            static_conditions: Static parameters
            num_generations: Number of generations with same conditions
            
        Returns:
            Consistency evaluation results
        """
        self.model.eval()
        
        consistency_results = {
            'intra_condition_variance': [],
            'inter_condition_variance': [],
            'consistency_scores': []
        }
        
        print("🔄 Evaluating generation consistency...")
        
        with torch.no_grad():
            for condition_idx in tqdm(range(min(static_conditions.size(0), 10)), desc="Testing consistency"):
                static_sample = static_conditions[condition_idx:condition_idx+1]
                
                # Generate multiple samples with same conditions
                generated_samples = []
                for _ in range(num_generations):
                    sample = self._generate_signals(static_sample, 
                                                  self.config.get('model', {}).get('signal_length', 200))
                    generated_samples.append(sample)
                
                # Analyze consistency
                consistency_analysis = self.analyzer.analyze_generation_consistency(generated_samples)
                consistency_results['consistency_scores'].append(consistency_analysis)
        
        self.results['analysis']['consistency'] = consistency_results
        return consistency_results
    
    def generate_evaluation_report(self, save_report: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_report: Whether to save report to disk
            
        Returns:
            Complete evaluation report
        """
        print("📊 Generating evaluation report...")
        
        report = {
            'evaluation_summary': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_config': dict(self.config),
                'evaluation_metrics': self.results.get('metrics', {}),
                'generation_stats': self.results.get('generation_stats', {}),
                'analysis_results': self.results.get('analysis', {})
            },
            'recommendations': self._generate_recommendations(),
            'visualizations': self._create_evaluation_plots()
        }
        
        if save_report:
            # Save JSON report
            report_path = self.output_dir / 'evaluation_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Save summary metrics
            if 'metrics' in self.results:
                metrics_df = pd.DataFrame([self.results['metrics']])
                metrics_df.to_csv(self.output_dir / 'summary_metrics.csv', index=False)
            
            print(f"✅ Evaluation report saved to {self.output_dir}")
        
        return report
    
    def _generate_signals(self, 
                         static_params: torch.Tensor, 
                         signal_length: int,
                         num_steps: int = 50,
                         cfg_scale: float = 1.0) -> torch.Tensor:
        """Generate signals using the trained model."""
        batch_size = static_params.size(0)
        in_channels = getattr(self.model, 'input_channels', 1)
        shape = (batch_size, in_channels, signal_length)
        
        generated = self.sampler.sample(
            self.model,
            shape=shape,
            static_params=static_params,
            device=self.device,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            progress=False
        )
        
        return generated
    
    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics across all samples."""
        aggregated = {}
        
        # Simple metrics (scalars)
        scalar_metrics = ['mse', 'mae', 'rmse', 'snr', 'psnr', 
                         'amplitude_envelope_similarity', 'phase_coherence']
        
        for metric in scalar_metrics:
            values = [m[metric] for m in all_metrics if metric in m]
            if values:
                aggregated[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        # Complex metrics (dictionaries)
        correlation_values = [m['correlation']['pearson_r'] for m in all_metrics 
                            if 'correlation' in m and 'pearson_r' in m['correlation']]
        if correlation_values:
            aggregated['correlation'] = {
                'pearson_r': {
                    'mean': float(np.mean(correlation_values)),
                    'std': float(np.std(correlation_values))
                }
            }
        
        # Frequency domain metrics
        freq_mse_values = [m['frequency_response']['frequency_mse'] for m in all_metrics 
                          if 'frequency_response' in m]
        if freq_mse_values:
            aggregated['frequency_response'] = {
                'frequency_mse': {
                    'mean': float(np.mean(freq_mse_values)),
                    'std': float(np.std(freq_mse_values))
                }
            }
        
        return aggregated
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        if 'metrics' in self.results:
            metrics = self.results['metrics']
            
            # SNR recommendations
            if 'snr' in metrics and metrics['snr']['mean'] < 10:
                recommendations.append(
                    "Low SNR detected. Consider increasing training epochs or adjusting loss weighting."
                )
            
            # Correlation recommendations
            if 'correlation' in metrics and metrics['correlation']['pearson_r']['mean'] < 0.8:
                recommendations.append(
                    "Low signal correlation. Consider improving model architecture or training data quality."
                )
            
            # Generation time recommendations
            if 'generation_stats' in self.results:
                avg_time = self.results['generation_stats'].get('avg_generation_time', 0)
                if avg_time > 1.0:  # More than 1 second per sample
                    recommendations.append(
                        "Generation time is high. Consider using fewer DDIM steps or model optimization."
                    )
        
        if not recommendations:
            recommendations.append("Model performance looks good across all evaluated metrics.")
        
        return recommendations
    
    def _create_evaluation_plots(self) -> List[str]:
        """Create evaluation visualization plots."""
        plot_paths = []
        
        if self.results.get('samples'):
            # Plot sample comparisons
            plot_path = self.visualizer.plot_sample_comparisons(
                self.results['samples']['generated'][:10],
                self.results['samples']['real'][:10]
            )
            plot_paths.append(plot_path)
            
            # Plot metrics distributions
            if 'metrics' in self.results:
                plot_path = self.visualizer.plot_metrics_distribution(self.results['metrics'])
                plot_paths.append(plot_path)
        
        return plot_paths