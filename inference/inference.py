#!/usr/bin/env python3
"""
ABR Signal Generation and Multi-Task Inference Pipeline

This module provides complete inference capabilities for the ABR S4+Transformer-based
diffusion model, including:
- Conditional signal generation with DDIM sampling
- Peak detection and estimation (existence, latency, amplitude)
- Hearing loss classification
- Hearing threshold estimation
- Classifier-free guidance (CFG) support
- Structured output generation

Author: AI Assistant
Date: January 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.hierarchical_unet import ProfessionalHierarchicalUNet
from diffusion.sampling import DDIMSampler
from diffusion.schedule import get_noise_schedule


class ABRInferenceEngine:
    """
    Complete inference engine for ABR signal generation and multi-task prediction.
    
    Supports:
    - Conditional signal generation with diffusion sampling
    - Multi-task prediction (peaks, classification, threshold)
    - Classifier-free guidance (CFG)
    - Batch processing with configurable parameters
    - Multiple output formats (JSON, CSV, visualizations)
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        cfg_scale: float = 2.0,
        sampling_steps: int = 50,
        batch_size: int = 8,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the ABR inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device for inference ('cuda', 'cpu', or 'auto')
            cfg_scale: Classifier-free guidance scale
            sampling_steps: Number of DDIM sampling steps
            batch_size: Batch size for inference
            class_names: List of hearing loss class names
        """
        self.model_path = model_path
        self.cfg_scale = cfg_scale
        self.sampling_steps = sampling_steps
        self.batch_size = batch_size
        self.class_names = class_names or ["NORMAL", "NÖROPATİ", "SNİK", "TOTAL", "İTİK"]
        
        # Setup device
        if device == 'auto' or device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Initializing ABR Inference Engine on {self.device}")
        
        # Load model and setup sampler
        self.model = self._load_model()
        self.sampler = self._setup_sampler()
        
        print(f"✅ Inference engine ready with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _load_model(self) -> nn.Module:
        """Load the trained ABR model from checkpoint."""
        print(f"📦 Loading model from: {self.model_path}")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model configuration
        if 'config' in checkpoint:
            model_config = checkpoint['config']
        else:
            # Default configuration for ABR model
            model_config = {
                'input_channels': 1,
                'static_dim': 4,
                'base_channels': 64,
                'n_levels': 4,
                'sequence_length': 200,
                'signal_length': 200,
                'num_classes': len(self.class_names),
                'n_transformer_layers': 3,  # Changed from num_transformer_layers
                'use_cross_attention': True,
                'use_positional_encoding': True,
                'film_dropout': 0.15,
                'use_cfg': True  # Enable CFG for inference
            }
            print("⚠️  Using default model configuration")
        
        # Create and load model
        model = ProfessionalHierarchicalUNet(**model_config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _setup_sampler(self) -> DDIMSampler:
        """Setup DDIM sampler for signal generation."""
        print("🎯 Setting up DDIM sampler...")
        
        # Create noise schedule
        noise_schedule = get_noise_schedule('cosine', num_timesteps=1000)
        
        # Initialize DDIM sampler
        sampler = DDIMSampler(noise_schedule, eta=0.0)  # Deterministic sampling
        
        return sampler
    
    def generate_signals(
        self,
        static_params: torch.Tensor,
        cfg_enabled: Optional[List[bool]] = None,
        signal_length: int = 200,
        use_cfg: bool = True,
        progress: bool = True
    ) -> torch.Tensor:
        """
        Generate ABR signals conditioned on static parameters.
        
        Args:
            static_params: Static parameters [batch, 4] (age, intensity, rate, fmp)
            cfg_enabled: Per-sample CFG enable flags
            signal_length: Length of generated signals
            use_cfg: Whether to use classifier-free guidance
            progress: Show progress bar
            
        Returns:
            Generated signals [batch, 1, signal_length]
        """
        batch_size = static_params.size(0)
        
        # Setup CFG masks
        if use_cfg and cfg_enabled is not None:
            cfg_mask = torch.tensor(cfg_enabled, dtype=torch.bool, device=self.device)
        elif use_cfg:
            cfg_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        else:
            cfg_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Generate signals using DDIM sampling
        with torch.no_grad():
            generated_signals = self.sampler.sample(
                model=self.model,
                shape=(batch_size, 1, signal_length),
                static_params=static_params,
                device=self.device,
                num_steps=self.sampling_steps,
                cfg_scale=self.cfg_scale if use_cfg else 1.0,
                cfg_mask=cfg_mask,
                progress=progress
            )
        
        return generated_signals
    
    def predict_diagnostics(
        self,
        signals: torch.Tensor,
        static_params: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict diagnostic outputs from generated signals.
        
        Args:
            signals: Generated signals [batch, 1, signal_length]
            static_params: Static parameters [batch, 4]
            
        Returns:
            Dictionary containing all diagnostic predictions
        """
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(signals, static_params)
            
            # Process outputs
            diagnostics = {}
            
            # Peak predictions
            if 'peak' in outputs and len(outputs['peak']) >= 3:
                peak_outputs = outputs['peak']
                diagnostics['peak_existence_logits'] = peak_outputs[0]  # [batch, 1]
                diagnostics['peak_existence'] = torch.sigmoid(peak_outputs[0]) > 0.5  # [batch, 1]
                diagnostics['peak_latency'] = peak_outputs[1]  # [batch, 1]
                diagnostics['peak_amplitude'] = peak_outputs[2]  # [batch, 1]
            
            # Classification predictions
            if 'class' in outputs:
                diagnostics['class_logits'] = outputs['class']  # [batch, num_classes]
                diagnostics['class_probabilities'] = F.softmax(outputs['class'], dim=1)
                diagnostics['predicted_class'] = torch.argmax(outputs['class'], dim=1)  # [batch]
            
            # Threshold predictions
            if 'threshold' in outputs:
                diagnostics['threshold_raw'] = outputs['threshold']  # [batch, 1] or [batch, 2]
                
                # Handle uncertainty-aware threshold prediction
                if outputs['threshold'].size(-1) == 2:
                    diagnostics['threshold_mean'] = outputs['threshold'][:, 0:1]
                    diagnostics['threshold_std'] = F.softplus(outputs['threshold'][:, 1:2]) + 1e-6
                    diagnostics['threshold'] = diagnostics['threshold_mean']
                else:
                    diagnostics['threshold'] = outputs['threshold']
                
                # Clamp threshold to reasonable range
                diagnostics['threshold'] = torch.clamp(diagnostics['threshold'], 0, 120)
            
            # Signal reconstruction (for evaluation)
            if 'recon' in outputs:
                diagnostics['reconstructed_signal'] = outputs['recon']
        
        return diagnostics
    
    def post_process_predictions(
        self,
        diagnostics: Dict[str, torch.Tensor],
        static_params: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply post-processing to diagnostic predictions.
        
        Args:
            diagnostics: Raw diagnostic predictions
            static_params: Static parameters for context
            
        Returns:
            Post-processed diagnostic predictions
        """
        processed = diagnostics.copy()
        
        # Post-process peak predictions
        if 'peak_latency' in processed:
            # Clamp latency to physiologically reasonable range (0-10ms for ABR)
            processed['peak_latency'] = torch.clamp(processed['peak_latency'], 0, 10)
        
        if 'peak_amplitude' in processed:
            # Clamp amplitude to reasonable range
            processed['peak_amplitude'] = torch.clamp(processed['peak_amplitude'], 0, 2.0)
        
        # Add clinical interpretation flags
        if 'threshold' in processed:
            # Clinical hearing loss categories based on threshold
            thresholds = processed['threshold'].squeeze(-1)
            
            # Define clinical categories
            normal_mask = thresholds <= 25
            mild_mask = (thresholds > 25) & (thresholds <= 40)
            moderate_mask = (thresholds > 40) & (thresholds <= 55)
            moderate_severe_mask = (thresholds > 55) & (thresholds <= 70)
            severe_mask = (thresholds > 70) & (thresholds <= 90)
            profound_mask = thresholds > 90
            
            # Create clinical category tensor
            clinical_category = torch.zeros_like(thresholds, dtype=torch.long)
            clinical_category[mild_mask] = 1
            clinical_category[moderate_mask] = 2
            clinical_category[moderate_severe_mask] = 3
            clinical_category[severe_mask] = 4
            clinical_category[profound_mask] = 5
            
            processed['clinical_category'] = clinical_category
            processed['clinical_category_names'] = [
                'Normal', 'Mild', 'Moderate', 'Moderate-Severe', 'Severe', 'Profound'
            ]
        
        return processed
    
    def format_results(
        self,
        signals: torch.Tensor,
        diagnostics: Dict[str, torch.Tensor],
        static_params: torch.Tensor,
        patient_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Format inference results into structured output.
        
        Args:
            signals: Generated signals [batch, 1, signal_length]
            diagnostics: Diagnostic predictions
            static_params: Input static parameters
            patient_ids: Optional patient identifiers
            
        Returns:
            List of formatted results for each sample
        """
        batch_size = signals.size(0)
        results = []
        
        for i in range(batch_size):
            # Basic information
            result = {
                'patient_id': patient_ids[i] if patient_ids else f'sample_{i:03d}',
                'static_parameters': {
                    'age': static_params[i, 0].item(),
                    'intensity': static_params[i, 1].item(),
                    'rate': static_params[i, 2].item(),
                    'fmp': static_params[i, 3].item()
                }
            }
            
            # Generated signal
            result['generated_signal'] = signals[i, 0].detach().cpu().numpy().tolist()
            
            # Peak predictions
            if 'peak_existence' in diagnostics:
                peak_exists = diagnostics['peak_existence'][i].item()
                result['v_peak'] = {
                    'exists': bool(peak_exists),
                    'confidence': diagnostics['peak_existence_logits'][i].sigmoid().item()
                }
                
                if peak_exists and 'peak_latency' in diagnostics:
                    result['v_peak']['latency'] = diagnostics['peak_latency'][i].item()
                    result['v_peak']['amplitude'] = diagnostics['peak_amplitude'][i].item()
                else:
                    result['v_peak']['latency'] = None
                    result['v_peak']['amplitude'] = None
            
            # Classification predictions
            if 'predicted_class' in diagnostics:
                class_idx = diagnostics['predicted_class'][i].item()
                class_probs = diagnostics['class_probabilities'][i].detach().cpu().numpy()
                
                result['predicted_class'] = self.class_names[class_idx] if class_idx < len(self.class_names) else f'Class_{class_idx}'
                result['class_confidence'] = float(class_probs[class_idx])
                result['class_probabilities'] = {
                    name: float(prob) for name, prob in zip(self.class_names, class_probs)
                }
            
            # Threshold predictions
            if 'threshold' in diagnostics:
                result['threshold_dB'] = diagnostics['threshold'][i].item()
                
                # Add uncertainty if available
                if 'threshold_std' in diagnostics:
                    result['threshold_uncertainty'] = diagnostics['threshold_std'][i].item()
                
                # Add clinical category
                if 'clinical_category' in diagnostics:
                    cat_idx = diagnostics['clinical_category'][i].item()
                    cat_names = diagnostics['clinical_category_names']
                    result['clinical_category'] = cat_names[cat_idx] if cat_idx < len(cat_names) else 'Unknown'
            
            # Quality metrics (if reconstruction available)
            if 'reconstructed_signal' in diagnostics:
                recon_signal = diagnostics['reconstructed_signal'][i, 0].detach().cpu().numpy()
                orig_signal = signals[i, 0].detach().cpu().numpy()
                
                # Compute reconstruction quality
                mse = np.mean((recon_signal - orig_signal) ** 2)
                correlation = np.corrcoef(recon_signal, orig_signal)[0, 1] if np.std(recon_signal) > 1e-8 else 0.0
                
                result['quality_metrics'] = {
                    'reconstruction_mse': float(mse),
                    'reconstruction_correlation': float(correlation) if not np.isnan(correlation) else 0.0
                }
            
            results.append(result)
        
        return results
    
    def run_inference(
        self,
        inputs: Union[str, List[Dict], torch.Tensor],
        static_params: Optional[torch.Tensor] = None,
        cfg_enabled: Optional[List[bool]] = None,
        patient_ids: Optional[List[str]] = None,
        use_cfg: bool = True,
        signal_length: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Run complete inference pipeline.
        
        Args:
            inputs: Input data (JSON file path, list of dicts, or tensor)
            static_params: Static parameters if inputs is tensor
            cfg_enabled: Per-sample CFG flags
            patient_ids: Patient identifiers
            use_cfg: Whether to use classifier-free guidance
            signal_length: Length of generated signals
            
        Returns:
            List of formatted inference results
        """
        # Parse inputs
        if isinstance(inputs, str):
            # Load from JSON file
            with open(inputs, 'r') as f:
                input_data = json.load(f)
            
            static_params, cfg_enabled, patient_ids = self._parse_json_inputs(input_data)
        
        elif isinstance(inputs, list):
            # Parse list of dictionaries
            static_params, cfg_enabled, patient_ids = self._parse_json_inputs(inputs)
        
        elif isinstance(inputs, torch.Tensor):
            # Use provided tensor
            if static_params is None:
                static_params = inputs
            static_params = static_params.to(self.device)
        
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
        
        # Ensure we have required data
        if static_params is None:
            raise ValueError("No static parameters provided")
        
        batch_size = static_params.size(0)
        all_results = []
        
        # Process in batches
        print(f"🔮 Running inference on {batch_size} samples...")
        
        for start_idx in tqdm(range(0, batch_size, self.batch_size), desc="Inference"):
            end_idx = min(start_idx + self.batch_size, batch_size)
            
            # Extract batch
            batch_static = static_params[start_idx:end_idx]
            batch_cfg = cfg_enabled[start_idx:end_idx] if cfg_enabled else None
            batch_ids = patient_ids[start_idx:end_idx] if patient_ids else None
            
            # Generate signals
            generated_signals = self.generate_signals(
                batch_static,
                cfg_enabled=batch_cfg,
                signal_length=signal_length,
                use_cfg=use_cfg,
                progress=False
            )
            
            # Predict diagnostics
            diagnostics = self.predict_diagnostics(generated_signals, batch_static)
            
            # Post-process predictions
            diagnostics = self.post_process_predictions(diagnostics, batch_static)
            
            # Format results
            batch_results = self.format_results(
                generated_signals, diagnostics, batch_static, batch_ids
            )
            
            all_results.extend(batch_results)
        
        print(f"✅ Inference completed for {len(all_results)} samples")
        return all_results
    
    def _parse_json_inputs(
        self, 
        input_data: List[Dict]
    ) -> Tuple[torch.Tensor, List[bool], List[str]]:
        """Parse JSON input format into tensors."""
        static_params = []
        cfg_enabled = []
        patient_ids = []
        
        for item in input_data:
            # Extract static parameters
            if 'static' in item:
                static_params.append(item['static'])
            else:
                raise ValueError(f"Missing 'static' field in input: {item}")
            
            # Extract CFG flag
            cfg_enabled.append(item.get('use_cfg', True))
            
            # Extract patient ID
            patient_ids.append(item.get('patient_id', f'sample_{len(patient_ids):03d}'))
        
        # Convert to tensors
        static_tensor = torch.tensor(static_params, dtype=torch.float32, device=self.device)
        
        return static_tensor, cfg_enabled, patient_ids
    
    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_dir: str,
        save_json: bool = True,
        save_csv: bool = True,
        save_signals: bool = False,
        save_visualizations: bool = False
    ) -> Dict[str, str]:
        """
        Save inference results in multiple formats.
        
        Args:
            results: Formatted inference results
            output_dir: Output directory
            save_json: Save detailed JSON results
            save_csv: Save summary CSV
            save_signals: Save individual signal files
            save_visualizations: Save signal visualizations
            
        Returns:
            Dictionary of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save detailed JSON results
        if save_json:
            json_path = output_path / "results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            saved_files['json'] = str(json_path)
            print(f"📄 Saved detailed results: {json_path}")
        
        # Save summary CSV
        if save_csv:
            csv_path = output_path / "results.csv"
            self._save_csv_summary(results, csv_path)
            saved_files['csv'] = str(csv_path)
            print(f"📊 Saved CSV summary: {csv_path}")
        
        # Save individual signal files
        if save_signals:
            signals_dir = output_path / "signals"
            signals_dir.mkdir(exist_ok=True)
            
            for result in results:
                patient_id = result['patient_id']
                signal = np.array(result['generated_signal'])
                
                # Save as numpy array
                signal_path = signals_dir / f"{patient_id}_signal.npy"
                np.save(signal_path, signal)
            
            saved_files['signals'] = str(signals_dir)
            print(f"🔊 Saved {len(results)} signal files: {signals_dir}")
        
        # Save visualizations
        if save_visualizations:
            viz_dir = output_path / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            for result in results:
                viz_path = self._create_signal_visualization(result, viz_dir)
                if viz_path:
                    continue
            
            saved_files['visualizations'] = str(viz_dir)
            print(f"📈 Saved visualizations: {viz_dir}")
        
        return saved_files
    
    def _save_csv_summary(self, results: List[Dict[str, Any]], csv_path: Path):
        """Save summary results as CSV."""
        with open(csv_path, 'w', newline='') as f:
            if not results:
                return
            
            # Define CSV columns
            fieldnames = [
                'patient_id', 'age', 'intensity', 'rate', 'fmp',
                'predicted_class', 'class_confidence', 'threshold_dB',
                'peak_exists', 'peak_confidence', 'peak_latency', 'peak_amplitude',
                'clinical_category'
            ]
            
            # Add uncertainty fields if available
            if 'threshold_uncertainty' in results[0]:
                fieldnames.append('threshold_uncertainty')
            
            # Add quality metrics if available
            if 'quality_metrics' in results[0]:
                fieldnames.extend(['reconstruction_mse', 'reconstruction_correlation'])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'patient_id': result['patient_id'],
                    'age': result['static_parameters']['age'],
                    'intensity': result['static_parameters']['intensity'],
                    'rate': result['static_parameters']['rate'],
                    'fmp': result['static_parameters']['fmp'],
                    'predicted_class': result.get('predicted_class', ''),
                    'class_confidence': result.get('class_confidence', ''),
                    'threshold_dB': result.get('threshold_dB', ''),
                    'peak_exists': result.get('v_peak', {}).get('exists', ''),
                    'peak_confidence': result.get('v_peak', {}).get('confidence', ''),
                    'peak_latency': result.get('v_peak', {}).get('latency', ''),
                    'peak_amplitude': result.get('v_peak', {}).get('amplitude', ''),
                    'clinical_category': result.get('clinical_category', '')
                }
                
                # Add optional fields
                if 'threshold_uncertainty' in result:
                    row['threshold_uncertainty'] = result['threshold_uncertainty']
                
                if 'quality_metrics' in result:
                    row['reconstruction_mse'] = result['quality_metrics']['reconstruction_mse']
                    row['reconstruction_correlation'] = result['quality_metrics']['reconstruction_correlation']
                
                writer.writerow(row)
    
    def _create_signal_visualization(
        self, 
        result: Dict[str, Any], 
        viz_dir: Path
    ) -> Optional[str]:
        """Create visualization for a single result."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            patient_id = result['patient_id']
            signal = np.array(result['generated_signal'])
            time_axis = np.linspace(0, 10, len(signal))  # 10ms ABR
            
            # Plot 1: Generated signal
            axes[0, 0].plot(time_axis, signal, 'b-', linewidth=2)
            axes[0, 0].set_title(f'Generated ABR Signal - {patient_id}')
            axes[0, 0].set_xlabel('Time (ms)')
            axes[0, 0].set_ylabel('Amplitude (μV)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add peak annotation if exists
            if result.get('v_peak', {}).get('exists') and result['v_peak'].get('latency') is not None:
                latency = result['v_peak']['latency']
                amplitude = result['v_peak']['amplitude']
                axes[0, 0].axvline(latency, color='red', linestyle='--', alpha=0.7)
                axes[0, 0].plot(latency, amplitude, 'ro', markersize=8)
                axes[0, 0].text(latency + 0.2, amplitude, f'Peak\n{latency:.1f}ms', 
                               fontsize=8, ha='left')
            
            # Plot 2: Class probabilities
            if 'class_probabilities' in result:
                class_probs = result['class_probabilities']
                classes = list(class_probs.keys())
                probs = list(class_probs.values())
                
                bars = axes[0, 1].bar(classes, probs, alpha=0.7)
                axes[0, 1].set_title('Hearing Loss Classification')
                axes[0, 1].set_ylabel('Probability')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Highlight predicted class
                pred_class = result.get('predicted_class', '')
                for bar, class_name in zip(bars, classes):
                    if class_name == pred_class:
                        bar.set_color('red')
                        bar.set_alpha(1.0)
            
            # Plot 3: Threshold visualization
            if 'threshold_dB' in result:
                threshold = result['threshold_dB']
                clinical_cat = result.get('clinical_category', 'Unknown')
                
                # Create threshold bar
                axes[1, 0].barh(['Threshold'], [threshold], color='orange', alpha=0.7)
                axes[1, 0].set_xlim(0, 120)
                axes[1, 0].set_xlabel('Threshold (dB SPL)')
                axes[1, 0].set_title(f'Hearing Threshold: {threshold:.1f} dB\nCategory: {clinical_cat}')
                
                # Add clinical range markers
                ranges = [(0, 25, 'Normal'), (25, 40, 'Mild'), (40, 55, 'Moderate'), 
                         (55, 70, 'Mod-Severe'), (70, 90, 'Severe'), (90, 120, 'Profound')]
                
                for start, end, label in ranges:
                    axes[1, 0].axvspan(start, end, alpha=0.1, 
                                      color='green' if label == clinical_cat else 'gray')
            
            # Plot 4: Summary information
            axes[1, 1].axis('off')
            
            # Create summary text
            summary_text = f"Patient: {patient_id}\n\n"
            summary_text += f"Static Parameters:\n"
            summary_text += f"  Age: {result['static_parameters']['age']:.1f}\n"
            summary_text += f"  Intensity: {result['static_parameters']['intensity']:.1f} dB\n"
            summary_text += f"  Rate: {result['static_parameters']['rate']:.1f} Hz\n"
            summary_text += f"  FMP: {result['static_parameters']['fmp']:.2f}\n\n"
            
            if 'predicted_class' in result:
                summary_text += f"Predicted Class: {result['predicted_class']}\n"
                summary_text += f"Confidence: {result['class_confidence']:.3f}\n\n"
            
            if 'v_peak' in result:
                peak_info = result['v_peak']
                summary_text += f"Peak Detection:\n"
                summary_text += f"  Exists: {peak_info['exists']}\n"
                summary_text += f"  Confidence: {peak_info['confidence']:.3f}\n"
                if peak_info['exists']:
                    summary_text += f"  Latency: {peak_info['latency']:.2f} ms\n"
                    summary_text += f"  Amplitude: {peak_info['amplitude']:.3f} μV\n"
            
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = viz_dir / f"{patient_id}_visualization.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            return str(viz_path)
            
        except Exception as e:
            print(f"⚠️  Failed to create visualization for {result.get('patient_id', 'unknown')}: {e}")
            return None


def create_sample_input(output_path: str = "sample_input.json"):
    """Create a sample input JSON file for testing."""
    sample_data = [
        {
            "patient_id": "P001",
            "static": [25, 70, 31, 2.8],  # age, intensity, rate, fmp
            "use_cfg": True
        },
        {
            "patient_id": "P002", 
            "static": [45, 85, 21, 1.5],
            "use_cfg": True
        },
        {
            "patient_id": "P003",
            "static": [60, 95, 11, 0.8],
            "use_cfg": False  # Disable CFG for this sample
        }
    ]
    
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"📝 Sample input file created: {output_path}")
    return output_path


def main():
    """Main inference function with CLI interface."""
    parser = argparse.ArgumentParser(
        description='ABR Signal Generation and Multi-Task Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    
    # Input arguments
    parser.add_argument(
        '--input_json', type=str, default=None,
        help='Path to input JSON file with patient data'
    )
    parser.add_argument(
        '--create_sample', action='store_true',
        help='Create sample input JSON file and exit'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir', type=str, default='outputs/inference',
        help='Output directory for results'
    )
    parser.add_argument(
        '--save_json', action='store_true', default=True,
        help='Save detailed JSON results'
    )
    parser.add_argument(
        '--save_csv', action='store_true', default=True,
        help='Save CSV summary'
    )
    parser.add_argument(
        '--save_signals', action='store_true',
        help='Save individual signal files (.npy)'
    )
    parser.add_argument(
        '--save_visualizations', action='store_true',
        help='Save signal visualizations (.png)'
    )
    
    # Model configuration
    parser.add_argument(
        '--cfg_scale', type=float, default=2.0,
        help='Classifier-free guidance scale'
    )
    parser.add_argument(
        '--steps', type=int, default=50,
        help='Number of DDIM sampling steps'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device for inference'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--signal_length', type=int, default=200,
        help='Length of generated signals'
    )
    
    # Feature flags
    parser.add_argument(
        '--no_cfg', action='store_true',
        help='Disable classifier-free guidance'
    )
    parser.add_argument(
        '--no_peak_pred', action='store_true',
        help='Skip peak prediction (not implemented - for future use)'
    )
    
    args = parser.parse_args()
    
    # Create sample input if requested
    if args.create_sample:
        create_sample_input("sample_input.json")
        return
    
    # Validate inputs
    if not args.input_json:
        print("❌ Error: --input_json is required (or use --create_sample to generate example)")
        return
    
    if not Path(args.model_path).exists():
        print(f"❌ Error: Model checkpoint not found: {args.model_path}")
        return
    
    if not Path(args.input_json).exists():
        print(f"❌ Error: Input JSON file not found: {args.input_json}")
        return
    
    try:
        # Initialize inference engine
        engine = ABRInferenceEngine(
            model_path=args.model_path,
            device=args.device,
            cfg_scale=args.cfg_scale,
            sampling_steps=args.steps,
            batch_size=args.batch_size
        )
        
        # Run inference
        results = engine.run_inference(
            inputs=args.input_json,
            use_cfg=not args.no_cfg,
            signal_length=args.signal_length
        )
        
        # Save results
        saved_files = engine.save_results(
            results=results,
            output_dir=args.output_dir,
            save_json=args.save_json,
            save_csv=args.save_csv,
            save_signals=args.save_signals,
            save_visualizations=args.save_visualizations
        )
        
        # Print summary
        print(f"\n🎉 Inference completed successfully!")
        print(f"📊 Processed {len(results)} samples")
        print(f"📁 Results saved to: {args.output_dir}")
        
        for file_type, path in saved_files.items():
            print(f"   {file_type}: {path}")
        
        # Print quick summary
        if results:
            print(f"\n📋 Quick Summary:")
            class_counts = {}
            threshold_sum = 0
            peak_count = 0
            
            for result in results:
                # Count classes
                pred_class = result.get('predicted_class', 'Unknown')
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                
                # Sum thresholds
                if 'threshold_dB' in result:
                    threshold_sum += result['threshold_dB']
                
                # Count peaks
                if result.get('v_peak', {}).get('exists'):
                    peak_count += 1
            
            print(f"   Average threshold: {threshold_sum / len(results):.1f} dB")
            print(f"   Peaks detected: {peak_count}/{len(results)} ({peak_count/len(results)*100:.1f}%)")
            print(f"   Class distribution: {dict(class_counts)}")
        
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code) 