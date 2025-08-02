#!/usr/bin/env python3
"""
ABR Training Diagnostic Script

Analyzes training convergence issues and provides specific recommendations.
Run this before training to check for potential problems.

Usage:
    python diagnose_training.py --config training/config_convergence_fix.yaml
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

# Import project modules
from models.hierarchical_unet import ProfessionalHierarchicalUNet
from diffusion.loss import ABRDiffusionLoss
from data.dataset import create_optimized_dataloaders
from training.enhanced_train import create_model


def analyze_loss_components(model: nn.Module, loss_fn: ABRDiffusionLoss, 
                           dataloader, device: torch.device, num_batches: int = 10) -> Dict[str, Any]:
    """Analyze loss component magnitudes and ratios."""
    model.eval()
    
    loss_components = {
        'signal_loss': [],
        'peak_exist_loss': [],
        'peak_latency_loss': [],
        'peak_amplitude_loss': [],
        'classification_loss': [],
        'threshold_loss': [],
        'total_loss': []
    }
    
    print("🔍 Analyzing loss component magnitudes...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            # Move to device
            for key in ['static_params', 'signal', 'v_peak', 'v_peak_mask', 'target']:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            outputs = model(batch['signal'], batch['static_params'])
            total_loss, loss_dict = loss_fn(outputs, batch)
            
            # Store individual components
            for key in loss_components.keys():
                if key in loss_dict:
                    loss_components[key].append(loss_dict[key].item())
                elif key == 'total_loss':
                    loss_components[key].append(total_loss.item())
                else:
                    loss_components[key].append(0.0)
    
    # Compute statistics
    analysis = {}
    for component, values in loss_components.items():
        if values:
            analysis[component] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    return analysis


def check_gradient_flow(model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
    """Check for gradient flow issues."""
    print("🔍 Checking gradient flow...")
    
    # Compute gradients
    loss.backward(retain_graph=True)
    
    gradient_analysis = {
        'has_gradients': 0,
        'zero_gradients': 0,
        'nan_gradients': 0,
        'large_gradients': 0,
        'gradient_norms': []
    }
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient_analysis['has_gradients'] += 1
            grad_norm = param.grad.norm().item()
            gradient_analysis['gradient_norms'].append(grad_norm)
            
            if grad_norm == 0:
                gradient_analysis['zero_gradients'] += 1
            elif np.isnan(grad_norm) or np.isinf(grad_norm):
                gradient_analysis['nan_gradients'] += 1
            elif grad_norm > 10.0:
                gradient_analysis['large_gradients'] += 1
        else:
            print(f"⚠️  No gradient for parameter: {name}")
    
    return gradient_analysis


def analyze_model_outputs(model: nn.Module, dataloader, device: torch.device, 
                         num_batches: int = 5) -> Dict[str, Any]:
    """Analyze model output distributions."""
    print("🔍 Analyzing model output distributions...")
    
    model.eval()
    output_analysis = {
        'signal_stats': {'mean': [], 'std': [], 'min': [], 'max': []},
        'class_logit_stats': {'mean': [], 'std': [], 'min': [], 'max': []},
        'peak_stats': {'exists_prob': [], 'latency_range': [], 'amplitude_range': []},
        'threshold_stats': {'mean': [], 'std': [], 'min': [], 'max': []}
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            # Move to device
            for key in ['static_params', 'signal', 'v_peak', 'v_peak_mask', 'target']:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            outputs = model(batch['signal'], batch['static_params'])
            
            # Analyze signal reconstruction
            if 'recon' in outputs:
                signal = outputs['recon']
                output_analysis['signal_stats']['mean'].append(signal.mean().item())
                output_analysis['signal_stats']['std'].append(signal.std().item())
                output_analysis['signal_stats']['min'].append(signal.min().item())
                output_analysis['signal_stats']['max'].append(signal.max().item())
            
            # Analyze classification logits
            if 'class' in outputs:
                logits = outputs['class']
                output_analysis['class_logit_stats']['mean'].append(logits.mean().item())
                output_analysis['class_logit_stats']['std'].append(logits.std().item())
                output_analysis['class_logit_stats']['min'].append(logits.min().item())
                output_analysis['class_logit_stats']['max'].append(logits.max().item())
            
            # Analyze peak predictions
            if 'peak' in outputs:
                peak_outputs = outputs['peak']
                if len(peak_outputs) >= 3:
                    exists_logits, latency, amplitude = peak_outputs[:3]
                    exists_prob = torch.sigmoid(exists_logits).mean().item()
                    output_analysis['peak_stats']['exists_prob'].append(exists_prob)
                    output_analysis['peak_stats']['latency_range'].append([latency.min().item(), latency.max().item()])
                    output_analysis['peak_stats']['amplitude_range'].append([amplitude.min().item(), amplitude.max().item()])
            
            # Analyze threshold predictions
            if 'threshold' in outputs:
                threshold = outputs['threshold']
                output_analysis['threshold_stats']['mean'].append(threshold.mean().item())
                output_analysis['threshold_stats']['std'].append(threshold.std().item())
                output_analysis['threshold_stats']['min'].append(threshold.min().item())
                output_analysis['threshold_stats']['max'].append(threshold.max().item())
    
    return output_analysis


def generate_recommendations(loss_analysis: Dict, gradient_analysis: Dict, 
                           output_analysis: Dict, config: Dict) -> List[str]:
    """Generate specific training recommendations."""
    recommendations = []
    
    # Check loss component balance
    if loss_analysis:
        total_loss_mean = loss_analysis.get('total_loss', {}).get('mean', 0)
        signal_loss_mean = loss_analysis.get('signal_loss', {}).get('mean', 0)
        class_loss_mean = loss_analysis.get('classification_loss', {}).get('mean', 0)
        
        if total_loss_mean > 10.0:
            recommendations.append("🚨 CRITICAL: Total loss too high (>10). Consider reducing learning rate.")
        
        if signal_loss_mean > 0 and class_loss_mean > 0:
            ratio = signal_loss_mean / class_loss_mean
            if ratio > 5.0:
                recommendations.append(f"⚠️  Signal loss dominates classification ({ratio:.2f}:1). Increase classification weight.")
            elif ratio < 0.2:
                recommendations.append(f"⚠️  Classification loss dominates signal ({ratio:.2f}:1). Increase signal weight.")
    
    # Check gradient flow
    if gradient_analysis:
        zero_ratio = gradient_analysis['zero_gradients'] / max(gradient_analysis['has_gradients'], 1)
        nan_ratio = gradient_analysis['nan_gradients'] / max(gradient_analysis['has_gradients'], 1)
        
        if zero_ratio > 0.1:
            recommendations.append(f"🚨 CRITICAL: {zero_ratio:.1%} of gradients are zero. Check for gradient flow issues.")
        
        if nan_ratio > 0:
            recommendations.append(f"🚨 CRITICAL: {nan_ratio:.1%} of gradients are NaN. Reduce learning rate immediately.")
        
        if gradient_analysis['gradient_norms']:
            avg_grad_norm = np.mean(gradient_analysis['gradient_norms'])
            if avg_grad_norm > 5.0:
                recommendations.append(f"⚠️  Large gradients (avg: {avg_grad_norm:.2f}). Increase gradient clipping.")
            elif avg_grad_norm < 1e-6:
                recommendations.append(f"⚠️  Very small gradients (avg: {avg_grad_norm:.2e}). Consider increasing learning rate.")
    
    # Check model outputs
    if output_analysis:
        # Check classification logits
        class_stats = output_analysis.get('class_logit_stats', {})
        if class_stats.get('std'):
            avg_std = np.mean(class_stats['std'])
            if avg_std < 0.1:
                recommendations.append("⚠️  Classification logits have low variance. Model may not be learning to classify.")
        
        # Check peak existence probabilities
        peak_stats = output_analysis.get('peak_stats', {})
        if peak_stats.get('exists_prob'):
            avg_prob = np.mean(peak_stats['exists_prob'])
            if avg_prob < 0.1 or avg_prob > 0.9:
                recommendations.append(f"⚠️  Peak existence predictions are extreme (avg: {avg_prob:.3f}). Check peak loss weighting.")
    
    # Check configuration
    current_lr = config.get('training', {}).get('learning_rate', 1e-4)
    if isinstance(current_lr, str):
        try:
            current_lr = float(current_lr)
        except ValueError:
            current_lr = 1e-4
    
    if current_lr < 1e-4:
        recommendations.append(f"⚠️  Learning rate might be too low ({current_lr:.1e}). Consider increasing to 2-3e-4.")
    
    current_weights = config.get('loss', {}).get('loss_weights', {})
    class_weight = current_weights.get('classification', 1.0)
    signal_weight = current_weights.get('signal', 1.0)
    if class_weight <= signal_weight:
        recommendations.append("⚠️  Classification weight should be higher than signal weight for multi-task learning.")
    
    return recommendations


def create_diagnostic_plots(loss_analysis: Dict, output_analysis: Dict, save_dir: Path):
    """Create diagnostic plots."""
    save_dir.mkdir(exist_ok=True)
    
    # Loss component comparison
    if loss_analysis:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss magnitudes
        components = []
        means = []
        stds = []
        
        for comp, stats in loss_analysis.items():
            if comp != 'total_loss' and 'mean' in stats:
                components.append(comp.replace('_loss', ''))
                means.append(stats['mean'])
                stds.append(stats['std'])
        
        ax[0].bar(components, means, yerr=stds, capsize=5)
        ax[0].set_title('Loss Component Magnitudes')
        ax[0].set_ylabel('Loss Value')
        ax[0].tick_params(axis='x', rotation=45)
        
        # Loss ratios
        if means:
            total_mean = sum(means)
            ratios = [m/total_mean for m in means]
            ax[1].pie(ratios, labels=components, autopct='%1.1f%%')
            ax[1].set_title('Loss Component Ratios')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'loss_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Diagnostic plots saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose ABR training issues')
    parser.add_argument('--config', type=str, default='training/config_convergence_fix.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--output_dir', type=str, default='training_diagnostics',
                       help='Output directory for diagnostic results')
    parser.add_argument('--num_batches', type=int, default=10,
                       help='Number of batches to analyze')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔬 ABR Training Diagnostic")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("🏗️  Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Create loss function
    print("🎯 Creating loss function...")
    loss_config = config.get('loss', {})
    loss_weights = loss_config.get('loss_weights', {})
    
    loss_fn = ABRDiffusionLoss(
        n_classes=config['model']['num_classes'],
        use_focal_loss=loss_config.get('use_focal_loss', False),
        focal_alpha=loss_config.get('focal_alpha', 1.0),
        focal_gamma=loss_config.get('focal_gamma', 2.0),
        peak_loss_type=loss_config.get('peak_loss_type', 'mse'),
        huber_delta=loss_config.get('huber_delta', 1.0),
        device=device,
        signal_weight=loss_weights.get('signal', 1.0),
        peak_weight=max(loss_weights.get('peak_latency', 2.0), loss_weights.get('peak_amplitude', 2.0)),
        class_weight=loss_weights.get('classification', 3.0),
        threshold_weight=loss_weights.get('threshold', 1.5),
        joint_generation_weight=loss_weights.get('static_params', 0.5)
    )
    
    # Create data loader
    print("📊 Creating data loader...")
    train_loader, val_loader, _, _ = create_optimized_dataloaders(
        data_path=config['data']['data_path'],
        config=config
    )
    
    # Run analyses
    print("\n📋 Running diagnostic analyses...")
    
    # 1. Analyze loss components
    loss_analysis = analyze_loss_components(model, loss_fn, train_loader, device, args.num_batches)
    
    # 2. Check gradient flow
    model.train()
    sample_batch = next(iter(train_loader))
    for key in ['static_params', 'signal', 'v_peak', 'v_peak_mask', 'target']:
        if key in sample_batch:
            sample_batch[key] = sample_batch[key].to(device)
    
    outputs = model(sample_batch['signal'], sample_batch['static_params'])
    total_loss, _ = loss_fn(outputs, sample_batch)
    gradient_analysis = check_gradient_flow(model, total_loss)
    
    # 3. Analyze model outputs
    output_analysis = analyze_model_outputs(model, train_loader, device, args.num_batches)
    
    # 4. Generate recommendations
    recommendations = generate_recommendations(loss_analysis, gradient_analysis, output_analysis, config)
    
    # 5. Create diagnostic plots
    create_diagnostic_plots(loss_analysis, output_analysis, output_dir)
    
    # Print results
    print("\n📊 DIAGNOSTIC RESULTS")
    print("=" * 50)
    
    print("\n🔍 Loss Component Analysis:")
    for component, stats in loss_analysis.items():
        print(f"  {component:20s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    print(f"\n🔍 Gradient Flow Analysis:")
    print(f"  Parameters with gradients: {gradient_analysis['has_gradients']}")
    print(f"  Zero gradients: {gradient_analysis['zero_gradients']}")
    print(f"  NaN gradients: {gradient_analysis['nan_gradients']}")
    print(f"  Large gradients (>10): {gradient_analysis['large_gradients']}")
    if gradient_analysis['gradient_norms']:
        print(f"  Average gradient norm: {np.mean(gradient_analysis['gradient_norms']):.4f}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Save results
    results = {
        'loss_analysis': loss_analysis,
        'gradient_analysis': gradient_analysis,
        'output_analysis': output_analysis,
        'recommendations': recommendations,
        'config_used': config
    }
    
    import json
    with open(output_dir / 'diagnostic_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"\n📁 Full diagnostic results saved to: {output_dir}")
    print("\n🚀 Run training with fixed configuration:")
    print(f"python run_training.py --config {args.config}")


if __name__ == "__main__":
    main()