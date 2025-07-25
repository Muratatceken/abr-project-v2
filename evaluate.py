#!/usr/bin/env python3
"""
Standalone Comprehensive Evaluation Script for ABR Models

Usage:
    python evaluate.py --checkpoint model.pth --split test
    python evaluate.py --checkpoint model.pth --data_path data.pkl --config eval_config.yaml

Author: AI Assistant
Date: January 2025
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
import json
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from evaluation.comprehensive_eval import ABRComprehensiveEvaluator, create_evaluation_config
from training.dataset import ABRDataset, collate_fn
from models.hierarchical_unet import ProfessionalHierarchicalUNet
from diffusion.sampling import DDIMSampler
from diffusion.schedule import get_noise_schedule


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load ABR model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint if available
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        # Default configuration
        model_config = {
            'input_channels': 1,
            'static_dim': 4,
            'base_channels': 64,
            'n_levels': 4,
            'sequence_length': 200,
            'signal_length': 200,
            'num_classes': 5,
            'num_transformer_layers': 3,
            'use_cross_attention': True,
            'use_positional_encoding': True,
            'film_dropout': 0.15,
            'use_cfg': False  # Disable CFG for evaluation
        }
    
    # Create model
    model = ProfessionalHierarchicalUNet(**model_config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def load_dataset(data_path: str, split: str = 'test', config: Dict[str, Any] = None) -> DataLoader:
    """
    Load evaluation dataset.
    
    Args:
        data_path: Path to dataset file
        split: Dataset split ('test', 'val', or 'all')
        config: Dataset configuration
        
    Returns:
        DataLoader for evaluation
    """
    print(f"Loading dataset from: {data_path}")
    
    # Load data
    import pickle
    with open(data_path, 'rb') as f:
        dataset_dict = pickle.load(f)
    
    data = dataset_dict['data']
    print(f"Total samples in dataset: {len(data)}")
    
    # Split data if requested
    if split == 'test':
        # Use last 20% as test set
        test_size = int(0.2 * len(data))
        data = data[-test_size:]
        print(f"Using test split: {len(data)} samples")
    elif split == 'val':
        # Use middle 20% as validation set
        val_start = int(0.6 * len(data))
        val_end = int(0.8 * len(data))
        data = data[val_start:val_end]
        print(f"Using validation split: {len(data)} samples")
    elif split == 'all':
        print(f"Using all data: {len(data)} samples")
    
    # Create dataset
    dataset = ABRDataset(data, mode='eval', augment=False)
    
    # Create dataloader
    batch_size = config.get('batch_size', 32) if config else 32
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Single process for evaluation
        collate_fn=collate_fn
    )
    
    print(f"Created dataloader with {len(dataloader)} batches")
    
    return dataloader


def run_comprehensive_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    evaluator: ABRComprehensiveEvaluator,
    device: torch.device,
    use_ddim: bool = False,
    save_visualizations: bool = True,
    args = None
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on the model with enhanced options.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader for evaluation
        evaluator: Comprehensive evaluator instance
        device: Device for computation
        use_ddim: Whether to use DDIM sampling for evaluation
        save_visualizations: Whether to save diagnostic visualizations
        args: Command line arguments for enhanced features
        
    Returns:
        Dictionary of evaluation results
    """
    print("\n🔬 Starting Enhanced Comprehensive Evaluation...")
    print("="*60)
    
    # Parse enhanced options
    no_visuals = getattr(args, 'no_visuals', False)
    only_clinical_flags = getattr(args, 'only_clinical_flags', False)
    bootstrap_ci = getattr(args, 'bootstrap_ci', False)
    save_json_only = getattr(args, 'save_json_only', False)
    limit_samples = getattr(args, 'limit_samples', None)
    diagnostic_cards = getattr(args, 'diagnostic_cards', False)
    quantile_analysis = getattr(args, 'quantile_analysis', False)
    
    # Update evaluator config based on CLI options
    if bootstrap_ci:
        evaluator.config['bootstrap'] = {
            'enabled': True,
            'n_samples': 500,
            'ci_percentile': 95
        }
    
    if no_visuals:
        save_visualizations = False
        print("📊 Visual plotting disabled for faster evaluation")
    
    if only_clinical_flags:
        print("⚠️  Clinical flags only mode - skipping detailed metrics")
    
    if limit_samples:
        print(f"🔍 Debug mode - limiting to {limit_samples} samples")
    
    # Setup DDIM sampler if requested
    ddim_sampler = None
    if use_ddim:
        print("🎯 Using DDIM sampling for realistic evaluation")
        noise_schedule = get_noise_schedule('cosine', num_timesteps=1000)
        ddim_sampler = DDIMSampler(noise_schedule, eta=0.0)
    
    model.eval()
    total_batches = len(dataloader)
    processed_samples = 0
    
    # Storage for clinical flags only mode
    clinical_flags_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}/{total_batches}", end='\r')
            
            # Move batch to device
            for key in ['static_params', 'signal', 'v_peak', 'v_peak_mask', 'target']:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            # Check sample limit
            if limit_samples and processed_samples >= limit_samples:
                print(f"\n✅ Reached sample limit: {limit_samples} samples")
                break
            
            # Generate model outputs
            if use_ddim and ddim_sampler:
                # Use DDIM sampling for realistic evaluation
                batch_size = batch['signal'].size(0)
                signal_length = batch['signal'].size(-1)
                
                generated_signals = ddim_sampler.sample(
                    model=model,
                    shape=(batch_size, 1, signal_length),
                    static_params=batch['static_params'],
                    device=device,
                    num_steps=50,
                    progress=False
                )
                
                # Get model outputs from generated signals
                outputs = model(generated_signals, batch['static_params'])
                
                # Replace reconstruction with generated signals for evaluation
                outputs['recon'] = generated_signals
            else:
                # Direct forward pass
                outputs = model(batch['signal'], batch['static_params'])
            
            if only_clinical_flags:
                # Only compute and store clinical failure flags
                if all(key in outputs for key in ['peak', 'class', 'threshold']):
                    failures = evaluator.compute_failure_modes(
                        outputs['peak'][0] if len(outputs['peak']) > 0 else torch.zeros(batch['signal'].size(0)),
                        batch['v_peak_mask'].any(dim=1).float() if 'v_peak_mask' in batch else torch.zeros(batch['signal'].size(0)),
                        outputs['threshold'],
                        batch.get('threshold', batch['static_params'][:, 1] * 100),
                        torch.argmax(outputs['class'], dim=1),
                        batch['target']
                    )
                    
                    # Store flagged patient IDs
                    for sample_idx in range(batch['signal'].size(0)):
                        patient_id = batch.get('patient_ids', [f"B{batch_idx}_S{sample_idx}"])[sample_idx] if 'patient_ids' in batch else f"B{batch_idx}_S{sample_idx}"
                        clinical_flags_data.append({
                            'patient_id': patient_id,
                            'batch_idx': batch_idx,
                            'sample_idx': sample_idx,
                            'failures': failures
                        })
            else:
                # Full evaluation
                batch_results = evaluator.evaluate_batch(batch, outputs, batch_idx)
                
                # Enhanced visualizations
                if save_visualizations and not no_visuals and batch_idx < 3:
                    # Standard visualizations
                    visualizations = evaluator.create_batch_diagnostics(
                        batch, outputs, batch_idx, n_samples=3
                    )
                    
                    # Save standard visualizations
                    for viz_name, viz_data in visualizations.items():
                        viz_path = evaluator.save_dir / "figures" / f"batch_{batch_idx}_{viz_name}.png"
                        with open(viz_path, 'wb') as f:
                            f.write(viz_data)
                    
                    # Diagnostic cards
                    if diagnostic_cards:
                        cards = evaluator.create_diagnostic_cards(batch, outputs, batch_idx, n_samples=2)
                        for card_name, card_data in cards.items():
                            card_path = evaluator.save_dir / "figures" / f"{card_name}.png"
                            with open(card_path, 'wb') as f:
                                f.write(card_data)
                    
                    # Quantile analysis
                    if quantile_analysis:
                        quantile_viz = evaluator.create_quantile_error_visualizations(batch, outputs)
                        for viz_name, viz_data in quantile_viz.items():
                            viz_path = evaluator.save_dir / "figures" / f"batch_{batch_idx}_{viz_name}.png"
                            with open(viz_path, 'wb') as f:
                                f.write(viz_data)
            
            processed_samples += batch['signal'].size(0)
    
    print(f"\n✅ Evaluation completed: {processed_samples} samples processed")
    
    if only_clinical_flags:
        # Print clinical flags summary
        print("\n⚠️  CLINICAL FLAGS SUMMARY")
        print("="*50)
        
        total_flagged = 0
        flag_counts = {}
        
        for patient_data in clinical_flags_data:
            failures = patient_data['failures']
            patient_flagged = False
            
            for flag_type, count in failures.items():
                if count > 0:
                    if flag_type not in flag_counts:
                        flag_counts[flag_type] = 0
                    flag_counts[flag_type] += count
                    if not patient_flagged:
                        print(f"🚨 Patient {patient_data['patient_id']}: {flag_type}")
                        patient_flagged = True
            
            if patient_flagged:
                total_flagged += 1
        
        print(f"\nTotal patients with clinical flags: {total_flagged}/{len(clinical_flags_data)}")
        for flag_type, count in flag_counts.items():
            print(f"  {flag_type}: {count} cases")
        
        return {'clinical_flags': clinical_flags_data, 'flag_summary': flag_counts}
    
    else:
        # Compute aggregate metrics
        print("📊 Computing aggregate metrics...")
        aggregate_metrics = evaluator.compute_aggregate_metrics()
        
        # Generate summary table
        if not save_json_only:
            print("📋 Generating summary table...")
            summary_table_path = evaluator.create_summary_table()
            if summary_table_path:
                print(f"📊 Summary table saved: {summary_table_path}")
        
        return aggregate_metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Comprehensive ABR Model Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint file'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_path', type=str, default='data/processed/ultimate_dataset.pkl',
        help='Path to dataset file'
    )
    parser.add_argument(
        '--split', type=str, default='test', choices=['test', 'val', 'all'],
        help='Dataset split to evaluate on'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to evaluation configuration YAML file'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--use_ddim', action='store_true',
        help='Use DDIM sampling for realistic evaluation'
    )
    parser.add_argument(
        '--no_visualizations', action='store_true',
        help='Skip saving diagnostic visualizations'
    )
    
    # Enhanced evaluation arguments
    parser.add_argument(
        '--no_visuals', action='store_true',
        help='Skip visual plotting for faster evaluation'
    )
    parser.add_argument(
        '--only_clinical_flags', action='store_true',
        help='Skip metrics computation and only report clinical failure flags'
    )
    parser.add_argument(
        '--bootstrap_ci', action='store_true',
        help='Enable bootstrap confidence intervals for all metrics'
    )
    parser.add_argument(
        '--save_json_only', action='store_true',
        help='Save only JSON results, skip CSV and visualizations'
    )
    parser.add_argument(
        '--limit_samples', type=int, default=None,
        help='Limit evaluation to specified number of samples (debug mode)'
    )
    parser.add_argument(
        '--diagnostic_cards', action='store_true',
        help='Generate multi-panel diagnostic cards for detailed analysis'
    )
    parser.add_argument(
        '--quantile_analysis', action='store_true',
        help='Generate quantile and error range visualizations'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir', type=str, default='outputs/evaluation',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--experiment_name', type=str, default='comprehensive_eval',
        help='Experiment name for output files'
    )
    
    # System arguments
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device to use (auto, cpu, cuda)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    # Logging arguments
    parser.add_argument(
        '--use_tensorboard', action='store_true',
        help='Log results to TensorBoard'
    )
    parser.add_argument(
        '--use_wandb', action='store_true',
        help='Log results to Weights & Biases'
    )
    parser.add_argument(
        '--wandb_project', type=str, default='abr-evaluation',
        help='W&B project name'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load evaluation configuration
    if args.config and Path(args.config).exists():
        print(f"Loading evaluation config from: {args.config}")
        with open(args.config, 'r') as f:
            eval_config = yaml.safe_load(f)
    else:
        print("Using default evaluation configuration")
        eval_config = create_evaluation_config()
    
    # Update config with command line arguments
    eval_config['batch_size'] = args.batch_size
    eval_config['save_dir'] = args.output_dir
    
    # Validate inputs
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        return 1
    
    if not Path(args.data_path).exists():
        print(f"❌ Dataset file not found: {args.data_path}")
        return 1
    
    try:
        # Load model
        model = load_model_from_checkpoint(args.checkpoint, device)
        
        # Load dataset
        dataloader = load_dataset(args.data_path, args.split, eval_config)
        
        # Create evaluator
        class_names = ["NORMAL", "NÖROPATİ", "SNİK", "TOTAL", "İTİK"]
        evaluator = ABRComprehensiveEvaluator(
            config=eval_config,
            class_names=class_names,
            save_dir=args.output_dir,
            device=device
        )
        
        # Initialize logging
        writer = None
        if args.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(args.output_dir) / "tensorboard"
                writer = SummaryWriter(log_dir)
                print(f"📊 TensorBoard logging enabled: {log_dir}")
            except ImportError:
                print("⚠️  TensorBoard not available")
        
        if args.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=f"{args.experiment_name}_{args.split}",
                    config={
                        'checkpoint': args.checkpoint,
                        'data_path': args.data_path,
                        'split': args.split,
                        'use_ddim': args.use_ddim,
                        **eval_config
                    }
                )
                print("📊 W&B logging enabled")
            except ImportError:
                print("⚠️  Weights & Biases not available")
        
        # Run evaluation
        results = run_comprehensive_evaluation(
            model=model,
            dataloader=dataloader,
            evaluator=evaluator,
            device=device,
            use_ddim=args.use_ddim,
            save_visualizations=not args.no_visualizations,
            args=args
        )
        
        # Save results
        print("\n💾 Saving evaluation results...")
        saved_files = evaluator.save_results(args.experiment_name)
        
        for file_type, file_path in saved_files.items():
            print(f"   {file_type}: {file_path}")
        
        # Log to external services
        if writer:
            evaluator.log_to_tensorboard(writer, 0)
            writer.close()
            print("📊 Results logged to TensorBoard")
        
        if args.use_wandb:
            evaluator.log_to_wandb(0)
            print("📊 Results logged to W&B")
        
        # Print summary
        evaluator.print_summary()
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 