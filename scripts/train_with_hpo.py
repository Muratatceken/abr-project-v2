#!/usr/bin/env python3
"""
Hyperparameter optimization script for ABR Transformer.

This script provides a command-line interface for running hyperparameter optimization
using Optuna with various sampling strategies and pruning algorithms.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import torch
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from train import main as train_main, load_config
from optimization.hyperparameter_optimization import (
    HPOConfig, HyperparameterSpace, HPOObjective, OptunaTuner, AutoML
)


def setup_logging(level: str = "INFO"):
    """Setup logging for HPO."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('hpo.log')
        ]
    )


def _extract_from_tensorboard_logs(config: Dict[str, Any]) -> float:
    """
    Extract validation loss from TensorBoard logs or checkpoint files.
    
    Args:
        config: Training configuration
        
    Returns:
        Best validation loss found, or inf if not found
    """
    try:
        # Try to read from TensorBoard event files
        import glob
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        log_dir = config.get('log_dir', 'logs')
        exp_name = config.get('exp_name', 'abr_transformer')
        tb_log_path = f"{log_dir}/{exp_name}"
        
        # Find event files
        event_files = glob.glob(f"{tb_log_path}/**/events.out.tfevents.*", recursive=True)
        
        if event_files:
            # Read the most recent event file
            latest_event_file = max(event_files, key=os.path.getmtime)
            
            ea = EventAccumulator(latest_event_file)
            ea.Reload()
            
            # Try to get validation loss
            scalar_keys = ea.Tags()['scalars']
            val_loss_keys = [k for k in scalar_keys if 'val' in k.lower() and 'loss' in k.lower()]
            
            if val_loss_keys:
                val_loss_events = ea.Scalars(val_loss_keys[0])
                if val_loss_events:
                    return min(event.value for event in val_loss_events)
        
        # Fallback: try to read from checkpoint metadata
        ckpt_dir = config.get('trainer', {}).get('ckpt_dir', 'checkpoints/abr_transformer')
        ckpt_files = glob.glob(f"{ckpt_dir}/*.yaml")
        
        if ckpt_files:
            latest_ckpt_meta = max(ckpt_files, key=os.path.getmtime)
            with open(latest_ckpt_meta, 'r') as f:
                ckpt_meta = yaml.safe_load(f)
                return ckpt_meta.get('val_loss', float('inf'))
                
    except Exception as e:
        logging.warning(f"Failed to extract metrics from logs: {e}")
    
    return float('inf')


def create_hpo_objective(base_config: Dict[str, Any], cv_folds: int = 3) -> HPOObjective:
    """
    Create HPO objective function.
    
    Args:
        base_config: Base training configuration
        cv_folds: Number of cross-validation folds
        
    Returns:
        HPO objective function
    """
    
    def train_fn(config: Dict[str, Any]):
        """Training function for HPO."""
        # Update configuration with HPO parameters
        updated_config = base_config.copy()
        updated_config.update(config)
        
        # Reduce epochs for faster HPO
        updated_config['trainer']['max_epochs'] = min(
            updated_config['trainer'].get('max_epochs', 100),
            config.get('max_epochs', 50)
        )
        
        # Disable expensive features during HPO
        updated_config['monitoring']['enabled'] = False
        updated_config['trainer']['sample_every_epochs'] = 999  # Disable sampling
        updated_config['viz']['plot_spectrogram'] = False
        
        # Save config temporarily
        temp_config_path = 'temp_hpo_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(updated_config, f)
        
        try:
            # Set up logging capture for metrics extraction
            import tempfile
            import json
            import re
            from io import StringIO
            import contextlib
            
            # Capture logging output to extract metrics
            log_capture = StringIO()
            
            # Run training with captured output
            original_argv = sys.argv
            sys.argv = ['train.py', '--config', temp_config_path]
            
            # Add logging handler to capture output
            logger = logging.getLogger()
            handler = logging.StreamHandler(log_capture)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            
            try:
                train_main()
            finally:
                logger.removeHandler(handler)
            
            # Extract metrics from captured logs
            log_content = log_capture.getvalue()
            
            # Parse validation loss from logs (look for patterns like "val(best) 0.1234")
            val_loss_pattern = r'val\(best\)\s+([\d\.e\-\+]+)'
            val_losses = re.findall(val_loss_pattern, log_content)
            
            # Parse training loss from logs (look for patterns like "train 0.1234")
            train_loss_pattern = r'train\s+([\d\.e\-\+]+)'
            train_losses = re.findall(train_loss_pattern, log_content)
            
            # Get the best (lowest) validation loss
            if val_losses:
                best_val_loss = min(float(loss) for loss in val_losses)
            else:
                # Fallback: try to read from TensorBoard logs or checkpoints
                best_val_loss = _extract_from_tensorboard_logs(updated_config)
            
            # Get final training loss
            final_train_loss = float(train_losses[-1]) if train_losses else float('inf')
            
            # Return actual metrics
            val_metrics = {'val_combined_score': best_val_loss}
            train_metrics = {'train_loss': final_train_loss}
            
            logging.info(f"HPO Trial Results - Val Loss: {best_val_loss:.6f}, Train Loss: {final_train_loss:.6f}")
            
            return val_metrics, train_metrics
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return {'val_combined_score': float('inf')}, {'train_loss': float('inf')}
        finally:
            sys.argv = original_argv
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    def eval_fn(model, config: Dict[str, Any]):
        """Evaluation function for HPO."""
        # Simplified evaluation - in practice would run proper validation
        return {'val_combined_score': 0.5}
    
    # Create hyperparameter space
    hyperparameter_space = HyperparameterSpace()
    hyperparameter_space.create_default_abr_space()
    
    return HPOObjective(
        train_fn=train_fn,
        eval_fn=eval_fn,
        hyperparameter_space=hyperparameter_space,
        base_config=base_config,
        cv_folds=cv_folds,
        use_early_stopping=True,
        max_epochs=50,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


def run_optuna_optimization(
    base_config: Dict[str, Any],
    n_trials: int = 100,
    timeout: Optional[int] = None,
    sampler: str = 'tpe',
    pruner: str = 'median',
    save_dir: str = 'hpo_results',
    cv_folds: int = 3
) -> Dict[str, Any]:
    """
    Run Optuna-based hyperparameter optimization.
    
    Args:
        base_config: Base training configuration
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        sampler: Sampling algorithm ('tpe', 'random', 'cmaes')
        pruner: Pruning algorithm ('median', 'hyperband', 'none')
        save_dir: Directory to save results
        cv_folds: Number of cross-validation folds
        
    Returns:
        Optimization results
    """
    # Create HPO configuration
    hpo_config = HPOConfig(
        study_name="abr_transformer_hpo",
        n_trials=n_trials,
        timeout=timeout,
        sampler=sampler,
        pruner=pruner,
        direction="minimize"  # Minimize validation loss
    )
    
    # Create objective function
    objective_fn = create_hpo_objective(base_config, cv_folds)
    
    # Create and run tuner
    tuner = OptunaTuner(hpo_config, objective_fn, save_dir)
    results = tuner.optimize()
    
    return results


def run_automl_pipeline(
    base_config: Dict[str, Any],
    n_trials: int = 100,
    cv_folds: int = 3,
    save_dir: str = 'automl_results'
) -> Dict[str, Any]:
    """
    Run complete AutoML pipeline.
    
    Args:
        base_config: Base training configuration
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        save_dir: Directory to save results
        
    Returns:
        AutoML results
    """
    # Create AutoML instance
    automl = AutoML(
        base_config=base_config,
        save_dir=save_dir
    )
    
    def train_fn(config: Dict[str, Any]):
        """Training function for AutoML."""
        return None, {'train_loss': 0.1}  # Simplified
    
    def eval_fn(model, config: Dict[str, Any]):
        """Evaluation function for AutoML."""
        return {'val_combined_score': 0.5}  # Simplified
    
    # Run AutoML pipeline
    results = automl.run_automl(
        train_fn=train_fn,
        eval_fn=eval_fn,
        n_trials=n_trials,
        cv_folds=cv_folds,
        n_jobs=1
    )
    
    return results


def create_search_space_config(save_path: str):
    """
    Create and save a sample search space configuration file.
    
    Args:
        save_path: Path to save the configuration
    """
    search_space_config = {
        # Model architecture parameters
        'd_model': {
            'type': 'int',
            'low': 128,
            'high': 512,
            'step': 64
        },
        'n_heads': {
            'type': 'int',
            'low': 4,
            'high': 16,
            'step': 2
        },
        'n_layers': {
            'type': 'int',
            'low': 4,
            'high': 12,
            'step': 2
        },
        'dropout': {
            'type': 'float',
            'low': 0.0,
            'high': 0.5,
            'step': 0.1
        },
        
        # Training parameters
        'learning_rate': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e-2,
            'log': True
        },
        'batch_size': {
            'type': 'int',
            'low': 8,
            'high': 64,
            'step': 8
        },
        'weight_decay': {
            'type': 'float',
            'low': 1e-6,
            'high': 1e-2,
            'log': True
        },
        
        # Loss parameters
        'use_focal_loss': {
            'type': 'categorical',
            'choices': [True, False]
        },
        'focal_alpha': {
            'type': 'float',
            'low': 0.1,
            'high': 0.9,
            'step': 0.1
        },
        'focal_gamma': {
            'type': 'float',
            'low': 0.5,
            'high': 5.0,
            'step': 0.5
        },
        
        # Augmentation parameters
        'mixup_prob': {
            'type': 'float',
            'low': 0.0,
            'high': 0.5,
            'step': 0.1
        },
        'cutmix_prob': {
            'type': 'float',
            'low': 0.0,
            'high': 0.5,
            'step': 0.1
        }
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(search_space_config, f, default_flow_style=False)
    
    print(f"Sample search space configuration saved to: {save_path}")


def analyze_hpo_results(results_dir: str):
    """
    Analyze and visualize HPO results.
    
    Args:
        results_dir: Directory containing HPO results
    """
    results_path = Path(results_dir) / 'hpo_results.json'
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    try:
        import json
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print("="*60)
        print("HPO RESULTS SUMMARY")
        print("="*60)
        
        print(f"Study Name: {results['study_name']}")
        print(f"Number of Trials: {results['n_trials']}")
        print(f"Optimization Time: {results['optimization_time']:.2f} seconds")
        print(f"Best Value: {results['best_value']:.6f}")
        
        print("\nBest Parameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        print("\nParameter Importance:")
        importance = results.get('parameter_importance', {})
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {imp:.4f}")
        
        # Try to create visualizations
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Optimization history
            history = results['optimization_history']
            if history:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(history)
                plt.title('Optimization History')
                plt.xlabel('Trial')
                plt.ylabel('Objective Value')
                plt.grid(True, alpha=0.3)
                
                # Parameter importance
                if importance:
                    plt.subplot(1, 2, 2)
                    params = list(importance.keys())
                    importances = list(importance.values())
                    
                    plt.barh(params, importances)
                    plt.title('Parameter Importance')
                    plt.xlabel('Importance')
                    plt.tight_layout()
                
                plot_path = Path(results_dir) / 'hpo_analysis.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"\nVisualization saved to: {plot_path}")
                
        except ImportError:
            print("\nMatplotlib not available for visualization")
        except Exception as e:
            print(f"\nVisualization failed: {e}")
            
    except Exception as e:
        print(f"Failed to analyze results: {e}")


def main():
    """Main HPO script."""
    parser = argparse.ArgumentParser(description="ABR Transformer Hyperparameter Optimization")
    parser.add_argument("--config", type=str, default="configs/train.yaml", 
                       help="Base training configuration file")
    parser.add_argument("--mode", type=str, choices=['hpo', 'automl', 'create_search_space', 'analyze'], 
                       default='hpo', help="Operation mode")
    
    # HPO parameters
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--sampler", type=str, choices=['tpe', 'random', 'cmaes'], 
                       default='tpe', help="Sampling algorithm")
    parser.add_argument("--pruner", type=str, choices=['median', 'hyperband', 'none'], 
                       default='median', help="Pruning algorithm")
    parser.add_argument("--cv_folds", type=int, default=3, help="Number of cross-validation folds")
    parser.add_argument("--save_dir", type=str, default="hpo_results", help="Results save directory")
    
    # Analysis parameters
    parser.add_argument("--results_dir", type=str, default="hpo_results", 
                       help="Directory containing HPO results for analysis")
    
    # Search space parameters
    parser.add_argument("--search_space_path", type=str, default="search_space.yaml",
                       help="Path for search space configuration file")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if args.mode == 'create_search_space':
        create_search_space_config(args.search_space_path)
        return
    
    if args.mode == 'analyze':
        analyze_hpo_results(args.results_dir)
        return
    
    # Load base configuration
    try:
        base_config = load_config(args.config)
        logging.info(f"Loaded base configuration from: {args.config}")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.mode == 'hpo':
            logging.info("Starting hyperparameter optimization...")
            results = run_optuna_optimization(
                base_config=base_config,
                n_trials=args.n_trials,
                timeout=args.timeout,
                sampler=args.sampler,
                pruner=args.pruner,
                save_dir=args.save_dir,
                cv_folds=args.cv_folds
            )
            
            logging.info("Hyperparameter optimization completed!")
            logging.info(f"Best parameters: {results['best_params']}")
            logging.info(f"Best value: {results['best_value']}")
            
        elif args.mode == 'automl':
            logging.info("Starting AutoML pipeline...")
            results = run_automl_pipeline(
                base_config=base_config,
                n_trials=args.n_trials,
                cv_folds=args.cv_folds,
                save_dir=args.save_dir
            )
            
            logging.info("AutoML pipeline completed!")
            logging.info(f"Best configuration saved to: {args.save_dir}")
            
    except KeyboardInterrupt:
        logging.info("Optimization interrupted by user")
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()
