#!/usr/bin/env python3
"""
Main entry point for the ABR CVAE training and evaluation pipeline.

This script provides a unified interface for training, evaluating, and running
inference with Conditional Variational Autoencoder models on ABR signals.

Usage:
    python main.py --mode train --config_path configs/default_config.json
    python main.py --mode evaluate --checkpoint_path checkpoints/best_model.pth
    python main.py --mode inference --checkpoint_path checkpoints/best_model.pth
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Project imports
from models.cvae import CVAE
from training.train import CVAETrainer
from training.dataset import ABRDataset
from utils.data_utils import (
    create_dataloaders, 
    get_dataset_info, 
    validate_data_config,
    create_inference_dataloader,
    generate_random_static_params
)
from evaluation.evaluate import CVAEEvaluator
# from utils.preprocessing import preprocess_and_save


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup logging configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO').upper())
    
    # Create logs directory
    log_file = log_config.get('log_file', 'logs/main.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")


def setup_device(config: Dict[str, Any]) -> torch.device:
    """
    Setup computation device.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        torch.device: Configured device
    """
    device_config = config.get('device', {})
    device_type = device_config.get('type', 'auto')
    
    if device_type == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_type == 'cuda':
        gpu_id = device_config.get('gpu_id', 0)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    
    # Set device properties
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = device_config.get('benchmark', True)
        if device_config.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
    
    logging.info(f"Device configured: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    return device


def setup_reproducibility(config: Dict[str, Any]) -> None:
    """
    Setup reproducibility settings.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    repro_config = config.get('reproducibility', {})
    seed = repro_config.get('seed', 42)
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    if repro_config.get('deterministic', False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"Reproducibility configured with seed: {seed}")


def create_output_directories(config: Dict[str, Any]) -> None:
    """
    Create output directories.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    output_dirs = [
        config['outputs']['base_dir'],
        config['outputs']['plots_dir'],
        config['outputs']['results_dir'],
        config['outputs']['models_dir'],
        config['outputs']['logs_dir'],
        config['checkpoints']['save_dir']
    ]
    
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created output directory: {directory}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logging.info(f"Configuration loaded from: {config_path}")
    return config


def create_model(config: Dict[str, Any], dataset_info: Dict[str, Any], device: torch.device) -> CVAE:
    """
    Create CVAE model based on configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        dataset_info (Dict[str, Any]): Dataset information
        device (torch.device): Device to place model on
        
    Returns:
        CVAE: Configured CVAE model
    """
    model_config = config['model']['architecture']
    
    model = CVAE(
        signal_length=dataset_info['signal_length'],
        static_dim=dataset_info['static_params_dim'],
        latent_dim=model_config['latent_dim'],
        predict_peaks=model_config['predict_peaks'],
        num_peaks=dataset_info.get('num_peaks', 6)
    ).to(device)
    
    # Initialize model weights
    init_config = config['model'].get('initialization', {})
    init_type = init_config.get('type', 'xavier_uniform')
    
    if init_type == 'xavier_uniform':
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=init_config.get('gain', 1.0))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model created:")
    logging.info(f"  Signal length: {dataset_info['signal_length']}")
    logging.info(f"  Static params dim: {dataset_info['static_params_dim']}")
    logging.info(f"  Latent dim: {model_config['latent_dim']}")
    logging.info(f"  Predict peaks: {model_config['predict_peaks']}")
    logging.info(f"  Total parameters: {total_params:,}")
    logging.info(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def create_optimizer(model: CVAE, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model (CVAE): Model to optimize
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        optim.Optimizer: Configured optimizer
    """
    opt_config = config['training']['optimizer']
    
    optimizer_type = opt_config.get('type', 'adam').lower()
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_config.get('learning_rate', 1e-3),
            weight_decay=opt_config.get('weight_decay', 1e-5),
            betas=opt_config.get('betas', [0.9, 0.999]),
            eps=opt_config.get('eps', 1e-8)
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_config.get('learning_rate', 1e-3),
            weight_decay=opt_config.get('weight_decay', 1e-5),
            betas=opt_config.get('betas', [0.9, 0.999]),
            eps=opt_config.get('eps', 1e-8)
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_config.get('learning_rate', 1e-3),
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.get('weight_decay', 1e-5)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    logging.info(f"Optimizer created: {optimizer_type}")
    logging.info(f"  Learning rate: {opt_config.get('learning_rate', 1e-3)}")
    logging.info(f"  Weight decay: {opt_config.get('weight_decay', 1e-5)}")
    
    return optimizer


def load_checkpoint(checkpoint_path: str, model: CVAE, optimizer: optim.Optimizer, device: torch.device) -> int:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (CVAE): Model to load weights into
        optimizer (optim.Optimizer): Optimizer to load state into
        device (torch.device): Device to load checkpoint on
        
    Returns:
        int: Starting epoch number
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get starting epoch
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    logging.info(f"Checkpoint loaded successfully")
    logging.info(f"  Epoch: {checkpoint.get('epoch', 0)}")
    logging.info(f"  Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    return start_epoch


def train_mode(config: Dict[str, Any], checkpoint_path: Optional[str] = None) -> None:
    """
    Run training mode.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        checkpoint_path (Optional[str]): Path to checkpoint for resuming training
    """
    logging.info("Starting training mode")
    
    # Setup device and reproducibility
    device = setup_device(config)
    setup_reproducibility(config)
    
    # Validate data configuration
    validate_data_config(config)
    
    # Get dataset information
    dataset_info = get_dataset_info(config)
    logging.info(f"Dataset info: {dataset_info}")
    
    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        config, 
        return_peaks=config['model']['architecture']['predict_peaks']
    )
    
    # Create model
    model = create_model(config, dataset_info, device)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Load checkpoint if specified
    start_epoch = 0
    if checkpoint_path:
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, device)
    
    # Create trainer configuration
    trainer_config = {
        **config['training'],
        'save_dir': config['checkpoints']['save_dir'],
        'log_interval': config['monitoring']['log_interval'],
        'save_interval': config['checkpoints']['save_frequency'],
        'device': device.type,
        'num_epochs': config['training']['epochs']
    }
    
    # Create trainer
    trainer = CVAETrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        config=trainer_config
    )
    
    # Load checkpoint state if resuming
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)
    
    # Start training
    logging.info(f"Starting training for {config['training']['epochs']} epochs")
    trainer.train(config['training']['epochs'])
    
    logging.info("Training completed successfully")


def evaluate_mode(config: Dict[str, Any], checkpoint_path: Optional[str] = None) -> None:
    """
    Run evaluation mode.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        checkpoint_path (Optional[str]): Path to checkpoint for evaluation
    """
    logging.info("Starting evaluation mode")
    
    # Use checkpoint path from config if not provided
    if checkpoint_path is None:
        checkpoint_path = config['evaluation']['checkpoint_path']
    
    # Create evaluation configuration
    eval_config = {
        'model': {
            'checkpoint_path': checkpoint_path,
            'device': config['device']['type']
        },
        'data': {
            'data_path': config['data']['processed_data_path'],
            'batch_size': config['data']['dataloader']['batch_size'],
            'num_workers': config['data']['dataloader']['num_workers']
        },
        'evaluation': {
            'compute_reconstruction_metrics': True,
            'compute_peak_metrics': config['model']['architecture']['predict_peaks'],
            'compute_advanced_metrics': True
        },
        'metrics': config['evaluation']['metrics'],
        'visualization': config['evaluation']['visualization'],
        'outputs': {
            'base_dir': os.path.join(config['outputs']['results_dir'], 'evaluation'),
            'plots_dir': os.path.join(config['outputs']['results_dir'], 'evaluation', 'plots'),
            'latent_space_dir': os.path.join(config['outputs']['results_dir'], 'evaluation', 'latent_space'),
            'generated_samples_dir': os.path.join(config['outputs']['results_dir'], 'evaluation', 'generated_samples'),
            'summary_file': os.path.join(config['outputs']['results_dir'], 'evaluation', 'summary.json')
        },
        'plotting': {
            'figure_size': [12, 8],
            'dpi': 300,
            'format': 'png',
            'style': 'seaborn-v0_8',
            'color_palette': 'husl'
        }
    }
    
    # Create evaluator
    evaluator = CVAEEvaluator(eval_config)
    
    # Run evaluation
    evaluator.run_evaluation()
    
    logging.info("Evaluation completed successfully")
    logging.info(f"Results saved to: {eval_config['outputs']['base_dir']}")


def inference_mode(config: Dict[str, Any], checkpoint_path: Optional[str] = None) -> None:
    """
    Run inference mode.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        checkpoint_path (Optional[str]): Path to checkpoint for inference
    """
    logging.info("Starting inference mode")
    
    # Setup device
    device = setup_device(config)
    
    # Use checkpoint path from config if not provided
    if checkpoint_path is None:
        checkpoint_path = config['inference']['checkpoint_path']
    
    # Get dataset information
    dataset_info = get_dataset_info(config)
    
    # Create model
    model = create_model(config, dataset_info, device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info(f"Model loaded from: {checkpoint_path}")
    
    # Generate inference parameters
    inference_config = config['inference']
    num_samples = inference_config['generation']['num_samples']
    
    # Generate random static parameters
    static_params = generate_random_static_params(config, num_samples)
    static_params = static_params.to(device)
    
    # Create output directory
    output_dir = inference_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples
    logging.info(f"Generating {num_samples} samples")
    
    generated_signals = []
    generated_peaks = []
    
    batch_size = inference_config['generation']['batch_size']
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_static = static_params[i:i+batch_size]
            
            # Generate samples
            if model.predict_peaks:
                batch_signals, batch_peaks = model.sample(batch_static, n_samples=1)
                generated_peaks.append(batch_peaks.cpu())
            else:
                batch_signals = model.sample(batch_static, n_samples=1)
            
            generated_signals.append(batch_signals.cpu())
    
    # Concatenate results
    generated_signals = torch.cat(generated_signals, dim=0)
    static_params = static_params.cpu()
    
    if model.predict_peaks:
        generated_peaks = torch.cat(generated_peaks, dim=0)
    
    # Save results
    save_format = inference_config['generation']['save_format']
    
    if save_format in ['signals', 'both']:
        signals_path = os.path.join(output_dir, 'generated_signals.pt')
        torch.save(generated_signals, signals_path)
        logging.info(f"Generated signals saved to: {signals_path}")
    
    if save_format in ['peaks', 'both'] and model.predict_peaks:
        peaks_path = os.path.join(output_dir, 'generated_peaks.pt')
        torch.save(generated_peaks, peaks_path)
        logging.info(f"Generated peaks saved to: {peaks_path}")
    
    # Save static parameters
    params_path = os.path.join(output_dir, 'static_parameters.pt')
    torch.save(static_params, params_path)
    logging.info(f"Static parameters saved to: {params_path}")
    
    # Save summary
    summary = {
        'num_samples': num_samples,
        'signal_shape': list(generated_signals.shape),
        'static_params_shape': list(static_params.shape),
        'model_config': config['model']['architecture'],
        'checkpoint_path': checkpoint_path,
        'generation_time': datetime.now().isoformat()
    }
    
    if model.predict_peaks:
        summary['peaks_shape'] = list(generated_peaks.shape)
    
    summary_path = os.path.join(output_dir, 'generation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info("Inference completed successfully")
    logging.info(f"Results saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='ABR CVAE Training and Evaluation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train a new model
    python main.py --mode train --config_path configs/default_config.json
    
    # Resume training from checkpoint
    python main.py --mode train --checkpoint_path checkpoints/epoch_050.pth
    
    # Evaluate a trained model
    python main.py --mode evaluate --checkpoint_path checkpoints/best_model.pth
    
    # Run inference
    python main.py --mode inference --checkpoint_path checkpoints/best_model.pth
        """
    )
    
    parser.add_argument(
        '--config_path',
        type=str,
        default='configs/default_config.json',
        help='Path to configuration file (default: configs/default_config.json)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'inference'],
        default='train',
        help='Mode to run: train, evaluate, or inference (default: train)'
    )
    
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to checkpoint file (optional, for resuming training or running evaluation/inference)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config_path)
        
        # Override logging level if verbose
        if args.verbose:
            config['logging']['level'] = 'DEBUG'
        
        # Setup logging
        setup_logging(config)
        
        # Create output directories
        create_output_directories(config)
        
        # Log startup information
        logging.info("="*60)
        logging.info("ABR CVAE Pipeline Starting")
        logging.info("="*60)
        logging.info(f"Mode: {args.mode}")
        logging.info(f"Configuration: {args.config_path}")
        logging.info(f"Checkpoint: {args.checkpoint_path or 'None'}")
        logging.info(f"Project: {config['project']['name']} v{config['project']['version']}")
        logging.info(f"Description: {config['project']['description']}")
        logging.info("="*60)
        
        # Run appropriate mode
        start_time = time.time()
        
        if args.mode == 'train':
            train_mode(config, args.checkpoint_path)
        elif args.mode == 'evaluate':
            evaluate_mode(config, args.checkpoint_path)
        elif args.mode == 'inference':
            inference_mode(config, args.checkpoint_path)
        
        # Log completion
        end_time = time.time()
        duration = end_time - start_time
        
        logging.info("="*60)
        logging.info(f"Pipeline completed successfully in {duration:.2f} seconds")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main() 