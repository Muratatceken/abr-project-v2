#!/usr/bin/env python3
"""
Configuration Loader for ABR Training Pipeline

Handles loading, validation, and management of training configurations
with support for environment variable overrides and configuration merging.

Author: AI Assistant
Date: January 2025
"""

import os
import yaml
import torch
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Structured configuration for ABR training."""
    
    # Project settings
    project_name: str = "ABR_HierarchicalUNet_Diffusion"
    experiment_name: str = "default_experiment"
    
    # Data settings
    dataset_path: str = "data/processed/ultimate_dataset_with_clinical_thresholds.pkl"
    signal_length: int = 200
    static_dim: int = 4
    n_classes: int = 5
    batch_size: int = 32
    num_workers: int = 4
    
    # Model architecture
    base_channels: int = 64
    n_levels: int = 4
    dropout: float = 0.1
    
    # Training parameters
    epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    
    # Loss weights
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'signal': 1.0,
        'peak_exist': 0.5,
        'peak_latency': 1.0,
        'peak_amplitude': 1.0,
        'classification': 1.0,
        'threshold': 0.8
    })
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"
    
    # Hardware
    device: str = "auto"
    mixed_precision: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to configuration file
        overrides: Dictionary of configuration overrides
        
    Returns:
        Loaded configuration as OmegaConf DictConfig
    """
    try:
        # Load base configuration
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to OmegaConf for advanced features
        config = OmegaConf.create(config_dict)
        
        # Apply environment variable overrides
        config = _apply_env_overrides(config)
        
        # Apply manual overrides
        if overrides:
            config = OmegaConf.merge(config, overrides)
        
        # Validate configuration
        config = _validate_config(config)
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def save_config(config: DictConfig, save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert to regular dict and save
        config_dict = OmegaConf.to_yaml(config)
        with open(save_path, 'w') as f:
            f.write(config_dict)
            
        logger.info(f"Configuration saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {save_path}: {e}")
        raise


def _apply_env_overrides(config: DictConfig) -> DictConfig:
    """Apply environment variable overrides to configuration."""
    
    # Common environment variable mappings
    env_mappings = {
        'ABR_DATA_PATH': 'data.dataset_path',
        'ABR_BATCH_SIZE': 'data.dataloader.batch_size',
        'ABR_LEARNING_RATE': 'training.optimizer.learning_rate',
        'ABR_EPOCHS': 'training.epochs',
        'ABR_DEVICE': 'hardware.device',
        'ABR_CHECKPOINT_DIR': 'paths.checkpoint_dir',
        'ABR_LOG_DIR': 'paths.log_dir',
        'ABR_MIXED_PRECISION': 'hardware.mixed_precision',
        'ABR_NUM_WORKERS': 'data.dataloader.num_workers'
    }
    
    for env_var, config_path in env_mappings.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            
            # Type conversion
            if config_path.endswith(('batch_size', 'epochs', 'num_workers')):
                value = int(value)
            elif config_path.endswith('learning_rate'):
                value = float(value)
            elif config_path.endswith('mixed_precision'):
                value = value.lower() in ('true', '1', 'yes')
            
            # Set nested configuration value
            OmegaConf.set(config, config_path, value)
            logger.info(f"Applied environment override: {env_var} -> {config_path} = {value}")
    
    return config


def _validate_config(config: DictConfig) -> DictConfig:
    """Validate and post-process configuration."""
    
    # Ensure required sections exist
    required_sections = ['data', 'model', 'training', 'paths']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Fix type conversions for numeric values that might be strings
    if 'training' in config and 'optimizer' in config.training:
        opt_config = config.training.optimizer
        # Convert string scientific notation to float
        if 'learning_rate' in opt_config:
            opt_config.learning_rate = float(opt_config.learning_rate)
        if 'weight_decay' in opt_config:
            opt_config.weight_decay = float(opt_config.weight_decay)
        if 'eps' in opt_config:
            opt_config.eps = float(opt_config.eps)
    
    if 'training' in config and 'scheduler' in config.training:
        sched_config = config.training.scheduler
        if 'eta_min' in sched_config:
            sched_config.eta_min = float(sched_config.eta_min)
    
    if 'diffusion' in config and 'noise_schedule' in config.diffusion:
        noise_config = config.diffusion.noise_schedule
        if 'beta_start' in noise_config:
            noise_config.beta_start = float(noise_config.beta_start)
        if 'beta_end' in noise_config:
            noise_config.beta_end = float(noise_config.beta_end)
    
    if 'training' in config and 'early_stopping' in config.training:
        es_config = config.training.early_stopping
        if 'min_delta' in es_config:
            es_config.min_delta = float(es_config.min_delta)
    
    # Auto-detect device if needed
    if config.hardware.device == "auto":
        config.hardware.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-detected device: {config.hardware.device}")
    
    # Ensure paths are absolute
    for path_key in ['checkpoint_dir', 'log_dir', 'output_dir']:
        if path_key in config.paths:
            path_val = config.paths[path_key]
            if not os.path.isabs(path_val):
                config.paths[path_key] = os.path.abspath(path_val)
    
    # Validate data paths
    if not os.path.exists(config.data.dataset_path):
        logger.warning(f"Dataset path does not exist: {config.data.dataset_path}")
    
    # Ensure output directories exist
    for path_key in ['checkpoint_dir', 'log_dir', 'output_dir']:
        if path_key in config.paths:
            os.makedirs(config.paths[path_key], exist_ok=True)
    
    # Validate batch size for multi-GPU training
    if hasattr(config.hardware, 'multi_gpu') and config.hardware.multi_gpu:
        gpu_count = torch.cuda.device_count()
        if config.data.dataloader.batch_size % gpu_count != 0:
            logger.warning(
                f"Batch size {config.data.dataloader.batch_size} is not divisible by "
                f"GPU count {gpu_count}. This may cause issues with DDP."
            )
    
    # Validate loss weights
    if 'loss' in config and 'weights' in config.loss:
        required_loss_keys = ['diffusion', 'peak_exist', 'peak_latency', 'peak_amplitude', 
                            'classification', 'threshold']
        for key in required_loss_keys:
            if key not in config.loss.weights:
                logger.warning(f"Missing loss weight for: {key}")
    
    return config


def create_experiment_config(
    base_config_path: str,
    experiment_name: str,
    overrides: Dict[str, Any]
) -> DictConfig:
    """
    Create experiment-specific configuration.
    
    Args:
        base_config_path: Path to base configuration
        experiment_name: Name of the experiment
        overrides: Experiment-specific overrides
        
    Returns:
        Experiment configuration
    """
    # Load base config
    config = load_config(base_config_path)
    
    # Set experiment name
    config.project.experiment_name = experiment_name
    
    # Apply experiment overrides
    config = OmegaConf.merge(config, overrides)
    
    # Create experiment-specific paths
    exp_checkpoint_dir = os.path.join(config.paths.checkpoint_dir, experiment_name)
    exp_log_dir = os.path.join(config.paths.log_dir, experiment_name)
    exp_output_dir = os.path.join(config.paths.output_dir, experiment_name)
    
    config.paths.checkpoint_dir = exp_checkpoint_dir
    config.paths.log_dir = exp_log_dir
    config.paths.output_dir = exp_output_dir
    
    # Ensure directories exist
    for path in [exp_checkpoint_dir, exp_log_dir, exp_output_dir]:
        os.makedirs(path, exist_ok=True)
    
    return config


def get_config_hash(config: DictConfig) -> str:
    """
    Generate a hash for configuration to track experiments.
    
    Args:
        config: Configuration to hash
        
    Returns:
        Configuration hash string
    """
    import hashlib
    import json
    
    # Convert to dict and sort for consistent hashing
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_str = json.dumps(config_dict, sort_keys=True)
    
    # Generate hash
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return config_hash


def print_config(config: DictConfig, title: str = "Configuration") -> None:
    """
    Pretty print configuration.
    
    Args:
        config: Configuration to print
        title: Title for the configuration display
    """
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(OmegaConf.to_yaml(config))
    print(f"{'='*60}\n")


# Utility functions for backwards compatibility
def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration as regular dict."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries."""
    base_omega = OmegaConf.create(base_config)
    override_omega = OmegaConf.create(override_config)
    merged = OmegaConf.merge(base_omega, override_omega)
    return OmegaConf.to_container(merged, resolve=True)