"""
Configuration loading and saving utilities.

This module provides functions to load and save configuration files
for the ABR signal generation model training and evaluation.
"""

import yaml
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, Any


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration as DictConfig
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML syntax
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to OmegaConf DictConfig for advanced features
        config = OmegaConf.create(config_dict)
        
        # Resolve any interpolations
        config = OmegaConf.to_container(config, resolve=True)
        config = OmegaConf.create(config)
        
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading config file {config_path}: {e}")


def save_config(config: Union[DictConfig, Dict[str, Any]], save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration file
        
    Raises:
        OSError: If unable to write to file
    """
    save_path = Path(save_path)
    
    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert DictConfig to regular dict if needed
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config
        
        with open(save_path, 'w') as f:
            yaml.safe_dump(config_dict, f, indent=2, default_flow_style=False)
            
    except Exception as e:
        raise OSError(f"Error saving config to {save_path}: {e}")


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = OmegaConf.merge(base_config, override_config)
    return merged


def validate_config(config: DictConfig) -> None:
    """
    Validate configuration contains required fields.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If required fields are missing
    """
    required_sections = ['model', 'data', 'training', 'optimization']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model section
    model_required = ['signal_length', 'static_dim', 'base_channels']
    for field in model_required:
        if field not in config.model:
            raise ValueError(f"Missing required field in model config: {field}")
    
    # Validate data section
    data_required = ['path', 'batch_size']
    for field in data_required:
        if field not in config.data:
            raise ValueError(f"Missing required field in data config: {field}")
    
    # Validate training section
    training_required = ['epochs', 'device']
    for field in training_required:
        if field not in config.training:
            raise ValueError(f"Missing required field in training config: {field}")
    
    # Validate optimization section
    opt_required = ['learning_rate']
    for field in opt_required:
        if field not in config.optimization:
            raise ValueError(f"Missing required field in optimization config: {field}")


def get_default_config() -> DictConfig:
    """
    Get default configuration.
    
    Returns:
        Default configuration
    """
    default_config = {
        'model': {
            'signal_length': 200,
            'static_dim': 4,
            'base_channels': 64,
            'n_levels': 4,
            'dropout': 0.1,
            's4_state_size': 64,
            'num_s4_layers': 2,
            'num_transformer_layers': 2,
            'num_heads': 8,
        },
        'data': {
            'path': 'data/processed/ultimate_dataset_with_clinical_thresholds.pkl',
            'batch_size': 32,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'num_workers': 4,
            'pin_memory': True,
        },
        'training': {
            'epochs': 100,
            'device': 'auto',
            'mixed_precision': True,
            'random_seed': 42,
        },
        'optimization': {
            'learning_rate': 0.0001,
            'weight_decay': 0.0001,
            'grad_clip_norm': 1.0,
            'accumulation_steps': 1,
            'warmup_epochs': 5,
            'ema_decay': 0.999,
        },
        'diffusion': {
            'schedule_type': 'cosine',
            'num_timesteps': 1000,
        },
        'logging': {
            'log_interval': 50,
            'val_interval': 1,
            'val_preview_samples': 8,
            'save_interval': 10,
        },
        'checkpoints': {
            'save_dir': 'checkpoints',
            'save_best': True,
            'early_stop_patience': 20,
        }
    }
    
    return OmegaConf.create(default_config)