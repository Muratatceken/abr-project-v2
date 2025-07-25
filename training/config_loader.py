#!/usr/bin/env python3
"""
Configuration Loader for Enhanced ABR Training

Utilities for loading and managing YAML-based configuration files
with validation and environment variable support.

Author: AI Assistant
Date: January 2025
"""

import yaml
import os
import argparse
from typing import Dict, Any, Optional, Union
from pathlib import Path


class ConfigLoader:
    """
    Configuration loader with YAML support and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Process environment variables
        self.config = self._process_env_vars(self.config)
        
        return self.config
    
    def _process_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process environment variables in configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Processed configuration with environment variables resolved
        """
        def process_value(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                # Extract environment variable name
                env_var = value[2:-1]
                default_value = None
                
                # Check for default value syntax: ${VAR:default}
                if ':' in env_var:
                    env_var, default_value = env_var.split(':', 1)
                
                # Get environment variable value
                env_value = os.getenv(env_var, default_value)
                
                # Try to convert to appropriate type
                if env_value is not None:
                    # Try to convert to int
                    try:
                        return int(env_value)
                    except ValueError:
                        pass
                    
                    # Try to convert to float
                    try:
                        return float(env_value)
                    except ValueError:
                        pass
                    
                    # Try to convert to bool
                    if env_value.lower() in ('true', 'false'):
                        return env_value.lower() == 'true'
                    
                    # Return as string
                    return env_value
                
                return None
            
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            else:
                return value
        
        return process_value(config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation like 'model.base_channels')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
        """
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
    
    def save_config(self, output_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def validate_config(self) -> bool:
        """
        Validate configuration for required fields and consistency.
        
        Returns:
            True if configuration is valid
        """
        required_fields = [
            'model.input_channels',
            'model.static_dim',
            'model.num_classes',
            'training.batch_size',
            'training.learning_rate',
            'training.num_epochs',
            'data.data_path'
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                print(f"Missing required configuration field: {field}")
                return False
        
        # Validate data types and ranges
        validations = [
            ('training.batch_size', int, lambda x: x > 0),
            ('training.learning_rate', (int, float), lambda x: x > 0),
            ('training.num_epochs', int, lambda x: x > 0),
            ('model.num_classes', int, lambda x: x > 0),
            ('data.val_split', (int, float), lambda x: 0 < x < 1),
        ]
        
        for field, expected_type, validator in validations:
            value = self.get(field)
            if value is not None:
                if not isinstance(value, expected_type):
                    print(f"Invalid type for {field}: expected {expected_type}, got {type(value)}")
                    return False
                if not validator(value):
                    print(f"Invalid value for {field}: {value}")
                    return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def flatten(self, separator: str = '.') -> Dict[str, Any]:
        """
        Flatten nested configuration dictionary.
        
        Args:
            separator: Separator for nested keys
            
        Returns:
            Flattened configuration dictionary
        """
        def _flatten(obj, parent_key='', sep='.'):
            items = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    items.extend(_flatten(v, new_key, sep=sep).items())
            else:
                return {parent_key: obj}
            return dict(items)
        
        return _flatten(self.config, sep=separator)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    def deep_merge(base_dict, update_dict):
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    result = {}
    for config in configs:
        result = deep_merge(result, config)
    
    return result


def create_config_from_args(args: argparse.Namespace, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create configuration dictionary from command line arguments.
    
    Args:
        args: Parsed command line arguments
        base_config: Base configuration to update
        
    Returns:
        Configuration dictionary
    """
    if base_config is None:
        base_config = {}
    
    # Map command line arguments to configuration structure
    arg_mapping = {
        # Data settings
        'data_path': 'data.data_path',
        'valid_peaks_only': 'data.valid_peaks_only',
        'val_split': 'data.val_split',
        'augment': 'data.augment',
        'cfg_dropout_prob': 'data.cfg_dropout_prob',
        
        # Model settings
        'base_channels': 'model.base_channels',
        'n_levels': 'model.n_levels',
        'num_transformer_layers': 'model.num_transformer_layers',
        'use_cross_attention': 'model.use_cross_attention',
        'film_dropout': 'model.film_dropout',
        'num_classes': 'model.num_classes',
        
        # Training settings
        'batch_size': 'training.batch_size',
        'learning_rate': 'training.learning_rate',
        'num_epochs': 'training.num_epochs',
        'weight_decay': 'training.weight_decay',
        'use_amp': 'training.use_amp',
        'patience': 'training.patience',
        'num_workers': 'training.num_workers',
        
        # Loss settings
        'use_focal_loss': 'loss.use_focal_loss',
        'use_class_weights': 'loss.use_class_weights',
        
        # Sampling settings
        'use_balanced_sampler': 'training.use_balanced_sampler',
        
        # Logging settings
        'output_dir': 'logging.output_dir',
        'use_wandb': 'logging.use_wandb',
        'wandb_project': 'logging.wandb_project',
        'experiment_name': 'experiment.name',
        
        # System settings
        'random_seed': 'experiment.random_seed'
    }
    
    # Create configuration loader and update with args
    config_loader = ConfigLoader()
    config_loader.config = base_config
    
    # Update configuration with command line arguments
    for arg_name, config_key in arg_mapping.items():
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is not None:
                config_loader.set(config_key, value)
    
    return config_loader.to_dict()


def load_config_with_overrides(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    args: Optional[argparse.Namespace] = None
) -> Dict[str, Any]:
    """
    Load configuration with overrides from multiple sources.
    
    Args:
        config_path: Path to base configuration file
        overrides: Dictionary of configuration overrides
        args: Command line arguments to override configuration
        
    Returns:
        Final configuration dictionary
    """
    # Load base configuration
    config_loader = ConfigLoader(config_path)
    base_config = config_loader.to_dict()
    
    # Apply overrides
    if overrides:
        config_loader.update(overrides)
    
    # Apply command line arguments
    if args:
        arg_config = create_config_from_args(args, config_loader.to_dict())
        config_loader.config = arg_config
    
    # Validate final configuration
    if not config_loader.validate_config():
        raise ValueError("Configuration validation failed")
    
    return config_loader.to_dict()


# Example usage and utility functions
def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration dictionary.
    
    Returns:
        Default configuration
    """
    return {
        'experiment': {
            'name': 'abr_training',
            'random_seed': 42
        },
        'model': {
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
            'dropout': 0.1,
            'use_cfg': True
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'use_amp': True,
            'patience': 15,
            'num_workers': 4
        },
        'data': {
            'data_path': 'data/processed/ultimate_dataset.pkl',
            'val_split': 0.2,
            'augment': True,
            'cfg_dropout_prob': 0.1
        },
        'loss': {
            'use_class_weights': True,
            'use_focal_loss': False
        },
        'logging': {
            'use_wandb': False,
            'use_tensorboard': True
        }
    }


if __name__ == '__main__':
    # Example usage
    config_loader = ConfigLoader('training/config.yaml')
    print("Loaded configuration:")
    print(yaml.dump(config_loader.to_dict(), default_flow_style=False, indent=2))
    
    # Test flattening
    flat_config = config_loader.flatten()
    print("\nFlattened configuration:")
    for key, value in flat_config.items():
        print(f"{key}: {value}")
    
    # Test validation
    is_valid = config_loader.validate_config()
    print(f"\nConfiguration is valid: {is_valid}") 