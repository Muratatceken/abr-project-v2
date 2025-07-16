"""
Data utilities for CVAE training and evaluation.

This module provides functions for creating dataloaders, managing data splits,
and handling data preprocessing workflows.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import logging
from pathlib import Path

from training.dataset import ABRDataset
from utils.preprocessing import load_and_preprocess_dataset


def setup_data_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary data directories.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    directories = [
        os.path.dirname(config['data']['processed_data_path']),
        os.path.dirname(config['data']['preprocessing']['scaler_path']),
        os.path.dirname(config['data']['preprocessing']['encoder_path']),
    ]
    
    for directory in directories:
        if directory:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")


def check_and_preprocess_data(config: Dict[str, Any]) -> str:
    """
    Check if processed data exists, and preprocess if necessary.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        str: Path to processed data file
    """
    processed_data_path = config['data']['processed_data_path']
    raw_data_path = config['data']['raw_data_path']
    
    # Create data directories
    setup_data_directories(config)
    
    # Check if processed data exists
    if os.path.exists(processed_data_path):
        logging.info(f"Found existing processed data: {processed_data_path}")
        return processed_data_path
    
    # Check if raw data exists
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
    
    logging.info(f"Processing raw data from: {raw_data_path}")
    
    # Preprocess data
    preprocessing_config = config['data']['preprocessing']
    
    try:
        # Load and preprocess the dataset
        data, scaler, onehot_encoder = load_and_preprocess_dataset(
            raw_data_path,
            verbose=preprocessing_config.get('verbose', True),
            save_transformers=preprocessing_config.get('save_transformers', True)
        )
        
        # Save processed data
        import joblib
        joblib.dump(data, processed_data_path)
        
        # Save transformers if paths are provided
        if preprocessing_config.get('scaler_path'):
            joblib.dump(scaler, preprocessing_config['scaler_path'])
        if preprocessing_config.get('encoder_path'):
            joblib.dump(onehot_encoder, preprocessing_config['encoder_path'])
        
        logging.info(f"Data preprocessing completed. Saved to: {processed_data_path}")
        return processed_data_path
        
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
        raise


def create_dataloaders(
    config: Dict[str, Any],
    return_peaks: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create training, validation, and test dataloaders.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        return_peaks (bool): Whether to return peak data
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders
    """
    # Get processed data path
    processed_data_path = check_and_preprocess_data(config)
    
    # Load dataset
    dataset = ABRDataset(
        data_path=processed_data_path,
        return_peaks=return_peaks
    )
    
    # Get dataloader configuration
    dataloader_config = config['data']['dataloader']
    
    # Calculate split sizes
    dataset_size = len(dataset)
    val_split = dataloader_config.get('val_split', 0.2)
    test_split = dataloader_config.get('test_split', 0.1)
    
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    # Create splits
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.get('reproducibility', {}).get('seed', 42))
    )
    
    # Create dataloaders
    batch_size = dataloader_config.get('batch_size', 32)
    num_workers = dataloader_config.get('num_workers', 4)
    pin_memory = dataloader_config.get('pin_memory', True)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=dataloader_config.get('shuffle', True),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=dataloader_config.get('drop_last', False)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # Log dataset information
    logging.info(f"Dataset splits created:")
    logging.info(f"  Total samples: {dataset_size}")
    logging.info(f"  Training samples: {train_size}")
    logging.info(f"  Validation samples: {val_size}")
    logging.info(f"  Test samples: {test_size}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Number of workers: {num_workers}")
    
    return train_dataloader, val_dataloader, test_dataloader


def get_dataset_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about the dataset.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Dict[str, Any]: Dataset information
    """
    processed_data_path = check_and_preprocess_data(config)
    
    # Load dataset
    dataset = ABRDataset(
        data_path=processed_data_path,
        return_peaks=config['model']['architecture'].get('predict_peaks', True)
    )
    
    # Get dataset information
    info = dataset.get_sample_info()
    
    # Add configuration information
    info.update({
        'processed_data_path': processed_data_path,
        'predict_peaks': config['model']['architecture'].get('predict_peaks', True),
        'batch_size': config['data']['dataloader'].get('batch_size', 32),
        'val_split': config['data']['dataloader'].get('val_split', 0.2),
        'test_split': config['data']['dataloader'].get('test_split', 0.1)
    })
    
    return info


def create_inference_dataloader(
    config: Dict[str, Any],
    static_params: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None
) -> DataLoader:
    """
    Create dataloader for inference.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        static_params (Optional[torch.Tensor]): Static parameters for inference
        batch_size (Optional[int]): Batch size for inference
        
    Returns:
        DataLoader: Inference dataloader
    """
    if batch_size is None:
        batch_size = config['inference']['generation'].get('batch_size', 10)
    
    if static_params is None:
        # Generate random static parameters based on config ranges
        static_params = generate_random_static_params(config)
    
    # Create a simple dataset from static parameters
    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, static_params):
            self.static_params = static_params
        
        def __len__(self):
            return len(self.static_params)
        
        def __getitem__(self, idx):
            return {'static_params': self.static_params[idx]}
    
    dataset = InferenceDataset(static_params)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )
    
    return dataloader


def generate_random_static_params(config: Dict[str, Any], num_samples: int = 50) -> torch.Tensor:
    """
    Generate random static parameters for inference.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        num_samples (int): Number of samples to generate
        
    Returns:
        torch.Tensor: Random static parameters
    """
    inference_config = config['inference']['static_params']
    
    # Generate continuous parameters
    age = np.random.uniform(
        inference_config['age_range'][0],
        inference_config['age_range'][1],
        num_samples
    )
    
    intensity = np.random.uniform(
        inference_config['intensity_range'][0],
        inference_config['intensity_range'][1],
        num_samples
    )
    
    stimulus_rate = np.random.uniform(
        inference_config['stimulus_rate_range'][0],
        inference_config['stimulus_rate_range'][1],
        num_samples
    )
    
    fmp = np.random.uniform(
        inference_config['fmp_range'][0],
        inference_config['fmp_range'][1],
        num_samples
    )
    
    res_no = np.random.uniform(
        inference_config['res_no_range'][0],
        inference_config['res_no_range'][1],
        num_samples
    )
    
    # Generate categorical parameters (one-hot encoded)
    hear_loss_categories = inference_config['hear_loss_categories']
    num_categories = len(hear_loss_categories)
    
    # Random category indices
    category_indices = np.random.randint(0, num_categories, num_samples)
    
    # Create one-hot encoding
    hear_loss_onehot = np.zeros((num_samples, num_categories))
    hear_loss_onehot[np.arange(num_samples), category_indices] = 1
    
    # Combine all parameters
    static_params = np.column_stack([
        age, intensity, stimulus_rate, fmp, res_no
    ])
    
    # Add one-hot encoded categorical parameters
    static_params = np.column_stack([static_params, hear_loss_onehot])
    
    # Normalize continuous parameters (assuming standard normalization)
    # Note: In practice, you should use the saved scaler from preprocessing
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    static_params[:, :5] = scaler.fit_transform(static_params[:, :5])
    
    return torch.from_numpy(static_params).float()


def load_data_transformers(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Load saved data transformers (scaler and encoder).
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Tuple[Any, Any]: Scaler and one-hot encoder
    """
    import joblib
    
    preprocessing_config = config['data']['preprocessing']
    
    scaler_path = preprocessing_config.get('scaler_path')
    encoder_path = preprocessing_config.get('encoder_path')
    
    scaler = None
    encoder = None
    
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logging.info(f"Loaded scaler from: {scaler_path}")
    
    if encoder_path and os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
        logging.info(f"Loaded encoder from: {encoder_path}")
    
    return scaler, encoder


def validate_data_config(config: Dict[str, Any]) -> None:
    """
    Validate data configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    data_config = config.get('data', {})
    
    # Check required paths
    if 'raw_data_path' not in data_config:
        raise ValueError("Missing 'raw_data_path' in data configuration")
    
    if 'processed_data_path' not in data_config:
        raise ValueError("Missing 'processed_data_path' in data configuration")
    
    # Check dataloader configuration
    dataloader_config = data_config.get('dataloader', {})
    
    batch_size = dataloader_config.get('batch_size', 32)
    if batch_size <= 0:
        raise ValueError(f"Invalid batch_size: {batch_size}")
    
    val_split = dataloader_config.get('val_split', 0.2)
    test_split = dataloader_config.get('test_split', 0.1)
    
    if val_split < 0 or val_split > 1:
        raise ValueError(f"Invalid val_split: {val_split}")
    
    if test_split < 0 or test_split > 1:
        raise ValueError(f"Invalid test_split: {test_split}")
    
    if val_split + test_split >= 1:
        raise ValueError(f"val_split + test_split must be < 1, got {val_split + test_split}")
    
    logging.info("Data configuration validated successfully")


def get_data_statistics(dataloader: DataLoader) -> Dict[str, Any]:
    """
    Compute statistics for a dataloader.
    
    Args:
        dataloader (DataLoader): DataLoader to analyze
        
    Returns:
        Dict[str, Any]: Data statistics
    """
    all_signals = []
    all_static_params = []
    all_peaks = []
    
    for batch in dataloader:
        all_signals.append(batch['signal'])
        all_static_params.append(batch['static_params'])
        
        if 'peaks' in batch:
            all_peaks.append(batch['peaks'])
    
    # Concatenate all data
    signals = torch.cat(all_signals, dim=0)
    static_params = torch.cat(all_static_params, dim=0)
    
    statistics = {
        'num_samples': len(signals),
        'signal_shape': signals.shape,
        'static_params_shape': static_params.shape,
        'signal_mean': signals.mean().item(),
        'signal_std': signals.std().item(),
        'signal_min': signals.min().item(),
        'signal_max': signals.max().item(),
        'static_params_mean': static_params.mean(dim=0).tolist(),
        'static_params_std': static_params.std(dim=0).tolist()
    }
    
    if all_peaks:
        peaks = torch.cat(all_peaks, dim=0)
        statistics.update({
            'peaks_shape': peaks.shape,
            'peaks_mean': peaks.mean().item(),
            'peaks_std': peaks.std().item()
        })
    
    return statistics 