import torch
from torch.utils.data import Dataset, Subset
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, Union, Tuple, List
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter
import logging
import warnings

warnings.filterwarnings('ignore')


class ABRDataset(Dataset):
    """
    PyTorch Dataset class for ABR (Auditory Brainstem Response) data.
    
    Compatible with S4 + Transformer-based Hierarchical U-Net model.
    Loads data from ultimate_dataset.pkl and provides proper tensor formatting
    for multi-task learning with FiLM conditioning.
    
    Each sample returns:
    - signal: ABR time series [200] for S4/Transformer processing
    - static: Static parameters [4] for FiLM conditioning  
    - target: Hearing loss classification label
    - v_peak: V peak data [2] for peak prediction
    - v_peak_mask: Validity mask [2] for missing peak handling
    - patient_id: For stratified splitting
    """
    
    def __init__(
        self, 
        data_path: str = "data/processed/ultimate_dataset.pkl",
        transform: Optional[Callable] = None,
        normalize_signal: bool = True,
        normalize_static: bool = True,
        return_peak_labels: bool = False
    ):
        """
        Initialize the ABRDataset.
        
        Args:
            data_path (str): Path to the ultimate_dataset.pkl file
            transform (Optional[Callable]): Optional transform to apply to each sample
            normalize_signal (bool): Whether to apply per-sample signal normalization
            normalize_static (bool): Whether static params are pre-normalized
        """
        self.data_path = data_path
        self.transform = transform
        self.normalize_signal = normalize_signal
        self.normalize_static = normalize_static
        self.return_peak_labels = return_peak_labels
        
        # Properties for training pipeline compatibility
        self.sequence_length = 200
        self.static_dim = 4
        self.static_names = ["Age", "Intensity", "StimulusRate", "FMP"]
        
        # Load the preprocessed data
        self._load_data()
        
        # Validate data structure
        self._validate_data()
        
        # Create patient mapping for stratified splitting
        self._create_patient_mapping()
        
        logging.info(f"ABRDataset initialized with {len(self.data)} samples from {len(self.patient_ids)} patients")
        logging.info(f"Signal shape: {self.data[0]['signal'].shape}")
        logging.info(f"Static params shape: {self.data[0]['static_params'].shape}")
        logging.info(f"Target distribution: {dict(Counter(self.targets))}")
    
    def _load_data(self) -> None:
        """Load data from pkl file."""
        try:
            dataset_dict = joblib.load(self.data_path)
            
            # Extract data based on ultimate dataset structure
            if isinstance(dataset_dict, dict) and 'data' in dataset_dict:
                self.data = dataset_dict['data']
                self.scaler = dataset_dict.get('scaler', None)
                self.label_encoder = dataset_dict.get('label_encoder', None)
                self.metadata = dataset_dict.get('metadata', {})
            else:
                # Fallback for direct list format
                self.data = dataset_dict
                self.scaler = None
                self.label_encoder = None
                self.metadata = {}
                
            logging.info(f"Loaded {len(self.data)} samples from {self.data_path}")
            
        except Exception as e:
            raise FileNotFoundError(f"Could not load data from {self.data_path}: {e}")
    
    def _validate_data(self) -> None:
        """Validate the loaded data structure."""
        if not isinstance(self.data, list) or len(self.data) == 0:
            raise ValueError("Dataset is empty or not in expected list format")
        
        # Check required keys (minimal feature set)
        required_keys = {'patient_id', 'static_params', 'signal', 'target'}
        sample_keys = set(self.data[0].keys())
        missing_keys = required_keys - sample_keys
        
        if missing_keys:
            raise ValueError(f"Missing required keys in data samples: {missing_keys}")
        
        # Validate shapes with strict assertion for T=200
        sample = self.data[0]
        if sample['static_params'].shape[0] != 4:
            raise ValueError(f"Expected 4 static parameters, got {sample['static_params'].shape[0]}")
        assert sample['signal'].shape[0] == 200, f"Expected exactly 200 time points, got {sample['signal'].shape[0]}. No resampling allowed in dataset."
            
        logging.info("Data validation passed")
    
    def _create_patient_mapping(self) -> None:
        """Create patient ID mapping and target lists for stratified splitting."""
        self.patient_ids = []
        self.targets = []
        
        for sample in self.data:
            self.patient_ids.append(sample['patient_id'])
            self.targets.append(sample['target'])
        
        # Get unique patients and their most common target class
        patient_df = pd.DataFrame({
            'patient_id': self.patient_ids,
            'target': self.targets
        })
        
        # For each patient, find their most common target class
        self.patient_targets = (patient_df.groupby('patient_id')['target']
                               .agg(lambda x: x.mode().iloc[0])
                               .to_dict())
        
        self.unique_patients = list(self.patient_targets.keys())
        
        logging.info(f"Found {len(self.unique_patients)} unique patients")
        logging.info(f"Patient-level target distribution: {dict(Counter(self.patient_targets.values()))}")
        
        # Track peak label distribution for multi-task training
        if self.return_peak_labels:
            self.peak_labels = []
            for sample in self.data:
                peak_exists = bool(sample['v_peak_mask'][0] and sample['v_peak_mask'][1])
                self.peak_labels.append(int(peak_exists))
            
            peak_counts = Counter(self.peak_labels)
            logging.info(f"Peak label distribution: {dict(peak_counts)}")
            logging.info(f"Peak positive ratio: {peak_counts.get(1, 0) / len(self.peak_labels):.3f}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a sample from the dataset for training pipeline.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Dict containing:
                - x0: ABR time series [1, 200] (normalized)
                - stat: Static parameters [4] in order [Age, Intensity, StimulusRate, FMP]
                - meta: Metadata dict with patient_id and target
                - peak_exists: Peak existence label [1] when return_peak_labels=True
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        sample = self.data[idx]
        
        # Extract and convert data
        signal = sample['signal'].astype(np.float32)
        static_params = sample['static_params'].astype(np.float32)
        target = int(sample['target'])
        patient_id = str(sample['patient_id'])
        
        # Compute peak existence label when multi-task training is enabled
        if self.return_peak_labels:
            peak_exists = bool(sample['v_peak_mask'][0] and sample['v_peak_mask'][1])
        
        # Strict length assertion (no resampling allowed)
        assert len(signal) == self.sequence_length, f"Signal length {len(signal)} != {self.sequence_length}. No resampling in dataset."
        
        # Apply signal normalization if requested
        if self.normalize_signal:
            # Z-score normalization per sample (already done in preprocessing, but ensure)
            signal_std = np.std(signal)
            if signal_std > 1e-8:  # Avoid division by zero
                signal = (signal - np.mean(signal)) / signal_std
        
        # Convert to required format for training pipeline
        result = {
            'x0': torch.tensor(signal, dtype=torch.float32).unsqueeze(0),  # [1, T] for conv1d
            'stat': torch.tensor(static_params, dtype=torch.float32),      # [S] static params
            'meta': {
                'patient_id': patient_id,
                'target': target,
                'sample_idx': idx
            }
        }
        
        # Add peak labels when multi-task training is enabled
        if self.return_peak_labels:
            result['peak_exists'] = torch.tensor(peak_exists, dtype=torch.float32)
        
        # Apply optional transform
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def denormalize_signal(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize signal for visualization in TensorBoard.
        
        Since we apply z-score normalization per sample, we cannot perfectly
        recover the original signal without storing per-sample statistics.
        This method provides a reasonable approximation for visualization.
        
        Args:
            x_norm: Normalized signal tensor of shape [..., T] 
            
        Returns:
            Denormalized signal (approximate) in physical units
        """
        # Since we can't perfectly denormalize per-sample z-score,
        # we apply a typical ABR amplitude scale for visualization
        # Typical ABR amplitudes range from -0.5 to +0.5 µV
        typical_scale = 0.2  # µV
        return x_norm * typical_scale
    
    def get_sample_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset structure.
        
        Returns:
            Dictionary with dataset statistics and structure info
        """
        if len(self.data) == 0:
            return {}
        
        sample = self.data[0]
        
        info = {
            'num_samples': len(self.data),
            'num_patients': len(self.unique_patients),
            'signal_length': sample['signal'].shape[0],
            'static_params_dim': sample['static_params'].shape[0],
            'num_classes': len(set(self.targets)),
            'class_distribution': dict(Counter(self.targets)),
            'patient_level_distribution': dict(Counter(self.patient_targets.values())),
            'data_path': self.data_path,
            'metadata': self.metadata
        }
        return info
    
    def get_class_weights(self, method: str = 'inverse_freq') -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        
        Args:
            method (str): Method to calculate weights ('inverse_freq' or 'balanced')
            
        Returns:
            torch.Tensor: Class weights for loss function
        """
        class_counts = Counter(self.targets)
        num_classes = len(class_counts)
        total_samples = len(self.targets)
        
        if method == 'inverse_freq':
            weights = []
            for i in range(num_classes):
                if i in class_counts:
                    weights.append(total_samples / class_counts[i])
                else:
                    weights.append(1.0)
        elif method == 'balanced':
            weights = []
            for i in range(num_classes):
                if i in class_counts:
                    weights.append(total_samples / (num_classes * class_counts[i]))
                else:
                    weights.append(1.0)
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_peak_class_weights(self, method: str = 'inverse_freq') -> torch.Tensor:
        """
        Calculate class weights for peak detection to handle class imbalance.
        
        Args:
            method (str): Method to calculate weights ('inverse_freq' or 'balanced')
            
        Returns:
            torch.Tensor: Class weights for [negative_class, positive_class]
        """
        if not self.return_peak_labels:
            raise ValueError("Dataset must be initialized with return_peak_labels=True to compute peak class weights")
        
        # Extract peak labels from all samples
        peak_labels = []
        for idx in range(len(self.data)):
            sample = self.data[idx]
            peak_exists = bool(sample['v_peak_mask'][0] and sample['v_peak_mask'][1])
            peak_labels.append(int(peak_exists))
        
        # Calculate class distribution
        class_counts = Counter(peak_labels)
        num_classes = 2  # Binary classification: 0 (no peak), 1 (peak exists)
        total_samples = len(peak_labels)
        
        if method == 'inverse_freq':
            weights = []
            for i in range(num_classes):
                if i in class_counts:
                    weights.append(total_samples / class_counts[i])
                else:
                    weights.append(1.0)
        elif method == 'balanced':
            weights = []
            for i in range(num_classes):
                if i in class_counts:
                    weights.append(total_samples / (num_classes * class_counts[i]))
                else:
                    weights.append(1.0)
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        return torch.tensor(weights, dtype=torch.float32)


def stratified_patient_split(
    dataset: ABRDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    min_samples_per_class: int = 1
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified train/validation/test splits ensuring no patient appears 
    in multiple splits.
    
    Args:
        dataset (ABRDataset): The ABR dataset
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set  
        test_ratio (float): Proportion for test set
        random_state (int): Random seed for reproducibility
        min_samples_per_class (int): Minimum samples per class in each split
        
    Returns:
        Tuple[List[int], List[int], List[int]]: Indices for train, val, test sets
    """
    # Validate split ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Get patient-level data
    patients = dataset.unique_patients
    patient_targets = [dataset.patient_targets[p] for p in patients]
    
    # First split: train vs (val + test)
    train_test_ratio = train_ratio / (train_ratio + val_ratio + test_ratio)
    
    if len(set(patient_targets)) < 2:
        # If only one class, do random split
        np.random.seed(random_state)
        shuffled_patients = np.random.permutation(patients)
        
        n_train = int(len(patients) * train_ratio)
        n_val = int(len(patients) * val_ratio)
        
        train_patients = shuffled_patients[:n_train]
        val_patients = shuffled_patients[n_train:n_train + n_val]
        test_patients = shuffled_patients[n_train + n_val:]
    else:
        # Stratified split for multiple classes
        from sklearn.model_selection import train_test_split
        
        # Split patients: train vs (val + test)
        train_patients, temp_patients, train_targets, temp_targets = train_test_split(
            patients, patient_targets,
            train_size=train_ratio,
            stratify=patient_targets,
            random_state=random_state
        )
        
        # Split temp into val and test
        if val_ratio > 0 and test_ratio > 0:
            val_test_ratio = val_ratio / (val_ratio + test_ratio)
            val_patients, test_patients = train_test_split(
                temp_patients,
                train_size=val_test_ratio,
                stratify=temp_targets,
                random_state=random_state + 1
            )
        elif val_ratio > 0:
            val_patients = temp_patients
            test_patients = []
        else:
            val_patients = []
            test_patients = temp_patients
    
    # Convert patient lists to sample indices
    def get_indices_for_patients(patient_list):
        indices = []
        for i, sample in enumerate(dataset.data):
            if sample['patient_id'] in patient_list:
                indices.append(i)
        return indices
    
    train_indices = get_indices_for_patients(train_patients)
    val_indices = get_indices_for_patients(val_patients) if val_ratio > 0 else []
    test_indices = get_indices_for_patients(test_patients) if test_ratio > 0 else []
    
    # Log split statistics
    def log_split_stats(name, indices):
        if len(indices) == 0:
            return
        targets = [dataset.targets[i] for i in indices]
        patients_in_split = set(dataset.patient_ids[i] for i in indices)
        logging.info(f"{name}: {len(indices)} samples, {len(patients_in_split)} patients")
        logging.info(f"  Class distribution: {dict(Counter(targets))}")
    
    log_split_stats("Train", train_indices)
    log_split_stats("Validation", val_indices)
    log_split_stats("Test", test_indices)
    
    # Verify no patient overlap
    train_patients_set = set(dataset.patient_ids[i] for i in train_indices)
    val_patients_set = set(dataset.patient_ids[i] for i in val_indices)
    test_patients_set = set(dataset.patient_ids[i] for i in test_indices)
    
    overlaps = []
    if train_patients_set & val_patients_set:
        overlaps.append("train-val")
    if train_patients_set & test_patients_set:
        overlaps.append("train-test")
    if val_patients_set & test_patients_set:
        overlaps.append("val-test")
    
    if overlaps:
        raise ValueError(f"Patient overlap detected in splits: {overlaps}")
    
    logging.info("✅ Patient-stratified split completed successfully - no patient overlap")
    
    return train_indices, val_indices, test_indices


def create_stratified_datasets(
    dataset: ABRDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    Create stratified dataset splits with no patient overlap.
    
    Args:
        dataset (ABRDataset): The base dataset
        train_ratio (float): Training set proportion
        val_ratio (float): Validation set proportion
        test_ratio (float): Test set proportion
        random_state (int): Random seed
        
    Returns:
        Tuple[Subset, Subset, Subset]: Train, validation, and test datasets
    """
    train_indices, val_indices, test_indices = stratified_patient_split(
        dataset, train_ratio, val_ratio, test_ratio, random_state
    )
    
    train_dataset = Subset(dataset, train_indices) if train_indices else None
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    test_dataset = Subset(dataset, test_indices) if test_indices else None
    
    return train_dataset, val_dataset, test_dataset


def load_ultimate_dataset(
    data_path: str = "data/processed/ultimate_dataset.pkl",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    **dataset_kwargs
) -> Tuple[ABRDataset, Subset, Subset, Subset]:
    """
    Load the ultimate ABR dataset and create stratified splits.
    
    Args:
        data_path (str): Path to ultimate_dataset.pkl
        train_ratio (float): Training set proportion
        val_ratio (float): Validation set proportion  
        test_ratio (float): Test set proportion
        random_state (int): Random seed
        **dataset_kwargs: Additional arguments for ABRDataset
        
    Returns:
        Tuple containing:
            - full_dataset: Complete ABRDataset
            - train_dataset: Training subset
            - val_dataset: Validation subset  
            - test_dataset: Test subset
    """
    # Load full dataset
    full_dataset = ABRDataset(data_path=data_path, **dataset_kwargs)
    
    # Create stratified splits
    train_dataset, val_dataset, test_dataset = create_stratified_datasets(
        full_dataset, train_ratio, val_ratio, test_ratio, random_state
    )
    
    # Log final statistics
    logging.info("="*60)
    logging.info("DATASET LOADING COMPLETE")
    logging.info("="*60)
    logging.info(f"Total samples: {len(full_dataset)}")
    logging.info(f"Unique patients: {len(full_dataset.unique_patients)}")
    logging.info(f"Training samples: {len(train_dataset) if train_dataset else 0}")
    logging.info(f"Validation samples: {len(val_dataset) if val_dataset else 0}")
    logging.info(f"Test samples: {len(test_dataset) if test_dataset else 0}")
    
    return full_dataset, train_dataset, val_dataset, test_dataset


def abr_collate_fn(batch):
    """Collate function for training pipeline format."""
    x0_batch = torch.stack([item['x0'] for item in batch])                 # [B, 1, 200]
    stat_batch = torch.stack([item['stat'] for item in batch])             # [B, 4]
    meta_batch = [item['meta'] for item in batch]                          # List of meta dicts

    result = {
        'x0': x0_batch,
        'stat': stat_batch,
        'meta': meta_batch
    }
    
    # Handle peak labels when present (multi-task training)
    if 'peak_exists' in batch[0]:
        peak_batch = torch.stack([item['peak_exists'] for item in batch])
        result['peak_exists'] = peak_batch
    
    return result


def create_optimized_dataloaders(
    data_path: str = "data/processed/ultimate_dataset.pkl",
    config: Dict[str, Any] = None,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    use_balanced_sampler: bool = False,
    use_peak_balanced_sampler: bool = False,
    augment: bool = True,
    cfg_dropout_prob: float = 0.1,
    random_state: int = 42,
    **kwargs
):
    """
    Create optimized DataLoaders with all performance enhancements.
    
    Args:
        data_path: Path to dataset file
        config: Configuration dictionary (takes precedence over individual args)
        batch_size: Batch size for training
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch
        persistent_workers: Keep workers alive between epochs
        use_balanced_sampler: Use weighted sampling for class balance
        use_peak_balanced_sampler: Use weighted sampling for peak detection class balance
        augment: Enable data augmentation for training
        cfg_dropout_prob: CFG dropout probability
        random_state: Random seed
        **kwargs: Additional arguments
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, full_dataset)
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import numpy as np
    
    # Use config values if provided, otherwise use function arguments
    if config is not None:
        batch_size = config.get('batch_size', batch_size)
        train_ratio = config.get('data', {}).get('train_ratio', train_ratio)
        val_ratio = config.get('data', {}).get('val_ratio', val_ratio) or config.get('val_split', val_ratio)
        test_ratio = config.get('data', {}).get('test_ratio', test_ratio)
        num_workers = config.get('num_workers', num_workers)
        pin_memory = config.get('pin_memory', pin_memory)
        prefetch_factor = config.get('prefetch_factor', prefetch_factor)
        persistent_workers = config.get('persistent_workers', persistent_workers)
        use_balanced_sampler = config.get('use_balanced_sampler', use_balanced_sampler)
        use_peak_balanced_sampler = config.get('loader', {}).get('use_peak_balanced_sampler', use_peak_balanced_sampler)
        augment = config.get('augment', augment)
        cfg_dropout_prob = config.get('cfg_dropout_prob', cfg_dropout_prob)
        random_state = config.get('random_seed', random_state)
    
    # Load dataset with stratified splits
    full_dataset, train_dataset, val_dataset, test_dataset = load_ultimate_dataset(
        data_path=data_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )
    
    # Use module-level collate function for multiprocessing compatibility
    
    # Create balanced sampler for training if requested
    train_sampler = None
    if use_balanced_sampler and train_dataset is not None:
        # Get targets from training dataset - use meta['target'] instead of sample['target']
        train_targets = []
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            train_targets.append(sample['meta']['target'].item() if torch.is_tensor(sample['meta']['target']) else sample['meta']['target'])
        
        # Calculate class weights
        class_counts = np.bincount(train_targets)
        class_weights = 1.0 / (class_counts + 1e-8)  # Add small epsilon to avoid division by zero
        sample_weights = [class_weights[target] for target in train_targets]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"✓ Created balanced sampler with class weights: {class_weights}")
    
    # Create peak-balanced sampler for multi-task training if requested
    if use_peak_balanced_sampler and train_dataset is not None:
        # Fix peak-balanced sampling to properly handle Subset objects
        try:
            # Get parent dataset and indices from Subset
            parent = train_dataset.dataset
            idxs = train_dataset.indices
            
            # Compute peak labels from parent dataset using Subset indices
            labels = [int(bool(parent.data[i]['v_peak_mask'][0] and parent.data[i]['v_peak_mask'][1])) for i in idxs]
            
            # Calculate peak class weights
            peak_class_counts = np.bincount(labels)
            peak_class_weights = 1.0 / (peak_class_counts + 1e-8)
            peak_sample_weights = [peak_class_weights[label] for label in labels]
            
            train_sampler = WeightedRandomSampler(
                weights=peak_sample_weights,
                num_samples=len(peak_sample_weights),
                replacement=True
            )
            print(f"✓ Created peak-balanced sampler with class weights: {peak_class_weights}")
            print(f"  Peak label distribution: {dict(zip(range(len(peak_class_counts)), peak_class_counts))}")
            print(f"  Training subset size: {len(idxs)}")
        except Exception as e:
            print(f"⚠️  Peak-balanced sampling failed: {e}")
            print("   Falling back to standard sampling")
            train_sampler = None
    
    # Create optimized data loaders
    train_loader = None
    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=abr_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
            persistent_workers=persistent_workers and num_workers > 0
        )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=abr_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
            persistent_workers=persistent_workers and num_workers > 0
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=abr_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
            persistent_workers=persistent_workers and num_workers > 0
        )
    
    print(f"✓ Created optimized dataloaders:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Workers: {num_workers}")
    print(f"  - Prefetch factor: {prefetch_factor}")
    print(f"  - Persistent workers: {persistent_workers}")
    print(f"  - Balanced sampling: {use_balanced_sampler}")
    print(f"  - Pin memory: {pin_memory}")
    
    return train_loader, val_loader, test_loader, full_dataset


# Backward compatibility function (simplified)
def create_dataloaders_compatible(
    data_path: str = "data/processed/ultimate_dataset.pkl",
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    random_state: int = 42
):
    """
    Backward compatibility wrapper for create_optimized_dataloaders.
    """
    train_loader, val_loader, test_loader, _ = create_optimized_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        num_workers=num_workers,
        pin_memory=pin_memory,
        random_state=random_state
    )
    return train_loader, val_loader, test_loader 