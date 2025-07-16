import torch
from torch.utils.data import Dataset
import joblib
import numpy as np
from typing import Dict, Any, Optional, Callable, Union


class ABRDataset(Dataset):
    """
    PyTorch Dataset class for ABR (Auditory Brainstem Response) data.
    
    This dataset loads preprocessed ABR data from a pickle file and provides
    PyTorch-compatible data loading with optional transforms and peak data handling.
    """
    
    def __init__(
        self, 
        data_path: str, 
        transform: Optional[Callable] = None, 
        return_peaks: bool = True
    ):
        """
        Initialize the ABRDataset.
        
        Args:
            data_path (str): Path to the .pkl file containing preprocessed data
            transform (Optional[Callable]): Optional transform to apply to each sample
            return_peaks (bool): If False, ignore peaks and peak_mask fields
        """
        self.data_path = data_path
        self.transform = transform
        self.return_peaks = return_peaks
        
        # Load the preprocessed data
        self.data = joblib.load(data_path)
        
        # Validate data structure
        if not isinstance(self.data, list):
            raise ValueError(f"Expected data to be a list, got {type(self.data)}")
        
        if len(self.data) == 0:
            raise ValueError("Dataset is empty")
        
        # Validate sample structure
        required_keys = {"patient_id", "static_params", "signal"}
        if self.return_peaks:
            required_keys.update({"peaks", "peak_mask"})
        
        sample_keys = set(self.data[0].keys())
        missing_keys = required_keys - sample_keys
        if missing_keys:
            raise ValueError(f"Missing required keys in data samples: {missing_keys}")
        
        # Cache patient IDs once for efficiency
        self.patient_ids = sorted(list({s["patient_id"] for s in self.data}))
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Dict containing the sample data with tensors converted to PyTorch tensors
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        # Get the raw sample
        sample = self.data[idx].copy()
        
        # Convert numpy arrays to PyTorch tensors
        sample['static_params'] = torch.from_numpy(sample['static_params']).float()
        sample['signal'] = torch.from_numpy(sample['signal']).float()
        
        if self.return_peaks:
            sample['peaks'] = torch.from_numpy(sample['peaks']).float()
            sample['peak_mask'] = torch.from_numpy(sample['peak_mask']).bool()
        else:
            # Remove peaks and peak_mask if not needed
            sample.pop('peaks', None)
            sample.pop('peak_mask', None)
        
        # Apply transform if provided
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def get_sample_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset structure.
        
        Returns:
            Dict containing dataset metadata
        """
        if len(self.data) == 0:
            return {"num_samples": 0}
        
        sample = self.data[0]
        info = {
            "num_samples": len(self.data),
            "static_params_dim": sample['static_params'].shape[0],
            "signal_length": sample['signal'].shape[0],
            "return_peaks": self.return_peaks
        }
        
        if self.return_peaks:
            info["num_peaks"] = sample['peaks'].shape[0]
        
        # Use cached patient IDs
        info["num_unique_patients"] = len(self.patient_ids)
        
        return info
    
    def get_patient_samples(self, patient_id: str) -> list:
        """
        Get all samples for a specific patient.
        
        Args:
            patient_id (str): Patient ID to filter by
            
        Returns:
            List of indices for samples belonging to the specified patient
        """
        indices = []
        for idx, sample in enumerate(self.data):
            if sample['patient_id'] == patient_id:
                indices.append(idx)
        return indices
    
    def get_all_patient_ids(self) -> list:
        """
        Get all unique patient IDs in the dataset.
        
        Returns:
            List of unique patient IDs (cached for efficiency)
        """
        return self.patient_ids 