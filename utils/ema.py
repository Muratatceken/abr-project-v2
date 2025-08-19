"""
Exponential Moving Average (EMA) for model parameters.

Maintains shadow weights that are exponentially averaged versions of the model parameters.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import copy


class EMA:
    """
    Exponential Moving Average helper for model parameters.
    
    Maintains shadow weights and provides methods to apply/restore EMA weights
    for improved sampling quality during inference.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        """
        Initialize EMA with a model.
        
        Args:
            model: PyTorch model to track
            decay: EMA decay factor (0.999 typical for diffusion models)
            device: Device to store shadow parameters on
        """
        self.decay = decay
        self.device = device or next(model.parameters()).device
        
        # Store shadow parameters
        self.shadow = {}
        self._register_model(model)
        
    def _register_model(self, model: nn.Module):
        """Register model parameters and create shadow copies."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
    
    def update(self, model: nn.Module):
        """
        Update EMA shadow parameters.
        
        Args:
            model: Model with current parameters
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # EMA update: shadow = decay * shadow + (1 - decay) * param
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Apply EMA shadow parameters to model and return original parameters.
        
        Args:
            model: Model to apply EMA weights to
            
        Returns:
            Dictionary of original parameters for restoration
        """
        original_params = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # Store original
                    original_params[name] = param.data.clone()
                    # Apply EMA
                    param.data.copy_(self.shadow[name])
        
        return original_params
    
    def restore(self, model: nn.Module, original_params: Dict[str, torch.Tensor]):
        """
        Restore original parameters to model.
        
        Args:
            model: Model to restore parameters to
            original_params: Original parameters returned by apply_to()
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in original_params:
                    param.data.copy_(original_params[name])
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state for checkpointing."""
        return {
            'shadow': self.shadow,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)
        
        # Move to correct device
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(self.device)


class EMAContextManager:
    """
    Context manager for temporary EMA application.
    
    Usage:
        with EMAContextManager(ema, model):
            # Model now uses EMA weights
            output = model(input)
        # Model weights restored automatically
    """
    
    def __init__(self, ema: EMA, model: nn.Module):
        self.ema = ema
        self.model = model
        self.original_params = None
    
    def __enter__(self):
        self.original_params = self.ema.apply_to(self.model)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_params is not None:
            self.ema.restore(self.model, self.original_params)


def create_ema(model: nn.Module, decay: float = 0.999, device: Optional[torch.device] = None) -> EMA:
    """
    Factory function to create EMA helper.
    
    Args:
        model: PyTorch model
        decay: EMA decay factor 
        device: Device for shadow parameters
        
    Returns:
        EMA instance
    """
    return EMA(model, decay, device)
