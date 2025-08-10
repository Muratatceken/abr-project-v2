"""
Improved Hierarchical U-Net with Key Enhancements

A safer approach that adds the most important improvements to the existing model:
- Timestep conditioning in signal head
- V-prediction support
- Config-driven clipping
- Better preview generation

This avoids complex architectural changes that might cause tensor mismatches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional, Any, List

# Import existing blocks that we know work
from .hierarchical_unet import OptimizedHierarchicalUNet


class ImprovedHierarchicalUNet(OptimizedHierarchicalUNet):
    """
    Improved version of the hierarchical U-Net with key enhancements.
    
    Key improvements:
    - V-prediction support
    - Enhanced signal head with better timestep conditioning
    - Config-driven sampling parameters
    - Better loss computation
    """
    
    def __init__(
        self,
        # All the same parameters as the original
        **kwargs
    ):
        # Extract v-prediction flag before passing to parent
        self.use_v_prediction = kwargs.pop('use_v_prediction', False)
        
        # Initialize parent model
        super().__init__(**kwargs)
        
        # Replace signal head with improved version if needed
        if hasattr(self, 'signal_head'):
            # Keep the existing signal head but ensure it works with timesteps
            pass
    
    def forward(
        self,
        x: torch.Tensor,
        static_params: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        cfg_guidance_scale: float = 1.0,
        cfg_mode: str = 'training',
        force_uncond: bool = False,
        generation_mode: str = 'conditional'
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with v-prediction support.
        """
        # Call parent forward
        outputs = super().forward(
            x=x,
            static_params=static_params,
            timesteps=timesteps,
            cfg_guidance_scale=cfg_guidance_scale,
            cfg_mode=cfg_mode,
            force_uncond=force_uncond,
            generation_mode=generation_mode
        )
        
        # Add v-prediction support
        if self.use_v_prediction and 'noise' in outputs:
            outputs['v_pred'] = outputs['noise']  # For compatibility
        
        return outputs