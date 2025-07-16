"""
CVAE Evaluation Package

This package provides comprehensive evaluation tools for CVAE models including:
- Quantitative metrics computation
- Visual diagnostics and plotting
- Modular evaluation pipeline
"""

from .evaluate import CVAEEvaluator

__version__ = "1.0.0"
__all__ = ["CVAEEvaluator"] 