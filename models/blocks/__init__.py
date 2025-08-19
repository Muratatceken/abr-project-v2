"""
Core components for ABR Transformer Generator

Exports only the essential components needed for the new transformer architecture.
Legacy U-Net and S4 components are kept for backward compatibility but not exported.
"""

# Core components for ABR Transformer
from .transformer_block import (
    TransformerBlock, MultiHeadAttention, MultiLayerTransformerBlock
)
from .film import (
    FiLMLayer, ConditionalEmbedding
)
from .positional import PositionalEmbedding, SinusoidalEmbedding
from .heads import (
    BaseHead, EnhancedSignalHead, AttentionPooling
)

__all__ = [
    # Core Transformer components for ABR Transformer Generator
    'TransformerBlock', 
    'MultiHeadAttention', 
    'MultiLayerTransformerBlock',
    'FiLMLayer', 
    'ConditionalEmbedding',
    'PositionalEmbedding', 
    'SinusoidalEmbedding',
    'BaseHead', 
    'EnhancedSignalHead', 
    'AttentionPooling',
] 