"""
Core components for ABR Transformer Generator
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