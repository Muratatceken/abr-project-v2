"""
Modular components for ABR Hierarchical U-Net

Professional implementation following SSSD-ECG patterns
"""

from .s4_layer import S4Layer, S4Block, EnhancedS4Layer
from .transformer_block import (
    TransformerBlock, MultiHeadAttention, MultiLayerTransformerBlock,
    DeepTransformerDecoder, EnhancedMultiHeadAttention
)
from .film import (
    FiLMLayer, ConditionalEmbedding, AdaptiveFiLMWithDropout, 
    MultiFiLM, CFGWrapper
)
from .positional import PositionalEmbedding, SinusoidalEmbedding
from .heads import (
    RobustPeakHead, RobustClassificationHead, RobustThresholdHead,
    EnhancedSignalHead, AttentionPooling, StaticParameterGenerationHead,
    # Aliases for backward compatibility
    EnhancedPeakHead, EnhancedClassificationHead, EnhancedThresholdHead
)
from .conv_blocks import Conv1dBlock, ResidualBlock, DownsampleBlock, UpsampleBlock, SkipConnection, EnhancedConvBlock, ResidualS4Block
from .encoder_block import EnhancedEncoderBlock, MultiScaleEncoderStack, OptimizedEncoderBlock, OptimizedBottleneckProcessor
from .decoder_block import EnhancedDecoderBlock, MultiScaleDecoderStack, BottleneckProcessor, OptimizedDecoderBlock, EnhancedSkipFusion, TaskSpecificFeatureExtractor

__all__ = [
    'S4Layer', 'S4Block', 'EnhancedS4Layer',
    'TransformerBlock', 'MultiHeadAttention', 'MultiLayerTransformerBlock',
    'DeepTransformerDecoder', 'EnhancedMultiHeadAttention',
    'FiLMLayer', 'ConditionalEmbedding', 'AdaptiveFiLMWithDropout', 
    'MultiFiLM', 'CFGWrapper',
    'PositionalEmbedding', 'SinusoidalEmbedding',
    'RobustPeakHead', 'RobustClassificationHead', 'RobustThresholdHead',
    'EnhancedSignalHead', 'AttentionPooling', 'StaticParameterGenerationHead',
    'EnhancedPeakHead', 'EnhancedClassificationHead', 'EnhancedThresholdHead', 
    'Conv1dBlock', 'ResidualBlock', 'DownsampleBlock', 'UpsampleBlock', 'SkipConnection',
    'EnhancedConvBlock', 'ResidualS4Block',
    'EnhancedEncoderBlock', 'MultiScaleEncoderStack',
    'EnhancedDecoderBlock', 'MultiScaleDecoderStack', 'BottleneckProcessor',
    'OptimizedEncoderBlock', 'OptimizedDecoderBlock', 'OptimizedBottleneckProcessor',
    'TaskSpecificFeatureExtractor', 'EnhancedSkipFusion'
] 