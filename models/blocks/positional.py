"""
Positional Embedding Implementations for ABR Signal Processing

Professional implementations of various positional encoding schemes
for transformer-based architectures.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings as used in "Attention Is All You Need".
    
    Uses sine and cosine functions of different frequencies to encode
    position information without requiring learned parameters.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        temperature: float = 10000.0,
        dropout: float = 0.0
    ):
        """
        Initialize sinusoidal positional embedding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            temperature: Temperature parameter for frequency scaling
            dropout: Dropout rate applied after adding positional encoding
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create frequency components
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(temperature) / d_model)
        )
        
        # Apply sine to even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd dimensions
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trainable parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            offset: Position offset for relative positioning
            
        Returns:
            Output tensor with positional encoding [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # If sequence is longer than max_len, extend the positional encoding
        if seq_len + offset > self.max_len:
            self._extend_positional_encoding(seq_len + offset)
        
        # Add positional encoding
        pos_encoding = self.pe[:, offset:offset + seq_len, :]
        x = x + pos_encoding
        
        return self.dropout(x)
    
    def _extend_positional_encoding(self, new_max_len: int):
        """Extend positional encoding for longer sequences."""
        pe = torch.zeros(new_max_len, self.d_model)
        position = torch.arange(0, new_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() *
            -(math.log(self.temperature) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        self.max_len = new_max_len


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings.
    
    Uses trainable embedding lookup table for position information.
    Can be more flexible than sinusoidal embeddings for specific tasks.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.0,
        padding_idx: Optional[int] = None
    ):
        """
        Initialize learned positional embedding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
            padding_idx: Index for padding token (no gradient updates)
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(
            max_len, d_model, padding_idx=padding_idx
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize position embeddings."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        if self.position_embeddings.padding_idx is not None:
            with torch.no_grad():
                self.position_embeddings.weight[self.position_embeddings.padding_idx].fill_(0)
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            offset: Position offset for relative positioning
            
        Returns:
            Output tensor with positional encoding [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        assert seq_len + offset <= self.max_len, \
            f"Sequence length {seq_len} + offset {offset} exceeds max_len {self.max_len}"
        
        # Create position indices
        positions = torch.arange(
            offset, offset + seq_len, 
            device=x.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Add positional embeddings
        pos_embeddings = self.position_embeddings(positions)
        x = x + pos_embeddings
        
        return self.dropout(x)


class RelativePositionalEmbedding(nn.Module):
    """
    Relative positional embeddings for transformer models.
    
    Encodes relative distances between positions rather than absolute positions.
    Useful for tasks where relative position matters more than absolute position.
    """
    
    def __init__(
        self,
        d_model: int,
        max_relative_distance: int = 128,
        num_heads: int = 8,
        bidirectional: bool = True
    ):
        """
        Initialize relative positional embedding.
        
        Args:
            d_model: Model dimension
            max_relative_distance: Maximum relative distance to encode
            num_heads: Number of attention heads
            bidirectional: Whether to use bidirectional relative positions
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_relative_distance = max_relative_distance
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.bidirectional = bidirectional
        
        # Vocabulary size for relative positions
        if bidirectional:
            vocab_size = 2 * max_relative_distance + 1
        else:
            vocab_size = max_relative_distance + 1
        
        # Relative position embeddings
        self.relative_position_embeddings = nn.Embedding(vocab_size, self.d_head)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize relative position embeddings."""
        nn.init.normal_(self.relative_position_embeddings.weight, mean=0.0, std=0.02)
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """
        Generate relative position indices.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position indices [seq_len, seq_len]
        """
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).expand(seq_len, seq_len)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        if self.bidirectional:
            # Clip to [-max_relative_distance, max_relative_distance]
            distance_mat_clipped = torch.clamp(
                distance_mat, 
                -self.max_relative_distance, 
                self.max_relative_distance
            )
            # Shift to positive indices [0, 2*max_relative_distance]
            final_mat = distance_mat_clipped + self.max_relative_distance
        else:
            # Only positive relative distances [0, max_relative_distance]
            final_mat = torch.clamp(distance_mat, 0, self.max_relative_distance)
        
        return final_mat
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate relative positional bias for attention.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensors on
            
        Returns:
            Relative position bias [num_heads, seq_len, seq_len]
        """
        # Get relative position indices
        relative_positions = self._get_relative_positions(seq_len).to(device)
        
        # Get relative position embeddings
        relative_embeddings = self.relative_position_embeddings(relative_positions)
        
        # Expand for multiple heads: [seq_len, seq_len, d_head] -> [num_heads, seq_len, seq_len]
        relative_embeddings = relative_embeddings.unsqueeze(0).expand(
            self.num_heads, -1, -1, -1
        ).reshape(self.num_heads, seq_len, seq_len)
        
        return relative_embeddings


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation.
    
    Applies rotation to queries and keys based on their absolute positions,
    which naturally encodes relative position information.
    
    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        base: float = 10000.0
    ):
        """
        Initialize rotary positional embedding.
        
        Args:
            d_model: Model dimension (should be even)
            max_len: Maximum sequence length
            base: Base for frequency calculation
        """
        super().__init__()
        
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Precompute frequency components
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute rotary embeddings for efficiency
        self._precompute_rotary_embeddings(max_len)
    
    def _precompute_rotary_embeddings(self, max_len: int):
        """Precompute rotary embeddings for efficiency."""
        positions = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(positions, self.inv_freq)
        
        # Create rotation matrices
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        
        self.register_buffer('cos_freqs', cos_freqs)
        self.register_buffer('sin_freqs', sin_freqs)
    
    def _apply_rotary_embedding(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor.
        
        Args:
            x: Input tensor [..., seq_len, d_model]
            cos: Cosine frequencies [seq_len, d_model//2]
            sin: Sine frequencies [seq_len, d_model//2]
            
        Returns:
            Rotated tensor [..., seq_len, d_model]
        """
        # Split into even and odd dimensions
        x1, x2 = x[..., 0::2], x[..., 1::2]
        
        # Apply rotation
        # [cos(θ) -sin(θ)] [x1]   [x1*cos(θ) - x2*sin(θ)]
        # [sin(θ)  cos(θ)] [x2] = [x1*sin(θ) + x2*cos(θ)]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Interleave back together
        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
        
        return rotated_x
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embedding to queries and keys.
        
        Args:
            q: Query tensor [..., seq_len, d_model]
            k: Key tensor [..., seq_len, d_model]
            offset: Position offset
            
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_len = q.size(-2)
        
        # Get frequency components for current sequence
        cos = self.cos_freqs[offset:offset + seq_len]
        sin = self.sin_freqs[offset:offset + seq_len]
        
        # Apply rotary embedding
        q_rotated = self._apply_rotary_embedding(q, cos, sin)
        k_rotated = self._apply_rotary_embedding(k, cos, sin)
        
        return q_rotated, k_rotated


class PositionalEmbedding(nn.Module):
    """
    Unified positional embedding module that supports multiple embedding types.
    
    Provides a single interface for different positional encoding schemes.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        embedding_type: str = 'sinusoidal',
        dropout: float = 0.0,
        **kwargs
    ):
        """
        Initialize positional embedding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            embedding_type: Type of positional embedding ('sinusoidal', 'learned', 'relative', 'rotary')
            dropout: Dropout rate
            **kwargs: Additional arguments for specific embedding types
        """
        super().__init__()
        
        self.embedding_type = embedding_type.lower()
        
        if self.embedding_type == 'sinusoidal':
            self.pos_embedding = SinusoidalEmbedding(
                d_model=d_model,
                max_len=max_len,
                dropout=dropout,
                **kwargs
            )
        elif self.embedding_type == 'learned':
            self.pos_embedding = LearnedPositionalEmbedding(
                d_model=d_model,
                max_len=max_len,
                dropout=dropout,
                **kwargs
            )
        elif self.embedding_type == 'relative':
            self.pos_embedding = RelativePositionalEmbedding(
                d_model=d_model,
                max_relative_distance=kwargs.get('max_relative_distance', 128),
                num_heads=kwargs.get('num_heads', 8),
                **kwargs
            )
        elif self.embedding_type == 'rotary':
            self.pos_embedding = RotaryPositionalEmbedding(
                d_model=d_model,
                max_len=max_len,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through positional embedding."""
        return self.pos_embedding(*args, **kwargs) 