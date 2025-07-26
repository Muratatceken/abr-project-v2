"""
Transformer Block Implementation for ABR Signal Processing

Custom implementation of multi-head attention and transformer blocks
for the hierarchical U-Net decoder path.

Based on professional patterns from SSSD-ECG project.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with support for self-attention and cross-attention.
    
    Implements scaled dot-product attention with multiple heads
    for enhanced representation learning.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        temperature: float = 1.0,
        is_cross_attention: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.temperature = temperature
        self.is_cross_attention = is_cross_attention
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor [batch, seq_len_q, d_model]
            key: Key tensor [batch, seq_len_k, d_model] (None for self-attention)
            value: Value tensor [batch, seq_len_k, d_model] (None for self-attention)
            mask: Attention mask [seq_len_q, seq_len_k] or [batch, seq_len_q, seq_len_k]
            key_padding_mask: Key padding mask [batch, seq_len_k]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Handle self-attention vs cross-attention
        if key is None:
            key = query
        if value is None:
            value = key
        
        batch_size, seq_len_q, d_model = query.shape
        seq_len_k = key.size(1)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, seq_len_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        output, attn_weights = self._scaled_dot_product_attention(
            q, k, v, mask, key_padding_mask
        )
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, d_model)
        output = self.out_proj(output)
        
        return output, attn_weights
    
    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            q: Query tensor [batch, n_heads, seq_len, d_head]
            k: Key tensor [batch, n_heads, seq_len, d_head]
            v: Value tensor [batch, n_heads, seq_len, d_head]
            mask: Attention mask
            key_padding_mask: Key padding mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        d_k = q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(d_k) * self.temperature)
        
        # Apply attention mask
        if mask is not None:
            if mask.dim() == 2:  # [seq_len, seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            elif mask.dim() == 3:  # [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply key padding mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(key_padding_mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Handle NaN values that might occur during softmax
        if torch.isnan(attn_weights).any():
            attn_weights = torch.where(torch.isnan(attn_weights), 
                                     torch.zeros_like(attn_weights), 
                                     attn_weights)
            # Renormalize
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Implements the FFN component of transformer blocks with
    configurable activation and dropout.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        activation: str = 'gelu',
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        
        # Two linear layers with activation in between
        self.linear1 = nn.Linear(d_model, self.d_ff, bias=bias)
        self.linear2 = nn.Linear(self.d_ff, d_model, bias=bias)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish' or activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.constant_(self.linear1.bias, 0.0)
        if self.linear2.bias is not None:
            nn.init.constant_(self.linear2.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through position-wise FFN.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Complete transformer block with multi-head attention and feed-forward network.
    
    Implements pre-layer normalization, residual connections, and dropout
    for stable and effective training.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-6,
        pre_norm: bool = True,
        bias: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.pre_norm = pre_norm
        
        # Multi-head attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias
        )
        
        # Position-wise feed-forward
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            bias=bias
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask [seq_len, seq_len] or [batch, seq_len, seq_len]
            key_padding_mask: Key padding mask [batch, seq_len]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        if self.pre_norm:
            # Pre-layer normalization
            # Self-attention sublayer
            x_norm = self.norm1(x)
            attn_output, _ = self.self_attention(x_norm, mask, key_padding_mask)
            x = x + self.dropout1(attn_output)
            
            # Feed-forward sublayer
            x_norm = self.norm2(x)
            ff_output = self.feed_forward(x_norm)
            x = x + self.dropout2(ff_output)
        else:
            # Post-layer normalization
            # Self-attention sublayer
            attn_output, _ = self.self_attention(x, mask, key_padding_mask)
            x = self.norm1(x + self.dropout1(attn_output))
            
            # Feed-forward sublayer
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for encoder-decoder attention.
    
    Allows decoder to attend to encoder representations
    while maintaining self-attention capabilities.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-6,
        pre_norm: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.pre_norm = pre_norm
        
        # Self-attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Cross-attention
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Feed-forward
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through cross-attention block.
        
        Args:
            x: Decoder input [batch, tgt_len, d_model]
            encoder_output: Encoder output [batch, src_len, d_model]
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
            
        Returns:
            Output tensor [batch, tgt_len, d_model]
        """
        if self.pre_norm:
            # Self-attention sublayer
            x_norm = self.norm1(x)
            self_attn_output, _ = self.self_attention(x_norm, tgt_mask, tgt_key_padding_mask)
            x = x + self.dropout1(self_attn_output)
            
            # Cross-attention sublayer
            x_norm = self.norm2(x)
            # Use x as query, encoder_output as key and value
            cross_attn_output, _ = self.cross_attention._scaled_dot_product_attention(
                x_norm, encoder_output, encoder_output, memory_mask, memory_key_padding_mask
            )
            x = x + self.dropout2(cross_attn_output)
            
            # Feed-forward sublayer
            x_norm = self.norm3(x)
            ff_output = self.feed_forward(x_norm)
            x = x + self.dropout3(ff_output)
        else:
            # Self-attention sublayer
            self_attn_output, _ = self.self_attention(x, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + self.dropout1(self_attn_output))
            
            # Cross-attention sublayer
            cross_attn_output, _ = self.cross_attention._scaled_dot_product_attention(
                x, encoder_output, encoder_output, memory_mask, memory_key_padding_mask
            )
            x = self.norm2(x + self.dropout2(cross_attn_output))
            
            # Feed-forward sublayer
            ff_output = self.feed_forward(x)
            x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class SequenceTransformer(nn.Module):
    """
    Stack of transformer blocks for sequence modeling.
    
    Configurable number of layers with optional positional encoding
    and flexible input/output format handling.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-6,
        pre_norm: bool = True,
        use_positional_encoding: bool = False,
        max_seq_len: int = 5000
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_positional_encoding = use_positional_encoding
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                pre_norm=pre_norm
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm (for pre-norm architecture)
        if pre_norm:
            self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        else:
            self.final_norm = nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer stack.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask
            key_padding_mask: Key padding mask
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask, key_padding_mask)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    
    Adds position information to input embeddings using
    sine and cosine functions of different frequencies.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding table
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding [batch, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x) 


class MultiLayerTransformerBlock(nn.Module):
    """
    Multi-layer Transformer block with enhanced architecture.
    
    Provides deeper processing with multiple stacked Transformer layers,
    improved attention patterns, and better normalization.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True,
        use_relative_position: bool = True,
        max_relative_position: int = 32
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.pre_norm = pre_norm
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])
        
        # Relative positional encoding
        if use_relative_position:
            self.relative_position = RelativePositionalEncoding(
                d_model=d_model,
                max_relative_position=max_relative_position
            )
        else:
            self.relative_position = None
        
        # Final layer norm if using pre-norm
        if pre_norm:
            self.final_norm = nn.LayerNorm(d_model)
        else:
            self.final_norm = nn.Identity()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-layer transformer.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Add relative positional information
        if self.relative_position is not None:
            x = self.relative_position(x)
        
        # Pass through all layers
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for improved attention patterns.
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 32):
        super().__init__()
        
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Learnable relative position embeddings
        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add relative positional encoding to input.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output with relative position information
        """
        batch_size, seq_len, _ = x.shape
        
        # Create relative position indices
        position = torch.arange(seq_len, device=x.device).unsqueeze(0)
        relative_position = position - position.transpose(0, 1)
        
        # Clamp to maximum range
        relative_position = torch.clamp(
            relative_position, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to positive indices
        relative_position = relative_position + self.max_relative_position
        
        # Get embeddings and average over sequence
        rel_embeddings = self.relative_position_embeddings(relative_position)
        rel_embeddings = rel_embeddings.mean(dim=0)  # [seq_len, d_model]
        
        # Add to input
        x = x + rel_embeddings.unsqueeze(0)  # Broadcast to [batch, seq_len, d_model]
        x = self.dropout(x)
        
        return x


class EnhancedMultiHeadAttention(nn.Module):
    """
    Enhanced Multi-Head Attention with improvements for ABR processing.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_rotary: bool = False,
        rotary_dim: Optional[int] = None
    ):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        self.use_rotary = use_rotary
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Rotary positional encoding
        if use_rotary:
            from .positional import RotaryPositionalEmbedding
            rotary_dim = rotary_dim or self.d_k
            self.rotary = RotaryPositionalEmbedding(rotary_dim)
        else:
            self.rotary = None
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.d_k * self.temperature) ** -0.5
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Enhanced attention forward pass.
        
        Args:
            query: Query tensor [batch, seq_len, d_model]
            key: Key tensor [batch, seq_len, d_model]
            value: Value tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Attention output [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply rotary positional encoding
        if self.rotary is not None:
            Q, K = self.rotary(Q, K)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(context)
        
        return output


class DeepTransformerDecoder(nn.Module):
    """
    Deep Transformer decoder for enhanced sequence processing.
    
    Features multiple layers, advanced attention mechanisms, and 
    architectural improvements specifically for ABR signal decoding.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_relative_position: bool = True,
        use_rotary: bool = False,
        cross_attention: bool = False
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.cross_attention = cross_attention
        
        # Main transformer layers
        self.layers = nn.ModuleList([
            DeepTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                use_relative_position=use_relative_position and (i == 0),  # Only first layer
                use_rotary=use_rotary,
                cross_attention=cross_attention
            )
            for i in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Optional position-wise feedforward at the end
        self.final_ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through deep transformer decoder.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            encoder_output: Optional encoder output for cross-attention
            mask: Optional attention mask
            
        Returns:
            Decoded output [batch, seq_len, d_model]
        """
        # Pass through all layers
        for layer in self.layers:
            x = layer(x, encoder_output=encoder_output, mask=mask)
        
        # Final processing
        residual = x
        x = self.final_norm(x)
        x = self.final_ff(x) + residual
        
        return x


class DeepTransformerLayer(nn.Module):
    """Single layer of the deep transformer decoder."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_relative_position: bool = False,
        use_rotary: bool = False,
        cross_attention: bool = False
    ):
        super().__init__()
        
        self.cross_attention = cross_attention
        
        # Self-attention
        self.self_attention = EnhancedMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_rotary=use_rotary
        )
        
        # Cross-attention (if needed)
        if cross_attention:
            self.cross_attention_layer = EnhancedMultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            )
            self.norm_cross = nn.LayerNorm(d_model)
        
        # Feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Relative positional encoding
        if use_relative_position:
            self.relative_position = RelativePositionalEncoding(d_model)
        else:
            self.relative_position = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through transformer layer."""
        
        # Add relative positional encoding
        if self.relative_position is not None:
            x = self.relative_position(x)
        
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, x, x, mask=mask)
        x = self.dropout(x) + residual
        
        # Cross-attention
        if self.cross_attention and encoder_output is not None:
            residual = x
            x = self.norm_cross(x)
            x = self.cross_attention_layer(x, encoder_output, encoder_output)
            x = self.dropout(x) + residual
        
        # Feedforward
        residual = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = x + residual
        
        return x


def get_activation(name: str):
    """Get activation function by name."""
    if name == 'gelu':
        return nn.GELU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'silu' or name == 'swish':
        return nn.SiLU()
    elif name == 'mish':
        return nn.Mish()
    else:
        return nn.GELU() 


class CrossAttentionTransformerBlock(nn.Module):
    """
    Transformer block with cross-attention support for encoder-decoder architecture.
    
    Features:
    - Self-attention on decoder features
    - Cross-attention to encoder features  
    - Position-wise feed-forward network
    - Residual connections and layer normalization
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_pre_norm: bool = True,
        use_relative_position: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_pre_norm = use_pre_norm
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Self-attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            is_cross_attention=False
        )
        
        # Cross-attention
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            is_cross_attention=True
        )
        
        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Relative position encoding (optional)
        if use_relative_position:
            self.relative_position = RelativePositionEncoding(d_model, n_heads)
        else:
            self.relative_position = None
    
    def forward(
        self,
        decoder_input: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through cross-attention transformer block.
        
        Args:
            decoder_input: Decoder features [batch, seq_len_decoder, d_model]
            encoder_output: Encoder features [batch, seq_len_encoder, d_model]
            self_attn_mask: Self-attention mask
            cross_attn_mask: Cross-attention mask
            key_padding_mask: Key padding mask
            
        Returns:
            Tuple of (output, self_attn_weights, cross_attn_weights)
        """
        # Self-attention
        if self.use_pre_norm:
            # Pre-normalization
            normed_input = self.norm1(decoder_input)
            self_attn_output, self_attn_weights = self.self_attention(
                query=normed_input,
                mask=self_attn_mask
            )
            decoder_input = decoder_input + self.dropout(self_attn_output)
        else:
            # Post-normalization
            self_attn_output, self_attn_weights = self.self_attention(
                query=decoder_input,
                mask=self_attn_mask
            )
            decoder_input = self.norm1(decoder_input + self.dropout(self_attn_output))
        
        # Cross-attention
        if self.use_pre_norm:
            # Pre-normalization
            normed_input = self.norm2(decoder_input)
            cross_attn_output, cross_attn_weights = self.cross_attention(
                query=normed_input,
                key=encoder_output,
                value=encoder_output,
                mask=cross_attn_mask,
                key_padding_mask=key_padding_mask
            )
            decoder_input = decoder_input + self.dropout(cross_attn_output)
        else:
            # Post-normalization
            cross_attn_output, cross_attn_weights = self.cross_attention(
                query=decoder_input,
                key=encoder_output,
                value=encoder_output,
                mask=cross_attn_mask,
                key_padding_mask=key_padding_mask
            )
            decoder_input = self.norm2(decoder_input + self.dropout(cross_attn_output))
        
        # Feed-forward network
        if self.use_pre_norm:
            # Pre-normalization
            normed_input = self.norm3(decoder_input)
            ff_output = self.feed_forward(normed_input)
            output = decoder_input + self.dropout(ff_output)
        else:
            # Post-normalization
            ff_output = self.feed_forward(decoder_input)
            output = self.norm3(decoder_input + self.dropout(ff_output))
        
        return output, self_attn_weights, cross_attn_weights


class EnhancedTransformerBlock(nn.Module):
    """
    Enhanced transformer block with multi-scale attention for ABR processing.
    
    Features:
    - Multi-scale self-attention for different temporal patterns
    - Optional cross-attention for encoder-decoder interaction
    - Enhanced positional encoding
    - Gated feed-forward network
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_multi_scale: bool = True,
        use_cross_attention: bool = False,
        use_gated_ffn: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_multi_scale = use_multi_scale
        self.use_cross_attention = use_cross_attention
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Multi-scale self-attention
        if use_multi_scale:
            self.multi_scale_attention = MultiScaleAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                scales=[1, 3, 5]  # Different attention scales
            )
        else:
            self.self_attention = MultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            )
        
        # Cross-attention (optional)
        if use_cross_attention:
            self.cross_attention = MultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                is_cross_attention=True
            )
        
        # Enhanced feed-forward network
        if use_gated_ffn:
            self.feed_forward = GatedFeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            )
        else:
            self.feed_forward = PositionwiseFeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        if use_cross_attention:
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        else:
            self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through enhanced transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            encoder_output: Encoder output for cross-attention [batch, enc_len, d_model]
            mask: Self-attention mask
            cross_mask: Cross-attention mask
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Multi-scale self-attention
        if self.use_multi_scale:
            attn_output = self.multi_scale_attention(x, mask=mask)
        else:
            attn_output, _ = self.self_attention(x, mask=mask)
        
        # Residual connection and normalization
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention (if enabled and encoder output provided)
        if self.use_cross_attention and encoder_output is not None:
            cross_attn_output, _ = self.cross_attention(
                query=x,
                key=encoder_output,
                value=encoder_output,
                mask=cross_mask
            )
            x = self.norm2(x + self.dropout(cross_attn_output))
            norm_idx = 3
        else:
            norm_idx = 2
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        
        # Final residual connection and normalization
        if norm_idx == 3:
            x = self.norm3(x + self.dropout(ff_output))
        else:
            x = self.norm2(x + self.dropout(ff_output))
        
        return x 


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism for capturing patterns at different temporal scales.
    
    This is particularly useful for ABR signals where peaks and morphology
    exist at different time scales.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        scales: List[int] = [1, 3, 5, 7]
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        self.num_scales = len(scales)
        
        # Ensure we can divide heads among scales AND that d_model is divisible by heads_per_scale
        valid_config_found = False
        for possible_scales in [self.num_scales, 4, 2, 1]:  # Try original first, then common divisors
            if n_heads % possible_scales == 0:
                heads_per_scale = n_heads // possible_scales
                if d_model % heads_per_scale == 0:  # Also check d_model divisibility
                    self.num_scales = possible_scales
                    self.scales = scales[:possible_scales]
                    self.heads_per_scale = heads_per_scale
                    valid_config_found = True
                    break
        
        if not valid_config_found:
            # Ultimate fallback: single scale with adjusted heads
            self.num_scales = 1
            self.scales = [scales[0]]
            # Find the largest divisor of d_model that's <= n_heads
            for h in range(min(n_heads, d_model), 0, -1):
                if d_model % h == 0:
                    self.heads_per_scale = h
                    break
            else:
                self.heads_per_scale = 1  # Final fallback
        
        # Ensure num_scales matches the actual number of scales we're using
        self.num_scales = len(self.scales)
        
        # Multi-scale attention heads
        self.scale_attentions = nn.ModuleList([
            MultiHeadAttention(
                d_model=d_model,
                n_heads=self.heads_per_scale,
                dropout=dropout
            )
            for _ in self.scales
        ])
        
        # Scale-specific 1D convolutions for multi-scale processing
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=scale, padding=scale//2, groups=d_model)
            for scale in self.scales
        ])
        
        # Fusion mechanism
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * self.num_scales, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-scale attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply multi-scale processing
        scale_outputs = []
        
        for scale_conv, scale_attention in zip(self.scale_convs, self.scale_attentions):
            # Apply scale-specific convolution
            x_conv = x.transpose(1, 2)  # [batch, d_model, seq_len]
            x_conv = scale_conv(x_conv)
            x_conv = x_conv.transpose(1, 2)  # [batch, seq_len, d_model]
            
            # Apply attention at this scale
            x_attn, _ = scale_attention(x_conv, mask=mask)
            scale_outputs.append(x_attn)
        
        # Fuse multi-scale features
        fused_features = torch.cat(scale_outputs, dim=-1)  # [batch, seq_len, d_model * num_scales]
        fused_output = self.scale_fusion(fused_features)  # [batch, seq_len, d_model]
        
        # Final projection
        output = self.output_proj(fused_output)
        
        return output


class GatedFeedForward(nn.Module):
    """
    Gated feed-forward network with enhanced expressivity.
    
    Uses gating mechanism similar to GLU (Gated Linear Units)
    for better information flow control.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Gated linear layers
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.value_proj = nn.Linear(d_model, d_ff)
        self.output_proj = nn.Linear(d_ff, d_model)
        
        # Activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()  # Default
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        for module in [self.gate_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through gated feed-forward network.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Compute gate and value
        gate = self.activation(self.gate_proj(x))  # [batch, seq_len, d_ff]
        value = self.value_proj(x)  # [batch, seq_len, d_ff]
        
        # Apply gating
        gated_value = gate * value  # Element-wise multiplication
        
        # Apply dropout
        gated_value = self.dropout(gated_value)
        
        # Output projection
        output = self.output_proj(gated_value)
        
        return output


class RelativePositionEncoding(nn.Module):
    """
    Relative positional encoding for transformer attention.
    
    Adds relative position information to attention scores
    instead of absolute positions.
    """
    
    def __init__(self, d_model: int, n_heads: int, max_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len = max_len
        
        # Relative position embeddings
        self.relative_positions = nn.Parameter(
            torch.randn(2 * max_len - 1, d_model // n_heads)
        )
        
        # Initialize
        nn.init.xavier_uniform_(self.relative_positions)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate relative position encodings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position encodings [seq_len, seq_len, d_head]
        """
        device = self.relative_positions.device
        
        # Create relative position indices
        positions = torch.arange(seq_len, device=device)
        relative_indices = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
        relative_indices = relative_indices + self.max_len - 1  # Shift to positive indices
        
        # Clamp to valid range
        relative_indices = torch.clamp(relative_indices, 0, 2 * self.max_len - 2)
        
        # Get relative position embeddings
        relative_encodings = self.relative_positions[relative_indices]  # [seq_len, seq_len, d_head]
        
        return relative_encodings 