"""
Lean diff embedding that combines token embedding and change-type signal.
Positional encoding is handled by RoPE in the attention mechanism.
Replaces the complex MultimodalEmbedding system with a simple, efficient implementation.
"""
import math
import torch
import torch.nn as nn
from .ChangeTypeEmbedding import ChangeTypeEmbedding


class DiffEmbedding(nn.Module):
    """
    Encoder embedding for diff text.

    Combines:
      1. Token embedding  — learns representations for BPE tokens
      2. ChangeTypeEmbedding — a lightweight learned bias conditioned on
         which structural change tags (<ADD>, <REMOVE>, etc.) are present

    Note: Positional encoding is handled by RoPE in MultiHeadAttention,
    so we don't add it here. The change-type bias is added on top of 
    the token embedding, giving every token a global context about what 
    kind of change this diff is.
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = None,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.change_type_emb = ChangeTypeEmbedding(d_model, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids: torch.Tensor,
                change_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids:       (batch, seq_len) long tensor
            change_features: (batch, 6) float tensor from CodeDiffDataset

        Returns:
            (batch, seq_len, d_model) encoder input tensor
        """
        # Standard token embedding scaled by sqrt(d_model) per Vaswani et al.
        x = self.token_emb(token_ids) * math.sqrt(self.d_model)

        # Add change-type bias if provided — broadcast across all positions
        if change_features is not None:
            seq_len = token_ids.size(1)
            change_bias = self.change_type_emb(change_features, seq_len)
            x = x + change_bias

        return self.dropout(x)
