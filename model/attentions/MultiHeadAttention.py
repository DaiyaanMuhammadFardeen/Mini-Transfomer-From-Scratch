import torch
import torch.nn as nn
import math
from ..positionalEnc import RotaryPositionalEncoding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible bynum_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Initialize rotary positional encoding with correct head dimension
        self.rotary_encoding = RotaryPositionalEncoding(head_dim=self.d_k)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        seq_len = Q.size(2)
        # Vectorized RoPE
        K = self.rotary_encoding.apply_rotary_emb(K, K.shape[2])
        Q = self.rotary_encoding.apply_rotary_emb(Q, Q.shape[2])
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask_value = -torch.finfo(attn_scores.dtype).max
            attn_scores = attn_scores.masked_fill(mask == 0, mask_value)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
