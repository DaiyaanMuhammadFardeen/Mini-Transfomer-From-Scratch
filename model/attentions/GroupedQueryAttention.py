# model/attentions/GroupedQueryAttention.py
import torch
import torch.nn as nn
import math
from ..positionalEnc import RotaryPositionalEncoding


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) — Ainslie et al. 2023.
    num_kv_heads << num_heads: K and V have fewer heads than Q.
    Each KV head is shared across (num_heads // num_kv_heads) query heads.

    Memory: O(batch × seq × d_model × num_kv_heads/num_heads) for KV cache.
    Quality: Essentially identical to MHA for generation tasks.
    """

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int = None,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads or max(1, num_heads // 4)
        assert num_heads % self.num_kv_heads == 0
        self.kv_groups    = num_heads // self.num_kv_heads

        self.d_model = d_model
        self.d_k     = d_model // num_heads
        self.dropout = dropout

        self.W_q = nn.Linear(d_model, num_heads    * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)

        self.rotary = RotaryPositionalEncoding(head_dim=self.d_k)

    def forward(self, Q_in, K_in, V_in, mask=None):
        B, S, _ = Q_in.shape
        T        = K_in.shape[1]

        # Project
        Q = self.W_q(Q_in).view(B, S, self.num_heads,    self.d_k).transpose(1, 2)
        K = self.W_k(K_in).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V_in).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        # RoPE on Q and K
        Q = self.rotary.apply_rotary_emb(Q, S)
        K = self.rotary.apply_rotary_emb(K, T)

        # Expand K, V to match num_heads by repeating each KV head kv_groups times
        K = K.repeat_interleave(self.kv_groups, dim=1)  # (B, num_heads, T, d_k)
        V = V.repeat_interleave(self.kv_groups, dim=1)

        # Scaled dot-product attention (uses Flash Attention backend in PyTorch 2.x)
        attn_mask = None
        if mask is not None:
            # Convert bool mask to additive float mask for sdpa
            attn_mask = torch.zeros(mask.shape, dtype=Q.dtype, device=Q.device)
            attn_mask = attn_mask.masked_fill(mask == 0, float('-inf'))

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False   # mask handles causality
        )

        # Merge heads
        out = attn_out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.d_k)
        return self.W_o(out)
