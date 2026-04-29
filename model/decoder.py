import torch
import torch.nn as nn
from .attentions.MultiHeadAttention import MultiHeadAttention
from .positionalFF import SwiGLU

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, drop_path_rate: float = 0.1):
        super(DecoderLayer, self).__init__()
        # Self-attention: standard MHA for AMD GPU compatibility
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Cross-attention: standard MHA for AMD GPU compatibility
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = SwiGLU(d_model, d_ff, bias=False)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.drop_path_rate = drop_path_rate
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Stochastic depth: skip this layer entirely with probability drop_path_rate
        if self.training and torch.rand(1).item() < self.drop_path_rate:
            return x   # Skip layer, pass residual unchanged

        # Pre-norm: normalize BEFORE each sublayer
        x_norm = self.norm1(x)
        attn_output = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout(attn_output)

        x_norm = self.norm2(x)
        attn_output = self.cross_attn(x_norm, enc_output, enc_output, src_mask)
        x = x + self.dropout(attn_output)

        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x
