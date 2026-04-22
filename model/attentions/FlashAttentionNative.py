import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# For PyTorch 2.1+ users - use native Flash Attention
class FlashAttentionNative(nn.Module):
    """
    Wrapper around PyTorch's native Flash Attention (2.1+)
    Most efficient implementation - uses CUDA kernel
    """
    def __init__(self, d_model, num_heads, dropout=0.0, causal=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal
        self.dropout = dropout

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.shape
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Use PyTorch's native scaled_dot_product_attention (Flash Attention backend)
        # Available in PyTorch 2.0+
        try:
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal
            )
        except AttributeError:
            # Fallback if native implementation not available
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if self.causal:
                seq_len = Q.shape[2]
                causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=Q.device), diagonal=1).bool()
                scores = scores.masked_fill(causal_mask, float('-inf'))
            scores = torch.softmax(scores, dim=-1)
            scores = F.dropout(scores, p=self.dropout, training=self.training)
            attn_output = torch.matmul(scores, V)

        output = self.combine_heads(attn_output)
        output = self.W_o(output)

        return output
