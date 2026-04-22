import torch
import torch.nn as nn
import math

class FlashAttentionOptimized(nn.Module):
    """
    Flash Attention v2 with block-wise computation for better memory efficiency
    Use this variant for very long sequences (>2048 tokens)
    """
    def __init__(self, d_model, num_heads, dropout=0.0, causal=False, block_size=64):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal
        self.dropout = dropout
        self.block_size = block_size  # Number of queries/keys per block
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.shape
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def flash_attention_block(self, Q, K, V, mask=None):
        """Block-wise Flash Attention computation"""
        B, num_heads, N_q, d_k = Q.shape
        _, _, N_k, _ = K.shape
        
        # Initialize output and normalization factors
        output = torch.zeros_like(Q)
        
        # Process queries in blocks
        for q_start in range(0, N_q, self.block_size):
            q_end = min(q_start + self.block_size, N_q)
            Q_block = Q[:, :, q_start:q_end, :]  # (B, num_heads, block_size, d_k)
            
            # Compute attention for this query block over all keys
            scores = torch.matmul(Q_block, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            # Apply causal mask if needed
            if self.causal:
                causal_mask = torch.triu(
                    torch.ones((q_end - q_start, N_k), device=Q.device, dtype=torch.bool),
                    diagonal=N_k - (q_end - q_start) + 1
                )
                scores = scores.masked_fill(causal_mask, float('-inf'))
            
            # Apply provided mask
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            # Numerically stable softmax
            scores = scores - scores.max(dim=-1, keepdim=True)[0]
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_weights = torch.nan_to_num(attn_weights, 0.0)
            
            # Apply attention to values
            output[:, :, q_start:q_end, :] = torch.matmul(attn_weights, V)
        
        return output
    
    def forward(self, Q, K, V, mask=None):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        attn_output = self.flash_attention_block(Q, K, V, mask)
        
        output = self.combine_heads(attn_output)
        output = self.W_o(output)
        
        return output
