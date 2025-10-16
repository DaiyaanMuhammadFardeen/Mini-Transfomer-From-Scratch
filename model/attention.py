import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e4)
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

class KernelizedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_causal_mask=False):
        super(KernelizedMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_causal_mask = use_causal_mask

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def feature_map(self, x):
        """ELU + 1 kernel approximation"""
        return (F.elu(x) + 1) / math.sqrt(self.d_k)

    def linear_attention(self, Q, K, V, mask=None):
        """
        Linear attention with O(n) complexity instead of O(n^2)
        Q, K, V shapes: (b, h, n, d_k)
        """
        b, h, n_q, d_k = Q.shape
        n_k = K.shape[2]
        
        phi_Q = self.feature_map(Q)  # (b, h, n_q, d_k)
        phi_K = self.feature_map(K)  # (b, h, n_k, d_k)

        if self.use_causal_mask:
            # Causal case: use cumulative sums
            # Compute cumulative KV: sum of (phi_K[i] * V[i]) for all i <= current position
            # Shape: (b, h, n_k, d_k, d_k) but we want (b, h, n_k, d_k)
            kv_product = torch.einsum('bhnd,bhne->bhde', phi_K, V)  # (b, h, d_k, d_k)
            k_sum = phi_K.sum(dim=2, keepdim=True)  # (b, h, 1, d_k)
            
            # For causal attention, compute for each query position up to that point
            output = []
            for i in range(n_q):
                # Sum of keys and values up to position i
                kv_i = torch.einsum('bhnd,bhne->bhde', phi_K[:, :, :i+1, :], V[:, :, :i+1, :])  # (b, h, d_k, d_k)
                k_sum_i = phi_K[:, :, :i+1, :].sum(dim=2)  # (b, h, d_k)
                
                # Query at position i
                numer = torch.matmul(phi_Q[:, :, i:i+1, :], kv_i)  # (b, h, 1, d_k)
                denom = torch.matmul(phi_Q[:, :, i:i+1, :], k_sum_i.unsqueeze(-1)) + 1e-3  # (b, h, 1, 1)
                
                output.append(numer / denom)
            
            return torch.cat(output, dim=2)  # (b, h, n_q, d_k)
        else:
            # Non-causal case: full attention over all keys
            kv_product = torch.einsum('bhnd,bhne->bhde', phi_K, V)  # (b, h, d_k, d_k)
            k_sum = phi_K.sum(dim=2, keepdim=True)  # (b, h, 1, d_k)
            
            numer = torch.matmul(phi_Q, kv_product)  # (b, h, n_q, d_k)
            denom = torch.matmul(phi_Q, k_sum.transpose(-1, -2)) + 1e-3  # (b, h, n_q, 1)
            
            return numer / denom

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

        attn_output = self.linear_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
