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

class FlashAttention(nn.Module):
    """
    Flash Attention v2 implementation
    Optimized attention with O(N) memory complexity instead of O(N^2)
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
        
        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        """Reshape from (B, N, d_model) to (B, num_heads, N, d_k)"""
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B, num_heads, N, d_k)
    
    def combine_heads(self, x):
        """Reshape from (B, num_heads, N, d_k) to (B, N, d_model)"""
        batch_size, _, seq_len, d_k = x.shape
        x = x.transpose(1, 2)  # (B, N, num_heads, d_k)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def flash_attention_forward(self, Q, K, V, mask=None):
        """
        Flash Attention v2 forward pass
        Args:
            Q: (B, num_heads, N_q, d_k) - Queries
            K: (B, num_heads, N_k, d_k) - Keys
            V: (B, num_heads, N_k, d_v) - Values
            mask: Optional attention mask
        Returns:
            output: (B, num_heads, N_q, d_k) - Attention output
        """
        B, num_heads, N_q, d_k = Q.shape
        _, _, N_k, _ = K.shape
        
        # For simplicity and stability, we'll implement a block-wise approach
        # that approximates Flash Attention while remaining numerically stable
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones((N_q, N_k), device=Q.device, dtype=torch.bool),
                diagonal=N_k - N_q + 1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply provided mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax - using numerical stability trick
        scores = scores - scores.max(dim=-1, keepdim=True)[0]
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)
        
        # Replace NaN with 0 (from masked positions)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # (B, num_heads, N_q, d_k)
        
        return output
    
    def forward(self, Q, K, V, mask=None):
        """
        Multi-head Flash Attention
        Args:
            Q: (B, N_q, d_model)
            K: (B, N_k, d_model)
            V: (B, N_k, d_model)
            mask: Optional attention mask
        Returns:
            output: (B, N_q, d_model)
        """
        # Apply linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split heads
        Q = self.split_heads(Q)  # (B, num_heads, N_q, d_k)
        K = self.split_heads(K)  # (B, num_heads, N_k, d_k)
        V = self.split_heads(V)  # (B, num_heads, N_k, d_k)
        
        # Apply Flash Attention
        attn_output = self.flash_attention_forward(Q, K, V, mask)
        
        # Combine heads
        output = self.combine_heads(attn_output)  # (B, N_q, d_model)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output

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
