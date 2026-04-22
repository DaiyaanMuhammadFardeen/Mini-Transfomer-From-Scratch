import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class FusionMechanism(nn.Module):
    """
    Fusion Mechanism for combining multiple embedding layers using
    learnable weights.
    """
    
    def __init__(
        self, 
        num_embeddings: int, 
        d_model: int, 
        fusion_method: str = 'learnable_weights'
    ):
        super(FusionMechanism, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.d_model = d_model
        self.fusion_method = fusion_method
        
        if fusion_method == 'learnable_weights':
            # Learnable weight fusion - default approach
            self.weights = nn.Parameter(torch.ones(num_embeddings) / num_embeddings)
        elif fusion_method == 'gated_fusion':
            # Gated fusion mechanism
            self.gate_weights = nn.Linear(d_model * num_embeddings, num_embeddings)
            self.projection = nn.Linear(d_model, d_model)
        else:
            # Simple weighted sum
            self.weights = nn.Parameter(torch.ones(num_embeddings) / num_embeddings)
        
        # Final projection layer
        self.final_projection = nn.Linear(d_model, d_model)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for fusion mechanism.
        
        Args:
            embeddings: List of embedding tensors, each with shape (batch_size, seq_len, d_model)
            
        Returns:
            Fused embedding tensor (batch_size, seq_len, d_model)
        """
        if not embeddings:
            raise ValueError("Embeddings list cannot be empty")
        
        batch_size, seq_len, d_model = embeddings[0].shape
        
        # Ensure all embeddings have the same shape
        for i, emb in enumerate(embeddings):
            if emb.shape != (batch_size, seq_len, d_model):
                raise ValueError(f"Embedding {i} has shape {emb.shape}, expected {(batch_size, seq_len, d_model)}")
        
        if self.fusion_method == 'learnable_weights':
            # Simple weighted sum with learnable weights
            weights = F.softmax(self.weights, dim=0)  # Normalize weights
            fused = sum(emb * weight for emb, weight in zip(embeddings, weights))
            
        elif self.fusion_method == 'gated_fusion':
            # Gated fusion mechanism
            # Concatenate all embeddings along the feature dimension
            concat_embeddings = torch.cat(embeddings, dim=-1)  # (batch_size, seq_len, d_model * num_embeddings)
            
            # Calculate gate values
            gates = torch.sigmoid(self.gate_weights(concat_embeddings))  # (batch_size, seq_len, num_embeddings)
            
            # Apply gates to each embedding
            fused = sum(emb * gates[:, :, i:i+1] for i, emb in enumerate(embeddings))
            
            # Apply projection
            fused = self.projection(fused)
        else:
            # Default to simple weighted sum
            weights = F.softmax(self.weights, dim=0)  # Normalize weights
            fused = sum(emb * weight for emb, weight in zip(embeddings, weights))
        
        # Apply normalization and dropout
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        # Final projection
        output = self.final_projection(fused)
        output = self.norm(output)
        
        return output