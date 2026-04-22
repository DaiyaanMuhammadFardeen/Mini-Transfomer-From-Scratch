import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PerformanceEmbedding(nn.Module):
    """
    Performance Embedding Layer that captures algorithm complexity,
    resource usage, and performance annotations.
    """
    
    def __init__(self, d_model: int):
        super(PerformanceEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # Algorithm complexity embedding
        self.algo_complexity_embedding = nn.Embedding(20, self.half_d_model)  # 20 different complexity classes
        
        # Resource usage embedding
        self.resource_usage_embedding = nn.Embedding(50, self.half_d_model)  # 50 different resource usage patterns
        
        # Performance annotation embedding
        self.perf_annotation_embedding = nn.Embedding(30, self.half_d_model)  # 30 different perf annotations
        
        # Performance feature extractor for continuous features
        self.perf_feature_extractor = nn.Sequential(
            nn.Linear(12, d_model),  # 12 performance-related continuous features
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Performance attention mechanism
        self.perf_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        
        # Performance combination layer
        self.perf_combination = nn.Linear(d_model * 3, d_model)
        
        # Final projection
        self.projection = nn.Linear(d_model * 2, d_model)  # For combining discrete and continuous features
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, perf_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for performance embedding.
        
        Args:
            perf_features: Performance features tensor (batch_size, seq_len, num_perf_features)
            
        Returns:
            Performance embedding tensor (batch_size, seq_len, d_model)
        """
        if perf_features is None:
            # If no performance features provided, return zero embeddings
            batch_size = 1
            seq_len = 10  # Default sequence length
            device = next(self.parameters()).device
            return torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=torch.float)
        
        batch_size, seq_len, num_features = perf_features.shape
        device = perf_features.device
        
        # Extract different performance features
        # Assume perf_features has shape (batch_size, seq_len, num_features) where:
        # 0: algorithm complexity ID
        # 1: resource usage ID
        # 2: performance annotation ID
        # 3-14: performance-related continuous features (12 features)
        
        # Algorithm Complexity Embedding
        if num_features > 0:
            algo_ids = perf_features[:, :, 0].long().clamp(0, 19)
            algo_emb = self.algo_complexity_embedding(algo_ids)  # (batch_size, seq_len, half_d_model)
            algo_emb = F.pad(algo_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            algo_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Resource Usage Embedding
        if num_features > 1:
            resource_ids = perf_features[:, :, 1].long().clamp(0, 49)
            resource_emb = self.resource_usage_embedding(resource_ids)  # (batch_size, seq_len, half_d_model)
            resource_emb = F.pad(resource_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            resource_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Performance Annotation Embedding
        if num_features > 2:
            annotation_ids = perf_features[:, :, 2].long().clamp(0, 29)
            annotation_emb = self.perf_annotation_embedding(annotation_ids)  # (batch_size, seq_len, half_d_model)
            annotation_emb = F.pad(annotation_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            annotation_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Combine discrete performance embeddings
        combined_discrete = torch.cat([algo_emb, resource_emb, annotation_emb], dim=-1)  # (batch_size, seq_len, d_model * 3)
        combined_discrete = self.perf_combination(combined_discrete)  # (batch_size, seq_len, d_model)
        
        # Process continuous performance features
        if num_features > 14:
            continuous_features = perf_features[:, :, 3:15]  # Extract 12 continuous features
        elif num_features > 3:
            continuous_features = perf_features[:, :, 3:]  # Extract available continuous features
            # Pad if needed
            if continuous_features.size(2) < 12:
                pad_size = 12 - continuous_features.size(2)
                continuous_features = F.pad(continuous_features, (0, pad_size))
            elif continuous_features.size(2) > 12:
                continuous_features = continuous_features[:, :, :12]
        else:
            continuous_features = torch.zeros(batch_size, seq_len, 12, device=device, dtype=torch.float)
        
        combined_continuous = self.perf_feature_extractor(continuous_features)  # (batch_size, seq_len, d_model)
        
        # Apply performance attention to the discrete embeddings
        attended_discrete, _ = self.perf_attention(
            combined_discrete, combined_discrete, combined_discrete
        )
        combined_discrete = combined_discrete + attended_discrete  # Residual connection
        
        # Combine discrete and continuous performance features
        combined_perf = torch.cat([combined_discrete, combined_continuous], dim=-1)  # (batch_size, seq_len, d_model * 2)
        output = self.projection(combined_perf)  # (batch_size, seq_len, d_model)
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        return output