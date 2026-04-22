import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChangeTypeEmbedding(nn.Module):
    """
    Change Type Embedding Layer that captures bug fixes, feature additions,
    refactoring, and optimization changes.
    """
    
    def __init__(self, d_model: int):
        super(ChangeTypeEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # Change type embedding
        self.change_type_embedding = nn.Embedding(10, self.half_d_model)  # 10 different change types
        
        # Change classification embedding
        self.change_class_embedding = nn.Embedding(8, self.half_d_model)  # 8 different change classifications
        
        # Change-specific feature extractor for continuous features
        self.change_feature_extractor = nn.Sequential(
            nn.Linear(12, d_model),  # 12 change-related continuous features
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Change type combination layer
        self.change_combination = nn.Linear(d_model * 2, d_model)
        
        # Change attention mechanism
        self.change_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        
        # Final projection
        self.projection = nn.Linear(d_model * 2, d_model)  # For combining discrete and continuous features
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, change_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for change type embedding.
        
        Args:
            change_features: Change features tensor (batch_size, seq_len, num_change_features)
            
        Returns:
            Change type embedding tensor (batch_size, seq_len, d_model)
        """
        if change_features is None:
            # If no change features provided, return zero embeddings
            batch_size = 1
            seq_len = 10  # Default sequence length
            device = next(self.parameters()).device
            return torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=torch.float)
        
        batch_size, seq_len, num_features = change_features.shape
        device = change_features.device
        
        # Extract different change features
        # Assume change_features has shape (batch_size, seq_len, num_features) where:
        # 0: change type ID (bug fix, feature add, etc.)
        # 1: change classification ID (ADDITION, MODIFICATION, etc.)
        # 2-13: change-related continuous features (12 features)
        
        # Change Type Embedding
        if num_features > 0:
            change_type_ids = change_features[:, :, 0].long().clamp(0, 9)
            change_type_emb = self.change_type_embedding(change_type_ids)  # (batch_size, seq_len, half_d_model)
            change_type_emb = F.pad(change_type_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            change_type_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Change Classification Embedding
        if num_features > 1:
            change_class_ids = change_features[:, :, 1].long().clamp(0, 7)
            change_class_emb = self.change_class_embedding(change_class_ids)  # (batch_size, seq_len, half_d_model)
            change_class_emb = F.pad(change_class_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            change_class_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Combine discrete change embeddings
        combined_discrete = change_type_emb + change_class_emb
        
        # Process continuous change features
        if num_features > 13:
            continuous_features = change_features[:, :, 2:14]  # Extract 12 continuous features
        elif num_features > 2:
            continuous_features = change_features[:, :, 2:]  # Extract available continuous features
            # Pad if needed
            if continuous_features.size(2) < 12:
                pad_size = 12 - continuous_features.size(2)
                continuous_features = F.pad(continuous_features, (0, pad_size))
            elif continuous_features.size(2) > 12:
                continuous_features = continuous_features[:, :, :12]
        else:
            continuous_features = torch.zeros(batch_size, seq_len, 12, device=device, dtype=torch.float)
        
        combined_continuous = self.change_feature_extractor(continuous_features)  # (batch_size, seq_len, d_model)
        
        # Apply change attention to the discrete embeddings
        attended_discrete, _ = self.change_attention(
            combined_discrete, combined_discrete, combined_discrete
        )
        combined_discrete = combined_discrete + attended_discrete  # Residual connection
        
        # Combine discrete and continuous change features
        combined_changes = torch.cat([combined_discrete, combined_continuous], dim=-1)  # (batch_size, seq_len, d_model * 2)
        output = self.projection(combined_changes)  # (batch_size, seq_len, d_model)
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        return output