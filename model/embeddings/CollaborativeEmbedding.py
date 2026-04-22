import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CollaborativeEmbedding(nn.Module):
    """
    Collaborative Embedding Layer that captures author, team, reviewer,
    and ownership context.
    """
    
    def __init__(self, d_model: int):
        super(CollaborativeEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # Author Embedding
        self.author_embedding = nn.Embedding(1000, self.half_d_model)  # 1000 different authors
        
        # Team Embedding
        self.team_embedding = nn.Embedding(100, self.half_d_model)  # 100 different teams
        
        # Reviewer Embedding
        self.reviewer_embedding = nn.Embedding(500, self.half_d_model)  # 500 different reviewers
        
        # Ownership Embedding
        self.ownership_embedding = nn.Embedding(200, self.half_d_model)  # 200 different ownership contexts
        
        # Style pattern embedding for coding style
        self.style_pattern_embedding = nn.Linear(15, self.half_d_model)  # 15 style-related features
        
        # Naming convention embedding
        self.naming_embedding = nn.Embedding(50, self.half_d_model)  # 50 different naming patterns
        
        # Collaboration combination layer
        self.collab_combination = nn.Linear(d_model * 4 + self.half_d_model, d_model)
        
        # Attention mechanism for collaborative context
        self.collab_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        
        # Final projection
        self.projection = nn.Linear(d_model, d_model)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, collab_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for collaborative embedding.
        
        Args:
            collab_features: Collaborative features tensor (batch_size, seq_len, num_collab_features)
            
        Returns:
            Collaborative embedding tensor (batch_size, seq_len, d_model)
        """
        if collab_features is None:
            # If no collaborative features provided, return zero embeddings
            batch_size = 1
            seq_len = 10  # Default sequence length
            device = next(self.parameters()).device
            return torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=torch.float)
        
        batch_size, seq_len, num_features = collab_features.shape
        device = collab_features.device
        
        # Extract different collaborative features
        # Assume collab_features has shape (batch_size, seq_len, num_features) where:
        # 0: author ID
        # 1: team ID
        # 2: reviewer ID
        # 3: ownership ID
        # 4-18: style-related continuous features (15 features)
        # 19: naming convention ID
        
        # Author Embedding
        if num_features > 0:
            author_ids = collab_features[:, :, 0].long().clamp(0, 999)
            author_emb = self.author_embedding(author_ids)  # (batch_size, seq_len, half_d_model)
            author_emb = F.pad(author_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            author_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Team Embedding
        if num_features > 1:
            team_ids = collab_features[:, :, 1].long().clamp(0, 99)
            team_emb = self.team_embedding(team_ids)  # (batch_size, seq_len, half_d_model)
            team_emb = F.pad(team_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            team_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Reviewer Embedding
        if num_features > 2:
            reviewer_ids = collab_features[:, :, 2].long().clamp(0, 499)
            reviewer_emb = self.reviewer_embedding(reviewer_ids)  # (batch_size, seq_len, half_d_model)
            reviewer_emb = F.pad(reviewer_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            reviewer_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Ownership Embedding
        if num_features > 3:
            ownership_ids = collab_features[:, :, 3].long().clamp(0, 199)
            ownership_emb = self.ownership_embedding(ownership_ids)  # (batch_size, seq_len, half_d_model)
            ownership_emb = F.pad(ownership_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            ownership_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Style pattern embedding
        if num_features > 18:
            style_features = collab_features[:, :, 4:19]  # Extract 15 style features
            style_emb = self.style_pattern_embedding(style_features)  # (batch_size, seq_len, half_d_model)
            style_emb = F.pad(style_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            style_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Naming convention embedding
        if num_features > 19:
            naming_ids = collab_features[:, :, 19].long().clamp(0, 49)
            naming_emb = self.naming_embedding(naming_ids)  # (batch_size, seq_len, half_d_model)
            naming_emb = F.pad(naming_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            naming_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Combine all collaborative embeddings
        combined_collab = author_emb + team_emb + reviewer_emb + ownership_emb + style_emb + naming_emb
        
        # Apply attention mechanism for collaborative context
        attended_collab, _ = self.collab_attention(
            combined_collab, combined_collab, combined_collab
        )
        combined_collab = combined_collab + attended_collab  # Residual connection
        
        # Apply normalization and dropout
        output = self.norm(combined_collab)
        output = self.dropout(output)
        
        # Final projection
        output = self.projection(output)
        output = self.norm(output)
        
        return output