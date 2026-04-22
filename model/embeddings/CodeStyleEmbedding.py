import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CodeStyleEmbedding(nn.Module):
    """
    Code Style Embedding Layer that captures naming conventions,
    formatting patterns, comment styles, and language idioms.
    """
    
    def __init__(self, d_model: int):
        super(CodeStyleEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # Naming convention embedding
        self.naming_convention_embedding = nn.Embedding(30, self.half_d_model)  # 30 different naming patterns
        
        # Formatting pattern embedding
        self.formatting_pattern_embedding = nn.Embedding(25, self.half_d_model)  # 25 different formatting patterns
        
        # Comment style embedding
        self.comment_style_embedding = nn.Embedding(20, self.half_d_model)  # 20 different comment styles
        
        # Language idiom embedding
        self.language_idiom_embedding = nn.Embedding(40, self.half_d_model)  # 40 different language idioms
        
        # Style feature extractor for continuous features
        self.style_feature_extractor = nn.Sequential(
            nn.Linear(10, d_model),  # 10 style-related continuous features
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Style attention mechanism
        self.style_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        
        # Style combination layer
        self.style_combination = nn.Linear(d_model * 4, d_model)
        
        # Final projection
        self.projection = nn.Linear(d_model * 2, d_model)  # For combining discrete and continuous features
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, style_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for code style embedding.
        
        Args:
            style_features: Style features tensor (batch_size, seq_len, num_style_features)
            
        Returns:
            Style embedding tensor (batch_size, seq_len, d_model)
        """
        if style_features is None:
            # If no style features provided, return zero embeddings
            batch_size = 1
            seq_len = 10  # Default sequence length
            device = next(self.parameters()).device
            return torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=torch.float)
        
        batch_size, seq_len, num_features = style_features.shape
        device = style_features.device
        
        # Extract different style features
        # Assume style_features has shape (batch_size, seq_len, num_features) where:
        # 0: naming convention ID
        # 1: formatting pattern ID
        # 2: comment style ID
        # 3: language idiom ID
        # 4-13: style-related continuous features (10 features)
        
        # Naming Convention Embedding
        if num_features > 0:
            naming_ids = style_features[:, :, 0].long().clamp(0, 29)
            naming_emb = self.naming_convention_embedding(naming_ids)  # (batch_size, seq_len, half_d_model)
            naming_emb = F.pad(naming_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            naming_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Formatting Pattern Embedding
        if num_features > 1:
            format_ids = style_features[:, :, 1].long().clamp(0, 24)
            format_emb = self.formatting_pattern_embedding(format_ids)  # (batch_size, seq_len, half_d_model)
            format_emb = F.pad(format_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            format_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Comment Style Embedding
        if num_features > 2:
            comment_ids = style_features[:, :, 2].long().clamp(0, 19)
            comment_emb = self.comment_style_embedding(comment_ids)  # (batch_size, seq_len, half_d_model)
            comment_emb = F.pad(comment_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            comment_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Language Idiom Embedding
        if num_features > 3:
            idiom_ids = style_features[:, :, 3].long().clamp(0, 39)
            idiom_emb = self.language_idiom_embedding(idiom_ids)  # (batch_size, seq_len, half_d_model)
            idiom_emb = F.pad(idiom_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            idiom_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Combine discrete style embeddings
        combined_discrete = torch.cat([naming_emb, format_emb, comment_emb, idiom_emb], dim=-1)  # (batch_size, seq_len, d_model * 4)
        combined_discrete = self.style_combination(combined_discrete)  # (batch_size, seq_len, d_model)
        
        # Process continuous style features
        if num_features > 13:
            continuous_features = style_features[:, :, 4:14]  # Extract 10 continuous features
        elif num_features > 4:
            continuous_features = style_features[:, :, 4:]  # Extract available continuous features
            # Pad if needed
            if continuous_features.size(2) < 10:
                pad_size = 10 - continuous_features.size(2)
                continuous_features = F.pad(continuous_features, (0, pad_size))
            elif continuous_features.size(2) > 10:
                continuous_features = continuous_features[:, :, :10]
        else:
            continuous_features = torch.zeros(batch_size, seq_len, 10, device=device, dtype=torch.float)
        
        combined_continuous = self.style_feature_extractor(continuous_features)  # (batch_size, seq_len, d_model)
        
        # Apply style attention to the discrete embeddings
        attended_discrete, _ = self.style_attention(
            combined_discrete, combined_discrete, combined_discrete
        )
        combined_discrete = combined_discrete + attended_discrete  # Residual connection
        
        # Combine discrete and continuous style features
        combined_styles = torch.cat([combined_discrete, combined_continuous], dim=-1)  # (batch_size, seq_len, d_model * 2)
        output = self.projection(combined_styles)  # (batch_size, seq_len, d_model)
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        return output