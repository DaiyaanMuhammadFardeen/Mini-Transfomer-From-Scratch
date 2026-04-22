import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ErrorExceptionEmbedding(nn.Module):
    """
    Error and Exception Embedding Layer that captures exception types,
    error codes, and validation logic changes.
    """
    
    def __init__(self, d_model: int):
        super(ErrorExceptionEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # Exception type embedding
        self.exception_type_embedding = nn.Embedding(100, self.half_d_model)  # 100 different exception types
        
        # Error code embedding
        self.error_code_embedding = nn.Embedding(200, self.half_d_model)  # 200 different error codes
        
        # Validation pattern embedding
        self.validation_pattern_embedding = nn.Embedding(50, self.half_d_model)  # 50 validation patterns
        
        # Error feature extractor for continuous features
        self.error_feature_extractor = nn.Sequential(
            nn.Linear(10, d_model),  # 10 error-related continuous features
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Error handling attention mechanism
        self.error_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        
        # Error combination layer
        self.error_combination = nn.Linear(d_model * 3, d_model)
        
        # Final projection
        self.projection = nn.Linear(d_model * 2, d_model)  # For combining discrete and continuous features
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, error_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for error and exception embedding.
        
        Args:
            error_features: Error features tensor (batch_size, seq_len, num_error_features)
            
        Returns:
            Error embedding tensor (batch_size, seq_len, d_model)
        """
        if error_features is None:
            # If no error features provided, return zero embeddings
            batch_size = 1
            seq_len = 10  # Default sequence length
            device = next(self.parameters()).device
            return torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=torch.float)
        
        batch_size, seq_len, num_features = error_features.shape
        device = error_features.device
        
        # Extract different error features
        # Assume error_features has shape (batch_size, seq_len, num_features) where:
        # 0: exception type ID
        # 1: error code ID
        # 2: validation pattern ID
        # 3-12: error-related continuous features (10 features)
        
        # Exception Type Embedding
        if num_features > 0:
            exception_ids = error_features[:, :, 0].long().clamp(0, 99)
            exception_emb = self.exception_type_embedding(exception_ids)  # (batch_size, seq_len, half_d_model)
            exception_emb = F.pad(exception_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            exception_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Error Code Embedding
        if num_features > 1:
            error_ids = error_features[:, :, 1].long().clamp(0, 199)
            error_emb = self.error_code_embedding(error_ids)  # (batch_size, seq_len, half_d_model)
            error_emb = F.pad(error_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            error_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Validation Pattern Embedding
        if num_features > 2:
            validation_ids = error_features[:, :, 2].long().clamp(0, 49)
            validation_emb = self.validation_pattern_embedding(validation_ids)  # (batch_size, seq_len, half_d_model)
            validation_emb = F.pad(validation_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            validation_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Combine discrete error embeddings
        combined_discrete = torch.cat([exception_emb, error_emb, validation_emb], dim=-1)  # (batch_size, seq_len, d_model * 3)
        combined_discrete = self.error_combination(combined_discrete)  # (batch_size, seq_len, d_model)
        
        # Process continuous error features
        if num_features > 12:
            continuous_features = error_features[:, :, 3:13]  # Extract 10 continuous features
        elif num_features > 3:
            continuous_features = error_features[:, :, 3:]  # Extract available continuous features
            # Pad if needed
            if continuous_features.size(2) < 10:
                pad_size = 10 - continuous_features.size(2)
                continuous_features = F.pad(continuous_features, (0, pad_size))
            elif continuous_features.size(2) > 10:
                continuous_features = continuous_features[:, :, :10]
        else:
            continuous_features = torch.zeros(batch_size, seq_len, 10, device=device, dtype=torch.float)
        
        combined_continuous = self.error_feature_extractor(continuous_features)  # (batch_size, seq_len, d_model)
        
        # Apply error attention to the discrete embeddings
        attended_discrete, _ = self.error_attention(
            combined_discrete, combined_discrete, combined_discrete
        )
        combined_discrete = combined_discrete + attended_discrete  # Residual connection
        
        # Combine discrete and continuous error features
        combined_errors = torch.cat([combined_discrete, combined_continuous], dim=-1)  # (batch_size, seq_len, d_model * 2)
        output = self.projection(combined_errors)  # (batch_size, seq_len, d_model)
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        return output