import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ComplexityEmbedding(nn.Module):
    """
    Complexity Embedding Layer that captures cyclomatic, cognitive, 
    and Halstead complexity metrics.
    """
    
    def __init__(self, d_model: int):
        super(ComplexityEmbedding, self).__init__()
        
        self.d_model = d_model
        
        # Complexity feature extractor for continuous complexity metrics
        self.complexity_feature_extractor = nn.Sequential(
            nn.Linear(3, d_model),  # 3 complexity metrics: cyclomatic, cognitive, halstead
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Additional complexity features extractor
        self.additional_complexity_extractor = nn.Sequential(
            nn.Linear(7, d_model),  # 7 additional complexity features
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Complexity attention mechanism
        self.complexity_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        
        # Complexity combination layer
        self.complexity_combination = nn.Linear(d_model * 2, d_model)
        
        # Final projection
        self.projection = nn.Linear(d_model, d_model)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, complexity_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for complexity embedding.
        
        Args:
            complexity_features: Complexity features tensor (batch_size, seq_len, num_complexity_features)
            
        Returns:
            Complexity embedding tensor (batch_size, seq_len, d_model)
        """
        if complexity_features is None:
            # If no complexity features provided, return zero embeddings
            batch_size = 1
            seq_len = 10  # Default sequence length
            device = next(self.parameters()).device
            return torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=torch.float)
        
        batch_size, seq_len, num_features = complexity_features.shape
        device = complexity_features.device
        
        # Extract complexity features
        # Assume complexity_features has shape (batch_size, seq_len, num_features) where:
        # 0-2: main complexity metrics (cyclomatic, cognitive, halstead)
        # 3-9: additional complexity features (7 features)
        
        # Extract main complexity metrics
        if num_features > 2:
            main_complexity = complexity_features[:, :, :3]  # Extract 3 main metrics
        else:
            main_complexity = torch.zeros(batch_size, seq_len, 3, device=device, dtype=torch.float)
        
        # Extract additional complexity features
        if num_features > 9:
            additional_complexity = complexity_features[:, :, 3:10]  # Extract 7 additional features
        elif num_features > 3:
            additional_complexity = complexity_features[:, :, 3:]  # Extract available features
            # Pad if needed
            if additional_complexity.size(2) < 7:
                pad_size = 7 - additional_complexity.size(2)
                additional_complexity = F.pad(additional_complexity, (0, pad_size))
            elif additional_complexity.size(2) > 7:
                additional_complexity = additional_complexity[:, :, :7]
        else:
            additional_complexity = torch.zeros(batch_size, seq_len, 7, device=device, dtype=torch.float)
        
        # Process main complexity metrics
        main_complexity_emb = self.complexity_feature_extractor(main_complexity)  # (batch_size, seq_len, d_model)
        
        # Process additional complexity features
        additional_complexity_emb = self.additional_complexity_extractor(additional_complexity)  # (batch_size, seq_len, d_model)
        
        # Combine main and additional complexity embeddings
        combined_complexity = main_complexity_emb + additional_complexity_emb
        
        # Apply complexity attention
        attended_complexity, _ = self.complexity_attention(
            combined_complexity, combined_complexity, combined_complexity
        )
        combined_complexity = combined_complexity + attended_complexity  # Residual connection
        
        # Apply normalization and dropout
        output = self.norm(combined_complexity)
        output = self.dropout(output)
        
        # Final projection
        output = self.projection(output)
        output = self.norm(output)
        
        return output