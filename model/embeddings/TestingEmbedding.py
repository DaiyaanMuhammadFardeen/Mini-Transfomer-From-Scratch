import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TestingEmbedding(nn.Module):
    """
    Testing Embedding Layer that captures test coverage, test types,
    assertion patterns, and mock usage.
    """
    
    def __init__(self, d_model: int):
        super(TestingEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # Test type embedding
        self.test_type_embedding = nn.Embedding(10, self.half_d_model)  # 10 different test types (unit, integration, etc.)
        
        # Assertion pattern embedding
        self.assertion_pattern_embedding = nn.Embedding(25, self.half_d_model)  # 25 different assertion patterns
        
        # Mock usage embedding
        self.mock_usage_embedding = nn.Embedding(20, self.half_d_model)  # 20 different mock usage patterns
        
        # Coverage impact embedding
        self.coverage_impact_embedding = nn.Linear(1, self.half_d_model)  # Continuous coverage value
        
        # Testing feature extractor for continuous features
        self.testing_feature_extractor = nn.Sequential(
            nn.Linear(8, d_model),  # 8 testing-related continuous features
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Testing attention mechanism
        self.testing_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        
        # Testing combination layer
        self.testing_combination = nn.Linear(d_model * 4, d_model)
        
        # Final projection
        self.projection = nn.Linear(d_model * 2, d_model)  # For combining discrete and continuous features
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, testing_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for testing embedding.
        
        Args:
            testing_features: Testing features tensor (batch_size, seq_len, num_testing_features)
            
        Returns:
            Testing embedding tensor (batch_size, seq_len, d_model)
        """
        if testing_features is None:
            # If no testing features provided, return zero embeddings
            batch_size = 1
            seq_len = 10  # Default sequence length
            device = next(self.parameters()).device
            return torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=torch.float)
        
        batch_size, seq_len, num_features = testing_features.shape
        device = testing_features.device
        
        # Extract different testing features
        # Assume testing_features has shape (batch_size, seq_len, num_features) where:
        # 0: test type ID
        # 1: assertion pattern ID
        # 2: mock usage ID
        # 3: coverage impact value (continuous)
        # 4-11: testing-related continuous features (8 features)
        
        # Test Type Embedding
        if num_features > 0:
            test_ids = testing_features[:, :, 0].long().clamp(0, 9)
            test_emb = self.test_type_embedding(test_ids)  # (batch_size, seq_len, half_d_model)
            test_emb = F.pad(test_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            test_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Assertion Pattern Embedding
        if num_features > 1:
            assertion_ids = testing_features[:, :, 1].long().clamp(0, 24)
            assertion_emb = self.assertion_pattern_embedding(assertion_ids)  # (batch_size, seq_len, half_d_model)
            assertion_emb = F.pad(assertion_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            assertion_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Mock Usage Embedding
        if num_features > 2:
            mock_ids = testing_features[:, :, 2].long().clamp(0, 19)
            mock_emb = self.mock_usage_embedding(mock_ids)  # (batch_size, seq_len, half_d_model)
            mock_emb = F.pad(mock_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            mock_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Coverage Impact Embedding (continuous)
        if num_features > 3:
            coverage_values = testing_features[:, :, 3:4]  # (batch_size, seq_len, 1)
            coverage_emb = self.coverage_impact_embedding(coverage_values)  # (batch_size, seq_len, half_d_model)
            coverage_emb = F.pad(coverage_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            coverage_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Combine discrete testing embeddings
        combined_discrete = test_emb + assertion_emb + mock_emb + coverage_emb
        
        # Process continuous testing features
        if num_features > 11:
            continuous_features = testing_features[:, :, 4:12]  # Extract 8 continuous features
        elif num_features > 4:
            continuous_features = testing_features[:, :, 4:]  # Extract available continuous features
            # Pad if needed
            if continuous_features.size(2) < 8:
                pad_size = 8 - continuous_features.size(2)
                continuous_features = F.pad(continuous_features, (0, pad_size))
            elif continuous_features.size(2) > 8:
                continuous_features = continuous_features[:, :, :8]
        else:
            continuous_features = torch.zeros(batch_size, seq_len, 8, device=device, dtype=torch.float)
        
        combined_continuous = self.testing_feature_extractor(continuous_features)  # (batch_size, seq_len, d_model)
        
        # Apply testing attention to the discrete embeddings
        attended_discrete, _ = self.testing_attention(
            combined_discrete, combined_discrete, combined_discrete
        )
        combined_discrete = combined_discrete + attended_discrete  # Residual connection
        
        # Combine discrete and continuous testing features
        combined_testing = torch.cat([combined_discrete, combined_continuous], dim=-1)  # (batch_size, seq_len, d_model * 2)
        output = self.projection(combined_testing)  # (batch_size, seq_len, d_model)
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        return output