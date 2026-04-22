import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class APIEmbedding(nn.Module):
    """
    API Embedding Layer that captures endpoint changes, parameter changes,
    response changes, and version updates.
    """
    
    def __init__(self, d_model: int):
        super(APIEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # Endpoint change embedding
        self.endpoint_change_embedding = nn.Embedding(100, self.half_d_model)  # 100 different endpoints
        
        # Parameter change embedding
        self.parameter_change_embedding = nn.Embedding(80, self.half_d_model)  # 80 different parameter patterns
        
        # Response change embedding
        self.response_change_embedding = nn.Embedding(60, self.half_d_model)  # 60 different response patterns
        
        # Version change embedding
        self.version_change_embedding = nn.Embedding(50, self.half_d_model)  # 50 different version patterns
        
        # API feature extractor for continuous features
        self.api_feature_extractor = nn.Sequential(
            nn.Linear(10, d_model),  # 10 API-related continuous features
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # API attention mechanism
        self.api_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        
        # API combination layer
        self.api_combination = nn.Linear(d_model * 4, d_model)
        
        # Final projection
        self.projection = nn.Linear(d_model * 2, d_model)  # For combining discrete and continuous features
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, api_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for API embedding.
        
        Args:
            api_features: API features tensor (batch_size, seq_len, num_api_features)
            
        Returns:
            API embedding tensor (batch_size, seq_len, d_model)
        """
        if api_features is None:
            # If no API features provided, return zero embeddings
            batch_size = 1
            seq_len = 10  # Default sequence length
            device = next(self.parameters()).device
            return torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=torch.float)
        
        batch_size, seq_len, num_features = api_features.shape
        device = api_features.device
        
        # Extract different API features
        # Assume api_features has shape (batch_size, seq_len, num_features) where:
        # 0: endpoint change ID
        # 1: parameter change ID
        # 2: response change ID
        # 3: version change ID
        # 4-13: API-related continuous features (10 features)
        
        # Endpoint Change Embedding
        if num_features > 0:
            endpoint_ids = api_features[:, :, 0].long().clamp(0, 99)
            endpoint_emb = self.endpoint_change_embedding(endpoint_ids)  # (batch_size, seq_len, half_d_model)
            endpoint_emb = F.pad(endpoint_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            endpoint_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Parameter Change Embedding
        if num_features > 1:
            param_ids = api_features[:, :, 1].long().clamp(0, 79)
            param_emb = self.parameter_change_embedding(param_ids)  # (batch_size, seq_len, half_d_model)
            param_emb = F.pad(param_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            param_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Response Change Embedding
        if num_features > 2:
            response_ids = api_features[:, :, 2].long().clamp(0, 59)
            response_emb = self.response_change_embedding(response_ids)  # (batch_size, seq_len, half_d_model)
            response_emb = F.pad(response_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            response_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Version Change Embedding
        if num_features > 3:
            version_ids = api_features[:, :, 3].long().clamp(0, 49)
            version_emb = self.version_change_embedding(version_ids)  # (batch_size, seq_len, half_d_model)
            version_emb = F.pad(version_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            version_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Combine discrete API embeddings
        combined_discrete = torch.cat([endpoint_emb, param_emb, response_emb, version_emb], dim=-1)  # (batch_size, seq_len, d_model * 4)
        combined_discrete = self.api_combination(combined_discrete)  # (batch_size, seq_len, d_model)
        
        # Process continuous API features
        if num_features > 13:
            continuous_features = api_features[:, :, 4:14]  # Extract 10 continuous features
        elif num_features > 4:
            continuous_features = api_features[:, :, 4:]  # Extract available continuous features
            # Pad if needed
            if continuous_features.size(2) < 10:
                pad_size = 10 - continuous_features.size(2)
                continuous_features = F.pad(continuous_features, (0, pad_size))
            elif continuous_features.size(2) > 10:
                continuous_features = continuous_features[:, :, :10]
        else:
            continuous_features = torch.zeros(batch_size, seq_len, 10, device=device, dtype=torch.float)
        
        combined_continuous = self.api_feature_extractor(continuous_features)  # (batch_size, seq_len, d_model)
        
        # Apply API attention to the discrete embeddings
        attended_discrete, _ = self.api_attention(
            combined_discrete, combined_discrete, combined_discrete
        )
        combined_discrete = combined_discrete + attended_discrete  # Residual connection
        
        # Combine discrete and continuous API features
        combined_api = torch.cat([combined_discrete, combined_continuous], dim=-1)  # (batch_size, seq_len, d_model * 2)
        output = self.projection(combined_api)  # (batch_size, seq_len, d_model)
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        return output