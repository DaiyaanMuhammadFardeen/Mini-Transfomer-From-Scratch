import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DomainSpecificEmbedding(nn.Module):
    """
    Domain-Specific Embedding Layer that captures business logic, 
    technical domain, and industry standard context.
    """
    
    def __init__(self, d_model: int):
        super(DomainSpecificEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # Business Logic Embedding
        self.business_logic_embedding = nn.Embedding(200, self.half_d_model)  # 200 business logic categories
        
        # Technical Domain Embedding
        self.technical_domain_embedding = nn.Embedding(50, self.half_d_model)  # 50 technical domains
        
        # Industry Standard Embedding
        self.industry_standard_embedding = nn.Embedding(100, self.half_d_model)  # 100 industry standards
        
        # Domain-specific feature extractor for continuous features
        self.domain_feature_extractor = nn.Sequential(
            nn.Linear(25, d_model),  # 25 domain-related continuous features
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Domain combination layer
        self.domain_combination = nn.Linear(d_model * 3, d_model)
        
        # Domain attention mechanism
        self.domain_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        
        # Final projection
        self.projection = nn.Linear(d_model * 2, d_model)  # For combining discrete and continuous features
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, domain_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for domain-specific embedding.
        
        Args:
            domain_features: Domain features tensor (batch_size, seq_len, num_domain_features)
            
        Returns:
            Domain-specific embedding tensor (batch_size, seq_len, d_model)
        """
        if domain_features is None:
            # If no domain features provided, return zero embeddings
            batch_size = 1
            seq_len = 10  # Default sequence length
            device = next(self.parameters()).device
            return torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=torch.float)
        
        batch_size, seq_len, num_features = domain_features.shape
        device = domain_features.device
        
        # Extract different domain features
        # Assume domain_features has shape (batch_size, seq_len, num_features) where:
        # 0: business logic ID
        # 1: technical domain ID
        # 2: industry standard ID
        # 3-27: domain-related continuous features (25 features)
        
        # Business Logic Embedding
        if num_features > 0:
            business_ids = domain_features[:, :, 0].long().clamp(0, 199)
            business_emb = self.business_logic_embedding(business_ids)  # (batch_size, seq_len, half_d_model)
            business_emb = F.pad(business_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            business_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Technical Domain Embedding
        if num_features > 1:
            tech_ids = domain_features[:, :, 1].long().clamp(0, 49)
            tech_emb = self.technical_domain_embedding(tech_ids)  # (batch_size, seq_len, half_d_model)
            tech_emb = F.pad(tech_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            tech_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Industry Standard Embedding
        if num_features > 2:
            industry_ids = domain_features[:, :, 2].long().clamp(0, 99)
            industry_emb = self.industry_standard_embedding(industry_ids)  # (batch_size, seq_len, half_d_model)
            industry_emb = F.pad(industry_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            industry_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Combine discrete domain embeddings
        combined_discrete = business_emb + tech_emb + industry_emb
        
        # Process continuous domain features
        if num_features > 27:
            continuous_features = domain_features[:, :, 3:28]  # Extract 25 continuous features
        elif num_features > 3:
            continuous_features = domain_features[:, :, 3:]  # Extract available continuous features
            # Pad if needed
            if continuous_features.size(2) < 25:
                pad_size = 25 - continuous_features.size(2)
                continuous_features = F.pad(continuous_features, (0, pad_size))
            elif continuous_features.size(2) > 25:
                continuous_features = continuous_features[:, :, :25]
        else:
            continuous_features = torch.zeros(batch_size, seq_len, 25, device=device, dtype=torch.float)
        
        combined_continuous = self.domain_feature_extractor(continuous_features)  # (batch_size, seq_len, d_model)
        
        # Apply domain attention to the discrete embeddings
        attended_discrete, _ = self.domain_attention(
            combined_discrete, combined_discrete, combined_discrete
        )
        combined_discrete = combined_discrete + attended_discrete  # Residual connection
        
        # Combine discrete and continuous domain features
        combined_domains = torch.cat([combined_discrete, combined_continuous], dim=-1)  # (batch_size, seq_len, d_model * 2)
        output = self.projection(combined_domains)  # (batch_size, seq_len, d_model)
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        return output