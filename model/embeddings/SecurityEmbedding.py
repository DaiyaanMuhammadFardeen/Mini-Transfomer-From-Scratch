import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SecurityEmbedding(nn.Module):
    """
    Security Embedding Layer that captures vulnerability patterns,
    security annotations, and input validation changes.
    """
    
    def __init__(self, d_model: int):
        super(SecurityEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # Vulnerability pattern embedding
        self.vulnerability_pattern_embedding = nn.Embedding(50, self.half_d_model)  # 50 different vulnerability patterns
        
        # Security annotation embedding
        self.security_annotation_embedding = nn.Embedding(30, self.half_d_model)  # 30 different security annotations
        
        # Input validation embedding
        self.input_validation_embedding = nn.Embedding(40, self.half_d_model)  # 40 different validation patterns
        
        # Authentication/authorization embedding
        self.auth_embedding = nn.Embedding(25, self.half_d_model)  # 25 different auth patterns
        
        # Security feature extractor for continuous features
        self.security_feature_extractor = nn.Sequential(
            nn.Linear(8, d_model),  # 8 security-related continuous features
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Security attention mechanism
        self.security_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        
        # Security combination layer
        self.security_combination = nn.Linear(d_model * 4, d_model)
        
        # Final projection
        self.projection = nn.Linear(d_model * 2, d_model)  # For combining discrete and continuous features
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, security_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for security embedding.
        
        Args:
            security_features: Security features tensor (batch_size, seq_len, num_security_features)
            
        Returns:
            Security embedding tensor (batch_size, seq_len, d_model)
        """
        if security_features is None:
            # If no security features provided, return zero embeddings
            batch_size = 1
            seq_len = 10  # Default sequence length
            device = next(self.parameters()).device
            return torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=torch.float)
        
        batch_size, seq_len, num_features = security_features.shape
        device = security_features.device
        
        # Extract different security features
        # Assume security_features has shape (batch_size, seq_len, num_features) where:
        # 0: vulnerability pattern ID
        # 1: security annotation ID
        # 2: input validation ID
        # 3: authentication/authorization ID
        # 4-11: security-related continuous features (8 features)
        
        # Vulnerability Pattern Embedding
        if num_features > 0:
            vuln_ids = security_features[:, :, 0].long().clamp(0, 49)
            vuln_emb = self.vulnerability_pattern_embedding(vuln_ids)  # (batch_size, seq_len, half_d_model)
            vuln_emb = F.pad(vuln_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            vuln_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Security Annotation Embedding
        if num_features > 1:
            annotation_ids = security_features[:, :, 1].long().clamp(0, 29)
            annotation_emb = self.security_annotation_embedding(annotation_ids)  # (batch_size, seq_len, half_d_model)
            annotation_emb = F.pad(annotation_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            annotation_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Input Validation Embedding
        if num_features > 2:
            validation_ids = security_features[:, :, 2].long().clamp(0, 39)
            validation_emb = self.input_validation_embedding(validation_ids)  # (batch_size, seq_len, half_d_model)
            validation_emb = F.pad(validation_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            validation_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Authentication/Authorization Embedding
        if num_features > 3:
            auth_ids = security_features[:, :, 3].long().clamp(0, 24)
            auth_emb = self.auth_embedding(auth_ids)  # (batch_size, seq_len, half_d_model)
            auth_emb = F.pad(auth_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            auth_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Combine discrete security embeddings
        combined_discrete = torch.cat([vuln_emb, annotation_emb, validation_emb, auth_emb], dim=-1)  # (batch_size, seq_len, d_model * 4)
        combined_discrete = self.security_combination(combined_discrete)  # (batch_size, seq_len, d_model)
        
        # Process continuous security features
        if num_features > 11:
            continuous_features = security_features[:, :, 4:12]  # Extract 8 continuous features
        elif num_features > 4:
            continuous_features = security_features[:, :, 4:]  # Extract available continuous features
            # Pad if needed
            if continuous_features.size(2) < 8:
                pad_size = 8 - continuous_features.size(2)
                continuous_features = F.pad(continuous_features, (0, pad_size))
            elif continuous_features.size(2) > 8:
                continuous_features = continuous_features[:, :, :8]
        else:
            continuous_features = torch.zeros(batch_size, seq_len, 8, device=device, dtype=torch.float)
        
        combined_continuous = self.security_feature_extractor(continuous_features)  # (batch_size, seq_len, d_model)
        
        # Apply security attention to the discrete embeddings
        attended_discrete, _ = self.security_attention(
            combined_discrete, combined_discrete, combined_discrete
        )
        combined_discrete = combined_discrete + attended_discrete  # Residual connection
        
        # Combine discrete and continuous security features
        combined_security = torch.cat([combined_discrete, combined_continuous], dim=-1)  # (batch_size, seq_len, d_model * 2)
        output = self.projection(combined_security)  # (batch_size, seq_len, d_model)
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        return output