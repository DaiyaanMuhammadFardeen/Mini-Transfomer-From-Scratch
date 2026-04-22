import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DependencyEmbedding(nn.Module):
    """
    Dependency Embedding Layer that captures import dependencies, 
    call graphs, data dependencies, and build dependencies.
    """
    
    def __init__(self, d_model: int):
        super(DependencyEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # Import dependency embedding
        self.import_dep_embedding = nn.Embedding(500, self.half_d_model)  # 500 different imports
        
        # Call graph embedding
        self.call_graph_embedding = nn.Embedding(1000, self.half_d_model)  # 1000 different call relationships
        
        # Data dependency embedding
        self.data_dep_embedding = nn.Embedding(300, self.half_d_model)  # 300 different data dependencies
        
        # Build dependency embedding
        self.build_dep_embedding = nn.Embedding(200, self.half_d_model)  # 200 different build dependencies
        
        # Dependency feature extractor for continuous features
        self.dep_feature_extractor = nn.Sequential(
            nn.Linear(15, d_model),  # 15 dependency-related continuous features
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Dependency combination layer
        self.dep_combination = nn.Linear(d_model * 4, d_model)
        
        # Graph convolution for dependency modeling
        self.graph_conv = nn.Linear(d_model, d_model)
        
        # Final projection
        self.projection = nn.Linear(d_model * 2, d_model)  # For combining discrete and continuous features
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, dep_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for dependency embedding.
        
        Args:
            dep_features: Dependency features tensor (batch_size, seq_len, num_dep_features)
            
        Returns:
            Dependency embedding tensor (batch_size, seq_len, d_model)
        """
        if dep_features is None:
            # If no dependency features provided, return zero embeddings
            batch_size = 1
            seq_len = 10  # Default sequence length
            device = next(self.parameters()).device
            return torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=torch.float)
        
        batch_size, seq_len, num_features = dep_features.shape
        device = dep_features.device
        
        # Extract different dependency features
        # Assume dep_features has shape (batch_size, seq_len, num_features) where:
        # 0: import dependency ID
        # 1: call graph ID
        # 2: data dependency ID
        # 3: build dependency ID
        # 4-18: dependency-related continuous features (15 features)
        
        # Import Dependency Embedding
        if num_features > 0:
            import_ids = dep_features[:, :, 0].long().clamp(0, 499)
            import_emb = self.import_dep_embedding(import_ids)  # (batch_size, seq_len, half_d_model)
            import_emb = F.pad(import_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            import_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Call Graph Embedding
        if num_features > 1:
            call_ids = dep_features[:, :, 1].long().clamp(0, 999)
            call_emb = self.call_graph_embedding(call_ids)  # (batch_size, seq_len, half_d_model)
            call_emb = F.pad(call_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            call_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Data Dependency Embedding
        if num_features > 2:
            data_ids = dep_features[:, :, 2].long().clamp(0, 299)
            data_emb = self.data_dep_embedding(data_ids)  # (batch_size, seq_len, half_d_model)
            data_emb = F.pad(data_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            data_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Build Dependency Embedding
        if num_features > 3:
            build_ids = dep_features[:, :, 3].long().clamp(0, 199)
            build_emb = self.build_dep_embedding(build_ids)  # (batch_size, seq_len, half_d_model)
            build_emb = F.pad(build_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model
        else:
            build_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Combine discrete dependency embeddings
        combined_discrete = torch.cat([import_emb, call_emb, data_emb, build_emb], dim=-1)  # (batch_size, seq_len, d_model * 4)
        combined_discrete = self.dep_combination(combined_discrete)  # (batch_size, seq_len, d_model)
        
        # Process continuous dependency features
        if num_features > 18:
            continuous_features = dep_features[:, :, 4:19]  # Extract 15 continuous features
        elif num_features > 4:
            continuous_features = dep_features[:, :, 4:]  # Extract available continuous features
            # Pad if needed
            if continuous_features.size(2) < 15:
                pad_size = 15 - continuous_features.size(2)
                continuous_features = F.pad(continuous_features, (0, pad_size))
            elif continuous_features.size(2) > 15:
                continuous_features = continuous_features[:, :, :15]
        else:
            continuous_features = torch.zeros(batch_size, seq_len, 15, device=device, dtype=torch.float)
        
        combined_continuous = self.dep_feature_extractor(continuous_features)  # (batch_size, seq_len, d_model)
        
        # Apply graph convolution to model dependencies
        convolved_discrete = self.graph_conv(combined_discrete)  # (batch_size, seq_len, d_model)
        
        # Combine discrete and continuous dependency features
        combined_deps = torch.cat([convolved_discrete, combined_continuous], dim=-1)  # (batch_size, seq_len, d_model * 2)
        output = self.projection(combined_deps)  # (batch_size, seq_len, d_model)
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        return output