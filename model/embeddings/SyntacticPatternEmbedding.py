import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SyntacticPatternEmbedding(nn.Module):
    """
    Syntactic Pattern Embedding Layer that captures design patterns,
    code smells, and refactoring patterns.
    """
    
    def __init__(self, d_model: int):
        super(SyntacticPatternEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # Design Pattern Embedding
        self.design_pattern_embedding = nn.Embedding(50, self.half_d_model)  # 50 different design patterns
        
        # Code Smell Embedding
        self.code_smell_embedding = nn.Embedding(30, self.half_d_model)  # 30 different code smells
        
        # Refactoring Pattern Embedding
        self.refactoring_pattern_embedding = nn.Embedding(40, self.half_d_model)  # 40 different refactoring patterns
        
        # Pattern combination layer
        self.pattern_combination = nn.Linear(d_model * 3, d_model)  # 3 pattern types
        
        # Pattern feature extractor for continuous pattern features
        self.pattern_feature_extractor = nn.Sequential(
            nn.Linear(20, d_model),  # 20 pattern-related continuous features
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Final projection
        self.projection = nn.Linear(d_model * 2, d_model)  # To maintain d_model after combining discrete and continuous
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: torch.Tensor, pattern_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for syntactic pattern embedding.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            pattern_features: Pattern features tensor (batch_size, seq_len, num_pattern_features)
            
        Returns:
            Syntactic pattern embedding tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize with zeros if pattern features not provided
        if pattern_features is None:
            design_pattern_ids = torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            code_smell_ids = torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            refactoring_pattern_ids = torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            continuous_pattern_features = torch.zeros(batch_size, seq_len, 20, device=input_ids.device, dtype=torch.float)
        else:
            # Extract pattern IDs and features from pattern_features tensor
            # Assume pattern_features has shape (batch_size, seq_len, num_features) where:
            # First 3 elements are pattern IDs, rest are continuous features
            design_pattern_ids = pattern_features[:, :, 0].long() if pattern_features.size(2) > 0 else torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            code_smell_ids = pattern_features[:, :, 1].long() if pattern_features.size(2) > 1 else torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            refactoring_pattern_ids = pattern_features[:, :, 2].long() if pattern_features.size(2) > 2 else torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            
            # Extract continuous pattern features (remaining columns)
            if pattern_features.size(2) > 3:
                continuous_pattern_features = pattern_features[:, :, 3:23] if pattern_features.size(2) >= 23 else pattern_features[:, :, 3:]
                # Ensure we have exactly 20 features
                if continuous_pattern_features.size(2) < 20:
                    pad_size = 20 - continuous_pattern_features.size(2)
                    continuous_pattern_features = F.pad(continuous_pattern_features, (0, pad_size))
                elif continuous_pattern_features.size(2) > 20:
                    continuous_pattern_features = continuous_pattern_features[:, :, :20]
            else:
                continuous_pattern_features = torch.zeros(batch_size, seq_len, 20, device=input_ids.device, dtype=torch.float)
        
        # Design Pattern Embedding
        design_emb = self.design_pattern_embedding(design_pattern_ids)  # (batch_size, seq_len, half_d_model)
        design_emb = F.pad(design_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model size
        
        # Code Smell Embedding
        smell_emb = self.code_smell_embedding(code_smell_ids)  # (batch_size, seq_len, half_d_model)
        smell_emb = F.pad(smell_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model size
        
        # Refactoring Pattern Embedding
        refactor_emb = self.refactoring_pattern_embedding(refactoring_pattern_ids)  # (batch_size, seq_len, half_d_model)
        refactor_emb = F.pad(refactor_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model size
        
        # Combine discrete pattern embeddings
        combined_discrete = torch.cat([design_emb, smell_emb, refactor_emb], dim=-1)  # (batch_size, seq_len, d_model * 3)
        combined_discrete = self.pattern_combination(combined_discrete)  # (batch_size, seq_len, d_model)
        
        # Process continuous pattern features
        combined_continuous = self.pattern_feature_extractor(continuous_pattern_features)  # (batch_size, seq_len, d_model)
        
        # Combine discrete and continuous pattern embeddings
        combined_patterns = torch.cat([combined_discrete, combined_continuous], dim=-1)  # (batch_size, seq_len, d_model * 2)
        output = self.projection(combined_patterns)  # (batch_size, seq_len, d_model)
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        return output