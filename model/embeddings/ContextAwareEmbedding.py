import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContextAwareEmbedding(nn.Module):
    """
    Context-Aware Embedding Layer that captures file, function, class, and module context.
    """
    
    def __init__(self, d_model: int):
        super(ContextAwareEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # Context type embeddings
        self.file_context_embedding = nn.Embedding(500, self.half_d_model)  # 500 different files
        self.function_context_embedding = nn.Embedding(1000, self.half_d_model)  # 1000 different functions
        self.class_context_embedding = nn.Embedding(500, self.half_d_model)  # 500 different classes
        self.module_context_embedding = nn.Embedding(200, self.half_d_model)  # 200 different modules
        
        # Context similarity attention layers
        self.file_similarity_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        self.function_similarity_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        
        # Context combination layer
        self.context_combination = nn.Linear(d_model * 4, d_model)  # 4 context types
        
        # Final projection
        self.projection = nn.Linear(d_model, d_model)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: torch.Tensor, context_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for context-aware embedding.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            context_info: Context information tensor (batch_size, seq_len, num_context_features)
            
        Returns:
            Context-aware embedding tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize with zeros if context info not provided
        if context_info is None:
            file_context_ids = torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            function_context_ids = torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            class_context_ids = torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            module_context_ids = torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
        else:
            # Extract context IDs from context_info tensor
            # Assume context_info has shape (batch_size, seq_len, 4) for file, function, class, module contexts
            file_context_ids = context_info[:, :, 0].long() if context_info.size(2) > 0 else torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            function_context_ids = context_info[:, :, 1].long() if context_info.size(2) > 1 else torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            class_context_ids = context_info[:, :, 2].long() if context_info.size(2) > 2 else torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            module_context_ids = context_info[:, :, 3].long() if context_info.size(2) > 3 else torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
        
        # File Context Embedding
        file_emb = self.file_context_embedding(file_context_ids)  # (batch_size, seq_len, half_d_model)
        file_emb = F.pad(file_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model size
        
        # Function Context Embedding
        function_emb = self.function_context_embedding(function_context_ids)  # (batch_size, seq_len, half_d_model)
        function_emb = F.pad(function_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model size
        
        # Class Context Embedding
        class_emb = self.class_context_embedding(class_context_ids)  # (batch_size, seq_len, half_d_model)
        class_emb = F.pad(class_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model size
        
        # Module Context Embedding
        module_emb = self.module_context_embedding(module_context_ids)  # (batch_size, seq_len, half_d_model)
        module_emb = F.pad(module_emb, (0, self.d_model - self.half_d_model))  # Pad to full d_model size
        
        # Combine all context embeddings
        combined_context = torch.cat([file_emb, function_emb, class_emb, module_emb], dim=-1)  # (batch_size, seq_len, d_model * 4)
        combined_context = self.context_combination(combined_context)  # (batch_size, seq_len, d_model)
        
        # Apply attention mechanisms for similarity-based context weighting
        # Using file and function context for similarity attention
        attended_context, _ = self.file_similarity_attention(
            combined_context, combined_context, combined_context
        )
        combined_context = combined_context + attended_context  # Residual connection
        
        # Apply normalization and dropout
        output = self.norm(combined_context)
        output = self.dropout(output)
        
        # Final projection
        output = self.projection(output)
        output = self.norm(output)
        
        return output