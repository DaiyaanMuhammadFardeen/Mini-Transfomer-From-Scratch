import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SemanticCodeEmbedding(nn.Module):
    """
    Semantic Code Embedding Layer that captures syntax tree structure,
    control flow, and data flow information.
    """
    
    def __init__(self, d_model: int):
        super(SemanticCodeEmbedding, self).__init__()
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        
        # AST Node Embedding
        self.ast_node_embedding = nn.Embedding(1000, self.half_d_model)  # Assuming 1000 different AST node types
        
        # Control Flow Embedding
        self.control_flow_embedding = nn.Embedding(20, self.half_d_model)  # 20 different control flow types
        
        # Data Flow Embedding
        self.data_flow_embedding = nn.Linear(10, self.half_d_model)  # 10 different data flow features
        
        # Projection layers to combine different semantic components
        self.ast_projection = nn.Linear(self.half_d_model, d_model)
        self.control_projection = nn.Linear(self.half_d_model, d_model)
        self.data_projection = nn.Linear(self.half_d_model, d_model)
        
        # Final combination layer
        self.combination_layer = nn.Linear(d_model * 3, d_model)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: torch.Tensor, ast_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for semantic code embedding.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            ast_features: AST features tensor (batch_size, seq_len, num_ast_features)
            
        Returns:
            Semantic code embedding tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize with zeros if AST features not provided
        if ast_features is None:
            ast_features = torch.zeros(batch_size, seq_len, 10, device=input_ids.device, dtype=torch.float)
            ast_node_ids = torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            control_flow_ids = torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
        else:
            # Assume ast_features contains concatenated AST node types, control flow types, and data flow features
            # First 1 element for AST node types, next 1 for control flow, rest for data flow
            ast_node_ids = ast_features[:, :, 0].long() if ast_features.size(2) > 0 else torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            control_flow_ids = ast_features[:, :, 1].long() if ast_features.size(2) > 1 else torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.long)
            data_flow_features = ast_features[:, :, 2:12] if ast_features.size(2) >= 12 else ast_features[:, :, 2:] if ast_features.size(2) > 2 else torch.zeros(batch_size, seq_len, 10, device=input_ids.device, dtype=torch.float)
        
        # AST Node Embedding
        ast_emb = self.ast_node_embedding(ast_node_ids)  # (batch_size, seq_len, half_d_model)
        ast_emb = self.ast_projection(ast_emb)  # (batch_size, seq_len, d_model)
        
        # Control Flow Embedding
        control_emb = self.control_flow_embedding(control_flow_ids)  # (batch_size, seq_len, half_d_model)
        control_emb = self.control_projection(control_emb)  # (batch_size, seq_len, d_model)
        
        # Data Flow Embedding
        if ast_features is not None and ast_features.size(2) >= 12:
            data_flow_features = ast_features[:, :, 2:12]  # Extract data flow features
        else:
            data_flow_features = torch.zeros(batch_size, seq_len, 10, device=input_ids.device, dtype=torch.float)
        
        data_emb = self.data_flow_embedding(data_flow_features)  # (batch_size, seq_len, half_d_model)
        data_emb = self.data_projection(data_emb)  # (batch_size, seq_len, d_model)
        
        # Combine all semantic embeddings
        combined_semantic = torch.cat([ast_emb, control_emb, data_emb], dim=-1)  # (batch_size, seq_len, d_model * 3)
        combined_semantic = self.combination_layer(combined_semantic)  # (batch_size, seq_len, d_model)
        
        # Apply normalization and dropout
        output = self.norm(combined_semantic)
        output = self.dropout(output)
        
        return output