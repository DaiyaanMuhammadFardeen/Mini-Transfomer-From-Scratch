import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from .SemanticCodeEmbedding import SemanticCodeEmbedding
from .ContextAwareEmbedding import ContextAwareEmbedding
from .SyntacticPatternEmbedding import SyntacticPatternEmbedding
from .TemporalEmbedding import TemporalEmbedding
from .CollaborativeEmbedding import CollaborativeEmbedding
from .DomainSpecificEmbedding import DomainSpecificEmbedding
from .ChangeTypeEmbedding import ChangeTypeEmbedding
from .DependencyEmbedding import DependencyEmbedding
from .ComplexityEmbedding import ComplexityEmbedding
from .ErrorExceptionEmbedding import ErrorExceptionEmbedding
from .PerformanceEmbedding import PerformanceEmbedding
from .TestingEmbedding import TestingEmbedding
from .CodeStyleEmbedding import CodeStyleEmbedding
from .SecurityEmbedding import SecurityEmbedding
from .APIEmbedding import APIEmbedding
from .FusionMechanism import FusionMechanism


class MultimodalEmbedding(nn.Module):
    """
    Comprehensive multimodal embedding system for commit message generation.
    Integrates multiple embedding layers to capture rich contextual information.
    """
    
    def __init__(
        self, 
        vocab_size: int,
        d_model: int,
        max_seq_length: int,
        device: str = 'cpu',
        fusion_method: str = 'learnable_weights'
    ):
        super(MultimodalEmbedding, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Basic token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Initialize all specialized embedding layers
        self.semantic_code_embedding = SemanticCodeEmbedding(d_model)
        self.context_embedding = ContextAwareEmbedding(d_model)
        self.pattern_embedding = SyntacticPatternEmbedding(d_model)
        self.temporal_embedding = TemporalEmbedding(d_model)
        self.collaborative_embedding = CollaborativeEmbedding(d_model)
        self.domain_embedding = DomainSpecificEmbedding(d_model)
        self.change_type_embedding = ChangeTypeEmbedding(d_model)
        self.dependency_embedding = DependencyEmbedding(d_model)
        self.complexity_embedding = ComplexityEmbedding(d_model)
        self.error_embedding = ErrorExceptionEmbedding(d_model)
        self.performance_embedding = PerformanceEmbedding(d_model)
        self.testing_embedding = TestingEmbedding(d_model)
        self.style_embedding = CodeStyleEmbedding(d_model)
        self.security_embedding = SecurityEmbedding(d_model)
        self.api_embedding = APIEmbedding(d_model)
        
        # Fusion mechanism to combine all embeddings
        self.fusion_mechanism = FusionMechanism(
            num_embeddings=16,  # 15 specialized embedding layers + 1 token embedding
            d_model=d_model,
            fusion_method=fusion_method
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, 
        input_ids: torch.Tensor,
        additional_features: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the multimodal embedding system.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            additional_features: Dictionary containing additional feature tensors
            
        Returns:
            Combined embedding tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape
        
        # Basic token embedding
        token_emb = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Initialize all specialized embeddings with zeros if features not provided
        if additional_features is None:
            additional_features = {}
        
        # Compute all specialized embeddings
        embeddings = [token_emb]
        
        # Semantic code embedding
        semantic_emb = self.semantic_code_embedding(input_ids, additional_features.get('ast_nodes', None))
        embeddings.append(semantic_emb)
        
        # Context embedding
        context_emb = self.context_embedding(input_ids, additional_features.get('context_info', None))
        embeddings.append(context_emb)
        
        # Pattern embedding
        pattern_emb = self.pattern_embedding(input_ids, additional_features.get('patterns', None))
        embeddings.append(pattern_emb)
        
        # Temporal embedding (if temporal features available)
        if 'temporal_features' in additional_features:
            temporal_emb = self.temporal_embedding(additional_features['temporal_features'])
            embeddings.append(temporal_emb)
        else:
            # Create zero embedding with same shape as token embedding
            temporal_emb = torch.zeros_like(token_emb)
            embeddings.append(temporal_emb)
        
        # Collaborative embedding
        if 'collaborative_features' in additional_features:
            collab_emb = self.collaborative_embedding(additional_features['collaborative_features'])
            embeddings.append(collab_emb)
        else:
            collab_emb = torch.zeros_like(token_emb)
            embeddings.append(collab_emb)
        
        # Domain embedding
        if 'domain_features' in additional_features:
            domain_emb = self.domain_embedding(additional_features['domain_features'])
            embeddings.append(domain_emb)
        else:
            domain_emb = torch.zeros_like(token_emb)
            embeddings.append(domain_emb)
        
        # Change type embedding
        if 'change_types' in additional_features:
            change_emb = self.change_type_embedding(additional_features['change_types'])
            embeddings.append(change_emb)
        else:
            change_emb = torch.zeros_like(token_emb)
            embeddings.append(change_emb)
        
        # Dependency embedding
        if 'dependencies' in additional_features:
            dep_emb = self.dependency_embedding(additional_features['dependencies'])
            embeddings.append(dep_emb)
        else:
            dep_emb = torch.zeros_like(token_emb)
            embeddings.append(dep_emb)
        
        # Complexity embedding
        if 'complexity_features' in additional_features:
            comp_emb = self.complexity_embedding(additional_features['complexity_features'])
            embeddings.append(comp_emb)
        else:
            comp_emb = torch.zeros_like(token_emb)
            embeddings.append(comp_emb)
        
        # Error/Exception embedding
        if 'error_features' in additional_features:
            error_emb = self.error_embedding(additional_features['error_features'])
            embeddings.append(error_emb)
        else:
            error_emb = torch.zeros_like(token_emb)
            embeddings.append(error_emb)
        
        # Performance embedding
        if 'performance_features' in additional_features:
            perf_emb = self.performance_embedding(additional_features['performance_features'])
            embeddings.append(perf_emb)
        else:
            perf_emb = torch.zeros_like(token_emb)
            embeddings.append(perf_emb)
        
        # Testing embedding
        if 'testing_features' in additional_features:
            test_emb = self.testing_embedding(additional_features['testing_features'])
            embeddings.append(test_emb)
        else:
            test_emb = torch.zeros_like(token_emb)
            embeddings.append(test_emb)
        
        # Code style embedding
        if 'style_features' in additional_features:
            style_emb = self.style_embedding(additional_features['style_features'])
            embeddings.append(style_emb)
        else:
            style_emb = torch.zeros_like(token_emb)
            embeddings.append(style_emb)
        
        # Security embedding
        if 'security_features' in additional_features:
            sec_emb = self.security_embedding(additional_features['security_features'])
            embeddings.append(sec_emb)
        else:
            sec_emb = torch.zeros_like(token_emb)
            embeddings.append(sec_emb)
        
        # API embedding
        if 'api_features' in additional_features:
            api_emb = self.api_embedding(additional_features['api_features'])
            embeddings.append(api_emb)
        else:
            api_emb = torch.zeros_like(token_emb)
            embeddings.append(api_emb)
        
        # Apply fusion mechanism
        fused_embedding = self.fusion_mechanism(embeddings)
        
        output = self.dropout(fused_embedding)
        
        return output