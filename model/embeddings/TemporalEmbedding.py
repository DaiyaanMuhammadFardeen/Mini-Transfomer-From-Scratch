import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalEmbedding(nn.Module):
    """
    Temporal Embedding Layer that captures time-based context like commit sequence,
    time gaps, and frequency patterns.
    """
    
    def __init__(self, d_model: int):
        super(TemporalEmbedding, self).__init__()
        
        self.d_model = d_model
        
        # Positional embedding for commit sequence position
        self.positional_embedding = nn.Embedding(10000, d_model)  # For sequence up to 10k commits
        
        # Time gap embedding
        self.time_gap_embedding = nn.Linear(1, d_model)  # For continuous time gap values
        
        # Frequency pattern embedding
        self.frequency_embedding = nn.Linear(1, d_model)  # For continuous frequency values
        
        # Time of day/week embedding
        self.time_of_day_embedding = nn.Embedding(24, d_model // 4)  # Hours of day
        self.day_of_week_embedding = nn.Embedding(7, d_model // 4)   # Days of week
        
        # Release cycle embedding
        self.release_cycle_embedding = nn.Embedding(50, d_model // 2)  # Up to 50 release cycles
        
        # Combination layer for all temporal features
        self.temporal_combination = nn.Linear(d_model * 3 + d_model // 2 + d_model // 2, d_model)
        
        # LSTM for temporal sequence modeling
        self.temporal_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Final projection
        self.projection = nn.Linear(d_model * 2, d_model)  # For bidirectional LSTM output
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, temporal_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for temporal embedding.
        
        Args:
            temporal_features: Temporal features tensor (batch_size, seq_len, num_temporal_features)
            
        Returns:
            Temporal embedding tensor (batch_size, seq_len, d_model)
        """
        if temporal_features is None:
            # If no temporal features provided, return zero embeddings
            batch_size = 1
            seq_len = 10  # Default sequence length
            device = next(self.parameters()).device
            return torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=torch.float)
        
        batch_size, seq_len, num_features = temporal_features.shape
        device = temporal_features.device
        
        # Extract different temporal features
        # Assume temporal_features has shape (batch_size, seq_len, num_features) where:
        # 0: commit sequence position
        # 1: time gap in days
        # 2: commit frequency
        # 3: hour of day (0-23)
        # 4: day of week (0-6)
        # 5: release cycle index
        
        # Commit sequence position embedding
        if num_features > 0:
            sequence_positions = temporal_features[:, :, 0].long().clamp(0, 9999)
            pos_emb = self.positional_embedding(sequence_positions)  # (batch_size, seq_len, d_model)
        else:
            pos_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Time gap embedding
        if num_features > 1:
            time_gaps = temporal_features[:, :, 1:2]  # (batch_size, seq_len, 1)
            gap_emb = self.time_gap_embedding(time_gaps)  # (batch_size, seq_len, d_model)
        else:
            gap_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Frequency pattern embedding
        if num_features > 2:
            frequencies = temporal_features[:, :, 2:3]  # (batch_size, seq_len, 1)
            freq_emb = self.frequency_embedding(frequencies)  # (batch_size, seq_len, d_model)
        else:
            freq_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Time of day and day of week embeddings
        time_components = []
        if num_features > 3:
            hours = temporal_features[:, :, 3].long().clamp(0, 23)
            hour_emb = F.pad(self.time_of_day_embedding(hours), (0, self.d_model - self.d_model // 4))
            time_components.append(hour_emb)
        else:
            time_components.append(torch.zeros(batch_size, seq_len, self.d_model, device=device))
        
        if num_features > 4:
            days = temporal_features[:, :, 4].long().clamp(0, 6)
            day_emb = F.pad(self.day_of_week_embedding(days), (0, self.d_model - self.d_model // 4))
            time_components.append(day_emb)
        else:
            time_components.append(torch.zeros(batch_size, seq_len, self.d_model, device=device))
        
        # Release cycle embedding
        if num_features > 5:
            cycles = temporal_features[:, :, 5].long().clamp(0, 49)
            cycle_emb = F.pad(self.release_cycle_embedding(cycles), (0, self.d_model - self.d_model // 2))
        else:
            cycle_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Combine all temporal embeddings
        combined_temporal = pos_emb + gap_emb + freq_emb + sum(time_components) + cycle_emb
        
        # Apply LSTM for temporal sequence modeling
        lstm_output, _ = self.temporal_lstm(combined_temporal)
        
        # Apply normalization and dropout
        output = self.norm(lstm_output)
        output = self.dropout(output)
        
        # Final projection to maintain d_model size
        output = self.projection(output)
        output = self.norm(output)
        
        return output