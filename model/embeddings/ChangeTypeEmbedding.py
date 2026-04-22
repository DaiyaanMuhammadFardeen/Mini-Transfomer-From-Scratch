# model/embeddings/ChangeTypeEmbedding.py

import torch
import torch.nn as nn


class ChangeTypeEmbedding(nn.Module):
    """
    Lightweight change-type embedding for code diffs.

    Reads a (batch, 6) binary feature tensor computed from the token sequence
    itself — no external metadata needed. Projects it to d_model and adds it
    as a position-independent bias to the encoder input at every position.

    The 6 features correspond to:
        0: <ADD> tag present in this diff
        1: <REMOVE> tag present in this diff
        2: <MODIFY> tag present in this diff
        3: <COMMENT_ADD> tag present
        4: <COMMENT_REMOVE> tag present
        5: <COMMENT_MODIFY> tag present

    These are detected from the token IDs inside CodeDiffDataset.__getitem__,
    so they cost nothing at inference time beyond a single Linear forward pass.
    """

    NUM_FEATURES = 6

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Project the 6-dim binary vector to d_model
        # Two-layer MLP with ReLU so the model can learn non-linear combinations
        # e.g. "ADD + REMOVE together → refactor signal"
        self.proj = nn.Sequential(
            nn.Linear(self.NUM_FEATURES, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize projection weights small so it starts as a gentle nudge
        # and grows only if it helps — stops it dominating the token embedding early
        nn.init.normal_(self.proj[0].weight, std=0.01)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.normal_(self.proj[2].weight, std=0.01)
        nn.init.zeros_(self.proj[2].bias)

    def forward(self, change_features: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            change_features: (batch_size, 6) float tensor of binary change-type flags
            seq_len:         int — length of the token sequence (needed to broadcast)

        Returns:
            (batch_size, seq_len, d_model) tensor — same shape as token embeddings,
            ready to be added directly to them.
        """
        # (batch, 6) → (batch, d_model)
        bias = self.proj(change_features)
        bias = self.norm(bias)
        bias = self.dropout(bias)

        # Expand to every position: (batch, d_model) → (batch, seq_len, d_model)
        # unsqueeze(1) adds the seq dimension, expand copies it across all positions
        bias = bias.unsqueeze(1).expand(-1, seq_len, -1)

        return bias
