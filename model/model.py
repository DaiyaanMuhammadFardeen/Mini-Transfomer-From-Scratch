import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .encoder import EncoderLayer
from .decoder import DecoderLayer
from .embeddings.MultimodalEmbedding import MultimodalEmbedding

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        
        # Replace the simple embedding with the multimodal embedding system
        self.encoder_embedding = MultimodalEmbedding(
            vocab_size=src_vocab_size, 
            d_model=d_model, 
            max_seq_length=max_seq_length
        )
        self.decoder_embedding = MultimodalEmbedding(
            vocab_size=tgt_vocab_size, 
            d_model=d_model, 
            max_seq_length=max_seq_length
        )

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        Generate attention masks for source and target sequences.
        Ensures all tensors are on the same device as input tensors.

        Args:
            src: Source sequence (B, S)
            tgt: Target sequence (B, T)

        Returns:
            src_mask: (B, 1, 1, S) - Mask for padding tokens in source
            tgt_mask: (B, 1, T, T) - Causal mask for target with padding mask
        """
        # Get device and dtype from input
        device = src.device
        dtype = src.dtype

        # Get sequence lengths
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)

        # Create padding mask for source: 1 for real tokens, 0 for padding
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

        # Create padding mask for target: 1 for real tokens, 0 for padding
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # (B, 1, T, 1)

        # Create causal mask on the correct device: upper triangular matrix is False (masked)
        nopeak_mask = torch.tril(
            torch.ones((tgt_seq_len, tgt_seq_len), device=device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Ensure all masks are on the correct device
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        nopeak_mask = nopeak_mask.to(device)

        # Combine target padding mask and causal mask
        # Both must be True for a position to be unmasked
        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt, src_features=None, tgt_features=None):
        """
        Args:
            src: Source sequence (B, S) - code diffs
            tgt: Target sequence (B, T) - commit messages
            src_features: Dictionary of additional source features for multimodal embedding
            tgt_features: Dictionary of additional target features for multimodal embedding

        Returns:
            output: Logits (B, T, tgt_vocab_size)
        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # Embed using multimodal embedding system
        src_embedded = self.encoder_embedding(src, src_features)
        tgt_embedded = self.decoder_embedding(tgt, tgt_features)

        # Apply dropout after embedding
        src_embedded = self.dropout(src_embedded)
        tgt_embedded = self.dropout(tgt_embedded)

        # Encoder pass
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = checkpoint(
                enc_layer,
                enc_output,
                src_mask,
                preserve_rng_state=True,
                use_reentrant=False
            )

        # Decoder pass
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = checkpoint(
                dec_layer,
                dec_output,
                enc_output,
                src_mask,
                tgt_mask,
                preserve_rng_state=True,
                use_reentrant=False
            )

        # Output projection
        output = self.fc(dec_output)
        return output