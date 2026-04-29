import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .encoder import EncoderLayer, RMSNorm
from .decoder import DecoderLayer
from .embeddings.DiffEmbedding import DiffEmbedding

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, tie_weights=True):
        super(Transformer, self).__init__()
        
        # Use lean DiffEmbedding for encoder (with change-type signal)
        self.encoder_embedding = DiffEmbedding(
            vocab_size=src_vocab_size, 
            d_model=d_model, 
            dropout=dropout
        )
        
        # Simple embedding for decoder (messages don't need change-type signal)
        # Positional encoding handled by RoPE in attention mechanism
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Final layer norms after all encoder/decoder layers
        self.encoder_norm = RMSNorm(d_model)
        self.decoder_norm = RMSNorm(d_model)

        self.fc = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Weight tying: fc.weight == decoder_embedding.weight
        # This works because both map between d_model and tgt_vocab_size
        if tie_weights:
            self.fc.weight = self.decoder_embedding.weight
            print("[Model] Decoder embedding and output projection weights are TIED")

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

    def forward(self, src, tgt, change_features=None):
        """
        Args:
            src: Source sequence (B, S) - code diffs
            tgt: Target sequence (B, T) - commit messages
            change_features: (B, 6) float tensor of binary change-type flags

        Returns:
            output: Logits (B, T, tgt_vocab_size)
        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Pass change_features into the encoder's DiffEmbedding
        src_embedded = self.encoder_embedding(src, change_features)
        src_embedded = self.dropout(src_embedded)
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = checkpoint(
                enc_layer,
                enc_output,
                src_mask,
                preserve_rng_state=True,
                use_reentrant=False
            )
        # Apply final encoder norm
        enc_output = self.encoder_norm(enc_output)
        
        # Decoder - simple embedding without positional encoding (RoPE handles it)
        tgt_embedded = self.decoder_embedding(tgt)
        tgt_embedded = self.dropout(tgt_embedded)
        
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
        # Apply final decoder norm
        dec_output = self.decoder_norm(dec_output)
        
        return self.fc(dec_output)
