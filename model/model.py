import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .encoder import EncoderLayer
from .decoder import DecoderLayer
from .positionalEnc import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Ensure num_heads is consistent
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # Size needed to allocate memory on GPU
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        nopeak_mask = torch.tril(torch.ones((1, 1, tgt_seq_len, tgt_seq_len))).bool()

        device = tgt.device
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        nopeak_mask = nopeak_mask.to(device)

        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = checkpoint(enc_layer, enc_output, src_mask, preserve_rng_state=True, use_reentrant=False)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = checkpoint(dec_layer, dec_output, enc_output, src_mask, tgt_mask, preserve_rng_state=True, use_reentrant=False)

        output = self.fc(dec_output)
        return output
