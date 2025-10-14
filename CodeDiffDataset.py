import torch
from torch.utils.data import Dataset

class CodeDiffDataset(Dataset):
    def __init__(self, messages, diffs, src_vocab, tgt_vocab, max_seq_length):
        self.messages = messages
        self.diffs = diffs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        src_text = self.messages[idx]
        tgt_text = self.diffs[idx]

        src_tokens = self.tokenize_text(src_text, self.src_vocab)
        tgt_tokens = self.tokenize_text(tgt_text, self.tgt_vocab)

        src_tokens = [self.src_vocab.stoi["<SOS>"]] + src_tokens + [self.src_vocab.stoi["<EOS>"]]
        tgt_tokens = [self.tgt_vocab.stoi["<SOS>"]] + tgt_tokens + [self.tgt_vocab.stoi["<EOS>"]]

        src_tokens = src_tokens[:self.max_seq_length] + [self.src_vocab.stoi["<PAD>"]] * (self.max_seq_length - len(src_tokens))
        tgt_tokens = tgt_tokens[:self.max_seq_length] + [self.tgt_vocab.stoi["<PAD>"]] * (self.max_seq_length - len(tgt_tokens))

        src_tokens = torch.tensor(src_tokens, dtype=torch.long)
        tgt_tokens = torch.tensor(tgt_tokens, dtype=torch.long)
        return src_tokens, tgt_tokens

    def tokenize_text(self, text, vocab):
        return vocab.numericalize(text)
