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
        ## Fixed: src should be diffs, tgt should be messages
        src_text = self.diffs[idx]
        tgt_text = self.messages[idx]

        src_tokens = self.tokenize_text(src_text, self.src_vocab)
        tgt_tokens = self.tokenize_text(tgt_text, self.tgt_vocab)

        src_tokens = [self.src_vocab.stoi["<SOS>"]] + src_tokens + [self.src_vocab.stoi["<EOS>"]]
        tgt_tokens = [self.tgt_vocab.stoi["<SOS>"]] + tgt_tokens + [self.tgt_vocab.stoi["<EOS>"]]

        src_tokens = src_tokens[:self.max_seq_length] + [self.src_vocab.stoi["<PAD>"]] * (self.max_seq_length - len(src_tokens))
        tgt_tokens = tgt_tokens[:self.max_seq_length] + [self.tgt_vocab.stoi["<PAD>"]] * (self.max_seq_length - len(tgt_tokens))

        src_tokens = torch.tensor(src_tokens, dtype=torch.long)
        tgt_tokens = torch.tensor(tgt_tokens, dtype=torch.long)
        
        # For now, return just the tokens since we don't have the additional features yet
        # In a full implementation, this would include additional feature tensors
        return src_tokens, tgt_tokens

    def tokenize_text(self, text, vocab):
        return vocab.numericalize(text)

    def get_multimodal_features(self, idx):
        """
        Placeholder method to return multimodal features for the given index.
        This would be implemented to extract semantic, contextual, and other features
        from the source code diffs and target messages.
        """
        # This is a placeholder - in a real implementation, this would extract
        # the various features for the multimodal embedding system
        return {
            'ast_nodes': None,
            'context_info': None,
            'patterns': None,
            'temporal_features': None,
            'collaborative_features': None,
            'domain_features': None,
            'change_types': None,
            'dependencies': None,
            'complexity_features': None,
            'error_features': None,
            'performance_features': None,
            'testing_features': None,
            'style_features': None,
            'security_features': None,
            'api_features': None
        }