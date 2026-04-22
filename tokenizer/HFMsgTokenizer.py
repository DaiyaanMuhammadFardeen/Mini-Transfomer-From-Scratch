"""
HuggingFace tokenizer wrapper for commit messages.
Provides compatible interface with old MsgVocabulary class.
"""
from tokenizers import Tokenizer


class HFMsgTokenizer:
    """Wrapper around HuggingFace tokenizer for commit messages."""
    
    def __init__(self, tokenizer_path: str = "message_tokenizer.json"):
        """Load tokenizer from JSON file."""
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.stoi = self.tokenizer.get_vocab()
        self.itos = {v: k for k, v in self.stoi.items()}
        
    def tokenize(self, text: str) -> list:
        """Tokenize text and return tokens as strings."""
        encoding = self.tokenizer.encode(text)
        return [self.itos[token_id] for token_id in encoding.ids]
    
    def numericalize(self, text: str) -> list:
        """Convert text to token IDs."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, token_ids: list) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids)
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.stoi)
