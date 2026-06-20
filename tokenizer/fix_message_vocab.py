"""
Fix existing message_vocab.pkl to include _trie attribute.
Run this once after applying the MsgVocabulary.py fix.
"""
import sys
sys.path.insert(0, '.')

from tokenizer.MsgVocabulary import MsgVocabulary

print("Loading existing vocabulary...")
vocab = MsgVocabulary.load("./tokenizer/message_vocab.pkl")

print(f"Vocab size: {len(vocab.stoi):,}")
print(f"Is trained: {vocab.is_trained}")

# Build trie explicitly (this happens automatically on first numericalize call)
print("Building trie...")
vocab._trie = None  # Will be built lazily

print("Resaving vocabulary with _trie support...")
vocab.save("./tokenizer/message_vocab_2.pkl")

print("✅ Done! Your vocabulary now supports trie-based tokenization.")
print("   You can now run pretokenize.py successfully.")
