"""
Test script to demonstrate the enhanced vocabulary builders
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MultimodalVocabulary import MultimodalVocabulary
from EnhancedMsgVocabulary import EnhancedMsgVocabulary


def test_multimodal_vocab():
    print("🧪 Testing MultimodalVocabulary...")
    
    # Create sample diff text
    sample_diffs = [
        "<ADD>def calculate_sum(a, b):</ADD>\n<ADD>    return a + b</ADD>",
        "<REMOVE>def old_function():</REMOVE>\n<ADD>def new_refactored_function():</ADD>",
        "Fixed security vulnerability in auth module",
        "Added performance optimization to data processing",
        "Updated documentation for API endpoints"
    ]
    
    # Create and build vocabulary
    vocab = MultimodalVocabulary(target_size=10000, freq_threshold=1)
    vocab.build_vocabulary(sample_diffs)
    
    print(f"✅ Vocabulary built with {len(vocab.stoi)} tokens")
    
    # Test tokenization
    test_text = "<ADD>def fix_security_bug():</ADD>\n<ADD>    # Added security check</ADD>"
    tokens = vocab.tokenize(test_text)
    print(f"📝 Original: {test_text}")
    print(f"🔤 Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
    
    # Test numericalization
    numericalized = vocab.numericalize(test_text)
    print(f"🔢 Numericalized: {numericalized[:10]}{'...' if len(numericalized) > 10 else ''}")
    
    # Check for embedding-specific tokens
    embedding_tokens = [
        '<BUG_FIX>', '<FEATURE_ADD>', '<REFACTOR>', '<OPTIMIZATION>',
        '<SECURITY_FIX>', '<PERFORMANCE>', '<API_CHANGE>', '<TEST_ADD>'
    ]
    
    found_tokens = [token for token in embedding_tokens if token in vocab.stoi]
    print(f"🔍 Found embedding tokens: {found_tokens}")
    
    return vocab


def test_enhanced_msg_vocab():
    print("\n🧪 Testing EnhancedMsgVocabulary...")
    
    # Create sample commit messages
    sample_messages = [
        "Fix critical security vulnerability in auth module",
        "Add new feature for user authentication",
        "Refactor data processing module for better performance",
        "Optimize database queries to reduce latency",
        "Update documentation for API endpoints",
        "Add unit tests for authentication module",
        "Fix typo in README file",
        "Implement caching mechanism for API responses"
    ]
    
    # Create and build vocabulary
    vocab = EnhancedMsgVocabulary(vocab_size=5000, min_frequency=1)
    vocab.build_vocab(sample_messages)
    
    print(f"✅ Vocabulary built with {len(vocab.stoi)} tokens")
    
    # Test tokenization
    test_text = "Fix security vulnerability in authentication module"
    tokens = vocab.tokenize(test_text)
    print(f"📝 Original: {test_text}")
    print(f"🔤 Tokens: {tokens}")
    
    # Test numericalization
    numericalized = vocab.numericalize(test_text)
    print(f"🔢 Numericalized: {numericalized}")
    
    # Check for embedding-specific tokens
    embedding_tokens = [
        '<BUG_FIX>', '<FEATURE_ADD>', '<REFACTOR>', '<OPTIMIZATION>',
        '<SECURITY_FIX>', '<PERFORMANCE>', '<API_CHANGE>', '<TEST_ADD>'
    ]
    
    found_tokens = [token for token in embedding_tokens if token in vocab.stoi]
    print(f"🔍 Found embedding tokens: {found_tokens}")
    
    return vocab


def main():
    print("🚀 Testing Enhanced Vocabulary Builders")
    print("="*50)
    
    # Test multimodal vocabulary
    diff_vocab = test_multimodal_vocab()
    
    # Test enhanced message vocabulary
    msg_vocab = test_enhanced_msg_vocab()
    
    print("\n" + "="*50)
    print("✅ All tests completed successfully!")
    
    # Save vocabularies
    print("\n💾 Saving vocabularies...")
    with open("test_diff_vocab.pkl", "wb") as f:
        import pickle
        pickle.dump(diff_vocab, f)
    print("✅ Diff vocabulary saved as test_diff_vocab.pkl")
    
    with open("test_msg_vocab.pkl", "wb") as f:
        pickle.dump(msg_vocab, f)
    print("✅ Message vocabulary saved as test_msg_vocab.pkl")


if __name__ == "__main__":
    main()