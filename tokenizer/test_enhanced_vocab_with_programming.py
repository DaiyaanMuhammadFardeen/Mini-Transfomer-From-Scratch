"""
Test script to verify the enhanced vocabulary builders with programming terms
"""
import sys
import os
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MultimodalVocabulary import MultimodalVocabulary
from EnhancedMsgVocabulary import EnhancedMsgVocabulary
from programming_terms import extract_programming_terms, create_programming_tokens


def test_programming_term_extraction():
    print("🧪 Testing Programming Term Extraction...")
    
    test_texts = [
        "Fix security vulnerability in JWT authentication with Python",
        "Add new feature using React and Node.js with PostgreSQL database",
        "Refactor data processing module for better performance with caching in Redis",
        "Update documentation for API endpoints in Express.js application",
        "Add unit tests for authentication module with Jest framework",
        "Fix bug in JavaScript code with async/await pattern",
        "Implemented caching with Redis for API responses in Django app",
        "Optimized database queries in MySQL using indexes"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {text}")
        terms = extract_programming_terms(text)
        tokens = create_programming_tokens(terms)
        print(f"  Extracted terms: {terms}")
        print(f"  Generated tokens: {tokens}")


def test_multimodal_vocab_with_programming():
    print("\n\n🧪 Testing MultimodalVocabulary with Programming Terms...")
    
    # Create sample diff text with programming terms
    sample_diffs = [
        "<ADD>def calculate_sum(a, b):</ADD>\n<ADD>    return a + b</ADD>",
        "<REMOVE>def old_function():</REMOVE>\n<ADD>def new_refactored_function():</ADD>",
        "Fixed security vulnerability in auth module using JWT authentication",
        "Added performance optimization to data processing with caching in Redis",
        "Updated documentation for API endpoints in React application using JavaScript",
        "Implemented new feature with Python and Django framework",
        "Fixed bug in JavaScript code with async/await pattern",
        "Added unit tests for PostgreSQL database integration"
    ]
    
    # Create and build vocabulary
    vocab = MultimodalVocabulary(target_size=10000, freq_threshold=1)
    vocab.build_vocabulary(sample_diffs)
    
    print(f"✅ Vocabulary built with {len(vocab.stoi)} tokens")
    
    # Test tokenization with programming terms
    test_text = "Fix security vulnerability in JWT authentication with Python and Django"
    tokens = vocab.tokenize(test_text)
    print(f"📝 Original: {test_text}")
    print(f"🔤 Tokens: {tokens}")
    
    # Test numericalization
    numericalized = vocab.numericalize(test_text)
    print(f"🔢 Numericalized: {numericalized}")
    
    # Check for programming-specific tokens
    programming_tokens = [token for token in tokens if '_LANG' in token or '_FRAMEWORK' in token or '_CMD' in token or '_TERM' in token or '_EXT' in token]
    print(f"💻 Found programming tokens: {programming_tokens}")
    
    return vocab


def test_enhanced_msg_vocab_with_programming():
    print("\n\n🧪 Testing EnhancedMsgVocabulary with Programming Terms...")
    
    # Create sample commit messages with programming terms
    sample_messages = [
        "Fix critical security vulnerability in auth module with JWT implementation",
        "Add new feature for user authentication using React and Node.js",
        "Refactor data processing module for better performance with caching",
        "Optimize database queries in PostgreSQL to reduce latency",
        "Update documentation for API endpoints in Express.js application",
        "Add unit tests for authentication module with Jest framework",
        "Fix typo in README.md file and improve Python code",
        "Implement caching mechanism with Redis for API responses"
    ]
    
    # Create and build vocabulary
    vocab = EnhancedMsgVocabulary(vocab_size=5000, min_frequency=1)
    vocab.build_vocab(sample_messages)
    
    print(f"✅ Vocabulary built with {len(vocab.stoi)} tokens")
    
    # Test tokenization
    test_text = "Fix security vulnerability in authentication module using Python and JWT"
    tokens = vocab.tokenize(test_text)
    print(f"📝 Original: {test_text}")
    print(f"🔤 Tokens: {tokens}")
    
    # Test numericalization
    numericalized = vocab.numericalize(test_text)
    print(f"🔢 Numericalized: {numericalized}")
    
    # Check for programming-specific tokens
    programming_tokens = [token for token in tokens if '_LANG' in token or '_FRAMEWORK' in token or '_CMD' in token or '_TERM' in token or '_EXT' in token]
    print(f"💻 Found programming tokens: {programming_tokens}")
    
    return vocab


def main():
    print("🚀 Testing Enhanced Vocabulary Builders with Programming Terms")
    print("="*70)
    
    # Test programming term extraction
    test_programming_term_extraction()
    
    # Test multimodal vocabulary
    diff_vocab = test_multimodal_vocab_with_programming()
    
    # Test enhanced message vocabulary
    msg_vocab = test_enhanced_msg_vocab_with_programming()
    
    print("\n" + "="*70)
    print("✅ All tests completed successfully!")
    
    # Save vocabularies for inspection
    print("\n💾 Saving test vocabularies...")
    with open("test_diff_vocab_with_programming.pkl", "wb") as f:
        import pickle
        pickle.dump(diff_vocab, f)
    print("✅ Diff vocabulary saved as test_diff_vocab_with_programming.pkl")
    
    with open("test_msg_vocab_with_programming.pkl", "wb") as f:
        pickle.dump(msg_vocab, f)
    print("✅ Message vocabulary saved as test_msg_vocab_with_programming.pkl")


if __name__ == "__main__":
    main()