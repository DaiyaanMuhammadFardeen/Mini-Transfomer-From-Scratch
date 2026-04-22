"""
Script to visualize both diff and message vocabularies after they have been trained.
"""
import pickle
import os
from visualization import generate_comprehensive_report


def load_vocabularies(diff_vocab_path: str = "diff_vocab.pkl", msg_vocab_path: str = "message_vocab.pkl"):
    """Load both vocabularies from pickle files."""
    print("Loading vocabularies...")
    
    with open(diff_vocab_path, 'rb') as f:
        diff_vocab = pickle.load(f)
    
    with open(msg_vocab_path, 'rb') as f:
        msg_vocab = pickle.load(f)
    
    print(f"Loaded DiffVocabulary with {len(diff_vocab.stoi)} tokens")
    print(f"Loaded MsgVocabulary with {len(msg_vocab.stoi)} tokens")
    
    return diff_vocab, msg_vocab


def main():
    """Main function to generate visualizations for both vocabularies."""
    print("Generating comprehensive visualizations for both vocabularies...")
    
    # Load vocabularies
    try:
        diff_vocab, msg_vocab = load_vocabularies()
    except FileNotFoundError as e:
        print(f"Error: Could not find vocabulary files: {e}")
        print("Please run the vocabulary generation scripts first:")
        print("1. python generateDiffToken.py")
        print("2. python generateMsgToken.py")
        return
    
    # Sample texts for visualization
    sample_texts = [
        "Fix security vulnerability in JWT authentication with Python Django and PostgreSQL",
        "Add new feature for user authentication using React and Node.js",
        "Refactor data processing module for better performance with caching",
        "Update documentation for API endpoints in Express.js application",
        "Add unit tests for authentication module with Jest framework"
    ]
    
    # Generate comprehensive visualizations
    generate_comprehensive_report(
        diff_vocab, "DiffVocabulary", 
        msg_vocab, "MsgVocabulary", 
        sample_texts
    )
    
    print("✅ All visualizations generated successfully!")
    print("Check the visualization_output directory for generated figures.")


if __name__ == "__main__":
    main()