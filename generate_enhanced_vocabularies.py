"""
Script to generate vocabulary pickle files for training.

This script demonstrates how to use the enhanced vocabulary builders
to create the necessary .pkl files for training your model.
"""
import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm
import gc

# Add the tokenizer directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tokenizer'))

from MultimodalVocabulary import MultimodalVocabulary
from EnhancedMsgVocabulary import EnhancedMsgVocabulary


def load_diff_data(file_path, sample_size=None):
    """
    Load diff data from parquet file.
    
    Args:
        file_path: Path to the parquet file containing diff data
        sample_size: Number of samples to load (None for all)
    
    Returns:
        List of diff texts
    """
    print(f"Loading diff data from {file_path}...")
    
    # Process data in chunks to reduce memory usage
    diffs = []
    
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        
        # Take a sample if specified
        if sample_size:
            df = df.head(sample_size)
        
        print(f"Processing {len(df)} diff entries...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading diffs"):
            diff_text = row.get('diff_text', '')
            if pd.notna(diff_text) and diff_text:
                diffs.append(str(diff_text))
            else:
                diffs.append('')
            
            # Periodic cleanup
            if idx % 10000 == 0:
                gc.collect()
        
        gc.collect()
        print(f"Loaded {len(diffs)} diff entries")
        return diffs
        
    except Exception as e:
        print(f"Error loading diff data: {e}")
        # Return sample data for testing
        return [
            "<ADD>def calculate_sum(a, b):</ADD>\n<ADD>    return a + b</ADD>",
            "<REMOVE>def old_function():</REMOVE>\n<ADD>def new_refactored_function():</ADD>",
            "Fixed security vulnerability in auth module",
            "Added performance optimization to data processing",
            "Updated documentation for API endpoints"
        ]


def load_message_data(file_path, sample_size=None):
    """
    Load commit message data from parquet file.
    
    Args:
        file_path: Path to the parquet file containing message data
        sample_size: Number of samples to load (None for all)
    
    Returns:
        List of commit messages
    """
    print(f"Loading message data from {file_path}...")
    
    # Process data in chunks to reduce memory usage
    messages = []
    
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        
        # Take a sample if specified
        if sample_size:
            df = df.head(sample_size)
        
        print(f"Processing {len(df)} message entries...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading messages"):
            message = row.get('message', '')  # Assuming 'message' column exists
            if pd.notna(message) and message:
                messages.append(str(message))
            else:
                messages.append('')
            
            # Periodic cleanup
            if idx % 10000 == 0:
                gc.collect()
        
        gc.collect()
        print(f"Loaded {len(messages)} message entries")
        return messages
        
    except Exception as e:
        print(f"Error loading message data: {e}")
        # Return sample data for testing
        return [
            "Fix critical security vulnerability in auth module",
            "Add new feature for user authentication",
            "Refactor data processing module for better performance",
            "Optimize database queries to reduce latency",
            "Update documentation for API endpoints",
            "Add unit tests for authentication module",
            "Fix typo in README file",
            "Implement caching mechanism for API responses"
        ]


def build_diff_vocabulary(diffs, vocab_size=75000):
    """
    Build vocabulary for diff data.
    
    Args:
        diffs: List of diff texts
        vocab_size: Target vocabulary size
    
    Returns:
        Trained MultimodalVocabulary instance
    """
    print(f"Building diff vocabulary with target size {vocab_size}...")
    
    # Create vocabulary instance
    diff_vocab = MultimodalVocabulary(
        target_size=vocab_size,
        freq_threshold=2,
        batch_size=4096
    )
    
    # Build vocabulary
    diff_vocab.build_vocabulary(diffs)
    
    return diff_vocab


def build_message_vocabulary(messages, vocab_size=10000):
    """
    Build vocabulary for commit messages.
    
    Args:
        messages: List of commit messages
        vocab_size: Target vocabulary size
    
    Returns:
        Trained EnhancedMsgVocabulary instance
    """
    print(f"Building message vocabulary with target size {vocab_size}...")
    
    # Create vocabulary instance
    msg_vocab = EnhancedMsgVocabulary(
        vocab_size=vocab_size,
        min_frequency=2
    )
    
    # Build vocabulary
    msg_vocab.build_vocab(messages)
    
    return msg_vocab


def save_vocabularies(diff_vocab, msg_vocab, diff_output_path, msg_output_path):
    """
    Save the trained vocabularies to pickle files.
    
    Args:
        diff_vocab: Trained diff vocabulary
        msg_vocab: Trained message vocabulary
        diff_output_path: Path to save diff vocabulary
        msg_output_path: Path to save message vocabulary
    """
    print("Saving vocabularies...")
    
    # Save diff vocabulary
    print(f"Saving diff vocabulary to {diff_output_path}...")
    with open(diff_output_path, 'wb') as f:
        pickle.dump(diff_vocab, f)
    print(f"✅ Diff vocabulary saved with {len(diff_vocab.stoi)} tokens")
    
    # Save message vocabulary
    print(f"Saving message vocabulary to {msg_output_path}...")
    with open(msg_output_path, 'wb') as f:
        pickle.dump(msg_vocab, f)
    print(f"✅ Message vocabulary saved with {len(msg_vocab.stoi)} tokens")


def main():
    print("🚀 Building Enhanced Vocabulary Files for Training")
    print("="*60)
    
    # Define file paths (update these to match your actual data paths)
    diff_data_path = "./diff_text.parquet"  # Update to your diff data path
    message_data_path = "./messages.parquet"  # Update to your message data path
    
    # Output paths for vocabulary files
    diff_vocab_path = "./diff_vocab_enhanced.pkl"
    msg_vocab_path = "./message_vocab_enhanced.pkl"
    
    # Check if data files exist, if not use sample data
    use_sample_data = False
    if not os.path.exists(diff_data_path):
        print(f"⚠️  Diff data file not found: {diff_data_path}")
        print("Using sample data for demonstration...")
        use_sample_data = True
    
    if not os.path.exists(message_data_path):
        print(f"⚠️  Message data file not found: {message_data_path}")
        print("Using sample data for demonstration...")
        use_sample_data = True
    
    # Load data
    if use_sample_data:
        diffs = [
            "<ADD>def calculate_sum(a, b):</ADD>\n<ADD>    return a + b</ADD>",
            "<REMOVE>def old_function():</REMOVE>\n<ADD>def new_refactored_function():</ADD>",
            "Fixed security vulnerability in auth module using JWT authentication",
            "Added performance optimization to data processing with caching",
            "Updated documentation for API endpoints in React application",
            "Implemented new feature with Python and Django framework",
            "Fixed bug in JavaScript code with async/await pattern",
            "Added unit tests for PostgreSQL database integration"
        ]
        messages = [
            "Fix critical security vulnerability in auth module with JWT implementation",
            "Add new feature for user authentication using React and Node.js",
            "Refactor data processing module for better performance with caching",
            "Optimize database queries in PostgreSQL to reduce latency",
            "Update documentation for API endpoints in Express.js application",
            "Add unit tests for authentication module with Jest framework",
            "Fix typo in README.md file and improve Python code",
            "Implement caching mechanism with Redis for API responses"
        ]
    else:
        # Load actual data (limit sample size for demonstration)
        diffs = load_diff_data(diff_data_path, sample_size=1000)  # Adjust as needed
        messages = load_message_data(message_data_path, sample_size=1000)  # Adjust as needed
    
    print(f"Sample diff: {diffs[0][:100]}...")
    print(f"Sample message: {messages[0]}")
    
    # Build vocabularies
    print("\n" + "="*60)
    diff_vocab = build_diff_vocabulary(diffs, vocab_size=25000)  # Reduced for demo
    print("\n" + "="*60)
    msg_vocab = build_message_vocabulary(messages, vocab_size=10000)  # Reduced for demo
    
    # Save vocabularies
    print("\n" + "="*60)
    save_vocabularies(diff_vocab, msg_vocab, diff_vocab_path, msg_vocab_path)
    
    print("\n" + "="*60)
    print("✅ All vocabulary files generated successfully!")
    print(f"📊 Diff vocabulary size: {len(diff_vocab.stoi)}")
    print(f"📊 Message vocabulary size: {len(msg_vocab.stoi)}")
    
    # Print some statistics
    print("\n📈 Vocabulary Statistics:")
    print(f"   Diff vocab special tokens: {len([k for k in diff_vocab.stoi.keys() if k.startswith('<') and k.endswith('>')])}")
    print(f"   Message vocab special tokens: {len([k for k in msg_vocab.stoi.keys() if k.startswith('<') and k.endswith('>')])}")
    
    # Show some example tokens
    print(f"\n🔤 Sample diff tokens: {list(diff_vocab.stoi.keys())[:10]}")
    print(f"🔤 Sample message tokens: {list(msg_vocab.stoi.keys())[:10]}")


if __name__ == "__main__":
    main()