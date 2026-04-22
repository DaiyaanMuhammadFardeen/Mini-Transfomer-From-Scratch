#!/usr/bin/env python3
"""
Analyze simple space-separated tokenization for commit messages.
This will help determine if BPE is overengineering or necessary.
"""

import pandas as pd
import re
from collections import Counter
from tqdm import tqdm
import sys

def load_data(filepath):
    """Load the training data"""
    print("Loading data...")
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} samples")
    return df

def simple_tokenize(text):
    """Simple space-separated tokenization with basic preprocessing"""
    if not isinstance(text, str):
        return []
    
    # Basic cleaning
    text = text.strip().lower()
    
    # Simple split by whitespace
    tokens = text.split()
    
    # Filter out empty tokens
    tokens = [token for token in tokens if token.strip()]
    
    return tokens

def advanced_tokenize(text):
    """More sophisticated tokenization that handles punctuation and special cases"""
    if not isinstance(text, str):
        return []
    
    # Clean and normalize
    text = ' '.join(text.split())  # Normalize whitespace
    
    # Pattern for better tokenization
    pattern = r"""
        (?:[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[a-z]+|[A-Z]+)|  # CamelCase and words
        (?:\w+(?:_\w+)+)|                                       # snake_case
        (?:\d+\.\d+(?:\.\d+)*)|                                # Version numbers
        (?:[a-zA-Z0-9]+(?:[/.][a-zA-Z0-9]+)+)|                 # Paths/URLs
        (?:[#@]\w+)|                                           # Hashtags, mentions
        (?:\w+)|                                               # Regular words
        (?:[^\w\s])                                            # Punctuation
    """
    
    tokens = re.findall(pattern, text, re.VERBOSE)
    return [token.lower() for token in tokens if token.strip()]

def analyze_tokenization(df, msg_column='msg'):
    """Analyze different tokenization approaches"""
    
    print("\n" + "="*60)
    print("TOKENIZATION ANALYSIS")
    print("="*60)
    
    # Sample a reasonable number for analysis
    sample_size = max(10000, len(df))
    sample_df = df
    
    print(f"\nAnalyzing {len(df):,} samples...")
    
    # Collect statistics
    simple_tokens_list = []
    advanced_tokens_list = []
    sequence_lengths_simple = []
    sequence_lengths_advanced = []
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Tokenizing"):
        msg = row[msg_column] if msg_column in row else str(row[0])
        
        # Simple tokenization
        simple_tokens = simple_tokenize(msg)
        simple_tokens_list.extend(simple_tokens)
        sequence_lengths_simple.append(len(simple_tokens))
        
        # Advanced tokenization
        advanced_tokens = advanced_tokenize(msg)
        advanced_tokens_list.extend(advanced_tokens)
        sequence_lengths_advanced.append(len(advanced_tokens))
    
    # Statistics for simple tokenization
    simple_vocab = set(simple_tokens_list)
    simple_token_counter = Counter(simple_tokens_list)
    
    # Statistics for advanced tokenization
    advanced_vocab = set(advanced_tokens_list)
    advanced_token_counter = Counter(advanced_tokens_list)
    
    print(f"\nSIMPLE TOKENIZATION (space-separated):")
    print(f"  Unique tokens: {len(simple_vocab):,}")
    print(f"  Total tokens: {len(simple_tokens_list):,}")
    print(f"  Average sequence length: {sum(sequence_lengths_simple)/len(sequence_lengths_simple):.1f}")
    print(f"  Max sequence length: {max(sequence_lengths_simple)}")
    print(f"  Min sequence length: {min(sequence_lengths_simple)}")
    
    print(f"\nADVANCED TOKENIZATION (regex-based):")
    print(f"  Unique tokens: {len(advanced_vocab):,}")
    print(f"  Total tokens: {len(advanced_tokens_list):,}")
    print(f"  Average sequence length: {sum(sequence_lengths_advanced)/len(sequence_lengths_advanced):.1f}")
    print(f"  Max sequence length: {max(sequence_lengths_advanced)}")
    print(f"  Min sequence length: {min(sequence_lengths_advanced)}")
    
    # Show most common tokens
    print(f"\nMOST COMMON SIMPLE TOKENS:")
    for token, count in simple_token_counter.most_common(20):
        print(f"  '{token}': {count:,}")
    
    print(f"\nMOST COMMON ADVANCED TOKENS:")
    for token, count in advanced_token_counter.most_common(20):
        print(f"  '{token}': {count:,}")
    
    # Vocabulary size analysis
    vocab_sizes = [1000, 5000, 10000, 20000, 50000]
    print(f"\nVOCABULARY COVERAGE ANALYSIS:")
    print(f"{'Vocab Size':<12} {'Simple Coverage %':<20} {'Advanced Coverage %':<20}")
    print("-" * 55)
    
    for vocab_size in vocab_sizes:
        simple_covered = sum(count for token, count in simple_token_counter.most_common(vocab_size))
        simple_coverage = (simple_covered / len(simple_tokens_list)) * 100
        
        advanced_covered = sum(count for token, count in advanced_token_counter.most_common(vocab_size))
        advanced_coverage = (advanced_covered / len(advanced_tokens_list)) * 100
        
        print(f"{vocab_size:<12} {simple_coverage:<20.2f} {advanced_coverage:<20.2f}")
    
    # Sequence length percentiles
    sequence_lengths_simple.sort()
    sequence_lengths_advanced.sort()
    
    percentiles = [50, 75, 90, 95, 99]
    print(f"\nSEQUENCE LENGTH PERCENTILES:")
    print(f"{'Percentile':<12} {'Simple Length':<15} {'Advanced Length':<15}")
    print("-" * 45)
    
    for p in percentiles:
        simple_idx = int((p/100) * len(sequence_lengths_simple)) - 1
        advanced_idx = int((p/100) * len(sequence_lengths_advanced)) - 1
        
        simple_val = sequence_lengths_simple[simple_idx]
        advanced_val = sequence_lengths_advanced[advanced_idx]
        
        print(f"{p}%{'tile':<9} {simple_val:<15} {advanced_val:<15}")
    
    return {
        'simple': {
            'vocab_size': len(simple_vocab),
            'total_tokens': len(simple_tokens_list),
            'avg_length': sum(sequence_lengths_simple)/len(sequence_lengths_simple),
            'max_length': max(sequence_lengths_simple),
            'token_counter': simple_token_counter
        },
        'advanced': {
            'vocab_size': len(advanced_vocab),
            'total_tokens': len(advanced_tokens_list),
            'avg_length': sum(sequence_lengths_advanced)/len(sequence_lengths_advanced),
            'max_length': max(sequence_lengths_advanced),
            'token_counter': advanced_token_counter
        }
    }

def analyze_full_dataset(df, msg_column='msg'):
    """Analyze the full dataset for comprehensive statistics"""
    print(f"\n" + "="*60)
    print("FULL DATASET ANALYSIS")
    print("="*60)
    
    print(f"Total samples: {len(df):,}")
    
    # Check for empty/null messages
    null_count = df[msg_column].isnull().sum() if msg_column in df.columns else 0
    empty_count = (df[msg_column] == '').sum() if msg_column in df.columns else 0
    
    print(f"Null messages: {null_count:,}")
    print(f"Empty messages: {empty_count:,}")
    
    # Analyze message lengths in characters
    if msg_column in df.columns:
        msg_lengths = df[msg_column].astype(str).str.len()
    else:
        msg_lengths = df.iloc[:, 0].astype(str).str.len()
    
    print(f"\nMESSAGE LENGTH STATISTICS (characters):")
    print(f"  Average: {msg_lengths.mean():.1f}")
    print(f"  Median: {msg_lengths.median():.1f}")
    print(f"  Std Dev: {msg_lengths.std():.1f}")
    print(f"  Min: {msg_lengths.min()}")
    print(f"  Max: {msg_lengths.max()}")
    
    # Length percentiles
    percentiles = [50, 75, 90, 95, 99]
    print(f"\nLENGTH PERCENTILES (characters):")
    for p in percentiles:
        val = msg_lengths.quantile(p/100)
        print(f"  {p}%: {val:.0f}")

def main():
    # Load data
    data_file = '/home/daiyaan2002/Desktop/Projects/Mini-Transfomer-From-Scratch/traindata.parquet'
    
    try:
        df = load_data(data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Analyze full dataset
    analyze_full_dataset(df)
    
    # Tokenization analysis
    results = analyze_tokenization(df)
    
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    simple_stats = results['simple']
    advanced_stats = results['advanced']
    
    print(f"\nBased on the analysis:")
    print(f"1. Simple space-separated tokenization yields {simple_stats['vocab_size']:,} unique tokens")
    print(f"2. Advanced regex tokenization yields {advanced_stats['vocab_size']:,} unique tokens")
    print(f"3. Average sequence length: {simple_stats['avg_length']:.1f} (simple) vs {advanced_stats['avg_length']:.1f} (advanced)")
    print(f"4. Maximum sequence length: {simple_stats['max_length']} (simple) vs {advanced_stats['max_length']} (advanced)")
    
    if simple_stats['vocab_size'] <= 20000:
        print(f"\n💡 RECOMMENDATION: Simple tokenization may be sufficient!")
        print(f"   With ~{simple_stats['vocab_size']:,} tokens, you can likely avoid BPE complexity.")
    else:
        print(f"\n⚠️  CONSIDERATION: Vocabulary size ({simple_stats['vocab_size']:,}) might benefit from subword tokenization.")

if __name__ == "__main__":
    main()