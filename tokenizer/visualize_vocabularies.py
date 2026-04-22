"""
Script to visualize both diff and message vocabularies after they have been trained.
"""
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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


def plot_vocab_coverage(word_freqs: dict, max_vocab: int = 50000, output: str = "vocab_coverage.png"):
    """
    Plot cumulative coverage as vocabulary size grows.
    Helps justify vocab_size choice in thesis.
    
    Args:
        word_freqs: Dictionary mapping words to their frequencies
        max_vocab: Maximum vocabulary size to plot
        output: Output file path
    """
    sorted_freqs = sorted(word_freqs.values(), reverse=True)
    total_tokens = sum(sorted_freqs)
    
    coverage = np.cumsum(sorted_freqs) / total_tokens * 100
    vocab_sizes = np.arange(1, min(len(coverage), max_vocab) + 1)
    coverage = coverage[:max_vocab]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(vocab_sizes, coverage, 'b-', linewidth=2, label='Cumulative coverage')
    
    # Mark common vocab sizes
    for vsize in [4000, 8000, 16000, 32000]:
        if vsize < len(coverage):
            cov = coverage[vsize - 1]
            ax.axvline(x=vsize, color='r', linestyle='--', alpha=0.5)
            ax.annotate(f'{vsize:,}\n({cov:.1f}%)', 
                       xy=(vsize, cov), xytext=(vsize + 500, cov - 5),
                       fontsize=9, color='red')
    
    ax.set_xlabel('Vocabulary Size', fontsize=12)
    ax.set_ylabel('Corpus Coverage (%)', fontsize=12)
    ax.set_title('Vocabulary Size vs. Corpus Coverage\n(Diff Tokenizer)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_token_distributions(diff_lengths: list, msg_lengths: list, output: str = "token_distributions.png"):
    """
    Distribution of tokenized sequence lengths for encoder (diff) and decoder (message).
    
    Args:
        diff_lengths: List of tokenized diff sequence lengths
        msg_lengths: List of tokenized message sequence lengths
        output: Output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Diff token lengths
    ax1.hist(diff_lengths, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(np.mean(diff_lengths), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(diff_lengths):.0f}')
    ax1.axvline(np.percentile(diff_lengths, 95), color='orange', linestyle='--', linewidth=2,
                label=f'95th pct: {np.percentile(diff_lengths, 95):.0f}')
    ax1.set_xlabel('Number of Tokens', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Diff (Encoder Input) Token Lengths', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Message token lengths
    ax2.hist(msg_lengths, bins=30, color='coral', edgecolor='white', alpha=0.8)
    ax2.axvline(np.mean(msg_lengths), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(msg_lengths):.0f}')
    ax2.axvline(np.percentile(msg_lengths, 95), color='orange', linestyle='--', linewidth=2,
                label=f'95th pct: {np.percentile(msg_lengths, 95):.0f}')
    ax2.set_xlabel('Number of Tokens', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Commit Message (Decoder Target) Token Lengths', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Token Length Distribution After Tokenization', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_zipf(word_freqs: dict, top_n: int = 100, output: str = "zipf_distribution.png"):
    """
    Log-log plot showing Zipf's Law holds in your tokenized corpus.
    
    Args:
        word_freqs: Dictionary mapping words to their frequencies
        top_n: Number of top tokens to plot
        output: Output file path
    """
    sorted_freqs = sorted(word_freqs.values(), reverse=True)[:top_n]
    ranks = np.arange(1, len(sorted_freqs) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale — shows long tail
    ax1.bar(ranks, sorted_freqs, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Token Rank', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Top-{top_n} Token Frequencies (Linear)', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Log-log scale — shows Zipf's law
    ax2.loglog(ranks, sorted_freqs, 'b-o', markersize=3, alpha=0.7, label='Actual')
    # Fit Zipf curve
    log_rank = np.log(ranks)
    log_freq = np.log(sorted_freqs)
    coeffs = np.polyfit(log_rank, log_freq, 1)
    fitted = np.exp(np.poly1d(coeffs)(log_rank))
    ax2.loglog(ranks, fitted, 'r--', linewidth=2, alpha=0.8,
               label=f'Zipf fit (α={-coeffs[0]:.2f})')
    ax2.set_xlabel('Token Rank (log)', fontsize=12)
    ax2.set_ylabel('Frequency (log)', fontsize=12)
    ax2.set_title("Zipf's Law Verification (Log-Log)", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.suptitle("Token Frequency Distribution — Zipf's Law", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


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
    
    # Generate comprehensive visualizations (existing)
    generate_comprehensive_report(
        diff_vocab, "DiffVocabulary", 
        msg_vocab, "MsgVocabulary", 
        sample_texts
    )
    
    # NEW: Plot A - Vocabulary Coverage
    print("\nGenerating vocabulary coverage plot...")
    if hasattr(diff_vocab, 'word_freqs') and diff_vocab.word_freqs:
        plot_vocab_coverage(
            diff_vocab.word_freqs,
            max_vocab=50000,
            output="visualization_output/vocab_coverage.png"
        )
    else:
        print("Warning: word_freqs not available in diff_vocab, skipping coverage plot")
    
    # NEW: Plot C - Zipf's Law
    print("\nGenerating Zipf's Law distribution plot...")
    if hasattr(diff_vocab, 'word_freqs') and diff_vocab.word_freqs:
        plot_zipf(
            diff_vocab.word_freqs,
            top_n=100,
            output="visualization_output/zipf_distribution.png"
        )
    else:
        print("Warning: word_freqs not available in diff_vocab, skipping Zipf plot")
    
    # NEW: Plot B - Token Length Distributions
    print("\nTo generate token length distributions, you need tokenized data.")
    print("Example usage:")
    print("  from CodeDiffDataset import CodeDiffDataset")
    print("  dataset = CodeDiffDataset(messages, diffs, src_vocab, tgt_vocab, max_seq_length)")
    print("  diff_lengths = [len(dataset[i][0]) for i in range(len(dataset))]")
    print("  msg_lengths = [len(dataset[i][1]) for i in range(len(dataset))]")
    print("  plot_token_distributions(diff_lengths, msg_lengths)")
    
    print("\n✅ All visualizations generated successfully!")
    print("Check the visualization_output directory for generated figures.")


if __name__ == "__main__":
    main()