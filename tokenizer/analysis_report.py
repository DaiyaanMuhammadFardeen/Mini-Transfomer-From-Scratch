"""
Comprehensive analysis report for the enhanced vocabulary building system.
"""
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
from visualization import setup_plot_style
import os


def generate_analysis_report(diff_vocab_path: str = "diff_vocab.pkl", msg_vocab_path: str = "message_vocab.pkl"):
    """Generate a comprehensive analysis report for both vocabularies."""
    print("Generating comprehensive analysis report...")
    
    # Load vocabularies
    try:
        with open(diff_vocab_path, 'rb') as f:
            diff_vocab = pickle.load(f)
        
        with open(msg_vocab_path, 'rb') as f:
            msg_vocab = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find vocabulary files: {e}")
        print("Please run the vocabulary generation scripts first.")
        return
    
    # Create output directory
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup plot style
    setup_plot_style()
    
    # 1. Basic Statistics
    print("\n📊 Basic Vocabulary Statistics:")
    print(f"Diff Vocabulary Size: {len(diff_vocab.stoi):,}")
    print(f"Message Vocabulary Size: {len(msg_vocab.stoi):,}")
    
    # Count special tokens
    diff_special = [t for t in diff_vocab.stoi.keys() if t.startswith('<') and t.endswith('>')]
    msg_special = [t for t in msg_vocab.stoi.keys() if t.startswith('<') and t.endswith('>')]
    
    print(f"Diff Special Tokens: {len(diff_special):,}")
    print(f"Message Special Tokens: {len(msg_special):,}")
    
    # 2. Token Length Analysis
    diff_token_lengths = [len(token) for token in diff_vocab.stoi.keys()]
    msg_token_lengths = [len(token) for token in msg_vocab.stoi.keys()]
    
    print(f"\n📏 Token Length Statistics:")
    print(f"Diff - Mean: {np.mean(diff_token_lengths):.2f}, Max: {max(diff_token_lengths)}, Min: {min(diff_token_lengths)}")
    print(f"Msg - Mean: {np.mean(msg_token_lengths):.2f}, Max: {max(msg_token_lengths)}, Min: {min(msg_token_lengths)}")
    
    # 3. Generate plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Vocabulary Analysis Report', fontsize=16, fontweight='bold')
    
    # Plot 1: Vocabulary size comparison
    sizes = [len(diff_vocab.stoi), len(msg_vocab.stoi)]
    special_counts = [len(diff_special), len(msg_special)]
    regular_counts = [len(diff_vocab.stoi) - len(diff_special), len(msg_vocab.stoi) - len(msg_special)]
    
    x = np.arange(2)
    width = 0.35
    axes[0, 0].bar(x - width/2, special_counts, width, label='Special', color='orange', alpha=0.7)
    axes[0, 0].bar(x + width/2, regular_counts, width, label='Regular', color='lightblue', alpha=0.7)
    axes[0, 0].set_title('Special vs Regular Tokens')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(['Diff', 'Msg'])
    axes[0, 0].legend()
    
    # Plot 2: Token length distributions
    axes[0, 1].hist(diff_token_lengths, bins=50, alpha=0.5, label='Diff', color='skyblue', edgecolor='black')
    axes[0, 1].hist(msg_token_lengths, bins=50, alpha=0.5, label='Msg', color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Token Length Distribution')
    axes[0, 1].set_xlabel('Token Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Plot 3: Special token categories for DiffVocabulary
    diff_special_categories = {}
    for token in diff_special:
        if '_' in token:
            category = token.split('_')[1].rstrip('>')
            diff_special_categories[category] = diff_special_categories.get(category, 0) + 1
        else:
            diff_special_categories['Other'] = diff_special_categories.get('Other', 0) + 1
    
    if diff_special_categories:
        categories = list(diff_special_categories.keys())[:10]  # Top 10 categories
        counts = [diff_special_categories[c] for c in categories]
        axes[0, 2].barh(range(len(categories)), counts)
        axes[0, 2].set_yticks(range(len(categories)))
        axes[0, 2].set_yticklabels(categories)
        axes[0, 2].set_title('Diff Vocabulary - Special Token Categories')
        axes[0, 2].set_xlabel('Count')
    
    # Plot 4: Programming term tokens analysis
    # Count programming-related tokens
    diff_prog_tokens = [t for t in diff_special if 
                       any(keyword in t.lower() for keyword in 
                           ['lang', 'framework', 'cmd', 'term', 'ext', 'construct', 'http'])]
    msg_prog_tokens = [t for t in msg_special if 
                      any(keyword in t.lower() for keyword in 
                          ['lang', 'framework', 'cmd', 'term', 'ext', 'construct', 'http'])]
    
    prog_counts = [len(diff_prog_tokens), len(msg_prog_tokens)]
    other_special_counts = [len(diff_special) - len(diff_prog_tokens), len(msg_special) - len(msg_prog_tokens)]
    
    axes[1, 0].bar(['Diff', 'Msg'], prog_counts, label='Programming Terms', color='green', alpha=0.7)
    axes[1, 0].bar(['Diff', 'Msg'], other_special_counts, bottom=prog_counts, 
                   label='Other Special', color='gray', alpha=0.7)
    axes[1, 0].set_title('Programming Terms vs Other Special Tokens')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # Plot 5: Top token lengths
    diff_length_counter = Counter(diff_token_lengths)
    msg_length_counter = Counter(msg_token_lengths)
    
    # Get top 15 most common lengths
    diff_top_lengths = diff_length_counter.most_common(15)
    msg_top_lengths = msg_length_counter.most_common(15)
    
    if diff_top_lengths:
        lengths, counts = zip(*diff_top_lengths)
        axes[1, 1].bar(lengths, counts, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('Top 15 Diff Token Lengths')
        axes[1, 1].set_xlabel('Token Length')
        axes[1, 1].set_ylabel('Frequency')
    
    # Plot 6: Vocabulary overlap analysis (if we had common tokens to compare)
    # For now, we'll show the distribution of token types
    diff_types = {
        'Programming': len([t for t in diff_special if any(kw in t.lower() for kw in ['lang', 'framework', 'cmd', 'term', 'ext'])),
        'Change Type': len([t for t in diff_special if 'FIX' in t or 'ADD' in t or 'REFACTOR' in t]),
        'Code Style': len([t for t in diff_special if 'CASE' in t or 'STYLE' in t]),
        'Other': len(diff_special) - len([t for t in diff_special if any(kw in t.lower() for kw in ['lang', 'framework', 'cmd', 'term', 'ext', 'FIX', 'ADD', 'REFACTOR', 'CASE', 'STYLE'])])
    }
    
    if any(diff_types.values()):
        axes[1, 2].pie(diff_types.values(), labels=diff_types.keys(), autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Diff Vocabulary - Token Type Distribution')
    
    plt.tight_layout()
    
    # Save the report
    report_path = os.path.join(output_dir, "vocabulary_analysis_report.png")
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Analysis report saved to: {report_path}")
    plt.close()
    
    # 4. Generate detailed statistics
    stats_path = os.path.join(output_dir, "vocabulary_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write("VOCABULARY STATISTICS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("DIFF VOCABULARY:\n")
        f.write(f"- Total tokens: {len(diff_vocab.stoi):,}\n")
        f.write(f"- Special tokens: {len(diff_special):,}\n")
        f.write(f"- Regular tokens: {len(diff_vocab.stoi) - len(diff_special):,}\n")
        f.write(f"- Mean token length: {np.mean(diff_token_lengths):.2f}\n")
        f.write(f"- Max token length: {max(diff_token_lengths)}\n")
        f.write(f"- Min token length: {min(diff_token_lengths)}\n\n")
        
        f.write("MESSAGE VOCABULARY:\n")
        f.write(f"- Total tokens: {len(msg_vocab.stoi):,}\n")
        f.write(f"- Special tokens: {len(msg_special):,}\n")
        f.write(f"- Regular tokens: {len(msg_vocab.stoi) - len(msg_special):,}\n")
        f.write(f"- Mean token length: {np.mean(msg_token_lengths):.2f}\n")
        f.write(f"- Max token length: {max(msg_token_lengths)}\n")
        f.write(f"- Min token length: {min(msg_token_lengths)}\n\n")
        
        f.write("SPECIAL TOKEN CATEGORIES (DiffVocabulary):\n")
        for category, count in sorted(diff_special_categories.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {category}: {count}\n")
        
        f.write(f"\nPROGRAMMING TERM TOKENS:\n")
        f.write(f"- Diff Vocabulary: {len(diff_prog_tokens)}\n")
        f.write(f"- Message Vocabulary: {len(msg_prog_tokens)}\n")
    
    print(f"📋 Detailed statistics saved to: {stats_path}")
    
    print("\n✅ Comprehensive analysis report generated successfully!")


def main():
    """Main function to run the analysis."""
    generate_analysis_report()


if __name__ == "__main__":
    main()