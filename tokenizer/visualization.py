"""
Visualization module for tokenization process analysis and research paper figures.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import pickle
from collections import Counter
import os


def setup_plot_style():
    """Set up the plotting style for research paper quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })


def visualize_vocabulary_statistics(vocab, title: str = "Vocabulary Statistics"):
    """Create comprehensive visualizations for vocabulary statistics."""
    setup_plot_style()
    
    # Prepare data
    vocab_size = len(vocab.stoi)
    special_tokens = [token for token in vocab.stoi.keys() if token.startswith('<') and token.endswith('>')]
    regular_tokens = [token for token in vocab.stoi.keys() if not (token.startswith('<') and token.endswith('>'))]
    
    # Count token types
    special_count = len(special_tokens)
    regular_count = len(regular_tokens)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title} - Vocabulary Analysis', fontsize=16, fontweight='bold')
    
    # 1. Pie chart of special vs regular tokens
    axes[0, 0].pie([special_count, regular_count], 
                    labels=['Special Tokens', 'Regular Tokens'], 
                    autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Token Type Distribution')
    
    # 2. Bar chart of special token categories
    if hasattr(vocab, 'special_tokens'):
        special_categories = {}
        for token in special_tokens:
            if '_' in token:
                category = token.split('_')[1].rstrip('>')
                special_categories[category] = special_categories.get(category, 0) + 1
            else:
                special_categories['Other'] = special_categories.get('Other', 0) + 1
        
        if special_categories:
            categories = list(special_categories.keys())
            counts = list(special_categories.values())
            axes[0, 1].bar(categories, counts)
            axes[0, 1].set_title('Special Token Categories')
            axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Token length distribution
    token_lengths = [len(token) for token in vocab.stoi.keys()]
    axes[1, 0].hist(token_lengths, bins=30, edgecolor='black')
    axes[1, 0].set_title('Token Length Distribution')
    axes[1, 0].set_xlabel('Token Length')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. Top frequent tokens (if model has frequency info)
    if hasattr(vocab, 'trained_model') and vocab.trained_model:
        # Sort tokens by frequency (inverse of model score)
        sorted_tokens = sorted(vocab.trained_model.items(), key=lambda x: x[1], reverse=True)
        top_tokens = sorted_tokens[:20]  # Top 20 most frequent
        
        if top_tokens:
            tokens, scores = zip(*top_tokens)
            tokens = [str(t)[:20] for t in tokens]  # Truncate long tokens
            axes[1, 1].barh(range(len(tokens)), scores)
            axes[1, 1].set_yticks(range(len(tokens)))
            axes[1, 1].set_yticklabels(tokens)
            axes[1, 1].set_title('Top 20 Frequent Tokens')
            axes[1, 1].set_xlabel('Frequency Score')
    
    plt.tight_layout()
    return fig


def visualize_tokenization_process(original_text: str, tokens: List[str], 
                                  title: str = "Tokenization Process Visualization"):
    """Visualize the tokenization process for a given text."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create a heatmap-like visualization
    token_positions = list(range(len(tokens)))
    token_lengths = [len(token) for token in tokens]
    
    # Create a color map based on token types
    colors = []
    for token in tokens:
        if token.startswith('<') and token.endswith('>'):
            colors.append('red')  # Special tokens
        elif token.startswith('.') or any(ext in token for ext in ['py', 'js', 'ts', 'java', 'cpp', 'c']):
            colors.append('blue')  # File extensions
        elif token in ['<BUG_FIX>', '<FEATURE_ADD>', '<REFACTOR>', '<OPTIMIZATION>']:
            colors.append('orange')  # Change type tokens
        elif token in ['<PYTHON_LANG>', '<JAVASCRIPT_LANG>', '<REACT_FRAMEWORK>', '<NODE_FRAMEWORK>']:
            colors.append('green')  # Programming terms
        else:
            colors.append('lightgray')  # Regular tokens
    
    # Plot the tokens
    for i, (token, pos, length, color) in enumerate(zip(tokens, token_positions, token_lengths, colors)):
        ax.barh(0, length, left=pos, color=color, alpha=0.7, edgecolor='black')
        ax.text(pos + length/2, 0, token[:10] + '..' if len(token) > 10 else token, 
                ha='center', va='center', fontsize=8, rotation=90)
    
    ax.set_yticks([0])
    ax.set_yticklabels(['Tokens'])
    ax.set_title(f'{title}\nOriginal: "{original_text[:50]}..."')
    ax.set_xlabel('Token Position')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Special Tokens'),
        Patch(facecolor='blue', label='File Extensions'),
        Patch(facecolor='orange', label='Change Types'),
        Patch(facecolor='green', label='Programming Terms'),
        Patch(facecolor='lightgray', label='Regular Tokens')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig


def compare_vocabulary_sizes(vocab1, name1: str, vocab2, name2: str):
    """Compare vocabulary sizes and compositions."""
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Vocabulary Comparison', fontsize=16, fontweight='bold')
    
    # 1. Vocabulary sizes
    sizes = [len(vocab1.stoi), len(vocab2.stoi)]
    names = [name1, name2]
    axes[0].bar(names, sizes, color=['skyblue', 'lightcoral'])
    axes[0].set_title('Vocabulary Size Comparison')
    axes[0].set_ylabel('Number of Tokens')
    
    # 2. Special token comparison
    special1 = len([t for t in vocab1.stoi.keys() if t.startswith('<') and t.endswith('>')])
    special2 = len([t for t in vocab2.stoi.keys() if t.startswith('<') and t.endswith('>')])
    regular1 = len(vocab1.stoi) - special1
    regular2 = len(vocab2.stoi) - special2
    
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width/2, [special1, special2], width, label='Special Tokens', color='orange')
    axes[1].bar(x + width/2, [regular1, regular2], width, label='Regular Tokens', color='lightblue')
    axes[1].set_title('Special vs Regular Tokens')
    axes[1].set_ylabel('Count')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].legend()
    
    # 3. Token length distributions
    lengths1 = [len(token) for token in vocab1.stoi.keys()]
    lengths2 = [len(token) for token in vocab2.stoi.keys()]
    
    axes[2].hist(lengths1, bins=30, alpha=0.5, label=name1, color='skyblue', edgecolor='black')
    axes[2].hist(lengths2, bins=30, alpha=0.5, label=name2, color='lightcoral', edgecolor='black')
    axes[2].set_title('Token Length Distribution')
    axes[2].set_xlabel('Token Length')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    return fig


def visualize_token_frequency_distribution(vocab, title: str = "Token Frequency Distribution"):
    """Visualize the frequency distribution of tokens in the vocabulary."""
    setup_plot_style()
    
    if not hasattr(vocab, 'trained_model') or not vocab.trained_model:
        print("Warning: No frequency model found in vocabulary")
        return None
    
    # Get frequency scores (inverse of model scores)
    freq_scores = [1/np.exp(score) for score in vocab.trained_model.values()]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{title} - Frequency Analysis', fontsize=16, fontweight='bold')
    
    # Histogram of frequency scores
    axes[0].hist(freq_scores, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribution of Token Frequencies')
    axes[0].set_xlabel('Frequency Score')
    axes[0].set_ylabel('Number of Tokens')
    
    # Log-log plot to show power-law distribution
    sorted_freqs = sorted(freq_scores, reverse=True)
    ranks = range(1, len(sorted_freqs) + 1)
    
    axes[1].loglog(ranks, sorted_freqs, marker='o', linestyle='', markersize=3, alpha=0.5)
    axes[1].set_title('Zipf\'s Law Visualization (Log-Log Plot)')
    axes[1].set_xlabel('Rank')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig


def save_visualizations(fig, filename: str, output_dir: str = "./visualization_output"):
    """Save visualization figures to file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {filepath}")
    plt.close(fig)


def generate_comprehensive_report(vocab1, name1: str, vocab2, name2: str, 
                                sample_texts: List[str] = None):
    """Generate a comprehensive visualization report."""
    print("Generating comprehensive visualization report...")
    
    # 1. Vocabulary statistics
    fig1 = visualize_vocabulary_statistics(vocab1, f"{name1} Vocabulary")
    save_visualizations(fig1, f"{name1.lower()}_statistics.png")
    
    fig2 = visualize_vocabulary_statistics(vocab2, f"{name2} Vocabulary")
    save_visualizations(fig2, f"{name2.lower()}_statistics.png")
    
    # 2. Comparison
    fig3 = compare_vocabulary_sizes(vocab1, name1, vocab2, name2)
    save_visualizations(fig3, "vocabulary_comparison.png")
    
    # 3. Frequency distributions if available
    fig4 = visualize_token_frequency_distribution(vocab1, f"{name1} Token Frequencies")
    if fig4:
        save_visualizations(fig4, f"{name1.lower()}_frequencies.png")
    
    fig5 = visualize_token_frequency_distribution(vocab2, f"{name2} Token Frequencies")
    if fig5:
        save_visualizations(fig5, f"{name2.lower()}_frequencies.png")
    
    # 4. Sample tokenizations if texts provided
    if sample_texts:
        for i, text in enumerate(sample_texts[:5]):  # First 5 samples
            try:
                tokens1 = vocab1.tokenize(text)
                fig = visualize_tokenization_process(text, tokens1, f"Tokenization: {name1}")
                save_visualizations(fig, f"{name1.lower()}_tokenization_sample_{i}.png")
                
                tokens2 = vocab2.tokenize(text)
                fig = visualize_tokenization_process(text, tokens2, f"Tokenization: {name2}")
                save_visualizations(fig, f"{name2.lower()}_tokenization_sample_{i}.png")
            except Exception as e:
                print(f"Error visualizing sample {i}: {e}")
    
    print("Comprehensive visualization report completed!")


# Example usage function
def example_usage():
    """Example of how to use the visualization module."""
    # This would typically be called after training vocabularies
    print("Example visualization usage - load vocabularies and call visualization functions")
    pass