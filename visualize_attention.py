"""
Visualize attention weights from the trained model.
Generates cross-attention and self-attention heatmaps for thesis presentation.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from model.model import Transformer


def extract_attention_weights(model, src_tokens: list, tgt_tokens: list, 
                               device, layer: int = 0, head: int = 0) -> np.ndarray:
    """
    Extract cross-attention weights from a specific decoder layer and head.
    
    Args:
        model: Trained Transformer model
        src_tokens: Source token IDs (list of ints)
        tgt_tokens: Target token IDs (list of ints)
        device: torch device
        layer: Decoder layer index
        head: Attention head index
    
    Returns:
        (tgt_len, src_len) attention matrix
    """
    model.eval()
    attention_weights = []
    
    def hook_fn(module, input, output):
        # For custom MultiHeadAttention, we need to modify it to return attention weights
        # This is a simplified version - you may need to adapt based on your architecture
        pass
    
    # Note: Your current MultiHeadAttention doesn't return attention weights
    # You'd need to modify scaled_dot_product_attention to return attn_probs
    # For now, this is a placeholder showing the intended API
    
    with torch.no_grad():
        src = torch.tensor([src_tokens], dtype=torch.long, device=device)
        tgt = torch.tensor([tgt_tokens], dtype=torch.long, device=device)
        
        # Forward pass
        _ = model(src, tgt)
    
    # Placeholder - returns zeros until attention weights are properly extracted
    return np.zeros((len(tgt_tokens), len(src_tokens)))


def plot_cross_attention(attn_weights: np.ndarray,
                          src_tokens: list,
                          tgt_tokens: list,
                          title: str = "Cross-Attention Heatmap",
                          output: str = "cross_attention.png"):
    """
    Visualize encoder-decoder cross-attention as a heatmap.
    Rows = generated tokens (decoder), Columns = input tokens (encoder).
    
    Args:
        attn_weights: (tgt_len, src_len) attention matrix
        src_tokens: List of source token strings
        tgt_tokens: List of target token strings
        title: Plot title
        output: Output file path
    """
    fig, ax = plt.subplots(figsize=(max(12, len(src_tokens) * 0.4),
                                    max(6, len(tgt_tokens) * 0.5)))
    
    sns.heatmap(attn_weights,
                xticklabels=src_tokens,
                yticklabels=tgt_tokens,
                cmap='Blues',
                annot=True if (attn_weights.shape[0] * attn_weights.shape[1] < 200) else False,
                fmt='.2f',
                linewidths=0.5,
                ax=ax)
    
    ax.set_xlabel('Encoder Input (Diff Tokens)', fontsize=12)
    ax.set_ylabel('Decoder Output (Message Tokens)', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_self_attention(attn_weights: np.ndarray,
                         tokens: list,
                         title: str = "Self-Attention Heatmap",
                         output: str = "self_attention.png"):
    """
    Encoder self-attention — shows which diff tokens attend to each other.
    
    Args:
        attn_weights: (seq_len, seq_len) attention matrix
        tokens: List of token strings
        title: Plot title
        output: Output file path
    """
    plot_cross_attention(attn_weights, tokens, tokens, title=title, output=output)


if __name__ == "__main__":
    print("Attention visualization module loaded.")
    print("To use: import visualize_attention and call plot_cross_attention()")
    print("Note: Requires model modifications to extract attention weights during forward pass.")
