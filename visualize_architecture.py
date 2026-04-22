"""
Generate clean model architecture diagram programmatically.
Creates publication-quality transformer architecture visualization for thesis.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


def plot_model_architecture(d_model: int = 256, num_heads: int = 8,
                              num_layers: int = 4, d_ff: int = 512,
                              src_vocab: int = 32000, tgt_vocab: int = 8000,
                              output: str = "model_architecture.png"):
    """
    Draw a clean transformer architecture diagram with your specific dimensions.
    
    Args:
        d_model: Model dimension (embedding size)
        num_heads: Number of attention heads
        num_layers: Number of encoder/decoder layers
        d_ff: Feed-forward dimension
        src_vocab: Source vocabulary size (diff tokens)
        tgt_vocab: Target vocabulary size (message tokens)
        output: Output file path
    """
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis('off')
    
    def draw_box(ax, x, y, width, height, label, sublabel='', color='#E3F2FD', fontsize=10):
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                              boxstyle="round,pad=0.1", facecolor=color,
                              edgecolor='#1565C0', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y + (0.1 if sublabel else 0), label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold')
        if sublabel:
            ax.text(x, y - 0.2, sublabel, ha='center', va='center',
                    fontsize=fontsize - 2, color='#555', style='italic')
    
    def draw_arrow(ax, x, y1, y2, color='#333'):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    # ── ENCODER STACK (left) ──
    enc_x = 3.0
    draw_box(ax, enc_x, 1.5, 3.5, 0.8, 'Diff Input Tokens',
             f'src_vocab = {src_vocab:,}', color='#FFF3E0')
    draw_arrow(ax, enc_x, 1.9, 2.8)
    draw_box(ax, enc_x, 3.2, 3.5, 0.8,
             f'Token Embedding + RoPE',
             f'd_model = {d_model}', color='#E8F5E9')
    draw_arrow(ax, enc_x, 3.6, 4.5)
    
    for i in range(num_layers):
        y_base = 5.0 + i * 3.2
        draw_box(ax, enc_x, y_base, 3.5, 0.7, f'Flash Attention (Self)',
                 f'{num_heads} heads × {d_model // num_heads}d', color='#E3F2FD')
        draw_box(ax, enc_x, y_base + 1.0, 3.5, 0.7, 'Add & LayerNorm', color='#F3E5F5')
        draw_box(ax, enc_x, y_base + 2.0, 3.5, 0.7, 'Feed-Forward',
                 f'{d_model} → {d_ff} → {d_model}', color='#E8F5E9')
        if i < num_layers - 1:
            draw_arrow(ax, enc_x, y_base + 2.35, y_base + 3.0)
    
    enc_top = 5.0 + num_layers * 3.2
    ax.text(enc_x, enc_top, 'Encoder Output\n(Memory)', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1565C0')
    
    # ── DECODER STACK (right) ──
    dec_x = 7.5
    draw_box(ax, dec_x, 1.5, 3.5, 0.8, 'Message Tokens (shifted right)',
             f'tgt_vocab = {tgt_vocab:,}', color='#FFF3E0')
    draw_arrow(ax, dec_x, 1.9, 2.8)
    draw_box(ax, dec_x, 3.2, 3.5, 0.8,
             f'Token Embedding + RoPE',
             f'd_model = {d_model}', color='#E8F5E9')
    draw_arrow(ax, dec_x, 3.6, 4.5)
    
    for i in range(num_layers):
        y_base = 5.0 + i * 3.2
        draw_box(ax, dec_x, y_base, 3.5, 0.7, 'Masked Self-Attention',
                 f'{num_heads} heads (causal)', color='#E3F2FD')
        draw_box(ax, dec_x, y_base + 1.0, 3.5, 0.7, 'Cross-Attention',
                 '← Encoder memory', color='#FCE4EC')
        draw_box(ax, dec_x, y_base + 2.0, 3.5, 0.7, 'Feed-Forward',
                 f'{d_model} → {d_ff} → {d_model}', color='#E8F5E9')
        # Cross-attention arrow from encoder
        ax.annotate('', xy=(dec_x - 1.75, y_base + 1.0), xytext=(enc_x + 1.75, enc_top - 0.5),
                    arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5,
                                   connectionstyle=f'arc3,rad=0.2'))
        if i < num_layers - 1:
            draw_arrow(ax, dec_x, y_base + 2.35, y_base + 3.0)
    
    dec_top = enc_top
    draw_arrow(ax, dec_x, dec_top - 0.3, dec_top + 0.8)
    draw_box(ax, dec_x, dec_top + 1.2, 3.5, 0.8,
             'Linear + Softmax',
             f'→ {tgt_vocab:,} output probs', color='#FFF3E0')
    
    # Titles and labels
    ax.text(enc_x, 0.5, f'ENCODER\n({num_layers} layers)', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#1565C0',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1565C0'))
    ax.text(dec_x, 0.5, f'DECODER\n({num_layers} layers)', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#B71C1C',
            bbox=dict(boxstyle='round', facecolor='#FCE4EC', edgecolor='#B71C1C'))
    
    # Model stats
    total_params = sum([
        src_vocab * d_model,  # encoder embedding
        tgt_vocab * d_model,  # decoder embedding
        num_layers * (4 * d_model**2 + 2 * d_model * d_ff),  # encoder layers
        num_layers * (6 * d_model**2 + 2 * d_model * d_ff),  # decoder layers (3 attn + FFN)
        tgt_vocab * d_model   # output projection
    ])
    
    ax.set_title(
        f'Mini-Transformer Architecture for Commit Message Generation\n'
        f'd_model={d_model}, {num_heads} heads, {num_layers} enc/dec layers, '
        f'd_ff={d_ff} | ~{total_params/1e6:.1f}M parameters',
        fontsize=14, fontweight='bold', pad=15
    )
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate model architecture diagram")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of encoder/decoder layers")
    parser.add_argument("--d-ff", type=int, default=512, help="Feed-forward dimension")
    parser.add_argument("--src-vocab", type=int, default=32000, help="Source vocabulary size")
    parser.add_argument("--tgt-vocab", type=int, default=8000, help="Target vocabulary size")
    parser.add_argument("--output", default="./model_architecture.png", help="Output file path")
    args = parser.parse_args()
    
    plot_model_architecture(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        src_vocab=args.src_vocab,
        tgt_vocab=args.tgt_vocab,
        output=args.output
    )
