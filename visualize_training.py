"""
Visualize training dynamics from CSV log.
Generates loss curves, learning rate schedule, and GPU memory usage plots.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(log_path: str, output: str = "loss_curves.png"):
    """
    Generate three-panel training dynamics visualization.
    
    Panel 1: Training and Validation Loss Curves
    Panel 2: Learning Rate Schedule (Noam)
    Panel 3: GPU Memory Usage
    
    Args:
        log_path: Path to training_log.csv
        output: Output image path
    """
    df = pd.read_csv(log_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curves
    ax = axes[0]
    # Per-step train loss (smoothed)
    smoothed = df['train_loss'].ewm(span=50).mean()
    ax.plot(df.index, df['train_loss'], color='steelblue', alpha=0.3, linewidth=0.5, label='Train loss (raw)')
    ax.plot(df.index, smoothed, color='steelblue', linewidth=2, label='Train loss (smoothed)')
    
    # Per-epoch validation loss
    epoch_df = df.groupby('epoch').last().reset_index()
    epoch_steps = epoch_df.index * len(df) / len(epoch_df)
    ax.scatter(epoch_steps, epoch_df['val_loss'], color='red', s=100, zorder=5, label='Val loss (epoch)')
    ax.plot(epoch_steps, epoch_df['val_loss'], color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate schedule
    ax = axes[1]
    ax.plot(df.index, df['lr'], color='green', linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule (Noam)', fontsize=13)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    # GPU Memory usage
    ax = axes[2]
    ax.fill_between(df.index, df['gpu_mem_mb'], alpha=0.4, color='orange')
    ax.plot(df.index, df['gpu_mem_mb'], color='darkorange', linewidth=1)
    ax.axhline(8192, color='red', linestyle='--', linewidth=2, label='8GB VRAM limit')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('GPU Memory (MB)', fontsize=12)
    ax.set_title('GPU Memory Usage During Training', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Dynamics — Mini-Transformer for Commit Message Generation',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize training logs")
    parser.add_argument("--log-path", default="./checkpoints/training_log.csv", help="Path to training log CSV")
    parser.add_argument("--output", default="./loss_curves.png", help="Output image path")
    args = parser.parse_args()
    
    plot_loss_curves(args.log_path, args.output)
