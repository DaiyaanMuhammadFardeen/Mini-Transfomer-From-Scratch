"""
Visualize evaluation results with comparison charts and qualitative examples.
Generates publication-quality figures for thesis presentation.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_metric_comparison(your_results: dict, output: str = "metric_comparison.png"):
    """
    Bar chart comparing your model against published baselines.
    
    Args:
        your_results: Dictionary with metric scores from evaluate.py
        output: Output file path
    """
    metrics = ['BLEU-4', 'METEOR', 'ROUGE-L']
    
    # Your model's scores - use actual values if available, otherwise defaults
    your_scores = [
        your_results.get('bleu_4', your_results.get('corpus_bleu_4', 0)) * 100 if your_results.get('bleu_4', your_results.get('corpus_bleu_4', 0)) < 1 else your_results.get('bleu_4', your_results.get('corpus_bleu_4', 0)),
        your_results.get('meteor', 0),
        your_results.get('rouge_l_f', your_results.get('mean_rouge_l_f', 0))
    ]
    
    # Published baseline scores from literature (verified from papers)
    # Sources: RACE (Shi et al. 2022), NNGen (Liu et al. 2018), CodeT5 (Wang et al. 2021)
    # Note: Scores are approximate averages across multiple datasets/languages
    baselines = {
        'NNGen (2018)':       [16.4, 10.1, 13.2],   # Retrieval-based, BLEU from cleaned dataset
        'CodeT5 (2021)':      [19.8, 21.7, 25.9],   # Pre-trained encoder-decoder
        'RACE (2022)':        [25.7, 18.2, 24.7],   # Retrieval-augmented SOTA
        'Our seq2seq Transformer': your_scores
    }
    
    x = np.arange(len(metrics))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    for i, (model_name, scores) in enumerate(baselines.items()):
        bars = ax.bar(x + i * width, scores, width, label=model_name,
                      color=colors[i], alpha=0.85, edgecolor='white')
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Evaluation Metric', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Commit Message Generation — Metric Comparison with Published Baselines',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(baselines) - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(s) for s in baselines.values()) * 1.15)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_length_analysis(references: list, hypotheses: list,
                          output: str = "length_analysis.png"):
    """
    Scatter plot: reference length vs hypothesis length.
    Shows if the model over-generates or under-generates.
    
    Args:
        references: List of reference commit messages
        hypotheses: List of generated commit messages
        output: Output file path
    """
    ref_lens = [len(r.split()) for r in references]
    hyp_lens = [len(h.split()) for h in hypotheses]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter
    ax1.scatter(ref_lens, hyp_lens, alpha=0.3, s=10, color='steelblue')
    max_len = max(max(ref_lens), max(hyp_lens))
    ax1.plot([0, max_len], [0, max_len], 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('Reference Length (words)', fontsize=12)
    ax1.set_ylabel('Predicted Length (words)', fontsize=12)
    ax1.set_title('Predicted vs. Reference Message Length', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Length error distribution
    length_errors = [h - r for r, h in zip(ref_lens, hyp_lens)]
    ax2.hist(length_errors, bins=30, color='coral', edgecolor='white', alpha=0.8)
    ax2.axvline(0, color='black', linewidth=2, linestyle='--', label='Perfect length')
    ax2.axvline(np.mean(length_errors), color='red', linewidth=2,
                label=f'Mean error: {np.mean(length_errors):+.1f} words')
    ax2.set_xlabel('Length Error (predicted - reference)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Commit Message Length Error Distribution', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Length Analysis of Generated Commit Messages', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_qualitative_examples(examples: list, output: str = "qualitative_examples.png"):
    """
    Table-style figure showing side-by-side reference and predicted messages.
    Perfect for thesis slides.
    
    Args:
        examples: List of dicts with keys: 'diff' (algorithm name), 'reference', 'predicted', 'bleu'
        output: Output file path
    """
    fig, ax = plt.subplots(figsize=(16, len(examples) * 1.5 + 2))
    ax.axis('off')
    
    col_labels = ['#', 'Algorithm', 'Reference Message', 'Predicted Message', 'BLEU']
    table_data = []
    for i, ex in enumerate(examples):
        # Use algorithm name directly (already formatted)
        algo_name = ex['diff'][:40] if len(ex['diff']) > 40 else ex['diff']
        ref = ex['reference'][:60] + '...' if len(ex['reference']) > 60 else ex['reference']
        pred = ex['predicted'][:60] + '...' if len(ex['predicted']) > 60 else ex['predicted']
        table_data.append([str(i+1), algo_name, ref, pred, f"{ex.get('bleu', 0):.3f}"])
    
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='left',
        loc='center',
        colWidths=[0.04, 0.18, 0.35, 0.35, 0.08]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2196F3')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    ax.set_title('Qualitative Examples — Generated vs. Reference Commit Messages',
                 fontsize=14, fontweight='bold', pad=20, y=0.95)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--metrics-path", default="./inference_results_metrics_summary.csv", 
                       help="Path to metrics CSV")
    parser.add_argument("--predictions-path", default="./inference_results.csv",
                       help="Path to predictions CSV")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    args = parser.parse_args()
    
    # Load metrics
    import pandas as pd
    metrics_df = pd.read_csv(args.metrics_path)
    metrics = metrics_df.iloc[0].to_dict()
    
    # Generate metric comparison
    plot_metric_comparison(metrics, output=f"{args.output_dir}/metric_comparison.png")
    
    # Load predictions for length analysis
    pred_df = pd.read_csv(args.predictions_path)
    references = pred_df['message'].tolist()
    hypotheses = pred_df['predicted_message'].tolist()
    
    plot_length_analysis(references, hypotheses, 
                        output=f"{args.output_dir}/length_analysis.png")
    
    print("✅ Evaluation visualizations generated!")
