#!/usr/bin/env python3
"""
generate_all_figures.py
Run this after training and evaluation to produce all thesis figures.

Usage:
    python generate_all_figures.py \
        --parquet ./traindata.parquet \
        --predictions ./inference_results.csv \
        --metrics ./inference_results_metrics_summary.csv \
        --log ./checkpoints/training_log.csv \
        --checkpoint ./checkpoints/transformer_best.pth \
        --output_dir ./thesis_figures/
"""

import argparse
import os
import pandas as pd
import pickle


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    fig = lambda name: os.path.join(args.output_dir, name)
    
    print("=" * 60)
    print("GENERATING ALL THESIS FIGURES")
    print("=" * 60)
    
    # 1. Architecture diagram
    print("\n[1/8] Model architecture diagram...")
    try:
        from visualize_architecture import plot_model_architecture
        plot_model_architecture(
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=512,
            src_vocab=32000,
            tgt_vocab=8000,
            output=fig("01_architecture.png")
        )
        print("✅ Architecture diagram generated")
    except Exception as e:
        print(f"⚠️  Skipped (error: {e})")
    
    # 2. Vocabulary visualizations
    print("\n[2/8] Vocabulary analysis...")
    try:
        from tokenizer.visualize_vocabularies import (
            load_vocabularies, plot_vocab_coverage, plot_zipf
        )
        
        diff_vocab_path = args.diff_vocab if hasattr(args, 'diff_vocab') else "./tokenizer/diff_vocab.pkl"
        msg_vocab_path = args.msg_vocab if hasattr(args, 'msg_vocab') else "./tokenizer/message_vocab.pkl"
        
        if os.path.exists(diff_vocab_path) and os.path.exists(msg_vocab_path):
            diff_vocab, msg_vocab = load_vocabularies(diff_vocab_path, msg_vocab_path)
            
            if hasattr(diff_vocab, 'word_freqs') and diff_vocab.word_freqs:
                plot_vocab_coverage(
                    diff_vocab.word_freqs,
                    max_vocab=50000,
                    output=fig("02a_vocab_coverage.png")
                )
                plot_zipf(
                    diff_vocab.word_freqs,
                    top_n=100,
                    output=fig("02b_zipf_distribution.png")
                )
                print("✅ Vocabulary plots generated")
            else:
                print("⚠️  word_freqs not available, skipping vocab plots")
        else:
            print("⚠️  Vocabulary files not found, skipping")
    except ImportError as e:
        print(f"⚠️  Skipped (import error: {e})")
        print("   This is expected if visualization dependencies are missing.")
    except Exception as e:
        print(f"⚠️  Skipped (error: {e})")
    
    # 3. Training curves
    print("\n[3/8] Training curves...")
    try:
        from visualize_training import plot_loss_curves
        if os.path.exists(args.log):
            plot_loss_curves(args.log, output=fig("03_training_curves.png"))
            print("✅ Training curves generated")
        else:
            print(f"⚠️  Log file not found: {args.log}")
    except Exception as e:
        print(f"⚠️  Skipped (error: {e})")
    
    # 4. Token length distributions (requires dataset)
    print("\n[4/8] Token length distributions...")
    try:
        from tokenizer.visualize_vocabularies import plot_token_distributions, load_pickle_with_redirect
        
        if os.path.exists(args.parquet):
            print("   Loading dataset to compute token lengths...")
            df = pd.read_parquet(args.parquet)
            
            # Load vocabularies for tokenization
            diff_vocab_path = args.diff_vocab if hasattr(args, 'diff_vocab') else "./tokenizer/diff_vocab.pkl"
            msg_vocab_path = args.msg_vocab if hasattr(args, 'msg_vocab') else "./tokenizer/message_vocab.pkl"
            
            if os.path.exists(diff_vocab_path) and os.path.exists(msg_vocab_path):
                # Use module redirector to load vocabularies
                diff_vocab = load_pickle_with_redirect(diff_vocab_path)
                msg_vocab = load_pickle_with_redirect(msg_vocab_path)
                
                # Sample subset for speed
                sample_size = min(1000, len(df))
                df_sample = df.head(sample_size)
                
                # Compute tokenized lengths - handle both column name variants
                diff_lengths = []
                msg_lengths = []
                
                for _, row in df_sample.iterrows():
                    diff_text = str(row.get('diff_text', ''))
                    # Try multiple possible column names for messages
                    msg_text = str(row.get('message', row.get('commit_message', row.get('reference_message', ''))))
                    
                    # Simple whitespace tokenization as approximation
                    diff_tokens = diff_text.split()
                    msg_tokens = msg_text.split()
                    
                    diff_lengths.append(len(diff_tokens))
                    msg_lengths.append(len(msg_tokens))
                
                plot_token_distributions(
                    diff_lengths,
                    msg_lengths,
                    output=fig("04_token_distributions.png")
                )
                print("✅ Token length distributions generated")
            else:
                print("⚠️  Vocabulary files not found, skipping")
        else:
            print(f"⚠️  Parquet file not found: {args.parquet}")
    except ImportError as e:
        print(f"⚠️  Skipped (import error: {e})")
        print("   This is expected if visualization dependencies are missing.")
    except Exception as e:
        print(f"⚠️  Skipped (error: {e})")
    
    # 5. Evaluation metrics comparison
    print("\n[5/8] Evaluation metrics comparison...")
    try:
        from visualize_evaluation import plot_metric_comparison
        
        if os.path.exists(args.metrics):
            metrics_df = pd.read_csv(args.metrics)
            metrics = metrics_df.iloc[0].to_dict()
            
            plot_metric_comparison(metrics, output=fig("05_metric_comparison.png"))
            print("✅ Metric comparison chart generated")
        else:
            print(f"⚠️  Metrics file not found: {args.metrics}")
    except Exception as e:
        print(f"⚠️  Skipped (error: {e})")
    
    # 6. Length analysis
    print("\n[6/8] Length analysis...")
    try:
        from visualize_evaluation import plot_length_analysis
        
        if os.path.exists(args.predictions):
            pred_df = pd.read_csv(args.predictions)
            # Handle different column naming conventions
            refs = pred_df['reference_message'].fillna('').tolist() if 'reference_message' in pred_df.columns else pred_df['message'].fillna('').tolist()
            hyps = pred_df['generated_message'].fillna('').tolist() if 'generated_message' in pred_df.columns else pred_df['predicted_message'].fillna('').tolist()
            
            plot_length_analysis(refs, hyps, output=fig("06_length_analysis.png"))
            print("✅ Length analysis generated")
        else:
            print(f"⚠️  Predictions file not found: {args.predictions}")
    except Exception as e:
        print(f"⚠️  Skipped (error: {e})")
    
    # 7. Qualitative examples
    print("\n[7/8] Qualitative examples...")
    try:
        from visualize_evaluation import plot_qualitative_examples
        
        if os.path.exists(args.predictions):
            pred_df = pd.read_csv(args.predictions)
            
            # Find the sample with highest BLEU score for each algorithm
            # Group by algorithm and find best example per algorithm
            examples = []
            
            # Check which column has the algorithm info
            if 'algorithm' in pred_df.columns:
                # Get unique algorithms
                algorithms = pred_df['algorithm'].unique()
                
                for algo in algorithms[:8]:  # Limit to first 8 algorithms
                    # Filter for this algorithm and get highest BLEU
                    algo_df = pred_df[pred_df['algorithm'] == algo]
                    
                    # Try different BLEU column names
                    bleu_col = None
                    for col in ['bleu_4', 'bleu', 'BLEU', 'corpus_bleu_4']:
                        if col in algo_df.columns:
                            bleu_col = col
                            break
                    
                    if bleu_col:
                        # Get row with highest BLEU for this algorithm
                        best_idx = algo_df[bleu_col].idxmax()
                        row = pred_df.loc[best_idx]
                    else:
                        # If no BLEU column, just take first example
                        row = algo_df.iloc[0]
                    
                    # Handle different column naming conventions
                    ref_msg = str(row.get('reference_message', row.get('message', '')))
                    gen_msg = str(row.get('generated_message', row.get('predicted_message', '')))
                    bleu_score = float(row.get(bleu_col, 0.0)) if bleu_col else 0.0
                    
                    examples.append({
                        'diff': str(algo),  # Use algorithm name instead of diff summary
                        'reference': ref_msg,
                        'predicted': gen_msg,
                        'bleu': bleu_score
                    })
            else:
                # Fallback: just take top 8 rows by BLEU score
                bleu_col = None
                for col in ['bleu_4', 'bleu', 'BLEU', 'corpus_bleu_4']:
                    if col in pred_df.columns:
                        bleu_col = col
                        break
                
                if bleu_col:
                    sorted_df = pred_df.nlargest(8, bleu_col)
                else:
                    sorted_df = pred_df.head(8)
                
                for i, row in sorted_df.iterrows():
                    ref_msg = str(row.get('reference_message', row.get('message', '')))
                    gen_msg = str(row.get('generated_message', row.get('predicted_message', '')))
                    bleu_score = float(row.get(bleu_col, 0.0)) if bleu_col else 0.0
                    
                    examples.append({
                        'diff': f'Sample {i+1}',
                        'reference': ref_msg,
                        'predicted': gen_msg,
                        'bleu': bleu_score
                    })
            
            plot_qualitative_examples(examples, output=fig("07_qualitative_examples.png"))
            print("✅ Qualitative examples table generated")
        else:
            print(f"⚠️  Predictions file not found: {args.predictions}")
    except Exception as e:
        print(f"⚠️  Skipped (error: {e})")
    
    # 8. Attention visualization (placeholder - requires model modifications)
    print("\n[8/8] Attention visualization...")
    print("       ⚠️  SKIPPED: Requires model architecture modifications")
    print()
    print("       To enable attention visualization, you need to:")
    print("       1. Modify MultiHeadAttention.scaled_dot_product_attention() to return attn_probs")
    print("       2. Register forward hooks on attention layers during inference")
    print("       3. Extract and save attention weights for specific examples")
    print()
    print("       See visualize_attention.py for detailed implementation notes.")
    print("       This is intentionally not automated as it requires code changes.")
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {args.output_dir}")
    print("=" * 60)
    
    # List generated files
    print("\nGenerated figures:")
    for filename in sorted(os.listdir(args.output_dir)):
        filepath = os.path.join(args.output_dir, filename)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  ✓ {filename} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all thesis figures from trained model and evaluation results"
    )
    parser.add_argument("--parquet", default="./traindata.parquet",
                       help="Path to parquet dataset file")
    parser.add_argument("--predictions", default="./inference_results.csv",
                       help="Path to predictions CSV from evaluate.py")
    parser.add_argument("--metrics", default="./inference_results_metrics_summary.csv",
                       help="Path to metrics summary CSV")
    parser.add_argument("--log", default="./checkpoints/training_log.csv",
                       help="Path to training log CSV")
    parser.add_argument("--checkpoint", default="./checkpoints/transformer_best.pth",
                       help="Path to best checkpoint")
    parser.add_argument("--diff-vocab", default="./tokenizer/diff_vocab.pkl",
                       help="Path to diff vocabulary pickle")
    parser.add_argument("--msg-vocab", default="./tokenizer/message_vocab.pkl",
                       help="Path to message vocabulary pickle")
    parser.add_argument("--output_dir", default="./thesis_figures/",
                       help="Output directory for all figures")
    
    args = parser.parse_args()
    main(args)
