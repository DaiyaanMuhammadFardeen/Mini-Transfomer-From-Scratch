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
        from tokenizer.visualize_vocabularies import plot_token_distributions
        
        if os.path.exists(args.parquet):
            print("   Loading dataset to compute token lengths...")
            df = pd.read_parquet(args.parquet)
            
            # Load vocabularies for tokenization
            diff_vocab_path = args.diff_vocab if hasattr(args, 'diff_vocab') else "./tokenizer/diff_vocab.pkl"
            msg_vocab_path = args.msg_vocab if hasattr(args, 'msg_vocab') else "./tokenizer/message_vocab.pkl"
            
            if os.path.exists(diff_vocab_path) and os.path.exists(msg_vocab_path):
                with open(diff_vocab_path, 'rb') as f:
                    diff_vocab = pickle.load(f)
                with open(msg_vocab_path, 'rb') as f:
                    msg_vocab = pickle.load(f)
                
                # Sample subset for speed
                sample_size = min(1000, len(df))
                df_sample = df.head(sample_size)
                
                # Compute tokenized lengths
                diff_lengths = []
                msg_lengths = []
                
                for _, row in df_sample.iterrows():
                    diff_text = str(row.get('diff_text', ''))
                    msg_text = str(row.get('message', ''))
                    
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
            refs = pred_df['message'].fillna('').tolist()
            hyps = pred_df['predicted_message'].fillna('').tolist()
            
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
            
            examples = []
            for i in range(min(8, len(pred_df))):
                row = pred_df.iloc[i]
                examples.append({
                    'diff': str(row.get('diff_text', ''))[:80],
                    'reference': str(row.get('message', '')),
                    'predicted': str(row.get('predicted_message', '')),
                    'bleu': 0.0  # Would need per-example BLEU scores
                })
            
            plot_qualitative_examples(examples, output=fig("07_qualitative_examples.png"))
            print("✅ Qualitative examples table generated")
        else:
            print(f"⚠️  Predictions file not found: {args.predictions}")
    except Exception as e:
        print(f"⚠️  Skipped (error: {e})")
    
    # 8. Attention visualization (placeholder - requires model modifications)
    print("\n[8/8] Attention visualization...")
    print("       Note: Requires model modifications to extract attention weights.")
    print("       See visualize_attention.py for implementation details.")
    print("       Skipping automated generation.")
    
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
