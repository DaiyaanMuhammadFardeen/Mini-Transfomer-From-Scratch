#!/usr/bin/env python3
"""
diagnostic.py

Diagnose why inference is failing - check vocabularies, tokenization, and model output.
"""

import torch
import torch.nn as nn
import pickle
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model.model import Transformer

# Custom unpickler
class PickleModuleRedirector(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'Vocabulary':
            try:
                from tokenizer.DiffVocabulary import DiffVocabulary
                if name == 'DiffVocabulary':
                    return DiffVocabulary
            except:
                pass
            try:
                from tokenizer.MsgVocabulary import MsgVocabulary
                if name == 'MsgVocabulary':
                    return MsgVocabulary
            except:
                pass
        return super().find_class(module, name)

def load_pickle_with_redirect(file_path):
    with open(file_path, 'rb') as f:
        unpickler = PickleModuleRedirector(f)
        try:
            obj = unpickler.load()
            return obj
        except Exception as e:
            f.seek(0)
            obj = pickle.load(f)
            return obj

def diagnose(parquet_path, checkpoint_path, diff_vocab_path, message_vocab_path):
    """Run diagnostics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}\n")

    # Load vocabularies
    print("=" * 80)
    print("1. VOCABULARY CHECK")
    print("=" * 80)
    src_vocab = load_pickle_with_redirect(diff_vocab_path)

    print(f"Source vocab size: {len(src_vocab.stoi)}")
    print(f"Source vocab type: {type(src_vocab)}")
    print(f"Source vocab has tokenize: {hasattr(src_vocab, 'tokenize')}")
    print(f"Source vocab has numericalize: {hasattr(src_vocab, 'numericalize')}")
    print(f"\nSource vocab special tokens: {[src_vocab.itos.get(i) for i in range(min(5, len(src_vocab.itos)))]}")

    # Load parquet and check diff_text
    print("\n" + "=" * 80)
    print("2. DATA CHECK")
    print("=" * 80)
    df = pd.read_parquet(parquet_path)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    if 'diff_text' in df.columns:
        non_null = df['diff_text'].notna().sum()
        print(f"Non-null diff_text: {non_null}")
        empty = (df['diff_text'].astype(str).str.strip() == '').sum()
        print(f"Empty diff_text: {empty}")

        # Sample a non-empty diff_text
        sample_diffs = df[df['diff_text'].notna() & (df['diff_text'].astype(str).str.len() > 10)].head(3)
        if len(sample_diffs) > 0:
            print(f"\nSample diff_text (first 200 chars):")
            for idx, row in sample_diffs.iterrows():
                print(f"  Row {idx}: {str(row['diff_text'])[:200]}...")

    # Test tokenization
    print("\n" + "=" * 80)
    print("3. TOKENIZATION TEST")
    print("=" * 80)

    test_diff = "<ADD> def hello world </ADD>"
    print(f"Test diff_text: {test_diff}")

    try:
        tokens = src_vocab.tokenize(test_diff)
        print(f"Tokens: {tokens[:20]}...")
        print(f"Num tokens: {len(tokens)}")

        ids = src_vocab.numericalize(test_diff)
        print(f"IDs: {ids[:20]}...")
        print(f"Num IDs: {len(ids)}")

        # Check if any IDs are in vocab
        in_vocab = sum(1 for id in ids if id < len(src_vocab.itos))
        print(f"IDs in vocab: {in_vocab}/{len(ids)}")
    except Exception as e:
        print(f"[!] Tokenization error: {e}")
        import traceback
        traceback.print_exc()

    # Load model
    print("\n" + "=" * 80)
    print("4. MODEL CHECK")
    print("=" * 80)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Checkpoint keys: {checkpoint.keys()}")

        model = Transformer(
            src_vocab_size=len(src_vocab.stoi),
            tgt_vocab_size=508,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2024,
            max_seq_length=1024,
            dropout=0.1
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"[!] Model loading error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test forward pass
    print("\n" + "=" * 80)
    print("5. FORWARD PASS TEST")
    print("=" * 80)

    try:
        # Create dummy input
        src_ids = torch.randint(0, len(src_vocab.stoi), (1, 1024)).to(device)
        tgt_ids = torch.randint(0, len(tgt_vocab.stoi), (1, 1024)).to(device)

        print(f"Input shapes: src={src_ids.shape}, tgt={tgt_ids.shape}")

        with torch.no_grad():
            output = model(src_ids, tgt_ids)

        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output min: {output.min().item():.4f}, max: {output.max().item():.4f}")
        print(f"Output contains NaN: {torch.isnan(output).any().item()}")
        print(f"Output contains Inf: {torch.isinf(output).any().item()}")

        # Check logits for first position
        logits = output[0, 0, :]
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, 5)
        print(f"\nTop 5 tokens at position 0:")
        for prob, idx in zip(top_k_probs, top_k_indices):
            token = tgt_vocab.itos.get(idx.item(), "<UNK>")
            print(f"  {token}: {prob.item():.4f}")

    except Exception as e:
        print(f"[!] Forward pass error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose inference issues")
    parser.add_argument("--parquet", required=True, help="Input parquet file")
    parser.add_argument("--checkpoint", default="./checkpoints/transformer_epoch_1.pth")
    parser.add_argument("--diff-vocab", default="./tokenizer/diff_text/diff_vocab.pkl")
    parser.add_argument("--message-vocab", default="./tokenizer/message/message_vocab.pkl")

    args = parser.parse_args()

    diagnose(args.parquet, args.checkpoint, args.diff_vocab, args.message_vocab)
