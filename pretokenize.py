#!/usr/bin/env python3
"""
pretokenize.py — Run this ONCE before training.
Converts raw parquet → compact int16 numpy arrays on disk.
Eliminates the ~20 GB RAM spike from loading raw strings.

Usage:
    python pretokenize.py \
        --data-path ./traindata.parquet \
        --diff-vocab-path ./tokenizer/diff_vocab.pkl \
        --message-vocab-path ./tokenizer/message_vocab.pkl \
        --output-dir ./tokenized_data \
        --max-seq-length 256
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import multiprocessing as mp
from functools import partial


class PickleModuleRedirector(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('Vocabulary', 'DiffVocabulary', 'tokenizer.diff_text.Vocabulary'):
            from tokenizer.DiffVocabulary import DiffVocabulary
            if name in ('DiffVocabulary', 'Vocabulary'):
                return DiffVocabulary
        elif module in ('MsgVocabulary', 'tokenizer.message.Vocabulary'):
            from tokenizer.MsgVocabulary import MsgVocabulary
            if name in ('MsgVocabulary', 'Vocabulary'):
                return MsgVocabulary
        return super().find_class(module, name)


def load_vocab(path):
    with open(path, 'rb') as f:
        return PickleModuleRedirector(f).load()


def tokenize_and_pad(text, vocab, max_len, dtype=np.int32):
    """Tokenize, add SOS/EOS, pad/truncate to max_len. Returns array with specified dtype."""
    ids = vocab.numericalize(str(text) if text else "")
    pad_id  = vocab.stoi["<PAD>"]
    sos_id  = vocab.stoi["<SOS>"]
    eos_id  = vocab.stoi["<EOS>"]

    # Leave room for SOS + EOS
    ids = ids[:max_len - 2]
    ids = [sos_id] + ids + [eos_id]
    # Pad
    ids += [pad_id] * (max_len - len(ids))
    return np.array(ids[:max_len], dtype=dtype)


def init_worker(src_vocab_path, tgt_vocab_path):
    """Initialize worker process by loading vocabularies once."""
    global _src_vocab, _tgt_vocab
    _src_vocab = load_vocab(src_vocab_path)
    _tgt_vocab = load_vocab(tgt_vocab_path)


def tokenize_chunk(args_tuple):
    """Worker function for parallel tokenization of a chunk.
    Uses globally initialized vocabularies to avoid reloading per chunk."""
    df_chunk, max_seq_length, dtype_str, start_idx = args_tuple
    
    # Use pre-loaded vocabularies from initializer
    src_vocab = _src_vocab
    tgt_vocab = _tgt_vocab
    
    # Convert dtype string back to numpy type
    dtype = np.int16 if dtype_str == 'int16' else np.int32
    
    df_chunk['diff_text'] = df_chunk['diff_text'].fillna('')
    df_chunk['message']   = df_chunk['message'].fillna('')
    
    num_rows = len(df_chunk)
    src_results = np.zeros((num_rows, max_seq_length), dtype=dtype)
    tgt_results = np.zeros((num_rows, max_seq_length), dtype=dtype)
    
    for i, (_, row) in enumerate(df_chunk.iterrows()):
        src_results[i] = tokenize_and_pad(row['diff_text'], src_vocab, max_seq_length, dtype)
        tgt_results[i] = tokenize_and_pad(row['message'], tgt_vocab, max_seq_length, dtype)
    
    return src_results, tgt_results, start_idx


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading vocabularies...")
    src_vocab = load_vocab(args.diff_vocab_path)
    tgt_vocab = load_vocab(args.message_vocab_path)
    print(f"  src_vocab: {len(src_vocab.stoi):,} tokens")
    print(f"  tgt_vocab: {len(tgt_vocab.stoi):,} tokens")

    # Check vocab size — switch to int32 if needed
    max_vocab = max(len(src_vocab.stoi), len(tgt_vocab.stoi))
    dtype = np.int16 if max_vocab <= 32767 else np.int32
    dtype_str = 'int16' if dtype == np.int16 else 'int32'
    print(f"  Using dtype: {dtype} (max vocab id: {max_vocab})")

    print(f"Reading parquet from {args.data_path}...")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.data_path)
    total_rows = pf.metadata.num_rows
    print(f"  Total rows: {total_rows:,}")

    # Determine number of workers
    num_workers = mp.cpu_count()
    print(f"  Using {num_workers} CPU cores for parallel tokenization")
    
    # Set chunk size
    chunk_size = 5000  # Process 100k rows per chunk

    # Pre-allocate output memmap arrays on disk
    src_path = os.path.join(args.output_dir, "src_tokens.npy")
    tgt_path = os.path.join(args.output_dir, "tgt_tokens.npy")
    src_mmap = np.lib.format.open_memmap(
        src_path, mode='w+', dtype=dtype,
        shape=(total_rows, args.max_seq_length)
    )
    tgt_mmap = np.lib.format.open_memmap(
        tgt_path, mode='w+', dtype=dtype,
        shape=(total_rows, args.max_seq_length)
    )

    # Use a generator to yield chunks one by one (lazy loading)
    def chunk_generator():
        curr_idx = 0
        for batch in pf.iter_batches(batch_size=chunk_size):
            df_chunk = batch.to_pandas()
            yield (df_chunk, args.max_seq_length, dtype_str, curr_idx)
            curr_idx += len(df_chunk)

    print(f"  Processing in parallel...")
    
    # Calculate expected number of chunks for accurate progress bar
    num_expected_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    # imap now consumes the generator lazily with pre-loaded vocabularies
    with mp.Pool(processes=num_workers, initializer=init_worker, 
                 initargs=(args.diff_vocab_path, args.message_vocab_path)) as pool:
        results = list(tqdm(pool.imap(tokenize_chunk, chunk_generator()), 
                           total=num_expected_chunks,
                           desc="Tokenizing chunks"))
    
    # Write results to memmap
    print("Writing tokenized data to disk...")
    for src_arr, tgt_arr, idx in tqdm(results, desc="Writing to disk"):
        num_rows = src_arr.shape[0]
        src_mmap[idx:idx+num_rows] = src_arr
        tgt_mmap[idx:idx+num_rows] = tgt_arr
    
    # Flush to disk
    src_mmap.flush()
    tgt_mmap.flush()
    
    # Clean up chunk dataframes
    del results
    gc.collect()
    
    # Compute actual src lengths (count non-PAD tokens)
    pad_id_src = src_vocab.stoi["<PAD>"]
    print("Computing sequence lengths...")
    src_lengths = np.sum(src_mmap != pad_id_src, axis=1).astype(np.int16)
    lengths_path = os.path.join(args.output_dir, "src_lengths.npy")
    np.save(lengths_path, src_lengths)
    print(f"Saved src_lengths.npy  mean={src_lengths.mean():.1f}  max={src_lengths.max()}")
    
    del src_mmap, tgt_mmap

    # Save metadata
    import json
    meta = {
        "total_rows": total_rows,
        "max_seq_length": args.max_seq_length,
        "src_vocab_size": len(src_vocab.stoi),
        "tgt_vocab_size": len(tgt_vocab.stoi),
        "dtype": str(dtype),
        "src_path": src_path,
        "tgt_path": tgt_path,
        "lengths_path": lengths_path,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Files written to {args.output_dir}/")
    print(f"  src_tokens.npy: {os.path.getsize(src_path) / 1e9:.2f} GB")
    print(f"  tgt_tokens.npy: {os.path.getsize(tgt_path) / 1e9:.2f} GB")
    print(f"\nNow run train.py with --use-pretokenized --tokenized-dir {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path",          default="./traindata.parquet")
    parser.add_argument("--diff-vocab-path",    default="./tokenizer/diff_vocab.pkl")
    parser.add_argument("--message-vocab-path", default="./tokenizer/message_vocab.pkl")
    parser.add_argument("--output-dir",         default="./tokenized_data")
    parser.add_argument("--max-seq-length",     type=int, default=256)
    main(parser.parse_args())
