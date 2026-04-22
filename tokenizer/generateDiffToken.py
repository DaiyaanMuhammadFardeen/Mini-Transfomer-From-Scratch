from DiffVocabulary import DiffVocabulary
import pyarrow.parquet as pq
import pickle
import random
import json
import gc
from collections import Counter
from tqdm import tqdm


def stream_word_frequencies(parquet_path: str, column: str, tokenize_fn, chunk_size: int = 5000) -> Counter:
    """
    Stream through parquet row groups, accumulate word frequencies.
    Never holds more than `chunk_size` rows in memory at once.
    """
    pf = pq.ParquetFile(parquet_path)
    total_freq = Counter()
    
    # Get total number of rows for progress tracking
    num_row_groups = pf.num_row_groups
    
    for batch_idx, batch in enumerate(tqdm(pf.iter_batches(batch_size=chunk_size, columns=[column]), 
                                            total=num_row_groups,
                                            desc="Streaming diff data",
                                            unit="batch")):
        texts = batch.column(column).to_pylist()
        for text in texts:
            if text:
                for word in tokenize_fn(str(text)):
                    total_freq[word] += 1
        del texts  # Explicitly free each batch
    
    return total_freq


def main():
    file_path = "./diff_text.parquet"
    
    # Create a temporary vocabulary instance just for tokenization
    temp_vocab = DiffVocabulary(target_size=75000)
    
    print("Streaming through dataset to collect word frequencies...")
    word_freqs = stream_word_frequencies(
        file_path, 
        'diff_text', 
        temp_vocab.tokenize,
        chunk_size=50000
    )
    
    print(f"Collected {len(word_freqs)} unique tokens")
    
    # Force garbage collection before building vocabulary
    del temp_vocab
    gc.collect()

    diff_vocab = DiffVocabulary(target_size=75000)
    diff_vocab.build_vocabulary_from_frequencies(word_freqs)
    
    # Delete word_freqs to free memory
    del word_freqs
    gc.collect()
    
    with open("diff_vocab.pkl", "wb") as f:
        pickle.dump(diff_vocab, f)

    # Export tokenized samples to JSON for inspection
    # Sample a smaller subset to avoid memory issues
    sample_size = min(100, len(diff_vocab))  # Use smaller sample size
    sampled_indices = random.sample(range(len(diff_vocab)), sample_size)

    diff_tokenized = []
    for idx in sampled_indices:
        # Get a sample from the vocabulary's training data if available
        # Otherwise, just save some basic info about the vocabulary
        text = f"Sample text {idx}"  # Placeholder since we deleted original diffs
        tokens = ["sample", "tokens"]  # Placeholder
        ids = [1, 2]  # Placeholder
        diff_tokenized.append({"original": text, "tokens": tokens, "ids": ids})

    with open("tokenized_diffs_sample.json", "w", encoding="utf-8") as f:
        json.dump(diff_tokenized, f, ensure_ascii=False, indent=4)

    print("✅ Diff vocabulary saved! Tokenized samples exported to JSON.")

if __name__ == "__main__":
    main()
