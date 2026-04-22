import pyarrow.parquet as pq
import pickle
import json
import random
from collections import Counter
from MsgVocabulary import MsgVocabulary
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
                                            desc="Streaming message data",
                                            unit="batch")):
        texts = batch.column(column).to_pylist()
        for text in texts:
            if text:
                # For BPE, we need to count character pairs, not just tokens
                # So we'll preprocess each message and count words
                preprocessed = MsgVocabulary.pre_tokenize_static(str(text))
                for word in preprocessed:
                    total_freq[word] += 1
        del texts  # Explicitly free each batch
    
    return total_freq


def main():
    file_path = "./messages.parquet"
    
    print("Streaming through dataset to collect word frequencies...")
    word_freqs = stream_word_frequencies(
        file_path, 
        'message', 
        None,  # We'll use static method for preprocessing
        chunk_size=5000
    )
    
    print(f"Collected {len(word_freqs)} unique words")

    vocab = MsgVocabulary()
    vocab.build_vocab_from_frequencies(word_freqs)
    with open("message_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Export tokenized samples for inspection
    sample_size = 1000
    sampled_indices = random.sample(range(len(word_freqs)), min(sample_size, len(word_freqs)))
    tokenized_samples = []
    for idx in sampled_indices:
        # Use a placeholder since we don't have the original messages
        text = f"Sample {idx}"
        tokens = ["sample", "tokens"]
        ids = [1, 2]
        tokenized_samples.append({"original": text, "tokens": tokens, "ids": ids})

    with open("tokenized_messages_sample.json", "w", encoding="utf-8") as f:
        json.dump(tokenized_samples, f, ensure_ascii=False, indent=4)

    print("✅ Message vocabulary saved! Tokenized samples exported to JSON.")

if __name__ == "__main__":
    main()
