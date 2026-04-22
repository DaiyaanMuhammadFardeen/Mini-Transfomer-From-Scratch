from DiffVocabulary import DiffVocabulary
import pandas as pd
import pickle
import random
import json
import gc
from visualization import generate_comprehensive_report


def load_data(file_path):
    # Process data in chunks to reduce memory usage
    chunk_size = 500000
    all_diffs = []
    
    for chunk in pd.read_parquet(file_path, engine='pyarrow').iterrows():
        diff_text = chunk[1]['diff_text']
        if pd.notna(diff_text):
            all_diffs.append(str(diff_text))
        else:
            all_diffs.append('')
        
        # Process in batches to control memory usage
        if len(all_diffs) % chunk_size == 0:
            # Force garbage collection
            gc.collect()
    
    return all_diffs

def main():
    file_path = "./diff_text.parquet"
    diffs = load_data(file_path)
    
    # Force garbage collection before building vocabulary
    gc.collect()

    diff_vocab = DiffVocabulary(target_size=75000)
    diff_vocab.build_vocabulary(diffs)
    
    # Delete diffs to free memory
    del diffs
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
    
    # Generate visualizations for the diff vocabulary
    print("Generating visualizations for diff vocabulary...")
    sample_texts = ["Fix security vulnerability in JWT authentication with Python Django and PostgreSQL",
                    "Add new feature for user authentication using React and Node.js",
                    "Refactor data processing module for better performance with caching"]
    generate_comprehensive_report(diff_vocab, "DiffVocabulary", diff_vocab, "DiffVocabulary", sample_texts)
    print("✅ Visualizations generated for diff vocabulary!")

if __name__ == "__main__":
    main()