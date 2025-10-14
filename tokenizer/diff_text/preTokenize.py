from Vocabulary import Vocabulary
import pandas as pd
import pickle
import json
import random

def load_data(file_path):
    df = pd.read_parquet(file_path)
    df['diff_text'] = df['diff_text'].fillna('')
    return df['diff_text'].tolist()

def main():
    file_path = "./dataset_part2.parquet"
    diffs = load_data(file_path)


    diff_vocab = Vocabulary(target_size=50000)
    diff_vocab.build_vocabulary(diffs)
    with open("diff_vocab.pkl", "wb") as f:
        pickle.dump(diff_vocab, f)

    # Export tokenized samples to JSON for inspection
    sample_size = 1000
    sampled_indices = random.sample(range(len(diffs)), min(sample_size, len(diffs)))

    diff_tokenized = []
    for idx in sampled_indices:
        text = diffs[idx]
        tokens = diff_vocab.tokenize(text)
        ids = diff_vocab.numericalize(text)
        diff_tokenized.append({"original": text, "tokens": tokens, "ids": ids})

    with open("tokenized_diffs_sample.json", "w", encoding="utf-8") as f:
        json.dump(diff_tokenized, f, ensure_ascii=False, indent=4)

    print("✅ Vocabularies saved! Tokenized samples exported to JSON.")

if __name__ == "__main__":
    main()
