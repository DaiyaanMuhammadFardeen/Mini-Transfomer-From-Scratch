import pandas as pd
import pickle
import json
import random
from MsgVocabulary import MsgVocabulary
from visualization import generate_comprehensive_report


def load_data(file_path):
    df = pd.read_parquet(file_path)
    df['message'] = df['message'].fillna('')
    return df['message'].tolist()

def main():
    file_path = "./messages.parquet"
    messages = load_data(file_path)

    vocab = MsgVocabulary()
    vocab.build_vocab(messages)
    with open("message_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Export tokenized samples for inspection
    sample_size = 1000
    sampled_indices = random.sample(range(len(messages)), min(sample_size, len(messages)))
    tokenized_samples = []
    for idx in sampled_indices:
        text = messages[idx]
        tokens = vocab.tokenize(text)
        ids = vocab.numericalize(text)
        tokenized_samples.append({"original": text, "tokens": tokens, "ids": ids})

    with open("tokenized_messages_sample.json", "w", encoding="utf-8") as f:
        json.dump(tokenized_samples, f, ensure_ascii=False, indent=4)

    print("✅ Message vocabulary saved! Tokenized samples exported to JSON.")
    
    # Generate visualizations for the message vocabulary
    print("Generating visualizations for message vocabulary...")
    sample_texts = ["Fix critical security vulnerability in auth module with JWT implementation",
                    "Add new feature for user authentication using React and Node.js",
                    "Refactor data processing module for better performance with caching"]
    generate_comprehensive_report(vocab, "MsgVocabulary", vocab, "MsgVocabulary", sample_texts)
    print("✅ Visualizations generated for message vocabulary!")

if __name__ == "__main__":
    main()
