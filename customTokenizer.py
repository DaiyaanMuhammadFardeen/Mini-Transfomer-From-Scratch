import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm
import pickle
from multiprocessing import Pool, cpu_count

def parallel_tokenize(args):
    instance, sentence = args
    return instance.tokenize(sentence)

def load_parquet(file_path):
    df = pd.read_parquet(file_path)
    # Handle missing values
    df['message'] = df['message'].fillna('')
    df['diff_text'] = df['diff_text'].fillna('')
    return df['message'].tolist(), df['diff_text'].tolist()

class Vocabulary:
    def __init__(self, freq_threshold=2):
        # Initialize special tokens, including dataset-specific diff tags
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>",
            4: "<ADD>",
            5: "</ADD>",
            6: "<REMOVE>",
            7: "</REMOVE>",
            8: "<COMMENT_ADD>",
            9: "</COMMENT_ADD>",
            10: "<COMMENT_REMOVE>",
            11: "</COMMENT_REMOVE>",
            12: "<MODIFY>",
            13: "</MODIFY>",
            14: "<COMMENT_MODIFY>",
            15: "</COMMENT_MODIFY>"
        }
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        print(f"🧠 Building vocabulary using {cpu_count()} CPU cores...")

        # Use multiprocessing to tokenize in parallel
        with Pool(cpu_count()) as pool:
            all_token_lists = list(
                tqdm(
                    pool.imap(parallel_tokenize, [(self, s) for s in sentence_list]),
                    total=len(sentence_list),
                    desc="🔤 Tokenizing",
                    ncols=80
                )
            )
        # Merge results
        for tokens in tqdm(all_token_lists, desc="📊 Counting frequencies", ncols=80):
            frequencies.update(tokens)

        # Filter tokens by frequency
        sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        idx = len(self.itos)
        for word, freq in tqdm(sorted_freq, desc="📚 Building vocab", ncols=80):
            if freq >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        print(f"✅ Vocabulary built! Total tokens: {len(self.stoi)}")

    def tokenize(self, text):
        # Custom tokenization to handle diff-specific tags
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == '<':
                # Look for a closing '>'
                j = text.find('>', i)
                if j == -1:
                    break  # Malformed tag
                tag = text[i:j+1]
                if tag in self.stoi:
                    tokens.append(tag)
                    i = j + 1
                else:
                    # Treat as regular text if not a known tag
                    tokens.append(text[i])
                    i += 1
            else:
                # Collect non-tag characters until a space or tag
                start = i
                while i < len(text) and text[i] != ' ' and text[i] != '<':
                    i += 1
                if start != i:
                    tokens.append(text[start:i])
                if i < len(text) and text[i] == ' ':
                    tokens.append(' ')
                    i += 1
        return tokens

    def numericalize(self, text):
        tokens = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

    def numericalize_batch(self, texts):
        """Tokenize and numericalize a batch of texts with progress bar."""
        print("🔢 Converting text to token IDs...")
        numericalized_data = []
        for text in tqdm(texts, desc="Numericalizing sentences", unit="sentence"):
            numericalized_data.append(self.numericalize(text))
        return numericalized_data

class CodeDiffDataset(Dataset):
    def __init__(self, messages, diffs, src_vocab, tgt_vocab, max_seq_length):
        self.messages = messages  # List of strings
        self.diffs = diffs  # List of strings
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        src_text = self.messages[idx]  # Raw source text (string)
        tgt_text = self.diffs[idx]     # Raw target text (string)

        # Tokenize the source and target text into lists of token indices
        src_tokens = self.tokenize_text(src_text, self.src_vocab)
        tgt_tokens = self.tokenize_text(tgt_text, self.tgt_vocab)

        # Add <SOS> and <EOS> tokens
        src_tokens = [self.src_vocab.stoi["<SOS>"]] + src_tokens + [self.src_vocab.stoi["<EOS>"]]
        tgt_tokens = [self.tgt_vocab.stoi["<SOS>"]] + tgt_tokens + [self.tgt_vocab.stoi["<EOS>"]]

        # Pad sequences to max_seq_length if necessary
        src_tokens = src_tokens[:self.max_seq_length] + [self.src_vocab.stoi["<PAD>"]] * (self.max_seq_length - len(src_tokens))
        tgt_tokens = tgt_tokens[:self.max_seq_length] + [self.tgt_vocab.stoi["<PAD>"]] * (self.max_seq_length - len(tgt_tokens))

        # Convert to tensors
        src_tokens = torch.tensor(src_tokens, dtype=torch.long)
        tgt_tokens = torch.tensor(tgt_tokens, dtype=torch.long)
        return src_tokens, tgt_tokens

    def tokenize_text(self, text, vocab):
        # Use the custom tokenization and numericalization from the Vocabulary class
        return vocab.numericalize(text)

def main():
    file_path = "./dataset_part2.parquet"  # Replace with your parquet file path
    messages, diffs = load_parquet(file_path)

# Build vocabularies
    src_vocab = Vocabulary(freq_threshold=2)
    tgt_vocab = Vocabulary(freq_threshold=2)
    src_vocab.build_vocabulary(messages)
    tgt_vocab.build_vocabulary(diffs)

    with open("src_vocab.pkl", "wb") as f:
        pickle.dump(src_vocab, f)
    with open("tgt_vocab.pkl", "wb") as f:
        pickle.dump(tgt_vocab, f)
    print("✅ Vocabularies saved successfully")

if __name__ == "__main__":
    main()
