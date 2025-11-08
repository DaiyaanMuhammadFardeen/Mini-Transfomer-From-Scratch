import pandas as pd
import pickle
import sys
from CodeDiffDataset import CodeDiffDataset

class PickleModuleRedirector(pickle.Unpickler):
    """Redirect old module paths to new ones during unpickling."""
    def find_class(self, module, name):
        if module == 'Vocabulary':
            try:
                from tokenizer.DiffVocabulary import DiffVocabulary
                if name == 'DiffVocabulary' or name == 'Vocabulary':
                    return DiffVocabulary
            except:
                pass
            try:
                from tokenizer.MsgVocabulary import MsgVocabulary
                if name == 'MsgVocabulary' or name == 'Vocabulary':
                    return MsgVocabulary
            except:
                pass
        elif module in ('DiffVocabulary', 'tokenizer.diff_text.Vocabulary'):
            from tokenizer.DiffVocabulary import DiffVocabulary
            return DiffVocabulary
        elif module in ('MsgVocabulary', 'tokenizer.message.Vocabulary'):
            from tokenizer.MsgVocabulary import MsgVocabulary
            return MsgVocabulary
        return super().find_class(module, name)

def load_pickle_with_redirect(file_path):
    """Load pickle file with module path redirection."""
    print(f"\033[96m🔓 Loading pickle from {file_path} with module redirection\033[0m", file=sys.stderr)
    with open(file_path, 'rb') as f:
        unpickler = PickleModuleRedirector(f)
        try:
            obj = unpickler.load()
            print(f"\033[92m✅ Successfully loaded {file_path}\033[0m", file=sys.stderr)
            return obj
        except Exception as e:
            print(f"\033[93m⚠️  Failed to load with redirection, trying standard pickle: {e}\033[0m", file=sys.stderr)
            f.seek(0)
            obj = pickle.load(f)
            return obj


def analyze_sequence_lengths(file_path, diff_vocab_path, message_vocab_path):
    """Analyze sequence lengths in the dataset."""
    print("Loading data from parquet file...")
    df = pd.read_parquet(file_path)
    df = df.sample(frac=0.1, random_state=42)  # Match your training script
    print(f"Parquet file loaded, shape: {df.shape}")
    
    df['diff_text'] = df['diff_text'].fillna('')
    df['message'] = df['message'].fillna('')
    print("NaN values filled")
    
    messages = df['message'].tolist()
    diffs = df['diff_text'].tolist()
    print(f"Extracted {len(messages)} messages and {len(diffs)} diffs")
    
    print("Loading vocabularies...")
    diff_vocab = load_pickle_with_redirect(diff_vocab_path)
    message_vocab = load_pickle_with_redirect(message_vocab_path)
    
    if not diff_vocab or not message_vocab:
        print("Failed to load vocabularies")
        return
    
    print("Creating dataset...")
    dataset = CodeDiffDataset(messages, diffs, diff_vocab, message_vocab, 512)
    
    diff_lengths = []
    message_lengths = []
    
    print("Analyzing sequence lengths...")
    for i in range(len(dataset)):
        src_tokens, tgt_tokens = dataset[i]
        diff_lengths.append((src_tokens != diff_vocab.stoi["<PAD>"]).sum().item())
        message_lengths.append((tgt_tokens != message_vocab.stoi["<PAD>"]).sum().item())
        
        if i % 1000 == 0:
            print(f"Processed {i}/{len(dataset)} samples")
    
    print("\n=== Sequence Length Analysis ===")
    print(f"Diff lengths - Min: {min(diff_lengths)}, Max: {max(diff_lengths)}, Mean: {sum(diff_lengths)/len(diff_lengths):.2f}")
    print(f"Message lengths - Min: {min(message_lengths)}, Max: {max(message_lengths)}, Mean: {sum(message_lengths)/len(message_lengths):.2f}")
    
    # Count very short sequences
    short_messages = sum(1 for l in message_lengths if l <= 5)
    long_messages = sum(1 for l in message_lengths if l > 20)
    
    print(f"\nMessages with <= 5 tokens: {short_messages} ({100*short_messages/len(message_lengths):.2f}%)")
    print(f"Messages with > 20 tokens: {long_messages} ({100*long_messages/len(message_lengths):.2f}%)")
    
    # Check for outliers
    message_lengths_sorted = sorted(message_lengths)
    q1 = message_lengths_sorted[len(message_lengths_sorted)//4]
    q3 = message_lengths_sorted[3*len(message_lengths_sorted)//4]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = sum(1 for l in message_lengths if l < lower_bound or l > upper_bound)
    print(f"\nOutliers: {outliers} ({100*outliers/len(message_lengths):.2f}%)")

if __name__ == "__main__":
    data_path = "./train_data.parquet"
    diff_vocab_path = "./tokenizer/diff_vocab.pkl"
    message_vocab_path = "./tokenizer/message_vocab.pkl"
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        diff_vocab_path = sys.argv[2]
    if len(sys.argv) > 3:
        message_vocab_path = sys.argv[3]
        
    analyze_sequence_lengths(data_path, diff_vocab_path, message_vocab_path)
