import pickle
import sys

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

def analyze_vocabulary(vocab_path, vocab_name):
    """Analyze vocabulary for potential imbalances."""
    print(f"\n=== Analyzing {vocab_name} ===")
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        print(f"Vocabulary size: {len(vocab.stoi)}")

        # Check for special tokens
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        for token in special_tokens:
            if token in vocab.stoi:
                print(f"Found {token} at index {vocab.stoi[token]}")
            else:
                print(f"Missing {token}")

        # Show first 20 tokens
        print("\nFirst 20 tokens:")
        for i in range(min(20, len(vocab.itos))):
            print(f"  {i}: {vocab.itos[i]}")

        # Check for potential issues
        if hasattr(vocab, 'freqs') and vocab.freqs:
            print("\nMost frequent tokens:")
            sorted_freqs = sorted(vocab.freqs.items(), key=lambda x: x[1], reverse=True)
            for token, freq in sorted_freqs[:20]:
                print(f"  {token}: {freq}")

    except Exception as e:
        print(f"Error loading {vocab_path}: {e}")

def compare_vocabularies(diff_vocab_path, msg_vocab_path):
    """Compare diff and message vocabularies."""
    print("=== Comparing Vocabularies ===")

    try:
        diff_vocab = load_pickle_with_redirect(diff_vocab_path)
        msg_vocab = load_pickle_with_redirect(msg_vocab_path)

        print(f"Diff vocab size: {len(diff_vocab.stoi)}")
        print(f"Message vocab size: {len(msg_vocab.stoi)}")

        # Check overlap
        diff_tokens = set(diff_vocab.stoi.keys())
        msg_tokens = set(msg_vocab.stoi.keys())
        common_tokens = diff_tokens.intersection(msg_tokens)

        print(f"Common tokens: {len(common_tokens)}")

    except Exception as e:
        print(f"Error comparing vocabularies: {e}")

if __name__ == "__main__":
    diff_vocab_path = "./tokenizer/diff_vocab.pkl"
    msg_vocab_path = "./tokenizer/message_vocab.pkl"

    if len(sys.argv) > 1:
        diff_vocab_path = sys.argv[1]
    if len(sys.argv) > 2:
        msg_vocab_path = sys.argv[2]

    analyze_vocabulary(diff_vocab_path, "Diff Vocabulary")
    analyze_vocabulary(msg_vocab_path, "Message Vocabulary")
    compare_vocabularies(diff_vocab_path, msg_vocab_path)
