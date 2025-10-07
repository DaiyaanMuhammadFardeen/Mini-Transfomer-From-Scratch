import torch
import pickle
from model.model import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_vocab_size = 5000  # Adjust based on vocabulary size
tgt_vocab_size = 5000  # Adjust based on vocabulary size
d_model = 256
num_heads = 8
num_layers = 6
d_ff = 1024
max_seq_length = 4096
dropout = 0.2
batch_size = 64
num_epochs = 50

transformer = Transformer(
    src_vocab_size,
    tgt_vocab_size,
    d_model,
    num_heads,
    num_layers,
    d_ff,
    max_seq_length,
    dropout
).to(device)

# Load vocabularies and model
with open("src_vocab.pkl", "rb") as f:
    src_vocab = pickle.load(f)
with open("tgt_vocab.pkl", "rb") as f:
    tgt_vocab = pickle.load(f)

transformer.load_state_dict(torch.load("transformer_model.pth"))
transformer.eval()

def translate_message(message, max_len=100):
    # Tokenize and numericalize input
    src_tokens = [src_vocab.stoi["<SOS>"]] + src_vocab.numericalize(message)[:max_len-2] + [src_vocab.stoi["<EOS>"]]
    src_tokens = src_tokens + [src_vocab.stoi["<PAD>"]] * (max_len - len(src_tokens))
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)

    # Generate target sequence
    tgt_tokens = [tgt_vocab.stoi["<SOS>"]]
    tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            output = transformer(src_tensor, tgt_tensor)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            tgt_tokens.append(next_token)
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            if next_token == tgt_vocab.stoi["<EOS>"]:
                break

    # Convert to text
    translated = [tgt_vocab.itos[idx] for idx in tgt_tokens[1:]]  # Skip <SOS>
    return " ".join(translated)

# Example
message = "Memoize the patient name lookup."
translated_diff = translate_message(message)
print(f"Input: {message}")
print(f"Output: {translated_diff}")
