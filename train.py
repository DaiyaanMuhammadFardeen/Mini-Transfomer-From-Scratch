import torch
from model.model import Transformer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from customTokenizer import CodeDiffDataset, Vocabulary, load_parquet
import pickle
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2024
max_seq_length = 512
dropout = 0.2
batch_size = 16
num_epochs = 3

file_path = "./dataset_part2.parquet"  # Replace with your parquet file path
messages, diffs = load_parquet(file_path)

# Load vocabularies and model
with open("src_vocab.pkl", "rb") as f:
    src_vocab = pickle.load(f)
with open("tgt_vocab.pkl", "rb") as f:
    tgt_vocab = pickle.load(f)

if len(src_vocab.stoi) > src_vocab_size:
    # Truncate itos/stoi to 0..4999
    src_vocab.itos = {k: src_vocab.itos[k] for k in range(src_vocab_size)}
    src_vocab.stoi = {v: k for k, v in src_vocab.itos.items()}
src_vocab_size = len(src_vocab.stoi)

if len(tgt_vocab.stoi) > tgt_vocab_size:
    tgt_vocab.itos = {k: tgt_vocab.itos[k] for k in range(tgt_vocab_size)}
    tgt_vocab.stoi = {v: k for k, v in tgt_vocab.itos.items()}
tgt_vocab_size = len(tgt_vocab.stoi)

# Step 5: Create Dataset and DataLoader
dataset = CodeDiffDataset(messages, diffs, src_vocab, tgt_vocab, max_seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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

criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.stoi["<PAD>"])
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (src_data, tgt_data) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", position=1, leave=False)):
        src_data, tgt_data = src_data.to(device), tgt_data.to(device)

        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])  # Exclude <EOS> for input
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))  # Exclude <SOS> for target
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}")

torch.save(transformer.state_dict(), "transformer_model.pth")
print("✅ Model saved successfully as transformer_model.pth")
