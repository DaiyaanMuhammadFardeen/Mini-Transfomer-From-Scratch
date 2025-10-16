import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from model.model import Transformer
from CodeDiffDataset import CodeDiffDataset
from tokenizer.diff_text.Vocabulary import Vocabulary
from tokenizer.message.Vocabulary import Vocabulary
from tqdm import tqdm

print("\033[94m🚀 Starting script imports and initialization\033[0m", file=sys.stderr)

# Custom unpickler to handle module name changes
class PickleModuleRedirector(pickle.Unpickler):
    """Redirect old module paths to new ones during unpickling."""
    def find_class(self, module, name):
        # Redirect old Vocabulary imports to new locations
        if module == 'Vocabulary':
            # Determine which vocabulary based on file context
            # This is a fallback - will try to import from the new locations
            try:
                from tokenizer.diff_text.Vocabulary import Vocabulary
                if name == 'Vocabulary':
                    return Vocabulary
            except:
                pass
            try:
                from tokenizer.message.Vocabulary import Vocabulary
                if name == 'Vocabulary':
                    return Vocabulary
            except:
                pass
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
            print(f"\033[93m⚠️  Failed to load with redirection, trying standard pickle\033[0m", file=sys.stderr)
            f.seek(0)
            obj = pickle.load(f)
            return obj

print("\033[94m⚙️  Setting device\033[0m", file=sys.stderr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\033[92m✨ Device set to: {device}\033[0m", file=sys.stderr)

print("\033[94m🔧 Defining hyperparameters\033[0m", file=sys.stderr)
src_vocab_size = 46173
tgt_vocab_size = 39625
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2024
max_seq_length = 1024
dropout = 0.2
batch_size = 10
num_epochs = 3
print(f"\033[92m📊 Hyperparameters set - src_vocab_size: {src_vocab_size}, tgt_vocab_size: {tgt_vocab_size}, epochs: {num_epochs}\033[0m", file=sys.stderr)

print("\033[94m📁 Creating checkpoint directory\033[0m", file=sys.stderr)
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"\033[92m✓ Checkpoint directory ready: {checkpoint_dir}\033[0m", file=sys.stderr)

print("\033[94m📝 Defining load_data function\033[0m", file=sys.stderr)
def load_data(file_path):
    print(f"\033[96m📂 load_data() called with file_path: {file_path}\033[0m", file=sys.stderr)
    print(f"\033[96m📖 Reading parquet file...\033[0m", file=sys.stderr)
    df = pd.read_parquet(file_path)
    print(f"\033[92m✓ Parquet file loaded, shape: {df.shape}\033[0m", file=sys.stderr)
    
    print(f"\033[96m🧹 Filling NaN values in diff_text and message columns\033[0m", file=sys.stderr)
    df['diff_text'] = df['diff_text'].fillna('')
    df['message'] = df['message'].fillna('')
    print(f"\033[92m✓ NaN values filled\033[0m", file=sys.stderr)
    
    messages = df['message'].tolist()
    diffs = df['diff_text'].tolist()
    print(f"\033[92m✓ Extracted {len(messages)} messages and {len(diffs)} diffs\033[0m", file=sys.stderr)
    return messages, diffs

print("\033[94m📥 Loading data from parquet file\033[0m", file=sys.stderr)
file_path = "./dataset_part2.parquet"
messages, diffs = load_data(file_path)
print(f"\033[92m🎉 Data loaded successfully - {len(messages)} samples\033[0m", file=sys.stderr)

# Try loading from tokenizer subdirectories first (where they were originally saved)
print("\033[94m🔍 Attempting to load vocabularies from tokenizer subdirectories\033[0m", file=sys.stderr)

diff_vocab_path = "./tokenizer/diff_text/diff_vocab.pkl"
message_vocab_path = "./tokenizer/message/message_vocab.pkl"

# Fallback to root directory if not found
if not os.path.exists(diff_vocab_path):
    diff_vocab_path = "./diff_vocab.pkl"
    print(f"\033[93m⚠️  diff_vocab.pkl not found in tokenizer/diff_text/, checking root: {diff_vocab_path}\033[0m", file=sys.stderr)

if not os.path.exists(message_vocab_path):
    message_vocab_path = "./message_vocab.pkl"
    print(f"\033[93m⚠️  message_vocab.pkl not found in tokenizer/message/, checking root: {message_vocab_path}\033[0m", file=sys.stderr)

print(f"\033[94m📥 Loading source vocabulary from {diff_vocab_path}\033[0m", file=sys.stderr)
try:
    src_vocab = load_pickle_with_redirect(diff_vocab_path)
    print(f"\033[92m✓ Source vocab loaded, size: {len(src_vocab.stoi)}\033[0m", file=sys.stderr)
except Exception as e:
    print(f"\033[91m❌ Failed to load source vocabulary: {e}\033[0m", file=sys.stderr)
    raise

print(f"\033[94m📥 Loading target vocabulary from {message_vocab_path}\033[0m", file=sys.stderr)
try:
    tgt_vocab = load_pickle_with_redirect(message_vocab_path)
    print(f"\033[92m✓ Target vocab loaded, size: {len(tgt_vocab.stoi)}\033[0m", file=sys.stderr)
except Exception as e:
    print(f"\033[91m❌ Failed to load target vocabulary: {e}\033[0m", file=sys.stderr)
    raise

print("\033[94m🔍 Checking if source vocab needs truncation\033[0m", file=sys.stderr)
if len(src_vocab.stoi) > src_vocab_size:
    print(f"\033[93m✂️  Truncating source vocab from {len(src_vocab.stoi)} to {src_vocab_size}\033[0m", file=sys.stderr)
    src_vocab.itos = {k: src_vocab.itos[k] for k in range(src_vocab_size)}
    src_vocab.stoi = {v: k for k, v in src_vocab.itos.items()}
    print(f"\033[92m✓ Source vocab truncated\033[0m", file=sys.stderr)
src_vocab_size = len(src_vocab.stoi)
print(f"\033[92m📊 Final source vocab size: {src_vocab_size}\033[0m", file=sys.stderr)

print("\033[94m🔍 Checking if target vocab needs truncation\033[0m", file=sys.stderr)
if len(tgt_vocab.stoi) > tgt_vocab_size:
    print(f"\033[93m✂️  Truncating target vocab from {len(tgt_vocab.stoi)} to {tgt_vocab_size}\033[0m", file=sys.stderr)
    tgt_vocab.itos = {k: tgt_vocab.itos[k] for k in range(tgt_vocab_size)}
    tgt_vocab.stoi = {v: k for k, v in tgt_vocab.itos.items()}
    print(f"\033[92m✓ Target vocab truncated\033[0m", file=sys.stderr)
tgt_vocab_size = len(tgt_vocab.stoi)
print(f"\033[92m📊 Final target vocab size: {tgt_vocab_size}\033[0m", file=sys.stderr)

print("\033[94m🏗️  Creating CodeDiffDataset\033[0m", file=sys.stderr)
dataset = CodeDiffDataset(messages, diffs, src_vocab, tgt_vocab, max_seq_length)
print(f"\033[92m✓ Dataset created with {len(dataset)} samples\033[0m", file=sys.stderr)

print("\033[94m🔗 Creating DataLoader\033[0m", file=sys.stderr)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)
print(f"\033[92m✓ DataLoader created with batch_size: {batch_size}\033[0m", file=sys.stderr)

print("\033[94m🤖 Initializing Transformer model\033[0m", file=sys.stderr)
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
print(f"\033[92m✓ Transformer model initialized with {sum(p.numel() for p in transformer.parameters())} and moved to {device}\033[0m", file=sys.stderr)

print("\033[94m📉 Initializing loss criterion\033[0m", file=sys.stderr)
criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.stoi["<PAD>"])
print(f"\033[92m✓ CrossEntropyLoss criterion created with PAD ignore index\033[0m", file=sys.stderr)

print("\033[94m🔄 Initializing optimizer\033[0m", file=sys.stderr)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
print(f"\033[92m✓ Adam optimizer initialized with lr=0.0001\033[0m", file=sys.stderr)

print("\033[94m🎛️  Initializing GradScaler\033[0m", file=sys.stderr)
scaler = GradScaler("cuda")
print(f"\033[92m✓ GradScaler initialized for CUDA\033[0m", file=sys.stderr)

print("\033[94m🎓 Setting model to training mode\033[0m", file=sys.stderr)
transformer.train()
print(f"\033[92m✓ Model training mode enabled\033[0m", file=sys.stderr)

print("\033[95m🔥 Starting training loop\033[0m", file=sys.stderr)
for epoch in range(num_epochs):
    print(f"\n\033[95m{'='*60}\033[0m", file=sys.stderr)
    print(f"\033[95m🌟 EPOCH {epoch+1}/{num_epochs} START 🌟\033[0m", file=sys.stderr)
    print(f"\033[95m{'='*60}\033[0m", file=sys.stderr)
    total_loss = 0
    
    print(f"\033[96m▶️  Starting batch iteration for epoch {epoch+1}\033[0m", file=sys.stderr)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", position=1, leave=False)
    for batch_idx, (src_data, tgt_data) in enumerate(pbar):
        src_data, tgt_data = src_data.to(device), tgt_data.to(device)
        
        optimizer.zero_grad()
        
        with autocast("cuda"):
            output = transformer(src_data, tgt_data[:, :-1])
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        
        scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        current_avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'batch_loss': f'{loss.item():.4f}', 'avg_loss': f'{current_avg_loss:.4f}'})
    
    print(f"\033[92m✓ Epoch {epoch+1}: All batches completed\033[0m", file=sys.stderr)
    avg_loss = total_loss / len(dataloader)
    print(f"\033[92m📈 Epoch {epoch+1}: Average loss calculated: {avg_loss:.4f}\033[0m", file=sys.stderr)
    print(f"Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    print(f"\033[94m💾 Epoch {epoch+1}: Creating checkpoint\033[0m", file=sys.stderr)
    checkpoint_path = os.path.join(checkpoint_dir, f"transformer_epoch_{epoch+1}.pth")
    print(f"\033[96m💿 Epoch {epoch+1}: Saving checkpoint to {checkpoint_path}\033[0m", file=sys.stderr)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'scaler_state_dict': scaler.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"\033[92m✅ Epoch {epoch+1}: Checkpoint saved successfully\033[0m", file=sys.stderr)
    print(f"\033[95m{'='*60}\033[0m", file=sys.stderr)
    print(f"\033[95m🎊 EPOCH {epoch+1}/{num_epochs} END 🎊\033[0m", file=sys.stderr)
    print(f"\033[95m{'='*60}\033[0m\n", file=sys.stderr)

print("\033[95m🏁 Training loop completed\033[0m", file=sys.stderr)
print("\033[94m💾 Saving final model state\033[0m", file=sys.stderr)
torch.save(transformer.state_dict(), "transformer_model.pth")
print("\033[92m✅ Final model saved to transformer_model.pth\033[0m", file=sys.stderr)
print("✅ Model saved successfully as transformer_model.pth")
print("\033[92m🎉 Script execution completed successfully\033[0m", file=sys.stderr)
