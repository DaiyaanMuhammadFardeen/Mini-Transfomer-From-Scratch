import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import os
import sys
import argparse
import csv
from torch.utils.data import DataLoader, random_split
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from model.model import Transformer
from CodeDiffDataset import CodeDiffDataset
from tqdm import tqdm

# Set working directory to project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("\033[94m🚀 Starting script imports and initialization\033[0m", file=sys.stderr)

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

def load_data(file_path):
    """Load data from parquet file."""
    print(f"\033[96m📂 load_data() called with file_path: {file_path}\033[0m", file=sys.stderr)
    print(f"\033[96m📖 Reading parquet file...\033[0m", file=sys.stderr)
    df = pd.read_parquet(file_path)
    # df = df.sample(frac=0.2, random_state=42) # random_state for reproducibility
    print(f"\033[92m✓ Parquet file loaded, shape: {df.shape}\033[0m", file=sys.stderr)
    df['diff_text'] = df['diff_text'].fillna('')
    df['message'] = df['message'].fillna('')
    print(f"\033[92m✓ NaN values filled\033[0m", file=sys.stderr)
    messages = df['message'].tolist()
    diffs = df['diff_text'].tolist()
    print(f"\033[92m✓ Extracted {len(messages)} messages and {len(diffs)} diffs\033[0m", file=sys.stderr)
    return messages, diffs

def length_regularization_loss(output, tgt_data, eos_idx, min_length=5, lambda_len=0.1):
    """
    Penalize sequences that end too early.
    output: Transformer output (batch_size, seq_len, vocab_size)
    tgt_data: Target tokens (batch_size, seq_len)
    eos_idx: Index of <EOS> token
    min_length: Minimum desired sequence length (tune based on dataset)
    lambda_len: Weight for length penalty
    """
    # Get predicted probabilities
    probs = torch.softmax(output, dim=-1)  # (batch_size, seq_len, vocab_size)
    eos_probs = probs[:, :, eos_idx]  # (batch_size, seq_len)

    # Estimate sequence length as first position where <EOS> prob is high
    max_probs, _ = torch.max(eos_probs, dim=1)  # (batch_size,)
    pred_lengths = torch.argmax((eos_probs > 0.5 * max_probs.unsqueeze(1)).float(), dim=1)  # (batch_size,)

    # Penalize if predicted length < min_length
    length_penalty = torch.clamp(min_length - pred_lengths.float(), min=0) ** 2
    return lambda_len * length_penalty.mean()

def contrastive_loss(embeddings, labels, temperature=0.1, margin=1.0):
    """
    Compute contrastive loss to encourage similar embeddings for similar messages
    and dissimilar embeddings for different messages.

    Args:
        embeddings: Encoded representations of inputs (batch_size, embedding_dim)
        labels: Ground truth labels (batch_size,)
        temperature: Temperature for scaling similarities
        margin: Margin for negative pairs

    Returns:
        contrastive_loss: Scalar loss value
    """
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    # Compute pairwise cosine similarities
    similarity_matrix = torch.matmul(embeddings, embeddings.t()) / temperature

    # Create positive and negative masks
    labels = labels.unsqueeze(0)
    positive_mask = (labels == labels.t()).float()
    negative_mask = 1 - positive_mask

    # Remove self-similarities
    eye = torch.eye(positive_mask.size(0), device=positive_mask.device)
    positive_mask = positive_mask - eye

    # Compute loss for positive pairs (pull similar closer)
    positive_loss = -torch.log(torch.exp(similarity_matrix) * positive_mask).sum() / positive_mask.sum().clamp(min=1)

    # Compute loss for negative pairs (push dissimilar apart)
    negative_loss = torch.relu(margin - similarity_matrix) * negative_mask
    negative_loss = negative_loss.sum() / negative_mask.sum().clamp(min=1)

    return positive_loss + negative_loss

def diversity_loss(output, tgt_data, vocab_size, diversity_weight=0.01):
    """
    Compute diversity loss to penalize repetitive outputs.

    Args:
        output: Model output logits (batch_size, seq_len, vocab_size)
        tgt_data: Target data (batch_size, seq_len)
        vocab_size: Size of vocabulary
        diversity_weight: Weight for diversity loss

    Returns:
        diversity_loss: Scalar loss value
    """
    batch_size, seq_len, _ = output.shape

    # Convert logits to probabilities
    probs = torch.softmax(output, dim=-1)

    # Compute entropy for each position
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

    # Average entropy across sequence
    avg_entropy = torch.mean(entropy)

    # Diversity loss is negative entropy (encourage higher entropy, more diverse outputs)
    return -diversity_weight * avg_entropy

def ngram_repetition_penalty(logits, generated_tokens, n=2, penalty=1.0):
    """
    Apply n-gram repetition penalty to logits.

    Args:
        logits: Current logits (batch_size, vocab_size)
        generated_tokens: Previously generated tokens (batch_size, seq_len)
        n: N-gram size
        penalty: Penalty factor

    Returns:
        penalized_logits: Logits with repetition penalty applied
    """
    if generated_tokens.size(1) < n:
        return logits

    batch_size, vocab_size = logits.shape

    for i in range(batch_size):
        # Get last n-1 tokens
        if generated_tokens.size(1) >= n-1:
            last_tokens = generated_tokens[i, -(n-1):].tolist()

            # Simple penalty - could be made more sophisticated
            for token in last_tokens:
                if 0 <= token < vocab_size:  # Make sure token is valid
                    logits[i, token] -= penalty

    return logits

def load_checkpoint(checkpoint_path, transformer, optimizer, scheduler, scaler, device):
    """Load checkpoint and resume training state."""
    print(f"\033[96m📥 Loading checkpoint from {checkpoint_path}\033[0m", file=sys.stderr)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    transformer.load_state_dict(checkpoint['model_state_dict'])
    print(f"\033[92m✓ Model state loaded\033[0m", file=sys.stderr)

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"\033[92m✓ Optimizer state loaded\033[0m", file=sys.stderr)

    # Load scaler state if available (only for float16 mode)
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"\033[92m✓ GradScaler state loaded\033[0m", file=sys.stderr)
    
    # Check if checkpoint was saved with bfloat16
    use_bfloat16_ckpt = checkpoint.get('use_bfloat16', False)
    if use_bfloat16_ckpt:
        print("\033[93m⚠ Checkpoint was trained with bfloat16\033[0m", file=sys.stderr)

    # Get the epoch number
    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('val_loss', float('inf'))

    print(f"\033[92m✅ Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}\033[0m", file=sys.stderr)

    return start_epoch, best_val_loss

def main(args):
    print("\033[94m⚙️  Setting device\033[0m", file=sys.stderr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\033[92m✨ Device set to: {device}\033[0m", file=sys.stderr)

    print("\033[94m🔧 Defining hyperparameters\033[0m", file=sys.stderr)
    d_model = args.d_model
    num_heads = args.num_heads
    num_layers = args.num_layers
    d_ff = args.d_ff
    max_seq_length = args.max_seq_length
    dropout = args.dropout
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    print(f"\033[92m📊 Hyperparameters set - d_model: {d_model}, num_epochs: {num_epochs}\033[0m", file=sys.stderr)

    print("\033[94m📁 Creating checkpoint directory\033[0m", file=sys.stderr)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"\033[92m✓ Checkpoint directory ready: {args.checkpoint_dir}\033[0m", file=sys.stderr)

    print("\033[94m📥 Loading data from parquet file\033[0m", file=sys.stderr)
    messages, diffs = load_data(args.data_path)
    print(f"\033[92m🎉 Data loaded successfully - {len(messages)} samples\033[0m", file=sys.stderr)

    print("\033[94m🔍 Loading vocabularies\033[0m", file=sys.stderr)
    diff_vocab_path = args.diff_vocab_path
    message_vocab_path = args.message_vocab_path
    if not os.path.exists(diff_vocab_path):
        diff_vocab_path = "./diff_vocab.pkl"
        print(f"\033[93m⚠️  diff_vocab.pkl not found in tokenizer/diff_text/, checking root: {diff_vocab_path}\033[0m", file=sys.stderr)
    if not os.path.exists(message_vocab_path):
        message_vocab_path = "./message_vocab.pkl"
        print(f"\033[93m⚠️  message_vocab.pkl not found in tokenizer/message/, checking root: {message_vocab_path}\033[0m", file=sys.stderr)

    print(f"\033[94m📥 Loading source vocabulary from {diff_vocab_path}\033[0m", file=sys.stderr)
    src_vocab = load_pickle_with_redirect(diff_vocab_path)
    print(f"\033[92m✓ Source vocab loaded, size: {len(src_vocab.stoi)}\033[0m", file=sys.stderr)

    print(f"\033[94m📥 Loading target vocabulary from {message_vocab_path}\033[0m", file=sys.stderr)
    tgt_vocab = load_pickle_with_redirect(message_vocab_path)
    print(f"\033[92m✓ Target vocab loaded, size: {len(tgt_vocab.stoi)}\033[0m", file=sys.stderr)

    src_vocab_size = len(src_vocab.stoi)
    tgt_vocab_size = len(tgt_vocab.stoi)
    print(f"\033[92m📊 Final vocab sizes - src: {src_vocab_size}, tgt: {tgt_vocab_size}\033[0m", file=sys.stderr)

    print("\033[94m🏗️  Creating CodeDiffDataset\033[0m", file=sys.stderr)
    dataset = CodeDiffDataset(messages, diffs, src_vocab, tgt_vocab, max_seq_length)
    print(f"\033[92m✓ Dataset created with {len(dataset)} samples\033[0m", file=sys.stderr)

    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"\033[92m✓ Dataset split: {train_size} train, {val_size} validation\033[0m", file=sys.stderr)

    print("\033[94m🔗 Creating DataLoaders\033[0m", file=sys.stderr)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,         # 6 workers can bottleneck on 16GB RAM; 4 is safer
        pin_memory=True,       # Faster CPU→GPU transfer
        prefetch_factor=2,     # Prefetch next 2 batches while GPU works
        persistent_workers=True # Don't restart workers each epoch
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    print(f"\033[92m✓ DataLoaders created with batch_size: {batch_size}, pin_memory=True\033[0m", file=sys.stderr)

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
    print(f"\033[92m✓ Transformer model initialized with {sum(p.numel() for p in transformer.parameters())} parameters\033[0m", file=sys.stderr)

    print("\033[94m📉 Initializing loss criterion\033[0m", file=sys.stderr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi["<PAD>"], label_smoothing=0.1)
    print(f"\033[92m✓ CrossEntropyLoss criterion created (label_smoothing=0.1)\033[0m", file=sys.stderr)

    print("\033[94m🔄 Initializing optimizer and scheduler\033[0m", file=sys.stderr)
    
    # Noam (transformer) learning rate schedule
    def noam_schedule(step, d_model, warmup_steps=4000):
        """Noam learning rate schedule from 'Attention Is All You Need'."""
        step = max(1, step)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    
    # Use AdamW with correct decoupled weight decay (Loshchilov & Hutter 2017)
    optimizer = optim.AdamW(
        transformer.parameters(),
        lr=1e-4,  # Peak LR; warmup will scale this
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01  # Standard for transformers
    )
    
    # LambdaLR scheduler with Noam schedule - steps every batch
    global_step = 0
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: noam_schedule(step, d_model=d_model, warmup_steps=4000)
    )
    
    print(f"\033[92m✓ AdamW optimizer and Noam scheduler initialized\033[0m", file=sys.stderr)

    print("\033[94m🎛️  Initializing mixed precision\033[0m", file=sys.stderr)
    # Use bfloat16 if supported (ROCm 5.2+), fallback to float16 with GradScaler
    use_bfloat16 = torch.cuda.is_bf16_supported() if device.type == 'cuda' else False
    if use_bfloat16:
        print("\033[92m✓ Using bfloat16 mixed precision (no GradScaler needed)\033[0m", file=sys.stderr)
        scaler = None
    else:
        print("\033[93m⚠ bfloat16 not supported, using float16 with GradScaler\033[0m", file=sys.stderr)
        scaler = GradScaler("cuda")
        print(f"\033[92m✓ GradScaler initialized for CUDA\033[0m", file=sys.stderr)

    print("\033[94m🎓 Setting model to training mode\033[0m", file=sys.stderr)
    transformer.train()
    print(f"\033[92m✓ Model training mode enabled\033[0m", file=sys.stderr)

    # Add new hyperparameters for contrastive learning
    contrastive_weight = 0.1  # Weight for contrastive loss
    temperature = 0.1  # Temperature for contrastive loss

    # Add new hyperparameters for diversity loss
    diversity_weight = 0.01  # Weight for diversity loss
    repetition_penalty = 1.0  # Penalty for repeated tokens

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, transformer, optimizer, scheduler, scaler, device
        )
        print(f"\033[95m🔥 Resuming training from epoch {start_epoch + 1}\033[0m", file=sys.stderr)
    else:
        print("\033[95m🔥 Starting training from scratch\033[0m", file=sys.stderr)
    
    # Initialize CSV logging
    log_path = os.path.join(args.checkpoint_dir, "training_log.csv")
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'step', 'train_loss', 'val_loss', 'lr', 'gpu_mem_mb'])

    for epoch in range(start_epoch, num_epochs):
        print(f"\n\033[95m{'='*60}\033[0m", file=sys.stderr)
        print(f"\033[95m🌟 EPOCH {epoch+1}/{num_epochs} START 🌟\033[0m", file=sys.stderr)
        print(f"\033[95m{'='*60}\033[0m", file=sys.stderr)
        total_train_loss = 0

        print(f"\033[96m▶️  Starting train batch iteration for epoch {epoch+1}\033[0m", file=sys.stderr)
        pbar = tqdm(train_dataloader, 
                    desc=f"Epoch {epoch+1}/{num_epochs} Train", 
                    position=0, 
                    leave=True,
                    unit="batch",
                    dynamic_ncols=True)
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch data - now includes change_features
            if len(batch_data) == 3:
                src_data, tgt_data, change_features = batch_data
            else:
                # Backward compatibility with old 2-tuple format
                src_data, tgt_data = batch_data
                change_features = None
            
            src_data = src_data.to(device)
            tgt_data = tgt_data.to(device)
            if change_features is not None:
                change_features = change_features.to(device)
            
            optimizer.zero_grad()

            # Training loop with ROCm/HIP Flash Attention context
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                if use_bfloat16:
                    # bfloat16 - no GradScaler needed
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = transformer(src_data, tgt_data[:, :-1], change_features)
                        ce_loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
                        len_loss = length_regularization_loss(output, tgt_data, tgt_vocab.stoi["<EOS>"], min_length=5, lambda_len=0.01)
                        loss = ce_loss + len_loss
                        loss = torch.clamp(loss, min=-1e6, max=1e6)
                    loss.backward()
                else:
                    # float16 - use GradScaler
                    with autocast("cuda" if device.type == 'cuda' else "cpu"):
                        output = transformer(src_data, tgt_data[:, :-1], change_features)
                        ce_loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
                        len_loss = length_regularization_loss(output, tgt_data, tgt_vocab.stoi["<EOS>"], min_length=5, lambda_len=0.01)
                        loss = ce_loss + len_loss
                        loss = torch.clamp(loss, min=-1e6, max=1e6)
                    scaler.scale(loss).backward()
            
            # Gradient clipping (prevents exploding gradients)
            if use_bfloat16:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()  # Noam schedule steps every batch
            global_step += 1
            total_train_loss += loss.item()
            current_avg_loss = total_train_loss / (batch_idx + 1)
            pbar.set_postfix({'batch_loss': f'{loss.item():.4f}', 'avg_loss': f'{current_avg_loss:.4f}'})
            
            # Log training metrics to CSV
            gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, global_step, loss.item(), '', optimizer.param_groups[0]['lr'], gpu_mem])

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"\033[92m✓ Epoch {epoch+1}: Train loss: {avg_train_loss:.4f}\033[0m", file=sys.stderr)

        # Validation loop
        transformer.eval()
        total_val_loss = 0
        print(f"\033[96m▶️  Starting validation batch iteration for epoch {epoch+1}\033[0m", file=sys.stderr)
        with torch.no_grad():
            pbar = tqdm(val_dataloader, 
                        desc=f"Epoch {epoch+1}/{num_epochs} Val", 
                        position=0, 
                        leave=True,
                        unit="batch",
                        dynamic_ncols=True)
            for batch_data in pbar:
                # Unpack batch data - now includes change_features
                if len(batch_data) == 3:
                    src_data, tgt_data, change_features = batch_data
                else:
                    # Backward compatibility with old 2-tuple format
                    src_data, tgt_data = batch_data
                    change_features = None
                
                src_data = src_data.to(device)
                tgt_data = tgt_data.to(device)
                if change_features is not None:
                    change_features = change_features.to(device)
                
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    if use_bfloat16:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            output = transformer(src_data, tgt_data[:, :-1], change_features)
                            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
                    else:
                        with autocast("cuda"):
                            output = transformer(src_data, tgt_data[:, :-1], change_features)
                            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
                total_val_loss += loss.item()
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"\033[92m✓ Epoch {epoch+1}: Validation loss: {avg_val_loss:.4f}\033[0m", file=sys.stderr)
        # Noam scheduler doesn't need epoch-level stepping
        
        # Update validation loss in CSV log (update last row for this epoch)
        import pandas as pd
        df_log = pd.read_csv(log_path)
        epoch_mask = df_log['epoch'] == epoch + 1
        if epoch_mask.any():
            df_log.loc[epoch_mask, 'val_loss'] = avg_val_loss
            df_log.to_csv(log_path, index=False)

        # Save checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"transformer_best.pth")
            print(f"\033[96m💿 Saving best checkpoint to {checkpoint_path}\033[0m", file=sys.stderr)
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'use_bfloat16': use_bfloat16
            }
            if scaler is not None:
                checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint_dict, checkpoint_path)

        # Save epoch checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"transformer_epoch_{epoch+1}.pth")
        print(f"\033[96m💿 Saving checkpoint to {checkpoint_path}\033[0m", file=sys.stderr)
        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'use_bfloat16': use_bfloat16
        }
        if scaler is not None:
            checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
        torch.save(checkpoint_dict, checkpoint_path)
        print(f"\033[92m✅ Epoch {epoch+1}: Checkpoint saved\033[0m", file=sys.stderr)

        transformer.train()
        print(f"Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("\033[95m🏁 Training loop completed\033[0m", file=sys.stderr)
    print("\033[94m💾 Saving final model state\033[0m", file=sys.stderr)
    torch.save(transformer.state_dict(), "transformer_model.pth")
    print("\033[92m✅ Final model saved to transformer_model.pth\033[0m", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model for commit message generation")
    parser.add_argument("--data-path", default="./traindata.parquet", help="Path to parquet file")
    parser.add_argument("--diff-vocab-path", default="./tokenizer/diff_vocab.pkl", help="Path to diff vocabulary")
    parser.add_argument("--message-vocab-path", default="./tokenizer/message_vocab.pkl", help="Path to message vocabulary")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Maximum sequence length") #avg for diff is 128, for message is 12, max is 120. It used to be 1024
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--resume", default=None, type=str, help="Path to checkpoint to resume training from (e.g., ./checkpoints/transformer_epoch_4.pth)")
    args = parser.parse_args()
    main(args)