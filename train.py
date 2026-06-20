import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import numpy as np
import os
import sys
import argparse
import csv
import gc
from torch.utils.data import DataLoader, random_split
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from model.model import Transformer
from CodeDiffDataset import CodeDiffDataset
from quicksave import QuickSaver
from bucket_sampler import BucketBatchSampler, CurriculumSampler
from ema import ModelEMA
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

def length_regularization_loss(output, tgt_data, eos_idx, pad_idx=0,
                                 min_length=5, max_length=50, lambda_len=0.01):
    """
    Two-sided length regularization:
    - Penalize if model predicts EOS too early (< min_length)
    - Penalize if model never predicts EOS (> max_length) — forces commitment
    """
    # output: (B, T, V), tgt_data: (B, T+1)
    # Log probs for EOS token at each position
    log_probs = torch.nn.functional.log_softmax(output, dim=-1)
    eos_log_probs = log_probs[:, :, eos_idx]   # (B, T)

    T = output.size(1)
    pos_weights = torch.arange(1, T + 1, device=output.device, dtype=torch.float)

    # Penalize high EOS probability early in sequence
    early_penalty = (eos_log_probs[:, :min_length].exp() *
                     (min_length - pos_weights[:min_length]) / min_length).mean()

    return lambda_len * early_penalty.clamp(min=0)

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
    accumulation_steps = args.accumulation_steps
    micro_batch_size = args.micro_batch_size
    print(f"\033[92m📊 Hyperparameters set - d_model: {d_model}, num_epochs: {num_epochs}\033[0m", file=sys.stderr)
    print(f"\033[92m📊 Gradient accumulation: micro_batch={micro_batch_size}, steps={accumulation_steps}, effective_batch={micro_batch_size * accumulation_steps}\033[0m", file=sys.stderr)

    print("\033[94m📁 Creating checkpoint directory\033[0m", file=sys.stderr)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"\033[92m✓ Checkpoint directory ready: {args.checkpoint_dir}\033[0m", file=sys.stderr)

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

    src_vocab_size = max(src_vocab.stoi.values()) + 1
    tgt_vocab_size = max(tgt_vocab.stoi.values()) + 1
    print(f"\033[92m📊 Final vocab sizes - src: {src_vocab_size}, tgt: {tgt_vocab_size}\033[0m", file=sys.stderr)

    print("\033[94m🏗️  Creating memory-mapped datasets\033[0m", file=sys.stderr)
    train_dataset = CodeDiffDataset(
        tokenized_dir=args.tokenized_dir,
        src_vocab=src_vocab,
        split='train',
        train_frac=0.9
    )
    val_dataset = CodeDiffDataset(
        tokenized_dir=args.tokenized_dir,
        src_vocab=src_vocab,
        split='val',
        train_frac=0.9
    )
    print(f"\033[92m✓ Datasets created - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}\033[0m", file=sys.stderr)

    print("\033[94m🔗 Creating DataLoaders with Curriculum Learning\033[0m", file=sys.stderr)
    
    # Initialize start_epoch for curriculum sampler
    start_epoch = 0
    
    # Load pre-computed sequence lengths for bucket batching
    src_lengths = np.load(os.path.join(args.tokenized_dir, "src_lengths.npy"))
    train_lengths = src_lengths[train_dataset.indices]
    print(f"\033[92m✓ Loaded sequence lengths - mean: {train_lengths.mean():.1f}, max: {train_lengths.max()}\033[0m", file=sys.stderr)
    
    # Create curriculum sampler (starts with easy examples, then full dataset)
    train_sampler = CurriculumSampler(
        lengths=train_lengths,
        batch_size=batch_size,
        bucket_size=batch_size * 50,  # Reduced bucket size to save memory
        warmup_epochs=1,  # First epoch uses shortest 50% of data
        current_epoch=start_epoch,
        shuffle=True
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,  # Replaces batch_size + shuffle args
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=max(1, batch_size // 4),   # halve again: 16 instead of 32
        shuffle=False,
        num_workers=0,                          # run in main process, no worker RAM
        pin_memory=False,                       # no point pinning without workers
        # remove persistent_workers and prefetch_factor entirely
    )
    print(f"\033[92m✓ DataLoaders created with bucket batching, batch_size: {batch_size}, train_workers=2, val_workers=0\033[0m", file=sys.stderr)

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

    print("\033[94m📉 Initializing loss criterion with class weights\033[0m", file=sys.stderr)
    
    # Build frequency-based class weights for rare token emphasis
    def build_class_weights(tgt_vocab, tgt_vocab_size, device, alpha=0.5):
        """
        Compute inverse-frequency class weights.
        alpha: exponent — 0.5 is square root smoothing (less extreme than full inverse)
        Tokens with no frequency info get weight 1.0.
        """
        weights = torch.ones(tgt_vocab_size, device=device)

        # If your MsgVocabulary has a word_freqs dict, use it
        if hasattr(tgt_vocab, 'word_freqs') and tgt_vocab.word_freqs:
            total = sum(tgt_vocab.word_freqs.values())
            for word, idx in tgt_vocab.stoi.items():
                freq = tgt_vocab.word_freqs.get(word, 1)
                if idx < tgt_vocab_size:  # guard against any stray IDs
                    weights[idx] = (total / (freq * tgt_vocab_size)) ** alpha

        # Always give zero weight to padding
        weights[tgt_vocab.stoi['<PAD>']] = 0.0
        return weights
    
    class_weights = build_class_weights(tgt_vocab, tgt_vocab_size, device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=tgt_vocab.stoi["<PAD>"],
        label_smoothing=0.1
    )
    print(f"\033[92m✓ CrossEntropyLoss with class weights and label_smoothing=0.1\033[0m", file=sys.stderr)

    print("\033[94m🔄 Initializing optimizer and OneCycleLR scheduler\033[0m", file=sys.stderr)
    
    # Use AdamW with correct decoupled weight decay (Loshchilov & Hutter 2017)
    optimizer = optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,  # Max LR for OneCycleLR
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01  # Standard for transformers
    )
    
    # OneCycleLR scheduler - better for short training runs (3-5 epochs)
    global_step = 0
    total_steps = (len(train_dataloader) * args.num_epochs) // accumulation_steps
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,              # Peak LR
        total_steps=total_steps,
        pct_start=0.05,            # 10% of training for warmup
        anneal_strategy='cos',    # Cosine decay
        div_factor=10.0,          # Initial LR = max_lr / 25
        final_div_factor=1e4      # Final LR = max_lr / 10000
    )
    
    print(f"\033[92m✓ AdamW optimizer and OneCycleLR scheduler initialized (total_steps={total_steps})\033[0m", file=sys.stderr)

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

    # Initialize EMA for better generalization
    ema = ModelEMA(transformer, decay=0.999)
    print(f"\033[92m✓ ModelEMA initialized (decay=0.999)\033[0m", file=sys.stderr)

    # Add new hyperparameters for contrastive learning
    contrastive_weight = 0.1  # Weight for contrastive loss
    temperature = 0.1  # Temperature for contrastive loss

    # Add new hyperparameters for diversity loss
    diversity_weight = 0.01  # Weight for diversity loss
    repetition_penalty = 1.0  # Penalty for repeated tokens

    # Initialize QuickSaver for power-outage proof checkpoints
    saver = QuickSaver(
        checkpoint_dir=args.checkpoint_dir,
        save_every_steps=args.quicksave_every,
        keep_last_n=3
    )

    # Auto-resume from latest checkpoint if requested
    global_step = 0
    best_val_loss = float('inf')
    resume_batch_idx = 0

    if args.auto_resume:
        state = saver.load_latest()
        if state:
            transformer.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            if 'scheduler_state_dict' in state and scheduler:
                scheduler.load_state_dict(state['scheduler_state_dict'])
            if 'scaler_state_dict' in state and scaler:
                scaler.load_state_dict(state['scaler_state_dict'])
            start_epoch = state.get('epoch', 0)
            global_step = state.get('global_step', 0)
            best_val_loss = state.get('best_val_loss', float('inf'))
            resume_batch_idx = state.get('batch_idx', 0)
            
            # Restore saver's internal counter
            saver.restore_state(state.get('saver_last_saved_step', global_step))
            print(f" QuickSaver synced to last_saved_step={saver.last_saved_step}", file=sys.stderr)
            print(f"\033[95m🔥 Resumed from step {global_step}, epoch {start_epoch}\033[0m", file=sys.stderr)
    elif args.resume:
        # Legacy manual resume (deprecated but kept for compatibility)
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
        
        # Update curriculum sampler for this epoch
        train_sampler.set_epoch(epoch)
        
        total_train_loss = 0
        accum_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at start of epoch

        print(f"\033[96m▶️  Starting train batch iteration for epoch {epoch+1}\033[0m", file=sys.stderr)
        pbar = tqdm(train_dataloader, 
                    desc=f"Epoch {epoch+1}/{num_epochs} Train", 
                    position=0, 
                    leave=True,
                    unit="batch",
                    dynamic_ncols=True)
        
        import itertools
        batch_iter = iter(pbar)
        
        # Reset resume skip after the first resumed epoch
        _skip_batches = resume_batch_idx if epoch == start_epoch else 0
        
        if _skip_batches > 0:
            print(f" Fast-forwarding {_skip_batches} batches...", file=sys.stderr)
            consumed = 0
            for _ in itertools.islice(batch_iter, _skip_batches):
                consumed += 1
            print(f" Skipped {consumed} batches", file=sys.stderr)
        
        for batch_idx, batch_data in enumerate(batch_iter, start=_skip_batches):
            
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

            # Training loop with ROCm/HIP Flash Attention context
            from torch.nn.attention import sdpa_kernel, SDPBackend
            
            with sdpa_kernel(SDPBackend.MATH):  # Safe fallback for ROCm
                if use_bfloat16:
                    # bfloat16 - no GradScaler needed
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = transformer(src_data, tgt_data[:, :-1], change_features)
                        ce_loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
                        len_loss = length_regularization_loss(output, tgt_data, tgt_vocab.stoi["<EOS>"], min_length=5, lambda_len=0.01)
                        loss = (ce_loss + len_loss) / accumulation_steps  # Scale loss for accumulation
                        loss = torch.clamp(loss, min=-1e6, max=1e6)
                    loss.backward()
                else:
                    # float16 - use GradScaler
                    with autocast("cuda" if device.type == 'cuda' else "cpu"):
                        output = transformer(src_data, tgt_data[:, :-1], change_features)
                        ce_loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
                        len_loss = length_regularization_loss(output, tgt_data, tgt_vocab.stoi["<EOS>"], min_length=5, lambda_len=0.01)
                        loss = (ce_loss + len_loss) / accumulation_steps  # Scale loss for accumulation
                        loss = torch.clamp(loss, min=-1e6, max=1e6)
                    scaler.scale(loss).backward()
            
            accum_loss += loss.item() * accumulation_steps  # Track unscaled loss for logging

            # Only update weights every `accumulation_steps` micro-batches
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping (prevents exploding gradients)
                if use_bfloat16:
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                
                scheduler.step()  # Noam schedule steps every accumulation step
                global_step += 1
                optimizer.zero_grad()  # Reset gradients after optimizer step
                
                # Update EMA weights
                ema.update(transformer)
                
                # ADD THIS: quicksave every N optimizer steps
                saver.step(
                    global_step=global_step,
                    epoch=epoch,
                    model=transformer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    val_loss=None,           # not known yet mid-epoch
                    best_val_loss=best_val_loss,
                    use_bfloat16=use_bfloat16,
                    batch_idx=batch_idx
                )
                
                total_train_loss += accum_loss
                current_avg_loss = total_train_loss / ((batch_idx // accumulation_steps) + 1)
                pbar.set_postfix({'batch_loss': f'{accum_loss:.4f}', 'avg_loss': f'{current_avg_loss:.4f}'})
                accum_loss = 0.0  # Reset accumulated loss
                
                # Log training metrics to CSV
                gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                with open(log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, global_step, accum_loss * accumulation_steps, '', optimizer.param_groups[0]['lr'], gpu_mem])
        
        # Handle remaining batches if not divisible by accumulation_steps
        if (batch_idx + 1) % accumulation_steps != 0:
            if use_bfloat16:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            
            scheduler.step()
            global_step += 1
            optimizer.zero_grad()
            
            total_train_loss += accum_loss

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"\033[92m✓ Epoch {epoch+1}: Train loss: {avg_train_loss:.4f}\033[0m", file=sys.stderr)

        # === MEMORY CLEANUP BEFORE VALIDATION ===
        # 1. Explicitly free last-batch tensors
        del src_data, tgt_data
        if 'output' in dir(): del output
        if 'loss' in dir(): del loss
        if 'ce_loss' in dir(): del ce_loss
        
        # 2. Kill train workers and flush CUDA cache
        train_dataloader._iterator = None  # kills prefetch queue
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\033[92m✓ Memory cleanup completed before validation\033[0m", file=sys.stderr)

        # Validation loop using EMA model for better generalization
        ema.model.to(device)
        ema.model.eval()
        total_val_loss = 0
        print(f"\033[96m▶️  Starting validation batch iteration for epoch {epoch+1}\033[0m", file=sys.stderr)
        try:
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
                    
                    from torch.nn.attention import sdpa_kernel, SDPBackend
                    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                        if use_bfloat16:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                output = ema.model(src_data, tgt_data[:, :-1], change_features)
                                loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
                        else:
                            with autocast("cuda"):
                                output = ema.model(src_data, tgt_data[:, :-1], change_features)
                                loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
                    total_val_loss += loss.item()
                    pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                    
                    # Free each batch immediately
                    del src_data, tgt_data, output, loss
                    if change_features is not None:
                        del change_features
        finally:
            # Always clean up after val
            torch.cuda.empty_cache()
            gc.collect()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"\033[92m✓ Epoch {epoch+1}: Validation loss: {avg_val_loss:.4f}\033[0m", file=sys.stderr)
        # Noam scheduler doesn't need epoch-level stepping
        
        # Update validation loss in CSV log (append summary row instead of reading entire file)
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, global_step, '', avg_val_loss, 
                             optimizer.param_groups[0]['lr'], 0])

        # Save epoch checkpoint using QuickSaver
        saver.save_epoch(
            epoch=epoch + 1,
            model=transformer,
            optimizer=optimizer,
            val_loss=avg_val_loss,
            best_val_loss=best_val_loss,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
            use_bfloat16=use_bfloat16
        )
        print(f"\033[92m✅ Epoch {epoch+1}: Checkpoint saved\033[0m", file=sys.stderr)

        transformer.train()
        print(f"Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("\033[95m🏁 Training loop completed\033[0m", file=sys.stderr)
    print("\033[94m💾 Saving final model state\033[0m", file=sys.stderr)
    torch.save(transformer.state_dict(), "transformer_model.pth")
    print("\033[92m✅ Final model saved to transformer_model.pth\033[0m", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model for commit message generation")
    parser.add_argument("--data-path", default="./traindata.parquet", help="Path to parquet file (legacy, not used with pretokenized data)")
    parser.add_argument("--tokenized-dir", default="./tokenized_data", help="Directory with pre-tokenized .npy files from pretokenize.py")
    parser.add_argument("--diff-vocab-path", default="./tokenizer/diff_vocab.pkl", help="Path to diff vocabulary")
    parser.add_argument("--message-vocab-path", default="./tokenizer/message_vocab.pkl", help="Path to message vocabulary")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Maximum sequence length") #avg for diff is 128, for message is 12, max is 120. It used to be 1024
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Effective batch size (micro_batch × accumulation_steps)")
    parser.add_argument("--micro-batch-size", type=int, default=16, help="Per-step batch size (memory budget)")
    parser.add_argument("--accumulation-steps", type=int, default=8, help="Accumulate this many micro-batches before optimizer step")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--quicksave-every", type=int, default=300,
                        help="Save checkpoint every N training steps")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Automatically resume from latest checkpoint in --checkpoint-dir")
    parser.add_argument("--resume", default=None, type=str, help="Path to checkpoint to resume training from (legacy)")
    args = parser.parse_args()
    main(args)
