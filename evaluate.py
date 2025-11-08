#!/usr/bin/env python3
"""
evaluate.py - Streamlined evaluation script for commit message generation.
Generates predictions and saves them to CSV with detailed logging.
"""

import torch
import pickle
import pandas as pd
import os
import sys
import argparse
from model.model import Transformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def print_header(text):
    """Print a nice header."""
    print(f"\n\033[95m{'='*70}\033[0m", file=sys.stderr)
    print(f"\033[95m{text}\033[0m", file=sys.stderr)
    print(f"\033[95m{'='*70}\033[0m", file=sys.stderr)

def print_step(text):
    """Print a step message."""
    print(f"\033[94m▶️  {text}\033[0m", file=sys.stderr)

def print_success(text):
    """Print a success message."""
    print(f"\033[92m✓ {text}\033[0m", file=sys.stderr)

def print_info(text):
    """Print an info message."""
    print(f"\033[96m   {text}\033[0m", file=sys.stderr)

def print_warning(text):
    """Print a warning message."""
    print(f"\033[93m⚠️  {text}\033[0m", file=sys.stderr)

def print_error(text):
    """Print an error message."""
    print(f"\033[91m✗ {text}\033[0m", file=sys.stderr)


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
    with open(file_path, 'rb') as f:
        unpickler = PickleModuleRedirector(f)
        return unpickler.load()


def load_vocabularies(diff_vocab_path, message_vocab_path):
    """Load source and target vocabularies."""
    print_step("Loading vocabularies...")
    
    src_vocab = load_pickle_with_redirect(diff_vocab_path)
    print_info(f"Source vocab: {len(src_vocab.stoi):,} tokens")
    
    tgt_vocab = load_pickle_with_redirect(message_vocab_path)
    print_info(f"Target vocab: {len(tgt_vocab.stoi):,} tokens")
    
    print_success("Vocabularies loaded")
    return src_vocab, tgt_vocab


def load_model(checkpoint_path, src_vocab_size, tgt_vocab_size, device, args):
    """Load model from checkpoint."""
    print_step("Loading model...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model first
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout
    ).to(device)
    
    # Check if checkpoint is a dict with metadata or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Training checkpoint format (from epoch saves)
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')
        
        print_info(f"Checkpoint type: Training checkpoint")
        print_info(f"Epoch: {epoch}")
        print_info(f"Validation loss: {val_loss}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state_dict format (from torch.save(model.state_dict(), ...))
        print_info(f"Checkpoint type: Direct state_dict")
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print_info(f"Model parameters: {num_params:,}")
    print_success("Model loaded and ready")
    
    return model


def tokenize_and_pad(text, vocab, max_length):
    """Convert text to padded token IDs."""
    if not text or pd.isna(text):
        text = ""
    
    # Tokenize and numericalize
    ids = vocab.numericalize(str(text))
    
    # Truncate if needed (leave room for SOS and EOS)
    if len(ids) > max_length - 2:
        ids = ids[:max_length - 2]
    
    # Add special tokens
    sos_id = vocab.stoi["<SOS>"]
    eos_id = vocab.stoi["<EOS>"]
    pad_id = vocab.stoi["<PAD>"]
    
    ids = [sos_id] + ids + [eos_id]
    
    # Pad to max_length
    ids = ids + [pad_id] * (max_length - len(ids))
    
    return ids[:max_length]


def beam_search_generate(model, diff_text, src_vocab, tgt_vocab, device,
                         max_seq_length, max_gen_length=100, beam_width=5, 
                         length_penalty=1.0, debug=True):
    """
    Generate commit message using beam search.
    
    Args:
        model: Trained transformer model
        diff_text: Source diff text
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: torch device
        max_seq_length: Maximum sequence length
        max_gen_length: Maximum generation length
        beam_width: Number of beams to keep
        length_penalty: Length penalty factor (higher = prefer longer sequences)
        debug: Print debug information
    
    Returns:
        Generated commit message string
    """
    # Tokenize source
    src_ids = tokenize_and_pad(diff_text, src_vocab, max_seq_length)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    # Special tokens
    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id = tgt_vocab.stoi["<EOS>"]
    pad_id = tgt_vocab.stoi["<PAD>"]
    
    if debug:
        print(f"\n\033[96m{'='*70}\033[0m", file=sys.stderr)
        print(f"\033[96m🔍 BEAM SEARCH - Width={beam_width}\033[0m", file=sys.stderr)
        print(f"\033[96m{'='*70}\033[0m", file=sys.stderr)
    
    # Initialize beams: (sequence, score)
    beams = [([sos_id], 0.0)]
    completed_beams = []
    
    with torch.no_grad():
        for step in range(max_gen_length):
            if debug and step < 3:
                print(f"\n\033[94m📍 Step {step + 1}\033[0m", file=sys.stderr)
                print(f"\033[96m   Active beams: {len(beams)}\033[0m", file=sys.stderr)
            
            candidates = []
            
            for beam_seq, beam_score in beams:
                # Prepare input
                current_tgt = beam_seq + [pad_id] * (max_seq_length - len(beam_seq))
                tgt_tensor = torch.tensor([current_tgt[:max_seq_length]], dtype=torch.long).to(device)
                
                # Forward pass
                output = model(src_tensor, tgt_tensor)
                logits = output[0, len(beam_seq) - 1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Get top-k candidates
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for log_prob, token_id in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                    new_seq = beam_seq + [token_id]
                    new_score = beam_score + log_prob
                    
                    # Apply length penalty: score / (length ** length_penalty)
                    normalized_score = new_score / (len(new_seq) ** length_penalty)
                    
                    # Check if completed
                    if token_id == eos_id or token_id == pad_id:
                        completed_beams.append((new_seq, normalized_score))
                    else:
                        candidates.append((new_seq, new_score, normalized_score))
            
            if not candidates:
                break
            
            # Sort by normalized score and keep top beam_width
            candidates.sort(key=lambda x: x[2], reverse=True)
            beams = [(seq, score) for seq, score, _ in candidates[:beam_width]]
            
            if debug and step < 3:
                print(f"\033[95m   Top 3 beam sequences:\033[0m", file=sys.stderr)
                for i, (seq, score) in enumerate(beams[:3], 1):
                    text = decode_ids(seq[1:], tgt_vocab)  # Skip SOS
                    print(f"\033[95m   {i}. Score={score:.2f} '{text}'\033[0m", file=sys.stderr)
            
            # Early stopping if we have enough completed beams
            if len(completed_beams) >= beam_width:
                break
    
    # Add remaining beams to completed
    for seq, score in beams:
        normalized_score = score / (len(seq) ** length_penalty)
        completed_beams.append((seq, normalized_score))
    
    if not completed_beams:
        if debug:
            print(f"\033[93m⚠️  No completed beams, returning empty\033[0m", file=sys.stderr)
        return ""
    
    # Select best beam
    best_beam = max(completed_beams, key=lambda x: x[1])
    best_seq = best_beam[0]
    
    if debug:
        print(f"\n\033[92m✅ Best beam selected:\033[0m", file=sys.stderr)
        print(f"\033[96m   Score: {best_beam[1]:.4f}\033[0m", file=sys.stderr)
        print(f"\033[96m   Length: {len(best_seq)}\033[0m", file=sys.stderr)
    
    # Decode (skip SOS)
    message = decode_ids(best_seq[1:], tgt_vocab)
    
    if debug:
        print(f"\033[92m   Message: '{message}'\033[0m", file=sys.stderr)
        print(f"\033[96m{'='*70}\033[0m", file=sys.stderr)
    
    return message


def decode_ids(ids, vocab):
    """Convert token IDs back to text."""
    tokens = []
    for token_id in ids:
        token = vocab.itos.get(int(token_id), '<UNK>')
        # Stop at special tokens
        if token in ['<EOS>', '<PAD>']:
            break
        if token not in ['<SOS>', '<UNK>']:
            tokens.append(token)
    
    return "".join(tokens).strip()


def generate_commit_message(model, diff_text, src_vocab, tgt_vocab, device, 
                           max_seq_length, max_gen_length=100, debug=False, interactive=False,
                           use_sampling=False, temperature=1.0, top_p=0.9, top_k=50):
    """
    Generate a commit message from diff text with sampling support.
    
    Args:
        model: Trained transformer model
        diff_text: Source diff text
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: torch device
        max_seq_length: Maximum sequence length for model
        max_gen_length: Maximum length of generated message
        debug: Print debug information
        interactive: Wait for Enter key after each step
        use_sampling: Use sampling instead of greedy decoding
        temperature: Temperature for sampling (higher = more random)
        top_p: Nucleus sampling parameter (0.0-1.0)
        top_k: Top-k sampling parameter (0 = disabled)
    
    Returns:
        Generated commit message string
    """
    # Tokenize source
    src_ids = tokenize_and_pad(diff_text, src_vocab, max_seq_length)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    # Initialize generation
    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id = tgt_vocab.stoi["<EOS>"]
    pad_id = tgt_vocab.stoi["<PAD>"]
    
    if debug:
        print(f"\n\033[96m{'='*70}\033[0m", file=sys.stderr)
        print(f"\033[96m🔍 DEBUG MODE - Generation Step by Step\033[0m", file=sys.stderr)
        print(f"\033[96m{'='*70}\033[0m", file=sys.stderr)
        print(f"\033[93m[INFO] Special tokens: SOS={sos_id}, EOS={eos_id}, PAD={pad_id}\033[0m", file=sys.stderr)
        print(f"\033[93m[INFO] Decoding: {'Sampling' if use_sampling else 'Greedy'}\033[0m", file=sys.stderr)
        if use_sampling:
            print(f"\033[93m[INFO] Temperature={temperature}, top_p={top_p}, top_k={top_k}\033[0m", file=sys.stderr)
        if interactive:
            print(f"\033[95m[INTERACTIVE] Press ENTER to proceed to next step...\033[0m", file=sys.stderr)
    
    tgt_ids = [sos_id]
    generated = []
    generated_text_so_far = ""
    
    with torch.no_grad():
        for step in range(max_gen_length):
            # Prepare target input
            current_tgt = tgt_ids + [pad_id] * (max_seq_length - len(tgt_ids))
            tgt_tensor = torch.tensor([current_tgt[:max_seq_length]], dtype=torch.long).to(device)
            
            if debug:
                print(f"\n\033[94m{'─'*70}\033[0m", file=sys.stderr)
                print(f"\033[94m📍 Step {step + 1}/{max_gen_length}\033[0m", file=sys.stderr)
                print(f"\033[96m   Current position: {len(tgt_ids) - 1}\033[0m", file=sys.stderr)
                print(f"\033[96m   Tokens generated so far: {len(generated)}\033[0m", file=sys.stderr)
                if generated_text_so_far:
                    print(f"\033[92m   Message so far: '{generated_text_so_far}'\033[0m", file=sys.stderr)
            
            # Forward pass
            if debug:
                print(f"\033[93m   🔄 Running forward pass...\033[0m", file=sys.stderr)
            
            output = model(src_tensor, tgt_tensor)
            
            # Get prediction for next token
            logits = output[0, len(tgt_ids) - 1, :]
            
            # Apply temperature scaling
            if use_sampling and temperature != 1.0:
                logits = logits / temperature
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            if debug:
                # Show top 5 predictions
                topk_probs, topk_indices = torch.topk(probs, 5)
                
                print(f"\033[95m   📊 Top 5 Predictions:\033[0m", file=sys.stderr)
                print(f"\033[95m   {'Rank':<6} {'Token ID':<10} {'Token':<20} {'Probability':<12}\033[0m", file=sys.stderr)
                print(f"\033[95m   {'-'*50}\033[0m", file=sys.stderr)
                
                for i, (prob, token_id) in enumerate(zip(topk_probs.tolist(), topk_indices.tolist()), 1):
                    token = tgt_vocab.itos.get(token_id, '<UNK>')
                    
                    # Format token for display
                    if token in ['<PAD>', '<EOS>', '<SOS>', '<UNK>']:
                        token_display = f"\033[91m{token}\033[0m"  # Red for special tokens
                    else:
                        token_display = repr(token) if len(token) <= 15 else repr(token[:15] + "...")
                    
                    print(f"\033[95m   {i:<6} {token_id:<10} {token_display:<28} {prob:.6f}\033[0m", file=sys.stderr)
            
            # Select next token based on decoding strategy
            if use_sampling:
                # Apply top-k filtering
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                    # Zero out probabilities outside top-k
                    probs_filtered = torch.zeros_like(probs)
                    probs_filtered.scatter_(0, top_k_indices, top_k_probs)
                    probs = probs_filtered
                
                # Apply nucleus (top-p) filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                    
                    # Find cutoff index where cumsum exceeds top_p
                    sorted_indices_to_remove = cumsum_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[0] = False
                    
                    # Zero out removed tokens
                    probs_filtered = probs.clone()
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs_filtered[indices_to_remove] = 0
                    probs = probs_filtered
                
                # Renormalize
                probs = probs / probs.sum()
                
                # Sample from distribution
                next_token = torch.multinomial(probs, 1).item()
                
                if debug:
                    print(f"\033[93m   🎲 Sampled token (not argmax)\033[0m", file=sys.stderr)
            else:
                # Greedy decoding
                next_token = torch.argmax(probs).item()
            
            next_token_str = tgt_vocab.itos.get(next_token, '<UNK>')
            
            if debug:
                print(f"\n\033[92m   ✓ Selected token: ID={next_token} Token={repr(next_token_str)}\033[0m", file=sys.stderr)
            
            # Stop conditions
            if next_token == eos_id:
                if debug:
                    print(f"\033[93m   🛑 Hit <EOS> token - stopping generation\033[0m", file=sys.stderr)
                break
            
            if next_token == pad_id:
                if debug:
                    print(f"\033[93m   🛑 Hit <PAD> token - stopping generation\033[0m", file=sys.stderr)
                break
            
            # Check for repetition (stuck in loop)
            if len(generated) >= 5 and all(t == next_token for t in generated[-5:]):
                if debug:
                    print(f"\033[93m   🛑 Stuck in repetition (same token 5 times) - stopping\033[0m", file=sys.stderr)
                break
            
            tgt_ids.append(next_token)
            generated.append(next_token)
            
            # Update generated text so far
            generated_text_so_far = decode_ids(generated, tgt_vocab)
            
            # Interactive mode - wait for user input
            if interactive and debug:
                print(f"\n\033[95m   Press ENTER to continue to next step (or Ctrl+C to skip)...\033[0m", file=sys.stderr)
                try:
                    input()
                except KeyboardInterrupt:
                    print(f"\n\033[93m   ⏭️  Skipping interactive mode for remaining steps\033[0m", file=sys.stderr)
                    interactive = False  # Disable for rest of generation
    
    if debug:
        print(f"\n\033[96m{'='*70}\033[0m", file=sys.stderr)
        print(f"\033[92m✅ Generation Complete!\033[0m", file=sys.stderr)
        print(f"\033[96m   Total tokens generated: {len(generated)}\033[0m", file=sys.stderr)
        print(f"\033[96m   Token IDs: {generated[:30]}{'...' if len(generated) > 30 else ''}\033[0m", file=sys.stderr)
    
    # Decode to text
    message = decode_ids(generated, tgt_vocab)
    
    if debug:
        print(f"\033[92m   Final decoded message: '{message}'\033[0m", file=sys.stderr)
        print(f"\033[96m{'='*70}\033[0m", file=sys.stderr)
    
    return message


def evaluate(args):
    """Main evaluation function."""
    
    print_header("🚀 Commit Message Generation Evaluation")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(f"Device: {device}")
    if torch.cuda.is_available():
        print_info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load vocabularies
    src_vocab, tgt_vocab = load_vocabularies(args.diff_vocab, args.message_vocab)
    
    # Load model
    model = load_model(
        args.checkpoint,
        len(src_vocab.stoi),
        len(tgt_vocab.stoi),
        device,
        args
    )
    
    # Load test data
    print_step("Loading test data...")
    df = pd.read_parquet(args.test_data)
    df['diff_text'] = df['diff_text'].fillna('')
    df['message'] = df['message'].fillna('')
    
    print_info(f"Total samples: {len(df):,}")
    
    if args.num_samples:
        df = df.head(args.num_samples)
        print_warning(f"Evaluating on first {args.num_samples:,} samples only")
    
    print_success("Test data loaded")
    
    # Generate predictions
    print_header("📝 Generating Commit Messages")
    
    results = []
    
    with tqdm(total=len(df), desc="Generating", file=sys.stderr) as pbar:
        for idx, row in df.iterrows():
            diff_text = row.get('diff_text', '')
            original_message = row.get('message', '')
            
            # Handle empty diffs
            if not diff_text or pd.isna(diff_text):
                generated_message = ""
            else:
                try:
                    generated_message = generate_commit_message(
                        model, 
                        str(diff_text),
                        src_vocab,
                        tgt_vocab,
                        device,
                        args.max_seq_length,
                        max_gen_length=args.max_gen_length
                    )
                except Exception as e:
                    print_error(f"Error at row {idx}: {e}")
                    generated_message = "<ERROR>"
            
            # Store result
            results.append({
                'index': idx,
                'original_message': original_message,
                'generated_message': generated_message,
                'diff_text_preview': str(diff_text)[:200] + "..." if len(str(diff_text)) > 200 else str(diff_text)
            })
            
            pbar.update(1)
            
            # Show progress every 100 samples
            if (idx + 1) % 100 == 0:
                pbar.set_postfix({'processed': f"{idx + 1}/{len(df)}"})
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    print_header("💾 Saving Results")
    
    print_step("Saving CSV file...")
    results_df.to_csv(args.output, index=False)
    print_success(f"Saved to: {args.output}")
    print_info(f"Total rows: {len(results_df):,}")
    
    # Show sample results
    print_header("📊 Sample Results")
    
    print_info("First 5 predictions:")
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n  \033[96mSample {i+1}:\033[0m", file=sys.stderr)
        print(f"  \033[93mOriginal:\033[0m {row['original_message']}", file=sys.stderr)
        print(f"  \033[92mGenerated:\033[0m {row['generated_message']}", file=sys.stderr)
    
    # Summary statistics
    print_header("📈 Summary Statistics")
    
    # Count empty generations
    empty_count = sum(1 for msg in results_df['generated_message'] if not msg or msg == "<ERROR>")
    print_info(f"Total samples: {len(results_df):,}")
    print_info(f"Successful generations: {len(results_df) - empty_count:,}")
    if empty_count > 0:
        print_warning(f"Empty/Error generations: {empty_count:,}")
    
    # Average lengths
    orig_lengths = [len(str(msg)) for msg in results_df['original_message'] if msg]
    gen_lengths = [len(str(msg)) for msg in results_df['generated_message'] if msg and msg != "<ERROR>"]
    
    if orig_lengths:
        print_info(f"Avg original message length: {sum(orig_lengths)/len(orig_lengths):.1f} chars")
    if gen_lengths:
        print_info(f"Avg generated message length: {sum(gen_lengths)/len(gen_lengths):.1f} chars")
    
    print_header("✅ Evaluation Complete!")
    print_info(f"Results saved to: {args.output}")
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate transformer model and generate commit messages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--test-data", required=True, 
                       help="Path to test parquet file")
    parser.add_argument("--checkpoint", required=True,
                       help="Path to model checkpoint (.pth file)")
    
    # Vocabulary paths
    parser.add_argument("--diff-vocab", default="./tokenizer/diff_vocab.pkl",
                       help="Path to diff vocabulary")
    parser.add_argument("--message-vocab", default="./tokenizer/message_vocab.pkl",
                       help="Path to message vocabulary")
    
    # Output
    parser.add_argument("--output", default="./predictions.csv",
                       help="Output CSV file path")
    
    # Sampling
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate (None = all)")
    
    # Generation parameters
    parser.add_argument("--max-gen-length", type=int, default=100,
                       help="Maximum length of generated message")
    
    # Model hyperparameters (must match training config)
    parser.add_argument("--d-model", type=int, default=1024,
                       help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4,
                       help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=2048,
                       help="Feed-forward dimension")
    parser.add_argument("--max-seq-length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    
    args = parser.parse_args()
    
    # Run evaluation
    try:
        results = evaluate(args)
    except KeyboardInterrupt:
        print_warning("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
