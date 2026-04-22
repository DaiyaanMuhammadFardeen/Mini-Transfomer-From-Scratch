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


def ngram_repetition_penalty(logits, generated_tokens, n=2, penalty=1.0):
    """
    Apply n-gram repetition penalty to logits during generation.
    
    Args:
        logits: Current logits (vocab_size,) or (batch_size, vocab_size)
        generated_tokens: Previously generated tokens (seq_len,) or (batch_size, seq_len)
        n: N-gram size
        penalty: Penalty factor (higher = stronger penalty)
        
    Returns:
        penalized_logits: Logits with repetition penalty applied
    """
    if generated_tokens.size(-1) < n:
        return logits
    
    # Handle both single sample and batch cases
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        generated_tokens = generated_tokens.unsqueeze(0)
        squeeze_end = True
    else:
        squeeze_end = False
    
    batch_size, vocab_size = logits.shape
    
    for i in range(batch_size):
        # Get last n-1 tokens
        if generated_tokens.size(-1) >= n-1:
            # Handle both 1D and 2D tensor indexing
            if generated_tokens.dim() == 1:
                last_tokens = generated_tokens[-(n-1):].tolist()
            else:
                last_tokens = generated_tokens[i, -(n-1):].tolist()
            
            # Apply penalty to repeated tokens
            for token in last_tokens:
                if 0 <= token < vocab_size:  # Make sure token is valid
                    logits[i, token] -= penalty
    
    return logits.squeeze(0) if squeeze_end else logits


def compute_similarity(text1, text2):
    """
    Compute similarity between two texts using simple word overlap.
    In practice, you might want to use more sophisticated methods like embeddings.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def mmr_rerank(candidates, diff_text, lambda_param=0.7):
    """
    Rerank candidates using Maximum Marginal Relevance.
    
    Args:
        candidates: List of (text, score) tuples
        diff_text: Source diff text
        lambda_param: Trade-off parameter between relevance and diversity (0=diverse, 1=relevant)
        
    Returns:
        Reranked list of candidates
    """
    if len(candidates) <= 1:
        return candidates
    
    # Start with the best candidate by original score
    selected = [candidates[0]]
    remaining = candidates[1:]
    
    while remaining and len(selected) < len(candidates):
        best_candidate = None
        best_mmr_score = -float('inf')
        
        # For each remaining candidate, compute MMR score
        for candidate in remaining:
            candidate_text, original_score = candidate
            
            # Relevance component (similarity to source diff)
            relevance = compute_similarity(diff_text, candidate_text)
            
            # Diversity component (distance to already selected candidates)
            min_distance = float('inf')
            for selected_text, _ in selected:
                distance = 1 - compute_similarity(candidate_text, selected_text)
                min_distance = min(min_distance, distance)
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * min_distance
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate = candidate
        
        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break
    
    return selected


def diversity_beam_search_generate(model, diff_text, src_vocab, tgt_vocab, device,
                                  max_seq_length, max_gen_length=100, beam_width=5, 
                                  length_penalty=1.0, repetition_penalty=1.2, 
                                  ngram_size=2, use_mmr=False, mmr_lambda=0.7,
                                  debug=True):
    """
    Generate commit message using beam search with diversity enhancements.
    
    Args:
        model: Trained transformer model
        diff_text: Source diff text
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: torch device
        max_seq_length: Maximum sequence length
        max_gen_length: Maximum generation length
        beam_width: Number of beams to keep
        length_penalty: Length penalty factor
        repetition_penalty: Penalty for repeated tokens
        ngram_size: N-gram size for repetition penalty
        use_mmr: Whether to use MMR reranking
        mmr_lambda: MMR trade-off parameter
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
        print(f"\033[96m🔍 DIVERSITY-ENHANCED BEAM SEARCH - Width={beam_width}\033[0m", file=sys.stderr)
        print(f"\033[96m{'='*70}\033[0m", file=sys.stderr)
    
    # Initialize beams: (sequence, score)
    beams = [([sos_id], 0.0)]
    completed_beams = []
    
    with torch.no_grad():
        # Get encoder output once
        src_mask, _ = model.generate_mask(src_tensor, src_tensor)
        src_embedded = model.dropout(model.encoder_embedding(src_tensor))
        enc_output = src_embedded
        for enc_layer in model.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
    
    for gen_step in range(max_gen_length):
        candidates = []
        
        with torch.no_grad():
            for seq, score in beams:
                # Prepare decoder input
                tgt_tensor = torch.tensor([seq], dtype=torch.long).to(device)
                
                # Create target mask
                _, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)
                
                # Forward pass
                tgt_embedded = model.dropout(model.decoder_embedding(tgt_tensor))
                dec_output = tgt_embedded
                for dec_layer in model.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
                
                # Get logits for last position
                logits = model.fc(dec_output[:, -1, :])
                
                # Apply repetition penalty
                seq_tensor = torch.tensor(seq, dtype=torch.long).to(device)
                logits = ngram_repetition_penalty(logits, seq_tensor, ngram_size, repetition_penalty)
                
                # Convert to log probabilities
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Get top-k candidates
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    token_id = top_k_indices[0, i].item()
                    token_score = top_k_probs[0, i].item()
                    
                    new_seq = seq + [token_id]
                    new_score = score + token_score
                    
                    # Apply length penalty
                    if length_penalty > 0:
                        new_score = new_score / (len(new_seq) ** length_penalty)
                    
                    candidates.append((new_seq, new_score))
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top beams
        beams = []
        for seq, score in candidates:
            if seq[-1] == eos_id:
                completed_beams.append((seq, score))
            else:
                beams.append((seq, score))
            
            if len(beams) == beam_width:
                break
        
        # If all beams are complete, stop
        if len(beams) == 0:
            break
    
    # Add any remaining beams to completed
    completed_beams.extend(beams)
    
    # Sort completed beams by score
    completed_beams.sort(key=lambda x: x[1], reverse=True)
    
    if debug:
        print(f"\n\033[96mTop {min(3, len(completed_beams))} candidates:\033[0m", file=sys.stderr)
        for i, (seq, score) in enumerate(completed_beams[:3]):
            # Convert to text
            tokens = []
            for idx in seq[1:]:  # Skip SOS
                if idx == eos_id:
                    break
                if idx in tgt_vocab.itos:
                    tokens.append(tgt_vocab.itos[idx])
            
            message = ' '.join(tokens)
            print(f"\033[96m  {i+1}. [{score:.4f}] {message}\033[0m", file=sys.stderr)
    
    # Return best sequence
    if completed_beams:
        # Convert sequences to text
        candidate_texts = []
        for seq, score in completed_beams[:10]:  # Consider top 10 candidates
            tokens = []
            for idx in seq[1:]:  # Skip SOS
                if idx == eos_id:
                    break
                if idx in tgt_vocab.itos:
                    tokens.append(tgt_vocab.itos[idx])
            
            message = ' '.join(tokens)
            candidate_texts.append((message, score))
        
        # Apply MMR reranking if requested
        if use_mmr and len(candidate_texts) > 1:
            candidate_texts = mmr_rerank(candidate_texts, diff_text, mmr_lambda)
        
        best_message, best_score = candidate_texts[0]
        return best_message
    else:
        return ""


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
        # Get encoder output once
        src_mask, _ = model.generate_mask(src_tensor, src_tensor)
        src_embedded = model.dropout(model.encoder_embedding(src_tensor))
        enc_output = src_embedded
        for enc_layer in model.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
    
    for gen_step in range(max_gen_length):
        candidates = []
        
        with torch.no_grad():
            for seq, score in beams:
                # Prepare decoder input
                tgt_tensor = torch.tensor([seq], dtype=torch.long).to(device)
                
                # Create target mask
                _, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)
                
                # Forward pass
                tgt_embedded = model.dropout(model.decoder_embedding(tgt_tensor))
                dec_output = tgt_embedded
                for dec_layer in model.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
                
                # Get logits for last position
                logits = model.fc(dec_output[:, -1, :])
                
                # Convert to log probabilities
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Get top-k candidates
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    token_id = top_k_indices[0, i].item()
                    token_score = top_k_probs[0, i].item()
                    
                    new_seq = seq + [token_id]
                    new_score = score + token_score
                    
                    # Apply length penalty
                    if length_penalty > 0:
                        new_score = new_score / (len(new_seq) ** length_penalty)
                    
                    candidates.append((new_seq, new_score))
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top beams
        beams = []
        for seq, score in candidates:
            if seq[-1] == eos_id:
                completed_beams.append((seq, score))
            else:
                beams.append((seq, score))
            
            if len(beams) == beam_width:
                break
        
        # If all beams are complete, stop
        if len(beams) == 0:
            break
    
    # Add any remaining beams to completed
    completed_beams.extend(beams)
    
    # Sort completed beams by score
    completed_beams.sort(key=lambda x: x[1], reverse=True)
    
    if debug:
        print(f"\n\033[96mTop {min(3, len(completed_beams))} candidates:\033[0m", file=sys.stderr)
        for i, (seq, score) in enumerate(completed_beams[:3]):
            # Convert to text
            tokens = []
            for idx in seq[1:]:  # Skip SOS
                if idx == eos_id:
                    break
                if idx in tgt_vocab.itos:
                    tokens.append(tgt_vocab.itos[idx])
            
            message = ' '.join(tokens)
            print(f"\033[96m  {i+1}. [{score:.4f}] {message}\033[0m", file=sys.stderr)
    
    # Return best sequence
    if completed_beams:
        best_seq, best_score = completed_beams[0]
        # Convert to text
        tokens = []
        for idx in best_seq[1:]:  # Skip SOS
            if idx == eos_id:
                break
            if idx in tgt_vocab.itos:
                tokens.append(tgt_vocab.itos[idx])
        
        return ' '.join(tokens)
    else:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer model for commit message generation")
    parser.add_argument("--data-path", default="./val_data.parquet", help="Path to parquet file")
    parser.add_argument("--diff-vocab-path", default="./tokenizer/diff_vocab.pkl", help="Path to diff vocabulary")
    parser.add_argument("--message-vocab-path", default="./tokenizer/message_vocab.pkl", help="Path to message vocabulary")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file path")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--max-gen-length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam search width")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty factor")
    parser.add_argument("--use-mmr", action="store_true", help="Use Maximum Marginal Relevance reranking")
    parser.add_argument("--mmr-lambda", type=float, default=0.7, help="MMR trade-off parameter (0=diverse, 1=relevant)")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Repetition penalty factor")
    parser.add_argument("--ngram-size", type=int, default=2, help="N-gram size for repetition penalty")
    parser.add_argument("--d-model", type=int, default=1024, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    args = parser.parse_args()
    
    print_header("COMMIT MESSAGE GENERATION EVALUATION")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(f"Device: {device}")
    
    # Load data
    print_step("Loading data...")
    df = pd.read_parquet(args.data_path)
    df = df.sample(frac=0.0002, random_state=42) # random_state for reproducibility
    print_info(f"Loaded {len(df):,} samples")
    print_success("Data loaded")
    
    # Load vocabularies
    src_vocab, tgt_vocab = load_vocabularies(args.diff_vocab_path, args.message_vocab_path)
    src_vocab_size = len(src_vocab.stoi)
    tgt_vocab_size = len(tgt_vocab.stoi)
    
    # Load model
    model = load_model(args.checkpoint, src_vocab_size, tgt_vocab_size, device, args)
    
    # Generate predictions
    print_step("Generating predictions...")
    predictions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        diff_text = row['diff_text'] if not pd.isna(row['diff_text']) else ""
        
        try:
            if args.use_mmr or args.repetition_penalty != 1.0:
                pred = diversity_beam_search_generate(
                    model, diff_text, src_vocab, tgt_vocab, device,
                    args.max_seq_length, args.max_gen_length, args.beam_width,
                    args.length_penalty, args.repetition_penalty, args.ngram_size,
                    args.use_mmr, args.mmr_lambda, debug=False
                )
            else:
                pred = beam_search_generate(
                    model, diff_text, src_vocab, tgt_vocab, device,
                    args.max_seq_length, args.max_gen_length, args.beam_width,
                    args.length_penalty, debug=False
                )
            
            predictions.append(pred)
        except Exception as e:
            print_warning(f"Error generating for row {idx}: {e}")
            predictions.append("")

    # Add predictions to dataframe
    df['predicted_message'] = predictions
    
    # Save to CSV
    print_step("Saving predictions...")
    df.to_csv(args.output, index=False)
    print_success(f"Predictions saved to {args.output}")
    
    # Print sample predictions
    print_step("Sample predictions:")
    for i in range(min(5, len(df))):
        actual = df.iloc[i]['message'] if not pd.isna(df.iloc[i]['message']) else ""
        predicted = df.iloc[i]['predicted_message']
        print(f"  Actual:    {actual}")
        print(f"  Predicted: {predicted}")
        print()

    print_header("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
