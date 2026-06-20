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

# Import search algorithms for benchmark evaluation
from search_algorithms import (
    greedy_search,
    beam_search,
    top_k_sampling,
    top_p_sampling,
    diverse_beam_search,
    contrastive_search,
    typical_sampling,
    min_p_sampling,
    temperature_sampling,
)

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


def batch_greedy_generate(model, diff_texts, src_vocab, tgt_vocab, device,
                           max_seq_length, max_gen_length=50, batch_size=32):
    """
    Generate commit messages for a batch of diffs in parallel.
    Fully utilizes GPU by running encoder + decoder on whole batches.

    Args:
        model:           Trained transformer
        diff_texts:      List of diff strings (length = batch_size)
        src_vocab:       Source vocabulary
        tgt_vocab:       Target vocabulary
        device:          torch.device
        max_seq_length:  Encoder input length
        max_gen_length:  Max tokens to generate
        batch_size:      How many samples per GPU batch

    Returns:
        List of generated strings (same length as diff_texts)
    """
    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id  = tgt_vocab.stoi["<EOS>"]
    pad_id  = tgt_vocab.stoi["<PAD>"]

    all_predictions = []

    for batch_start in range(0, len(diff_texts), batch_size):
        batch_texts = diff_texts[batch_start : batch_start + batch_size]
        B = len(batch_texts)

        # Tokenise & pad all sources
        src_ids_list = [
            tokenize_and_pad(t, src_vocab, max_seq_length) for t in batch_texts
        ]
        src_tensor = torch.tensor(src_ids_list, dtype=torch.long).to(device)  # (B, S)

        # Change-type features
        change_feats = torch.stack([
            extract_change_features(ids, src_vocab) for ids in src_ids_list
        ]).to(device)  # (B, 6)

        with torch.no_grad():
            # Encode once
            src_mask, _ = model.generate_mask(src_tensor, src_tensor)
            src_emb     = model.dropout(model.encoder_embedding(src_tensor, change_feats))
            enc_output  = src_emb
            for layer in model.encoder_layers:
                enc_output = layer(enc_output, src_mask)
            enc_output = model.encoder_norm(enc_output)          # (B, S, D)

            # Decode step-by-step for all samples in parallel
            # tgt_seqs: current decoder input for each sample  (B, T)
            tgt_seqs  = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
            done_mask = torch.zeros(B, dtype=torch.bool, device=device)  # True when EOS hit

            for _ in range(max_gen_length):
                _, tgt_mask = model.generate_mask(src_tensor, tgt_seqs)

                tgt_emb    = model.dropout(model.decoder_embedding(tgt_seqs))
                dec_output = tgt_emb
                for layer in model.decoder_layers:
                    dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
                dec_output = model.decoder_norm(dec_output)

                # Greedy: argmax over last token position
                logits     = model.fc(dec_output[:, -1, :])   # (B, V)
                next_token = logits.argmax(dim=-1)              # (B,)

                # Force PAD for already-finished sequences
                next_token = next_token.masked_fill(done_mask, pad_id)

                tgt_seqs  = torch.cat([tgt_seqs, next_token.unsqueeze(1)], dim=1)
                done_mask = done_mask | (next_token == eos_id)

                if done_mask.all():
                    break

        # Decode token IDs -> strings
        for i in range(B):
            tokens = []
            for tok_id in tgt_seqs[i, 1:].tolist():   # skip SOS
                if tok_id == eos_id:
                    break
                if tok_id == pad_id:
                    continue
                if tok_id in tgt_vocab.itos:
                    tokens.append(tgt_vocab.itos[tok_id])
            all_predictions.append(" ".join(tokens))

    return all_predictions


def load_model(checkpoint_path, src_vocab_size, tgt_vocab_size, device, args):
    """Load model from checkpoint."""
    print_step("Loading model...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if checkpoint has vocab size info or adjust based on checkpoint shape
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Extract actual vocab sizes from checkpoint if they differ
    if 'decoder_embedding.weight' in state_dict:
        checkpoint_tgt_vocab_size = state_dict['decoder_embedding.weight'].shape[0]
        if checkpoint_tgt_vocab_size != tgt_vocab_size:
            print_info(f"Adjusting target vocab size: {tgt_vocab_size} -> {checkpoint_tgt_vocab_size}")
            tgt_vocab_size = checkpoint_tgt_vocab_size
    
    if 'encoder_embedding.weight' in state_dict:
        checkpoint_src_vocab_size = state_dict['encoder_embedding.weight'].shape[0]
        if checkpoint_src_vocab_size != src_vocab_size:
            print_info(f"Adjusting source vocab size: {src_vocab_size} -> {checkpoint_src_vocab_size}")
            src_vocab_size = checkpoint_src_vocab_size
    
    # Build model with corrected vocab sizes
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
    
    # Compile the model (PyTorch 2.0+) — gives 20–40% speedup on modern GPUs
    if hasattr(torch, 'compile'):
        print_info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
        print_success("Model compiled")
    
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


def extract_change_features(token_ids, src_vocab):
    """
    Extract 6-dimensional change-type features from token IDs.
    Matches the logic in CodeDiffDataset._extract_change_features.
    
    Args:
        token_ids: List or tensor of token IDs
        src_vocab: Source vocabulary with stoi mapping
        
    Returns:
        torch.Tensor of shape (6,) with binary flags
    """
    CHANGE_TYPE_TAGS = ['<ADD>', '<REMOVE>', '<MODIFY>',
                        '<COMMENT_ADD>', '<COMMENT_REMOVE>', '<COMMENT_MODIFY>']
    
    features = torch.zeros(6, dtype=torch.float32)
    
    # Convert to tensor if needed
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids)
    
    for i, tag in enumerate(CHANGE_TYPE_TAGS):
        tag_id = src_vocab.stoi.get(tag, -1)
        if tag_id != -1 and (token_ids == tag_id).any():
            features[i] = 1.0
    
    return features


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
    
    # Extract change features
    change_features = extract_change_features(src_ids, src_vocab).unsqueeze(0).to(device)
    
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
        src_embedded = model.dropout(model.encoder_embedding(src_tensor, change_features))
        enc_output = src_embedded
        for enc_layer in model.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        # Apply final encoder norm
        enc_output = model.encoder_norm(enc_output)
    
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
                # Apply final decoder norm
                dec_output = model.decoder_norm(dec_output)
                
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
    Generate commit message using beam search with KV-caching.
    
    Args:
        model: Trained transformer model
        diff_text: Source diff text
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: torch device
        max_seq_length: Maximum sequence length
        max_gen_length: Maximum generation length
        beam_width: Number of beams to keep
        length_penalty: Length penalty factor (Wu et al. 2016)
        debug: Print debug information
    
    Returns:
        Generated commit message string
    """
    def length_penalty_score(score, length, alpha=1.0):
        """Wu et al. (2016) length penalty - apply only to completed sequences."""
        return score / ((5 + length) ** alpha / (5 + 1) ** alpha)
    
    # Tokenize source
    src_ids = tokenize_and_pad(diff_text, src_vocab, max_seq_length)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    # Extract change features
    change_features = extract_change_features(src_ids, src_vocab).unsqueeze(0).to(device)
    
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
        src_embedded = model.encoder_embedding(src_tensor, change_features)
        src_embedded = model.dropout(src_embedded)
        enc_output = src_embedded
        for enc_layer in model.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        # Apply final encoder norm
        enc_output = model.encoder_norm(enc_output)
    
    for gen_step in range(max_gen_length):
        candidates = []
        
        with torch.no_grad():
            for seq, score in beams:
                # Prepare decoder input - full sequence up to now
                tgt_tensor = torch.tensor([seq], dtype=torch.long).to(device)
                
                # Create target mask
                _, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)
                
                # Forward pass through decoder
                batch_size, tgt_seq_len = tgt_tensor.shape
                tgt_embedded = model.decoder_embedding(tgt_tensor)
                dec_output = model.dropout(tgt_embedded)
                
                for dec_layer in model.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
                # Apply final decoder norm
                dec_output = model.decoder_norm(dec_output)
                
                # Get logits for last position only
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
                    # NO length penalty during expansion - applied at the end
                    
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
    
    # Apply Wu et al. (2016) length penalty ONLY to completed sequences
    completed_beams = [(seq, length_penalty_score(score, len(seq), alpha=length_penalty)) 
                       for seq, score in completed_beams]
    
    # Sort completed beams by penalized score
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
        
        return " ".join(tokens)
    else:
        return ""


# ============================================================================
# SEARCH ALGORITHM REGISTRY FOR BENCHMARK EVALUATION
# ============================================================================

ALGORITHM_REGISTRY = {
    "greedy":            lambda m, d, sv, tv, dev, msl: greedy_search(m, d, sv, tv, dev, msl, max_gen_length=50),
    "beam_search_k5":    lambda m, d, sv, tv, dev, msl: beam_search(m, d, sv, tv, dev, msl, max_gen_length=50, beam_width=5, length_penalty=1.0),
    "beam_search_k3":    lambda m, d, sv, tv, dev, msl: beam_search(m, d, sv, tv, dev, msl, max_gen_length=50, beam_width=3, length_penalty=1.0),
    "top_k_50":          lambda m, d, sv, tv, dev, msl: top_k_sampling(m, d, sv, tv, dev, msl, max_gen_length=50, k=50, temperature=0.8),
    "top_k_20":          lambda m, d, sv, tv, dev, msl: top_k_sampling(m, d, sv, tv, dev, msl, max_gen_length=50, k=20, temperature=0.8),
    "top_p_0.9":         lambda m, d, sv, tv, dev, msl: top_p_sampling(m, d, sv, tv, dev, msl, max_gen_length=50, p=0.9, temperature=0.8),
    "top_p_0.95":        lambda m, d, sv, tv, dev, msl: top_p_sampling(m, d, sv, tv, dev, msl, max_gen_length=50, p=0.95, temperature=0.8),
    "temperature_0.7":   lambda m, d, sv, tv, dev, msl: temperature_sampling(m, d, sv, tv, dev, msl, max_gen_length=50, temperature=0.7),
    "temperature_1.0":   lambda m, d, sv, tv, dev, msl: temperature_sampling(m, d, sv, tv, dev, msl, max_gen_length=50, temperature=1.0),
    "diverse_beam":      lambda m, d, sv, tv, dev, msl: diverse_beam_search(m, d, sv, tv, dev, msl, max_gen_length=50, beam_width=6, group_beam_width=2, diversity_penalty=0.5),
    "contrastive_approx":lambda m, d, sv, tv, dev, msl: contrastive_search(m, d, sv, tv, dev, msl, max_gen_length=50, penalty_alpha=0.6, top_k=4),
    "typical_tau0.9":    lambda m, d, sv, tv, dev, msl: typical_sampling(m, d, sv, tv, dev, msl, max_gen_length=50, tau=0.9, temperature=0.8),
    "min_p_0.05":        lambda m, d, sv, tv, dev, msl: min_p_sampling(m, d, sv, tv, dev, msl, max_gen_length=50, min_p=0.05, temperature=0.8),
}


# ============================================================================
# EVALUATION METRICS - SOTA for Commit Message Generation
# ============================================================================

def compute_bleu(references: list, hypotheses: list) -> float:
    """
    Compute corpus BLEU-4 score.
    Used in: ATOM, CommitGen, NNGen, RACE, FIRA — essentially every paper.
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
    
    Returns:
        BLEU-4 score (0-100)
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        
        smoothing = SmoothingFunction().method3
        refs_tokenized = [[ref.lower().split()] for ref in references]
        hyps_tokenized = [hyp.lower().split() for hyp in hypotheses]
        
        bleu_score = corpus_bleu(
            refs_tokenized, 
            hyps_tokenized,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing
        ) * 100
        
        return bleu_score
    except Exception as e:
        print_warning(f"BLEU computation failed: {e}")
        return 0.0


def compute_meteor(references: list, hypotheses: list) -> float:
    """
    Compute METEOR score (synonym-aware).
    Used in: RLGC (2023), FIRA (2021), recent papers prefer METEOR over BLEU.
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
    
    Returns:
        METEOR score (0-100)
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        
        # Download required data if not present
        try:
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            print_info("Downloading WordNet for METEOR...")
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            print_success("WordNet downloaded successfully")
        
        scores = []
        for ref, hyp in zip(references, hypotheses):
            score = meteor_score([ref.lower().split()], hyp.lower().split())
            scores.append(score)
        
        return sum(scores) / len(scores) * 100 if scores else 0.0
    except Exception as e:
        print_warning(f"METEOR computation failed: {e}")
        return 0.0


def compute_rouge_l(references: list, hypotheses: list) -> dict:
    """
    Compute ROUGE-L (Longest Common Subsequence F1).
    Used in: FIRA, RLGC, CMG-LSTM (2019), NNGen.
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
    
    Returns:
        Dictionary with rouge_l_f, rouge_l_p, rouge_l_r
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        
        avg_f = sum(s['rougeL'].fmeasure for s in scores) / len(scores) * 100 if scores else 0.0
        avg_p = sum(s['rougeL'].precision for s in scores) / len(scores) * 100 if scores else 0.0
        avg_r = sum(s['rougeL'].recall for s in scores) / len(scores) * 100 if scores else 0.0
        
        return {
            'rouge_l_f': avg_f,
            'rouge_l_p': avg_p,
            'rouge_l_r': avg_r
        }
    except Exception as e:
        print_warning(f"ROUGE-L computation failed: {e}")
        return {'rouge_l_f': 0.0, 'rouge_l_p': 0.0, 'rouge_l_r': 0.0}


def compute_cider(references: list, hypotheses: list) -> float:
    """
    Compute CIDEr score (Consensus-based Image Description Evaluation).
    Penalizes common/generic outputs more than BLEU.
    Used in: Recent NLP generation papers.
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
    
    Returns:
        CIDEr score (0-100)
    """
    try:
        from pycocoevalcap.cider.cider import Cider
        
        gts = {i: [{'caption': r}] for i, r in enumerate(references)}
        res = {i: [{'caption': h}] for i, h in enumerate(hypotheses)}
        
        scorer = Cider()
        score, _ = scorer.compute_score(gts, res)
        
        return score * 100
    except Exception as e:
        print_warning(f"CIDEr computation failed: {e}")
        return 0.0


def compute_exact_match(references: list, hypotheses: list) -> float:
    """
    Compute Exact Match rate.
    Used in: NNGen (2018), RACE (2021).
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
    
    Returns:
        Exact match percentage (0-100)
    """
    matches = sum(1 for r, h in zip(references, hypotheses) 
                  if r.strip().lower() == h.strip().lower())
    return matches / len(references) * 100 if references else 0.0


def compute_token_accuracy(references: list, hypotheses: list) -> float:
    """
    Compute token-level accuracy (precision).
    Used in: NNGen (2018), RACE (2021).
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
    
    Returns:
        Token accuracy percentage (0-100)
    """
    total_correct = total_tokens = 0
    for ref, hyp in zip(references, hypotheses):
        ref_toks = ref.lower().split()
        hyp_toks = hyp.lower().split()
        min_len = min(len(ref_toks), len(hyp_toks))
        correct = sum(r == h for r, h in zip(ref_toks[:min_len], hyp_toks[:min_len]))
        total_correct += correct
        total_tokens += len(ref_toks)
    return total_correct / max(total_tokens, 1) * 100


def compute_bert_score(references: list, hypotheses: list) -> dict:
    """
    Compute BERTScore (semantic similarity using contextual embeddings).
    Used in: RLGC (2023), recent papers that go beyond n-gram overlap.
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
    
    Returns:
        Dictionary with bertscore_p, bertscore_r, bertscore_f1
    """
    try:
        from bert_score import score as bert_score_func
        
        P, R, F1 = bert_score_func(hypotheses, references, lang='en', verbose=False)
        return {
            'bertscore_p': P.mean().item() * 100,
            'bertscore_r': R.mean().item() * 100,
            'bertscore_f1': F1.mean().item() * 100
        }
    except Exception as e:
        print_warning(f"BERTScore computation failed: {e}")
        return {'bertscore_p': 0.0, 'bertscore_r': 0.0, 'bertscore_f1': 0.0}


def compute_all_metrics(references: list, hypotheses: list) -> dict:
    """
    Compute all evaluation metrics at once.
    
    Args:
        references: List of ground truth commit messages
        hypotheses: List of generated commit messages
    
    Returns:
        Dictionary with all metric scores
    """
    print_step("Computing evaluation metrics...")
    
    # Filter out empty predictions
    valid_pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses) if hyp.strip()]
    if not valid_pairs:
        print_warning("No valid predictions to evaluate")
        return {}
    
    valid_refs, valid_hyps = zip(*valid_pairs)
    valid_refs = list(valid_refs)
    valid_hyps = list(valid_hyps)
    
    print_info(f"Evaluating {len(valid_hyps)} valid predictions (skipped {len(references) - len(valid_hyps)} empty)")
    
    metrics = {}
    metrics['n_samples'] = len(valid_hyps)
    metrics['n_empty_predictions'] = len(references) - len(valid_hyps)
    
    # BLEU-4
    print_info("Computing BLEU-4...")
    metrics['bleu_4'] = compute_bleu(valid_refs, valid_hyps)
    print_info(f"  BLEU-4: {metrics['bleu_4']:.2f}")
    
    # METEOR
    print_info("Computing METEOR...")
    metrics['meteor'] = compute_meteor(valid_refs, valid_hyps)
    print_info(f"  METEOR: {metrics['meteor']:.2f}")
    
    # ROUGE-L
    print_info("Computing ROUGE-L...")
    rouge_scores = compute_rouge_l(valid_refs, valid_hyps)
    metrics.update(rouge_scores)
    print_info(f"  ROUGE-L F1: {metrics['rouge_l_f']:.2f}, P: {metrics['rouge_l_p']:.2f}, R: {metrics['rouge_l_r']:.2f}")
    
    # CIDEr
    print_info("Computing CIDEr...")
    metrics['cider'] = compute_cider(valid_refs, valid_hyps)
    print_info(f"  CIDEr: {metrics['cider']:.2f}")
    
    # Exact Match
    print_info("Computing Exact Match...")
    metrics['exact_match'] = compute_exact_match(valid_refs, valid_hyps)
    print_info(f"  Exact Match: {metrics['exact_match']:.2f}%")
    
    # Token Accuracy
    print_info("Computing Token Accuracy...")
    metrics['token_accuracy'] = compute_token_accuracy(valid_refs, valid_hyps)
    print_info(f"  Token Accuracy: {metrics['token_accuracy']:.2f}%")
    
    # BERTScore (optional - requires bert-score package)
    print_info("Computing BERTScore (may take a while)...")
    bert_scores = compute_bert_score(valid_refs, valid_hyps)
    metrics.update(bert_scores)
    print_info(f"  BERTScore F1: {metrics['bertscore_f1']:.2f}, P: {metrics['bertscore_p']:.2f}, R: {metrics['bertscore_r']:.2f}")
    
    print_success("All metrics computed")
    
    return metrics


def save_metrics_report(metrics: dict, output_path: str):
    """
    Save metrics to a readable report file.
    
    Args:
        metrics: Dictionary of metric scores
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMMIT MESSAGE GENERATION - EVALUATION METRICS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Samples evaluated: {metrics.get('n_samples', 0)}\n")
        f.write(f"Empty predictions skipped: {metrics.get('n_empty_predictions', 0)}\n\n")
        
        f.write("N-GRAM OVERLAP METRICS:\n")
        f.write("-"*70 + "\n")
        f.write(f"BLEU-4:          {metrics.get('bleu_4', 0):8.2f}\n")
        f.write(f"METEOR:          {metrics.get('meteor', 0):8.2f}\n\n")
        
        f.write("LONGEST COMMON SUBSEQUENCE:\n")
        f.write("-"*70 + "\n")
        f.write(f"ROUGE-L F1:      {metrics.get('rouge_l_f', 0):8.2f}\n")
        f.write(f"ROUGE-L Precision: {metrics.get('rouge_l_p', 0):8.2f}\n")
        f.write(f"ROUGE-L Recall:    {metrics.get('rouge_l_r', 0):8.2f}\n\n")
        
        f.write("CONSENSUS-BASED METRIC:\n")
        f.write("-"*70 + "\n")
        f.write(f"CIDEr:           {metrics.get('cider', 0):8.2f}\n\n")
        
        f.write("EXACT MATCH & TOKEN ACCURACY:\n")
        f.write("-"*70 + "\n")
        f.write(f"Exact Match:     {metrics.get('exact_match', 0):8.2f}\n")
        f.write(f"Token Accuracy:  {metrics.get('token_accuracy', 0):8.2f}\n\n")
        
        f.write("SEMANTIC SIMILARITY (BERTScore):\n")
        f.write("-"*70 + "\n")
        f.write(f"BERTScore F1:    {metrics.get('bertscore_f1', 0):8.2f}\n")
        f.write(f"BERTScore Precision: {metrics.get('bertscore_p', 0):8.2f}\n")
        f.write(f"BERTScore Recall:    {metrics.get('bertscore_r', 0):8.2f}\n\n")
        
        # Comparison table for thesis
        f.write("="*70 + "\n")
        f.write("THESIS COMPARISON TABLE\n")
        f.write("="*70 + "\n")
        f.write(f"{'Metric':<25} {'Your Model':>15} {'ATOM (2020)':>15} {'NNGen (2018)':>15}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'BLEU-4':<25} {metrics.get('bleu_4', 0):>14.2f} {'~30.2':>15} {'~21.3':>15}\n")
        f.write(f"{'METEOR':<25} {metrics.get('meteor', 0):>14.2f} {'~27.1':>15} {'~18.4':>15}\n")
        f.write(f"{'ROUGE-L':<25} {metrics.get('rouge_l_f', 0):>14.2f} {'~35.4':>15} {'~25.1':>15}\n")
        f.write(f"{'Exact Match':<25} {metrics.get('exact_match', 0):>14.2f} {'~15.8':>15} {'~12.1':>15}\n")
        f.write("="*70 + "\n")
        f.write("Note: Reference values are approximate from published papers.\n")
        f.write("Exact figures depend on dataset and preprocessing choices.\n\n")
        
        f.write("="*70 + "\n")
        f.write("INTERPRETATION GUIDE:\n")
        f.write("-"*70 + "\n")
        f.write("BLEU-4 > 20:     Meaningful overlap with human messages\n")
        f.write("BLEU-4 > 28:     State-of-the-art performance (ATOM level)\n")
        f.write("METEOR > 25:     Good semantic similarity\n")
        f.write("ROUGE-L F1 > 30: Strong structural overlap\n")
        f.write("Exact Match > 10: Excellent exact reproduction\n")
        f.write("Token Accuracy > 40: Good token-level precision\n")
        f.write("BERTScore F1 > 85: High semantic similarity\n")
        f.write("CIDEr > 50:      High consensus with human references\n")
        f.write("="*70 + "\n")
    
    print_success(f"Metrics report saved to {output_path}")


def run_benchmark(args):
    """
    Run comprehensive benchmark across all search algorithms.
    Produces two CSV files: predictions_all_algorithms.csv and benchmark_summary.csv
    """
    print_header("COMPREHENSIVE SEARCH ALGORITHM BENCHMARK")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(f"Device: {device}")
    
    # Step 1: Load data, vocabularies, model
    print_step("Loading evaluation data...")
    df = pd.read_parquet(args.data_path)
    # Use larger sample for reliable statistics (fixed n=500)
    df = df.sample(n=250, random_state=42)
    print_info(f"Loaded {len(df):,} samples for benchmark")
    print_success("Data loaded")
    
    # Load vocabularies
    src_vocab, tgt_vocab = load_vocabularies(args.diff_vocab_path, args.message_vocab_path)
    src_vocab_size = len(src_vocab.stoi)
    tgt_vocab_size = len(tgt_vocab.stoi)
    
    # Load model
    model = load_model(args.checkpoint, src_vocab_size, tgt_vocab_size, device, args)
    
    max_seq_length = args.max_seq_length
    
    # Step 3: Outer loop over samples, inner loop over algorithms
    print_step(f"Running benchmark across {len(ALGORITHM_REGISTRY)} algorithms...")
    rows = []
    
    total_iterations = len(df) * len(ALGORITHM_REGISTRY)
    iteration_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking samples"):
        diff_text = row['diff_text'] if not pd.isna(row['diff_text']) else ""
        ref_message = row['message'] if not pd.isna(row['message']) else ""
        
        for algo_name, algo_fn in ALGORITHM_REGISTRY.items():
            iteration_count += 1
            
            try:
                gen_message, gen_time = algo_fn(
                    model, diff_text, src_vocab, tgt_vocab, device, max_seq_length
                )
            except Exception as e:
                print_warning(f"Error with {algo_name} on sample {idx}: {e}")
                gen_message, gen_time = "", 0.0
            
            # Per-sample metrics (sentence-level)
            if gen_message.strip():
                bleu = compute_bleu([ref_message], [gen_message])
                meteor = compute_meteor([ref_message], [gen_message])
                rouge = compute_rouge_l([ref_message], [gen_message])
                exact = int(ref_message.strip().lower() == gen_message.strip().lower())
            else:
                bleu = 0.0
                meteor = 0.0
                rouge = {"rouge_l_f": 0.0}
                exact = 0
            
            rows.append({
                "sample_id":         idx,
                "diff_text":         diff_text[:200],
                "reference_message": ref_message,
                "algorithm":         algo_name,
                "generated_message": gen_message,
                "bleu_4":            bleu,
                "meteor":            meteor,
                "rouge_l_f":         rouge["rouge_l_f"],
                "exact_match":       exact,
                "gen_time_ms":       gen_time * 1000,  # Convert to milliseconds
                "output_length":     len(gen_message.split()) if gen_message else 0,
            })
    
    # Step 4: Save predictions CSV
    print_step("Saving detailed predictions...")
    df_pred = pd.DataFrame(rows)
    df_pred.to_csv("predictions_all_algorithms.csv", index=False)
    print_success(f"Saved {len(df_pred):,} prediction rows to predictions_all_algorithms.csv")
    
    # Step 5: Compute corpus-level BLEU and aggregate statistics per algorithm
    print_step("Computing aggregate statistics...")
    summary_rows = []
    
    for algo_name, group in df_pred.groupby("algorithm"):
        refs = group["reference_message"].tolist()
        hyps = group["generated_message"].tolist()
        
        # Corpus-level BLEU (not averaged sentence BLEU)
        corpus_bleu_score = compute_bleu(refs, hyps)
        
        # Mean metrics
        mean_meteor = group["meteor"].mean()
        mean_rouge = group["rouge_l_f"].mean()
        exact_match_pct = group["exact_match"].mean() * 100
        
        # Generation time statistics
        times = group["gen_time_ms"]
        mean_gen_time = times.mean()
        p50_gen_time = times.quantile(0.50)
        p95_gen_time = times.quantile(0.95)
        
        # Output length and empty count
        mean_output_len = group["output_length"].mean()
        n_empty = int((group["generated_message"].str.strip() == "").sum())
        n_samples = len(group)
        
        summary_rows.append({
            "algorithm":        algo_name,
            "corpus_bleu_4":    corpus_bleu_score,
            "mean_meteor":      mean_meteor,
            "mean_rouge_l_f":   mean_rouge,
            "exact_match_pct":  exact_match_pct,
            "mean_gen_time_ms": mean_gen_time,
            "p50_gen_time_ms":  p50_gen_time,
            "p95_gen_time_ms":  p95_gen_time,
            "mean_output_len":  mean_output_len,
            "n_empty":          n_empty,
            "n_samples":        n_samples,
        })
    
    # Create summary dataframe sorted by BLEU-4
    df_summary = pd.DataFrame(summary_rows).sort_values("corpus_bleu_4", ascending=False)
    
    # Step 6: Save summary CSV and print table
    df_summary.to_csv("benchmark_summary.csv", index=False)
    print_success("Saved benchmark_summary.csv")
    
    # Print summary table to terminal
    print_header("BENCHMARK SUMMARY TABLE")
    print(df_summary.to_string(index=False))
    
    print_header("BENCHMARK COMPLETE")
    print_info(f"Detailed predictions: predictions_all_algorithms.csv")
    print_info(f"Aggregate summary: benchmark_summary.csv")


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
    parser.add_argument("--algorithm", type=str, default="beam_search_k5", 
                        choices=list(ALGORITHM_REGISTRY.keys()),
                        help=f"Search algorithm to use. Available: {', '.join(ALGORITHM_REGISTRY.keys())}")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run all search algorithms and produce benchmark CSVs")
    
    args = parser.parse_args()
    
    # If benchmark mode, run comprehensive evaluation
    if args.benchmark:
        # Set seed for reproducibility with stochastic algorithms
        torch.manual_seed(42)
        
        print_header("COMPREHENSIVE SEARCH ALGORITHM BENCHMARK")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print_info(f"Device: {device}")
        
        print_step("Loading evaluation data...")
        df = pd.read_parquet(args.data_path)
        df = df.sample(n=250, random_state=42)
        print_info(f"Loaded {len(df):,} samples for benchmark")
        print_success("Data loaded")
        
        src_vocab, tgt_vocab = load_vocabularies(args.diff_vocab_path, args.message_vocab_path)
        src_vocab_size = len(src_vocab.stoi)
        tgt_vocab_size = len(tgt_vocab.stoi)
        
        model = load_model(args.checkpoint, src_vocab_size, tgt_vocab_size, device, args)
        max_seq_length = args.max_seq_length
        
        print_step(f"Running benchmark across {len(ALGORITHM_REGISTRY)} algorithms...")
        rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking samples"):
            diff_text = row['diff_text'] if not pd.isna(row['diff_text']) else ""
            ref_message = row['message'] if not pd.isna(row['message']) else ""
            
            for algo_name, algo_fn in ALGORITHM_REGISTRY.items():
                try:
                    gen_message, gen_time = algo_fn(
                        model, diff_text, src_vocab, tgt_vocab, device, max_seq_length
                    )
                except Exception as e:
                    print_warning(f"Error with {algo_name} on sample {idx}: {e}")
                    gen_message, gen_time = "", 0.0
                
                if gen_message.strip():
                    bleu = compute_bleu([ref_message], [gen_message])
                    meteor = compute_meteor([ref_message], [gen_message])
                    rouge = compute_rouge_l([ref_message], [gen_message])
                    exact = int(ref_message.strip().lower() == gen_message.strip().lower())
                else:
                    bleu = 0.0
                    meteor = 0.0
                    rouge = {"rouge_l_f": 0.0}
                    exact = 0
                
                rows.append({
                    "sample_id":         idx,
                    "diff_text":         diff_text[:200],
                    "reference_message": ref_message,
                    "algorithm":         algo_name,
                    "generated_message": gen_message,
                    "bleu_4":            bleu,
                    "meteor":            meteor,
                    "rouge_l_f":         rouge["rouge_l_f"],
                    "exact_match":       exact,
                    "gen_time_ms":       gen_time * 1000,
                    "output_length":     len(gen_message.split()) if gen_message else 0,
                })
        
        print_step("Saving detailed predictions...")
        df_pred = pd.DataFrame(rows)
        df_pred.to_csv("predictions_all_algorithms.csv", index=False)
        print_success(f"Saved {len(df_pred):,} prediction rows to predictions_all_algorithms.csv")
        
        print_step("Computing aggregate statistics...")
        summary_rows = []
        
        for algo_name, group in df_pred.groupby("algorithm"):
            refs = group["reference_message"].tolist()
            hyps = group["generated_message"].tolist()
            
            corpus_bleu_score = compute_bleu(refs, hyps)
            mean_meteor = group["meteor"].mean()
            mean_rouge = group["rouge_l_f"].mean()
            exact_match_pct = group["exact_match"].mean() * 100
            
            times = group["gen_time_ms"]
            mean_gen_time = times.mean()
            p50_gen_time = times.quantile(0.50)
            p95_gen_time = times.quantile(0.95)
            
            mean_output_len = group["output_length"].mean()
            n_empty = int((group["generated_message"].str.strip() == "").sum())
            n_samples = len(group)
            
            summary_rows.append({
                "algorithm":        algo_name,
                "corpus_bleu_4":    corpus_bleu_score,
                "mean_meteor":      mean_meteor,
                "mean_rouge_l_f":   mean_rouge,
                "exact_match_pct":  exact_match_pct,
                "mean_gen_time_ms": mean_gen_time,
                "p50_gen_time_ms":  p50_gen_time,
                "p95_gen_time_ms":  p95_gen_time,
                "mean_output_len":  mean_output_len,
                "n_empty":          n_empty,
                "n_samples":        n_samples,
            })
        
        df_summary = pd.DataFrame(summary_rows).sort_values("corpus_bleu_4", ascending=False)
        df_summary.to_csv("benchmark_summary.csv", index=False)
        print_success("Saved benchmark_summary.csv")
        
        print_header("BENCHMARK SUMMARY TABLE")
        print(df_summary.to_string(index=False))
        
        print_header("BENCHMARK COMPLETE")
        print_info(f"Detailed predictions: predictions_all_algorithms.csv")
        print_info(f"Aggregate summary: benchmark_summary.csv")
        return
    
    print_header("COMMIT MESSAGE GENERATION EVALUATION")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(f"Device: {device}")
    
    # Load data
    print_step("Loading data...")
    df = pd.read_parquet(args.data_path)
    df = df.sample(n=250, random_state=42)  # Fixed n=500 for consistent evaluation
    print_info(f"Loaded {len(df):,} samples")
    print_success("Data loaded")
    
    # Load vocabularies
    src_vocab, tgt_vocab = load_vocabularies(args.diff_vocab_path, args.message_vocab_path)
    src_vocab_size = len(src_vocab.stoi)
    tgt_vocab_size = len(tgt_vocab.stoi)
    
    # Load model
    model = load_model(args.checkpoint, src_vocab_size, tgt_vocab_size, device, args)
    
    # Get selected algorithm from registry
    if args.algorithm not in ALGORITHM_REGISTRY:
        print_error(f"Unknown algorithm: {args.algorithm}")
        print_info(f"Available algorithms: {', '.join(ALGORITHM_REGISTRY.keys())}")
        sys.exit(1)
    
    algorithm_fn = ALGORITHM_REGISTRY[args.algorithm]
    print_info(f"Using algorithm: {args.algorithm}")
    
    # Generate predictions using batched generation for greedy algorithms
    # For non-greedy algorithms (beam search, etc.), use per-sample loop
    BATCH_SIZE = 768  # Tune this — increase until VRAM is ~90% full
    
    if args.algorithm == "greedy":
        # Batched generation for maximum GPU utilization
        diff_texts = [
            row['diff_text'] if not pd.isna(row['diff_text']) else ""
            for _, row in df.iterrows()
        ]
        
        print_step(f"Generating {len(diff_texts)} predictions (batch_size={BATCH_SIZE})...")
        import time
        t0 = time.time()
        
        predictions = batch_greedy_generate(
            model, diff_texts, src_vocab, tgt_vocab, device,
            max_seq_length=args.max_seq_length,
            max_gen_length=args.max_gen_length,
            batch_size=BATCH_SIZE,
        )
        
        total_time = time.time() - t0
        generation_times = [total_time / len(predictions)] * len(predictions)  # approximate per-sample
        print_success(f"Generated {len(predictions)} predictions in {total_time:.1f}s "
                      f"({total_time/len(predictions)*1000:.1f} ms/sample)")
    else:
        # Per-sample generation for beam search and other algorithms
        print_step("Generating predictions...")
        predictions = []
        generation_times = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating ({args.algorithm})"):
            diff_text = row['diff_text'] if not pd.isna(row['diff_text']) else ""
            
            try:
                pred, gen_time = algorithm_fn(
                    model, diff_text, src_vocab, tgt_vocab, device, args.max_seq_length
                )
                
                predictions.append(pred)
                generation_times.append(gen_time)
            except Exception as e:
                print_warning(f"Error generating for row {idx}: {e}")
                predictions.append("")
                generation_times.append(0.0)

    # Add predictions and generation times to dataframe
    df['predicted_message'] = predictions
    df['generation_time'] = generation_times
    
    # Print generation time statistics
    valid_times = [t for t in generation_times if t > 0]
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        total_time = sum(valid_times)
        print_info(f"Average generation time: {avg_time:.3f}s per sample")
        print_info(f"Total generation time: {total_time:.1f}s for {len(valid_times)} samples")
    
    # Compute evaluation metrics
    print_header("COMPUTING EVALUATION METRICS")
    references = [row['message'] if not pd.isna(row['message']) else "" for _, row in df.iterrows()]
    hypotheses = df['predicted_message'].tolist()
    
    metrics = compute_all_metrics(references, hypotheses)
    
    if metrics:
        # Save metrics report
        metrics_path = args.output.replace('.csv', '_metrics.txt')
        save_metrics_report(metrics, metrics_path)
        
        # Also save metrics to CSV metadata
        metrics_df = pd.DataFrame([metrics])
        metrics_csv_path = args.output.replace('.csv', '_metrics_summary.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        print_success(f"Metrics summary saved to {metrics_csv_path}")
    
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
