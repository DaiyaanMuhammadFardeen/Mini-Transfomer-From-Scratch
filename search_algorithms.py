"""
Search algorithms for Transformer-based text generation.
Implements various search strategies for sequence generation.
"""

import torch
import time
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Safety utility
# ---------------------------------------------------------------------------

def _safe_multinomial(probs: torch.Tensor) -> int:
    """
    Sample one token from a probability distribution, with full guards.

    Problems that cause the ROCm/CUDA assertion crash:
      1. NaN / Inf values anywhere in the tensor
      2. Negative values (possible after masking edge-cases)
      3. All-zero mass (empty typical set, over-aggressive min-p, etc.)

    Strategy:
      - Replace non-finite values with 0
      - Clamp negatives to 0
      - If the resulting mass is still zero → fall back to argmax of the
        *original* probs (i.e. greedy) so we never pass a degenerate
        distribution to multinomial.
    """
    # Work on a clean clone so we don't corrupt the caller's tensor
    safe = probs.clone().float()

    # Kill NaN / Inf
    safe = torch.nan_to_num(safe, nan=0.0, posinf=0.0, neginf=0.0)

    # Kill negatives
    safe = safe.clamp(min=0.0)

    total = safe.sum()
    if total <= 0.0:
        # Degenerate distribution → greedy fallback on original probs
        safe_orig = torch.nan_to_num(probs.clone().float(), nan=0.0,
                                     posinf=0.0, neginf=0.0).clamp(min=0.0)
        if safe_orig.sum() <= 0.0:
            # Absolute last resort: uniform
            safe_orig = torch.ones_like(probs)
        return torch.argmax(safe_orig).item()

    # Renormalise (multinomial does NOT require a normalised input, but
    # being explicit avoids floating-point edge cases on some backends)
    safe = safe / safe.sum()

    # multinomial on CPU is safer on ROCm for small vocabs; GPU for big ones.
    # Keeping on the same device as input but CPU fallback if needed.
    try:
        return torch.multinomial(safe, 1).item()
    except RuntimeError:
        # Last-resort CPU fallback
        return torch.multinomial(safe.cpu(), 1).item()


# ---------------------------------------------------------------------------
# Greedy search
# ---------------------------------------------------------------------------

def greedy_search(model, diff_text, src_vocab, tgt_vocab, device, max_seq_length,
                  max_gen_length=50):
    start_time = time.time()

    src_ids = _tokenize_and_pad(diff_text, src_vocab, max_seq_length)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    change_features = _extract_change_features(src_ids, src_vocab).unsqueeze(0).to(device)

    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id = tgt_vocab.stoi["<EOS>"]

    tgt_tokens = [sos_id]

    with torch.no_grad():
        for _ in range(max_gen_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)

            src_embedded = model.dropout(model.encoder_embedding(src_tensor, change_features))
            tgt_embedded = model.dropout(model.decoder_embedding(tgt_tensor))

            enc_output = src_embedded
            for enc_layer in model.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)
            enc_output = model.encoder_norm(enc_output)

            dec_output = tgt_embedded
            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            dec_output = model.decoder_norm(dec_output)

            logits = model.fc(dec_output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).item()
            tgt_tokens.append(next_token)

            if next_token == eos_id:
                break

    return _tokens_to_text(tgt_tokens, tgt_vocab), time.time() - start_time


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

def beam_search(model, diff_text, src_vocab, tgt_vocab, device, max_seq_length,
                max_gen_length=50, beam_width=5, length_penalty=1.0):

    def _lp(score, length, alpha=1.0):
        return score / ((5 + length) ** alpha / 6 ** alpha)

    start_time = time.time()

    src_ids = _tokenize_and_pad(diff_text, src_vocab, max_seq_length)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    change_features = _extract_change_features(src_ids, src_vocab).unsqueeze(0).to(device)

    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id = tgt_vocab.stoi["<EOS>"]

    beams = [([sos_id], 0.0)]
    completed_beams = []

    with torch.no_grad():
        src_mask, _ = model.generate_mask(src_tensor, src_tensor)
        src_embedded = model.dropout(model.encoder_embedding(src_tensor, change_features))
        enc_output = src_embedded
        for enc_layer in model.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        enc_output = model.encoder_norm(enc_output)   # applied ONCE

    for _ in range(max_gen_length):
        candidates = []

        with torch.no_grad():
            for seq, score in beams:
                tgt_tensor = torch.tensor([seq], dtype=torch.long).to(device)
                _, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)

                tgt_embedded = model.dropout(model.decoder_embedding(tgt_tensor))
                dec_output = tgt_embedded
                for dec_layer in model.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
                dec_output = model.decoder_norm(dec_output)

                logits = model.fc(dec_output[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    token_id = top_k_indices[0, i].item()
                    token_score = top_k_probs[0, i].item()
                    candidates.append((seq + [token_id], score + token_score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        beams = []
        for seq, score in candidates:
            if seq[-1] == eos_id:
                completed_beams.append((seq, score))
            else:
                beams.append((seq, score))
            if len(beams) == beam_width:
                break

        if not beams:
            break

    completed_beams.extend(beams)
    completed_beams = [
        (seq, _lp(score, len(seq), alpha=length_penalty))
        for seq, score in completed_beams
    ]
    completed_beams.sort(key=lambda x: x[1], reverse=True)

    message = _tokens_to_text(completed_beams[0][0], tgt_vocab) if completed_beams else ""
    return message, time.time() - start_time


# ---------------------------------------------------------------------------
# Top-k sampling
# ---------------------------------------------------------------------------

def top_k_sampling(model, diff_text, src_vocab, tgt_vocab, device, max_seq_length,
                   max_gen_length=50, k=50, temperature=1.0):
    start_time = time.time()

    src_ids = _tokenize_and_pad(diff_text, src_vocab, max_seq_length)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    change_features = _extract_change_features(src_ids, src_vocab).unsqueeze(0).to(device)

    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id = tgt_vocab.stoi["<EOS>"]
    tgt_tokens = [sos_id]

    with torch.no_grad():
        for _ in range(max_gen_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)

            src_embedded = model.dropout(model.encoder_embedding(src_tensor, change_features))
            tgt_embedded = model.dropout(model.decoder_embedding(tgt_tensor))

            enc_output = src_embedded
            for enc_layer in model.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)
            enc_output = model.encoder_norm(enc_output)

            dec_output = tgt_embedded
            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            dec_output = model.decoder_norm(dec_output)

            logits = model.fc(dec_output[:, -1, :]) / temperature
            top_k_logits, top_k_indices = torch.topk(logits, k)
            probs = torch.softmax(top_k_logits, dim=-1)

            next_token_idx = _safe_multinomial(probs[0])
            next_token = top_k_indices[0, next_token_idx].item()
            tgt_tokens.append(next_token)

            if next_token == eos_id:
                break

    return _tokens_to_text(tgt_tokens, tgt_vocab), time.time() - start_time


# ---------------------------------------------------------------------------
# Top-p (nucleus) sampling
# ---------------------------------------------------------------------------

def top_p_sampling(model, diff_text, src_vocab, tgt_vocab, device, max_seq_length,
                   max_gen_length=50, p=0.9, temperature=1.0):
    start_time = time.time()

    src_ids = _tokenize_and_pad(diff_text, src_vocab, max_seq_length)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    change_features = _extract_change_features(src_ids, src_vocab).unsqueeze(0).to(device)

    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id = tgt_vocab.stoi["<EOS>"]
    tgt_tokens = [sos_id]

    with torch.no_grad():
        for _ in range(max_gen_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)

            src_embedded = model.dropout(model.encoder_embedding(src_tensor, change_features))
            tgt_embedded = model.dropout(model.decoder_embedding(tgt_tensor))

            enc_output = src_embedded
            for enc_layer in model.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)
            enc_output = model.encoder_norm(enc_output)

            dec_output = tgt_embedded
            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            dec_output = model.decoder_norm(dec_output)

            logits = model.fc(dec_output[:, -1, :]) / temperature
            probs = torch.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)

            # Keep the first token that pushes cumulative past p
            # (always keep at least 1 token)
            cutoff = (cumulative > p).nonzero(as_tuple=True)[1]
            cutoff_idx = cutoff[0].item() + 1 if len(cutoff) else sorted_probs.size(-1)

            nucleus_probs = sorted_probs[:, :cutoff_idx]
            nucleus_indices = sorted_indices[:, :cutoff_idx]

            next_token_idx = _safe_multinomial(nucleus_probs[0])
            next_token = nucleus_indices[0, next_token_idx].item()
            tgt_tokens.append(next_token)

            if next_token == eos_id:
                break

    return _tokens_to_text(tgt_tokens, tgt_vocab), time.time() - start_time


# ---------------------------------------------------------------------------
# Diverse beam search
# ---------------------------------------------------------------------------

def diverse_beam_search(model, diff_text, src_vocab, tgt_vocab, device, max_seq_length,
                        max_gen_length=50, beam_width=6, group_beam_width=2,
                        diversity_penalty=0.5, length_penalty=1.0):

    def _lp(score, length, alpha=1.0):
        return score / ((5 + length) ** alpha / 6 ** alpha)

    start_time = time.time()
    num_groups = beam_width // group_beam_width

    src_ids = _tokenize_and_pad(diff_text, src_vocab, max_seq_length)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    change_features = _extract_change_features(src_ids, src_vocab).unsqueeze(0).to(device)

    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id = tgt_vocab.stoi["<EOS>"]

    group_beams = [[([sos_id], 0.0)] * group_beam_width for _ in range(num_groups)]
    group_completed = [[] for _ in range(num_groups)]

    with torch.no_grad():
        src_mask, _ = model.generate_mask(src_tensor, src_tensor)
        src_embedded = model.dropout(model.encoder_embedding(src_tensor, change_features))
        enc_output = src_embedded
        for enc_layer in model.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        enc_output = model.encoder_norm(enc_output)

    for _ in range(max_gen_length):
        selected_tokens_by_group = [set() for _ in range(num_groups)]

        with torch.no_grad():
            for g in range(num_groups):
                candidates = []
                for seq, score in group_beams[g]:
                    tgt_tensor = torch.tensor([seq], dtype=torch.long).to(device)
                    _, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)

                    tgt_embedded = model.dropout(model.decoder_embedding(tgt_tensor))
                    dec_output = tgt_embedded
                    for dec_layer in model.decoder_layers:
                        dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
                    dec_output = model.decoder_norm(dec_output)

                    logits = model.fc(dec_output[:, -1, :])

                    for prev_g in range(g):
                        for tok in selected_tokens_by_group[prev_g]:
                            if 0 <= tok < logits.size(-1):
                                logits[0, tok] -= diversity_penalty

                    log_probs = torch.log_softmax(logits, dim=-1)
                    top_k_p, top_k_i = torch.topk(log_probs, group_beam_width)

                    for i in range(group_beam_width):
                        tok = top_k_i[0, i].item()
                        candidates.append((seq + [tok], score + top_k_p[0, i].item()))

                candidates.sort(key=lambda x: x[1], reverse=True)
                new_beams = []
                for seq, score in candidates:
                    if seq[-1] == eos_id:
                        group_completed[g].append((seq, score))
                    else:
                        new_beams.append((seq, score))
                        selected_tokens_by_group[g].add(seq[-1])
                    if len(new_beams) == group_beam_width:
                        break
                group_beams[g] = new_beams

    all_completed = [b for gc in group_completed for b in gc]
    for gb in group_beams:
        all_completed.extend(gb)

    all_completed = [
        (seq, _lp(score, len(seq), alpha=length_penalty))
        for seq, score in all_completed
    ]
    all_completed.sort(key=lambda x: x[1], reverse=True)

    message = _tokens_to_text(all_completed[0][0], tgt_vocab) if all_completed else ""
    return message, time.time() - start_time


# ---------------------------------------------------------------------------
# Contrastive search (approximate)
# ---------------------------------------------------------------------------

def contrastive_search(model, diff_text, src_vocab, tgt_vocab, device, max_seq_length,
                       max_gen_length=50, penalty_alpha=0.6, top_k=4):
    """
    Approximate contrastive search (Su et al., 2022).
    Uses the current hidden state as proxy for all candidates — see original
    docstring for caveats.
    """
    start_time = time.time()

    src_ids = _tokenize_and_pad(diff_text, src_vocab, max_seq_length)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    change_features = _extract_change_features(src_ids, src_vocab).unsqueeze(0).to(device)

    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id = tgt_vocab.stoi["<EOS>"]
    tgt_tokens = [sos_id]
    past_hidden_states = []

    with torch.no_grad():
        for _ in range(max_gen_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)

            src_embedded = model.dropout(model.encoder_embedding(src_tensor, change_features))
            tgt_embedded = model.dropout(model.decoder_embedding(tgt_tensor))

            enc_output = src_embedded
            for enc_layer in model.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)
            enc_output = model.encoder_norm(enc_output)

            dec_output = tgt_embedded
            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            dec_output = model.decoder_norm(dec_output)

            current_hidden = dec_output[:, -1, :]
            logits = model.fc(current_hidden)
            top_k_logits, top_k_indices = torch.topk(logits, top_k)

            contrastive_scores = []
            for i in range(top_k):
                token_logit = top_k_logits[0, i].item()
                if past_hidden_states:
                    past = torch.stack(past_hidden_states)
                    expanded = current_hidden.expand(past.size(0), -1)
                    max_sim = torch.cosine_similarity(expanded, past, dim=1).max().item()
                else:
                    max_sim = 0.0
                contrastive_scores.append(token_logit - penalty_alpha * max_sim)

            best_idx = int(torch.argmax(torch.tensor(contrastive_scores)).item())
            next_token = top_k_indices[0, best_idx].item()
            tgt_tokens.append(next_token)
            past_hidden_states.append(current_hidden.squeeze(0))

            if next_token == eos_id:
                break

    return _tokens_to_text(tgt_tokens, tgt_vocab), time.time() - start_time


# ---------------------------------------------------------------------------
# Typical sampling  ← PRIMARY FIX
# ---------------------------------------------------------------------------

def typical_sampling(model, diff_text, src_vocab, tgt_vocab, device, max_seq_length,
                     max_gen_length=50, tau=0.9, temperature=1.0):
    """
    Typical sampling (Meister et al., 2022).

    Key fix vs. original:
      - All distribution math is done on CPU to avoid ROCm assertion crashes
        caused by near-zero std_dev producing an all-zero typical-set mask.
      - _safe_multinomial() provides a final safety net.
    """
    start_time = time.time()

    src_ids = _tokenize_and_pad(diff_text, src_vocab, max_seq_length)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    change_features = _extract_change_features(src_ids, src_vocab).unsqueeze(0).to(device)

    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id = tgt_vocab.stoi["<EOS>"]
    tgt_tokens = [sos_id]

    with torch.no_grad():
        for _ in range(max_gen_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)

            src_embedded = model.dropout(model.encoder_embedding(src_tensor, change_features))
            tgt_embedded = model.dropout(model.decoder_embedding(tgt_tensor))

            enc_output = src_embedded
            for enc_layer in model.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)
            enc_output = model.encoder_norm(enc_output)

            dec_output = tgt_embedded
            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            dec_output = model.decoder_norm(dec_output)

            logits = (model.fc(dec_output[:, -1, :]) / temperature).squeeze(0)  # [V]

            # ---- All distribution work on CPU to avoid ROCm assertion ----
            logits_cpu = logits.float().cpu()
            logits_cpu = torch.nan_to_num(logits_cpu, nan=0.0, posinf=0.0, neginf=0.0)

            probs_cpu = torch.softmax(logits_cpu, dim=-1)
            log_probs_cpu = torch.log(probs_cpu.clamp(min=1e-10))

            # Shannon entropy H
            entropy = -(probs_cpu * log_probs_cpu).sum()          # scalar

            # Surprise of each token  s_i = -log p_i
            surprise = -log_probs_cpu                              # [V]

            # |s_i - H|
            typicality = torch.abs(surprise - entropy)            # [V]

            # Sort by typicality and keep the smallest-typicality tokens
            # until their cumulative probability mass >= tau
            sorted_typ, sorted_idx = torch.sort(typicality)
            sorted_probs = probs_cpu[sorted_idx]
            cumulative = torch.cumsum(sorted_probs, dim=0)

            cutoff = (cumulative >= tau).nonzero(as_tuple=True)[0]
            cutoff_idx = cutoff[0].item() + 1 if len(cutoff) else len(sorted_probs)

            # Keep at least 1 token
            cutoff_idx = max(cutoff_idx, 1)

            nucleus_probs = sorted_probs[:cutoff_idx]
            nucleus_indices = sorted_idx[:cutoff_idx]

            next_token_local = _safe_multinomial(nucleus_probs)
            next_token = nucleus_indices[next_token_local].item()
            # ---------------------------------------------------------------

            tgt_tokens.append(next_token)

            if next_token == eos_id:
                break

    return _tokens_to_text(tgt_tokens, tgt_vocab), time.time() - start_time


# ---------------------------------------------------------------------------
# Min-p sampling
# ---------------------------------------------------------------------------

def min_p_sampling(model, diff_text, src_vocab, tgt_vocab, device, max_seq_length,
                   max_gen_length=50, min_p=0.05, temperature=1.0):
    start_time = time.time()

    src_ids = _tokenize_and_pad(diff_text, src_vocab, max_seq_length)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    change_features = _extract_change_features(src_ids, src_vocab).unsqueeze(0).to(device)

    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id = tgt_vocab.stoi["<EOS>"]
    tgt_tokens = [sos_id]

    with torch.no_grad():
        for _ in range(max_gen_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)

            src_embedded = model.dropout(model.encoder_embedding(src_tensor, change_features))
            tgt_embedded = model.dropout(model.decoder_embedding(tgt_tensor))

            enc_output = src_embedded
            for enc_layer in model.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)
            enc_output = model.encoder_norm(enc_output)

            dec_output = tgt_embedded
            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            dec_output = model.decoder_norm(dec_output)

            logits = model.fc(dec_output[:, -1, :]) / temperature
            probs = torch.softmax(logits, dim=-1)[0]  # [V]

            # Guard NaN before threshold comparison
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)

            max_prob = probs.max()
            threshold = min_p * max_prob
            mask = probs >= threshold

            if mask.sum() == 0:
                mask = torch.ones_like(probs, dtype=torch.bool)

            masked_probs = probs * mask.float()
            next_token = _safe_multinomial(masked_probs)
            tgt_tokens.append(next_token)

            if next_token == eos_id:
                break

    return _tokens_to_text(tgt_tokens, tgt_vocab), time.time() - start_time


# ---------------------------------------------------------------------------
# Temperature sampling
# ---------------------------------------------------------------------------

def temperature_sampling(model, diff_text, src_vocab, tgt_vocab, device, max_seq_length,
                         max_gen_length=50, temperature=1.0):
    start_time = time.time()

    src_ids = _tokenize_and_pad(diff_text, src_vocab, max_seq_length)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    change_features = _extract_change_features(src_ids, src_vocab).unsqueeze(0).to(device)

    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id = tgt_vocab.stoi["<EOS>"]
    tgt_tokens = [sos_id]

    with torch.no_grad():
        for _ in range(max_gen_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)

            src_embedded = model.dropout(model.encoder_embedding(src_tensor, change_features))
            tgt_embedded = model.dropout(model.decoder_embedding(tgt_tensor))

            enc_output = src_embedded
            for enc_layer in model.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)
            enc_output = model.encoder_norm(enc_output)

            dec_output = tgt_embedded
            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            dec_output = model.decoder_norm(dec_output)

            logits = model.fc(dec_output[:, -1, :]) / temperature
            probs = torch.softmax(logits, dim=-1)[0]

            next_token = _safe_multinomial(probs)
            tgt_tokens.append(next_token)

            if next_token == eos_id:
                break

    return _tokens_to_text(tgt_tokens, tgt_vocab), time.time() - start_time


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _tokenize_and_pad(text, vocab, max_length):
    if not text:
        text = ""

    ids = vocab.numericalize(str(text))

    if len(ids) > max_length - 2:
        ids = ids[:max_length - 2]

    sos_id = vocab.stoi["<SOS>"]
    eos_id = vocab.stoi["<EOS>"]
    pad_id = vocab.stoi["<PAD>"]

    ids = [sos_id] + ids + [eos_id]
    ids = ids + [pad_id] * (max_length - len(ids))

    return ids[:max_length]


def _extract_change_features(token_ids, src_vocab):
    CHANGE_TYPE_TAGS = ['<ADD>', '<REMOVE>', '<MODIFY>',
                        '<COMMENT_ADD>', '<COMMENT_REMOVE>', '<COMMENT_MODIFY>']

    features = torch.zeros(6, dtype=torch.float32)

    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids)

    for i, tag in enumerate(CHANGE_TYPE_TAGS):
        tag_id = src_vocab.stoi.get(tag, -1)
        if tag_id != -1 and (token_ids == tag_id).any():
            features[i] = 1.0

    return features


def _tokens_to_text(tokens, vocab):
    eos_id = vocab.stoi["<EOS>"]
    sos_id = vocab.stoi["<SOS>"]
    pad_id = vocab.stoi["<PAD>"]

    text_tokens = []
    for idx in tokens:
        if idx == eos_id:
            break
        if idx not in (pad_id, sos_id) and idx in vocab.itos:
            text_tokens.append(vocab.itos[idx])

    return ' '.join(text_tokens)
