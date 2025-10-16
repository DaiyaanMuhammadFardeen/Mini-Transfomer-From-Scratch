import torch
import re
import sys
import gc
import random
from collections import defaultdict
from math import log
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np

def parallel_tokenize(args):
    instance, sentence = args
    return instance.tokenize(sentence)

class Vocabulary:
    """Stochastic vocabulary builder with GPU-accelerated EM training for commit messages."""
    
    def __init__(self, freq_threshold=2, target_size=50000, percent_to_remove=0.1, stochastic_prob=0.1, max_expansion_depth=2):
        # Simplified special tokens for messages (remove diff-specific tags)
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.target_size = target_size
        self.percent_to_remove = percent_to_remove
        self.stochastic_prob = stochastic_prob  # StochasTok: Probability of random split
        self.max_expansion_depth = max_expansion_depth  # StochasTok: Max recursive splits

    def tokenize(self, text):
        """Tokenize commit message with stochastic expansion."""
        # Simple regex pre-tokenization for natural language
        words = re.findall(r"\w+|[^\w\s]|\'\w+", text.lower())  # Lowercase, preserve contractions
        tokens = []
        for word in words:
            # Apply stochastic expansion per token
            expanded = self._stochastic_expand(word)
            tokens.extend(expanded)
        return tokens

    def _stochastic_expand(self, token, depth=0):
        """StochasTok: Randomly split token into subword pairs with probability."""
        if depth >= self.max_expansion_depth or len(token) <= 1:
            return [token]  # Base case: No further splits
        
        if random.random() < self.stochastic_prob:  # Randomly decide to split
            # Simple random split at a position (paper suggests mid-word for simplicity)
            split_pos = random.randint(1, len(token) - 1)
            left = token[:split_pos]
            right = token[split_pos:]
            # Recurse for potential deeper splits
            return self._stochastic_expand(left, depth + 1) + self._stochastic_expand(right, depth + 1)
        else:
            return [token]  # Keep whole token

    def _encode_word(self, word, model):
        """Encode word using unigram segmentation with DP, incorporating stochastic splits."""
        best_segmentations = [{"start": 0, "score": 0}] + [{"start": None, "score": None} for _ in range(len(word))]
        for start_idx in range(len(word)):
            best_score_at_start = best_segmentations[start_idx]["score"]
            if best_score_at_start is None:
                continue
            for end_idx in range(start_idx + 1, len(word) + 1):
                token = word[start_idx:end_idx]
                if token in model:
                    score = model[token] + best_score_at_start
                    if best_segmentations[end_idx]["score"] is None or best_segmentations[end_idx]["score"] > score:
                        best_segmentations[end_idx] = {"start": start_idx, "score": score}
        
        segmentation = best_segmentations[-1]
        if segmentation["score"] is None:
            # For OOV, try stochastic splitting before defaulting to <UNK>
            if random.random() < self.stochastic_prob:
                split_pos = random.randint(1, len(word) - 1) if len(word) > 1 else len(word)
                left, right = word[:split_pos], word[split_pos:]
                left_tokens, left_score = self._encode_word(left, model)
                right_tokens, right_score = self._encode_word(right, model)
                if left_score is not None and right_score is not None:
                    return (left_tokens + right_tokens, left_score + right_score)
            return (["<UNK>"], None)
        
        score = segmentation["score"]
        tokens = []
        pos = len(word)
        while pos > 0:
            start = best_segmentations[pos]["start"]
            tokens.insert(0, word[start:pos])
            pos = start
        return (tuple(tokens), score)

    def _encode_word_batch_gpu(self, words, model, device='cuda'):
        """Encode multiple words in parallel on GPU using vectorized DP."""
        if not torch.cuda.is_available() and device == 'cuda':
            device = 'cpu'
            print("⚠️ CUDA not available, falling back to CPU", file=sys.stderr)
        
        batch_size = 8192
        max_word_len = max(len(w) for w in words) if words else 1
        results = []
        
        for batch_start in range(0, len(words), batch_size):
            batch_words = words[batch_start:batch_start + batch_size]
            results.extend(self._encode_batch_vectorized(batch_words, model, max_word_len, device))
        
        return results

    def _encode_batch_vectorized(self, words, model, max_len, device):
        """Vectorized DP for batch of words."""
        batch_size = len(words)
        dp_scores = torch.full((batch_size, max_len + 1), float('inf'), 
                              dtype=torch.float32, device=device)
        dp_scores[:, 0] = 0
        dp_backptr = np.full((batch_size, max_len + 1), -1, dtype=np.int32)
        
        token_lens = sorted(set(len(t) for t in model.keys()))
        
        for start_idx in range(max_len):
            for token_len in token_lens:
                end_idx = start_idx + token_len
                if end_idx > max_len:
                    continue
                
                tokens, valid_mask = [], []
                for i, word in enumerate(words):
                    if end_idx <= len(word):
                        token = word[start_idx:end_idx]
                        tokens.append(token)
                        valid_mask.append(i)
                
                if not tokens:
                    continue
                
                token_scores = torch.tensor(
                    [model.get(t, float('inf')) for t in tokens],
                    dtype=torch.float32, device=device
                )
                
                for idx, word_idx in enumerate(valid_mask):
                    prev_score = dp_scores[word_idx, start_idx]
                    new_score = prev_score + token_scores[idx]
                    if new_score < dp_scores[word_idx, end_idx]:
                        dp_scores[word_idx, end_idx] = new_score
                        dp_backptr[word_idx, end_idx] = start_idx
        
        results = []
        dp_scores_cpu = dp_scores.cpu().numpy()
        for i, word in enumerate(words):
            word_len = len(word)
            final_score = dp_scores_cpu[i, word_len]
            
            if np.isinf(final_score):
                # StochasTok: Try splitting OOV word
                if random.random() < self.stochastic_prob and len(word) > 1:
                    split_pos = random.randint(1, len(word) - 1)
                    left, right = word[:split_pos], word[split_pos:]
                    left_result, left_score = self._encode_word(left, model)
                    right_result, right_score = self._encode_word(right, model)
                    if left_score is not None and right_score is not None:
                        results.append((left_result + right_result, left_score + right_score))
                        continue
                results.append((["<UNK>"], None))
                continue
            
            tokens = []
            pos = word_len
            while pos > 0:
                start = dp_backptr[i, pos]
                if start == -1:
                    break
                tokens.insert(0, word[start:pos])
                pos = start
            results.append((tokens, final_score))
        
        return results

    def _compute_loss_gpu(self, model, word_freqs, device='cuda', batch_size=1024):
        """GPU-accelerated loss computation."""
        if not torch.cuda.is_available():
            device = 'cpu'
        
        words = list(word_freqs.keys())
        freqs = [word_freqs[w] for w in words]
        total_loss = 0.0
        
        for i in tqdm(range(0, len(words), batch_size), desc="Computing loss (GPU)", ncols=160):
            batch_words = words[i:i + batch_size]
            batch_freqs = freqs[i:i + batch_size]
            batch_results = self._encode_word_batch_gpu(batch_words, model, device)
            
            for (tokens, score), freq in zip(batch_results, batch_freqs):
                if score is not None:
                    total_loss += freq * score
        
        return total_loss

    def _compute_scores_gpu(self, model, word_freqs, device='cuda', batch_size=1024):
        """GPU-accelerated exact score computation."""
        print(f"Computing exact scores on {device.upper()}...", file=sys.stderr)
        if not torch.cuda.is_available():
            device = 'cpu'
        
        model_loss = self._compute_loss_gpu(model, word_freqs, device, batch_size)
        scores = {}
        tokens_to_score = [t for t in model.keys() if len(t) > 1]
        
        for token in tqdm(tokens_to_score, desc="Scoring tokens", ncols=160):
            token_score_val = model.pop(token)
            loss_without = self._compute_loss_gpu(model, word_freqs, device, batch_size)
            scores[token] = loss_without - model_loss
            model[token] = token_score_val
        
        return scores

    def _smart_prune_with_caching(self, model, word_freqs, device='cuda'):
        """Optimized pruning with heuristic warm-start and batch scoring."""
        print("🔥 Smart pruning with optimizations...", file=sys.stderr)
        iteration = 0
        
        while len(model) > self.target_size:
            iteration += 1
            print(f"  Pruning iteration {iteration}: {len(model)} → {self.target_size}", file=sys.stderr)
            
            if iteration == 1:
                print("    Phase 1: Heuristic filtering", file=sys.stderr)
                heuristic_scores = {
                    token: (-model[token] / max(1, len(token)))
                    for token in model.keys()
                    if len(token) > 1
                }
                num_candidates = int(len(model) * 0.5)
                sorted_by_heuristic = sorted(heuristic_scores.items(), key=lambda x: x[1])
                candidates_to_score = [t for t, _ in sorted_by_heuristic[:num_candidates]]
            else:
                candidates_to_score = [t for t in model.keys() if len(t) > 1]
            
            if candidates_to_score:
                print("    Phase 2: Exact scoring", file=sys.stderr)
                scores = self._compute_scores_gpu(model, word_freqs, device)
            else:
                print("    No scoreable tokens, stopping pruning", file=sys.stderr)
                break
            
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])
            num_to_remove = max(1, int(len(model) * self.percent_to_remove))
            
            for i in range(min(num_to_remove, len(sorted_scores))):
                token_to_remove, score = sorted_scores[i]
                model.pop(token_to_remove, None)
                print(f"    Removed: {token_to_remove} (score: {score:.4f})")
        
        return model

    def _train_unigram_model(self, word_freqs, model, num_iters=2, device='cuda'):
        """GPU-accelerated EM training with stochastic expansions."""
        if not torch.cuda.is_available():
            device = 'cpu'
            print("⚠️ CUDA not available, using CPU", file=sys.stderr)
        
        words = list(word_freqs.keys())
        freqs = torch.tensor([word_freqs[w] for w in words], dtype=torch.float32, device=device)
        print(f"Preparing GPU tensors for {len(words)} words...", file=sys.stderr)
        
        for iteration in range(num_iters):
            print(f"🧩 EM Iteration {iteration+1}/{num_iters}", file=sys.stderr)
            token_counts = defaultdict(float)
            total_loss = 0.0
            
            batch_size = 8192
            for batch_start in tqdm(range(0, len(words), batch_size), 
                                   desc="E-step (GPU)", ncols=160, leave=False):
                batch_words = words[batch_start:batch_start + batch_size]
                batch_freqs = freqs[batch_start:batch_start + batch_size]
                batch_results = self._encode_word_batch_gpu(batch_words, model, device)
                
                for (tokens, score), freq in zip(batch_results, batch_freqs.cpu().numpy()):
                    if score is not None:
                        total_loss += float(freq) * score
                        for tok in tokens:
                            token_counts[tok] += float(freq)
            
            total_count = sum(token_counts.values())
            if total_count > 0:
                model = {tok: -log(cnt / total_count) for tok, cnt in token_counts.items()}
            
            print(f"  Loss: {total_loss:.4f} | Vocab size: {len(model)}", file=sys.stderr)
            
            if len(model) > self.target_size:
                num_to_remove = int(len(model) * 0.1)
                sorted_toks = sorted(model.items(), key=lambda x: x[1], reverse=True)
                for t, _ in sorted_toks[:num_to_remove]:
                    del model[t]
        
        return model

    def _heuristic_preprune(self, token_freqs):
        """Fast heuristic pre-pruning before EM."""
        print("🪓 Pre-pruning tokens with heuristics...", file=sys.stderr)
        pruned = {tok: freq for tok, freq in token_freqs.items()
                 if freq >= 3 and 1 <= len(tok) <= 8}
        
        scored = {tok: freq / (1 + len(tok)**1.2) for tok, freq in pruned.items()}
        target_size = int(self.target_size * 2)
        sorted_toks = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        kept = {tok: token_freqs[tok] for tok, _ in sorted_toks[:target_size]}
        
        print(f"✨ Prepruned to {len(kept)} tokens (from {len(token_freqs)})", file=sys.stderr)
        return kept

    def build_vocabulary(self, sentence_list):
        """Build vocabulary with GPU acceleration and stochastic tokenization."""
        print(f"🧠 Building vocabulary using GPU acceleration with StochasTok...", file=sys.stderr)
        
        with Pool(cpu_count()) as pool:
            all_token_lists = list(
                tqdm(
                    pool.imap(parallel_tokenize, [(self, s) for s in sentence_list]),
                    total=len(sentence_list),
                    desc="🔤 Tokenizing with StochasTok",
                    ncols=160
                )
            )
        
        all_tokens = [token for tokens in all_token_lists for token in tokens]
        del all_token_lists
        gc.collect()
        
        word_freqs = defaultdict(int)
        for token in all_tokens:
            word_freqs[token] += 1
        del all_tokens
        gc.collect()
        
        MAX_SUBWORD_LEN = 8
        character_freqs = defaultdict(int)
        subwords_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            word = str(word)
            for i in range(len(word)):
                character_freqs[word[i]] += freq
                for j in range(i + 2, min(i + MAX_SUBWORD_LEN + 1, len(word) + 1)):
                    subwords_freqs[word[i:j]] += freq
        
        sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
        token_freqs = list(character_freqs.items()) + sorted_subwords[:2 * self.target_size - len(character_freqs)]
        del character_freqs, subwords_freqs
        gc.collect()
        
        token_freqs = {token: freq for token, freq in token_freqs if freq >= self.freq_threshold}
        total_sum = sum(token_freqs.values())
        token_freqs = self._heuristic_preprune(token_freqs)
        model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self._train_unigram_model(word_freqs, model, num_iters=5, device=device)
        model = self._smart_prune_with_caching(model, word_freqs, device=device)
        
        del token_freqs
        gc.collect()
        
        sorted_tokens = sorted(model, key=model.get)
        idx = len(self.itos)
        for word in sorted_tokens:
            if word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        
        print(f"✅ Vocabulary built! Total tokens: {len(self.stoi)}", file=sys.stderr)

    def numericalize(self, text):
        """Convert text to token IDs, using stochastic splitting for OOV."""
        tokens = self.tokenize(text)
        subword_tokens = []
        for token in tokens:
            if token in self.stoi:
                subword_tokens.append(token)
            else:
                # StochasTok: Try splitting OOV word
                if random.random() < self.stochastic_prob and len(token) > 1:
                    expanded = self._stochastic_expand(token)
                    subword_tokens.extend(expanded)
                else:
                    subword_tokens.append("<UNK>")
        return [self.stoi.get(t, self.stoi["<UNK>"]) for t in subword_tokens]
