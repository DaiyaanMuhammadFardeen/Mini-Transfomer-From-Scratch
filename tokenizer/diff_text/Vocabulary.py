import torch
import re
import ctypes
import os
import sys
import gc
from collections import defaultdict
from math import log
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from tree_sitter import Language, Parser
import numpy as np

lib_path = "../build/python.so"
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Cannot find {lib_path}")
lib = ctypes.cdll.LoadLibrary(lib_path)
lib.tree_sitter_python.restype = ctypes.c_void_p
PY_LANGUAGE = Language(lib.tree_sitter_python())
parser = Parser(language=PY_LANGUAGE)

def get_node_text(node, source_code):
    return source_code[node.start_byte:node.end_byte].decode('utf8')

def parallel_tokenize(args):
    instance, sentence = args
    return instance.tokenize(sentence)

class Vocabulary:
    """GPU-accelerated vocabulary builder with EM training and optimized pruning."""
    
    def __init__(self, freq_threshold=2, target_size=50000, percent_to_remove=0.1):
        self.itos = {
            0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>",
            4: "<ADD>", 5: "</ADD>", 6: "<REMOVE>", 7: "</REMOVE>",
            8: "<COMMENT_ADD>", 9: "</COMMENT_ADD>",
            10: "<COMMENT_REMOVE>", 11: "</COMMENT_REMOVE>",
            12: "<MODIFY>", 13: "</MODIFY>",
            14: "<COMMENT_MODIFY>", 15: "</COMMENT_MODIFY>"
        }
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.target_size = target_size
        self.percent_to_remove = percent_to_remove

    def tokenize_diff_line(self, line):
        """Tokenize a single diff line using tree-sitter."""
        tokens = []
        code = line.strip()
        if code:
            try:
                tree = parser.parse(bytes(code, "utf8"))
                tokens.extend(self._extract_code_tokens(tree.root_node, bytes(code, "utf8")))
            except Exception as e:
                print(f"⚠️ Parsing error: {e}, falling back to simple split", file=sys.stderr)
                tokens.extend(code.split())
        return tokens

    def _extract_code_tokens(self, node, source_code):
        """Extract tokens from tree-sitter node."""
        tokens = []
        if node.type == 'identifier':
            tokens.append(get_node_text(node, source_code))
        elif node.type in ('string', 'integer', 'float'):
            tokens.append(get_node_text(node, source_code))
        elif node.type in ('def', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'from', 'as'):
            tokens.append(node.type)
        elif node.type == 'comment':
            comment_text = get_node_text(node, source_code).lstrip('#').strip()
            tokens.extend(re.split(r'\s+', comment_text))
        elif node.type in ('operator', 'punctuation', 'keyword'):
            tokens.append(get_node_text(node, source_code))
        for child in node.children:
            tokens.extend(self._extract_code_tokens(child, source_code))
        return tokens

    def tokenize(self, text):
        """Tokenize diff text."""
        lines = text.split('\n')
        tokens = []
        for line in lines:
            if line.strip():
                tokens.extend(self.tokenize_diff_line(line))
        return tokens

    def _encode_word(self, word, model):
        """Encode word using unigram segmentation with DP."""
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
            return ("<UNK>", None)
        
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
        
        for i in tqdm(range(0, len(words), batch_size), desc="Computing loss (GPU)", ncols=80):
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
        
        for token in tqdm(tokens_to_score, desc="Scoring tokens", ncols=80):
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
        """GPU-accelerated EM training."""
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
                                   desc="E-step (GPU)", ncols=80, leave=False):
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
        """Build vocabulary with GPU acceleration."""
        print(f"🧠 Building vocabulary using GPU acceleration...", file=sys.stderr)
        
        with Pool(cpu_count()) as pool:
            all_token_lists = list(
                tqdm(
                    pool.imap(parallel_tokenize, [(self, s) for s in sentence_list]),
                    total=len(sentence_list),
                    desc="🔤 Tokenizing",
                    ncols=80
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
        model = self._train_unigram_model(word_freqs, model, num_iters=2, device=device)
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
