import torch
import re
import ctypes
import os
import sys
import gc
import pickle
from collections import defaultdict
from math import log
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from tree_sitter import Language, Parser
import numpy as np

# Lazy initialization - only load when needed
_parser = None
MAX_RECURSION_DEPTH = 5  # Fix #1: Prevent infinite recursion

def _get_parser():
    """Lazy load tree-sitter parser with correct path resolution."""
    global _parser
    if _parser is None:
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        lib_path = os.path.join(current_dir, "build", "python.so")

        if not os.path.exists(lib_path):
            print(f"[ERROR] Cannot find {lib_path}", file=sys.stderr)
            raise FileNotFoundError(f"Cannot find tree-sitter library at {lib_path}")

        try:
            lib = ctypes.cdll.LoadLibrary(lib_path)
            lib.tree_sitter_python.restype = ctypes.c_void_p
            PY_LANGUAGE = Language(lib.tree_sitter_python())
            _parser = Parser(language=PY_LANGUAGE)
            print(f"[DEBUG] Tree-sitter parser loaded successfully from {lib_path}", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Failed to load tree-sitter: {e}", file=sys.stderr)
            raise

    return _parser

def get_node_text(node, source_code):
    """Safely extract node text with error handling."""
    try:
        return source_code[node.start_byte:node.end_byte].decode('utf8')
    except (UnicodeDecodeError, IndexError) as e:
        print(f"[WARNING] Failed to extract node text: {e}", file=sys.stderr)
        return ""

def parallel_tokenize(args):
    instance, sentence = args
    return instance.tokenize(sentence)

class DiffVocabulary:
    """GPU-accelerated vocabulary builder with EM training and optimized pruning."""

    def __init__(self, freq_threshold=2, target_size=50000, percent_to_remove=0.1,
                 checkpoint_dir="./tokenizer_checkpoints", batch_size=4096, max_recursion=MAX_RECURSION_DEPTH):
        self.itos = {
            0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>",
            4: "<ADD>", 5: "</ADD>", 6: "<REMOVE>", 7: "</REMOVE>",
            8: "<COMMENT_ADD>", 9: "</COMMENT_ADD>",
            10: "<COMMENT_REMOVE>", 11: "</COMMENT_REMOVE>",
            12: "<MODIFY>", 13: "</MODIFY>",
            14: "<COMMENT_MODIFY>", 15: "</COMMENT_MODIFY>",
            16: "<STR>", 17: "</STR>"
            # 18: "<FUNC_", 19: "_FUNC>",
            # 20: "<CLASS_", 21: "_CLASS>",
            # 22: "<MOD_", 23: "_MOD>",
            # 24: "<VAR_", 25: "_VAR>"
        }
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.target_size = target_size
        self.percent_to_remove = percent_to_remove
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size  # Fix #8: Make batch size configurable
        self.max_recursion = max_recursion  # Fix #1: Add recursion depth limit
        self.trained_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _split_identifier(self, name):
        """Split identifier into subparts based on naming conventions."""
        if not name:
            return []

        parts = [p for p in name.split('_') if p]
        subparts = []
        for part in parts:
            camel_splits = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', part)
            subparts.extend([s for s in camel_splits])

        return subparts

    def _strip_string_quotes(self, text):
        """Strip surrounding quotes from string literals."""
        if text.startswith(('"', "'", '"""', "'''")) and text.endswith(('"', "'", '"""', "'''")):
            return text[1:-1] if len(text) > 2 and text[0] == text[-1] and text[0] in ('"', "'") else text[3:-3]
        return text

    def _split_string_content(self, stripped_text):
        if '/' in stripped_text or '\\' in stripped_text:
            parts = [p.strip() for p in re.split(r'[/\\]', stripped_text) if p.strip()]
            return parts
        elif ' ' in stripped_text:
            parts = re.split(r'\s+', stripped_text.strip())
            return parts
        else:
            return [stripped_text]

    def tokenize_diff_line(self, line):
        """Tokenize a single diff line using tree-sitter, handling special tags."""
        tokens = []

        try:
            code = line.strip()

            if not code:
                return tokens

            special_tags = {
                '<ADD>': '</ADD>',
                '<REMOVE>': '</REMOVE>',
                '<MODIFY>': '</MODIFY>',
                '<COMMENT_ADD>': '</COMMENT_ADD>',
                '<COMMENT_REMOVE>': '</COMMENT_REMOVE>',
                '<COMMENT_MODIFY>': '</COMMENT_MODIFY>',
            }

            opening_tag = None
            closing_tag = None
            inner_code = code

            for open_tag, close_tag in special_tags.items():
                if code.startswith(open_tag) and code.endswith(close_tag):
                    opening_tag = open_tag
                    closing_tag = close_tag
                    inner_code = code[len(open_tag):-len(close_tag)].strip()
                    break

            if opening_tag:
                tokens.append(opening_tag)

            if inner_code:
                # Fix #11: Validate UTF-8 before parsing
                try:
                    inner_code.encode('utf-8')
                    parser = _get_parser()
                    tree = parser.parse(bytes(inner_code, "utf8"))
                    tokens.extend(self._extract_code_tokens(tree.root_node, bytes(inner_code, "utf8"), depth=0))
                except UnicodeDecodeError as e:
                    print(f"[WARNING] Invalid UTF-8 in diff line: {e}, falling back to simple split", file=sys.stderr)
                    tokens.extend(inner_code.split())
                except Exception as e:
                    print(f"[WARNING] Parsing error: {e}, falling back to simple split", file=sys.stderr)
                    tokens.extend(inner_code.split())

            if closing_tag:
                tokens.append(closing_tag)

        except Exception as e:
            print(f"[ERROR] Unexpected error in tokenize_diff_line: {e}", file=sys.stderr)
            return []

        return tokens

    def _aggressive_prefilter(self, word_freqs):
        """Aggressively filter vocabulary before EM training."""
        print("🔥 Aggressive pre-filtering...", file=sys.stderr)

        # Remove very rare words (increase threshold)
        filtered = {w: f for w, f in word_freqs.items() if f >= 5}  # Increase from 2

        # Remove very long words (likely noise)
        filtered = {w: f for w, f in filtered.items() if len(w) <= 20}

        # Remove words with unusual characters
        filtered = {w: f for w, f in filtered.items()
            if not any(ord(c) > 127 for c in str(w))}

        # Keep only top N most frequent words for EM training
        MAX_WORDS_FOR_EM = 1000000  # Reduce to 50K from 2.2M!
        if len(filtered) > MAX_WORDS_FOR_EM:
            sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
            filtered = dict(sorted_words[:MAX_WORDS_FOR_EM])

        print(f"✨ Filtered to {len(filtered)} words (from {len(word_freqs)})", file=sys.stderr)
        return filtered

    def _extract_code_tokens(self, node, source_code, depth=0):
        """Extract tokens from tree-sitter node with recursion depth limit."""
        # Fix #1: Prevent infinite recursion
        if depth > self.max_recursion:
            return []

        tokens = []
        try:
            if node.type == 'identifier':
                # parent_type = node.parent.type if node.parent else None
                # if parent_type == 'function_definition':
                #     prefix, suffix = '<FUNC_', '_FUNC>'
                # elif parent_type == 'class_definition':
                #     prefix, suffix = '<CLASS_', '_CLASS>'
                # elif parent_type in ('import_statement', 'import_from_statement'):
                #     prefix, suffix = '<MOD_', '_MOD>'
                # else:
                #     prefix, suffix = '<VAR_', '_VAR>'

                identifier_text = get_node_text(node, source_code)
                split_parts = self._split_identifier(identifier_text)
                # split_parts = [prefix + part + suffix for part in split_parts]
                tokens.extend(split_parts)

            elif node.type in ('integer', 'float'):
                tokens.append(get_node_text(node, source_code))

            elif node.type in ('def', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'from', 'as'):
                tokens.append(node.type)

            elif node.type == 'comment':
                comment_text = get_node_text(node, source_code).lstrip('#').strip()
                if comment_text and re.match(r'^\s*[a-zA-Z_]\w*\s*[=+]', comment_text):
                    try:
                        tree = _get_parser().parse(bytes(comment_text, "utf8"))
                        # Fix #1: Increase depth to prevent excessive recursion
                        tokens.extend(self._extract_code_tokens(tree.root_node, bytes(comment_text, "utf8"), depth=depth+1))
                    except Exception as e:
                        print(f"[WARNING] Failed to parse comment: {e}", file=sys.stderr)
                        tokens.extend(re.split(r'\s+', comment_text))
                else:
                    tokens.extend(re.split(r'\s+', comment_text))

            elif node.type in ('operator', 'punctuation', 'keyword'):
                tokens.append(get_node_text(node, source_code))

            elif node.type == 'string':
                string_text = get_node_text(node, source_code)
                stripped_text = self._strip_string_quotes(string_text)
                if stripped_text:
                    parts = self._split_string_content(stripped_text)
                    subworded_parts = []
                    for part in parts:
                        if len(part) > 5 and self.trained_model is not None:
                            encoded, _ = self._encode_word(part, self.trained_model)
                            subworded_parts.extend(encoded)
                        else:
                            subworded_parts.append(part)
                    tokens.extend(['<STR>'] + subworded_parts + ['</STR>'])

            # Recursively process children
            for child in node.children:
                tokens.extend(self._extract_code_tokens(child, source_code, depth=depth+1))

        except Exception as e:
            print(f"[WARNING] Error extracting tokens from node type {node.type}: {e}", file=sys.stderr)

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
        # Fix #9: Validate model is not empty
        if not model:
            print(f"[WARNING] Model is empty, returning UNK for word: {word}", file=sys.stderr)
            return (["<UNK>"], None)

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
            return (["<UNK>"], None)

        score = segmentation["score"]
        tokens = []
        pos = len(word)
        while pos > 0:
            start = best_segmentations[pos]["start"]
            if start is None:
                print(f"[WARNING] Incomplete segmentation for word: {word}", file=sys.stderr)
                return (["<UNK>"], None)
            tokens.insert(0, word[start:pos])
            pos = start
        return (tokens, score)

    def _encode_word_batch_gpu(self, words, model, device=None):
        """Encode multiple words in parallel on GPU using vectorized DP."""
        # Fix #6: Use local variable instead of mutating self.device
        if device is None:
            device = self.device

        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("⚠️ CUDA not available, falling back to CPU", file=sys.stderr)

        if not words:
            return []

        results = []
        for batch_start in range(0, len(words), self.batch_size):
            batch_words = words[batch_start:batch_start + self.batch_size]
            results.extend(self._encode_batch_vectorized(batch_words, model, device))

        return results

    def _encode_batch_vectorized(self, words, model, device):
        """Vectorized DP for batch of words."""
        # Fix #9: Validate model
        if not model:
            return [(["<UNK>"], None) for _ in words]

        batch_size = len(words)
        max_len = max(len(w) for w in words) if words else 1

        dp_scores = torch.full((batch_size, max_len + 1), float('inf'),
                               dtype=torch.float32, device=device)
        dp_scores[:, 0] = 0
        # Fix #4: Initialize backpointer with proper sentinels
        dp_backptr = np.full((batch_size, max_len + 1), -1, dtype=np.int32)

        # OPTIMIZATION: Pre-compute all valid tokens for this batch
        valid_tokens_cache = {}
        for start_idx in range(max_len):
            for word_idx, word in enumerate(words):
                for end_idx in range(start_idx + 1, min(start_idx + 9, len(word) + 1)):  # Max token length = 8
                    token = word[start_idx:end_idx]
                    if token in model:
                        key = (word_idx, start_idx, end_idx)
                        valid_tokens_cache[key] = model[token]

        # Use cached lookups
        for (word_idx, start_idx, end_idx), token_score in valid_tokens_cache.items():
            prev_score = dp_scores[word_idx, start_idx]
            if torch.isinf(prev_score):
                continue
            new_score = prev_score + token_score
            if new_score < dp_scores[word_idx, end_idx]:
                dp_scores[word_idx, end_idx] = new_score
                dp_backptr[word_idx, end_idx] = start_idx

        token_lens = sorted(set(len(t) for t in model.keys())) if model else []

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

            # Fix #4: Handle incomplete segmentation properly
            if np.isinf(final_score):
                results.append((["<UNK>"], None))
                continue

            tokens = []
            pos = word_len
            while pos > 0:
                start = dp_backptr[i, pos]
                if start == -1:
                    print(f"[WARNING] Failed to backtrack for word: {word}", file=sys.stderr)
                    results.append((["<UNK>"], None))
                    break
                tokens.insert(0, word[start:pos])
                pos = start
            else:
                results.append((tokens, final_score))

        return results

    def _compute_loss_gpu(self, model, word_freqs, device='cuda', batch_size=None):
        """GPU-accelerated loss computation."""
        if batch_size is None:
            batch_size = self.batch_size

        if not torch.cuda.is_available():
            device = 'cpu'

        # Fix #9: Validate inputs
        if not model or not word_freqs:
            return 0.0

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

    def _compute_scores_gpu(self, model, word_freqs, device='cuda'):
        """GPU-accelerated exact score computation with safe model handling."""
        # Fix #2: Don't mutate the model; compute scores safely
        print(f"Computing exact scores on {device.upper()}...", file=sys.stderr)
        if not torch.cuda.is_available():
            device = 'cpu'

        model_loss = self._compute_loss_gpu(model, word_freqs, device)
        scores = {}
        tokens_to_score = [t for t in model.keys() if len(t) > 1]

        for token in tqdm(tokens_to_score, desc="Scoring tokens", ncols=80):
            # Fix #2: Create a copy without this token instead of mutating original
            model_without_token = {t: model[t] for t in model if t != token}
            try:
                loss_without = self._compute_loss_gpu(model_without_token, word_freqs, device)
                scores[token] = loss_without - model_loss
            except Exception as e:
                print(f"[ERROR] Failed to score token {token}: {e}", file=sys.stderr)
                scores[token] = 0.0

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

    def _fast_approximate_prune(self, model, word_freqs):
        """Fast pruning using heuristics only, no exact scoring."""
        print("⚡ Fast approximate pruning...", file=sys.stderr)

        while len(model) > self.target_size:
            # Use only heuristic scores (no exact computation)
            heuristic_scores = {}
            for token in model.keys():
                if len(token) > 1:
                    # Score based on: negative log prob / length + rarity penalty
                    freq_in_words = sum(1 for w in word_freqs if token in str(w))
                    heuristic_scores[token] = -model[token] / len(token) + (1.0 / max(freq_in_words, 1))

            # Remove worst 10%
            num_to_remove = max(1, int(len(model) * 0.1))
            sorted_tokens = sorted(heuristic_scores.items(), key=lambda x: x[1])

            for token, _ in sorted_tokens[:num_to_remove]:
                model.pop(token, None)

            print(f"  Pruned to {len(model)} tokens", file=sys.stderr)

        return model

    def _train_unigram_model(self, word_freqs, model, num_iters=5, device='cuda'):
        """GPU-accelerated EM training."""
        if not torch.cuda.is_available():
            device = 'cpu'
            print("⚠️ CUDA not available, using CPU", file=sys.stderr)

        # Fix #10: Validate word frequencies are positive
        for word, freq in word_freqs.items():
            if freq <= 0:
                print(f"[WARNING] Invalid frequency for word '{word}': {freq}, setting to 1", file=sys.stderr)
                word_freqs[word] = 1

        words = list(word_freqs.keys())
        freqs = torch.tensor([word_freqs[w] for w in words], dtype=torch.float32, device=device)
        print(f"Preparing GPU tensors for {len(words)} words...", file=sys.stderr)

        for iteration in range(num_iters):
            print(f"🧩 EM Iteration {iteration+1}/{num_iters}", file=sys.stderr)
            token_counts = defaultdict(float)
            total_loss = 0.0

            for batch_start in tqdm(range(0, len(words), self.batch_size),
                                    desc="E-step (GPU)", ncols=80, leave=False):
                batch_words = words[batch_start:batch_start + self.batch_size]
                batch_freqs = freqs[batch_start:batch_start + self.batch_size]
                batch_results = self._encode_word_batch_gpu(batch_words, model, device)

                for (tokens, score), freq in zip(batch_results, batch_freqs.cpu().numpy()):
                    if score is not None:
                        total_loss += float(freq) * score
                        for tok in tokens:
                            token_counts[tok] += float(freq)

            total_count = sum(token_counts.values())
            if total_count > 0:
                model = {tok: -log(cnt / total_count) for tok, cnt in token_counts.items()}
            else:
                print("[ERROR] No valid tokens after E-step", file=sys.stderr)
                break

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
        # Fix #5: Ensure target_size doesn't exceed available tokens
        target_size = min(int(self.target_size * 2), len(scored))
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

        word_freqs = defaultdict(int)
        for token_list in tqdm(all_token_lists, desc="Counting frequencies", ncols=80):
            for token in token_list:
                word_freqs[token] += 1

        del all_token_lists
        gc.collect()
        word_freqs = self._aggressive_prefilter(word_freqs)

        MAX_SUBWORD_LEN = 10
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

        # Fix #10: Validate we have tokens
        if total_sum <= 0:
            print("[ERROR] No valid tokens after subword extraction", file=sys.stderr)
            return

        token_freqs = self._heuristic_preprune(token_freqs)
        model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self._train_unigram_model(word_freqs, model, num_iters=5, device=device)
        self.trained_model = model
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
        """Convert text to numerical IDs."""
        tokens = self.tokenize(text)

        # Fix #3: Use proper model instead of dummy one
        if self.trained_model is None:
            print("[WARNING] trained_model not set, using stoi for encoding", file=sys.stderr)
            return [self.stoi.get(t, self.stoi["<UNK>"]) for t in tokens]

        subword_tokens = []
        for token in tokens:
            if token in self.stoi:
                subword_tokens.append(token)
            else:
                try:
                    encoded, _ = self._encode_word(token, self.trained_model)
                    subword_tokens.extend(encoded)
                except Exception as e:
                    print(f"[WARNING] Failed to encode token '{token}': {e}", file=sys.stderr)
                    subword_tokens.append("<UNK>")

        return [self.stoi.get(t, self.stoi["<UNK>"]) for t in subword_tokens]

    def numericalize_batch(self, texts):
        """Convert batch of texts to numerical IDs."""
        numericalized_data = []
        for text in tqdm(texts, desc="Numericalizing sentences", unit="sentence"):
            try:
                numericalized_data.append(self.numericalize(text))
            except Exception as e:
                print(f"[ERROR] Failed to numericalize text: {e}", file=sys.stderr)
                # Return empty sequence on error
                numericalized_data.append([self.stoi["<PAD>"]])
        return numericalized_data
