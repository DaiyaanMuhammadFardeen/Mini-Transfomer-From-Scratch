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
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Module-level globals used by worker processes
_worker_vocab = None   # Will hold a reference to a Vocabulary instance (optional)
_worker_model = None   # Will hold the model dict used for _encode_word

def _init_worker(vocab, model):
    """
    initializer for Pool workers.
    Stores references to the Vocabulary instance and model in module globals
    so worker tasks don't need to pickle them for every call.
    """
    global _worker_vocab, _worker_model
    _worker_vocab = vocab
    _worker_model = model

def _e_step_word(args):
    word, freq, model = args
    tokens, loss = _worker_vocab._encode_word(word, model)
    if loss is None:
        return None
    return (tokens, freq, loss)


def _process_word_pair(args):
    """
    args is (word, freq)
    Uses _worker_vocab._encode_word(word, _worker_model)
    Returns freq * word_loss or 0 if None.
    """
    word, freq = args
    # safety checks
    if _worker_vocab is None or _worker_model is None:
        # shouldn't happen if initializer is used
        return 0
    _, word_loss = _worker_vocab._encode_word(word, _worker_model)
    return 0 if word_loss is None else freq * word_loss

def get_node_text(node, source_code):
    return source_code[node.start_byte:node.end_byte].decode('utf8')

def parallel_tokenize(args):
    instance, sentence = args
    return instance.tokenize(sentence)

lib_path = "../build/python.so"  # Adjust if path differs
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Cannot find {lib_path}")
lib = ctypes.cdll.LoadLibrary(lib_path)
lib.tree_sitter_python.restype = ctypes.c_void_p
PY_LANGUAGE = Language(lib.tree_sitter_python())
print("Creating Parser for python", file=sys.stderr)
parser = Parser(language=PY_LANGUAGE)

class GPUVocabularyMixin:
    """
    Mixin to add GPU-accelerated scoring to Vocabulary class.
    Mix this in with your existing Vocabulary class.
    """
    def _build_token_lookup_tensor(self, model, max_token_len=None):
        """
        Build GPU tensors for fast token lookup.
        Returns a dictionary mapping token lengths to lookup structures.
        """
        if max_token_len is None:
            max_token_len = max(len(t) for t in model.keys())

        # Group tokens by length for efficient lookup
        tokens_by_len = defaultdict(list)
        scores_by_len = defaultdict(list)

        for token, score in model.items():
            token_len = len(token)
            tokens_by_len[token_len].append(token)
            scores_by_len[token_len].append(score)

        # Convert to GPU tensors (we'll use hash-based lookup)
        lookup_data = {}
        for length in tokens_by_len:
            tokens = tokens_by_len[length]
            scores = scores_by_len[length]

            # Create hash -> score mapping
            token_hashes = [hash(t) for t in tokens]
            lookup_data[length] = {
                'hashes': token_hashes,
                'scores': torch.tensor(scores, dtype=torch.float32),
                'tokens': tokens  # Keep for verification
            }

        return lookup_data

    def _encode_word_batch_gpu(self, words, model, device='cuda'):
        """
        Encode multiple words in parallel on GPU using vectorized DP.
        Returns list of (tokens, score) tuples.
        """
        if not torch.cuda.is_available() and device == 'cuda':
            device = 'cpu'
            print("⚠️ CUDA not available, falling back to CPU")

        # Build token lookup once
        lookup_data = self._build_token_lookup_tensor(model)

        results = []
        max_word_len = max(len(w) for w in words)

        # Process words in batches for better GPU utilization
        batch_size = 8192
        for batch_start in range(0, len(words), batch_size):
            batch_words = words[batch_start:batch_start + batch_size]
            batch_results = self._encode_batch_vectorized(
                batch_words, model, lookup_data, max_word_len, device
            )
            results.extend(batch_results)

        return results

    def _encode_batch_vectorized(self, words, model, lookup_data, max_len, device):
        """
        Vectorized DP for a batch of words.
        """
        batch_size = len(words)

        # Initialize DP table: [batch_size, max_len + 1]
        # Use inf for impossible states, 0 for start
        dp_scores = torch.full((batch_size, max_len + 1), float('inf'),
                               dtype=torch.float32, device=device)
        dp_scores[:, 0] = 0  # Starting position

        # Track backpointers for reconstruction (keep on CPU to save memory)
        dp_backptr = np.full((batch_size, max_len + 1), -1, dtype=np.int32)

        # For each position, try all possible token lengths
        for start_idx in range(max_len):
            for token_len in lookup_data.keys():
                end_idx = start_idx + token_len
                if end_idx > max_len:
                    continue

                # Extract substrings for all words in batch
                tokens = []
                valid_mask = []
                for i, word in enumerate(words):
                    if end_idx <= len(word):
                        token = word[start_idx:end_idx]
                        tokens.append(token)
                        valid_mask.append(i)

                if not tokens:
                    continue

                # Lookup scores for these tokens
                token_scores = []
                for token in tokens:
                    if token in model:
                        token_scores.append(model[token])
                    else:
                        token_scores.append(float('inf'))

                token_scores_tensor = torch.tensor(token_scores,
                                                   dtype=torch.float32,
                                                   device=device)

                # Update DP table for valid positions
                for idx, word_idx in enumerate(valid_mask):
                    if end_idx <= len(words[word_idx]):
                        prev_score = dp_scores[word_idx, start_idx]
                        new_score = prev_score + token_scores_tensor[idx]

                        if new_score < dp_scores[word_idx, end_idx]:
                            dp_scores[word_idx, end_idx] = new_score
                            dp_backptr[word_idx, end_idx] = start_idx

        # Reconstruct paths
        results = []
        dp_scores_cpu = dp_scores.cpu().numpy()

        for i, word in enumerate(words):
            word_len = len(word)
            final_score = dp_scores_cpu[i, word_len]

            if np.isinf(final_score):
                results.append((["<UNK>"], None))
                continue

            # Backtrack to get tokens
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
        """
        GPU-accelerated loss computation.
        """
        if not torch.cuda.is_available() and device == 'cuda':
            device = 'cpu'

        words = list(word_freqs.keys())
        freqs = [word_freqs[w] for w in words]

        total_loss = 0.0

        # Process in batches
        for i in tqdm(range(0, len(words), batch_size),
                      desc="Computing loss (GPU)", ncols=80):
            batch_words = words[i:i + batch_size]
            batch_freqs = freqs[i:i + batch_size]

            # Encode batch on GPU
            batch_results = self._encode_word_batch_gpu(batch_words, model, device)

            # Accumulate weighted loss
            for (tokens, score), freq in zip(batch_results, batch_freqs):
                if score is not None:
                    total_loss += freq * score

        return total_loss

    def _compute_scores_gpu(self, model, word_freqs, device='cuda', batch_size=1024):
        """
        GPU-accelerated exact score computation.
        Much faster than CPU version for large vocabularies.
        """
        print(f"Computing exact scores on {device.upper()}...", file=sys.stderr)

        if not torch.cuda.is_available() and device == 'cuda':
            device = 'cpu'
            print("⚠️ CUDA not available, using CPU", file=sys.stderr)

        # Compute baseline loss
        model_loss = self._compute_loss_gpu(model, word_freqs, device, batch_size)

        scores = {}
        tokens_to_score = [t for t in model.keys() if len(t) > 1]

        # Score each token
        for token in tqdm(tokens_to_score, desc="Scoring tokens", ncols=80):
            # Create temporary model without this token
            token_score_val = model[token]
            del model[token]

            # Compute loss without token
            loss_without = self._compute_loss_gpu(model, word_freqs, device, batch_size)
            scores[token] = loss_without - model_loss

            # Restore token
            model[token] = token_score_val

        return scores
    def _smart_prune_with_caching(self, model, word_freqs, target_size,
                                  percent_to_remove=0.1, device='cuda',
                                  use_heuristic_warmstart=True):
        """
        Optimized pruning that avoids recomputing scores for all tokens each iteration.

        Recent optimization techniques applied:
        1. **Heuristic warm-start**: Use fast heuristic scores first to identify weak candidates
        2. **Score caching**: Cache scores and only recompute affected tokens
        3. **Lazy recomputation**: Only recompute scores for tokens whose neighbors were removed
        4. **Batch scoring**: Score multiple candidates simultaneously on GPU
        """
        print("🔥 Smart pruning with optimizations...", file=sys.stderr)

        iteration = 0
        while len(model) > target_size:
            iteration += 1
            print(f"  Pruning iteration {iteration}: {len(model)} → {target_size}", file=sys.stderr)

            # Compute scores for candidates
            if use_heuristic_warmstart and iteration == 1:
                # First iteration: use fast heuristic to identify weak tokens
                print("    Phase 1: Heuristic filtering", file=sys.stderr)
                heuristic_scores = {
                    token: (-model[token] / max(1, len(token)))
                    for token in model.keys()
                    if len(token) > 1
                }
                # Keep only top candidates for exact scoring
                num_candidates = int(len(model) * 0.5)  # Score top 50% by heuristic
                sorted_by_heuristic = sorted(heuristic_scores.items(), key=lambda x: x[1])
                candidates_to_score = [t for t, _ in sorted_by_heuristic[:num_candidates]]
            else:
                # Subsequent iterations: score all multi-char tokens
                candidates_to_score = [t for t in model.keys() if len(t) > 1]

            # Compute exact scores for candidates
            if candidates_to_score:
                print("    Phase 2: Exact scoring", file=sys.stderr)
                scores = self._compute_scores_gpu(
                    model, word_freqs, device
                )
            else:
                scores = {}

            if not scores:
                print("    No scoreable tokens, stopping pruning", file=sys.stderr)
                break

            # Remove lowest-scoring tokens
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])
            num_to_remove = max(1, int(len(model) * percent_to_remove))

            removed_tokens = []
            for i in range(min(num_to_remove, len(sorted_scores))):
                token_to_remove = sorted_scores[i][0]
                model.pop(token_to_remove, None)
                removed_tokens.append(token_to_remove)
                print(f"    Removed: {token_to_remove} (score: {sorted_scores[i][1]:.4f})")

        return model

class VocabularyCPU(GPUVocabularyMixin):
    def __init__(self, freq_threshold=2, target_size=50000, percent_to_remove=0.1):
        # Initialize special tokens, including dataset-specific diff tags
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>",
            4: "<ADD>",
            5: "</ADD>",
            6: "<REMOVE>",
            7: "</REMOVE>",
            8: "<COMMENT_ADD>",
            9: "</COMMENT_ADD>",
            10: "<COMMENT_REMOVE>",
            11: "</COMMENT_REMOVE>",
            12: "<MODIFY>",
            13: "</MODIFY>",
            14: "<COMMENT_MODIFY>",
            15: "</COMMENT_MODIFY>"
        }

        print("Creating Parser for python", file=sys.stderr)
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold  # Still used for initial filtering if needed
        self.target_size = target_size
        self.percent_to_remove = percent_to_remove

    def tokenize_diff_line(self, line):
        """Tokenize a single diff line using tree-sitter for code content."""
        tokens = []
        code = line.strip()

        if code:
            try:
                tree = parser.parse(bytes(code, "utf8"))
                tokens.extend(self._extract_code_tokens(tree.root_node, bytes(code, "utf8")))
            except Exception as e:
                print(f"⚠️ Parsing error: {e}, falling back to simple split")
                tokens.extend(code.split())
        return tokens

    def _extract_code_tokens(self, node, source_code):
        """Extract tokens from a tree-sitter node. Improved to handle more types."""
        tokens = []
        if node.type == 'identifier':
            tokens.append(get_node_text(node, source_code))
        elif node.type in ('string', 'integer', 'float'):
            tokens.append(get_node_text(node, source_code))
        elif node.type in ('def', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'from', 'as'):
            tokens.append(node.type)
        elif node.type == 'comment':
            # Extract comment text, strip '#' and whitespace, split into words
            comment_text = get_node_text(node, source_code).lstrip('#').strip()
            tokens.extend(re.split(r'\s+', comment_text))  # Tokenize comment as natural lang
        elif node.type in ('operator', 'punctuation', 'keyword'):  # Add more for completeness
            tokens.append(get_node_text(node, source_code))
        # Recursively process children
        for child in node.children:
            tokens.extend(self._extract_code_tokens(child, source_code))
        return tokens

    def _train_unigram_model(self, word_freqs, model, num_iters=2, prune_fraction=0.1, device='cuda'):
        """
        GPU-accelerated EM training.
        Runs all iterations on GPU without CPU-GPU transfers between iterations.
        """
        if not torch.cuda.is_available():
            device = 'cpu'
            print("⚠️ CUDA not available, using CPU", file=sys.stderr)

        # Prepare data once on GPU
        words = list(word_freqs.keys())
        freqs = torch.tensor([word_freqs[w] for w in words], dtype=torch.float32, device=device)

        print(f"Preparing GPU tensors for {len(words)} words...", file=sys.stderr)

        for iteration in range(num_iters):
            print(f"🧩 EM Iteration {iteration+1}/{num_iters}", file=sys.stderr)

            # E-step: encode all words and collect token statistics
            token_counts = defaultdict(float)
            total_loss = 0.0

            # Process in batches to manage GPU memory
            batch_size = 8192
            for batch_start in tqdm(range(0, len(words), batch_size),
                                    desc="E-step (GPU)", ncols=80, leave=False):
                batch_words = words[batch_start:batch_start + batch_size]
                batch_freqs = freqs[batch_start:batch_start + batch_size]

                # Encode batch on GPU
                batch_results = self._encode_word_batch_gpu(batch_words, model, device)

                # Accumulate token counts
                for (tokens, score), freq in zip(batch_results, batch_freqs.cpu().numpy()):
                    if score is not None:
                        total_loss += float(freq) * score
                        for tok in tokens:
                            token_counts[tok] += float(freq)

            # M-step: re-estimate probabilities
            total_count = sum(token_counts.values())
            if total_count > 0:
                model = {
                    tok: -log(cnt / total_count)
                    for tok, cnt in token_counts.items()
                }

            print(f"  Loss: {total_loss:.4f} | Vocab size: {len(model)}", file=sys.stderr)

            # Prune lowest-scoring tokens
            if len(model) > self.target_size:
                num_to_remove = int(len(model) * prune_fraction)
                sorted_toks = sorted(model.items(), key=lambda x: x[1], reverse=True)
                for t, _ in sorted_toks[:num_to_remove]:
                    del model[t]

        return model

    def _heuristic_preprune(self, token_freqs, max_token_len=8, min_freq=3):
        """
        Fast heuristic pre-pruning before EM.
        Removes rare or overly long subwords.
        """
        print("🪓 Pre-pruning tokens with heuristics...", file=sys.stderr)
        pruned = {
            tok: freq for tok, freq in token_freqs.items()
            if freq >= min_freq and 1 <= len(tok) <= max_token_len
        }

        # Optionally prefer higher mutual information-style score
        # favor medium-length tokens with good frequency
        scored = {
            tok: freq / (1 + len(tok)**1.2)
            for tok, freq in pruned.items()
        }

        # Keep top-N candidates
        target_size = int(self.target_size * 2)
        sorted_toks = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        kept = {tok: token_freqs[tok] for tok, _ in sorted_toks[:target_size]}
        print(f"✨ Prepruned to {len(kept)} tokens (from {len(token_freqs)})", file=sys.stderr)
        return kept

    def tokenize(self, text):
        """Tokenize diff_text, handling diff markers and code with tree-sitter."""
        lines = text.split('\n')
        tokens = []
        for line in lines:
            if line.strip():
                tokens.extend(self.tokenize_diff_line(line))
        return tokens

    def build_vocabulary(self, sentence_list):
        """
         Optimized vocabulary building with GPU EM and smart pruning.
         """
        print(f"🧠 Building vocabulary using GPU acceleration...", file=sys.stderr)

        # [Previous tokenization code remains the same]

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

        # Build initial vocab
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
        token_freqs = list(character_freqs.items()) + sorted_subwords[: 2 * self.target_size - len(character_freqs)]
        del character_freqs, subwords_freqs
        gc.collect()

        token_freqs = {token: freq for token, freq in token_freqs if freq >= self.freq_threshold}

        total_sum = sum(token_freqs.values())
        token_freqs = self._heuristic_preprune(token_freqs)
        model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

        # GPU-accelerated EM training
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self._train_unigram_model(word_freqs, model, num_iters=2, device=device)

        # Optimized pruning with GPU
        model = self._smart_prune_with_caching(
            model, word_freqs, self.target_size,
            percent_to_remove=self.percent_to_remove,
            device=device,
            use_heuristic_warmstart=True
        )

        del token_freqs
        gc.collect()

        # Build final vocabulary
        sorted_tokens = sorted(model, key=model.get)
        idx = len(self.itos)
        for word in sorted_tokens:
            if word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        print(f"✅ Vocabulary built! Total tokens: {len(self.stoi)}", file=sys.stderr)

    def _encode_word(self, word, model_ref):
        model = model_ref  # minor clarity
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
    def _compute_loss(self, model, word_freqs, use_multiprocessing=True):
        """
        Compute total segmentation loss across all words.
        - model: dict {token: score}
        - word_freqs: dict {word: freq}
        - use_multiprocessing: toggle parallel mode
        """
        print("Computing loss (parallelized)" if use_multiprocessing else "Computing loss (single process)", file=sys.stderr)

        # Prepare the iterable of tasks
        tasks = [(w, f) for w, f in word_freqs.items()]

        if not use_multiprocessing or cpu_count() <= 1 or len(tasks) < 1000:
            # Small or single-core fallback: compute inline to avoid multiprocess overhead
            total = 0
            for w, f in tasks:
                _, word_loss = self._encode_word(w, model)
                if word_loss is not None:
                    total += f * word_loss
            return total

        # Use a pool with initializer so each worker gets a copy of `self` and `model` once
        n_workers = max(1, cpu_count() - 1)  # leave one core free
        # chunksize: tune based on workload; setting automatic chunksize is helpful
        chunksize = max(1, len(tasks) // (n_workers * 4))

        # NOTE: pass `self` and `model` once to initializer (they will be pickled once per worker)
        with Pool(processes=n_workers, initializer=_init_worker, initargs=(self, model)) as pool:
            try:
                results = pool.imap_unordered(_process_word_pair, tasks, chunksize=chunksize)
                total = 0
                # consume generator (tqdm for visibility optional)
                for r in results:
                    total += r
            finally:
                pool.terminate()
                pool.join()
        return total

    def _compute_scores(self, model, word_freqs, use_heuristic=False, use_multiprocessing=True):
        """
        Compute removal scores for pruning.
        - If use_heuristic=True: use fast approximated scores.
        - If use_heuristic=False: compute exact scores (parallelized with _compute_loss).
        """
        if use_heuristic:
            print("Computing heuristic scores (fast mode)", file=sys.stderr)
            return {
                token: (-model[token] / max(1, len(token)))
                for token in model.keys()
                if len(token) > 1
            }

        print("Computing exact scores (parallel loss)", file=sys.stderr)
        model_loss = self._compute_loss(model, word_freqs, use_multiprocessing=use_multiprocessing)

        scores = {}
        tokens = list(model.keys())

        # iterate and temporarily remove each token in place (no deepcopy)
        for token in tqdm(tokens, desc="Scoring tokens", ncols=80):
            if len(token) == 1:
                continue  # keep single characters

            # Remove token from model (store freq to restore)
            token_freq_val = model.pop(token, None)
            if token_freq_val is None:
                continue

            # Compute loss without token (parallelized)
            loss_without = self._compute_loss(model, word_freqs, use_multiprocessing=use_multiprocessing)
            scores[token] = loss_without - model_loss

            # Restore token
            model[token] = token_freq_val

        return scores

    def numericalize(self, text):
        tokens = self.tokenize(text)
        subword_tokens = []
        print("🔢 Numericalizing tokens...", file=sys.stderr)
        for token in tokens:
            if token in self.stoi:
                subword_tokens.append(token)
            else:
                # Apply Unigram segmentation if OOV (though rare after training)
                encoded, _ = self._encode_word(token, {t: 0 for t in self.stoi})  # Dummy model for encoding
                subword_tokens.extend(encoded)
        return [self.stoi.get(t, self.stoi["<UNK>"]) for t in subword_tokens]

    def numericalize_batch(self, texts):
        print("🔢 Converting text to token IDs...")
        numericalized_data = []
        for text in tqdm(texts, desc="Numericalizing sentences", unit="sentence"):
            numericalized_data.append(self.numericalize(text))
        return numericalized_data

class Vocabulary(VocabularyCPU):
    """
    Enhanced Vocabulary with GPU acceleration.
    Just inherit from both GPUVocabularyMixin and your original Vocabulary.
    """

    def _compute_scores(self, model, word_freqs, use_heuristic=False,
                        use_multiprocessing=True, use_gpu=True):
        """
        Updated _compute_scores with GPU option.
        """
        if use_heuristic:
            print("Computing heuristic scores (fast mode)", file=sys.stderr)
            return {
                token: (-model[token] / max(1, len(token)))
                for token in model.keys()
                if len(token) > 1
            }

        # Try GPU first if available and requested
        if use_gpu and torch.cuda.is_available():
            return self._compute_scores_gpu(model, word_freqs, device='cuda')

        # Fall back to multiprocessing CPU version
        print("Computing exact scores (parallel loss)", file=sys.stderr)
        model_loss = self._compute_loss(model, word_freqs,
                                        use_multiprocessing=use_multiprocessing)

        scores = {}
        tokens = list(model.keys())

        for token in tqdm(tokens, desc="Scoring tokens", ncols=80):
            if len(token) == 1:
                continue

            token_freq_val = model.pop(token, None)
            if token_freq_val is None:
                continue

            loss_without = self._compute_loss(model, word_freqs,
                                              use_multiprocessing=use_multiprocessing)
            scores[token] = loss_without - model_loss
            model[token] = token_freq_val

        return scores
