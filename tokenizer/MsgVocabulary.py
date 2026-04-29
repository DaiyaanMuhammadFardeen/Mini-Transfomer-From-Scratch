"""
Enhanced message vocabulary builder that incorporates information from multiple embedding types.
This vocabulary builder creates tokens that are specifically designed to work well with
the various embedding layers in the model, especially for commit messages.
"""
import re
import sys
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import json

# Import programming terms
try:
    from tokenizer.programming_terms import extract_programming_terms, create_programming_tokens
except ImportError:
    from programming_terms import extract_programming_terms, create_programming_tokens


class TrieNode:
    __slots__ = ('children', 'token_id')
    def __init__(self):
        self.children = {}
        self.token_id = None   # None = not a complete token

class VocabTrie:
    """
    Prefix trie built from vocabulary for O(L) max-match tokenization.
    L = length of the longest matching token.
    """
    def __init__(self, stoi: dict):
        self.root = TrieNode()
        self.unk_id = stoi.get('<UNK>', 1)
        # Insert all non-special tokens
        for word, idx in stoi.items():
            if word.startswith('<') and word.endswith('>'):
                continue   # skip special tokens
            node = self.root
            for ch in word.lower():
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.token_id = idx

    def tokenize(self, text: str) -> list[int]:
        import re
        ids = []
        for word in re.findall(r'\w+|[^\w\s]|\s+', text):
            word_lower = word.lower()
            i = 0
            while i < len(word_lower):
                node = self.root
                last_match_end = -1
                last_match_id  = None
                j = i
                while j < len(word_lower) and word_lower[j] in node.children:
                    node = node.children[word_lower[j]]
                    j += 1
                    if node.token_id is not None:
                        last_match_end = j
                        last_match_id  = node.token_id
                if last_match_id is not None:
                    ids.append(last_match_id)
                    i = last_match_end
                else:
                    ids.append(self.unk_id)
                    i += 1
        return ids


class MsgVocabulary:
    """
    Enhanced BPE tokenizer that incorporates embedding-specific tokens for commit messages.
    Uses chunked parallel processing to maximize CPU utilization.
    """

    def __init__(self, vocab_size: int = 10000, min_frequency: int = 500):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.n_workers = max(1, cpu_count())

        self.merges = {}
        self.merge_order = []
        self.stoi = {}
        self.itos = {}

        # Enhanced special tokens relevant to different embeddings
        self.special_tokens = [
            '<PAD>', '<UNK>', '<SOS>', '<EOS>',
            # Change type tokens
            '<BUG_FIX>', '<FEATURE_ADD>', '<REFACTOR>', '<OPTIMIZATION>',
            '<DOC_UPDATE>', '<TEST_ADD>', '<CONFIG_CHANGE>',
            # Code style tokens
            '<CAMEL_CASE>', '<SNAKE_CASE>', '<PASCAL_CASE>', '<CONSTANT_CASE>',
            '<INDENT_STYLE>', '<LINE_LENGTH>', '<COMMENT_STYLE>',
            # Dependency tokens
            '<IMPORT>', '<EXPORT>', '<DEPENDENCY>', '<LIBRARY>',
            '<PACKAGE>', '<MODULE>', '<FRAMEWORK>',
            # Security tokens
            '<SECURITY_FIX>', '<VULNERABILITY>', '<PERMISSION>', '<AUTH>',
            # Performance tokens
            '<PERFORMANCE>', '<BOTTLENECK>', '<OPTIMIZATION>',
            # Error/Exception tokens
            '<ERROR>', '<EXCEPTION>', '<TRY_CATCH>', '<RAISE>',
            # API tokens
            '<API_CHANGE>', '<ENDPOINT>', '<PARAMETER>', '<RESPONSE>',
            # Complexity tokens
            '<COMPLEXITY>', '<CYCLOMATIC>', '<COGNITIVE>',
            # Domain-specific tokens
            '<DOMAIN>', '<BUSINESS_LOGIC>', '<UI_CHANGE>', '<BACKEND>',
            # Testing tokens
            '<TEST>', '<UNIT_TEST>', '<INTEGRATION_TEST>', '<MOCK>',
            # Syntactic tokens
            '<FUNCTION>', '<CLASS>', '<METHOD>', '<VARIABLE>',
            '<LOOP>', '<CONDITIONAL>', '<EXPRESSION>', '<STATEMENT>',
            # Temporal tokens
            '<TIMESTAMP>', '<VERSION>', '<RELEASE>', '<DEPRECATED>',
            # Collaborative tokens
            '<REVIEWED>', '<APPROVED>', '<WIP>', '<CO_AUTHOR>'
        ]

        self._initialize_special_tokens()
        self.is_trained = False
        self._trie = None   # Built lazily on first numericalize call

        print(f"\033[94m⚡ Enhanced BPE ({self.n_workers} workers) with embedding-aware tokens\033[0m", file=sys.stderr)

    def _initialize_special_tokens(self):
        for idx, token in enumerate(self.special_tokens):
            self.stoi[token] = idx
            self.itos[idx] = token

    def _extract_change_type_tokens(self, text: str) -> List[str]:
        """Extract tokens related to change types from commit message"""
        change_patterns = {
            # Bug fix related
            r'\b(fix|bug|error|issue|patch|correct|resolve|solve)\b': '<BUG_FIX>',
            r'\b(bugfix|hotfix|fixup|debug)\b': '<BUG_FIX>',
            # Feature addition related
            r'\b(add|feature|implement|new|create|build|develop)\b': '<FEATURE_ADD>',
            r'\b(feature_|new_|add_)\w+': '<FEATURE_ADD>',
            # Refactoring related
            r'\b(refactor|refactoring|restructure|reorganize|rework|cleanup|clean)\b': '<REFACTOR>',
            r'\b(refactor_|_refactor)\w*': '<REFACTOR>',
            # Optimization related
            r'\b(optimize|optimization|perf|performance|speed|fast|efficient)\b': '<OPTIMIZATION>',
            r'\b(opt_|perf_|speed_)\w+': '<OPTIMIZATION>',
            # Documentation related
            r'\b(doc|documentation|readme|comment|explain|update|docs)\b': '<DOC_UPDATE>',
            # Test related
            r'\b(test|unittest|pytest|assert|mock|spec|testing)\b': '<TEST_ADD>',
            r'\b(test_|_test|spec_|_spec)\w+': '<TEST_ADD>',
        }

        tokens = []
        for pattern, token in change_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                tokens.append(token)

        return tokens

    def _extract_code_style_tokens(self, text: str) -> List[str]:
        """Extract tokens related to code style from commit message"""
        style_tokens = []

        # Check for naming conventions in commit message
        if re.search(r'[a-z]+(?:[A-Z][a-z]*)+', text):  # camelCase
            style_tokens.append('<CAMEL_CASE>')
        if re.search(r'[a-z]+(?:_[a-z]+)+', text):  # snake_case
            style_tokens.append('<SNAKE_CASE>')
        if re.search(r'[A-Z]+(?:[A-Z][a-z]*)+', text):  # PascalCase
            style_tokens.append('<PASCAL_CASE>')
        if re.search(r'[A-Z]+(?:_[A-Z]+)+', text):  # CONSTANT_CASE
            style_tokens.append('<CONSTANT_CASE>')

        # Check for common style patterns in commit message
        if re.search(r'\bdef\s+\w+\s*\(', text):  # Function definitions
            style_tokens.append('<FUNCTION>')
        if re.search(r'\bclass\s+\w+', text):  # Class definitions
            style_tokens.append('<CLASS>')
        if re.search(r'\s{4}|\t', text):  # Indentation style
            style_tokens.append('<INDENT_STYLE>')
        if re.search(r'#.*', text):  # Comments
            style_tokens.append('<COMMENT_STYLE>')

        return style_tokens

    def _extract_security_tokens(self, text: str) -> List[str]:
        """Extract tokens related to security from commit message"""
        security_patterns = {
            r'\b(auth|authenticate|authorization|authz|login|logout|session|token|jwt|oauth)\b': '<AUTH>',
            r'\b(security|secure|encrypt|decrypt|cipher|crypto|hash|password|key)\b': '<SECURITY_FIX>',
            r'\b(vuln|vulnerability|exploit|attack|secure|insecure|permission|privilege)\b': '<VULNERABILITY>',
            r'\b(permission|access|grant|revoke|admin|root|sudo)\b': '<PERMISSION>',
        }

        tokens = []
        for pattern, token in security_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                tokens.append(token)

        return tokens

    def _extract_dependency_tokens(self, text: str) -> List[str]:
        """Extract tokens related to dependencies from commit message"""
        dep_patterns = {
            r'\b(import|from|include|require|using)\b': '<IMPORT>',
            r'\b(export|module|package|library|framework)\b': '<LIBRARY>',
            r'\b(dependency|depend|requires?|requirement)\b': '<DEPENDENCY>',
            r'\b(pip|npm|yarn|gradle|maven|poetry|conda)\b': '<PACKAGE>',
        }

        tokens = []
        for pattern, token in dep_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                tokens.append(token)

        return tokens

    def _extract_performance_tokens(self, text: str) -> List[str]:
        """Extract tokens related to performance from commit message"""
        perf_patterns = {
            r'\b(performance|perf|speed|fast|slow|efficient|optimize|optimization)\b': '<PERFORMANCE>',
            r'\b(bottleneck|memory|cpu|gpu|latency|throughput|bandwidth)\b': '<BOTTLENECK>',
            r'\b(cache|buffer|memory|allocation|gc|garbage collection)\b': '<OPTIMIZATION>',
        }

        tokens = []
        for pattern, token in perf_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                tokens.append(token)

        return tokens

    def _extract_error_tokens(self, text: str) -> List[str]:
        """Extract tokens related to errors and exceptions from commit message"""
        error_patterns = {
            r'\b(try|except|catch|finally|raise|throw|exception|error)\b': '<TRY_CATCH>',
            r'\b(error|fail|failure|exception|bug|crash|abort|terminate)\b': '<ERROR>',
            r'\b(AssertionError|ValueError|TypeError|RuntimeError|Exception)\b': '<EXCEPTION>',
            r'\b(warning|warn|log|debug|trace|info|error)\b': '<ERROR>',
        }

        tokens = []
        for pattern, token in error_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                tokens.append(token)

        return tokens

    def _extract_api_tokens(self, text: str) -> List[str]:
        """Extract tokens related to API changes from commit message"""
        api_patterns = {
            r'\b(api|endpoint|route|url|path|request|response|http|rest|graphql)\b': '<API_CHANGE>',
            r'\b(endpoint|route|path|uri|url)\b': '<ENDPOINT>',
            r'\b(param|parameter|query|header|body|payload|json|xml)\b': '<PARAMETER>',
            r'\b(response|status|code|success|error|ok|200|404|500)\b': '<RESPONSE>',
        }

        tokens = []
        for pattern, token in api_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                tokens.append(token)

        return tokens

    def _extract_complexity_tokens(self, text: str) -> List[str]:
        """Extract tokens related to code complexity from commit message"""
        complexity_patterns = {
            r'\b(complex|complexity|nested|deep|simple|easy|hard|difficult)\b': '<COMPLEXITY>',
            r'\b(cyclomatic|branch|if|elif|else|switch|case)\b': '<CYCLOMATIC>',
            r'\b(cognitive|thinking|understand|readable|clear|obvious)\b': '<COGNITIVE>',
        }

        tokens = []
        for pattern, token in complexity_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                tokens.append(token)

        return tokens

    def _extract_domain_tokens(self, text: str) -> List[str]:
        """Extract tokens related to domain-specific concepts from commit message"""
        domain_patterns = {
            r'\b(domain|business|logic|model|entity|service|repository|controller)\b': '<DOMAIN>',
            r'\b(business|logic|rule|workflow|process|procedure)\b': '<BUSINESS_LOGIC>',
            r'\b(ui|user|interface|frontend|view|component|display|render)\b': '<UI_CHANGE>',
            r'\b(backend|server|api|database|model|engine|core|service)\b': '<BACKEND>',
        }

        tokens = []
        for pattern, token in domain_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                tokens.append(token)

        return tokens

    def _extract_testing_tokens(self, text: str) -> List[str]:
        """Extract tokens related to testing from commit message"""
        testing_patterns = {
            r'\b(test|unit|integration|e2e|mock|stub|spy|fake)\b': '<TEST>',
            r'\b(unittest|pytest|nose|mocha|jest|junit|testng)\b': '<UNIT_TEST>',
            r'\b(integration|integration_test|end_to_end|e2e)\b': '<INTEGRATION_TEST>',
            r'\b(mock|stub|fake|spy|double|test_double)\b': '<MOCK>',
        }

        tokens = []
        for pattern, token in testing_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                tokens.append(token)

        return tokens

    def _extract_syntactic_tokens(self, text: str) -> List[str]:
        """Extract tokens related to syntactic patterns from commit message"""
        syntactic_patterns = {
            r'\b(def|function|func|method)\b': '<FUNCTION>',
            r'\b(class|type|struct|interface)\b': '<CLASS>',
            r'\b(def\s+\w+\s*\()': '<METHOD>',
            r'\b(var|let|const|variable|attr|field)\b': '<VARIABLE>',
            r'\b(for|while|do|loop|foreach)\b': '<LOOP>',
            r'\b(if|elif|else|switch|case|ternary)\b': '<CONDITIONAL>',
            r'\b(return|yield|break|continue)\b': '<STATEMENT>',
        }

        tokens = []
        for pattern, token in syntactic_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                tokens.append(token)

        return tokens

    def _extract_programming_tokens(self, text: str) -> List[str]:
        """Extract programming language, framework, and technical terms from commit message."""
        # Extract programming terms from the text
        terms = extract_programming_terms(text)
        # Convert them to special tokens
        tokens = create_programming_tokens(terms)
        return tokens

    def _extract_embedding_tokens(self, text: str) -> List[str]:
        """Extract all embedding-specific tokens from commit message"""
        tokens = []
        tokens.extend(self._extract_change_type_tokens(text))
        tokens.extend(self._extract_code_style_tokens(text))
        tokens.extend(self._extract_security_tokens(text))
        tokens.extend(self._extract_dependency_tokens(text))
        tokens.extend(self._extract_performance_tokens(text))
        tokens.extend(self._extract_error_tokens(text))
        tokens.extend(self._extract_api_tokens(text))
        tokens.extend(self._extract_complexity_tokens(text))
        tokens.extend(self._extract_domain_tokens(text))
        tokens.extend(self._extract_testing_tokens(text))
        tokens.extend(self._extract_syntactic_tokens(text))
        tokens.extend(self._extract_programming_tokens(text))  # NEW: Add programming terms

        return list(set(tokens))  # Remove duplicates

    def pre_tokenize(self, text: str) -> List[str]:
        """Enhanced pre-tokenization that extracts embedding tokens and normalizes text."""
        # Extract embedding-specific tokens first
        embedding_tokens = self._extract_embedding_tokens(text)

        # Normalize and split the main text
        text = ' '.join(text.split())
        pattern = r"""
            (?:[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[a-z]+|[A-Z]+)|
            (?:\w+(?:_\w+)+)|
            (?:\d+\.\d+(?:\.\d+)*)|
            (?:[a-zA-Z0-9]+(?:[/.][a-zA-Z0-9]+)+)|
            (?:[#@]\w+)|
            (?:\w+)|
            (?:[^\w\s])
        """
        main_tokens = re.findall(pattern, text, re.VERBOSE)

        # Combine embedding tokens with main tokens
        return embedding_tokens + main_tokens

    @staticmethod
    def pre_tokenize_static(text: str) -> List[str]:
        """Static version of pre_tokenize for streaming frequency counting."""
        # Normalize and split the main text
        text = ' '.join(text.split())
        pattern = r"""
            (?:[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[a-z]+|[A-Z]+)|
            (?:\w+(?:_\w+)+)|
            (?:\d+\.\d+(?:\.\d+)*)|
            (?:[a-zA-Z0-9]+(?:[/.][a-zA-Z0-9]+)+)|
            (?:[#@]\w+)|
            (?:\w+)|
            (?:[^\w\s])
        """
        return re.findall(pattern, text, re.VERBOSE)

    @staticmethod
    def _count_pairs_chunk(word_chunk: List[Tuple[tuple, int]]) -> Dict[Tuple[str, str], int]:
        """Count pairs in a chunk of words (for parallel processing)."""
        pairs = defaultdict(int)
        for word, freq in word_chunk:
            for i in range(len(word) - 1):
                # Skip pairs involving space characters or </w> tokens
                if word[i].isspace() or word[i+1].isspace() or word[i] == '</w>' or word[i+1] == '</w>':
                    continue
                pairs[(word[i], word[i + 1])] += freq
        return dict(pairs)

    @staticmethod
    def _merge_chunk(args):
        """Merge pair in a chunk of words (for parallel processing)."""
        word_chunk, pair = args
        pair_0, pair_1 = pair
        replacement = ''.join(pair)

        result = []
        for word, freq in word_chunk:
            new_word = []
            i = 0
            while i < len(word):
                # Skip merging involving space characters or </w> tokens
                if i < len(word) - 1 and word[i] == pair_0 and word[i + 1] == pair_1:
                    # Check if either token is a space or </w>
                    if pair_0.isspace() or pair_1.isspace() or pair_0 == '</w>' or pair_1 == '</w>':
                        new_word.append(word[i])
                        i += 1
                    else:
                        new_word.append(replacement)
                        i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            result.append((tuple(new_word), freq))
        return result

    def _parallel_pair_stats(self, word_freqs: Dict[tuple, int]) -> Dict[Tuple[str, str], int]:
        """
        Compute pair statistics in parallel across word chunks.
        MASSIVE speedup for large vocabularies.
        """
        # Convert to list for chunking
        word_items = list(word_freqs.items())

        # Split into chunks for parallel processing
        chunk_size = max(100, len(word_items) // (self.n_workers * 2))
        chunks = [word_items[i:i + chunk_size] for i in range(0, len(word_items), chunk_size)]

        # Process chunks in parallel
        with Pool(processes=self.n_workers) as pool:
            chunk_results = pool.map(self._count_pairs_chunk, chunks)

        # Merge results
        all_pairs = defaultdict(int)
        for chunk_pairs in chunk_results:
            for pair, count in chunk_pairs.items():
                all_pairs[pair] += count

        return dict(all_pairs)

    def _parallel_merge(self, word_freqs: Dict[tuple, int], pair: tuple) -> Dict[tuple, int]:
        """
        Perform merge in parallel across word chunks.
        MASSIVE speedup for large vocabularies.
        """
        # Convert to list for chunking
        word_items = list(word_freqs.items())

        # Split into chunks
        chunk_size = max(100, len(word_items) // (self.n_workers * 2))
        chunks = [word_items[i:i + chunk_size] for i in range(0, len(word_items), chunk_size)]

        # Merge in parallel
        merge_fn = partial(self._merge_chunk)
        args = [(chunk, pair) for chunk in chunks]

        with Pool(processes=self.n_workers) as pool:
            chunk_results = pool.map(merge_fn, args)

        # Combine results
        new_word_freqs = {}
        for chunk_result in chunk_results:
            for word, freq in chunk_result:
                new_word_freqs[word] = freq

        return new_word_freqs

    @staticmethod
    def _process_text_batch(texts: List[str]) -> Counter:
        """Preprocess batch with embedding-aware tokenization."""
        word_counts = Counter()
        for text in texts:
            if text and text.strip():
                # Extract embedding tokens first
                embedding_tokens = []
                # Add programming tokens
                terms = extract_programming_terms(text)
                embedding_tokens.extend(create_programming_tokens(terms))

                # Add other embedding tokens
                embedding_tokens.extend([
                    '<BUG_FIX>', '<FEATURE_ADD>', '<REFACTOR>', '<OPTIMIZATION>',
                    '<DOC_UPDATE>', '<TEST_ADD>', '<CONFIG_CHANGE>', '<PERFORMANCE>',
                    '<SECURITY_FIX>', '<ERROR>', '<API_CHANGE>', '<COMPLEXITY>',
                    '<DOMAIN>', '<TEST>', '<FUNCTION>', '<CLASS>', '<VARIABLE>'
                ])

                # Check for embedding tokens in text
                for token in embedding_tokens:
                    if token.lower().replace('<', '').replace('>', '').replace('_', ' ') in text.lower():
                        word_counts[token] += 1

                # Process the actual text - preserve special tokens, lowercase rest
                text = ' '.join(text.split())
                # Preserve uppercase special tokens like <TYPE_FIX>
                parts = re.split(r'(<[^>]+>)', text)
                normalized = ''.join(p if p.startswith('<') and p.endswith('>') else p.lower() for p in parts)
                
                pattern = r"""
                    (?:[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[a-z]+|[A-Z]+)|
                    (?:\w+(?:_\w+)+)|
                    (?:\d+\.\d+(?:\.\d+)*)|
                    (?:[a-zA-Z0-9]+(?:[/.][a-zA-Z0-9]+)+)|
                    (?:[#@]\w+)|
                    (?:\w+)|
                    (?:[^\w\s])
                """
                tokens = re.findall(pattern, normalized, re.VERBOSE)
                word_counts.update(tokens)
        return word_counts

    def build_vocab(self, texts: List[str], verbose: bool = True):
        """
        Build vocabulary with embedding-aware tokenization and ultra-fast parallel BPE.
        """
        if verbose:
            print(f"\033[95m{'='*70}\033[0m", file=sys.stderr)
            print(f"\033[95m⚡ Enhanced BPE Training with Embedding Awareness\033[0m", file=sys.stderr)
            print(f"\033[95m{'='*70}\033[0m", file=sys.stderr)
            print(f"\033[96m📊 Dataset: {len(texts):,} messages\033[0m", file=sys.stderr)
            print(f"\033[96m🎯 Target vocab: {self.vocab_size:,}\033[0m", file=sys.stderr)
            print(f"\033[96m⚙️  Workers: {self.n_workers}\033[0m", file=sys.stderr)

        # Step 1: Parallel preprocessing with embedding awareness
        print(f"\n\033[94m🔄 Step 1: Preprocessing with embedding awareness...\033[0m", file=sys.stderr)
        batch_size = max(100, len(texts) // (self.n_workers * 4))
        text_batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        word_counts = Counter()
        with Pool(processes=self.n_workers) as pool:
            for batch_counter in tqdm(
                pool.imap(self._process_text_batch, text_batches),
                total=len(text_batches),
                desc="Preprocessing",
                file=sys.stderr
            ):
                word_counts.update(batch_counter)

        if verbose:
            print(f"\033[92m✓ {len(word_counts):,} unique words\033[0m", file=sys.stderr)

        # Step 2: Character conversion
        print(f"\n\033[94m🔄 Step 2: Character conversion...\033[0m", file=sys.stderr)
        word_freqs = {tuple(list(word) + ['</w>']): freq for word, freq in word_counts.items()}

        # Initialize vocabulary
        next_idx = len(self.special_tokens)
        all_chars = set()
        for word in word_freqs.keys():
            all_chars.update(word)

        for char in sorted(all_chars):
            # Skip adding space characters to vocabulary
            if char != ' ' and char != '</w>' and char not in self.stoi:
                self.stoi[char] = next_idx
                self.itos[next_idx] = char
                next_idx += 1

        if verbose:
            print(f"\033[92m✓ Initial vocab: {len(self.stoi):,}\033[0m", file=sys.stderr)

        # Step 3: ULTRA-FAST PARALLEL BPE MERGES
        print(f"\n\033[94m🔄 Step 3: Parallel BPE merges...\033[0m", file=sys.stderr)
        print(f"\033[96m⚡ Using {self.n_workers} parallel workers per operation\033[0m", file=sys.stderr)

        num_merges = self.vocab_size - len(self.stoi)
        merge_count = 0

        # Optimization: Recompute stats only when needed
        pairs_cache = None
        cache_valid = False

        with tqdm(total=num_merges, desc="Merging", file=sys.stderr) as pbar:
            for i in range(num_merges):
                # PARALLEL: Get pair statistics
                if not cache_valid:
                    pairs_cache = self._parallel_pair_stats(word_freqs)
                    cache_valid = True

                if not pairs_cache:
                    if verbose:
                        print(f"\n\033[93m⚠️  No more pairs\033[0m", file=sys.stderr)
                    break

                # Find best pair (deterministic)
                best_pair = max(pairs_cache.items(), key=lambda x: (x[1], x[0]))[0]
                best_freq = pairs_cache[best_pair]

                # Skip merging involving space characters or </w> tokens
                if best_pair[0].isspace() or best_pair[1].isspace() or best_pair[0] == '</w>' or best_pair[1] == '</w>':
                    del pairs_cache[best_pair]
                    cache_valid = False
                    continue

                if best_freq < self.min_frequency:
                    if verbose:
                        print(f"\n\033[93m⚠️  Frequency threshold\033[0m", file=sys.stderr)
                    break

                # PARALLEL: Merge operation
                word_freqs = self._parallel_merge(word_freqs, best_pair)

                # Invalidate cache (stats changed after merge)
                cache_valid = False

                # Record merge
                merged_token = ''.join(best_pair)
                self.merges[best_pair] = merged_token
                self.merge_order.append(best_pair)

                if merged_token not in self.stoi:
                    self.stoi[merged_token] = next_idx
                    self.itos[next_idx] = merged_token
                    next_idx += 1

                merge_count += 1

                # Update progress less frequently for speed
                if merge_count % 5 == 0:
                    pbar.set_postfix({
                        'freq': f"{best_freq:,}",
                        'vocab': f"{len(self.stoi):,}",
                        'words': f"{len(word_freqs):,}"
                    })
                    pbar.update(5)

                # Periodic logging
                if verbose and merge_count % 50 == 0:
                    print(f"\n\033[96m📊 {merge_count:,}/{num_merges:,} ({100*merge_count/num_merges:.1f}%)\033[0m", file=sys.stderr)
                    print(f"\033[96m   '{best_pair[0]}'+'{best_pair[1]}' → '{merged_token}' (freq: {best_freq:,})\033[0m", file=sys.stderr)
                    print(f"\033[96m   Unique word forms: {len(word_freqs):,}\033[0m", file=sys.stderr)

        # Final update
        pbar.update(merge_count % 5)

        self.is_trained = True

        if verbose:
            print(f"\n\033[95m{'='*70}\033[0m", file=sys.stderr)
            print(f"\033[92m✅ Complete!\033[0m", file=sys.stderr)
            print(f"\033[92m   • Final vocab size: {len(self.stoi):,}\033[0m", file=sys.stderr)
            print(f"\033[92m   • Merges performed: {merge_count:,}\033[0m", file=sys.stderr)
            print(f"\033[92m   • Final word forms: {len(word_freqs):,}\033[0m", file=sys.stderr)

    def build_vocab_from_frequencies(self, word_freqs: Counter, verbose: bool = True):
        """
        Build vocabulary from pre-computed word frequencies (streaming approach).
        This avoids loading the entire dataset into memory.
        
        Args:
            word_freqs: Counter mapping words to their frequencies
            verbose: Whether to print progress information
        """
        if verbose:
            print(f"\033[95m{'='*70}\033[0m", file=sys.stderr)
            print(f"\033[95m⚡ Enhanced BPE Training from Frequencies\033[0m", file=sys.stderr)
            print(f"\033[95m{'='*70}\033[0m", file=sys.stderr)
            print(f"\033[96m📊 Unique words: {len(word_freqs):,}\033[0m", file=sys.stderr)
            print(f"\033[96m🎯 Target vocab: {self.vocab_size:,}\033[0m", file=sys.stderr)
            print(f"\033[96m⚙️  Workers: {self.n_workers}\033[0m", file=sys.stderr)

        # Step 1: Character conversion
        print(f"\n\033[94m🔄 Step 1: Character conversion...\033[0m", file=sys.stderr)
        word_freqs_tuple = {tuple(list(word) + ['</w>']): freq for word, freq in word_freqs.items()}

        # Initialize vocabulary
        next_idx = len(self.special_tokens)
        all_chars = set()
        for word in word_freqs_tuple.keys():
            all_chars.update(word)

        for char in sorted(all_chars):
            # Skip adding space characters to vocabulary
            if char != ' ' and char != '</w>' and char not in self.stoi:
                self.stoi[char] = next_idx
                self.itos[next_idx] = char
                next_idx += 1

        if verbose:
            print(f"\033[92m✓ Initial vocab: {len(self.stoi):,}\033[0m", file=sys.stderr)

        # Step 2: ULTRA-FAST PARALLEL BPE MERGES
        print(f"\n\033[94m🔄 Step 2: Parallel BPE merges...\033[0m", file=sys.stderr)
        print(f"\033[96m⚡ Using {self.n_workers} parallel workers per operation\033[0m", file=sys.stderr)

        num_merges = self.vocab_size - len(self.stoi)
        merge_count = 0

        # Optimization: Recompute stats only when needed
        pairs_cache = None
        cache_valid = False

        with tqdm(total=num_merges, desc="Merging", file=sys.stderr) as pbar:
            for i in range(num_merges):
                # PARALLEL: Get pair statistics
                if not cache_valid:
                    pairs_cache = self._parallel_pair_stats(word_freqs_tuple)
                    cache_valid = True

                if not pairs_cache:
                    if verbose:
                        print(f"\n\033[93m⚠️  No more pairs\033[0m", file=sys.stderr)
                    break

                # Find best pair (deterministic)
                best_pair = max(pairs_cache.items(), key=lambda x: (x[1], x[0]))[0]
                best_freq = pairs_cache[best_pair]

                # Skip merging involving space characters or </w> tokens
                if best_pair[0].isspace() or best_pair[1].isspace() or best_pair[0] == '</w>' or best_pair[1] == '</w>':
                    del pairs_cache[best_pair]
                    cache_valid = False
                    continue

                if best_freq < self.min_frequency:
                    if verbose:
                        print(f"\n\033[93m⚠️  Frequency threshold\033[0m", file=sys.stderr)
                    break

                # PARALLEL: Merge operation
                word_freqs_tuple = self._parallel_merge(word_freqs_tuple, best_pair)

                # Invalidate cache (stats changed after merge)
                cache_valid = False

                # Record merge
                merged_token = ''.join(best_pair)
                self.merges[best_pair] = merged_token
                self.merge_order.append(best_pair)

                if merged_token not in self.stoi:
                    self.stoi[merged_token] = next_idx
                    self.itos[next_idx] = merged_token
                    next_idx += 1

                merge_count += 1

                # Update progress less frequently for speed
                if merge_count % 5 == 0:
                    pbar.set_postfix({
                        'freq': f"{best_freq:,}",
                        'vocab': f"{len(self.stoi):,}",
                        'words': f"{len(word_freqs_tuple):,}"
                    })
                    pbar.update(5)

                # Periodic logging
                if verbose and merge_count % 50 == 0:
                    print(f"\n\033[96m📊 {merge_count:,}/{num_merges:,} ({100*merge_count/num_merges:.1f}%)\033[0m", file=sys.stderr)
                    print(f"\033[96m   '{best_pair[0]}'+'{best_pair[1]}' → '{merged_token}' (freq: {best_freq:,})\033[0m", file=sys.stderr)
                    print(f"\033[96m   Unique word forms: {len(word_freqs_tuple):,}\033[0m", file=sys.stderr)

        # Final update
        pbar.update(merge_count % 5)

        self.is_trained = True

        if verbose:
            print(f"\n\033[95m{'='*70}\033[0m", file=sys.stderr)
            print(f"\033[92m✅ Complete!\033[0m", file=sys.stderr)
            print(f"\033[92m   • Final vocab size: {len(self.stoi):,}\033[0m", file=sys.stderr)
            print(f"\033[92m   • Merges performed: {merge_count:,}\033[0m", file=sys.stderr)
            print(f"\033[92m   • Final word forms: {len(word_freqs_tuple):,}\033[0m", file=sys.stderr)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize using embedding-aware BPE."""
        if not self.is_trained:
            return list(text)

        # Extract embedding tokens first
        embedding_tokens = self._extract_embedding_tokens(text)

        # Preserve special tokens (uppercase), lowercase everything else
        # This prevents <TYPE_FIX> from becoming <type_fix> and turning into UNK
        parts = re.split(r'(<[^>]+>)', text)
        normalized = ''.join(p if p.startswith('<') and p.endswith('>') else p.lower() for p in parts)
        
        # Tokenize the main text
        words = self.pre_tokenize(normalized)
        # Filter out the embedding tokens from the main text to avoid duplication
        main_tokens = [w for w in words if w not in self.special_tokens]

        tokens = embedding_tokens  # Add embedding tokens first

        for word in main_tokens:
            word_tokens = list(word) + ['</w>']

            # Apply merges
            for pair in self.merge_order:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        # Skip merging involving space characters or </w> tokens
                        if pair[0].isspace() or pair[1].isspace() or pair[0] == '</w>' or pair[1] == '</w>':
                            i += 1
                        else:
                            word_tokens = word_tokens[:i] + [''.join(pair)] + word_tokens[i + 2:]
                            i += 1
                    else:
                        i += 1

            tokens.extend(word_tokens)

        return tokens

    def max_match_numericalize(self, text: str) -> list[int]:
        """
        Word-boundary-aware max-match tokenization.
        Splits on whitespace/punctuation first, then max-matches within each word.
        This prevents 'fixindentation' from being matched if 'fixindentation' isn't in vocab.
        
        For a word like "indentation", if 'indentation' is in the vocabulary,
        it will be matched as a single token rather than split into "indent" + "ation".
        This consistently produces fewer, larger tokens.

        Algorithm:
            1. Split text into words and separators (preserving order)
            2. For each word, try exact match first
            3. If no exact match, use longest-token-first greedy matching
            4. Separators are handled individually
        """
        import re
        unk_id = self.stoi.get('<UNK>', 1)

        # Cache max token length for efficiency
        if not hasattr(self, '_max_token_len'):
            # Exclude special tokens (those wrapped in <>) from length calculation
            real_tokens = [t for t in self.stoi if not (t.startswith('<') and t.endswith('>'))]
            self._max_token_len = max((len(t) for t in real_tokens), default=1)

        # Split text into words and non-word separators, preserving order
        tokens = re.findall(r'\w+|[^\w\s]|\s+', text)
        ids = []

        for token in tokens:
            token_lower = token.lower()

            # First, check if the full word is in vocab (exact match, no splitting needed)
            if token_lower in self.stoi:
                ids.append(self.stoi[token_lower])
                continue

            # Otherwise, max-match within this word
            i = 0
            while i < len(token_lower):
                matched = False
                max_j = min(i + self._max_token_len, len(token_lower))
                for j in range(max_j, i, -1):
                    candidate = token_lower[i:j]
                    if candidate in self.stoi:
                        ids.append(self.stoi[candidate])
                        i = j
                        matched = True
                        break
                if not matched:
                    ids.append(unk_id)
                    i += 1

        return ids

    def numericalize(self, text: str) -> List[int]:
        """
        Public tokenization entrypoint.
        Uses trie-based max-match for O(L) tokenization speed.
        Falls back to character-level if not trained.
        """
        if not self.is_trained or not self.stoi:
            return [self.stoi.get(c, self.stoi.get('<UNK>', 1)) for c in text.lower()]

        # Build trie lazily on first call
        if self._trie is None:
            self._trie = VocabTrie(self.stoi)
        
        return self._trie.tokenize(text)

    def decode(self, indices: List[int]) -> str:
        """Decode indices to text."""
        tokens = [self.itos.get(idx, '<UNK>') for idx in indices]
        tokens = [t for t in tokens if t not in self.special_tokens]
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text

    def save(self, filepath: str):
        """Save vocabulary."""
        print(f"\033[96m💾 Saving to {filepath}...\033[0m", file=sys.stderr)
        data = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'merges': self.merges,
            'merge_order': self.merge_order,
            'stoi': self.stoi,
            'itos': {int(k): v for k, v in self.itos.items()},
            'special_tokens': self.special_tokens,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"\033[92m✅ Saved\033[0m", file=sys.stderr)

    @classmethod
    def load(cls, filepath: str):
        """Load vocabulary."""
        print(f"\033[96m📥 Loading from {filepath}...\033[0m", file=sys.stderr)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        vocab = cls(vocab_size=data['vocab_size'], min_frequency=data['min_frequency'])
        vocab.merges = data['merges']
        vocab.merge_order = data['merge_order']
        vocab.stoi = data['stoi']
        vocab.itos = {int(k): v for k, v in data['itos'].items()}
        vocab.special_tokens = data['special_tokens']
        vocab.is_trained = data['is_trained']
        vocab._trie = None  # Initialize trie lazily
        ##TODO: Please remove the above line once you hve retrained msgvocab

        print(f"\033[92m✅ Loaded (vocab: {len(vocab.stoi):,})\033[0m", file=sys.stderr)
        return vocab

    def __len__(self):
        return len(self.stoi)

    def __getstate__(self):
        """Custom pickle serialization."""
        state = self.__dict__.copy()
        # Don't pickle _trie, rebuild it on load
        state['_trie'] = None
        return state

    def __setstate__(self, state):
        """Custom pickle deserialization - ensures _trie exists."""
        self.__dict__.update(state)
        # Ensure _trie attribute exists for backward compatibility
        if '_trie' not in self.__dict__:
            self._trie = None
