"""
Enhanced vocabulary builder that incorporates information from multiple embedding types.
This vocabulary builder creates tokens that are specifically designed to work with
the various embedding layers in the model.
"""
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
from typing import Dict, List, Set, Tuple, Optional, Any

# Import programming terms
from programming_terms import extract_programming_terms, create_programming_tokens


class DiffVocabulary:
    """
    Enhanced vocabulary builder that creates tokens specifically designed to work with
    multiple embedding layers. It incorporates information from different embedding types
    to create a more comprehensive vocabulary.
    """
    
    def __init__(self, 
                 freq_threshold=2, 
                 target_size=50000, 
                 percent_to_remove=0.1,
                 checkpoint_dir="./tokenizer_checkpoints", 
                 batch_size=4096,
                 max_recursion_depth=5):
        # Initialize with special tokens relevant to different embeddings
        self.itos = {
            0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>",
            # Diff-specific tokens
            4: "<ADD>", 5: "</ADD>", 6: "<REMOVE>", 7: "</REMOVE>",
            8: "<COMMENT_ADD>", 9: "</COMMENT_ADD>",
            10: "<COMMENT_REMOVE>", 11: "</COMMENT_REMOVE>",
            12: "<MODIFY>", 13: "</MODIFY>",
            14: "<COMMENT_MODIFY>", 15: "</COMMENT_MODIFY>",
            16: "<STR>", 17: "</STR>",
            # Change type tokens
            18: "<BUG_FIX>", 19: "<FEATURE_ADD>", 20: "<REFACTOR>", 21: "<OPTIMIZATION>",
            22: "<DOC_UPDATE>", 23: "<TEST_ADD>", 24: "<CONFIG_CHANGE>",
            # Code style tokens
            25: "<CAMEL_CASE>", 26: "<SNAKE_CASE>", 27: "<PASCAL_CASE>", 28: "<CONSTANT_CASE>",
            29: "<INDENT_STYLE>", 30: "<LINE_LENGTH>", 31: "<COMMENT_STYLE>",
            # Dependency tokens
            32: "<IMPORT>", 33: "<EXPORT>", 34: "<DEPENDENCY>", 35: "<LIBRARY>",
            36: "<PACKAGE>", 37: "<MODULE>", 38: "<FRAMEWORK>",
            # Security tokens
            39: "<SECURITY_FIX>", 40: "<VULNERABILITY>", 41: "<PERMISSION>", 42: "<AUTH>",
            # Performance tokens
            43: "<PERFORMANCE>", 44: "<BOTTLENECK>", 45: "<OPTIMIZATION>",
            # Error/Exception tokens
            46: "<ERROR>", 47: "<EXCEPTION>", 48: "<TRY_CATCH>", 49: "<RAISE>",
            # API tokens
            50: "<API_CHANGE>", 51: "<ENDPOINT>", 52: "<PARAMETER>", 53: "<RESPONSE>",
            # Complexity tokens
            54: "<COMPLEXITY>", 55: "<CYCLOMATIC>", 56: "<COGNITIVE>",
            # Domain-specific tokens
            57: "<DOMAIN>", 58: "<BUSINESS_LOGIC>", 59: "<UI_CHANGE>", 60: "<BACKEND>",
            # Testing tokens
            61: "<TEST>", 62: "<UNIT_TEST>", 63: "<INTEGRATION_TEST>", 64: "<MOCK>",
            # Syntactic tokens
            65: "<FUNCTION>", 66: "<CLASS>", 67: "<METHOD>", 68: "<VARIABLE>",
            69: "<LOOP>", 70: "<CONDITIONAL>", 71: "<EXPRESSION>", 72: "<STATEMENT>",
            # Temporal tokens
            73: "<TIMESTAMP>", 74: "<VERSION>", 75: "<RELEASE>", 76: "<DEPRECATED>",
            # Collaborative tokens
            77: "<REVIEWED>", 78: "<APPROVED>", 79: "<WIP>", 80: "<CO_AUTHOR>"
        }
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.target_size = target_size
        self.percent_to_remove = percent_to_remove
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.max_recursion = max_recursion_depth
        self.trained_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize embedding-specific token collectors
        self.change_type_tokens = set()
        self.code_style_tokens = set()
        self.semantic_code_tokens = set()
        self.dependency_tokens = set()
        self.security_tokens = set()
        self.performance_tokens = set()
        self.error_tokens = set()
        self.api_tokens = set()
        self.complexity_tokens = set()
        self.domain_tokens = set()
        self.testing_tokens = set()
        self.syntactic_tokens = set()
        
        # Parser for tree-sitter
        self._parser = None

    def _get_parser(self):
        """Lazy load tree-sitter parser with correct path resolution."""
        if self._parser is None:
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
                self._parser = Parser(language=PY_LANGUAGE)
                # print(f"[DEBUG] Tree-sitter parser loaded successfully from {lib_path}", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] Failed to load tree-sitter: {e}", file=sys.stderr)
                raise

        return self._parser

    def get_node_text(self, node, source_code):
        """Safely extract node text with error handling."""
        try:
            return source_code[node.start_byte:node.end_byte].decode('utf8')
        except (UnicodeDecodeError, IndexError) as e:
            print(f"[WARNING] Failed to extract node text: {e}", file=sys.stderr)
            return ""

    def _extract_change_type_tokens(self, text: str) -> List[str]:
        """Extract tokens related to change types (bug fixes, features, etc.)"""
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
        """Extract tokens related to code style (naming conventions, formatting, etc.)"""
        style_tokens = []
        
        # Check for naming conventions
        if re.search(r'[a-z]+(?:[A-Z][a-z]*)+', text):  # camelCase
            style_tokens.append('<CAMEL_CASE>')
        if re.search(r'[a-z]+(?:_[a-z]+)+', text):  # snake_case
            style_tokens.append('<SNAKE_CASE>')
        if re.search(r'[A-Z]+(?:[A-Z][a-z]*)+', text):  # PascalCase
            style_tokens.append('<PASCAL_CASE>')
        if re.search(r'[A-Z]+(?:_[A-Z]+)+', text):  # CONSTANT_CASE
            style_tokens.append('<CONSTANT_CASE>')
        
        # Check for common style patterns
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
        """Extract tokens related to security"""
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
        """Extract tokens related to dependencies"""
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
        """Extract tokens related to performance"""
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
        """Extract tokens related to errors and exceptions"""
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
        """Extract tokens related to API changes"""
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
        """Extract tokens related to code complexity"""
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
        """Extract tokens related to domain-specific concepts"""
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
        """Extract tokens related to testing"""
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
        """Extract tokens related to syntactic patterns"""
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
        """Extract programming language, framework, and technical terms."""
        # Extract programming terms from the text
        terms = extract_programming_terms(text)
        # Convert them to special tokens
        tokens = create_programming_tokens(terms)
        return tokens

    def _extract_embedding_tokens(self, text: str) -> List[str]:
        """Extract all embedding-specific tokens from text"""
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
                try:
                    inner_code.encode('utf-8')
                    parser = self._get_parser()
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

    def _extract_code_tokens(self, node, source_code, depth=0):
        """Extract tokens from tree-sitter node with recursion depth limit."""
        if depth > self.max_recursion:
            return []

        tokens = []
        try:
            if node.type == 'identifier':
                identifier_text = self.get_node_text(node, source_code)
                split_parts = self._split_identifier(identifier_text)
                tokens.extend(split_parts)

            elif node.type in ('integer', 'float'):
                tokens.append(self.get_node_text(node, source_code))

            elif node.type in ('def', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'from', 'as'):
                tokens.append(node.type)

            elif node.type == 'comment':
                comment_text = self.get_node_text(node, source_code).lstrip('#').strip()
                if comment_text and re.match(r'^\s*[a-zA-Z_]\w*\s*[=+]', comment_text):
                    try:
                        tree = self._get_parser().parse(bytes(comment_text, "utf8"))
                        tokens.extend(self._extract_code_tokens(tree.root_node, bytes(comment_text, "utf8"), depth=depth+1))
                    except Exception as e:
                        print(f"[WARNING] Failed to parse comment: {e}", file=sys.stderr)
                        tokens.extend(re.split(r'\s+', comment_text))
                else:
                    tokens.extend(re.split(r'\s+', comment_text))

            elif node.type in ('operator', 'punctuation', 'keyword'):
                tokens.append(self.get_node_text(node, source_code))

            elif node.type == 'string':
                string_text = self.get_node_text(node, source_code)
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
        """Tokenize diff text with embedding-aware tokenization."""
        # Extract embedding-specific tokens first
        embedding_tokens = self._extract_embedding_tokens(text)
        
        # Then tokenize the actual diff content
        lines = text.split('\n')
        content_tokens = []
        for line in lines:
            if line.strip():
                content_tokens.extend(self.tokenize_diff_line(line))
        
        # Combine both types of tokens
        all_tokens = embedding_tokens + content_tokens
        return all_tokens

    def _encode_word(self, word, model):
        """Encode word using unigram segmentation with DP."""
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
        """Encode multiple words in parallel on GPU using vectorized DP with memory efficiency."""
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
            batch_results = self._encode_batch_vectorized(batch_words, model, device)
            results.extend(batch_results)
            
            # Clean up temporary batch results
            del batch_words, batch_results
            
            # Periodic garbage collection
            if batch_start % (self.batch_size * 10) == 0:
                gc.collect()

        return results

    def _encode_batch_vectorized(self, words, model, device):
        """Vectorized DP for batch of words."""
        if not model:
            return [(["<UNK>"], None) for _ in words]

        batch_size = len(words)
        max_len = max(len(w) for w in words) if words else 1

        dp_scores = torch.full((batch_size, max_len + 1), float('inf'),
                               dtype=torch.float32, device=device)
        dp_scores[:, 0] = 0
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
        """GPU-accelerated exact score computation with safe model handling and memory efficiency."""
        print(f"Computing exact scores on {device.upper()}...", file=sys.stderr)
        if not torch.cuda.is_available():
            device = 'cpu'

        model_loss = self._compute_loss_gpu(model, word_freqs, device)
        scores = {}
        tokens_to_score = [t for t in model.keys() if len(t) > 1]

        for token in tqdm(tokens_to_score, desc="Scoring tokens", ncols=80):
            model_without_token = {t: model[t] for t in model if t != token}
            try:
                loss_without = self._compute_loss_gpu(model_without_token, word_freqs, device)
                scores[token] = loss_without - model_loss
            except Exception as e:
                print(f"[ERROR] Failed to score token {token}: {e}", file=sys.stderr)
                scores[token] = 0.0
            
            # Clean up temporary model copy
            del model_without_token
            
            # Periodic garbage collection
            if len(scores) % 1000 == 0:
                gc.collect()

        return scores

    def _smart_prune_with_caching(self, model, word_freqs, device='cuda'):
        """Optimized pruning with heuristic warm-start and batch scoring, memory efficient."""
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
                
                # Clean up temporary variables
                del heuristic_scores, sorted_by_heuristic
                gc.collect()
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
            
            # Clean up temporary variables
            del scores, sorted_scores
            gc.collect()

        return model

    def _train_unigram_model(self, word_freqs, model, num_iters=5, device='cuda'):
        """GPU-accelerated EM training with memory-efficient processing."""
        if not torch.cuda.is_available():
            device = 'cpu'
            print("⚠️ CUDA not available, using CPU", file=sys.stderr)

        # Validate word frequencies are positive
        for word, freq in word_freqs.items():
            if freq <= 0:
                print(f"[WARNING] Invalid frequency for word '{word}': {freq}, setting to 1", file=sys.stderr)
                word_freqs[word] = 1

        words = list(word_freqs.keys())
        print(f"Preparing GPU tensors for {len(words)} words...", file=sys.stderr)

        for iteration in range(num_iters):
            print(f"🧩 EM Iteration {iteration+1}/{num_iters}", file=sys.stderr)
            token_counts = defaultdict(float)
            total_loss = 0.0

            # Process in smaller batches to manage memory
            for batch_start in tqdm(range(0, len(words), self.batch_size),
                                    desc="E-step (GPU)", ncols=80, leave=False):
                batch_words = words[batch_start:batch_start + self.batch_size]
                batch_freqs = torch.tensor([word_freqs[w] for w in batch_words], 
                                          dtype=torch.float32, device=device)
                batch_results = self._encode_word_batch_gpu(batch_words, model, device)

                for (tokens, score), freq in zip(batch_results, batch_freqs.cpu().numpy()):
                    if score is not None:
                        total_loss += float(freq) * score
                        for tok in tokens:
                            token_counts[tok] += float(freq)
                    
                    # Clean up processed batch results to free memory
                    del tokens, score
                
                # Clean up batch tensors
                del batch_words, batch_freqs, batch_results
                
                # Periodic garbage collection
                if batch_start % (self.batch_size * 10) == 0:
                    gc.collect()

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

            # Clean up intermediate variables
            del token_counts
            gc.collect()

        return model

    def _heuristic_preprune(self, token_freqs):
        """Fast heuristic pre-pruning before EM."""
        print("🪓 Pre-pruning tokens with heuristics...", file=sys.stderr)
        pruned = {tok: freq for tok, freq in token_freqs.items()
            if freq >= 3 and 1 <= len(tok) <= 8}

        scored = {tok: freq / (1 + len(tok)**1.2) for tok, freq in pruned.items()}
        # Ensure target_size doesn't exceed available tokens
        target_size = min(int(self.target_size * 2), len(scored))
        sorted_toks = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        kept = {tok: token_freqs[tok] for tok, _ in sorted_toks[:target_size]}

        print(f"✨ Prepruned to {len(kept)} tokens (from {len(token_freqs)})", file=sys.stderr)
        return kept

    def build_vocabulary(self, sentence_list):
        """Build vocabulary with embedding-aware tokenization and GPU acceleration."""
        print(f"🧠 Building multimodal vocabulary with embedding-aware tokenization...", file=sys.stderr)
        
        # Process sentence_list in chunks to manage memory
        chunk_size = 100000
        all_token_lists = []
        
        for i in tqdm(range(0, len(sentence_list), chunk_size), 
                      desc="Processing sentence chunks", 
                      total=(len(sentence_list) + chunk_size - 1) // chunk_size,
                      ncols=80):
            chunk = sentence_list[i:i + chunk_size]
            
            with Pool(cpu_count()) as pool:
                chunk_token_lists = list(
                    tqdm(
                        pool.imap(self._parallel_tokenize, [(self, s) for s in chunk]),
                        total=len(chunk),
                        desc="🔤 Tokenizing chunk",
                        ncols=80,
                        leave=False
                    )
                )
            
            all_token_lists.extend(chunk_token_lists)
            del chunk, chunk_token_lists  # Free memory
            gc.collect()

        word_freqs = defaultdict(int)
        for token_list in tqdm(all_token_lists, desc="Counting frequencies", ncols=80):
            for token in token_list:
                word_freqs[token] += 1

        del all_token_lists  # Free memory after counting
        gc.collect()
        
        # Apply aggressive pre-filtering to focus on embedding-relevant tokens
        word_freqs = self._aggressive_prefilter(word_freqs)

        MAX_SUBWORD_LEN = 10
        character_freqs = defaultdict(int)
        subwords_freqs = defaultdict(int)
        
        # Process subwords in batches to manage memory with progress tracking
        word_freqs_items = list(word_freqs.items())
        sentences_processed = 0
        for word, freq in tqdm(word_freqs_items, desc="Extracting subwords", unit="word"):
            word = str(word)
            for i in range(len(word)):
                character_freqs[word[i]] += freq
                for j in range(i + 2, min(i + MAX_SUBWORD_LEN + 1, len(word) + 1)):
                    subwords_freqs[word[i:j]] += freq
            
            sentences_processed += 1
            if sentences_processed % 10000 == 0:  # Periodic garbage collection
                gc.collect()

        sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
        token_freqs = list(character_freqs.items()) + sorted_subwords[:2 * self.target_size - len(character_freqs)]
        del character_freqs, subwords_freqs  # Free memory
        gc.collect()

        token_freqs = {token: freq for token, freq in token_freqs if freq >= self.freq_threshold}
        total_sum = sum(token_freqs.values())

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
        for word in tqdm(sorted_tokens, desc="Building final vocabulary", unit="token"):
            if word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        print(f"✅ Multimodal vocabulary built! Total tokens: {len(self.stoi)}", file=sys.stderr)

    def build_vocabulary_from_frequencies(self, word_freqs):
        """
        Build vocabulary from pre-computed word frequencies (streaming approach).
        This avoids loading the entire dataset into memory.
        
        Args:
            word_freqs: Counter or dict mapping tokens to their frequencies
        """
        print(f"🧠 Building multimodal vocabulary from pre-computed frequencies...", file=sys.stderr)
        
        # Apply aggressive pre-filtering to focus on embedding-relevant tokens
        word_freqs = self._aggressive_prefilter(word_freqs)

        MAX_SUBWORD_LEN = 10
        character_freqs = defaultdict(int)
        subwords_freqs = defaultdict(int)
        
        # Process subwords in batches to manage memory with progress tracking
        word_freqs_items = list(word_freqs.items())
        sentences_processed = 0
        for word, freq in tqdm(word_freqs_items, desc="Extracting subwords", unit="word"):
            word = str(word)
            for i in range(len(word)):
                character_freqs[word[i]] += freq
                for j in range(i + 2, min(i + MAX_SUBWORD_LEN + 1, len(word) + 1)):
                    subwords_freqs[word[i:j]] += freq
            
            sentences_processed += 1
            if sentences_processed % 10000 == 0:  # Periodic garbage collection
                gc.collect()

        sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
        token_freqs = list(character_freqs.items()) + sorted_subwords[:2 * self.target_size - len(character_freqs)]
        del character_freqs, subwords_freqs  # Free memory
        gc.collect()

        token_freqs = {token: freq for token, freq in token_freqs if freq >= self.freq_threshold}
        total_sum = sum(token_freqs.values())

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
        for word in tqdm(sorted_tokens, desc="Building final vocabulary", unit="token"):
            if word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        print(f"✅ Multimodal vocabulary built! Total tokens: {len(self.stoi)}", file=sys.stderr)

    def _parallel_tokenize(self, args):
        """Helper for parallel tokenization."""
        instance, sentence = args
        return instance.tokenize(sentence)

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
        MAX_WORDS_FOR_EM = 1000000  # Reduce to 1M from 2.2M!
        if len(filtered) > MAX_WORDS_FOR_EM:
            sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
            filtered = dict(sorted_words[:MAX_WORDS_FOR_EM])

        print(f"✨ Filtered to {len(filtered)} words (from {len(word_freqs)})", file=sys.stderr)
        return filtered

    def numericalize(self, text):
        """Convert text to numerical IDs."""
        tokens = self.tokenize(text)

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
        """Convert batch of texts to numerical IDs with memory-efficient processing."""
        numericalized_data = []
        chunk_size = 100000  # Process in smaller chunks to manage memory
        
        for i in tqdm(range(0, len(texts), chunk_size), 
                      desc="Processing numericalization chunks", 
                      total=(len(texts) + chunk_size - 1) // chunk_size,
                      ncols=80):
            chunk = texts[i:i + chunk_size]
            
            for text in chunk:
                try:
                    numericalized_data.append(self.numericalize(text))
                except Exception as e:
                    print(f"[ERROR] Failed to numericalize text: {e}", file=sys.stderr)
                    # Return empty sequence on error
                    numericalized_data.append([self.stoi["<PAD>"]])
            
            # Clean up chunk and perform garbage collection
            del chunk
            if i % (chunk_size * 5) == 0:
                gc.collect()
        
        return numericalized_data
