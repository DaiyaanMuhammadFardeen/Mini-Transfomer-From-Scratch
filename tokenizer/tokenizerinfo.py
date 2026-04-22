#!/usr/bin/env python3
"""
tokenizer_diagnose.py

Comprehensive diagnostic tool for inspecting a pickled tokenizer (.pkl file).
Prints:
- Object type
- Top-level attributes
- Any nested vocabularies or models
- Vocabulary size and a few sample tokens
"""

import pickle
import sys
import types

def describe_obj(obj, indent=0, seen=None):
    """Recursively describe an object to find vocab-like dicts/lists."""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))

    prefix = "  " * indent
    obj_type = type(obj).__name__

    if isinstance(obj, dict):
        print(f"{prefix}🔹 dict with {len(obj)} entries")
        sample = list(obj.keys())[:5]
        print(f"{prefix}   sample keys: {sample}")
    elif isinstance(obj, (list, tuple, set)):
        print(f"{prefix}🔹 {obj_type} of length {len(obj)}")
    elif hasattr(obj, "__dict__"):
        print(f"{prefix}📦 {obj_type} object with attributes:")
        for attr, val in vars(obj).items():
            if isinstance(val, (dict, list, tuple, set)):
                print(f"{prefix}  • {attr}: {type(val).__name__} ({len(val)})")
            else:
                print(f"{prefix}  • {attr}: {type(val).__name__}")
    else:
        print(f"{prefix}{obj_type}")

def try_len(obj):
    """Return len(obj) if possible, else None."""
    try:
        return len(obj)
    except Exception:
        return None

def find_vocab_like(obj, path="", found=None, seen=None):
    """Recursively search for dict-like attributes that may represent a vocabulary."""
    if found is None:
        found = []
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return found
    seen.add(id(obj))

    # If it's a dict with string keys, treat it as vocab
    if isinstance(obj, dict) and all(isinstance(k, str) for k in list(obj.keys())[:10]):
        found.append((path or "root", len(obj), obj))
        return found

    # Check attributes for classes
    if hasattr(obj, "__dict__"):
        for attr, val in vars(obj).items():
            subpath = f"{path}.{attr}" if path else attr
            find_vocab_like(val, subpath, found, seen)

    # Recurse into lists/tuples
    if isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            find_vocab_like(val, f"{path}[{i}]", found, seen)

    return found

def main():
    if len(sys.argv) < 2:
        print("Usage: python tokenizer_diagnose.py <tokenizer.pkl>")
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"🔍 Loading tokenizer from: {file_path}")
    with open(file_path, "rb") as f:
        tokenizer = pickle.load(f)

    print("\n=== BASIC INFO ===")
    print("Type:", type(tokenizer))
    print("Attributes:", dir(tokenizer))

    print("\n=== STRUCTURE OVERVIEW ===")
    describe_obj(tokenizer, indent=1)

    print("\n=== SEARCHING FOR VOCAB-LIKE OBJECTS ===")
    vocab_candidates = find_vocab_like(tokenizer)
    if not vocab_candidates:
        print("⚠️  No dict-like vocabularies found.")
    else:
        for path, size, vocab in vocab_candidates:
            print(f"✅ Found possible vocab at '{path}' with size {size}")
            sample_keys = list(vocab.keys())[:10]
            print(f"   Sample tokens: {sample_keys}")

    print("\n=== EMBEDDING-SPECIFIC TOKENS ANALYSIS ===")
    # Check for embedding-specific tokens
    embedding_tokens = [
        '<BUG_FIX>', '<FEATURE_ADD>', '<REFACTOR>', '<OPTIMIZATION>',
        '<DOC_UPDATE>', '<TEST_ADD>', '<CONFIG_CHANGE>',
        '<CAMEL_CASE>', '<SNAKE_CASE>', '<PASCAL_CASE>', '<CONSTANT_CASE>',
        '<INDENT_STYLE>', '<LINE_LENGTH>', '<COMMENT_STYLE>',
        '<IMPORT>', '<EXPORT>', '<DEPENDENCY>', '<LIBRARY>',
        '<PACKAGE>', '<MODULE>', '<FRAMEWORK>',
        '<SECURITY_FIX>', '<VULNERABILITY>', '<PERMISSION>', '<AUTH>',
        '<PERFORMANCE>', '<BOTTLENECK>', '<OPTIMIZATION>',
        '<ERROR>', '<EXCEPTION>', '<TRY_CATCH>', '<RAISE>',
        '<API_CHANGE>', '<ENDPOINT>', '<PARAMETER>', '<RESPONSE>',
        '<COMPLEXITY>', '<CYCLOMATIC>', '<COGNITIVE>',
        '<DOMAIN>', '<BUSINESS_LOGIC>', '<UI_CHANGE>', '<BACKEND>',
        '<TEST>', '<UNIT_TEST>', '<INTEGRATION_TEST>', '<MOCK>',
        '<FUNCTION>', '<CLASS>', '<METHOD>', '<VARIABLE>',
        '<LOOP>', '<CONDITIONAL>', '<EXPRESSION>', '<STATEMENT>',
        '<TIMESTAMP>', '<VERSION>', '<RELEASE>', '<DEPRECATED>',
        '<REVIEWED>', '<APPROVED>', '<WIP>', '<CO_AUTHOR>'
    ]
    
    found_embedding_tokens = []
    for token in embedding_tokens:
        if token in tokenizer.stoi:
            found_embedding_tokens.append(token)
    
    print(f"✅ Found {len(found_embedding_tokens)} embedding-specific tokens:")
    for token in found_embedding_tokens[:20]:  # Show first 20
        print(f"   - {token}: {tokenizer.stoi[token]}")
    if len(found_embedding_tokens) > 20:
        print(f"   ... and {len(found_embedding_tokens) - 20} more")

    print("\n=== SUMMARY ===")
    if vocab_candidates:
        print(f"Total possible vocabularies found: {len(vocab_candidates)}")
        largest = max(vocab_candidates, key=lambda x: x[1])
        print(f"📏 Largest vocabulary at '{largest[0]}' with {largest[1]} entries")
    else:
        print("No vocabulary detected. You may have a different serialization format.")

if __name__ == "__main__":
    main()