import pandas as pd
import os
import sys
import ctypes
from tree_sitter import Language, Parser
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import difflib
import gc

# Lazy initialization for parser
_parser = None

# Recursion limit safety
RECURSION_SAFETY_THRESHOLD = int(sys.getrecursionlimit() * 0.9)  # 90% of max

def _get_parser():
    """Lazy load tree-sitter parser with correct path resolution."""
    global _parser
    if _parser is None:
        # Assuming the build/python.so is in the same directory structure as provided
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        lib_path = os.path.join(current_dir, "build", "python.so")

        if not os.path.exists(lib_path):
            print(f"[ERROR] Cannot find {lib_path}", file=sys.stderr)
            raise FileNotFoundError(f"Cannot find tree-sitter library at {lib_path}")

        lib = ctypes.cdll.LoadLibrary(lib_path)
        lib.tree_sitter_python.restype = ctypes.c_void_p
        PY_LANGUAGE = Language(lib.tree_sitter_python())
        _parser = Parser(language=PY_LANGUAGE)
        print(f"[DEBUG] Tree-sitter parser loaded successfully from {lib_path}", file=sys.stderr)
    return _parser

def check_recursion_depth():
    """Check if we're approaching recursion limit."""
    current_depth = len([frame for frame in sys._current_frames().values()])
    return current_depth >= RECURSION_SAFETY_THRESHOLD

def get_node_text(node, source_code):
    """Safely extract node text."""
    try:
        return source_code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"[WARNING] Failed to extract node text: {e}", file=sys.stderr)
        return ""

def get_ordered_leaves_with_gaps(root_node, source_bytes, max_depth=75):
    leaves = []
    stack = [(root_node, 0)]  # (node, depth)

    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            print(f"[WARNING] Max depth {max_depth} reached, skipping subtree", file=sys.stderr)
            continue
        if not node.children:
            leaves.append(node)
        else:
            for child in reversed(node.children):
                stack.append((child, depth + 1))

    result = []
    current_pos = 0

    for leaf in leaves:
        # Add gap before leaf, but only if it's non-empty and not just whitespace
        if leaf.start_byte > current_pos:
            try:
                gap = source_bytes[current_pos:leaf.start_byte].decode('utf-8', errors='ignore')
                if gap.strip():  # Only append if gap has non-whitespace content
                    result.append(gap)
            except Exception:
                pass

        # Add leaf text
        leaf_text = get_node_text(leaf, source_bytes)
        if leaf_text:
            result.append(leaf_text)
        current_pos = leaf.end_byte

    # Add trailing gap, but only if it's non-empty and not just whitespace
    if current_pos < len(source_bytes):
        try:
            gap = source_bytes[current_pos:].decode('utf-8', errors='ignore')
            if gap.strip():  # Only append if gap has non-whitespace content
                result.append(gap)
        except Exception:
            pass

    return result

def process_modify_line(line):
    # Check for both <MODIFY> and <COMMENT_MODIFY> tags
    if not (
        (line.startswith('<MODIFY>') and line.endswith('</MODIFY>')) or
        (line.startswith('<COMMENT_MODIFY>') and line.endswith('</COMMENT_MODIFY>'))
    ) or ' → ' not in line:
        return line

    # Determine the tag type and extract content
    if line.startswith('<MODIFY>'):
        tag_start = '<MODIFY>'
        tag_end = '</MODIFY>'
    else:  # <COMMENT_MODIFY>
        tag_start = '<COMMENT_MODIFY>'
        tag_end = '</COMMENT_MODIFY>'

    content = line[len(tag_start):-len(tag_end)]
    try:
        before, after = content.split(' → ', 1)
    except ValueError:
        print(f"[WARNING] Invalid format in {tag_start}: {line}", file=sys.stderr)
        return None

    if not before or not after:
        print(f"[WARNING] Empty before or after in {tag_start}: {line}", file=sys.stderr)
        return None

    if len(before) > 2000 or len(after) > 2000:
        print(f"[WARNING] Line too long (before: {len(before)}, after: {len(after)}) in {tag_start}, skipping", file=sys.stderr)
        return None

    # Split into lines for diffing
    tokens1 = before.splitlines()
    tokens2 = after.splitlines()

    MAX_TOKENS = 5000
    if len(tokens1) > MAX_TOKENS or len(tokens2) > MAX_TOKENS:
        print(f"[WARNING] Too many tokens ({len(tokens1)}, {len(tokens2)}) in {tag_start}, skipping", file=sys.stderr)
        return None

    # Use difflib.SequenceMatcher to compare the two sequences
    sm = difflib.SequenceMatcher(None, tokens1, tokens2, autojunk=False)
    result = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            # Keep unchanged lines
            result.extend(tokens1[i1:i2])
        elif tag == 'delete':
            # Wrap deleted lines in <REMOVE> tags
            tokens = tokens1[i1:i2]
            if any(t.strip() for t in tokens):  # Skip empty/whitespace-only
                result.append('<REMOVE>')
                result.extend(tokens)
                result.append('</REMOVE>')
        elif tag == 'insert':
            # Wrap inserted lines in <ADD> tags
            tokens = tokens2[j1:j2]
            if any(t.strip() for t in tokens):  # Skip empty/whitespace-only
                result.append('<ADD>')
                result.extend(tokens)
                result.append('</ADD>')
        elif tag == 'replace':
            # Wrap deleted lines in <REMOVE> and inserted lines in <ADD>
            tokens_before = tokens1[i1:i2]
            tokens_after = tokens2[j1:j2]
            if any(t.strip() for t in tokens_before):
                result.append('<REMOVE>')
                result.extend(tokens_before)
                result.append('</REMOVE>')
            if any(t.strip() for t in tokens_after):
                result.append('<ADD>')
                result.extend(tokens_after)
                result.append('</ADD>')

    # Join the result back into a single string
    new_content = '\n'.join(result)
    if len(new_content) > 5000:
        print(f"[WARNING] Result too long ({len(new_content)}) in {tag_start}, skipping", file=sys.stderr)
        return None

    return f'{tag_start}{new_content}{tag_end}'

def process_diff(diff_text):
    """Process a single diff text, returns None if any line fails processing."""
    try:
        # Sanity check on diff size
        if len(diff_text) > 100000:
            print(f"[WARNING] Diff too large ({len(diff_text)} chars), skipping", file=sys.stderr)
            return None

        lines = diff_text.split('\n')

        # Limit number of lines
        if len(lines) > 2500:
            print(f"[WARNING] Too many lines ({len(lines)}), skipping", file=sys.stderr)
            return None

        new_lines = []
        for line in lines:
            processed = process_modify_line(line)
            if processed is None:
                # If any line fails, mark entire diff as failed
                return None
            new_lines.append(processed)

        return '\n'.join(new_lines)

    except RecursionError:
        print(f"[ERROR] Recursion limit in diff processing", file=sys.stderr)
        return None
    except MemoryError:
        print(f"[ERROR] Memory error in diff processing", file=sys.stderr)
        gc.collect()
        return None
    except Exception as e:
        print(f"[WARNING] Unexpected error in diff: {e}", file=sys.stderr)
        return None

def process_batch(batch_data):
    """Process a batch of diffs for memory efficiency."""
    results = []
    for idx, diff_text in batch_data:
        result = process_diff(diff_text)
        results.append((idx, result))

        # Periodic garbage collection
        if len(results) % 100 == 0:
            gc.collect()

    return results

def main():
    input_path = "./rebalanced_data_3.parquet"
    output_path = "./rebalanced_data_fixed.parquet"

    print("📖 Loading dataset...", file=sys.stderr)
    df = pd.read_parquet(input_path)
    df['diff_text'] = df['diff_text'].fillna('')

    total_rows = len(df)
    print(f"📊 Total rows: {total_rows}", file=sys.stderr)

    # Process in batches for memory efficiency
    BATCH_SIZE = 500
    all_results = []

    # Prepare batches
    batches = []
    for i in range(0, len(df), BATCH_SIZE):
        batch = list(enumerate(df['diff_text'][i:i+BATCH_SIZE], start=i))
        batches.append(batch)

    print(f"🔄 Processing {len(batches)} batches with {cpu_count()} workers...", file=sys.stderr)

    # Use multiprocessing Pool with smaller chunks
    with Pool(processes=max(cpu_count(), 8)) as pool:  # Limit workers to reduce memory
        for batch_results in tqdm(
            pool.imap(process_batch, batches),
            total=len(batches),
            desc="Processing batches",
            unit="batch"
        ):
            all_results.extend(batch_results)

            # Periodic garbage collection
            if len(all_results) % 5000 == 0:
                gc.collect()

    # Separate successful and failed rows
    print("🔍 Filtering results...", file=sys.stderr)
    valid_indices = []
    processed_diffs = []

    for idx, result in all_results:
        if result is not None:
            valid_indices.append(idx)
            processed_diffs.append(result)

    removed_count = total_rows - len(valid_indices)
    print(f"❌ Removed {removed_count} rows ({removed_count/total_rows*100:.2f}%) due to processing errors", file=sys.stderr)
    print(f"✅ Kept {len(valid_indices)} rows ({len(valid_indices)/total_rows*100:.2f}%)", file=sys.stderr)

    # Create filtered dataframe
    df_filtered = df.iloc[valid_indices].copy()
    df_filtered['diff_text'] = processed_diffs

    # Clear memory
    del df
    del all_results
    gc.collect()

    # Save result
    print(f"💾 Saving to {output_path}...", file=sys.stderr)
    df_filtered.to_parquet(output_path, index=False)

    print(f"✅ Cleaned dataset saved to {output_path}")
    print(f"📊 Final dataset size: {len(df_filtered)} rows")

if __name__ == "__main__":
    # Increase recursion limit slightly but with safety checks
    sys.setrecursionlimit(2000)
    main()
