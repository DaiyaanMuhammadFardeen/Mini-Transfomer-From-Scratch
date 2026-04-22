import pandas as pd
import re
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

def get_node_text(node, source_code):
    return source_code[node.start_byte:node.end_byte].decode('utf-8')

def get_ordered_leaves_with_gaps(root_node, source_bytes):
    leaves = []
    
    def traverse(node):
        if not node.children:
            leaves.append(node)
            return
        for child in node.children:
            traverse(child)
    
    traverse(root_node)
    
    # Leaves are in order
    result = []
    current_pos = 0
    for leaf in leaves:
        if leaf.start_byte > current_pos:
            gap = source_bytes[current_pos:leaf.start_byte].decode('utf-8')
            result.append(gap)
        result.append(get_node_text(leaf, source_bytes))
        current_pos = leaf.end_byte
    
    if current_pos < len(source_bytes):
        gap = source_bytes[current_pos:].decode('utf-8')
        result.append(gap)
    
    return result

def process_modify_line(line):
    if not (line.startswith('<MODIFY>') and line.endswith('</MODIFY>') and ' → ' in line):
        return line
    
    content = line[len('<MODIFY>'):-len('</MODIFY>')].strip()
    if ' → ' not in content:
        return line
    
    before, after = content.split(' → ', 1)
    before = before.strip()
    after = after.strip()
    
    if not before or not after:
        return line
    
    try:
        parser = _get_parser()
        tree_before = parser.parse(bytes(before, 'utf-8'))
        tree_after = parser.parse(bytes(after, 'utf-8'))
        
        source_before = bytes(before, 'utf-8')
        source_after = bytes(after, 'utf-8')
        
        tokens1 = get_ordered_leaves_with_gaps(tree_before.root_node, source_before)
        tokens2 = get_ordered_leaves_with_gaps(tree_after.root_node, source_after)
        
        sm = difflib.SequenceMatcher(None, tokens1, tokens2)
        
        result = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                result.extend(tokens1[i1:i2])
            elif tag == 'delete':
                result.append('<REMOVE>')
                result.extend(tokens1[i1:i2])
                result.append('</REMOVE>')
            elif tag == 'insert':
                result.append('<ADD>')
                result.extend(tokens2[j1:j2])
                result.append('</ADD>')
            elif tag == 'replace':
                result.append('<REMOVE>')
                result.extend(tokens1[i1:i2])
                result.append('</REMOVE>')
                result.append('<ADD>')
                result.extend(tokens2[j1:j2])
                result.append('</ADD>')
        
        new_content = ''.join(result)
        return new_content
    except Exception as e:
        print(f"⚠️ Error processing line: {e}", file=sys.stderr)
        return line

def process_diff(diff_text):
    """Process a single diff text, splitting and processing lines."""
    lines = diff_text.split('\n')
    new_lines = [process_modify_line(line) for line in lines]
    return '\n'.join(new_lines)

def main():
    input_path = "./commitpack_cleaned.parquet"
    output_path = "./cleaned_dataset_part1.parquet"
    
    df = pd.read_parquet(input_path)
    df['diff_text'] = df['diff_text'].fillna('')
    
    # Use multiprocessing Pool to process diffs in parallel
    with Pool(processes=cpu_count()) as pool:
        processed_diffs = list(tqdm(
            pool.imap(process_diff, df['diff_text']),
            total=len(df['diff_text']),
            desc="Processing diffs",
            unit="diff"
        ))
    
    df['diff_text'] = processed_diffs
    df.to_parquet(output_path, index=False)
    
    gc.collect()
    print(f"✅ Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    main()
