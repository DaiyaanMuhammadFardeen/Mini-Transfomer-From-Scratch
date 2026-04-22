#!/usr/bin/env python3
"""
analyze_dataset.py

Comprehensive statistics analyzer for commit dataset with diff_text and message columns.
Memory-efficient streaming analysis for large datasets.
"""

import pandas as pd
import pyarrow.parquet as pq
import argparse
import re
from collections import Counter
from tqdm import tqdm
import gc

class DatasetAnalyzer:
    def __init__(self):
        self.stats = {
            'total_rows': 0,
            'message_stats': {
                'total_messages': 0,
                'empty_messages': 0,
                'total_chars': 0,
                'total_words': 0,
                'total_lines': 0,
                'min_length': float('inf'),
                'max_length': 0,
            },
            'diff_stats': {
                'total_diffs': 0,
                'empty_diffs': 0,
                'total_chars': 0,
                'total_lines': 0,
                'min_length': float('inf'),
                'max_length': 0,
            },
            'tag_counts': Counter(),
            'line_type_counts': {
                'add_lines': 0,
                'remove_lines': 0,
                'context_lines': 0,
                'other_lines': 0,
            },
            'file_extensions': Counter(),
        }
    
    def analyze_message(self, msg):
        """Analyze a single commit message."""
        if pd.isna(msg) or not isinstance(msg, str) or msg.strip() == "":
            self.stats['message_stats']['empty_messages'] += 1
            return
        
        self.stats['message_stats']['total_messages'] += 1
        msg_len = len(msg)
        self.stats['message_stats']['total_chars'] += msg_len
        self.stats['message_stats']['min_length'] = min(self.stats['message_stats']['min_length'], msg_len)
        self.stats['message_stats']['max_length'] = max(self.stats['message_stats']['max_length'], msg_len)
        
        words = msg.split()
        self.stats['message_stats']['total_words'] += len(words)
        
        lines = msg.split('\n')
        self.stats['message_stats']['total_lines'] += len(lines)
    
    def analyze_diff(self, diff):
        """Analyze a single diff_text."""
        if pd.isna(diff) or not isinstance(diff, str) or diff.strip() == "":
            self.stats['diff_stats']['empty_diffs'] += 1
            return
        
        self.stats['diff_stats']['total_diffs'] += 1
        diff_len = len(diff)
        self.stats['diff_stats']['total_chars'] += diff_len
        self.stats['diff_stats']['min_length'] = min(self.stats['diff_stats']['min_length'], diff_len)
        self.stats['diff_stats']['max_length'] = max(self.stats['diff_stats']['max_length'], diff_len)
        
        lines = diff.split('\n')
        self.stats['diff_stats']['total_lines'] += len(lines)
        
        # Find all tags (anything in angle brackets)
        tags = re.findall(r'<(/?\w+)>', diff)
        self.stats['tag_counts'].update(tags)
        
        # Analyze line types
        for line in lines:
            line_stripped = line.strip()
            
            # Count line modifications
            if line_stripped.startswith('+') and not line_stripped.startswith('+++'):
                self.stats['line_type_counts']['add_lines'] += 1
            elif line_stripped.startswith('-') and not line_stripped.startswith('---'):
                self.stats['line_type_counts']['remove_lines'] += 1
            elif line_stripped.startswith(' '):
                self.stats['line_type_counts']['context_lines'] += 1
            else:
                self.stats['line_type_counts']['other_lines'] += 1
            
            # Extract file extensions from diff headers
            file_match = re.search(r'[ab]/.*?\.(\w+)', line)
            if file_match:
                ext = file_match.group(1).lower()
                self.stats['file_extensions'][ext] += 1
    
    def print_report(self):
        """Print a comprehensive, pretty-formatted report."""
        msg_stats = self.stats['message_stats']
        diff_stats = self.stats['diff_stats']
        
        print("\n" + "="*80)
        print("DATASET ANALYSIS REPORT".center(80))
        print("="*80)
        
        # Overall stats
        print(f"\n📊 OVERALL STATISTICS")
        print(f"{'─'*80}")
        print(f"  Total Rows:                {self.stats['total_rows']:,}")
        
        # Message statistics
        print(f"\n📝 MESSAGE STATISTICS")
        print(f"{'─'*80}")
        print(f"  Total Messages:            {msg_stats['total_messages']:,}")
        print(f"  Empty Messages:            {msg_stats['empty_messages']:,}")
        if msg_stats['total_messages'] > 0:
            print(f"  Total Characters:          {msg_stats['total_chars']:,}")
            print(f"  Total Words:               {msg_stats['total_words']:,}")
            print(f"  Total Lines:               {msg_stats['total_lines']:,}")
            print(f"  Avg Message Length:        {msg_stats['total_chars'] / msg_stats['total_messages']:.2f} chars")
            print(f"  Avg Words per Message:     {msg_stats['total_words'] / msg_stats['total_messages']:.2f}")
            print(f"  Avg Lines per Message:     {msg_stats['total_lines'] / msg_stats['total_messages']:.2f}")
            if msg_stats['min_length'] != float('inf'):
                print(f"  Min Message Length:        {msg_stats['min_length']} chars")
                print(f"  Max Message Length:        {msg_stats['max_length']} chars")
        
        # Diff statistics
        print(f"\n🔍 DIFF_TEXT STATISTICS")
        print(f"{'─'*80}")
        print(f"  Total Diffs:               {diff_stats['total_diffs']:,}")
        print(f"  Empty Diffs:               {diff_stats['empty_diffs']:,}")
        if diff_stats['total_diffs'] > 0:
            print(f"  Total Characters:          {diff_stats['total_chars']:,}")
            print(f"  Total Lines:               {diff_stats['total_lines']:,}")
            print(f"  Avg Diff Length:           {diff_stats['total_chars'] / diff_stats['total_diffs']:.2f} chars")
            print(f"  Avg Lines per Diff:        {diff_stats['total_lines'] / diff_stats['total_diffs']:.2f}")
            if diff_stats['min_length'] != float('inf'):
                print(f"  Min Diff Length:           {diff_stats['min_length']} chars")
                print(f"  Max Diff Length:           {diff_stats['max_length']} chars")
        
        # Line type statistics
        print(f"\n➕➖ CODE MODIFICATION STATISTICS")
        print(f"{'─'*80}")
        line_counts = self.stats['line_type_counts']
        total_mod_lines = line_counts['add_lines'] + line_counts['remove_lines']
        print(f"  Added Lines (+):           {line_counts['add_lines']:,}")
        print(f"  Removed Lines (-):         {line_counts['remove_lines']:,}")
        print(f"  Context Lines:             {line_counts['context_lines']:,}")
        print(f"  Other Lines:               {line_counts['other_lines']:,}")
        if line_counts['remove_lines'] > 0:
            print(f"  Add/Remove Ratio:          {line_counts['add_lines'] / line_counts['remove_lines']:.2f}")
        print(f"  Net Lines Changed:         {line_counts['add_lines'] - line_counts['remove_lines']:,}")
        
        # Tag statistics
        if self.stats['tag_counts']:
            print(f"\n🏷️  XML/HTML TAG STATISTICS")
            print(f"{'─'*80}")
            total_tags = sum(self.stats['tag_counts'].values())
            print(f"  Total Tags Found:          {total_tags:,}")
            print(f"  Unique Tags:               {len(self.stats['tag_counts'])}")
            print(f"\n  Top 30 Tags:")
            for tag, count in self.stats['tag_counts'].most_common(30):
                percentage = (count / total_tags) * 100
                print(f"    <{tag}>:".ljust(30) + f"{count:,}".rjust(15) + f" ({percentage:.2f}%)")
        
        # File extension statistics
        if self.stats['file_extensions']:
            print(f"\n📄 FILE EXTENSION STATISTICS (Python Files)")
            print(f"{'─'*80}")
            total_files = sum(self.stats['file_extensions'].values())
            print(f"  Total File References:     {total_files:,}")
            print(f"  Unique Extensions:         {len(self.stats['file_extensions'])}")
            print(f"\n  Top 20 Extensions:")
            for ext, count in self.stats['file_extensions'].most_common(20):
                percentage = (count / total_files) * 100
                print(f"    .{ext}:".ljust(30) + f"{count:,}".rjust(15) + f" ({percentage:.2f}%)")
        
        print(f"\n{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze commit dataset statistics")
    parser.add_argument("-i", "--input", required=True, help="Input parquet file")
    parser.add_argument("-b", "--batch-size", type=int, default=50000, 
                       help="Rows per batch (adjust based on RAM)")
    args = parser.parse_args()
    
    print(f"[+] Analyzing dataset: {args.input}")
    print(f"[+] Batch size: {args.batch_size:,} rows")
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer()
    
    # Get total row count
    parquet_file = pq.ParquetFile(args.input)
    total_rows = parquet_file.metadata.num_rows
    analyzer.stats['total_rows'] = total_rows
    print(f"[+] Total rows: {total_rows:,}")
    
    # Calculate number of batches
    num_batches = (total_rows + args.batch_size - 1) // args.batch_size
    print(f"[+] Processing in {num_batches} batches...\n")
    
    # Process in streaming batches
    batch_iter = parquet_file.iter_batches(batch_size=args.batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc="Analyzing"):
        try:
            batch = next(batch_iter)
            batch_df = batch.to_pandas()
        except StopIteration:
            break
        
        # Analyze messages
        if 'message' in batch_df.columns:
            for msg in batch_df['message']:
                analyzer.analyze_message(msg)
        
        # Analyze diffs
        if 'diff_text' in batch_df.columns:
            for diff in batch_df['diff_text']:
                analyzer.analyze_diff(diff)
        
        # Cleanup
        del batch_df, batch
        gc.collect()
    
    # Print final report
    analyzer.print_report()

if __name__ == "__main__":
    main()
