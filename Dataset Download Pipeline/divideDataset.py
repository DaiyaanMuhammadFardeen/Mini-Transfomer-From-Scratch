#!/usr/bin/env python3
"""
divideDataset.py
Memory-efficient Parquet splitter: splits a dataset into N equal parts using chunked reading.
Handles large datasets that don't fit in memory by processing in chunks.
"""
import os
import gc
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_parquet_metadata(data_path):
    """Get total row count without loading entire dataset."""
    parquet_file = pq.ParquetFile(data_path)
    total_rows = parquet_file.metadata.num_rows
    return total_rows, parquet_file


def split_parquet_memory_efficient(data_path, num_splits, output_dir, chunk_size=50000):
    """
    Split parquet file into N parts using chunked reading for memory efficiency.
    
    Args:
        data_path: Path to input parquet file
        num_splits: Number of parts to split into
        output_dir: Directory to save split files
        chunk_size: Number of rows to read at once (adjust based on available RAM)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total rows without loading data
    print("Reading parquet metadata...")
    total_rows, parquet_file = get_parquet_metadata(data_path)
    print(f"Dataset has {total_rows:,} rows. Splitting into {num_splits} parts...")
    
    # Calculate split boundaries
    split_size = total_rows // num_splits
    split_boundaries = []
    for i in range(num_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < num_splits - 1 else total_rows
        split_boundaries.append((start, end))
        print(f"Part {i+1}: rows {start:,} to {end-1:,} ({end-start:,} rows)")
    
    # Initialize output writers for each split
    output_files = [
        os.path.join(output_dir, f"dataset_part{i+1}.parquet")
        for i in range(num_splits)
    ]
    writers = [None] * num_splits
    schemas = [None] * num_splits
    
    try:
        # Process data in chunks
        print(f"\nProcessing in chunks of {chunk_size:,} rows...")
        rows_processed = 0
        
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
            # Convert batch to pandas DataFrame
            df_chunk = batch.to_pandas()
            chunk_start = rows_processed
            chunk_end = rows_processed + len(df_chunk)
            
            print(f"Processing chunk {batch_idx+1}: rows {chunk_start:,}-{chunk_end-1:,}")
            
            # Determine which splits this chunk belongs to
            for split_idx, (split_start, split_end) in enumerate(split_boundaries):
                # Check if this chunk overlaps with current split
                if chunk_end <= split_start or chunk_start >= split_end:
                    continue
                
                # Calculate overlap
                overlap_start = max(0, split_start - chunk_start)
                overlap_end = min(len(df_chunk), split_end - chunk_start)
                
                # Extract relevant rows for this split
                df_split_chunk = df_chunk.iloc[overlap_start:overlap_end].copy()
                
                if len(df_split_chunk) > 0:
                    # Initialize writer on first write
                    if writers[split_idx] is None:
                        schemas[split_idx] = pa.Schema.from_pandas(df_split_chunk)
                        writers[split_idx] = pq.ParquetWriter(
                            output_files[split_idx],
                            schemas[split_idx],
                            compression='snappy'
                        )
                    
                    # Write chunk to appropriate split file
                    table = pa.Table.from_pandas(df_split_chunk, schema=schemas[split_idx])
                    writers[split_idx].write_table(table)
                    
                    # Clean up
                    del df_split_chunk
                    del table
            
            rows_processed = chunk_end
            
            # Aggressive garbage collection
            del df_chunk
            del batch
            gc.collect()
        
        print(f"\nSuccessfully processed all {rows_processed:,} rows")
        
    finally:
        # Close all writers
        print("\nFinalizing output files...")
        for i, writer in enumerate(writers):
            if writer is not None:
                writer.close()
                file_size = os.path.getsize(output_files[i]) / (1024**2)
                print(f"Part {i+1}: {output_files[i]} ({file_size:.2f} MB)")
        
        # Final cleanup
        gc.collect()


def estimate_chunk_size(available_ram_gb=4, safety_factor=0.5):
    """
    Estimate safe chunk size based on available RAM.
    
    Args:
        available_ram_gb: Available RAM in GB
        safety_factor: Safety multiplier (0.5 = use 50% of available RAM)
    
    Returns:
        Estimated safe chunk size in rows
    """
    # Rough estimate: assume ~1KB per row on average
    bytes_per_row = 1024
    available_bytes = available_ram_gb * 1024**3 * safety_factor
    estimated_rows = int(available_bytes / bytes_per_row)
    
    # Round to nearest 10k for cleaner numbers
    return max(10000, (estimated_rows // 10000) * 10000)


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient Parquet dataset splitter for large files"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Input Parquet file path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="splits",
        help="Directory to save split files"
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=5,
        help="Number of parts to split into"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Number of rows per chunk (default: auto-estimate based on RAM)"
    )
    parser.add_argument(
        "--available_ram",
        type=float,
        default=4.0,
        help="Available RAM in GB for auto-estimation (default: 4.0)"
    )
    
    args = parser.parse_args()
    
    # Estimate chunk size if not provided
    if args.chunk_size is None:
        args.chunk_size = estimate_chunk_size(args.available_ram)
        print(f"Auto-estimated chunk size: {args.chunk_size:,} rows")
        print(f"(Based on {args.available_ram}GB available RAM)")
        print("You can override with --chunk_size if needed\n")
    
    split_parquet_memory_efficient(
        args.data_path,
        args.num_splits,
        args.output_dir,
        args.chunk_size
    )
    
    print("\n✓ Split complete!")


if __name__ == "__main__":
    main()
