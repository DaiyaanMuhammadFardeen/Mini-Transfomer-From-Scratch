import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys

def is_ascii(text):
    """Check if text contains only ASCII characters."""
    if pd.isna(text):
        return True  # Keep NaN values
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def clean_dataset_streaming(input_path, output_path, message_column='message', chunk_size=5000):
    """
    Clean dataset by removing rows with non-ASCII characters in message column.
    Uses streaming to minimize memory usage.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to save cleaned parquet file
        message_column: Name of the column to clean (default: 'message')
        chunk_size: Number of rows to process at a time (default: 5000)
    """
    print(f"Starting to clean dataset from {input_path}", file=sys.stderr)
    print(f"Output will be saved to {output_path}", file=sys.stderr)
    
    total_rows = 0
    removed_rows = 0
    processed_rows = 0
    writer = None
    schema = None
    
    # Read parquet file in chunks using pyarrow
    parquet_file = pq.ParquetFile(input_path)
    num_batches = parquet_file.num_row_groups
    total_rows = parquet_file.metadata.num_rows
    
    print(f"Total rows in dataset: {total_rows}", file=sys.stderr)
    print(f"Processing in {chunk_size}-row chunks...", file=sys.stderr)
    
    # Process each row group
    for batch_idx in range(num_batches):
        # Read one row group at a time
        batch = parquet_file.read_row_group(batch_idx).to_pandas()
        
        # Filter rows with ASCII-only messages
        ascii_mask = batch[message_column].apply(is_ascii)
        batch_removed = (~ascii_mask).sum()
        removed_rows += batch_removed
        
        # Keep only ASCII rows
        clean_batch = batch[ascii_mask]
        
        if len(clean_batch) > 0:
            # Write to parquet incrementally
            clean_batch = clean_batch.reset_index(drop=True)
            table = pa.Table.from_pandas(clean_batch)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
            writer.write_table(table)
        
        processed_rows += len(batch)
        print(f"Processed {processed_rows}/{total_rows} rows. Removed this batch: {batch_removed}", file=sys.stderr)
    
    # Close the writer
    if writer is not None:
        writer.close()
    
    # Print summary
    print("\n" + "="*60, file=sys.stderr)
    print(f"Cleaning complete!", file=sys.stderr)
    print(f"Original rows: {total_rows}", file=sys.stderr)
    print(f"Removed rows: {removed_rows}", file=sys.stderr)
    print(f"Kept rows: {total_rows - removed_rows}", file=sys.stderr)
    print(f"Removal rate: {(removed_rows/total_rows)*100:.2f}%", file=sys.stderr)
    print("="*60, file=sys.stderr)

if __name__ == "__main__":
    # Configuration
    input_file = "./commitpack_cleaned.parquet"
    output_file = "./cleaned.parquet"
    message_col = "message"
    
    clean_dataset_streaming(input_file, output_file, message_col)
