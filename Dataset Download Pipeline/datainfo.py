import pandas as pd
import sys

def count_lines_in_column(parquet_path, column_name='diff_text'):
    """
    Count the number of lines in a text column of a parquet file.
    
    Args:
        parquet_path: Path to the parquet file
        column_name: Name of the column to analyze (default: 'diff_text')
    """
    print(f"Reading {parquet_path}...", file=sys.stderr)
    df = pd.read_parquet(parquet_path)
    
    print(f"Analyzing column '{column_name}'...", file=sys.stderr)
    
    # Count lines in each row (split by newline)
    def count_lines(text):
        if pd.isna(text):
            return 0
        return len(str(text).split('\n'))
    
    df[f'{column_name}_lines'] = df[column_name].apply(count_lines)
    
    # Print statistics
    print("\n" + "="*60)
    print(f"Statistics for '{column_name}':")
    print("="*60)
    print(f"Total rows: {len(df)}")
    print(f"Total lines (sum): {df[f'{column_name}_lines'].sum()}")
    print(f"Average lines per row: {df[f'{column_name}_lines'].mean():.2f}")
    print(f"Median lines per row: {df[f'{column_name}_lines'].median():.2f}")
    print(f"Min lines: {df[f'{column_name}_lines'].min()}")
    print(f"Max lines: {df[f'{column_name}_lines'].max()}")
    print(f"Std dev: {df[f'{column_name}_lines'].std():.2f}")
    print("="*60)

if __name__ == "__main__":
    parquet_file = "./cleaned.parquet"  # Change to your file
    column = "diff_text"  # Change if needed
    
    count_lines_in_column(parquet_file, column)
