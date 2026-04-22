"""
Memory-efficient diff tokenizer using HuggingFace tokenizers library.
Replaces custom BPE implementation with Rust-backed ByteLevel BPE.
"""
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tqdm import tqdm


def build_diff_tokenizer(
    parquet_path: str = "./diff_text.parquet",
    vocab_size: int = 32000,
    output_path: str = "diff_tokenizer.json",
    chunk_size: int = 5000
):
    """
    Build diff tokenizer using HuggingFace tokenizers with streaming.
    
    Args:
        parquet_path: Path to parquet file containing diff_text column
        vocab_size: Target vocabulary size
        output_path: Where to save the trained tokenizer JSON
        chunk_size: Number of rows to process at once
    
    Returns:
        Trained Tokenizer object
    """
    # Define special tokens for diffs
    special_tokens = [
        "<PAD>", "<UNK>", "<SOS>", "<EOS>",
        # Diff-specific tokens
        "<ADD>", "</ADD>", "<REMOVE>", "</REMOVE>",
        "<MODIFY>", "</MODIFY>",
        "<COMMENT_ADD>", "</COMMENT_ADD>",
        "<COMMENT_REMOVE>", "</COMMENT_REMOVE>",
        "<COMMENT_MODIFY>", "</COMMENT_MODIFY>",
        # Change type tokens
        "<BUG_FIX>", "<FEATURE_ADD>", "<REFACTOR>", "<OPTIMIZATION>",
        "<DOC_UPDATE>", "<TEST_ADD>", "<CONFIG_CHANGE>",
        # Code style tokens
        "<CAMEL_CASE>", "<SNAKE_CASE>", "<PASCAL_CASE>", "<CONSTANT_CASE>",
        "<INDENT_STYLE>", "<LINE_LENGTH>", "<COMMENT_STYLE>",
        # Dependency tokens
        "<IMPORT>", "<EXPORT>", "<DEPENDENCY>", "<LIBRARY>",
        "<PACKAGE>", "<MODULE>", "<FRAMEWORK>",
        # Security tokens
        "<SECURITY_FIX>", "<VULNERABILITY>", "<PERMISSION>", "<AUTH>",
        # Performance tokens
        "<PERFORMANCE>", "<BOTTLENECK>",
        # Error/Exception tokens
        "<ERROR>", "<EXCEPTION>", "<TRY_CATCH>", "<RAISE>",
        # API tokens
        "<API_CHANGE>", "<ENDPOINT>", "<PARAMETER>", "<RESPONSE>",
        # Complexity tokens
        "<COMPLEXITY>", "<CYCLOMATIC>", "<COGNITIVE>",
        # Domain-specific tokens
        "<DOMAIN>", "<BUSINESS_LOGIC>", "<UI_CHANGE>", "<BACKEND>",
        # Testing tokens
        "<TEST>", "<UNIT_TEST>", "<INTEGRATION_TEST>", "<MOCK>",
        # Syntactic tokens
        "<FUNCTION>", "<CLASS>", "<METHOD>", "<VARIABLE>",
        "<LOOP>", "<CONDITIONAL>", "<EXPRESSION>", "<STATEMENT>",
        # Temporal tokens
        "<TIMESTAMP>", "<VERSION>", "<RELEASE>", "<DEPRECATED>",
        # Collaborative tokens
        "<REVIEWED>", "<APPROVED>", "<WIP>", "<CO_AUTHOR>"
    ]
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    
    # Use ByteLevel pre-tokenizer (preserves case, handles Unicode)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Configure trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True
    )
    
    # Streaming iterator - never loads full dataset into memory
    def diff_iterator():
        print(f"Streaming from {parquet_path}...")
        pf = pq.ParquetFile(parquet_path)
        total_rows = 0
        
        # Get total number of row groups for progress tracking
        num_row_groups = pf.num_row_groups
        
        for batch in tqdm(pf.iter_batches(batch_size=chunk_size, columns=['diff_text']),
                          total=num_row_groups,
                          desc="Processing diff data",
                          unit="batch"):
            texts = batch.column('diff_text').to_pylist()
            for text in texts:
                if text:
                    yield str(text)
                    total_rows += 1
            
            if total_rows % 50000 == 0:
                print(f"  Processed {total_rows:,} rows...")
        
        print(f"Total rows streamed: {total_rows:,}")
    
    # Train from iterator (memory-efficient)
    print("Training BPE tokenizer...")
    tokenizer.train_from_iterator(diff_iterator(), trainer=trainer)
    
    # Save tokenizer
    tokenizer.save(output_path)
    print(f"✅ Tokenizer saved to {output_path}")
    print(f"   Vocabulary size: {tokenizer.get_vocab_size()}")
    
    return tokenizer


if __name__ == "__main__":
    tokenizer = build_diff_tokenizer()
