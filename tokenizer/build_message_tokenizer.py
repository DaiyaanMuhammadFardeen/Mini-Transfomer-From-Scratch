"""
Memory-efficient message tokenizer using HuggingFace tokenizers library.
Replaces custom BPE implementation with Rust-backed ByteLevel BPE.
Includes conventional commit message normalization.
"""
import pyarrow.parquet as pq
import re
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tqdm import tqdm

# Conventional commit pattern: type(scope)!: description
CONV_COMMIT = re.compile(r'^(\w+)(\([^)]+\))?(!?):\s*(.+)', re.DOTALL)


def normalize_commit_message(msg: str) -> str:
    """
    Normalize conventional commit messages by extracting type prefix.
    Example: 'fix(auth): correct null pointer' -> '<TYPE_FIX> (auth) correct null pointer'
    """
    m = CONV_COMMIT.match(msg.strip())
    if m:
        ctype, scope, breaking, body = m.groups()
        scope = scope or ''
        breaking_marker = '!' if breaking else ''
        return f"<TYPE_{ctype.upper()}{breaking_marker}> {scope} {body}".strip()
    return msg


def build_message_tokenizer(
    parquet_path: str = "./messages.parquet",
    vocab_size: int = 8000,
    output_path: str = "message_tokenizer.json",
    chunk_size: int = 10000,
    use_conventional_commits: bool = True
):
    """
    Build message tokenizer using HuggingFace tokenizers with streaming.
    
    Args:
        parquet_path: Path to parquet file containing message column
        vocab_size: Target vocabulary size (smaller for messages: 8000-16000)
        output_path: Where to save the trained tokenizer JSON
        chunk_size: Number of rows to process at once
        use_conventional_commits: Whether to normalize conventional commit format
    
    Returns:
        Trained Tokenizer object
    """
    # Define special tokens for commit messages
    special_tokens = [
        "<PAD>", "<UNK>", "<SOS>", "<EOS>",
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
    
    # Use Whitespace pre-tokenizer for natural language (simpler than ByteLevel)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Configure trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True
    )
    
    # Streaming iterator - never loads full dataset into memory
    def message_iterator():
        print(f"Streaming from {parquet_path}...")
        pf = pq.ParquetFile(parquet_path)
        total_rows = 0
        normalized_count = 0
        
        # Get total number of row groups for progress tracking
        num_row_groups = pf.num_row_groups
        
        for batch in tqdm(pf.iter_batches(batch_size=chunk_size, columns=['message']),
                          total=num_row_groups,
                          desc="Processing message data",
                          unit="batch"):
            texts = batch.column('message').to_pylist()
            for text in texts:
                if text:
                    text = str(text)
                    # Apply conventional commit normalization if enabled
                    if use_conventional_commits:
                        normalized = normalize_commit_message(text)
                        if normalized != text:
                            normalized_count += 1
                        yield normalized
                    else:
                        yield text
                    total_rows += 1
            
            if total_rows % 50000 == 0:
                print(f"  Processed {total_rows:,} rows...")
        
        print(f"Total rows streamed: {total_rows:,}")
        if use_conventional_commits:
            print(f"Conventional commits normalized: {normalized_count:,} ({100*normalized_count/total_rows:.1f}%)")
    
    # Train from iterator (memory-efficient)
    print("Training BPE tokenizer...")
    tokenizer.train_from_iterator(message_iterator(), trainer=trainer)
    
    # Save tokenizer
    tokenizer.save(output_path)
    print(f"✅ Tokenizer saved to {output_path}")
    print(f"   Vocabulary size: {tokenizer.get_vocab_size()}")
    
    return tokenizer


if __name__ == "__main__":
    tokenizer = build_message_tokenizer()
