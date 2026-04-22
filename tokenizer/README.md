# Enhanced Tokenizer System

This directory contains the enhanced tokenizer system with multimodal vocabulary building capabilities and comprehensive visualization tools.

## Directory Structure

```
tokenizer/
├── DiffVocabulary.py          # Enhanced vocabulary for code diffs (with embedding awareness)
├── MsgVocabulary.py           # Enhanced vocabulary for commit messages (with embedding awareness)
├── programming_terms.py       # Comprehensive programming terminology database
├── visualization.py           # Visualization tools for research paper figures
├── analysis_report.py         # Comprehensive analysis report generator
├── visualize_vocabularies.py  # Script to visualize both vocabularies
├── generateDiffToken.py       # Script to generate diff vocabulary
├── generateMsgToken.py        # Script to generate message vocabulary
├── old/                      # Old tokenizer implementations (preserved for reference)
│   ├── DiffVocabulary.py
│   └── MsgVocabulary.py
├── visualization_output/     # Generated visualization figures
└── analysis_output/          # Generated analysis reports
```

## Enhanced Features

### 1. Multimodal Vocabulary Building
- **Embedding-aware tokenization**: Special tokens for each embedding layer
- **Programming term recognition**: Over 500+ programming languages, frameworks, commands, and technical terms
- **Semantic token extraction**: Automatic detection of change types, security issues, performance concerns, etc.

### 2. Programming Terms Database
- **Languages**: Python, JavaScript, Java, Go, Rust, and 30+ more
- **Frameworks**: Django, React, Spring, Express, and 100+ more
- **System Commands**: Git, Docker, Kubernetes, and 50+ more
- **Technical Terms**: Architecture patterns, DevOps tools, security concepts, and 200+ more

### 3. Research Paper Visualizations
- Vocabulary statistics and distributions
- Tokenization process visualization
- Comparative analysis between vocabularies
- Frequency distribution analysis (Zipf's law)
- Token type categorization

## Usage

### Generate Diff Vocabulary
```bash
python generateDiffToken.py
```

### Generate Message Vocabulary
```bash
python generateMsgToken.py
```

### Generate Comprehensive Analysis
```bash
python visualize_vocabularies.py
```

### Generate Analysis Report
```bash
python analysis_report.py
```

## Key Improvements

1. **Enhanced Semantic Understanding**: The new tokenizers recognize and preserve important programming concepts
2. **Better Embedding Alignment**: Special tokens directly correspond to embedding layer concepts
3. **Comprehensive Analysis Tools**: Built-in visualization and analysis capabilities
4. **Research-Ready Output**: High-quality figures suitable for research papers
5. **Backward Compatibility**: Maintains original functionality while adding new capabilities

## Visualization Output

The system generates several types of visualizations:
- Vocabulary size comparisons
- Token type distributions
- Token length analysis
- Frequency distributions
- Tokenization process visualization
- Special token categorization

All visualizations are saved in the `visualization_output/` directory with high-resolution quality suitable for research papers.