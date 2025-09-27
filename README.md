# Byte-Pair Encoding (BPE) Tokenizer

A from-scratch implementation of Byte-Pair Encoding (BPE) tokenization with comprehensive evaluation and analysis. This project demonstrates the fundamental concepts behind modern tokenization used in language models like GPT and BERT.

## Overview

Byte-Pair Encoding is a subword tokenization algorithm that iteratively merges the most frequently occurring pairs of characters or character sequences in a corpus. This implementation provides:

- A complete BPE tokenizer built from first principles
- Support for special tokens that remain atomic during training
- Comprehensive evaluation across different vocabulary sizes
- Performance analysis and visualization tools

## Implementation Details

### Core Algorithm

The `BPE_Tokenizer` class implements the standard BPE algorithm:

1. **Initialization**: Start with a base vocabulary of all 256 possible byte values
2. **Corpus Preprocessing**: Convert text to UTF-8 bytes and handle special tokens
3. **Iterative Merging**: Find the most frequent adjacent pair and merge into a new token
4. **Vocabulary Building**: Continue until reaching the desired vocabulary size

### Key Features

- **Special Token Support**: Tokens like `<eos>` and `<pad>` are preserved as atomic units
- **UTF-8 Encoding**: Handles arbitrary text through byte-level representation  
- **Efficient Statistics**: Uses `defaultdict` for fast pair frequency counting
- **Merge Logging**: Tracks all learned merge rules for analysis
- **Round-trip Fidelity**: Ensures perfect encode/decode consistency

### Code Structure

```python
class BPE_Tokenizer:
    def __init__(self):
        self.vocab = {}          # Token bytes -> token ID mapping
        self.merges = {}         # (token1, token2) -> merged_token_id
        self.special_tokens = [] # Protected tokens
        self.inverse_vocab = {}  # token ID -> token bytes mapping
        
    def train(self, texts, vocab_size, special_tokens)
    def encode(self, text) -> List[int]
    def decode(self, ids) -> str
```

## Evaluation Framework

The project includes comprehensive evaluation across multiple vocabulary sizes (1K, 2K, 4K, 8K, 32K tokens) using the AI arXiv papers dataset.

### Metrics Analyzed

1. **Training Time**: Time required to learn merge rules
2. **Sequence Length**: Average tokens per document after tokenization
3. **Compression Ratio**: Relative reduction in sequence length vs baseline
4. **Round-trip Fidelity**: Percentage of documents that decode perfectly
5. **Bytes per Token**: Information density of learned tokens
6. **Merge Pattern Analysis**: Evolution of learned subwords

### Key Findings

| Vocab Size | Mean Sequence Length | Bytes/Token | Compression Ratio | Training Time |
|------------|---------------------|-------------|-------------------|---------------|
| 1,000      | 397.99             | 2.91        | 1.00x (baseline)  | 88.16s        |
| 2,000      | 300.80             | 3.85        | 0.76x             | 178.84s       |
| 4,000      | 234.86             | 4.93        | 0.59x             | 341.72s       |
| 8,000      | 187.27             | 6.18        | 0.47x             | 590.93s       |
| 32,000     | 109.70             | 10.55       | 0.28x             | 1788.84s      |

### Observations

- **Perfect Fidelity**: All vocabulary sizes achieved 100% round-trip accuracy
- **Power Law Scaling**: Sequence length decreases approximately as a power law with vocabulary size
- **Diminishing Returns**: Compression benefits level off at higher vocabulary sizes
- **Merge Evolution**: Early merges focus on common character pairs (`'e '`, `'in'`, `'th'`), while later merges capture longer phrases and domain-specific terms

## Usage

### Basic Training and Encoding

```python
from bpe import BPE_Tokenizer

# Initialize tokenizer
tokenizer = BPE_Tokenizer()

# Train on corpus with special tokens
corpus = ["Hello world!", "This is a test."]
tokenizer.train(corpus, vocab_size=1000, special_tokens=["<eos>", "<pad>"])

# Encode and decode text
text = "Hello there!"
token_ids = tokenizer.encode(text)
decoded_text = tokenizer.decode(token_ids)

print(f"Original: {text}")
print(f"Token IDs: {token_ids}")  
print(f"Decoded: {decoded_text}")
```

### Running the Full Analysis

```bash
python bpe.py
```

This will:
- Download the AI arXiv dataset (10K documents)
- Train tokenizers for all vocabulary sizes
- Generate detailed analysis report (`bpe_analysis_results.txt`)
- Create visualization plot (`sequence_length_vs_vocab.png`)

## Files Structure

- `bpe.py` - Main implementation and evaluation script
- `bpe_analysis_results.txt` - Detailed evaluation results
- `sequence_length_vs_vocab.png` - Visualization of compression performance
- `README.md` - This documentation

## Dependencies

```
datasets
tqdm  
numpy
matplotlib
```

Install with: `pip install datasets tqdm numpy matplotlib`

## Technical Notes

### Merge Selection Strategy

The algorithm selects merges using a lexicographically-stable tie-breaking rule:
```python
best_pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
```
This ensures deterministic behavior when multiple pairs have the same frequency.

### Memory Efficiency

The implementation uses several optimizations:
- In-place token list updates during training
- Efficient pair statistics using `defaultdict(int)`
- Minimal memory footprint for inverse vocabulary mapping

### Special Token Handling

Special tokens are handled through regex-based pre-tokenization:
```python
special_pattern = f'({"|".join(map(re.escape, self.special_tokens))})'
parts = re.split(special_pattern, text)
```

This ensures special tokens remain atomic throughout the BPE process.

## Applications

This BPE implementation demonstrates concepts fundamental to:

- **Language Model Tokenization**: Similar to GPT/BERT tokenizers
- **Neural Machine Translation**: Handling out-of-vocabulary words
- **Text Compression**: Subword-level encoding for efficiency
- **Cross-lingual NLP**: Byte-level representation handles any language

## Future Extensions

Potential improvements could include:

- **Regularization**: Dropout during training for robustness
- **Vocabulary Pruning**: Remove infrequent tokens to reduce size  
- **Parallel Training**: Multi-threaded merge computation
- **Incremental Learning**: Update existing vocabularies with new data
- **Alternative Merge Criteria**: Explore different pair selection strategies

## References

- Sennrich, R., Haddow, B., & Birch, A. (2015). Neural Machine Translation of Rare Words with Subword Units.
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.

## License

This implementation is provided for educational and research purposes.