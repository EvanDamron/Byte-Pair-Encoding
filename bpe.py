import re
import time
import random
import logging
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# --- Configure Logging ---
# Set to WARNING to reduce console noise during the analysis run
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BPE_Tokenizer:
    """
    A from-scratch Byte-Pair Encoding (BPE) Tokenizer.
    This tokenizer learns merge rules from a corpus and uses them to tokenize new text.
    It supports special tokens that are kept atomic during training and encoding.
    """

    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.special_tokens = []
        self.inverse_vocab = {}
        self.learned_merges_log = []

    def _get_stats(self, ids_list):
        counts = defaultdict(int)
        for ids in ids_list:
            for pair in zip(ids, ids[1:]):
                counts[pair] += 1
        return counts

    def _merge(self, ids, pair, new_id):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, texts, vocab_size, special_tokens):
        if not texts: raise ValueError("Cannot train on an empty list of texts.")
        if vocab_size < 256 + len(special_tokens): raise ValueError("Vocab size is too small.")

        self.special_tokens = special_tokens
        self.learned_merges_log = []

        # 1. --- Initial Vocabulary & Inverse Vocab ---
        self.vocab = {i: i for i in range(256)}
        self.inverse_vocab = {i: bytes([i]) for i in range(256)}
        for token in special_tokens:
            id = len(self.vocab)
            encoded_token = token.encode('utf-8')
            self.vocab[encoded_token] = id
            self.inverse_vocab[id] = encoded_token

        # 2. --- Pre-tokenize the training data ---
        special_pattern = f'({"|".join(map(re.escape, self.special_tokens))})'
        tokenized_chunks = []
        for text in tqdm(texts, desc="Preprocessing texts", leave=False):
            parts = re.split(special_pattern, text)
            chunk_ids = []
            for part in parts:
                if not part: continue
                encoded_part = part.encode('utf-8')
                if encoded_part in self.vocab:
                    chunk_ids.append(self.vocab[encoded_part])
                else:
                    chunk_ids.extend(list(encoded_part))
            tokenized_chunks.append(chunk_ids)

        # 3. --- Learn Merge Rules ---
        num_merges = vocab_size - len(self.vocab)
        for i in tqdm(range(num_merges), desc=f"Learning merges for V={vocab_size}", leave=False):
            stats = self._get_stats(tokenized_chunks)
            if not stats:
                logging.warning("No more pairs to merge. Stopping training early.")
                break

            best_pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            p0, p1 = best_pair
            new_id = 256 + len(self.special_tokens) + i

            # Log the merge rule using repr() for clear string representation
            token1_repr = repr(self.inverse_vocab[p0].decode('utf-8', 'replace'))
            token2_repr = repr(self.inverse_vocab[p1].decode('utf-8', 'replace'))
            new_token_bytes = self.inverse_vocab[p0] + self.inverse_vocab[p1]
            new_token_repr = repr(new_token_bytes.decode('utf-8', 'replace'))
            self.learned_merges_log.append((i + 1, token1_repr, token2_repr, new_token_repr))

            tokenized_chunks = [self._merge(ids, best_pair, new_id) for ids in tokenized_chunks]
            self.merges[best_pair] = new_id
            self.inverse_vocab[new_id] = new_token_bytes

        self.vocab = {v: k for k, v in self.inverse_vocab.items()}

    def encode(self, text):
        if not self.vocab: raise RuntimeError("Tokenizer has not been trained.")
        final_ids = []
        special_pattern = f'({"|".join(map(re.escape, self.special_tokens))})'
        parts = re.split(special_pattern, text)
        for part in parts:
            if not part: continue
            encoded_part = part.encode('utf-8')
            if encoded_part in self.vocab:
                final_ids.append(self.vocab[encoded_part])
                continue
            chunk_ids = list(encoded_part)
            while True:
                stats = self._get_stats([chunk_ids])
                possible_merges = {pair: self.merges[pair] for pair in stats if pair in self.merges}
                if not possible_merges: break
                best_pair = min(possible_merges, key=lambda p: self.merges[p])
                chunk_ids = self._merge(chunk_ids, best_pair, self.merges[best_pair])
            final_ids.extend(chunk_ids)
        return final_ids

    def decode(self, ids):
        if not self.inverse_vocab: raise RuntimeError("Tokenizer is not trained.")
        tokens_bytes = b"".join(self.inverse_vocab[id] for id in ids)
        return tokens_bytes.decode('utf-8', errors='replace')


def format_merges_table(title, merges):
    """Formats a list of merge rules into a clean, aligned plain-text table."""
    header_line = f"--- {title} ---"

    if not merges:
        return f"{header_line}\n(No merges to display)\n"

    headers = ["Merge #", "Token 1", "Token 2", "Resulting Token"]

    # Calculate the maximum width for each column
    col_widths = [len(h) for h in headers]
    for num, t1, t2, res in merges:
        col_widths[0] = max(col_widths[0], len(str(num)))
        col_widths[1] = max(col_widths[1], len(t1))
        col_widths[2] = max(col_widths[2], len(t2))
        col_widths[3] = max(col_widths[3], len(res))

    # Build the table string
    table_lines = [header_line]

    # Header row
    header_row = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    table_lines.append(header_row)

    # Separator row
    separator_row = "-+-".join("-" * w for w in col_widths)
    table_lines.append(separator_row)

    # Data rows
    for num, t1, t2, res in merges:
        row_items = [str(num), t1, t2, res]
        data_row = " | ".join(f"{item:<{w}}" for item, w in zip(row_items, col_widths))
        table_lines.append(data_row)

    return "\n".join(table_lines)


if __name__ == '__main__':
    # --- Configuration ---
    DATASET_NAME = "jamescalam/ai-arxiv"
    NUM_DOCS_TO_TRAIN = 10000
    NUM_SAMPLES_FOR_TEST = 50
    SPECIAL_TOKENS = ["<eos>", "<pad>"]
    VOCAB_SIZES_TO_TEST = [1000, 2000, 4000, 8000, 32000]


    RESULTS_FILE = "bpe_analysis_results.txt"
    PLOT_FILE = "sequence_length_vs_vocab.png"

    # --- Load Dataset ---
    logging.info(f"Loading {NUM_DOCS_TO_TRAIN} documents from '{DATASET_NAME}'...")
    ds = load_dataset(DATASET_NAME, split=f"train[:{NUM_DOCS_TO_TRAIN}]")
    training_texts = [doc['summary'] for doc in ds]
    logging.info("Dataset loaded.")

    analysis_data = {}

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write("BPE Tokenizer Analysis Report\n")
        f.write("=============================\n\n")

        # --- Main Training and Analysis Loop ---
        for vocab_size in VOCAB_SIZES_TO_TEST:
            section_header = f"Analysis for Vocab Size: {vocab_size}"
            logging.info(f"\n{'=' * len(section_header)}\n{section_header}\n{'=' * len(section_header)}")
            f.write(f"## {section_header}\n\n")

            # 1. Train Tokenizer
            tokenizer = BPE_Tokenizer()
            start_time = time.time()
            tokenizer.train(training_texts, vocab_size, SPECIAL_TOKENS)
            duration = time.time() - start_time
            logging.info(f"Training completed in {duration:.2f} seconds.")
            f.write(f"**Training Time:** {duration:.2f} seconds\n\n")

            # 2. Show Merges
            if tokenizer.learned_merges_log:
                f.write(format_merges_table("First 20 Merges", tokenizer.learned_merges_log[:20]))
                f.write("\n\n")
                f.write(format_merges_table("Last 20 Merges", tokenizer.learned_merges_log[-20:]))
                f.write("\n\n")

            # 3. Perform Round-trip Tests
            test_samples = random.sample(training_texts, min(NUM_SAMPLES_FOR_TEST, len(training_texts)))
            passed_count = sum(1 for text in test_samples if tokenizer.decode(tokenizer.encode(text)) == text)
            accuracy = (passed_count / len(test_samples)) * 100
            logging.info(f"Round-trip Fidelity ({NUM_SAMPLES_FOR_TEST} samples): {accuracy:.2f}%")
            f.write(f"**Round-trip Fidelity ({NUM_SAMPLES_FOR_TEST} samples):** {accuracy:.2f}%\n\n")

            random_doc = random.choice(training_texts)
            decoded_doc = tokenizer.decode(tokenizer.encode(random_doc))
            f.write("**Single Document Round-trip Example:**\n")
            f.write(f"  - **Original:** {random_doc}\n")
            f.write(f"  - **Decoded:** {decoded_doc}\n")
            f.write(f"  - **Success:** {random_doc == decoded_doc}\n\n")

            # 4. Analyze token lengths
            logging.info("Analyzing token lengths across the dataset...")
            token_lengths = [len(tokenizer.encode(text)) for text in
                             tqdm(training_texts, desc="Encoding all docs", leave=False)]
            total_tokens = sum(token_lengths)
            total_bytes = sum(len(text.encode('utf-8')) for text in training_texts)

            analysis_data[vocab_size] = {
                'training_time': duration,
                'test_accuracy': accuracy,
                'token_lengths': token_lengths,
                'mean_len': np.mean(token_lengths),
                'p5_len': np.percentile(token_lengths, 5),
                'p95_len': np.percentile(token_lengths, 95),
                'bytes_per_token': total_bytes / total_tokens if total_tokens > 0 else 0,
            }
            f.write("---\n\n")

    # --- Final Summary and Plotting ---
    logging.info("\nGenerating final summary and plot...")

    baseline_vocab = min(VOCAB_SIZES_TO_TEST)
    baseline_tokens = np.mean(analysis_data[baseline_vocab]['token_lengths'])
    for v_size in VOCAB_SIZES_TO_TEST:
        mean_tokens = analysis_data[v_size]['mean_len']
        analysis_data[v_size]['compression_ratio'] = mean_tokens / baseline_tokens

    # Write final summary table to file
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write("## Final Summary\n\n")

        summary_headers = ["Vocab Size", "Mean Seq. Len", "5th P-tile", "95th P-tile", "Bytes/Token", "Comp. Ratio"]
        summary_data = []
        for v_size, data in analysis_data.items():
            summary_data.append([
                str(v_size),
                f"{data['mean_len']:.2f}",
                f"{data['p5_len']:.2f}",
                f"{data['p95_len']:.2f}",
                f"{data['bytes_per_token']:.2f}",
                f"{data.get('compression_ratio', 1.0):.2f}"
            ])

        col_widths = [len(h) for h in summary_headers]
        for row in summary_data:
            for i, item in enumerate(row):
                col_widths[i] = max(col_widths[i], len(item))

        header_row = " | ".join(f"{h:<{w}}" for h, w in zip(summary_headers, col_widths))
        f.write(header_row + "\n")
        separator_row = "-+-".join("-" * w for w in col_widths)
        f.write(separator_row + "\n")
        for row in summary_data:
            data_row = " | ".join(f"{item:<{w}}" for item, w in zip(row, col_widths))
            f.write(data_row + "\n")

    # Generate Plot
    vocab_sizes = list(analysis_data.keys())
    mean_lengths = [d['mean_len'] for d in analysis_data.values()]
    p5_lengths = [d['p5_len'] for d in analysis_data.values()]
    p95_lengths = [d['p95_len'] for d in analysis_data.values()]

    lower_errors = np.array(mean_lengths) - np.array(p5_lengths)
    upper_errors = np.array(p95_lengths) - np.array(mean_lengths)
    error_bars = [lower_errors, upper_errors]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(vocab_sizes, mean_lengths, yerr=error_bars, fmt='-o', capsize=5,
                label='Mean Length (5th-95th percentile error)', color='royalblue', ecolor='skyblue')

    ax.set_title('Tokenized Sequence Length vs. Vocabulary Size', fontsize=16)
    ax.set_xlabel('Vocabulary Size', fontsize=12)
    ax.set_ylabel('Average Sequence Length (Tokens)', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="--")
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)

    logging.info(f"Analysis complete. Full report saved to '{RESULTS_FILE}'.")
    logging.info(f"Plot saved to '{PLOT_FILE}'.")