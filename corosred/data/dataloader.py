"""
Dataset streaming and memory mapping for COROSred.
"""

from pathlib import Path
import numpy as np


def load_memmap_dataset(bin_path: str, seq_len: int, vocab_size: int = 8192):
    """Loads token stream from memory-mapped binary file, falling back to synthetic data."""
    path = Path(bin_path)
    if path.exists():
        raw = np.memmap(path, dtype=np.int32, mode="r")
        n_seqs = len(raw) // seq_len
        return raw[: n_seqs * seq_len].reshape(n_seqs, seq_len)

    # Synthetic fallback for dry runs
    return np.random.randint(0, vocab_size, (10000, seq_len), dtype=np.uint16)
