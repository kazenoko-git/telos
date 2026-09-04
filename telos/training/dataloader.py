"""
Unified DataLoader utilities for PyTorch and MLX.
Handles memmap-based contiguous sequence streaming.
"""

import numpy as np


def get_global_targets_contiguous(dataset_matrix: np.ndarray, idx_ptr: int, total_batch: int, seq_len: int) -> tuple[np.ndarray, int]:
    """
    Fetches contiguous batches from the memory-mapped token array with wraparound.
    Returns standard numpy array.
    """
    n_rows = dataset_matrix.shape[0]
    end_ptr = idx_ptr + total_batch
    
    if end_ptr <= n_rows:
        batch = dataset_matrix[idx_ptr:end_ptr, :seq_len]
        next_ptr = end_ptr % n_rows
    else:
        first = dataset_matrix[idx_ptr:n_rows, :seq_len]
        remainder = end_ptr - n_rows
        second = dataset_matrix[:remainder, :seq_len]
        batch = np.concatenate((first, second), axis=0)
        next_ptr = remainder
        
    return batch, next_ptr


def get_global_targets_contiguous_mlx(dataset_matrix, idx_ptr, total_batch, seq_len):
    """Fetches contiguous batches and returns MLX array."""
    import mlx.core as mx
    batch, next_ptr = get_global_targets_contiguous(dataset_matrix, idx_ptr, total_batch, seq_len)
    return mx.array(batch, dtype=mx.int32), next_ptr


def get_global_targets_contiguous_pytorch(dataset_matrix, idx_ptr, total_batch, seq_len, device):
    """Fetches contiguous batches and returns PyTorch tensor."""
    import torch
    batch, next_ptr = get_global_targets_contiguous(dataset_matrix, idx_ptr, total_batch, seq_len)
    return torch.from_numpy(batch.astype(np.int64)).to(device), next_ptr
