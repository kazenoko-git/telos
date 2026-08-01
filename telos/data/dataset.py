"""
PyTorch Dataset & DataLoader for Python Code Autocomplete with Dynamic Masking.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from telos.diffusion.forward_process import apply_masking
from telos.data.tokenizer import PAD_TOKEN_ID, MASK_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from telos.diffusion.forward_process import apply_masking
from telos.data.tokenizer import PAD_TOKEN_ID, MASK_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID


class PythonCodeDataset(Dataset):
    """PyTorch Dataset with zero-copy NumPy storage for ultra-low RAM footprint."""

    def __init__(self, sequences: list | np.ndarray, max_seq_len: int = 512):
        """
        Args:
            sequences: list of token ID lists OR a contiguous 2D NumPy array [num_samples, max_seq_len].
            max_seq_len: maximum sequence length.
        """
        if isinstance(sequences, list):
            # Pre-pad into contiguous int32 numpy array to free CPython list RAM overhead
            num_samples = len(sequences)
            arr = np.full((num_samples, max_seq_len), PAD_TOKEN_ID, dtype=np.int32)
            for i, seq in enumerate(sequences):
                trunc_seq = seq[:max_seq_len]
                arr[i, :len(trunc_seq)] = trunc_seq
            self.sequences = arr
        else:
            self.sequences = sequences

        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]
        return torch.from_numpy(seq.astype(np.int64))


def collate_fn_dynamic_masking(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """custom dataloader collate function that applies dynamic mdlm masking per batch.

    args:
        batch: list of 1d tensors [seq_len] of clean token ids.

    returns:
        masked_input_ids: tensor [batch_size, seq_len] with [MASK] tokens inserted.
        targets: tensor [batch_size, seq_len] containing original clean token ids.
        mask_positions: boolean tensor [batch_size, seq_len] indicating masked positions.
        t_values: tensor [batch_size, 1] containing per-example sampled time step t.
    """
    # stack batch tensors [batch_size, seq_len]
    targets = torch.stack(batch, dim=0)

    # define special tokens that MUST NOT be masked
    special_tokens = {PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID}

    # apply MDLM forward masking process dynamically
    masked_input_ids, mask_positions, t_values = apply_masking(
        input_ids=targets,
        mask_token_id=MASK_TOKEN_ID,
        special_token_ids=special_tokens
    )

    return masked_input_ids, targets, mask_positions, t_values


def create_dataloader(
    sequences: list[list[int]],
    batch_size: int = 32,
    max_seq_len: int = 256,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """creates a PyTorch DataLoader with dynamic masking enabled."""
    dataset = PythonCodeDataset(sequences, max_seq_len=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_dynamic_masking,
        num_workers=num_workers,
        pin_memory=True
    )
