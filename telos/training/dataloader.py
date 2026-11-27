"""
Unified DataLoader utilities for PyTorch and MLX.
Handles memmap-based contiguous sequence streaming and multi-shard `.bin` directories.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def get_global_targets_contiguous(dataset_matrix: np.ndarray, idx_ptr: int, total_batch: int, seq_len: int) -> tuple[np.ndarray, int]:
    """
    Fetches contiguous batches from the memory-mapped token array with wraparound.
    Returns standard numpy array.
    """
    if dataset_matrix.ndim != 2 or not 0 < seq_len <= dataset_matrix.shape[1]:
        raise ValueError("dataset_matrix must be 2D and contain seq_len columns")
    n_rows = dataset_matrix.shape[0]
    if n_rows == 0 or total_batch < 1:
        raise ValueError("dataset must be nonempty and total_batch must be positive")
    idx_ptr %= n_rows
    end_ptr = idx_ptr + total_batch
    
    if end_ptr <= n_rows:
        batch = dataset_matrix[idx_ptr:end_ptr, :seq_len]
        next_ptr = end_ptr % n_rows
    else:
        # A large tuned batch may span the dataset more than once.
        rows = (idx_ptr + np.arange(total_batch)) % n_rows
        batch = dataset_matrix[rows, :seq_len]
        next_ptr = end_ptr % n_rows
        
    return batch, next_ptr


def get_global_targets_contiguous_mlx(dataset_matrix, idx_ptr, total_batch, seq_len):
    """Fetches contiguous batches and returns MLX array."""
    import mlx.core as mx
    batch, next_ptr = get_global_targets_contiguous(dataset_matrix, idx_ptr, total_batch, seq_len)
    return mx.array(batch, dtype=mx.int32), next_ptr


def get_global_targets_contiguous_pytorch(dataset_matrix, idx_ptr: int, total_batch: int, seq_len: int, device, non_blocking: bool = True):
    """
    Fetches contiguous batches and transfers to PyTorch tensor.
    Avoids redundant memory copies when dataset_matrix is already int32 or int64,
    and uses non-blocking pinned transfers to halve PCIe bus transfer time.
    """
    import torch
    batch, next_ptr = get_global_targets_contiguous(dataset_matrix, idx_ptr, total_batch, seq_len)
    
    # Ensure contiguous and writable memory to prevent PyTorch non-writable array warning and undefined behavior
    if not batch.flags.writeable or not batch.flags.c_contiguous:
        batch = np.ascontiguousarray(batch).copy()

    if batch.dtype in (np.int32, np.int64):
        tensor = torch.from_numpy(batch)
    elif batch.dtype == np.uint16:
        try:
            tensor = torch.from_numpy(batch)
        except TypeError:
            tensor = torch.from_numpy(batch.astype(np.int32))
    else:
        tensor = torch.from_numpy(batch.astype(np.int32))

    if str(device).startswith("cuda") and torch.cuda.is_available():
        if not tensor.is_pinned():
            tensor = tensor.pin_memory()
            
    res = tensor.to(device, dtype=torch.long, non_blocking=non_blocking)
    # Explicitly unbind local host memory references to avoid retention in glibc arenas
    del tensor, batch
    return res, next_ptr


def _detect_shard_dtype(bin_path: Path, vocab_size: int = 32768) -> np.dtype:
    """Detect uint16/int32 dtype from sidecar JSON metadata, falling back to vocab_size."""
    for meta_cand in (Path(str(bin_path) + ".json"), bin_path.with_suffix(".json")):
        if meta_cand.exists():
            try:
                with open(meta_cand, "r", encoding="utf-8") as mf:
                    dt_str = json.load(mf).get("dtype", "uint16")
                    return np.dtype(np.int32 if dt_str == "int32" else np.uint16)
            except Exception:
                pass
    if "mac" in str(bin_path) or vocab_size > 65536:
        return np.dtype(np.int32)
    return np.dtype(np.uint16)


def _head_tail_sha256(path: Path, chunk_bytes: int = 65536) -> str:
    """Fast content fingerprint of first and last 64 KiB of a shard file."""
    h = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as f:
        h.update(f.read(min(chunk_bytes, size)))
        if size > chunk_bytes:
            f.seek(max(0, size - chunk_bytes))
            h.update(f.read(chunk_bytes))
    return h.hexdigest()


class ShardedMemmapDataset:
    """
    Lazy multi-shard `.bin` token dataset with exact `(shard_idx, row_idx)` resume.

    Discovers all `*.bin` files in a directory (sorted lexicographically) and keeps
    only one `np.memmap` open at a time so host RSS stays flat across 100+ GB corpora
    mounted read-only under `/kaggle/input`.
    """

    def __init__(self, data_dir: str | Path, seq_len: int, vocab_size: int = 32768):
        self.data_dir = Path(data_dir)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")

        if self.data_dir.is_dir():
            shards = sorted(p for p in self.data_dir.rglob("*.bin") if p.is_file())
        else:
            raise FileNotFoundError(f"Shard directory not found: {self.data_dir}")

        if not shards:
            raise FileNotFoundError(f"No .bin shards found in directory: {self.data_dir}")

        self.shard_paths: list[Path] = []
        self.shard_dtypes: list[np.dtype] = []
        self.shard_rows: list[int] = []
        for p in shards:
            dtype = _detect_shard_dtype(p, vocab_size=self.vocab_size)
            n_tokens = p.stat().st_size // dtype.itemsize
            n_rows = n_tokens // self.seq_len
            if n_rows > 0:
                self.shard_paths.append(p)
                self.shard_dtypes.append(dtype)
                self.shard_rows.append(n_rows)

        if not self.shard_paths:
            raise ValueError(
                f"Found {len(shards)} .bin file(s) in {self.data_dir}, but none had at least "
                f"one complete sequence of length {self.seq_len}."
            )

        self.num_shards = len(self.shard_paths)
        self.cumulative_rows = np.cumsum([0] + self.shard_rows).tolist()
        self.total_rows = int(self.cumulative_rows[-1])
        self._open_shard_idx: int | None = None
        self._open_memmap: np.memmap | None = None

    def __len__(self) -> int:
        return self.total_rows

    def _get_shard_matrix(self, shard_idx: int) -> np.ndarray:
        if self._open_shard_idx == shard_idx and self._open_memmap is not None:
            return self._open_memmap
        self._open_memmap = None
        self._open_shard_idx = None
        path = self.shard_paths[shard_idx]
        dtype = self.shard_dtypes[shard_idx]
        n_rows = self.shard_rows[shard_idx]
        mm = np.memmap(path, dtype=dtype, mode="r", shape=(n_rows, self.seq_len))
        self._open_memmap = mm
        self._open_shard_idx = shard_idx
        return mm

    def close(self) -> None:
        self._open_memmap = None
        self._open_shard_idx = None

    def cursor_from_global_row(self, global_row: int) -> tuple[int, int]:
        g = int(global_row) % self.total_rows
        lo, hi = 0, self.num_shards - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.cumulative_rows[mid] <= g:
                lo = mid
            else:
                hi = mid - 1
        return lo, g - self.cumulative_rows[lo]

    def global_row_from_cursor(self, shard_idx: int, row_idx: int) -> int:
        s = int(shard_idx) % self.num_shards
        r = int(row_idx) % self.shard_rows[s]
        return (self.cumulative_rows[s] + r) % self.total_rows

    def advance_cursor(self, shard_idx: int, row_idx: int, delta_rows: int) -> tuple[int, int]:
        g = self.global_row_from_cursor(shard_idx, row_idx)
        return self.cursor_from_global_row(g + int(delta_rows))

    def get_batch_numpy(
        self,
        shard_idx: int,
        row_idx: int,
        total_batch: int,
        seq_len: int | None = None,
    ) -> tuple[np.ndarray, int, int]:
        """
        Reads `total_batch` contiguous rows starting at `(shard_idx, row_idx)`,
        seamlessly spanning shard boundaries and wrapping around the final shard.
        Returns `(batch_np, next_shard_idx, next_row_idx)`.
        """
        if total_batch < 1:
            raise ValueError("total_batch must be positive")
        width = self.seq_len if seq_len is None else int(seq_len)
        if not 0 < width <= self.seq_len:
            raise ValueError(f"seq_len ({width}) must be in (0, {self.seq_len}]")

        s = int(shard_idx) % self.num_shards
        r = int(row_idx) % self.shard_rows[s]

        chunks: list[np.ndarray] = []
        remaining = total_batch
        while remaining > 0:
            mat = self._get_shard_matrix(s)
            avail = self.shard_rows[s] - r
            take = min(remaining, avail)
            chunks.append(np.ascontiguousarray(mat[r : r + take, :width]))
            remaining -= take
            r += take
            if r >= self.shard_rows[s]:
                s = (s + 1) % self.num_shards
                r = 0

        if len(chunks) == 1:
            batch = chunks[0].copy()
        else:
            batch = np.concatenate(chunks, axis=0)
        return batch, s, r

    def get_batch_pytorch(
        self,
        shard_idx: int,
        row_idx: int,
        total_batch: int,
        seq_len: int,
        device,
        non_blocking: bool = True,
    ):
        """Fetches a batch from `(shard_idx, row_idx)` and transfers to a PyTorch tensor."""
        import torch

        batch, next_s, next_r = self.get_batch_numpy(shard_idx, row_idx, total_batch, seq_len=seq_len)
        if batch.dtype in (np.int32, np.int64):
            tensor = torch.from_numpy(batch)
        elif batch.dtype == np.uint16:
            try:
                tensor = torch.from_numpy(batch)
            except TypeError:
                tensor = torch.from_numpy(batch.astype(np.int32))
        else:
            tensor = torch.from_numpy(batch.astype(np.int32))

        if str(device).startswith("cuda") and torch.cuda.is_available():
            if not tensor.is_pinned():
                tensor = tensor.pin_memory()

        res = tensor.to(device, dtype=torch.long, non_blocking=non_blocking)
        del tensor, batch
        return res, next_s, next_r

    def signature(self) -> dict:
        """Fast deterministic dataset manifest signature for resume validation."""
        entries = []
        for p, dt, n_rows in zip(self.shard_paths, self.shard_dtypes, self.shard_rows):
            rel = str(p.relative_to(self.data_dir))
            entries.append({
                "name": rel,
                "size_bytes": p.stat().st_size,
                "dtype": str(dt),
                "rows": n_rows,
                "head_tail_sha256": _head_tail_sha256(p),
            })
        return {
            "kind": "sharded_memmap_v1",
            "seq_len": self.seq_len,
            "total_rows": self.total_rows,
            "num_shards": self.num_shards,
            "shards": entries,
        }

