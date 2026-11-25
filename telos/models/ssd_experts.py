"""Read-only Safetensors ranges and bounded expert residency on Apple Silicon."""

import json
import math
import os
import re
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

_MIB = 2**20
_EXPERT_KEY = re.compile(
    r"model\.layers\.\d+\.(?:mlp|feed_forward)\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases)"
)
_DTYPES = {"F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": ("<u2", 2), "U32": ("<u4", 4)}


@dataclass(frozen=True)
class TensorRange:
    path: Path
    offset: int
    size: int
    dtype: str
    shape: tuple[int, ...]

    def expert(self, index):
        size = self.size // self.shape[0]
        return TensorRange(self.path, self.offset + index * size, size, self.dtype, self.shape[1:])

    def load(self, *, uncached=False):
        import mlx.core as mx
        import numpy as np

        if uncached:
            import fcntl

        with self.path.open("rb", buffering=0) as file:
            if uncached:
                # Avoid replacing an explicit expert budget with unbounded file caching.
                fcntl.fcntl(file.fileno(), fcntl.F_NOCACHE, 1)
            data = os.pread(file.fileno(), self.size, self.offset)
        if len(data) != self.size:
            raise OSError(f"Short Safetensors read in {self.path.name} at byte {self.offset}")
        array = mx.array(np.frombuffer(data, dtype=_DTYPES[self.dtype][0]).reshape(self.shape))
        return array.view(mx.bfloat16) if self.dtype == "BF16" else array


def _read_header(path):
    with path.open("rb") as file:
        header_size = int.from_bytes(file.read(8), "little")
        if not 2 <= header_size <= 100_000_000:
            raise ValueError(f"Invalid Safetensors header size in {path.name}")
        header = json.loads(file.read(header_size))
        payload_offset = 8 + header_size
        file_size = os.fstat(file.fileno()).st_size
    if not isinstance(header, dict):
        raise ValueError(f"Invalid Safetensors header in {path.name}")  # noqa: TRY004
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        try:
            shape = tuple(entry["shape"])
            dtype = entry["dtype"]
            start, end = entry["data_offsets"]
            itemsize = _DTYPES[dtype][1]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported Safetensors tensor {name}") from exc
        if (any(type(d) is not int or d <= 0 for d in shape)
                or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= file_size - payload_offset
                or end - start != math.prod(shape) * itemsize):
            raise ValueError(f"Invalid Safetensors range for {name}")
        tensors[name] = TensorRange(path, payload_offset + start, end - start, dtype, shape)
    ordered = sorted(tensors.values(), key=lambda entry: entry.offset)
    if any(a.offset + a.size > b.offset for a, b in pairwise(ordered)):
        raise ValueError(f"Overlapping Safetensors ranges in {path.name}")
    return tensors


def tensor_ranges(checkpoint):
    if checkpoint.suffix == ".safetensors":
        return _read_header(checkpoint)
    if checkpoint.name != "model.safetensors.index.json":
        raise ValueError("Flash-MoE requires MLX-format Safetensors")
    index = json.loads(checkpoint.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Invalid Safetensors weight index")
    root = checkpoint.parent.resolve()
    shards = {}
    for name, filename in weight_map.items():
        if not isinstance(filename, str):
            raise ValueError("Invalid Safetensors shard name")  # noqa: TRY004
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or path.suffix != ".safetensors":
            raise ValueError(f"Invalid Safetensors shard path: {filename}")
        shards.setdefault(path, set()).add(name)
    tensors = {}
    for path, names in shards.items():
        shard = _read_header(path)
        if shard.keys() != names:
            raise ValueError(f"Safetensors weight index does not match {path.name}")
        tensors.update(shard)
    return tensors


def is_expert_tensor(name):
    return _EXPERT_KEY.fullmatch(name) is not None


def _command(*args):
    return subprocess.check_output(args, text=True, timeout=3)


def _free_percent():
    match = re.search(r"System-wide memory free percentage: (\d+)%", _command("memory_pressure", "-Q"))
    if not match:
        raise RuntimeError("Cannot determine available macOS memory for Flash-MoE")
    return int(match[1])


def _swapout_bytes():
    stats = _command("vm_stat")
    page_size = re.search(r"page size of (\d+) bytes", stats)
    pages = re.search(r"^Swapouts:\s+(\d+)\.", stats, re.MULTILINE)
    if not page_size or not pages:
        raise RuntimeError("Cannot read macOS swap-out counters for Flash-MoE")
    return int(page_size[1]) * int(pages[1])


class AppleMemoryGuard:
    """Use live headroom for admission and stop on new swap writes during streaming."""

    def __init__(self):
        physical = int(_command("sysctl", "-n", "hw.memsize"))
        free = _free_percent()
        self.budget = physical * max(0, free - 15) // 100
        self._swap_start = _swapout_bytes()
        self._checked_at = 0.0

    def check(self):
        now = time.monotonic()
        if now - self._checked_at < 0.25:
            return
        self._checked_at = now
        if _free_percent() < 15 or _swapout_bytes() - self._swap_start >= 128 * _MIB:
            raise RuntimeError("Flash-MoE stopped because of system memory pressure or new swap writes")


class SSDExpertCache:
    """Retain recently selected experts up to a shared byte budget."""

    def __init__(self, byte_limit, guard):
        self.byte_limit = byte_limit
        self.guard = guard
        self.resident_bytes = 0
        self.bytes_read = 0
        self._entries = OrderedDict()

    def get(self, layer, expert, ranges):
        self.guard.check()
        key = (layer, expert)
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key][0]

        slices = {name: entry.expert(expert) for name, entry in ranges.items()}
        size = sum(entry.size for entry in slices.values())
        if size > self.byte_limit:
            raise ValueError(f"One expert needs {size} bytes, over the Flash-MoE cache budget")
        while self.resident_bytes + size > self.byte_limit:
            oldest = next(iter(self._entries))
            evicted_size = self._entries.pop(oldest)[1]
            self.resident_bytes -= evicted_size

        import mlx.core as mx

        # MLX can retain freed buffers in its allocator cache, independently of
        # this cache. Reclaim them before staging the next on-disk expert.
        mx.clear_cache()
        arrays = {}
        for name, entry in slices.items():
            arrays[name] = entry.load(uncached=True)
            self.bytes_read += entry.size
        mx.eval(arrays)
        self._entries[key] = (arrays, size)
        self.resident_bytes += size
        return arrays
