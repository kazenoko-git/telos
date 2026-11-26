"""Read-only Safetensors ranges and bounded expert residency on Apple Silicon."""

import json
import math
import os
import re
import subprocess
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from threading import Event, Thread

_MIB = 2**20
_EXPERT_KEY = re.compile(
    r"model\.layers\.\d+\.(?:mlp|feed_forward)\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases)"
)
_DTYPES = {"F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": ("<u2", 2), "U32": ("<u4", 4), "U8": ("u1", 1)}


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
        if uncached:
            import fcntl

        with self.path.open("rb", buffering=0) as file:
            if uncached:
                # Avoid replacing an explicit expert budget with unbounded file caching.
                fcntl.fcntl(file.fileno(), fcntl.F_NOCACHE, 1)
            data = os.pread(file.fileno(), self.size, self.offset)
        if len(data) != self.size:
            raise OSError(f"Short Safetensors read in {self.path.name} at byte {self.offset}")
        return self.array_from_buffer(data)

    def array_from_buffer(self, data):
        import mlx.core as mx
        import numpy as np

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


def expert_layer_sizes(ranges, experts):
    sizes = {}
    for name, entry in ranges.items():
        if is_expert_tensor(name):
            layer = int(name.split(".")[2])
            if len(entry.shape) != 3 or entry.shape[0] != experts:
                raise ValueError(f"Expected an expert-major tensor for {name}")
            sizes[layer] = sizes.get(layer, 0) + entry.size // experts
        elif any(f".{module}.{bank}." in name
                 for module in ("mlp", "feed_forward") for bank in ("switch_mlp", "experts")):
            raise ValueError(f"Unsupported expert tensor {name}")
    return sizes


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


def apple_memory_budget(physical_bytes, free_percent):
    """Available budget after the native loader's system headroom reserve."""
    return physical_bytes * max(0, free_percent - 15) // 100


class AppleMemoryGuard:
    """Use live headroom for admission and stop on new swap writes during streaming."""

    def __init__(self):
        physical = int(_command("sysctl", "-n", "hw.memsize"))
        free = _free_percent()
        self.budget = apple_memory_budget(physical, free)
        self._swap_start = _swapout_bytes()
        self._checked_at = 0.0
        self._background = False
        self._background_error = None

    def _sample(self):
        if _free_percent() < 15 or _swapout_bytes() - self._swap_start >= 128 * _MIB:
            raise RuntimeError("Flash-MoE stopped because of system memory pressure or new swap writes")

    def check(self):
        if self._background:
            if self._background_error is not None:
                raise RuntimeError("Flash-MoE background memory check failed") from self._background_error
            return
        now = time.monotonic()
        if now - self._checked_at < 0.25:
            return
        self._checked_at = now
        self._sample()

    @contextmanager
    def sample_in_background(self):
        """Keep monitoring steady tokens without waiting on subprocesses there."""
        if self._background:
            raise RuntimeError("Flash-MoE background sampling is already active")
        self._sample()
        stop = Event()
        self._background_error = None

        def monitor():
            while not stop.wait(0.25):
                try:
                    self._sample()
                except Exception as exc:
                    self._background_error = exc
                    return

        thread = Thread(target=monitor, name="telos-flash-moe-guard", daemon=True)
        thread.start()
        self._background = True
        try:
            yield
        finally:
            stop.set()
            thread.join()
            self._background = False
            self._checked_at = 0.0
        if self._background_error is not None:
            raise RuntimeError("Flash-MoE background memory check failed") from self._background_error


def allocate_layer_slots(byte_limit, layer_sizes, layer_weights=None):
    """Assign whole-expert per-layer slot quotas within a shared byte budget."""
    if not layer_sizes:
        return {}
    if layer_weights is None:
        if len(set(layer_sizes.values())) != 1:
            return {}
        slots = byte_limit // next(iter(layer_sizes.values()))
        n = len(layer_sizes)
        if slots < n:
            return {}
        return {layer: (i + 1) * slots // n - i * slots // n for i, layer in enumerate(sorted(layer_sizes))}

    counts = dict.fromkeys(layer_sizes, 0)
    used = 0
    ordered = sorted(layer_sizes)
    while True:
        best_layer = None
        best_score = -1.0
        for layer in ordered:
            weight = layer_weights.get(layer, 1.0)
            size = layer_sizes[layer]
            if weight > 0 and used + size <= byte_limit:
                score = weight / (counts[layer] + 1)
                if score > best_score:
                    best_score = score
                    best_layer = layer
        if best_layer is None:
            break
        counts[best_layer] += 1
        used += layer_sizes[best_layer]
    return counts


class SSDExpertCache:
    """Retain recently selected experts up to a shared byte budget."""

    def __init__(self, byte_limit, guard, *, layer_sizes=None, layer_weights=None):
        self.byte_limit = byte_limit
        self.guard = guard
        self.resident_bytes = 0
        self.bytes_read = 0
        self.hits = 0
        self.misses = 0
        self._entries = OrderedDict()
        self._protected = set()
        self.layer_slots = allocate_layer_slots(byte_limit, layer_sizes, layer_weights)

    @contextmanager
    def visit(self, layer, experts):
        """Protect unconsumed resident experts for the current routed layer visit."""
        self._protected = {(layer, expert) for expert in experts}
        try:
            yield
        finally:
            self._protected.clear()

    def _evict(self, key):
        self.resident_bytes -= self._entries.pop(key)[1]

    def _layer_victim(self, resident, requested):
        return resident[0]

    def get(self, layer, expert, ranges):
        key = (layer, expert)
        self._protected.discard(key)
        if key in self._entries:
            self.hits += 1
            self._entries.move_to_end(key)
            return self._entries[key][0]
        self.guard.check()

        slices = {name: entry.expert(expert) for name, entry in ranges.items()}
        size = sum(entry.size for entry in slices.values())
        if size > self.byte_limit:
            raise ValueError(f"One expert needs {size} bytes, over the Flash-MoE cache budget")
        self.misses += 1
        if layer in self.layer_slots:
            slots = self.layer_slots[layer]
            if slots == 0:
                return self._read_expert(slices, size)
            resident = [k for k in self._entries if k[0] == layer]
            if len(resident) >= slots:
                candidates = [k for k in resident if k not in self._protected]
                if not candidates:
                    return self._read_expert(slices, size)
                victim = self._layer_victim(candidates, key)
                if victim is None:
                    return self._read_expert(slices, size)
                self._evict(victim)
        while self.resident_bytes + size > self.byte_limit:
            victim = next((k for k in self._entries if k not in self._protected), None)
            if victim is None:
                return self._read_expert(slices, size)
            self._evict(victim)

        arrays = self._read_expert(slices, size)
        self._entries[key] = (arrays, size)
        self.resident_bytes += size
        return arrays

    def _read_expert(self, slices, size):
        import mlx.core as mx

        arrays = self._load_slices(slices)
        mx.eval(arrays)
        self.bytes_read += size
        return arrays

    def _load_slices(self, slices):
        import mlx.core as mx

        # MLX may retain freed buffers independently of the expert cache.
        mx.clear_cache()
        return {name: entry.load(uncached=True) for name, entry in slices.items()}
