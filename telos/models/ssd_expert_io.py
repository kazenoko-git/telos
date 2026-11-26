"""Bounded asynchronous demand reads for LFM's SSD expert cache."""

import fcntl
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock

from .ssd_experts import SSDExpertCache

_MIB = 2**20


class _ExpertReadSession:
    def __init__(self, fragment_size, async_size, profile=None):
        self._sync_buffer = memoryview(bytearray(fragment_size))
        self._async_buffer = memoryview(bytearray(async_size))
        self._files = {}
        self._pending = {}
        self._workers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="telos-expert-read")
        self._profile = profile

    def _fd(self, path):
        # Only the inference thread opens and closes descriptors.
        if path not in self._files:
            fd = os.open(path, os.O_RDONLY)
            try:
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            except BaseException:
                os.close(fd)
                raise
            self._files[path] = fd
        return self._files[path]

    @staticmethod
    def _read(fd, entry, view):
        if os.preadv(fd, [view], entry.offset) != entry.size:
            raise OSError(f"Short Safetensors read in {entry.path.name} at byte {entry.offset}")
        return view

    def queue(self, entries):
        if self._pending:
            raise RuntimeError("Expert read queue was not drained before the next routed layer")
        offset = 0
        for entry in entries:
            end = offset + entry.size
            if end > len(self._async_buffer):
                break
            fd = self._fd(entry.path)
            self._pending[entry] = self._workers.submit(self._read, fd, entry, self._async_buffer[offset:end])
            offset = end

    def load(self, entry):
        future = self._pending.pop(entry, None)
        if future is None:
            fd, view = self._fd(entry.path), self._sync_buffer[:entry.size]
            if self._profile is None:
                data = self._read(fd, entry, view)
            else:
                data = self._profile.read(self._read, fd, entry, view, asynchronous=False)
        elif self._profile is not None and not future.done():
            data = self._profile.read(future.result, asynchronous=True)
        else:
            data = future.result()
        # MLX copies the NumPy input before another read can reuse this storage.
        return entry.array_from_buffer(data)

    def close(self):
        try:
            self._workers.shutdown(wait=True, cancel_futures=True)
        finally:
            for fd in self._files.values():
                os.close(fd)


class LFMSSDExpertCache(SSDExpertCache):
    """Per-layer eviction with request-scoped, reusable read resources."""

    def __init__(self, byte_limit, guard, *, layer_sizes, fragment_size, top_k,
                 layer_weights=None, decode_decay=1.0):
        super().__init__(byte_limit, guard, layer_sizes=layer_sizes, layer_weights=layer_weights)
        self._layer_sizes = layer_sizes.copy()
        self._window_slots = None
        self._fragment_size = fragment_size
        self._async_size = min(24 * _MIB, max(layer_sizes.values()) * top_k)
        self._decode_decay = decode_decay
        self._session = None
        self.profile = None
        self._decode_uses = None
        self._lock = Lock()

    def reserve_window_slots(self, slots):
        if self._session is not None:
            raise RuntimeError("Configure window expert slots before starting an SSD request")
        if slots.keys() != self._layer_sizes.keys() or any(n < 1 for n in slots.values()):
            raise ValueError("Window expert pinning requires a positive slot count for every bank")
        required = sum(self._layer_sizes[layer] * count for layer, count in slots.items())
        if required > self.byte_limit:
            raise ValueError(f"Window experts need {required / _MIB:.2f} MiB but the cache has {self.byte_limit / _MIB:.2f} MiB")
        self._window_slots = slots.copy()
        return required

    @contextmanager
    def request(self, *, profile=None):
        with self._lock:
            prefill_slots = self.layer_slots
            session = _ExpertReadSession(self._fragment_size, self._async_size, profile)
            self._session = session
            self.profile = profile
            try:
                yield
            finally:
                self._decode_uses = None
                self.layer_slots = prefill_slots
                self._session = None
                self.profile = None
                session.close()

    def begin_decode(self):
        if self._session is not None and self._window_slots is not None:
            self.layer_slots = self._window_slots
            self._decode_uses = None
            kept = defaultdict(int)
            for key in reversed(list(self._entries)):
                layer, _ = key
                if kept[layer] == self.layer_slots[layer]:
                    self._evict(key)
                else:
                    kept[layer] += 1
            return
        if self._session is not None and self.layer_slots:
            self._decode_uses = defaultdict(float if self._decode_decay != 1.0 else int)

    def step_decode(self):
        if self._decode_uses is not None and self._decode_decay != 1.0:
            for key in list(self._decode_uses):
                self._decode_uses[key] *= self._decode_decay

    def get(self, layer, expert, ranges):
        if self._decode_uses is not None:
            self._decode_uses[layer, expert] += 1
        return super().get(layer, expert, ranges)

    def _layer_victim(self, resident, requested):
        if self._decode_uses is None:
            return super()._layer_victim(resident, requested)
        # The resident list is already in LRU order, which breaks frequency ties.
        victim = min(resident, key=self._decode_uses.__getitem__)
        if self._decode_uses[requested] < self._decode_uses[victim]:
            return None
        return victim

    def prepare(self, layer, experts, ranges):
        if self._session is None:
            return
        self.guard.check()
        self._session.queue(entry.expert(expert) for expert in sorted(experts)
                            if (layer, expert) not in self._entries for entry in ranges.values())

    def _load_slices(self, slices):
        if self._session is None:
            return super()._load_slices(slices)

        import mlx.core as mx

        # Reuse the allocator's free buffers up to the scratch allowance.
        if mx.get_cache_memory() > 128 * _MIB:
            mx.clear_cache()
        return {name: self._session.load(entry) for name, entry in slices.items()}
