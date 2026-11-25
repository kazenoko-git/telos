"""Bounded asynchronous demand reads for LFM's SSD expert cache."""

import fcntl
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock

from .ssd_experts import SSDExpertCache

_MIB = 2**20


class _ExpertReadSession:
    def __init__(self, fragment_size, async_size):
        self._sync_buffer = memoryview(bytearray(fragment_size))
        self._async_buffer = memoryview(bytearray(async_size))
        self._files = {}
        self._pending = {}
        self._workers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="telos-expert-read")

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
            data = self._read(self._fd(entry.path), entry, self._sync_buffer[:entry.size])
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

    def __init__(self, byte_limit, guard, *, layer_sizes, fragment_size, top_k):
        super().__init__(byte_limit, guard, layer_sizes=layer_sizes)
        self._fragment_size = fragment_size
        self._async_size = min(24 * _MIB, max(layer_sizes.values()) * top_k)
        self._session = None
        self._lock = Lock()

    @contextmanager
    def request(self):
        with self._lock:
            session = _ExpertReadSession(self._fragment_size, self._async_size)
            self._session = session
            try:
                yield
            finally:
                self._session = None
                session.close()

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
