"""Shared memory admission and runtime checks for MLX post-training."""

import os
import time

from telos.models.ssd_experts import _MIB, _command, _free_percent, _swapout_bytes


def _rss_bytes():
    return int(_command("ps", "-o", "rss=", "-p", str(os.getpid()))) * 1024


class MLXPosttrainMemory:
    def __init__(self, budget_mib):
        import mlx.core as mx

        if budget_mib is not None and (type(budget_mib) is not int or budget_mib < 1):
            raise ValueError("Post-training memory budget must be a positive number of MiB")
        self.mx = mx
        self._rss_start = _rss_bytes()
        physical = int(_command("sysctl", "-n", "hw.memsize"))
        headroom = physical * max(0, _free_percent() - 15) // 100
        self.budget = min(self._rss_start + headroom, mx.device_info()["max_recommended_working_set_size"])
        if budget_mib is not None:
            self.budget = min(self.budget, budget_mib * _MIB)
        self._swap_start = _swapout_bytes()
        self._checked_at = float("-inf")

    def __enter__(self):
        self.mx.clear_cache()
        self.check(force=True)
        self._allocator_limit = self.mx.set_memory_limit(self.budget)
        self._cache_limit = self.mx.set_cache_limit(0)
        return self

    def __exit__(self, *exc):
        self.mx.set_cache_limit(self._cache_limit)
        self.mx.set_memory_limit(self._allocator_limit)

    def model_budget_mib(self, *, separate_teacher):
        # Inference planners reserve workspace within each share. Leave the
        # existing process footprint outside those shares.
        available = self.budget - max(self._rss_start, self.mx.get_active_memory())
        share = available // ((2 if separate_teacher else 1) * _MIB)
        if share < 1:
            raise RuntimeError("No RAM remains for checkpoint weights under the post-training memory budget")
        return share

    def check(self, *, force=False):
        now = time.monotonic()
        if not force and now - self._checked_at < 0.25:
            return
        self._checked_at = now
        if _swapout_bytes() > self._swap_start or _free_percent() < 15:
            raise RuntimeError("MLX post-training stopped because of low system memory or new swap writes")
        rss = _rss_bytes()
        mlx_bytes = self.mx.get_active_memory() + self.mx.get_cache_memory()
        if max(rss, mlx_bytes) >= self.budget:
            raise RuntimeError(
                f"MLX post-training exceeded its {self.budget / _MIB:.0f} MiB memory budget "
                f"(RSS {rss / _MIB:.0f} MiB, MLX {mlx_bytes / _MIB:.0f} MiB)"
            )
