import sys
import types
from .runner import benchmark, main

class _BenchModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        """Benchmarks model directly when calling telos.bench(...)"""
        return benchmark(*args, **kwargs)

sys.modules[__name__].__class__ = _BenchModule

__all__ = [
    "benchmark",
    "main",
]
