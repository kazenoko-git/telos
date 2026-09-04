import sys
import types
from .runner import evaluate, main
from .probes import PROBE_SUITE_100

class _EvalModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        """Evaluates model directly when calling telos.eval(...)"""
        return evaluate(*args, **kwargs)

sys.modules[__name__].__class__ = _EvalModule

__all__ = [
    "evaluate",
    "main",
    "PROBE_SUITE_100",
]
