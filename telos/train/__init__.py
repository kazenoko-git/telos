import sys
import types
from .cli import train, main

class _TrainModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        """Trains model directly when calling telos.train(...)"""
        return train(*args, **kwargs)

sys.modules[__name__].__class__ = _TrainModule

__all__ = [
    "train",
    "main",
]
