import sys
import types
from .prepare import prepare_dataset, iterate_text_sources, main

class _DataPrepModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        """Prepares dataset directly when calling telos.dataprep(...)"""
        return prepare_dataset(*args, **kwargs)

sys.modules[__name__].__class__ = _DataPrepModule

__all__ = [
    "prepare_dataset",
    "iterate_text_sources",
    "main",
]
