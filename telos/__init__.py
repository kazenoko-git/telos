"""
télos (τέλος): Discrete Diffusion & Autoregressive Language Modeling Framework.
"""

from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("telos")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0.dev0"

from . import dataprep
from . import train
from . import eval
from . import bench
from .training.hardware import HardwareProfile

from .models.config import TelosConfig

# Primary callable aliases
prepare_dataset = dataprep.prepare_dataset
run_train = train.train
evaluate = eval.evaluate
benchmark = bench.benchmark

_TORCH_IMPORTS = {
    "TelosTransformer": ".models.transformer",
    "UnifiedPyTorchTrainer": ".training.trainer_pytorch",
}

def __getattr__(name: str):
    if name == "afm":
        import importlib
        return importlib.import_module(".afm", __package__)
    if name == "MLXTelosTransformer":
        from .models import MLXTelosTransformer
        return MLXTelosTransformer
    if name == "UnifiedMLXTrainer":
        from .training import UnifiedMLXTrainer
        return UnifiedMLXTrainer
    if name in _TORCH_IMPORTS:
        try:
            import importlib
            mod = importlib.import_module(_TORCH_IMPORTS[name], __package__)
            return getattr(mod, name)
        except ImportError as err:
            raise ImportError(
                f"{name} requires 'torch', which is not available in this environment. "
                "Install it via `pip install torch`."
            ) from err
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "dataprep",
    "train",
    "run_train",
    "eval",
    "bench",
    "prepare_dataset",
    "evaluate",
    "benchmark",
    "afm",
    "TelosTransformer",
    "TelosConfig",
    "MLXTelosTransformer",
    "UnifiedMLXTrainer",
    "UnifiedPyTorchTrainer",
    "HardwareProfile",
]

