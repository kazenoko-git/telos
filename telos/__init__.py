"""
télos (τέλος): Discrete Diffusion & Autoregressive Language Modeling Framework.
"""

from . import dataprep
from . import train
from . import eval
from . import bench
from .models.transformer import TelosTransformer, TelosConfig
from .training.trainer_pytorch import UnifiedPyTorchTrainer
from .training.hardware import HardwareProfile

# Primary callable aliases
prepare_dataset = dataprep.prepare_dataset
run_train = train.train
evaluate = eval.evaluate
benchmark = bench.benchmark

def __getattr__(name: str):
    if name == "MLXTelosTransformer":
        from .models import MLXTelosTransformer
        return MLXTelosTransformer
    if name == "UnifiedMLXTrainer":
        from .training import UnifiedMLXTrainer
        return UnifiedMLXTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "dataprep",
    "train",
    "run_train",
    "eval",
    "bench",
    "prepare_dataset",
    "evaluate",
    "benchmark",
    "TelosTransformer",
    "TelosConfig",
    "MLXTelosTransformer",
    "UnifiedMLXTrainer",
    "UnifiedPyTorchTrainer",
    "HardwareProfile",
]

