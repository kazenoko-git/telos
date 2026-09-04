"""
télos (τέλος): Discrete Diffusion & Autoregressive Language Modeling Framework.
"""

from .models import TelosTransformer, TelosConfig, MLXTelosTransformer
from .training import UnifiedMLXTrainer, UnifiedPyTorchTrainer, HardwareProfile
from .dataprep.prepare import prepare_dataset as dataprep
from .train.cli import train
from .eval.runner import evaluate
from .bench.runner import benchmark

__all__ = [
    "dataprep",
    "train",
    "evaluate",
    "benchmark",
    "TelosTransformer",
    "TelosConfig",
    "MLXTelosTransformer",
    "UnifiedMLXTrainer",
    "UnifiedPyTorchTrainer",
    "HardwareProfile",
]

