"""Universal Model Adapters for Télos Evaluation."""

from .base import BaseModelAdapter
from .telos_native import TelosNativeAdapter
from .openai_api import OpenAIAPIAdapter
from .gemini_api import GeminiAPIAdapter
from .mlx_lm import MLXLMAdapter
from .swift_afm import SwiftAFMAdapter
from .registry import load_adapter

__all__ = [
    "BaseModelAdapter",
    "TelosNativeAdapter",
    "OpenAIAPIAdapter",
    "GeminiAPIAdapter",
    "MLXLMAdapter",
    "SwiftAFMAdapter",
    "load_adapter",
]

