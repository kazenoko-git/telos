"""
Base Model Adapter for Télos Evaluation Framework.

Defines the abstract interface that all model backends must satisfy:
- Télos Native (COROSred, AR, MDLM, UNDLM)
- Hugging Face Transformers (Gemma 4 12B, Ternary Bonsai 27B, etc.)
- Apple Silicon MLX (mlx_lm)
- OpenAI-compatible REST API (vLLM, SGLang, Ollama, AFM 3 Core)
- Google Gemini API (Gemini 4 26B A4B, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseModelAdapter(ABC):
    """Abstract Base Class for Universal Model Evaluation."""

    def __init__(self, model_name: str, backend_name: str):
        self.model_name = model_name
        self.backend_name = backend_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generates a text completion for the provided prompt.

        Args:
            prompt: Text prompt input to the model.
            max_new_tokens: Maximum number of newly generated tokens.
            temperature: Sampling temperature (0.0 = deterministic greedy).
            stop: List of stop strings that terminate generation.

        Returns:
            The generated continuation string.
        """
        pass

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generates completions for a batch of prompts.
        Default implementation sequentially invokes generate().
        Subclasses can override for vector or parallel execution.
        """
        return [
            self.generate(
                p,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop=stop,
                **kwargs
            )
            for p in prompts
        ]

    def get_logprobs(self, prompt: str, target: str) -> float:
        """
        Computes the log probability of a target continuation given the prompt.
        Applicable to local open-weights / native models for probe evaluations.
        For black-box APIs without logprob support, returns -float('inf').
        """
        raise NotImplementedError(
            f"Logprob evaluation is not supported by adapter: {self.__class__.__name__}"
        )
