"""
Universal Model Adapter Registry & Factory for Télos.

Automatically routes model strings and checkpoints to the appropriate adapter:
- AFM 3 Core / AFM 3 Core Advanced -> native Swift FoundationModels bridge
  (see ``telos.afm``), or an OpenAI-compatible endpoint when ``api_base`` is given
- Ternary Bonsai 27B -> HuggingFace or OpenAI API adapter
- Gemma 4 e4b / Gemma 4 12B -> HuggingFace or MLX-LM adapter
- Gemini 4 26B A4B -> Google Gemini REST adapter
- Local .pt / .safetensors checkpoints -> Télos Native adapter
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from .base import BaseModelAdapter
from .telos_native import TelosNativeAdapter
from .openai_api import OpenAIAPIAdapter
from .gemini_api import GeminiAPIAdapter


def load_adapter(
    model_identifier: str,
    backend: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    concurrency: int = 8,
    device: str = "auto",
    torch_dtype: str = "bfloat16",
    quantization: Optional[str] = None,
    **kwargs
) -> BaseModelAdapter:
    """
    Factory function to instantiate the proper BaseModelAdapter.

    Args:
        model_identifier: Checkpoint path, Hugging Face repo ID, or model name
                          (e.g. 'afm-3-core', 'ternary-bonsai-27b', 'gemini-4-26b-a4b', 'gemma-4-12b').
        backend: Explicit backend override ('telos_native', 'openai_api', 'gemini_api', 'huggingface', 'mlx_lm').
        api_base: Base URL for OpenAI-compatible endpoints (e.g. 'http://localhost:8000/v1').
        api_key: API key for OpenAI or Gemini.
        concurrency: Number of concurrent worker threads for API adapters.
        device: Torch device ('cpu', 'cuda', 'mps', 'auto').
        torch_dtype: Model weights precision ('bfloat16', 'float16', 'float32').
        quantization: Quantization mode ('4bit', '8bit', None).
    """
    m_lower = model_identifier.lower().strip()

    # Explicit MLX-LM backend override
    if backend == "mlx_lm":
        from .mlx_lm import MLXLMAdapter
        return MLXLMAdapter(model_path=model_identifier, **kwargs)

    # Google Gemini / Gemma API routing
    if backend == "gemini_api" or "gemini" in m_lower or "gemma-4" in m_lower:
        gemini_model = model_identifier
        if "gemini-4" in m_lower and "gemma" not in m_lower:
            gemini_model = os.environ.get("GEMINI_4_MODEL_NAME", "gemini-2.5-flash")
        return GeminiAPIAdapter(
            model_name=gemini_model,
            api_key=api_key,
            concurrency=concurrency,
        )

    # Explicit Swift FoundationModels backend override
    if backend in ["swift", "swift_afm"]:
        from .swift_afm import SwiftAFMAdapter
        return SwiftAFMAdapter(model_identifier=model_identifier, **kwargs)

    # AFM models: route to native Swift FoundationModels on Apple Silicon by default
    if "afm-3" in m_lower or "afm_3" in m_lower or "afm" in m_lower:
        if api_base is not None or backend == "openai_api":
            return OpenAIAPIAdapter(
                model_name=model_identifier,
                api_base=api_base,
                api_key=api_key,
                concurrency=concurrency,
            )
        if backend == "mlx_lm":
            from .mlx_lm import MLXLMAdapter
            return MLXLMAdapter(model_path=model_identifier, **kwargs)
        from .swift_afm import SwiftAFMAdapter
        return SwiftAFMAdapter(model_identifier=model_identifier, **kwargs)

    # OpenAI-compatible REST endpoints
    if (
        backend == "openai_api"
        or api_base is not None
        or m_lower.startswith("http://")
        or m_lower.startswith("https://")
    ):
        base_url = api_base or "http://localhost:8000/v1"
        return OpenAIAPIAdapter(
            model_name=model_identifier,
            api_base=base_url,
            api_key=api_key,
            concurrency=concurrency,
        )

    # Local Télos Checkpoints (.safetensors, .pt, or directory containing them)
    path_obj = Path(model_identifier)
    if (
        backend == "telos_native"
        or path_obj.is_file()
        or (path_obj.is_dir() and (
            (path_obj / "model.safetensors").exists()
            or (path_obj / "checkpoint_final.pt").exists()
            or list(path_obj.glob("checkpoint_step_*.pt"))
        ))
    ):
        return TelosNativeAdapter(checkpoint_path=path_obj, **kwargs)

    # Hugging Face Transformers (Default for outside open-weights: Gemma 4, Ternary Bonsai 27B, etc.)
    try:
        from .huggingface import HuggingFaceAdapter
        return HuggingFaceAdapter(
            model_name_or_path=model_identifier,
            device=device,
            torch_dtype=torch_dtype,
            quantization=quantization,
            **kwargs
        )
    except Exception as exc:
        # Fall back to OpenAIAPIAdapter if HF import/download fails or user intends local vLLM serving
        print(f"Notice: HuggingFace adapter failed to initialize ({exc}). Routing to OpenAI-compatible REST adapter...")
        return OpenAIAPIAdapter(
            model_name=model_identifier,
            api_base=api_base or "http://localhost:8000/v1",
            api_key=api_key,
            concurrency=concurrency,
        )
