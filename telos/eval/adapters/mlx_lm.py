"""
Apple Silicon MLX-LM Model Adapter.

Provides high-performance native inference on Apple Silicon Macs using mlx-lm.
"""

from typing import List, Optional
from .base import BaseModelAdapter


class MLXLMAdapter(BaseModelAdapter):
    """Adapter for models running via mlx-lm on Apple Silicon."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_name=model_path, backend_name="mlx_lm")
        import mlx_lm

        print(f"Loading MLX-LM model from: {model_path}...")
        from pathlib import Path
        revision = kwargs.get('revision')
        if not Path(model_path).exists():
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_path, revision=revision)
        self.model_path = str(Path(model_path).resolve())
        self.model_revision = Path(self.model_path).name if 'snapshots' in Path(self.model_path).parts else revision
        self.model, self.tokenizer = mlx_lm.load(self.model_path)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generates completion using mlx_lm."""
        import mlx_lm

        sampler = None
        if temperature > 0.0:
            import mlx_lm.sample_utils as su
            sampler = su.make_sampler(temp=temperature)

        response = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
        )

        if stop:
            for s in stop:
                if s in response:
                    response = response[:response.index(s)]

        return response
