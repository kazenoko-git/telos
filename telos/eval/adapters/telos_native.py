"""
Télos Native Model Adapter.

Wraps native Télos architectures:
- Continuous Relaxed Diffusion (COROSred)
- Causal Autoregressive (AR)
- Masked Discrete Diffusion (MDLM)
- Uniform Discrete Diffusion (UNDLM)

Supports both Apple Silicon MLX and PyTorch runtimes.
"""

from typing import List, Optional, Any, Dict
from pathlib import Path
import numpy as np

from .base import BaseModelAdapter


class TelosNativeAdapter(BaseModelAdapter):
    """Adapter wrapping native Télos models (MLX or PyTorch)."""

    def __init__(self, checkpoint_path: str | Path, config: dict | None = None, tokenizer=None):
        super().__init__(model_name=str(checkpoint_path), backend_name="telos_native")
        from telos.eval.runner import load_model_from_checkpoint
        from telos.data.tokenizer import load_tokenizer

        self.model, self.backend, self.vocab_size = load_model_from_checkpoint(checkpoint_path, config)
        self.tokenizer = tokenizer or load_tokenizer()
        self.paradigm = getattr(self.model, "paradigm", "ar")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generates continuation using native greedy decoding."""
        from telos.eval.runner import _generate_greedy_completion
        return _generate_greedy_completion(
            model=self.model,
            tokenizer=self.tokenizer,
            backend=self.backend,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    def get_logprobs(self, prompt: str, target: str) -> float:
        """Computes cross-entropy / logprob of target continuation given prompt."""
        p_ids = self.tokenizer.encode(prompt).ids
        full_ids = self.tokenizer.encode(prompt + target).ids
        if len(full_ids) <= len(p_ids):
            return -float("inf")

        target_len = len(full_ids) - len(p_ids)

        if self.backend == "mlx":
            import mlx.core as mx
            x = mx.array([full_ids], dtype=mx.int32)
            logits = self.model(x)  # [1, seq_len, vocab_size]
            logits_np = np.array(logits[0].astype(mx.float32))
        else:
            import torch
            device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else torch.device("cpu")
            x = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = self.model(x)
            logits_np = logits[0].cpu().float().numpy()

        # Compute log-softmax along sequence for target tokens
        log_probs = 0.0
        for i in range(len(p_ids) - 1, len(full_ids) - 1):
            target_token_id = full_ids[i + 1]
            pos_logits = logits_np[i]
            max_l = np.max(pos_logits)
            lse = max_l + np.log(np.sum(np.exp(pos_logits - max_l)))
            token_logp = float(pos_logits[target_token_id] - lse)
            log_probs += token_logp

        return log_probs / max(1, target_len)
