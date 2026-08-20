"""
Standalone high-level Model Hub inference API for télos Autoregressive Language Models (AR).

Provides the TelosModel.from_pretrained() API for loading checkpoints and performing
causal next-token code completion.
"""

from pathlib import Path
import json
import numpy as np
from tokenizers import Tokenizer

try:
    import mlx.core as mx
    import mlx.nn as nn
    from ar.model.mlx_components import MLXCausalTransformer
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class TelosModel:
    """High-level standalone inference wrapper for Autoregressive (AR) causal language models."""

    def __init__(self, model, tokenizer: Tokenizer, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    def from_pretrained(cls, model_path_or_repo: str | Path):
        """Loads a pretrained AR model, tokenizer, and config from a local checkpoint directory."""
        model_path = Path(model_path_or_repo)
        assert model_path.exists(), f"Model path not found: {model_path}"

        # 1. Load config
        cfg_file = model_path / "config.json"
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                config = json.load(f)
        else:
            config = {"vocab_size": 8192, "d_model": 256, "n_layers": 13, "n_heads": 4, "n_kv_heads": 4, "seq_len": 512}

        # 2. Load tokenizer
        tok_file = None
        for p in [model_path / "tokenizer.json", Path("configs/shared/tokenizer_mac.json"), Path("configs/tokenizer_mac.json")]:
            if p.exists():
                tok_file = p
                break
        assert tok_file is not None, "Tokenizer file could not be located"
        tokenizer = Tokenizer.from_file(str(tok_file))

        # 3. Load model weights
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for Apple Silicon AR standalone inference.")

        model = MLXCausalTransformer(
            vocab_size=config.get("vocab_size", 8192),
            d_model=config.get("d_model", 256),
            n_layers=config.get("n_layers", 13),
            n_heads=config.get("n_heads", 4),
            n_kv_heads=config.get("n_kv_heads", 4)
        )

        weights_file = None
        for name in ["model.safetensors", "model.pt"]:
            if (model_path / name).exists():
                weights_file = model_path / name
                break
        if weights_file is None:
            st_files = sorted(list(model_path.glob("*.safetensors")), key=lambda p: p.stat().st_mtime)
            if st_files:
                weights_file = st_files[-1]

        assert weights_file is not None, f"No model weights found in {model_path}"
        model.load_weights(str(weights_file), strict=False)
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())

        return cls(model=model, tokenizer=tokenizer, config=config)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
        """Executes causal autoregressive next-token code generation."""
        tokens = self.tokenizer.encode(prompt).ids
        prompt_len = len(tokens)
        eos_id = self.tokenizer.token_to_id("<|endoftext|>") or 3

        for _ in range(max_tokens):
            seq_mx = mx.array([tokens], dtype=mx.int32)
            logits = self.model(seq_mx)[0, -1]  # [V]

            if temperature > 0:
                scaled_logits = logits.astype(mx.float32) / temperature
                probs = nn.softmax(scaled_logits, axis=-1)
                next_token = int(mx.random.categorical(scaled_logits))
            else:
                next_token = int(mx.argmax(logits))

            tokens.append(next_token)
            if next_token == eos_id or next_token == 0:
                break

        completion_tokens = tokens[prompt_len:]
        full_text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)

        for stop_str in ["[EOS]", "[PAD]", "<|endoftext|>"]:
            if stop_str in full_text:
                full_text = full_text.split(stop_str)[0]

        return full_text.rstrip()
