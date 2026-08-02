"""
télos MDLM — Apple MLX Inference & Code Generation Sampler
===========================================================
Generates Python code completions using discrete reverse diffusion (MDLM sampler)
from a trained MLX safetensors checkpoint.
"""

import argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer


# Import MLX Model from train_mlx.py
from train_mlx import MLXTelosTransformer, tree_flatten


def sample_mdlm_mlx(
    model: MLXTelosTransformer,
    prompt_tokens: list[int],
    seq_len: int = 512,
    mask_token_id: int = 1,
    steps: int = 64,
    temperature: float = 0.8
) -> list[int]:
    """MDLM discrete reverse diffusion sampling in MLX."""
    prompt_len = len(prompt_tokens)
    assert prompt_len < seq_len, f"Prompt length {prompt_len} exceeds max seq len {seq_len}"

    # Initialize sequence: prompt + [MASK] tokens
    x = np.full((1, seq_len), mask_token_id, dtype=np.int32)
    x[0, :prompt_len] = prompt_tokens
    x = mx.array(x)

    # Reverse diffusion steps
    timesteps = np.linspace(1.0, 1e-4, steps)

    for i in range(steps - 1):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]

        # Predict logits
        logits = model(x)
        B, T, V = logits.shape

        # Scale logits by temperature for sampling diversity
        if temperature != 1.0:
            logits = logits / temperature

        # Sample categorical token predictions
        probs = mx.softmax(logits, axis=-1)
        pred_tokens = mx.random.categorical(logits.reshape(-1, V)).reshape(B, T)

        # Unmask probability from t_curr -> t_next
        unmask_prob = 1.0 - (t_next / t_curr)
        unmask_rand = mx.random.uniform(0.0, 1.0, (B, T))

        # Identify currently masked tokens (excluding prompt)
        is_masked = (x == mask_token_id)
        prompt_mask = mx.array([[True if idx < prompt_len else False for idx in range(seq_len)]])
        is_masked = is_masked & (~prompt_mask)

        # Decide which masked positions to reveal in this step
        should_unmask = is_masked & (unmask_rand < unmask_prob)
        x = mx.where(should_unmask, pred_tokens, x)

    # Final step: fill any remaining masks
    logits = model(x)
    pred_tokens = mx.argmax(logits, axis=-1)
    is_masked = (x == mask_token_id)
    x = mx.where(is_masked, pred_tokens, x)

    return x[0].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase_b_25m_mlx/checkpoint_step_15000.safetensors")
    parser.add_argument("--tokenizer", type=str, default="configs/tokenizer_mac.json")
    parser.add_argument("--prompt", type=str, default="def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.8)
    args = parser.parse_args()

    print("=" * 65)
    print("  télos MDLM — Apple MLX Code Completion Sampler")
    print("=" * 65)

    # Load Tokenizer
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        print(f"Error: Tokenizer not found at {tok_path}")
        return
    tokenizer = Tokenizer.from_file(str(tok_path))

    # Initialize 25M Model
    model = MLXTelosTransformer(
        vocab_size=8192,
        d_model=512,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        return

    print(f"  Loading weights from {ckpt_path}...")
    model.load_weights(str(ckpt_path))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())

    prompt_encoding = tokenizer.encode(args.prompt)
    prompt_tokens = prompt_encoding.ids

    print(f"\n  Prompt: {repr(args.prompt)}")
    print(f"  Prompt tokens: {len(prompt_tokens)} | Reverse Diffusion Steps: {args.steps}")
    print("─" * 65)
    print("  Generating completion...")

    generated_tokens = sample_mdlm_mlx(
        model=model,
        prompt_tokens=prompt_tokens,
        seq_len=512,
        mask_token_id=1,
        steps=args.steps,
        temperature=args.temp
    )

    generated_code = tokenizer.decode(generated_tokens)

    print("\n" + "=" * 65)
    print("  GENERATED CODE COMPLETION:")
    print("=" * 65)
    print(generated_code)
    print("=" * 65)


if __name__ == "__main__":
    main()
