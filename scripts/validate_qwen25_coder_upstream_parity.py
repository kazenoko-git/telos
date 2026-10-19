#!/usr/bin/env python3
"""Validate Qwen2.5-Coder custom implementation against upstream Hugging Face endpoint.

Validations:
1. Mask Token Round-Trip: Read mask ID dynamically from the diffusion tokenizer and assert round trip.
2. Actual Masked Example: Compare logits, cosine similarity (>0.999), and top token predictions
   on an actual code sequence containing the mask token.
3. End-to-End Generation: Compare outputs of the documented native decoder using upstream model vs
   custom implementation, asserting exact token equality.
"""
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import sys
import types
from pathlib import Path

# Inject dummy dllm module to prevent HF modeling_qwen2 import crash
if "dllm" not in sys.modules:
    sys.modules["dllm"] = types.ModuleType("dllm")

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qwen25_coder_model import (
    load_qwen25_coder_model,
    get_diff_weights_path,
)


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Documented native transfer schedule from official dLLM README."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, :remainder[i]] += 1
    return num_transfer


@torch.no_grad()
def native_generate(
    model_fn,
    prompt_tensor: torch.Tensor,
    prompt_lens: torch.Tensor,
    pad_id: int,
    mask_id: int,
    steps: int = 64,
    max_new_tokens: int = 64,
    block_size: int = 64,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Documented native decoder preserving committed tokens and handling every position."""
    batch_size = prompt_tensor.size(0)
    total_length = int(prompt_lens.max().item() + max_new_tokens)
    x = torch.full((batch_size, total_length), pad_id, dtype=torch.long, device=prompt_tensor.device)
    for i, length in enumerate(prompt_lens.tolist()):
        x[i, :length] = prompt_tensor[i, :length]
        x[i, length : length + max_new_tokens] = mask_id

    positions = torch.arange(total_length, device=x.device)
    num_blocks = max_new_tokens // block_size
    assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_lens + num_block * block_size
        block_end = block_start + block_size
        init_block_mask = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
            & (x == mask_id)
        )
        num_transfer_tokens = get_num_transfer_tokens(init_block_mask, steps_per_block)

        for i in range(steps_per_block):
            # Only unmasked/uncommitted positions are eligible for transfer
            block_mask = (
                (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                & (x == mask_id)
            )

            logits = model_fn(x)
            x0 = torch.argmax(logits, dim=-1)

            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            confidence = torch.full_like(x0_p, -float("inf"))
            confidence = torch.where(block_mask, x0_p, confidence)
            x0 = torch.where(block_mask, x0, x)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k == 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

        # Handle every output position: fallback ensure no masks remain in this block
        block_remain = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
            & (x == mask_id)
        )
        if block_remain.any():
            logits = model_fn(x)
            x0 = torch.argmax(logits, dim=-1)
            x = torch.where(block_remain, x0, x)

    return x


def test_mask_token_round_trip():
    print("=" * 60)
    print("1. VALIDATING MASK TOKEN ID DYNAMIC READ & ROUND TRIP")
    print("=" * 60)
    diff_weights = get_diff_weights_path()
    diff_dir = diff_weights.parent

    # Test via HF AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(str(diff_dir))
    mask_token_str = hf_tok.mask_token
    mask_id = hf_tok.mask_token_id
    print(f"Hugging Face Tokenizer: mask_token='{mask_token_str}', mask_token_id={mask_id}")
    assert mask_token_str == "<|mask|>", f"Unexpected mask token: {mask_token_str}"
    assert hf_tok.convert_ids_to_tokens(mask_id) == "<|mask|>", "HF ID-to-token round trip failed"
    assert hf_tok.convert_tokens_to_ids("<|mask|>") == mask_id, "HF token-to-ID round trip failed"

    # Test via tokenizers.Tokenizer (fast tokenizer)
    fast_tok = Tokenizer.from_file(str(diff_dir / "tokenizer.json"))
    fast_mask_id = fast_tok.token_to_id("<|mask|>")
    print(f"Fast Tokenizer: token_to_id('<|mask|>') = {fast_mask_id}")
    assert fast_mask_id == mask_id, f"ID mismatch: fast={fast_mask_id} vs HF={mask_id}"
    assert fast_tok.id_to_token(fast_mask_id) == "<|mask|>", "Fast ID-to-token round trip failed"

    print(f">>> MASK TOKEN ROUND TRIP PASSED: mask_id={mask_id} ('<|mask|>') <<<\n")
    return mask_id


def test_actual_masked_example(device: str, mask_id: int):
    print("=" * 60)
    print("2. VALIDATING ACTUAL MASKED EXAMPLE AGAINST UPSTREAM")
    print("=" * 60)
    diff_weights = get_diff_weights_path()
    diff_dir = diff_weights.parent
    fast_tok = Tokenizer.from_file(str(diff_dir / "tokenizer.json"))

    prefix_str = "def is_even(n: int) -> bool:\n    return n % 2 "
    suffix_str = " 0\n"
    prefix_ids = fast_tok.encode(prefix_str, add_special_tokens=False).ids
    suffix_ids = fast_tok.encode(suffix_str, add_special_tokens=False).ids

    # Assemble sequence with exact mask token
    full_seq = prefix_ids + [mask_id] + suffix_ids
    mask_pos = len(prefix_ids)
    print(f"Constructed sequence length: {len(full_seq)}, Mask Position: {mask_pos}")

    input_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)

    # 1. Upstream evaluation
    print(f"Loading upstream AutoModelForMaskedLM on {device}...")
    up_model = AutoModelForMaskedLM.from_pretrained(
        str(diff_dir),
        dtype=torch.float32,
        trust_remote_code=True,
    ).to(device).eval()

    with torch.no_grad():
        up_logits = up_model(input_tensor).logits[0, mask_pos, :].float()

    del up_model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # 2. Custom evaluation
    print(f"Loading custom Qwen25CoderModel on {device}...")
    cu_model = load_qwen25_coder_model(diff_weights, device=device)

    with torch.no_grad():
        cu_logits, _ = cu_model(input_tensor, is_causal=False)
        cu_logits = cu_logits[0, mask_pos, :].float()

    # Comparison
    cos_sim = float(F.cosine_similarity(up_logits.unsqueeze(0), cu_logits.unsqueeze(0)).item())
    max_diff = float((up_logits - cu_logits).abs().max().item())
    up_top5 = torch.topk(up_logits, 5).indices.tolist()
    cu_top5 = torch.topk(cu_logits, 5).indices.tolist()

    print(f"  Cosine Similarity: {cos_sim:.8f} (Threshold: > 0.999)")
    print(f"  Max Absolute Diff: {max_diff:.4f}")
    print(f"  Upstream Top-5:    {up_top5} ({[fast_tok.decode([i]) for i in up_top5]})")
    print(f"  Custom Top-5:      {cu_top5} ({[fast_tok.decode([i]) for i in cu_top5]})")

    assert cos_sim > 0.999, f"Cosine similarity failed: {cos_sim} < 0.999"
    assert up_top5[0] == cu_top5[0], f"Top prediction mismatch: {up_top5[0]} vs {cu_top5[0]}"

    print(">>> ACTUAL MASKED EXAMPLE PARITY PASSED! <<<\n")
    del cu_model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def test_end_to_end_generation(device: str, mask_id: int):
    print("=" * 60)
    print("3. VALIDATING END-TO-END GENERATION AGAINST UPSTREAM")
    print("=" * 60)
    diff_weights = get_diff_weights_path()
    diff_dir = diff_weights.parent
    fast_tok = Tokenizer.from_file(str(diff_dir / "tokenizer.json"))
    pad_id = fast_tok.token_to_id("<|endoftext|>")

    prompt_str = "def square(x):\n    return"
    p_ids = fast_tok.encode(prompt_str, add_special_tokens=False).ids
    p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
    p_lens = torch.tensor([len(p_ids)], dtype=torch.long, device=device)

    # 1. Upstream generation
    print(f"Running upstream generation on {device} (steps=64, block_size=64)...")
    up_model = AutoModelForMaskedLM.from_pretrained(
        str(diff_dir),
        dtype=torch.float32,
        trust_remote_code=True,
    ).to(device).eval()

    up_out = native_generate(
        lambda x: up_model(x).logits,
        p_tensor,
        p_lens,
        pad_id,
        mask_id,
        steps=64,
        max_new_tokens=64,
        block_size=64,
        temperature=0.0,
    )
    up_tokens = up_out[0, len(p_ids):].tolist()
    del up_model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # 2. Custom generation
    print(f"Running custom generation on {device} (steps=64, block_size=64)...")
    cu_model = load_qwen25_coder_model(diff_weights, device=device)
    cu_out = native_generate(
        lambda x: cu_model(x, is_causal=False)[0],
        p_tensor,
        p_lens,
        pad_id,
        mask_id,
        steps=64,
        max_new_tokens=64,
        block_size=64,
        temperature=0.0,
    )
    cu_tokens = cu_out[0, len(p_ids):].tolist()

    exact_match = (up_tokens == cu_tokens)
    print(f"  Exact Token Match: {exact_match}")
    if not exact_match:
        diffs = sum(1 for a, b in zip(up_tokens, cu_tokens) if a != b)
        print(f"  Mismatched tokens: {diffs} / {len(up_tokens)}")

    assert exact_match, "End-to-end generated tokens did not match between upstream and custom!"
    print(">>> END-TO-END GENERATION PARITY PASSED! <<<\n")
    del cu_model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running validation on {device}...")
    mask_id = test_mask_token_round_trip()
    test_actual_masked_example(device, mask_id)
    test_end_to_end_generation(device, mask_id)
    print("=" * 60)
    print("ALL THREE UPSTREAM PARITY VALIDATIONS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
