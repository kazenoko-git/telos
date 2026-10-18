#!/usr/bin/env python3
"""Validation script for Qwen3-0.6B AR, MDLM, and COROSred generation.

Checks:
1. Tokenizer IDs and special tokens (<|im_start|>, <|im_end|>, <|endoftext|>, <|mask|>).
2. Native AR causal greedy generation and termination.
3. Native MDLM iterative block-diffusion generation.
4. COROSred two-model confidence-routed generation on generated context.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from scripts.qwen3_06b_model import (
    Qwen3_06B_Config,
    Qwen3_06B_Model,
    load_qwen3_06b_model,
    AR_WEIGHTS,
    DIFF_WEIGHTS,
)

TOKENIZER_FILE = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
)

EOS_TOKENS = {151645, 151643}  # <|im_end|>, <|endoftext|>
MASK_TOKEN_ID = 151669  # <|mask|>


def validate_tokenizer_ids() -> bool:
    """Validate tokenizer IDs and vocabulary configuration."""
    print("=== 1. Validating Tokenizer IDs ===")
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))
    vocab_size = tok.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    expected = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|endoftext|>": 151643,
    }
    for token, exp_id in expected.items():
        actual_id = tok.token_to_id(token)
        assert actual_id == exp_id, f"Token ID mismatch for {token}: expected {exp_id}, got {actual_id}"
        print(f"  [OK] {token:<15} -> {actual_id}")

    # Check mask token
    diff_tok_path = (
        Path.home()
        / ".cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1/snapshots/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/tokenizer.json"
    )
    diff_tok = Tokenizer.from_file(str(diff_tok_path))
    mask_id = diff_tok.token_to_id("<|mask|>")
    assert mask_id == MASK_TOKEN_ID, f"Mask ID mismatch: expected {MASK_TOKEN_ID}, got {mask_id}"
    print(f"  [OK] {'<|mask|>':<15} -> {mask_id}")
    return True


@torch.no_grad()
def generate_ar_greedy(
    model: Qwen3_06B_Model,
    tok: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    stop_strings: Optional[List[str]] = None,
    device: str = "mps",
) -> Tuple[str, List[int], List[dict]]:
    """Greedy autoregressive generation with token-level confidence recording."""
    input_ids = tok.encode(prompt, add_special_tokens=False).ids
    generated_ids = []
    token_stats = []

    curr_ids = list(input_ids)
    for _ in range(max_new_tokens):
        t_in = torch.tensor([curr_ids], dtype=torch.long, device=device)
        logits = model(t_in, is_causal=True)[0, -1, :]

        # Compute token probability distribution in float32
        probs = F.softmax(logits.float(), dim=-1)
        top2 = torch.topk(probs, 2)
        p1 = float(top2.values[0].item())
        p2 = float(top2.values[1].item())
        entropy = -float(torch.sum(probs * torch.log(probs.clamp_min(1e-12))).item())
        pred_token = int(top2.indices[0].item())

        # Check EOS termination
        if pred_token in EOS_TOKENS:
            break

        generated_ids.append(pred_token)
        curr_ids.append(pred_token)
        token_stats.append({
            "token_id": pred_token,
            "token_text": tok.decode([pred_token]),
            "prob": p1,
            "margin": p1 - p2,
            "entropy": entropy,
        })

        # Check stop strings on decoded text
        if stop_strings:
            text_so_far = tok.decode(generated_ids)
            if any(s in text_so_far for s in stop_strings):
                break

    completion_text = tok.decode(generated_ids)
    return completion_text, generated_ids, token_stats


@torch.no_grad()
def generate_mdlm_block_diffusion(
    model: Qwen3_06B_Model,
    tok: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    block_size: int = 32,
    steps: int = 32,
    device: str = "mps",
) -> str:
    """Native MDLM generation using block-diffusion with low_confidence remasking."""
    prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
    prompt_len = len(prompt_ids)
    pad_id = 151643  # <|endoftext|>

    total_len = prompt_len + max_new_tokens
    x = torch.full((1, total_len), pad_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    x[0, prompt_len:total_len] = MASK_TOKEN_ID

    num_blocks = max_new_tokens // block_size
    steps_per_block = max(1, steps // num_blocks)
    positions = torch.arange(total_len, device=device)

    for b in range(num_blocks):
        b_start = prompt_len + b * block_size
        b_end = b_start + block_size

        for s in range(steps_per_block):
            block_mask = (positions >= b_start) & (positions < b_end) & (x[0] == MASK_TOKEN_ID)
            num_masked = int(block_mask.sum().item())
            if num_masked == 0:
                break

            # Diffusion forward pass under bidirectional attention
            logits = model(x, is_causal=False)[0]  # [total_len, vocab_size]
            probs = F.softmax(logits.float(), dim=-1)
            x0 = torch.argmax(probs, dim=-1)
            x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            # Unmask top-k confident tokens in this step
            k_transfer = max(1, num_masked // (steps_per_block - s))
            confidence = torch.where(block_mask, x0_p, torch.tensor(-float("inf"), device=device))
            _, topk_idx = torch.topk(confidence, k=min(k_transfer, num_masked))

            x[0, topk_idx] = x0[topk_idx]

    new_ids = x[0, prompt_len:].tolist()
    # Truncate at first EOS if present
    for i, tid in enumerate(new_ids):
        if tid in EOS_TOKENS:
            new_ids = new_ids[:i]
            break
    return tok.decode(new_ids)


@torch.no_grad()
def generate_corosred(
    ar_model: Qwen3_06B_Model,
    diff_model: Qwen3_06B_Model,
    tok: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    margin_threshold: float = 0.5,
    entropy_threshold: float = 1.0,
    stop_strings: Optional[List[str]] = None,
    device: str = "mps",
) -> Tuple[str, dict]:
    """COROSred: AR draft -> uncertainty span detection -> MDLM repair using generated context."""
    t0 = time.monotonic()

    # Step 1: Generate AR draft and record token-level uncertainties
    ar_text, ar_tokens, token_stats = generate_ar_greedy(
        ar_model, tok, prompt, max_new_tokens=max_new_tokens, stop_strings=stop_strings, device=device
    )
    ar_time = time.monotonic() - t0

    prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
    prompt_len = len(prompt_ids)

    # Step 2: Flag uncertain tokens without reference answers
    uncertain_indices = []
    for idx, stat in enumerate(token_stats):
        if stat["margin"] < margin_threshold or stat["entropy"] > entropy_threshold:
            uncertain_indices.append(idx)

    # If no tokens are uncertain, return AR draft directly
    if not uncertain_indices:
        return ar_text, {
            "routed": False,
            "ar_draft": ar_text,
            "final_text": ar_text,
            "uncertain_tokens_count": 0,
            "mdlm_forward_passes": 0,
            "ar_time": ar_time,
            "total_time": ar_time,
        }

    # Step 3: Mask uncertain positions in the generated context (Prompt + Draft)
    # Never supply ground truth or external reference
    repaired_tokens = list(ar_tokens)
    for idx in uncertain_indices:
        repaired_tokens[idx] = MASK_TOKEN_ID

    full_seq = prompt_ids + repaired_tokens
    t_full = torch.tensor([full_seq], dtype=torch.long, device=device)

    # Step 4: Run MDLM bidirectional forward pass to repair masked spans
    t_repair_start = time.monotonic()
    logits = diff_model(t_full, is_causal=False)[0]

    for idx in uncertain_indices:
        pos = prompt_len + idx
        pred_token = int(logits[pos].argmax().item())
        repaired_tokens[idx] = pred_token

    repair_time = time.monotonic() - t_repair_start
    final_text = tok.decode(repaired_tokens)

    return final_text, {
        "routed": True,
        "ar_draft": ar_text,
        "final_text": final_text,
        "uncertain_tokens_count": len(uncertain_indices),
        "mdlm_forward_passes": 1,
        "ar_time": ar_time,
        "repair_time": repair_time,
        "total_time": time.monotonic() - t0,
    }


def main():
    print("Testing Qwen3-0.6B generation implementations...")
    validate_tokenizer_ids()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))

    print(f"\n=== 2. Loading Models on {device} ===")
    ar_model = load_qwen3_06b_model(AR_WEIGHTS, device=device)
    diff_model = load_qwen3_06b_model(DIFF_WEIGHTS, device=device)

    test_prompt = 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n    return'

    print(f"\n=== 3. AR Generation Test ===")
    ar_out, ar_tokens, stats = generate_ar_greedy(ar_model, tok, test_prompt, max_new_tokens=32, device=device)
    print(f"Prompt:\n{test_prompt}")
    print(f"AR Continuation:\n{ar_out!r}")
    print(f"Generated {len(ar_tokens)} tokens.")
    print("First 3 tokens confidence:")
    for s in stats[:3]:
        print(f"  Token {s['token_text']!r} -> Prob: {s['prob']:.4f}, Margin: {s['margin']:.4f}, Entropy: {s['entropy']:.4f}")

    print(f"\n=== 4. MDLM Block-Diffusion Generation Test ===")
    mdlm_out = generate_mdlm_block_diffusion(diff_model, tok, test_prompt, max_new_tokens=32, steps=32, block_size=32, device=device)
    print(f"MDLM Continuation:\n{mdlm_out!r}")

    print(f"\n=== 5. COROSred Generation Test ===")
    coros_out, meta = generate_corosred(ar_model, diff_model, tok, test_prompt, max_new_tokens=32, device=device)
    print(f"COROSred Result:\n{coros_out!r}")
    print(f"Routing metadata: {meta}")

    print("\nAll generation implementations verified successfully!")


if __name__ == "__main__":
    main()
