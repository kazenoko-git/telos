#!/usr/bin/env python3
"""Validate custom Qwen2.5-Coder model against official Hugging Face endpoints.

Tests:
1. AR parity: Qwen/Qwen2.5-Coder-0.5B-Instruct vs Custom Causal Implementation.
2. MDLM parity: dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1 vs Custom Bidirectional Implementation.
Checks cosine similarity (> 0.999), mean logit diff, and greedy token agreement.
"""
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import sys
import types
sys.modules["dllm"] = types.ModuleType("dllm")

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

from scripts.qwen25_coder_model import (
    load_qwen25_coder_model,
    get_ar_weights_path,
    get_diff_weights_path,
)

TEST_PROMPTS = [
    "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is prime.\"\"\"\n",
    "def quicksort(arr: list[int]) -> list[int]:\n",
    "def count_vowels(s: str) -> int:\n",
]


def test_ar_endpoint_parity(device: str = "cpu"):
    print("\n" + "=" * 60)
    print("1. VALIDATING AR ENDPOINT PARITY (Qwen2.5-Coder-0.5B-Instruct)")
    print("=" * 60)

    repo_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    print(f"Loading official HF model: {repo_id}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    ar_weights = get_ar_weights_path()
    print(f"Loading custom implementation from {ar_weights.name}...")
    custom_model = load_qwen25_coder_model(ar_weights, device=device)

    tok_path = ar_weights.parent / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))

    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        token_ids = tok.encode(prompt, add_special_tokens=False).ids
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            hf_out = hf_model(input_tensor).logits[0, -1, :].float()
            custom_out, _ = custom_model(input_tensor, is_causal=True)
            custom_logits = custom_out[0, -1, :].float()

        cos_sim = float(F.cosine_similarity(hf_out.unsqueeze(0), custom_logits.unsqueeze(0)).item())
        mean_diff = float((hf_out - custom_logits).abs().mean().item())
        max_diff = float((hf_out - custom_logits).abs().max().item())

        hf_top5 = torch.topk(hf_out, 5).indices.tolist()
        custom_top5 = torch.topk(custom_logits, 5).indices.tolist()
        top5_match = (hf_top5 == custom_top5)

        print(f"Prompt {idx}: {repr(prompt.splitlines()[0])}")
        print(f"  Cosine Similarity: {cos_sim:.8f} (Threshold: > 0.999)")
        print(f"  Mean Logit Diff:   {mean_diff:.4e}")
        print(f"  Max Logit Diff:    {max_diff:.4e}")
        print(f"  Top-5 Token Match: {top5_match} (Tokens: {custom_top5})")

        assert cos_sim > 0.999, f"AR Cosine similarity failed: {cos_sim}"
        assert top5_match, f"Top-5 token mismatch: {hf_top5} vs {custom_top5}"

    print(">>> AR Official Endpoint Parity: PASSED <<<\n")
    del hf_model, custom_model
    gc.collect()


def test_diff_endpoint_parity(device: str = "cpu"):
    print("\n" + "=" * 60)
    print("2. VALIDATING MDLM ENDPOINT PARITY (Qwen2.5-Coder-0.5B-Instruct-diffusion)")
    print("=" * 60)

    repo_id = "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"
    print(f"Loading official HF model: {repo_id}...")
    hf_model = AutoModelForMaskedLM.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    ).eval()

    diff_weights = get_diff_weights_path()
    print(f"Loading custom implementation from {diff_weights.name}...")
    custom_model = load_qwen25_coder_model(diff_weights, device=device)

    tok_path = diff_weights.parent / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    mask_token_id = 151669

    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        token_ids = tok.encode(prompt, add_special_tokens=False).ids
        # Mask the last token
        masked_ids = list(token_ids)
        masked_ids[-1] = mask_token_id
        input_tensor = torch.tensor([masked_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            hf_out = hf_model(input_tensor).logits[0, -1, :].float()
            custom_out, _ = custom_model(input_tensor, is_causal=False)
            custom_logits = custom_out[0, -1, :].float()

        cos_sim = float(F.cosine_similarity(hf_out.unsqueeze(0), custom_logits.unsqueeze(0)).item())
        mean_diff = float((hf_out - custom_logits).abs().mean().item())
        max_diff = float((hf_out - custom_logits).abs().max().item())

        hf_top5 = torch.topk(hf_out, 5).indices.tolist()
        custom_top5 = torch.topk(custom_logits, 5).indices.tolist()
        top5_match = (hf_top5 == custom_top5)

        print(f"Prompt {idx} (Masked pos {len(masked_ids)-1}): {repr(prompt.splitlines()[0])}")
        print(f"  Cosine Similarity: {cos_sim:.8f} (Threshold: > 0.999)")
        print(f"  Mean Logit Diff:   {mean_diff:.4e}")
        print(f"  Max Logit Diff:    {max_diff:.4e}")
        print(f"  Top-5 Token Match: {top5_match} (Tokens: {custom_top5})")

        assert cos_sim > 0.999, f"MDLM Cosine similarity failed: {cos_sim}"
        assert top5_match, f"Top-5 token mismatch: {hf_top5} vs {custom_top5}"

    print(">>> MDLM Official Endpoint Parity: PASSED <<<\n")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running validation on {device}...")
    test_ar_endpoint_parity(device=device)
    test_diff_endpoint_parity(device=device)
    print("=" * 60)
    print("ALL QWEN2.5-CODER ENDPOINT VALIDATIONS PASSED PERFECTLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
