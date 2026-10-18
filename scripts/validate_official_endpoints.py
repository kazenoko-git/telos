#!/usr/bin/env python3
"""Validate custom Qwen3-0.6B AR and MDLM implementations against official Hugging Face endpoints.

Compares:
1. Official Qwen/Qwen3-0.6B (AutoModelForCausalLM) vs. custom Qwen3_06B_Model (causal)
2. Official dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 (AutoModelForMaskedLM) vs. custom Qwen3_06B_Model (bidirectional)
Checks:
- Logits numerical parity (max absolute diff, mean absolute diff, cosine similarity)
- Top-k prediction alignment
- Generation token-by-token parity
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qwen3_06b_model import (
    load_qwen3_06b_model,
    AR_WEIGHTS,
    DIFF_WEIGHTS,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TEST_PROMPT = 'def is_even(n: int) -> bool:\n    """Return True if n is even."""\n'


def validate_ar_endpoint():
    print("=" * 80)
    print("1. VALIDATING AUTOREGRESSIVE (AR) ENDPOINT AGAINST OFFICIAL HUGGING FACE")
    print("=" * 80)
    
    # Load official model
    print("Loading official AutoModelForCausalLM Qwen/Qwen3-0.6B...")
    official_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    official_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    
    # Load custom model
    print("Loading custom Qwen3_06B_Model (AR weights)...")
    custom_model = load_qwen3_06b_model(AR_WEIGHTS, device=DEVICE)
    
    # Tokenize test prompt
    input_ids = official_tok.encode(TEST_PROMPT, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        official_out = official_model(input_ids)
        official_logits = official_out.logits.float()
        custom_logits = custom_model(input_ids, is_causal=True).float()
        
    diff = (official_logits - custom_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = F.cosine_similarity(official_logits.flatten(), custom_logits.flatten(), dim=0).item()
    
    # Compare top-5 predictions at last position
    off_top5 = torch.topk(official_logits[0, -1], 5).indices.tolist()
    cust_top5 = torch.topk(custom_logits[0, -1], 5).indices.tolist()
    
    print(f"Max Absolute Logit Diff:  {max_diff:.6e}")
    print(f"Mean Absolute Logit Diff: {mean_diff:.6e}")
    print(f"Cosine Similarity:        {cos_sim:.8f}")
    print(f"Official Top-5 Tokens:    {off_top5} ({[official_tok.decode([t]) for t in off_top5]})")
    print(f"Custom Top-5 Tokens:      {cust_top5} ({[official_tok.decode([t]) for t in cust_top5]})")
    
    # Token-by-token greedy generation parity check (30 tokens)
    print("\nGreedy Generation Parity (30 tokens):")
    # Official generation
    off_gen = official_model.generate(
        input_ids,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=official_tok.pad_token_id or official_tok.eos_token_id,
    )[0, input_ids.shape[1]:].tolist()
    
    # Custom generation with KV cache
    cust_gen = []
    logits, past_kvs = custom_model(input_ids, is_causal=True, use_cache=True)
    next_token = int(logits[0, -1].argmax().item())
    prompt_len = input_ids.shape[1]
    for step in range(30):
        cust_gen.append(next_token)
        t_curr = torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)
        logits, past_kvs = custom_model(
            t_curr,
            is_causal=True,
            past_key_values=past_kvs,
            start_pos=prompt_len + step,
            use_cache=True,
        )
        next_token = int(logits[0, -1].argmax().item())
        
    match = (off_gen == cust_gen)
    print(f"Token Match Parity: {'PASSED (100% Identical)' if match else 'FAILED'}")
    if not match:
        print(f"Official: {off_gen}")
        print(f"Custom:   {cust_gen}")
    print(f"Decoded Generation: {official_tok.decode(cust_gen)!r}")
    
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cos_sim": cos_sim,
        "tokens_match": match,
    }


def validate_mdlm_endpoint():
    print("\n" + "=" * 80)
    print("2. VALIDATING MASKED DIFFUSION (MDLM) ENDPOINT AGAINST OFFICIAL HUGGING FACE")
    print("=" * 80)
    
    # Load official model class directly from local snapshot
    import importlib.util
    snap_dir = Path.home() / ".cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1/snapshots/c8d24a3f4adaeef46881b450e1bf7d1005203bd7"
    spec = importlib.util.spec_from_file_location("modeling_qwen3", str(snap_dir / "modeling_qwen3.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    official_tok = AutoTokenizer.from_pretrained(str(snap_dir))
    official_model = mod.A2DQwen3LMHeadModel.from_pretrained(str(snap_dir), dtype=torch.bfloat16).to(DEVICE).eval()
    
    # Load custom model
    print("Loading custom Qwen3_06B_Model (MDLM weights)...")
    custom_model = load_qwen3_06b_model(DIFF_WEIGHTS, device=DEVICE)
    
    mask_id = official_tok.mask_token_id  # 151669
    prompt_ids = official_tok.encode("def add(a, b):\n    return a + b\n")
    test_seq = prompt_ids + [mask_id, mask_id, mask_id]
    input_ids = torch.tensor([test_seq], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        official_out = official_model(input_ids)
        official_logits = official_out.logits.float()
        
        custom_logits = custom_model(input_ids, is_causal=False).float()
        
    diff = (official_logits - custom_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = F.cosine_similarity(official_logits.flatten(), custom_logits.flatten(), dim=0).item()
    
    off_argmax = official_logits[0, -3:].argmax(dim=-1).tolist()
    cust_argmax = custom_logits[0, -3:].argmax(dim=-1).tolist()
    
    print(f"Max Absolute Logit Diff:  {max_diff:.6e}")
    print(f"Mean Absolute Logit Diff: {mean_diff:.6e}")
    print(f"Cosine Similarity:        {cos_sim:.8f}")
    print(f"Official Argmax on Masks: {off_argmax} ({[official_tok.decode([t]) for t in off_argmax]})")
    print(f"Custom Argmax on Masks:   {cust_argmax} ({[official_tok.decode([t]) for t in cust_argmax]})")
    
    match = (off_argmax == cust_argmax)
    print(f"Argmax Match Parity: {'PASSED (100% Identical)' if match else 'FAILED'}")
    
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cos_sim": cos_sim,
        "argmax_match": match,
    }


def main():
    ar_res = validate_ar_endpoint()
    mdlm_res = validate_mdlm_endpoint()
    print("\n" + "=" * 80)
    print("SUMMARY OF PARITY VALIDATION:")
    print(f"AR Model:   Cosine Sim = {ar_res['cos_sim']:.8f}, Max Diff = {ar_res['max_diff']:.2e}, Greedy Parity = {ar_res['tokens_match']}")
    print(f"MDLM Model: Cosine Sim = {mdlm_res['cos_sim']:.8f}, Max Diff = {mdlm_res['max_diff']:.2e}, Mask Parity   = {mdlm_res['argmax_match']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
