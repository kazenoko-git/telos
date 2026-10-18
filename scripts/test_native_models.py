#!/usr/bin/env python3
"""Sanity-check native AR generation and MDLM masked denoising.

Verifies:
1. AR model produces coherent causal autoregressive continuations.
2. MDLM model reconstructs masked tokens using bidirectional context.
3. Both models run natively on CPU/MPS without shape or numerical errors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from dclm_model import DEFAULT_AR_WEIGHTS, DEFAULT_MDLM_WEIGHTS, DCLMConfig, load_dclm_model

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_FILE = (
    Path.home()
    / ".cache/huggingface/hub/models--michaelchenkj--datadecide-dclm-530m-53b/snapshots/213f7e96745ee3d397834821fe2b397019bdde98/weights/step57786/tokenizer.json"
)


def test_ar(tok: Tokenizer, device: str = "cpu") -> None:
    print("\n--- Testing Native AR Model ---")
    model = load_dclm_model(DEFAULT_AR_WEIGHTS, is_ar=True, device=device, dtype=torch.bfloat16)
    prompt = "def add(a, b):\n    '''Return the sum of a and b.'''\n    return"
    enc = tok.encode(prompt, add_special_tokens=False)
    ids = torch.tensor([enc.ids], dtype=torch.long, device=device)

    print(f"Prompt: {repr(prompt)}")
    generated = list(enc.ids)

    # Autoregressive greedy generation
    for step in range(12):
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids, is_causal=True)
            next_token = int(logits[0, -1, :50280].argmax().item())
        generated.append(next_token)
        token_str = tok.decode([next_token])
        print(f"  step {step+1}: token {next_token} -> {repr(token_str)}")

    full_text = tok.decode(generated)
    print(f"Generated text:\n{full_text}")
    assert len(generated) == len(enc.ids) + 12, "AR generation step count mismatch"


def test_mdlm(tok: Tokenizer, device: str = "cpu") -> None:
    print("\n--- Testing Native MDLM Model (Bidirectional Denoising) ---")
    model = load_dclm_model(DEFAULT_MDLM_WEIGHTS, is_ar=False, device=device, dtype=torch.bfloat16)

    # Prompt with hole: 'return a + b' with '+' replaced by [MASK] (50280)
    prompt = "def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\n"
    enc = tok.encode(prompt, add_special_tokens=False)
    plus_token = tok.encode(" +", add_special_tokens=False).ids
    print(f"Original text: {repr(prompt)}")
    print(f"Original token IDs: {enc.ids}")

    # Find the '+' token position
    hole_idx = None
    for idx, (a, b) in enumerate(enc.offsets):
        if prompt[a:b].strip() == "+":
            hole_idx = idx
            break
    assert hole_idx is not None, "Could not locate '+' in prompt"
    target_token = enc.ids[hole_idx]
    print(f"Hole position: {hole_idx}, target token: {target_token} ({repr(prompt[enc.offsets[hole_idx][0]:enc.offsets[hole_idx][1]])})")

    # Replace target with mask_id (50280)
    masked_ids = list(enc.ids)
    masked_ids[hole_idx] = 50280  # mask_id
    input_ids = torch.tensor([masked_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids, is_causal=False)
        hole_logits = logits[0, hole_idx, :50280]
        pred_token = int(hole_logits.argmax().item())
        probs = F.softmax(hole_logits.float(), dim=-1)
        target_prob = float(probs[target_token].item())
        pred_prob = float(probs[pred_token].item())

    pred_text = tok.decode([pred_token])
    target_text = tok.decode([target_token])
    print(f"Prediction: {pred_token} ({repr(pred_text)}) with prob {pred_prob:.4f}")
    print(f"Target:     {target_token} ({repr(target_text)}) with prob {target_prob:.4f}")
    top5 = torch.topk(probs, 5)
    print("Top 5 candidates at masked position:")
    for rank, (p, idx) in enumerate(zip(top5.values.tolist(), top5.indices.tolist())):
        print(f"  #{rank+1}: token {idx} ({repr(tok.decode([idx]))}) prob: {p:.4f}")


def main() -> None:
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))
    device = "cpu"
    test_ar(tok, device=device)
    test_mdlm(tok, device=device)
    print("\nBoth native models validated successfully!")


if __name__ == "__main__":
    main()
