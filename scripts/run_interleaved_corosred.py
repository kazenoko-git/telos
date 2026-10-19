#!/usr/bin/env python3
"""Standalone inference entry point for Interleaved COROSred decoder.

Usage:
    .venv/bin/python3 scripts/run_interleaved_corosred.py --prompt "def fibonacci(n: int) -> int:\n"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qwen25_coder_model import (
    get_ar_weights_path,
    get_diff_weights_path,
    load_qwen25_coder_model,
)
from telos.decoding.interleaved_corosred import (
    InterleavedConfig,
    ar_baseline_decode,
    get_mask_token_id,
    interleaved_corosred_decode,
)


def main():
    parser = argparse.ArgumentParser(description="Run Interleaved COROSred Inference")
    parser.add_argument("--prompt", type=str, default="def is_prime(n: int) -> bool:\n", help="Prompt string")
    parser.add_argument("--block-size", type=int, default=32, help="AR block drafting size")
    parser.add_argument("--overlap", type=int, default=8, help="Inspection window bounded overlap")
    parser.add_argument("--max-masks", type=int, default=3, help="Max remasked positions per block")
    parser.add_argument("--max-refine-passes", type=int, default=3, help="Max sequential refinement passes")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Maximum generated tokens")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: auto mps/cpu)")
    parser.add_argument("--baseline", action="store_true", help="Run AR baseline comparison")
    parser.add_argument("--json-out", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading models on {device}...")

    ar_weights = get_ar_weights_path()
    diff_weights = get_diff_weights_path()
    tokenizer = Tokenizer.from_file(str(diff_weights.parent / "tokenizer.json"))
    mask_id = get_mask_token_id(tokenizer)

    ar_model = load_qwen25_coder_model(ar_weights, device=device, dtype=torch.bfloat16)
    diff_model = load_qwen25_coder_model(diff_weights, device=device, dtype=torch.bfloat16)

    config = InterleavedConfig(
        block_size=args.block_size,
        overlap=args.overlap,
        max_masks_per_block=args.max_masks,
        max_refinement_passes=args.max_refine_passes,
        max_new_tokens=args.max_new_tokens,
        mask_id=mask_id,
    )

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids

    print("\nExecuting Interleaved COROSred decoding...")
    res = interleaved_corosred_decode(
        prompt_ids=prompt_ids,
        ar_model=ar_model,
        diff_model=diff_model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )

    print("\n" + "=" * 60)
    print("INTERLEAVED COROSRED OUTPUT:")
    print("=" * 60)
    print(res.cleaned_output)
    print("\n" + "-" * 60)
    print(f"Total Latency: {res.total_latency * 1000:.2f} ms")
    print(f"Timing Breakdown: {res.timing_breakdown}")
    print(f"Forward Passes: {res.forward_passes}")
    print(f"Blocks Traced: {len(res.block_traces)}")

    if args.baseline:
        print("\nExecuting AR baseline decoding...")
        base = ar_baseline_decode(
            prompt_ids=prompt_ids,
            ar_model=ar_model,
            tokenizer=tokenizer,
            config=config,
            device=device,
        )
        print("\n" + "=" * 60)
        print("AR BASELINE OUTPUT:")
        print("=" * 60)
        print(base.cleaned_output)
        print("\n" + "-" * 60)
        print(f"Baseline Latency: {base.total_latency * 1000:.2f} ms")

    if args.json_out:
        out_dict = {
            "prompt": args.prompt,
            "raw_output": res.raw_output,
            "cleaned_output": res.cleaned_output,
            "total_latency": res.total_latency,
            "timing_breakdown": res.timing_breakdown,
            "forward_passes": res.forward_passes,
            "block_traces": [
                {
                    "block_idx": t.block_idx,
                    "drafted_tokens": t.drafted_tokens,
                    "inspected_positions": t.inspected_positions,
                    "remasked_positions": t.remasked_positions,
                    "refinement_updates": t.refinement_updates,
                    "cache_rebuilt": t.cache_rebuilt,
                    "continued_draft_len": t.continued_draft_len,
                }
                for t in res.block_traces
            ],
        }
        with open(args.json_out, "w") as f:
            json.dump(out_dict, f, indent=2)
        print(f"\nSaved inference record to {args.json_out}")


if __name__ == "__main__":
    main()
