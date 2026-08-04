"""Unified Master Benchmarking Suite for télos MDLM.

Modes:
  --mode throughput : Measures generation & training throughput (tokens/sec).
  --mode samplers   : Compares Cosine vs Non-Monotonic vs Windowed samplers on prompt suite.
  --mode schedules  : Evaluates timestep schedule trade-offs across steps (16, 32, 64, 128).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
import torch

from telos.hub.inference import TelosModel
from telos.diffusion.sampler import MDLMSampler, NonMonotonicMDLMSampler, WindowedMDLMSampler


PROMPT_SUITE = [
    "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "def bubble_sort(arr: list) -> list:\n    \"\"\"Sort an array in ascending order.\"\"\"\n",
    "def read_json_file(file_path: str) -> dict:\n    \"\"\"Read and parse a JSON file.\"\"\"\n",
    "class Node:\n    def __init__(self, val=0, next=None):\n",
    "import math\n\ndef calculate_std_dev(data: list) -> float:\n",
]


def run_throughput_benchmark(model_obj: TelosModel):
    """Measures raw sampling throughput (tok/sec) on target device."""
    print("\n" + "=" * 80)
    print("RUNNING THROUGHPUT BENCHMARK")
    print("=" * 80)

    prompt = PROMPT_SUITE[0]
    steps_list = [16, 32, 64, 128]
    target_len = 64

    for steps in steps_list:
        # Warmup
        model_obj.complete(prompt, max_tokens=16, num_steps=16)

        t0 = time.time()
        res = model_obj.complete(prompt, max_tokens=target_len, num_steps=steps)
        elapsed = time.time() - t0

        tok_per_sec = target_len / elapsed
        print(f"Steps: {steps:3d} | Time: {elapsed:.2f}s | Throughput: {tok_per_sec:.1f} tok/s")


def run_sampler_comparison(model_obj: TelosModel):
    """Compares Cosine, Non-Monotonic, and Windowed samplers on prompt suite."""
    print("\n" + "=" * 80)
    print("RUNNING SAMPLER COMPARISON (Cosine vs Non-Monotonic vs Windowed)")
    print("=" * 80)

    model = model_obj.model
    tokenizer = model_obj.tokenizer
    mask_token_id = tokenizer.token_to_id("[MASK]") or 4

    cosine_sampler = MDLMSampler(model, mask_token_id, num_steps=64, schedule="cosine")
    non_mono_sampler = NonMonotonicMDLMSampler(model, mask_token_id, num_steps=64, remask_threshold=0.15)
    windowed_sampler = WindowedMDLMSampler(model, mask_token_id, window_size=32, num_steps_per_window=16)

    for i, prompt in enumerate(PROMPT_SUITE, 1):
        print(f"\n--- Prompt {i}: {prompt.splitlines()[0]} ---")
        prompt_enc = tokenizer.encode(prompt)
        prompt_ids = torch.tensor([prompt_enc.ids], device=model_obj.device)

        # 1. Cosine Sampler
        t0 = time.time()
        seq_cos = cosine_sampler.sample(seq_len=64 + len(prompt_enc.ids), prompt_ids=prompt_ids, device=model_obj.device)
        t_cos = time.time() - t0
        text_cos = tokenizer.decode(seq_cos[0].tolist())

        # 2. Non-Monotonic Sampler
        t0 = time.time()
        seq_nm = non_mono_sampler.sample(seq_len=64 + len(prompt_enc.ids), prompt_ids=prompt_ids, device=model_obj.device)
        t_nm = time.time() - t0
        text_nm = tokenizer.decode(seq_nm[0].tolist())

        # 3. Windowed Sampler
        t0 = time.time()
        seq_win = windowed_sampler.sample(target_tokens=64, prompt_ids=prompt_ids, device=model_obj.device)
        t_win = time.time() - t0
        text_win = tokenizer.decode(seq_win[0].tolist())

        print(f"[Cosine Sampler - {t_cos:.2f}s]:")
        print(text_cos[:150].replace("\n", " "))
        print(f"[Non-Monotonic - {t_nm:.2f}s]:")
        print(text_nm[:150].replace("\n", " "))
        print(f"[Windowed Sampler - {t_win:.2f}s]:")
        print(text_win[:150].replace("\n", " "))


def main():
    parser = argparse.ArgumentParser(description="Master Benchmarking CLI for télos MDLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase_c_tpu_125m/checkpoint_tpu_125M_final_step_238.pt", help="Path to checkpoint")
    parser.add_argument("--mode", type=str, choices=["throughput", "samplers", "all"], default="all", help="Benchmark mode")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}...")
    model_obj = TelosModel.from_pretrained(args.checkpoint)

    if args.mode in ["throughput", "all"]:
        run_throughput_benchmark(model_obj)

    if args.mode in ["samplers", "all"]:
        run_sampler_comparison(model_obj)


if __name__ == "__main__":
    main()
