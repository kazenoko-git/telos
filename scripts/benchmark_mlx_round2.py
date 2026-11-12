"""Bounded, reversed-order scheduler and microbatch trials for 50M AR training.

Uses the existing real-corpus profiler in fresh processes. No training defaults
are changed. MLX scheduling overrides must be set before importing MLX.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


# Supported by MLX 0.32.2 mlx/utils.h and backend/metal/device.cpp. These
# control scheduling, not tensor precision or model/checkpoint geometry.
# https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/utils.h
VARIANTS = {
    "reference": (8, {}),
    "buffers16": (8, {"MLX_MAX_OPS_PER_BUFFER": "16", "MLX_MAX_MB_PER_BUFFER": "16"}),
    "buffers128": (8, {"MLX_MAX_OPS_PER_BUFFER": "128", "MLX_MAX_MB_PER_BUFFER": "128"}),
    "buffers1000": (8, {"MLX_MAX_OPS_PER_BUFFER": "1000", "MLX_MAX_MB_PER_BUFFER": "400"}),
    "microbatch16": (16, {}),
    "fastsync": (8, {"MLX_METAL_FAST_SYNCH": "1"}),
    "defaults": (8, {}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=["reference", "buffers128", "buffers1000", "microbatch16"])
    args = parser.parse_args()
    if not 0 <= args.warmup < args.updates:
        parser.error("warmup must be nonnegative and smaller than updates")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    sequence = args.variants + list(reversed(args.variants))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER", "MLX_METAL_FAST_SYNCH"):
        env.pop(key, None)
    (output / "power.txt").write_text(subprocess.check_output(["pmset", "-g", "custom"], text=True))
    for index, variant in enumerate(sequence):
        microbatch, overrides = VARIANTS[variant]
        trial = output / f"{index:02d}_{variant}"
        command = [sys.executable, "scripts/profile_mlx_training.py",
                   "--output", str(trial), "--tokens", str(args.updates * 16384),
                   "--microbatch", str(microbatch), "--grad-accum", str(64 // microbatch),
                   "--max-mlx-gib", "3.2",
                   "--critical-memory-free-percent", "30", "--critical-swap-free-mib", "768"]
        if variant != "defaults":
            command += ["--compiled-accumulation", "--force-fused-attention",
                        "--experimental-packed-swiglu"]
        print(f"Starting {index + 1}/{len(sequence)}: {variant}", flush=True)
        with (output / f"{index:02d}_{variant}.log").open("w") as log:
            process = subprocess.run(command, cwd=root, env={**env, **overrides},
                                     stdout=log, stderr=subprocess.STDOUT, timeout=180)
        summary_path = trial / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        rows = [json.loads(line) for line in (trial / "steps.jsonl").read_text().splitlines()] if (trial / "steps.jsonl").exists() else []
        warmed = rows[args.warmup:]
        result = {"variant": variant, "environment": overrides, "microbatch": microbatch,
                  "grad_accum": 64 // microbatch, "returncode": process.returncode,
                  "summary": summary, "warmup_updates": args.warmup,
                  "warmed_tokens_per_second": len(warmed) * 16384 / sum(r["training_step_seconds"] for r in warmed) if warmed else None,
                  "peak_mlx_gib": max((r["mlx_peak_bytes"] for r in rows), default=0) / 1024**3}
        results.append(result)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if process.returncode or summary.get("status") != "complete":
            raise SystemExit("Trial failed or memory guard stopped; remaining trials cancelled")


if __name__ == "__main__":
    main()
