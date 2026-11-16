"""Run fresh-process matched M5 backbone trials, committing each evidence bundle."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit-results", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    for key in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER", "MLX_METAL_FAST_SYNCH"):
        env.pop(key, None)
    for index, implementation in enumerate(["telos", "mlx_lm", "mlx_lm", "telos"]):
        trial = args.output / f"{index:02d}_{implementation}"
        command = [sys.executable, str(root / "scripts/benchmark_m5_matched.py"),
                   "--implementation", implementation, "--data", args.data,
                   "--updates", "64", "--warmup", "12", "--output", str(trial)]
        print(f"Starting {trial}", flush=True)
        with (args.output / f"{index:02d}_{implementation}.log").open("w") as stream:
            result = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=300)
        if args.commit_results:
            subprocess.run(["git", "add", "--", str(trial),
                            str(args.output / f"{index:02d}_{implementation}.log")], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m",
                            f"docs: record M5 {implementation} trial {index + 1}"], cwd=root, check=True)
        if result.returncode:
            raise SystemExit(f"Trial failed: {trial}; remaining trials cancelled")
        print(json.loads((trial / "result.json").read_text()), flush=True)


if __name__ == "__main__":
    main()
