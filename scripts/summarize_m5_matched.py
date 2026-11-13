"""Validate and summarize a completed A/B/B/A matched-backbone experiment."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    trials = []
    signatures = set()
    for name in ["00_telos", "01_mlx_lm", "02_mlx_lm", "03_telos"]:
        path = args.directory / name
        result = json.loads((path / "result.json").read_text())
        meta = json.loads((path / "metadata.json").read_text())
        steps = [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]
        check = json.loads((path / "parity.json").read_text())
        assert result["status"] == "complete" and check["passed"]
        assert len(steps) == meta["updates"] == result["updates"] == 64
        signatures.add(json.dumps({k: meta[k] for k in
                                  ["parameters", "architecture", "microbatch", "accumulation",
                                   "data_prefix_sha256", "packages", "script_sha256"]}, sort_keys=True))
        warmed = steps[meta["warmup"]:]
        trials.append({"trial": name, "implementation": result["implementation"],
                       "input_tokens_per_second": result["warmed_input_tokens_per_second"],
                       "target_tokens_per_second": result["warmed_target_tokens_per_second"],
                       "warmed_seconds": sum(s["seconds"] for s in warmed),
                       "median_step_seconds": statistics.median(s["seconds"] for s in warmed),
                       "peak_mlx_gib": result["training_peak_mlx_bytes"] / 1024**3,
                       "initial_heldout_loss": result["initial_heldout_loss"],
                       "final_heldout_loss": result["final_heldout_loss"]})
    assert len(signatures) == 1, "Trial configurations differ"
    assert len({t["initial_heldout_loss"] for t in trials}) == 1, "Initial BF16 losses differ"
    grouped = {}
    for implementation in ["telos", "mlx_lm"]:
        rows = [t for t in trials if t["implementation"] == implementation]
        grouped[implementation] = {
            "pooled_input_tokens_per_second": 2 * 52 * 16384 / sum(t["warmed_seconds"] for t in rows),
            "mean_peak_mlx_gib": statistics.mean(t["peak_mlx_gib"] for t in rows),
            "mean_final_heldout_loss": statistics.mean(t["final_heldout_loss"] for t in rows)}
    summary = {"trials": trials, "pooled": grouped,
               "telos_throughput_change_pct": 100 * (grouped["telos"]["pooled_input_tokens_per_second"] /
                                                      grouped["mlx_lm"]["pooled_input_tokens_per_second"] - 1),
               "limitations": ["Common backbone harness; not native trainer ranking",
                               "Short run; not time to quality or convergence evidence",
                               "Two repetitions per implementation; no statistical superiority claim"]}
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
