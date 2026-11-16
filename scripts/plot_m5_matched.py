"""Plot completed-step timing drift and loss for the exploratory M5 pairs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    colors = {"telos": "#3455db", "mlx_lm": "#ba542c"}
    for i, name in enumerate(["00_telos", "01_mlx_lm", "02_mlx_lm", "03_telos"]):
        rows = [json.loads(line) for line in (args.directory / name / "steps.jsonl").read_text().splitlines()]
        implementation = "telos" if "telos" in name else "mlx_lm"
        label = ("Télos" if implementation == "telos" else "MLX-LM Llama") + f" · trial {i + 1}"
        style = "-" if i < 2 else "--"
        x = [row["update"] for row in rows]
        axes[0].plot(x, [r["seconds"] for r in rows], style, color=colors[implementation], label=label, lw=1.5)
        axes[1].plot(x, [r["loss"] for r in rows], style, color=colors[implementation], lw=1.5)
    axes[0].axvspan(0, 12, color="#dddddd", alpha=.35)
    axes[0].set(title="Timing varied during and between trials", ylabel="Completed update (seconds)", xlabel="Optimizer update")
    axes[0].legend(fontsize=8)
    axes[1].set(title="Short training loss trajectories", ylabel="Mean next-token cross entropy", xlabel="Optimizer update")
    for ax in axes:
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("M5 Pro · matched backbones in a common harness", fontsize=13)
    fig.savefig(args.directory / "timing_and_loss.svg")
    fig.savefig(args.directory / "timing_and_loss.png", dpi=160)


if __name__ == "__main__":
    main()
