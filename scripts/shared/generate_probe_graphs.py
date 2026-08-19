"""Generates publication-quality figures for Telos scaling laws and probe benchmarks."""

import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Setup output directory
Path("figures").mkdir(exist_ok=True)

# Define clean styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

def parse_probe_file(file_path):
    """Extracts overall metrics and category-level metrics from probe txt files."""
    if not file_path.exists():
        return None
    with open(file_path, "r") as f:
        text = f.read()

    categories = {}
    if "Category-Specific Breakdown:" in text:
        cat_section = text.split("Category-Specific Breakdown:")[-1].split("====")[0]
        for line in cat_section.strip().split("\n"):
            m = re.search(r"-\s+([^:]+):\s+Target CE\s+=\s+([\d\.]+)\s+\|\s+Avg Rank\s+=\s+([\d\.]+)", line)
            if m:
                cat_name = m.group(1).strip()
                ce = float(m.group(2))
                rank = float(m.group(3))
                categories[cat_name] = {"ce": ce, "rank": rank}

    overall_ce, overall_rank, top5_acc = None, None, None
    m_ce = re.search(r"Overall Average Target CE\s*:\s*([\d\.]+)", text)
    if m_ce: overall_ce = float(m_ce.group(1))
    m_rank = re.search(r"Overall Average Rank\s*:\s*([\d\.]+)", text)
    if m_rank: overall_rank = float(m_rank.group(1))
    m_top5 = re.search(r"Top-5 Accuracy\s*:\s*([\d\.]+)%", text)
    if m_top5: top5_acc = float(m_top5.group(1))

    return {
        "overall_ce": overall_ce,
        "overall_rank": overall_rank,
        "top5_acc": top5_acc,
        "categories": categories
    }


def find_probe(filename: str) -> Path:
    p1 = Path("evals/masked/probes") / filename
    if p1.exists():
        return p1
    return Path("probes_output") / filename

# Complete Model definitions with all updated peak ratios
ROPE_MODELS = {
    "12.5M": [
        (1, find_probe("masked_telos_12m_r1_probes.txt")),
        (5, find_probe("masked_telos_12m_r5_probes.txt")),
        (10, find_probe("masked_telos_12m_r10_probes.txt")),
        (15, find_probe("masked_telos_12m_r15_probes.txt")),
        (20, find_probe("masked_telos_12m_r20_probes.txt")),
        (25, find_probe("masked_telos_12m_r25_probes.txt")),
        (30, find_probe("masked_telos_12m_r30_probes.txt")),
    ],
    "25M": [
        (1, find_probe("masked_telos_25m_r1_probes.txt")),
        (10, find_probe("masked_telos_25m_r10_probes.txt")),
        (15, find_probe("masked_telos_25m_r15_probes.txt")),
        (20, find_probe("masked_telos_25m_r20_probes.txt")),
        (25, find_probe("masked_telos_25m_r25_probes.txt")),
        (30, find_probe("masked_telos_25m_r30_probes.txt")),
        (35, find_probe("masked_telos_25m_r35_probes.txt")),
        (40, find_probe("masked_telos_25m_r40_probes.txt")),
    ],
    "50M": [
        (25, find_probe("masked_telos_50m_r25_probes.txt")),
        (35, find_probe("masked_telos_50m_r35_probes.txt")),
        (40, find_probe("masked_telos_50m_r40_probes.txt")),
        (45, find_probe("masked_telos_50m_r45_probes.txt")),
    ]
}

# -------------------------------------------------------------
# Figure 1: Scaling Laws — Target Cross-Entropy vs Token Ratio
# -------------------------------------------------------------
def plot_scaling_ce():
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    
    colors = {"12.5M": "#2563eb", "25M": "#10b981", "50M": "#8b5cf6"}
    markers = {"12.5M": "o", "25M": "s", "50M": "D"}
    
    for tier, data in ROPE_MODELS.items():
        ratios, ces = [], []
        for r, p in data:
            res = parse_probe_file(p)
            if res and res["overall_ce"] is not None:
                ratios.append(r)
                ces.append(res["overall_ce"])
        if ratios:
            ax.plot(ratios, ces, marker=markers[tier], color=colors[tier], linewidth=2.4, markersize=8, label=f"télos-{tier} (RoPE)")
            for r, c in zip(ratios, ces):
                ax.annotate(f"{c:.2f}", (r, c), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, fontweight="bold", color=colors[tier])
    
    ax.set_title("télos MDLM Scaling Laws: Target Cross-Entropy vs. Token Multiplier", pad=15, fontweight="bold")
    ax.set_xlabel("Token Over-Training Multiplier (1:N Ratio)", labelpad=10)
    ax.set_ylabel("Average Target Cross-Entropy (nats)", labelpad=10)
    ax.set_ylim(7.1, 8.6)
    ax.set_xlim(0, 48)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig("figures/scaling_cross_entropy.png")
    plt.close()
    print("  Saved figures/scaling_cross_entropy.png")

# -------------------------------------------------------------
# Figure 2: Scaling Laws — Top-5 Accuracy vs Token Ratio
# -------------------------------------------------------------
def plot_scaling_top5():
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    
    colors = {"12.5M": "#2563eb", "25M": "#10b981", "50M": "#8b5cf6"}
    markers = {"12.5M": "o", "25M": "s", "50M": "D"}
    
    for tier, data in ROPE_MODELS.items():
        ratios, accs = [], []
        for r, p in data:
            res = parse_probe_file(p)
            if res and res["top5_acc"] is not None:
                ratios.append(r)
                accs.append(res["top5_acc"])
        if ratios:
            ax.plot(ratios, accs, marker=markers[tier], color=colors[tier], linewidth=2.4, markersize=8, label=f"télos-{tier} (RoPE)")
            for r, a in zip(ratios, accs):
                if a > 0:
                    ax.annotate(f"{a:.1f}%", (r, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, fontweight="bold", color=colors[tier])
    
    ax.set_title("télos MDLM Scaling: Top-5 Probe Accuracy vs. Token Multiplier", pad=15, fontweight="bold")
    ax.set_xlabel("Token Over-Training Multiplier (1:N Ratio)", labelpad=10)
    ax.set_ylabel("Top-5 Probe Accuracy (%)", labelpad=10)
    ax.set_ylim(-1, 22)
    ax.set_xlim(0, 48)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig("figures/scaling_top5_accuracy.png")
    plt.close()
    print("  Saved figures/scaling_top5_accuracy.png")

# -------------------------------------------------------------
# Figure 3: Scaling Laws — Average Probe Rank vs Token Ratio
# -------------------------------------------------------------
def plot_scaling_rank():
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    
    colors = {"12.5M": "#2563eb", "25M": "#10b981", "50M": "#8b5cf6"}
    markers = {"12.5M": "o", "25M": "s", "50M": "D"}
    
    for tier, data in ROPE_MODELS.items():
        ratios, ranks = [], []
        for r, p in data:
            res = parse_probe_file(p)
            if res and res["overall_rank"] is not None:
                ratios.append(r)
                ranks.append(res["overall_rank"])
        if ratios:
            ax.plot(ratios, ranks, marker=markers[tier], color=colors[tier], linewidth=2.4, markersize=8, label=f"télos-{tier} (RoPE)")
            for r, rk in zip(ratios, ranks):
                ax.annotate(f"{rk:.0f}", (r, rk), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, fontweight="bold", color=colors[tier])
    
    ax.set_title("télos MDLM Probe Quality: Average Prediction Rank vs. Token Multiplier", pad=15, fontweight="bold")
    ax.set_xlabel("Token Over-Training Multiplier (1:N Ratio)", labelpad=10)
    ax.set_ylabel("Average Target Token Rank (Lower is Better, /8192)", labelpad=10)
    ax.set_ylim(500, 1900)
    ax.set_xlim(0, 48)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig("figures/scaling_average_rank.png")
    plt.close()
    print("  Saved figures/scaling_average_rank.png")

# -------------------------------------------------------------
# Figure 4: Syntactic Category Breakdown Matrix (50M 1:45 vs 25M 1:40)
# -------------------------------------------------------------
def plot_category_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    
    # 50M 1:45
    p1 = find_probe("phase_b_50m_1to45_mlx_rope_ft_probes.txt")
    res1 = parse_probe_file(p1)
    
    # 25M 1:40
    p2 = find_probe("phase_b_25m_1to40_mlx_probes.txt")
    res2 = parse_probe_file(p2)
    
    if not res1 or not res2: return
    
    cats = sorted(list(res1["categories"].keys()), key=lambda k: res1["categories"][k]["rank"])
    ranks1 = [res1["categories"][c]["rank"] for c in cats]
    ranks2 = [res2["categories"][c]["rank"] for c in cats]
    
    y_pos = np.arange(len(cats))
    height = 0.38
    
    # Horizontal Bars for Ranks comparison
    rects1 = ax1.barh(y_pos - height/2, ranks1, height, label="50M 1:45 (2.25B tok)", color="#8b5cf6", alpha=0.9)
    rects2 = ax1.barh(y_pos + height/2, ranks2, height, label="25M 1:40 (1.0B tok)", color="#10b981", alpha=0.9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(cats, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Average Prediction Rank (/8192, Lower is Better)", labelpad=10)
    ax1.set_title("Average Prediction Rank by Category", pad=12, fontweight="bold")
    ax1.legend(frameon=True, loc="lower right")
    
    for rect in rects1:
        w = rect.get_width()
        ax1.text(w + 30, rect.get_y() + rect.get_height()/2, f"#{w:.0f}", va="center", fontsize=8.5, fontweight="bold", color="#5b21b6")
    for rect in rects2:
        w = rect.get_width()
        ax1.text(w + 30, rect.get_y() + rect.get_height()/2, f"#{w:.0f}", va="center", fontsize=8.5, fontweight="bold", color="#065f46")
    ax1.set_xlim(0, max(max(ranks1), max(ranks2)) * 1.25)
    
    # Cross-Entropy comparison
    ces1 = [res1["categories"][c]["ce"] for c in cats]
    ces2 = [res2["categories"][c]["ce"] for c in cats]
    
    r_ce1 = ax2.barh(y_pos - height/2, ces1, height, label="50M 1:45 (2.25B tok)", color="#8b5cf6", alpha=0.9)
    r_ce2 = ax2.barh(y_pos + height/2, ces2, height, label="25M 1:40 (1.0B tok)", color="#10b981", alpha=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Target Cross-Entropy (nats)", labelpad=10)
    ax2.set_title("Target Cross-Entropy by Category", pad=12, fontweight="bold")
    ax2.legend(frameon=True, loc="lower right")
    
    for rect in r_ce1:
        w = rect.get_width()
        ax2.text(w + 0.1, rect.get_y() + rect.get_height()/2, f"{w:.2f}", va="center", fontsize=8.5, fontweight="bold", color="#5b21b6")
    for rect in r_ce2:
        w = rect.get_width()
        ax2.text(w + 0.1, rect.get_y() + rect.get_height()/2, f"{w:.2f}", va="center", fontsize=8.5, fontweight="bold", color="#065f46")
    ax2.set_xlim(0, max(max(ces1), max(ces2)) * 1.22)
    
    plt.tight_layout()
    plt.savefig("figures/category_breakdown_50m.png")
    plt.close()
    print("  Saved figures/category_breakdown_50m.png")


if __name__ == "__main__":
    print("Generating updated publication-quality figures...")
    plot_scaling_ce()
    plot_scaling_top5()
    plot_scaling_rank()
    plot_category_breakdown()
    print("All figures successfully created in figures/ directory!")
