"""
Generates publication-quality figures comparing all 12 models across the 3 paradigms:
- AR 12.5M (1:1, 1:5, 1:10, 1:15)
- MDLM 12.5M (1:1, 1:5, 1:10, 1:15)
- UNDLM 12.5M (1:1, 1:5, 1:10, 1:15)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup output directories
Path("figures").mkdir(exist_ok=True)
artifact_dir = Path("/Users/ivansamuel/.gemini/antigravity-ide/brain/41b584d5-6756-4262-977b-2d3c9b81f8a1")

# Clean styling configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

PROBE_FILES = {
    "AR 1:1": Path("evals/masked/probes/ar_telos_12m_r1_probes.txt"),
    "AR 1:5": Path("evals/masked/probes/ar_telos_12m_r5_probes.txt"),
    "AR 1:10": Path("evals/masked/probes/ar_telos_12m_r10_probes.txt"),
    "AR 1:15": Path("evals/masked/probes/ar_telos_12m_r15_probes.txt"),
    "MDLM 1:1": Path("evals/masked/probes/masked_telos_12m_r1_probes.txt"),
    "MDLM 1:5": Path("evals/masked/probes/masked_telos_12m_r5_probes.txt"),
    "MDLM 1:10": Path("evals/masked/probes/masked_telos_12m_r10_probes.txt"),
    "MDLM 1:15": Path("evals/masked/probes/masked_telos_12m_r15_probes.txt"),
    "UNDLM 1:1": Path("evals/masked/probes/uniform_telos_12m_r1_probes.txt"),
    "UNDLM 1:5": Path("evals/masked/probes/uniform_telos_12m_r5_probes.txt"),
    "UNDLM 1:10": Path("evals/masked/probes/uniform_telos_12m_r10_probes.txt"),
    "UNDLM 1:15": Path("evals/masked/probes/uniform_telos_12m_r15_probes.txt"),
}

def parse_probe_file(file_path: Path) -> dict:
    """Extracts overall and category-level metrics from a probe report text file."""
    if not file_path.exists():
        return None
    text = file_path.read_text()
    
    categories = {}
    if "Category-Specific Breakdown:" in text:
        cat_section = text.split("Category-Specific Breakdown:")[-1].split("====")[0]
        for line in cat_section.strip().split("\n"):
            m = re.search(r"-\s+([^:]+):\s+Target CE\s+=\s+([\d\.]+)\s+\|\s+Avg Rank\s+=\s+([\d\.]+)\s+\|\s+Top-5 Acc\s+=\s+([\d\.]+)%", line)
            if m:
                cat_name = m.group(1).strip()
                ce = float(m.group(2))
                rank = float(m.group(3))
                top5 = float(m.group(4))
                categories[cat_name] = {"ce": ce, "rank": rank, "top5": top5}

    overall_ce, overall_rank, top1_acc, top5_acc = None, None, None, None
    m_ce = re.search(r"Overall Average Target CE\s*:\s*([\d\.]+)", text)
    if m_ce: overall_ce = float(m_ce.group(1))
    m_rank = re.search(r"Overall Average Rank\s*:\s*([\d\.]+)", text)
    if m_rank: overall_rank = float(m_rank.group(1))
    m_top1 = re.search(r"Top-1 Accuracy\s*:\s*([\d\.]+)%", text)
    if m_top1: top1_acc = float(m_top1.group(1))
    m_top5 = re.search(r"Top-5 Accuracy\s*:\s*([\d\.]+)%", text)
    if m_top5: top5_acc = float(m_top5.group(1))

    return {
        "overall_ce": overall_ce,
        "overall_rank": overall_rank,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "categories": categories
    }

def generate_all_plots():
    parsed_data = {}
    for name, path in PROBE_FILES.items():
        data = parse_probe_file(path)
        if data:
            parsed_data[name] = data
            
    ratios = [1, 5, 10, 15]
    paradigms = {
        "Autoregressive (AR)": {
            "color": "#2563eb", "marker": "o",
            "keys": ["AR 1:1", "AR 1:5", "AR 1:10", "AR 1:15"]
        },
        "Masked Diffusion (MDLM)": {
            "color": "#10b981", "marker": "s",
            "keys": ["MDLM 1:1", "MDLM 1:5", "MDLM 1:10", "MDLM 1:15"]
        },
        "Uniform Diffusion (UNDLM)": {
            "color": "#f59e0b", "marker": "D",
            "keys": ["UNDLM 1:1", "UNDLM 1:5", "UNDLM 1:10", "UNDLM 1:15"]
        }
    }
    
    # -------------------------------------------------------------
    # Figure 1: Scaling Laws — Target Cross-Entropy vs Token Ratio
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for p_name, cfg in paradigms.items():
        r_list, ce_list = [], []
        for r, k in zip(ratios, cfg["keys"]):
            if k in parsed_data and parsed_data[k]["overall_ce"] is not None:
                r_list.append(r)
                ce_list.append(parsed_data[k]["overall_ce"])
        if r_list:
            ax.plot(r_list, ce_list, marker=cfg["marker"], color=cfg["color"], linewidth=2.5, markersize=8, label=p_name)
            for r, c in zip(r_list, ce_list):
                ax.annotate(f"{c:.2f}", (r, c), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold", color=cfg["color"])
                
    ax.set_title("12.5M Scaling Laws: Target Cross-Entropy vs. Token Multiplier", pad=15, fontweight="bold")
    ax.set_xlabel("Token Over-Training Multiplier (1:N Ratio)", labelpad=10)
    ax.set_ylabel("Average Target Cross-Entropy (nats, lower is better)", labelpad=10)
    ax.set_xticks(ratios)
    ax.set_xlim(0, 17)
    ax.set_ylim(6.8, 9.5)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig1_path = Path("figures/scaling_cross_entropy.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig1_path}")
    
    # -------------------------------------------------------------
    # Figure 2: Scaling Laws — Top-5 Accuracy vs Token Ratio
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for p_name, cfg in paradigms.items():
        r_list, acc_list = [], []
        for r, k in zip(ratios, cfg["keys"]):
            if k in parsed_data and parsed_data[k]["top5_acc"] is not None:
                r_list.append(r)
                acc_list.append(parsed_data[k]["top5_acc"])
        if r_list:
            ax.plot(r_list, acc_list, marker=cfg["marker"], color=cfg["color"], linewidth=2.5, markersize=8, label=p_name)
            for r, a in zip(r_list, acc_list):
                if a > 0:
                    ax.annotate(f"{a:.1f}%", (r, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold", color=cfg["color"])
                
    ax.set_title("12.5M Scaling Laws: Top-5 Probe Accuracy vs. Token Multiplier", pad=15, fontweight="bold")
    ax.set_xlabel("Token Over-Training Multiplier (1:N Ratio)", labelpad=10)
    ax.set_ylabel("Top-5 Probe Accuracy (%, higher is better)", labelpad=10)
    ax.set_xticks(ratios)
    ax.set_xlim(0, 17)
    ax.set_ylim(-1, 20)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    fig2_path = Path("figures/scaling_top5_accuracy.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig2_path}")
    
    # -------------------------------------------------------------
    # Figure 3: Scaling Laws — Average Prediction Rank vs Token Ratio
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for p_name, cfg in paradigms.items():
        r_list, rank_list = [], []
        for r, k in zip(ratios, cfg["keys"]):
            if k in parsed_data and parsed_data[k]["overall_rank"] is not None:
                r_list.append(r)
                rank_list.append(parsed_data[k]["overall_rank"])
        if r_list:
            ax.plot(r_list, rank_list, marker=cfg["marker"], color=cfg["color"], linewidth=2.5, markersize=8, label=p_name)
            for r, rk in zip(r_list, rank_list):
                ax.annotate(f"#{rk:.0f}", (r, rk), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold", color=cfg["color"])
                
    ax.set_title("12.5M Scaling Laws: Average Target Token Rank vs. Token Multiplier", pad=15, fontweight="bold")
    ax.set_xlabel("Token Over-Training Multiplier (1:N Ratio)", labelpad=10)
    ax.set_ylabel("Average Target Token Rank (/8192, lower is better)", labelpad=10)
    ax.set_xticks(ratios)
    ax.set_xlim(0, 17)
    ax.set_ylim(600, 3600)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig3_path = Path("figures/scaling_average_rank.png")
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig3_path}")
    
    # -------------------------------------------------------------
    # Figure 4: 1:15 Peak Ratio Category Comparison (AR vs MDLM vs UNDLM)
    # -------------------------------------------------------------
    categories = sorted(list(parsed_data["MDLM 1:15"]["categories"].keys()))
    n_cats = len(categories)
    
    fig, (ax_ce, ax_rank) = plt.subplots(1, 2, figsize=(16, 6))
    y = np.arange(n_cats)
    bar_height = 0.25
    
    comp_models = [
        ("AR 1:15 (188M tok)", "AR 1:15", "#2563eb", -1),
        ("MDLM 1:15 (188M tok)", "MDLM 1:15", "#10b981", 0),
        ("UNDLM 1:15 (188M tok)", "UNDLM 1:15", "#f59e0b", 1),
    ]
    
    for label, key, color, offset in comp_models:
        ces = [parsed_data[key]["categories"][c]["ce"] for c in categories]
        ranks = [parsed_data[key]["categories"][c]["rank"] for c in categories]
        ax_ce.barh(y + offset * bar_height, ces, bar_height, label=label, color=color, alpha=0.88)
        ax_rank.barh(y + offset * bar_height, ranks, bar_height, label=label, color=color, alpha=0.88)
        
    ax_ce.set_yticks(y)
    ax_ce.set_yticklabels(categories, fontweight="bold")
    ax_ce.invert_yaxis()
    ax_ce.set_xlabel("Target Cross-Entropy (nats, lower is better)")
    ax_ce.set_title("Syntactic Category Target CE (1:15 Ratio)", fontweight="bold")
    ax_ce.legend(frameon=True, loc="lower right")
    ax_ce.set_xlim(4.0, 11.5)
    
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels([])
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("Average Prediction Rank (/8192, lower is better)")
    ax_rank.set_title("Syntactic Category Average Rank (1:15 Ratio)", fontweight="bold")
    ax_rank.legend(frameon=True, loc="lower right")
    ax_rank.set_xlim(0, 5000)
    
    plt.suptitle("Syntactic Category Breakdown: 12.5M Models at 1:15 Ratio (188.0M Tokens)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig4_path = Path("figures/category_breakdown_12m_r15.png")
    plt.savefig(fig4_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig4_path}")
    
    # Copy all figures to artifact directory
    if artifact_dir.exists():
        import shutil
        for f in [fig1_path, fig2_path, fig3_path, fig4_path]:
            shutil.copy(f, artifact_dir / f.name)
            print(f"  Copied {f.name} to artifact directory.")

if __name__ == "__main__":
    generate_all_plots()
