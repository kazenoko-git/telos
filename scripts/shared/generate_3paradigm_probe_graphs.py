"""
Generates publication-quality figures comparing all 6 models across the 3 paradigms:
- AR 12.5M (1:1 & 1:5)
- MDLM 12.5M (1:1 & 1:5)
- UNDLM 12.5M (1:1 & 1:5)
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
    "MDLM 1:1": Path("evals/masked/probes/masked_telos_12m_r1_probes.txt"),
    "MDLM 1:5": Path("evals/masked/probes/masked_telos_12m_r5_probes.txt"),
    "UNDLM 1:1": Path("evals/masked/probes/uniform_telos_12m_r1_probes.txt"),
    "UNDLM 1:5": Path("evals/masked/probes/uniform_telos_12m_r5_probes.txt"),
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
            
    models = list(parsed_data.keys())
    
    # -------------------------------------------------------------
    # Figure 1: 3-Paradigm Overview (Target CE, Avg Rank, Top-5 Acc)
    # -------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = ["#2563eb", "#1d4ed8", "#10b981", "#059669", "#f59e0b", "#d97706"]
    x = np.arange(len(models))
    width = 0.6
    
    ces = [parsed_data[m]["overall_ce"] for m in models]
    ranks = [parsed_data[m]["overall_rank"] for m in models]
    top5s = [parsed_data[m]["top5_acc"] for m in models]
    
    # CE Plot
    rects1 = ax1.bar(x, ces, width, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Target Cross-Entropy (nats, lower is better)")
    ax1.set_title("Overall Target Cross-Entropy", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right")
    ax1.set_ylim(6.0, 10.5)
    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 0.1, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    # Rank Plot
    rects2 = ax2.bar(x, ranks, width, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_ylabel("Average Prediction Rank (/8192, lower is better)")
    ax2.set_title("Overall Average Prediction Rank", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha="right")
    ax2.set_ylim(0, 3600)
    for rect in rects2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 50, f"#{h:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    # Top-5 Acc Plot
    rects3 = ax3.bar(x, top5s, width, color=colors, edgecolor="black", alpha=0.85)
    ax3.set_ylabel("Top-5 Probe Accuracy (%, higher is better)")
    ax3.set_title("Overall Top-5 Probe Accuracy", fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=30, ha="right")
    ax3.set_ylim(0, 3.0)
    for rect in rects3:
        h = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width()/2., h + 0.05, f"{h:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    plt.suptitle("12.5M Parameter 3-Paradigm Comparison: AR vs. MDLM vs. UNDLM (1:1 & 1:5 Ratios)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = Path("figures/all_6_models_overview.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig1_path}")
    
    # -------------------------------------------------------------
    # Figure 2: Category Breakdown across all 6 models
    # -------------------------------------------------------------
    categories = sorted(list(parsed_data["MDLM 1:1"]["categories"].keys()))
    n_cats = len(categories)
    
    fig, (ax_ce, ax_rank) = plt.subplots(1, 2, figsize=(16, 7))
    
    y = np.arange(n_cats)
    bar_height = 0.13
    
    paradigm_styles = [
        ("AR 1:1", "#3b82f6", -2.5),
        ("AR 1:5", "#1d4ed8", -1.5),
        ("MDLM 1:1", "#10b981", -0.5),
        ("MDLM 1:5", "#047857", 0.5),
        ("UNDLM 1:1", "#f59e0b", 1.5),
        ("UNDLM 1:5", "#b45309", 2.5),
    ]
    
    for m_name, color, offset in paradigm_styles:
        cat_ces = [parsed_data[m_name]["categories"][c]["ce"] for c in categories]
        cat_ranks = [parsed_data[m_name]["categories"][c]["rank"] for c in categories]
        
        ax_ce.barh(y + offset * bar_height, cat_ces, bar_height, label=m_name, color=color, alpha=0.9)
        ax_rank.barh(y + offset * bar_height, cat_ranks, bar_height, label=m_name, color=color, alpha=0.9)
        
    ax_ce.set_yticks(y)
    ax_ce.set_yticklabels(categories, fontweight="bold")
    ax_ce.invert_yaxis()
    ax_ce.set_xlabel("Target Cross-Entropy (nats)")
    ax_ce.set_title("Target CE by Syntactic Category (All 6 Models)", fontweight="bold")
    ax_ce.legend(frameon=True, loc="lower right")
    ax_ce.set_xlim(5.5, 12.0)
    
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels([])
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("Average Prediction Rank (/8192)")
    ax_rank.set_title("Prediction Rank by Syntactic Category (All 6 Models)", fontweight="bold")
    ax_rank.legend(frameon=True, loc="lower right")
    ax_rank.set_xlim(0, 5500)
    
    plt.suptitle("Syntactic Category Benchmark: Complete 6-Model Paradigm Matrix (12.5M Scale)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig2_path = Path("figures/all_6_models_category_matrix.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig2_path}")
    
    # -------------------------------------------------------------
    # Figure 3: Paradigm Scaling Trajectory (1:1 -> 1:5)
    # -------------------------------------------------------------
    fig, (ax_tce, ax_trank) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    ratios = [1, 5]
    paradigms = [
        ("Autoregressive (AR)", [parsed_data["AR 1:1"]["overall_ce"], parsed_data["AR 1:5"]["overall_ce"]], [parsed_data["AR 1:1"]["overall_rank"], parsed_data["AR 1:5"]["overall_rank"]], "#2563eb", "o"),
        ("Masked Diffusion (MDLM)", [parsed_data["MDLM 1:1"]["overall_ce"], parsed_data["MDLM 1:5"]["overall_ce"]], [parsed_data["MDLM 1:1"]["overall_rank"], parsed_data["MDLM 1:5"]["overall_rank"]], "#10b981", "s"),
        ("Uniform Diffusion (UNDLM)", [parsed_data["UNDLM 1:1"]["overall_ce"], parsed_data["UNDLM 1:5"]["overall_ce"]], [parsed_data["UNDLM 1:1"]["overall_rank"], parsed_data["UNDLM 1:5"]["overall_rank"]], "#f59e0b", "D"),
    ]
    
    for p_name, ce_vals, rank_vals, col, mark in paradigms:
        ax_tce.plot(ratios, ce_vals, marker=mark, color=col, linewidth=2.5, markersize=8, label=p_name)
        for r, val in zip(ratios, ce_vals):
            ax_tce.annotate(f"{val:.2f}", (r, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9.5, fontweight="bold", color=col)
            
        ax_trank.plot(ratios, rank_vals, marker=mark, color=col, linewidth=2.5, markersize=8, label=p_name)
        for r, val in zip(ratios, rank_vals):
            ax_trank.annotate(f"#{val:.0f}", (r, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9.5, fontweight="bold", color=col)
            
    ax_tce.set_title("Target Cross-Entropy Scaling Trajectory (1:1 to 1:5)", fontweight="bold")
    ax_tce.set_xlabel("Token Multiplier Ratio (1:N)")
    ax_tce.set_ylabel("Target Cross Entropy (nats)")
    ax_tce.set_xticks([1, 2, 3, 4, 5])
    ax_tce.set_ylim(7.5, 9.8)
    ax_tce.legend(frameon=True)
    
    ax_trank.set_title("Average Prediction Rank Trajectory (1:1 to 1:5)", fontweight="bold")
    ax_trank.set_xlabel("Token Multiplier Ratio (1:N)")
    ax_trank.set_ylabel("Average Target Token Rank (/8192)")
    ax_trank.set_xticks([1, 2, 3, 4, 5])
    ax_trank.set_ylim(500, 3600)
    ax_trank.legend(frameon=True)
    
    plt.suptitle("12.5M Paradigm Scaling Sensitivity: 1:1 vs. 1:5 Multipliers", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3_path = Path("figures/all_6_models_scaling_trajectory.png")
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig3_path}")
    
    # Copy all figures to artifact directory
    if artifact_dir.exists():
        import shutil
        for f in [fig1_path, fig2_path, fig3_path]:
            shutil.copy(f, artifact_dir / f.name)
            print(f"  Copied {f.name} to artifact directory.")

if __name__ == "__main__":
    generate_all_plots()
