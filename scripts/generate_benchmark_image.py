#!/usr/bin/env python3
"""
Télos Benchmark Comparison Image Generator (Paper & Website Cream Aesthetic).

Generates publication-quality, warm cream/ivory themed comparison graphs (PNG)
matching the exact design system of https://telos.research.wingit.tech/pages/telos-preliminary-2.
Composites the official Télos logo (`logos/telos_logo.png`) seamlessly on the cream canvas.

Outputs high-DPI images:
  1. `figures/benchmark_comparison_graph.png` (Grouped Vertical Bar Chart - Fig 4/9 style)
  2. `figures/benchmark_breakdown_horizontal.png` (Horizontal Category Breakdown - Fig 14 style)
  3. `figures/benchmark_radar_graph.png` (Capability Envelope Radar Fingerprint)
"""

import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGO_PATH = PROJECT_ROOT / "logos" / "telos_logo.png"
FIGURES_DIR = PROJECT_ROOT / "figures"

OUTPUT_BAR_PNG = FIGURES_DIR / "benchmark_comparison_graph.png"
OUTPUT_HORIZONTAL_PNG = FIGURES_DIR / "benchmark_breakdown_horizontal.png"
OUTPUT_RADAR_PNG = FIGURES_DIR / "benchmark_radar_graph.png"

# Color tokens extracted directly from the Télos research paper design system (index-B-kcauLn.css)
COLOR_BG_PAGE = "#FDF9F5"     # var(--color-bg): Warm cream / ivory page background
COLOR_BG_CARD = "#F4EBE1"     # var(--color-bg-alt): Soft warm parchment card background
COLOR_TEXT_MAIN = "#2D2A26"   # var(--color-text-main): Deep espresso charcoal text
COLOR_TEXT_MUTED = "#5C554D"  # var(--color-text-muted): Muted warm taupe / sepia text
COLOR_BORDER = "#E8DFD5"      # var(--color-border): Delicate border tone
COLOR_GRID = "#E5D9C5"        # CartesianGrid stroke="#E5D9C5" strokeDasharray="3 3"
COLOR_SPINE = "#C8BFB4"       # Subtle axis spine color

# Configure font hierarchy matching Outfit & clean modern macOS sans-serif typography
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Outfit", "Avenir Next", "Helvetica Neue", "Arial", "sans-serif"]
FONT_FAMILY_SANS = "sans-serif"


# Model categorical palette from paper figures
MODEL_COLORS = {
    "AFM-3 Core Advanced": "#2563EB",   # Royal Blue
    "IBM Granite 4.2 3B": "#D97706",    # Warm Amber / Terracotta
    "LiquidAI LFM 8B A1B": "#059669",   # Forest Emerald Green
    "Google Gemma 4 E4B": "#7C3AED",    # Deep Amethyst Purple
    "Microsoft Phi-4 Mini": "#DC2626",  # Crimson / Carmine Red
}

SUITE_KEYS = [
    "humaneval",
    "tooluse",
    "cyber",
    "gpqa_diamond",
    "mmlu_science",
    "competition_math",
    "arc"
]


def load_master_summaries() -> Dict[str, Dict[str, Any]]:
    """Loads all master summary JSON files from `logs/` and normalizes metrics."""
    master_files = sorted(glob.glob(str(LOGS_DIR / "*master_summary.json")))
    models_data = {}

    model_name_map = {
        "afm3": "AFM-3 Core Advanced",
        "granite_mlx": "IBM Granite 4.2 3B",
        "lfm8b": "LiquidAI LFM 8B A1B",
        "phi4_mini": "Microsoft Phi-4 Mini",
        "gemma4_e4b": "Google Gemma 4 E4B"
    }

    for mf in master_files:
        path = Path(mf)
        key = path.name.replace("_master_summary.json", "").replace("eval_report_", "")
        with open(path) as f:
            try:
                data = json.load(f)
            except Exception:
                continue

        model_label = model_name_map.get(key, key.replace("_", " ").title())
        suites = {}

        for suite_key, info in data.items():
            if isinstance(info, dict) and "pass_at_1_pct" in info:
                suites[suite_key] = {
                    "label": info.get("label", suite_key),
                    "pass_at_1_pct": float(info.get("pass_at_1_pct", 0.0)),
                }

        # Strict Filter: Only include models that have completed ALL 7 benchmark suites!
        # Unfinished benchmarks like Phi-4 Mini or in-progress runs must NOT be displayed.
        completed_suites = [s for s in SUITE_KEYS if s in suites]
        if len(completed_suites) == len(SUITE_KEYS):
            models_data[model_label] = suites
        else:
            print(f"ℹ Skipping unfinished model '{model_label}' ({len(completed_suites)}/{len(SUITE_KEYS)} suites completed)")

    return models_data


def get_clean_logo_image() -> Image.Image:
    """Extracts the Télos logo and removes white background for seamless blending on cream."""
    if not LOGO_PATH.exists():
        return None
    
    img = Image.open(LOGO_PATH).convert("RGBA")
    arr = np.array(img).copy()
    
    # White pixels (RGB > 240) become fully transparent
    mask = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
    arr[mask, 3] = 0

    # Non-white pixels tinted towards deep espresso #2D2A26 for crisp high-contrast text
    non_transparent = ~mask
    for c, target_val in enumerate([45, 42, 38]):
        arr[non_transparent, c] = target_val

    return Image.fromarray(arr)


def generate_paper_vertical_comparison(data: Dict[str, Dict[str, Any]]):
    """
    Renders grouped vertical bar chart matching Section 3 / Figure 4 & 9 Recharts styling.
    """
    suite_names = [
        ("humaneval", "HumanEval\n(Code)"),
        ("tooluse", "Tool-Use\n(BFCL)"),
        ("cyber", "Cybersecurity\n(OWASP)"),
        ("gpqa_diamond", "GPQA Diamond\n(PhD Science)"),
        ("mmlu_science", "MMLU Science\n(STEM)"),
        ("competition_math", "MATH\n(Hendrycks)"),
        ("arc", "ARC-Challenge\n(Reasoning)")
    ]

    suites = [s[0] for s in suite_names]
    labels = [s[1] for s in suite_names]
    models = [m for m in MODEL_COLORS.keys() if m in data]
    n_suites = len(suites)
    n_models = len(models)

    # Clean figure matching the warm card container in the paper
    fig = plt.figure(figsize=(13.5, 7.8), dpi=300, facecolor=COLOR_BG_CARD)

    # 1. Dedicated Header Band (Y: 0.86 to 0.98) - strictly above plot area
    logo = get_clean_logo_image()
    if logo is not None:
        logo_ax = fig.add_axes([0.06, 0.905, 0.13, 0.055], anchor="NW")
        logo_ax.imshow(logo)
        logo_ax.axis("off")

    fig.text(0.20, 0.925, "Télos Benchmark Evaluation: Pass@1 Across Model Architectures",
             fontsize=14.5, fontweight="bold", fontfamily=FONT_FAMILY_SANS, color=COLOR_TEXT_MAIN, va="center")

    # Plot axes strictly bounded between Y: 0.12 and Y: 0.80
    ax = fig.add_axes([0.07, 0.12, 0.88, 0.68])
    ax.set_facecolor(COLOR_BG_CARD)

    # Dotted horizontal grid lines matching Recharts CartesianGrid strokeDasharray="3 3"
    ax.grid(True, axis="y", linestyle=(0, (3, 3)), color=COLOR_GRID, linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_SPINE)
    ax.spines["bottom"].set_color(COLOR_SPINE)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # X positions
    x = np.arange(n_suites)
    bar_width = 0.75 / n_models

    for i, model_name in enumerate(models):
        scores = [data[model_name].get(s, {}).get("pass_at_1_pct", 0.0) for s in suites]
        color = MODEL_COLORS.get(model_name, "#2563EB")
        offset = x + (i - (n_models - 1) / 2) * bar_width
        
        bars = ax.bar(
            offset, scores, bar_width * 0.90,
            label=model_name,
            color=color,
            edgecolor="none",
            zorder=3
        )

        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold",
                    color=COLOR_TEXT_MAIN,
                    fontfamily=FONT_FAMILY_SANS
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold", fontfamily=FONT_FAMILY_SANS, color=COLOR_TEXT_MAIN)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Pass@1 Accuracy (%)", fontsize=11, fontfamily=FONT_FAMILY_SANS, color=COLOR_TEXT_MUTED, labelpad=10)
    ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=10)

    # Recharts-style Legend (centered, no box border, clean square swatches)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.11),
        ncol=len(models),
        frameon=False,
        fontsize=10,
        handlelength=1.0,
        handleheight=1.0,
        labelcolor=COLOR_TEXT_MAIN,
        prop={"family": FONT_FAMILY_SANS, "weight": "600"}
    )

    plt.savefig(OUTPUT_BAR_PNG, facecolor=COLOR_BG_CARD, edgecolor="none", dpi=300)
    plt.close()
    print(f"[OK] Saved Paper Vertical Comparison Graph: {OUTPUT_BAR_PNG}")


def generate_paper_horizontal_breakdown(data: Dict[str, Dict[str, Any]]):
    """
    Renders horizontal bar chart matching Screenshot 2 / Section 3 Category Rank Breakdown.
    """
    suite_names = [
        ("humaneval", "HumanEval (Code)"),
        ("tooluse", "Tool-Use (BFCL)"),
        ("cyber", "Cybersecurity (OWASP)"),
        ("gpqa_diamond", "GPQA Diamond (PhD Science)"),
        ("mmlu_science", "MMLU Science (STEM)"),
        ("competition_math", "MATH (Hendrycks)"),
        ("arc", "ARC-Challenge (Reasoning)")
    ]

    # Invert so top benchmark in list appears at the top of the Y-axis
    suite_names.reverse()
    suites = [s[0] for s in suite_names]
    labels = [s[1] for s in suite_names]
    models = [m for m in MODEL_COLORS.keys() if m in data]
    n_suites = len(suites)
    n_models = len(models)

    fig = plt.figure(figsize=(13.0, 7.8), dpi=300, facecolor=COLOR_BG_CARD)

    # 1. Dedicated Header Band (Y: 0.86 to 0.98) - strictly above plot area
    logo = get_clean_logo_image()
    if logo is not None:
        logo_ax = fig.add_axes([0.06, 0.905, 0.13, 0.055], anchor="NW")
        logo_ax.imshow(logo)
        logo_ax.axis("off")

    fig.text(0.20, 0.925, "Benchmark Accuracy Breakdown: Multi-Model Evaluation",
             fontsize=14.5, fontweight="bold", fontfamily=FONT_FAMILY_SANS, color=COLOR_TEXT_MAIN, va="center")

    # Plot axes strictly bounded between Y: 0.10 and Y: 0.80
    ax = fig.add_axes([0.23, 0.10, 0.71, 0.70])
    ax.set_facecolor(COLOR_BG_CARD)

    # Dotted vertical grid lines matching Recharts CartesianGrid strokeDasharray="3 3"
    ax.grid(True, axis="x", linestyle=(0, (3, 3)), color=COLOR_GRID, linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_SPINE)
    ax.spines["bottom"].set_color(COLOR_SPINE)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Y positions
    y = np.arange(n_suites)
    bar_height = 0.75 / n_models

    for i, model_name in enumerate(models):
        scores = [data[model_name].get(s, {}).get("pass_at_1_pct", 0.0) for s in suites]
        color = MODEL_COLORS.get(model_name, "#2563EB")
        offset = y + (i - (n_models - 1) / 2) * bar_height
        
        bars = ax.barh(
            offset, scores, bar_height * 0.90,
            label=model_name,
            color=color,
            edgecolor="none",
            zorder=3
        )

        for bar in bars:
            w = bar.get_width()
            if w > 0:
                ax.annotate(
                    f"{w:.1f}%",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left", va="center",
                    fontsize=8.5, fontweight="bold",
                    color=COLOR_TEXT_MAIN,
                    fontfamily=FONT_FAMILY_SANS
                )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5, fontweight="500", fontfamily=FONT_FAMILY_SANS, color=COLOR_TEXT_MAIN)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Pass@1 Accuracy (%)", fontsize=11, fontfamily=FONT_FAMILY_SANS, color=COLOR_TEXT_MUTED, labelpad=10)
    ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=10)

    # Legend centered above plot area
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.11),
        ncol=len(models),
        frameon=False,
        fontsize=10,
        handlelength=1.0,
        handleheight=1.0,
        labelcolor=COLOR_TEXT_MAIN,
        prop={"family": FONT_FAMILY_SANS, "weight": "600"}
    )

    plt.savefig(OUTPUT_HORIZONTAL_PNG, facecolor=COLOR_BG_CARD, edgecolor="none", dpi=300)
    plt.close()
    print(f"[OK] Saved Paper Horizontal Breakdown Graph: {OUTPUT_HORIZONTAL_PNG}")


def generate_paper_radar_graph(data: Dict[str, Dict[str, Any]]):
    """
    Renders capability radar fingerprint chart in the exact paper palette.
    """
    suite_labels = {
        "humaneval": "Code\n(HumanEval)",
        "tooluse": "Tool-Use\n(BFCL)",
        "cyber": "Cybersecurity\n(OWASP)",
        "gpqa_diamond": "PhD Science\n(GPQA)",
        "mmlu_science": "STEM\n(MMLU)",
        "competition_math": "MATH\n(Hendrycks)",
        "arc": "Reasoning\n(ARC)"
    }

    suites = list(suite_labels.keys())
    categories = [suite_labels[s] for s in suites]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure(figsize=(9.5, 9.2), dpi=300, facecolor=COLOR_BG_CARD)
    
    # Dedicated Header Band (strictly above polar plot)
    logo = get_clean_logo_image()
    if logo is not None:
        logo_ax = fig.add_axes([0.06, 0.91, 0.14, 0.055], anchor="NW")
        logo_ax.imshow(logo)
        logo_ax.axis("off")

    fig.text(0.22, 0.93, "Model Capability Radar: Multi-Domain Skill Envelope",
             fontsize=13.5, fontweight="bold", fontfamily=FONT_FAMILY_SANS, color=COLOR_TEXT_MAIN, va="center")

    # Polar axis with proper margin clearance strictly below header
    ax = fig.add_axes([0.12, 0.06, 0.76, 0.72], polar=True)
    ax.set_facecolor(COLOR_BG_CARD)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color=COLOR_TEXT_MAIN, size=10, weight="600", family=FONT_FAMILY_SANS)
    ax.tick_params(pad=14)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80], ["20%", "40%", "60%", "80%"], color=COLOR_TEXT_MUTED, size=8.5, family=FONT_FAMILY_SANS)
    plt.ylim(0, 100)
    ax.grid(color=COLOR_GRID, linestyle=(0, (3, 3)), linewidth=1.2)

    for model_name, suites_dict in data.items():
        values = [suites_dict.get(s, {}).get("pass_at_1_pct", 0.0) for s in suites]
        values += values[:1]
        color = MODEL_COLORS.get(model_name, "#2563EB")
        ax.plot(angles, values, linewidth=2.2, linestyle="solid", label=model_name, color=color)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(data),
        frameon=False,
        fontsize=9.5,
        prop={"family": FONT_FAMILY_SANS, "weight": "600"},
        labelcolor=COLOR_TEXT_MAIN
    )

    plt.savefig(OUTPUT_RADAR_PNG, facecolor=COLOR_BG_CARD, edgecolor="none", dpi=300)
    plt.close()
    print(f"[OK] Saved Paper Radar Graph: {OUTPUT_RADAR_PNG}")



def main():
    """Generates all benchmark comparison images in the paper cream aesthetic."""
    print("=" * 80)
    print("  TÉLOS BENCHMARK IMAGE GENERATOR (WEBSITE / PAPER CREAM AESTHETIC)")
    print("=" * 80 + "\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_master_summaries()
    print(f"[OK] Loaded evaluation data for {len(data)} models:")
    for model, suites in data.items():
        print(f"  · {model:<24}: {len(suites)} suites")

    generate_paper_vertical_comparison(data)
    generate_paper_horizontal_breakdown(data)
    generate_paper_radar_graph(data)

    print("\n[OK] Publication-grade image generation complete!")


if __name__ == "__main__":
    main()
