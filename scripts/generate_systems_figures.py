#!/usr/bin/env python3
"""
Télos Systems & Quality Diagram Generator: Throughput, Memory & Degenerate Infinite Loop Rates.

Strictly covers the three evaluated models:
  1. Apple AFM-3 Core Advanced
  2. IBM Granite 4.2 3B
  3. LiquidAI LFM 8B A1B

Generates:
  1. `figures/benchmark_throughput_memory.png` (Decoding Throughput & Memory Footprint)
  2. `figures/benchmark_repetition_rate.png` (Degenerate Infinite Loop Rates Across All 3,067 Tasks & Lexical Repetition)
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGO_PATH = PROJECT_ROOT / "logos" / "telos_logo.png"

OUTPUT_THROUGHPUT_PNG = FIGURES_DIR / "benchmark_throughput_memory.png"
OUTPUT_REPETITION_PNG = FIGURES_DIR / "benchmark_repetition_rate.png"

# Cream/Parchment Paper Design Tokens
COLOR_BG_CARD = "#F4EBE1"     # Warm soft parchment card container
COLOR_TEXT_MAIN = "#2D2A26"   # Deep espresso charcoal text
COLOR_TEXT_MUTED = "#5C554D"  # Muted warm taupe text
COLOR_BORDER = "#E8DFD5"      # Delicate border tone
COLOR_GRID = "#E5D9C5"        # Dotted grid lines
COLOR_SPINE = "#C8BFB4"       # Subtle axis spine color

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Outfit", "Avenir Next", "Helvetica Neue", "Arial", "sans-serif"]
FONT_FAMILY_SANS = "sans-serif"

# Model Colors
MODEL_COLORS = {
    "Apple AFM-3 Core Advanced": "#2563EB",  # Royal Blue
    "IBM Granite 4.2 3B": "#D97706",         # Warm Terracotta Amber
    "LiquidAI LFM 8B A1B": "#059669",        # Forest Emerald Green
}


def get_clean_logo_image() -> Image.Image:
    """Extracts and tints the Télos logo for seamless blending on cream."""
    if not LOGO_PATH.exists():
        return None
    img = Image.open(LOGO_PATH).convert("RGBA")
    arr = np.array(img).copy()
    mask = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
    arr[mask, 3] = 0
    non_transparent = ~mask
    for c, target_val in enumerate([45, 42, 38]):
        arr[non_transparent, c] = target_val
    return Image.fromarray(arr)


def generate_throughput_and_memory_figure():
    """
    Renders dual-panel figure for AFM-3, IBM Granite, and LiquidAI LFM 8B:
      Panel A: Decoding Throughput (tok/sec) [Sustained Code vs Peak Burst]
      Panel B: Resident Unified Memory Footprint (GB) & Efficiency (tok/s·GB)
    """
    models = [
        "Apple AFM-3 Core Advanced",
        "IBM Granite 4.2 3B",
        "LiquidAI LFM 8B A1B"
    ]
    
    # Grounded empirical measurements on Apple M5 Pro (24 GB Unified Memory)
    sustained_tok_s = [58.9, 61.1, 150.9]
    peak_tok_s = [86.2, 103.2, 166.2]
    memory_gb = [1.33, 2.07, 4.85]
    efficiency = [peak / mem for peak, mem in zip(peak_tok_s, memory_gb)]

    fig = plt.figure(figsize=(14.0, 7.4), dpi=300, facecolor=COLOR_BG_CARD)

    logo = get_clean_logo_image()
    if logo is not None:
        logo_ax = fig.add_axes([0.06, 0.908, 0.12, 0.055], anchor="NW")
        logo_ax.imshow(logo)
        logo_ax.axis("off")

    fig.text(0.19, 0.925, "On-Device Inference: Unified Memory Utilisation & Decoding Throughput",
             fontsize=14.5, fontweight="bold", fontfamily=FONT_FAMILY_SANS, color=COLOR_TEXT_MAIN, va="center")
    fig.text(0.19, 0.895, "Apple M5 Pro (24 GB Unified Memory) | AFM-3 Core Advanced vs. IBM Granite 4.2 3B vs. LiquidAI LFM 8B A1B",
             fontsize=9.8, fontfamily=FONT_FAMILY_SANS, color=COLOR_TEXT_MUTED, va="center")

    short_labels = [
        "Apple AFM-3\nCore Advanced",
        "IBM Granite\n4.2 3B",
        "LiquidAI\nLFM 8B A1B"
    ]

    # Panel A: Decoding Throughput
    ax1 = fig.add_axes([0.08, 0.13, 0.40, 0.68])
    ax1.set_facecolor(COLOR_BG_CARD)
    ax1.grid(True, axis="y", linestyle=(0, (3, 3)), color=COLOR_GRID, linewidth=1.1, zorder=0)
    ax1.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax1.spines[spine].set_color(COLOR_SPINE)

    x = np.arange(len(models))
    width = 0.30

    for i, m in enumerate(models):
        c = MODEL_COLORS[m]
        # Sustained bar
        ax1.bar(x[i] - width/2, sustained_tok_s[i], width * 0.90,
                color=c, alpha=0.72, edgecolor="none", zorder=3)
        # Peak bar
        ax1.bar(x[i] + width/2, peak_tok_s[i], width * 0.90,
                color=c, alpha=1.0, edgecolor="none", zorder=3)

        # Label peak values
        ax1.annotate(f"{peak_tok_s[i]:.1f}",
                     xy=(x[i] + width/2, peak_tok_s[i]),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9.0, fontweight="bold",
                     color=COLOR_TEXT_MAIN)
        # Label sustained values
        ax1.annotate(f"{sustained_tok_s[i]:.1f}",
                     xy=(x[i] - width/2, sustained_tok_s[i]),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5,
                     color=COLOR_TEXT_MUTED)

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=10.0, fontweight="600", color=COLOR_TEXT_MAIN)
    ax1.set_ylabel("Decoding Throughput (tokens / second)", fontsize=10.5, color=COLOR_TEXT_MUTED, labelpad=10)
    ax1.set_ylim(0, 190)
    ax1.set_title("Panel A: Sustained vs. Peak Decoding Speed", fontsize=11.5, fontweight="bold",
                  color=COLOR_TEXT_MAIN, pad=14)

    p_sust = patches.Patch(facecolor="#5C554D", alpha=0.72, label="Sustained Code")
    p_peak = patches.Patch(facecolor="#2D2A26", alpha=1.0, label="Peak Decode")
    ax1.legend(handles=[p_sust, p_peak], loc="upper left", frameon=False, fontsize=9.2)

    # Panel B: Memory Utilisation & Efficiency
    ax2 = fig.add_axes([0.56, 0.13, 0.39, 0.68])
    ax2.set_facecolor(COLOR_BG_CARD)
    ax2.grid(True, axis="y", linestyle=(0, (3, 3)), color=COLOR_GRID, linewidth=1.1, zorder=0)
    ax2.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax2.spines[spine].set_color(COLOR_SPINE)

    bars_mem = ax2.bar(x, memory_gb, width * 1.3,
                       color=[MODEL_COLORS[m] for m in models],
                       alpha=0.88, edgecolor="none", zorder=3)

    for i, bar in enumerate(bars_mem):
        h = bar.get_height()
        ax2.annotate(f"{h:.2f} GB",
                     xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                     color=COLOR_TEXT_MAIN)
        ax2.annotate(f"{efficiency[i]:.1f} tok/s·GB",
                     xy=(bar.get_x() + bar.get_width()/2, h / 2),
                     ha="center", va="center", fontsize=8.8, fontweight="bold",
                     color="#FFFFFF" if h > 1.8 else COLOR_TEXT_MAIN)

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=10.0, fontweight="600", color=COLOR_TEXT_MAIN)
    ax2.set_ylabel("Resident Unified Memory (GB)", fontsize=10.5, color=COLOR_TEXT_MUTED, labelpad=10)
    ax2.set_ylim(0, 6.0)
    ax2.set_title("Panel B: Resident RAM & Efficiency (tok/s / GB RAM)", fontsize=11.5, fontweight="bold",
                  color=COLOR_TEXT_MAIN, pad=14)

    plt.savefig(OUTPUT_THROUGHPUT_PNG, facecolor=COLOR_BG_CARD, edgecolor="none", dpi=300)
    plt.close()
    print(f"✓ Saved Throughput & Memory Figure: {OUTPUT_THROUGHPUT_PNG}")


def generate_repetition_rate_figure():
    """
    Renders dedicated Behavioral Quality figure with Degenerate Infinite Loop Rates as the primary focus:
      Panel A (Left): Degenerate Infinite Loop Rate (%) across all 5 benchmark tracks and Overall (3,067 tasks)
      Panel B (Right): Multi-Granular Lexical Repetition (2-Gram, 3-Gram, 4-Gram, Line Repetition) on Code
    """
    models = [
        "Apple AFM-3 Core Advanced",
        "IBM Granite 4.2 3B",
        "LiquidAI LFM 8B A1B"
    ]

    # Exact empirical degenerate loop rates across all evaluated suites
    # Evaluated across: HumanEval (164), GPQA Diamond (198), MATH (700), MMLU Science (833), ARC (1172)
    categories = [
        "Code\n(HumanEval)",
        "PhD Science\n(GPQA)",
        "Competition\nMATH",
        "STEM\n(MMLU)",
        "Overall\n(3,067 Tasks)"
    ]

    # Exact empirical degenerate runaway loop rates across all evaluated suites:
    # HumanEval (164), GPQA Diamond (198), Competition MATH (700), MMLU Science (833), ARC (1,172)
    categories = [
        "Code\n(HumanEval)",
        "PhD Science\n(GPQA)",
        "Competition\nMATH",
        "STEM\n(MMLU)",
        "Overall\n(3,067 Tasks)"
    ]

    # Ground-truth runaway loop percentages (cycling loops & token blowup):
    # Granite was the heaviest runaway victim overall (1,129 runaway tasks, 36.8%),
    # failing 97.0% of GPQA and 75.4% of MATH in circular reasoning.
    loops_afm =     [2.4,  67.2, 73.3,  6.7, 23.4]
    loops_granite = [1.2,  97.0, 75.4, 48.9, 36.8]
    loops_lfm =     [20.1, 90.9, 61.6, 26.5, 28.5]

    # Lexical repetition metrics on HumanEval code generation:
    rep_metrics = ["2-Gram", "3-Gram", "4-Gram", "Line Rep"]
    lex_afm =     [24.6, 13.2,  8.2, 2.7]
    lex_granite = [19.2, 11.0,  6.9, 2.0]
    lex_lfm =     [34.5, 21.6, 14.5, 3.6]

    fig = plt.figure(figsize=(15.8, 7.8), dpi=300, facecolor=COLOR_BG_CARD)

    # 1. Header with Télos logo
    logo = get_clean_logo_image()
    if logo is not None:
        logo_ax = fig.add_axes([0.05, 0.912, 0.12, 0.055], anchor="NW")
        logo_ax.imshow(logo)
        logo_ax.axis("off")

    fig.text(0.18, 0.932, "Comparative Behavioral Quality: Degenerate Infinite Loop Rates & Repetition",
             fontsize=14.5, fontweight="bold", fontfamily=FONT_FAMILY_SANS, color=COLOR_TEXT_MAIN, va="center")
    fig.text(0.18, 0.902, "Empirical Evaluation Across 3,067 Standardized Benchmark Tasks | Lower is Better (0.0% = Flawless)",
             fontsize=9.8, fontfamily=FONT_FAMILY_SANS, color=COLOR_TEXT_MUTED, va="center")

    # Panel A: Degenerate Infinite Loop Rate (%)
    ax1 = fig.add_axes([0.07, 0.13, 0.45, 0.68])
    ax1.set_facecolor(COLOR_BG_CARD)
    ax1.grid(True, axis="y", linestyle=(0, (3, 3)), color=COLOR_GRID, linewidth=1.1, zorder=0)
    ax1.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax1.spines[spine].set_color(COLOR_SPINE)

    x1 = np.arange(len(categories))
    w1 = 0.26

    # Model loop bars
    bars1_afm = ax1.bar(x1 - w1, loops_afm, w1 * 0.90, label="Apple AFM-3 Core Advanced",
                        color=MODEL_COLORS["Apple AFM-3 Core Advanced"], alpha=0.95, edgecolor="none", zorder=3)
    bars1_gra = ax1.bar(x1,      loops_granite, w1 * 0.90, label="IBM Granite 4.2 3B",
                        color=MODEL_COLORS["IBM Granite 4.2 3B"], alpha=0.95, edgecolor="none", zorder=3)
    bars1_lfm = ax1.bar(x1 + w1, loops_lfm, w1 * 0.90, label="LiquidAI LFM 8B A1B",
                        color=MODEL_COLORS["LiquidAI LFM 8B A1B"], alpha=0.95, edgecolor="none", zorder=3)

    for bars in [bars1_afm, bars1_gra, bars1_lfm]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.1f}%",
                         xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8.2, fontweight="bold",
                         color=COLOR_TEXT_MAIN)

    ax1.set_xticks(x1)
    ax1.set_xticklabels(categories, fontsize=9.5, fontweight="600", color=COLOR_TEXT_MAIN)
    ax1.set_ylabel("Degenerate Runaway Loop Rate (%)", fontsize=10.5, color=COLOR_TEXT_MUTED, labelpad=10)
    ax1.set_ylim(0, 115)
    ax1.set_title("Panel A: Runaway Loop Rate by Domain & Overall (3,067 Tasks)",
                  fontsize=11.2, fontweight="bold", color=COLOR_TEXT_MAIN, pad=18)

    # Position horizontal legend cleanly above bars with no overlaps
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
               fontsize=8.6, prop={"weight": "600"})

    # Highlight Overall bar with subtle background tint
    ax1.axvspan(3.5, 4.5, color="#EDE2D5", alpha=0.45, zorder=1)
    ax1.text(4.0, 105, "Full Suite", ha="center", fontsize=8.5, fontweight="bold", color=COLOR_TEXT_MUTED)

    # Panel B: Lexical Repetition Dynamics (Code Generation)
    ax2 = fig.add_axes([0.58, 0.13, 0.38, 0.68])
    ax2.set_facecolor(COLOR_BG_CARD)
    ax2.grid(True, axis="y", linestyle=(0, (3, 3)), color=COLOR_GRID, linewidth=1.1, zorder=0)
    ax2.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax2.spines[spine].set_color(COLOR_SPINE)

    x2 = np.arange(len(rep_metrics))
    w2 = 0.26

    bars2_afm = ax2.bar(x2 - w2, lex_afm, w2 * 0.90, label="Apple AFM-3",
                        color=MODEL_COLORS["Apple AFM-3 Core Advanced"], alpha=0.95, edgecolor="none", zorder=3)
    bars2_gra = ax2.bar(x2,      lex_granite, w2 * 0.90, label="IBM Granite",
                        color=MODEL_COLORS["IBM Granite 4.2 3B"], alpha=0.95, edgecolor="none", zorder=3)
    bars2_lfm = ax2.bar(x2 + w2, lex_lfm, w2 * 0.90, label="LiquidAI LFM",
                        color=MODEL_COLORS["LiquidAI LFM 8B A1B"], alpha=0.95, edgecolor="none", zorder=3)

    for bars in [bars2_afm, bars2_gra, bars2_lfm]:
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.1f}%",
                         xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8.2, fontweight="bold",
                         color=COLOR_TEXT_MAIN)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(rep_metrics, fontsize=10.0, fontweight="600", color=COLOR_TEXT_MAIN)
    ax2.set_ylabel("Repetition Rate (%) — Lower is Better", fontsize=10.5, color=COLOR_TEXT_MUTED, labelpad=10)
    ax2.set_ylim(0, 42)
    ax2.set_title("Panel B: Multi-Granular Lexical Repetition (Code)",
                  fontsize=11.2, fontweight="bold", color=COLOR_TEXT_MAIN, pad=14)

    plt.savefig(OUTPUT_REPETITION_PNG, facecolor=COLOR_BG_CARD, edgecolor="none", dpi=300)
    plt.close()
    print(f"✓ Saved Repetition & Degenerate Loop Rate Figure: {OUTPUT_REPETITION_PNG}")


if __name__ == "__main__":
    generate_throughput_and_memory_figure()
    generate_repetition_rate_figure()
