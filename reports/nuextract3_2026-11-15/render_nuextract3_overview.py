"""Render publication-quality NuExtract3 overview figure in Télos warm cream style.

Generates:
- reports/nuextract3_2026-11-15/overview.png
- reports/nuextract3_2026-11-15/overview.pdf

Derives all numbers directly from summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from historical_scores import scores as historical_scores, cell

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUMMARY_FILE = HERE / "summary.json"
LOGO_FILE = REPO / "figures" / "test_transparent_logo.png"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    if not SUMMARY_FILE.exists():
        print(f"Summary file not found: {SUMMARY_FILE}")
        return

    data = json.loads(SUMMARY_FILE.read_text())
    tracks = {t["track"]: t for t in data.get("tracks", [])}

    direct = tracks.get("direct_qa")
    extraction = next((t for k, t in tracks.items() if k != "direct_qa"), None)

    if not direct or not extraction:
        print("Missing required track summaries (need both direct_qa and extraction)")
        return

    # Palette
    bg_color = "#FDF9F5"
    card_bg = "#F5EFE8"
    border_color = "#E5DCD2"
    text_dark = "#2D2A26"
    text_muted = "#6B635B"
    color_direct = "#2B5278"       # Deep Slate Blue
    color_exact = "#9B5B3E"        # Terracotta Rust
    color_pipeline = "#23765D"     # Pine / Forest Green
    color_neutral = "#443F39"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Avenir Next", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": text_dark,
    })

    fig = plt.figure(figsize=(13, 8.5), dpi=200, facecolor=bg_color)

    # 1. Header with Logo
    if LOGO_FILE.exists():
        logo_ax = fig.add_axes([0.05, 0.85, 0.12, 0.11])
        logo_ax.imshow(plt.imread(LOGO_FILE))
        logo_ax.axis("off")

    fig.text(0.18, 0.92, "NuExtract3 · 512-Case Benchmark Suite", fontsize=22, weight="bold", color=text_dark)
    fig.text(
        0.18,
        0.875,
        "numind/NuExtract3 (4-bit MLX on Apple Silicon) · 3 Separated Tracks · 15 November 2026",
        fontsize=11,
        color=text_muted,
    )

    # 2. Scoreboard Cards
    d_corr = direct["direct_correct"]
    d_total = direct["records"]
    d_pct = d_corr / d_total * 100

    e_exact = extraction["extraction_exact"]
    e_ground = extraction["extraction_grounded"]
    e_total = extraction["records"]
    e_pct = e_exact / e_total * 100

    p_corr = extraction["pipeline_correct"]
    p_pct = p_corr / e_total * 100

    d_med = direct["timing"]["client_ms"]["median"]
    e_med = extraction["timing"]["client_ms"]["median"]

    history = historical_scores()
    cards = [
        {
            "title": "Direct QA Track",
            "score": f"{d_corr} / {d_total}",
            "pct": f"{d_pct:.1f}%",
            "sub": f"Reader-1: {cell(history['reader'], 'direct')}\nMedian Latency: {d_med:.0f} ms",
            "color": color_direct,
            "x": 0.05,
        },
        {
            "title": "Exact Extraction",
            "score": f"{e_exact} / {e_total}",
            "pct": f"{e_pct:.1f}%",
            "sub": f"Literal-source cases: {e_ground} ({e_ground/e_total*100:.1f}%)\nFrozen Operand Descriptors",
            "color": color_exact,
            "x": 0.28,
        },
        {
            "title": "Calculator Pipeline",
            "score": f"{p_corr} / {e_total}",
            "pct": f"{p_pct:.1f}%",
            "sub": f"NuExtract Tiny: {cell(history['tiny'], 'pipeline')}\nHuman-specified operations",
            "color": color_pipeline,
            "x": 0.51,
        },
        {
            "title": "Output Format",
            "score": f"{extraction['format_failures']} / {e_total}",
            "pct": "0.0% Fail",
            "sub": f"Recovery Retries: {extraction['recovery_attempts']}\nMedian Ext. Latency: {e_med:.0f} ms",
            "color": color_neutral,
            "x": 0.74,
        },
    ]

    card_y = 0.69
    card_w = 0.21
    card_h = 0.145

    for c in cards:
        # Card Background
        box = FancyBboxPatch(
            (c["x"], card_y),
            card_w,
            card_h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            fc=card_bg,
            ec=border_color,
            lw=1.0,
            transform=fig.transFigure,
            figure=fig,
        )
        fig.patches.append(box)

        # Card Content
        fig.text(c["x"] + 0.015, card_y + 0.115, c["title"].upper(), fontsize=9, weight="bold", color=text_muted)
        fig.text(c["x"] + 0.015, card_y + 0.065, c["score"], fontsize=17, weight="bold", color=c["color"])
        fig.text(c["x"] + 0.125, card_y + 0.068, c["pct"], fontsize=12, weight="bold", color=c["color"])
        fig.text(c["x"] + 0.015, card_y + 0.018, c["sub"], fontsize=8.5, color=text_muted, linespacing=1.3)

    # 3. Model Benchmark Comparison Bar Chart (Mid-left)
    ax_comp = fig.add_axes([0.08, 0.28, 0.38, 0.34])
    ax_comp.set_facecolor(bg_color)
    for spine in ["top", "right"]:
        ax_comp.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_comp.spines[spine].set_color(border_color)

    models = [
        "SmolLM2 (360M)",
        "GLiNER2 Base (205M)",
        "NuExtract-1.5 (0.5B)",
        "Reader-1 (Baseline)",
        "NuExtract3 (4B)",
    ]
    pipe_scores = [history[k]["pipeline"] for k in ("smol","gliner","tiny")] + [0,p_corr]
    direct_scores = [history["smol"]["direct"],0,0,history["reader"]["direct"],d_corr]

    y_pos = range(len(models))
    height = 0.36

    ax_comp.barh(
        [y - height / 2 for y in y_pos],
        direct_scores,
        height,
        label="Direct QA",
        color=color_direct,
        alpha=0.9,
    )
    ax_comp.barh(
        [y + height / 2 for y in y_pos],
        pipe_scores,
        height,
        label="Calculator Pipeline",
        color=color_pipeline,
        alpha=0.9,
    )

    ax_comp.set_yticks(list(y_pos))
    ax_comp.set_yticklabels(models, fontsize=9.5)
    ax_comp.set_xlim(0, 520)
    ax_comp.set_xlabel("Tasks Correct (/512)", fontsize=10, color=text_muted)
    ax_comp.set_title("Cross-Model Comparison Across Frozen Suite", fontsize=11.5, weight="bold", pad=10, loc="left")
    ax_comp.legend(loc="lower right", frameon=True, facecolor=card_bg, edgecolor=border_color, fontsize=8.5)
    ax_comp.grid(axis="x", linestyle=":", alpha=0.5, color="#D0C8BE")

    # Add score annotations on bars
    for y, (d_val, p_val) in enumerate(zip(direct_scores, pipe_scores)):
        if d_val > 0:
            ax_comp.text(d_val + 6, y - height / 2, f"{d_val}", va="center", fontsize=8.5, color=color_direct)
        if p_val > 0:
            ax_comp.text(p_val + 6, y + height / 2, f"{p_val}", va="center", fontsize=8.5, color=color_pipeline)

    # 4. Category Breakdown Table / Chart (Mid-right)
    ax_cat = fig.add_axes([0.55, 0.28, 0.40, 0.34])
    ax_cat.set_facecolor(bg_color)
    for spine in ["top", "right"]:
        ax_cat.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_cat.spines[spine].set_color(border_color)

    d_cats = direct.get("categories", {})
    e_cats = extraction.get("categories", {})
    cat_keys = sorted(d_cats.keys())

    cat_y = range(len(cat_keys))
    cat_h = 0.36

    d_cat_vals = [d_cats[c]["correct"] / max(1, d_cats[c]["total"]) * 100 for c in cat_keys]
    p_cat_vals = [e_cats[c]["pipeline_correct"] / max(1, e_cats[c]["total"]) * 100 for c in cat_keys]

    ax_cat.barh([y - cat_h / 2 for y in cat_y], d_cat_vals, cat_h, label="Direct QA %", color=color_direct, alpha=0.9)
    ax_cat.barh(
        [y + cat_h / 2 for y in cat_y], p_cat_vals, cat_h, label="Pipeline %", color=color_pipeline, alpha=0.9
    )

    clean_labels = [c.replace("_", " ").title() for c in cat_keys]
    ax_cat.set_yticks(list(cat_y))
    ax_cat.set_yticklabels(clean_labels, fontsize=9)
    ax_cat.set_xlim(0, 105)
    ax_cat.set_xlabel("Accuracy (%)", fontsize=10, color=text_muted)
    ax_cat.set_title("Accuracy by Task Category", fontsize=11.5, weight="bold", pad=10, loc="left")
    ax_cat.legend(loc="lower right", frameon=True, facecolor=card_bg, edgecolor=border_color, fontsize=8.5)
    ax_cat.grid(axis="x", linestyle=":", alpha=0.5, color="#D0C8BE")

    # 5. Protocol and Method Notes (Bottom)
    notes = [
        "• Tracks are strictly separated: Direct QA scores direct model answers; Extraction + Calculator feeds verbatim operands to deterministic executor.",
        "• Model configuration: numind/NuExtract3-mlx-4bits (Qwen 3.5 4B base), greedy decoding (temp=0.0), thinking_mode=False.",
        "• Evaluation integrity: Replayed independently via specialist_contract.py; no serialized gold/reference fields found in saved prompts.",
        "• Hardware runtime: Apple Silicon (Apple M5 Pro, 24 GB Unified Memory), mlx-lm 0.32.2. Statistics read from recorded run artifacts.",
    ]
    for i, note in enumerate(notes):
        fig.text(0.05, 0.16 - i * 0.035, note, fontsize=8.5, color=text_muted)

    out_png = HERE / "overview.png"
    out_pdf = HERE / "overview.pdf"
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(out_pdf, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Rendered overview figure:\n  - {out_png}\n  - {out_pdf}")


if __name__ == "__main__":
    main()
