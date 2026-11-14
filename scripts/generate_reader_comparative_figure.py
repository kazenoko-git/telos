#!/usr/bin/env python3
"""Generate comparative evaluation overview figure in the Télos research palette."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def load_comparative_data(root: Path, input_path: Path) -> dict:
    """Load benchmark data from combined file or fallback files."""
    if input_path.exists():
        return json.loads(input_path.read_text())
    fallback = root / "reports/xerj_reader_2026-11-14/comparative_512_v2_run.json"
    if fallback.exists():
        return json.loads(fallback.read_text())
    raise FileNotFoundError(f"Could not find comparative run data at {input_path}")


def get_model_display_name(model_id: str) -> str:
    """Return clean human-readable name for figure display."""
    names = {
        "xerj-reader-1": "Reader-1 (API + Native Offsets)",
        "ministral-3-8b-instruct-2512": "Ministral 3 8B (Q4 Local)",
        "google/gemma-4-e4b": "Gemma 4 E4B (Q4 Local)",
        "lfm3-omni-2b": "LFM3 Omni 2B (MLX Local)",
        "ternary-bonsai-4b-mlx": "Ternary Bonsai 4B (1.58b)",
        "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite (Incomplete)",
    }
    return names.get(model_id, model_id)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "reports/xerj_reader_2026-11-14/comparative_512_v2_run.json",
        help="Path to comparative run JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "figures/xerj_reader_comparative_2026-11-14.png",
        help="Path to output PNG figure",
    )
    args = parser.parse_args()

    data = load_comparative_data(root, args.input)
    model_runs = data.get("model_runs", {})

    # Palette definition conforming to Télos research styling
    ink, muted, page, card, border = "#2D2A26", "#5C554D", "#FDF9F5", "#F4EBE1", "#E8DFD5"
    rust, green, blue = "#9B5B3E", "#23765D", "#2563AB"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Outfit", "Avenir Next", "Helvetica Neue", "Arial"],
        "text.color": ink,
    })

    fig = plt.figure(figsize=(16, 9), dpi=220, facecolor=page)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.axis("off")

    # Background cards for visual hierarchy
    # Left main card: Accuracy overview for full 512-case models
    bg.add_patch(FancyBboxPatch((0.04, 0.09), 0.44, 0.77, boxstyle="round,pad=0.012,rounding_size=0.015",
                                facecolor=card, edgecolor=border))
    # Top-right card: Latency vs Accuracy trade-off
    bg.add_patch(FancyBboxPatch((0.51, 0.48), 0.45, 0.38, boxstyle="round,pad=0.012,rounding_size=0.015",
                                facecolor=card, edgecolor=border))
    # Bottom-right card: Verification & system diagnostics
    bg.add_patch(FancyBboxPatch((0.51, 0.09), 0.45, 0.36, boxstyle="round,pad=0.012,rounding_size=0.015",
                                facecolor=card, edgecolor=border))

    # Top banner with Télos logo
    logo_path = root / "figures/test_transparent_logo.png"
    if logo_path.exists():
        logo_ax = fig.add_axes([0.042, 0.885, 0.13, 0.08])
        logo_ax.imshow(plt.imread(logo_path))
        logo_ax.axis("off")

    fig.text(0.185, 0.925, "Reader-1 comparative evaluation (v2 audited)", fontsize=20, weight="medium")
    fig.text(0.185, 0.892, "Frozen 512-case suite · Disaggregated subset validation · Native vs reconstructed grounding",
             fontsize=10.5, color=muted)
    fig.text(0.96, 0.925, "14 NOV 2026", fontsize=9, ha="right", color=muted)
    fig.text(0.96, 0.895, "TÉLOS RESEARCH", fontsize=9, ha="right", color=muted, weight="medium")

    # Collect model metrics for models with full 512 evaluations
    completed_models = []
    

    for m_id, r in model_runs.items():
        summary = r.get("summary", {})
        c_all = summary.get("combined_512", {})
        c_240 = summary.get("inspected_240", {})
        c_272 = summary.get("new_272", {})
        tot = c_all.get("total_cases", 0)

        if m_id == "gemini-3.5-flash-lite":
            # Excluded from primary 512 comparison; tracked in dedicated note
            continue

        if tot == 512:
            completed_models.append({
                "id": m_id,
                "name": get_model_display_name(m_id),
                "acc": c_all.get("accuracy_pct", 0.0),
                "corr": c_all.get("correct_count", 0),
                "tot": tot,
                "acc_240": c_240.get("accuracy_pct", 0.0),
                "acc_272": c_272.get("accuracy_pct", 0.0),
                "lat_ms": c_all.get("latencies_ms", {}).get("median", 0.0),
                "valid_offset_pct": c_all.get("evidence", {}).get("valid_offsets_pct", 0.0),
            })

    # Sort descending by accuracy
    completed_models.sort(key=lambda x: x["acc"], reverse=True)

    # Subplot 1: Left Card - Horizontal Bar Chart (5 Completed Models)
    fig.text(0.065, 0.815, "Full benchmark accuracy (512 cases)", fontsize=14, weight="medium")
    fig.text(0.065, 0.785, "Combined status-and-value correctness across all 12 categories", fontsize=9.5, color=muted)

    ax_left = fig.add_axes([0.18, 0.22, 0.28, 0.54], facecolor=card)
    y_pos = list(range(len(completed_models)))
    labels = [m["name"] for m in completed_models]
    values = [m["acc"] for m in completed_models]
    bar_colors = [green if v >= 85 else rust if v >= 70 else "#6B7280" for v in values]

    ax_left.barh(y_pos, values, color=bar_colors, height=0.48, zorder=3)

    for i, m in enumerate(completed_models):
        ax_left.scatter(m["acc_240"], i - 0.12, marker="s", color=ink, s=28, zorder=4, alpha=0.85)
        ax_left.scatter(m["acc_272"], i + 0.12, marker="o", color=rust, s=28, zorder=4, alpha=0.85)
        ax_left.text(min(m["acc"] + 2.5, 102), i, f"{m['acc']:.1f}% ({m['corr']}/{m['tot']})",
                     va="center", fontsize=9.5, color=ink)

    ax_left.set_yticks(y_pos, labels, fontsize=10.5, color=ink)
    ax_left.invert_yaxis()
    ax_left.set_xlim(0, 118)
    ax_left.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"], fontsize=9, color=muted)
    ax_left.grid(axis="x", color="#DDD0C0", linestyle=(0, (3, 4)), zorder=0)
    ax_left.tick_params(axis="both", length=0, pad=8)
    for spine in ax_left.spines.values():
        spine.set_visible(False)

    # Legend for markers inside left card
    ax_left.scatter([], [], marker="s", color=ink, s=30, label="Inspected (240)")
    ax_left.scatter([], [], marker="o", color=rust, s=30, label="New (272)")
    leg = ax_left.legend(loc="lower right", frameon=True, facecolor=page, edgecolor=border, fontsize=8.5)
    leg.get_frame().set_linewidth(0.8)

    # Note on Gemini incomplete status inside left card
    note_text = "Note: Gemini 3.5 Flash-Lite evaluation is incomplete (107/512 completed; 25 quota errors; 380 unattempted due to daily cap; 106/107 correct)."
    fig.text(0.065, 0.12, note_text, fontsize=8.2, color=muted, style="italic")
    fig.text(0.065, 0.12, note_text, fontsize=8.2, color=muted, style="italic")
    # Subplot 2: Top-right Card - Latency vs Accuracy
    fig.text(0.535, 0.815, "Latency vs accuracy trade-off", fontsize=14, weight="medium")
    fig.text(0.535, 0.785, "Median model response latency (ms, log scale) vs 512 accuracy (%)",
             fontsize=9.5, color=muted)

    ax_tr = fig.add_axes([0.56, 0.54, 0.38, 0.22], facecolor=card)

    scatter_colors = {
        "xerj-reader-1": blue,
        "ministral-3-8b-instruct-2512": green,
        "google/gemma-4-e4b": "#D35400",
        "lfm3-omni-2b": "#16A085",
        "ternary-bonsai-4b-mlx": rust,
    }

    for m in completed_models:
        lat = max(m["lat_ms"], 50.0)
        c = scatter_colors.get(m["id"], ink)
        ax_tr.scatter(lat, m["acc"], s=100, color=c, zorder=4, edgecolor=ink, linewidth=0.8)
        dx = 15 if m["id"] != "xerj-reader-1" else -65
        dy = 2 if m["id"] not in ["google/gemma-4-e4b", "lfm3-omni-2b"] else -5
        ax_tr.annotate(
            m["name"].split(" (")[0],
            (lat, m["acc"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.5,
            color=ink,
            weight="medium",
        )

    ax_tr.set_xscale("log")
    ax_tr.set_xlim(180, 5000)
    ax_tr.set_ylim(55, 100)
    ax_tr.set_xticks([200, 500, 1000, 2000, 3000, 4000], ["200ms", "500ms", "1s", "2s", "3s", "4s"], fontsize=8.5, color=muted)
    ax_tr.set_yticks([60, 70, 80, 90, 100], ["60%", "70%", "80%", "90%", "100%"], fontsize=8.5, color=muted)
    ax_tr.grid(True, color="#DDD0C0", linestyle=(0, (3, 4)), zorder=0)
    ax_tr.tick_params(axis="both", length=0, pad=6)
    for spine in ax_tr.spines.values():
        spine.set_visible(False)

    # Subplot 3: Bottom-right Card - Key Diagnostics Callout (Updated with exact v2 audited values)
    fig.text(0.535, 0.405, "Audited verification & execution findings", fontsize=14, weight="medium")

    diagnostics = [
        ("Ministral 3 8B Instruct", "91.0%", "Highest overall accuracy (466/512)\n100% on tables, conflicts, lists, missing, corrections"),
        ("Google Gemma 4 E4B", "84.2%", "Balanced local 4B model (431/512)\n95% arithmetic, 100% tables, lists, corrections"),
        ("LiquidAI LFM3 Omni 2B", "579 ms", "Fastest resident local MLX runtime (430/512)\nFixed list parsing; 100% tables, lists, missing"),
        ("Ternary Bonsai 4B", "80.5%", "1.58-bit ternary runtime (412/512)\nFixed money schema; 1,107 ms latency, 95% tables"),
        ("Reader-1 Baseline", "100%", "128/128 sampled executor replays match\nNative character offsets; 293 ms model latency"),
        ("Evidence Grounding", "Provenance", "Reader-1: Native model character coordinates\nGeneral models: Reconstructed literal quote search"),
    ]

    card_x = [0.535, 0.685, 0.835]
    card_y = [0.265, 0.125]

    for idx, (title, stat, desc) in enumerate(diagnostics):
        col = idx % 3
        row = idx // 3
        px = card_x[col]
        py = card_y[row]
        fig.text(px, py + 0.08, title, fontsize=9.5, weight="medium", color=ink)
        fig.text(px, py + 0.035, stat, fontsize=17, weight="medium",
                 color=green if "91" in stat or "100" in stat else blue if "579" in stat else rust)
        fig.text(px, py, desc, fontsize=7.8, color=muted, va="top", linespacing=1.25)

    # Footer note
    fig.text(0.045, 0.045, "Controlled comparative diagnostics across 12 structured document extraction categories. Not an estimate of general web accuracy.",
             fontsize=9, color=muted)
    fig.text(0.96, 0.045, "Source: comparative_512_v2_run.json", fontsize=9, ha="right", color=muted)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=page, dpi=220)
    plt.close(fig)
    print(f"Saved comparative evaluation figure to {args.output}")


if __name__ == "__main__":
    main()
