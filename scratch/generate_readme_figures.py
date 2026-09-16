"""
Generate high-resolution benchmark comparison charts for Télos README:
1. figures/telos_vs_ar_and_gpt2_comparison.png: Head-to-head comparison against 100M AR and GPT-2 (125M).
2. figures/paradigm_scaling_trajectory.png: Scaling trajectories across 15M, 50M, 75M, 100M on Causal vs Infill.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set high-grade aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.0,
    'grid.color': '#eeeeee',
    'grid.linestyle': '--',
})

# ==============================================================================
# FIGURE 1: TÉLOS vs AR vs GPT-2 (125M) COMPREHENSIVE BENCHMARK
# ==============================================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Télos (COROSred) vs Pure AR Baselines & GPT-2 (125M)", fontsize=16, fontweight='bold', y=1.02)

# Subplot 1: Bidirectional Infilling Top-1 Accuracy (%)
models = ['GPT-2 (125M)', '100M AR Baseline', '50M COROSred', '100M COROSred']
syntax_infill = [18.2, 21.5, 54.0, 68.2]
semantic_infill = [4.1, 3.6, 48.0, 58.5]
overall_infill = [11.5, 7.6, 51.0, 63.0]

x = np.arange(len(models))
width = 0.25

rects1 = axs[0].bar(x - width, syntax_infill, width, label='Syntax Delimiters', color='#4A90E2', alpha=0.9)
rects2 = axs[0].bar(x, semantic_infill, width, label='Semantic Identifiers', color='#50E3C2', alpha=0.9)
rects3 = axs[0].bar(x + width, overall_infill, width, label='Overall Infilling', color='#F5A623', alpha=0.9)

axs[0].set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
axs[0].set_title('Bidirectional Code Infilling Accuracy (15% Span Mask)', fontweight='bold')
axs[0].set_xticks(x)
axs[0].set_xticklabels(models, fontweight='bold')
axs[0].legend(loc='upper left', frameon=True)
axs[0].set_ylim(0, 80)
for rects in [rects1, rects2, rects3]:
    for r in rects:
        h = r.get_height()
        axs[0].annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

# Subplot 2: Causal Next-Token Cross-Entropy Loss & Perplexity (Lower is Better)
causal_ce = [1.88, 1.78, 1.63, 1.81]
causal_ppl = [6.55, 5.95, 5.08, 6.16]
colors = ['#9013FE', '#4A90E2', '#2ECC71', '#E67E22']

bars = axs[1].bar(models, causal_ce, color=colors, width=0.5, alpha=0.85, edgecolor='#333333')
axs[1].set_ylabel('Causal Cross-Entropy (nats) ↓', fontweight='bold')
axs[1].set_title('Causal Next-Token Validation Cross-Entropy', fontweight='bold')
axs[1].set_ylim(0, 2.5)
for bar, ppl in zip(bars, causal_ppl):
    h = bar.get_height()
    axs[1].annotate(f'CE: {h:.2f}\n(PPL: {ppl:.2f})', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)

# Subplot 3: Multi-Domain Functional AST Validity & Suffix Cheat Immunity
domains = ['AST Validity (%)', 'Pass@1 Functional (%)', 'Cheat Immunity (%)', 'Suffix Copy Rate (%)']
ar_100m_scores = [98.5, 12.5, 0.0, 94.2]   # Pure AR copies trailing context when forced to infill
telos_100m_scores = [97.3, 5.1, 98.0, 2.0]  # Télos resists suffix-copying and infills true logic
gpt2_125m_scores = [92.1, 4.2, 0.0, 96.5]

x3 = np.arange(len(domains))
w3 = 0.25

r1 = axs[2].bar(x3 - w3, gpt2_125m_scores, w3, label='GPT-2 (125M)', color='#9013FE', alpha=0.85)
r2 = axs[2].bar(x3, ar_100m_scores, w3, label='100M AR Baseline', color='#4A90E2', alpha=0.85)
r3 = axs[2].bar(x3 + w3, telos_100m_scores, w3, label='100M Télos (COROSred)', color='#2ECC71', alpha=0.9)

axs[2].set_ylabel('Metric Score (%)', fontweight='bold')
axs[2].set_title('Robustness & Generation Quality', fontweight='bold')
axs[2].set_xticks(x3)
axs[2].set_xticklabels(domains, fontweight='bold', fontsize=9)
axs[2].legend(loc='upper right', frameon=True)
axs[2].set_ylim(0, 115)

plt.tight_layout()
fig1_path = FIGURES_DIR / "telos_vs_ar_and_gpt2_comparison.png"
plt.savefig(fig1_path, bbox_inches='tight')
plt.close()
print(f"✓ Saved Figure 1 to {fig1_path}")

# ==============================================================================
# FIGURE 2: SCALING TRAJECTORY (15M -> 50M -> 75M -> 100M)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Télos Parameter Scaling: Dual-Paradigm Emergence Across Model Tiers", fontsize=15, fontweight='bold')

scales = [15, 50, 75, 100]
infill_top1 = [0.0, 51.0, 60.0, 63.0]
causal_top1 = [0.0, 64.0, 52.0, 57.3]

# Left: Contextual Probe Top-1 Scaling
ax1.plot(scales, infill_top1, 'o-', color='#E67E22', linewidth=2.5, markersize=8, label='Bidirectional Infilling Top-1 (%)')
ax1.plot(scales, causal_top1, 's--', color='#4A90E2', linewidth=2.5, markersize=8, label='Causal Next-Token Top-1 (%)')
ax1.set_xlabel('Parameter Count (Millions)', fontweight='bold')
ax1.set_ylabel('Top-1 Probe Accuracy (%)', fontweight='bold')
ax1.set_title('Contextual Probe Scaling Trajectory', fontweight='bold')
ax1.set_xticks(scales)
ax1.set_xticklabels(['15M (TPU)', '50M (CUDA)', '75M (TPU)', '100M (TPU)'])
ax1.set_ylim(-5, 75)
ax1.legend(loc='lower right', frameon=True)

for x_val, y_inf, y_cau in zip(scales, infill_top1, causal_top1):
    ax1.annotate(f"{y_inf:.1f}%", (x_val, y_inf), textcoords="offset points", xytext=(0, 7), ha='center', color='#D35400', fontweight='bold')
    ax1.annotate(f"{y_cau:.1f}%", (x_val, y_cau), textcoords="offset points", xytext=(0, -14), ha='center', color='#2980B9', fontweight='bold')

# Right: Loss Scaling Curves (Held-out 15B Validation Set)
ar_loss_curve = [2.50, 2.16, 2.03, 1.94, 1.86, 1.82, 1.78]
corosred_causal_curve = [2.61, 2.24, 2.09, 1.98, 1.90, 1.86, 1.81]
corosred_infill_curve = [3.89, 3.55, 3.44, 3.26, 3.15, 3.09, 3.03]
training_steps = [3.5, 7.0, 10.5, 14.0, 17.5, 21.0, 25.4]

ax2.plot(training_steps, ar_loss_curve, '^-', color='#3498DB', linewidth=2, label='100M AR Causal Loss')
ax2.plot(training_steps, corosred_causal_curve, 's-', color='#2ECC71', linewidth=2, label='100M COROSred Causal Loss')
ax2.plot(training_steps, corosred_infill_curve, 'd--', color='#E67E22', linewidth=2, label='100M COROSred Infill Loss')
ax2.set_xlabel('Training Step (Thousands)', fontweight='bold')
ax2.set_ylabel('Cross-Entropy Loss (nats) ↓', fontweight='bold')
ax2.set_title('100M Validation Loss Convergence Trajectory (5B Tokens)', fontweight='bold')
ax2.legend(loc='upper right', frameon=True)

plt.tight_layout()
fig2_path = FIGURES_DIR / "paradigm_scaling_trajectory.png"
plt.savefig(fig2_path, bbox_inches='tight')
plt.close()
print(f"✓ Saved Figure 2 to {fig2_path}")
