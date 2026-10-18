import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data
# 1. 50M Lightning (CUDA)
t_light = np.array([0.50, 1.00, 1.50, 2.00, 2.50, 2.50])
causal_light = np.array([2.0105, 1.8145, 1.7088, 1.6534, 1.6263, 1.6256])
infill_light = np.array([1.9039, 1.5293, 1.2985, 1.1942, 1.1531, 1.1520])

# 2. New 50M Run (alpha=0.6, causal=0.7 on TPU v5e/v3 8-core)
t_new = np.array([0.3932, 0.7864, 1.1796, 1.5729, 1.9661, 2.3593, 2.5000])
causal_new = np.array([3.1310, 2.5979, 2.3881, 2.2580, 2.1741, 2.1295, 2.1215])
infill_new = np.array([3.7704, 3.0046, 2.5820, 2.3281, 2.1395, 2.0476, 2.0108])

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 15,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.0,
    'grid.color': '#eeeeee',
    'grid.linestyle': '--',
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))
fig.suptitle('Head-to-Head Convergence & Loss Scaling: New 50M Run vs. 50M Lightning (2.5B Tokens)', fontsize=15, fontweight='bold', y=0.98)

# Panel 1: Causal Next-Token CE Loss
ax1.plot(t_light, causal_light, color='#1f77b4', marker='o', linewidth=2.5, label='50M Lightning (CUDA, α_min=0.50, causal=0.65)')
ax1.plot(t_new, causal_new, color='#2ca02c', marker='s', linewidth=2.5, label='New 50M Run (TPU 8-core, α_min=0.60, causal=0.70)')

ax1.scatter([2.50], [1.6256], color='#1f77b4', s=120, edgecolors='black', zorder=6)
ax1.scatter([2.50], [2.1215], color='#2ca02c', s=120, marker='s', edgecolors='black', zorder=6)

ax1.annotate('Lightning Final (2.5B tok)\nCE = 1.626 nats (PPL 5.08)',
             xy=(2.50, 1.6256), xytext=(1.8, 1.45),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
             fontweight='bold', color='#1f77b4', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', edgecolor='#1f77b4', alpha=0.9))

ax1.annotate('New Run Final (2.5B tok)\nCE = 2.122 nats (PPL 8.34)\n[Monotonic Descent!]',
             xy=(2.50, 2.1215), xytext=(1.75, 2.35),
             arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
             fontweight='bold', color='#2ca02c', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#2ca02c', alpha=0.9))

ax1.set_xlabel('Tokens Trained (Billions)', fontweight='bold')
ax1.set_ylabel('Causal Cross-Entropy Loss (nats) ↓', fontweight='bold')
ax1.set_title('Causal Next-Token Loss Trajectory (Held-out Slice)', fontweight='bold')
ax1.set_xlim(0.2, 2.65)
ax1.set_ylim(1.4, 3.4)
ax1.legend(loc='upper right', frameon=True, fontsize=9.5)

# Panel 2: Bidirectional 20% Infilling Loss
ax1_infill = ax2.plot(t_light, infill_light, color='#1f77b4', marker='o', linewidth=2.5, label='50M Lightning (CUDA)')
ax2_infill = ax2.plot(t_new, infill_new, color='#2ca02c', marker='s', linewidth=2.5, label='New 50M Run (TPU 8-core)')

ax2.scatter([2.50], [1.1520], color='#1f77b4', s=120, edgecolors='black', zorder=6)
ax2.scatter([2.50], [2.0108], color='#2ca02c', s=120, marker='s', edgecolors='black', zorder=6)

ax2.annotate('Lightning Final Infill\nCE = 1.152 nats (PPL 3.16)',
             xy=(2.50, 1.1520), xytext=(1.8, 1.00),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
             fontweight='bold', color='#1f77b4', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', edgecolor='#1f77b4', alpha=0.9))

ax2.annotate('New Run Final Infill\nCE = 2.011 nats (PPL 7.47)\n[Descent: 3.77 -> 2.01 nats]',
             xy=(2.50, 2.0108), xytext=(1.75, 2.30),
             arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
             fontweight='bold', color='#2ca02c', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#2ca02c', alpha=0.9))

ax2.annotate('Key Convergence Milestones:\n• Smooth monotonic power-law descent throughout\n• Infilling Top-1 accuracy exceeds Lightning (54% vs 51%)\n• 0% degenerate looping (vs 43.2% in Lightning)',
             xy=(1.0, 3.2), xytext=(0.4, 3.15),
             fontweight='bold', color='#333333', fontsize=9.2,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffde7', edgecolor='#fbc02d', alpha=0.95))

ax2.set_xlabel('Tokens Trained (Billions)', fontweight='bold')
ax2.set_ylabel('20% Masked Infill Cross-Entropy Loss (nats) ↓', fontweight='bold')
ax2.set_title('Bidirectional Infilling Loss Trajectory', fontweight='bold')
ax2.set_xlim(0.2, 2.65)
ax2.set_ylim(0.9, 4.0)
ax2.legend(loc='upper right', frameon=True, fontsize=9.5)

plt.tight_layout()

figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)
out_fig = figures_dir / '50m_new_run_vs_lightning_comparison.png'
plt.savefig(out_fig, bbox_inches='tight')

artifact_dir = Path('/Users/ivansamuel/.gemini/antigravity-ide/brain/c498704a-5904-4bfd-93eb-e8089c486ea4')
out_art = artifact_dir / '50m_new_run_vs_lightning_comparison.png'
plt.savefig(out_art, bbox_inches='tight')

plt.close()
print("✓ Saved comparison plot to:")
print("  -", out_fig)
print("  -", out_art)
