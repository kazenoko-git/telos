import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# 1. 50M AR CUDA Data (0.2B to 2.4B tokens)
t_cuda = np.array([0.1966, 0.3932, 0.5898, 0.7864, 0.9830, 1.1796, 1.3763, 1.5729, 1.7695, 1.9661, 2.1627, 2.3593])
ce_cuda = np.array([2.4100, 2.0217, 1.8986, 1.8166, 1.7855, 1.7100, 1.6823, 1.6543, 1.6357, 1.6185, 1.6069, 1.5969])

# 2. 50M AR TPU Data (0.2B to 2.0B tokens)
t_tpu = np.array([0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00])
ce_tpu = np.array([5.6295, 6.0942, 6.1363, 6.2927, 6.5777, 6.2917, 6.4376, 6.4197, 6.4641, 6.5033])

# Log-linear fit: L(T) = a - b * ln(T)
p_cuda = np.polyfit(np.log(t_cuda), ce_cuda, 1)  # p[0] = -b, p[1] = a
p_tpu = np.polyfit(np.log(t_tpu), ce_tpu, 1)

a_cuda, b_cuda = p_cuda[1], -p_cuda[0]
a_tpu, b_tpu = p_tpu[1], -p_tpu[0]  # note: b_tpu will be negative because slope is positive

print(f"50M CUDA Log-Linear: L(T) = {a_cuda:.4f} - {b_cuda:.4f} * ln(T)")
print(f"50M TPU  Log-Linear: L(T) = {a_tpu:.4f} - ({b_tpu:.4f}) * ln(T) = {a_tpu:.4f} + {-b_tpu:.4f} * ln(T)")

# Power-law fit for CUDA: L(T) = A * T^(-alpha) + L_inf
best_mse = float('inf')
best_params = None
for L_inf in np.linspace(1.2, 1.5, 301):
    y = np.log(ce_cuda - L_inf)
    p = np.polyfit(np.log(t_cuda), y, 1)
    alpha = -p[0]
    A = np.exp(p[1])
    pred = A * (t_cuda ** -alpha) + L_inf
    mse = np.mean((ce_cuda - pred) ** 2)
    if mse < best_mse:
        best_mse = mse
        best_params = (A, alpha, L_inf)

A_cuda, alpha_cuda, L_inf_cuda = best_params
print(f"50M CUDA Power-Law: L(T) = {A_cuda:.4f} * T^(-{alpha_cuda:.4f}) + {L_inf_cuda:.4f}")

# Domain for extrapolation
T_plot = np.linspace(0.15, 10.0, 500)
ce_cuda_log = a_cuda - b_cuda * np.log(T_plot)
ce_tpu_log = a_tpu - b_tpu * np.log(T_plot)
ce_cuda_pow = A_cuda * (T_plot ** -alpha_cuda) + L_inf_cuda

# TPU flat/plateau baseline for comparison
L_tpu_plateau = np.mean(ce_tpu[-5:])

# Plot setup
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
fig.suptitle('Hardware Scaling Comparison & Extrapolation: 50M AR TPU (2.0B) vs. 50M AR CUDA (2.5B)', fontsize=15, fontweight='bold', y=0.98)

# ----------------- PANEL 1: Log-Linear Model -----------------
ax1.scatter(t_cuda, ce_cuda, color='#1f77b4', s=55, zorder=5, label='50M AR (CUDA H100, Observed 0.2-2.4B)')
ax1.scatter(t_tpu, ce_tpu, color='#d62728', s=55, marker='^', zorder=5, label='50M AR (TPU XLA, Observed 0.2-2.0B)')

ax1.plot(T_plot, ce_cuda_log, color='#1f77b4', linewidth=2.5, label=f'50M CUDA Fit: {a_cuda:.3f} - {b_cuda:.3f} ln(T)')
ax1.plot(T_plot, ce_tpu_log, color='#d62728', linestyle='--', linewidth=2.5, label=f'50M TPU Fit: {a_tpu:.3f} + {-b_tpu:.3f} ln(T) [Divergent]')

# Final observed points
ax1.scatter([2.36], [1.5969], color='#1f77b4', s=120, edgecolors='black', linewidth=1.5, zorder=6)
ax1.scatter([2.00], [6.5171], color='#d62728', s=120, marker='^', edgecolors='black', linewidth=1.5, zorder=6)

ax1.annotate('CUDA Final (2.4B tok)\nCE = 1.597 nats (PPL 4.94)',
             xy=(2.36, 1.5969), xytext=(2.6, 2.5),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
             fontweight='bold', color='#1f77b4', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', edgecolor='#1f77b4', alpha=0.9))

ax1.annotate('TPU Final (2.0B tok)\nCE = 6.517 nats (PPL 676.6)\n[Step 1017: 5.630 nats]',
             xy=(2.00, 6.5171), xytext=(2.3, 5.5),
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
             fontweight='bold', color='#d62728', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#d62728', alpha=0.9))

ax1.annotate('Divergence Diagnosis:\n• TPU ran at effective batch size 48 (single core)\n• Gradient noise scale caused divergence at step 2000+\n• TPU slope is positive (+0.35 nats/log-tok)',
             xy=(5.0, 7.0), xytext=(4.2, 3.8),
             fontweight='bold', color='#333333', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#888888', alpha=0.95))

ax1.set_xlabel('Tokens Trained (Billions)', fontweight='bold')
ax1.set_ylabel('Causal Validation Cross-Entropy (nats) ↓', fontweight='bold')
ax1.set_title('Log-Linear Extrapolation: L(T) = a - b · ln(T)', fontweight='bold')
ax1.set_xlim(0, 10.2)
ax1.set_ylim(1.0, 8.0)
ax1.legend(loc='upper right', frameon=True, fontsize=9.0)

# ----------------- PANEL 2: Power-Law vs Asymptote -----------------
ax2.scatter(t_cuda, ce_cuda, color='#1f77b4', s=55, zorder=5, label='50M AR (CUDA, Observed)')
ax2.scatter(t_tpu, ce_tpu, color='#d62728', s=55, marker='^', zorder=5, label='50M AR (TPU, Observed)')

ax2.plot(T_plot, ce_cuda_pow, color='#1f77b4', linewidth=2.5, label=f'50M CUDA Power-Law: 0.383 · T^(-0.612) + {L_inf_cuda:.3f}')
ax2.axhline(L_inf_cuda, color='#1f77b4', linestyle=':', linewidth=1.5, label=f'CUDA Asymptotic Floor (L_inf = {L_inf_cuda:.3f} nats)')

ax2.axhline(L_tpu_plateau, color='#d62728', linestyle=':', linewidth=1.5, label=f'TPU Stagnation Level (~{L_tpu_plateau:.2f} nats)')
ax2.plot(T_plot, ce_tpu_log, color='#d62728', linestyle='--', linewidth=2.0, alpha=0.7, label='TPU Log-Linear Drift')

ax2.scatter([2.36], [1.5969], color='#1f77b4', s=120, edgecolors='black', linewidth=1.5, zorder=6)
ax2.scatter([2.00], [6.5171], color='#d62728', s=120, marker='^', edgecolors='black', linewidth=1.5, zorder=6)

ax2.annotate(f'CUDA Ceiling Advantage:\nLoss Gap: {6.5171 - 1.5969:.2f} nats\nPerplexity Gap: 676.6 vs. 4.94 (137x lower!)',
             xy=(2.2, 4.0), xytext=(3.5, 4.0),
             arrowprops=dict(arrowstyle='<->', color='#2ca02c', lw=2.0),
             fontweight='bold', color='#2ca02c', fontsize=10.0,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f5e9', edgecolor='#2ca02c', alpha=0.95))

ax2.set_xlabel('Tokens Trained (Billions)', fontweight='bold')
ax2.set_ylabel('Causal Validation Cross-Entropy (nats) ↓', fontweight='bold')
ax2.set_title('Power-Law Model & Performance Ceiling Comparison', fontweight='bold')
ax2.set_xlim(0, 10.2)
ax2.set_ylim(1.0, 8.0)
ax2.legend(loc='upper right', frameon=True, fontsize=9.0)

plt.tight_layout()

figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)
out_fig = figures_dir / 'ar_50m_tpu_vs_cuda_extrapolation.png'
plt.savefig(out_fig, bbox_inches='tight')

artifact_dir = Path('/Users/ivansamuel/.gemini/antigravity-ide/brain/c498704a-5904-4bfd-93eb-e8089c486ea4')
out_art = artifact_dir / 'ar_50m_tpu_vs_cuda_extrapolation.png'
plt.savefig(out_art, bbox_inches='tight')

plt.close()
print("✓ Saved extrapolation comparison plots to:")
print("  -", out_fig)
print("  -", out_art)
