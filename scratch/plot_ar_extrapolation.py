import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Load data
with open('scratch/ar_models_1to50_loss_curves.json') as f:
    data = json.load(f)

res_50m = data['50M_AR_CUDA']
res_100m = data['100M_AR_TPU']

t_50 = np.array([r['tokens_b'] for r in res_50m])
ce_50 = np.array([r['causal_ce'] for r in res_50m])
ppl_50 = np.array([r['ppl'] for r in res_50m])

t_100 = np.array([r['tokens_b'] for r in res_100m])
ce_100 = np.array([r['causal_ce'] for r in res_100m])
ppl_100 = np.array([r['ppl'] for r in res_100m])

# Fitted parameters (Power law with asymptote: A * T^(-alpha) + L_inf)
A_50, a_50, L_inf_50 = 0.3738, 0.6211, 1.3772
A_100, a_100, L_inf_100 = 0.9845, 0.4795, 1.3248

def fit_50m(T):
    return A_50 * (T ** -a_50) + L_inf_50

def fit_100m(T):
    return A_100 * (T ** -a_100) + L_inf_100

# Extrapolation domain
T_ext = np.linspace(0.2, 25.0, 500)
ce_50_ext = fit_50m(T_ext)
ce_100_ext = fit_100m(T_ext)

ppl_50_ext = np.exp(ce_50_ext)
ppl_100_ext = np.exp(ce_100_ext)

# Plotting
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))
fig.suptitle('Mathematical Extrapolation of 50M AR (CUDA) vs 100M AR (TPU) Scaling Trajectories', fontsize=15, fontweight='bold', y=0.98)

# Panel 1: Cross-Entropy Loss Extrapolation
ax1.scatter(t_50, ce_50, color='#1f77b4', s=45, zorder=5, label='50M AR (CUDA, Observed 0.2-2.4B)')
ax1.scatter(t_100, ce_100, color='#ff7f0e', s=55, marker='s', zorder=5, label='100M AR (TPU, Observed 0.7-5.0B)')

# Extrapolated curves
ax1.plot(T_ext, ce_50_ext, color='#1f77b4', linestyle='-', linewidth=2.5, label='50M Fit: 0.374 · T^(-0.621) + 1.377')
ax1.plot(T_ext, ce_100_ext, color='#ff7f0e', linestyle='-', linewidth=2.5, label='100M Fit: 0.985 · T^(-0.480) + 1.325')

# Asymptotes
ax1.axhline(L_inf_50, color='#1f77b4', linestyle=':', linewidth=1.5, alpha=0.8, label=f'50M Asymptote (L_inf = {L_inf_50:.3f} nats)')
ax1.axhline(L_inf_100, color='#ff7f0e', linestyle=':', linewidth=1.5, alpha=0.8, label=f'100M Asymptote (L_inf = {L_inf_100:.3f} nats)')

# Milestone points
ax1.scatter([2.36], [1.5969], color='#1f77b4', s=100, edgecolors='black', linewidth=1.5, zorder=6)
ax1.scatter([5.0], [1.7842], color='#ff7f0e', s=120, marker='s', edgecolors='black', linewidth=1.5, zorder=6)

# Milestone: When 100M reaches 50M 2.4B loss (at 14.6B tokens)
T_match = 14.61
ce_match = 1.5969
ax1.scatter([T_match], [ce_match], color='#d62728', s=110, marker='X', zorder=6, label=f'100M matches 50M 2.4B Loss ({T_match:.1f}B tok)')
ax1.hlines(ce_match, 2.36, T_match, colors='#d62728', linestyles='--', alpha=0.6, linewidth=1.3)

# Annotations
ax1.annotate('50M Final (2.4B tok)\nCE = 1.597 nats',
             xy=(2.36, 1.5969), xytext=(2.8, 1.48),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
             fontweight='bold', color='#1f77b4', fontsize=9.5)

ax1.annotate('100M Final (5.0B tok)\nCE = 1.784 nats',
             xy=(5.0, 1.7842), xytext=(5.5, 1.92),
             arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5),
             fontweight='bold', color='#d95f02', fontsize=9.5)

ax1.annotate('100M requires 14.6B tokens\n(6.2x compute) to reach 50M 2.4B loss!',
             xy=(T_match, ce_match), xytext=(T_match - 4.2, ce_match + 0.18),
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
             fontweight='bold', color='#d62728', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#d62728', alpha=0.9))

ax1.set_xlabel('Tokens Trained (Billions)', fontweight='bold')
ax1.set_ylabel('Causal Validation Cross-Entropy (nats) ↓', fontweight='bold')
ax1.set_title('Cross-Entropy Loss Extrapolation (0 to 25B Tokens)', fontweight='bold')
ax1.set_xlim(0, 25.5)
ax1.set_ylim(1.25, 2.6)
ax1.legend(loc='upper right', frameon=True, fontsize=8.5)

# Panel 2: Perplexity Extrapolation
ax2.scatter(t_50, ppl_50, color='#1f77b4', s=45, zorder=5, label='50M AR (CUDA)')
ax2.scatter(t_100, ppl_100, color='#ff7f0e', s=55, marker='s', zorder=5, label='100M AR (TPU)')

ax2.plot(T_ext, ppl_50_ext, color='#1f77b4', linestyle='-', linewidth=2.5, label=f'50M PPL (Floor: {np.exp(L_inf_50):.2f})')
ax2.plot(T_ext, ppl_100_ext, color='#ff7f0e', linestyle='-', linewidth=2.5, label=f'100M PPL (Floor: {np.exp(L_inf_100):.2f})')

ax2.axhline(np.exp(L_inf_50), color='#1f77b4', linestyle=':', linewidth=1.5, alpha=0.8)
ax2.axhline(np.exp(L_inf_100), color='#ff7f0e', linestyle=':', linewidth=1.5, alpha=0.8)

ax2.scatter([2.36], [np.exp(1.5969)], color='#1f77b4', s=100, edgecolors='black', linewidth=1.5, zorder=6)
ax2.scatter([5.0], [np.exp(1.7842)], color='#ff7f0e', s=120, marker='s', edgecolors='black', linewidth=1.5, zorder=6)
ax2.scatter([T_match], [np.exp(ce_match)], color='#d62728', s=110, marker='X', zorder=6)

ax2.annotate('50M Final PPL: 4.94',
             xy=(2.36, np.exp(1.5969)), xytext=(3.0, 5.8),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
             fontweight='bold', color='#1f77b4', fontsize=9.5)

ax2.annotate('100M Final PPL: 5.95',
             xy=(5.0, np.exp(1.7842)), xytext=(5.8, 7.2),
             arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5),
             fontweight='bold', color='#d95f02', fontsize=9.5)

ax2.set_xlabel('Tokens Trained (Billions)', fontweight='bold')
ax2.set_ylabel('Validation Perplexity (PPL) ↓', fontweight='bold')
ax2.set_title('Perplexity Extrapolation & Asymptotic Floors', fontweight='bold')
ax2.set_xlim(0, 25.5)
ax2.set_ylim(3.0, 13.0)
ax2.legend(loc='upper right', frameon=True, fontsize=9.5)

plt.tight_layout()

out_fig = Path('figures/ar_50m_vs_100m_extrapolation.png')
plt.savefig(out_fig, bbox_inches='tight')

out_artifact = Path('/Users/ivansamuel/.gemini/antigravity-ide/brain/c498704a-5904-4bfd-93eb-e8089c486ea4/ar_50m_vs_100m_extrapolation.png')
plt.savefig(out_artifact, bbox_inches='tight')

plt.close()
print('✓ Successfully generated clean extrapolation plots!')
