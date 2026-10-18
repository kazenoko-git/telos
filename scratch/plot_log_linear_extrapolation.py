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

t_100 = np.array([r['tokens_b'] for r in res_100m])
ce_100 = np.array([r['causal_ce'] for r in res_100m])

# Log-linear fit: L(T) = a - b * ln(T)
p_50 = np.polyfit(np.log(t_50), ce_50, 1)    # p[0] = -b, p[1] = a
p_100 = np.polyfit(np.log(t_100), ce_100, 1)

a_50, b_50 = p_50[1], -p_50[0]
a_100, b_100 = p_100[1], -p_100[0]

print(f'50M Log-Linear:  L(T) = {a_50:.4f} - {b_50:.4f} * ln(T)')
print(f'100M Log-Linear: L(T) = {a_100:.4f} - {b_100:.4f} * ln(T)')

# Crossover calculation
ln_T_cross = (a_100 - a_50) / (b_100 - b_50)
T_cross_B = np.exp(ln_T_cross)
L_cross = a_50 - b_50 * ln_T_cross

print(f'Log-Linear Crossover: T = {T_cross_B:,.1f} Billion tokens ({T_cross_B/1000:.1f} Trillion tokens)')
print(f'Loss at crossover: {L_cross:.4f} nats')

# When does 100M reach 50M final (1.5969 nats)?
ln_T_match = (a_100 - 1.5969) / b_100
T_match = np.exp(ln_T_match)
print(f'100M reaches 1.5969 nats at T = {T_match:.2f}B tokens (Step {int(T_match*1e9/196608):,})')

# Evaluation domain for plotting (up to 30B tokens)
T_plot = np.linspace(0.2, 30.0, 500)
ce_50_log = a_50 - b_50 * np.log(T_plot)
ce_100_log = a_100 - b_100 * np.log(T_plot)

# Power law for comparison
A_50, alpha_50, L_inf_50 = 0.3738, 0.6211, 1.3772
A_100, alpha_100, L_inf_100 = 0.9845, 0.4795, 1.3248
ce_50_pow = A_50 * (T_plot ** -alpha_50) + L_inf_50
ce_100_pow = A_100 * (T_plot ** -alpha_100) + L_inf_100

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Log-Linear vs. Chinchilla Power-Law Extrapolation: 50M AR (CUDA) vs 100M AR (TPU)', fontsize=15, fontweight='bold', y=0.98)

# Panel 1: Log-Linear Extrapolation
ax1.scatter(t_50, ce_50, color='#1f77b4', s=45, zorder=5, label='50M AR (CUDA, Observed)')
ax1.scatter(t_100, ce_100, color='#ff7f0e', s=55, marker='s', zorder=5, label='100M AR (TPU, Observed)')

ax1.plot(T_plot, ce_50_log, color='#1f77b4', linewidth=2.5, label=f'50M Log-Linear: {a_50:.3f} - {b_50:.3f} ln(T)')
ax1.plot(T_plot, ce_100_log, color='#ff7f0e', linewidth=2.5, label=f'100M Log-Linear: {a_100:.3f} - {b_100:.3f} ln(T)')

# Milestone points
ax1.scatter([2.36], [1.5969], color='#1f77b4', s=100, edgecolors='black', linewidth=1.5, zorder=6)
ax1.scatter([5.0], [1.7842], color='#ff7f0e', s=120, marker='s', edgecolors='black', linewidth=1.5, zorder=6)
ax1.scatter([T_match], [1.5969], color='#d62728', s=110, marker='X', zorder=6, label=f'100M matches 50M loss at {T_match:.1f}B tok')
ax1.hlines(1.5969, 2.36, T_match, colors='#d62728', linestyles='--', alpha=0.6, linewidth=1.3)

ax1.annotate(f'100M reaches 50M (2.4B) Loss\nat {T_match:.1f}B tokens (Step 38,950)\n[3.2x Token Multiplier]',
             xy=(T_match, 1.5969), xytext=(T_match + 1.5, 1.75),
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
             fontweight='bold', color='#d62728', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#d62728', alpha=0.9))

ax1.annotate('Log-Linear Crossover Point:\n26,423 Billion Tokens (26.4 Trillion)\n[Slope difference is only 0.051 nats/log-tok]',
             xy=(20.0, 0.95), xytext=(12.0, 0.65),
             fontweight='bold', color='#555555', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#888888', alpha=0.9))

ax1.set_xlabel('Tokens Trained (Billions)', fontweight='bold')
ax1.set_ylabel('Causal Validation Cross-Entropy (nats) ↓', fontweight='bold')
ax1.set_title('Log-Linear Model: L(T) = a - b · ln(T)', fontweight='bold')
ax1.set_xlim(0, 30.5)
ax1.set_ylim(0.5, 2.6)
ax1.legend(loc='upper right', frameon=True, fontsize=9.5)

# Panel 2: Direct Model Comparison (Log-Linear vs Power Law)
ax2.plot(T_plot, ce_50_log, color='#1f77b4', linestyle='--', linewidth=2.0, label='50M Log-Linear (No floor)')
ax2.plot(T_plot, ce_50_pow, color='#1f77b4', linestyle='-', linewidth=2.5, label='50M Power-Law (Floor: 1.377 nats)')
ax2.plot(T_plot, ce_100_log, color='#ff7f0e', linestyle='--', linewidth=2.0, label='100M Log-Linear (No floor)')
ax2.plot(T_plot, ce_100_pow, color='#ff7f0e', linestyle='-', linewidth=2.5, label='100M Power-Law (Floor: 1.325 nats)')

ax2.axhline(L_inf_50, color='#1f77b4', linestyle=':', alpha=0.7)
ax2.axhline(L_inf_100, color='#ff7f0e', linestyle=':', alpha=0.7)

ax2.scatter(t_50, ce_50, color='#1f77b4', s=35, zorder=5)
ax2.scatter(t_100, ce_100, color='#ff7f0e', s=45, marker='s', zorder=5)

ax2.set_xlabel('Tokens Trained (Billions)', fontweight='bold')
ax2.set_ylabel('Causal Validation Cross-Entropy (nats) ↓', fontweight='bold')
ax2.set_title('Model Comparison: Log-Linear vs. Power-Law with Asymptote', fontweight='bold')
ax2.set_xlim(0, 30.5)
ax2.set_ylim(1.0, 2.6)
ax2.legend(loc='upper right', frameon=True, fontsize=9.5)

plt.tight_layout()

out_fig = Path('figures/ar_50m_vs_100m_log_linear.png')
plt.savefig(out_fig, bbox_inches='tight')

out_artifact = Path('/Users/ivansamuel/.gemini/antigravity-ide/brain/c498704a-5904-4bfd-93eb-e8089c486ea4/ar_50m_vs_100m_log_linear.png')
plt.savefig(out_artifact, bbox_inches='tight')

plt.close()
print('✓ Saved log-linear comparison plot to:', out_fig)
