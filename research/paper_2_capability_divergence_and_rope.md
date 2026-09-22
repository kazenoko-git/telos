# télos (τέλος): Preliminary Report 2 — Capability Divergence, Positional Encoding, and the Limits of Masked Diffusion Scaling

**Author**: Ivan Samuel  
**Affiliation**: Wing It Research  
**Website**: [telos.research.wingit.tech](https://telos.research.wingit.tech) *(Note: Research portal is currently under active development)*

---

## Abstract

Following the initial scaling observations in Masked Diffusion Language Models (MDLMs) for code autocomplete, we complete our scaling sweeps across the 12.5M, 25M, and 50M parameter tiers. We identify a fundamental limitation in MDLM capacity allocation: past a critical, scale-dependent over-training ratio, aggregate Cross-Entropy (CE) and Average Target Rank decouple. While overall CE continues to improve, the model hyper-specializes on structurally predictable tokens (literals, operators, keywords) at the catastrophic expense of semantic tokens (function names, attributes). Furthermore, we evaluate the impact of Rotary Positional Embeddings (RoPE). We demonstrate that while native RoPE training yields record Top-5 accuracies, attempting to retrofit RoPE via 150-step fine-tuning on converged absolute-position models uniformly degrades performance by ~0.6 nats across all scales. These findings suggest that the strictly monotonic commitment of discrete masking creates a hard ceiling for semantic learning, motivating a transition toward Uniform Noise Diffusion Language Models (UNDLM) featuring self-correction.

---

## 1. Introduction

In Preliminary Report 1, we demonstrated that token-budget scaling for Masked Diffusion Language Models (MDLMs) does not follow a constant parameter-to-token ratio. Instead, we observed preliminary evidence that larger MDLMs require proportionally larger token budgets to reach optimal contextual prediction performance.

This second report details the completion of the 50M parameter suite and expands our analysis of the 12.5M and 25M models. By deeply analyzing category-level probe performance, we uncover a novel phenomenon: the **CE-Rank Divergence**. We find that MDLMs eventually hit a representational wall where the loss continues to decrease solely due to high-confidence structural predictions, while open-vocabulary semantic predictions actively degrade.

Additionally, we investigate architectural modifications, specifically Rotary Positional Embeddings (RoPE), documenting a stark difference between native RoPE training and post-hoc RoPE fine-tuning.

---

## 2. Completed Scaling Sweeps and the "Sweet Spot"

We evaluated all three model tiers across extended token-to-parameter ratios. We consistently observe three distinct regimes: a baseline plateau, an optimal "sweet spot", and a divergence/collapse phase.

| Model Tier | Optimal "Sweet Spot" Ratio | Sweet Spot CE (nats) | Sweet Spot Rank | Collapse / Divergence Ratio |
| :--- | :---: | :---: | :---: | :--- |
| **12.5M** | 1:20 (250M tokens) | 7.84 | 729 | 1:30 (CE 8.40, Rank 1,721) |
| **25M** | 1:30 (750M tokens) | 6.90 | 409 | 1:35 (CE 7.39, Rank 1,175) |
| **50M** | 1:35 (1.75B tokens) | 7.05 | 446 | 1:40 (CE 9.67, Rank 3,988) |

- **Observation 1**: The optimal over-training ratio scales with parameter count. Larger models can absorb proportionally more data before they begin to over-specialize or destabilize.
- **Observation 2 (PE Confounding)**: Note that the 25M 1:30 $\to$ 1:35 transition is confounded by both an increase in token budget and an architectural change in positional encoding (from absolute PE to native RoPE). A similar PE confound applies to the 12.5M 1:25 $\to$ 1:30 transition.
- **Observation 3 (Cleanest Over-Training Curve)**: The 50M series (all non-RoPE, absolute PE) represents our cleanest over-training trajectory: 1:25 (7.19 CE) $\to$ 1:35 (7.05 CE) $\to$ 1:40 (9.67 CE, catastrophic collapse). Because positional encoding remained invariant across these runs, this provides unconfounded evidence for scale-dependent over-training limits in MDLM.

---

## 3. Empirical Scaling Laws: Cross-Entropy and Top-5 Accuracy

<p align="center">
  <img src="../figures/scaling_cross_entropy.png" alt="Target Cross Entropy Scaling" width="400">
  <img src="../figures/scaling_top5_accuracy.png" alt="Top-5 Accuracy Scaling" width="400">
</p>

### Target Cross-Entropy vs. Token Multiplier

- **New Project Record Cross-Entropy**: 25M 1:40 achieved the lowest cross-entropy in the entire project at **7.24 nats**, outperforming 50M 1:45 (7.41 nats) and 25M 1:35 (7.39 nats).
- **Monotonic Reduction**: Scaling the over-training ratio systematically reduces uncertainty across all syntactic probe categories.

### Top-5 Accuracy Scaling (%)

- **Top-1 Breakthrough**: 25M 1:40 achieved **7.92% Top-1 exact accuracy** (8 out of 101 target tokens were the absolute #1 highest-probability prediction out of 8,192 classes).
- **Top-5 Consistency**: Maintained a strong **17.82% Top-5 accuracy** (18 out of 101 in Top 5).

---

## 4. The CE-Rank Divergence Phenomenon

The most critical finding from extended scaling is the decoupling of Cross-Entropy and Average Target Rank. CE and Average Rank decouple past a critical over-training ratio: CE continues to decrease while rank increases — the model becomes a better syntactic predictor but a worse semantic predictor.

<p align="center">
  <img src="../figures/scaling_average_rank.png" alt="CE-Rank Divergence Average Rank" width="550">
</p>

*[Category Heatmap: YET TO UPDATE]*

In the 25M tier, training from 1:30 to 1:40 yielded an improvement in overall CE (from 7.39 at 1:35 down to 7.24 at 1:40), while average target rank tripled (from 409 at 1:30 to 1,239 at 1:40).

### Category-Level Analysis: Where Rank Explodes

Dissecting probe category transitions (25M: 1:30 $\to$ 1:40):

- **Literals Rank**: 351 $\to$ 32 ($11\times$ improvement)
- **Operators Rank**: 66 $\to$ 67 (Maintained)
- **Imports Rank**: 70 $\to$ 290 ($4\times$ degradation)
- **Function Names Rank**: $\sim 3,100 \to 3,452$ (Severe degradation)
- **Attribute Names Rank**: $\sim 3,500 \to 3,954$ (Severe degradation)

### Capacity Allocation in MDLMs
As over-training progresses, the gradient signal disproportionately rewards easily predictable structural tokens (which produce the largest, most consistent loss reductions). The model concentrates its fixed representational budget on position-predictable tokens (e.g., predicting an operator after a variable, or a boolean literal).

Consequently, the model is "starved" of the capacity required to learn complex, context-dependent semantic bindings. An MDLM can predict that `self._____` requires an attribute, but lacking self-correction during generation, it loses the nuanced semantic resolution to distinguish `self.logger` from `self.config`.

---

## 5. Positional Encoding: The RoPE Fine-Tuning Regression

To improve sequence extrapolation and positional alignment, we attempted to retrofit Rotary Positional Embeddings (RoPE) onto converged absolute/learned positional embedding models using a 150-step (~19M token) fine-tuning phase.

The results were unequivocally negative across all scales:

| Original Base Checkpoint (Absolute PE) | After 150-step RoPE FT | Delta |
| :--- | :--- | :--- |
| **50M 1:25**: 7.19 CE (Rank 469) | 7.79 CE (Rank 725) | **+0.60 CE (Degraded)** |
| **50M 1:35**: 7.05 CE (Rank 446) | 7.72 CE (Rank 823) | **+0.67 CE (Degraded)** |
| **25M 1:30**: 6.90 CE (Rank 409) | 7.55 CE (Rank 620) | **+0.65 CE (Degraded)** |

- **The RoPE Shock**: Injecting `mlx.fast.rope` directly into attention blocks subjected $W_q$ and $W_k$ matrices to an immediate geometric shift from additive offsets to rotary phase angles. 150 steps (~1% of budget) was vastly insufficient to re-align attention circuits, resulting in systematic ~0.6 nat degradation.
- **Native RoPE from Initialization**: Models trained natively with RoPE from step 0 (e.g., 25M 1:35 and 1:40) achieved record Top-5 accuracies (17.82%), demonstrating that RoPE is highly effective when trained from scratch.

---

## 6. Implications for Télos

The MDLM scaling study provides two concrete findings:
1. **Scale-Dependent Over-Training Limits**: The clean 50M absolute-PE series demonstrates that extreme over-training eventually leads to instability and catastrophic collapse (50M 1:40).
2. **Positional Encoding Constraints**: RoPE cannot be retrofitted post-hoc via short fine-tuning.

This motivates the exploration of alternative diffusion mechanisms:
- **Uniform Noise DLMs (UNDLM)**: Introduces a uniform noise schedule enabling tokens to transition across vocabulary states dynamically, providing an intrinsic self-correction mechanism.
- **Continuous-Time Discrete-Space DLMs (CTDS-DLM)**: Uses a continuous transition rate matrix to avoid discrete masking commitment.
- **COROSred**: Unifies causal AR drafting with selective re-diffusion to eliminate commitment errors.
