# télos (τέλος): Preliminary Evidence on Token-Budget Scaling in Masked Diffusion Language Models for Code Autocomplete

**Author**: Ivan Samuel  
**Affiliation**: Wing It Research  
**Website**: [telos.research.wingit.tech](https://telos.research.wingit.tech)

---

## Abstract

Autoregressive language models currently dominate code generation, yet they are fundamentally constrained by strict left-to-right generation. Masked Diffusion Language Models (MDLMs) offer a compelling alternative via bidirectional context and non-monotonic iterative decoding. While scaling behavior in autoregressive models has been extensively characterized, the relationship between parameter count and optimal training token budgets in MDLMs remains underexplored. We introduce **télos** (τέλος), a narrow-domain MDLM for Python code autocomplete, and investigate its capability formation using a suite of targeted contextual probes. Evaluating models at 12.5M and 25M parameters under a $\text{Beta}(1.5, 1.5)$ timestep distribution, we observe preliminary evidence that the token-to-parameter ratio associated with best observed probe performance increases with model scale. Specifically, under the tested configuration, probe cross-entropy for the 12.5M model reached a minimum at a 1:15 ratio (~187.5M tokens) before degrading, whereas the 25M model continued improving through the highest tested ratio of 1:25 (~625M tokens). Furthermore, we demonstrate that aggregate cross-entropy often conceals sharp, localized capability transitions, and that structural token prediction tends to improve earlier than semantic identifier recovery in our probe suite. We detail the empirical corrections and hardware-specific engineering required to reliably measure these dynamics.

---

## 1. Introduction

Autoregressive (AR) language models have established the paradigm for modern code generation. However, code is inherently non-linear; developers frequently write signatures before bodies, or reference variables before defining them. Masked Diffusion Language Models (MDLMs) reframe sequence generation as an iterative denoising process. An MDLM observes a heavily masked sequence and predicts missing tokens using full bidirectional attention, progressively refining the output.

The scaling laws of AR models—most notably formalised by Hoffmann et al. (Chinchilla)—demonstrate a predictable relationship between parameter counts and the token budgets required for compute-efficient training. Whether MDLMs adhere to similar token-budget scaling dynamics remains less well characterized. Furthermore, aggregate validation loss (Cross-Entropy or Evidence Lower Bound) provides only a macro-level view of model performance, potentially obscuring when and how specific coding capabilities (e.g., syntax closure vs. variable binding) emerge.

In this paper, we present **télos**, an open-source, narrow-domain Python MDLM. We focus our empirical investigation on the following research questions:

- **RQ1**: How does contextual code prediction evolve with additional training tokens?
- **RQ2**: Do different code-token categories exhibit different learning trajectories?
- **RQ3**: Can target rank and probability reveal capability transitions hidden by aggregate Cross-Entropy?
- **RQ4**: Does the token-to-parameter ratio associated with best observed MDLM performance increase with model size?
- **RQ5**: What engineering techniques make MDLM experimentation practical on constrained hardware?

---

## 2. Background and Related Work

### Masked Diffusion Language Models
Continuous-time diffusion models have recently been adapted for discrete categorical data. Sahoo et al. (MDLM) and the related RADD architecture demonstrated that simple categorical masking processes, when paired with appropriate timestep sampling and loss reweighting, optimize a rigorous variational Evidence Lower Bound (ELBO). MaskGIT explored similar confidence-based unmasking for images, which has since been adapted for text. Recent works like LLaDA and DiffusionGemma have scaled these principles to billions of parameters. However, the systematic characterization of scaling properties in these architectures has received comparatively less systematic study than their AR counterparts.

### Scaling Laws
The Chinchilla scaling laws describe compute-efficient training for AR models, identifying a constant optimal parameter-to-token ratio under a specific training and compute envelope. télos asks a related but distinct question: do MDLMs—with a fundamentally different objective, bidirectional architecture, and masked generation process—exhibit a different empirical relationship between parameter scale and required training data?

---

## 3. Model Architecture

We treat the repo-grounded implementation of télos as the authoritative architecture. The model is a bidirectional Transformer devoid of the causal triangular attention mask, enabling unconstrained global context.

```
Step 1: Clean Code Sequence
   │
   ▼
Step 2: ByteLevel BPE Tokenization
   │
   ▼
Step 3: Beta(1.5, 1.5) Masking
   │
   ▼
Step 4: Bidirectional Transformer (RMSNorm, SwiGLU, GQA)
   │
   ▼
Step 5: Full Sequence Logits (No Timestep Conditioning)
```

**Core design choices:**
- **Normalization**: RMSNorm (Root Mean Square Layer Normalization) for training stability.
- **Position Encoding**: Standard learnable absolute positional embeddings (1D).
- **Feed-Forward Network**: SwiGLU activation with expansion factor $\approx 2.67$, dimensions aligned to multiples of 64 for hardware efficiency.
- **Attention**: Full bidirectional multi-head self-attention with GQA (Grouped Query Attention) support.
- **Embeddings**: Untied input token embeddings (`self.emb`) and output projection matrices (`self.head`).
- **No Timestep Conditioning**: The model receives no information about the masking ratio $t$. This follows findings that time-agnostic MDLM architectures achieve near-optimal ELBO performance.

### Model Configurations
- **12.5M**: $d_{\text{model}} = 256$, Layers = 13, Heads = 4, KV Heads = 4, Seq Len = 512.
- **25M**: $d_{\text{model}} = 512$, Layers = 8, Heads = 8, KV Heads = 8, Seq Len = 512.
- **50M**: $d_{\text{model}} = 768$, Layers = 8, Heads = 12, KV Heads = 12, Seq Len = 512.

---

## 4. Diffusion Objective

The forward process independently replaces tokens with a `[MASK]` token with probability $t$. Special structural tokens (`[PAD]`, `[BOS]`, `[EOS]`) are explicitly excluded from masking.

```
Phase 1: Partially Masked Sequence
   │
   ▼
Phase 2: Parallel Model Prediction
   │
   ▼
Phase 3: Confidence-Based Selection
   │
   ▼
Phase 4: Permanent Unmasking of Top Tokens
```

The training objective utilizes a reweighted cross-entropy loss:

$$\mathcal{L} = \frac{1}{t} \cdot \text{CE}_{\text{masked}}$$

The timestep $t$ is clamped ($t \in [10^{-3}, 1.0]$) to maintain numerical stability against exploding gradients as $t \to 0$.

---

## 5. Data and Tokenization

- **Dataset**: CodeParrot-Clean corpus, restricted to Python source code. Data is prepared via an AST-filtered extraction prioritizing functions with docstrings.
- **Tokenization**: Custom ByteLevel BPE tokenizer with a vocabulary of 8,192. Byte-level pre-tokenization preserves critical Python indentation.
- **Storage**: Memory-mapped binary arrays (`.bin`) storing `int32` token IDs.

---

## 6. Experimental Setup and Corrections

### 6.1 Timestep Sampling
We sample the masking ratio from a Beta distribution: $t \sim \text{Beta}(1.5, 1.5)$, concentrating probability mass around $t \approx 0.5$. This corrected an earlier implementation that used a cosine transform resulting in an invalid $\text{Beta}(0.5, 0.5)$ arcsine distribution that oversampled extreme values ($t \approx 0$ and $t \approx 1$).

### 6.2 Validated Experimental Runs

| Model | Params | Ratio | Tokens | Init | Status | Mean Probe CE (nats) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| télos-12.5M | 12.5M | 1:1 | 12.5M | scratch | complete | 8.1754 |
| télos-12.5M | 12.5M | 1:5 | 62.5M | scratch | complete | 8.0957 |
| télos-12.5M | 12.5M | 1:10 | 125M | scratch | complete | 7.9897 |
| télos-12.5M | 12.5M | 1:15 | 187.5M | scratch | complete | **7.5604** |
| télos-12.5M | 12.5M | 1:20 | 250M | scratch | complete | 7.7388 |
| télos-12.5M | 12.5M | 1:25 | 312.5M | scratch | complete | 7.8470 |
| télos-25M | 25M | 1:1 | 25M | Net2Net | complete | 7.8068 |
| télos-25M | 25M | 1:10 | 250M | Net2Net | complete | 7.5627 |
| télos-25M | 25M | 1:15 | 375M | Net2Net | complete | 7.6237 |
| télos-25M | 25M | 1:20 | 500M | Net2Net | complete | 7.6413 |
| télos-25M | 25M | 1:25 | 625M | Net2Net | complete | **7.4361** |
| télos-50M | 50M | 1:25 | 1.25B | Net2Net | complete | **7.1873** |

---

## 7. Token-Budget Scaling Observations

- **12.5M Saturation**: Probe CE reaches minimum at **1:15** (187.5M tokens, CE $\approx 7.5604$), then degrades at higher ratios.
- **25M Monotonic Progress**: Continuous improvement through **1:25** (625M tokens, CE $\approx 7.4361$) without saturation.

### Candidate Scaling Formulae
Fitting optimal token ratio $R^*(N)$ against parameter count $N$ (in millions):

1. **Logarithmic**: $R^*(N) = 14.43 \cdot \ln(N) - 21.44 \implies$ 50M optimal $\approx 1:35$
2. **Sqrt + Constant**: $R^*(N) = 6.83 \cdot \sqrt{N} - 9.14 \implies$ 50M optimal $\approx 1:39$
3. **Power Law**: $R^*(N) = 2.39 \cdot N^{0.74} \implies$ 50M optimal $\approx 1:42$

---

## 8. Structural vs. Semantic Learning Trajectories

| Category | 12.5M (1:15) CE | 25M (1:25) CE | 50M (1:25) CE | Trend |
| :--- | :---: | :---: | :---: | :--- |
| **Operators** | 6.26 | 6.50 | 6.03 | ↓ Consistent improvement |
| **Keywords** | 6.82 | 6.56 | 6.05 | ↓ Rapid improvement |
| **Literals** | 7.02 | 6.83 | 5.87 | ↓ Largest improvement |
| **Imports** | 7.05 | 6.12 | 6.16 | ↓ Near-saturated |
| **Identifiers** | 7.77 | 7.64 | 6.29 | ↓ Improving |
| **Functions** | 8.33 | 8.43 | 8.45 | → Stalled |
| **Attributes** | 8.21 | 8.98 | 8.90 | → Stalled |
| **Class names** | 9.54 | 8.39 | 9.75 | ↑↓ Inconsistent |

**Key finding**: Structural syntax is learned significantly earlier than semantic identifier recovery, creating a ~4 nat gap between structural tokens and semantic tokens.

---

## 9. Systems and Hardware Implementation

- **Apple Silicon (MLX)**: Microbatch gradient accumulation and `mx.eval` graph management to prevent memory leaks. Fusing SwiGLU gates followed by `mx.split()` caused an 8.7–11.5% throughput regression due to strided non-contiguity; reverting to separate linear projections restored full tensor throughput.
- **TPU v6e-1 (PyTorch-XLA)**: Strategic placement of `xm.mark_step()` immediately following dynamic random masking operations prevented graph recompilation crashes.

---

## References

1. Sahoo, S., et al. (2024). *Masked Diffusion Language Models*.
2. Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models (Chinchilla)*.
3. Chang, H., et al. (2022). *MaskGIT: Masked Generative Image Transformer*.
4. Zheng, L., et al. (2024). *RADD: Reparameterized Absorbing Discrete Diffusion*.
5. Nie, J., et al. (2025). *LLaDA: Large Language Diffusion Models*.
6. Google DeepMind (2026). *DiffusionGemma: An experimental discrete diffusion model based on Gemma*.
