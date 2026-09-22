# COROSred: Confidence-Routed Selective Re-Diffusion

**Author**: Ivan Samuel  
**Affiliation**: Wing It Research  
**Website**: [telos.research.wingit.tech](https://telos.research.wingit.tech) *(Note: Research portal is currently under active development)*

---

## Architecture Overview

**COROSred** (**CO**nfidence **RO**uted **S**elective **ReD**iffusion) unites **Causal Autoregressive (AR) drafting** with **Bidirectional Masked Diffusion infilling** and **Confidence-Guided Selective Re-Diffusion** in a single dynamic training process and runtime architecture.

```
                               ┌────────────────────────────────────────────────┐
                               │             TÉLOS MODEL BACKBONE               │
                               │  (RoPE, SwiGLU, RMSNorm, Weight Tying, GQA)    │
                               └──────────────────────┬─────────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
                       ▼                                                             ▼
         ┌───────────────────────────┐                                 ┌───────────────────────────┐
         │  Causal AR Next-Token     │                                 │ Learned Reliability Head  │
         │  Loss: L_causal (alpha)   │                                 │ LRH Predictability Gate   │
         └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                       │                                                             │
                       │                        Routing Decision                     │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │ High Confidence (> 0.65):   │ ──> Accept Draft Token
                                       │ Fast Causal Autoregression  │
                                       ├─────────────────────────────┤
                                       │ Low Confidence (Ambiguous): │ ──> Route to Bidirectional
                                       │ Masked Infill & Re-Diff     │     Refinement Denoiser
                                       └─────────────────────────────┘
```

---

## Key Components

### 1. Unified Dynamic Loss Schedule
COROSred optimizes a three-part composite loss function:

$$\mathcal{L}_{\text{total}} = \alpha(t) \mathcal{L}_{\text{causal}} + \beta(t) \mathcal{L}_{\text{infill}} + \gamma(t) \mathcal{L}_{\text{rediff}}$$

- **$\alpha(t)$ (Causal AR Loss)**: Next-token prediction ensuring fast, coherent causal sequence drafting.
- **$\beta(t)$ (Uniform Masked Diffusion Loss)**: Random absorbing-state infilling over masked spans, enforcing full bidirectional attention context.
- **$\gamma(t)$ (Selective Re-Diffusion Loss)**: Targeted masking and refinement of ambiguous or low-confidence draft tokens predicted by the model's own preliminary passes.

### 2. Learned Reliability Head (LRH)
Unlike standard diffusion models that rely on heuristic entropy cutoffs, COROSred incorporates an auxiliary classification head trained to output a scalar confidence score $c_i \in [0, 1]$ per token.
- High-confidence predictions ($c_i \ge 0.65$) are accepted immediately in 1 step.
- Low-confidence tokens ($c_i < 0.65$) trigger a bidirectional re-diffusion pass, giving the model iterative self-correction capabilities.

---

## Performance Summary

<p align="center">
  <img src="../figures/corosred_50m_2b_benchmark.png" alt="COROSred 50M Benchmark" width="750">
</p>

*[Detailed Ablation Diagrams: YET TO UPDATE]*

Across 512-task evaluation suites, COROSred eliminates the structural vs. semantic trade-off seen in pure MDLMs:
- **Causal Validation Perplexity**: $5.08$ ($1.29\times$ lower PPL than compute-matched pure AR).
- **Bidirectional Infill Top-1**: $63.0\%$ (vs. $7.6\%$ for pure AR baselines).
- **Anti-Cheat Span Immunity**: $52.0\%$ exact match with $2.0\%$ suffix copy rate under multi-token chunk masking ($K \in \{1, 2, 4, 8, 16\}$).
