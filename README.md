# τέλος (télos) — Exploring Language Modeling Paradigms at Scale

<p align="center">
  <a href="https://telos.research.wingit.tech"><strong> Research Page & Demos: telos.research.wingit.tech</strong></a>
</p>

**τέλος** (or **telos**) is an open-source AI research... thing... by me, designed to systematically evaluate, optimize, and compare foundational language modeling paradigms:
1. **Autoregressive Language Models (AR)** — Standard causal left-to-right next-token prediction.
2. **Masked Diffusion Language Models (MDLM)** — Non-autoregressive generation via continuous absorbing-state ($[\text{MASK}]$) diffusion.
3. **Uniform Noise Diffusion Language Models (UNDLM)** — Non-autoregressive generation via discrete uniform vocabulary corruption with **reversible self-correction**.

All architectures are built from scratch (with help of Opus 4.6, and a lot of research papers), sharing identical transformer backbones (RoPE, SwiGLU, RMSNorm, Weight Tying) and trained under controlled token-to-parameter scaling ratios on Apple Silicon (**Apple MLX / Metal**) and Google Cloud TPUs (**PyTorch-XLA**).


## Core Research Questions

**τέλος** (or **telos**) is a group of AI models trained for Research purposes to figure out what types of models are truly the best.

## Current Research includes

1. How well do Masked Diffusion Language Models scale with increase in tokens per parameter?
2. How can we increase the training performance with negligible model quality on MLX *(Apple M series Architecture)* and XLA *(Google TPU Architecture)*?
3. How do AR, MD and UND models compare with each other?
4. How well do Uniform Noise Diffusion Language Models scale with increase in tokens per paramater?
5. How to bring Diffusion Models to a level of Autoregressive models?


## Empirical Scaling Benchmark (12.5M Scale)

Below are the benchmark results evaluated across 12 models trained under identical architectures at the 12.5M scale across token over-training multipliers ($1:1$ up to $1:15$):

| Model Paradigm | Token Multiplier | Total Tokens Trained | Target CE (nats) ↓ | Avg Prediction Rank (/8192) ↓ | Top-1 Accuracy (%) ↑ | Top-5 Accuracy (%) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AR** | **1:1** | 12.58M | 8.1410 | 1341.7 | 0.00% | 0.99% |
| **AR** | **1:5** | 62.65M | 7.6211 | 1580.8 | 1.98% | 13.86% |
| **AR** | **1:10** | 125.3M | 7.3242 | 1436.5 | **5.94%** | 14.85% |
| **AR** | **1:15** | 188.0M | **7.0781** | **1135.0** | **5.94%** | **16.83%** |
| **MDLM** | **1:1** | 12.58M | 8.0734 | **831.5** | 0.00% | 0.00% |
| **MDLM** | **1:5** | 62.65M | 8.1652 | 1261.4 | 0.99% | 1.98% |
| **MDLM** | **1:10** | 125.3M | 8.0160 | 965.2 | 0.00% | 0.99% |
| **MDLM** | **1:15** | 188.0M | 7.8588 | 1176.5 | 0.99% | 5.94% |
| **UNDLM** | **1:1** | 12.58M | 9.0167 | 3240.1 | 0.00% | 0.00% |
| **UNDLM** | **1:5** | 62.65M | 8.5228 | 2018.5 | 0.00% | 0.00% |
| **UNDLM** | **1:10** | 125.3M | 8.3288 | 1999.6 | 0.00% | 2.97% |
| **UNDLM** | **1:15** | 188.0M | 8.3147 | 1750.3 | 0.00% | 0.99% |

### Cross Entropy Scaling
![Cross Entropy Scaling](figures/scaling_cross_entropy.png)

### Average Rank Scaling
![Average Rank Scaling](figures/scaling_average_rank.png)

### Top 5% Accuracy
![Scaling Top 5% Accuracy](figures/scaling_top5_accuracy.png)

### Key Research Insights

- **Autoregressive (AR)** excel at low-entropy directional continuations: leads on **Imports** (4.52 nats, #68 rank), **Operators** (5.25 nats, #92 rank), **Class Names** (6.13 nats, #100 rank), and **Literals** (4.99 nats, #43 rank).
- **Masked Diffusion (MDLM)** excels at structural and bidirectional syntactic constraints: leads on **Keywords** (#233 rank), **Function signatures** (#625 rank), and **Attribute definitions** (#646 rank).
- **Uniform Noise Diffusion (UNDLM)** exhibits continuous monotonic learning across all categories when evaluated with Monte Carlo noise marginalization, achieving the lowest cross-entropy in **Identifier recovery** (7.96 nats at 1:15).


## Technical Highlights

- **Unified Transformer Core**: Pre-LayerNorm architecture with **RMSNorm**, **SwiGLU** feed-forward activation, **Rotary Position Embeddings (RoPE)**, and weight-tied embeddings.
- **Continuous Noise Schedules**: Continuous timestep diffusion $t \sim \text{Beta}(\alpha, \beta)$ (default $\text{Beta}(1.5, 1.5)$) ensuring balanced noise allocation between global structure and fine token details.
- **ELBO-Consistent Loss**: Reweighted cross-entropy with $1/t$ timestep weighting matching discrete variational lower bounds.
- **Iterative Inference Samplers**:
  - *MDLM*: Confidence-based iterative unmasking over 16–128 steps with Gumbel noise temperature annealing.
  - *UNDLM*: Self-correcting reverse diffusion with dynamic re-noising and categorical sampling.
- **101 Contextual Probe Suite**: Standardized probing framework evaluating prediction rank, target cross-entropy, Top-1, and Top-5 accuracies across 8 code syntax categories.

## Publications

- **Research and Interactive Demo**: [telos.research.wingit.tech](https://telos.research.wingit.tech)
- **Detailed Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## Reproduction (Quickstart)

### 1. Installation
```bash
git clone https://github.com/kazenoko-git/telos.git
cd telos
uv sync  # or: pip install -e . && pip install mlx tokenizers torch pyyaml matplotlib
```

### 2. Training (AR, MDLM, UNDLM)
Train all three paradigms under identical configurations:
```bash
jupyter notebook notebooks/shared/Unified_Training_Suite.ipynb
```

### 3. Evaluate Contextual Probes
Evaluate any trained model checkpoint against the 101-probe suite:
```bash
uv run python notebooks/shared/Evaluation.py --checkpoint checkpoints/masked/12m/telos_12m_r15 --mode probes
```

### 4. Generate Graphs & Visualizations
```bash
uv run python scripts/shared/generate_3paradigm_probe_graphs.py
```

### 5. Quick Inference
```python
from mdiff.hub import TelosModel

model = TelosModel.from_pretrained("checkpoints/masked/12m/telos_12m_r15")
print(model.complete("def fibonacci(n: int) -> int:\n", max_tokens=64))
```

For advanced cluster setups, dataset tokenization, and TPU training, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Citation

```bibtex
@article{samuel2026telos,
  title   = {télos: Exploring Scaling Laws, Hardware Optimizations, and Paradigm Trade-offs in Discrete Diffusion and Autoregressive Language Models},
  author  = {Ivan Samuel},
  journal = {telos Research},
  year    = {2026},
  url     = {https://telos.research.wingit.tech}
}
```

---

## License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
