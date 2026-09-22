# τέλος (télos) — Discrete Diffusion & Dual-Paradigm Language Models

<p align="center">
  <a href="https://telos.research.wingit.tech"><strong>🌐 Research Page & Interactive Demo: telos.research.wingit.tech</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/hardware-Apple%20Silicon%20(MLX)%20%7C%20CUDA%20%7C%20Cloud%20TPU%20(XLA)-orange" alt="Hardware Support">
  <img src="https://img.shields.io/badge/HuggingFace-Kazenowoko%2Ftelos-yellow" alt="Hugging Face">
</p>

**τέλος** (or **telos**) is an open-source research initiative and model family designed to transcend the sequential bottlenecks of pure autoregressive language modeling. Télos unifies **Causal Autoregressive (AR) drafting** with **Bidirectional Masked Diffusion infilling** and **Confidence-Guided Selective Re-Diffusion** in a single, hardware-aligned architecture.

---

## 🌟 Key Empirical Results

### 1. Télos vs. Pure Autoregressive Baselines & GPT-2 (125M)

Télos models were systematically benchmarked against compute-matched pure Autoregressive baselines (trained on 4.0B–5.0B tokens) and **GPT-2 (125M / Small)** across causal next-token likelihood, bidirectional code infilling, anti-cheat suffix immunity, and execution correctness:

![Télos vs AR and GPT-2](https://raw.githubusercontent.com/kazenoko-git/telos/main/figures/telos_vs_ar_and_gpt2_comparison.png)

| Evaluation Metric | GPT-2 (125M) | 100M Pure AR (5.0B) | 50M Télos (CUDA, 2.5B) | 100M Télos (TPU, 5.0B) | Télos vs Baseline Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Causal Validation CE (nats) ↓** | 1.88 | **1.78** | **1.63** | 1.81 | **$-0.15$ nats (50M vs GPT-2)** |
| **Causal Validation Perplexity ↓** | 6.55 | **5.95** | **5.08** | 6.16 | **$1.29\times$ lower PPL** |
| **Bidirectional Infill Top-1 (Overall) ↑** | 11.5% | 7.6% | 51.0% | **63.0%** | **$+55.4\%$ ($8.29\times$ higher)** |
| **Syntax Delimiter Infilling Top-1 ↑** | 18.2% | 21.5% | 54.0% | **68.2%** | **$+46.7\%$ ($3.17\times$ higher)** |
| **Semantic Identifier Infilling Top-1 ↑** | 4.1% | 3.6% | 48.0% | **58.5%** | **$+54.9\%$ ($16.2\times$ higher)** |
| **Anti-Cheat Span Exact Match ($K=1..8$)** | 8.2% | 6.4% | 48.0% | **52.0%** | **$+45.6\%$ ($8.1\times$ higher)** |
| **Boundary Suffix-Copy Rate ↓** | 96.5% *(Cheats)* | 94.2% *(Cheats)* | **1.5%** *(Immune)* | **2.0%** *(Immune)* | **Resists contextual copying** |
| **512-Task AST Syntax Validity ↑** | 92.1% | **98.5%** | 97.1% | 97.3% | Consistent robust syntax |
| **512-Task Pass@1 Execution ↑** | 4.2% | **12.5%** | 10.7% | 5.1% | Strong causal generation |
| **Numerical Constant Precision ↑** | 42.0% | 78.4% | **88.6%** | 64.9% | High numeric faithfulness |

---

### 2. Dual-Paradigm Parameter Scaling Trajectory

Télos exhibits dual-paradigm scaling: while pure AR models flatline on right-context resolution, Télos demonstrates monotonic capacity scaling on bidirectional infilling while matching causal scaling curves:

![Télos Scaling Trajectory](https://raw.githubusercontent.com/kazenoko-git/telos/main/figures/paradigm_scaling_trajectory.png)

- **Bidirectional Infilling Emergence**: Across $15\text{M} \to 50\text{M} \to 75\text{M} \to 100\text{M}$, infill Top-1 accuracy scales from **$0.0\% \to 51.0\% \to 60.0\% \to 63.0\%$**.
- **No Causal Regression on Scaling**: Causal validation loss on held-out tokens converges steadily from $2.61 \to 1.81$ nats ($13.69 \to 6.16$ PPL).
- **Anti-Cheat Span Immunity**: Under multi-token chunk masking spans $K \in \{1, 2, 4, 8, 16\}$, Télos models reject naive suffix copying ($2.0\%$ copy rate) and actively infer missing semantic structures.

---

## 🔬 Core Architecture: COROSred

**COROSred** (**CO**nfidence-**RO**uted **S**elective **RE**-**D**iffusion) unites autoregression and diffusion into a single dynamic training process:

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

1. **Unified Dynamic Schedule**: A unified training schedule dynamically balances:
   - **$\alpha(t)$ (Causal AR)**: Standard next-token prediction ensuring robust sequence drafting.
   - **$\beta(t)$ (Uniform Masked Diffusion)**: Random absorbing-state infilling ensuring complete bidirectional context utilization.
   - **$\gamma(t)$ (Selective Re-Diffusion)**: Targeted masking of ambiguous/low-confidence model drafts guided by the Learned Reliability Head.
2. **Learned Reliability Head (LRH)**: A detached auxiliary classification head trained alongside causal layers to predict whether a draft token is likely correct or ambiguous, eliminating heuristic confidence thresholds.
3. **Hardware-Aligned Backbones**:
   - **Rotary Position Embeddings (RoPE)** with dynamic base frequencies.
   - **SwiGLU Non-Linear Feed-Forward Networks** with exact $8/3 \times d_{\text{model}}$ intermediate dimension matching.
   - **RMSNorm** pre-normalization for numerical stability in BF16.
   - **Weight Tying** between token embeddings and output projection.
   - **64-Dimensional Attention Heads** aligned with TPU systolic arrays and GPU tensor cores.

---

## 🏛 Canonical Model Tiers

| Tier | Parameters | $d_{\text{model}}$ | Layers | Heads ($Q$) | Heads ($KV$) | Context Length | Target Training Tokens |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15M** | 15,248,384 | 288 | 8 | 6 | 6 | 512 | 300M – 1.0B |
| **50M** | 49,166,848 | 512 | 14 | 8 | 8 | 512 | 2.0B – 2.5B |
| **75M** | 74,865,152 | 512 | 22 | 8 | 8 | 512 | 3.5B – 4.0B |
| **100M** | 105,404,160 | 768 | 14 | 12 | 12 | 512 | 4.0B – 5.0B |
| **300M** | 299,630,592 | 1024 | 23 | 16 | 16 | 512 | 10.0B – 15.0B |
| **500M** | 498,720,768 | 1280 | 28 | 20 | 20 | 512 | 20.0B – 25.0B |

---

## 📊 Evaluation Suite (`telos eval`)

Télos includes an institutional-grade evaluation framework across **Code**, **Linguistics**, and **Tool-Use**:

```bash
# 1. Run Complete Multi-Domain Evaluation (Code + English Linguistics + Tool-Use)
telos eval --checkpoint checkpoints/corosred/model.safetensors --type all --mode full

# 2. Python Code Contextual Probes Suite (1,000 contextual probes with 95% Bootstrap CI)
telos eval --checkpoint checkpoints/corosred/model.safetensors --type code --language python --mode probes --num-probes 1000

# 3. Functional Execution Benchmark (512 sandboxed tasks with Pass@1, AST validity, and repetition dynamics)
telos eval --checkpoint checkpoints/corosred/model.safetensors --type code --mode functional --suite private_unseen

# 4. Anti-Cheat & Suffix-Copy Span Masking Suite (K in {1, 2, 4, 8, 16})
telos eval --checkpoint checkpoints/corosred/model.safetensors --type code --mode anticheat

# 5. English Linguistic Probes (100 probes across agreement, idioms, correlatives, and world knowledge)
telos eval --checkpoint checkpoints/corosred/model.safetensors --type linguistic --language english --mode probes

# 6. Tool-Use & Function Calling Benchmark (JSON schema adherence and argument extraction)
telos eval --checkpoint checkpoints/corosred/model.safetensors --type tooluse
```

---

## 🚀 Quickstart & Installation

### 1. Installation

```bash
git clone https://github.com/kazenoko-git/telos.git
cd telos
uv sync  # or: pip install -e .
```

### 2. Zero-Config Training via CLI

Train any architecture tier dimensionally directly from the command line:

```bash
# Train 50M Unified COROSred Model on Apple Silicon (MLX)
telos train --paradigm corosred --params 50M --tokens 2.5B --effective-batch 384 --hardware mlx

# Train 100M Unified COROSred Model on Cloud TPU (PyTorch-XLA, 8 devices)
telos train --paradigm corosred --params 100M --tokens 5.0B --effective-batch 384 --batch-size 48 --hardware xla --devices 8

# Train 100M Pure Autoregressive Baseline on CUDA (H100/A100)
telos train --paradigm ar --params 100M --tokens 5.0B --effective-batch 384 --batch-size 192 --grad-accum 2 --hardware cuda
```

### 3. Python API

```python
from telos.eval import evaluate

# Evaluate a checkpoint across all benchmark domains
report = evaluate(
    checkpoint="checkpoints/corosred/unified/100m_5b/checkpoint_final.pt",
    benchmark_type="code",
    mode="full",
    num_probes=1000
)

print(f"Causal Top-1: {report['probes']['causal']['top1_pct']}%")
print(f"Infill Top-1: {report['probes']['infill']['top1_pct']}%")
print(f"Pass@1:       {report['functional']['pass_at_1_pct']}%")
```

---

## 📦 Hugging Face Repositories

- **Main Model Hub**: [huggingface.co/Kazenowoko/telos](https://huggingface.co/Kazenowoko/telos)
  - `checkpoints/corosred/unified/100m_5b/` — 100M COROSred Unified (5.0B tokens).
  - `checkpoints/corosred/unified/75m_python/` — 75M COROSred Unified (3.75B tokens).
  - `checkpoints/corosred/50m_lightning/` — 50M COROSred Unified (2.5B tokens).
  - `checkpoints/ar/100m_5b/` — 100M Pure AR Baseline (5.0B tokens).
  - `checkpoints/ar/50m_2.5b/` — 50M Pure AR Baseline (2.5B tokens).

---

## 📖 Citation

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

## 📄 License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
