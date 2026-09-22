# τέλος (télos) — Discrete Diffusion & Dual-Paradigm Language Models

<p align="center">
  <a href="https://telos.research.wingit.tech">
    <img src="https://raw.githubusercontent.com/kazenoko-git/telos/main/figures/benchmark_comparison_graph.png" alt="Télos Research Portal Benchmark Dashboard" width="880">
  </a>
</p>

<p align="center">
  <a href="https://telos.research.wingit.tech"><strong>🌐 Interactive Research Portal & Live Demos: telos.research.wingit.tech</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/organization-Wing%20It%20Research-blueviolet" alt="Wing It Research">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/hardware-Apple%20Silicon%20(MLX)%20%7C%20CUDA%20%7C%20Cloud%20TPU%20(XLA)-orange" alt="Hardware Support">
</p>

---

## 1. What Télos by Wing It Research is About

**télos** (τέλος) is an open-source research initiative by **Wing It Research** exploring non-monotonic generation, discrete diffusion, and dual-paradigm language modeling for code autocomplete and reasoning.

Autoregressive (AR) language models are inherently constrained by left-to-right generation. Télos introduces architectures and training dynamics that unify **Causal Autoregressive drafting**, **Bidirectional Masked Diffusion infilling**, and **Confidence-Guided Selective Re-Diffusion** (COROSred) into a single, hardware-aligned model family.

Interactive research visualizations, scaling charts, and live code completion playgrounds are available online at **[telos.research.wingit.tech](https://telos.research.wingit.tech)**.

---

## 2. Latest Findings

### a. Latest Benchmarks (AFM vs LFM vs IBM Prelim)

We conducted an institutional-grade 7-suite comparative evaluation comparing premier edge and open-weights models: **Apple AFM-3 Core Advanced** (native Swift IPC), **IBM Granite 4.2 3B (MLX)**, and **LiquidAI LFM 8B A1B**:

<p align="center">
  <img src="https://raw.githubusercontent.com/kazenoko-git/telos/main/figures/benchmark_breakdown_horizontal.png" alt="Benchmark Breakdown Across Models" width="880">
</p>

| Suite | Tasks | Apple AFM-3 Core Adv. | IBM Granite 4.2 3B | LiquidAI LFM 8B A1B | Domain Metric |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **OpenAI HumanEval** | 164 | **62.80%** | 60.37% | 45.73% | Pass@1 (Code Execution) |
| **BFCL Tool-Use** | 30 | **73.33%** | 70.00% | 63.33% | Pass@1 (JSON Schema Match) |
| **Cybersecurity Auditing** | 50 | 12.00% | 22.00% | **24.00%** | Pass@1 (CWE Remediation) |
| **GPQA Diamond** | 198 | **39.39%** | 28.79% | 20.71% | Pass@1 (Multiple Choice) |
| **MMLU Science** | 119 | 69.75% | 61.82% | **73.11%** | Pass@1 (Multiple Choice) |
| **Competition MATH** | 140 | 52.00% | **64.71%** | 43.57% | Pass@1 (LaTeX / Boxed Math) |
| **ARC-Challenge** | 150 | 85.32% | **86.60%** | 59.56% | Pass@1 (Reasoning Choice) |
| **Macro Average** | **841** | **56.37%** | **56.33%** | **47.14%** | 7-Suite Completion |

*For complete breakdown and evaluation scripts, see [`benchmarks/README.md`](benchmarks/README.md).*

---

### b. Latest Research (COROSred Prelim)

**COROSred** (**CO**nfidence-**RO**uted **S**elective **RE**-**D**iffusion) unites causal drafting and bidirectional diffusion refinement:

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

- **Perplexity Advantage**: Achieves **5.08 PPL** ($1.29\times$ lower validation perplexity than compute-matched pure AR).
- **Bidirectional Infilling**: Reaches **63.0% Top-1 infill accuracy** (vs $7.6\%$ for pure AR).
- **Anti-Cheat Span Immunity**: Maintains **52.0% exact match** with a **2.0% suffix copy rate** under multi-token chunk masking ($K \in \{1, 2, 4, 8, 16\}$).
- **CE-Rank Divergence**: Discovered that extreme MDLM over-training leads to Cross-Entropy/Target-Rank divergence, where loss decreases on structural tokens while semantic rank explodes.

*For paper preprints and scaling laws, see [`research/README.md`](research/README.md).*

---

### c. Latest Models

**None** *(No pre-trained model weights are currently hosted or distributed publicly. Users can train architectures from scratch using zero-config CLI commands).*

---

## 3. Documentation Directory Routing

| Directory | Topic & Contents |
| :--- | :--- |
| 🔬 **[Research (`/research`)](research/README.md)** | **[Paper 1: Token-Budget Scaling](research/paper_1_token_budget_scaling.md)** • **[Paper 2: Capability Divergence & RoPE](research/paper_2_capability_divergence_and_rope.md)** • **[COROSred Architecture](research/corosred_architecture.md)** |
| 📊 **[Benchmarks (`/benchmarks`)](benchmarks/README.md)** | **[AFM vs LFM vs IBM Comparison](benchmarks/afm_vs_lfm_vs_ibm.md)** • **[Apple AFM Report](benchmarks/afm.md)** • **[IBM Granite Report](benchmarks/ibm_granite.md)** • **[LiquidAI LFM Report](benchmarks/liquid_lfm.md)** |
| 🖥️ **[CLI Usage (`/cli`)](cli/README.md)** | **[CLI Command Guide & Zero-Config Training](cli/README.md)** *(Official replacement for DEPLOYMENT.md)* |

---

## 4. Basic CLI Usage

### Installation

```bash
# Standard package install
pip install telos

# With Apple Silicon Metal support (MLX)
pip install "telos[mlx]"
```

### Core CLI Commands

```bash
# 1. Data Preparation (.bin token streams)
telos dataprep --input raw_data/ --output data/python.bin --vocab-size 8192

# 2. Zero-Config Training (50M COROSred on Apple Silicon MLX)
telos train --paradigm corosred --params 50M --tokens 2.5B --hardware mlx

# 3. Multi-Domain Evaluation
telos eval --checkpoint checkpoints/corosred/model.safetensors --type all --mode full

# 4. Hardware Benchmark (Capped at 5 minutes)
telos bench --paradigm corosred --params 50M --hardware mlx

# 5. Apple Foundation Models (AFM-3 Core Advanced on Apple Silicon)
telos afm status
telos afm generate "Explain rotary position embeddings in two sentences."

# 6. Verification Suite
telos test
```

### Python API

```python
import telos

# Probe on-device Apple Foundation Models
if telos.afm.probe().available:
    response = telos.afm.generate("Write a python function for quicksort.")
    print(response)
```

*For complete CLI documentation, flags, and hardware setup, see [`cli/README.md`](cli/README.md).*

---

## 5. Citations & References

### Primary Citation

```bibtex
@article{samuel2026telos,
  title   = {télos: Exploring Scaling Laws, Hardware Optimizations, and Paradigm Trade-offs in Discrete Diffusion and Autoregressive Language Models},
  author  = {Ivan Samuel},
  journal = {Wing It Research},
  year    = {2026},
  url     = {https://telos.research.wingit.tech}
}
```

### Framework & Related AI Research Citations

- **Apple MLX Framework**:
  ```bibtex
  @software{mlx2023,
    author  = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Tristan Lifchitz and Gautier Izacard and Bowen Baker and Guillermo Izquierdo and Maryam Fasihpanah and Michael Brabandere and others},
    title   = {{MLX}: Efficient machine learning on Apple silicon},
    url     = {https://github.com/ml-explore/mlx},
    version = {0.22.0},
    year    = {2023}
  }
  ```
- **Masked Diffusion Language Models (MDLM)**: Sahoo et al., *Masked Diffusion Language Models*, 2024.
- **Compute-Optimal Scaling (Chinchilla)**: Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022.
- **Large Language Diffusion Models (LLaDA)**: Nie et al., *LLaDA: Large Language Diffusion Models*, 2025.
- **Reparameterized Discrete Diffusion (RADD)**: Zheng et al., *RADD: Reparameterized Absorbing Discrete Diffusion*, 2024.
- **Masked Image Generation (MaskGIT)**: Chang et al., *MaskGIT: Masked Generative Image Transformer*, 2022.
- **Discrete Diffusion Language Modeling (DiffusionGemma)**: Google DeepMind, *DiffusionGemma: An experimental discrete diffusion model based on Gemma*, 2026.

---

## 📄 License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
