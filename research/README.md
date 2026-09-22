# Télos Research Papers & Scaling Studies

This directory contains the foundational research papers, empirical scaling laws, and technical reports for **télos** (τέλος), developed by **Wing It Research**.

---

## 📚 Technical Reports & Publications

| Report | Title | Core Focus & Findings |
| :--- | :--- | :--- |
| **[Paper 1: Token-Budget Scaling](paper_1_token_budget_scaling.md)** | *Preliminary Evidence on Token-Budget Scaling in Masked Diffusion Language Models for Code Autocomplete* | MDLM architecture definition, Beta(1.5, 1.5) timestep distribution, token-to-parameter ratio scaling, capability transitions, and hardware acceleration on Apple Silicon (MLX) & Cloud TPU (PyTorch-XLA). |
| **[Paper 2: Capability Divergence & RoPE](paper_2_capability_divergence_and_rope.md)** | *Capability Divergence, Positional Encoding, and the Limits of Masked Diffusion Scaling* | 12.5M, 25M, 50M completed sweeps, CE-Rank divergence phenomenon, structural vs semantic token capacity allocation, post-hoc RoPE fine-tuning regression, and motivation for Uniform Noise (UNDLM) and COROSred. |
| **[COROSred Architecture](corosred_architecture.md)** | *Confidence-Routed Selective Re-Diffusion* | Dual-paradigm unification of Causal Autoregressive drafting ($\alpha$), Uniform Masked Diffusion infilling ($\beta$), and Learned Reliability Head (LRH) guided selective re-diffusion ($\gamma$). |

---

## 🔬 Key Empirical Discoveries

### 1. Scale-Dependent Over-Training Ratios
MDLM token budgets do not scale at a constant parameter-to-token ratio. Larger models absorb higher data volumes before hit with capacity degradation:
- **12.5M Tier**: Peak performance at **1:20** (250M tokens). Diverges at 1:30.
- **25M Tier**: Peak performance at **1:30** (750M tokens).
- **50M Tier**: Peak performance at **1:35** (1.75B tokens). Collapse at 1:40.

### 2. The CE-Rank Divergence Phenomenon
Past a critical over-training ratio, aggregate Cross-Entropy (CE) and Average Target Rank decouple:
- CE continues to decrease due to high-confidence structural predictions (operators, keywords, delimiters).
- Target rank explodes on open-vocabulary semantic predictions (function names, attribute names, class names).
- Fixed representational capacity is starved of semantic bindings without iterative self-correction.

### 3. Rotary Positional Embeddings (RoPE) Dynamics
- **Native RoPE**: Models trained natively with RoPE from initialization achieved record Top-5 accuracies (17.82%).
- **Post-Hoc RoPE Retrofit**: Fine-tuning converged absolute-PE models with RoPE uniformly degrades cross-entropy by **~0.6 nats** across all scales.

---

## 🌐 Online Interactive Visualizations

Visit **[telos.research.wingit.tech](https://telos.research.wingit.tech)** to explore interactive scaling charts, model checkpoints, and live code completion demos.
