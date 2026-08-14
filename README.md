# télos (τέλος) — Small Masked Diffusion Language Model for Code Autocomplete

**télos** (or **τέλος**) is a Masked Diffusion Language Model (MDLM) built and trained from scratch, specialized for narrow-domain Python code autocomplete. Unlike traditional autoregressive (AR) language models that generate code left-to-right with causal attention, télos utilizes full **bidirectional self-attention** and an iterative absorbing-state diffusion process with **Beta(1.5, 1.5)** schedule masking to complete code blocks in parallel.

---

## Technical Highlights

- **Architecture**: Decoder-style Transformer with full bidirectional self-attention (no causal mask).
- **Framework Support**: Dual backend support for **PyTorch / PyTorch-XLA (TPU)** and **Apple MLX (Metal GPU)** optimized training & inference.
- **Positional Encoding**: Rotary Positional Embeddings (RoPE).
- **Activation & Norm**: SwiGLU ($\approx 2.67\times$ expansion) with RMSNorm.
- **Embeddings**: Weight-tied input/output token embeddings.
- **Masking Schedule**: Flexible **Beta($\alpha, \beta$)** continuous masking schedule (default $\text{Beta}(1.5, 1.5)$) alongside standard linear/cosine schedules.
- **Objective**: Masked Cross-Entropy reweighted by $1/t$ where $t \sim \text{Uniform}(0, 1)$ or $t \sim \text{Beta}(1.5, 1.5)$ (ELBO-consistent weighting).
- **Sampling**: Confidence-based iterative unmasking over configurable denoising steps (16–128 steps) with Gumbel noise temperature control and repetition penalties.
- **Evaluation Suite**: 101 contextual probes across 8 categories (Imports, Keywords, Operators, Literals, Function Names, Class Names, Attribute Names, Identifier Recovery) measuring rank and cross-entropy metrics.

---

## Model Scaling Suites & Configs

The repository includes pre-configured scaling study definitions across model sizes and parameter-to-token ratios:

| Model Suite | Parameters | Target Ratios ($N:D$) | Framework | Config Location |
| :--- | :--- | :--- | :--- | :--- |
| **12M Suite** | ~12.5M | 1:1, 1:5, 1:10, 1:15, 1:20, 1:25 | MLX / Metal | `configs/phase_b_12m_*_mlx.yaml` |
| **25M Suite** | ~25M | 1:1, 1:3, 1:5, 1:10, 1:15, 1:20, 1:25, 1:30 | MLX / Metal | `configs/phase_b_25m_*_mlx.yaml` |
| **50M Suite** | ~50M | 1:1, 1:10, 1:15, 1:20, 1:25, 1:30, 1:35, 1:40, 1:45 | MLX / TPU | `configs/phase_b_50m_*_mlx.yaml` |

---

## Project Structure

```
telos/
├── DEPLOYMENT.md                  # Comprehensive reproduction and deployment guide
├── pyproject.toml                 # Dependencies and build configuration
├── configs/                       # YAML configurations for 12M, 25M, and 50M parameter scaling suites
├── notebooks/                     # Interactive Jupyter notebooks for training and evaluation
│   ├── Training_Suites.ipynb      # Main execution notebook for 50M/25M training pipelines
│   ├── Evaluation.py              # CLI and module for running the 101-probe benchmark suite
│   ├── Evaluation.ipynb           # Notebook version of evaluation and probing benchmarks
│   └── Benchmarking.ipynb         # Throughput & latency benchmarking
├── telos/
│   ├── model/                     # PyTorch & MLX bidirectional transformers (RoPE, RMSNorm, SwiGLU)
│   ├── diffusion/                 # Forward masking, loss functions, confidence-based MDLMSampler
│   ├── data/                      # BPE tokenizer, dataset preparation scripts, dataset matrix loader
│   ├── training/                  # Unified PyTorch, PyTorch-XLA (TPU), and MLX trainers
│   ├── eval/                      # Metrics computation and qualitative sampling scripts
│   └── hub/                       # Standalone inference API (`TelosModel.from_pretrained`)
├── probes_output/                 # Benchmark scorecards and detailed probe analysis outputs
└── scripts/                       # Command-line entry points for training, sampling, and data preparation
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/kazenoko/telos.git
cd telos
pip install -e .
```

For Apple Silicon MLX GPU support:
```bash
pip install mlx tokenizers torch pyyaml
```

### High-Level Standalone Inference

```python
from telos.hub import TelosModel

# Load model from local checkpoint directory or HuggingFace Hub
model = TelosModel.from_pretrained("checkpoints/phase_b_50m_1to35_mlx_20260811_231951")

prompt = """def binary_search(arr: list[int], target: int) -> int:
    \"\"\"Perform binary search on a sorted list.\"\"\"
"""

# Execute iterative unmasking completion
completion = model.complete(
    prompt=prompt,
    max_tokens=128,
    num_steps=64,
    temperature=0.3,
    schedule="linear"
)
print(completion)
```

### Running Evaluation & Probes

To evaluate a trained checkpoint against the 101 contextual probes suite:

```bash
python notebooks/Evaluation.py --checkpoint checkpoints/phase_b_50m_1to40_mlx --mode probes
```

---

## License

Apache-2.0
