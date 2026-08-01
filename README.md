# télos (τέλος) — Small Masked Diffusion Language Model for Code Autocomplete

**télos** (or **τέλος**) is a Masked Diffusion Language Model (MDLM) built and trained from scratch, specialized for narrow-domain Python code autocomplete. Unlike traditional autoregressive (AR) language models that generate code left-to-right with causal attention, télos utilizes full **bidirectional self-attention** and an iterative absorbing-state diffusion process to complete code blocks.

---

## Technical Highlights

- **Architecture**: Decoder-style Transformer with full bidirectional self-attention (no causal mask).
- **Positional Encoding**: Rotary Positional Embeddings (RoPE).
- **Activation**: SwiGLU ($\approx 2.67\times$ expansion).
- **Normalization**: RMSNorm.
- **Embeddings**: Weight-tied input/output embeddings.
- **Time Conditioning**: Omitted (time-agnostic optimal ELBO per RADD/MDLM findings).
- **Objective**: Masked Cross-Entropy reweighted by $1/t$ where $t \sim \text{Uniform}(0, 1)$ (ELBO-consistent weighting).
- **Sampling**: Confidence-based iterative unmasking over configurable denoising steps (16–128 steps).

---

## Project Structure

```
telos/
├── DEPLOYMENT.md                  # Detailed reproduction and deployment guide
├── pyproject.toml                 # Dependencies and build configuration
├── configs/
│   ├── phase_a.yaml               # Phase A local validation config (~3M params)
│   └── phase_b.yaml               # Phase B H100 production config (~85M params)
├── telos/
│   ├── model/                     # Bidirectional transformer, RoPE, RMSNorm, SwiGLU
│   ├── diffusion/                 # Forward masking process, 1/t loss, iterative sampler
│   ├── data/                      # BPE tokenizer, dataset loader, function extractor
│   ├── training/                  # PyTorch trainer, warmup+cosine decay, checkpointing
│   ├── eval/                      # Held-out perplexity, qualitative code sampling
│   └── hub/                       # HuggingFace Hub export & standalone inference package
├── scripts/                       # Executable entry points (train, sample, prepare)
└── tests/                         # Comprehensive unit test suite
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/kazenoko/telos.git
cd telos
pip install -e .
```

### Inference Example

```python
from telos.hub import TelosModel

# Load model from HuggingFace Hub or local checkpoint
model = TelosModel.from_pretrained("kazenoko/telos-85m")

prompt = """def binary_search(arr, target):
    \"\"\"Perform binary search on a sorted list.\"\"\"
"""

# Run iterative unmasking denoising
completion = model.complete(prompt, max_tokens=128, num_steps=64, temperature=0.8)
print(completion)
```

---

## License

Apache-2.0
