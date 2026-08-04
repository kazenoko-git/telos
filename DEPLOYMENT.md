# DEPLOYMENT — télos (τέλος) MDLM

This document details how to set up, reproduce, train, and deploy **télos (τέλος)** — a Masked Diffusion Language Model for Python code autocomplete, natural language instructions, and shell command execution.

---

## 1. Model Phase & Hardware Specifications

| Parameter | Phase A (Local M5 Pro) | **Phase B (Flagship Autocomplete)** | **Phase C (1.08B Flagship Coder)** |
| :--- | :--- | :--- | :--- |
| **Model Size** | **~12.48 Million** ($1.25 \times 10^7$) | **~232.4 Million** ($2.32 \times 10^8$) | **~1.08 Billion** ($1.08 \times 10^9$) |
| **Architecture** | Deep & Narrow (10L, $d=320$) | Deep & Narrow (16L, $d=1152$) | **Deep & Narrow (24L, $d=2048$)** |
| **Attention** | GQA (8 Query, 2 KV) | GQA (16 Query, 4 KV) | **GQA (32 Query, 8 KV)** |
| **Vocabulary** | 4,096 BPE Tokens | 8,192 BPE Tokens | **32,768 BPE Tokens** (Multi-Domain) |
| **Context Length** | 256 tokens | 512 tokens | **512 tokens** |
| **Domain Mixture** | Pure Python | **100% Pure Python Code** | **60% Python, 25% English, 15% Shell** |
| **Token Budget** | 120 Million tokens | **8.0 Billion tokens** (34:1 ratio) | **60.0 Billion tokens** (55:1 overtraining ratio) |
| **Target Steps** | 7,500 steps (~60 mins) | **60,000 steps (~8.0B Tokens, 34:1 to 50:1 ratio)** | **375,000 steps (~1.8 Hours on TPU v6e-1)** |
| **Hardware** | Apple Silicon MPS | Lightning AI TPU v6e-1 / 4x T4 | **Lightning AI TPU v6e-1 / Kaggle TPU** |

---

## 2. Environment Setup

### Prerequisites
- Python >= 3.10
- `uv` package manager (`curl -LsSf https://ast.sh/uv/install.sh | sh`)

### Installation
```bash
git clone https://github.com/kazenoko-git/telos.git
cd telos
uv sync
```

---

## 3. Training Suite

All training is executed through the master script `scripts/train.py` with device auto-detection (`tpu`, `cuda`, `mps`, `cpu`):

### Phase B: Flagship Pure Python Autocomplete Model (~232.4M Params)
```bash
uv run python scripts/prepare_data.py --config configs/phase_b.yaml
uv run python scripts/train_tokenizer.py --config configs/phase_b.yaml
uv run python scripts/train.py --config configs/phase_b.yaml --device tpu
```

### Phase C: TPU v6e-1 Sequential Scaling Suite (125M / 250M / 500M @ Eff Batch 1024)
```bash
uv run python scripts/train.py --config configs/phase_c_tpu_v6e.yaml --device tpu
```

### Local Mac Training (M5 Pro)
```bash
uv run python scripts/train.py --config configs/phase_b_mac.yaml --device mps
```

---

## 4. Master Benchmarking & Evaluation Suite

### Benchmarking
Run throughput measurements and comparative sampler tests:
```bash
uv run python scripts/benchmark.py --checkpoint checkpoints/phase_c_tpu_125m/checkpoint_tpu_125M_final_step_238.pt --mode all
```

### Evaluation & Contextual Probes
Run loss, category cross-entropy breakdown, and contextual probe token rank analysis:
```bash
uv run python scripts/eval.py --checkpoint checkpoints/phase_c_tpu_125m/checkpoint_tpu_125M_final_step_238.pt --mode full
```

---

## 5. Model Publishing & Interactive Gradio Web UI

### HuggingFace Hub Upload
```bash
python -m telos.hub.upload --model-dir checkpoints/phase_c --repo-id kazenoko/telos-1b-coder
```

### Standalone Inference
```python
from telos.hub import TelosModel

model = TelosModel.from_pretrained("checkpoints/phase_c_tpu_125m/checkpoint_tpu_125M_final_step_238.pt")
code = model.complete(
    "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    max_tokens=64,
    num_steps=64,
    temperature=0.3
)
print(code)
```

### Interactive Gradio Application
```bash
uv run python telos/hub/gradio_app.py
```
