# Télos CLI & Deployment Guide (`telos`)

This document is the official command line interface (CLI) and package deployment guide for **τέλος** (télos). It details installation, data preparation, zero-config model training, multi-domain evaluation, hardware benchmarking, Apple Foundation Models (`telos afm`), and PyPI distribution.

---

## 1. Installation

Install Télos directly via `pip` or `uv`:

```bash
# Standard installation
pip install telos-ml

# With Apple Silicon Metal support (MLX)
pip install "telos-ml[mlx]"

# Complete development environment
pip install "telos-ml[all]"
```

For local development in editable mode:
```bash
git clone https://github.com/kazenoko-git/telos.git
cd telos
uv sync
uv pip install -e .
```

Verify installation:
```bash
telos --version
telos --help
```

---

## 2. Master CLI Commands Overview (`telos <command>`)

Télos provides a unified command line interface for data preparation, pretraining, post-training, evaluation, and utilities:

| Command | Module | Description |
| :--- | :--- | :--- |
| **`telos dataprep`** | `telos.dataprep` | High-efficiency data processing for raw text, code directories, JSONL, or Hugging Face datasets into chunked binary memory-mapped arrays (`.bin`). |
| **`telos pretrain`** | `telos.train` | Model pretraining across supported objectives (AR, MDLM, UNDLM, COROSred, custom). `telos train` remains an alias. |
| **`telos posttrain`** | `telos.posttrain` | Full-parameter supervised fine-tuning (SFT) or direct preference optimization (DPO) for native causal Télos checkpoints. |
| **`telos eval`** | `telos.eval` | Multi-domain evaluation suite across 100 contextual probes, 512 functional code execution tasks, anti-cheat span masking, linguistics, and tool-use. |
| **`telos bench`** | `telos.bench` | Hardware throughput, step latency, and memory benchmark engine strictly capped at 5 minutes. |
| **`telos afm`** | `telos.afm` | Apple on-device Foundation Models: availability status and text generation on Apple Silicon. |
| **`telos test`** | `telos.testing` | Unified verification suite verifying model contracts, causality, losses, and samplers. |

---

## 3. Model Pretraining (`telos pretrain`)

User directly specifies the fundamental training dimensions from the command line:

### The 6 Fundamental Training Dimensions

1. **Amount of Parameters**: `--params` (e.g. `12M`, `25M`, `50M`, `100M`, `500M`, or raw integer). An analytical geometry solver automatically computes optimal $(d_{\text{model}}, n_{\text{layers}}, n_{\text{heads}})$. Custom architectures can be explicitly specified via `--d-model` and `--n-layers`.
2. **Amount of Training Tokens**: `--tokens` (e.g. `2.5B`, `300M`, `50M`). Total steps are automatically calculated from effective batch tokens per step.
3. **Batch Size**: `--effective-batch` (sequences or token count) with automatic gradient accumulation calculation, OR direct `--batch-size` + `--grad-accum`.
4. **Tokenizer**: `--tokenizer` (path to custom JSON, Hugging Face model, or default) with automatic `--vocab-size` inference.
5. **Hardware Target**: `--hardware` (`auto`, `mlx`, `cuda`, `mps`, `xla`, `cpu`). Auto-detects Apple Silicon Metal, NVIDIA GPUs, or Cloud TPUs.
6. **Hardware Count**: `--devices` (e.g. `1`, `4`, `8`, or `auto` for all available devices).

### Automatic Training Dynamics
- **Max LR**: Auto-scaled with model width: $\text{max\_lr} = 6.0 \times 10^{-4} \times \sqrt{256 / d_{\text{model}}}$.
- **Min LR**: Auto-calculated as $0.1 \times \text{max\_lr}$ (standard cosine floor).
- **Warmup Steps**: Auto-calculated as $\max(50, \min(2000, 0.02 \times \text{max\_steps}))$.
- **Weight Decay**: Default $0.1$.
- **PyTorch Compilation**: `--compile` / `--no-compile` toggles `torch.compile` kernel fusion on CUDA.

### Cadence & Device Synchronization Controls
- `--default-cadence`: Base step cadence across all cadence operations (defaults to **25** on TPU/XLA, and **1** on CUDA/CPU).
- `--lr-cadence`: Step frequency for learning rate updates.
- `--sched-step` / `--sched-cadence`: Step frequency for COROSred dynamic schedule updates.
- `--cadence`: Step frequency for cross-replica metric synchronization and EMAs.

### CLI Training Examples

```bash
# 1. Train a 25M MDLM model on 300M tokens on Apple Silicon (MLX)
telos pretrain --paradigm mdlm --params 25M --tokens 300M --effective-batch 32

# 2. Train a 50M UNDLM model on 4x NVIDIA GPUs (CUDA) with torch.compile
telos pretrain --paradigm undlm --params 50M --tokens 500M --hardware cuda --devices 4 --compile

# 3. Train Unified COROSred 50M on Cloud TPU (8 cores)
telos pretrain --paradigm corosred --params 50M --tokens 2.5B --causal-ratio 0.65 --mask-prob 0.20 --hardware xla --devices 8 --data data/python_corpus_5b.bin

# 4. Train 100M AR Baseline on 5B Tokens on Cloud TPU
telos pretrain --paradigm ar --params 100M --tokens 5B --hardware xla --devices 8 --data data/python_corpus_5b.bin
```

### Resuming pretraining

Resume from a complete checkpoint without repeating the original configuration:

```bash
# Apple Silicon / MLX
telos pretrain --resume checkpoints/ar/checkpoint_step_1000.resume.npz

# PyTorch CPU or a single CUDA device
telos pretrain --resume checkpoints/ar/checkpoint_step_1000.pt
```

Télos restores the model, optimizer moments, fp32 master weights when used,
learning-rate position, random generators, diffusion sampling state, and the next
unconsumed data row. COROSred also restores its metric tracker and routing cache
on PyTorch. Checkpoints are written atomically so a failed save preserves the
previous complete file. MLX continues to export `.safetensors` weights for evaluation;
its separate `.resume.npz` checkpoint contains the complete training state.

Resume uses the saved architecture, objective, batching, seed, and original total
step budget. `--checkpoint-dir` can select a new output directory. `--data` can
relocate the same corpus; its token contents, shape, and dtype must match the saved
fingerprint. Changed data, tokenizers, backends, and incompatible configurations
are rejected. Custom tokenizer files must remain available at their recorded paths.
Data fingerprints read the full corpus once when training starts.

v1.5.5 supports pretraining resume with the original CUDA/XLA worker topology.
Keep every rank checkpoint together: rank zero uses the normal `.pt` filename;
other workers use `.rankN.pt`. Each worker restores its own RNG, optimizer and cursor.
A topology change or missing rank file aborts resume. XLA stochastic updates use
step-indexed seeds to avoid depending on graph-tracing RNG consumption; their
physical TPU continuation retest is still outstanding. CPU interruption tests are bit-exact; native MLX tests
verify state restoration and continuation, with rounding tolerance for parallel GPU
reductions. Identical numerical results across hardware or runtime versions are not
guaranteed. CUDA and single-device XLA paths require hardware validation.

Only checkpoints written with complete training state can resume.
Stochastic XLA continuation across the v1.5.5 seeding change is not equivalent to
an uninterrupted older run; begin a new run to validate the new seed strategy. Older
checkpoints and exported weights can initialize a new run with `--init-checkpoint`.
The legacy `--resume-step N` syntax resolves a complete checkpoint in the selected
checkpoint directory; it no longer skips updates with fresh training state.
After an interruption, resume the most recent completed checkpoint; work after that
checkpoint is repeated. `--save-every 0` disables intermediate saves.

---

## 4. Post-training (`telos posttrain`)

SFT records are JSONL objects with `prompt` and `response` strings. DPO records use `prompt`, `chosen`, and `rejected`. The command performs full-parameter updates and writes native checkpoints that can be evaluated with `telos eval`.

```bash
# Supervised fine-tuning
telos posttrain --method sft --checkpoint checkpoints/ar/checkpoint_final.pt \
  --data data/instructions.jsonl --output checkpoints/ar-sft

# Direct preference optimization
telos posttrain --method dpo --checkpoint checkpoints/ar-sft/checkpoint_final.pt \
  --data data/preferences.jsonl --output checkpoints/ar-dpo
```

Choose `--hardware mlx`, `cuda`, `xla`, or `cpu` (default: `auto`). MLX runs natively on Apple Silicon; PyTorch MPS is not used. CUDA/XLA/CPU consume native PyTorch checkpoints, while MLX accepts native MLX or PyTorch checkpoints. DPO retains a frozen reference copy of the starting checkpoint during training. XLA launches all visible TPU devices, checks each worker's rank, partitions global batches by ordinal, and coordinates gradient reduction and checkpoint saves. A requested device count that differs from the runtime fails before training. For TPU post-training, use `--hardware xla --devices auto` to require the complete runtime topology.

---

## 5. Multi-Domain Evaluation (`telos eval`)

```bash
# 1. Run Complete Multi-Domain Evaluation (Code + English Linguistics + Tool-Use)
telos eval --checkpoint checkpoints/corosred/model.safetensors --type all --mode full

# 2. Python Code Contextual Probes Suite (1,000 contextual probes with 95% Bootstrap CI)
telos eval --checkpoint checkpoints/corosred/model.safetensors --type code --language python --mode probes --num-probes 1000

# 3. Functional Execution Benchmark (512 sandboxed tasks with Pass@1, AST validity)
telos eval --checkpoint checkpoints/corosred/model.safetensors --type code --mode functional --suite private_unseen

# 4. Anti-Cheat & Suffix-Copy Span Masking Suite (K in {1, 2, 4, 8, 16})
telos eval --checkpoint checkpoints/corosred/model.safetensors --type code --mode anticheat

# 5. Native Apple Foundation Models (AFM-3 Core Advanced) Evaluation
telos eval --model afm-3-core-advanced --backend swift --type all

# 6. IBM Granite 4.2 3B Evaluation
telos eval --model ibm-granite-4.2-3b --backend mlx --type all
```

---

## 5. Hardware Benchmarking (`telos bench`)

Strictly capped at 5 minutes to prevent compute waste:

```bash
# Benchmark 50M COROSred model throughput on Apple Silicon
telos bench --paradigm corosred --params 50M --hardware mlx

# Benchmark 100M AR baseline on CUDA
telos bench --paradigm ar --params 100M --hardware cuda
```

---

## 6. Apple Foundation Models (`telos afm`)

Access Apple's on-device Foundation Models (AFM 3) on Apple Silicon:

```bash
# Check on-device model availability
telos afm status

# Generate completion via native Swift bridge
telos afm generate "Explain rotary position embeddings in two sentences."
```

Python API:
```python
import telos.afm

# Probe availability
status = telos.afm.probe()
print("AFM Available:", status.available)

# Generate completion
text = telos.afm.generate("Write a haiku about gradients.")
print(text)
```

---

## 7. Python API Overview

```python
import telos

# 1. Prepare Dataset
telos.prepare_dataset(
    input_path="data/raw_python/",
    output_bin="data/python_corpus.bin",
    vocab_size=8192
)

# 2. Train Model
telos.run_train(
    paradigm="corosred",
    params="50M",
    tokens="2.5B",
    hardware="mlx"
)

# 3. Evaluate Checkpoint
results = telos.evaluate(
    checkpoint="checkpoints/corosred/model.safetensors",
    benchmark_type="code",
    mode="full"
)
print("Pass@1:", results["functional"]["pass_at_1_pct"])
```

---

## 8. Packaging & PyPI Distribution

```bash
# Build wheels and source distribution
rm -rf dist
uv build

# Inspect archive contents
python -m zipfile -l dist/telos_ml-*.whl
tar tzf dist/telos_ml-*.tar.gz

# Rehearse in isolated smoke venv
uv venv /tmp/telos-smoke
uv pip install --python /tmp/telos-smoke/bin/python \
  --extra-index-url https://pypi.org/simple/ dist/telos_ml-1.0.0-py3-none-any.whl
/tmp/telos-smoke/bin/telos --version
/tmp/telos-smoke/bin/telos afm status

# Publish to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/

# Publish to PyPI
uv publish
```

### Resuming SFT and DPO (v1.5.5)

Post-training writes full `.resume.pt` (CPU/CUDA/XLA) or `.resume.npz` (MLX)
checkpoints alongside exported model weights. Repeat the original source checkpoint,
data and training options, adding the full resume checkpoint:

```bash
telos posttrain --method dpo --checkpoint checkpoints/ar/checkpoint_final.pt \
  --data data/preferences.jsonl --output checkpoints/ar-dpo-resumed \
  --hardware cuda --devices 2 --max-steps 1000 --batch-size 4 \
  --resume checkpoints/ar-dpo/checkpoint_step_250.resume.pt
```

The model, optimizer, random generators, sampler and frozen DPO reference are
restored. Data, tokenizer, run settings and worker identity are checked before
continuation. XLA writes one resume file per rank. CUDA post-training uses one
process with DataParallel across the selected GPUs; batch size is per GPU.
`--devices auto` selects all visible CUDA GPUs. MLX remains a single-device backend.

CPU and native MLX SFT/DPO interruption tests pass. Multi-GPU CUDA and physical
TPU post-training/resume remain hardware-unverified. Simulated rank tests do not
establish collective or physical multi-device correctness.
