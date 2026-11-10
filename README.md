# télos (τέλος)

<p align="center">
  <img src="logos/telos_logo.png" alt="Télos logo" width="300">
</p>

<p align="center">
  <a href="https://telos.wingit.tech"><strong>telos.wingit.tech</strong></a>
  · <a href="https://github.com/kazenoko-git/telos">GitHub</a>
  · <a href="https://pypi.org/project/telos-ml/">PyPI</a>
</p>

**Build, train, and evaluate language models with one Python toolkit.** Télos helps developers and researchers prepare text data, run autoregressive or discrete-diffusion experiments, and compare model behavior across hardware backends.

## Why choose Télos?

Télos is for people who want to **study how language models are trained**, rather than primarily adapt an existing chat model. Set a model-size and token budget, choose a training objective and hardware backend, then prepare data, train, benchmark, and evaluate from one toolkit. Its parameter-budget solver fills in a compatible transformer shape, so experiments can start from the research question instead of hand-calculating layer dimensions.

The distinct research angle is comparing autoregressive training with masked- and uniform-noise-diffusion objectives in the same codebase. Training targets include Apple Silicon (MLX), NVIDIA GPUs (CUDA), and Google Cloud TPUs (XLA); the current research mainline is autoregressive modeling. Télos also includes contextual probes and task evaluations so experiments can measure more than training loss.

### How it compares

These tools are often easier choices when the job is adapting a capable model that already exists. Télos fits a different job: building small models from a controlled configuration and testing training-objective or scaling questions.

| Tool | Reach for it when… | Télos is a better fit when… |
| :--- | :--- | :--- |
| [Unsloth](https://unsloth.ai/docs) | You want a guided, memory-efficient way to fine-tune or continue-pretrain supported open models, use RL workflows, or export and run the result. Unsloth also supports full pretraining. | You want parameter and token budgets to drive a compact from-scratch model setup, and want to compare AR with discrete-diffusion objectives. |
| [Axolotl](https://docs.axolotl.ai/) | You want configurable post-training recipes such as SFT, LoRA/QLoRA, preference optimization, or RL across supported models. | You want model-size/token budgets and pretraining-objective comparisons at the center of the research workflow, with Télos's evaluation suites alongside it. |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | You want to run or fine-tune language models locally on Apple Silicon. | You want one research codebase spanning MLX, CUDA, and TPU training, or you want to study both autoregressive and discrete-diffusion objectives. |

For the common “fine-tune this existing model on my data” task, start with Unsloth or Axolotl. Choose Télos when the question is “what happens if I change the model size, token budget, or training objective?”

## What you can do with Télos

- **Get from corpus to training run:** prepare text, code, JSONL, or Hugging Face datasets as token streams, then train with a parameter budget and token target instead of assembling a training stack by hand.
- **Run on the hardware you have:** use Apple Silicon through MLX, NVIDIA GPUs through PyTorch/CUDA, Google Cloud TPUs through PyTorch-XLA, or CPU for compatible workflows.
- **Compare model behavior:** run contextual probes, code execution, reasoning, tool-use, cybersecurity, and other evaluation suites; save reports for later analysis.

## Get started

Install the package:

```bash
pip install telos-ml
```

For Apple Silicon MLX support, install the optional extra:

```bash
pip install "telos-ml[mlx]"
```

Prepare a corpus and start an autoregressive training run:

```bash
telos dataprep --corpus ./my_text_or_code --output data/corpus.bin
telos train --paradigm ar --params 50M --tokens 300M --data data/corpus.bin
```

Explore the CLI with `telos --help` or see the [CLI guide](cli/README.md) for data preparation, training, evaluation, and hardware benchmark options.

### Python API

```python
from telos.models import TelosConfig, TelosTransformer

config = TelosConfig(
    vocab_size=8192,
    d_model=512,
    n_layers=8,
    n_heads=8,
    n_kv_heads=8,
    max_seq_len=512,
    is_causal=True,
)
model = TelosTransformer(config)
```

## Research with Télos

Télos supports experiments in autoregressive language modeling and discrete diffusion, including masked and uniform-noise diffusion. The research program has examined training data budgets, scaling behavior, positional encodings, and evaluation methods. The current training and research mainline is autoregressive modeling.

Browse the [research index](research/README.md) for papers, reports, and experiment notes. See [deployment and hardware guidance](DEPLOYMENT.md) for setup details.

## Model evaluations

We evaluated three compact models across seven suites. Scores below are Pass@1; the macro average gives each suite equal weight. These results describe the evaluated configurations and suites, not a universal ranking of the models.

| Suite | Tasks | Apple AFM-3 Core Advanced | IBM Granite 4.2 3B | LiquidAI LFM 8B A1B |
| :--- | ---: | ---: | ---: | ---: |
| HumanEval | 164 | **62.80%** | 60.37% | 45.73% |
| BFCL tool-use | 30 | **73.33%** | 70.00% | 63.33% |
| Cybersecurity auditing | 50 | 12.00% | 22.00% | **24.00%** |
| GPQA Diamond | 198 | **39.39%** | 28.79% | 20.71% |
| MMLU Science | 119 | 69.75% | 61.82% | **73.11%** |
| Competition MATH | 140 | 52.00% | **64.71%** | 43.57% |
| ARC-Challenge | 150 | 85.32% | **86.60%** | 59.56% |
| **Macro average** | **841** | **56.37%** | **56.33%** | **47.14%** |

<p align="center">
  <img src="figures/benchmark_radar_graph.png" alt="Seven-suite comparison of Apple AFM-3, IBM Granite, and LiquidAI LFM" width="600">
</p>

The full [benchmark report](benchmarks/README.md) includes methods, throughput and memory measurements, repetition analysis, and per-model reports.

## Project guides

| Guide | What it covers |
| :--- | :--- |
| [CLI guide](cli/README.md) | Data preparation, training, evaluation, and hardware benchmarking |
| [Research index](research/README.md) | Papers, scaling studies, and architecture notes |
| [Benchmark reports](benchmarks/README.md) | Model comparisons, evaluation details, and performance measurements |

## Citation

```bibtex
@article{samuel2026telos,
  title   = {télos: Exploring Scaling Laws, Hardware Optimizations, and Paradigm Trade-offs in Discrete Diffusion and Autoregressive Language Models},
  author  = {Ivan Samuel},
  journal = {Wing It Research},
  year    = {2026},
  url     = {https://telos.wingit.tech}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
