# télos (τέλος)

<p align="center">
  <img src="logos/telos_logo.png" alt="Télos logo" width="300">
</p>

<p align="center">
  <a href="https://telos.wingit.tech"><strong>telos.wingit.tech</strong></a>
  · <a href="https://github.com/kazenoko-git/telos">GitHub</a>
  · <a href="https://pypi.org/project/telos-ml/">PyPI</a>
</p>

**Train language models, then benchmark what they learned.** Télos brings data preparation, training workflows, and model evaluation together, with support for autoregressive and discrete-diffusion research across Apple Silicon, NVIDIA GPUs, and Google Cloud TPUs.

## Why choose Télos?

Télos is for people who want to run and study language-model training processes, rather than focus mainly on adapting an existing chat model. Prepare data, configure a training run, choose an objective and backend, then evaluate the resulting model in the same toolkit. Model-size and token budgets are useful controls; they are not the product's central purpose.

Télos supports training workflows across its supported objectives and hardware backends; the current research mainline is autoregressive modeling, with masked- and uniform-noise-diffusion research also represented in the codebase. Its benchmark and evaluation suite tests model behavior—not just training loss—including contextual code probes, executable code tasks, anti-cheat checks, reasoning, tool-use, cybersecurity, and other task suites. Post-training workflows such as SFT, preference optimization, and RL are outside Télos's core focus.

### How it compares

Télos overlaps with general training toolkits, focused end-to-end training projects, and Apple Silicon training stacks. The best fit depends on whether you need a particular training workflow, research flexibility, or a standard evaluation ecosystem.

| Tool | Reach for it when… | Télos is a better fit when… |
| :--- | :--- | :--- |
| [LitGPT](https://github.com/Lightning-AI/litgpt) | You want a broad PyTorch toolkit for pretraining, continued pretraining, fine-tuning, and evaluation across many supported model architectures. Its evaluation command connects to the EleutherAI LM Evaluation Harness. | You want Télos's training and evaluation workflows together, including its custom and public task suites and support for multiple training objectives/backends. |
| [nanochat](https://github.com/karpathy/nanochat) | You want a compact, hackable end-to-end GPT-style training pipeline, with a public time-to-CORE quality target. | You want a broader research workflow across supported training objectives and hardware, plus Télos's multi-domain model evaluation. |
| [MLX Examples](https://github.com/ml-explore/mlx-examples) / [MLX-Pretrain](https://github.com/N8python/mlx-pretrain) | You want Apple-Silicon-focused examples or pretraining workflows built around MLX. | You want a training and evaluation toolkit that also targets CUDA and TPU backends, with task suites for comparing resulting model behavior. |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | You want to run or fine-tune language models locally on Apple Silicon. | You want the broader training research workflow and evaluation suite; MLX-LM's main focus is generation and fine-tuning. |
| [Unsloth](https://unsloth.ai/docs) | You want guided, memory-efficient workflows for fine-tuning, continued pretraining, full pretraining, or RL on supported models. | You want a research toolkit centered on training-process comparisons and evaluating models on Télos's task suites. |
| [Axolotl](https://docs.axolotl.ai/) | You want configurable fine-tuning and post-training recipes across supported models. | You want the training-research workflow and Télos's custom/public benchmark suites in one project. |

Télos's evaluation suite is not limited to Télos-trained models. It can evaluate external models through supported adapters and run both custom tasks and public benchmark suites. For standardized public scores, LitGPT's LM Evaluation Harness integration and standalone harnesses such as [LightEval](https://huggingface.co/docs/lighteval/main/index) offer large task catalogs and established comparison workflows. Télos adds its own domain-focused suites, including sandboxed functional code evaluation. Benchmark data is generated or fetched separately from the evaluator; this is a setup requirement, not a missing evaluation capability.

## What you can do with Télos

- **Get from corpus to training run:** prepare text, code, JSONL, or Hugging Face datasets as token streams, then train with a parameter budget and token target instead of assembling a training stack by hand.
- **Run on the hardware you have:** use Apple Silicon through MLX, NVIDIA GPUs through PyTorch/CUDA, Google Cloud TPUs through PyTorch-XLA, or CPU for compatible workflows.
- **Benchmark trained and external models:** run custom and public task suites for contextual code behavior, sandboxed code execution, reasoning, tool-use, cybersecurity, and other domains; inspect and save per-task results. Some suites must first be generated or fetched with the benchmark builder.
- **Measure training throughput:** use the separate `telos bench` command for short hardware-throughput measurements. This is distinct from evaluating model quality on benchmark tasks.

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

Build or fetch the evaluation datasets as needed, then run an evaluation:

```bash
python scripts/build_evaluation_benchmarks.py --all
telos eval --checkpoint checkpoints/ar/model.safetensors --type all --mode full
```

Run the short hardware benchmark separately:

```bash
telos bench --paradigm ar --params 50M --hardware mlx
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

Télos runs its own evaluation suite against external models as well as its own checkpoints. The full [benchmark report](benchmarks/README.md) includes evaluation methods, per-suite results, repetition analysis, and model reports. Its model-evaluation throughput figures describe model execution; `telos bench` separately measures training throughput.

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
