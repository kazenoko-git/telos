# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.0] - 2026-11-12

### Added
- `telos pretrain` command with `telos train` retained as a compatibility alias.
- Full-parameter SFT and DPO post-training for native Télos causal checkpoints.
- MLX post-training for Apple Silicon, including PyTorch-to-MLX checkpoint weight conversion.
- CUDA, CPU, and PyTorch-XLA post-training paths; XLA validates rank identity, partitions batches by ordinal, reduces gradients across replicas, and synchronizes checkpoint writes.

### Fixed
- XLA master-rank detection now fails closed instead of assuming the current process is rank zero when rank APIs fail.
- Avoid re-importing the optional MLX runtime in diffusion modules when Metal is unavailable, preventing duplicate native-extension initialization in headless CPU/TPU environments.

## [1.3.2] - 2026-11-12

### Added
- Reproducible, memory-guarded round-two MLX scheduling and microbatch benchmarks, with paired run evidence and a detailed report.

### Performance
- Optional Metal command-buffer limits of 16 operations / 16 MiB reduced measured peak MLX allocation by 19.3% (1.58 to 1.28 GiB) on the existing experimental 50M AR fast path, at similar throughput. The measured +2.8% throughput difference remains within run variation; training defaults are unchanged and long-run convergence is unverified.

## [1.3.1] - 2026-11-12

### Added
- Opt-in MLX compiled backward accumulation and forced fused SDPA attention.
- Experimental packed SwiGLU Metal VJP, retaining existing checkpoint parameter shapes.
- Bounded profiling tools and reports for MLX training throughput, memory use, and GPU execution.

### Performance
- Deferring final accumulated-gradient materialization improves measured throughput by 0.29% while keeping memory bounded.
- A short paired 50M AR benchmark measured 42.0k tokens/s with all experimental options enabled versus 36.5k on defaults (+15.1%); measured peak MLX allocation was 1.59 versus 1.73 GiB. Experimental options remain disabled by default, and the short run does not establish long-run training equivalence.

## [1.3.0] - 2026-11-10

### Added
- Fused Metal cross entropy with a custom first-order backward pass for MLX language-model objectives.
- Compiled MLX optimizer updates and isolated microbatch tuning that preserves the configured effective batch.
- Experimental packed SwiGLU gate/up projection with automatic loading of legacy split checkpoints.
- Experimental tiled tied-embedding AR vocabulary loss, disabled by default.

### Fixed
- Preserve transformer-layer gradients when MLX gradient checkpointing is enabled.
- Bound gradient-accumulation memory by evaluating each microbatch before constructing the next graph.
- Wrap data batches correctly when the requested batch exceeds the token dataset.

## [1.2.0] - 2026-10-21

### Security
- Replaced unsafe `eval()` in tool-use evaluation math parser with an explicit AST arithmetic evaluator (`_eval_ast_math`), eliminating sandbox escape vectors.
- Enforced `weights_only=True` across `torch.load()` checkpoint deserialization in trainer, CLI, and evaluation runners.
- Defaulted `trust_remote_code=False` in HuggingFace model adapter to prevent automatic execution of unreviewed Hub code.
- Passed Google Gemini API keys via `x-goog-api-key` HTTP request headers instead of query parameters.
- Replaced `shell=True` in remote system monitor subprocess invocation.

### Added
- Dedicated PyPI package documentation (`README_PYPI.md`) focusing on installation, hardware acceleration, CLI, and Python API usage.
- Unit regression tests verifying safe arithmetic parsing and rejection of sandbox escape payloads.

## [1.1.0] - 2026-10-20

### Fixed
- TPU replica rank detection now uses `torch_xla.runtime.global_ordinal()`.
  Rank lookup failures abort training instead of silently treating every worker
  as rank zero and duplicating input slices. Invalid replicated ranks are rejected.
- Added regression coverage for distinct eight-worker ranks and data slices,
  rank lookup failures, and out-of-range replica identities.

### Added
- Interleaved COROSred decoding.
- Research tooling and evidence documenting historical TPU rank collapse and
  matched evaluation of the corrected TPU and CUDA checkpoints.

## [1.0.0] - 2026-09-23

First public release on PyPI as `telos-ml` (`pip install telos-ml`).

### Added
- **`telos-ml` PyPI Distribution** — Published package to PyPI under the name
  `telos-ml`, preserving the `telos` Python module import and the `telos` CLI
  command.
- **`telos dataprep`** — prepares raw text, code directories, JSONL, or Hugging
  Face datasets into memory-mapped binary token streams (`.bin`).
- **`telos train`** — zero-config dimensional trainer. Specify parameters and
  tokens; the geometry solver derives `d_model`, layers, and heads. Supports the
  AR, MDLM, UNDLM, and COROSred paradigms on MLX (Apple Silicon), CUDA, TPU via
  PyTorch-XLA, and CPU.
- **COROSred** — the unified dual-paradigm trainer, combining causal AR drafting,
  uniform masked-diffusion infilling, and confidence-guided selective
  re-diffusion, with a Learned Reliability Head driving routing.
- **`telos eval`** — multi-domain evaluation: contextual probes with bootstrap
  confidence intervals, sandboxed functional execution (Pass@1, AST validity),
  anti-cheat span masking, linguistic probes, and tool-use / function-calling
  benchmarks.
- **`telos bench`** — hardware throughput and step-latency benchmark, capped at
  five minutes.
- **`telos test`** — unified verification suite for model contracts, causality,
  losses, and samplers.
- **`telos.afm`** — Python access to Apple's on-device Foundation Models
  (AFM 3 Core / Core Advanced) on Apple Silicon. Exposes `probe()`, `require()`,
  `generate()`, and a persistent `AFMBridge`, plus a `telos afm status` /
  `telos afm generate` CLI. The Swift bridge is compiled on first use into a user
  cache directory.
- Bundled default ByteLevel BPE tokenizer, so an installed package works with no
  manual downloads.
- Graceful degradation when MLX is installed but Metal is unavailable (headless,
  virtualized, or sandboxed macOS sessions): `telos train`, `telos bench` and
  `telos test` fall back to the PyTorch CPU path instead of aborting the process.

[Unreleased]: https://github.com/kazenoko-git/telos/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/kazenoko-git/telos/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/kazenoko-git/telos/releases/tag/v1.0.0
