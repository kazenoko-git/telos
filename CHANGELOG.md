# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-09-22

First public release.

### Added
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

[Unreleased]: https://github.com/kazenoko-git/telos/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/kazenoko-git/telos/releases/tag/v1.0.0
