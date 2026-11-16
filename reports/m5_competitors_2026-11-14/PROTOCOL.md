# M5 Pro competitor tests — 14 November 2026

## Scope and order

Use a clean checkout of Télos v1.5.5 plus documentation changes. Preserve the
original checkout's uncommitted optimization experiments. Run GPU jobs serially.
Commit each script, protocol change, result bundle, and report separately.

First compare the Télos MLX backbone with the MLX-LM Llama backbone used by
MLX-Pretrain. Match architecture, initial tensors, corpus tokens, loss,
optimizer, precision, microbatch, accumulation, and completed update count.
Use a common training harness to isolate model implementation differences.
This is an implementation comparison, not a full-toolkit speed ranking.

Then measure native trainer workflows separately. Apple MLX Examples' stock
transformer uses sinusoidal positions, LayerNorm, and a different feed-forward
network; its raw throughput cannot establish a matched Télos speed advantage.
LitGPT/nanochat device support must be checked before planning native M5 runs.
Evaluation harness parity remains a separate experiment using identical saved
completions and task/scoring policies.

## Controlled implementation comparison

- Apple M5 Pro, 24 GB unified memory, AC power; record OS/runtime/power settings.
- MLX 0.32.2 in an isolated environment; record all installed package versions.
- 8,192 vocabulary, width 512, 14 layers, 8 equal query/KV heads, hidden MLP
  width 1,408, tied embeddings, RMSNorm epsilon 1e-5, traditional RoPE base 10,000.
- BF16 model tensors; FP32 master weights and AdamW moments, LR 4e-4,
  betas 0.9/0.999, epsilon 1e-8, weight decay 0.01, global norm clipping 1.0.
- Shared initial weights mapped explicitly; fail on missing/extra tensors or
  parameter count mismatch. Check pre-training FP32 forward and gradient parity
  before interpreting speed. Record BF16 initial loss differences.
- Real corpus: a hashed prefix of `data/english_corpus_5b.bin` (uint16 IDs).
  256-token sequences, microbatch 8, accumulation 8: 16,384 input tokens and
  16,320 next-token targets per update. No special-token masking in this shared
  loss; therefore this differs from the native Télos objective's masking policy.
- Start with a short smoke run, then fresh processes in A/B/B/A order, 64
  updates each; exclude 12 warmup updates. Run longer only after these pass.
- Both implementations process the same full input sequence, then discard the
  final logits. Common compiled microbatch loss/gradient function and bounded
  accumulation; materialize all model, master and optimizer arrays per update.
- Report completed input tokens/s and target tokens/s, update timings, MLX peak
  allocation, process peak RSS, wall time, initial/final held-out loss, and failures.
  MLX allocation and process RSS are different memory counters.
- Advisory 3.2 GiB MLX allocation guard and stop on low system memory; capture
  whole-system memory pressure, power settings, and thermal status where available.

## Interpretation

Short runs establish execution and local throughput, not convergence or time to
useful model quality. The shared harness does not exercise either toolkit's
data preparation, checkpoint/resume, or evaluator. Report these limits alongside
every result. Never pool changing contexts, runtime versions, or architectures.

## Pinned upstream sources

- MLX Examples: `796f5b53cab69a3d48a44233ce21aae889e94a08`.
- MLX-Pretrain: `8f5bb3c7a5c407de9e7ccdb2a8b25c0e585cc436`.
- MLX-Pretrain `train.py:setup_model` imports `mlx_lm.models.llama` when no
  local architecture module exists. Record the installed MLX-LM version and
  source hashes with each controlled run.
