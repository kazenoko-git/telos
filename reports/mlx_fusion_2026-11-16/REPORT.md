# MLX projection, attention backward and QKV investigation

Measured on 2026-11-16, Apple M5 Pro / 24 GiB, AC power with High Power Mode
configured (`powermode 2`). These are workload-specific measurements, not
convergence or model-quality results. No experimental option is enabled by
default by this work.

## Projection/loss correctness

The historical opt-in vocabulary-tiled loss calculated its partition from BF16
GEMM logits but its target score from an FP32 dot product. The unchanged method
from `f558146` reproduces CE **-0.109375** where the full projection gives zero.
A separate confident-target case loses the BF16 target gradient entirely.
The corrected vocabulary path derives both quantities from the same rounded
logits, then subtracts the target in FP32 in its custom first-order backward.
See [the executable reproduction](../../scripts/check_mlx_legacy_projection.py)
and [raw results](legacy_projection_gate.json).

The token-chunked prototype recomputes full-vocabulary logits for token tiles,
uses the existing CE backward and accumulates weight gradients in FP32. Select
it with `--fused-vocab-loss --fused-vocab-chunk-axis token
--fused-vocab-chunk-size 512` on MLX. The default axis remains `vocab`; the
feature remains off. PyTorch rejects the MLX-only token option.

Both paths retain full transformer context before dropping the unused final
AR hidden position. Tests compare losses and parameter gradients with the full
head in FP32/BF16, checkpointing on/off, special-token masks, non-divisible
tiles and signed/zero cotangents. Each axis also passes exact tiny-run
checkpoint replay for model/master/optimizer/MLX RNG. Chunked GEMM reductions
can still change BF16 gradients and subsequent weights versus a full head.
Only first-order training is supported; no bitwise full-head or convergence
equivalence is claimed.

## Measured projection allocation

Fresh-process MLX 0.32.2, BF16, 2,048 hidden rows, width 512, uniform synthetic
targets, four warmups and 24 synchronized loss/gradient evaluations. These
rates are **head-only diagnostics**, not pretraining token rates. Every case
passed scalar loss and gradient gates; the 65,536-vocabulary token-512 check
additionally matched all per-token losses exactly in this fixture.

| Vocabulary | Full head peak MiB | Vocab tiles 4096 | Token tiles 512 |
|---:|---:|---:|---:|
| 8,192 | 92.0 | 98.1 | 134.0 |
| 32,768 | 356.0 | 374.3 | 518.0 |
| 65,536 | 708.0 | 746.6 | 1,011.5 |

All chunked cases ran slower than the corresponding full-head case in this
single-order sweep. These implementations have **not established an
allocation or throughput win**. The lazy graph can retain multiple logits or
gradient buffers despite Python tiling. An attempted `mx.depends` serialization
increased allocation further and was removed. Its raw measurements remain in
`projection_matrix_serial`; they do not describe the final implementation.
An evaluator/primitive that controls buffer lifetimes is a more promising next
step than exposing a claimed O(chunk) memory bound.

Raw data: [initial sweep](projection_matrix/results.json),
[rejected dependency experiment](projection_matrix_serial/results.json),
[per-token gate](projection_per_token_gate.json).

## Shape-aware attention backward preparation

The isolated MLX 0.32.4 probe compares native and fallback backward on the same
runtime. It checks 24 causal BF16 shapes: lengths 64–2048, head dimensions
64/128, eight query heads, and eight or two KV heads. Batch × length is 2048.
Each mode's exported graph is checked for the actual dedicated
`ScaledDotProductAttentionVJP` primitive, and outputs/dQ/dK/dV are compared with
an independent FP32 attention reference. Maximum observed relative L2 was
0.006189 (gate 0.015).

Twelve alternating-order measurement pairs per shape show that a single
backward choice is inappropriate. For equal-head dimension-64 attention,
native gain over fallback was -45.5% at length 64, -22.8% at 256, -2.9% at 1024
(inconclusive under the probe's 3%/75%-wins rule), and +135.9% at 2048. GQA and
dimension 128 have different crossover behavior. These percentages describe
the attention output-plus-VJP component only.

The initial full-training ABBA confirmation was attempted on merged v1.7.4 code with
the same 49M architecture, length 2048 and effective batch eight. The planner
selected microbatch one, accumulation eight and checkpointing. Its first run
hit the sustained swap-growth guard at update 22, so no full-update speed
claim was made at that stage. The native/fallback environment
flag is changed only inside the single-threaded diagnostic or between fresh
processes; production training does not mutate a process-wide flag by shape.

The [subsequent full-training validation](FULL_ATTENTION_VALIDATION.md) completed
all eight ABBA jobs with a fixed post-warmup swap baseline. It measured native
+76.9% at context 2048 and fallback +6.0% at 256 across two adjacent pairs per
shape. Checkpoint tensors differ between modes despite close held-out CE; no
automatic dispatcher or default is promoted. See that report for allocation,
timing drift, numerical distances and the limits of these short-run checks.

Raw data: [component samples and numerical errors](attention_shapes/results.json),
[runtime/device metadata](attention_shapes/metadata.json), and
[invalid full-update attempt](full_attention/results.json).
Backend behavior was checked against the version-pinned
[MLX 0.32.4 Metal implementation](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp).

## Sustained QKV/RoPE and command buffers

Four valid frozen-source (`f558146`) runs each completed 192 optimizer updates
over the same nonwrapping real-corpus prefix: 49,166,848 parameters, length
256, microbatch eight × accumulation eight, MLX 0.32.2. The first 16 updates
are excluded from rates. Compiled accumulation, packed SwiGLU and forced fused
SDPA are held fixed. A byte-verified 32 MiB corpus prefix avoids hashing the
entire 10 GB corpus at every setup; production fingerprinting is unchanged.

| Setting | Warmed input tok/s | Peak MLX GiB |
|---|---:|---:|
| Default buffers, reference QKV | 41,154 | 1.642 |
| 16/16 buffers, reference QKV | 35,457 | 1.297 |
| 16/16 buffers, fused QKV/RoPE | 33,071 | 1.235 |
| Default buffers, fused QKV/RoPE | 40,313 | 1.596 |

All four initial weight files have SHA256
`b95399e433484c6b0b87d846f5a89cd136d68900592debadac9dd0123a466be3`.
Each final checkpoint's 86 model leaves, 86 master leaves, 174 optimizer leaves
and MLX RNG leaf matches the default/reference run bit for bit, with equal
shapes/dtypes and finite values. This establishes exact tensor-state parity
for these completed runs, not arbitrary datasets or long-run model quality.

16/16 reduces the reference peak by 21.0%. No speed win is established:
reverse-order repeats were invalidated by other local Python validation jobs
or sustained swap growth. Interrupted runs, including earlier full-corpus
attempts, are retained and excluded. The historical +7% QKV speed claim has
not been reproduced here. Neither buffer setting nor QKV default is changed.

See [timing/telemetry results](qkv_buffers_prefix/results.json),
[exact final-state comparison](qkv_buffers_prefix/state_comparison.json),
[frozen source hashes](qkv_buffers_prefix/source_manifest.json) and
[corpus provenance](corpus_prefix_provenance.json).

## Reproduction

Run from the source checkout with the shared environment's Python, setting
`PYTHONPATH=.`. Use a new output directory for each probe. Only one GPU workload
should run at a time.

```sh
python scripts/prepare_mlx_corpus_prefix.py data/english_corpus_5b.bin reports/mlx_fusion_2026-11-16/corpus_prefix.bin
python scripts/check_mlx_legacy_projection.py
python scripts/run_mlx_projection_matrix.py --output /tmp/mlx-head-sweep
# Only when the isolated 0.32.4 runtime is available:
PYTHONPATH=/private/tmp/telos-mlx-0.32.4:. python scripts/probe_mlx_attention_backward.py --output /tmp/mlx-attention-shapes
```

The attention confirmation driver uses fresh processes, excludes 16 warmups,
records actual planner choices, and cancels its own trial on competing Python
workers, memory/swap guards or timeout. It is intentionally allowed to fail
rather than turn an interrupted run into a performance claim. The frozen QKV
driver remains under `scratch/mlx_fusion_20261116`. Recreate its
`/private/tmp/telos-fusion-frozen-f558146` source directory with `git archive
f558146`, then apply [the instrumentation patch](frozen_profiler.patch) to that
export using `patch -p0`. Its source hash must match the recorded manifest.
The profiler's `commit` field describes the orchestration checkout; the frozen
import path and source hashes identify the code actually trained. No claim is
made that a different profiler or orchestration checkout gives identical timing.
