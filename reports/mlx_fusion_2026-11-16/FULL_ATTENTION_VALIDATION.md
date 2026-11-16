# Full-training attention backward validation

Completed on 2026-11-16, Apple M5 Pro / 24 GiB, AC power with High Power Mode
configured. All eight fresh-process jobs completed: **896 optimizer updates
and 14,680,064 input tokens** in total. Both compared modes use isolated MLX
0.32.4; these are not comparisons against MLX 0.32.2.

## Matched protocol

The model has 49,166,848 parameters, 14 layers, width 512, eight equal Q/K/V
heads and head dimension 64. BF16 parameters, FP32 masters/AdamW, real training
tokens, initialization, learning-rate schedule and experimental packed SwiGLU
are matched within each context length. Forward SDPA is forced fused in both
modes; only native versus fallback backward is changed before MLX import.

Fresh-process order per context is native, fallback, fallback, native. The
first 32 updates are excluded. The production planner selected:

| Context | Microbatch | Accumulation | Effective batch | Checkpointing | Updates per job |
|---:|---:|---:|---:|---|---:|
| 2048 | 1 | 8 | 8 | On | 128 |
| 256 | 4 | 16 | 64 | Off | 96 |

Every initial weight export has SHA256
`b95399e433484c6b0b87d846f5a89cd136d68900592debadac9dd0123a466be3`.
Every run within each shape has the same recorded memory plan. Production
training code was unchanged from `2f0d393`; the instrumented profiler and
controller hashes are [recorded separately](full_attention_steady/source_sha256.txt).

The earlier unchanged-protocol retry again stopped on model/optimizer setup
swap growth and remains invalid. Before starting the complete comparison, we
fixed the swap-growth baseline at update 32 for **all** runs. Allocation and
absolute free-memory/swap guards remained active throughout setup. See the
[full protocol](FULL_ATTENTION_PROTOCOL.md) and
[excluded retry](full_attention_retry/results.json).

## Completed-update throughput and allocation

Rates count input tokens over synchronized, completed optimizer-update times;
they exclude process/model setup and final checkpoint serialization. These are
full-model training updates, not attention-only kernel rates.

| Context | Native tok/s, two jobs | Fallback tok/s, two jobs | Adjacent-pair result | Native peak GiB | Fallback peak GiB |
|---:|---|---|---|---|---|
| 2048 | 20,819; 17,777 | 10,135; 11,674 | Native +105.4%, +52.3% | 2.130; 2.258 | 1.633; 1.684 |
| 256 | 27,674; 28,917 | 28,551; 31,493 | Fallback +3.2%, +8.9% | 1.219; 1.219 | 1.181; 1.181 |

The geometric means of adjacent-pair speed ratios are **native +76.9% at
2048**, and **fallback +6.0% at 256**. With only two pairs per shape, these are
descriptive measurements, not population estimates or confidence intervals.
There is substantial timing drift: the first fallback long-context job's
first/second measured halves were 8,759 and 12,025 tok/s. All half-window rates
are retained in the [summary](full_attention_steady/comparison_summary.json).
Both long-context pairs nevertheless favor native; both short-context pairs
favor fallback. Native's long-context speed gain uses more allocation.

All jobs passed their guards without telemetry errors or competing Python
workers. Across the complete runs, minimum sampled system free memory was
28%, minimum swap headroom was 585.9 MiB, and maximum post-warmup swap growth
was 361.2 MiB. Peak MLX allocation was 2.258 GiB, below the 4 GiB guard. These
are sampled/advisory safeguards, not an OS memory cap. Other interactive
application GPU activity is not excluded by Python-worker detection.

## Full-model graph and checkpoint checks

Full-model gradient graphs reconstructed at the actual microbatch shapes,
with matching checkpointing and kernel options, contain **14 dedicated
`ScaledDotProductAttentionVJP` nodes** in native mode and **zero** in fallback
mode at both lengths. The requested modes therefore select different
backward implementations at the measured model shapes.

Final model, master, optimizer and MLX RNG groups were checked for tensor keys,
shapes, dtypes and finite values. Repetitions within each mode reproduce those
tensor groups bit for bit. Between native and fallback, RNG matches but model,
master and optimizer tensors differ:

| Context | Model relative L2 | Model maximum absolute difference | FP32 master relative L2 |
|---:|---:|---:|---:|
| 2048 | 0.012322 | 0.009766 | 0.012173 |
| 256 | 0.003087 | 0.007813 | 0.002680 |

Thus this is **not a bitwise-equivalent replacement** for fallback training.
See [all group distances and graph checks](full_attention_steady/checkpoint_validation.json).

## Held-out diagnostic

Every checkpoint was evaluated through the same inference path on 32
contiguous real-corpus sequences at its context length. The data start at
token offset 16,711,680, beyond either training range; token hashes and
per-sequence losses are saved. Both repetitions of each mode give the same
held-out losses.

| Context | Native mean CE | Fallback mean CE | Native minus fallback |
|---:|---:|---:|---:|
| 2048 | 6.303634 | 6.304272 | -0.000638 |
| 256 | 6.658892 | 6.661068 | -0.002175 |

All values are finite. These 32-sequence checks quantify short-run differences;
they do not establish equal convergence, broad quality or an improvement in
generalization. In particular, similar held-out CE does not cancel the
checkpoint-weight differences above.

## Decision and reproduction

The full-training results support shape-dependent experimental choices on this
model/runtime/device: native at context 2048 for speed when the additional
allocation is acceptable, fallback at 256 for both speed and allocation. They
do not establish a general crossover threshold, GQA policy, other head sizes,
other microbatches, bidirectional objectives or other hardware. No automatic
dispatcher or production default is changed.

With the existing corpus fixture and isolated runtime, run one GPU workload at
a time from the source checkout, using an unused output directory:

```sh
PYTHONPATH=. python scripts/run_mlx_attention_confirmation.py \
  --output /tmp/mlx-full-attention \
  --data reports/mlx_fusion_2026-11-16/corpus_prefix.bin \
  --warmup 32 --swap-growth-baseline-updates 32
MLX_ENABLE_TF32=0 PYTHONPATH=/private/tmp/telos-mlx-0.32.4:. \
  python scripts/validate_mlx_attention_checkpoints.py /tmp/mlx-full-attention \
  --data reports/mlx_fusion_2026-11-16/corpus_prefix.bin
python scripts/summarize_mlx_attention_confirmation.py /tmp/mlx-full-attention
```

The controller uses the isolated runtime for its fresh children; the validator
requires it explicitly. Checkpoints and the corpus binary remain local;
telemetry, numerical comparisons, data hashes and exported graphs are retained
in the source report. [Raw run results](full_attention_steady/results.json)
include the executed arguments, actual plans, runtime paths and guard metadata.
