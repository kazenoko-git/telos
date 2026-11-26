# Inkling 2 Mini expert route trace and cache replay

This report documents native Inkling 2 Mini route tracing, offline cache replay, and the guarded SSD-only cache update on commit `4252455` using the pinned MLX checkpoint (`9c0e2960609ba7cb40daf92c6f7bf8ed56e1dd62`, 2 GiB admission budget, 512-token context).

## Guarded baseline trace capture

A single guarded SSD-only trace run (`250.86 MiB` expert cache, `answer_prefix` prompt of 28 tokens, 16 greedy tokens, opt-in tracing enabled at commit `a18f424`) reproduced the exact 16 token IDs from [the prior guarded run](../inkling2_mini_ssd_guarded_2026-11-26/runs.json):

| Metric | Value |
| --- | ---: |
| Checkpoint load | 0.66 s (`963.55 MiB` disk0 reads) |
| Time to first sampled token | 7.16 s |
| Decode speed (15 post-prefill steps) | 1.92 tok/s |
| Logical expert bytes (prefill / per decode token) | 12.72 GiB / 1.71 GiB (`1,754.2 MiB`) |
| Expert cache hits / misses (prefill + decode) | 0 / 8,587 (`41,249,192,192` bytes) |
| Whole-drive `disk0` reads (prefill / decode / total) | 10.95 GiB / 14.50 GiB / 25.45 GiB |
| Whole-drive `disk0` writes during generation | 10.40 MiB |
| Peak sampled RSS / peak MLX generation memory | 1,594.95 MiB / 1,286.91 MiB |
| Swap used / swap-outs (origin → final) | 2,708.19 MiB / 37,338,859 (0 growth) |
| Free memory (origin → minimum) | 61% → 39% |

Across the 15 post-prefill decode tokens, the 12 looped banks (layers 6–17) averaged `4.86/8` overlapping experts between pass 0 and pass 1, `4.82/8` between pass 1 and pass 2, `3.71/8` between pass 0 and pass 2, and `13.86/24` unique experts across all three passes of a token. Global LRU suffered `0` hits because the four trailing MXFP8 banks (`250.60 MiB` per token) plus a single loop pass (`410 MiB`) each exceed the `250.86 MiB` cache allowance.

## Offline replay comparison

Replaying the captured 768-visit trace (`48` prefill visits, `720` decode visits) via `telos.models.ssd_replay` reproduced the live legacy baseline (`0 / 8,587` hits, `41,249,192,192` bytes) identically.

### 2 GiB budget (`250.86 MiB` / `263,043,072` bytes cache)

| Policy | Prefill hits | Prefill read | Decode hits | Decode packed read / tok |
| --- | ---: | ---: | ---: | ---: |
| Global LRU (legacy baseline) | 0 / 2,827 (0.0%) | 12.72 GiB | 0 / 5,760 (0.0%) | 1,754.2 MiB |
| Global LRU + visit protection | 0 / 2,827 (0.0%) | 12.72 GiB | 0 / 5,760 (0.0%) | 1,754.2 MiB |
| Equal-byte quotas + LRU (no protection) | 0 / 2,827 (0.0%) | 12.72 GiB | 0 / 5,760 (0.0%) | 1,754.2 MiB |
| Equal-byte quotas + LRU + visit protection | 37 / 2,827 (1.3%) | 12.57 GiB | 734 / 5,760 (12.7%) | 1,540.2 MiB |
| Equal-byte quotas + LFU + visit protection | 37 / 2,827 (1.3%) | 12.57 GiB | 977 / 5,760 (17.0%) | 1,464.1 MiB |
| Visit-weighted (`visits^1`) + LFU + visit protection | 66 / 2,827 (2.3%) | 12.45 GiB | 1,447 / 5,760 (25.1%) | 1,330.2 MiB |
| Loop-weighted (`visits^2`) + LRU + visit protection | 90 / 2,827 (3.2%) | 12.35 GiB | 1,470 / 5,760 (25.5%) | 1,335.6 MiB |
| Loop-weighted (`visits^2`) + LFU + visit protection | 90 / 2,827 (3.2%) | 12.35 GiB | 1,668 / 5,760 (29.0%) | 1,279.2 MiB |
| Loop-weighted (`visits^2`) + decayed LFU (`0.5`) + visit protection | 90 / 2,827 (3.2%) | 12.35 GiB | 1,719 / 5,760 (29.8%) | 1,264.6 MiB |
| Offline Belady MIN reference | 116 / 2,827 (4.1%) | 12.24 GiB | 2,385 / 5,760 (41.4%) | 1,075.0 MiB |

### 3 GiB budget (`1,024.00 MiB` / `1,073,741,824` bytes cache, offline only)

| Policy | Prefill hits | Prefill read | Decode hits | Decode packed read / tok |
| --- | ---: | ---: | ---: | ---: |
| Global LRU (legacy baseline) | 0 / 2,827 (0.0%) | 12.72 GiB | 2,488 / 5,760 (43.2%) | 1,045.7 MiB |
| Equal-byte quotas + LRU + visit protection | 174 / 2,827 (6.2%) | 11.99 GiB | 3,062 / 5,760 (53.2%) | 849.9 MiB |
| Visit-weighted (`visits^1`) + LFU + visit protection | 284 / 2,827 (10.0%) | 11.54 GiB | 3,363 / 5,760 (58.4%) | 757.8 MiB |
| Loop-weighted (`visits^2`) + decayed LFU (`0.5`) + visit protection | 359 / 2,827 (12.7%) | 11.22 GiB | 3,525 / 5,760 (61.2%) | 738.5 MiB |
| Offline Belady MIN reference | 478 / 2,827 (16.9%) | 10.73 GiB | 4,322 / 5,760 (75.0%) | 450.1 MiB |

## Cache implementation and guarded A/B status

Commit `4252455` implemented whole-expert per-physical-layer quota allocation weighted by `visits ** 2`, visit-scoped protection of already-selected unconsumed resident experts, and per-token decode frequency decay (`0.5`). Tiny sharded fixtures verify exact resident-vs-SSD hidden-state, logit, KV/convolution-state, and token-ID parity, plus exact replay agreement across 24 targeted tests.

Matched guarded SSD-only A/B runs on the real checkpoint could not complete because host background memory pressure dropped starting free memory from `61%` (during trace capture) to `49%–53%`; during prefill of the baseline run (`1,370.23 MiB` peak RSS, `0` swap growth), system free memory reached `34%` and tripped the `35%` runtime guard, after which system free memory remained at `43%–49%` (at or below the safe headroom margin).

[Replay metrics](replay.json) · [Trace and guard records](trace_run.json)
