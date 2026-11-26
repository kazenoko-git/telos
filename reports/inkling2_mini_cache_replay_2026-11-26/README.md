# Inkling 2 Mini expert route trace, replay, and guarded A/B

This report documents native Inkling 2 Mini route tracing, offline cache replay, and the matched guarded SSD-only A/B comparison on commit `42524557` using the pinned MLX checkpoint (`9c0e2960609ba7cb40daf92c6f7bf8ed56e1dd62`, `2 GiB` admission budget, `512`-token context, `250.86 MiB` expert cache, no MLX-LM or resident model).

## Guarded baseline trace capture

A single guarded SSD-only trace run (`250.86 MiB` expert cache, `answer_prefix` prompt of `28` tokens, `16` greedy tokens, opt-in tracing enabled at commit `a18f4241`) reproduced the exact 16 token IDs from [the prior guarded run](../inkling2_mini_ssd_guarded_2026-11-26/runs.json):

| Metric | Value |
| --- | ---: |
| Checkpoint load | 0.66 s (`963.55 MiB` `disk0` reads) |
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

## Matched guarded SSD-only A/B on the real checkpoint

Commit `42524557` implemented whole-expert per-physical-layer quota allocation weighted by `visits ** 2`, visit-scoped protection of already-selected unconsumed resident experts, and per-token decode frequency decay (`0.5`). Tiny sharded fixtures verify exact resident-vs-SSD hidden-state, logit, KV/convolution-state, and token-ID parity, plus exact replay agreement across 24 targeted tests.

With host background memory cleared (`71%–75%` free before launch), all four sequential single-process guarded A/B runs (`answer_prefix` baseline → `answer_prefix` optimized → `chat` optimized → `chat` baseline, `2 GiB` budget, `512` context, `16` greedy tokens) completed with zero guard trips and exact generated token-ID parity against [the saved baseline runs](../inkling2_mini_ssd_guarded_2026-11-26/runs.json):

| Measurement | `answer_prefix` baseline | `answer_prefix` optimized | `chat` baseline | `chat` optimized |
| --- | ---: | ---: | ---: | ---: |
| Generated token-ID parity | Exact | Exact | Exact | Exact |
| Decode speed | 1.85 tok/s | **3.09 tok/s** (+66.9%) | 1.92 tok/s | **3.01 tok/s** (+56.8%) |
| Time to first sampled token (TTFT) | 5.94 s | 5.85 s | 5.99 s | 5.75 s |
| Total generation request time | 14.34 s | 10.97 s | 14.08 s | 11.02 s |
| Cache hits (prefill / decode) | 0 / 0 | 90 / 1,719 (29.8%) | 0 / 0 | 91 / 1,594 (27.7%) |
| Logical expert bytes (prefill) | 12.72 GiB | 12.35 GiB | 12.31 GiB | 11.93 GiB |
| Logical expert bytes / decode token | 1,754.2 MiB | **1,264.6 MiB** (-27.9%) | 1,754.2 MiB | **1,300.2 MiB** (-25.9%) |
| Physical `disk0` reads (prefill / decode / total) | 10.92 / 14.45 / 25.37 GiB | **6.70 / 9.77 / 16.47 GiB** | 6.61 / 13.94 / 20.55 GiB | **6.42 / 10.33 / 16.74 GiB** |
| Physical `disk0` writes (generation) | 1.09 MiB | 0.08 MiB | 4.01 MiB | 7.15 MiB |
| Peak process RSS | 1,737.0 MiB | 1,672.4 MiB | 1,764.7 MiB | 1,670.3 MiB |
| Peak MLX memory during generation | 1,286.9 MiB | 1,284.7 MiB | 1,286.9 MiB | 1,284.7 MiB |
| Free memory (origin → minimum) | 72% → 66% | 74% → 68% | 75% → 66% | 74% → 67% |
| Swap used / swap-outs (origin → final) | 1,638.56 MiB / 37,350,756 (0) | 1,638.56 MiB / 37,350,756 (0) | 1,638.56 MiB / 37,350,756 (0) | 1,638.56 MiB / 37,350,756 (0) |

On `answer_prefix`, live `optimized` hit counts (`90` prefill, `1,719` decode) and logical expert bytes (`12.35 GiB` prefill, `1,264.6 MiB/tok` decode) matched `telos.models.ssd_replay` identically. Physical `disk0` decode reads fell `-32.4%` (`14.45 → 9.77 GiB`) on `answer_prefix` and `-25.9%` (`13.94 → 10.33 GiB`) on `chat`.

[Matched A/B runs](ab_runs.json) · [Replay metrics](replay.json) · [Trace and aborted-guard records](trace_run.json)
