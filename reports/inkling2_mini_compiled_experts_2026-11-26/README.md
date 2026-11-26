# Inkling pinned routes, compiled experts, and refresh intervals

Native Telos ran the trained adapter on an M5 Pro at 3 GiB and 512-token context with fused NVFP4 and relative masks. Grouped attention projections stayed off. Each comparison used the same three [prompts](../inkling2_mini_pinned_w16_2026-11-26/prompts.jsonl). Completed W=16 variants matched all 146 sampled IDs, and steady tokens read zero expert bytes. No completed run grew swap or increased swap-outs.

## Remove steady route materialization

Each physical bank now has an opt-in host route plan that lasts until the next window. It reuses the selected IDs in sorted order and the per-expert row and slot arrays. Gate weights still update on every token. On 134 steady steps, the old path made 6,432 `.tolist()` calls and the reused path made zero. The complete decode dropped from 6,864 calls to 216, all at initial selection or refresh. Steady throughput moved from 16.02 to 16.21 tok/s in that comparison. The count measures host materializations. The speed result gives no evidence of a full GPU wait for each previously materialized pinned array. [Route reuse comparison](host_routes.json)

## Batch the resident expert block

The batched NVFP4 kernel binds eight existing packed expert buffers, with one launch per projection and no packed weight stack. Across 44 steady NVFP4 visits per token, that changes the two projections from 704 to 88 launches. The four MXFP8 visits use the existing path. Bitwise fixture tests cover both the shared activation input and per-expert activation input layouts. The compiled variant uses `mx.compile` on the resident expert projection and SwiGLU block. The full layer still owns host cache operations and variable-length attention state.

| W=16 path with background memory sampling | Overall | Between refreshes |
| --- | ---: | ---: |
| Cached routes, separate experts | 5.44 tok/s | 6.03 tok/s |
| Batched experts, eager | 10.14 tok/s | 11.97 tok/s |
| Batched experts, compiled | **16.76 tok/s** | **21.34 tok/s** |
| Compiled batches plus fused sorted sum | 16.16 tok/s | 20.88 tok/s |

The separate-expert control was unusually slow in this run. The independent comparison below ran the regular synchronous model guard and placed the separate-expert control last. It confirmed a 27.5% overall and 34.9% steady gain from compiled batching at W=16. The fused sorted sum matched the same tokens and showed no extra throughput gain in its one trial. Peak MLX memory remained about 2,161 MiB across these variants. [Background sampling comparison](background_guard.json)

## Refresh frequency

| Runtime route window | Overall | Between refreshes | Decode steps with expert reads | Physical decode reads |
| --- | ---: | ---: | ---: | ---: |
| W=16, cached routes and separate experts | 13.20 tok/s | 15.76 tok/s | 9 | 3.14 GiB |
| W=16, compiled batches | 16.83 tok/s | 21.27 tok/s | 9 | 3.16 GiB |
| W=32, compiled batches | **17.94 tok/s** | 21.21 tok/s | 5 | 2.00 GiB |
| W=64, compiled batches | 16.75 tok/s | 18.65 tok/s | 3 | 1.39 GiB |

W=32 and W=64 reused the adapter trained at W=16. Both changed outputs on the two longer prompts, with the first differing tokens at positions 21 and 23. These are throughput and output probes. A wider window requires a held-out quality comparison before adoption. Exact lookahead at a fresh bank depends on the preceding layer output. Prefetching before that output is ready would require a prediction and a measurement of overfetch. [Refresh-width comparison and W=16 repeat](refresh_widths.json)

## Memory admission

An early attempt stopped at 34% reported headroom under the previous fixed runtime rule, with no swap growth before termination. The Inkling runner now uses the native loader’s budget to admit a worker, which translates to roughly 28% reported headroom for a 3 GiB request on this 24 GiB Mac. It also requires normal macOS pressure, samples pressure and both swap counters throughout the run, and caps worker RSS at the requested budget. Optional model-side background sampling moves the subprocess waits off the decode thread. The independent benchmark watchdog keeps its any-growth swap stop in both modes. [Stopped attempt](guarded_stop.json)
