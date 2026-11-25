# LFM2.5 router-aware expert prefetch trial

Tested native Telos on the installed LFM2.5-8B-A1B MLX 4-bit checkpoint on the M5 Pro on AC in High Power mode. The integrated demand-read path supplied both the trace and the baseline. Each guarded process used a 2 GiB admission budget, 512 MiB per-layer cache, 128-token context, and the same 16-token raw prompt and 16-token greedy budget. Every process ran alone and MLX-LM was absent.

## Offline gate

A demand-only run recorded 330 decode layer visits, expert IDs, per-layer cache residency, router inputs, and fragment read timings. It matched all 16 resident token IDs and text. The offline NumPy reference reproduced the real top-four set at all 330 visits. The captured fragment reads give a median interval of 0.84 ms from the earliest read start to the latest read finish for one demand-missed expert. That is a service-time proxy; it does not model contention from additional reads.

Replay compared previous-token choices, recent expert frequency, and using a later layer's router on an earlier layer's input. A prediction counted only when its expert was absent at the target layer and had enough lead time by the service-time proxy. The replay held no more than four staged experts within 24 MiB. The live-trial gate required coverage of at least 10% of demand misses and less than 5% extra expert bytes on each segment. Calibration used decode steps 1–7, and the held-out segment used steps 8–15 of the same prompt.

The conservative one-layer-ahead predictor used a 0.10 margin over the fifth routing score. It covered 75 of 400 demand misses with seven wrong reads on calibration, and 92 of 428 with four wrong reads on the held-out segment. That is 21.5% held-out coverage with 0.9% estimated extra bytes. Previous-token and recent-frequency rules covered only 1.6% of held-out misses. The trace represents one prompt.

## Live result

The candidate staged one predicted expert at a time after the current layer's final demand lookup. The actual router still selected and weighted the experts. A matched prediction replaced the corresponding demand read, and speculative staging did not alter eviction order.

| Measurement | Demand-only baseline | Next-layer predictor |
| --- | ---: | ---: |
| Decode trials | 10.57, 10.60 tok/s | 9.70, 10.98 tok/s |
| Median decode | 10.59 tok/s | 10.34 tok/s |
| Peak MLX memory | 964 MiB | 964 MiB |
| Whole-drive host reads per request | 4.64–5.45 GiB | 4.69–4.71 GiB |
| Whole-drive host writes per request | 16.3–24.3 MiB | 16.0–18.3 MiB |

Each predictor run staged 197 experts, used 185, and read 70.875 MiB for 12 unused experts. That is 1.45% extra application bytes relative to decode demand. About 97–98% of used speculative bytes were already ready at consumption. The earlier availability did not produce a repeatable end-to-end gain. The timing comparison does not isolate the added router and scheduling overhead.

All four comparison runs reproduced all 16 IDs and exact text from the [resident reference](../lfm25_flash_moe_2026-11-25/resident.json). The watchdog would stop the process and cancel remaining trials at the first measured increase in used swap or cumulative swap-outs. Both stayed flat, and sampled free memory stayed at or above 58%. Drive counters include background traffic. The prototype was not enabled in the runtime.

[Offline replay](replay.json) · [Compact routing trace](routing.jsonl) · [Live trials](live_trials.jsonl) · [Swap guard record](guard.json)
