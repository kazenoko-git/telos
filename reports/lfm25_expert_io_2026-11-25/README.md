# Guarded LFM2.5 expert read prototypes

Tested the installed LFM2.5-8B-A1B MLX 4-bit checkpoint natively on the M5 Pro with MLX 0.32.2 and High Power mode on AC. Each configuration ran twice in fresh processes, in baseline, reuse, async, async, reuse, baseline order. The model was unloaded between processes. All trials used the same 512 MiB per-layer LRU cache, 2 GiB admission budget, 128-token context, 16-token prompt and 16-token greedy limit. MLX-LM was absent. Both variants were experimental and the native runtime defaults were unchanged.

- **Allocation reuse** kept a read-only descriptor and one 1.75 MiB staging buffer. It read into that buffer with positioned vectored I/O and relied on the same 128 MiB MLX free-cache cap instead of clearing the allocator on every miss. Synchronous loading and evaluation order were preserved.
- **Async demand reads** used up to four workers and staged at most 24 MiB for the current layer. They queued known misses after the router selected experts during decode. Prefill, cache eviction, accumulation order and MLX conversion stayed on the existing path. No predicted expert was read. Cache hits later evicted inside the same layer used the synchronous path as needed.

| Median of two runs | Baseline | Allocation reuse | Async demand |
| --- | ---: | ---: | ---: |
| Decode | 4.28 tok/s | 5.34 tok/s | 6.90 tok/s |
| First token | 1,686 ms | 1,361 ms | 1,637 ms |
| MLX peak during generation | 964 MiB | 964 MiB | 964 MiB |
| Process peak RSS | 1,263 MiB | 1,259 MiB | 1,283 MiB |
| Backing SSD host reads per request | 5.29 GiB | 4.85 GiB | 4.65 GiB |
| Backing SSD host writes per request | 40.7 MiB | 22.1 MiB | 16.2 MiB |

Allocation reuse increased median decode speed 25% and async demand reads increased it 61% on this workload. Both async trials requested exactly 4,254,695,424 bytes through the worker queue and spent about 359 ms waiting on queued reads. This is measured waiting time. A GPU timeline is needed to quantify compute overlap independently from read concurrency or lower dispatch overhead.

All runs matched the [resident reference](../lfm25_flash_moe_2026-11-25/resident.json) for all 16 token IDs and exact decoded text. Telos requested the same 7,447.8 MiB of expert payload per request in every run, including 2,557.4 MiB for prefill and 326.0 MiB per decode token. All six recorded the same 492 decode hits out of 1,320 lookups.

The device metrics are before-and-after IOKit block-driver counters for the whole physical disk storing the checkpoint. They cover the generation request including prefill and exclude model loading. They include OS and application background traffic and do not measure NAND-internal writes. During a separate 2.25-second no-model sample, the same disk recorded 32.7 MiB read and 16.1 MiB written. Device-level totals varied between trials even with the same application-requested bytes, so differences in the table do not establish SSD write savings.

The independent watchdog sampled about every 100 ms and was configured to terminate the worker and cancel remaining trials on any measured growth in used swap or cumulative swap-outs. Used swap stayed at 3,877.50 MiB and cumulative swap-outs stayed at 37,061,020 pages. Free memory stayed at or above 54% throughout model runs.

[Raw trials](trials.jsonl) · [Guard and idle disk sample](guard_and_idle.json)
