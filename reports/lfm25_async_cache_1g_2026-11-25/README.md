# LFM2.5 demand-only async reads at 768 MiB and 1 GiB

Native Telos at `7048e138` ran the LFM2.5-8B-A1B MLX 4-bit checkpoint on an M5 Pro with MLX 0.32.2 and AC High Power mode. Four fresh processes ran in 768, 1,024, 1,024, 768 MiB order using the [previous comparison's workload and guards](../lfm25_async_cache_cap_2026-11-25/README.md). Each used a 2 GiB admission budget, 22 per-layer cache partitions, a 128-token context, the same 16-token raw prompt, and a 16-token greedy generation limit. The worker applied each cap in process. The production default remains 512 MiB. No MLX-LM was used.

| Median of two trials | 768 MiB | 1 GiB |
| --- | ---: | ---: |
| Decode after first token | 12.85 tok/s | 12.74 tok/s |
| Time to first token | 1,499 ms | 1,464 ms |
| Decode cache hit rate | 52.95% | 60.83% |
| Expert payload read per decode token | 244.52 MiB | 203.57 MiB |
| Host SSD reads per generation request | 4.50 GiB | 3.49 GiB |
| Host SSD writes per generation request | 19.06 MiB | 9.98 MiB |
| Peak MLX memory during generation | 1,224 MiB | 1,478 MiB |
| Peak process RSS | 1,541 MiB | 1,795 MiB |

Decode speed overlapped across trials at 12.50–13.19 tok/s for 768 MiB and 12.36–13.11 tok/s for 1 GiB. The larger cache reduced expert payload reads per decode token by 16.7% and added 254 MiB of peak process RSS. Host SSD reads ranged from 3.93 to 5.08 GiB for 768 MiB and 3.48 to 3.49 GiB for 1 GiB. These whole-drive IOKit counters cover prefill and decode, exclude load, and include background activity. The idle sample recorded 16.52 MiB of host writes in 2.23 seconds, so the measured write difference cannot be attributed to Telos.

All trials reproduced all 16 token IDs and the exact decoded text from the [resident reference](../lfm25_flash_moe_2026-11-25/resident.json). Across 126 watchdog samples, used swap stayed at 3,781.50 MiB and cumulative swap-outs stayed at 37,061,020 pages. The lowest reported free memory was 50%.

A header-only planner check with the cache cap lifted found 1,057.88 MiB available after existing reserves at this checkpoint, budget and context. That admits 179 whole expert entries across 22 layers, compared with 173 at 1 GiB. The ceiling is a planner estimate. The largest live cache cap tested was 1 GiB.

[Raw trials](trials.jsonl) · [Guard and idle disk measurements](guard_and_idle.json)
