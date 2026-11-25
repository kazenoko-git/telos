# LFM2.5 demand-only async reads at 512 and 768 MiB

Native Telos at `3a9e68e4` ran LFM2.5-8B-A1B MLX 4-bit on an M5 Pro with MLX 0.32.2 and AC High Power mode. Four fresh processes ran one at a time in 512, 768, 768, 512 MiB order. Each used a 2 GiB admission budget, a 128-token context, a 16-token raw prompt, and a 16-token greedy generation limit. There was no warmup or MLX-LM. The worker changed only the cache cap in memory for the 768 MiB trials. Both sizes used 22 per-layer partitions and the same async demand reads, reusable staging buffers, and open file descriptors. The production default remains 512 MiB.

| Median of two trials | 512 MiB | 768 MiB |
| --- | ---: | ---: |
| Decode after first token | 10.06 tok/s | 13.69 tok/s |
| Time to first token | 1,434 ms | 1,372 ms |
| Decode cache hit rate | 37.27% | 52.95% |
| Expert payload read per decode token | 326.03 MiB | 244.52 MiB |
| Host SSD reads per generation request | 5.23 GiB | 3.86 GiB |
| Host SSD writes per generation request | 16.47 MiB | 8.22 MiB |
| Peak MLX memory during generation | 964 MiB | 1,224 MiB |
| Peak process RSS | 1,280 MiB | 1,541 MiB |

The larger cache decoded 36.2% faster at the median. Per-trial speeds were 9.30 and 10.82 tok/s at 512 MiB and 13.45 and 13.94 tok/s at 768 MiB. Telos requested 25% less expert payload per decode token. All four trials reproduced all 16 token IDs and exact decoded text from the [resident reference](../lfm25_flash_moe_2026-11-25/resident.json).

Host SSD counts are whole-drive IOKit block deltas for the generation window, including prefill and excluding checkpoint load. Observed reads fell 26.1% at the median and ranged from 4.62 to 5.83 GiB at 512 MiB and 3.85 to 3.87 GiB at 768 MiB. The idle sample also recorded 16.60 MiB of host writes in 2.23 seconds. Host writes include background activity, so the measured write difference cannot be attributed to Telos or used to estimate NAND wear. This comparison covers one short prompt.

The watchdog checked used swap and cumulative swap-outs about every 100 ms and would stop the worker and cancel the remaining trials at the first measured increase. Across 128 logged samples, used swap stayed at 3,869.50 MiB, cumulative swap-outs stayed at 37,061,020 pages, and reported free memory stayed at or above 57%.

[Raw trials](trials.jsonl) · [Guard and idle disk measurements](guard_and_idle.json)
