# Combined LFM2.5 expert read optimization

Tested native Telos on the installed LFM2.5-8B-A1B MLX 4-bit checkpoint on the M5 Pro with AC power, High Power mode, and MLX 0.32.2. All requests used a 512 MiB per-layer cache, a 2 GiB admission budget, a 128-token context, and the same 16-token prompt and 16-token greedy limit. Every trial ran in a fresh process and exited before the next process started. No MLX-LM was used.

The prototype comparison ran baseline, async alone, combined, combined, async alone, baseline. The integrated path then ran twice through normal Telos generation. Both combined paths reused descriptors and read buffers, and issued concurrent reads for current-layer experts selected by the actual router. They preserved expert accumulation order.

| Median of two runs | Baseline | Async alone | Combined prototype | Integrated path |
| --- | ---: | ---: | ---: | ---: |
| Decode | 4.39 tok/s | 6.58 tok/s | 10.87 tok/s | 11.36 tok/s |
| Time to first token | 1,594 ms | 1,705 ms | 1,351 ms | 1,385 ms |
| MLX peak during generation | 964 MiB | 964 MiB | 964 MiB | 964 MiB |
| Peak process RSS | 1,264 MiB | 1,288 MiB | 1,283 MiB | 1,280 MiB |
| Host SSD reads per request | 4.65 GiB | 4.65 GiB | 4.76 GiB | 4.73 GiB |
| Host SSD writes per request | 28.2 MiB | 26.8 MiB | 20.2 MiB | 0.54 MiB |

The combined prototype decoded 2.48× as fast as the baseline in the interleaved comparison. The integrated path decoded at 11.18 and 11.53 tok/s in the separate verification runs. All eight runs reproduced all 16 token IDs and exact decoded text from the [resident reference](../lfm25_flash_moe_2026-11-25/resident.json). Each run recorded 492 cache hits out of 1,320 decode lookups and requested 7,447.8 MiB of expert payload including prefill. The optimization preserved cache behavior and expert payload traffic.

Device-level deltas came from whole-drive IOKit block counters on the checkpoint's backing SSD and cover the full generation request without load. They include background jobs and do not represent internal NAND writes. Idle samples also recorded host writes. The integrated verification occurred after the interleaved comparison, so the write difference does not establish an attributable reduction.

The watchdog checked used swap and cumulative swap-outs about every 100 ms and would stop the run and cancel the remaining trials at the first measured increase. It also tracked process RSS and system free memory. Used swap remained at 3,869.50 MiB and cumulative swap-outs remained at 37,061,020 pages. The completed runs stayed at or above 58% free memory. An initial comparison stopped before its second arm at the 60% preflight floor with swap flat. The completed comparison used a 55% preflight floor. Both used a 40% runtime free-memory floor, a 2,500 MiB RSS ceiling, and a 768 MiB free-swap floor.

[Raw trials](trials.jsonl) · [Guard and idle disk measurements](guard_and_idle.json)
