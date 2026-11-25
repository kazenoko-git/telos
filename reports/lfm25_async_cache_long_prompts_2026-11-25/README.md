# LFM2.5 demand-only expert cache on longer prompts

Native Telos at `aa64f0d9` ran LFM2.5-8B-A1B MLX 4-bit on an M5 Pro in AC High Power mode with MLX 0.32.2. Each trial ran in a fresh process with 22 per-layer cache partitions, the 2 GiB admission budget, async demand reads, greedy decoding, and no warmup or MLX-LM. The three prompts had 94, 190 and 333 input tokens. They generated 32, 40 and 40 tokens respectively. The raw trials preserve prompt text, token IDs, exact output and run order.

The requested 2,048 MiB cap exercised the planner ceiling. At 128-token context it admitted 1,057.88 MiB, or 179 expert entries. At 512-token context the larger KV and workspace reserves reduced the ceiling to 1,027.88 MiB, or 174 entries. The 1 GiB cap admitted 173 entries and 768 MiB admitted 130. A true 2 GiB cache at the 512-token context requires a planner budget of at least 3,068.12 MiB. A decimal 2 GB cache requires at least 2,927.47 MiB. These are header-only planner estimates for higher budgets. All live trials stayed within the 2 GiB admission budget. Production code remains demand-only with its 512 MiB default.

| Prompt tokens | Context | Effective cache cap | Decode (tok/s) | SSD request reads (GiB) | SSD decode reads (GiB) | SSD request writes (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 94 | 128 | 768 MiB | 11.90 | 7.45 | 5.28 | 32.90 |
| 94 | 128 | 1 GiB | 13.29 | 6.70 | 4.52 | 34.86 |
| 94 | 128 | 1,057.88 MiB | 12.42 | 6.76 | 4.59 | 25.77 |
| 190 | 512 | 768 MiB | 11.71 | 10.39 | 6.69 | 40.88 |
| 190 | 512 | 1 GiB | 12.86 | 7.71 | 5.40 | 40.46 |
| 190 | 512 | 1,027.88 MiB | 13.26 | 7.48 | 5.15 | 24.06 |
| 333 | 512 | 768 MiB | 11.23 | 9.07 | 6.68 | 49.19 |
| 333 | 512 | 1 GiB | 12.86 | 8.12 | 5.36 | 32.30 |
| 333 | 512 | 1,027.88 MiB | 12.74 | 7.82 | 5.32 | 25.98 |

Averaged across the three prompts, 1 GiB decoded 12.0% faster and reduced measured whole-drive SSD reads during decode by 18.1% versus 768 MiB. Telos requested 18.4% less expert payload per decode token, from 273.42 to 223.24 MiB. The ceiling cut expert payload by another 1.3%, with mixed decode-speed results across the three prompts. Peak process RSS reached 1,541 MiB at 768 MiB, 1,795 MiB at 1 GiB, and 1,831 MiB at the ceiling. Each prompt and cap combination has one trial.

SSD counts come from whole-drive IOKit block counters, including background activity. The request window excludes checkpoint load. The decode window starts at a disk sample immediately after the first token, and decode timing excludes that sample. Idle reads and writes also occurred, so request writes cannot be attributed to Telos or used to estimate NAND wear.

Every cap reproduced the exact generated token IDs, decoded text and finish reason for each prompt. The watchdog checked used swap and cumulative swap-outs about every 100 ms and would kill the worker and cancel remaining trials at the first measured increase. Across 494 logged samples, used swap stayed at 3,741.50 MiB, cumulative swap-outs stayed at 37,061,020 pages, and reported free memory never fell below 49%.

[Raw trials](trials.jsonl) · [Guard, planner and idle disk measurements](guard_and_idle.json)
