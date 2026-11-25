# Frequency admission at the 512 MiB LFM2.5 default

Native Telos at `785bf980` ran LFM2.5-8B-A1B MLX 4-bit on an M5 Pro with MLX 0.32.2 in AC High Power mode. Each fresh process used the production 512 MiB cache cap, 86 expert slots across 22 layers, the same 2 GiB admission budget, and a 512-token context. Each request generated 160 greedy tokens. The 190- and 333-token prompts were carried over from the [1 GiB comparison](../lfm25_frequency_admission_2026-11-26/README.md).

The native frequency arm used the production cache policy. The LRU control disabled the decode admission hook on its cache instance so that all its requests used the existing LRU path. Both arms used the same demand-only async reader. There were two trials per prompt and policy, with reversed policy order on the second pass. MLX-LM was absent.

| Prompt | LRU decode (tok/s) | Frequency decode (tok/s) | LRU SSD reads during decode (GiB) | Frequency SSD reads during decode (GiB) |
| --- | ---: | ---: | ---: | ---: |
| 190 input tokens | 10.26 | 14.26 | 34.77 | 27.41 |
| 333 input tokens | 10.80 | 14.12 | 35.15 | 29.64 |
| Mean | 10.53 | 14.19 | 34.96 | 28.52 |

Frequency admission improved mean decode speed by 34.8% and reduced measured whole-drive SSD reads during decode by 18.4%. Expert payload per decoded token fell from 351.20 to 287.23 MiB. Average whole-request SSD reads fell from 37.75 to 30.88 GiB. Every frequency trial was faster and read less from the SSD than either LRU trial on the same prompt. All eight trials matched all 160 generated token IDs, exact text, prompt IDs and finish reason against the corresponding 1 GiB reference.

The swap monitor checked used swap and cumulative swap-outs about every 100 ms and would cancel the comparison on the first measured increase. Both counters stayed flat across 1,159 logged samples. Used swap remained at 3,557.44 MiB and cumulative swap-outs at 37,061,020 pages. Reported free memory reached a low of 44%. Peak worker RSS was about 1,288 MiB. No guard was triggered.

SSD traffic came from whole-drive IOKit counters and includes background activity. The decode window starts immediately after the first token. The request window excludes checkpoint load. Telos's expert-read counts provide a separate measure of the application payload. This was a benchmark-only change.

[Raw trials](trials.jsonl) · [Guard and idle disk measurements](guard_and_idle.json)
