# Frequency-based expert admission on longer LFM2.5 generations

Native Telos ran LFM2.5-8B-A1B MLX 4-bit on an M5 Pro with MLX 0.32.2 in AC High Power mode. Each fresh process used a 2 GiB admission budget, a 1 GiB per-layer-partitioned expert cache, 512-token context, and 160 generated tokens at temperature zero. The benchmark used the same 190- and 333-token [prompts](../lfm25_demand_optimizations_2026-11-26/README.md) as the earlier short-generation comparison. Reads were demand-only and MLX-LM was absent.

The comparison used a process-local 1 GiB cap and frequency policy prototype on `77d2ac62`, with two trials per prompt and policy. It reversed policy order on the second pass for each prompt. The LRU and frequency arms used the same timing and read accounting.

| Prompt | LRU decode (tok/s) | Frequency decode (tok/s) | LRU SSD reads during decode (GiB) | Frequency SSD reads during decode (GiB) |
| --- | ---: | ---: | ---: | ---: |
| 190 input tokens | 14.13 | 16.42 | 21.29 | 16.10 |
| 333 input tokens | 14.36 | 15.95 | 22.21 | 18.83 |
| Mean | 14.25 | 16.19 | 21.75 | 17.46 |

Across both prompts and repeats, frequency admission improved decode speed by 13.6% and reduced measured whole-drive SSD reads during decode by 19.7%. Expert payload per decode token fell from 217.16 to 175.03 MiB. On a miss, the policy uses request-local decode access counts to pick an eviction victim. LRU breaks ties, and a requested expert with a lower count than the victim is read for the current computation without entering the cache. Prefill retains LRU. Production code keeps the existing 512 MiB cap and enables frequency admission for LFM's per-layer cache during decode.

Two follow-up runs with the integrated policy in `73a1a8e` matched all 160 token IDs, exact decoded text, and every expert-read step against the corresponding prototype output. They used the same process-local benchmark cap. The 190-token prompt recorded 19.07 tok/s and 16.06 GiB of host SSD reads during decode. The 333-token prompt recorded 17.95 tok/s and 19.00 GiB. These follow-ups verified integration in a separate batch. The interleaved comparison supplies the throughput estimate.

Every completed trial matched prompt IDs, generated IDs, text, and finish reason for its prompt. The initial baseline was stopped when reported free memory fell to 39%, below its 40% runtime floor. Swap remained flat. The completed comparison used a 45% preflight floor, a 35% runtime floor, a 768 MiB free-swap floor, and a 2,048 MiB RSS ceiling. It retained the first-increase swap stop. Across 1,308 samples including the stopped attempt, used swap stayed at 3,557.44 MiB and cumulative swap-outs stayed at 37,061,020 pages.

SSD traffic is from whole-drive IOKit counters and includes background activity. The decode window starts immediately after the first token. Read counts from the expert cache establish application payload separately. The request window excludes checkpoint load. Tests covering LFM2.5, inference, and Qwen3 reported 21 passed and one skipped.

[Raw trials](trials.jsonl) · [Guard and idle disk measurements](guard_and_idle.json)
