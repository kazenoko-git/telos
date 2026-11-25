# LFM2.5 demand-only expert read experiments

Native Telos at `0d30b5e9` ran the LFM2.5-8B-A1B MLX 4-bit checkpoint on an M5 Pro with MLX 0.32.2 in AC High Power mode. Each inference trial used a fresh process, a 2 GiB admission budget, a 1 GiB per-layer-partitioned cache with 173 total expert slots, and the same three [longer-prompt workloads](../lfm25_async_cache_long_prompts_2026-11-25/README.md). The 94-, 190-, and 333-token prompts generated 32, 40, and 40 greedy tokens respectively. The variants ran independently with demand-only reads and no MLX-LM.

| Average across the three prompts | Equal-quota LRU | Packed expert reads | Uneven layer quotas | Frequency admission | One evaluation per decode layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 15.83 | 14.70 | 15.61 | 17.06 | 16.05 |
| Time to first token (ms) | 2,304 | 1,053 | 2,267 | 2,207 | 2,204 |
| Expert payload per decode token (MiB) | 223.24 | 223.24 | 218.77 | 195.91 | 223.24 |
| Whole-drive reads during decode (GiB) | 5.00 | 7.89 | 4.90 | 4.37 | 5.00 |
| Whole-drive reads per request (GiB) | 7.74 | 11.44 | 7.16 | 6.62 | 7.25 |
| Whole-drive writes per request (MiB) | 3.41 | 1.23 | 16.03 | 2.76 | 5.31 |
| Expert read calls during decode | 12,312 | 1,368 | 12,066 | 10,746 | 12,312 |
| Routed output evaluations during decode | 3,197 | 3,197 | 3,197 | 3,197 | 799 |

There was one trial per prompt and variant, plus a repeat of the 190-token baseline at the end. That baseline measured 16.29 and 16.12 tok/s. All 16 trials matched prompt IDs, generated IDs, exact text, and finish reason per prompt. The three baselines also matched the prior longer-prompt comparison.

## One read per expert

The prototype wrote an uncompressed temporary 4.06 GiB sidecar of the exact checkpoint expert slices. It reused the existing staging allowance and read one contiguous block per miss in place of nine tensor reads. Expert payload and misses were identical to baseline. Time to first token fell 54% on average and total request time fell 21% on these short generations. During decode, average throughput fell 7% and measured whole-drive reads rose 58%. The original layout registered fewer device bytes than application payload, while the packed layout registered approximately one device byte per requested byte. The experiment did not isolate the source of that difference. No cache purge was performed.

## Reuse at the same memory cap

Uneven layer quotas were fitted on the other two traces for each held-out prompt. The optimization assigned 4 to 16 entries per layer by minimizing recorded decode misses at the same 173-slot total, using the original quotas as a tie-breaker. Frequency admission counted decode accesses, evicted the least frequently accessed resident with LRU tie-breaking, and bypassed insertion for a miss with fewer accesses than that resident. Prefill retained standard LRU.

| Decode misses | Equal-quota LRU | Uneven layer quotas | Frequency admission |
| --- | ---: | ---: | ---: |
| 94 prompt tokens | 1,233 | 1,208 | 1,158 |
| 190 prompt tokens | 1,414 | 1,399 | 1,203 |
| 333 prompt tokens | 1,457 | 1,415 | 1,221 |
| Total | 4,104 | 4,022 | 3,582 |

The live runs matched replay miss counts for every prompt and policy. Frequency admission reduced misses by 12.7% and whole-drive decode reads by 12.7%. Decode throughput rose from 16.29 to 18.89 tok/s on the 190-token prompt and from 15.67 to 17.20 on the 333-token prompt. It fell from 15.54 to 15.08 on the 94-token prompt. Uneven quotas reduced misses by 2.0% with mixed throughput.

## Routed evaluation and read overlap

The prototype deferred the routed output evaluation to the end of each single-token layer, preserving the same expert accumulation order and holding at most one layer's selected weights. Prefill retained the original evaluation schedule. The change removed 75% of routed output evaluation calls during decode. Measured time inside those calls fell about 0.27 seconds per request, while time waiting at the read consumer rose about 0.27 seconds. The average decode speed rose 1.4% and the result varied by prompt.

## Memory and measurement

The watchdog enforced a 2,048 MiB process RSS ceiling, a 768 MiB free-swap floor, and free-memory floors of 50% before launch and 40% while running. It would stop work at the first measured increase in used swap or cumulative swap-outs. Across 1,248 logged samples there was no increase in either. Free memory reached a low of 42%. Peak process RSS over all requests was 1,804 MiB. The packed sidecar was removed after the comparison.

SSD traffic came from whole-drive IOKit counters and includes background activity. The request window excludes load and sidecar construction. The decode window starts just after the first token. Idle runs also recorded writes, so request write totals cannot be attributed to Telos.

The production demand-only implementation was unchanged. Its default cap remains 512 MiB.

[Raw trials](trials.jsonl) · [Cache policy replay](replay.json) · [Routing traces](routing.jsonl) · [Memory guards and disk accounting](guard_and_idle.json)
