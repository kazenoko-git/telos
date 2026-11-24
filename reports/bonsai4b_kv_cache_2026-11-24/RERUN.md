# Bonsai 4B matched rerun and cache outlier

The M5 Pro ran on AC in High Power mode. Ardia was paused, and the pre-run CPU sample was 89.7% idle. A background Git process still used roughly one core, and the sampled GPU utilization before the run was 33%. The system reported no thermal or performance warnings. These measurements represent a quieter desktop session with residual activity.

## Native Telos and LM Studio

Both runtimes loaded the original Ternary Bonsai 4B 2-bit Safetensors weights. The file matched SHA-256 `1f956eefc2fa23fc21048ac2d1a8e76d78558666d81bf6b1fa89d83ba59698ac`. The [same benchmark harness](../bonsai4b_native_inference_2026-11-24/benchmark.py) verified identical 212 prompt IDs, a 1,024-token context, a 192-token budget, greedy sampling and zero drafted tokens. Four fresh loads per runtime followed Telos, LM Studio, LM Studio, Telos twice, with an 8-token warmup for each trial. Each runtime held the model alone. The LM Studio client SDK was 1.5.0 and native Telos used MLX 0.32.2.

| Median | Native Telos | LM Studio |
| --- | ---: | ---: |
| Time to first delivered token | 113.5 ms | 197.7 ms |
| Decode, shared 191-token window | 145.19 tok/s | 124.33 tok/s |
| Request time | 1.427 s | 1.723 s |

Telos decoded 16.8% faster in the shared window. Its MLX allocator peaked at 1,607 MiB. The LM Studio sampled process-tree RSS rose by a median of 1,868 MiB over its unloaded baseline. The memory metrics measure different scopes.

Every native run reproduced all 192 greedy token IDs and exact text from the preceding [matched benchmark](../bonsai4b_decode_2026-11-24/README.md). Every LM Studio run reproduced its prior text and matched the first 813 shared characters after its leading-space trim. LM Studio streamed 191 tokens and Telos streamed 192. The LM Studio SDK stream exposes fragment text and token counts for this check.

## Cache A/B

Twelve pairs compared indexed writes at `2cd27b53` with the concatenating cache at `01b6faac`. Both variants shared the loaded weights in one process, ran an 8-token warmup, and alternated order in successive pairs. Decode rates here span tokens 1 through 192.

| Median | Concatenating cache | Indexed cache |
| --- | ---: | ---: |
| Decode | 109.49 tok/s | 111.19 tok/s |
| Time to first token | 154.5 ms | 159.5 ms |
| MLX allocator peak | 1,607 MiB | 1,607 MiB |

The median paired decode gain was **1.4%**, with eight faster pairs out of twelve. The first pair was 22.7% slower with indexed writes. Keeping the other eleven pairs gives a 2.6% median gain. This run did not reproduce the earlier 3.6% estimate and leaves the size of the effect uncertain. All 24 runs matched the expected 192 token IDs.

## Slow trial

The original 50.7 tok/s trial took 125.8 ms to deliver its first token and about 3.77 s to deliver the remaining tokens. Its paired baseline took 116.9 ms and 1.45 s. The original trial retained aggregate timing only, so the exact cause is unresolved.

A separate earlier fresh-process run at 63.5 tok/s retained a token trace. Its final 32 token intervals consumed 1.47 s of the 3.01 s decode window, with the largest gaps at tokens 181–185. The cache crossed its only growth boundary around token 46. In the new first pair, the indexed run began with slower token delivery and sped up across the run. Its largest gap was 14.8 ms, and the median gap at token 46 across indexed runs was 9.1 ms, close to the surrounding 9.2 ms median. The traces show multi-token rate changes and give no evidence of a recurring stall at cache growth.

[Matched trials](matched_rerun.jsonl) · [Paired cache trials with token timing](cache_rerun.jsonl)
