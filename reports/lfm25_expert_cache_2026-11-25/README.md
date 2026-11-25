# LFM2.5 Flash-MoE expert cache comparison

The installed LiquidAI LFM2.5-8B-A1B 4-bit model ran on the M5 Pro with MLX 0.32.2, AC power and High Power mode. All four SSD variants used the same 2 GiB admission budget, 128-token context, 16-token prompt and 16-token greedy limit. Each configuration ran once in a fresh process and exited before the next load. The benchmark used native Telos without MLX-LM. The experiments varied the expert cache only and left the runtime defaults in place.

The per-layer prototypes divided whole-expert slots as evenly as possible across the 22 routed layers and evicted the least recently used expert within each layer. The same overall RAM budget and original admission reserves applied to both cache sizes.

| Policy and cache cap | Decode hits | Mean read per decode token | Decode speed | MLX peak |
| --- | ---: | ---: | ---: | ---: |
| Global LRU, 512 MiB | 11 / 1,320 (0.8%) | 515.4 MiB | 2.90 tok/s | 964 MiB |
| Global LRU, 768 MiB | 681 / 1,320 (51.6%) | 251.6 MiB | 5.25 tok/s | 1,224 MiB |
| Per-layer LRU, 512 MiB | 492 / 1,320 (37.3%) | 326.0 MiB | 4.35 tok/s | 964 MiB |
| Per-layer LRU, 768 MiB | 699 / 1,320 (53.0%) | 244.5 MiB | 5.26 tok/s | 1,224 MiB |

Decode covers the 15 steps after the first generated token. All four configurations read 2,557.4 MiB of expert payload for prefill. The byte totals count bytes requested by Telos and do not measure physical drive traffic.

Without hits, each decode step reads 519.75 MiB of experts. The 512 MiB global policy averaged 515.4 MiB. Under the same cap, per-layer eviction cut reads by 36.7% with the same measured MLX peak. At 768 MiB, per-layer eviction cut reads by a further 2.8% relative to global LRU. Their single-run speeds were essentially tied. The 2.90 tok/s baseline also differs from the previous 4.82 tok/s on this prompt, so the hit and byte counts are stronger evidence than the timing deltas.

All four runs reproduced all 16 greedy token IDs and exact text from the [resident reference](../lfm25_flash_moe_2026-11-25/resident.json). The watchdog checked used swap and cumulative swap-outs about every 100 ms from before each load. Used swap stayed at 4,309.50 MiB, the cumulative swap-out count stayed at 37,061,020 pages, and reported free memory stayed at or above 67%.

[Global 512 MiB](global_512.json) · [Global 768 MiB](global_768.json) · [Per-layer 512 MiB](layer_512.json) · [Per-layer 768 MiB](layer_768.json) · [Guard samples](samples.jsonl)
