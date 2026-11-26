# Inkling 2 Mini W=16 with all selected experts pinned

The trained step-32 adapter ran in separate native MLX 0.32.2 processes on the M5 Pro in High Power mode. Both variants used a 3 GiB budget, 512-token context, 1,024 MiB expert cache, restored NVFP4 projection, and cold expert cache per request. The pinned variant reserved eight slots in each of 24 physical banks, with a total packed capacity of 934.03 MiB. Looped visits share the selected bank. The prefill cache policy is the same in both variants. [Prompts](prompts.jsonl) · [Guarded runner](../../scripts/evaluate_inkling_window_adapter.py)

| Measurement | Normal W=16 cache | Pin all selected experts |
| --- | ---: | ---: |
| Decode speed across 143 steps | 4.89 tok/s | **8.43 tok/s** |
| Decode speed between refresh boundaries | 5.11 tok/s | **9.34 tok/s** |
| Mean time to first sampled token | 6.67 s | 6.56 s |
| Logical expert reads during decode | 62.11 GiB | **6.16 GiB** |
| Physical SSD reads during decode | 27.43 GiB | **3.38 GiB** |
| Physical SSD writes during decode | 37.87 MiB | 8.98 MiB |
| Exposed SSD wait during decode | 5.55 s | 0.48 s |
| MLX evaluation wall time during decode | 17.01 s | 12.93 s |
| Peak sampled RSS | 1,804 MiB | 2,331 MiB |
| Peak MLX memory | 2,329 MiB | 2,251 MiB |

Both long probes generated 65 tokens and passed three refreshes. The short Dolly probe stopped after 16 sampled tokens. All 146 sampled IDs, including the stop token, matched across the two policies. The seasons answer became repetitive, the food-waste list had garbled items, and the short factual answer gave the correct surname. Pinning preserved those outputs. The [earlier held-out response-loss evaluation](../inkling2_mini_w16_adapter_2026-11-26/README.md) provides a separate quality comparison against stock Inkling.

Pinned decode read experts at step 1 and at refresh steps 17, 33, and 49 on both long prompts. It read zero expert bytes on all 134 other decode steps. The first load fetched 866–900 MiB per probe, and the six later refreshes fetched 581–663 MiB each. Whole-drive measurements include background I/O. Swap stayed at 2,035.62 MiB across both variants, with zero new swap-outs and zero guard trips. [Raw timing, read, and guard records](comparison.json)
