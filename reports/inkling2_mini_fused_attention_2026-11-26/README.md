# Attention and routing in fused pinned Inkling decode

Native Telos ran the trained W=16 adapter on the M5 Pro in High Power mode with fused NVFP4, eight resident experts per physical bank, a 3 GiB budget, and a 512-token context. Each worker used the same three [prompts](../inkling2_mini_pinned_w16_2026-11-26/prompts.jsonl). In all four workers, the 146 sampled IDs matched the previous fused benchmark exactly. The 134 steady decode steps per worker had zero logical expert reads and zero exposed SSD wait. Swap used and cumulative swap-outs stayed flat.

| Timing method | Matched control | Profiled steady speed | Attention per token | Routing per token |
| --- | ---: | ---: | ---: | ---: |
| All 48 layer visits timed on every token | 15.27 tok/s | 8.76 tok/s | 21.43 ms | 9.23 ms |
| Four of 48 visits timed, rotating across tokens | 16.08 tok/s | 15.96 tok/s | **20.83 ms estimated** | **8.89 ms estimated** |

The sampled report scales 536 timed visits to the 48 visits per token. Attention was about 33% of the sampled run’s 62.65 ms per steady token, and routing was about 14%. The sampled run stayed within 0.74% of its matched control throughput. The full instrumentation added synchronization at every layer visit and slowed decode substantially.

Attention timing covers the attention module, including input projections, short Q/K/V convolutions where configured, attention scoring, gating, and output projection. The residual mix and normalization before attention and before the router are fenced separately. Routing timing covers evaluation of the per-token gate weights and materialization of expert IDs. The IDs are already pinned between refreshes. These are host-wall estimates around MLX evaluation, with dispatch and synchronization included. Python graph construction occurs outside the measured sections.

[Full profile and matched control](full_profile.json) · [Sampled profile and matched control](sampled_profile.json)
