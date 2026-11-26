# Inkling attention components and grouped projections

Native Telos compared the trained W=16 adapter with fused NVFP4 and pinned experts on an M5 Pro in High Power mode. Every worker used a 3 GiB budget, 512-token context, MLX 0.32.2, and the same three [prompts](../inkling2_mini_pinned_w16_2026-11-26/prompts.jsonl). The component run and both grouped-projection comparisons matched all 146 sampled IDs. Expert reads were zero on the 134 decode steps between refreshes in every worker. Used swap and cumulative swap-outs stayed flat, and no guard tripped.

## Attention components

The profiler rotates through four of 48 layer visits per token. It fences each component separately for 536 sampled visits and scales the results by 12. The matched control sustained 15.19 tok/s between refreshes, and the component profile sustained 13.78 tok/s. [Component profile and control](component_profile.json)

| Separately fenced component | Extrapolated wall time per token |
| --- | ---: |
| Packed linear projections, including LoRA and output projection | 31.50 ms |
| Attention setup, including KV state, relative bias, mask and gating | 21.26 ms |
| Short K/V convolutions, including dynamic coefficient projection | 9.62 ms |
| Scaled dot-product attention kernel | 7.21 ms |

These figures rank the instrumented sections. Separate fences add dispatch and synchronization to each section, and the projections use three fences. The extrapolated numbers do not sum to the earlier [20.8 ms whole-module estimate](../inkling2_mini_fused_attention_2026-11-26/README.md). The probe measures the K/V convolution inside attention. The separate post-attention residual convolution is outside this module.

## Group the query-side projections

The experiment packs Q, relative features, sink, and gate rows into one affine matrix per attention layer. Those four projections share the query input even when loop messages give K and V separate inputs. The individual base projections use views of the grouped packed storage, and the grouped output preserves each projection’s existing LoRA delta. The experimental path was active in all 24 physical layers. Tiny model tests with active LoRA matched hidden state bits and logits. A direct base-projection probe also found bit equality on real layers 6 and 23 using 16 seeded BF16 inputs.

| Comparison | Separate overall | Grouped overall | Separate between refreshes | Grouped between refreshes |
| --- | ---: | ---: | ---: | ---: |
| Separate first | 12.98 tok/s | 13.34 tok/s | 15.73 tok/s | 15.88 tok/s |
| Grouped first | 13.48 tok/s | 13.32 tok/s | 16.18 tok/s | 16.16 tok/s |
| Pooled across both orders | 13.23 tok/s | 13.33 tok/s | 15.95 tok/s | 16.02 tok/s |

The pooled steady-decode gain was 0.42%, and the direction changed when the order reversed. Peak MLX memory was about 2,160–2,161 MiB in all four workers. Physical SSD reads during decode ranged from 3.15 to 3.30 GiB across the three prompts per worker. The grouped path is opt-in through the benchmark runner. [Separate first](grouped_forward.json) · [Grouped first](grouped_reversed.json)
