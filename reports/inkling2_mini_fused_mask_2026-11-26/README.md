# Fused Inkling relative bias and decode mask

The experimental Metal kernel gathers existing projected relative logits, applies the sliding-window cutoff or full-attention bias rules, and appends the learned sink logit in one pass. The relative projection and scaled dot-product attention use the existing MLX operations. The kernel runs during single-token decode. Tiny BF16 and FP32 fixtures matched mask bits across sliding and full attention, including context and window boundaries. Tiny model fixtures also matched hidden states, logits, and sampled token IDs.

Native Telos ran the trained W=16 adapter on the M5 Pro in High Power mode at 3 GiB and 512-token context. Both variants used fused NVFP4, eight pinned experts per physical bank, separate query projections, and the same three [prompts](../inkling2_mini_pinned_w16_2026-11-26/prompts.jsonl). One comparison ran the control first and the other ran the fused mask first. Both long prompts crossed three refreshes. The short prompt stopped after 16 sampled tokens. All 146 sampled IDs matched across all four workers and matched the earlier fused NVFP4 benchmark.

| Comparison | Existing overall | Fused overall | Existing between refreshes | Fused between refreshes |
| --- | ---: | ---: | ---: | ---: |
| Existing first | 12.67 tok/s | 13.01 tok/s | 15.31 tok/s | 15.94 tok/s |
| Fused first | 12.91 tok/s | 13.24 tok/s | 15.69 tok/s | 16.09 tok/s |
| Pooled across both orders | 12.79 tok/s | **13.12 tok/s** | 15.50 tok/s | **16.01 tok/s** |
| Pooled long prompts only | 13.14 tok/s | 13.21 tok/s | 15.61 tok/s | 15.89 tok/s |

The pooled gain was 2.6% overall and 3.3% between refreshes. The short prompt accounted for much of the improvement. On the two longer prompts, the gains were 0.5% overall and 1.8% between refreshes.

Peak MLX memory stayed about 2,161 MiB. Logical expert reads remained 6.16 GiB across the three decode probes per worker, including zero expert reads between refreshes. Physical decode reads ranged from 3.23 to 3.32 GiB, with whole-drive background I/O included. Used swap stayed at 2,011.62 MiB and cumulative swap-outs stayed flat across all workers. The fused mask is opt-in through the benchmark runner. [Existing first](forward.json) · [Fused first](reversed.json)
