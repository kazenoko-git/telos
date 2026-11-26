# Fused NVFP4 projection for pinned Inkling decode

The experimental Metal kernel unpacks each NVFP4 weight, applies its block and global scales, and rounds to BF16 before multiplying. It uses the non-split dense MLX GEMV lane accumulation and reduction order. It runs on supported single-token BF16 decode projections. Prefill, MXFP8, training, and other shapes use the existing projection paths. [MLX 0.32.2 dequantization](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/fp_quantized.h#L2162) · [MLX 0.32.2 GEMV](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/gemv.h)

The kernel matched BF16 output bits on four synthetic matrix shapes across three seeded inputs and global scales. The shapes include both actual Inkling Mini projection dimensions. Tiny model fixtures also checked hidden states, logits, and greedy IDs before real evaluation.

Two isolated native Telos workers then ran the same trained W=16 adapter with eight pinned experts per physical bank on the M5 Pro in High Power mode. Both used a 3 GiB budget, 512-token context, MLX 0.32.2, and the same three [prompts](../inkling2_mini_pinned_w16_2026-11-26/prompts.jsonl). Two prompts generated 65 tokens across three refreshes each. The shorter probe returned 16 sampled tokens. All 146 sampled IDs were identical, including the stop token.

| Metric | Dequantize then matmul | Fused Metal |
| --- | ---: | ---: |
| Decode speed across 143 steps | 8.79 tok/s | **13.40 tok/s** |
| Decode speed across 134 steps between refreshes | 9.78 tok/s | **16.24 tok/s** |
| Mean time to first sampled token | 6.41 s | 6.40 s |
| Logical expert reads during decode | 6.16 GiB | 6.16 GiB |
| Physical SSD reads during decode | 3.26 GiB | 3.34 GiB |
| Peak sampled RSS | 2,474 MiB | 1,716 MiB |
| Peak MLX memory | 2,251 MiB | 2,161 MiB |

Decode throughput increased by 52.5% across the same steps. Both paths read zero expert bytes between refreshes. Whole-drive counters include background I/O. Used swap fell by 8 MiB during the control and stayed flat during the fused run. Cumulative swap-outs stayed flat and neither guard tripped. The kernel remains experimental and opt-in through the benchmark runner. [Raw measurements and guard records](comparison.json)
