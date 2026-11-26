# Inkling 2 Mini W=16 adapter evaluation

Telos evaluated stock Inkling, untrained W=16 routing, and Gemini’s saved step-32 W=16 LoRA adapter on the M5 Pro in High Power mode. Each variant ran in its own native MLX 0.32.2 process with a 2 GiB budget, 512-token context, and a 250.86 MiB expert cache. All variants used the restored NVFP4 projection. Stock Inkling generated fresh FP32 teacher logits, and each decode probe started with an empty expert cache. [Guarded runner](../../scripts/evaluate_inkling_window_adapter.py)

## Response scoring

The original Dolly validation file had 16 examples. One prompt also appeared in the 256-row training file with a different response, so the primary comparison excludes that row. The 15 remaining examples contain 280 scored response tokens. NLL and teacher metrics average across examples.

| Measurement | Stock | Untrained W=16 | Trained W=16 |
| --- | ---: | ---: | ---: |
| Response NLL, 15 unseen prompts | 3.697 | 4.092 | **2.089** |
| Response NLL, original 16 examples | 3.633 | 4.077 | **2.119** |
| KL from stock teacher, 15 unseen prompts | 0.000 | 1.520 | 1.017 |
| Top-1 agreement with teacher, 15 unseen prompts | 100.0% | 62.6% | 74.1% |

The trained adapter lowered response NLL on 13 of 15 unseen prompts versus stock and 15 of 15 versus untrained W=16. The short free-generation probes were mixed. The trained adapter left the grain-production answer incomplete and garbled the satellite’s name. It identified the president by surname without the stock output’s middle initial. The teacher-forced gains apply to this small Dolly split.

## Guarded decode probes

The first three validation prompts ran greedily with a cap of 16 generated tokens. Every probe used all 16 tokens. Throughput and SSD traffic cover the 15 decode steps after the first sampled token. The probes end inside the first routing window, so they do not measure the cost of a refresh.

| Measurement | Stock | Untrained W=16 | Trained W=16 |
| --- | ---: | ---: | ---: |
| Mean decode tok/s | 2.51 | 2.83 | 2.81 |
| Mean time to first sampled token | 6.63 s | 6.43 s | 6.18 s |
| Logical expert reads per decode token | 1.299 GiB | 1.002 GiB | 1.001 GiB |
| Physical SSD reads per decode token | 0.696 GiB | 0.518 GiB | 0.514 GiB |
| Physical SSD writes across the three decode windows | 11.35 MiB | 12.92 MiB | 6.42 MiB |
| Decode wall time exposed as SSD wait | 32.3% | 24.4% | 28.7% |
| Decode wall time in MLX evaluation calls | 49.6% | 56.5% | 53.2% |
| Peak sampled RSS | 1,683 MiB | 1,700 MiB | 1,377 MiB |
| Peak MLX memory | 1,375 MiB | 1,426 MiB | 1,480 MiB |

The trained adapter averaged 12.0% higher decode throughput and 26.1% lower physical decode reads versus stock. Whole-drive counters include background activity, and MLX evaluation time includes host dispatch and synchronization. Used swap fell from 2,043.62 to 2,035.62 MiB across the evaluation, with no new swap-outs and no guard trips. [Raw comparison and guard records](comparison.json)
