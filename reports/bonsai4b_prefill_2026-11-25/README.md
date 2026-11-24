# Bonsai 4B final-position prefill

Native Qwen3 generation now computes one query in its final layer during prompt prefill. The final layer still builds keys and values for every prompt position. Earlier positions in that layer need no attention output or MLP result because generation only reads the last hidden state. The API that returns all hidden states still computes them all.

## Public generation A/B

Tested the original Bonsai 4B 2-bit weights on the M5 Pro with MLX 0.32.2, AC power and High Power mode. The comparison used eight alternating pairs per workload in a single process with shared model weights and an 8-token warmup per implementation. The baseline Qwen3 model came from `24d13502`. The timings include tokenizing and streaming through `TextGenerator.generate`. A background Git task and desktop activity were present.

Both workloads used greedy decoding and a 1,024-token context. The first used the [matched benchmark prompt](../bonsai4b_native_inference_2026-11-24/benchmark.py). The second joined four copies of the same prompt with blank lines and requested eight output tokens.

| Workload | Baseline median first token | Final-position median first token | Median paired change |
| --- | ---: | ---: | ---: |
| 212 prompt tokens, 192 generated | 115.4 ms | 111.5 ms | −2.5% |
| 848 prompt tokens, 8 generated | 347.1 ms | 342.5 ms | −1.7% |

The first token arrived faster in six of eight pairs for each workload. On the 192-token workload, sustained decode was 141.8 versus 142.1 tok/s and the MLX allocation peak stayed at 1,607 MiB. With the 848-token prompt, the peak moved from 1,989.5 to 1,986.2 MiB. The trial set reproduced all 192 token IDs and exact text from the previous matched benchmark. All eight generated IDs and exact text also matched in every long-prompt pair.

## Other screened changes

- Delivering the first token before queuing lookahead reduced first-token latency by a paired median of 2.3% over eight pairs. Total request time rose by 0.4% and peak MLX allocation was unchanged. The runtime keeps the existing lookahead schedule.
- Folding the YaRN query and key amplitude into the attention scale raised paired median decode speed by 2.4% over eight pairs. All 192 IDs matched on the benchmark, but peak MLX allocation increased by 42 MiB and the rearrangement changes low-precision rounding. The runtime keeps the existing YaRN arithmetic.

All trials used native Telos and Bonsai 4B. [Public generation timing and parity records](trials.jsonl)
