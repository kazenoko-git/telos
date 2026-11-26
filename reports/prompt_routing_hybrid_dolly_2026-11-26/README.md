# Periodic expert refresh on Dolly instructions

This experiment reruns the fixed-prompt and token-level baselines alongside two hybrids. The hybrids select one expert per layer from the prompt and refresh after consuming every four or eight response tokens. The new expert begins predicting the next token. Each refresh uses the current causal hidden state, and the forward pass reuses the chosen expert until the next boundary in training and decode. A fixture checks the router call cadence and compares incremental logits with teacher forcing across multiple refreshes.

The four modes share the same 336,704-parameter model, tokenizer, 2,940 training examples, 351 held-out examples, and the same 32 decode prompts as the previous Dolly comparison. Each seed used identical initialization and training batches across its four modes, with 1,000 AdamW updates at effective batch 32, microbatch 16, and two accumulation steps. The run verifies token counts against the previous experiment for each seed. [Previous Dolly setup and results](../prompt_routing_dolly_2026-11-26/README.md)

| Measurement | Token-level | Prompt-fixed | Refresh every 4 | Refresh every 8 |
| --- | ---: | ---: | ---: | ---: |
| Mean held-out response NLL | 4.318 | 4.655 | 4.629 | 4.632 |
| Mean held-out next-token accuracy | 19.72% | 17.72% | 17.73% | 17.17% |
| Mean greedy word overlap F1 | 14.35% | 12.65% | 13.75% | 14.03% |
| Exact greedy matches across seeds | 0/96 | 0/96 | 0/96 | 0/96 |
| Mean SSD-backed decode | 650 tok/s | 996 tok/s | 938 tok/s | 988 tok/s |
| Mean time to first token | 4.25 ms | 2.37 ms | 1.99 ms | 2.14 ms |
| Mean whole-drive reads per 32 requests | 191.24 MiB | 16.58 MiB | 32.90 MiB | 21.79 MiB |
| Mean expert payload reread during decode per 32 requests | 87.22 MiB | 0 MiB | 20.09 MiB | 8.13 MiB |

Refresh-4 and refresh-8 recovered about 8% and 7% of the fixed-prompt loss gap to token routing, respectively. Refresh-8 kept approximately fixed-prompt decode speed and cut expert rereads by 91% versus token-level routing. The greedy F1 difference was smaller than the NLL gap, and the tiny models produced no exact reference matches.

Benchmarks ran on the M5 Pro under AC High Power mode with MLX 0.32.2. Each seed timed fixed/token/refresh-4/refresh-8 and then reversed the order. All trials used a fresh 192 KiB expert cache per request and a fixed 24-token decode; text quality was scored up to EOS or the cap. Whole-drive reads include background activity and exclude checkpoint copies and model load. Two token-level trials recorded about 300 MiB of whole-drive reads, while the corresponding repeats recorded about 132 and 143 MiB. Application-level expert bytes give the more direct account of rereads. All three prompt-based modes read exactly 6 MiB of expert payload per 32 prefills.

The watchdog stopped the first attempt after cumulative swap-outs advanced. The comparison restarted after swap stabilized, with a 2 GiB MLX allocator limit and a 128 MiB allocator-cache limit. The completed attempt recorded no growth in used swap or cumulative swap-outs across 3,063 samples. Resident and SSD logits matched exactly at 96 checked token steps per checkpoint across all 12 trained checkpoints. Targeted tests passed 24 with one skip.

```bash
telos pretrain routing --routing all --data /path/to/databricks-dolly-15k.jsonl \
  --steps 1000 --seed 42 --output /tmp/telos-dolly-hybrid-42
```

[Raw measurements](results.jsonl) · [Dataset and guard metadata](metadata.json)
