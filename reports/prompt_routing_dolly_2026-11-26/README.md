# Prompt-fixed routing on real instructions

We trained matched 336,704-parameter causal MLX models from scratch on short, complete examples from Databricks Dolly 15k, revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` (CC BY-SA 3.0). The models share a two-layer backbone and have four routed experts and a shared dense FFN per layer. Prompt-fixed routing selects one expert per layer from the prompt. Token-level routing selects one expert per layer from every token's causal state. [Dataset card](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a)

Normalized instructions determine the split, keeping repeated instructions with different contexts together. The shared 512-token byte-level BPE tokenizer was trained only on the training split. We kept 2,940 training and 351 held-out instruction/response pairs that fit 96 tokens without truncation. Loss and token accuracy cover response targets and EOS. Right padding contributes neither targets nor token-routing balance. Both modes used the same initialization and sampled batches within each seed. Seeds 42, 43, and 44 each used 1,000 AdamW updates with a microbatch of 16 and two accumulation steps. Each matched pair saw roughly 1.91 million non-padding tokens and 1.00 million supervised response tokens per model.

| Measurement | Token-level | Prompt-fixed |
| --- | ---: | ---: |
| Mean held-out response NLL | 4.327 | 4.655 |
| Mean held-out response token accuracy | 19.80% | 17.72% |
| Mean greedy word overlap F1 | 13.71% | 12.65% |
| Exact greedy matches across seeds | 0/96 | 0/96 |
| SSD-backed decode | 658 tok/s | 979 tok/s |
| Mean time to first token | 4.42 ms | 2.56 ms |
| Whole-drive SSD reads per 32 requests | 136.22 MiB | 7.69 MiB |
| Expert payload during prefill per 32 requests | 23.41 MiB | 6.00 MiB |
| Expert payload reread during decode per 32 requests | 88.13 MiB | 0 MiB |

Teacher-forced quality covers 10,735 response targets per seed. Token-level routing had lower held-out loss on every seed. Greedy outputs from both small models frequently contained invented words or generic fragments. Word overlap and exact matching against one reference are limited measures of instruction quality.

The M5 Pro ran with AC High Power mode and MLX 0.32.2. Both routes started each request with an empty 192 KiB expert cache, enough for one expert per layer. Each seed benchmarked the same 32 held-out prompts in prompt/token/token/prompt order. Every run timed 24 generated tokens for matched work, while greedy quality scored the text before the first EOS or the cap. Sustained decode excludes prefill. Whole-drive IOKit reads include background traffic and exclude checkpoint copies and model load. These tiny-model timings measure relative behavior on this workload.

Resident and SSD logits matched exactly on 96 token steps per checkpoint, with six checkpoints tested. Used swap and cumulative swap-outs stayed flat across 939 watchdog samples. Targeted tests produced 18 passes and one skip. The synthetic CLI path also passed a smoke run.

```bash
dataset=$(hf download databricks/databricks-dolly-15k databricks-dolly-15k.jsonl \
  --repo-type dataset --revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a)
telos pretrain routing --data "$dataset" --steps 1000 --seed 42 --output /tmp/telos-dolly-42
```

[Raw trials and quality metrics](results.jsonl) · [Dataset, tokenizer and guard metadata](metadata.json)
