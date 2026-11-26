# Tiny prompt-fixed vs. token-level expert routing

This MLX experiment jointly trains a causal Telos backbone and top-one routers. Both variants have 306,496 parameters, two transformer layers, a shared dense SwiGLU and a four-expert routed SwiGLU bank at each layer. The prompt variant selects one expert per layer from the mean of prompt embeddings and reuses it throughout the continuation. The token variant selects from each token's causal hidden state. Both use a straight-through router and the same load-balancing loss. The prompt's pooled representation excludes continuation tokens, and only continuation targets enter the language-model loss. This uses prompt-fixed routing as an architectural idea from Apple's work. [Instruction-Following Pruning paper](https://machinelearning.apple.com/research/pruning-large-language)

Each seed used the same initialization and sampled training batches for both modes. Seeds 42, 43, and 44 trained for 1,000 AdamW updates with a microbatch of 16 and two accumulation steps, for 32,000 sampled sequences per model. The synthetic corpus has four fixed symbol transformations with 819 training examples and 205 held-out prompt combinations. Each example has five prompt tokens and 24 continuation tokens. Teacher-forced quality uses all held-out combinations. Greedy checks use the same fixed, stratified sample of 32 held-out prompts per seed.

| Measurement | Token-level | Prompt-fixed |
| --- | ---: | ---: |
| Mean held-out next-token accuracy across three seeds | 100.00% | 99.99% |
| Mean held-out NLL | 0.000147 | 0.000514 |
| Exact greedy completions across three seeds | 96/96 | 96/96 |
| Mean SSD-backed decode speed | 593 tok/s | 963 tok/s |
| Mean time to first token | 3.57 ms | 2.27 ms |
| Mean whole-drive reads per 32 requests | 144.47 MiB | 6.76 MiB |
| Mean expert payload read during decode per 32 requests | 111.34 MiB | 0 MiB |
| Mean expert payload read during prefill per 32 requests | 16.34 MiB | 6.00 MiB |

Benchmarked native MLX 0.32.2 on the M5 Pro in AC High Power mode. Six timing trials per mode used prompt/token/token/prompt order within each seed. Both started every request with an empty 192 KiB expert cache, enough for one expert per layer. Only the backbone and routers were loaded before each trial. Both variants read selected expert slices from SSD through Telos's demand cache. Prompt-fixed routing loaded the two selected experts during prefill and reread neither during decode. Whole-drive IOKit reads include background traffic and exclude checkpoint copies and model load. Decode speed excludes prefill. The tiny models and prompt-aligned synthetic tasks establish the residency mechanism and do not predict performance or quality on natural language.

For each trained model, 96 logits vectors were compared element for element across resident and SSD paths and matched exactly. All 410 watchdog samples had unchanged used swap and cumulative swap-outs. Targeted tests reported 16 passes and one skip.

```bash
telos pretrain routing --routing both --steps 1000 --seed 42 --output /tmp/telos-routing-42
```

[Raw results](results.jsonl) · [Swap guard summary](guard.json)
