# Matched Ternary Bonsai 4B inference comparison

On the Apple M5 Pro, Telos's in-process MLX-LM adapter delivered **144.5 tok/s** and LM Studio delivered **127.7 tok/s** over the same 191-token measurement window. The direct path was **13.1% faster** for this prompt. LM Studio used its installed backend and localhost SDK stream. The native Telos transformer does not handle this Qwen3 checkpoint.

| Median of four trials | Telos via MLX-LM | LM Studio |
| --- | ---: | ---: |
| Time to first delivered token | 120 ms | 198 ms |
| Time to first token reported by engine | 120 ms | 165 ms |
| Delivered decode rate over shared 191 tokens | 144.5 tok/s | 127.7 tok/s |
| Engine-reported decode rate | 145.3 tok/s | 125.7 tok/s |
| RSS high-water mark | 1,675 MiB, one process | 2,478 MiB, sampled process tree |

Delivered decode rate is 190 divided by the time from the first to the 191st token. The direct range was 143.3–149.9 tok/s and the LM Studio range was 122.2–131.5 tok/s. The first direct trial took 383 ms to deliver the first token. Its other trials took 117–121 ms. Direct MLX allocator peak was 1,623 MiB. LM Studio's sampled RSS rose by 1,842 MiB above its unloaded service baseline. The memory figures have different measurement scopes.

## Inputs and protocol

- Both engines used the local 2-bit file indexed by LM Studio as `prism-ml/Ternary-Bonsai-4B-mlx-2bit`. The SHA-256 of `model.safetensors` was `1f956eefc2fa23fc21048ac2d1a8e76d78558666d81bf6b1fa89d83ba59698ac`.
- Both encoded the literal prompt to the same **212 token IDs**. Both used a **1,024-token context**, a **192-token output limit**, greedy selection, and no extra stop strings or speculative decoding. The SDK confirmed neutral repetition penalty and filtering. Transformers raised a tokenizer-regex diagnostic on load. Exact prompt IDs were checked despite that diagnostic.
- Each trial used a fresh model load and one short warmup with a different first token. Engines ran in the order Telos, LM Studio, LM Studio, Telos, repeated twice. Only one model was loaded for each trial. Direct MLX and MLX-LM were both 0.32.2. The LM Studio Python SDK was 1.5.0. The SDK supplies raw completion streaming and exposes the loaded model tokenizer. [Raw completion SDK](https://lmstudio.ai/docs/python/llm-prediction/completion) · [Tokenizer SDK](https://lmstudio.ai/docs/python/tokenization)
- Each runtime produced identical text in all four of its trials. Both returned the same **813 characters** over the shared output. LM Studio emitted 191 tokens and signaled its token limit. MLX-LM emitted 192 and added one token. This establishes output agreement for the tested prompt.

[Benchmark harness](benchmark.py) · [Eight raw trials](trials.jsonl)
