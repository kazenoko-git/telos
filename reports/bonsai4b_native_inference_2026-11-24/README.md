# Native Telos versus LM Studio on Ternary Bonsai 4B

The direct benchmark loaded Bonsai with `TextGenerator.from_checkpoint` and decoded through Telos's Qwen3 implementation and MLX primitives. The harness asserts that MLX-LM was never imported. LM Studio ran the other arm through its own localhost SDK stream.

| Median of four trials | Native Telos on MLX | LM Studio |
| --- | ---: | ---: |
| Time to first delivered token | 111 ms | 198 ms |
| Decode over the same 191-token window | 122.6 tok/s | 127.6 tok/s |
| Request wall time including prefill | 1.669 s | 1.692 s |
| RSS high-water mark | 1,296 MiB, one process | 2,459 MiB, sampled process tree |

Native Telos delivered the first token 44% sooner and its median decode rate was 4% lower. The ranges were 113.2–126.8 tok/s for Telos and 123.9–133.1 tok/s for LM Studio. Native MLX allocator peak was 1,607 MiB. LM Studio's sampled process-tree RSS increased 1,841 MiB above its unloaded baseline. These memory counters have different measurement scopes.

## Output parity

- Both tokenizers produced the same 212 prompt IDs. Both runtimes used a 1,024-token context, greedy decoding, a 192-token output budget, and no extra stop strings or speculation. Native output was deterministic across all four trials.
- The **813 shared characters matched** after accounting for the initial space trimmed by LM Studio. LM Studio emitted 191 tokens at the configured limit. Telos emitted 192 and appended one token. This is output parity on the tested prompt, with the token-limit and whitespace behavior noted.
- An independent FP32 fixture with mixed query and KV head counts, an explicit head dimension, an untied output head, and YaRN matched Hugging Face Transformers logits to a maximum absolute difference of 6.71 × 10⁻⁷. Unit tests compare packed projections with dequantized reference matrices and cover cache replay and sharded loading. The full checkpoint's GPU logits were not exported from LM Studio. [Qwen3 architecture](https://huggingface.co/docs/transformers/model_doc/qwen3) · [YaRN](https://arxiv.org/abs/2309.00071)

## Reproduction

This uses the exact 2-bit Safetensors file from the [previous matched test](../bonsai4b_inference_2026-11-24/README.md), SHA-256 `1f956eefc2fa23fc21048ac2d1a8e76d78558666d81bf6b1fa89d83ba59698ac`. Both backends used the original packed weights. On an M5 Pro with MLX 0.32.2, each trial loaded fresh weights and ran one unrelated warmup. The evaluation order was Telos, LM Studio, LM Studio, Telos, repeated twice, with only one backend holding the model at a time. The LM Studio SDK version was 1.5.0. Decode rate is 190 divided by the time between delivery of the first and 191st generated tokens.

[Benchmark harness](benchmark.py) · [Raw trials](trials.jsonl)
