# Native Bonsai 4B decode profile and rerun

Tested the original Ternary Bonsai 4B 2-bit Safetensors checkpoint on the M5 Pro with native Telos and MLX 0.32.2. The weights and prompts match the [previous native comparison](../bonsai4b_native_inference_2026-11-24/README.md). Neither MLX-LM nor either 27B model was loaded.

## Profile and screening

A baseline `cProfile` run recorded 1.624 s in the greedy sampling call for 200 tokens including warmup. That call synchronizes the lazy MLX graph with `argmax(...).item()`, so the time includes Metal execution and synchronization. Python graph construction in the cached forward accounted for 0.158 s. The profile does not isolate individual GPU kernels.

- Packing compatible QKV and gated MLP projections into wider quantized operations moved median throughput from 120.6 to 118.5 tok/s in a short A/B/B/A screen. It raised MLX peak allocation from 1,607 to 1,776 MiB and was dropped.
- Keeping the greedy ID on device and submitting one cached decode step before waiting for the current ID raised median throughput from 117.5 to 138.1 tok/s in a separate A/B/B/A screen. All 192 greedy IDs matched. Allocator peak stayed at 1,607 MiB.

The shipped path pipelines greedy cached MLX generation only. It can submit one extra forward when EOS or a text stop ends the stream, and the loop never schedules past the requested token budget. Sampling and uncached execution use their original paths.

## Matched LM Studio rerun

Four trials per runtime, alternating in A/B/B/A order twice, with fresh loaded weights and an unrelated warmup. Both used the same 212 prompt IDs, a 1,024-token context, a 192-token budget, greedy decoding, and no speculative decoding. The benchmark timed delivery from token 1 through token 191 on both runtimes.

| Median | Native Telos before | Native Telos with lookahead | LM Studio rerun |
| --- | ---: | ---: | ---: |
| Time to first delivered token | 111 ms | 113 ms | 206 ms |
| Decode over shared token window | 122.6 tok/s | **139.5 tok/s** | 122.7 tok/s |
| Total request time | 1.669 s | **1.479 s** | 1.755 s |

Native Telos gained 13.8% over the previous native run and was 13.7% faster than the LM Studio rerun. The prior LM Studio run reached 127.6 tok/s, 4% higher than its rerun. The native improvement also held against that prior rate. This result covers one prompt and four trials per runtime.

The optimized native path reproduced all **192 token IDs** and exact text from the prior native run on every trial. LM Studio returned the same **813 shared characters** after its leading-space trim and emitted 191 tokens at its limit. Telos streamed one further token. Native allocator peak remained 1,607 MiB. LM Studio's sampled process-tree RSS increased by a median of 1,853 MiB above its unloaded baseline. These memory counters have different scopes.

[Screening trials](screening.jsonl) · [Matched rerun](trials.jsonl) · [Benchmark harness](../bonsai4b_native_inference_2026-11-24/benchmark.py)
