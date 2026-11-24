# Telos 1.9.0 pre-release inference audit

Apple M5 Pro, 24 GiB unified memory. Telos native runs used MLX 0.32.2 or PyTorch 2.13.0 with six CPU threads. The tested inference implementation is in local commit `07886c8`.

## Native trained checkpoints

The 50M and 105.4M checkpoints came from completed TPU AR runs on Python corpus data. GPU inference used the 49.2M BF16 MLX checkpoint from a separate 1,536-update archived run. It used a different corpus prefix, model size and context limit. The CPU and MLX figures therefore describe distinct checkpoints.

All native measurements used the local 8,192-token `configs/shared/tokenizer_mac.json` override. The first 105M invocation without it failed vocabulary validation because the archived checkpoint has no tokenizer reference and the package fallback holds 4,096 tokens. Both local corpus samples contain token IDs above 4,095 and decode coherently with the override. The archived MLX resume state records the 4,096-token fallback hash despite the 8,192-token corpus and model. These artifacts lack a verified tokenizer identity in inference metadata.

### Performance

Each warmed result is the median of five public streaming requests per shape. TTFT measures the first generated non-EOS token. First visible text was effectively the same for these prompts. Decode rate excludes the first token and includes public streaming and sampling overhead. Tail rate measures the last 32 tokens. Peak RSS is for the checkpoint audit process, including loading. MLX peak allocation is a separate allocator figure and should not be added to RSS.

| Checkpoint | Prompt → output | Warm TTFT | Decode tok/s | Last 32 tok/s | Process peak RSS | MLX allocator peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TPU AR 105.4M, CPU | 51 → 144 | 21.7 ms | 197 | 196 | 1,067 MiB | — |
| TPU AR 105.4M, CPU | 219 → 144 | 46.5 ms | 193 | 193 | 1,067 MiB | — |
| TPU AR 49.2M, CPU | 51 → 144 | 9.3 ms | 323 | 326 | 647 MiB | — |
| TPU AR 49.2M, CPU | 219 → 144 | 26.9 ms | 273 | 292 | 647 MiB | — |
| Archived AR 49.2M, MLX BF16 | 51 → 144 | 2.2 ms | 658 | 670 | 325 MiB | 113 MiB |
| Archived AR 49.2M, MLX BF16 | 219 → 37 | 3.1 ms | 623 | 624 | 325 MiB | 162 MiB |

Fresh CLI process to first byte took 1.36 s, 1.33 s, and 0.64 s respectively. Model load alone was 975 ms, 447 ms, and 33 ms. Warmed TTFT excludes checkpoint load. The MLX long-prompt output cap was 37 tokens because the checkpoint context is 256.

On one 72-token parity prompt, CPU continuations matched across uncached, dynamic, and reserved caches. Single-pass speeds for these modes were 31, 190, and 217 tok/s on the 105M checkpoint, and 27, 179, and 273 tok/s on the 50M checkpoint. The trained BF16 MLX model returned different cached and uncached continuations on this prompt.

### Quality

Six Python function-completion cases were tested with greedy decoding and once with a fixed seed. A continuation had to parse as a single function and pass sandboxed tests. The 105M model passed one of six with seeded sampling and zero with greedy decoding. Both 49M-class models passed zero under both settings. The seeded 105M `is_even` completion passed. The greedy completion used `return n < 0 or n % 2 == 0` and failed on negative odd input. All three failed the one-word France-capital instruction, and several outputs repeated text.

These probes assess small pretrained base models on a small set of prompts. They are not a benchmark of a native instruction-tuned checkpoint. Raw continuations and outcomes are in [105M CPU](native-100m-cpu.json), [50M CPU](native-50m-cpu.json), and [49M MLX](native-49m-mlx.json).

The built-in Telos functional evaluator passed **0/4 HumanEval cases per native checkpoint**. This is a smoke slice, with a 192-token output cap and two-second execution timeout. The separate six-prompt scores above use the public inference API and smaller hand-written checks. [105M](telos-eval-native-100m.json), [50M](telos-eval-native-50m.json), [49M](telos-eval-native-49m-mlx.json).

The MLX parity follow-up used identical forced prefixes at every step. On two of three prompts, the first BF16 greedy difference occurred when a 0.03125 full-prefix margin became an exact tie with caching. The largest BF16 logit delta across all tested positions was 0.125. Recasting the same saved weights to FP32 preserved the greedy choice at all 216 positions, with maximum logit delta 6.56 × 10⁻⁶. This supports precision sensitivity on the tested artifact. [Raw MLX checks](mlx-cache-parity.json).

## Models in LM Studio

Quality uses the Telos HumanEval evaluator through its OpenAI-compatible adapter to the local LM Studio worker. The audit caps the adapter at 384 new tokens per task because the evaluator otherwise applies a 2,048-token minimum to external models. This is a four-case smoke test. API replies that exhaust their budget on reasoning can lose the visible answer.

Timing comes from LM Studio’s local streaming telemetry. Telos’s current OpenAI-compatible adapter buffers responses and does not expose first-token timing. The table uses three 160-token-budget performance prompts, with one short prompt and two longer prompts. Reported short and long time to first token are engine timings. The decode result is the median of all three engine decode rates, including reasoning tokens when applicable. Prompt token counts include chat templates and vary by model. Fresh model load is measured separately in the raw data.

Memory is the sampled increase in aggregate LM Studio process-tree RSS above its unloaded baseline. RSS includes service and worker overhead and can omit allocations outside the sampled processes. Telos native peak RSS and MLX allocator figures above use different measurement scopes.

| LM Studio model | Context configured | Short TTFT | Longer TTFT | Decode tok/s | Peak RSS increase | Telos HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 0.8B | 2,048 | 30 ms | 90 ms | 365.7 | 1.84 GiB | 2/4 |
| Ternary Bonsai 4B | 2,048 | 104 ms | 326 ms | 114.3 | 1.69 GiB | 4/4 |
| Ternary Bonsai 2 2B | 2,048 | 124 ms | 320 ms | 209.6 | 1.70 GiB | 2/4 |
| Bonsai 8B | 2,048 | 219 ms | 520 ms | 98.6 | 1.92 GiB | 3/4 |
| Phi 4 Mini Reasoning | 2,048 | 101 ms | 252 ms | 79.8 | 1.18 GiB | 0/4 |
| Qwen3.5 2B OptiQ | 2,048 | 37 ms | 118 ms | 139.3 | 2.59 GiB | 2/4 |
| LFM2.5 8B A1B | 2,048 | 94 ms | 274 ms | 127.0 | 1.82 GiB | 0/4 |
| Ministral 3 8B Instruct 2512 | 2,048 | 107 ms | 1,272 ms | 27.2 | 3.57 GiB | 3/4 |
| Gemma 4 E4B | 2,048 | 74 ms | 797 ms | 14.2 | 4.12 GiB | 3/4 |
| Bonsai 27B | 1,024 | 275 ms | 3,972 ms | 12.2 | 4.56 GiB | 4/4 |
| Ternary Bonsai 2 27B | 1,024 | 490 ms | 3,528 ms | 14.3 | 5.41 GiB | 0/4 |

Raw data includes [runtime trials](lmstudio.jsonl), [Telos evaluator results](lmstudio-telos-eval.jsonl), [27B runs](lmstudio-27b.jsonl), and [embedding and load retries](lmstudio-retries.jsonl). The 27B models ran individually at the reduced context after checking host memory.

Nomic Embed Text v1.5 returned 768-dimensional embeddings in a 137 ms localhost batch. Duplicate inputs had identical vectors and the distinct input differed. LFM3 Omni 2B failed to load because the installed LM Studio runtime lacks `mlx_vlm.speculative.drafters.lfm3_omni`. The local LFM3 1.7B directory has tokenizer and configuration files but is missing the `model.safetensors` named in its weight index. The Bonsai image directory contains a Flux2 autoencoder artifact and was outside the text-generation workload.

The native `telos generate` entry point rejected a Qwen3.5 0.8B MLX folder as expected. The Telos adapter evaluated the model through LM Studio successfully. External evaluator token-length and repetition fields were excluded because no matching per-model tokenizer was attached to that adapter in this run. HumanEval pass counts use sandboxed outputs and do not depend on those fields.

## Release work to prioritize

1. Make every native inference export carry its tokenizer and identity metadata. Keep the explicit override for legacy archives and improve the error when a bundled tokenizer does not match. Avoid guessing based only on vocabulary size.
2. Document the BF16 cache parity boundary and cover close-logit cases on trained weights. The FP32 follow-up is a useful reference for diagnosing numerical differences.
3. Verify quality on a representative native instruction-tuned checkpoint before making instruction or chat quality claims. The checkpoints tested here are pretrained base models. The inference runtime and model quality require separate acceptance criteria.
4. Add a checkpoint inference benchmark surface with first-token timing, decode tokens per second, and documented memory counters. The existing `telos bench` measures training, and the Telos external adapter currently returns buffered generations.
5. Profile cold checkpoint loading if one-shot CLI latency matters. The fresh process took roughly 0.64–1.36 s to emit its first byte on these artifacts, while model-resident requests were much faster. Physical CUDA inference acceptance is still open.

Run `native.py` for native public-API timing and quality prompts, `mlx_parity.py` for forced-prefix parity, and `telos_eval.py` for the Telos HumanEval slice. `lmstudio.py` runs the LM Studio sweeps against a locally started server and unloads its own model after each entry.
