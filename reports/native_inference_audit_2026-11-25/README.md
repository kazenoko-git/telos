# Native inference optimization audit

Reviewed on 2026-11-25 for Telos 1.9.0, with native Bonsai 4B 2-bit on the M5 Pro as the reference workload. The inventory covers inference for existing weights, optional quality tradeoffs, and architecture choices for future Telos-trained models. Paper results are external evidence. This audit did not run additional models or change the runtime.

## What is already in the checkout

| Change | Commit | Local evidence |
| --- | --- | --- |
| Packed affine projections and native Qwen3 checkpoint loading | `06498977`, `2f7ce764`, `ebc3f8f6` | Original Bonsai 4B checkpoint runs through Telos and MLX primitives |
| One-token greedy decode lookahead | `99c7cded` | 13.8% improvement in the preceding matched native comparison |
| Indexed KV writes | `f3205be9` | 1.4% median paired decode gain in the quieter rerun, with meaningful variability and unchanged allocator peak |
| Last-position final-layer prefill | `59c37b4e` | 2.5% and 1.7% median paired first-token latency reductions for the two tested prompt lengths |

The latest matched comparison measured 145.2 tok/s for native Telos and 124.3 tok/s for LM Studio with the same checkpoint and prompt. The prefill patch was benchmarked separately against the preceding native implementation. The current native Qwen3 path already uses packed quantized matmul, fused RMSNorm, RoPE, grouped-query SDPA, and a request-local KV cache. Sampling still synchronizes for each token. Qwen3 cached updates accept one token per row. [Matched rerun](../bonsai4b_kv_cache_2026-11-24/RERUN.md) · [Prefill results](../bonsai4b_prefill_2026-11-25/README.md) · [Qwen3 implementation](../../telos/models/qwen3_mlx.py) · [Generation loop](../../telos/inference/generation.py)

## Short-context decode and fresh-prompt latency

| Candidate | Where it could help | Telos assessment |
| --- | --- | --- |
| GPU-resident sampled decode | Sampled local generation | Extend one-step lookahead to consume the sampled ID and RNG state on device. Verify seeded parity and stopping. Measure greedy and sampled modes separately. |
| Fused sampling and logit processing | Sampling with top-k, top-p, penalties or constraints | Profile logits processing at the actual vocabulary size. Options include fused top-k and sampling, and avoiding an independent full sort when the sampling algorithm permits. Preserve tie and RNG semantics. |
| Fused output projection and greedy reduction | Batch-one greedy decode | Measure output-head time first. A packed projection with a local max-and-index reduction could reduce dispatches and intermediate logits traffic while preserving the head's arithmetic and argmax tie-breaks. The weight reads remain necessary. |
| Specialized affine 2-bit Metal matvec | Batch-one decode with existing Bonsai weights | Benchmark real projection shapes and packed layout against MLX. Measure bandwidth, kernel occupancy and time for attention, MLP and vocabulary projections separately. Extend by format only after a win. [BaseRT](https://arxiv.org/abs/2607.00501) · [Apple Silicon portability study](https://arxiv.org/abs/2610.31479) |
| Pointwise fusion and residual plus normalization | Launch-bound layers | Screen SiLU times gate, residual add plus RMSNorm, and RoPE plus cache append separately. Earlier gate and projection-packing screens do not establish a useful default. MLX exposes custom Metal kernels when profiling identifies a worthwhile boundary. [Apple MLX session](https://developer.apple.com/videos/play/wwdc2025/315/?time=752) |
| Fixed-shape graph reuse and command submission | Host-bound decode | Measure graph construction and command gaps before adding complexity. Reuse shapes and scratch buffers when possible. Shape bucketing needs correct masking, and larger token submission windows can delay streaming and cancellation. [BaseRT](https://arxiv.org/abs/2607.00501) |
| M5 TensorOps prefill GEMM and attention | Fresh medium or long prompts | Evaluate a single projection shape and the actual packed format first. Keep a regular MLX path for other chips and compare transfer, conversion and dispatch overhead end to end. BaseRT's M5 study puts the tensor units on compute-heavy prefill. [BaseRT M5](https://arxiv.org/abs/2607.19438) · [Apple Metal TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/?time=21) |
| Prefill tiling, buckets and chunk sizes | Prompt processing or peak memory | Separate an isolated throughput test from a memory-budget test. Smaller chunks can improve responsiveness or cap scratch allocations and can cost single-request throughput. Cached multi-token suffix processing needs absolute query positions and an offset-aware causal mask. [Sarathi-Serve](https://arxiv.org/abs/2403.02310) |
| Exact attention kernel dispatch | Different prefill and decode shapes | Compare MLX SDPA against alternative tiling only where a trace shows a gap. FlashAttention-2 and -4 offer useful work-partitioning ideas. FlashAttention-4's reported kernel targets a different hardware stack. [FlashAttention-2](https://arxiv.org/abs/2307.08691) · [FlashAttention-4](https://openreview.net/pdf/b9ff21f07bc47ea9383be2ebf862284823a5b908.pdf) |
| Compilation and persistent weights | Repeated requests and cold starts | Keep loaded weights and tokenizer available across requests. Measure cold load, first warm request and subsequent warm requests separately. Consider file mapping and kernel warmup only where startup traces justify them. |
| Tokenizer and stream overhead | Small models, constrained output, many stops | Profile encode, incremental detokenization, stop matching and the final text flush. Preserve byte boundaries and exact stop behavior. The existing decode stream already avoids decoding the full output after each token. |

## Exact reuse and speculative decoding

| Candidate | Where it could help | Telos assessment |
| --- | --- | --- |
| Session-local exact prefix KV reuse | Chat turns, repeated documents and stable tool prompts | Highest-value architectural addition for repeated prefixes. Keep one contiguous cache per session first, match token IDs, append the suffix, and retain state only for the matching checkpoint and effective positional configuration. Measure realistic hit rates and memory. [SGLang](https://arxiv.org/abs/2312.07104) |
| Persistent prompt snapshots | Shared system prompts and warm restart | Add a bounded in-memory cache first. Consider serialized state and SSD only if measured load-plus-restore beats recompute. Account for checkpoint identity, prompt divergence and cache lifetimes. [SiliconBench](https://arxiv.org/abs/2609.19169) |
| Chunk reuse with selective recomputation | RAG documents reordered or reused in different prompts | Different from exact prefix hits. A block's KV can depend on earlier context. CacheBlend selectively recomputes for this scenario, so quality needs a separate gate. [CacheBlend](https://arxiv.org/abs/2405.16444) |
| Prompt lookup, n-gram and suffix drafts | Repetitive code edits, extraction and structured output | Lowest-overhead speculative candidate for current weights. The target verifies a short token block and retains only accepted state. Measure acceptance and total verifier cost against ordinary decode. [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding) · [SAM-Decoding](https://github.com/hyx1999/SAM-Decoding) |
| Small draft model and adaptive lookahead | Larger or slower targets | Benchmark the verifier alone at 2, 4 and 8 tokens before adding a drafter. For greediness, require exact IDs. For sampling, correct rejection sampling preserves the target distribution. Draft latency, cache rollback and memory can erase the gain. [Speculative decoding](https://arxiv.org/abs/2211.17192) · [Apple Silicon study](https://arxiv.org/abs/2610.31479) |
| EAGLE-3, DFlash and test-time adaptation | Compatible target-specific speculative checkpoints | Trained feature drafters and parallel block drafters need compatible weights and different integration work. Reported accelerator results need fresh evaluation on local hardware. Test-time adaptation is research for long outputs after basic verification works. [EAGLE-3](https://arxiv.org/abs/2503.01840) · [DFlash](https://arxiv.org/abs/2602.06036) · [Test-Time Speculation](https://arxiv.org/abs/2605.09329) |

## Long context and constrained memory

For Bonsai 4B, a hypothetical two-byte cache payload is 144 KiB per token across all 36 layers, eight KV heads and 128 channels per head for both K and V. That works out to about 1.125 GiB at 8,192 tokens before padding and allocator overhead. This is a shape-based estimate. Measure the actual memory and decode curve before picking a crossover for a new default.

| Candidate | Where it could help | Telos assessment |
| --- | --- | --- |
| Split-KV decode and online softmax reduction | Long-context batch-one decode | Parallelize a single query across context sections and combine the partial reductions. Inspect what installed MLX already dispatches and compare by context length. [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) · [FlashInfer](https://arxiv.org/abs/2501.01005) |
| Cache layout, block size and update fusion | Longer or irregular contexts | Sweep the existing block size and look for cache copies and allocation spikes. Combine K/V append or use a layout suitable for GQA only if attention and allocator traces justify it. |
| 8-bit then 4-bit KV cache with a recent full-precision tail | Long-context capacity and possibly bandwidth | Best compression starting point. Quantize older blocks and consume packed values directly in attention. Dequantizing the entire cache every step can erase the speed gain. Use quality gates and an opt-in policy initially. [KIVI](https://arxiv.org/abs/2402.02750) · [KVQuant](https://arxiv.org/abs/2401.18079) |
| Sub-4-bit or rotated KV representations | Tighter memory budgets | Compare KIVI, KVQuant and TurboQuant only after setting a simple 4-bit baseline. TurboQuant and Fast-TurboQuant introduce additional transforms or reconstruction work that needs native kernel measurements. [TurboQuant](https://arxiv.org/abs/2504.19874) · [Fast-TurboQuant](https://arxiv.org/abs/2606.21448) |
| KV pruning, eviction and query-aware selection | Very long context where cache traffic dominates | StreamingLLM keeps sinks and a recent window, SnapKV selects prompt positions, ThinK prunes key channels, and RocketKV combines eviction with query-aware selection. These can change answers. Test retrieval at multiple positions and exact code or document references. [StreamingLLM](https://arxiv.org/abs/2309.17453) · [SnapKV](https://arxiv.org/abs/2404.14469) · [ThinK](https://arxiv.org/abs/2407.21018) · [RocketKV](https://arxiv.org/abs/2502.14051) |
| Sparse prefill and lower-precision attention math | Long-context first-token latency | A separate approximate path. Compare dynamic sparse attention or reduced precision against exact SDPA using real long-context tasks. [MInference](https://www.proceedings.com/content/079/079017-1663open.pdf) |
| Tiered KV and weight offload | Contexts or models that exceed practical RAM | Capacity option. Benchmark page faults, prefetch and storage traffic under pressure. Unified memory changes the economics of CPU/GPU offload, and SSD reads in the token-critical path can dominate latency. [TierKV](https://arxiv.org/abs/2609.21172) · [LLM in a Flash](https://arxiv.org/abs/2312.11514) |
| Prompt compression or retrieval selection | Very long input with redundant context | Reduces tokens fed to the model and changes its input. Evaluate end-to-end task quality separately from equivalent-kernel optimizations. |

## Serving and future checkpoints

| Candidate | Where it could help | Telos assessment |
| --- | --- | --- |
| Continuous batching and chunked-prefill scheduling | Several simultaneous requests | Benchmark users per second and p95 first-token and inter-token latency across concurrency levels. Keep local single-stream results distinct. [Sarathi-Serve](https://arxiv.org/abs/2403.02310) · [SiliconBench](https://arxiv.org/abs/2609.19169) |
| PagedAttention | Many live sequences with varied lengths | Optional runtime cache layout for serving or memory-pressure workloads. Its allocation and sharing benefits need a workload with the relevant pressure. [PagedAttention](https://arxiv.org/abs/2309.06180) |
| RadixAttention | Many cached branches with shared prefixes | Optional runtime prefix indexing and reuse strategy. Exact single-session reuse can use a contiguous cache. Paged allocation and radix lookup solve distinct problems. [SGLang](https://arxiv.org/abs/2312.07104) |
| Expert gather and grouped GEMM | Native MoE prefill or batched decode | Once native MoE exists, profile direct expert gathers at batch one and grouped GEMM on larger token batches. MARLIN's multi-row weight reuse also matters more when the runtime has multiple rows to process. [BaseRT M5](https://arxiv.org/abs/2607.19438) · [MARLIN](https://arxiv.org/abs/2408.11743) |
| Distributed pipeline or tensor parallelism | Memory capacity or several Macs | Separate deployment option, with communication, partitioning and topology costs. Apple's MLX/RDMA work is evidence for Mac Studio clusters. [Apple MLX and RDMA](https://machinelearning.apple.com/research/mlx-distributed-inference) |
| Post-training or mixed-bit weight quantization | Memory and bandwidth across a model library | AWQ-style calibration and choices such as higher precision for sensitive tensors produce a separate checkpoint. Preserve exact-weight Bonsai comparisons as their own result. [AWQ](https://arxiv.org/abs/2306.00978) |
| GQA, MLA, local attention and hybrid recurrent layers | Future Telos-trained model designs | Model configuration choices with quality and portability implications. Bonsai 4B already uses GQA. MLA or recurrent state needs architecture-aware training and inference. [GQA](https://arxiv.org/abs/2305.13245) · [DeepSeek-V2](https://arxiv.org/abs/2405.04434) |
| Ternary-aware training, distillation, pruning and smaller models | Future checkpoints for constrained hardware | Evaluate task quality, footprint and speed together. A native 1-bit loader for installed weights is separate from training a ternary model. [BitNet b1.58](https://arxiv.org/abs/2402.17764) |
| Multi-token prediction training | Future checkpoints that can draft with their own heads | Make it an optional objective and retain target verification in inference. Model quality, head acceptance and runtime speed all need measurement. [DeepSeek-V3](https://arxiv.org/abs/2412.19437) |

## Test order for Telos

1. **Single-stream speed.** Trace the real Bonsai shapes and compare packed matvec, output projection, attention and host gaps. Extend device-resident sampling and test M5 TensorOps separately for prefill.
2. **Repeated prompts.** Implement the smallest exact session cache around multi-token suffix append and measure shared-prefix TTFT against full recompute.
3. **Long context.** Measure uncompressed attention and memory at 1K, 4K, 8K and 16K where feasible. Try split-KV only where the baseline dispatch leaves a gap. Compare opt-in 8-bit and 4-bit KV with retrieval and code quality tests.
4. **Speculation.** Benchmark target block verification and rollback alone before trying prompt lookup. Measure its win rate across repetitive inputs and ordinary chat before considering a learned drafter.
5. **Serving.** Reassess optional paging, radix indexing and batching after there is a concurrent workload to measure. Keep default single-user inference simple.

Across changes intended to preserve behavior, record greedy IDs and text alongside first-token latency, inter-token latency, sustained decode, peak memory and request latency. Use seeded checks for exact sampled scheduling changes, distribution checks for speculative sampling, and task evaluations for intentionally approximate methods.
