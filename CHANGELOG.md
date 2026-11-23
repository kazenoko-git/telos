# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Optional incremental KV-cache support (`hidden_states_with_cache`) on PyTorch `TelosTransformer` and MLX `MLXTelosTransformer`, plus length-bucketed shared-prefill `collect_rollouts_cached` in `telos/posttrain/grpo.py`.
- Sparse masked-only `mdlm_loss_from_model_pytorch` helper in `telos/diffusion/mdlm.py`.
- Unit and parity tests in `tests/test_v189_optimizations.py` and M5 Pro Metal screening script in `scratch/bench_v189_m5pro.py`.

### Fixed
- Use KV-cached shared-prefill rollouts in PyTorch and MLX GRPO post-training (`7.79×` faster rollout generation on PyTorch CPU with exact parity).
- Project only valid (`labels != -100`) hidden states in PyTorch SFT post-training (`1.185×` E2E on CPU, `0.0` loss/grad diff).
- Project only masked positions in PyTorch MDLM training (`1.103×` E2E on CPU, `0.0` loss/grad diff) and in `MDLMSampler` / `MLXMDLMSampler` (`1.055×` on CPU, exact token parity).

## [1.8.8] - 2026-11-22

### Added
- `disable_lora_adapters` context managers in `telos/posttrain/adapters.py` and `telos/posttrain/mlx_adapters.py` to evaluate frozen base-model log-probs without duplicating base weights.
- Unit and gradient-parity tests in `tests/test_v188_posttrain_optimizations.py`.

### Fixed
- Project only valid (`labels != -100`) hidden states through `output_projection` and use fused `-F.cross_entropy(..., reduction="none")` instead of materializing full FP32 `F.log_softmax` in `_torch_token_logps`.
- Reuse the frozen base model via `disable_lora_adapters` instead of `copy.deepcopy(model)` for LoRA/QLoRA DPO and GRPO reference log-probs on PyTorch and MLX.
- Skip zero-variance GRPO `policy_updates` when `beta == 0` and all group advantages are zero.
- Use `int32` indices instead of `int64` in `NF4Linear.dequantize()` and enable CUDA TF32/cuDNN benchmark mode in post-training.

### Validation
- Full local suite passed (`486 passed, 7 skipped`), including MLX QLoRA GRPO rollout/resume parity; CI and wheel smoke tests passed on Python 3.10–3.13.
- Apple M5 Pro measurements recorded exact output parity for the zero-copy MLX QLoRA reference, with 49.2% lower active model memory and 36.1% lower peak GPU memory. Physical CUDA throughput remains unverified; see [v1.8.8 validation](docs/releases/v1.8.8-validation.md).


## [1.8.7] - 2026-11-22

### Added
- `hidden_states()` and `reliability_scores()` helpers on `TelosTransformer` and `MLXTelosTransformer` to skip full-vocabulary LM head projections when only hidden states or reliability scores are needed.
- Opt-in Fast-dLLM `confidence_threshold` parallel unmasking and Prophet `early_commit_margin` early-commit controls on `MDLMSampler` and `MLXMDLMSampler`.
- Unit and gradient-parity tests in `tests/test_v187_diffusion_decoding_optimizations.py`.

### Fixed
- Project only masked tokens through `output_projection` in `corosred_unified_step_pytorch` (3.57× faster LM head+CE on CPU with 0.0 exact diff) and pass `next_token_only=True` on causal passes.
- Skip full-vocabulary projection in `RoutingMaskCache.refresh` via `reliability_scores()`.
- Eliminate wasted vocab projection and double `mx.argsort` in `corosred_unified_loss_fn_mlx`, remove per-microbatch `torch.clear_autocast_cache()`, and replace `torch.argsort` with `torch.topk` in COROSred Phase B/C.
- Use `next_token_only=True` in `DualMetricMonitor.evaluate`.
- Vectorize repetition penalty and slice masked positions before softmax/topk in `MDLMSampler`; compute top-2 margin on device in `MLXMDLMSampler`.
- Vectorize interleaved COROSred probe without redundant `F.log_softmax`/`.clone()` (4.01× faster at V=8192), bound stop-string decoding to a tail window, decode lexical tokens once, and truncate AR KV-cache to the earliest edited position on rebuild.

### Validation
- PyTorch CPU diffusion and decoding suite passed (`12 passed, 3 skipped`). Metal/CUDA/TPU throughput remains pending physical runs; see [v1.8.7 validation](docs/releases/v1.8.7-validation.md).


## [1.8.6] - 2026-11-22

### Added
- Opt-in token-chunked checkpointed cross-entropy (`token_chunked_cross_entropy` and `chunk_axis="token"`) in `telos/models/vocab_loss.py` and `UnifiedPyTorchTrainer`.
- Opt-in `CautiousAdamW` optimizer (`training.cautious_optimizer: true`, ICLR 2026 sign-aligned update masking).
- Unit tests (`tests/test_v186_pretraining_optimizations.py`) and M5 Pro Metal ABBA screening harness (`scratch/exp_v186_mlx_candidates.py`).

### Fixed
- Hoist XLA gradient checkpoint import out of the per-layer loop in `TelosTransformer.forward`.
- Replace double `torch.argsort` with a single sort in `compute_vectorized_roc_auc`.
- Hoist microbatch cadence invariants out of the accumulation loop in `UnifiedPyTorchTrainer`.
- Hoist BF16 scale cast in `clip_grad_norm_mlx`, skip initial zero-add on first microbatch, and precompute attention scale in `MLXBlock`.
- Refactor MLX Metal `cross_entropy` backward to row-per-threadgroup hoisted loads and hoist per-row scalars into threadgroup memory in M5 MPP fused projection CE backward.

### Validation
- PyTorch CPU unit/parity suite passed (`27 passed`). Token-chunked CE measured **1.31×** forward speedup and single-sort ROC-AUC measured **2.18×** speedup on CPU. Metal/CUDA/TPU hardware throughput remains pending physical runs; see [v1.8.6 validation](docs/releases/v1.8.6-validation.md).


## [1.8.5] - 2026-11-20

### Added
- Opt-in persistent XLA compilation cache helper (`configure_xla_persistent_cache` via `XLA_PERSISTENT_CACHE_PATH` or `training.xla_cache_dir`).
- Native `torch_xla.utils.checkpoint.checkpoint` dispatch on XLA devices.

### Fixed
- Fuse COROSred cadence `xm.all_reduce` before `xm.mark_step()` on TPU so metric reduction stays inside the main step HLO graph instead of launching a second graph execution.
- Skip per-step `xm.set_rng_state` reseeding on deterministic AR runs (`dropout == 0.0`).

### Validation
- New XLA cache, graph ordering, and checkpoint dispatch are checked with mocks. Physical TPU compilation counts and throughput for these changes remain unverified. Dynamic `XLAAdamW` and per-microbatch graph splitting shipped in v1.8.2; its archived physical results remain available. See [v1.8.5 validation](docs/releases/v1.8.5-validation.md).


## [1.8.4] - 2026-11-20

### Fixed
- Replace `sigmoid(x) >= 0.5` with `x >= 0.0` in the COROSred reliability threshold check.
- Hoist RoPE `cos`/`sin` dtype casts and unsqueezes once per `TelosTransformer.forward`, and remove the dead `is_xla` check in the forward path.
- Use native `F.rms_norm` in PyTorch `RMSNorm` (with FP32-variance fallback), single-shot `F.cross_entropy(..., reduction="mean")` in `ar_loss_fn_pytorch`, and broadcast `expand`+`reshape` instead of `repeat_interleave` for GQA.
- Replace full `torch.argsort` with `torch.topk(..., largest=False)` in `RoutingMaskCache.refresh`.
- Stage `CUDAPrefetcher` pinned host buffers in `int32` when token dtypes fit, enable cuDNN TF32 and `cudnn.benchmark`, and pass `gradient_as_bucket_view=True` to `DistributedDataParallel`.

### Validation
- Local PyTorch structural regressions passed as part of the full suite: **471 passed, 7 skipped**. Physical CUDA throughput remains unverified until hardware runs.

- Precision handling, disabled clipping, and unused diagnostic fixes shipped in v1.8.2. See [v1.8.4 validation](docs/releases/v1.8.4-validation.md).


## [1.8.3] - 2026-11-20

### Fixed
- Reduce MLX COROSred `reliability_metrics` peak allocation by using `next_token_only=True`, Metal `cross_entropy`, and materializing/releasing causal logits before the infill pass.


### Validation
- Fresh-process M5 Pro ABBA validation of `mlx_reliability.py` (`d_model=512`, 14 layers, `vocab=8192`, `T=256`, `batch=8`) measured **-12.53% peak MLX memory** (`771.06 MB` to `674.47 MB`) with **0.0 exact parity** across all four metrics; no speed advantage is claimed.

- Dataprep, MLX tuning, optimizer overlap and configurable probe cadence shipped in v1.8.1/v1.8.2 and remain available. See [v1.8.3 validation](docs/releases/v1.8.3-validation.md).


## [1.8.2] - 2026-11-19

### Fixed
- Honor disabled gradient clipping and native CUDA BF16 support consistently.
- Align MLX autotuning with training compilation, evaluation and asynchronous timing.
- Skip unused COROSred diagnostic reductions while preserving reliability loss normalization.

### Added
- Optional accumulated MLX optimizer overlap; remains disabled by default after no measured speed benefit.
- Optional dynamic XLA AdamW controls and physical TPU compilation, numerical, gradient and checkpoint validation scripts. Native AdamW remains the default.
- Configurable MLX COROSred reliability probe cadence; changing feedback cadence can alter training trajectories.


### Validation
- Physical short AR trainer runs passed on a single TPU v5e-1 runtime (BF16) and Kaggle 2× T4 DDP (BF16 request falling back to FP16 with GradScaler): 49.17M parameters, T=256, microbatch 8, accumulation 4, and exact checkpoint/resume reported by notebook comparisons. CUDA replicas also matched model, optimizer and scaler state; no overflow updates were skipped.
- Small TPU optimizer checks found compilation stability for both native and candidate AdamW. No sustained throughput, DDP scaling, long-run stability, or non-AR hardware acceptance is claimed. See [v1.8.2 validation](docs/releases/v1.8.2-validation.md) for archived evidence and limits.


## [1.8.1] - 2026-11-18

### Fixed
- Stream large plain-text/code files in bounded chunks, cap tokenizer batches by character volume, and write each encoding directly instead of building a flattened Python token list.
- Preserve short trailing chunks and apply license cleanup only at document starts.

### Added
- Dataprep `--chunk-chars` and `--max-batch-chars` memory controls; default to eight chunks per batch in both corpus CLIs. See `docs/dataprep.md` for boundary behavior and record-reader limits.

### Validation
- Full suite on the v1.8.1 release checkout: 445 passed, 7 skipped. Wheel/sdist checks and packaged dataprep smoke passed. Real-tokenizer local runs completed for 2, 32 and 128 MiB files; Kaggle remains untested. See [bounded dataprep](docs/dataprep.md).

## [1.8.0] - 2026-11-18

### Added
- Online GRPO reinforcement learning for causal checkpoints on single-device MLX, CUDA and CPU. Supports sampled completion groups, exact-match and custom Python rewards, clipped policy updates, optional frozen-reference KL, full/LoRA/QLoRA training and complete checkpoint resume.

### Validation
- CPU and MLX learned a synthetic verified one-token task. Full training-state replay passed on CPU full training, MLX BF16 full training and MLX QLoRA. Physical CUDA and real-dataset learning remain unverified. See [v1.8.0 validation](docs/releases/v1.8.0-validation.md).

## [1.7.8] - 2026-11-18

### Added
- Opt-in MLX Metal projection and cross entropy for tied-head AR pretraining. Enable with `--fused-vocab-loss --fused-vocab-chunk-axis metal`. M5 tensor operations accelerate aligned FP32 and BF16 heads, and a SIMD kernel covers other shapes and Metal devices.

### Validation
- At 2,048 tokens and a 65,024-token vocabulary in BF16 on M5 Pro, optimized fusion used 270 MiB and 54.74 ms per head call. The full head used 703 MiB and 24.92 ms. The original fused kernel used 294 MiB and 202.57 ms. See [v1.7.8 validation](docs/releases/v1.7.8-validation.md).

## [1.7.7] - 2026-11-18

### Changed
- Enable compiled MLX accumulation by default; disable with `--no-compiled-accumulation` or `training.compiled_accumulation: false`.
- Compile accumulated-gradient scaling and clipping with the MLX optimizer update. Preserve explicit clipping/accumulation controls, first-step moment initialization and master-weight behavior.

### Added
- Installed-runtime attention graph audit and isolated optimizer-boundary probe.

### Validation
- Compiled accumulation with async measured +43.8% pooled throughput on one planner-selected 1×64 workload, with identical saved numerical arrays; clocks varied and the option remains separately configurable.
- The optimizer boundary probe used 18.9% less update time. Short full-training replays preserved exact numerical arrays; clock differences prevent a stable end-to-end speed claim. 401 tests passed, 7 skipped; see `reports/mlx_update_paths_2026-11-18.md`.

- Two 1,536-update complementary switchbacks preserved all 347 saved numerical arrays exactly and flat active memory. Compiled won 43/44 pairs; strict clock/thermal matching yielded only four accepted pairs, below planned coverage. The release decision accepts this limitation; see `docs/releases/v1.7.7-validation.md`.

## [1.7.6] - 2026-11-17

### Added
- Bounded asynchronous MLX gradient accumulation, enabled by default. Disable with `--no-async-accumulation` or `training.async_accumulation: false`. Eager memory policy and stage profiling retain synchronous accumulation.
- Warmed GPU timeline capture, CPU/command-buffer correlation, hardware telemetry, and interleaved/sustained MLX comparison tools.

### Fixed
- Release MLX checkpoint arrays without waiting for cyclic garbage collection; preserve the state format and verify exact nested-state round-trip.

### Changed
- Make the MLX profiling cache configurable with `--cache-mib`, defaulting to 256 MiB after repeated short-run validation.

### Validation
- Four 768-update runs preserved model, optimizer, master-weight and MLX-RNG array state exactly with bounded memory, but did not establish a stable sustained speedup. This separate-process measurement is retained alongside the clock/thermal follow-up; see `reports/mlx_async_sustained_2026-11-17.md`.

- Two complementary 1,536-update randomized switchbacks favored async by 4.99% overall, preserving exact saved array states. Strict clock/thermal matching favored async but missed planned coverage (10 pairs versus 16 required, with fewer than eight in each trial); clocks were not locked. Enable async by default following the release decision; the result does not establish always-on thermal behavior or gains across all workloads. See `reports/mlx_clock_controlled_2026-11-17.md`.
- Full suite after the checkpoint retention fix: 399 passed, 7 skipped after the default change.

## [1.7.5] - 2026-11-16

### Fixed
- Correct experimental MLX vocabulary-tiled CE target rounding and confident-target gradients in BF16; preserve the reference AR transformer context.
- Report actual memory-planned microbatch/accumulation in the MLX profiler and normalize sampled loss using the executed accumulation count.

### Added
- Default-off MLX token-chunked projection/CE with recomputed logits, first-order gradient checks and exact per-path checkpoint replay. Reject its use on other backends.
- Fresh-process projection diagnostics and a numerical/graph-checked, shape-aware native/fallback attention-backward probe.

### Validation
- Correctness checks and measured limits are recorded in `docs/releases/v1.7.5-validation.md` and `reports/mlx_fusion_2026-11-16/REPORT.md`. Chunking has not established a memory or speed win; attention dispatch and all experimental defaults remain unchanged.
- Four completed 192-update QKV/buffer runs matched final model, master, optimizer and MLX RNG tensors bit for bit. 16/16 lowered allocation, but incomplete reverse-order repetitions do not establish a new speedup.
- Complete eight same-runtime full-training attention ABBA jobs: native backward was faster at context 2048, while fallback was faster at 256. Native used more allocation and changed final weights; close short-run held-out CE is not quality-equivalence evidence. See `reports/mlx_fusion_2026-11-16/FULL_ATTENTION_VALIDATION.md`.

## [1.7.4] - 2026-11-16

### Added
- Plan MLX pretraining microbatches within 60% of system RAM, including at least 1 GiB of headroom. Preserve effective batch size through variable gradient accumulation; use checkpointing when necessary and permitted.
- Select the largest fitting measured microbatch under the same allocation budget, including warmup peaks. Save estimated and measured memory-plan metadata.
- Merge the MLX pretraining profiling tools and matched runtime investigation, retaining their workload-specific evidence and limitations.

### Fixed
- Preserve explicit activation-checkpointing policies and measured memory-plan identity across checkpoint loading.

### Validation
- See `docs/releases/v1.7.4-validation.md` for combined-code checks. Memory estimates and MLX allocation limits are advisory; this release makes no new throughput or exact peak-memory guarantee.

## [1.7.3] - 2026-11-16

### Fixed
- Use smooth, numerically stable MLX DPO softplus so an initially identical policy/reference pair receives the correct nonzero gradient at zero preference margin.

### Added
- Stage MLX post-training batches directly as int32 arrays, avoiding per-example Torch tensors and int64 conversion.
- Reuse bounded frozen-reference response scores across reshuffled MLX DPO batches; preserve padded-width identity, full-batch miss evaluation and cache rebuilding after resume.
- Add loss/gradient, cache-capacity, masking, full/LoRA/QLoRA, optimizer/master/RNG and resume regression coverage.

### Validation
- Paired M5 Pro synthetic 47M-model runs measured 5.55% faster SFT and 25.46% faster DPO end to end, with unchanged peak MLX allocation. Both DPO versions include the loss fix; final model, master, optimizer, reference and MLX RNG tensors matched bit for bit.
- These short repeated-example measurements do not establish sustained throughput, convergence, real-dataset gains or gains on other hardware. See `docs/releases/v1.7.3-validation.md` for checks and reproduction details.

## [1.7.2] - 2026-11-16

### Fixed
- Exclude emulated BF16 from CUDA automatic precision selection; T4 uses scaled FP16. Inspect all GPUs in pretraining profiles and retain explicit BF16 opt-in.
- Normalize verified single-chip v5e TPU bounds before topology detection and the post-training launcher. Refuse conflicting metadata; preserve unknown and multi-chip topologies.
- Recompute frozen NF4 projections during backward instead of retaining dense dequantized matrices for input gradients.
- Seed Python/NumPy post-training RNGs before resume restoration, making full state reproducible across fresh control processes.

### Validation
- Replace the external zero-LR benchmark loop with a packaged runner using production updates, scheduler/weight-change assertions, synchronized aggregate timing and matched compile off/on/on/off runs.
- Add a fail-closed full-checkpoint comparator, real process termination/restart tests for all pretraining objectives and full/LoRA/QLoRA SFT/DPO, frozen adapter/buffer checks and an NF4 retention/allocator probe.
- Preserve subprocess logs, commands, source/fixture hashes and environment/import provenance. Correct the historical hardware report's unsupported performance, resume, memory and distributed claims in `docs/releases/v1.7.2-validation.md`.
- Physical v1.7.2 CUDA/TPU reruns remain pending; short historical single-device runs do not establish multi-rank correctness or representative training performance.

## [1.7.1] - 2026-11-15

### Fixed
- Resolve sandbox entry-point symlinks and temporary scratch paths before launching. Permit macOS dyld to read the root directory itself without granting access to its descendants; keep network, outside-scratch writes and Python process forking denied.
- Default MLX FP32 matmuls to full precision before backend initialization instead of implicit TF32 on M5. Preserve an explicit `MLX_ENABLE_TF32=1` opt-in; BF16/FP16 GEMMs keep their acceleration.
- Accumulate BF16/FP16 embedding lookup gradients in FP32 before rounding once, avoiding repeated-token low-precision atomic accumulation errors.

### Tests
- Retain the original FP32/BF16 checkpoint-gradient assertions and tolerances; add matched eager/compiled checkpoint coverage, exact repeated-token gradient checks and a Float64 GEMM reference.
- Add real sandbox launch, scratch-alias, outside-file, network, environment and process-fork regressions.

## [1.7.0] - 2026-11-15

### Added
- Native LoRA and QLoRA SFT/DPO on PyTorch and MLX, with adapter-only artifacts, merged native exports, base-checkpoint hashes and complete resume state. PyTorch uses portable packed NF4; MLX uses grouped affine int4. Double quantization and paged optimizers are not included.
- CUDA post-training mixed precision, FP16 scaler resume, fused AdamW, optional compilation and torchrun DDP with per-rank resume artifacts.
- Opt-in PyTorch packed QKV/SwiGLU projections and checkpointed tiled AR vocabulary loss; canonical checkpoint keys remain compatible.
- Integrate the existing opt-in MLX QKV/RoPE kernel and its gradient/update checks.
- Fresh-process synthetic post-training benchmark runner with source hashes.

### Changed
- Pretokenize post-training records, collate batches on CPU, concatenate DPO pairs, and bound frozen-reference caching.
- Compile MLX post-training updates and use class-index cross entropy instead of materializing FP32 log-softmax intermediates.
- Use fixed XLA post-training shapes, read loss only at logging cadence, and keep distributed SFT normalization token-weighted.
- Use event-controlled CUDA prefetch with reusable pinned staging, native CUDA GQA, explicit activation-checkpointing policies, and configurable XLA cleanup cadence.

### Fixed
- Fence benchmark updates on CUDA, MLX and XLA; exclude warmup and use slowest-rank timing for distributed reporting/stopping. Label input-token units and insufficient-sample reports.
- Preserve native checkpoint attention-head configuration, maximum context and tied-embedding settings during loading.
- Update isolated MLX accumulation experiments for the production step-hook signature.
- Allow same-directory MLX post-training resume; select checkpoint steps numerically and exclude adapter/resume sidecars.

### Validation
- CPU and Metal validation only; CUDA/TPU hardware, distributed execution and sustained convergence remain unverified. See the release validation report for exact results and reproduced baseline failures.

## [1.6.2] - 2026-11-15

### Added
- Ship the frozen 512-case synthetic document diagnostic suite, operand plans, integrity manifest and historical response snapshots inside the package.
- Run provider-neutral document evaluations through `telos eval --suite document512`, with separate direct QA, exact source extraction and deterministic calculator tracks.
- Export tasks and replay reference or imported responses offline with `telos eval documents`; render each track in the cream report viewer.
- Save raw completions, request attempts, model revisions and local weight hashes; resume only matching run configurations.

### Fixed
- Respect explicit local evaluation backends and revision pins. Document runs disable hidden HTTP retries and retain malformed outputs and backend failures.

## [1.6.1] - 2026-11-14

### Fixed
- Use the actual Télos logo in the Reader-1 overview figure.
- Remove decorative dashboard and figure copy; retain results, provenance, controls, and essential scoring context.

## [1.6.0] - 2026-11-14

### Added
- Evaluation Beautification Display: standalone, offline HTML reports in the Télos research figures' cream/parchment style, with category scores, filters, CSV/JSON exports, and case-level evidence inspection.
- `telos eval report` for saved native, tool-use, contextual-probe, audited historical, and Reader-1 reports; `--html-report` exports after evaluation.
- Adaptive live progress for functional, tool-use, and contextual-probe evaluation, with plain CI output, `NO_COLOR`, and `--display auto|plain|off`.
- Reproducible Reader-1 diagnostic artifacts: 34 pilot cases, six follow-up probes, and 240 parameter-generated synthetic cases. The extension scored 162/240 under its fixed checks; all 60 sampled formula replays matched. These are diagnostic results, not production accuracy estimates.

### Fixed
- Preserve missing metrics and unscored records in displays instead of inventing zero values or task outcomes.
- Avoid counting native code aliases and dual-track summaries twice; keep historical saved-success labels and model identities.
- Escape report strings and validate Unicode evidence offsets before highlighting source text.

## [1.5.5] - 2026-11-13

### Fixed
- Add complete SFT/DPO resume on CPU, CUDA, MLX and XLA, preserving frozen DPO references, optimizer state, RNGs and samplers.
- Save and restore per-rank pretraining checkpoints with topology validation.
- Seed XLA stochastic updates by global step and rank to avoid graph-tracing RNG drift; reject older stochastic checkpoints lacking this strategy. Physical TPU continuation retesting remains outstanding.
- Preserve adaptive MLX COROSred loss tracking and correct MLX SFT collation.

## [1.5.4] - 2026-11-13

### Fixed
- Enable multi-GPU CUDA SFT and DPO with device-count validation, per-device batching, and unwrapped portable model checkpoints.

## [1.5.3] - 2026-11-13

### Fixed
- Train the MLX COROSred reliability head before competence gating and track measured accuracy/AUC across resume.
- Blend confidence-targeted and exploratory infilling masks; apply ambiguity exclusion to unreduced reliability loss.

## [1.5.2] - 2026-11-13

### Fixed
- Initialize the XLA runtime explicitly for native evaluation on requested or automatically detected TPU devices.
- Propagate evaluation placement failures instead of silently continuing on CPU.
- Abort on multi-core XLA launch or worker failure instead of falling back to single-core training.

## [1.5.1] - 2026-11-13

### Added
- Complete pretraining resume checkpoints on MLX and single-process PyTorch: model, optimizer, master weights, scheduler, RNGs, data cursor, diffusion state, and PyTorch COROSred tracking/routing state.
- `telos pretrain --resume PATH` restores the original run configuration and checks dataset and tokenizer fingerprints. Legacy numeric resume resolves an actual checkpoint instead of skipping steps.

### Fixed
- Write training checkpoints atomically and propagate save failures, preserving the previous complete checkpoint.
- Restore PyTorch scheduler state without advancing it a second time; preserve MLX optimizer initialization and fp32 master weights.
- Reject incomplete/legacy and distributed checkpoints for full resume; retain weights-only initialization through `--init-checkpoint`.
- Create published GitHub releases with changelog notes and distribution assets after tagged PyPI publication.

## [1.5.0] - 2026-11-13

### Security
- Run generated Python, JavaScript, and Rust evaluation code inside a network-denying OS sandbox on macOS and Linux; fail closed when the sandbox backend is unavailable.
- Prevent JavaScript candidates from calling `process.exit` to skip the test harness.

### Fixed
- Do not count structural-only polyglot checks as functional execution passes when a runtime is unavailable.
- Enforce `dataprep --tokens` exactly, including zero-token output.

## [Unreleased]

## [1.4.0] - 2026-11-12

### Added
- `telos pretrain` command with `telos train` retained as a compatibility alias.
- Full-parameter SFT and DPO post-training for native Télos causal checkpoints.
- MLX post-training for Apple Silicon, including PyTorch-to-MLX checkpoint weight conversion.
- CUDA, CPU, and PyTorch-XLA post-training paths; XLA validates rank identity, partitions batches by ordinal, reduces gradients across replicas, and synchronizes checkpoint writes.

### Fixed
- XLA master-rank detection now fails closed instead of assuming the current process is rank zero when rank APIs fail.
- Avoid re-importing the optional MLX runtime in diffusion modules when Metal is unavailable, preventing duplicate native-extension initialization in headless CPU/TPU environments.

## [1.3.2] - 2026-11-12

### Added
- Reproducible, memory-guarded round-two MLX scheduling and microbatch benchmarks, with paired run evidence and a detailed report.

### Performance
- Optional Metal command-buffer limits of 16 operations / 16 MiB reduced measured peak MLX allocation by 19.3% (1.58 to 1.28 GiB) on the existing experimental 50M AR fast path, at similar throughput. The measured +2.8% throughput difference remains within run variation; training defaults are unchanged and long-run convergence is unverified.

## [1.3.1] - 2026-11-12

### Added
- Opt-in MLX compiled backward accumulation and forced fused SDPA attention.
- Experimental packed SwiGLU Metal VJP, retaining existing checkpoint parameter shapes.
- Bounded profiling tools and reports for MLX training throughput, memory use, and GPU execution.

### Performance
- Deferring final accumulated-gradient materialization improves measured throughput by 0.29% while keeping memory bounded.
- A short paired 50M AR benchmark measured 42.0k tokens/s with all experimental options enabled versus 36.5k on defaults (+15.1%); measured peak MLX allocation was 1.59 versus 1.73 GiB. Experimental options remain disabled by default, and the short run does not establish long-run training equivalence.

## [1.3.0] - 2026-11-10

### Added
- Fused Metal cross entropy with a custom first-order backward pass for MLX language-model objectives.
- Compiled MLX optimizer updates and isolated microbatch tuning that preserves the configured effective batch.
- Experimental packed SwiGLU gate/up projection with automatic loading of legacy split checkpoints.
- Experimental tiled tied-embedding AR vocabulary loss, disabled by default.

### Fixed
- Preserve transformer-layer gradients when MLX gradient checkpointing is enabled.
- Bound gradient-accumulation memory by evaluating each microbatch before constructing the next graph.
- Wrap data batches correctly when the requested batch exceeds the token dataset.

## [1.2.0] - 2026-10-21

### Security
- Replaced unsafe `eval()` in tool-use evaluation math parser with an explicit AST arithmetic evaluator (`_eval_ast_math`), eliminating sandbox escape vectors.
- Enforced `weights_only=True` across `torch.load()` checkpoint deserialization in trainer, CLI, and evaluation runners.
- Defaulted `trust_remote_code=False` in HuggingFace model adapter to prevent automatic execution of unreviewed Hub code.
- Passed Google Gemini API keys via `x-goog-api-key` HTTP request headers instead of query parameters.
- Replaced `shell=True` in remote system monitor subprocess invocation.

### Added
- Dedicated PyPI package documentation (`README_PYPI.md`) focusing on installation, hardware acceleration, CLI, and Python API usage.
- Unit regression tests verifying safe arithmetic parsing and rejection of sandbox escape payloads.

## [1.1.0] - 2026-10-20

### Fixed
- TPU replica rank detection now uses `torch_xla.runtime.global_ordinal()`.
  Rank lookup failures abort training instead of silently treating every worker
  as rank zero and duplicating input slices. Invalid replicated ranks are rejected.
- Added regression coverage for distinct eight-worker ranks and data slices,
  rank lookup failures, and out-of-range replica identities.

### Added
- Interleaved COROSred decoding.
- Research tooling and evidence documenting historical TPU rank collapse and
  matched evaluation of the corrected TPU and CUDA checkpoints.

## [1.0.0] - 2026-09-23

First public release on PyPI as `telos-ml` (`pip install telos-ml`).

### Added
- **`telos-ml` PyPI Distribution** — Published package to PyPI under the name
  `telos-ml`, preserving the `telos` Python module import and the `telos` CLI
  command.
- **`telos dataprep`** — prepares raw text, code directories, JSONL, or Hugging
  Face datasets into memory-mapped binary token streams (`.bin`).
- **`telos train`** — zero-config dimensional trainer. Specify parameters and
  tokens; the geometry solver derives `d_model`, layers, and heads. Supports the
  AR, MDLM, UNDLM, and COROSred paradigms on MLX (Apple Silicon), CUDA, TPU via
  PyTorch-XLA, and CPU.
- **COROSred** — the unified dual-paradigm trainer, combining causal AR drafting,
  uniform masked-diffusion infilling, and confidence-guided selective
  re-diffusion, with a Learned Reliability Head driving routing.
- **`telos eval`** — multi-domain evaluation: contextual probes with bootstrap
  confidence intervals, sandboxed functional execution (Pass@1, AST validity),
  anti-cheat span masking, linguistic probes, and tool-use / function-calling
  benchmarks.
- **`telos bench`** — hardware throughput and step-latency benchmark, capped at
  five minutes.
- **`telos test`** — unified verification suite for model contracts, causality,
  losses, and samplers.
- **`telos.afm`** — Python access to Apple's on-device Foundation Models
  (AFM 3 Core / Core Advanced) on Apple Silicon. Exposes `probe()`, `require()`,
  `generate()`, and a persistent `AFMBridge`, plus a `telos afm status` /
  `telos afm generate` CLI. The Swift bridge is compiled on first use into a user
  cache directory.
- Bundled default ByteLevel BPE tokenizer, so an installed package works with no
  manual downloads.
- Graceful degradation when MLX is installed but Metal is unavailable (headless,
  virtualized, or sandboxed macOS sessions): `telos train`, `telos bench` and
  `telos test` fall back to the PyTorch CPU path instead of aborting the process.

[Unreleased]: https://github.com/kazenoko-git/telos/compare/v1.8.5...HEAD
[1.5.5]: https://github.com/kazenoko-git/telos/compare/v1.5.4...v1.5.5
[1.5.4]: https://github.com/kazenoko-git/telos/compare/v1.5.3...v1.5.4
[1.5.3]: https://github.com/kazenoko-git/telos/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/kazenoko-git/telos/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/kazenoko-git/telos/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/kazenoko-git/telos/compare/v1.4.0...v1.5.0
[1.1.0]: https://github.com/kazenoko-git/telos/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/kazenoko-git/telos/releases/tag/v1.0.0

[1.7.3]: https://github.com/kazenoko-git/telos/releases/tag/v1.7.3

[1.7.4]: https://github.com/kazenoko-git/telos/compare/v1.7.3...v1.7.4
[1.7.5]: https://github.com/kazenoko-git/telos/compare/v1.7.4...v1.7.5
[1.7.8]: https://github.com/kazenoko-git/telos/compare/v1.7.7...v1.7.8
[1.8.0]: https://github.com/kazenoko-git/telos/compare/v1.7.8...v1.8.0
