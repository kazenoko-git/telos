# MLX 50M GPU profile (2026-11-11)

## Workload

- 50M autoregressive transformer, MBS 8, gradient accumulation 8, sequence length 256
- Real `data/english_corpus_5b.bin`, 262,144 input tokens, 16 optimizer updates
- Apple M5 Pro, MLX 0.32.2; Xcode Metal System Trace
- Swap was already about 5.5 GiB used before the run.

## Measurements

- Training harness wall time: 8.96 s, about 29.3k input tokens/s including setup and trace launch.
- Sum of measured optimizer-step times: 8.19 s, about 32.0k input tokens/s within the measured steps; median step 0.497 s, last eight average 0.494 s.
- Peak MLX allocation: 1.67 GiB. System memory headroom was 37% at the final sample; swap usage stayed at 5.50 GiB during the run.
- System GPU telemetry had only two samples: device/renderer utilization was 35%/35% initially and 89%/76% at the final sample. These are system-wide samples, not process-specific averages.
- The trace contained 40,520 GPU activity intervals and 8,474 command-buffer completion events. Its Metal shader timeline was disabled, so the shader interval table contained no rows or kernel names.

## Interpretation and limits

The run reached high system GPU activity while training and used 1.67 GiB of MLX-managed memory. This does not establish sustained GPU saturation: telemetry was sparse and system-wide. The capture cannot identify which MLX kernels dominate or distinguish compute, memory-bandwidth, and launch-overhead limits. No Neural Engine occupancy counter was captured.

The next useful profile must enable Metal shader timeline or GPU counters and collect enough samples to attribute GPU time to kernels. The attempted counter-table export grew past 268 MB while free swap fell to about 327 MB; it was stopped to protect memory headroom. No second training run was started.

## Follow-up shader profile

A second, shorter capture used the saved shader-enabled Instruments template on the same 50M workload (MBS 8, accumulation 8, sequence length 256), for 8 optimizer updates and 131,072 input tokens. It completed in 5.54 s wall time; measured update calls summed to 4.64 s (about 0.58 s each). Peak MLX allocation was 1.64 GiB. The final system-wide GPU sample was 89% device / 77% renderer, while free memory was 42% and only about 519 MiB of swap was free.

Instruments attributed 112.37 ms of Compute-channel activity to the Python training process across 86 intervals (4.22 ms average CPU-to-GPU latency, 2.37 ms longest interval). It attributed 109.02 ms across 75 Metal encoders. These are profiler channel/encoder measurements, not a direct fraction of end-to-end step time or a GPU utilization percentage. Shader Timeline recorded short kernels: the two largest grouped families were Steel GEMM variants (48 intervals / 4.32 ms and 17 / 2.39 ms); fused elementwise families accounted for 1.88 ms and 1.80 ms, and cross-entropy gradient intervals for 0.57 ms. These shader interval durations are samples and should not be summed as a complete GPU-time accounting.

The brief per-process Compute activity compared with multi-hundred-millisecond update calls makes host launch, synchronization, or other non-GPU work a strong bottleneck candidate. The system-wide utilization samples do not contradict that: they may include other applications and do not give sustained Python-process utilization. This trace alone cannot distinguish CPU-side work from GPU queue gaps, command-buffer waits, or profiler accounting limits, so the bottleneck is not yet conclusively identified. The Time Profiler samples were dominated by the Python main thread and shutdown/finalization, but are too sparse in this short capture to quantify the training loop's CPU cost.

Processing the shader trace temporarily drove Instruments to roughly 7 GB RSS and increased swap use substantially; system memory later recovered to 56% free. Keep the 755 MB trace and summary for reference, but avoid more trace exports or training captures until there is ample memory headroom.

## Synchronized stage timing

To break down the update without another Instruments trace, an opt-in stage profiler synchronized at each boundary and ran 8 updates of the same workload. It measured the corpus batch slice plus per-microbatch conversion, forward/backward completion, gradient accumulation/materialization, and optimizer update. The first two updates were excluded from steady-state averages because they include initial compilation and optimizer setup.

Across updates 3–8, mean times per optimizer update were 0.42 ms for batch preparation, 415.81 ms for forward/backward, 11.57 ms for gradient materialization, and 12.96 ms for the optimizer update. End-to-end update time averaged 441.79 ms. Peak MLX allocation was 1.90 GiB; system memory ended at 60% free with about 1.20 GiB swap free.

Forward/backward completion accounted for about 94% of the measured update time. Batch preparation, materialization, and optimizer work are comparatively small; eliminating or overlapping accumulation materialization alone is unlikely to close the speed gap. This directs optimization toward the model's forward/backward kernels and shapes. The explicit synchronization changes scheduling and can affect timings, so these stage numbers diagnose cost distribution rather than serve as an uninstrumented throughput benchmark. This run does not explain why the earlier Instruments Compute-channel accounting reported only 112 ms; that measurement appears incomplete for total device work in this workload.

The diagnostic is opt-in (`--stage-profile`); without it, training follows the existing execution path. It forces completed outputs between microbatches to keep the memory bound while timing each stage.

## Sustained one-minute profile

Two low-overhead profiles ran the same 50M AR model on the real corpus (MBS 8, accumulation 8, sequence length 256). The first completed 128 optimizer updates in 55.07 s. The follow-up completed 140 updates and 2,293,760 tokens in 60.59 s, about 37.91k input tokens/s end-to-end. Across its 140 training-step calls, the mean was 432.23 ms (median 427.20 ms; p95 477.32 ms), equivalent to 37.91k tokens/s; excluding the first eight updates gives about 38.02k tokens/s. This confirms warmed-up throughput around 38k tokens/s for this workload, not 100k tokens/s.

Peak MLX allocation in the 60-second run was 1.77 GiB and process peak RSS was about 0.25 GB. System-wide free memory ranged from 50% to 66%; free swap stayed between 1,303 and 1,391 MiB and ended at 1,391 MiB. The low-overhead monitor sampled system telemetry every five seconds. During active training, system-wide device utilization was usually 93–95% (median 94%); renderer utilization had a median of 58%. These are whole-system readings, not per-process utilization or NAX-specific activity. This run used no Instruments trace, avoiding the high memory and trace-size cost of collecting GPU counters continuously for a minute. Raw timings and telemetry are in `logs/mlx_sustained_profile_20261111_60s/`.

## Kernel-level candidate check

The existing experimental tiled vocabulary loss was compared with the standard output projection plus Metal cross-entropy on the real 50M AR corpus. Each run used the same seed, MBS 8, accumulation 8, sequence length 256, and 24 updates; throughput below excludes the first eight update calls. The existing parity test passed both loss and gradient checks on its small float32 model, within its documented reduction-order tolerances.

| Path | Steady update throughput | Peak MLX allocation |
| --- | ---: | ---: |
| Standard projection + Metal CE | 38.1k tokens/s | 1.72 GiB |
| Tiled loss, chunk 256 | 31.7k tokens/s | 1.86 GiB |
| Tiled loss, chunk 1,024 | 33.6k tokens/s | 1.79 GiB |
| Tiled loss, chunk 8,192 | 33.0k tokens/s | 1.78 GiB |

The tiled path was 12–17% slower after warmup and did not reduce peak MLX allocation in these runs. Keep it experimental and disabled for throughput-oriented training. Short-run final losses were close, but this comparison is not a full convergence or 50M gradient-equivalence test.

An isolated compiled BF16 block microbenchmark used the 50M block shape (batch 8, sequence 256, width 512). Parameter-gradient timings averaged 1.94 ms for attention, 1.99 ms for MLP, and 3.80 ms for the full block. The isolated branch timings omit gradients with respect to the incoming hidden state, so they are useful for relative kernel-shape triage, not end-to-end attribution. Attention and MLP were similar; neither emerged as a single obvious target. These measurements favor investigating lower-level kernel/launch behavior while preserving the current model math.

## Compiled-call, evaluation, and wait costs

The synchronized stage profiler was extended to time the Python call into the already-compiled microbatch function, the subsequent `mx.eval` call, and `mx.synchronize` separately. Two short 8-update runs were recorded in each mode, all at MBS 8 / accumulation 8 / sequence length 256; averages below exclude the first two updates. The asynchronous diagnostic substitutes `mx.async_eval` before the same explicit synchronization, so it still materializes gradients at each microbatch boundary.

| Evaluation mode | Run | Compiled function call | Eval call | Explicit sync wait | Forward/backward total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mx.eval` | 1 | 7.3 ms | 548.6 ms | 4.7 ms | 560.7 ms |
| `mx.eval` | 2 | 5.8 ms | 419.7 ms | 5.1 ms | 430.6 ms |
| `mx.async_eval` | 1 | 4.8 ms | 277.2 ms | 115.4 ms | 397.4 ms |
| `mx.async_eval` | 2 | 5.7 ms | 307.0 ms | 116.7 ms | 429.4 ms |

The repeated pair is the informative one: `mx.eval` call plus explicit wait took 424.8 ms/update; `mx.async_eval` plus wait took 423.7 ms/update. Async evaluation mostly moved about 110 ms from the eval call into the synchronization wait; it did not produce a repeatable speedup when the profiler synchronized at every microbatch. Calling the compiled function itself cost only about 5–7 ms/update, roughly 1–2% of forward/backward time. This does not support Python invocation of the compiled graph as the main bottleneck. The remaining eval-call time includes MLX evaluation and command submission work; this timer alone cannot divide host graph construction from device execution.

The earlier shader trace provides partial, not whole-run, kernel-launch evidence. The Instruments recording lasted 7.165 s, but its shader intervals span only 6.697–7.141 s (445 ms), near the final optimizer update and process shutdown. In that slice, the Python process had 75 command-buffer submissions with one encoder each; their encoder times summed to 109.0 ms (median 1.28 ms each). Shader Timeline listed 923 intervals over the slice, with 44.0 ms summed interval duration. Those intervals show many short dispatched kernels, but the capture does not establish that launch gaps dominate steady-state training.

A follow-up Metal System Trace attached during warmed-up training and captured 4.872 s of active work, overlapping optimizer updates 32–42 of a 64-update run. The run completed 1,048,576 tokens in 30.40 s; peak MLX memory was about 1.9 GiB and free swap stayed around 1.15–1.26 GiB. In the trace, the training process submitted 2,688 command buffers, 2,601 of them with one encoder. Across those encoded submissions, encoder time summed to 3,089.6 ms over the 4.872 s trace; median encoder time was 1.01 ms and p95 was 2.40 ms. This is about 236 encoded command buffers per optimizer update, or roughly 29.5 per microbatch. Submission start-to-start cadence had a 1.20 ms median and 2.69 ms p95. These are encoder and submission timing measures, not a utilization percentage: their aggregate does not tell us whether the remaining trace time was GPU queue idle, host work, waiting, or other unaccounted GPU activity.

As a separate check, the command-buffer timeline intervals themselves span 4.46 s and their union covers 99.7% of that window. The median interval gap is about 4 microseconds, the 95th percentile is about 9 microseconds, and the largest is about 53 microseconds. This supports the view that command-buffer activity is almost continuous in the recorded timeline. It still does not prove the GPU was executing kernels for that whole time: the table marks command-buffer submission events and their durations, not a dedicated GPU busy/idle counter.

The active trace's Time Profiler contained 4,025 process samples, with 3,371 on the Python main thread. Sampled stacks during training include MLX evaluation (`eval_impl`, `gpu::eval`, `Compiled::eval_gpu`), Metal command encoding/submission (`CommandEncoder`, `AGX::performEnqueueKernel`), Metal allocation/free, and Steel GEMM paths. This confirms that evaluation, command submission, allocation, and GEMM are all present in the active path, but the symbolized sample volume is insufficient to assign exact wall-time shares. The shader-interval export had schema only and no event rows, so this capture does not attribute individual GPU kernels. The earlier tail-only Time Profiler evidence is superseded for the question of whether those dispatch stacks occur during active training; it remains valid that the tail trace included teardown.

Together with the stage timings, this points away from Python's call into the compiled function as the main cost (about 1–2% of forward/backward time), and toward MLX evaluation/device completion with substantial command-buffer dispatch volume. It does not yet prove launch overhead is the limiting factor: many 1 ms encoders can contain significant GPU work, and aggregate encoder time is not a GPU busy-time counter. A more targeted low-memory next experiment is a bounded one-microbatch-ahead async accumulation path, compared against the current path at identical effective batch and seed, with memory and gradient/loss equivalence checked. The current async diagnostic synchronized immediately after each microbatch and therefore measured no useful overlap.

A later 2-second repeat capture during a 16-update run did not yield usable trace tables: Instruments hit an internal Metal residency assertion. The training run itself completed with 1.93 GiB peak MLX allocation, 36% system memory free, and about 850 MiB free swap. The failed 1.1 GiB trace bundle was removed; its contents are not used for the interval analysis above.

## Bandwidth hypothesis follow-up

The shader trace includes MLX `steel_gemm_fused_nax` and `steel_gemm_splitk_nax` BF16 kernels, confirming that this training run dispatched NAX GEMM work. Two corrected counter captures enabled the `Performance Limiters` counter set for the same 8-update, 131,072-token 50M AR workload (MBS 8, accumulation 8, sequence length 256). The runs completed in 4.93 s and 4.23 s, with peak MLX allocations of 1.68 GiB and 1.81 GiB. The first run ended at 45% free system memory and 878 MiB free swap.

**Correction: the counter traces are partial captures, not whole-run measurements.** Their saved run intervals span only 0.445 s and 0.451 s, respectively, versus the 4.93 s and 4.23 s training-harness wall times. The displayed means therefore describe a small captured slice, not full-run averages. The counter exports contain 4,013 and 4,307 GPU-bandwidth sample intervals, covering about 209 ms and 220 ms of sampled intervals within those short slices.

Across the recorded samples, combined GPU bandwidth had means of 124.78 and 142.38 GB/s, medians of 111.27 and 133.08 GB/s, and 95th percentiles of 285.25 and 307.09 GB/s. The maxima were 413.65 and 432.66 GB/s, each lasting about 50 microseconds. Only 3.1% and 5.0% of samples exceeded Apple's published 307 GB/s M5 Pro bandwidth figure. These peaks are isolated counter windows, not evidence of sustained external-memory throughput above the system specification. Read and write means add to the combined mean; separate read/write maxima happen at different times and must not be added.

In the repeat, samples overlapping Steel GEMM/NAX shader intervals averaged 182.1 GB/s, versus 118.6 GB/s when no shader interval was recorded; 12.1% of GEMM-overlapping samples exceeded 307 GB/s. The repeat's maximum sample coincided with a `steel_gemm_fused_nax_tn_bfloat16` interval. In the first corrected slice, GEMM-overlapping samples averaged 185.9 GB/s, but its maximum only overlapped two tiny non-GEMM shader intervals by roughly 1 microsecond each. This suggests GEMM bursts contribute to some high samples, but the captured intervals cover only a small part of each run and counter/shader alignment is approximate. It does not quantify full-run kernel bandwidth or establish a sustained bottleneck.

The partial-slice means for `Memory Request Stalls Influence` were 4.55% and 6.08%; `Last Level Cache Limiter` means were 69.65% and 81.46%; and `Neural Accelerator Utilization` means were 19.92% and 32.31%, with maxima near 99.5%. These are aggregated counter samples, not per-kernel or full-run values. They confirm NAX activity in the slices but do not establish continuous bandwidth saturation. A useful next measurement needs a counter trace whose recorded interval spans the workload; verify its duration against the training wall-time log before interpreting averages. Avoid another large export while swap headroom is low.

## Follow-up tooling

An Instruments user template named `Telos MLX GPU Profile` is saved with the `Performance Limiters` counter set, Shader Timeline enabled, and a five-second capture window. The macOS `metalperftrace` lookback for the same period contained only `Arc` and `stable`, not the Python training process, so its always-on graphics metrics cannot answer this MLX compute-kernel question. The second capture's counter summary was read in Instruments; its full GPU counter-value table was not exported, to avoid unnecessary memory use.
