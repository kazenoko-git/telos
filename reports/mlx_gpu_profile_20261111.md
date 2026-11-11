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

## Bandwidth hypothesis follow-up

The shader trace includes MLX `steel_gemm_fused_nax` and `steel_gemm_splitk_nax` BF16 kernels, confirming that this training run dispatched NAX GEMM work. A corrected Instruments capture enabled the `Performance Limiters` counter set in both Metal instruments and produced a populated GPU counter summary for an 8-update, 131,072-token 50M AR run (MBS 8, accumulation 8, sequence length 256). The run completed in 4.93 s; peak MLX allocation was 1.68 GiB. Its final system sample reported 45% free memory and 878 MiB free swap. The trace itself was 384 MiB on disk after processing.

For the selected whole-capture interval, Instruments reported mean GPU bandwidth of 124.78 GB/s (read 76.75 + write 48.03 GB/s) and a maximum combined sample of 413.65 GB/s. Apple documents these bandwidth counter units as GB/s and says the counter summary means are aggregated over the selected time range. Apple publishes 307 GB/s memory bandwidth for M5 Pro. The mean is about 41% of that figure and is not evidence of saturation. The short-window maximum exceeds the published figure; the trace summary does not explain the counter's sampling interval or why the peak exceeds the published system bandwidth, so do not interpret it as sustainable physical throughput. The read and write maxima (329.90 and 322.31 GB/s) are separate maxima and should not be added together.

The same whole-capture summary reported mean `Memory Request Stalls Influence` of 4.55% (maximum 92.66%), mean `Last Level Cache Limiter` of 69.65%, mean `Neural Accelerator Utilization` of 19.92% (maximum 99.43%), and mean `Neural Accelerator Limiter` of 20.01%. These averages include the full selected capture and are not per-kernel values. They show NAX activity and some high memory/cache-limiter samples, but do not isolate the training kernels or establish a sustained external-memory-bandwidth bottleneck. This evidence supports investigating per-kernel timing and counter aggregation before changing training behavior.

A repeat of the same 8-update workload completed in 4.23 s with peak MLX allocation of 1.68 GiB. Its whole-capture mean GPU bandwidth was 142.38 GB/s (read 92.11 + write 50.27 GB/s), with a maximum combined sample of 432.66 GB/s; read and write maxima were 343.45 and 340.30 GB/s. Mean memory-request stall influence was 6.08% (maximum 89.77%), mean LLC limiter 81.46%, and mean Neural Accelerator utilization 32.31% (maximum 99.46%). Thus >400 GB/s peaks recurred in a second capture, so it was not unique to the first run. The variation in means and repeated above-spec maxima still leave sampling/aggregation semantics unresolved; neither run demonstrates sustained transfer above the published bandwidth.

## Follow-up tooling

An Instruments user template named `Telos MLX GPU Profile` is saved with the `Performance Limiters` counter set, Shader Timeline enabled, and a five-second capture window. The macOS `metalperftrace` lookback for the same period contained only `Arc` and `stable`, not the Python training process, so its always-on graphics metrics cannot answer this MLX compute-kernel question. The second capture's counter summary was read in Instruments; its full GPU counter-value table was not exported, to avoid unnecessary memory use.
