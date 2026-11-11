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

## Follow-up tooling

An Instruments user template named `Telos MLX GPU Profile` is now saved with the `Performance Limiters` counter set, Shader Timeline enabled, and a five-second capture window. The macOS `metalperftrace` lookback for the same period contained only `Arc` and `stable`, not the Python training process, so its always-on graphics metrics cannot answer this MLX compute-kernel question.
