# ANE/GPU linear projection pipeline probe — 2026-11-11

## Setup

Prototype an accumulation-window schedule with fixed weights: Core ML (`CPU_AND_NE`) predicts the next 2048x512 by 512x512 linear projection on a single worker thread while MLX evaluates the analytic backward for the previous microbatch on the GPU. A one-item future queue bounds prefetch to one microbatch. No optimizer update occurs during these 16 microbatches. Compare this pipeline against serial Core ML-forward/GPU-backward and all-GPU FP16 forward/backward. The GPU backward computes input and weight gradients plus mean-square loss; ANE and GPU modes use the same Core ML forward outputs within each hybrid comparison.

## Timing

Two short runs per mode were interleaved. Total time for 16 microbatches:

| Mode | Run times | Median total | Throughput |
|---|---:|---:|---:|
| Serial Core ML forward + GPU backward | 39.10 ms / 19.93 ms | 29.52 ms | 542 microbatches/s |
| One-microbatch overlap pipeline | 21.48 ms / 14.61 ms | 18.05 ms | 887 microbatches/s |
| GPU-only forward + backward | 16.71 ms / 16.59 ms | 16.65 ms | 961 microbatches/s |

The overlapped schedule reduced elapsed time about 39% versus serial Core ML/GPU handoff in this tiny projection probe. It remained about 8% slower than GPU-only. Timing varied substantially between the two trials, so this needs repetition before further investment.

Mean hybrid loss was 0.00319395 for both serial and pipeline modes. GPU-only was 0.00319334, reflecting small FP16 forward differences. The prototype computed GPU input and weight gradients; it did not train a transformer or apply an optimizer update.

Peak MLX allocations stayed under 0.19 GiB and system swap did not change during the probe (861 MiB free at both endpoints). This excludes Core ML/ANE allocations from the MLX peak; system memory free went from 35% to 34%.

## Limits

This establishes a plausible host-scheduled pipeline for one independent projection and its backward pass, not verified simultaneous ANE and GPU execution. `CPU_AND_NE` allows Core ML to use the CPU and Neural Engine, and the captured endpoint telemetry did not sample utilization during the run. No Xcode trace was taken. Thus the time reduction over serial mode shows overlap or scheduling benefit, but does not prove the projection ran on the ANE while the GPU was executing kernels.

This is a scheduling prototype, not an MLX trainer integration. A follow-up would need an Instruments trace showing ANE and GPU activity overlap, then a real accumulated-gradient check and end-to-end training benchmark with the same weights held through the accumulation window. The GPU-only path was already faster here, so there is no current basis to integrate the pipeline into Télos training.
