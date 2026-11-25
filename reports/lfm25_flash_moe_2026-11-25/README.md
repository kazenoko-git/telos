# Guarded LFM2.5-8B-A1B resident and SSD comparison

The installed LiquidAI LFM2.5-8B-A1B MLX 4-bit checkpoint ran in separate fresh processes on the M5 Pro with MLX 0.32.2, AC power and High Power mode. Native Telos used the same 16-token raw prompt, 128-token context and 16-token greedy generation limit in both modes. Each process served one request without warmup. The resident process exited before SSD mode loaded. MLX-LM was never imported.

| Measurement | Resident | SSD experts |
| --- | ---: | ---: |
| Load time | 1.30 s | 0.31 s |
| First token | 490 ms | 992 ms |
| Decode, tokens 1–16 | 50.96 tok/s | 4.82 tok/s |
| Total request | 0.785 s | 4.101 s |
| MLX peak during generation | 4,758 MiB | 964 MiB |
| Process peak RSS | 2,769 MiB | 1,269 MiB |
| Minimum sampled macOS free memory | 50% | 66% |

All 16 generated token IDs and the decoded text matched. The resident run used an 8 GiB admission budget and the SSD run used 2 GiB. The SSD cache held about 508 MiB after generation under its 512 MiB cap. Telos recorded 10.05 GiB of expert reads for this one request, including prefill. This is an application read count. No device-level I/O was measured. First-token timings include cold startup effects within the first generation request. Resident generation uses lookahead and SSD generation submits one token at a time.

The watchdog sampled about every 100 ms, including during weight load. It would stop both the active process and the remaining comparison at any measured growth in used swap or cumulative swap-outs. It also required 65% free memory before starting either worker, 40% while running, and at least 768 MiB free swap, with per-process RSS ceilings of 6.5 GiB and 2,500 MiB. MLX's allocator guidelines were 6 GiB and 2 GiB with a 128 MiB free cache cap for both.

An initial resident attempt ended at 49% reported free memory under a stricter 50% floor. Swap use and cumulative swap-outs stayed flat. After the worker exited, free memory recovered to 71% and the comparison restarted with the 40% floor. In the completed comparison used swap stayed at 4,424.38 MiB and the cumulative swap-out count stayed at 37,039,719 pages.

[Resident record](resident.json) · [SSD record](ssd_experts.json) · [Guard samples](samples.jsonl)
