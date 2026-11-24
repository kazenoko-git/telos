# Native Bonsai 4B KV cache benchmark

The M5 Pro ran the original Ternary Bonsai 4B 2-bit Safetensors checkpoint through native Telos with MLX 0.32.2. The [native benchmark workload](../bonsai4b_native_inference_2026-11-24/benchmark.py) uses 212 prompt tokens, a 1,024-token context and a budget of 192 greedy tokens. Decode timing spans delivery of tokens 1 through 192.

The baseline at `01b6faac` grew every layer's KV cache by concatenating the previous prefix with each new token. The candidate keeps prefill compact, reserves space in blocks of up to 256 tokens during decode, and writes new entries by index. Rotary position and the attention range follow the number of valid entries.

## Alternating trials with shared weights

Both implementations used the same loaded weight arrays in one process. Each ran an 8-token warmup. The order alternated across pairs.

| Pair | Baseline tok/s | Indexed tok/s | Change |
| --- | ---: | ---: | ---: |
| 1 | 131.63 | 50.70 | −61.5% |
| 2 | 88.51 | 92.02 | +4.0% |
| 3 | 85.53 | 88.96 | +4.0% |
| 4 | 86.44 | 91.40 | +5.7% |
| 5 | 84.91 | 90.86 | +7.0% |
| 6 | 87.90 | 90.69 | +3.2% |
| 7 | 87.63 | 88.71 | +1.2% |
| 8 | 88.24 | 88.49 | +0.3% |
| Median | 87.77 | 89.83 | +3.6% paired |

The candidate was faster in seven of eight pairs. Median time to first token was 205.7 ms for the baseline and 208.2 ms for indexed writes. Both peaked at 1,607 MiB of MLX allocations. All 16 runs emitted the same 192 token IDs. The SHA-256 of the compact JSON token array is `18b3b514746e060bf1c5895359d31f1bd226ca9cc27e60ca9dc895e01dfdb874`.

Four earlier fresh-process pairs had nearly identical medians at 82.64 tok/s for the baseline and 82.43 tok/s for indexed writes. Individual runs across those pairs ranged from 63.54 to 134.72 tok/s. The shared-weights result is evidence of a small gain on this workload, with substantial run-to-run variability. LM Studio was outside this round of testing.

An initial screen also tried compiling the gated MLP activation. That variant raised the measured peak from 1,607 to 1,625 MiB. An early cache reservation variant reached 1,559 MiB in that screen. The version selected for the runtime reserves cache space only as decode needs it, which preserved the baseline peak in the paired trials.
