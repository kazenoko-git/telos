# Inkling W=16 whole-layer compilation

The compiled expert block remains the baseline. Compiling the whole steady-state layer failed exact token parity on the real Inkling 2 Mini W=16 adapter. Its faster warm steps came with a cold compilation cost and higher peak process RSS.

## Guarded comparison

Native Telos used MLX 0.32.2 on an Apple M5 Pro in High Power mode. Both workers used the same 3 GiB budget, 512-token context, pinned W=16 experts, greedy sampling, and [three prompts](../inkling2_mini_pinned_w16_2026-11-26/prompts.jsonl) with a 65-token cap. Workers ran one at a time, compiled experts first, with checks that abort on growth in swap use or cumulative swap-outs. Both counters held steady. [Raw comparison](comparison.json)

| Metric | Compiled expert block | Compiled whole steady layer |
| --- | ---: | ---: |
| Decode throughput across all 143 post-prefill steps | 17.20 tok/s | 14.50 tok/s |
| Steady throughput across all 134 no-read steps | 21.82 tok/s | 17.58 tok/s |
| Steady throughput excluding the first steady step in each worker | 21.81 tok/s | 25.69 tok/s |
| First steady step | 44 ms | 2,445 ms |
| Peak process RSS | 1,618 MiB | 2,598 MiB |
| Peak MLX allocations | 2,161 MiB | 2,161 MiB |
| Physical decode reads | 3,254 MiB | 3,592 MiB |
| Physical decode writes | 11.0 MiB | 12.4 MiB |

Both workers sampled 65, 65, and 16 IDs. Whole-layer compilation changed the first prompt at sampled position 32, where the compiled expert block selected ID 161458 and the whole layer selected ID 220. The other two prompts matched exactly. The divergent continuation limits what the speed and SSD totals can establish about an equivalent workload. Physical counters cover the system disk.

## Parity checks

A tiny deterministic fixture with active attention LoRA produced bitwise identical hidden states, logits, live KV entries, and convolution history across 20 decode tokens, five routing refreshes, and the fixture's context limit. On the real prompt that diverged, a guarded 40-token replay using the same fixed-size tensor KV code in eager mode matched all 40 IDs against the compiled expert baseline. That isolates the observed failure to introducing whole-layer compilation on this prompt. The exact operation responsible remains unverified. MLX documents that compiled functions agree with ordinary execution up to numerical precision. [Eager diagnostic](eager_tensor_diagnostic.json) · [MLX compilation guide](https://github.com/ml-explore/mlx/blob/v0.32.2/docs/src/usage/compile.rst)

The inference regression suite passed 66 tests with one skip. Whole-layer compilation is opt-in for experimentation.
