# Inkling 2 Mini pinned decode profile

Native Telos profiled the trained W=16 adapter on the M5 Pro in High Power mode at 3 GiB and 512-token context. Both isolated workers used the same three [benchmark prompts](../inkling2_mini_pinned_w16_2026-11-26/prompts.jsonl), eight resident expert slots per physical bank, and a 65-token cap. All 146 sampled IDs matched between workers and matched the previous pinned run. Swap used and cumulative swap-outs stayed flat. [Full model measurements](comparison.json)

The 134 decode steps between refreshes had zero packed bytes read and zero exposed SSD wait. The normal timing path sustained 9.31 tok/s between refreshes and 8.32 tok/s across all decode steps. Extra evaluation fences made the section-timing run slower at 7.89 tok/s between refreshes. The following breakdown describes the fenced run and includes MLX dispatch and synchronization time.

| Section | Mean ms per steady token | Share of fenced wall time |
| --- | ---: | ---: |
| Expert projection and accumulation | 65.84 | 51.9% |
| Backbone and routing through gate weights | 36.65 | 28.9% |
| Final state evaluation | 1.32 | 1.0% |
| Output projection and sampling | 3.06 | 2.4% |
| Other wall time | 19.94 | 15.7% |
| Total | 126.81 | 100% |

An isolated probe used the real packed weights from an eight-expert NVFP4 bank with synthetic BF16 hidden inputs and routing weights. All three paths produced exactly equal BF16 outputs on three seeded inputs. Each median comes from six 24-visit measurements with alternating order. [Bank probe](../../scripts/profile_inkling_expert_bank.py) · [Raw bank measurements](expert_bank.json)

| Projection and accumulation | Median time per bank visit |
| --- | ---: |
| Current packed path with row scatter | 1.523 ms |
| Packed path with direct single-row accumulation | 1.562 ms |
| Pre-expanded BF16 weights with direct accumulation | 0.839 ms |

Pre-expanding saved 45% per visit in isolation and costs 121.5 MiB of BF16 matrices per bank. Expanding all 20 NVFP4 banks would add about 2.37 GiB alongside the packed cache. Across the 44 NVFP4 visits, the current projection path can materialize 5.22 GiB of BF16 matrices per token. Those matrices are temporary and are reused by the allocator.

The measured candidates are retaining expanded matrices for a small number of repeated banks under the same memory guard and fusing weight expansion with projection while preserving the pre-multiply BF16 rounding. The previous fused projection changed greedy IDs, so an exact fixture check and a guarded end-to-end comparison are required for a different kernel. The direct single-row path showed no benefit in this probe.
