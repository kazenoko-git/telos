# First guarded Inkling 2 Mini SSD-only run

Native Telos at `6063529e` used the pinned Inkling 2 Mini MLX checkpoint on the M5 Pro under AC High Power mode with MLX 0.32.2. Each probe ran in a fresh process, with a 2 GiB admission budget and a 512-token context. The loader was required to select SSD expert storage before reading any weights. It admitted a 250.86 MiB expert cache and loaded about 958 MiB of active MLX text weights. Both probes decoded greedily for 16 tokens. No MLX-LM or resident model was used.

| Measurement | Chat prompt | Answer channel opened |
| --- | ---: | ---: |
| Prompt tokens | 26 | 28 |
| Checkpoint load | 0.87 s | 0.93 s |
| Time to first sampled token | 8.41 s | 7.92 s |
| Time to first decoded text | 10.11 s | 7.92 s |
| Decode speed | 1.34 tok/s | 1.25 tok/s |
| SSD reads for prefill | 11.17 GiB | 9.67 GiB |
| SSD reads during decode | 15.17 GiB | 14.93 GiB |
| SSD reads for full generation | 26.34 GiB | 24.60 GiB |
| Peak process RSS | 1,603 MiB | 1,389 MiB |
| Peak MLX memory during generation | 1,287 MiB | 1,287 MiB |

Both probes asked for a short explanation of why ice floats. The first used the checkpoint's chat framing with thinking effort 0 and ended at the token limit while still emitting thinking text. The second added the model's answer-channel prefix to check answer quality within the same generation limit. It produced “Ice floats because it is less dense than liquid water, a result of the” before the limit. The core explanation was correct and the sentence was incomplete. These two probes are a functionality and memory check for one prompt.

Telos recorded exactly 1,839,368,960 expert bytes on every post-prefill decode step in both probes, or 1.71 GiB per token. This equals the full packed payload of all eight selected experts at each of the 48 visits. The 250.86 MiB cache avoided no expert reads during the 30 observed decode steps. The measured whole-drive read volumes were approximately 1.0 GiB per decode token.

Whole-drive SSD counters include background activity. The generation window excludes checkpoint load. The decode window and sustained speed measurement start at the disk sample just after the first model token, excluding the sample's timing overhead. The first sampled token can be a chat marker.

The watchdog checked used swap, cumulative swap-outs, worker RSS and system memory headroom about every 100 ms. It would stop at the first measured swap increase, and enforced a 2,048 MiB RSS ceiling, 45% preflight and 35% runtime free-memory floors, and a 768 MiB free-swap floor. All 364 worker samples had 3,773.06 MiB swap in use and 37,323,500 cumulative swap-out pages. Reported free memory stayed at or above 39%. Neither probe tripped a guard.

[Raw runs and guard records](runs.json)
