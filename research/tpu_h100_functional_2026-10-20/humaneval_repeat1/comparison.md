# TPU versus H100: functional evaluation

Greedy FP32 generation, correct 8192-token tokenizer, maximum 256 new tokens in a 512-token context. Strict scores use function-boundary stops without source repair. Telos-cleaned scores are secondary.

| Model | Suite | Strict passed | Strict pass@1 | Telos-cleaned passed |
|---|---|---:|---:|---:|
| 100m_r50_modal_h100_seed42_cad1 | humaneval | 4/164 | 2.44% | 4/164 |
| 100m_r50_tpu_v5e8_seed42_cad1 | humaneval | 3/164 | 1.83% | 3/164 |
| 100m_r50_modal_h100_seed42_cad1 | humaneval (completed prior run) | 4/164 | 2.44% | 4/164 |

| Model | Probes | Top-1 | Top-5 | Mean CE |
|---|---|---:|---:|---:|

Private_unseen is the local 512-task generated suite, evaluated separately with base specifications and algorithmic hints. Its name does not establish disjointness from training data. Probes measure first-target-token prediction, not full solution correctness. Infill probes include the supplied suffix; causal probes see only the prefix.
