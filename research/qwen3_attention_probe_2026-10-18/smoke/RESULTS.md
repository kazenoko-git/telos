# Frozen Qwen3-4B attention probe: measured results

Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`. Original BF16 weights, no training or FIM.

Validated 112 score records from 4 examples.
Native/custom causal maximum logit difference: 0.03125.
Paired causal suffix-independence controls passed.

Numbers below are descriptive results for this fixed, small probe. The paired
examples use two templates; they are not a broad code benchmark. Local code
examples are not verified absent from Qwen pretraining.

## zero placeholder; shifted readout

| Attention | Paired token top-1 | Both top-1 | Both prefer required operator | Local token top-1 | Local top-5 | Local NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| causal | 2/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| full | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| future_bias_4 | 2/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| future_bias_2 | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| last_1_full | 2/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| last_4_full | 2/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| query_only | 2/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |

## zero placeholder; same_position readout

| Attention | Paired token top-1 | Both top-1 | Both prefer required operator | Local token top-1 | Local top-5 | Local NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| causal | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| full | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| future_bias_4 | 2/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| future_bias_2 | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| last_1_full | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| last_4_full | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| query_only | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |

## underscore placeholder; shifted readout

| Attention | Paired token top-1 | Both top-1 | Both prefer required operator | Local token top-1 | Local top-5 | Local NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| causal | 2/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| full | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| future_bias_4 | 2/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| future_bias_2 | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| last_1_full | 2/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| last_4_full | 2/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| query_only | 2/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |

## underscore placeholder; same_position readout

| Attention | Paired token top-1 | Both top-1 | Both prefer required operator | Local token top-1 | Local top-5 | Local NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| causal | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| full | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| future_bias_4 | 2/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| future_bias_2 | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| last_1_full | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| last_4_full | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |
| query_only | 0/4 | 0/2 | 0/2 | 0/0 | 0/0 | n/a |

## Interpretation limits

A changed prediction demonstrates sensitivity, not necessarily correct use of right context.
Both pair members favoring their respective required operator is a stronger diagnostic;
recovering both as vocabulary-wide top-1 is stronger still. Neither proves a working
multi-token diffusion sampler, iterative convergence, calibration, or an inference speedup.
Any promising condition needs a new evaluation set before claiming generalization.
