# Frozen Qwen3-4B attention probe: measured results

Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`. Original BF16 weights, no training or FIM.

Validated 3136 score records from 112 examples.
Native/custom causal maximum logit difference: 0.03125.
Paired causal suffix-independence controls passed.

Numbers below are descriptive results for this fixed, small probe. The paired
examples use two templates; they are not a broad code benchmark. Local code
examples are not verified absent from Qwen pretraining.

## zero placeholder; shifted readout

| Attention | Paired token top-1 | Both top-1 (strict vs alternative) | Both prefer required operator | Local token top-1 | Local top-5 | Local NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| causal | 17/48 | 0/24 | 0/24 | 54/64 | 58/64 | 0.813 |
| full | 0/48 | 0/24 | 0/24 | 2/64 | 23/64 | 8.534 |
| future_bias_4 | 4/48 | 0/24 | 0/24 | 31/64 | 44/64 | 4.007 |
| future_bias_2 | 0/48 | 0/24 | 0/24 | 20/64 | 45/64 | 5.597 |
| last_1_full | 17/48 | 0/24 | 0/24 | 54/64 | 58/64 | 0.806 |
| last_4_full | 17/48 | 0/24 | 0/24 | 54/64 | 59/64 | 0.650 |
| query_only | 21/48 | 0/24 | 0/24 | 53/64 | 60/64 | 0.823 |

## zero placeholder; same_position readout

| Attention | Paired token top-1 | Both top-1 (strict vs alternative) | Both prefer required operator | Local token top-1 | Local top-5 | Local NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| causal | 0/48 | 0/24 | 0/24 | 22/64 | 41/64 | 3.846 |
| full | 0/48 | 0/24 | 0/24 | 2/64 | 8/64 | 10.986 |
| future_bias_4 | 4/48 | 0/24 | 0/24 | 13/64 | 26/64 | 7.309 |
| future_bias_2 | 0/48 | 0/24 | 0/24 | 4/64 | 15/64 | 10.818 |
| last_1_full | 0/48 | 0/24 | 0/24 | 22/64 | 41/64 | 3.675 |
| last_4_full | 0/48 | 0/24 | 0/24 | 23/64 | 42/64 | 3.226 |
| query_only | 0/48 | 0/24 | 1/24 | 9/64 | 37/64 | 4.412 |

## underscore placeholder; shifted readout

| Attention | Paired token top-1 | Both top-1 (strict vs alternative) | Both prefer required operator | Local token top-1 | Local top-5 | Local NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| causal | 17/48 | 0/24 | 0/24 | 54/64 | 58/64 | 0.813 |
| full | 0/48 | 0/24 | 0/24 | 2/64 | 23/64 | 8.534 |
| future_bias_4 | 4/48 | 0/24 | 0/24 | 31/64 | 44/64 | 4.007 |
| future_bias_2 | 0/48 | 0/24 | 0/24 | 20/64 | 45/64 | 5.597 |
| last_1_full | 17/48 | 0/24 | 0/24 | 54/64 | 58/64 | 0.806 |
| last_4_full | 17/48 | 0/24 | 0/24 | 54/64 | 59/64 | 0.650 |
| query_only | 21/48 | 0/24 | 0/24 | 53/64 | 60/64 | 0.823 |

## underscore placeholder; same_position readout

| Attention | Paired token top-1 | Both top-1 (strict vs alternative) | Both prefer required operator | Local token top-1 | Local top-5 | Local NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| causal | 0/48 | 0/24 | 0/24 | 27/64 | 37/64 | 6.061 |
| full | 0/48 | 0/24 | 0/24 | 2/64 | 10/64 | 11.685 |
| future_bias_4 | 8/48 | 0/24 | 0/24 | 17/64 | 29/64 | 7.018 |
| future_bias_2 | 0/48 | 0/24 | 0/24 | 4/64 | 18/64 | 10.857 |
| last_1_full | 0/48 | 0/24 | 0/24 | 25/64 | 37/64 | 5.692 |
| last_4_full | 0/48 | 0/24 | 0/24 | 25/64 | 37/64 | 5.225 |
| query_only | 0/48 | 0/24 | 1/24 | 13/64 | 35/64 | 5.896 |

## Interpretation limits

A changed prediction demonstrates sensitivity, not necessarily correct use of right context.
Both pair members favoring their respective required operator is a stronger diagnostic;
The table excludes apparent paired top-1 successes that tie with the alternative operator.
Recovering both as vocabulary-wide top-1 is stronger still. Neither proves a working
multi-token diffusion sampler, iterative convergence, calibration, or an inference speedup.
Any promising condition needs a new evaluation set before claiming generalization.
