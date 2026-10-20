# Completed anti-cheat, contextual probes and generation diagnostics

Both 100M-tier final checkpoints evaluated; CPU FP32 for native anti-cheat, MPS FP32 for previously completed probes and functional generation. Same 8192-token tokenizer and pinned local datasets.

## Anti-cheat: 1,000 probes per model

| Model | Requested K | Exact match | Token accuracy | Suffix boundary match | Prefix boundary match |
|---|---:|---:|---:|---:|---:|
| H100 | span_1 | 20.6% | 20.6% | 1.7% | 0.9% |
| H100 | span_2 | 20.7% | 22.3% | 2.2% | 1.0% |
| H100 | span_4 | 20.7% | 22.4% | 2.2% | 1.0% |
| H100 | span_8 | 20.7% | 22.4% | 2.2% | 1.0% |
| TPU | span_1 | 19.4% | 19.4% | 3.0% | 2.3% |
| TPU | span_2 | 19.4% | 21.05% | 3.0% | 2.4% |
| TPU | span_4 | 19.4% | 21.18% | 3.1% | 2.4% |
| TPU | span_8 | 19.4% | 21.18% | 3.1% | 2.4% |

Neither model triggers the native anti-cheat heuristic. This does not establish absence of shortcut learning. Boundary-token matches can be legitimate; no adversarial suffix intervention was performed.

**Span limitation:** 914/1,000 targets are one token; only86 have at least2 tokens,11 at least3, and2 at least4. None reaches8. Requested K=8 therefore does not constitute an eight-token reconstruction test. Actual-length strata are saved in each raw report. Native target tokenization differs from contextual probes, so the two accuracy columns are not directly interchangeable.

## Contextual probes

| Model | Mode | Count | Top-1 | Top-5 | Mean CE |
|---|---|---:|---:|---:|---:|
| H100 | causal | 864 | 43.4% | 60.65% | 3.41 |
| H100 | infill | 1000 | 55.1% | 73.3% | 2.36 |
| TPU | causal | 864 | 43.52% | 60.42% | 3.42 |
| TPU | infill | 1000 | 53.4% | 73.8% | 2.35 |

## Generation diagnostics

Computed from existing full-suite runs, with no new generation. Token lengths, n-gram repetition and loop flags use raw continuations; line repetition, numeric metrics and AST validity use native-cleaned completions.

| Model | Suite | Avg tokens | 3-gram repetition | Line repetition | Loop heuristic flagged | Numeric recall | Numeric precision |
|---|---|---:|---:|---:|---:|---:|---:|
| H100 | humaneval | 109.1 | 35.0% | 23.3% | 69.5% | 27.4% | 38.8% |
| H100 | private_unseen_base | 108.0 | 30.7% | 20.3% | 62.9% | 20.1% | 25.2% |
| H100 | private_unseen_hint | 135.8 | 39.2% | 26.2% | 57.4% | 31.8% | 39.4% |
| TPU | humaneval | 73.9 | 23.1% | 13.6% | 74.4% | 21.2% | 32.2% |
| TPU | private_unseen_base | 97.1 | 26.8% | 15.8% | 74.6% | 23.3% | 27.9% |
| TPU | private_unseen_hint | 118.6 | 32.8% | 19.3% | 72.9% | 33.7% | 41.8% |

The loop heuristic flags any three consecutive repetitions of a token cycle of length1–32; ordinary repeated syntax/whitespace may trigger it. It is not a measured rate of semantically degenerate programs. Numeric metrics compare prompt literals with emitted literals; they do not establish mathematical correctness.

## Functional code: completed

| Suite | H100 strict passes | TPU strict passes |
|---|---:|---:|
| HumanEval164 | 4 | 3 |
| Private base512 | 10 | 10 |
| Private hint512 | 20 | 22 |

Both full HumanEval reruns reproduce all164 raw completions exactly for their respective checkpoint. H100 native-cleaned hint scoring passes21 versus20 under strict scoring; all other displayed pass counts agree between scoring methods.

These measurements describe the two checkpoints. They do not identify which change resolved the historical training gap. Raw results, source hashes, actual-span strata and individual probe records are retained beside this report.
