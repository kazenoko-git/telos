# Saved 50M CUDA/TPU checkpoint comparison — 18 October 2026

Actual CPU measurements of three saved checkpoints at step 12,716. This compares learned endpoints under a common evaluator; it is not an accelerator execution test or a causal explanation of training divergence.

## Evaluation contract

- Same 32 sequences across two disjoint 16-sequence slices; 512 tokens per sequence, 16,384 input tokens total.
- Local `data/python_corpus_15b.bin`, explicitly interpreted as uint16. Slice offsets: 6,000,000,000 and 7,000,000,000 tokens. These offsets do not establish document-level absence from training.
- Identical seed-42 corruption mask across checkpoints and slices: probability 0.20, first position preserved. Only masked positions contribute to infill loss.
- Explicit causal attention for next-token prediction; explicit bidirectional attention for masked infill.
- CPU FP32 evaluation, dropout and gradient checkpointing disabled. No GPU allocation, optimizer update, or training performed.
- Every task gradient is the per-token mean over the complete slice, accumulated through four microbatches. This avoids averaging unequal masked-token counts as if equal.
- Gradient geometry uses the same normalized coefficients `alpha=0.50`, `beta=0.35`. It describes the corrected per-task-mean objective, not the historical loss implementation. Reliability-head loss is excluded.
- Gradient cosine is causal versus infill **within each model**, not a cross-model comparison of potentially permuted parameter coordinates.

## Loss and accuracy

Losses and accuracies below are token-weighted over both slices.

| Saved checkpoint | Causal CE | Causal Top-1 | Infill CE | Infill Top-1 |
|---|---:|---:|---:|---:|
| CUDA Lightning | 1.5647 | 67.01% | 1.0862 | 77.05% |
| TPU baseline | 2.0696 | 60.68% | 1.7863 | 67.55% |
| TPU schedule-step-1 | 2.0445 | 61.14% | 1.7120 | 68.20% |

CUDA has lower causal and infill CE in every one of the eight paired microbatches for both TPU comparisons. TPU schedule-step-1 improves on baseline in the pooled metrics but retains a substantial deficit. These are new slices, so values should not be substituted into the older validation trajectory.

## Gradient geometry

Ranges below retain the two separate slice measurements; they are not gradients formed by averaging the two cosine values.

| Checkpoint | Causal gradient norm | Infill gradient norm | Causal–infill cosine | Weighted causal/infill norm ratio |
|---|---:|---:|---:|---:|
| CUDA Lightning | 0.874–0.947 | 2.897–2.955 | 0.072–0.099 | 0.423–0.467 |
| TPU baseline | 1.095–1.122 | 3.089–3.322 | 0.143–0.172 | 0.483–0.506 |
| TPU schedule-step-1 | 0.989–1.006 | 2.845–3.203 | 0.125–0.132 | 0.441–0.505 |

Overall task gradients have positive cosine in every checkpoint on both slices. No measured layer group in either TPU checkpoint has negative task cosine. CUDA has one slightly negative layer on the second slice (`layers.1`, cosine -0.01255). The measurements do not show a TPU-specific endpoint task-conflict pattern.

Under the declared weights, the infill gradient contribution has larger norm than the causal contribution in all three models. This also occurs in CUDA. Gradient norm is not a percentage of learning: direction, sampling noise, clipping and AdamW state matter. The masked task has fewer target positions, which also affects gradient-estimation noise.

The TPU backbone gradients are finite and nonzero, with somewhat larger aggregate causal gradient norms than CUDA on these slices. This does not demonstrate what gradient or parameter-update scales occurred during the original training.

## What this changes

1. The saved-model quality deficit is reproduced with correct explicit attention modes and identical evaluation inputs.
2. The interrupted historical audit did not provide evidence that TPU infill gradients were fighting causal gradients. The corrected endpoint measurements show positive overall alignment.
3. Larger infill-gradient magnitude is shared across these endpoints under the declared objective; it is not sufficient to explain the backend difference.
4. Training-time task conflict, BF16 arithmetic differences, clipping/reduction behavior and AdamW updates remain unmeasured here. The original training cause remains unresolved.

## Reproduction and artifacts

```bash
.venv/bin/python scripts/compare_saved_50m_gradients.py --output logs/saved_50m_gradient_comparison_20261018.json
.venv/bin/python scripts/compare_saved_50m_gradients.py --offset-tokens 7000000000 --output logs/saved_50m_gradient_comparison_20261018_slice2.json
```

Each JSON includes checkpoint hashes, embedded configs, exact input/mask hashes, source hashes, per-microbatch metrics and per-layer geometry. The corresponding `.fixture.pt` files store the exact tokens and masks.

- `logs/saved_50m_gradient_comparison_20261018.json`
- `logs/saved_50m_gradient_comparison_20261018_slice2.json`
- `logs/saved_50m_gradient_comparison_20261018_summary.json`

Validation: four regression tests passed, covering attention-mode selection, full-fixture versus microbatch gradient equivalence for both objectives, and detection of opposing task gradients. All six actual model/slice evaluations completed successfully.
