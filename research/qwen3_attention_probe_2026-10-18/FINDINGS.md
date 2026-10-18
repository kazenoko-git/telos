# Can frozen Qwen3-4B recover code tokens by changing attention alone?

**These interventions did not produce a useful bidirectional denoiser.** Full
bidirectional attention severely damaged recovery. More limited changes showed
some right-context sensitivity and occasional beneficial changes, but no robust
two-sided recovery on the counterfactual pairs.

## Fixed experiment

- Original BF16 Qwen/Qwen3-4B, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`, on this Mac's MLX backend.
- No training, adapters, FIM tokens, instruction prompts, reordering, candidate
  generation, or sequence reranking. The true target was removed before embedding.
- 24 counterfactual pairs (48 examples) and 64 local solution-token reconstructions.
- Seven attention settings, two placeholder controls, two output-position readouts.
- 3,136 recorded scores from 1,568 forward passes, approximately 142 seconds after
  model loading. Qwen processes exited after the experiment and numerical follow-up.

The original-order next-token-aligned readout is primary. The table shows its
zero-embedding placeholder condition. Literal-underscore results were identical
for this readout, as expected because masked-position keys were blocked in every
condition. These are controls, not independent extra examples.

| Attention | Local token top-1 | Local target NLL (lower is better) | Counterfactual pairs both recovered, excluding alternative-score ties |
| --- | ---: | ---: | ---: |
| Causal | 54/64 (84.4%) | 0.813 | 0/24 |
| Full bidirectional | 2/64 (3.1%) | 8.534 | 0/24 |
| Future attention penalty 4 | 31/64 (48.4%) | 4.007 | 0/24 |
| Future attention penalty 2 | 20/64 (31.3%) | 5.597 | 0/24 |
| Final layer bidirectional | 54/64 (84.4%) | 0.806 | 0/24 |
| Final four layers bidirectional | 54/64 (84.4%) | 0.650 | 0/24 |
| Readout queries bidirectional | 53/64 (82.8%) | 0.823 | 0/24 |

All conditions and both readout positions are in [full/RESULTS.md](full/RESULTS.md).

## What changed usefully, and what did not

The final-four-layer intervention corrected `counts` in local task 0063 but
broke a previously correct subtraction operator in task 0036. Its total accuracy
did not improve. Its lower target loss is an interesting descriptive signal,
not proof of broad repair ability; no shuffled-suffix control was run on these
local examples to isolate why their target probabilities improved.

Query-only attention corrected four local tokens and broke five. A change in
prediction alone is not evidence that the model understands the suffix.

Same-position readouts were poor. Query-only attention did make both suffixes
of one arithmetic pair favor their respective operators over the alternative,
including when the output head was recomputed in FP32. But the vocabulary-wide
prediction was the digit `3`, not either required operator. This is a weak
conditional-preference signal, not successful recovery.

## Numerical follow-up

Original BF16 query-only output appeared to recover both members of one pair.
One required operator was exactly tied with its alternative, and argmax chose
the required token by token-ID ordering. We reran all 48 paired examples with
the BF16 backbone unchanged and the final tied output projection in FP32.

All 96 original BF16 candidate margins reproduced exactly. On the tied case,
the FP32 margin for the required subtraction operator versus addition was
**-0.08698**: addition was actually preferred. There were zero pairs with both
top-1 answers correct in this follow-up. This was an exploratory precision
diagnostic selected after seeing the original run, not a fresh evaluation set.
It does not check every possible numerical issue in the BF16 attention backbone.

## Verification and limits

- All three weight SHA-256 digests match pinned Hugging Face LFS metadata.
- Custom causal forward reproduced the native causal argmax; maximum logit
  difference was 0.03125, within the preset 0.05 tolerance.
- Causal paired outputs were independent of suffix changes, as required.
- Pair token prefixes, lengths, hole positions, and differing answers were
  checked; both complete synthetic programs passed their generated assertions.
- All expected records are present with no duplicate case/condition keys.
- Full-run source hash matches the archived `full/probe_source.py`.

These pairs use only two templates. The local examples are not verified absent
from Qwen pretraining, and their strong causal baseline shows that many missing
tokens are already predictable from the prefix. The findings reject these
specific settings as a working conversion recipe; they do not establish that
every inference-only intervention is impossible. No iterative diffusion,
multi-hole recovery, confidence routing, or deployment speedup was demonstrated.

Files: `full/results.jsonl`, `full/analysis.json`, `full/runtime.json`,
`full/completion.json`, `full/fp32_head_check.json`, `weights_integrity.json`.
