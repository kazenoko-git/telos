# Frozen Qwen3-4B right-context probe

Status: **completed after the user authorized inference.** See
[FINDINGS.md](FINDINGS.md) for conclusions and [full/RESULTS.md](full/RESULTS.md)
for every condition. All three original BF16 weight shards were downloaded and
their SHA-256 digests matched the pinned Hugging Face LFS metadata. The download
cache is under `scratch/qwen_attention_probe/hf_cache/` (ignored by Git).
No Qwen process remains running. Bonsai's process was not changed.

## Question and scope

Can changing attention at inference make frozen Qwen3-4B use right context to
recover a missing code token? This is a one-pass capability probe, not a
demonstration of a diffusion sampler, COROSred conversion, or a speedup.

Use original BF16 Qwen/Qwen3-4B at revision
`1cfa9a7208912126459214e8b04321603b3df60c`. No training, adapters, instruction
prompts, FIM markers, input rearrangement, candidate generation, or sequence
likelihood reranking. Token positions and RoPE stay unchanged. No KV cache.
Two candidate logits are compared only as an evaluation diagnostic.

## Prepared cases

- 24 counterfactual pairs / 48 examples: 12 arithmetic-operator pairs and 12
  comparison-operator pairs. Pair members have identical token prefixes and
  hole positions, different required answers, equal total token lengths, and
  different suffix assertions. The missing operator is not explicitly repeated
  in the suffix. Both complete programs were executed to check their assertions.
- 64 local code reconstructions sampled deterministically from the repository's
  `private_unseen_suite.json`, targeting one solution token per task. This is a
  local reconstruction sample, **not verified exclusion from Qwen pretraining**.
  Corpus code was not executed. These examples are secondary to the paired
  intervention, since prefixes can already strongly suggest their answers.

The raw original text and token IDs are saved in `cases.json`. Original target
IDs are evaluation labels only. The runtime replaces the target before
embedding. The complete programs are never supplied to the model.

## Attention and placeholder controls

Each example uses zero-embedding and literal-underscore placeholders separately.
No fresh or FIM-specific mask token is introduced. All conditions block the
unknown position as an attention key, including the causal control, so the
placeholder cannot spread through attention; its local residual is retained in
the underscore diagnostic. No condition contains the true target embedding at
the missing position.

Seven fixed conditions, with no selection using results:

| Condition | Future context access |
| --- | --- |
| causal | None |
| full | Every layer, all valid positions |
| future_bias_4 | Every layer, future attention logits penalized by 4 |
| future_bias_2 | Every layer, future attention logits penalized by 2 |
| last_1_full | Only final layer is bidirectional |
| last_4_full | Only final four layers are bidirectional |
| query_only | Every layer, only the two readout rows may see future positions |

Primary readout: the logit at the position **before** the hole, preserving Qwen's
next-token alignment. Secondary readout: the hole's own logit, testing a
same-position reconstruction hypothesis. They are reported separately.

Before scoring, the run checks that the custom causal path reproduces native
Qwen's shifted readout within the configured numerical tolerance and with the
same argmax. This numerical check **passed**, with maximum logit difference
0.03125. Pure CPU mask checks and paired causal suffix-independence checks also
passed. A failing control would invalidate the comparison.

## Evidence to look for

Report vocabulary-wide top-1/top-5 recovery and target negative log probability.
For each counterfactual pair also report:

1. Both required tokens recovered as vocabulary-wide top-1 predictions.
2. Both suffixes favor their required operator over the alternative operator.
3. Whether the top-1 prediction changes when the suffix changes.

Item 3 alone is sensitivity, not useful understanding. A useful signal needs
correctly directed changes (items 1/2), improvement over causal attention, and
consistency across examples and placeholder controls. An identical-prefix
causal model cannot correctly select both different pair targets.

All masks, readouts, and groups must remain visible in the report. Do not pick
the best condition and present its in-sample score as a held-out finding. If a
condition looks promising, create fresh counterfactual families for a follow-up.
Poor results would reject these particular interventions, not prove every
possible inference-only conversion impossible. Good single-token results would
still leave multi-hole reconstruction and iterative stability untested.

## Reproduction commands

An isolated temporary environment was installed; the project's `.venv` was not
modified. If `/tmp/telos-qwen-attention-probe` is gone, recreate it:

```sh
cd /Users/ivansamuel/Documents/Projects/telos
uv venv /tmp/telos-qwen-attention-probe --python .venv/bin/python
uv pip install --python /tmp/telos-qwen-attention-probe/bin/python -r research/qwen3_attention_probe_2026-10-18/requirements.txt
```

Preparation only (default; does not import MLX or load weights):

```sh
/tmp/telos-qwen-attention-probe/bin/python scripts/probe_qwen3_attention.py
```

The implementation check was run as follows. For a repeat, choose a fresh output
directory; the script refuses to overwrite results. The download flag fetches
only pinned files missing from the local cache:

```sh
/tmp/telos-qwen-attention-probe/bin/python scripts/probe_qwen3_attention.py --run --download-weights --max-cases 4 --output research/qwen3_attention_probe_2026-10-18/smoke
```

The completed full experiment command was:

```sh
/tmp/telos-qwen-attention-probe/bin/python scripts/probe_qwen3_attention.py --run --output research/qwen3_attention_probe_2026-10-18/full
```

The full design used 1,568 single-example forward passes (112 examples, two
placeholders, seven masks); each pass produced both readouts. Time from model
load through scoring was approximately 142 seconds (excluding download and
initial model loading). Results streamed to JSONL and summaries updated after
each case. Existing results are protected against overwrite. No automatic run
or monitor is set up.

Generate the validated tables:

```sh
/tmp/telos-qwen-attention-probe/bin/python scripts/analyze_qwen3_attention_probe.py research/qwen3_attention_probe_2026-10-18/full
```

An exploratory numerical follow-up reran all 48 paired examples in query-only
mode, preserving the BF16 backbone but recomputing the output head in FP32. Its
96 BF16 candidate margins reproduced exactly. The apparent paired top-1 success
was a rounding tie; the FP32 output head preferred the wrong operator. This
follow-up was selected after observing the original results and is explicitly
not an independent held-out experiment. See `full/fp32_head_check.json` and
`scripts/check_qwen3_attention_precision.py`.

Model architecture reference:
https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py
Installed backend: MLX / MLX-LM 0.32.2. Runtime compatibility was verified. Two
harness issues were corrected before scoring: frozen parameters must be checked
as flattened arrays rather than nested containers, and attention masks must
match BF16 activations. Failed startup logs were preserved. The actual full-run
source is archived as `full/probe_source.py` and matches its runtime source hash.
