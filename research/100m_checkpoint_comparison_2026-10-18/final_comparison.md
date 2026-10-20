# Completed100M checkpoint comparison: common losses and full HumanEval

All5 completed unified endpoints were freshly evaluated. The legacy one-step synthetic PhaseA checkpoint was excluded.

## Results

Lower cross-entropy is better. HumanEval uses164 official tasks, greedy decoding and one completion per task. The primary pass count below uses standard function-boundary stops, without AST back-trimming, indentation repair or function replacement.

| Checkpoint | Causal CE | Infill CE | Strict HumanEval | Pass@1 | Telos-cleaned passes |
|---|---:|---:|---:|---:|---:|
| 100m_r50_modal_h100_seed42_cad1 | 1.360218 | 0.589983 | 4/164 | 2.44% | 4/164 |
| 100m_5b | 1.726202 | 1.195184 | 1/164 | 0.61% | 1/164 |
| 100m_5b_alpha075 | 1.741622 | 1.295387 | 0/164 | 0.00% | 0/164 |
| 100m_schedstep1 | 1.730354 | 1.305765 | 0/164 | 0.00% | 0/164 |
| 100m_tpu_2 | 1.737497 | 1.210185 | 0/164 | 0.00% | 0/164 |

## Protocol and boundaries

- Common losses: CPU FP32;128 sequences of512 tokens across4 fixed corpus slices, with identical20% random masking on ordinary tokens. Causal attention and shifted next-token targets; bidirectional infilling scored only at masked positions. Exact fixture and per-sequence metrics are saved.
- HumanEval generation: native Telos greedy generator, PyTorch MPS FP32,8192-token configs/tokenizer_mac.json, maximum256 new tokens inside a512-token total context. The longest prompt receives a smaller completion allowance; no task was omitted.
- Official dataset: all164 local prompts/entrypoints match; local test harness ASTs match the official tests plus check(entry_point). Dataset and tokenizer hashes are saved.
- Execution: fresh isolated subprocess,5-second timeout, network sockets disabled,512MB address-space limit attempted. Standard-library executor canonical and deliberately incorrect controls behaved as expected. This is process isolation, not an OS security sandbox.
- Secondary score: the existing Telos evaluator applies its usual completion cleanup and AST back-trimming. Both raw completions and cleaned versions are preserved.
- All endpoints have matching declared geometry and strict state loading. Actual unique parameters105,995,521. Older recipes differ in alpha floor, causal ratio and/or cadence. Exact historical source/runtime identity is incomplete. This does not isolate the cause of a CUDA/TPU gap.
- Corpus slices exceed5B positions in the15B file; train/eval document disjointness and public-benchmark contamination are not established. Loss metrics are not asserted to be a held-out generalization benchmark.
- The new queued v5e-8 run is not part of this table.

## Passed HumanEval tasks

- 100m_r50_modal_h100_seed42_cad1: HumanEval/23 (strlen), HumanEval/28 (concatenate), HumanEval/35 (max_element), HumanEval/53 (add).
- 100m_5b: HumanEval/23 (strlen).
- 100m_5b_alpha075: none.
- 100m_schedstep1: none.
- 100m_tpu_2: none.

## Reliability head

- 100m_r50_modal_h100_seed42_cad1: common-fixture causal-correctness AUROC0.4957.
- 100m_5b: common-fixture causal-correctness AUROC0.4918.
- 100m_5b_alpha075: common-fixture causal-correctness AUROC0.4060.
- 100m_schedstep1: common-fixture causal-correctness AUROC0.4086.
- 100m_tpu_2: common-fixture causal-correctness AUROC0.4868.

## Artifacts

- ../results.json: common-fixture metrics, checkpoint hashes and embedded configurations.
- summary.json: full native HumanEval reports and generation protocol.
- strict_summary.json: primary stop-only outcomes for every generated completion.
- *.generations.jsonl: raw and cleaned text, task IDs, token budgets and timings.
- HumanEval.jsonl.gz and dataset_provenance.json: exact benchmark source.
