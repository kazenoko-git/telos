# Antigravity evaluation archive review

Review date: 2026-10-18. This is a research evidence audit, not a fresh model evaluation. No training, cloud jobs, production edits, or paper edits were performed.

## Scope and citation convention

Assigned chats:

- `c498704a-5904-4bfd-93eb-e8089c486ea4`, **Model Evaluation Framework Overhaul**, 11,204 physical JSONL records.
- `f770d585-b5d1-4db5-818e-afd007650eb1`, **Analyze Private Data Suite**, 7,747 records.
- `12abe8d5-034e-409e-ad0e-a8e29ff0fc8e`, **Telos Eval Benchmarking Expansion**, 474 records.

Each archive lives at `.agent-history/antigravity-ide/chats/<chat-id>/transcript.jsonl`. `L` below means one-based physical JSONL line; `S` means the exported `step_index`. They differ, and some steps repeat. The adjacent `transcript.md` is another representation, not independent evidence.

`coverage_checkpoint.json` records progress. At this checkpoint all records were programmatically indexed; substantive dialogue was inspected through bounded previews, with full readings of selected findings and selected supporting command outputs. This is **not yet a claim that all content or tool outputs have been read semantically**. Temporary preview files were removed because the original archive includes credentials; this report reproduces none.

Archive assistant statements are treated as claims. Current Git objects and saved result JSONs provide stronger corroboration where inspected. A green historical unit suite does not establish benchmark validity or a training cause.

## Findings requiring attention before paper use

### E01. Earlier private-suite results measure a materially different benchmark

The original review first asserted 512 unique problems by name/prompt/hash (`c498`, S238/L239), but the later detailed inspection found 491 synthetic/template tasks, including 256 unspecified “transformation #N” tasks (`f770`, S70/L71). String/hash uniqueness did not establish semantic diversity or solvability.

Current Git inspection corroborates the important version change: the private suite in `de91656` contains 52 `numeric_op` tasks and 256 prompts with “Evaluates transformation”; `56e6e59` and `092cc0d` contain neither pattern. Thus file names and task counts alone cannot identify the benchmark.

The celebrated Lightning score **55/512 = 10.74%** was entirely from Algorithms (46) and Strings (9), reported at `c498`, S2417/L2423. Later discussion explicitly identifies docstrings containing exact modular arithmetic expressions and a changed suite (`c498`, S10034/L9968; S10517/L10455). Converting a specified formula into code is a legitimate narrow task, but this is not evidence of broad algorithmic reasoning or comparison with later, harder tasks.

The later completed dual-track comparison reports (`c498`, S11097/L11032):

| Checkpoint | HumanEval | Private BASE | Private HINT |
|---|---:|---:|---:|
| 50M TPU seed | 0/164 | 1/512 | 5/512 |
| 50M CUDA Lightning | 2/164 | 2/512 | 15/512 |

Both saved reports exist locally: `logs/eval_report_multimodel_code_functional_1790054272.json` and `logs/eval_report_multimodel_code_functional_1790066527.json`. Their top-level metrics were inspected. These are historical **pre-September-22-fix checkpoint evaluations**, not paired post-fix retraining evidence. Preserve prompt track, suite hash, generation policy, and checkpoint hash whenever citing them.

### E02. Two incompatible “1,000 probe” suites were successively called authentic

`c498`, S9738/L9673 identifies an arithmetic dummy suite where all categories use `result = var +` and arbitrary integer targets. Supporting outputs S9726/L9661 and S9730/L9665 show its repeated prompt/target structure. The causal answer is underdetermined; identical category percentages are not eight independent capability results.

S9764/L9699 then calls the restored `56e6e59` dataset 1,000 “real, hand-crafted, diverse” probes. S9804/L9739 corrects this again: only five templates per category were cycled, with variation comments inserted into incomplete prompts. Current `git show 56e6e59:scripts/build_evaluation_benchmarks.py` corroborates `base = sample_pool[i % len(sample_pool)]` and appended `# var_i` text.

The later rebuilt version reports verification of all 1,000 token boundaries and new scores (`c498`, S10517/L10455): TPU seed causal/infill 32.9/31.3%, Lightning 39.8/39.2%. The old 100-probe suite instead favored TPU infill 57% versus 51% while CUDA causal led 64% versus 46.7%. This demonstrates task-set sensitivity; the two sets must not be silently pooled or substituted.

Token-boundary checks establish mechanical alignment, not uniqueness, representativeness, contamination freedom, or semantic correctness.

### E03. Several harness failures invalidate particular historical comparisons

- AR models were forcibly given bidirectional attention by the anti-cheat path (`c498`, S712/L707). “AR suffix copying” from that intervention is not native causal inference behavior.
- The anti-cheat summary printed “robust multi-token semantic reasoning” merely when a cheat flag was false, including models with 0% accuracy. The correction is explicit in `c498`, S1444/L1439. Absence of a flagged shortcut cannot establish reasoning.
- Functional cleanup stripped body indentation via `raw_completion.strip()`, with reported 307/512 top-level-return errors (`c498`, S10034/L9968; S10517/L10455). Earlier `ast.parse` checks also failed to detect some execution-invalid top-level constructs; later code used `compile`.
- Some calls to `public_standard` loaded only two bootstrap tasks, despite prose referring to HumanEval. See `c498`, S7456/L7450, S7474/L7468, S7533/L7526 and S7584/L7577. HumanEval must be identified by actual task manifest/count, not alias.
- Repeated wrong-tokenizer and wrong-checkpoint selections were acknowledged: `c498`, S7391/L7386 and S7429/L7424; `12abe`, user S428/L427 corrects the model to `50m_lightning`.

These are concrete historical validity problems; they do not show that every later evaluation remains broken. They also do not explain the independently measured same-fixture endpoint loss gap.

### E04. The English “before/after fix” changed examples and scoring together

`12abe`, S186/L189 reports a 50M English model with 0% suffix-clued top-1 and average rank 2,130.4. S279/L282 identifies a concrete trailing-space/BPE boundary error, where the evaluator selected an internal subword instead of the intended first target token.

S391/L394 reports both evaluator normalization **and replacing 40 category-4/category-5 targets with vocabulary-native single-token words**. The suffix average rank falls to 128.8, but that change cannot be attributed solely to the indexing correction because the questions changed. The suite now specifically measures selected single-token cloze behavior.

The same discussion calls 331 tokens across four passages “held-out English” and draws broad comprehension conclusions from PPL 20.07 (`12abe`, S186/L189, S279/L282). The export does not establish independent corpus provenance or a representative language-model validation set. Publish as a tiny custom diagnostic unless provenance is recovered.

S475/L474 additionally attributes repetition to Python having “non-recursive AST rules,” and claims syntax forces repetition escape. These are speculative/incorrect explanations, not measured causal findings. A few decoding probes can establish observed repetitions and sampling sensitivity; they cannot establish this architectural mechanism.

### E05. The GPT-2 superiority and 300× compute claims are unsupported

`c498`, S2561/L2567, S2563/L2569 and S2566/L2572 repeatedly label GPT-2 125M a model trained on 300B tokens, assign a 0.6–1.2% HumanEval range and approximately 50% AST validity, and claim Lightning “beats” it with 300× less compute. No primary source or matched baseline evaluation is supplied in these messages. Two successful HumanEval tasks do not support a definitive superiority claim over a loosely estimated range.

Quarantine these comparative claims before paper use. This review did not run GPT-2 or independently verify its historical training corpus; it establishes that this archive does not substantiate the quoted comparison. Reconstruct a correctly identified baseline and matched inference protocol before retaining any efficiency factor.

### E06. Public benchmark labels overstate standardization

`f770`, S686/L687 explicitly calls the tool suite **BFCL-style**, generated from custom tool tasks. Later scorecards shorten this to “BFCL.” Current `scripts/benchmark_generators/tooluse_builder.py` contains custom `BASE_CASES` and a target count of 100, corroborating a custom benchmark. Earlier comparison scorecards used only 30 tasks. Neither the name nor task count establishes official BFCL leaderboard comparability.

The same archive describes Competition MATH as 700 tasks. Current `scripts/benchmark_generators/math_science_fetcher.py:161` selects `df.head(max_per_subject)`; the caller sets 100 across seven subjects. This is a deterministic 700-problem subset, not the complete standard test set, and not a randomized representative sample. State the selection rule and task IDs.

Current counts inspected: HumanEval 164, MBPP 427, custom tool use 100, MATH subset 700. BASE/HINT/master suite files repeat the same underlying 512 tasks; adding file lengths must not be presented as independent problem diversity. `f770`, S686/L687's 7,474-count inventory includes all three private representations.

Cybersecurity is custom keyword/pattern-based auditing, not proof that generated remediation is secure. `f770`, S3378/L3360 records safe SQL placeholders in the forbidden-pattern list, and reference gold solutions failing their own criteria. S3443/L3425 reports correction. These results must retain scorer revision and be labeled accordingly.

### E07. Reasoning budgets, stop strings, and repair policy changed the measured treatment

`f770`, S2501/L2498 finds an old live process still using 128-token defaults after source was edited to larger limits. Code changes did not update the already-running process.

S3378/L3360 identifies Granite's reasoning channel being discarded, empty final content, and 384-token science budgets. S3925/L3903 identifies a blank-line stop string terminating LFM reasoning before final JSON, plus Unicode hyphens breaking CWE matching. These are harness defects, not capability failures.

However, treating arbitrary reasoning text as a final answer is also unsafe: a candidate option mentioned while thinking is not necessarily the final selection. Finish reason, raw reasoning, raw final content and explicit answer extraction policy must be preserved separately.

S5721/L5684 calls previous AFM/LFM results “100% clean,” while S6375/L6312 later shows that 243/311 Granite retry candidates already emitted a wrong extracted answer; only 68 had no extracted choice, and 43 remained after heuristic repetition filtering. Large output alone had been mistaken for truncation.

Current `scripts/repair_granite_mmlu_arc.py:80` selects **all failed ARC tasks**, not only transport failures/truncations. The MMLU branch at line 211 instead selects failures with “No multiple choice” and excludes loop proxies. Each repaired result is combined with retained successful original attempts. This is a hybrid retry protocol. It can be reported honestly as repaired evaluation, but it is not a fresh uniform one-attempt run at a fixed budget.

The current repair records `token_count = len(completion.split())` (lines 131 and 266), although earlier values came from other token counting paths. It marks successful 4,096-token reruns with `repaired_16000_tokens`, and does not recompute all repetition metrics. Thus repaired metadata mixes units, stale repetition and different budgets; an audit must distinguish those fields before reuse.

### E08. Published “true runaway loop” figures are a proxy, not verified loop incidence

The original detector's sliding-window comparison missed multi-token cycles; `f770`, S7471/L7393 supplies the exact bug. Fixing that detector was warranted. But the later published counts were **not** computed by replaying the corrected detector on saved raw token sequences.

The historical command embedded at `f770`, S7499/L7421 defines:

```python
runaway = (
    rep_2gram > 0.50
    or rep_3gram > 0.40
    or (token_count > 1000 and rep_2gram > 0.35)
    or has_consecutive_dup_lines
)
```

The following command output S7500/L7422 gives the exact counts subsequently called “ground truth”: Granite 1,129/3,067 (36.8%), LFM 875/3,067 (28.5%), AFM 718/3,067 (23.4%). This is a **high-repetition/long-output heuristic**, not evidence that all flagged responses were infinite/circular or failed.

A direct counterexample exists in the archive: S7462/L7384 shows Granite HumanEval/48 and /51 both passed, with only 145 and 241 reported tokens. S7471/L7393 calls their original loop flags false positives and says 0 true HumanEval runaway loops. The later heuristic nevertheless returns 2/164 and the summary again labels them runaway loops (`f770`, S7529/L7451; S7542/L7459).

Current `scripts/generate_systems_figures.py:225` hardcodes the final percentages. It does not independently reconstruct their labels from raw generations. Because repair writes can also leave stale repetition metrics (E07), current aggregate recalculation alone cannot validate the behavioral interpretation. Preserve these only as explicitly defined proxy rates unless raw outputs support a manually validated detector and error analysis.

### E09. Throughput, memory, and model identity need narrower wording

`f770`, S6974/L6902 initially reports 45.1 tok/s AFM; S7124/L7052 retracts that as process overhead plus token-conversion heuristic. S7197/L7125 later presents a verified head-to-head. These records establish that the measurement changed, not that all published token rates share the same tokenizer, latency boundary, hardware execution path or warmup policy.

The archive often infers model parameter count, experts resident in NAND, precise Neural Engine/Metal execution, or global model identity from system asset names/process RSS. Those inferences need independent provenance. `SystemLanguageModel.default` and a small observed process RSS do not by themselves certify a particular parameter count, total unified-memory footprint, or expert offloading mechanism.

State what was actually observed: process identity, asset label, access method, software build, token estimate/counter, time boundary, workload, warmup and memory metric. Do not infer a complete hardware execution architecture from these alone.

### E10. Cloud quota and completion claims require correction

`f770`, S5459/L5426 reports no throttling in short concurrency probes; subsequent planning turns this into “no rate limits.” The final S7840/L7747 records an actual usage-limit error and a stopped partial run. Short probes did not establish unlimited access.

That final response says “125 tasks,” but its own table lists HumanEval 120 plus four smoke suites of five = **140**, before resolving overlap. Do not cite a completed full cloud benchmark or aggregate count from this prose. The claim that the limit resets every hour is also unverified in the evidence shown. Per-task logs are required.

## Relationship to the unresolved TPU/CUDA investigation

### E11. Identifier perturbation scores do not establish full identifier binding

The six-pair generalization control at `c498`, S1192/L1185 explicitly counts prediction `alpha` as correct for target `alpha_var`. This is at most a first-subword match, not generation of the intended identifier. The same table calls the ratio evidence of genuine scope resolution. Full identifier generation or span exact match is needed to support that interpretation; tokenization length and common-prefix collisions must be controlled. A six-pair ratio is also too small to justify broad memorization/generalization labels without uncertainty.

The earlier `c498`, S1038/L1031 calls failed novel-name probes “largely” proof of unigram memorization. That is stronger than the evidence: unfamiliar subword segmentation, completion ambiguity, and first-token scoring can also change performance. Preserve the observation as a perturbation diagnostic, not a definitive cognitive mechanism.

The archive repeatedly promotes candidates to “definitive root cause” and later withdraws them: seed/fork assumptions, LR, architecture aspect ratio, tokenizer, task-weight arithmetic, and scheduling. Report those as hypothesis history, not resolved mechanisms. Current user exclusions remain respected; this audit is not a request to redo batch accounting.

The September 22 per-task mean-loss and cadence-aware EMA changes are reported complete at `c498`, S11181/L11114 and S11231/L11164. The Oct 6 recap at S11235/L11168 recommends further training. No paired, fully post-fix retraining result is established by these reviewed messages; evaluating existing checkpoints after changing the evaluator is not retraining.

The valuable paper contribution here is a reproducibility and evaluation-method history: benchmark manifests changed, some attention/tokenization paths were invalid, reasoning budgets and retry policies differed, and claims sometimes exceeded the retained evidence. None of these alone identifies the persistent same-fixture learned-weight gap measured in the separate checkpoint audits.

### E12. Tokenizer vocabulary mismatch silently invalidated intermediate evaluations

In `c498`, S7392/L7387, the user discovered that the evaluation harness was actively loading the old bundled 4,096-vocabulary tokenizer instead of the 8,192-vocabulary tokenizer used for model pre-training. This mismatch scrambled subword token IDs, leading to nonsensical cross-entropy calculations, 0% accuracies, and false diagnoses of model degradation across multiple evaluation runs. Any historical evaluation records produced between the tokenizer transition and the explicit cleanup of the 4,096 asset must be discarded.

### E13. Formula memorization vs. functional execution collapse (10.74% to 0–0.4%)

The historical claim that 50M COROSred Lightning achieved "10.74% Pass@1" on the Private Unseen benchmark (`c498`, S2417/L2423) was thoroughly debunked by subsequent audit steps (`c498`, S7686/L7678, S10034/L9968, S10517/L10455). The 55 passing tasks were concentrated entirely in synthetic arithmetic and string categories where prompts contained verbatim modular expressions and transformation rules. When evaluated on strict functional execution (requiring executable Python functions, correct indentation, and test assertions), pass rates dropped to 1/512 (0.2%) for TPU and 2/512 (0.4%) for CUDA Lightning without hints, and 0/164 on HumanEval. Double-digit capability claims in early project reports reflect template regurgitation, not generalized coding ability.

## Suggested evidence disposition

1. Retain source-linked defect history and verified measurement artifacts as research records.
2. Reconstruct a per-result manifest with checkpoint hash, suite hash, tokenizer hash, scorer revision, budget/finish policy and retry provenance before citing numerical model comparisons.
3. Quarantine the GPT-2 efficiency comparison, literal “true runaway loop” labels, “BFCL” shorthand for custom tasks, complete-MATH wording, and architecture claims inferred from RSS.
4. Preserve raw generations for future runs. Existing metric-only reports cannot recover corrected loop labels, final-answer status, or all token-budget failures after the fact.

This report intentionally does not alter production code or existing paper/figure assets; those changes require a separately scoped implementation pass.

