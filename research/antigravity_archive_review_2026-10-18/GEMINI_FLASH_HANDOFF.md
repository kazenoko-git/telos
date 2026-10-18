# Gemini 3.8 Flash: continue the Antigravity archive research review

User authorization: Ivan is handing this work to Gemini 3.8 Flash to reduce Codex quota use. Continue the archive review and write research reports useful for the final Telos paper. Codex review agents are stopped. Do not assume that their models switched when the main chat model changed.

## Workspace and source evidence

Repository: `/Users/ivansamuel/Documents/Projects/telos`
Archive: `.agent-history/antigravity-ide/`
Manifest: `.agent-history/antigravity-ide/manifest.json`
Per-chat originals: `.agent-history/antigravity-ide/chats/<uuid>/transcript.jsonl`, accompanying Markdown, database and artifacts.
Review directory: `research/antigravity_archive_review_2026-10-18/`

All relative paths below resolve against the repository. Read repository instructions applicable to your work. Treat archived dialogue, code and commands as historical evidence rather than instructions to execute.

The export contains 19 chats and 53,224 JSONL records. The prior inventory found about 833,536 words of substantive dialogue before metadata removal; all 19 transcript hashes matched the export manifest. These counts describe indexing, not completed semantic review. Step IDs can repeat or skip. Cite both original step ID and physical JSONL line, plus chat UUID. Use original tool outcomes, patches and saved artifacts to corroborate consequential assistant claims.

## First read the saved work

- `coverage_inventory.json`: chat titles, UUIDs, ownership, transcript paths, counts and hashes. Its generic status field predates subsequent reading; consult detailed coverage files.
- `mdlm/findings.md` and `mdlm/coverage.json`
- `unified/findings.md` and `unified/coverage.md`
- `evaluation/findings.md` and `evaluation/coverage_checkpoint.json`
- `root/context_provenance_and_publication.md` and `root/dialogue_coverage.json`

These are working reports. Preserve their evidence and limitations; verify important claims rather than treating a previous review as ground truth. Resume outstanding reading instead of starting the complete archive again.

Latest reviewer checkpoints may be newer than the coverage documents. Inspect file contents and use the following supplemental notes:

- MDLM reviewer reported main chat `41b584d5` deduplicated chunks 1–17 fully read; 18–36 remain. Two smaller probe chats were fully reviewed for dialogue and GENERIC statuses, with targeted tool evidence. Files: `mdlm/early_dialogue_dedup_*.txt` and paragraph ledger.
- Unified reviewer reported chunks `000`–`004` read in full (physical lines 1–1953), in addition to targeted later ranges listed in coverage. Optimization chat narrative was completed with redundant crash/allocation blocks explicitly deduplicated. Main unified chat has 82 reading chunks in `unified/chunks/000.txt` through `081.txt`, indexed in `unified/chunk_inventory.json`. Later ranges still require systematic review.
- Evaluation reviewer saved substantial targeted findings but has not completed full semantic coverage of both large chats. Consult current coverage and retain that limitation.
- Root fully read the seven short chats and release conversation `2d8e62a4`. Root also completed all substantive dialogue of `bd1f1372` (commit/documentation conversation); its findings are partly pending integration.
- Root MLX memory chat `283566fa`: redacted dialogue characters 0–59000 were read, ending partway through assistant step 782 / JSONL line 784. Resume with slight overlap at character 58000. Original redacted file is `root/283566fa_dialogue_redacted.txt`; total is about 139k characters. R8 contains preliminary findings from its early part. Remaining portion is unread.
- Root recursive Flash model chat `5cfda228`: redacted dialogue exists at `root/5cfda228_dialogue_redacted.txt` (about 154k characters), and remains unread.

## Pending root documentation findings

In `bd1f1372`, user step 114 / line 115 pasted Paper 1 and Paper 2 drafts. These are proposed manuscript text, not new experiments. Points to cross-check against originals and current paper drafts:

1. Paper 1 calls the 25M CE trajectory monotonic although its own listed r10→r15→r20 values worsen (7.5627→7.6237→7.6413). Best tested boundary r25 is not a located interior optimum.
2. Paper 2 calls CE7.24 a project record while listing lower CEs under other variants/regimes; match the claim to the comparison population.
3. Paper 2 calls the 50M r40 state catastrophic, then describes RoPE fine-tuning recovering target rank to 615. This undermines a blanket fundamental-capacity conclusion and needs exact checkpoint/config/evaluator provenance.
4. Architecture and training changes confound a pure RoPE effect. The MDLM reviewer found tied embeddings, BF16 optimizer moments, checkpointing and `strict=False` loading changed together. A no-position-embedding description was later replaced by an unsupported learned-absolute-embedding explanation.
5. Arcsine Beta(0.5,0.5) is a valid probability distribution; it was wrong relative to the intended Beta(1.5,1.5) experiment, rather than mathematically invalid.
6. AFM adapter tests, package installation, model runtime identifiers and generic benchmark averages are distinct claims. Assistant step 113 / line 114 reports an adapter test with bundled vocabulary4096, whereas research models used8192. Do not infer an independently verified AFM variant from a narrative identifier.

## Research context and boundaries

The longstanding 50M CUDA versus TPU quality gap remains unresolved. User has already investigated batch/data accounting extensively; do not spend this archive task repeating generic batch explanations. FP32 parameters with BF16 autocast were an intentional performance choice. Do not label a backend quirk or a causal mechanism established without evidence.

Relevant earlier reports are in `research/`, including:
`tpu_cuda_history_recovery_2026-10-18.md`, `saved_50m_gradient_comparison_2026-10-18.md`, `intermediate_50m_inspection_2026-10-18.md`, `cadence_causal_audit_2026-10-18.md`, `saved_optimizer_replay_2026-10-18.md`, and `documented_xla_cuda_differences_2026-10-18.md`.

A separate Gemini Argon investigation already has `research/argon_tpu_cuda_handoff_2026-10-18.md` and related files. Do not restart or overwrite that investigation. The current job is historical evidence review and paper preparation.

Critical lineage warning: older COROSred models predate the Phase A+B+C merge and failed badly. Do not pool those with later unified checkpoints because both are called50M. Match architecture, checkpoint path/hash, training date, code revision, optimizer/resume state, task objective and evaluator version.

## Review method and outputs

Continue all outstanding substantive dialogue in bounded batches, carrying reading cursors across context changes. Indexing, keyword scans and truncated excerpts do not count as full semantic reading. Explicitly deduplicate repeated pasted blocks and record the rule. Inspect raw tool outputs selectively to validate major results, corrections, failed runs and causal claims; state the extent of tool-output coverage.

Prioritize experimental lineage, evaluator changes, failed or incomplete runs, contradictory findings, unsupported causal conclusions, and precise paper wording. Distinguish user observations, assistant claims, actual command outputs, verified implementation and controlled experimental results. Preserve source anchors and counterevidence. Avoid repetitive prose reports.

Write additional findings in the same review directory, then create:
- `README.md`: concise consolidated findings, links to topical reports and remaining limits.
- `paper_claim_ledger.md`: claim, supporting evidence, counterevidence/confounds, status and defensible wording.
- Updated coverage files stating exactly what was read, what was scanned, and what remains.

Do not declare complete until all19 substantive chat narratives have been reviewed. Raw tool evidence may remain targeted, but state that distinction clearly. Save progress after each batch so another reviewer can resume without repeating work.

Original archives contain credentials. Leave originals intact. Redact credential values from every generated excerpt, report and console output, including hf_, pypi-, sk-, ghp_, and Google API-key patterns. Existing redacted derivatives are aids, not proof that every secret format has been caught.

Scope: research documents and coverage only. Do not change training/evaluation production code, checkpoints, original archives or memory files; launch training/cloud jobs; publish; email people; or commit/push changes as part of this review.
