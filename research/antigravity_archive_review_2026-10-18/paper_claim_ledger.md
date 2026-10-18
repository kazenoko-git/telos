# Paper Claim Ledger: Antigravity Archive Review

Date: 2026-10-18  
Scope: Comprehensive claim-by-claim audit of all historical research assertions, paper drafts (`paper_1_token_budget_scaling.md`, `paper_2_capability_divergence_and_rope.md`), benchmark reports, and assistant diagnoses across the 19 archived Antigravity conversations.

Each entry classifies the claim, supporting conversational or artifact evidence, counterevidence or confounding factors uncovered in the code and transcripts, current evidential status, and defensible wording for publication.

---

## 1. 25M MDLM Cross-Entropy Trajectory Monotonicity

- **Paper Draft / Location:** `research/paper_1_token_budget_scaling.md`, Section 7; `bd1f1372`, line 692.
- **Original Claim:** The 25M MDLM model improved monotonically with token budget up to the highest tested ratio of 1:25 (~625M tokens, CE $\approx 7.4361$).
- **Supporting Evidence:** Reported table in Paper 1 showing 1:1 (7.8068) and 1:25 (7.4361).
- **Counterevidence & Confounds:**
  - The paper's own table lists non-monotonic intermediate values: $r_{10}=7.5627 \to r_{15}=7.6237 \to r_{20}=7.6413$ before dropping to $r_{25}=7.4361$.
  - Early 25M runs suffered from an "Upscaling Initialization Shock" caused by Net2Net zero-padding with unadjusted RMSNorm variance scaling (`41b584d5`, Chunk 25, L7715/S7765).
  - $r_{25}$ is merely the largest tested budget in that series, not a located interior minimum.
- **Status:** **Refuted as monotonic; unconfirmed as optimum.**
- **Defensible Publication Wording:**
  > "In the 25M parameter sweep, evaluation cross-entropy at the largest tested budget ($r_{25}$, 625M tokens) reached 7.4361 nats, compared to 7.8068 nats at $r_1$. However, intermediate ratios exhibited non-monotonic variance ($r_{10} \to r_{20}$ range: 7.56–7.64 nats). Because $r_{25}$ represents the boundary of the evaluated grid, it should be characterized as the best observed endpoint rather than an identified interior optimum."

---

## 2. Parameter-Dependent Optimal Token Ratio Scaling Laws ($R^*(N)$)

- **Paper Draft / Location:** `research/paper_1_token_budget_scaling.md`, Section 9.3; `bd1f1372`, lines 805–883.
- **Original Claim:** The optimal token-to-parameter ratio $R^*(N)$ follows a predictive scaling relationship (fit via logarithmic, square-root, or power-law forms) predicting $R^*(50\text{M}) > 1:35$.
- **Supporting Evidence:** Two evaluated data points: 12.5M observed minimum at 1:15 (~187.5M tokens) and 25M lowest tested value at 1:25 (~625M tokens).
- **Counterevidence & Confounds:**
  - Two points cannot mathematically distinguish between logarithmic, power-law, or linear scaling forms.
  - The 25M $r_{25}$ point is a censored boundary endpoint, not an established optimum.
  - Model configurations varied depth, width, and attention heads simultaneously ($d=256, L=13$ for 12.5M vs. $d=512, L=8$ for 25M), confounding parameter scaling with aspect-ratio variations.
  - Net2Net initialization was used for 25M, whereas 12.5M was trained from scratch.
- **Status:** **Exploratory hypothesis only; mathematically underdetermined.**
- **Defensible Publication Wording:**
  > "Evaluating 12.5M and 25M parameter tiers indicates that the token-to-parameter budget associated with lowest probe cross-entropy was higher for 25M ($r_{25}$) than for 12.5M ($r_{15}$). However, with only two completed tiers and censored boundary evaluations, candidate functional forms ($R^*(N)$) remain exploratory hypotheses rather than empirically validated scaling laws."

---

## 3. "Project Record Cross-Entropy" 7.24 nats at 25M 1:40

- **Paper Draft / Location:** `research/paper_2_capability_divergence_and_rope.md`, Section 3; `bd1f1372`, line 323.
- **Original Claim:** 25M 1:40 achieved a "New Project Record Cross-Entropy" of 7.24 nats, outperforming all other runs in the project.
- **Supporting Evidence:** Reported contextual probe evaluation in Section 3 of Paper 2 draft.
- **Counterevidence & Confounds:**
  - Section 2 of the exact same draft explicitly reports lower cross-entropies: 25M 1:30 at 6.90 nats and 50M 1:35 at 7.05 nats.
  - Calling 7.24 an unqualified project record directly contradicts the paper's own summary table.
- **Status:** **Factually contradicted by draft's own internal data.**
- **Defensible Publication Wording:**
  > "Within the native RoPE 25M series, the 1:40 configuration reached 7.24 nats. Lower cross-entropies were observed under earlier non-RoPE regimes (notably 6.90 nats for 25M 1:30 and 7.05 nats for 50M 1:35); comparisons across these series must account for differences in positional encoding, tokenizer revisions, and initialization."

---

## 4. CE-Rank Divergence as a Fundamental Masked Diffusion Capacity Ceiling

- **Paper Draft / Location:** `research/paper_2_capability_divergence_and_rope.md`, Abstract and Section 4; `bd1f1372`, lines 344–399.
- **Original Claim:** Extended overtraining causes aggregate CE to continue improving while average target rank degrades (CE-Rank divergence), revealing a fundamental representational capacity ceiling in discrete masked diffusion.
- **Supporting Evidence:** Category-level probe breakdown showing literal/operator rank improvements alongside function/attribute rank degradation.
- **Counterevidence & Confounds:**
  - The 25M 1:30 $\to$ 1:40 transition was confounded by the switch from absolute PE to native RoPE (`bd1f1372`, line 301).
  - Single-step cloze probes test unigram prediction under high ground-truth ambiguity (e.g. multiple valid function names), which artificially inflates target rank.
  - The 50M 1:40 "collapse" (rank 3,988) was recovered to rank 615 via RoPE fine-tuning (`bd1f1372`, line 393), indicating training instability rather than an immutable representational ceiling.
- **Status:** **Descriptive category trend confirmed; causal capacity ceiling unproven.**
- **Defensible Publication Wording:**
  > "Under extended token-to-parameter ratios, single-step contextual probes show diverging trajectories: syntactic tokens (operators, keywords, literals) continue to gain probability mass while open-vocabulary semantic tokens (function names, attributes) exhibit increased rank variance. While this highlights challenges in resolving semantic ambiguity under fixed masking budgets, simultaneous modifications in positional encoding and learning rate prevent attributing this solely to an inherent capacity ceiling of discrete diffusion."

---

## 5. RoPE Fine-Tuning Regression (~0.6 nats)

- **Paper Draft / Location:** `research/paper_2_capability_divergence_and_rope.md`, Section 5; `bd1f1372`, lines 400–424.
- **Original Claim:** Attempting to retrofit RoPE via 150-step fine-tuning on converged absolute-PE models systematically regresses cross-entropy by ~0.6 nats due to violent rotational coordinate shock on attention matrices.
- **Supporting Evidence:** Reported delta tables showing +0.60 to +0.67 nat degradation across 12.5M, 25M, and 50M base checkpoints.
- **Counterevidence & Confounds:**
  - The RoPE retrofit patch (`41b584d5`, Chunk 10, L4592/S4603) simultaneously introduced:
    1. Weight tying (discarding separate `head.weight` and projecting from `emb.weight`),
    2. Gradient checkpointing,
    3. BF16 Adam optimizer moments,
    4. `strict=False` state dict loading.
  - Earlier MLX models had no positional encoding (permutation-equivariant) rather than learned absolute PE (`41b584d5`, L4504/S4515).
- **Status:** **Empirical regression confirmed; causal attribution to rotary geometry alone refuted.**
- **Defensible Publication Wording:**
  > "Applying a 150-step fine-tuning recipe that introduced RoPE alongside input/output weight tying, BF16 optimizer state storage, and activation checkpointing resulted in a ~0.60–0.67 nat regression across evaluated checkpoints. Because architectural tying and optimizer precision changed concurrently with positional encoding, this performance delta reflects compound adaptation effects rather than an isolated rotary coordinate shock."

---

## 6. Timestep Sampler Mathematical Validity

- **Paper Draft / Location:** `research/paper_1_token_budget_scaling.md`, Section 6.1; `bd1f1372`, lines 580–628.
- **Original Claim:** Historical cosine sampling $t = 0.5 - 0.5\cos(\pi u)$ accidentally produced an "invalid Beta(0.5, 0.5) arcsine distribution", which was corrected to Beta(1.5, 1.5).
- **Supporting Evidence:** Analytical density derivation of the inverse-cosine transform yielding $\frac{1}{\pi\sqrt{t(1-t)}}$.
- **Counterevidence & Confounds:**
  - Beta(0.5, 0.5) is a mathematically valid, well-behaved continuous probability distribution (the standard arcsine distribution on $(0, 1)$).
  - It was mismatched with the experimental objective (which sought to sample intermediate noise levels $t \approx 0.5$ rather than boundary noise $t \to 0, 1$), but was not mathematically invalid.
- **Status:** **Distribution derivation correct; "mathematically invalid" label false.**
- **Defensible Publication Wording:**
  > "The initial cosine-based timestep transform produced an arcsine Beta(0.5, 0.5) distribution that concentrated sampling density heavily near the boundaries ($t \to 0$ and $t \to 1$). To allocate training budget evenly across intermediate masking rates, the pipeline was updated to sample from Beta(1.5, 1.5), which places maximum density around $t \approx 0.5$."

---

## 7. 50M COROSred Lightning 10.74% Pass@1 on Private Unseen Benchmark

- **Archive / Chat:** `c498704a-5904-4bfd-93eb-e8089c486ea4`, S2417/L2423; `benchmarks/README.md`.
- **Original Claim:** 50M COROSred Lightning achieved 10.74% (55/512) Pass@1 on the held-out private code benchmark.
- **Supporting Evidence:** Benchmark scorecard generated in early evaluation sessions.
- **Counterevidence & Confounds:**
  - The 55 passing solutions were restricted entirely to synthetic arithmetic and string categories where task docstrings provided explicit modular formulas and transformation rules (`c498`, S7686/L7678).
  - Evaluator body indentation stripping (`raw_completion.strip()`) caused 307/512 false execution errors (`c498`, S10034/L9968).
  - When re-evaluated under strict functional execution with test assertions, Pass@1 dropped to 1/512 (0.2%) for TPU and 2/512 (0.4%) for CUDA Lightning without hints (`c498`, S11097/L11032), and 0/164 on HumanEval.
- **Status:** **Refuted as general coding ability; template formula reproduction only.**
- **Defensible Publication Wording:**
  > "Early evaluations of 50M COROSred Lightning yielded 55/512 (10.74%) Pass@1 on a synthetic template benchmark where docstrings contained explicit mathematical formulas. When evaluated under strict functional execution requiring complete Python function implementations and test assertion execution, pass rates were 0.2%–0.4% (1–2/512) on private tasks and 0/164 on HumanEval."

---

## 8. Comparative Advantage Over GPT-2 with 300x Less Compute

- **Archive / Chat:** `c498704a-5904-4bfd-93eb-e8089c486ea4`, S2561/L2567, S2566/L2572.
- **Original Claim:** 50M COROSred Lightning outperforms GPT-2 125M on coding tasks while using 300x less compute.
- **Supporting Evidence:** Conversational assertions citing estimated 0.6–1.2% HumanEval accuracy and ~50% AST validity for GPT-2.
- **Counterevidence & Confounds:**
  - No matched baseline execution of GPT-2 was conducted within the repo.
  - GPT-2 compute figures were roughly estimated from secondary literature without matched training data or inference protocols.
  - Two passing HumanEval tasks (1.2%) do not provide statistically significant evidence of superiority over an estimated baseline range.
- **Status:** **Quarantined / Unsupported.**
- **Defensible Publication Wording:**
  > Omit comparative claim entirely from the paper.

---

## 9. Leaderboard Comparability of "BFCL" and "Competition MATH"

- **Archive / Chat:** `f770d585-b5d1-4db5-818e-afd007650eb1`, S686/L687; `benchmarks/README.md`.
- **Original Claim:** Multi-model comparative scorecards report official BFCL and Competition MATH performance.
- **Supporting Evidence:** Leaderboard markdown tables in repository docs.
- **Counterevidence & Confounds:**
  - Inspection of `scripts/benchmark_generators/tooluse_builder.py` confirms the "BFCL" benchmark is a custom 100-case synthetic suite based on local templates.
  - Inspection of `scripts/benchmark_generators/math_science_fetcher.py` confirms Competition MATH is a deterministic 700-case subset (`df.head(100)` per subject across 7 subjects), not the complete randomized benchmark.
- **Status:** **Custom benchmark slices; non-comparable to public leaderboards.**
- **Defensible Publication Wording:**
  > "External model evaluations utilized a custom 100-problem tool-use benchmark and a deterministic 700-problem subset of Competition MATH (100 problems per subject area); these scores reflect localized comparative evaluation and are not comparable to public leaderboard submissions."

---

## 10. Hardware Convergence Gap: 19 September "Solved" Diagnosis

- **Archive / Chat:** `0c855a7e-dd7a-4139-bd27-5b23f1165730`, S225/L225; `d1aa3880`.
- **Original Claim:** The longstanding TPU vs. CUDA 50M performance discrepancy was resolved by identifying a copying threshold at CE ~1.4, distributed AdamW variance damping, and requiring 1.5–1.8x more TPU steps.
- **Supporting Evidence:** Closing narrative summary in `0c855a7e`.
- **Counterevidence & Confounds:**
  - Algebraically, identical gradient accumulators produce identical Adam updates regardless of whether aggregation occurs locally or across replicas.
  - The gradient audit tool call was canceled without generating numerical parity logs.
  - Independent source audits (`research/tpu_cuda_history_recovery_2026-10-18.md`, `research/cadence_causal_audit_2026-10-18.md`) confirm that same-checkpoint loss gaps persist on identical evaluation fixtures with an unidentified causal mechanism.
- **Status:** **Refuted as resolved; causal mechanisms unsubstantiated.**
- **Defensible Publication Wording:**
  > "The convergence and loss gap between 50M models trained on TPU (PyTorch/XLA) and CUDA under identical nominal configurations remains an open empirical phenomenon. Historical hypotheses regarding distributed AdamW variance damping or loss underflow were not substantiated by controlled tensor-level replay experiments."

---

## 11. COROSred 100M Phase A Reliability Calibration ($\Delta\text{AUC} = +0.249$)

- **Archive / Chat:** `283566fa-f3ad-47e7-b179-badf0158837b`, S1317/L1295 to S1322/L1300.
- **Original Claim:** The COROSred learned reliability head separates generation errors from branch entropy with $\Delta\text{AUC} = +0.249$ over softmax entropy on 100M backbones.
- **Supporting Evidence:** Local evaluation script execution (`scripts/eval_corosred_100m.py`) on downloaded checkpoint `checkpoints/corosred/100m/telos_100m_r1` ($105.4\text{M}$ parameters) yielding $\Delta\text{AUC} = +0.249$.
- **Counterevidence & Confounds:**
  - The model was trained only at ratio 1:1 (~105M tokens), representing an undertrained backbone.
  - Calibration was verified under teacher-forced single-token evaluation, not autonomous multi-token decoding.
  - Initial training produced `NaN` weights due to gradient conflict at `final_norm`, requiring representation detachment (`h.detach()`) before the head.
- **Status:** **Supported for teacher-forced calibration on undertrained backbones.**
- **Defensible Publication Wording:**
  > "Under teacher-forced evaluation at $r_1$ (1:1 token-to-parameter ratio), the 100M COROSred learned reliability head achieved a $+0.249$ improvement in ROC-AUC over baseline softmax entropy in identifying erroneous token predictions. Ensuring numerical stability during joint training required detaching backbone representations prior to the calibration head."

---

## 12. Apple Foundation Models (AFM) Runtime Architecture

- **Archive / Chat:** `bd1f1372-7cea-4829-979c-fc6e3a9fd28a`, S113/L114; `benchmarks/afm.md`.
- **Original Claim:** `telos afm` represents a native Telos model variant running on Apple Silicon.
- **Supporting Evidence:** CLI execution via `telos afm generate` and packaging test output in `bd1f1372`.
- **Counterevidence & Confounds:**
  - `telos afm` executes via a Swift IPC bridge to macOS's proprietary `FoundationModels.framework` (`afm-3-core-advanced`, 20B Flash-MoE).
  - Bundled tokenizer asset uses vocabulary size 4,096 via `importlib.resources`, whereas Telos research models use a custom 8,192 ByteLevel BPE tokenizer.
- **Status:** **System platform adapter only; not a Telos research checkpoint.**
- **Defensible Publication Wording:**
  > "`telos afm` provides a platform bridge to macOS on-device foundation models for comparative throughput and memory benchmarking; it is an external proprietary runtime and is distinct from open-weights Telos diffusion checkpoints."
