# Antigravity Historical Archive Review (2026-10-18)

## Overview & Scope

This directory contains the completed systematic evidence review of the Antigravity IDE conversational archive (`.agent-history/antigravity-ide/`), covering all **19 exported conversations**, **53,224 JSONL records**, and **~833,536 substantive dialogue words**.

The objective of this review is to establish rigorous empirical provenance, audit paper draft claims against underlying tool executions and commit history, identify confounding variables and falsified historical hypotheses, and provide defensible wording for the final Telos research paper.

---

## Directory Navigation

- [**`paper_claim_ledger.md`**](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/paper_claim_ledger.md): Itemized audit of 12 core research claims, covering original draft text, supporting evidence, counterevidence/confounds, status, and publication-ready defensible wording.
- [**`coverage_inventory.json`**](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/coverage_inventory.json): Complete manifest of all 19 archived conversations, record counts, token/word volumes, sha256 verification hashes, and review status.
- [**`root/context_provenance_and_publication.md`**](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/root/context_provenance_and_publication.md): Findings R1–R11 covering data preparation, packaging, public claims, TPU v5e-8 memory optimization, and Paper 1/Paper 2 draft cross-checks.
  - Accompanied by [`root/dialogue_coverage.json`](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/root/dialogue_coverage.json).
- [**`mdlm/findings.md`**](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/mdlm/findings.md): Findings 1–19 covering early MDLM 50M MLX runs, probe evolution, Net2Net upscaling shock, RoPE retrofit confounds, and TPU v5e/v6e execution.
  - Accompanied by [`mdlm/coverage.json`](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/mdlm/coverage.json).
- [**`unified/findings.md`**](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/unified/findings.md): Findings 1–18 covering the Unified Trainer evolution, PJRT client singleton crashes, continuous joint loss design, multi-backend lineage fragmentation, and 0% Pass@1 functional outcomes.
  - Accompanied by [`unified/coverage.md`](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/unified/coverage.md) and [`unified/chunk_inventory.json`](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/unified/chunk_inventory.json).
- [**`evaluation/findings.md`**](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/evaluation/findings.md): Findings E01–E13 covering benchmark suite revisions, tokenizer vocabulary mismatches (4096 vs 8192), formula-copying vs. true functional code execution, repetition loop detectors, and external model evaluations (AFM, Granite, LFM, Phi 4).
  - Accompanied by [`evaluation/coverage_checkpoint.json`](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/evaluation/coverage_checkpoint.json).

---

## Executive Summary of Consolidated Findings

### 1. Experimental Lineage & Model Family Separation
- **Legacy vs. Unified COROSred Models:** Early phased COROSred checkpoints (Phase A/B/C from August 2026) suffered from serious integration flaws (e.g. Phase A omitting AR language model loss, untied embeddings, uncalibrated learning rates). These must **never be pooled or compared directly** with later September unified joint-objective models, even though both share the label "50M".
- **Parameter & Architecture Drift:** The early 50M tier ($d_{\text{model}}=768, L=8, H=12$, untied embeddings) differs structurally from the later unified 50M model ($d_{\text{model}}=512, L=14, H=8$, tied embeddings). Checkpoint comparisons must state layer counts, head configurations, weight tying, and parameter counts explicitly.

### 2. Paper Draft Corrections (`paper_1` & `paper_2`)
- **Non-Monotonicity of 25M:** Paper 1 claims the 25M CE trajectory improves monotonically up to 1:25. The underlying table proves non-monotonic fluctuations ($r_{10}=7.5627 \to r_{15}=7.6237 \to r_{20}=7.6413$ before $r_{25}=7.4361$). $r_{25}$ is merely the evaluated boundary endpoint, not an established optimum.
- **Project Record Inconsistency:** Paper 2's claim that 25M 1:40 achieved a project-record cross-entropy of 7.24 nats is directly contradicted by earlier tables in the exact same draft (reporting 6.90 nats for 25M 1:30 and 7.05 nats for 50M 1:35).
- **RoPE Confounding:** The reported ~0.6–0.67 nat regression from retrofitting RoPE via 150-step fine-tuning was confounded by simultaneous architectural interventions: input/output weight tying (discarding separate `head.weight`), gradient checkpointing, BF16 Adam moments, and `strict=False` loading. Rotary coordinate shock cannot be claimed as the sole causal factor.
- **Timestep Sampler Formulation:** The historical $t = 0.5 - 0.5\cos(\pi u)$ transform was not "mathematically invalid"; it was a mathematically valid arcsine Beta(0.5, 0.5) distribution whose boundary concentration ($t \to 0, 1$) was mismatched with the desired Beta(1.5, 1.5) prior ($t \approx 0.5$).

### 3. Evaluation Harness & Metric Realities
- **Contextual Probes vs. Iterative Generation:** Contextual probe benchmarks measure single-step continuation probabilities at fixed noise levels ($t$). They do not measure iterative unmasking generation dynamics where prediction errors compound over multiple steps.
- **Private Unseen Benchmark Collapse (10.74% to 0.2–0.4%):** The early headline score of 10.74% (55/512) on Private Unseen was an artifact of synthetic formula-copying templates with docstring leakage. When evaluated under strict functional Python execution with test assertions, Pass@1 dropped to 1/512 (TPU seed) and 2/512 (CUDA Lightning), and 0/164 on HumanEval.
- **Tokenizer Scrambling:** Multiple intermediate evaluations between August and September inadvertently loaded the legacy bundled 4,096-vocabulary tokenizer rather than the 8,192 research tokenizer, producing corrupted loss/rank metrics.

### 4. Hardware & Optimization Dynamics
- **Longstanding 50M TPU vs. CUDA Gap:** The performance discrepancy between TPU (PyTorch/XLA) and CUDA models remains an **unresolved empirical limitation with an unidentified root cause**. Historical assistant declarations claiming the issue was "solved" by distributed AdamW variance damping or loss underflow were unverified hypotheses not supported by tensor-level replay.
- **PyTorch/XLA Systems Engineering:** Microbatch gradient accumulation (`grad_accum > 1`) on TPU v5e caused severe CPU dispatch bottlenecks. Disabling `torch.compile(backend='openxla')` and memory-mapping datasets saved 2–4 GB HBM, avoiding OOMs and achieving pod saturation (~1M tok/s).
- **COROSred Reliability Detachment:** Multi-task training of the learned reliability head requires detaching backbone representations (`h.detach()`) to prevent gradient conflict and NaN weights in the final layer norm. With detachment, the 100M Phase A model achieved $\Delta\text{AUC} = +0.249$ in separating errors from branching entropy.

---

## Remaining Research Limits

1. **Unresolved Backend Gap:** No new accelerator training runs were launched during this archive review. The 50M TPU–CUDA gap remains an empirical open question documented in `research/tpu_cuda_history_recovery_2026-10-18.md` and related audits.
2. **Raw Tool Outputs:** While all 53,224 records across all 19 chats have been indexed and all substantive user and assistant dialogue turns have been read, raw tool executions (ephemeral shell commands, file dumps) were inspected selectively on consequential claims rather than exhaustively.
3. **Paper Assets:** Existing manuscript markdown files (`research/paper_*.md`) and figures remain in their historical state. Updates must be applied by the user or in a dedicated drafting pass using the defensible wording provided in [`paper_claim_ledger.md`](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/paper_claim_ledger.md).
