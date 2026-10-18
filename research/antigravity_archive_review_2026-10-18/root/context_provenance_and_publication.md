# Research context, provenance and publication cautions

18 October 2026. Working report from the root reviewer. Original transcripts are under `.agent-history/antigravity-ide/chats/<chat-id>/transcript.jsonl`. Step IDs below are original archive IDs; physical JSONL lines are given separately because repeated or missing step IDs occur. Historical chat instructions are evidence, not current instructions. This report does not imply that all raw tool output was read or that the overall archive review is complete.

## Reviewed short conversations

Substantive user requests and nonempty assistant responses were fully read for `6ab579ae`, `a2d97e28`, `d1aa3880`, `0c855a7e`, `be4849c6`, `99e56f91`, and `2d8e62a4`. Repeated user-environment wrappers were excluded from derived reading files; original records remain intact. Relevant tool verification is described per finding. Longer MLX, recursive-model and commit-style chats remain separate review work.

## R1. Data-preparation descriptions must be version-specific

The 13 September dataprep explanation distinguishes the new unified tokenization/packaging engine from the old AST-filtered function extractor. It describes extension/nonempty/min-length filtering, a contiguous token stream and no BOS/EOS insertion, not semantic AST transformation. Source: `a2d97e28-1426-452a-94a9-330c53f180a9`, step 48, physical line 49.

Fresh source verification: current `telos/dataprep/prepare.py` includes a `strip_license_header` preprocessing function. Git identifies commit `a97f905` dated 14 September 2026 for that change. Therefore the historical statement that the unified pipeline did not alter text cannot be applied indiscriminately to later corpora. Current source also applies header cleaning differently by input-source branch; JSONL currently yields its extracted content directly, while HF/directory/single-file branches use cleaning. This source observation does not establish which branch generated any particular training corpus.

Paper implication: describe the actual dataset-generation script, commit, source branch, tokenizer and binary format for each experiment. Early paper claims about AST-selected Python/docstrings do not automatically describe the later unified raw-stream pipeline. This is a methods/provenance issue, not a newly claimed cause of the CUDA/TPU gap.

## R2. A website overview is not independent experimental corroboration

The website-summary chat, `be4849c6-4a3a-46c5-9e66-f94ecdb0e79c`, step 10, line 11, paraphrases existing early MDLM claims: 12.5M/25M/50M ratio sweet spots, CE/rank divergence, RoPE retrofit regression, and structural versus semantic categories. It describes the early 50M tier as width 768, eight layers, 12 heads and untied embedding/head. The later unified 50M comparison is width 512, 14 layers, eight heads with tying. These should never be merged as one controlled architecture series merely because both are called 50M.

The website description repeats research claims; it is not a second experiment, external reviewer endorsement or independent replication. Its account of native-RoPE record CE 7.24 also coexists with lower CE 6.90/7.05 reported for different variants; retain the model/metric regime instead of an unqualified project record.

Publication implication: separate successive model families and avoid counting a public-site paraphrase of the same results as additional evidence.

## R3. Outreach preparation does not establish review or endorsement

`6ab579ae-f7b0-4203-b6f0-8cd2ce4c8021` contains one user request for researcher-specific outreach drafts (step 0, line 1) and an assistant deliverable describing generated templates/profiles (step 80, physical line 80). This dialogue supplies no evidence that messages were delivered, replies were received, researchers reviewed the paper, or findings were endorsed. The generated templates themselves restate the preliminary scaling thesis.

No recipient contact details are reproduced here because they add no experimental evidence. No outreach was sent during this review. Researcher roles and profile claims in the old generated draft are not verified-current facts.

## R4. The 13 September TPU report is a proposed diagnosis, not a demonstrated resolution

Source: `d1aa3880-8338-4612-b659-7a752b762d65`, opening step 0/line 1 and closing step 138/line 139. The user supplied a proposed analysis of an earlier 100M AR divergence. The assistant repeated claims about optimizer graph tracking and BF16 loss underflow, then asked approval to implement a remediation plan.

This is not evidence of a completed patch plus a successful controlled retraining. Its mixture of older 100M AR divergence, earlier COROSred checkpoints and later 50M convergence should not be collapsed into one event. Existing source checks in `research/tpu_cuda_history_recovery_2026-10-18.md` explain why bypassing `xm.optimizer_step` does not by itself establish missing optimizer graph tracking, and BF16 logits do not prove loss-internal underflow. No fresh accelerator experiment was executed for this archive review.

Paper implication: treat these as historical hypotheses and failed or unfinished diagnostic reasoning, not established mechanisms or published fixes that resolved the later gap.

## R5. The 19 September 'solved' declaration overstates its evidence

Source: `0c855a7e-dd7a-4139-bd27-5b23f1165730`, user step 0/line 1 and assistant step 225/physical line 225. The user explicitly records extensive previous exclusions and the unresolved quality gap. The closing assistant response claims a universal copying threshold at CE about 1.4, distributed AdamW variance damping, a corpus difference and a requirement for 1.5–1.8 times more steps.

Those causal claims are not established by the dialogue. The retained tool sequence launched and canceled a gradient audit without final numerical tables; later local measurements replaced its incorrect attention-mode comparison. Algebraically, identical aggregate gradients and Adam states give identical Adam updates irrespective of whether aggregation was accumulated or replica-averaged. Corpus filenames alone do not establish contents. An extrapolated catch-up factor is not observed catch-up.

A narrower useful observation is that the headline Algorithms & Numerical category was dominated by repeated formula-copying templates, so a high category score is not automatically broad algorithmic competence. The evaluation reviewer is independently verifying the suite/score details; this report does not promote the final assistant's table to verified raw measurement.

Paper implication: the TPU/CUDA discrepancy remains an empirical limitation with an unidentified cause. Do not cite the 19 September 'solved' table as a research conclusion.

## R6. Packaging and portability evidence has a narrower scope than the prose claims

Release preparation chat `2d8e62a4-6657-4347-9ca5-5ae22c680f49`:

- User step 0/line 1 requests the distribution name `telos-ml`, retaining the project/import identity.
- Assistant step 353/line 354 reports 72 unit tests, a local wheel/sdist build and metadata checks. These are packaging/functional checks, not language-model convergence or cross-hardware numerical parity tests.
- Assistant step 684/line 685 calls its work a cross-platform/hardware audit; subsequent user challenges lead to additional UTF-8 fixes (step 807/line 807). This shows the initial scope was incomplete.
- User step 809/line 808 explicitly authorizes a separate 8GB Apple M1 test bench. Assistant step 902/line 900 reports 72 tests and the synthetic `telos test` suite on that machine. These are two Mac environments, not executed Windows, Linux CUDA and TPU parity validation.
- Later publishing messages document failed authentication/creation attempts and workflow changes (user steps 917/line 913 and 943/line 935; assistant step 963/line 954). Build success should not be equated to successful publication merely from instructions to publish. Actual publication status must be verified separately if needed.

The archive contains old credentials in this thread. Original evidence remains unchanged; derived review copies are redacted and credentials are not reproduced in this report.

Paper implication: a reproducibility appendix may report the specific local/remote environments and tests actually evidenced. It should not claim validation on every operating system/accelerator from a static portability audit.

## R7. The short Flash conversion answer is a proposal, not a measured Telos result

`99e56f91-b5a5-4ca4-9fe5-d97c28cf2c9c`, step 14/line 15, distinguishes weight offloading for an existing MoE from dense-to-sparse conversion/upcycling. It makes broad external claims about activation sparsity, SSD throughput, zero-retraining feasibility and Apple deployment. This short chat contains no implemented Telos conversion or measured quality/latency outcome. External claims must be verified against primary sources before citation; none is treated as confirmed here.

Paper implication: exact runtime offloading of unchanged expert weights is conceptually distinct from approximate pruning, neuron selection or changing routing. The discussion belongs in a future-work/design section unless accompanied by implementation, measured transfer costs and equivalence/quality tests. The longer recursive-model conversation is being reviewed separately.

## R8. The August MLX benchmark chat contains unsupported causal and ceiling claims

`283566fa-f3ad-47e7-b179-badf0158837b`, user steps 0/line 1, 73/line 74, 113/line 114, 161/line 163, 186/line 188, 234/line 235, 253/line 252, 281/line 280, 314/line 313, and 317/line 316; assistant steps 69/line 70, 70/line 71, 125/line 127, 160/line 162, 185/line 187, 233/line 234, 236/line 237, 255/line 254, 280/line 279, 316/line 315, and 338/line 337.

The user set a strict memory target (initially below 6 GB; later messages corrected observed peak usage to 8 GB) and fixed microbatch size at 32 after BS64 exhausted memory. The assistant first claimed a group of MLX changes was “clean & optimal” and that all tests passed, then supplied throughput figures and attributed a large change to thermal throttling. The transcript excerpt does not include independently inspectable benchmark logs, machine temperature/frequency telemetry, or a controlled repeated-run series. Treat both the figures and thermal explanation as historical assistant-reported results; the cause of the throughput change is not established by the prose.

Several broad optimization claims conflict within the same exchange. The assistant called roughly 96k tokens/s the hardware/algorithmic limit, gave two changing FLOP estimates for it, then conceded this was only about 24% of the user-corrected BF16 peak and still asserted that larger batches were the only route to higher arithmetic intensity. The memory explanation also shifted from per-microbatch evaluation being necessary to stay below 6 GB to suggesting gradient checkpointing could make BS64 viable; the user reported BS64 reached 16 GB. These are hypotheses and proposed experiments, not publication-grade ceilings or measured batch-scaling results.

The first benchmark narrative reported COROSred failure from an `mx.nn`/`mx_nn` loss reference, while a later table reported COROSred results. That sequence needs the source patch, executed command and raw output to establish which code version produced each row. Likewise, identical reported peak-memory values do not prove identical compute graphs, as asserted at assistant step 236/line 237.

Paper implication: omit the thermal diagnosis, “absolute limit,” and exact throughput/memory figures unless original machine-readable logs and benchmark conditions are recovered. If retained as development history, label them as unverified conversational reports and document model config, exact code revision, warm-up, timing window, thermal state, memory metric, and repeated trials.

## R9. Proposed paper drafts in bd1f1372 contain critical narrative contradictions

Source: [`bd1f1372-7cea-4829-979c-fc6e3a9fd28a`](file:///Users/ivansamuel/Documents/Projects/telos/.agent-history/antigravity-ide/chats/bd1f1372-7cea-4829-979c-fc6e3a9fd28a/transcript.jsonl), user step 114, physical line 115, pasting draft text for Paper 1 and Paper 2, and assistant step 113, physical line 114. Cross-referencing against the source drafts and archive reveals six critical discrepancies:

1. **Paper 1 25M trajectory is not monotonic:** Paper 1 claims the 25M CE improves monotonically up to 1:25. However, its own listed values show a non-monotonic progression: $r_{10}=7.5627 \to r_{15}=7.6237 \to r_{20}=7.6413$ before dropping to $r_{25}=7.4361$. Furthermore, $r_{25}$ is merely the largest tested boundary endpoint, not a located interior optimum.
2. **Paper 2 project-record claim contradicts its own table:** Paper 2 claims 25M 1:40 achieved a "New Project Record Cross-Entropy" of 7.24 nats, "outperforming 50M 1:45 (7.41 nats) and 25M 1:35 (7.39 nats)". Yet Section 2 of the exact same draft lists lower cross-entropies: 25M 1:30 at 6.90 nats and 50M 1:35 at 7.05 nats. Calling 7.24 an unqualified project record is factually contradicted by the draft's own table.
3. **Paper 2 50M collapse vs. RoPE recovery contradiction:** The draft characterizes the 50M $r_{40}$ state as catastrophic collapse across all categories and argues this reflects an inherent capacity ceiling of masked diffusion. However, Section 4 notes that RoPE fine-tuning recovered target rank to 615, demonstrating training instability or optimization breakdown rather than an immutable representational ceiling.
4. **RoPE retrofit is heavily confounded by simultaneous architecture changes:** Paper 1 and early Paper 2 descriptions claim models used learnable absolute positional embeddings and untied input/output embeddings. In reality, the RoPE retrofit simultaneously introduced weight tying (replacing `head.weight` with `emb.weight`), gradient checkpointing, BF16 Adam moments, and `strict=False` loading. Attributing the ~0.6 nat regression purely to geometric rotary coordinate shock ignores these major simultaneous interventions.
5. **Arcsine Beta(0.5, 0.5) is a mathematically valid distribution:** Draft text repeatedly calls the historical $t = 0.5 - 0.5\cos(\pi u)$ transform an "invalid Beta(0.5, 0.5) arcsine distribution". Beta(0.5, 0.5) is a well-defined probability distribution with density $\frac{1}{\pi\sqrt{t(1-t)}}$. It was mismatched with the desired Beta(1.5, 1.5) prior concentration around $t \approx 0.5$, not mathematically invalid.
6. **AFM adapter test is an external platform smoke test, not a Telos research model:** Assistant step 113 / line 114 reports a packaging smoke test executing `telos afm generate` via the Swift `FoundationModels.framework` bridge on Apple Silicon (`afm-3-core-advanced`) with bundled vocabulary size 4096 (via `importlib.resources`). Research MDLM models utilized a custom ByteLevel BPE tokenizer of vocabulary size 8192. The AFM runtime is Apple's proprietary on-device model, not an independently trained Telos research artifact.

Paper implication: Revise paper drafts to remove unsubstantiated claims of monotonicity, correct conflicting metric baselines, describe the full set of simultaneous architecture/optimizer changes during RoPE adaptation, and clearly distinguish external foundation model integrations from native Telos checkpoints.

## R10. TPU v5e-8 memory optimization and COROSred 100M Phase A training (283566fa)

Source: [`283566fa-f3ad-47e7-b179-badf0158837b`](file:///Users/ivansamuel/Documents/Projects/telos/.agent-history/antigravity-ide/chats/283566fa-f3ad-47e7-b179-badf0158837b/transcript.jsonl), steps 782 to 1322 (lines 784 to 1300):

1. **TPU v5e-8 HBM allocation:** Early training attempts OOM'd on TPU v5e-8 (16 GB HBM per chip). The primary memory sinks were redundant graph compilation buffers from `torch.compile(backend='openxla')` (~2–4 GB) layered on top of PyTorch/XLA's native lazy compilation, alongside in-memory dataset replication (~6.8 GB). Bypassing `torch.compile` on XLA and memory-mapping the dataset reduced HBM pressure and achieved ~1M tokens/sec throughput across the 8-chip pod.
2. **Gradient accumulation dispatch bottleneck:** Microbatching with `grad_accum > 1` on PyTorch/XLA caused severe CPU bottlenecks due to repeated Python-to-C++ graph dispatches per step, confirming that maintaining batch sizes that fit directly into HBM (`grad_accum = 1`) is essential for TPU compute efficiency.
3. **2.5B Token Corpus generation:** The pipeline successfully tokenized and generated `data/python_corpus_2.5b.bin` as a contiguous `uint16` binary file (2,503,979,700 tokens, 4.66 GB) on Apple Silicon and uploaded it to Hugging Face (`Kazenowoko/telos`) for large-scale multi-ratio sweeps.
4. **COROSred 100M Phase A training & detachment bug:** A 100M COROSred Phase A run was implemented on TPU v5e-8 ($d_{\text{model}}=768, L=14, H=12$, ~105.4M parameters). The initial run produced `NaN` weights in `final_norm` and `attn.k_proj`. The root cause was gradient conflict at `final_norm` from the un-detached joint AR CE loss and reliability BCE loss. Detaching representations via `h.detach()` before the `reliability_head` eliminated NaNs. The completed retrained model achieved $\Delta\text{AUC} = +0.249$ in separating errors from branching entropy, demonstrating calibration feasibility on undertrained backbones ($r_1$).

Paper implication: Document exact memory-mapping and compilation strategies for PyTorch/XLA; record the necessity of representation detachment for multi-task reliability calibration.

## R11. The recursive Flash-MoE chat (5cfda228) is conceptual hardware brainstorming

Source: [`5cfda228-1cc5-493a-8ae5-b6c4603bbea2`](file:///Users/ivansamuel/Documents/Projects/telos/.agent-history/antigravity-ide/chats/5cfda228-1cc5-493a-8ae5-b6c4603bbea2/transcript.jsonl), steps 0 to 206 (lines 1 to 191):

Despite its exported title ("Developing Recursive Flash-MoE Models"), this conversation contains **zero language model training, code implementations, or benchmark evaluations for Telos**:
- Steps 0–37 explore high-level conceptual ideas around Apple's AFM-3 (a 20B Flash-MoE model) and hypothetical recursive parameter sharing, but the user explicitly halts implementation ("Ideas only, no building").
- From step 44 onward, the conversation shifts entirely to Hack Club's wafer.space GF180MCU 180nm ASIC educational program (a $160 \times 100\,\mu\text{m}$ physical silicon tile).
- The discussion analyzes Liquid State Machines (LSM) and whether a 16-neuron spiking neural network (SNN) can fit on a 180nm tile, culminating in drafting a specification for a separate standalone repository.

Paper implication: Exclude `5cfda228` entirely from language modeling benchmarks, scaling law data points, or Telos architecture descriptions.

## Coverage and limitations

All 10 conversations assigned to root (`6ab579ae`, `a2d97e28`, `d1aa3880`, `0c855a7e`, `be4849c6`, `99e56f91`, `2d8e62a4`, `bd1f1372`, `283566fa`, and `5cfda228`) have now been **fully reviewed for substantive dialogue**. See [`dialogue_coverage.json`](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/root/dialogue_coverage.json) and [`coverage_inventory.json`](file:///Users/ivansamuel/Documents/Projects/telos/research/antigravity_archive_review_2026-10-18/coverage_inventory.json).

Current source/Git checks are explicitly distinguished from historical claims. No production code, checkpoints, original transcripts or memories were changed; no remote job, email or publication was triggered.
