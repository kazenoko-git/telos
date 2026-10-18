# Early MDLM, probe evolution, and paper-evidence audit

Review date: 2026-10-18. Reviewer: parallel MDLM archive review agent. This is an evidence audit, not a new model experiment or a finding of the cause of the later unified COROSred TPU–CUDA gap.

## Scope and citation key

Original sources are `.agent-history/antigravity-ide/chats/<ID>/transcript.jsonl`. `L` means **one-based physical JSONL line**; `S` means archived `step_index`. They are not interchangeable; duplicates and missing steps exist.

- **E**: `41b584d5-6756-4262-977b-2d3c9b81f8a1`, *MDLM 50M MLX Overnight Runs*, 2026-08-06–23. 11,013 records. Large early development history.
- **P**: `8ea83f59-d4ff-4096-aed7-26924954fdb0`, *Probe Execution And Plotting*, 2026-08-23–26. 705 records.
- **N**: `04d3c66e-a260-4469-a56d-082ff1de11ea`, 2026-08-27–31 (some tool-local timestamps cross midnight). 1,155 records.

See `coverage.json` for exact review coverage. Entire user/model dialogue and all GENERIC records were read for P and N; selected original code actions and command outputs were inspected. E review remains **in progress**. Merely extracting/indexing E does not count as semantic review. Derived dialogue files strip wrapper metadata when a `<USER_REQUEST>` exists; original source remains authoritative. Credential-shaped strings were redacted in derived text, not in the original archive.

## Findings suitable for a historical methods or limitations section

### 1. Probe scores changed substantially when the measurement procedure changed

**Direct evidence:** N L442/S442 is a completed diagnostic task output listing “Correct Target IDs + No 500-pad” results. Its 12M masked r25 CE is 6.7970 and r30 CE 7.6908. N L461/S461 records the concrete protocol changes: context-sensitive target ID resolution (previously unconditional space prefix, claimed 32/101 affected), removal of ~500 padded mask/noise positions, and UNDLM draw count K=16. N L75/S74 describes the preceding 512-token padded protocol and K=32. These are different measurement protocols, not exchangeable points on one curve.

P L91/S90 and L206/S206 report earlier 100-probe tables; N L461/S461 reports much different later values. For example AR 12M r25 CE 6.8032 in P versus 4.5463 in N. Several changes occurred between them. Do not attribute this entire difference to training progress or any one patch.

**Paper implication:** Every table/figure needs evaluator source revision, exact probe/token IDs, input padding and masking, random seeds/draw counts, checkpoint SHA and tokenizer identity. Single-step continuation probes do not establish multi-step generation competence. Probe counts are called 100, 101 and 98 active in different places; retain the actual scored denominator.

### 2. The archived RoPE “fix” explanation is internally inconsistent

N L255/S254 and L295/S294 claim torch half-split RoPE corresponds to MLX `traditional=True`, describing `False` as interleaved. However, original tool output N L214/S213 shows torch `_rotate_half` splitting the feature vector into halves. N L223/S222 prints a toy MLX comparison at nonzero position:

- `traditional=False`: `[-0.301169, 0.98995, 1.38177, 1.00995]`
- `traditional=True`: `[-0.301169, 1.38177, 0.98995, 1.00995]`

Those values show that **False is the half-split layout in that toy execution**, contrary to the narrative. N L226/S225 shows the old False calls. This is evidence against copying the archived natural-language diagnosis verbatim. It does not by itself establish which convention each historical checkpoint was trained with, nor whether the subsequent overall probe gains came from this change. Need a full torch-versus-MLX tensor parity check tied to each checkpoint lineage before presenting a repaired RoPE bug as confirmed causal history.

### 3. A concrete silent MLX→PyTorch source-weight loading problem is documented

**Direct evidence:** N L1138/S1157 prints an upscaler that copies only if `src_key in src_state`, silently leaving other target tensors at their initialization. N L1141/S1160 prints MLX source keys (`emb.weight`, `norm1`, `norm2`, `qkv_proj`, `mlp.w1/w2/w3`). N L1108/S1127 prints torch projection and SwiGLU naming (`attn.q_proj`, `mlp.v`, `mlp.w2`). N L1147/S1166 is the actual adaptation diff translating embeddings, norms, projections and splitting QKV. N L1150/S1169 records commit `0683a2a`; L1153/S1172 records its push.

**Limit:** N closing L1155/S1174 says “90% ignored” and calls this the root cause. Exact fraction is an agent claim, not a verified per-parameter load accounting table. Nor was the predicted CE 4.5–5.0 after retraining measured in this closing message.

**Paper implication:** Exclude affected checkpoints from claims of smooth token-budget saturation or inherent MDLM collapse unless initialization was independently reconstructed. Successful-looking “initialized with zero-shock parity” messages were not sufficient validation. This is an early upscaled AR/MDLM/UNDLM result, not proof about later unified COROSred 50M runs.

### 4. Historical collapse diagnoses were repeatedly revised

N L590/S593 first claims missing TPU clipping, high LR and fallback initialization explain r30/r35 collapse. N L612/S615 gives the proposed clipping patch; L640/S643 reports it applied. N L956/S977 later claims three pre-fix bugs fully explain the downloaded checkpoints. Nevertheless N L1155/S1174 says the freshly downloaded replacements still had CE ~9.11/9.60 and identifies the naming mismatch above.

The direct conversion evidence is useful; the successive declarations of “root cause,” “models are fine” or guaranteed recovery are not independently sufficient. P L415/S416 calls the code mathematically flawless before P L464/S466 identifies the hardcoded mask ID problem. This archive is valuable precisely because it preserves falsified explanations.

### 5. Initialization, backend, optimizer recipe and architecture changed with ratio

P L253/S253 and duplicate L255/S253 explicitly propose 50M r1/r30 upscaled locally, r35 upscaled on T4, r40/r45 from scratch on TPU. P L256/S254 recognizes the initialization confound. P L150/S153 and L156/S153 contain the user-pasted actual launch output falling back to 12M r25 for 25M r30, despite the request for scratch when same-ratio source is absent. N L777/S782 shows fallback behavior persisted/reappeared later; L811/S816 reports removing it.

N L568/S571 initially mistakes duplicate tied embedding storage for a real parameter-count difference, then retracts that within the same message. Parameter count must mean unique learned parameters, not sum of checkpoint tensor entries.

P L498/S501 reports a broad LR/weight-decay rewrite; P L646/S642 reports another LR and warmup rewrite, and L661/S657 lists the settings. N L853/S858 reports a diffusion-only 0.1 LR multiplier (commit `d3302c9`). N L947/S968 reports a later 50M LR change. These require revision-specific run metadata. A generic tier/ratio label cannot identify a controlled experiment.

### 6. Wrong mask ID is supported, but “weight decay collapse” was overclaimed

P L464/S466 states tokenizer ID1=[MASK], ID4=[UNK], while the loss used4. This requires validating the exact tokenizer and source revision for each run; using the trained corruption token at inference may salvage an existing checkpoint, as the same conversation notes at L470/S473. It is not automatically proof every model requires retraining.

The adjacent claim that AdamW weight decay is an L2 loss penalty is inaccurate. Its claimed catastrophic feature collapse from global0.1 decay was not supported there by norm trajectories or a controlled ablation. An exemption for 1D parameters can be a sensible recipe choice without being established as the measured cause of failure. Do not cite these broad claims as experimental findings.

### 7. Timestep sampler correction is a real historical boundary; its claimed scaling invariance is not proved

E L2629/S2645 contains the user's detailed audit of a cosine transform mislabeled Beta(1.5,1.5) and of a dummy `optimizer.update` on synthetic inputs before step1. E L2667/S2683 reports removal of that synthetic update and explicit cosine/uniform strategy separation. E L2980/S2978 and L2982/S2980 discuss the later addition of an actual NumPy Beta sampler; E L2996/S2994 reports its addition.

For `t=(1-cos(pi*u))/2`, uniform u, the density is **1/(pi sqrt(t(1-t)))**, Beta(1/2,1/2). The different density printed in E L2980/S2978, `(pi/2) sin(pi*t)`, is incorrect despite its correct Beta label. Claims in E L3002/S3000 and L3141/S3137 that changing the distribution only multiplies token efficiency by a constant, preserves curve shape/rankings, or proves a particular scaling law are **hypotheses without a paired experiment**. Do not promote them to theory or results. “Paper-standard Beta1.5” also needs verification against the actual cited objective and any importance correction.

### 8. Overnight progress prose is demonstrably unsafe as completion evidence

E L3298/S3294 declares all11 runs finished. Immediately afterward E L3299/S3295 supplies the actual `Path.stat().mtime` traceback at the transition from 12M to25M. E L3319/S3315 acknowledges that 25M never started and that earlier monitoring inferred progress from directory existence. The chronology preserves a concrete false-completion episode.

Later genuine runs may exist; this episode does not invalidate all checkpoints. It means final paper run completeness must be backed by endpoint weights, optimizer/step metadata and logs, rather than conversational progress updates. E L3431/S3428 also corrects a supposed625M-token run to6502steps/~852M tokens; exact saved configuration must settle the run budget.

### 9. Early COROSred design provenance includes important rejected claims

P L677/S675 is the user's explicit statement of CRSR differences and tests: learned reliability head, trained correction loop, attention geometry, full-sequence refine cost and stochastic-diversity evaluation. P L690/S688 contains detailed corrections: iid categorical k-sample agreement is only a distribution functional; bidirectional refinement cannot simply reuse causal K/V; AR token steps and full-sequence refinement passes are not equal cost units; restrict novelty claims and include prior art.

P L695/S693 reports these changes applied. P L696/S694 adds router exposure bias, ambiguity-exclusion k sweep, effect sizes/multiple seeds, sequence-length-dependent cost and prose OOD provenance. P L705/S703 reports v0.3 gates (deltaAUC>=.05 and>=3seeds; exposure-bias deltaAUC<.10; SelfBLEU increase<=.02). These are **design requirements**, not results demonstrating the gates were passed.

The archive thus supports a design-evolution narrative. It does not support “proved superior,” guaranteed noncollapse, demonstrated hidden-state error identifiability, or the assertion that same-distribution sample agreement validates semantic correctness.

## Existing paper drafts: corrections needed before publication

Inspected `research/paper_1_token_budget_scaling.md` and `research/paper_2_capability_divergence_and_rope.md` directly. They remain unchanged by this review.

1. **Paper1 says 25M progress is monotonic, but its own CE table rises7.5627→7.6237→7.6413 before7.4361.** Say “best observed endpoint at largest tested ratio,” not monotonic improvement.
2. **Paper1's ratio fit has too few and censored optima.** A largest-tested best point is not an identified optimum, and upscaled initialization/pretraining tokens differ. Candidate log/sqrt/power fits are descriptive extrapolations, not an empirical compute-optimal law.
3. **Paper2 calls7.24 the project record despite listing6.90 at25M r30 and7.05 at50M r35.** Scope it to a specific comparable subset if warranted, otherwise remove.
4. **Paper2's “clean/unconfounded”50M absolute-PE series is not established by holding PE alone fixed.** Scratch/upscale, token count, backend, loss recipe and conversion provenance still need controls.
5. **Paper2's causal account of CE-rank divergence as capacity starvation or a fundamental MDLM ceiling is unproven.** Ground-truth token ambiguity, probe-token extraction, evaluator geometry, optimization, initialization and corrupted load paths remain alternatives. Aggregate CE/rank can diverge without proving token-category capacity tradeoffs.
6. **Paper2's “RoPE cannot be retrofitted” is universal language from a150-step experiment.** Supported scope would be the specific adaptation recipe and checkpoints after loader/parity validation. Likewise native-RoPE superiority cannot be separated from budget and initializer changes without matched controls.
7. **Architecture is temporal.** Paper1 lists absolute PE and untied embeddings, whereas E L3527/S3525 describes RoPE and tied embeddings. Do not resolve this by choosing whichever report sounds authoritative; inspect each retained checkpoint/source revision.
8. **Structural versus semantic category comparison is descriptive, not proof of representation or impossible targets.** E L2614/S2630 labels >55% “impossible” based largely on ambiguity/rank; those categories still need contextual task construction and acceptance of multiple valid completions.
9. **All public-literature names/novelty claims in the archive need fresh primary-source checking.** This review audits the historical conversation, not the truth of every search result it contained.

## Relation to the current TPU–CUDA issue

The early archive demonstrates that real integration, conversion and evaluation defects occurred and that confident historical explanations frequently overshot evidence. It therefore strengthens the need for source/checkpoint lineage and tensor-level parity in the later study. It does **not** supply a newly proven cause of the later unified COROSred50M deficit. Respect the user's exclusion of pre-A+B+C-merge COROSred and earlier failed paradigms as controls for that later comparison.

## Additional findings from continued E review (chunks 2–17)

### 10. The RoPE retrofit also changed weight tying and optimizer storage

E L4592/S4603 reports adding RoPE **together with** weight tying (discarding/replacing the independent head on load with embedding-derived output), gradient checkpointing, BF16 Adam moments and `strict=False` loading. E L4532/S4543 had explicitly warned that the old `head.weight` would be replaced by `emb.weight`. E L4678/S4693 then reports the post-adaptation scores. Thus the observed ~0.6-nat regression is not a clean isolated RoPE ablation. The report must list simultaneous changes and inspect actual loading/parameter parity before attributing the regression solely to rotational-coordinate adaptation.

### 11. “No RoPE” was repeatedly turned into an unsupported “learned absolute PE” story

E L4504/S4515 and L4510/S4521 explicitly describe the old MLX model as having **no positional encoding of any kind**. Yet E L4840/S4857 later asserts the original50M1:35 used learned absolute PE and that fine-tuning ripped those learned positions out. E L4887/S4895 then infers every historical model used RoPE from inspecting the **current** hardcoded model, which the user immediately corrects at L4888/S4896. The closing correction L4889/S4897 again calls old models absolute/learned PE without showing the original weight keys or source.

This is an unresolved architecture-provenance contradiction. **Do not repeat the draft's absolute-PE description as confirmed.** The distinction matters: introducing RoPE into a permutation-equivariant model differs from replacing learned absolute positions. Later code does not establish earlier checkpoint architecture.

### 12. The flagship50M1:40 “collapse” score conflicts with its earlier evaluation

E L4464/S4474 reports50M1:40 at step15000 with CE7.0563, rank510.3, Top5=5.94%, and category-level values; it identifies checkpoint `phase_b_50m_1to40_mlx_20260813_222750`. E L4830/S4847 later calls the “native”50M1:40 a catastrophic collapse (CE9.67, rank3988) and treats its RoPE-FT score615 as recovery. No matched evaluator/checkpoint-hash comparison has been established in the reviewed dialogue. Given that the MLX model code had changed meanwhile, this **must be resolved before Paper2 uses9.67 as the cleanest overtraining evidence**. Candidate explanations include different evaluated weights or evaluating a non-RoPE checkpoint under the changed architecture. These are hypotheses, not established diagnosis.

### 13. Paper2's own supporting category comparison changed numbers

E L4398/S4408 reports25M1:30 function-name rank550.5 and attribute-name rank1242.8. E L4820/S4837 later describes function names approximately3100→3450 and attributes3500→3950 in its25M trajectory, while still retaining the same aggregate1:30 CE6.90/rank409. Paper2 repeats those later approximate category comparisons. The category endpoints need re-extraction from the exact raw per-probe results; conversation prose is not an adequate source for those deltas.

### 14. Long50M runs were interrupted and resumed using saved weights

E L4161/S4169 describes adding `resume_step`, loading a safetensors weight checkpoint and fast-forwarding the dataset pointer for50M1:35 step5000. L4283/S4293 resumes again at10000. E L4427/S4437 resumes50M1:40 at5000, L4485/S4495 resumes50M1:45 at6000, and L4712/S4728 again at14000. The dialogue does not establish restoration of the Adam moments/RNG alongside those weights. Exact optimizer-state continuity needs source/checkpoint verification before calling these independent schedules uninterrupted and equivalent. This is a paper-methods limitation, not a renewed proposal to explain the later unified TPU–CUDA gap through batch accounting.

### 15. The user's original paper requirements already prohibited these overclaims

E L3574/S3578 explicitly requires separating observed results, engineering observations, interpretations, hypotheses, prior literature, invalidated and historical experiments; forbids merging old85M/125M/250M experiments with current Beta runs; calls the25M endpoint best-tested rather than optimum; and demands exact sources for figures. E L3638/S3642 further rejects universal scaling forms from two points, causal “optimal learning signal” wording, unlabeled category provenance and misidentifying MHA configurations as actually using GQA. These are useful editorial requirements to preserve in the eventual paper, not evidence that the requested controls or claims audit were actually completed.

### 16. Repository split into three self-contained paradigm stacks created code divergence

E L4934/S4944 and L4936/S4946 implement a directory restructuring where `mdiff/` (masked diffusion), `undiff/` (uniform diffusion), and `ar/` (autoregressive) became independent self-contained stacks rather than inheriting from shared modules. While this avoided coupling, it created divergence: bug fixes, tokenizer updates, and optimizer settings applied to one paradigm were not automatically mirrored in the others. Paper analyses comparing AR vs. MDLM vs. UNDLM must state the specific code revision and file implementation for each paradigm.

### 17. Single-step continuation probes do not measure multi-step iterative generation

E L7414/S7459 records the user's direct challenge: *"Does MDLM probe test only single step denoising or full iterative generation? WHY? ESPECIALLY WHEN WE'RE GONNA BE USING MULTI STEP ITERATIVE GENERATION FOR INFERENCE?"* The assistant acknowledges that all contextual probe benchmarks evaluate only single-step unmasking at fixed masking ratios ($t$), not multi-step iterative denoising loops. A low cross-entropy or high target rank under a single-pass probe does not demonstrate capability in autonomous iterative generation, where errors compound across steps. Paper claims must describe probes as single-step contextual cloze metrics, not full generative competence.

### 18. The 25M "Upscaling Initialization Shock" was a variance scaling bug, and source checkpoints were missing

E L7715/S7765 and L7738/S7788 document that early 25M models performed worse than 12.5M models across ratios. The assistant identified an "Upscaling Initialization Shock" caused by zero-padding weights without adjusting RMSNorm scaling. Furthermore, E L7998/S8017 explicitly documents that 12.5M 1:30 and 1:35 checkpoints did not exist; thus 25M 1:30 and 1:35 runs could not have been initialized from matched same-ratio 12.5M predecessors, introducing an initialization asymmetry across ratios within the 25M series.

### 19. Accelerator execution on Kaggle TPU v5e-8 and Lightning AI TPU v6e-1 introduced hardware confounds

E L8034/S8124 to L8593/S8616 document extensive Kaggle TPU v5e-8 failures: single-worker container constraints caused `xmp.spawn` crashes, which left `libtpu.so` holding hardware locks (`open(/dev/vfio/1): Device or resource busy`), misdiagnosed by the assistant as memory exhaustion. Moving to Lightning AI TPU v6e-1 (E L9268/S9289) encountered non-linear step-count scaling (1:10 vs 1:15 having $3\times$ steps), matrix-dimension padding requirements (multiples of 128/256), and artificial throughput degradation caused by redundant `torch_xla.sync()` calls that dropped TPU utilization to 70%. These engineering dynamics affected throughput and training stability, reinforcing that hardware environments cannot be treated as transparently interchangeable.

