# Unified trainer evolution: source-backed archive findings

Review date: 2026-10-18. This is a historical research audit, not a new training experiment or a diagnosis of the unresolved backend gap. Coverage is recorded separately in `coverage.md`; the main chat is still being reviewed. Source paths below are relative to the repository. `L` identifies the physical JSONL line, `S` the original `step_index`. Original step IDs are not always consecutive and sometimes recur after edited user turns, so line and step are both provided.

## Assigned sources

- U: `.agent-history/antigravity-ide/chats/84c05c0f-f22e-42ce-9600-57e1ed7ec419/transcript.jsonl`, **Unified Trainer Development Plan**, 16,775 records, September 1–13.
- O: `.agent-history/antigravity-ide/chats/e6c6c734-808a-495a-bb4a-e08af8cc2927/transcript.jsonl`, **Optimization Review For Transformers**, 964 records, September 6–7.
- R: `.agent-history/antigravity-ide/chats/99809d60-d4ff-4e3e-a457-624189bfa764/transcript.jsonl`, resume audit, 37 records, September 7.

An archived assistant assertion is treated as a claim, not automatically as a measurement. Original patches prove a code edit occurred locally; unit tests prove only their exercised path; user-pasted accelerator logs are historical observations. Matching a change to a particular saved checkpoint needs independent lineage evidence.

## Findings already supported by inspected original records

### 1. The word “unified” originally described a trainer, not the later joint COROSred objective

U L1/S0 requests one trainer for AR, MDLM, UNDLM and COROSred across MLX, CUDA and XLA. It does not establish that COROSred was already trained under a continuous joint loss. Early logs explicitly label COROSred Phase A/B/C. The substantive proposal to abandon sequential phases and train a weighted causal/infill/head objective continuously appears on **September 10**, U L9181/S9264 (repeated edited turn L9183/S9264). Implementation is announced at U L9548/S9633.

Consequently a September 5–9 checkpoint described as using the “unified trainer” may still be a legacy phased COROSred model. A paper must classify the objective using embedded config and actual loss implementation, not the conversation title, folder basename, or the presence of a generic unified trainer banner.

### 2. An early Phase A implementation omitted backbone language-model loss

U L3397/S3422 explicitly reports that `crsr_phase_a_loss_fn` returned reliability BCE without AR CE. Since the head consumed detached hidden states, training this objective from scratch supplied no backbone gradient. The reported fix is `L_AR + L_rel`, alongside auto-chaining Phase A weights into Phase B. This is a concrete historical implementation defect in the older phased architecture; it cannot explain a later verified continuous unified objective without showing that it survived there.

The record also contains repeated user corrections about what Phase A means (U L3342/S3367, L3362/S3387, L3368/S3393, L3374/S3399). These corrections are important design history. Do not infer consistent phase semantics across all dates.

### 3. A performance optimization actually changed the TPU diffusion timestep distribution

O L423/S422 says the TPU beta-buffer indexing was replaced with `0.5*(1-cos(pi*u))` for uniform `u`, claiming a stable bell-shaped sampler. O L726/S735 identifies the mathematical error: this is the arcsine distribution, Beta(1/2,1/2), not Beta(1.5,1.5). The original code patch at O **L839/S848** replaces that transform with pre-sampled float32 Beta(1.5,1.5) values accessed using device-computed indices; the inspected patch includes both the deleted transform and replacement.

This supports a narrow historical claim: an attempted XLA speed optimization briefly altered the noise distribution for the MDLM/UNDLM path. It does not establish an error in later fixed-mask unified COROSred training. Avoid folding this into the unresolved 50M joint-objective discrepancy.

### 4. Precision policy changed several times, and the archive’s explanations contradict one another

O L218/S217 announces explicit native BF16 TPU parameters as an optimization. O L726/S735 then reports lack of FP32 master weights; O L839/S848 adds FP32 master copies while retaining BF16 model parameters. O L947/S956 adds master-weight synchronization after checkpoint loading.

U L7923/S7985 and L8144/S8208 later attribute poor learning to a manual `p.data.copy_` bridge and “lazy graph aliasing,” without a controlled XLA update trace in those passages. U L8166/S8231 announces removal of master weights while retaining native BF16 parameters and asserts that the optimizer supplies FP32 moments automatically. U L8706/S8787 subsequently identifies the resulting native-BF16 parameter policy as an update-rounding problem; the inspected patch U **L8729/S8810** enables XLA BF16 autocast while keeping parameters FP32. U L8833/S8914 announces that policy on September 9.

The verified source edit establishes the FP32-parameter/BF16-autocast policy, not the literal claims that every earlier update was zero or that `.copy_` attenuated updates by a measured percentage. U L9012/S9093 reports moved `final_norm` values in retrained checkpoints, yet causal evaluation still remains poor (Phase A causal CE7.8585, Phase C6.3605 versus a differently trained AR checkpoint1.3407). This is evidence that “weights can update” is insufficient to demonstrate training parity.

For the current 50M audit, do not revive “parameters were BF16” after actual saved tensors and runtime policy have been checked. The archive contains distinct precision generations.

### 5. An actual local compiled-MLX learning-rate experiment passed

O **L734/S743** and **L819/S828** contain command output comparing eager and compiled MLX optimizer steps with LR changing from1e-3 to100. The printed result is `step1 matches eager: True` and `lr change picked up by compiled fn: True`. This supports that specific MLX compiled-LR test; it says nothing about XLA’s LR tracing or historical accelerator optimizer kernels.

O L867/S876 and L911/S920 show19 local Darwin tests passed. O **L957/S966** shows21 local Darwin tests passed after the resume-related changes. “All tests passed” in the surrounding assistant summary must not be reported as physical CUDA/TPU validation.

### 6. A real master-weight resume defect was patched; a later independent audit found it already fixed

O L917/S926 reports that loaded BF16 model weights could be overwritten by stale FP32 master copies on the first optimizer step. O **L947/S956** is the actual patch adding `mp.data.copy_(p.data.float())` after loading. R L31/S30, L33/S32, L35/S34 and L37/S36 inspect the code and conclude this synchronization, MLX wired-memory setting and SPMD throughput multiplier fixes are already present.

The resume audit also corrects a claim that nested `tpu:/mac:/gpu:/lightning:` config blocks were unused: builder profile inheritance actively consumes them (R L31/S30, L37/S36). Thus the phrase “dead config” in earlier assistant/user audit prose is not a stable fact across commits.

### 7. The continuous objective design has arithmetic and theory claims that should not enter the paper uncorrected

At U L9230/S9314 and L9231/S9315, the discussion treats causal and infill token counts as proving a roughly7× per-parameter gradient-strength mismatch under separate mean losses. Counts alone do not prove gradient norms or interference. U L9231/S9315 also contains `511+76=687`; the sum is587. These are design-discussion mistakes, not fresh empirical findings.

U **L9323/S9408** proposes processing **K×Bm upcoming sequences every K steps** for cached routing but divides the arithmetic overhead byK a second time, claiming0.16% atK50. If the refreshed cache contains distinct future sequences, its average extra model-forward work is `Bm` sequences per step, so under the passage’s own “forward1, forward+backward3” approximation the overhead is `Bm/(3B)`—8.3% atBm/B=.25, or16.7% at.5. Batching may reduce launch overhead, but it does not remove the arithmetic. U L9548/S9633 repeats the0.16% implementation claim.

This is a paper-cost-accounting correction, separate from the repeatedly investigated batch/data issue and not evidence of the backend convergence cause. Actual cost also depends on whether routing executes; the later routing-cache defect must be considered before assigning overhead to a real run.

The polynomial description in these passages also needs independent mathematical specification rather than narrative quotation: `a_min+(a_max-a_min)*(1-t)^p` withp>1 falls faster early on the decay segment, not slower. A preceding hold interval changes when the decay begins; it does not change that derivative.

### 8. Performance guarantees were repeatedly disproven by the next accelerator log

O L532/S531 interprets one benchmark as full TPU saturation and zero recompilation, then the real AR training at O L533/S532 OOMs. O L618/S617 identifies XLA backward padding of separately sliced chunked CE as the cause and bypasses chunking on XLA. O L725/S725 addresses `torch.utils.checkpoint` trying to access `torch.xla`. These are genuine backend integration concerns, but the specific causal proof varies: an allocation traceback is stronger than a throughput-based statement of saturation.

U L9966/S10042 attributes host spin to a `libtpu` flag and says it was applied. U **L10842/S10920** subsequently acknowledges that the flag is unrecognized by the deployed runtime and caused startup aborts. U L10091/S10168 labels several CUDA/XLA paths “Zero Risk” after CPU/MLX smokes; later user logs expose further issues. A paper should avoid retrospective wording that these changes were accelerator-validated when they were not.

U L10975/S11053 also compares a sum of allocations across8cores to16GB per-core HBM to assert a host spill cause; the cross-core sum does not establish per-core capacity overflow. Treat that host-memory explanation as a historical hypothesis unless independent memory measurements establish it.

### 9. The September 11 50M joint run had explicit interruptions and changing resume semantics

U **L12715/S12727** records C-CE1.882, I-CE1.375 atstep6025, then an immediate pause due host-RAM growth. U L12732/S12744 says checkpoint6000 is saved; U L12734/S12746 says uploaded. U L12892/S12904 announces restoring optimizer state on resume. The next user log U **L12893/S12906** reports C-CE3.379/I-CE3.224 at6025 and very slow steps.

U L12924/S12939 then attributes this to CPU optimizer-state stranding and missing EMA telemetry and disables optimizer-state restoration on TPU. The claim that load_state_dict necessarily left moments stranded on CPU is not established merely by a CPU map_location; it needs the actual loaded-state/device evidence. Do not cite that narrative as a proven cause.

U **L13014/S13031** proposes replacing checkpoint saving with a CPU model-state snapshot **without optimizer moments**; U L13030/S13047 announces that change applied. This gives a historical reason why later TPU checkpoints may omit optimizer state. It also means repeated warm resumes are not exact training resumes. It does not establish that the later distinct50m_python model had this interruption history.

U L13031/S13048 asks about an explicit gamma0 restart; U L13052/S13069 recommends gamma.1 instead. An assistant recommendation is not proof of the exact command ultimately executed. Preserve this distinction when assigning gamma provenance.

### 10. Double reduction was correctly retracted, then incorrectly resurrected as an explanation

U **L14069/S14083** explicitly corrects two diagnoses: mean-reducing identical already-reduced gradients again is mathematically a no-op, and a proposed tying-order change was not demonstrated causal. U **L14192/S14206** repeats the correction and restores `xm.optimizer_step` while changing the automatic checkpointing threshold.

Yet U **L16317/S16310** later asserts that double reduction divides gradients by64, reintroducing the same rejected explanation. This is algebraically inconsistent with the mean-of-identical-replicas correction. It should not be retained as paper evidence.

U L14069/S14083 proposes activation checkpointing as the “actual” cause; U L14192/S14206 calls it the “true smoking gun,” but those excerpts provide no same-state TPU on/off gradient comparison. It remains a historical hypothesis, not an identified cause of the50M gap. In particular, the purported50M case is claimed not to use that checkpointing path.

### 11. Claims about attention overflow and stale loss logs were confident, but not measured there

U **L16317/S16310** and **L16353/S16346** attribute the poor100M TPU trajectory to unfused attention exponential overflow, absent GradScaler and lazy clipping; they also claim `.item()` can return stale cached HLO constants. The displayed loss trajectory is evidence that checkpoint evaluations differ; it is not an operator trace proving those mechanisms. Stable softmax does not imply naive unshifted exp simply because FlashAttention is absent. BF16 CUDA autocast also does not generally imply a GradScaler is active. Treat these as unverified historical stories.

These passages should be paired with actual saved-checkpoint measurements if retained. They cannot diagnose later unified50M convergence on their own.

### 12. Model lineage was repeatedly confused, including hardware labels

Examples include U L6530/S6613 (a mistakenly selected month-old Phase C checkpoint), L7113/S7199 and L7140/S7226 (overwritten/deleted50M paths), L15284/S15274 (multiple Phase C15M models versus one unified15M), L16226/S16217 (12.5M1:30 mistaken for15M1:50), and U L16317/S16310 acknowledging that wrong selection.

A particularly important hardware counterexample: U L7924/S7986 says the100M AR run stopped on TPU atstep12000 and continued on H100; U L8067/S8130 says it uploaded. Later U L10317/S10395 and L16353/S16346 describe the healthy100M1:40 baseline as H100-trained. A full-lifecycle ledger is needed before calling it CUDA-only. User correction U L16354/S16347 says it was TPU too, consistent with the earlier mixed-backend continuation history.

This does not invalidate the later correctly identified50M CUDA/TPU comparison. It invalidates inferring provenance from old assistant tables without checking the actual chain.

## Implications for a future paper

The archive supports an engineering-history narrative: phase semantics and loss alignment changed; real masking/precision/resume/backend integration bugs existed; local tests did not certify physical accelerator equivalence; and successful evaluations were sometimes attributed to the wrong checkpoint. It does **not** support declaring the unresolved backend gap to be intrinsic TPU numerical weakness.

Use distinct experiment rows for: legacyPhaseA/B/C, initial joint objective, later loss normalization/cadence fixes, and any future controlled parity run. Never combine old failed models with later joint-objective checkpoints merely because both are “COROSred50M.” Archive findings are leads and provenance corrections; final quantitative claims need checkpoint hashes, source revision, exact evaluation contract, data identity and reproducible raw results.

## Additional findings from early-construction review

### 13. Canonical RoPE and model reconstruction were explicit design changes

U L104/S103 approves retraining rather than preserving old state-dict compatibility and accepts standardizing MLX RoPE to `traditional=True` to match the PyTorch formulation. The same turn approves a separate phase flag. Thus old checkpoints cannot be presumed mathematically compatible solely from matching parameter shapes.

U L781/S783 reports that the early evaluation loader failed to restore `is_causal` and reliability-head presence and used `strict=False`. U L1049/S1051 announces stricter loading, but U L1693/S1698 then identifies that causal metadata still was not persisted and that path inference could silently change behavior after a rename. U L1903/S1908 announces persistence of paradigm/causality and embedded-config fallback. These are evaluation provenance defects, not proof that a specific headline table was wrong: the old notebook and CLI evaluators differed, and that distinction must be recovered for each result.

### 14. CUDA fused AdamW was initially a silently inactive optimization

U L1904/S1909 points out that checking `hasattr(torch.optim.AdamW,"fused")` tests a class attribute rather than the constructor signature. U L1952/S1957 reports the local check returnedFalse and changes detection to `inspect.signature`. This is useful optimizer dispatch history; it is not an explanation for a later backend deficit after dispatch was separately verified.

### 15. Version assumptions were already unstable in the initial audit

U L1086/S1089 calls XLA autocast unsupported and U L1241/S1244 disables it, describing native TPU BF16 as sufficient. September6–9 then goes through explicit BF16 parameter casting, master copies and ultimately FP32 parameters with BF16 autocast(finding4). These archived statements are contradictory integration advice, not authoritative runtime documentation. The actual local patches and historical deployed revision are stronger evidence than the conversational explanation.

U L1336/S1339 also says an absent `__init__.py` necessarily prevents Python from recognizing a directory as a package; namespace packages make that blanket assertion too strong. Adding initialization for predictable packaging can still be appropriate. This illustrates why paper-facing methodological claims must be independently checked even when the engineering change was harmless.

### 16. PJRT computation client singleton crashes required runtime process caching

U L2100/S2107 documents a fatal TPU `SIGABRT` caused by PyTorch/XLA's PJRT C++ runtime: `InitializeComputationClient()` enforces a hard singleton assertion (`ABSL_CHECK(!g_computation_client_initialized)`). Running hardware auto-detection in `builder.py` followed by `UnifiedPyTorchTrainer` instantiation invoked the client twice, crashing multi-phase runs and notebook kernels. The solution created `xla_utils.py` to enforce a process-lifetime device singleton. This is a systems engineering defect, not a model convergence issue, but it explains why early multi-phase training scripts failed on Cloud TPU VMs.

### 17. Multi-backend training coordination led to lineage fragmentation and frozen metrics

Across U L5129/S5226 to L13658/S13670 (Chunks 021–068), training was opportunistically shifted between Kaggle TPU v5e-8, Lightning AI TPU v6e-1, Modal H100, and local Mac MLX due to TPU quota exhaustion and queue delays:
- **Mixed-hardware trajectories:** The 100M AR run was trained on TPU up to step 12,000, then resumed and continued on an H100 GPU (U L7924/S7986, L8067/S8130), confounding hardware-specific numerical analyses.
- **Lineage confusion:** Stale month-old Phase C checkpoints were repeatedly selected as baselines by the assistant (U L6530/S6613), and file paths were overwritten.
- **Metric freezing on accelerator path:** In U L11646/S11723, the user discovered that the `AUC per step` metric in the COROSred training logs was completely frozen at `0.7500` across all 3,815 training steps. The metric tracker update had been omitted from the accelerated TPU loop, proving that logged training telemetry did not reflect live calibration dynamics.
- **Dataset memorization concerns:** In U L7508/S7595 and L16674/S16667, the user noted validation perplexity falling to 2.14 ("near memorization territory") and explicitly questioned whether the evaluation and training splits were contaminated.

### 18. Functional benchmarks revealed 0% Pass@1 despite favorable token ranks

In U L16472/S16466 to L16775/S16768 (Chunks 079–081), comprehensive evaluation of 15M, 50M, and 100M models across AR and COROSred demonstrated that **Pass@1 was 0% across all models and tasks** on full functional execution benchmarks (U L16585/S16578). Furthermore, individual token evaluations revealed severe asymmetric divergence: 50M COROSred achieved rank 1 on causal continuation prompts while collapsing to rank 23 on bidirectional infilling for the exact same variable contexts (U L14760/S14755). This provides concrete historical proof that low cross-entropy and high single-token rank do not translate into executable code generation competence.

