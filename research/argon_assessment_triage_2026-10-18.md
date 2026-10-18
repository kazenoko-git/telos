# Triage of Gemini Argon’s TPU/CUDA assessment — 18 October 2026

Source: user-supplied assessment in `/Users/ivansamuel/.codex/attachments/64310357-9ba6-4373-ac98-47eb2fa1fefb/Pasted text.txt`. This is a bounded source/document review; no new training, checkpoint evaluation or hardware experiment was run. No Codex review agents were resumed.

## Accepted correction: endpoint content identity

The pinned inventory `scratch/backend_parity/hf_50m_inventory.json` lists HF `50m_python/checkpoint_final.pt` with SHA-256 `5e62ca9fedc1aaf2ad3951c5a4f7da6085bbf661cf0b25b4cb28bda0c69ae781`. The handoff’s candidate local `50m_tpu_lrcadence_schedstep1` endpoint has that same hash. The original intermediate report incorrectly said this endpoint was absent from HF without distinguishing path from content. Corrected `intermediate_50m_inspection_2026-10-18.md` accordingly. Argon’s proposed local overwrite/move history and uninterrupted-run claim remain unverified; file identity alone does not establish either.

## Hypothesis 1: replica initialization remains unproven

Different final weights across independent training runs do not demonstrate different initial weights across replicas *within* a run. Cosine and relative L2 are also mathematically related measures of the same vectors, not two independent signatures of initialization. Missing explicit broadcast is a reason to inspect launch/initialization semantics, not proof of replica divergence. Actual launch topology, source revision, inherited RNG/checkpoint state and per-replica initial tensors are missing.

Argon’s replica-offset decay equation is conditional on identical reduced gradients, optimizer states, hyperparameters and update paths. It is not a measurement. Likewise, head-weight similarity across checkpoints could inform initialization and head activity, but cannot establish within-run replica mismatch. No new forensics were executed here.

## Hypothesis 2: a concrete conditional code hazard, not an established historical cause

Current `trainer_pytorch.py:113–118` catches failures of `xm.get_ordinal()` and assigns rank0. Lines704–723 derive rank-based data offsets from `self.rank` in replicated mode. Thus, if that API were missing in the actual runtime and this replicated code path ran, rank/data duplication would follow. SPMD explicitly handles world/rank differently. The historical runtime version, launch mode and affected run revision remain unknown. This triage does not independently verify Argon’s asserted API-removal version.

Its claimed eightfold effective-step match is not a controlled fit. Duplicated samples do not make optimization exactly equivalent to dividing step count by eight: update count, LR trajectory, repeated exposure, stochastic corruption and optimizer state still matter. Identical corruption RNG across ranks is not by itself duplicate-data evidence when ranks process different sequences. Identical model initialization across data-parallel replicas is desirable.

## Hypothesis 3: genuine recipe confounds, unsupported causal attribution

The historical loss-normalization change was already documented in `tpu_cuda_history_recovery_2026-10-18.md`. A change in the intended task weighting is not automatically a backend defect. Upload dates do not prove the exact trainer revision or chronology of training. A fixed9.1x factor depends on task fractions, mask realization and normalization conventions; it cannot explain the causal deficit by assertion.

A gated head can lack direct training signal while gamma is zero. However, its input representation still evolves, so its metric need not remain fixed; threshold crossing is not mathematically impossible. A high final evaluation head metric does not prove when gamma opened in training, especially if metric definitions differ. Existing cadence audit found exact zero backbone-gradient change before clipping when toggling gamma under its fixtures; head changes can still affect global clipping. The task remains unresolved.

## Diagnostic criticisms: mostly stated scope limits

The cadence replay deliberately supplies synthetic observations and labels them synthetic; it was not presented as an actual TPU trajectory. The common-fixture endpoint-gradient comparison explicitly does not reconstruct the historical objective or updates. Single-replica eager tensor captures cannot validate multi-replica launch behavior or a production fused XLA graph; those are scope limits and reasons for an additional experiment, not proof that the original measurement was invalid.

The history report already records evaluation of existing `50m_seed` weights; it does not claim a controlled paired retraining after all fixes. Seeded-run improvement alone would also not isolate initialization if other recipe/code changes occurred.

## Conclusion

One confirmed reporting correction; one conditional rank/API hazard worthy of checking against historical runtime evidence; initialization and recipe hypotheses remain unproven. Argon has not established the cause of the TPU/CUDA quality gap. The archive review can incorporate these distinctions without launching another broad round of expensive diagnostics.
