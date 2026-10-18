# Independent investigation: Telos 50M unified COROSred TPU–CUDA quality gap

Prepared 18 October 2026. This document is self-contained: it includes the key observations, historical context, checkpoint metadata, measurements, relevant source code, diagnostic implementations and source references. You do not have access to the author's filesystem. Local paths below identify provenance; their relevant contents are embedded here. Raw checkpoint weights and token corpora are not embedded.

## Request to the independent investigator

Investigate why the TPU-trained 50M unified COROSred models substantially underperform a CUDA-trained counterpart. Previous investigators have not established the cause. Do not adopt their conclusions as settled. Audit the evidence and code independently, including the diagnostic code itself.

Return:

1. Your best-supported explanation, or an explicit statement that the evidence does not identify one.
2. At most three ranked hypotheses, each with the exact mechanism, relevant code, supporting evidence, conflicting evidence and a predicted observable result.
3. Any errors in previous experimental interpretation or diagnostic implementation. A defective diagnostic can invalidate an inference without explaining the training gap.
4. The smallest discriminating next test for each surviving hypothesis. Distinguish a test that can run locally now from one that actually requires a TPU. Avoid proposing another full training run by default.
5. A clear distinction between a confirmed shared implementation defect and a demonstrated backend-specific cause.

The user has been investigating this for a long time with Gemini, Claude and Codex. Effective batch size, token/data accounting and LR scaling have already been investigated repeatedly. Do not restart generic accounting checklists. Reopen an earlier exclusion only if you identify a concrete overlooked mechanism or evidence that invalidates it. The prior exclusions are user-reported historical findings, not a fresh complete independent proof supplied in this packet.

FP32 parameters with BF16 autocast are an intentional performance choice on both training paths. Do not assume the TPU optimizer updates BF16 weights merely from the configuration saying bf16. Do not prescribe blanket FP32 training as the answer. Kaggle TPU access is scarce and currently unavailable to the user. The local Mac has CPU/MPS, no CUDA and no torch_xla; Kaggle credentials are not configured in the investigation session.

## System and architecture

Telos is a custom PyTorch language model. COROSred unified training combines causal next-token prediction, bidirectional masked-token infill and a learned reliability head in one loop. It is not a pretrained Gemma/Qwen evaluation problem. Earlier conversations about Gemma/Qwen throughput are unrelated to this training discrepancy.

The eligible compared models share:

- Vocabulary 8,192; context 512; width 512; 14 transformer layers; 8 query heads and 8 KV heads; tied embedding/output weights; RoPE; pre-RMSNorm; SwiGLU; reliability head enabled.
- Current strict reconstruction counts **49,430,017 unique parameters**, including the reliability head. Embedded `_resolved_params=49,166,848` is a different recorded count; do not confuse that field with the measured unique-parameter count or silently interpret this discrepancy as untied embeddings.
- Saved final global step 12,716.
- Max LR 0.0004242640687119285; min LR 0.000042426406871192855; warmup 254 steps; AdamW betas (0.9, 0.95), epsilon 1e-8; weight decay 0.1 for decayed parameters; global gradient clipping threshold 1.
- Intended FP32 parameters, BF16 autocast forward/loss. Current code disables manual master copies, re-ties after device transfer and deduplicates optimizer parameters.
- Unified settings: alpha max 0.85, alpha min 0.50, beta min 0.15, beta max 0.35, gamma max 0.10, hold fraction 0.20, polynomial power 2.5, gamma ranking threshold 0.55, fallback accuracy threshold 0.65, causal ratio 0.65, mask probability 0.20, ambiguity top-k 5, routing-cache refresh interval 50, adaptive rebalancing false.
- CUDA configuration records batch 192 and accumulation 2, one device. TPU records batch 48 and accumulation 1, eight devices. These facts are included for completeness, not as a request to repeat accounting work.
- Corpus filenames differ between historical configurations; filenames alone do not establish content differences. No causal claim from this distinction has been established.

Do not confuse the eligible unified models with older separate Phase A/B/C models. Those earlier models failed badly and are excluded. Another old experiment named `50m_2.5b` has different task settings and unequal saved embedding/output tensors; it is excluded from the backend comparison even though its configuration has a unified marker. Its optimizer moments are not used as evidence of the later TPU run's behavior.

## Checkpoint identities and lineage warning

Public model repository: https://huggingface.co/Kazenowoko/telos

Pinned inventory revision: `d728840aada52094748903ddc56256807a0bfd61`.

Relevant checkpoint labels:

- CUDA: `checkpoints/corosred/50m_lightning/checkpoint_final.pt`.
- TPU baseline endpoint: `checkpoints/corosred/unified/50m_python/checkpoint_final.pt`.
- TPU objective-cadence-1 endpoint: `checkpoints/corosred/unified/50m_tpu_schedstep1/checkpoint_final.pt`.
- TPU LR-and-objective-cadence-1 candidate endpoint: `checkpoints/corosred/unified/50m_tpu_lrcadence_schedstep1/checkpoint_final.pt` (available locally; absent from the pinned HF inventory).

All three endpoints in the original comparison store FP32 tensors and equal embedding/output tensor values. Equality in a saved state does not establish original in-memory aliasing. CUDA saves optimizer moments; the eligible TPU endpoints/intermediates omit them.

Crucially, **the `50m_python` directory cannot be assumed to represent one continuous run**. Its step-4,000/8,000/12,000 files explicitly record LR/objective cadence 1; its final file records LR cadence 10 and lacks those overrides. The step-12,000 weights have relative L2 distance 0.024999 and cosine 0.999688 to the local LR-and-objective-cadence-1 final, but distance 1.240007/cosine 0.228330 to the final stored beside them. The separate matching endpoint is a strong lineage candidate, not a proven resume chain. Weight proximity alone is not a training manifest.

The eligible HF files were hash-verified locally against the pinned remote inventory. CUDA files were uploaded 14 September; TPU Python files 18 September 2026. Upload dates are not training dates. The saved files do not identify exact source commits or runtime versions.

## What actually happened in training, versus present-day evaluation

Older validation trajectory records at step 12,716:

| Endpoint | Historical recorded causal CE | Historical recorded infill CE |
|---|---:|---:|
| CUDA | 1.3870304823 | 1.0212213993 |
| TPU baseline | 1.9093503952 | 1.7630840540 |
| TPU objective-cadence-1 | 1.8766576052 | 1.6967291832 |

These are retained historical records, not regenerated with today's evaluator. Exact original validation inputs/implementation are not reconstructed. New evaluations below use different specified inputs and must not replace these numbers in the historical trajectory.

## New endpoint measurements, actually executed locally

Common CPU FP32 evaluator; dropout/checkpointing disabled; explicit causal attention for next-token prediction and explicit bidirectional attention for infill; no optimizer update. Two corpus slices of 16 sequences each, length 512, at token offsets 6 billion and 7 billion in local `python_corpus_15b.bin`. Uint16 interpretation was explicit, consistent with the earlier scripts; a sidecar proving this format was unavailable. Document-level absence from training is not established.

Four microbatches per slice, same seed-42 corruption mask across models; probability 0.20, first position preserved. Causal and infill gradients are separately token-weighted means over the entire slice. Reliability loss excluded. Geometry uses normalized alpha 0.50 and beta 0.35. This is the current declared task objective, not a reconstruction of historical training loss.

Pooled across both slices:

| Endpoint | Causal CE | Causal top-1 | Infill CE | Infill top-1 |
|---|---:|---:|---:|---:|
| CUDA | 1.56465 | 67.01% | 1.08619 | 77.05% |
| TPU baseline | 2.06959 | 60.68% | 1.78634 | 67.55% |
| TPU objective-cadence-1 | 2.04450 | 61.14% | 1.71204 | 68.20% |

CUDA has lower loss for both tasks in all eight paired microbatches against both TPU endpoints. All measured gradients were finite. Overall causal-versus-infill gradient cosine within each model was positive on both slices. No TPU layer group showed negative cosine. CUDA had one slightly negative layer on slice 2 (layers.1, -0.01255). The infill weighted gradient norm exceeded the causal weighted norm in every endpoint, including CUDA. No TPU-specific endpoint task-conflict pattern was found. None of these observations establishes what happened during historical training.

## Intermediate measurements, same first-slice evaluator

| Checkpoint | Step | Causal CE | Infill CE | Within-model task-gradient cosine |
|---|---:|---:|---:|---:|
| CUDA | 1,271 | 2.20181 | 2.40955 | 0.1571 |
| CUDA | 3,813 | 1.74667 | 1.65492 | 0.1252 |
| CUDA | 7,626 | 1.55809 | 1.23012 | 0.0874 |
| CUDA | 11,439 | 1.48908 | 1.09837 | 0.0968 |
| TPU cadence-1 intermediate | 4,000 | 2.43701 | 2.82393 | 0.2688 |
| TPU cadence-1 intermediate | 8,000 | 2.11615 | 2.11556 | 0.2155 |
| TPU cadence-1 intermediate | 12,000 | 1.99400 | 1.80241 | 0.1678 |
| TPU cadence-1 candidate final | 12,716 | 1.99400 | 1.77821 | 0.1858 |

The deficit is present by the earliest eligible TPU intermediate at step 4,000, even relative to CUDA step 1,271. This does not locate the first divergent update. TPU continues improving. These sparse observations do not prove an irreducible loss floor, establish continuous-run lineage, or rule out between-save instability. Nearby steps are not exact matched steps.

## Cadence findings and their limits

There are distinct LR, objective-step and metric-feedback cadences. `sched_cadence=1` alone did not isolate LR cadence: the objective-only endpoint still saves LR scheduler cadence 10. The LR-and-objective-cadence-1 candidate saves LR cadence 1 and still has a deficit. No independently recovered complete metric-cadence trace exists.

A synthetic replay fed identical per-step metrics: 0.50 until step 999, then 0.60. This is a deliberately invented step response, not recovered training telemetry.

| Tracker | Metric-update cadence | First gamma-active step |
|---|---:|---:|
| Historical EMA | 1 | 1,069 |
| Historical EMA | 25 | 2,701 |
| Current cadence-adjusted EMA | 1 | 1,069 |
| Current cadence-adjusted EMA | 25 | 1,051 |

Current correction is `decay_eff=max(0,1-cadence*(1-0.99))`; it is a linear time-constant approximation, not exact equivalence to sampling all fine-grained measurements. Alpha/beta were identical across these metric-only replays because nominal alpha/beta depend on progress rather than reliability metrics, with adaptive rebalancing disabled.

Quantized LR cadence 10 changes summed applied LR by +0.0580%; cadence 25 by +0.1547%. Max nominal alpha differences from objective-step quantization: 0.000773 at cadence 10, 0.002059 at cadence 25. Small schedule differences alone do not prove small long-run training effects.

Actual source inspection and controlled backward tests found:

1. Reliability scores use `reliability_head(h.detach())`; BCE has no direct gradient into the backbone.
2. The trainer refresh guard calls `schedule.get_weights(step)` **without EMA arguments**. That always returns mask_blend 0. A new empty routing cache is therefore not populated through this guard. Both the inspected historical and current sources share this defect.
3. With an empty cache, random-mask fallback ignores the requested positive blend.
4. In a tiny controlled model, changing gamma 0 to 0.1 and blend 0 to 0.8 left backbone gradients exactly identical before clipping, in both current and representative historical loss implementations.
5. Global clipping includes head and backbone. Head gradients can therefore indirectly rescale backbone updates when clipping is active. Clipping threshold 1 was inactive in the tiny probe; forcing threshold 0.25 demonstrated this indirect effect. The measured max change 0.00009583 is a tiny-model mechanism check, not the historical 50M effect size.
6. CUDA/CPU validation probes additionally feed ROC-AUC into the tracker at checkpoint boundaries; multi-core TPU skips these probes. Training accelerator feedback uses balanced accuracy as its ranking proxy. This is a real source asymmetry, not a demonstrated explanation of the gap.

Do not conclude that a shared routing defect explains TPU-specific degradation, or that the detached head proves all cadence effects impossible: the global-clipping route remains.

## Optimizer and numerical probes actually executed

A fixed 50M CPU probe from the CUDA final used eight 64-token sequences, a frozen mask and weights alpha .7/beta .3/gamma .1. Independent captures exactly matched across 267 recorded forward/gradient/update tensors. Diagnostic loss 1.6790300608; preclip norm 7.4013557434. This was a fresh-optimizer, one-device operator probe, not a distributed trainer or original resume. A CPU BF16 smoke probe detected differences, confirming the comparator was not always returning zero. No actual TPU/CUDA capture exists.

A subsequent CPU-only ordinary-versus-fused AdamW replay used the CUDA final's saved optimizer moments at step 12,716, independently cloned weights/states and the same captured FP32 gradient tensors. Eight prescribed scaled/sign-changed gradients were applied, seven clipped at threshold 1 and one unclipped. All **49,430,017 parameters and both moments matched bit for bit after every update**. Dispatch instrumentation confirmed 16 ordinary and 16 fused calls. Aliasing checks and tests verified states were independent; deliberately different gradients produced detectable differences. Parameter group sizes and moment shapes match current order, but original checkpoint has no per-parameter name manifest.

This tests CPU implementations in local PyTorch 2.13.0. It does NOT test CUDA fused kernels, XLA lowering, distributed reduction, original TPU moments, or BF16 forward/backward execution. Treating it as accelerator parity would be an error.

A separate CPU float64 algebra check compared gradient averaging with scaled sequential accumulation: 20 AdamW updates, eight prescribed contributions, identical aggregate gradients/states produced zero parameter/moment differences. Re-averaging identical replicas changed values by at most 2.22e-16. This rejects a generic algebraic "distributed Adam variance damping" claim, not actual runtime defects.

## Historical investigations and corrections

The retained 19 September user statement reports prior investigations of FP32 causal-mask infinities, identical loss coefficients across cores, loss-pooling algebra, fixed task-slot effects, lexical skew, effective batch/token/LR accounting and coherent raw model outputs. Precision and FP32/master-weight handling had already been investigated. These are coverage reports; the packet does not recreate every historical experiment.

Earlier proposed explanations were overclaimed:

- Distributed averaging before Adam does not intrinsically change moments compared with equivalent accumulation when resulting gradients/state are identical.
- Repeated averaging of already identical gradients does not introduce a universal additional factor 1/world_size. A current trainer comment claims otherwise; it is mathematically misleading. Repeated collectives can still affect communication/numerical rounding.
- `xm.optimizer_step` is not known to supply mysterious additional optimizer graph tracking. Reviewed v2.8 code reduces gradients, calls optimizer.step, and optionally synchronizes. Exact historical installed build was not recovered.
- BF16 output logits do not prove BF16 cross-entropy intermediates under autocast, much less actual underflow.
- An earlier gradient audit was canceled before final tables yet followed by a claimed diagnosis. It also omitted explicit bidirectional attention and compared 50M CUDA against larger TPU models. Its claimed resolution is not trusted.
- A claimed universal copying threshold around CE 1.4 is unsupported.

Verified source chronology:

- 14 September: `29840d2`, on-device re-tying and optimizer deduplication.
- 17 September: `a2f09a0`, explicit cadence controls; used as representative historical source in the local audit, not proven source for every checkpoint.
- 20 September: `def9953`, seed initialization and XLA RNG synchronization.
- 22 September: `a8a2386`, configurable shared cadence defaults; `ebc7cae`, corrected task-mean objective; `3c1331b`, cadence-dependent EMA; `44ccf63`, pass actual metric cadence.

The reviewed October 6 archive recap still proposed new training with corrected code. Completed evaluations of older weights do not establish successful paired post-fix retraining. No such completed paired retraining was found in the reviewed archive segments; this is not proof none exists anywhere.

## Historical versus current objective

Representative historical code at a2f09a0 uses:

```python
w_c = B_c / B
w_m = B_m / B
causal_seq_norm = causal_loss_sum / (B_c * (T - 1))
infill_seq_norm = infill_loss_sum / (B_m * (T - 1))
pooled = (alpha*w_c*causal_seq_norm + beta*w_m*infill_seq_norm) / (alpha*w_c + beta*w_m)
loss = pooled + gamma * reliability_bce
```

Current corrected code uses:

```python
causal_mean = causal_loss_sum / causal_target_count
infill_mean = infill_loss_sum / masked_target_count
pooled = (alpha*causal_mean + beta*infill_mean) / (alpha + beta)
loss = pooled + gamma * reliability_bce
```

New endpoint-gradient measurements use separate current task means, not the historical pooling. A common historical objective defect is not automatically a backend differentiator.

## Environment evidence

The old Kaggle bootstrap cell shown by the user clears selected stale TPU environment variables, kills lingering VFIO device holders, clones/fetches the repository and resets to origin/main, then executes `pip install -q --no-deps -e .`. The screenshot shows clearing `TPU_PROCESS_ADDRESSES`, `CLOUD_TPU_TASK_ID`, `XLA_PERSISTENT_CACHE_PATH` and `LIBTPU_INIT_ARGS`. The screenshot is cropped; do not infer the complete cleared-variable list.

It preserves Kaggle's preinstalled Torch/XLA stack rather than upgrading dependencies. This establishes the intended environment source, not exact torch/torch_xla/libtpu versions. Pulling origin/main means identical notebook cells could execute different Telos commits. No usable historical runtime-version output was found in the targeted archived chats. Do not demand that the currently away-from-home user produce inaccessible Kaggle output.

## Documented external precedents: context, not diagnosis

- https://github.com/pytorch/xla/pull/8094 — merged September 2024, enabled FP32 cross-entropy autocasting. Example HLO (AWS Neuron) shows why a final FP32 loss could hide BF16 intermediates. It is a shared XLA policy fix, not a reproduction of this TPU failure. Reviewed v2.8 policy already includes it.
- https://github.com/pytorch/xla/pull/7878 — August 2024 norm-result dtype fix under AMP, tagged XLA GPU. Confirms historical integration defects; does not establish a TPU clipping bug in our runtime.
- https://github.com/huggingface/pytorch-image-models/discussions/934 — researchers reported lower short-run TPU DeiT accuracy; recipe/LR differences complicated the comparison, and longer training reportedly caught up. Not proof of an inherent TPU penalty.
- https://github.com/pytorch/xla/issues/5755 — unresolved visible report of accuracy drop after TPU load/save, even without training. A checkpoint report, not the same demonstrated convergence mechanism.
- https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html — numerical/kernel differences are documented; ordinary rounding differences alone do not explain this sustained observed gap.
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html — attention implementations can differ numerically. Telos uses this API; no kernel-level comparison has been captured.
- https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/core/xla_model.py — reviewed reduction/optimizer wrapper semantics.
- https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/csrc/autocast_mode.cpp — reviewed FP32 loss autocast registrations.

## What the evidence currently permits

A substantial learned-quality deficit is confirmed under a common evaluator. It is already present by the earliest eligible TPU save. Simple LR/objective cadence-1 experiments did not remove it. No TPU-specific endpoint task-gradient conflict was found. Several shared code defects and historical experimental interpretation problems are documented. Actual matched CUDA-versus-XLA forward/backward/reduction/clipping/update captures are absent.

The remaining possibilities include an unmeasured runtime/operator/precision difference, historical execution differences not recorded in configs, or effects of optimization trajectories not isolated by endpoint tests. This packet does not establish which is responsible. Challenge those possibilities rather than rebranding uncertainty as "TPU quirk" or claiming a cause merely because its mechanism is conceivable.

The appendices below provide the actual source and compact machine-readable evidence so you can reason beyond this summary.


# Appendix A — Checkpoint metadata and machine-readable measurements

These are compact extracts of actual local artifacts. Numbers are preserved at recorded precision. They are measurements, not an independent accelerator comparison.


## A1. Common evaluator contract

```json
{
  "purpose": "Saved endpoint diagnostic, not runtime backend parity or training causality",
  "device": "cpu",
  "compute_dtype": "float32",
  "torch": "2.13.0",
  "seed": 42,
  "threads": 2,
  "sequence_length": 512,
  "data_path": "/Users/ivansamuel/Documents/Projects/telos/data/python_corpus_15b.bin",
  "offset_tokens": 6000000000,
  "data_dtype": "uint16 (explicit assumption matching prior audit)",
  "evaluation_sequences": 16,
  "evaluation_tokens": 8192,
  "input_sha256": "3f4ad4c384eb9f556e34c6b75ce24eeeefe1886fcaaa6cd0fad8f0b05edd5410",
  "mask_sha256": "154efef123c0c15e0c18705ac8bf9dd707ea550ef3ce2baf2df3945b6b772e6b",
  "fixture": "logs/saved_50m_gradient_comparison_20261018.fixture.pt",
  "alpha": 0.5,
  "beta": 0.35,
  "reliability_loss": "excluded; compare causal and infill backbone gradients only",
  "gradient_aggregation": "token-weighted mean over the complete fixture per task",
  "source_sha256": {
    "scripts/compare_saved_50m_gradients.py": "56ee4909e26f099d1e58edbf32b1ad15aa529df7ead37755d407e7d8999859a1",
    "telos/models/transformer.py": "7bb232b4f5d95dd22f8d5ba903672eccd4cc4e6c2985ef20e9a4a751f79b4524",
    "telos/models/components.py": "ed3f82e902a2d94d2e9df3179516c794d0035a3fee51417d2555fa437458f0b6"
  },
  "git_commit": "38dbd652c4a4821b59979a97bc45aeb276848390"
}
```


## A2. Full endpoint metadata/configurations

```json
{
  "cuda_lightning": {
    "checkpoint": "/Users/ivansamuel/Documents/Projects/telos/checkpoints/corosred/50m_lightning/checkpoint_final.pt",
    "checkpoint_sha256": "e2e560508672880c45c72d9e28f459a7d3ca649032aa31cad12dccebb3b332ee",
    "step": 12716,
    "config": {
      "model": {
        "vocab_size": 8192,
        "seq_len": 512,
        "max_seq_len": 512,
        "d_model": 512,
        "n_layers": 14,
        "n_heads": 8,
        "n_kv_heads": 8,
        "tied_embeddings": true,
        "use_reliability_head": true,
        "mask_token_id": 1,
        "paradigm": "corosred",
        "is_causal": true
      },
      "training": {
        "compile": false,
        "batch_size": 192,
        "gradient_accumulation": 2,
        "max_steps": 12716,
        "max_lr": 0.0004242640687119285,
        "min_lr": 4.2426406871192855e-05,
        "warmup_steps": 254,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "precision": "bf16"
      },
      "checkpoint": {
        "checkpoint_dir": "checkpoints/corosred/50m_lightning",
        "save_every_steps": 1271
      },
      "data": {
        "train_path": "data/python_corpus_2.5b.bin"
      },
      "_resolved_params": 49166848,
      "_backend": "pytorch",
      "_device": "cuda",
      "_device_count": 1,
      "corosred": {
        "unified": true,
        "phase": "unified",
        "alpha_max": 0.85,
        "alpha_min": 0.5,
        "beta_min": 0.15,
        "beta_max": 0.35,
        "gamma_max": 0.1,
        "hold_fraction": 0.2,
        "decay_power": 2.5,
        "gamma_gate_auc": 0.55,
        "acc_gate_threshold": 0.65,
        "causal_ratio": 0.65,
        "routing_cache_steps": 50,
        "adaptive_rebalance": false,
        "mask_prob": 0.2,
        "k_amb": 5
      },
      "paradigm": "corosred"
    },
    "parameter_count": 49430017,
    "saved_tensor_dtypes": [
      "torch.float32"
    ]
  },
  "tpu_baseline": {
    "checkpoint": "/Users/ivansamuel/Documents/Projects/telos/checkpoints/corosred/unified/50m_python/checkpoint_final.pt",
    "checkpoint_sha256": "59ee1988e001a7f094e707ac60cee1d4162ded69c1be7f1436089ad260644442",
    "step": 12716,
    "config": {
      "model": {
        "vocab_size": 8192,
        "seq_len": 512,
        "max_seq_len": 512,
        "d_model": 512,
        "n_layers": 14,
        "n_heads": 8,
        "n_kv_heads": 8,
        "tied_embeddings": true,
        "use_reliability_head": true,
        "mask_token_id": 1,
        "paradigm": "corosred",
        "is_causal": true
      },
      "training": {
        "batch_size": 48,
        "gradient_accumulation": 1,
        "max_steps": 12716,
        "max_lr": 0.0004242640687119285,
        "min_lr": 4.2426406871192855e-05,
        "warmup_steps": 254,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "precision": "bf16"
      },
      "checkpoint": {
        "checkpoint_dir": "checkpoints/corosred/unified/50m_python",
        "save_every_steps": 4000
      },
      "data": {
        "train_path": "data/python_corpus_5b.bin"
      },
      "_resolved_params": 49166848,
      "_backend": "pytorch",
      "_device": "xla",
      "_device_count": 8,
      "corosred": {
        "unified": true,
        "phase": "unified",
        "alpha_max": 0.85,
        "alpha_min": 0.5,
        "beta_min": 0.15,
        "beta_max": 0.35,
        "gamma_max": 0.1,
        "hold_fraction": 0.2,
        "decay_power": 2.5,
        "gamma_gate_auc": 0.55,
        "acc_gate_threshold": 0.65,
        "causal_ratio": 0.65,
        "routing_cache_steps": 50,
        "adaptive_rebalance": false,
        "mask_prob": 0.2,
        "k_amb": 5
      },
      "paradigm": "corosred"
    },
    "parameter_count": 49430017,
    "saved_tensor_dtypes": [
      "torch.float32"
    ]
  },
  "tpu_schedstep1": {
    "checkpoint": "/Users/ivansamuel/Documents/Projects/telos/checkpoints/corosred/unified/50m_tpu_schedstep1/checkpoint_final.pt",
    "checkpoint_sha256": "f1db63c89ba579af959996342306ca75a07b8f71fe87db57372c0ba1a55d4eb9",
    "step": 12716,
    "config": {
      "model": {
        "vocab_size": 8192,
        "seq_len": 512,
        "max_seq_len": 512,
        "d_model": 512,
        "n_layers": 14,
        "n_heads": 8,
        "n_kv_heads": 8,
        "tied_embeddings": true,
        "use_reliability_head": true,
        "mask_token_id": 1,
        "paradigm": "corosred",
        "is_causal": true
      },
      "training": {
        "batch_size": 48,
        "gradient_accumulation": 1,
        "max_steps": 12716,
        "max_lr": 0.0004242640687119285,
        "min_lr": 4.2426406871192855e-05,
        "warmup_steps": 254,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "precision": "bf16",
        "sched_cadence": 1
      },
      "checkpoint": {
        "checkpoint_dir": "checkpoints/corosred/unified/50m_python",
        "save_every_steps": 4000
      },
      "data": {
        "train_path": "data/python_corpus_5b.bin"
      },
      "_resolved_params": 49166848,
      "_backend": "pytorch",
      "_device": "xla",
      "_device_count": 8,
      "corosred": {
        "unified": true,
        "phase": "unified",
        "alpha_max": 0.85,
        "alpha_min": 0.5,
        "beta_min": 0.15,
        "beta_max": 0.35,
        "gamma_max": 0.1,
        "hold_fraction": 0.2,
        "decay_power": 2.5,
        "gamma_gate_auc": 0.55,
        "acc_gate_threshold": 0.65,
        "causal_ratio": 0.65,
        "routing_cache_steps": 50,
        "adaptive_rebalance": false,
        "mask_prob": 0.2,
        "k_amb": 5,
        "sched_cadence": 1
      },
      "paradigm": "corosred"
    },
    "parameter_count": 49430017,
    "saved_tensor_dtypes": [
      "torch.float32"
    ]
  }
}
```


## A3. First slice endpoint metrics and geometry

```json
{
  "cuda_lightning": {
    "causal": {
      "ce": 1.4752874505029965,
      "top1_pct": 68.68884540117416,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 1.663659381306568,
          "top1_pct": 64.48140900195695
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 1.4604099221425513,
          "top1_pct": 70.7436399217221
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 1.3489604434855063,
          "top1_pct": 70.89041095890411
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 1.4281200550773605,
          "top1_pct": 68.6399217221135
        }
      ]
    },
    "infill": {
      "ce": 1.0732028747937694,
      "top1_pct": 77.0807453416149,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 1.3575331744025736,
          "top1_pct": 70.83333333333333
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 1.0176181135506466,
          "top1_pct": 80.04926108374384
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 1.0375996907552083,
          "top1_pct": 80.0
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 0.8711055355608616,
          "top1_pct": 77.49360613810742
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 0.873914216051399,
      "infill_gradient_norm": 2.954790568324735,
      "causal_infill_cosine": 0.09926178980446186,
      "weighted_causal_norm": 0.5140671859125876,
      "weighted_infill_norm": 1.2166784693101849,
      "weighted_causal_to_infill_ratio": 0.4225168759697426,
      "combined_gradient_norm": 1.3670182994193794,
      "causal_loss_directional_derivative": -0.5547929641620797,
      "infill_loss_directional_derivative": -3.745804840640841
    },
    "layer_task_cosines": {
      "tok_embeddings": 0.2120957630380768,
      "layers.0": 0.12576350831898675,
      "layers.1": 0.015467980217768832,
      "layers.2": 0.038122357787805136,
      "layers.3": 0.06548948990450541,
      "layers.4": 0.05443351910378428,
      "layers.5": 0.0515104372260621,
      "layers.6": 0.053884111505816616,
      "layers.7": 0.06445025120639516,
      "layers.8": 0.09003720313978976,
      "layers.9": 0.11784911886955292,
      "layers.10": 0.11491857416207778,
      "layers.11": 0.13170914925095664,
      "layers.12": 0.15344937709444495,
      "layers.13": 0.16484786491205158,
      "final_norm": 0.28022180598799135
    }
  },
  "tpu_baseline": {
    "causal": {
      "ce": 1.9869742496139615,
      "top1_pct": 62.010763209393346,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 2.3296464595309443,
          "top1_pct": 56.55577299412916
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 1.9685084871116683,
          "top1_pct": 64.18786692759295
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 1.7980705007414994,
          "top1_pct": 64.87279843444227
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 1.8516715510717343,
          "top1_pct": 62.426614481409004
        }
      ]
    },
    "infill": {
      "ce": 1.860621254488548,
      "top1_pct": 67.3913043478261,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 2.3689229628619026,
          "top1_pct": 58.333333333333336
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 1.7415669258004927,
          "top1_pct": 70.6896551724138
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 1.800122371720679,
          "top1_pct": 68.14814814814815
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 1.5165061853121005,
          "top1_pct": 72.63427109974424
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 1.1221614486085958,
      "infill_gradient_norm": 3.322226679093891,
      "causal_infill_cosine": 0.17202758280911484,
      "weighted_causal_norm": 0.6600949697697622,
      "weighted_infill_norm": 1.3679756913916021,
      "weighted_causal_to_infill_ratio": 0.4825341370636979,
      "combined_gradient_norm": 1.6179501128780844,
      "causal_loss_directional_derivative": -1.0048108779433353,
      "infill_loss_directional_derivative": -4.921979267503452
    },
    "layer_task_cosines": {
      "tok_embeddings": 0.2797806696750064,
      "layers.0": 0.20843774358074568,
      "layers.1": 0.052784101382355834,
      "layers.2": 0.054152289466002194,
      "layers.3": 0.06977320599424795,
      "layers.4": 0.08949204268296984,
      "layers.5": 0.06642473043155303,
      "layers.6": 0.06508590838455379,
      "layers.7": 0.0669111276899249,
      "layers.8": 0.05116441994283639,
      "layers.9": 0.06380374688288079,
      "layers.10": 0.052941830660708525,
      "layers.11": 0.09536041778808042,
      "layers.12": 0.11872861538365809,
      "layers.13": 0.22746791275380704,
      "final_norm": 0.3710282526604749
    }
  },
  "tpu_schedstep1": {
    "causal": {
      "ce": 1.963892031555773,
      "top1_pct": 62.69569471624266,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 2.2823063501406557,
          "top1_pct": 58.12133072407045
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 1.938703742279232,
          "top1_pct": 64.53033268101761
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 1.781204611821184,
          "top1_pct": 65.45988258317026
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 1.8533534219820205,
          "top1_pct": 62.67123287671233
        }
      ]
    },
    "infill": {
      "ce": 1.766005844210986,
      "top1_pct": 68.4472049689441,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 2.206333833582261,
          "top1_pct": 61.27450980392157
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 1.6204312329221828,
          "top1_pct": 73.15270935960591
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 1.678834816261574,
          "top1_pct": 69.38271604938272
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 1.5479847159227142,
          "top1_pct": 70.076726342711
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 0.9888376094051959,
      "infill_gradient_norm": 3.203406502674644,
      "causal_infill_cosine": 0.13152197725295192,
      "weighted_causal_norm": 0.5816691820030565,
      "weighted_infill_norm": 1.3190497363954414,
      "weighted_causal_to_infill_ratio": 0.4409759283168352,
      "combined_gradient_norm": 1.509984063276923,
      "causal_loss_directional_derivative": -0.7467238963238814,
      "infill_loss_directional_derivative": -4.470520407102294
    },
    "layer_task_cosines": {
      "tok_embeddings": 0.23964273434684386,
      "layers.0": 0.13783489206639515,
      "layers.1": 0.053212761428630524,
      "layers.2": 0.04855034232100172,
      "layers.3": 0.05209656656933216,
      "layers.4": 0.06871648183907493,
      "layers.5": 0.05852276993391618,
      "layers.6": 0.0523817275499079,
      "layers.7": 0.048453885250935455,
      "layers.8": 0.039916947479307796,
      "layers.9": 0.054280096580005044,
      "layers.10": 0.07577489413822973,
      "layers.11": 0.08683041912117968,
      "layers.12": 0.14718165852565962,
      "layers.13": 0.20199212365485678,
      "final_norm": 0.4454876247807213
    }
  }
}
```


## A4. Second slice endpoint metrics and geometry

```json
{
  "cuda_lightning": {
    "causal": {
      "ce": 1.6540164536925912,
      "top1_pct": 65.32534246575342,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 1.5448198710402397,
          "top1_pct": 67.31898238747554
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 1.6908834498455847,
          "top1_pct": 62.81800391389432
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 1.6294241530088063,
          "top1_pct": 65.06849315068493
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 1.7509383408757337,
          "top1_pct": 66.0958904109589
        }
      ]
    },
    "infill": {
      "ce": 1.0991804987747478,
      "top1_pct": 77.01863354037268,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 1.1125943800982308,
          "top1_pct": 77.94117647058823
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 1.0026787067281788,
          "top1_pct": 75.1231527093596
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 1.1774374879436729,
          "top1_pct": 76.54320987654322
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 1.104328282348945,
          "top1_pct": 78.51662404092072
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 0.947024686092751,
      "infill_gradient_norm": 2.897061689114531,
      "causal_infill_cosine": 0.07228313018902743,
      "weighted_causal_norm": 0.5570733447604418,
      "weighted_infill_norm": 1.1929077543412774,
      "weighted_causal_to_infill_ratio": 0.46698778068389474,
      "combined_gradient_norm": 1.352563960438038,
      "causal_loss_directional_derivative": -0.6092214079282369,
      "infill_loss_directional_derivative": -3.572583351572393
    },
    "layer_task_cosines": {
      "tok_embeddings": 0.18785735117924166,
      "layers.0": 0.06895928112105305,
      "layers.1": -0.012546290377344371,
      "layers.2": 0.01795871959041029,
      "layers.3": 0.04344736350269698,
      "layers.4": 0.025157524188770236,
      "layers.5": 0.028629372491079955,
      "layers.6": 0.04715385292384338,
      "layers.7": 0.05477587750470914,
      "layers.8": 0.07151333670306062,
      "layers.9": 0.08023668875736356,
      "layers.10": 0.09238012452059804,
      "layers.11": 0.10800506728463874,
      "layers.12": 0.12277922361264187,
      "layers.13": 0.1459742424640742,
      "final_norm": 0.20704563009395174
    }
  },
  "tpu_baseline": {
    "causal": {
      "ce": 2.1522019119411997,
      "top1_pct": 59.34442270058708,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 2.0148192403834395,
          "top1_pct": 60.86105675146771
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 2.1607286188233856,
          "top1_pct": 57.681017612524464
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 2.174486253592833,
          "top1_pct": 57.92563600782779
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 2.2587735349651417,
          "top1_pct": 60.90998043052838
        }
      ]
    },
    "infill": {
      "ce": 1.7120578173524845,
      "top1_pct": 67.70186335403727,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 1.7472282858455883,
          "top1_pct": 66.42156862745098
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 1.5881209349984606,
          "top1_pct": 66.25615763546799
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 1.7646048840181328,
          "top1_pct": 66.91358024691358
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 1.7496211449508472,
          "top1_pct": 71.35549872122762
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 1.0949880502242324,
      "infill_gradient_norm": 3.0885744455137956,
      "causal_infill_cosine": 0.14284171714618005,
      "weighted_causal_norm": 0.6441106177789603,
      "weighted_infill_norm": 1.2717659481527392,
      "weighted_causal_to_infill_ratio": 0.5064694637519911,
      "combined_gradient_norm": 1.5054192012038154,
      "causal_loss_directional_derivative": -0.9042103075467071,
      "infill_loss_directional_derivative": -4.212110776790884
    },
    "layer_task_cosines": {
      "tok_embeddings": 0.2372428135706245,
      "layers.0": 0.18321118106123074,
      "layers.1": 0.04017547638149852,
      "layers.2": 0.04562823070757456,
      "layers.3": 0.048783256634018515,
      "layers.4": 0.03669801694664853,
      "layers.5": 0.05277934399271582,
      "layers.6": 0.019821505196963705,
      "layers.7": 0.048008533578573716,
      "layers.8": 0.03682073230530975,
      "layers.9": 0.046637274044107134,
      "layers.10": 0.0430504638365724,
      "layers.11": 0.07372675004115938,
      "layers.12": 0.10398416637901123,
      "layers.13": 0.21446166752401796,
      "final_norm": 0.615566094321209
    }
  },
  "tpu_schedstep1": {
    "causal": {
      "ce": 2.1251126940702973,
      "top1_pct": 59.57681017612524,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 2.00661042991683,
          "top1_pct": 60.714285714285715
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 2.1426442961870107,
          "top1_pct": 57.58317025440313
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 2.128653031739237,
          "top1_pct": 58.46379647749511
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 2.2225430184381114,
          "top1_pct": 61.545988258317024
        }
      ]
    },
    "infill": {
      "ce": 1.6580699849572982,
      "top1_pct": 67.95031055900621,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 1.6814495909447764,
          "top1_pct": 68.38235294117646
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 1.6150861467633928,
          "top1_pct": 67.24137931034483
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 1.7160823869116513,
          "top1_pct": 65.92592592592592
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 1.618217136548913,
          "top1_pct": 70.33248081841433
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 1.0055261049323374,
      "infill_gradient_norm": 2.845471569412806,
      "causal_infill_cosine": 0.1253930724762349,
      "weighted_causal_norm": 0.5914859440778456,
      "weighted_infill_norm": 1.1716647638758613,
      "weighted_causal_to_infill_ratio": 0.5048252386810823,
      "combined_gradient_norm": 1.3771182111286562,
      "causal_loss_directional_derivative": -0.7424850899722366,
      "infill_loss_directional_derivative": -3.5449823923506956
    },
    "layer_task_cosines": {
      "tok_embeddings": 0.2528604575994789,
      "layers.0": 0.09036065731194154,
      "layers.1": 0.03985783287300938,
      "layers.2": 0.04596679805439238,
      "layers.3": 0.05000166151015639,
      "layers.4": 0.05419214719451856,
      "layers.5": 0.07047655927147979,
      "layers.6": 0.06653191046417722,
      "layers.7": 0.05235812539236399,
      "layers.8": 0.03887232964136874,
      "layers.9": 0.04667175143322942,
      "layers.10": 0.0872235724054287,
      "layers.11": 0.10150202053265851,
      "layers.12": 0.14932887460920477,
      "layers.13": 0.2313897888597559,
      "final_norm": 0.6216541978194058
    }
  }
}
```


## A5. Intermediate checkpoint configurations, schedulers and measurements

```json
{
  "cuda_1271": {
    "metadata": {
      "checkpoint": "/Users/ivansamuel/Documents/Projects/telos/checkpoints/corosred/50m_lightning/checkpoint_step_1271.pt",
      "checkpoint_sha256": "d71aa19d36c6373f1296cac02d8f3a61ed0b9f9dfd3e6d1b272224cd4e610552",
      "step": 1271,
      "config": {
        "model": {
          "vocab_size": 8192,
          "seq_len": 512,
          "max_seq_len": 512,
          "d_model": 512,
          "n_layers": 14,
          "n_heads": 8,
          "n_kv_heads": 8,
          "tied_embeddings": true,
          "use_reliability_head": true,
          "mask_token_id": 1,
          "paradigm": "corosred",
          "is_causal": true
        },
        "training": {
          "compile": false,
          "batch_size": 192,
          "gradient_accumulation": 2,
          "max_steps": 12716,
          "max_lr": 0.0004242640687119285,
          "min_lr": 4.2426406871192855e-05,
          "warmup_steps": 254,
          "weight_decay": 0.1,
          "grad_clip": 1.0,
          "precision": "bf16"
        },
        "checkpoint": {
          "checkpoint_dir": "checkpoints/corosred/50m_lightning",
          "save_every_steps": 1271
        },
        "data": {
          "train_path": "data/python_corpus_2.5b.bin"
        },
        "_resolved_params": 49166848,
        "_backend": "pytorch",
        "_device": "cuda",
        "_device_count": 1,
        "corosred": {
          "unified": true,
          "phase": "unified",
          "alpha_max": 0.85,
          "alpha_min": 0.5,
          "beta_min": 0.15,
          "beta_max": 0.35,
          "gamma_max": 0.1,
          "hold_fraction": 0.2,
          "decay_power": 2.5,
          "gamma_gate_auc": 0.55,
          "acc_gate_threshold": 0.65,
          "causal_ratio": 0.65,
          "routing_cache_steps": 50,
          "adaptive_rebalance": false,
          "mask_prob": 0.2,
          "k_amb": 5
        },
        "paradigm": "corosred"
      },
      "parameter_count": 49430017,
      "saved_tensor_dtypes": [
        "torch.float32"
      ]
    },
    "causal": {
      "ce": 2.2018101164039567,
      "top1_pct": 58.7573385518591,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 2.6403132548770794,
          "top1_pct": 52.98434442270059
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 2.0980754934411694,
          "top1_pct": 62.13307240704501
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 2.015199784430039,
          "top1_pct": 61.64383561643836
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 2.053651932867539,
          "top1_pct": 58.268101761252446
        }
      ]
    },
    "infill": {
      "ce": 2.409550637191867,
      "top1_pct": 57.01863354037267,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 2.826215856215533,
          "top1_pct": 51.96078431372549
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 2.4391509558766935,
          "top1_pct": 57.881773399014776
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 2.2417958200713732,
          "top1_pct": 61.23456790123457
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 2.1177950417599103,
          "top1_pct": 57.033248081841435
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 0.8637223519004213,
      "infill_gradient_norm": 2.9453224581895876,
      "causal_infill_cosine": 0.15712454392557743,
      "weighted_causal_norm": 0.5080719717061303,
      "weighted_infill_norm": 1.2127798357251243,
      "weighted_causal_to_infill_ratio": 0.4189317440311437,
      "combined_gradient_norm": 1.386580645363293,
      "causal_loss_directional_derivative": -0.6034218718997837,
      "infill_loss_directional_derivative": -3.807154477805088
    },
    "scheduler_state": {
      "warmup_steps": 254,
      "max_steps": 12716,
      "min_lr": 4.2426406871192855e-05,
      "update_cadence": 1,
      "base_lrs": [
        0.0004242640687119285,
        0.0004242640687119285
      ],
      "last_epoch": 1271,
      "_step_count": 1272,
      "_is_initial": false,
      "_get_lr_called_within_step": false,
      "_last_lr": [
        0.0004180237788693835,
        0.0004180237788693835
      ]
    },
    "saved_keys": [
      "global_step",
      "model_state_dict",
      "optimizer_state_dict",
      "scheduler_state_dict",
      "config"
    ],
    "lrh_acc_ema": null,
    "lrh_auc_ema": null,
    "optimizer_present": true
  },
  "cuda_3813": {
    "metadata": {
      "checkpoint": "/Users/ivansamuel/Documents/Projects/telos/checkpoints/corosred/50m_lightning/checkpoint_step_3813.pt",
      "checkpoint_sha256": "18484bf2aebade9c34853940c56e607e2ec53125fb3a3b4c30649603384c5f8c",
      "step": 3813,
      "config": {
        "model": {
          "vocab_size": 8192,
          "seq_len": 512,
          "max_seq_len": 512,
          "d_model": 512,
          "n_layers": 14,
          "n_heads": 8,
          "n_kv_heads": 8,
          "tied_embeddings": true,
          "use_reliability_head": true,
          "mask_token_id": 1,
          "paradigm": "corosred",
          "is_causal": true
        },
        "training": {
          "compile": false,
          "batch_size": 192,
          "gradient_accumulation": 2,
          "max_steps": 12716,
          "max_lr": 0.0004242640687119285,
          "min_lr": 4.2426406871192855e-05,
          "warmup_steps": 254,
          "weight_decay": 0.1,
          "grad_clip": 1.0,
          "precision": "bf16"
        },
        "checkpoint": {
          "checkpoint_dir": "checkpoints/corosred/50m_lightning",
          "save_every_steps": 1271
        },
        "data": {
          "train_path": "data/python_corpus_2.5b.bin"
        },
        "_resolved_params": 49166848,
        "_backend": "pytorch",
        "_device": "cuda",
        "_device_count": 1,
        "corosred": {
          "unified": true,
          "phase": "unified",
          "alpha_max": 0.85,
          "alpha_min": 0.5,
          "beta_min": 0.15,
          "beta_max": 0.35,
          "gamma_max": 0.1,
          "hold_fraction": 0.2,
          "decay_power": 2.5,
          "gamma_gate_auc": 0.55,
          "acc_gate_threshold": 0.65,
          "causal_ratio": 0.65,
          "routing_cache_steps": 50,
          "adaptive_rebalance": false,
          "mask_prob": 0.2,
          "k_amb": 5
        },
        "paradigm": "corosred"
      },
      "parameter_count": 49430017,
      "saved_tensor_dtypes": [
        "torch.float32"
      ]
    },
    "causal": {
      "ce": 1.7466748679920652,
      "top1_pct": 63.62524461839531,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 2.015829485689824,
          "top1_pct": 58.365949119373774
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 1.7664995585402397,
          "top1_pct": 65.21526418786692
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 1.5678353804198264,
          "top1_pct": 66.73189823874755
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 1.6365350473183709,
          "top1_pct": 64.18786692759295
        }
      ]
    },
    "infill": {
      "ce": 1.654918864943226,
      "top1_pct": 67.01863354037268,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 2.00561897427428,
          "top1_pct": 59.31372549019608
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 1.5966881061422413,
          "top1_pct": 69.95073891625616
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 1.6188051576967593,
          "top1_pct": 70.12345679012346
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 1.3868423813139386,
          "top1_pct": 68.79795396419438
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 0.8684980043923416,
      "infill_gradient_norm": 2.836501076408881,
      "causal_infill_cosine": 0.12520767200817054,
      "weighted_causal_norm": 0.5108811790543186,
      "weighted_infill_norm": 1.1679710314624805,
      "weighted_causal_to_infill_ratio": 0.4374091182849084,
      "combined_gradient_norm": 1.3321326312785557,
      "causal_loss_directional_derivative": -0.5707075066794994,
      "infill_loss_directional_derivative": -3.4943914053708847
    },
    "scheduler_state": {
      "warmup_steps": 254,
      "max_steps": 12716,
      "min_lr": 4.2426406871192855e-05,
      "update_cadence": 1,
      "base_lrs": [
        0.0004242640687119285,
        0.0004242640687119285
      ],
      "last_epoch": 3813,
      "_step_count": 3814,
      "_is_initial": false,
      "_get_lr_called_within_step": false,
      "_last_lr": [
        0.000352440303983523,
        0.000352440303983523
      ]
    },
    "saved_keys": [
      "global_step",
      "model_state_dict",
      "optimizer_state_dict",
      "scheduler_state_dict",
      "config"
    ],
    "lrh_acc_ema": null,
    "lrh_auc_ema": null,
    "optimizer_present": true
  },
  "cuda_7626": {
    "metadata": {
      "checkpoint": "/Users/ivansamuel/Documents/Projects/telos/checkpoints/corosred/50m_lightning/checkpoint_step_7626.pt",
      "checkpoint_sha256": "3504d99d0e4b7efd18c85d95c33e0b9688dc3d83cc0e882e4e947584ea581582",
      "step": 7626,
      "config": {
        "model": {
          "vocab_size": 8192,
          "seq_len": 512,
          "max_seq_len": 512,
          "d_model": 512,
          "n_layers": 14,
          "n_heads": 8,
          "n_kv_heads": 8,
          "tied_embeddings": true,
          "use_reliability_head": true,
          "mask_token_id": 1,
          "paradigm": "corosred",
          "is_causal": true
        },
        "training": {
          "compile": false,
          "batch_size": 192,
          "gradient_accumulation": 2,
          "max_steps": 12716,
          "max_lr": 0.0004242640687119285,
          "min_lr": 4.2426406871192855e-05,
          "warmup_steps": 254,
          "weight_decay": 0.1,
          "grad_clip": 1.0,
          "precision": "bf16"
        },
        "checkpoint": {
          "checkpoint_dir": "checkpoints/corosred/50m_lightning",
          "save_every_steps": 1271
        },
        "data": {
          "train_path": "data/python_corpus_2.5b.bin"
        },
        "_resolved_params": 49166848,
        "_backend": "pytorch",
        "_device": "cuda",
        "_device_count": 1,
        "corosred": {
          "unified": true,
          "phase": "unified",
          "alpha_max": 0.85,
          "alpha_min": 0.5,
          "beta_min": 0.15,
          "beta_max": 0.35,
          "gamma_max": 0.1,
          "hold_fraction": 0.2,
          "decay_power": 2.5,
          "gamma_gate_auc": 0.55,
          "acc_gate_threshold": 0.65,
          "causal_ratio": 0.65,
          "routing_cache_steps": 50,
          "adaptive_rebalance": false,
          "mask_prob": 0.2,
          "k_amb": 5
        },
        "paradigm": "corosred"
      },
      "parameter_count": 49430017,
      "saved_tensor_dtypes": [
        "torch.float32"
      ]
    },
    "causal": {
      "ce": 1.5580894335841946,
      "top1_pct": 67.4779843444227,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 1.7783775254938234,
          "top1_pct": 63.013698630136986
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 1.591765700487708,
          "top1_pct": 68.54207436399217
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 1.4064747909277153,
          "top1_pct": 70.15655577299412
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 1.4557397174275317,
          "top1_pct": 68.19960861056751
        }
      ]
    },
    "infill": {
      "ce": 1.2301199018585016,
      "top1_pct": 74.40993788819875,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 1.5078786214192708,
          "top1_pct": 65.93137254901961
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 1.193903279421952,
          "top1_pct": 77.0935960591133
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 1.2266181098090279,
          "top1_pct": 77.28395061728395
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 0.9815178990668958,
          "top1_pct": 77.49360613810742
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 0.7788113794330954,
      "infill_gradient_norm": 2.7833007594327754,
      "causal_infill_cosine": 0.08740204253154674,
      "weighted_causal_norm": 0.4581243408429973,
      "weighted_infill_norm": 1.1460650185899663,
      "weighted_causal_to_infill_ratio": 0.3997367805594831,
      "combined_gradient_norm": 1.2708745510418074,
      "causal_loss_directional_derivative": -0.43480475792448753,
      "infill_loss_directional_derivative": -3.3012897910017553
    },
    "scheduler_state": {
      "warmup_steps": 254,
      "max_steps": 12716,
      "min_lr": 4.2426406871192855e-05,
      "update_cadence": 1,
      "base_lrs": [
        0.0004242640687119285,
        0.0004242640687119285
      ],
      "last_epoch": 7626,
      "_step_count": 7627,
      "_is_initial": false,
      "_get_lr_called_within_step": false,
      "_last_lr": [
        0.00017918366243500276,
        0.00017918366243500276
      ]
    },
    "saved_keys": [
      "global_step",
      "model_state_dict",
      "optimizer_state_dict",
      "scheduler_state_dict",
      "config"
    ],
    "lrh_acc_ema": null,
    "lrh_auc_ema": null,
    "optimizer_present": true
  },
  "cuda_11439": {
    "metadata": {
      "checkpoint": "/Users/ivansamuel/Documents/Projects/telos/checkpoints/corosred/50m_lightning/checkpoint_step_11439.pt",
      "checkpoint_sha256": "4882ca2c8a89c43ab7737479b94b8bcdad054a650f09ea487e8f2cb14e638c13",
      "step": 11439,
      "config": {
        "model": {
          "vocab_size": 8192,
          "seq_len": 512,
          "max_seq_len": 512,
          "d_model": 512,
          "n_layers": 14,
          "n_heads": 8,
          "n_kv_heads": 8,
          "tied_embeddings": true,
          "use_reliability_head": true,
          "mask_token_id": 1,
          "paradigm": "corosred",
          "is_causal": true
        },
        "training": {
          "compile": false,
          "batch_size": 192,
          "gradient_accumulation": 2,
          "max_steps": 12716,
          "max_lr": 0.0004242640687119285,
          "min_lr": 4.2426406871192855e-05,
          "warmup_steps": 254,
          "weight_decay": 0.1,
          "grad_clip": 1.0,
          "precision": "bf16"
        },
        "checkpoint": {
          "checkpoint_dir": "checkpoints/corosred/50m_lightning",
          "save_every_steps": 1271
        },
        "data": {
          "train_path": "data/python_corpus_2.5b.bin"
        },
        "_resolved_params": 49166848,
        "_backend": "pytorch",
        "_device": "cuda",
        "_device_count": 1,
        "corosred": {
          "unified": true,
          "phase": "unified",
          "alpha_max": 0.85,
          "alpha_min": 0.5,
          "beta_min": 0.15,
          "beta_max": 0.35,
          "gamma_max": 0.1,
          "hold_fraction": 0.2,
          "decay_power": 2.5,
          "gamma_gate_auc": 0.55,
          "acc_gate_threshold": 0.65,
          "causal_ratio": 0.65,
          "routing_cache_steps": 50,
          "adaptive_rebalance": false,
          "mask_prob": 0.2,
          "k_amb": 5
        },
        "paradigm": "corosred"
      },
      "parameter_count": 49430017,
      "saved_tensor_dtypes": [
        "torch.float32"
      ]
    },
    "causal": {
      "ce": 1.4890785590543205,
      "top1_pct": 68.50538160469668,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 1.6990397055788282,
          "top1_pct": 63.64970645792564
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 1.4813669581702544,
          "top1_pct": 70.64579256360078
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 1.3407328805344911,
          "top1_pct": 71.1839530332681
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 1.4351746919337085,
          "top1_pct": 68.54207436399217
        }
      ]
    },
    "infill": {
      "ce": 1.0983665513696137,
      "top1_pct": 76.8944099378882,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 1.3795037363089768,
          "top1_pct": 71.07843137254902
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 1.0490895539081742,
          "top1_pct": 79.80295566502463
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 1.0572339470003858,
          "top1_pct": 79.01234567901234
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 0.898778812964554,
          "top1_pct": 77.74936061381074
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 0.877480365597298,
      "infill_gradient_norm": 2.9828784059893176,
      "causal_infill_cosine": 0.09677061047418488,
      "weighted_causal_norm": 0.5161649209395871,
      "weighted_infill_norm": 1.228244049525013,
      "weighted_causal_to_infill_ratio": 0.42024622153813695,
      "combined_gradient_norm": 1.3775740471042932,
      "causal_loss_directional_derivative": -0.5572200803197288,
      "infill_loss_directional_derivative": -3.812696219448977
    },
    "scheduler_state": {
      "warmup_steps": 254,
      "max_steps": 12716,
      "min_lr": 4.2426406871192855e-05,
      "update_cadence": 1,
      "base_lrs": [
        0.0004242640687119285,
        0.0004242640687119285
      ],
      "last_epoch": 11439,
      "_step_count": 11440,
      "_is_initial": false,
      "_get_lr_called_within_step": false,
      "_last_lr": [
        5.223419103701787e-05,
        5.223419103701787e-05
      ]
    },
    "saved_keys": [
      "global_step",
      "model_state_dict",
      "optimizer_state_dict",
      "scheduler_state_dict",
      "config"
    ],
    "lrh_acc_ema": null,
    "lrh_auc_ema": null,
    "optimizer_present": true
  },
  "tpu_4000": {
    "metadata": {
      "checkpoint": "/Users/ivansamuel/Documents/Projects/telos/checkpoints/corosred/unified/50m_python/checkpoint_step_4000.pt",
      "checkpoint_sha256": "41591e31fd7a37c01bcfc0de48032ea9e052899f1162120239c428a0978b3054",
      "step": 4000,
      "config": {
        "model": {
          "vocab_size": 8192,
          "seq_len": 512,
          "max_seq_len": 512,
          "d_model": 512,
          "n_layers": 14,
          "n_heads": 8,
          "n_kv_heads": 8,
          "tied_embeddings": true,
          "use_reliability_head": true,
          "mask_token_id": 1,
          "paradigm": "corosred",
          "is_causal": true
        },
        "training": {
          "batch_size": 48,
          "gradient_accumulation": 1,
          "max_steps": 12716,
          "max_lr": 0.0004242640687119285,
          "min_lr": 4.2426406871192855e-05,
          "warmup_steps": 254,
          "weight_decay": 0.1,
          "grad_clip": 1.0,
          "precision": "bf16",
          "lr_cadence": 1,
          "sched_cadence": 1
        },
        "checkpoint": {
          "checkpoint_dir": "checkpoints/corosred/unified/50m_python",
          "save_every_steps": 4000
        },
        "data": {
          "train_path": "data/python_corpus_5b.bin"
        },
        "_resolved_params": 49166848,
        "_backend": "pytorch",
        "_device": "xla",
        "_device_count": 8,
        "corosred": {
          "unified": true,
          "phase": "unified",
          "alpha_max": 0.85,
          "alpha_min": 0.5,
          "beta_min": 0.15,
          "beta_max": 0.35,
          "gamma_max": 0.1,
          "hold_fraction": 0.2,
          "decay_power": 2.5,
          "gamma_gate_auc": 0.55,
          "acc_gate_threshold": 0.65,
          "causal_ratio": 0.65,
          "routing_cache_steps": 50,
          "adaptive_rebalance": false,
          "mask_prob": 0.2,
          "k_amb": 5,
          "sched_cadence": 1
        },
        "paradigm": "corosred"
      },
      "parameter_count": 49430017,
      "saved_tensor_dtypes": [
        "torch.float32"
      ]
    },
    "causal": {
      "ce": 2.437014406207956,
      "top1_pct": 56.37230919765166,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 2.817714385090509,
          "top1_pct": 51.85909980430528
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 2.3873940783237524,
          "top1_pct": 58.414872798434445
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 2.253015686154599,
          "top1_pct": 58.95303326810176
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 2.289933475262965,
          "top1_pct": 56.262230919765166
        }
      ]
    },
    "infill": {
      "ce": 2.823928131672166,
      "top1_pct": 50.993788819875775,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 3.194792803596048,
          "top1_pct": 46.568627450980394
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 2.8863832069735222,
          "top1_pct": 51.724137931034484
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 2.6894272038966047,
          "top1_pct": 53.82716049382716
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 2.51140467651055,
          "top1_pct": 51.91815856777494
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 0.9995564483002508,
      "infill_gradient_norm": 3.2538072021912905,
      "causal_infill_cosine": 0.2688249047488322,
      "weighted_causal_norm": 0.5879743813530888,
      "weighted_infill_norm": 1.3398029656081785,
      "weighted_causal_to_infill_ratio": 0.43885138072238017,
      "combined_gradient_norm": 1.6013526011919414,
      "causal_loss_directional_derivative": -0.9477262338463838,
      "infill_loss_directional_derivative": -4.873764324055359
    },
    "scheduler_state": {
      "warmup_steps": 254,
      "max_steps": 12716,
      "min_lr": 4.2426406871192855e-05,
      "update_cadence": 1,
      "base_lrs": [
        0.0004242640687119285,
        0.0004242640687119285
      ],
      "last_epoch": 4000,
      "_step_count": 4001,
      "_is_initial": false,
      "_get_lr_called_within_step": false,
      "_last_lr": [
        0.0003452761780797113,
        0.0003452761780797113
      ]
    },
    "saved_keys": [
      "global_step",
      "model_state_dict",
      "scheduler_state_dict",
      "config",
      "lrh_acc_ema",
      "lrh_auc_ema"
    ],
    "lrh_acc_ema": 0.5279628646544104,
    "lrh_auc_ema": 0.4872594623964844,
    "optimizer_present": false
  },
  "tpu_8000": {
    "metadata": {
      "checkpoint": "/Users/ivansamuel/Documents/Projects/telos/checkpoints/corosred/unified/50m_python/checkpoint_step_8000.pt",
      "checkpoint_sha256": "b760b4446d2e98470662bd4001a6a9d8b6b501186f890b30826404d6b7548751",
      "step": 8000,
      "config": {
        "model": {
          "vocab_size": 8192,
          "seq_len": 512,
          "max_seq_len": 512,
          "d_model": 512,
          "n_layers": 14,
          "n_heads": 8,
          "n_kv_heads": 8,
          "tied_embeddings": true,
          "use_reliability_head": true,
          "mask_token_id": 1,
          "paradigm": "corosred",
          "is_causal": true
        },
        "training": {
          "batch_size": 48,
          "gradient_accumulation": 1,
          "max_steps": 12716,
          "max_lr": 0.0004242640687119285,
          "min_lr": 4.2426406871192855e-05,
          "warmup_steps": 254,
          "weight_decay": 0.1,
          "grad_clip": 1.0,
          "precision": "bf16",
          "lr_cadence": 1,
          "sched_cadence": 1
        },
        "checkpoint": {
          "checkpoint_dir": "checkpoints/corosred/unified/50m_python",
          "save_every_steps": 4000
        },
        "data": {
          "train_path": "data/python_corpus_5b.bin"
        },
        "_resolved_params": 49166848,
        "_backend": "pytorch",
        "_device": "xla",
        "_device_count": 8,
        "corosred": {
          "unified": true,
          "phase": "unified",
          "alpha_max": 0.85,
          "alpha_min": 0.5,
          "beta_min": 0.15,
          "beta_max": 0.35,
          "gamma_max": 0.1,
          "hold_fraction": 0.2,
          "decay_power": 2.5,
          "gamma_gate_auc": 0.55,
          "acc_gate_threshold": 0.65,
          "causal_ratio": 0.65,
          "routing_cache_steps": 50,
          "adaptive_rebalance": false,
          "mask_prob": 0.2,
          "k_amb": 5,
          "sched_cadence": 1
        },
        "paradigm": "corosred"
      },
      "parameter_count": 49430017,
      "saved_tensor_dtypes": [
        "torch.float32"
      ]
    },
    "causal": {
      "ce": 2.1161538738093966,
      "top1_pct": 60.31066536203522,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 2.480411178678449,
          "top1_pct": 55.72407045009785
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 2.075158524186644,
          "top1_pct": 63.160469667318985
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 1.9232405869694837,
          "top1_pct": 62.62230919765166
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 1.9858052054030089,
          "top1_pct": 59.73581213307241
        }
      ]
    },
    "infill": {
      "ce": 2.115556799847147,
      "top1_pct": 62.2360248447205,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 2.5907939836090685,
          "top1_pct": 55.3921568627451
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 2.0074737999826815,
          "top1_pct": 64.77832512315271
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 1.977672134211034,
          "top1_pct": 65.4320987654321
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 1.8747082488311222,
          "top1_pct": 63.42710997442455
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 1.0004161425950213,
      "infill_gradient_norm": 3.1510922398733485,
      "causal_infill_cosine": 0.2155035187957699,
      "weighted_causal_norm": 0.5884800838794243,
      "weighted_infill_norm": 1.297508569359614,
      "weighted_causal_to_infill_ratio": 0.4535462021417469,
      "combined_gradient_norm": 1.5358828560438509,
      "causal_loss_directional_derivative": -0.8684589986929406,
      "infill_loss_directional_derivative": -4.488189217198666
    },
    "scheduler_state": {
      "warmup_steps": 254,
      "max_steps": 12716,
      "min_lr": 4.2426406871192855e-05,
      "update_cadence": 1,
      "base_lrs": [
        0.0004242640687119285,
        0.0004242640687119285
      ],
      "last_epoch": 8000,
      "_step_count": 8001,
      "_is_initial": false,
      "_get_lr_called_within_step": false,
      "_last_lr": [
        0.00016218888376663026,
        0.00016218888376663026
      ]
    },
    "saved_keys": [
      "global_step",
      "model_state_dict",
      "scheduler_state_dict",
      "config",
      "lrh_acc_ema",
      "lrh_auc_ema"
    ],
    "lrh_acc_ema": 0.5208321310835856,
    "lrh_auc_ema": 0.5097788692438723,
    "optimizer_present": false
  },
  "tpu_12000": {
    "metadata": {
      "checkpoint": "/Users/ivansamuel/Documents/Projects/telos/checkpoints/corosred/unified/50m_python/checkpoint_step_12000.pt",
      "checkpoint_sha256": "69765017879a9125f39352f0a906edae32dd4c8bfb764e0698640464678bd14e",
      "step": 12000,
      "config": {
        "model": {
          "vocab_size": 8192,
          "seq_len": 512,
          "max_seq_len": 512,
          "d_model": 512,
          "n_layers": 14,
          "n_heads": 8,
          "n_kv_heads": 8,
          "tied_embeddings": true,
          "use_reliability_head": true,
          "mask_token_id": 1,
          "paradigm": "corosred",
          "is_causal": true
        },
        "training": {
          "batch_size": 48,
          "gradient_accumulation": 1,
          "max_steps": 12716,
          "max_lr": 0.0004242640687119285,
          "min_lr": 4.2426406871192855e-05,
          "warmup_steps": 254,
          "weight_decay": 0.1,
          "grad_clip": 1.0,
          "precision": "bf16",
          "lr_cadence": 1,
          "sched_cadence": 1
        },
        "checkpoint": {
          "checkpoint_dir": "checkpoints/corosred/unified/50m_python",
          "save_every_steps": 4000
        },
        "data": {
          "train_path": "data/python_corpus_5b.bin"
        },
        "_resolved_params": 49166848,
        "_backend": "pytorch",
        "_device": "xla",
        "_device_count": 8,
        "corosred": {
          "unified": true,
          "phase": "unified",
          "alpha_max": 0.85,
          "alpha_min": 0.5,
          "beta_min": 0.15,
          "beta_max": 0.35,
          "gamma_max": 0.1,
          "hold_fraction": 0.2,
          "decay_power": 2.5,
          "gamma_gate_auc": 0.55,
          "acc_gate_threshold": 0.65,
          "causal_ratio": 0.65,
          "routing_cache_steps": 50,
          "adaptive_rebalance": false,
          "mask_prob": 0.2,
          "k_amb": 5,
          "sched_cadence": 1
        },
        "paradigm": "corosred"
      },
      "parameter_count": 49430017,
      "saved_tensor_dtypes": [
        "torch.float32"
      ]
    },
    "causal": {
      "ce": 1.9939997424818066,
      "top1_pct": 62.30430528375734,
      "tokens": 8176,
      "batches": [
        {
          "batch": 0,
          "tokens": 2044,
          "ce": 2.3388557210127203,
          "top1_pct": 57.92563600782779
        },
        {
          "batch": 1,
          "tokens": 2044,
          "ce": 1.9728803242722603,
          "top1_pct": 64.38356164383562
        },
        {
          "batch": 2,
          "tokens": 2044,
          "ce": 1.7953708596425513,
          "top1_pct": 64.87279843444227
        },
        {
          "batch": 3,
          "tokens": 2044,
          "ce": 1.8688920649996943,
          "top1_pct": 62.03522504892368
        }
      ]
    },
    "infill": {
      "ce": 1.8024140731148097,
      "top1_pct": 67.01863354037268,
      "tokens": 1610,
      "batches": [
        {
          "batch": 0,
          "tokens": 408,
          "ce": 2.2785669962565103,
          "top1_pct": 58.8235294117647
        },
        {
          "batch": 1,
          "tokens": 406,
          "ce": 1.6792214680187807,
          "top1_pct": 69.70443349753694
        },
        {
          "batch": 2,
          "tokens": 405,
          "ce": 1.7057080680941359,
          "top1_pct": 69.87654320987654
        },
        {
          "batch": 3,
          "tokens": 391,
          "ce": 1.5336461372082801,
          "top1_pct": 69.82097186700767
        }
      ]
    },
    "all_backbone": {
      "causal_gradient_norm": 1.0096267840328983,
      "infill_gradient_norm": 3.4333860844882227,
      "causal_infill_cosine": 0.16780533837601694,
      "weighted_causal_norm": 0.5938981082546461,
      "weighted_infill_norm": 1.4137472112598564,
      "weighted_causal_to_infill_ratio": 0.420087907883755,
      "combined_gradient_norm": 1.6227082852256978,
      "causal_loss_directional_derivative": -0.8391335698960183,
      "infill_loss_directional_derivative": -5.196108763288849
    },
    "scheduler_state": {
      "warmup_steps": 254,
      "max_steps": 12716,
      "min_lr": 4.2426406871192855e-05,
      "update_cadence": 1,
      "base_lrs": [
        0.0004242640687119285,
        0.0004242640687119285
      ],
      "last_epoch": 12000,
      "_step_count": 12001,
      "_is_initial": false,
      "_get_lr_called_within_step": false,
      "_last_lr": [
        4.55280343750454e-05,
        4.55280343750454e-05
      ]
    },
    "saved_keys": [
      "global_step",
      "model_state_dict",
      "scheduler_state_dict",
      "config",
      "lrh_acc_ema",
      "lrh_auc_ema"
    ],
    "lrh_acc_ema": 0.5168264424727654,
    "lrh_auc_ema": 0.5096296930652147,
    "optimizer_present": false
  }
}
```


## A6. Local LR-and-objective-cadence-1 candidate final

```json
{
  "metadata": {
    "checkpoint": "checkpoints/corosred/unified/50m_tpu_lrcadence_schedstep1/checkpoint_final.pt",
    "checkpoint_sha256": "5e62ca9fedc1aaf2ad3951c5a4f7da6085bbf661cf0b25b4cb28bda0c69ae781",
    "step": 12716,
    "config": {
      "model": {
        "vocab_size": 8192,
        "seq_len": 512,
        "max_seq_len": 512,
        "d_model": 512,
        "n_layers": 14,
        "n_heads": 8,
        "n_kv_heads": 8,
        "tied_embeddings": true,
        "use_reliability_head": true,
        "mask_token_id": 1,
        "paradigm": "corosred",
        "is_causal": true
      },
      "training": {
        "batch_size": 48,
        "gradient_accumulation": 1,
        "max_steps": 12716,
        "max_lr": 0.0004242640687119285,
        "min_lr": 4.2426406871192855e-05,
        "warmup_steps": 254,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "precision": "bf16",
        "lr_cadence": 1,
        "sched_cadence": 1
      },
      "checkpoint": {
        "checkpoint_dir": "checkpoints/corosred/unified/50m_python",
        "save_every_steps": 4000
      },
      "data": {
        "train_path": "data/python_corpus_5b.bin"
      },
      "_resolved_params": 49166848,
      "_backend": "pytorch",
      "_device": "xla",
      "_device_count": 8,
      "corosred": {
        "unified": true,
        "phase": "unified",
        "alpha_max": 0.85,
        "alpha_min": 0.5,
        "beta_min": 0.15,
        "beta_max": 0.35,
        "gamma_max": 0.1,
        "hold_fraction": 0.2,
        "decay_power": 2.5,
        "gamma_gate_auc": 0.55,
        "acc_gate_threshold": 0.65,
        "causal_ratio": 0.65,
        "routing_cache_steps": 50,
        "adaptive_rebalance": false,
        "mask_prob": 0.2,
        "k_amb": 5,
        "sched_cadence": 1
      },
      "paradigm": "corosred"
    },
    "parameter_count": 49430017,
    "saved_tensor_dtypes": [
      "torch.float32"
    ]
  },
  "causal": {
    "ce": 1.9940016535630198,
    "top1_pct": 62.25538160469667,
    "tokens": 8176,
    "batches": [
      {
        "batch": 0,
        "tokens": 2044,
        "ce": 2.3473896541707435,
        "top1_pct": 57.28962818003914
      },
      {
        "batch": 1,
        "tokens": 2044,
        "ce": 1.9769128250749144,
        "top1_pct": 64.77495107632095
      },
      {
        "batch": 2,
        "tokens": 2044,
        "ce": 1.7873775235827423,
        "top1_pct": 65.2641878669276
      },
      {
        "batch": 3,
        "tokens": 2044,
        "ce": 1.8643266114236792,
        "top1_pct": 61.69275929549902
      }
    ]
  },
  "infill": {
    "ce": 1.778208022976514,
    "top1_pct": 67.26708074534162,
    "tokens": 1610,
    "batches": [
      {
        "batch": 0,
        "tokens": 408,
        "ce": 2.2232504451976105,
        "top1_pct": 60.78431372549019
      },
      {
        "batch": 1,
        "tokens": 406,
        "ce": 1.6788932913042642,
        "top1_pct": 69.45812807881774
      },
      {
        "batch": 2,
        "tokens": 405,
        "ce": 1.693412724247685,
        "top1_pct": 69.38271604938272
      },
      {
        "batch": 3,
        "tokens": 391,
        "ce": 1.5047721374980019,
        "top1_pct": 69.56521739130434
      }
    ]
  },
  "fixture_sha256": "03d264b09f452304161145bc1808ff0f79d30b122b5cbc0eca1c38e17dd8a7fa",
  "all_backbone": {
    "causal_gradient_norm": 1.0474779106327128,
    "infill_gradient_norm": 3.398152469054507,
    "causal_infill_cosine": 0.18580758910274017,
    "weighted_causal_norm": 0.6161634768427723,
    "weighted_infill_norm": 1.3992392519636203,
    "weighted_causal_to_infill_ratio": 0.4403560548905987,
    "combined_gradient_norm": 1.6303127620083302,
    "causal_loss_directional_derivative": -0.9177506507375182,
    "infill_loss_directional_derivative": -5.143875489438248
  }
}
```


## A7. Lineage distances

```json
[
  {
    "path": "checkpoints/corosred/unified/50m_python/checkpoint_step_8000.pt",
    "relative_l2_distance_from_python_step12000": 0.13659786135932145,
    "weight_cosine": 0.9906386262420259,
    "scheduler": {
      "warmup_steps": 254,
      "max_steps": 12716,
      "min_lr": 4.2426406871192855e-05,
      "update_cadence": 1,
      "base_lrs": [
        0.0004242640687119285,
        0.0004242640687119285
      ],
      "last_epoch": 8000,
      "_step_count": 8001,
      "_is_initial": false,
      "_get_lr_called_within_step": false,
      "_last_lr": [
        0.00016218888376663026,
        0.00016218888376663026
      ]
    },
    "config": {
      "model": {
        "vocab_size": 8192,
        "seq_len": 512,
        "max_seq_len": 512,
        "d_model": 512,
        "n_layers": 14,
        "n_heads": 8,
        "n_kv_heads": 8,
        "tied_embeddings": true,
        "use_reliability_head": true,
        "mask_token_id": 1,
        "paradigm": "corosred",
        "is_causal": true
      },
      "training": {
        "batch_size": 48,
        "gradient_accumulation": 1,
        "max_steps": 12716,
        "max_lr": 0.0004242640687119285,
        "min_lr": 4.2426406871192855e-05,
        "warmup_steps": 254,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "precision": "bf16",
        "lr_cadence": 1,
        "sched_cadence": 1
      },
      "checkpoint": {
        "checkpoint_dir": "checkpoints/corosred/unified/50m_python",
        "save_every_steps": 4000
      },
      "data": {
        "train_path": "data/python_corpus_5b.bin"
      },
      "_resolved_params": 49166848,
      "_backend": "pytorch",
      "_device": "xla",
      "_device_count": 8,
      "corosred": {
        "unified": true,
        "phase": "unified",
        "alpha_max": 0.85,
        "alpha_min": 0.5,
        "beta_min": 0.15,
        "beta_max": 0.35,
        "gamma_max": 0.1,
        "hold_fraction": 0.2,
        "decay_power": 2.5,
        "gamma_gate_auc": 0.55,
        "acc_gate_threshold": 0.65,
        "causal_ratio": 0.65,
        "routing_cache_steps": 50,
        "adaptive_rebalance": false,
        "mask_prob": 0.2,
        "k_amb": 5,
        "sched_cadence": 1
      },
      "paradigm": "corosred"
    }
  },
  {
    "path": "checkpoints/corosred/unified/50m_python/checkpoint_final.pt",
    "relative_l2_distance_from_python_step12000": 1.2400065243512395,
    "weight_cosine": 0.22832975648949932,
    "scheduler": {
      "warmup_steps": 254,
      "max_steps": 12716,
      "min_lr": 4.2426406871192855e-05,
      "update_cadence": 10,
      "base_lrs": [
        0.0004242640687119285,
        0.0004242640687119285
      ],
      "last_epoch": 12716,
      "_step_count": 12717,
      "_is_initial": false,
      "_get_lr_called_within_step": false,
      "_last_lr": [
        4.2426406871192855e-05,
        4.2426406871192855e-05
      ]
    },
    "config": {
      "model": {
        "vocab_size": 8192,
        "seq_len": 512,
        "max_seq_len": 512,
        "d_model": 512,
        "n_layers": 14,
        "n_heads": 8,
        "n_kv_heads": 8,
        "tied_embeddings": true,
        "use_reliability_head": true,
        "mask_token_id": 1,
        "paradigm": "corosred",
        "is_causal": true
      },
      "training": {
        "batch_size": 48,
        "gradient_accumulation": 1,
        "max_steps": 12716,
        "max_lr": 0.0004242640687119285,
        "min_lr": 4.2426406871192855e-05,
        "warmup_steps": 254,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "precision": "bf16"
      },
      "checkpoint": {
        "checkpoint_dir": "checkpoints/corosred/unified/50m_python",
        "save_every_steps": 4000
      },
      "data": {
        "train_path": "data/python_corpus_5b.bin"
      },
      "_resolved_params": 49166848,
      "_backend": "pytorch",
      "_device": "xla",
      "_device_count": 8,
      "corosred": {
        "unified": true,
        "phase": "unified",
        "alpha_max": 0.85,
        "alpha_min": 0.5,
        "beta_min": 0.15,
        "beta_max": 0.35,
        "gamma_max": 0.1,
        "hold_fraction": 0.2,
        "decay_power": 2.5,
        "gamma_gate_auc": 0.55,
        "acc_gate_threshold": 0.65,
        "causal_ratio": 0.65,
        "routing_cache_steps": 50,
        "adaptive_rebalance": false,
        "mask_prob": 0.2,
        "k_amb": 5
      },
      "paradigm": "corosred"
    }
  },
  {
    "path": "checkpoints/corosred/unified/50m_tpu_schedstep1/checkpoint_final.pt",
    "relative_l2_distance_from_python_step12000": 1.2414784812200907,
    "weight_cosine": 0.2320370586101902,
    "scheduler": {
      "warmup_steps": 254,
      "max_steps": 12716,
      "min_lr": 4.2426406871192855e-05,
      "update_cadence": 10,
      "base_lrs": [
        0.0004242640687119285,
        0.0004242640687119285
      ],
      "last_epoch": 12716,
      "_step_count": 12717,
      "_is_initial": false,
      "_get_lr_called_within_step": false,
      "_last_lr": [
        4.2426406871192855e-05,
        4.2426406871192855e-05
      ]
    },
    "config": {
      "model": {
        "vocab_size": 8192,
        "seq_len": 512,
        "max_seq_len": 512,
        "d_model": 512,
        "n_layers": 14,
        "n_heads": 8,
        "n_kv_heads": 8,
        "tied_embeddings": true,
        "use_reliability_head": true,
        "mask_token_id": 1,
        "paradigm": "corosred",
        "is_causal": true
      },
      "training": {
        "batch_size": 48,
        "gradient_accumulation": 1,
        "max_steps": 12716,
        "max_lr": 0.0004242640687119285,
        "min_lr": 4.2426406871192855e-05,
        "warmup_steps": 254,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "precision": "bf16",
        "sched_cadence": 1
      },
      "checkpoint": {
        "checkpoint_dir": "checkpoints/corosred/unified/50m_python",
        "save_every_steps": 4000
      },
      "data": {
        "train_path": "data/python_corpus_5b.bin"
      },
      "_resolved_params": 49166848,
      "_backend": "pytorch",
      "_device": "xla",
      "_device_count": 8,
      "corosred": {
        "unified": true,
        "phase": "unified",
        "alpha_max": 0.85,
        "alpha_min": 0.5,
        "beta_min": 0.15,
        "beta_max": 0.35,
        "gamma_max": 0.1,
        "hold_fraction": 0.2,
        "decay_power": 2.5,
        "gamma_gate_auc": 0.55,
        "acc_gate_threshold": 0.65,
        "causal_ratio": 0.65,
        "routing_cache_steps": 50,
        "adaptive_rebalance": false,
        "mask_prob": 0.2,
        "k_amb": 5,
        "sched_cadence": 1
      },
      "paradigm": "corosred"
    }
  },
  {
    "path": "checkpoints/corosred/unified/50m_tpu_lrcadence_schedstep1/checkpoint_final.pt",
    "relative_l2_distance_from_python_step12000": 0.024998521939436382,
    "weight_cosine": 0.9996875310715473,
    "scheduler": {
      "warmup_steps": 254,
      "max_steps": 12716,
      "min_lr": 4.2426406871192855e-05,
      "update_cadence": 1,
      "base_lrs": [
        0.0004242640687119285,
        0.0004242640687119285
      ],
      "last_epoch": 12716,
      "_step_count": 12717,
      "_is_initial": false,
      "_get_lr_called_within_step": false,
      "_last_lr": [
        4.2426406871192855e-05,
        4.2426406871192855e-05
      ]
    },
    "config": {
      "model": {
        "vocab_size": 8192,
        "seq_len": 512,
        "max_seq_len": 512,
        "d_model": 512,
        "n_layers": 14,
        "n_heads": 8,
        "n_kv_heads": 8,
        "tied_embeddings": true,
        "use_reliability_head": true,
        "mask_token_id": 1,
        "paradigm": "corosred",
        "is_causal": true
      },
      "training": {
        "batch_size": 48,
        "gradient_accumulation": 1,
        "max_steps": 12716,
        "max_lr": 0.0004242640687119285,
        "min_lr": 4.2426406871192855e-05,
        "warmup_steps": 254,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "precision": "bf16",
        "lr_cadence": 1,
        "sched_cadence": 1
      },
      "checkpoint": {
        "checkpoint_dir": "checkpoints/corosred/unified/50m_python",
        "save_every_steps": 4000
      },
      "data": {
        "train_path": "data/python_corpus_5b.bin"
      },
      "_resolved_params": 49166848,
      "_backend": "pytorch",
      "_device": "xla",
      "_device_count": 8,
      "corosred": {
        "unified": true,
        "phase": "unified",
        "alpha_max": 0.85,
        "alpha_min": 0.5,
        "beta_min": 0.15,
        "beta_max": 0.35,
        "gamma_max": 0.1,
        "hold_fraction": 0.2,
        "decay_power": 2.5,
        "gamma_gate_auc": 0.55,
        "acc_gate_threshold": 0.65,
        "causal_ratio": 0.65,
        "routing_cache_steps": 50,
        "adaptive_rebalance": false,
        "mask_prob": 0.2,
        "k_amb": 5,
        "sched_cadence": 1
      },
      "paradigm": "corosred"
    }
  }
]
```


## A8. Cadence replay and gradient reachability

```json
{
  "purpose": "Synthetic CPU replay and source reachability; does not identify cause of saved CUDA/TPU deficit",
  "historical_revision": "a2f09a06eefa836c91efe8f937646127f754ece5",
  "historical_source_sha256": {
    "schedule.py": "0fd7d967e975115f0420dbca3e46bce7108ed66907c3c061b93246b6c2cf45f7",
    "corosred_unified.py": "94acb4e53ee3b85e04761535285d2ccacb9897ee483f9b84d3831a723f565e64"
  },
  "current_source_sha256": {
    "telos/training/schedule.py": "7864ee13bbcdf622039c14a55dcd47ddf21159bf273a0025beb69c216a18b465",
    "telos/training/lr_schedule.py": "b873a6da8edb11b8c388f732be1797061d0c9500fb62867bd71c826b5de21269",
    "telos/training/trainer_pytorch.py": "f2435e74dc0462172c5f3f86da0cf7e077f2986dbf50e1fa330d21be2f757d4a",
    "telos/diffusion/corosred_unified.py": "d51b77df038b69eebb0d420fecaa6e243f356a8c12fb522dd0d653501834df9e",
    "telos/models/transformer.py": "7bb232b4f5d95dd22f8d5ba903672eccd4cc4e6c2985ef20e9a4a751f79b4524"
  },
  "metric_input": "Synthetic identical per-step .50 until step 999, .60 thereafter; not recovered training observations",
  "metric_replays": {
    "historical_every1": {
      "first_gamma_step": 1069,
      "gamma_active_steps": 11648,
      "final_auc_ema": 0.5999999999999945,
      "max_difference_vs_current_every1": {
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.0,
        "mask_blend": 0.0
      }
    },
    "historical_every25": {
      "first_gamma_step": 2701,
      "gamma_active_steps": 10016,
      "final_auc_ema": 0.599102764876595,
      "max_difference_vs_current_every1": {
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.1,
        "mask_blend": 0.3357503872425167
      }
    },
    "current_every1": {
      "first_gamma_step": 1069,
      "gamma_active_steps": 11648,
      "final_auc_ema": 0.5999999999999945,
      "max_difference_vs_current_every1": {
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.0,
        "mask_blend": 0.0
      }
    },
    "current_every25": {
      "first_gamma_step": 1051,
      "gamma_active_steps": 11666,
      "final_auc_ema": 0.5999999999999999,
      "max_difference_vs_current_every1": {
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.1,
        "mask_blend": 0.09600000000000009
      }
    }
  },
  "lr_quantization": {
    "1": {
      "integrated_lr": 2.9618086768814087,
      "max_absolute_difference_vs_1": 0.0,
      "integrated_relative_difference_vs_1": 0.0
    },
    "10": {
      "integrated_lr": 2.963526944903716,
      "max_absolute_difference_vs_1": 4.3316494697200254e-07,
      "integrated_relative_difference_vs_1": 0.0005801414641395386
    },
    "25": {
      "integrated_lr": 2.966390712383119,
      "max_absolute_difference_vs_1": 1.155104873568351e-06,
      "integrated_relative_difference_vs_1": 0.001547039664471006
    }
  },
  "objective_step_quantization": {
    "10": {
      "alpha": 0.0007728336790072676,
      "beta": 0.0004416062071980753
    },
    "25": {
      "alpha": 0.0020586088883263187,
      "beta": 0.0011745758662078387
    }
  },
  "gradient_paths_current": {
    "max_backbone_gradient_difference_before_clip": 0.0,
    "max_backbone_gradient_difference_after_clip": 0.0,
    "max_backbone_gradient_difference_with_forced_active_clip_point25": 9.583309292793274e-05,
    "global_norm_gamma0": 0.7178520560264587,
    "global_norm_gamma_point1": 0.7191993594169617,
    "head_gradient_norm_gamma0": 0.0,
    "head_gradient_norm_gamma_point1": 0.044002377050307837
  },
  "gradient_paths_historical_loss": {
    "max_backbone_gradient_difference_before_clip": 0.0,
    "max_backbone_gradient_difference_after_clip": 0.0,
    "max_backbone_gradient_difference_with_forced_active_clip_point25": 0.0001206323504447937,
    "global_norm_gamma0": 0.6765214800834656,
    "global_norm_gamma_point1": 0.6779509782791138,
    "head_gradient_norm_gamma0": 0.0,
    "head_gradient_norm_gamma_point1": 0.044002377050307837
  },
  "routing_refresh_gate": {
    "without_metrics": 0.0,
    "with_auc_point8": 1.0
  }
}
```


## A9. Saved CUDA optimizer replay on CPU

```json
{
  "device": "cpu",
  "torch": "2.13.0",
  "physical_cuda_xla_comparison": false,
  "checkpoint_sha256": "e2e560508672880c45c72d9e28f459a7d3ca649032aa31cad12dccebb3b332ee",
  "capture_sha256": "41946b7da502903b14d4b2061fdabdc8c5bc07f35cf9541db843ad0f4661da7c",
  "capture_metadata": {
    "fixture_sha256": "dee919df8f96d074d1c152dca54b5ac636bd67b7e9c45ac160e21b74dc3b0265",
    "device": "cpu",
    "precision": "fp32",
    "torch": "2.13.0",
    "git_commit": "38dbd652c4a4821b59979a97bc45aeb276848390",
    "hardware": "cpu",
    "torch_xla": null,
    "probe_source_sha256": "792198caf6567ec197a189ee181278bf97b5c27dab30733e1fb6623f09e05b44",
    "objective_source_sha256": "d51b77df038b69eebb0d420fecaa6e243f356a8c12fb522dd0d653501834df9e",
    "model_source_sha256": "7bb232b4f5d95dd22f8d5ba903672eccd4cc4e6c2985ef20e9a4a751f79b4524",
    "components_source_sha256": "ed3f82e902a2d94d2e9df3179516c794d0035a3fee51417d2555fa437458f0b6",
    "fresh_optimizer": true,
    "diagnostic_weights": {
      "alpha": 0.7,
      "beta": 0.3,
      "gamma": 0.1,
      "mask_blend": 0.0
    },
    "loss": 1.6790300607681274,
    "preclip_gradient_norm": 7.401355743408203,
    "metrics": {
      "loss": 1.6790300607681274,
      "unweighted_ce": 1.6028671264648438,
      "causal_ce": 1.725919246673584,
      "infill_ce": 1.479814887046814,
      "r_loss": 0.2694208323955536,
      "lrh_acc": 0.8821292519569397,
      "lrh_bal_acc": 0.8007569313049316,
      "lrh_auc": 0.9323801513877208,
      "alpha": 0.7,
      "beta": 0.3,
      "gamma": 0.1,
      "mask_blend": 0.0,
      "rebal_clamped": false
    }
  },
  "source_sha256": "e64e533c3fa3ec79eb46e695bf1d6af8328f1d987fe74d0e74ef0c0fe4d82b8c",
  "optimizer_state": "Saved CUDA final moments at step 12716, cloned identically for both CPU modes",
  "mapping_caveat": "Parameter group sizes and every moment shape verified against current named-parameter order; checkpoint has no saved parameter-name manifest",
  "gradient_stream": "Same captured FP32 gradients, repeated with prescribed sign/scale changes; not a continuation of training",
  "steps": [
    {
      "step": 1,
      "prescribed_gradient_scale": 1.0,
      "preclip_norm": 7.401355266571045,
      "clip_coefficient": 0.13511037220524008,
      "parameters": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": 0.0,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg_sq": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      }
    },
    {
      "step": 2,
      "prescribed_gradient_scale": 0.5,
      "preclip_norm": 3.7006776332855225,
      "clip_coefficient": 0.2702207079008597,
      "parameters": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": 0.0,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg_sq": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      }
    },
    {
      "step": 3,
      "prescribed_gradient_scale": 1.5,
      "preclip_norm": 11.102038383483887,
      "clip_coefficient": 0.09007354103676347,
      "parameters": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": 0.0,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg_sq": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      }
    },
    {
      "step": 4,
      "prescribed_gradient_scale": -0.25,
      "preclip_norm": 1.8503388166427612,
      "clip_coefficient": 0.540441269763297,
      "parameters": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": 0.0,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg_sq": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      }
    },
    {
      "step": 5,
      "prescribed_gradient_scale": 1.0,
      "preclip_norm": 7.401355266571045,
      "clip_coefficient": 0.13511037220524008,
      "parameters": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": 0.0,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg_sq": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      }
    },
    {
      "step": 6,
      "prescribed_gradient_scale": 0.75,
      "preclip_norm": 5.551019191741943,
      "clip_coefficient": 0.1801470658470428,
      "parameters": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": 0.0,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg_sq": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      }
    },
    {
      "step": 7,
      "prescribed_gradient_scale": -0.5,
      "preclip_norm": 3.7006776332855225,
      "clip_coefficient": 0.2702207079008597,
      "parameters": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": 0.0,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg_sq": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      }
    },
    {
      "step": 8,
      "prescribed_gradient_scale": 0.01,
      "preclip_norm": 0.07401356101036072,
      "clip_coefficient": 1.0,
      "parameters": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": 0.0,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      },
      "exp_avg_sq": {
        "max_absolute": 0.0,
        "relative_l2_vs_tensor": 0.0,
        "relative_l2_vs_plain_cumulative_update": null,
        "unequal_elements": 0,
        "elements": 49430017
      }
    }
  ],
  "dispatch_counts": {
    "ordinary": 16,
    "fused": 16
  }
}
```


## A10. HF checkpoint inventory

```json
{
  "repo": "Kazenowoko/telos",
  "revision": "d728840aada52094748903ddc56256807a0bfd61",
  "files": [
    {
      "path": "checkpoints/corosred/50m_lightning/checkpoint_final.pt",
      "size": 610093100,
      "sha256": "e2e560508672880c45c72d9e28f459a7d3ca649032aa31cad12dccebb3b332ee"
    },
    {
      "path": "checkpoints/corosred/50m_lightning/checkpoint_step_10168.pt",
      "size": 610095775,
      "sha256": "150ac2b74b9b3ffc252aa794477df3838b5f60de366b09ffd35ec90cc531766e"
    },
    {
      "path": "checkpoints/corosred/50m_lightning/checkpoint_step_11439.pt",
      "size": 610095775,
      "sha256": "4882ca2c8a89c43ab7737479b94b8bcdad054a650f09ea487e8f2cb14e638c13"
    },
    {
      "path": "checkpoints/corosred/50m_lightning/checkpoint_step_1271.pt",
      "size": 610095240,
      "sha256": "d71aa19d36c6373f1296cac02d8f3a61ed0b9f9dfd3e6d1b272224cd4e610552"
    },
    {
      "path": "checkpoints/corosred/50m_lightning/checkpoint_step_12710.pt",
      "size": 610095775,
      "sha256": "e6c0502c452b86de20cd6f41b7e4e18120ed361aab24d26adc80aaa8b5b83ae3"
    },
    {
      "path": "checkpoints/corosred/50m_lightning/checkpoint_step_2542.pt",
      "size": 610095240,
      "sha256": "dc482ba764d9fde8d15d9ae3eb0e199d15a76a443c1cc4f4659b8767508882c7"
    },
    {
      "path": "checkpoints/corosred/50m_lightning/checkpoint_step_3813.pt",
      "size": 610095240,
      "sha256": "18484bf2aebade9c34853940c56e607e2ec53125fb3a3b4c30649603384c5f8c"
    },
    {
      "path": "checkpoints/corosred/50m_lightning/checkpoint_step_5084.pt",
      "size": 610095240,
      "sha256": "f212635a25b8f7003a296c4ab818182e8d6309426b4c545f7c45f54e7dbfa6ad"
    },
    {
      "path": "checkpoints/corosred/50m_lightning/checkpoint_step_6355.pt",
      "size": 610095240,
      "sha256": "df86a3baadb5c557fada95c27e3b38c2650e1af7a565ead5b05e153d5d18772f"
    },
    {
      "path": "checkpoints/corosred/50m_lightning/checkpoint_step_7626.pt",
      "size": 610095240,
      "sha256": "3504d99d0e4b7efd18c85d95c33e0b9688dc3d83cc0e882e4e947584ea581582"
    },
    {
      "path": "checkpoints/corosred/50m_lightning/checkpoint_step_8897.pt",
      "size": 610095240,
      "sha256": "6078544701980c8eb53a53c47779d7669ad22d0085dbf24562f5f5954fb08e66"
    },
    {
      "path": "checkpoints/corosred/50m_lightning/config.json",
      "size": 1314,
      "sha256": null
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_final.pt",
      "size": 214541336,
      "sha256": "25fb6e2de255cf890979667e7822760e1fe573ca1be692272b8feea366241438"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_step_1000.pt",
      "size": 643650537,
      "sha256": "cc0d39fb4a294bed8d4522defb38172b31e46a3dea5f4a43d7235f5bdcd3f7a5"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_step_10000.pt",
      "size": 214542031,
      "sha256": "3e14423402a8f420da8df813fb0adfbcf3aeff530033a24357e789b26c3a9d63"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_step_11000.pt",
      "size": 214542031,
      "sha256": "ce232ce0bbcfc9673fe38a7a71e2a28f932d7abe5969927af863b385d62bb273"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_step_12000.pt",
      "size": 214542031,
      "sha256": "4445a7e7cd395ac93c639775911827ad2d776f5b150c108e0c97026d3f27d651"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_step_2000.pt",
      "size": 643650537,
      "sha256": "189da8a8fc1d166e269f741fff139d14885c2833a898769710d0ee0b619372a6"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_step_3000.pt",
      "size": 214541892,
      "sha256": "031725a7d6c5b9288f41cfc48da12a1d178aeeeac158c298a69e8d74cd23a2a3"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_step_4000.pt",
      "size": 643650537,
      "sha256": "a048fbe5bfb222814d33ad71535d7f76f573b68e57168b9e2c521d3aad1172e8"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_step_5000.pt",
      "size": 643650537,
      "sha256": "bae866af7172e4eaba1378ec7f2cdfff9a8543693e0d2f22a68dbf4ef070eb89"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_step_6000.pt",
      "size": 214541892,
      "sha256": "decc6e7cf56177ee748430f4f088736a34b8ebc90d923502209877429688179a"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_step_7000.pt",
      "size": 214541892,
      "sha256": "668d1a3e4a05f872252b1b4ac05fb7d0964bfa89e2c40d07d73d962a6c2bb493"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_step_8000.pt",
      "size": 214541892,
      "sha256": "fa894b0c9d645200adeeaf1214805b7df89f7c0116d81a87da6309fcca49a183"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/checkpoint_step_9000.pt",
      "size": 214541892,
      "sha256": "381ee2a3c7fbc084ae47bd08cf823882a4a55670bf8f07191f70708f4a7c5011"
    },
    {
      "path": "checkpoints/corosred/unified/50m_2.5b/config.json",
      "size": 1289,
      "sha256": null
    },
    {
      "path": "checkpoints/corosred/unified/50m_english/checkpoint_final.pt",
      "size": 214541336,
      "sha256": "3ff3681a69e788550eb5dd57460db91e8cd6d000a8004eacdfad15fd86ff7738"
    },
    {
      "path": "checkpoints/corosred/unified/50m_english/checkpoint_step_12000.pt",
      "size": 214542031,
      "sha256": "369b423a472cb44f94b64cb46e58d4d705eeed359047e44cb36ece1867f717c6"
    },
    {
      "path": "checkpoints/corosred/unified/50m_english/checkpoint_step_3000.pt",
      "size": 214541892,
      "sha256": "cf8a6e6974f79d6efa72b2e3ba5da698912324913776462061634c51e182b494"
    },
    {
      "path": "checkpoints/corosred/unified/50m_english/checkpoint_step_6000.pt",
      "size": 214541892,
      "sha256": "348056ab5428da791231bced36d2d5d25bbb5b583475240d306d0d7517450701"
    },
    {
      "path": "checkpoints/corosred/unified/50m_english/checkpoint_step_9000.pt",
      "size": 214541892,
      "sha256": "57706753fb4382b1e0b0cc2c778791dad1429bf456a8ef1dcfc3299375636ec4"
    },
    {
      "path": "checkpoints/corosred/unified/50m_english/config.json",
      "size": 1295,
      "sha256": null
    },
    {
      "path": "checkpoints/corosred/unified/50m_python/checkpoint_final.pt",
      "size": 214541336,
      "sha256": "5e62ca9fedc1aaf2ad3951c5a4f7da6085bbf661cf0b25b4cb28bda0c69ae781"
    },
    {
      "path": "checkpoints/corosred/unified/50m_python/checkpoint_step_12000.pt",
      "size": 214542095,
      "sha256": "69765017879a9125f39352f0a906edae32dd4c8bfb764e0698640464678bd14e"
    },
    {
      "path": "checkpoints/corosred/unified/50m_python/checkpoint_step_4000.pt",
      "size": 214541956,
      "sha256": "41591e31fd7a37c01bcfc0de48032ea9e052899f1162120239c428a0978b3054"
    },
    {
      "path": "checkpoints/corosred/unified/50m_python/checkpoint_step_8000.pt",
      "size": 214541956,
      "sha256": "b760b4446d2e98470662bd4001a6a9d8b6b501186f890b30826404d6b7548751"
    },
    {
      "path": "checkpoints/corosred/unified/50m_python/config.json",
      "size": 1362,
      "sha256": null
    },
    {
      "path": "checkpoints/corosred/unified/50m_seed/checkpoint_final.pt",
      "size": 214541336,
      "sha256": "a4345bbf0a6c267decec247ab404021f4ded63008d50e70685787d8091ce887d"
    },
    {
      "path": "checkpoints/corosred/unified/50m_seed/checkpoint_step_10000.pt",
      "size": 214542095,
      "sha256": "47079555e5d2d82d6fd1b1166a6cf86343e89b822ff81c57b0ff06c2af1f35e2"
    },
    {
      "path": "checkpoints/corosred/unified/50m_seed/checkpoint_step_12000.pt",
      "size": 214542095,
      "sha256": "1b696b7459468e919ca8626f14b129abb9840f4cb88387ee0b0bf3dc0d9a539a"
    },
    {
      "path": "checkpoints/corosred/unified/50m_seed/checkpoint_step_2000.pt",
      "size": 214541892,
      "sha256": "f0c489d103a322c44334395700d851bede1e444d3e01a63f1d08da48515e0f95"
    },
    {
      "path": "checkpoints/corosred/unified/50m_seed/checkpoint_step_4000.pt",
      "size": 214541892,
      "sha256": "1a3cf62b6c4cd5afce15024dee80869def93eba0a864b85dd6e5b5d5c6d3fec6"
    },
    {
      "path": "checkpoints/corosred/unified/50m_seed/checkpoint_step_6000.pt",
      "size": 214541892,
      "sha256": "640d079c69fc8c3353cf72a22f2fb31c182276ee132dfcce7d6a2ab831fe3775"
    },
    {
      "path": "checkpoints/corosred/unified/50m_seed/checkpoint_step_8000.pt",
      "size": 214541892,
      "sha256": "2bf1eb3a74d8469cbd4375240960b2eef159a054f0fb88f2b5b7cf3710a5ff9f"
    },
    {
      "path": "checkpoints/corosred/unified/50m_seed/config.json",
      "size": 1369,
      "sha256": null
    }
  ]
}
```


## A11. HF file upload history

```json
[
  {
    "path": "checkpoints/corosred/50m_lightning/checkpoint_final.pt",
    "last_commit": "LastCommitInfo(oid='23d32ce9e8253dfa0f50c4ae4566339505e9fed4', title='Upload folder using huggingface_hub', date=datetime.datetime(2026, 9, 14, 15, 18, 27, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/50m_lightning/checkpoint_step_10168.pt",
    "last_commit": "LastCommitInfo(oid='23d32ce9e8253dfa0f50c4ae4566339505e9fed4', title='Upload folder using huggingface_hub', date=datetime.datetime(2026, 9, 14, 15, 18, 27, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/50m_lightning/checkpoint_step_11439.pt",
    "last_commit": "LastCommitInfo(oid='23d32ce9e8253dfa0f50c4ae4566339505e9fed4', title='Upload folder using huggingface_hub', date=datetime.datetime(2026, 9, 14, 15, 18, 27, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/50m_lightning/checkpoint_step_1271.pt",
    "last_commit": "LastCommitInfo(oid='23d32ce9e8253dfa0f50c4ae4566339505e9fed4', title='Upload folder using huggingface_hub', date=datetime.datetime(2026, 9, 14, 15, 18, 27, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/50m_lightning/checkpoint_step_12710.pt",
    "last_commit": "LastCommitInfo(oid='23d32ce9e8253dfa0f50c4ae4566339505e9fed4', title='Upload folder using huggingface_hub', date=datetime.datetime(2026, 9, 14, 15, 18, 27, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/50m_lightning/checkpoint_step_2542.pt",
    "last_commit": "LastCommitInfo(oid='23d32ce9e8253dfa0f50c4ae4566339505e9fed4', title='Upload folder using huggingface_hub', date=datetime.datetime(2026, 9, 14, 15, 18, 27, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/50m_lightning/checkpoint_step_3813.pt",
    "last_commit": "LastCommitInfo(oid='23d32ce9e8253dfa0f50c4ae4566339505e9fed4', title='Upload folder using huggingface_hub', date=datetime.datetime(2026, 9, 14, 15, 18, 27, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/50m_lightning/checkpoint_step_5084.pt",
    "last_commit": "LastCommitInfo(oid='23d32ce9e8253dfa0f50c4ae4566339505e9fed4', title='Upload folder using huggingface_hub', date=datetime.datetime(2026, 9, 14, 15, 18, 27, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/50m_lightning/checkpoint_step_6355.pt",
    "last_commit": "LastCommitInfo(oid='23d32ce9e8253dfa0f50c4ae4566339505e9fed4', title='Upload folder using huggingface_hub', date=datetime.datetime(2026, 9, 14, 15, 18, 27, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/50m_lightning/checkpoint_step_7626.pt",
    "last_commit": "LastCommitInfo(oid='23d32ce9e8253dfa0f50c4ae4566339505e9fed4', title='Upload folder using huggingface_hub', date=datetime.datetime(2026, 9, 14, 15, 18, 27, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/50m_lightning/checkpoint_step_8897.pt",
    "last_commit": "LastCommitInfo(oid='23d32ce9e8253dfa0f50c4ae4566339505e9fed4', title='Upload folder using huggingface_hub', date=datetime.datetime(2026, 9, 14, 15, 18, 27, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/50m_lightning/config.json",
    "last_commit": "LastCommitInfo(oid='23d32ce9e8253dfa0f50c4ae4566339505e9fed4', title='Upload folder using huggingface_hub', date=datetime.datetime(2026, 9, 14, 15, 18, 27, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/unified/50m_python/checkpoint_final.pt",
    "last_commit": "LastCommitInfo(oid='6a0f9509916c9b6f88bec92940c4559f9e29be4a', title='add: 50M Unified COROSred 2.5B-token checkpoints and config with lrcadence and schedstep 1', date=datetime.datetime(2026, 9, 18, 6, 24, 9, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/unified/50m_python/checkpoint_step_12000.pt",
    "last_commit": "LastCommitInfo(oid='6a0f9509916c9b6f88bec92940c4559f9e29be4a', title='add: 50M Unified COROSred 2.5B-token checkpoints and config with lrcadence and schedstep 1', date=datetime.datetime(2026, 9, 18, 6, 24, 9, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/unified/50m_python/checkpoint_step_4000.pt",
    "last_commit": "LastCommitInfo(oid='6a0f9509916c9b6f88bec92940c4559f9e29be4a', title='add: 50M Unified COROSred 2.5B-token checkpoints and config with lrcadence and schedstep 1', date=datetime.datetime(2026, 9, 18, 6, 24, 9, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/unified/50m_python/checkpoint_step_8000.pt",
    "last_commit": "LastCommitInfo(oid='6a0f9509916c9b6f88bec92940c4559f9e29be4a', title='add: 50M Unified COROSred 2.5B-token checkpoints and config with lrcadence and schedstep 1', date=datetime.datetime(2026, 9, 18, 6, 24, 9, tzinfo=datetime.timezone.utc))"
  },
  {
    "path": "checkpoints/corosred/unified/50m_python/config.json",
    "last_commit": "LastCommitInfo(oid='6a0f9509916c9b6f88bec92940c4559f9e29be4a', title='add: 50M Unified COROSred 2.5B-token checkpoints and config with lrcadence and schedstep 1', date=datetime.datetime(2026, 9, 18, 6, 24, 9, tzinfo=datetime.timezone.utc))"
  }
]
```


# Appendix B — Exact current model, loss and scheduling sources

These are the inspected working-copy sources, not proof of the source used during historical training. Current Git HEAD: `38dbd652c4a4821b59979a97bc45aeb276848390`. Comments in the sources are not automatically correct; the double-reduction attenuation comment in particular is disputed above.


### telos/models/config.py

SHA-256 of embedded excerpt/source: `0aa3f512839d9be40d9ae45f66a8c0dfcb4562a15a9feb776559bb5ad28e8b96`

```python
"""
Model configuration dataclasses for Télos.
"""

from dataclasses import dataclass


@dataclass
class TelosConfig:
    """Model configuration parameters."""
    vocab_size: int = 8192       # Size of tokenizer vocabulary
    d_model: int = 512           # Hidden embedding dimension
    n_layers: int = 8            # Number of transformer block layers
    n_heads: int = 8             # Number of query attention heads
    n_kv_heads: int | None = None # Number of key/value heads for GQA (defaults to n_heads if None)
    max_seq_len: int = 512       # Maximum supported sequence length
    seq_len: int | None = None   # Alias for max_seq_len
    dropout: float = 0.0         # Attention dropout rate
    tied_embeddings: bool = True # Tie input embeddings with output linear projection
    is_causal: bool = False      # Causal autoregressive mask vs bidirectional diffusion
    use_reliability_head: bool = False  # Scalar reliability head for COROSred
    use_grad_checkpoint: bool = False   # Enable gradient checkpointing across transformer layers

    def __post_init__(self):
        if self.seq_len is not None:
            self.max_seq_len = self.seq_len
```


### telos/models/components.py

SHA-256 of embedded excerpt/source: `ed3f82e902a2d94d2e9df3179516c794d0035a3fee51417d2555fa437458f0b6`

```python
"""
Unified PyTorch components for the Télos Transformer backbone.

Includes:
- RMSNorm: root mean square normalization
- RoPE (Rotary Position Embeddings): relative positional encodings applied to Q/K
- SwiGLU: Feed-forward network
- Attention: Multi-head attention with configurable causal masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Simplifies LayerNorm by eliminating mean-centering and relying
    solely on the root mean square of feature activations.
    Uses standard elementary ops compatible across TPU (PyTorch-XLA), CUDA, MPS, and CPU.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotates query and key vectors in 2D pairs to encode relative position.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings to query and key tensors."""
    cos = cos.to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    SwiGLU(x) = (x * W1) * SiLU(x * V) * W2
    Hidden dim is set to roughly 8/3 * d_model rounded to nearest multiple of 64.
    """
    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * 4 * d_model / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # Gate projection
        self.v = nn.Linear(d_model, hidden_dim, bias=False)   # Up projection
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)  # Down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.v(x))


class Attention(nn.Module):
    """Multi-Head Attention with configurable causal masking (Bidirectional or Causal).
    
    Supports:
    - Multi-Head Attention (MHA): n_kv_heads == n_heads
    - Grouped-Query Attention (GQA): n_kv_heads < n_heads
    - Multi-Query Attention (MQA): n_kv_heads == 1
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None, dropout: float = 0.0, is_causal: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.is_causal = is_causal
        assert n_heads % self.n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        
        self.num_queries_per_kv = n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_causal_override: bool | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = apply_rope(q, k, cos, sin)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        causal_flag = is_causal_override if is_causal_override is not None else self.is_causal

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal_flag
        )

        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)
```


### telos/models/transformer.py

SHA-256 of embedded excerpt/source: `7bb232b4f5d95dd22f8d5ba903672eccd4cc4e6c2985ef20e9a4a751f79b4524`

```python
"""
Unified PyTorch Transformer backbone for Télos.
"""

import math
import torch
import torch.nn as nn
from .components import RMSNorm, RotaryEmbedding, SwiGLU, Attention
from .config import TelosConfig



class TransformerBlock(nn.Module):
    """Transformer block with Pre-RMSNorm and Bidirectional/Causal attention."""

    def __init__(self, config: TelosConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config.d_model, config.n_heads, config.n_kv_heads, config.dropout, is_causal=config.is_causal)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask_override: bool | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, is_causal_override=mask_override)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TelosTransformer(nn.Module):
    """Unified Telos Transformer (AR, MDLM, UNDLM, COROSred)."""

    def __init__(self, config: TelosConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            # Filter kwargs to match TelosConfig fields
            valid_keys = {
                "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                "max_seq_len", "seq_len", "dropout", "tied_embeddings", "is_causal",
                "use_reliability_head", "use_grad_checkpoint"
            }
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            self.config = TelosConfig(**cfg_kwargs)
        else:
            self.config = config

        self.use_grad_checkpoint = getattr(self.config, "use_grad_checkpoint", False)
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.dropout = nn.Dropout(self.config.dropout)
        self.rope = RotaryEmbedding(dim=self.config.d_model // self.config.n_heads, max_seq_len=self.config.max_seq_len)
        self.layers = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layers)])
        self.final_norm = RMSNorm(self.config.d_model)
        self.output_projection = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        
        # Scalar reliability head for COROSred (only instantiated when enabled)
        if self.config.use_reliability_head:
            self.reliability_head = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model),
                nn.SiLU(),
                nn.Linear(self.config.d_model, 1)
            )
        else:
            self.reliability_head = None

        self.tie_weights()
        self.apply(self._init_weights)

    def tie_weights(self):
        """Explicitly re-ties output projection weights to token embeddings."""
        if self.config.tied_embeddings:
            self.output_projection.weight = self.tok_embeddings.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, return_reliability: bool = False, mask_override: bool | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len, f"Sequence length {seq_len} exceeds max allowed {self.config.max_seq_len}"

        h = self.tok_embeddings(input_ids)
        h = self.dropout(h)
        cos, sin = self.rope(h, seq_len)

        is_xla = h.device.type == "xla" or str(h.device).startswith("xla")
        for layer in self.layers:
            # Memory Optimization: Recompute activations during backward pass when gradient checkpointing is enabled
            if self.use_grad_checkpoint and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, cos, sin, mask_override,
                    use_reentrant=False
                )
            else:
                h = layer(h, cos, sin, mask_override=mask_override)


        h = self.final_norm(h)
        logits = self.output_projection(h)
        
        if return_reliability:
            r_scores = self.reliability_head(h.detach()).squeeze(-1)
            return logits, r_scores
            
        return logits
```


### telos/training/lr_schedule.py

SHA-256 of embedded excerpt/source: `b873a6da8edb11b8c388f732be1797061d0c9500fb62867bd71c826b5de21269`

```python
"""
Unified LR Schedule for PyTorch.
(MLX has its own functional schedule in the trainer loop).
"""

import math
try:
    from torch.optim.lr_scheduler import LRScheduler as _LRSchedulerBase
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as _LRSchedulerBase


class WarmupCosineLR(_LRSchedulerBase):
    """Linear Warmup followed by Cosine Annealing decay with optional step quantization for XLA graph stability."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 3e-5,
        update_cadence: int = 1,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.update_cadence = max(1, update_cadence)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch

        if step < self.warmup_steps:
            alpha = step / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]

        if step >= self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        # Quantize step to update_cadence to prevent XLA from tracing distinct float constants every step
        effective_step = step if self.update_cadence == 1 else (step - (step % self.update_cadence))
        progress = (effective_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]
```


### telos/training/schedule.py

SHA-256 of embedded excerpt/source: `7864ee13bbcdf622039c14a55dcd47ddf21159bf273a0025beb69c216a18b465`

```python
"""
Continuous Multi-Objective Scheduling and Dynamic Metric Tracking for COROSred.

Provides:
- COROSredSchedule: Hold-then-decay polynomial progression with permanent alpha floor
- DynamicMetricTracker: Zero-overhead running EMAs for LRH accuracy, ROC-AUC, and safe bounded loss rebalancing
- compute_vectorized_roc_auc: Fast, differentiable-free rank-sum ROC-AUC computation
"""

import math
import torch


def compute_vectorized_roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes Wilcoxon-Mann-Whitney ROC-AUC efficiently using torch.argsort / rank-sum.
    
    Args:
        scores: 1D tensor of continuous logit predictions / reliability scores
        labels: 1D binary tensor (0.0 or 1.0) indicating true correctness
        
    Returns:
        float: ROC-AUC score in [0.0, 1.0]. Returns 0.5 if all labels are uniform.
    """
    with torch.no_grad():
        # Flatten and filter any NaN values
        s = scores.reshape(-1).float()
        y = labels.reshape(-1).float()

        n_pos = torch.sum(y == 1.0).item()
        n_neg = torch.sum(y == 0.0).item()

        # If batch contains only positive or only negative labels, AUC is undefined -> return chance (0.50)
        if n_pos == 0 or n_neg == 0:
            return 0.50

        # Sort scores ascending to assign rank (1-indexed ranks)
        # Add tiny jitter to break ties consistently without sorting nondeterminism
        ranks = torch.argsort(torch.argsort(s)) + 1

        # Sum of ranks for positive class (R_1)
        r1 = torch.sum(ranks[y == 1.0]).float().item()

        # Mann-Whitney U statistic: U = R_1 - (n_pos * (n_pos + 1)) / 2
        # AUC = U / (n_pos * n_neg)
        u1 = r1 - (n_pos * (n_pos + 1.0)) / 2.0
        auc = u1 / (n_pos * n_neg)
        return float(max(0.0, min(1.0, auc)))


class COROSredSchedule:
    """
    Computes smooth continuous task weights alpha(t), beta(t), gamma(t) and masking blend ratio.
    
    Curve Design:
    - Hold-Then-Decay: Holds alpha at alpha_max during the initial representation-learning window (hold_fraction),
      then decays via a polynomial curve (1 - tau)^p down to alpha_min.
    - Permanent Floor: alpha(t) is bounded by alpha_min > 0 across the entire training duration.
    - Gated Gamma: LRH auxiliary loss is held at 0.0 until empirical LRH accuracy crosses acc_gate_threshold.
    - Gated Masking Blend: Ratio of LRH confidence-routed masks vs uniform random masks is smoothly
      driven by running LRH ROC-AUC ranking quality.
    """

    def __init__(
        self,
        max_steps: int,
        alpha_max: float = 0.85,
        alpha_min: float = 0.20,
        beta_min: float = 0.15,
        beta_max: float = 0.70,
        gamma_max: float = 0.10,
        hold_fraction: float = 0.20,
        decay_power: float = 2.5,
        gamma_gate_auc: float = 0.55,
        acc_gate_threshold: float = 0.65,
        auc_gate_min: float = 0.50,
        auc_gate_target: float = 0.75,
    ):
        assert max_steps > 0, "max_steps must be strictly positive"
        assert alpha_min > 0.0, "alpha_min floor must be strictly positive (never abandon causal)"
        assert alpha_max >= alpha_min, "alpha_max must be >= alpha_min"
        assert beta_max >= beta_min, "beta_max must be >= beta_min"

        self.max_steps = max_steps
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma_max = gamma_max
        self.hold_fraction = max(0.0, min(1.0, hold_fraction))
        self.decay_power = decay_power
        self.gamma_gate_auc = gamma_gate_auc
        self.acc_gate_threshold = acc_gate_threshold
        self.auc_gate_min = auc_gate_min
        self.auc_gate_target = auc_gate_target

        self.head_ready_override = False

    def set_head_ready(self, ready: bool = True):
        """Allows external evaluation probe or trainer state to explicitly override head warmup gate."""
        self.head_ready_override = ready

    def get_weights(
        self,
        step: int,
        lrh_acc_ema: float | None = None,
        lrh_auc_ema: float | None = None,
    ) -> dict[str, float]:
        """
        Computes the scheduled loss weights for a given training step.
        
        Args:
            step: Current global training step (0 to max_steps)
            lrh_acc_ema: Optional running EMA of LRH accuracy (fallback gate)
            lrh_auc_ema: Optional running EMA of LRH ROC-AUC / balanced accuracy (primary gamma gate and mask blend)
            
        Returns:
            dict containing:
                "alpha": Causal next-token prediction weight
                "beta": Bidirectional infilling weight
                "gamma": Learned Reliability Head BCE weight
                "mask_blend": Fraction of infill tokens routed via confidence (0.0 to 1.0)
        """
        t = max(0.0, min(1.0, step / float(self.max_steps)))

        # 1. Compute alpha(t) and beta(t) via hold-then-decay polynomial progression
        if t <= self.hold_fraction:
            # Hold phase: maximum causal budget for representation learning
            alpha = self.alpha_max
            beta = self.beta_min
        else:
            # Decay phase: polynomial decay with exponent p
            decay_range = max(1e-6, 1.0 - self.hold_fraction)
            tau = (t - self.hold_fraction) / decay_range
            decay_factor = math.pow(max(0.0, 1.0 - tau), self.decay_power)
            ramp_factor = math.pow(min(1.0, tau), self.decay_power)

            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * decay_factor
            beta = self.beta_min + (self.beta_max - self.beta_min) * ramp_factor

        # Strict permanent floor enforcement
        alpha = max(self.alpha_min, alpha)
        beta = max(self.beta_min, min(self.beta_max, beta))

        # Gate gamma on ranking metric (AUC or balanced accuracy) to handle class imbalance
        head_is_competent = self.head_ready_override
        if not head_is_competent and lrh_auc_ema is not None:
            head_is_competent = (lrh_auc_ema >= self.gamma_gate_auc)
        elif not head_is_competent and lrh_acc_ema is not None and self.acc_gate_threshold is not None:
            head_is_competent = (lrh_acc_ema >= self.acc_gate_threshold)
        elif not head_is_competent and lrh_auc_ema is None and lrh_acc_ema is None:
            head_is_competent = False

        if head_is_competent:
            # Ramp gamma over warmup fraction
            warmup_frac = 0.05
            progress_after_gate = max(0.0, min(1.0, (t - 0.10) / warmup_frac)) if (lrh_auc_ema is None and lrh_acc_ema is None) else 1.0
            gamma = self.gamma_max * progress_after_gate
        else:
            gamma = 0.0

        # Compute mask blend ratio based on ranking performance
        if lrh_auc_ema is not None and self.gamma_max > 0.0:
            auc_range = max(1e-6, self.auc_gate_target - self.auc_gate_min)
            mask_blend = max(0.0, min(1.0, (lrh_auc_ema - self.auc_gate_min) / auc_range))
        else:
            mask_blend = 0.0

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "mask_blend": float(mask_blend),
        }


class DynamicMetricTracker:
    """
    Maintains running Exponential Moving Averages (EMA) of training-time metrics
    and performs safe, trust-region-bounded dynamic loss rebalancing.
    """

    def __init__(
        self,
        ema_decay: float = 0.99,
        trust_region: float = 0.20,
        rebalance_temp: float = 0.25,
        cadence: int = 1,
    ):
        self.raw_decay = ema_decay
        self.cadence = max(1, int(cadence))
        # Effective decay: decay_eff = 1.0 - cadence * (1.0 - ema_decay)
        # Keeps effective time constant identical regardless of whether updates occur every step or every 25 steps
        self.decay = max(0.0, 1.0 - self.cadence * (1.0 - self.raw_decay))
        self.trust_region = trust_region
        self.rebalance_temp = rebalance_temp

        # LRH performance statistics
        self.lrh_acc_ema: float | None = None
        self.lrh_auc_ema: float | None = None

        # Task loss statistics for safe rebalancing
        self.loss_c_ema: float | None = None
        self.loss_m_ema: float | None = None
        self.initial_loss_c: float | None = None
        self.initial_loss_m: float | None = None

        self.step_count = 0

    def update_lrh(self, acc: float, auc: float, cadence: int | None = None):
        """Updates LRH classification accuracy and ROC-AUC running EMAs."""
        if math.isnan(acc) or math.isnan(auc):
            return

        cadence_step = self.cadence if cadence is None else max(1, int(cadence))
        decay_eff = max(0.0, 1.0 - cadence_step * (1.0 - self.raw_decay))

        if self.lrh_acc_ema is None:
            self.lrh_acc_ema = float(acc)
            self.lrh_auc_ema = float(auc)
        else:
            self.lrh_acc_ema = decay_eff * self.lrh_acc_ema + (1.0 - decay_eff) * float(acc)
            self.lrh_auc_ema = decay_eff * self.lrh_auc_ema + (1.0 - decay_eff) * float(auc)

    def update_losses(self, loss_c: float, loss_m: float, cadence: int | None = None):
        """Updates causal and masked reconstruction per-token loss EMAs."""
        if math.isnan(loss_c) or math.isnan(loss_m):
            return

        cadence_step = self.cadence if cadence is None else max(1, int(cadence))
        decay_eff = max(0.0, 1.0 - cadence_step * (1.0 - self.raw_decay))

        if self.initial_loss_c is None:
            self.initial_loss_c = max(1e-4, float(loss_c))
            self.initial_loss_m = max(1e-4, float(loss_m))
            self.loss_c_ema = self.initial_loss_c
            self.loss_m_ema = self.initial_loss_m
        else:
            self.loss_c_ema = decay_eff * self.loss_c_ema + (1.0 - decay_eff) * float(loss_c)
            self.loss_m_ema = decay_eff * self.loss_m_ema + (1.0 - decay_eff) * float(loss_m)

        self.step_count += 1

    def get_balanced_weights(
        self,
        alpha_nominal: float,
        beta_nominal: float,
    ) -> tuple[float, float, dict[str, float | bool]]:
        """
        Computes safe, trust-region-bounded dynamic loss rebalancing weights.
        
        Uses relative rate-of-change (r_i(t) = L_i(t) / L_i(0)) rather than raw loss magnitudes.
        Strictly bounds adjustments to +/- trust_region (default +/- 20%) to prevent
        dynamic rebalancing from fighting the macro curriculum schedule.
        
        Returns:
            alpha_effective: Adjusted causal weight
            beta_effective: Adjusted infilling weight
            telemetry: Dict containing nominal_ratio, effective_ratio, factor, and clamped status
        """
        nominal_ratio = beta_nominal / max(1e-6, alpha_nominal)

        # If loss statistics have not yet warmed up, return nominal weights directly
        if self.initial_loss_c is None or self.loss_c_ema is None or self.step_count < 10:
            return alpha_nominal, beta_nominal, {
                "nominal_ratio": nominal_ratio,
                "effective_ratio": nominal_ratio,
                "factor": 1.0,
                "clamped": False,
            }

        # Relative rate-of-change (GradNorm formulation)
        rate_c = max(1e-4, self.loss_c_ema) / max(1e-4, self.initial_loss_c)
        rate_m = max(1e-4, self.loss_m_ema) / max(1e-4, self.initial_loss_m)

        # Raw adjustment factor based on relative training pace: (rate_m / rate_c)^temp
        raw_factor = math.pow(rate_m / rate_c, self.rebalance_temp)

        # Strict Trust-Region Clamping: factor must remain within [1 - delta, 1 + delta]
        min_factor = 1.0 - self.trust_region
        max_factor = 1.0 + self.trust_region
        clamped_factor = max(min_factor, min(max_factor, raw_factor))
        was_clamped = (clamped_factor != raw_factor)

        # Apply bounded adjustment to beta relative to alpha
        alpha_effective = alpha_nominal
        beta_effective = beta_nominal * clamped_factor
        effective_ratio = beta_effective / max(1e-6, alpha_effective)

        telemetry = {
            "nominal_ratio": nominal_ratio,
            "effective_ratio": effective_ratio,
            "factor": clamped_factor,
            "clamped": was_clamped,
        }
        return alpha_effective, beta_effective, telemetry
```


### telos/diffusion/corosred_unified.py

SHA-256 of embedded excerpt/source: `d51b77df038b69eebb0d420fecaa6e243f356a8c12fb522dd0d653501834df9e`

```python
"""
Unified COROSred Single-Loop Training Step and Heterogeneous Minibatch Execution.

Combines causal autoregression, bidirectional infilling, and Learned Reliability Head
routing into a single continuous training loop from init.

Key Features:
- Heterogeneous Minibatch Partition: microbatch B is partitioned into B_c causal and B_m masked sequences.
  Maintains exact 1x FLOP budget while executing native SDPA / FlashAttention kernels on both sub-batches.
- Unified Pooled Token Normalization: Evaluated tokens are pooled across tasks to eliminate the ~7x
  count-driven loss scale imbalance.
- Periodic Cached Routing: Refreshes confidence-routed mask caches periodically to achieve 0.16%
  amortized overhead with static TPU/XLA graphs.
- Teacher-Forced In-Loop Metrics: Computes batch LRH accuracy and ROC-AUC for free during causal step.
"""

from collections import deque
import torch
import torch.nn.functional as F

from telos.training.schedule import compute_vectorized_roc_auc


class RoutingMaskCache:
    """
    Asynchronous / periodic cache for Learned Reliability Head confidence-routed masks.
    
    Refreshes top-k low-confidence token masks every K steps (e.g. K=50) to eliminate
    the need for an extra live causal forward pass on B_m during the inner training step.
    Amortized FLOP overhead is ~0.16%, preserving 1.0x training compute budget and static TPU graphs.
    """

    def __init__(self, refresh_every_steps: int = 50, k_targeted_ratio: float = 0.70):
        self.refresh_every_steps = refresh_every_steps
        self.k_targeted_ratio = k_targeted_ratio
        self.cached_masks = deque()
        self.last_refresh_step = -1

    def should_refresh(self, current_step: int) -> bool:
        """Returns True if the cache should be replenished at current step."""
        if not self.cached_masks:
            return True
        return (current_step - self.last_refresh_step) >= self.refresh_every_steps

    def refresh(
        self,
        model,
        upcoming_seqs: torch.Tensor,
        current_step: int,
        mask_prob: float = 0.20,
    ):
        """
        Evaluates a block of upcoming infill sequences through the causal model
        under torch.no_grad() to identify lowest-confidence token positions.
        
        Args:
            model: Transformer model
            upcoming_seqs: Tensor of shape (K * B_m, T)
            current_step: Global step integer
            mask_prob: Fraction of tokens to mask (default 0.20)
        """
        raw_model = getattr(model, "module", model)
        has_reliability = (
            getattr(raw_model, "reliability_head", None) is not None
            or getattr(getattr(raw_model, "config", None), "use_reliability_head", False)
        )
        if not has_reliability:
            return

        B_total, T = upcoming_seqs.shape
        with torch.no_grad():
            causal_out = raw_model(upcoming_seqs, return_reliability=True, mask_override=True)
            if isinstance(causal_out, tuple):
                _, raw_r = causal_out
            else:
                return

            # Shift r_scores to align with next-token prediction
            r_scores = raw_r[:, :-1]
            # Prepend high confidence score for token 0 (BOS) so it is never masked
            r_with_bos = torch.cat(
                [torch.full((B_total, 1), 1e9, device=upcoming_seqs.device, dtype=r_scores.dtype), r_scores],
                dim=1,
            )

            total_k = max(1, int(T * mask_prob))
            k_targeted = max(1, int(total_k * self.k_targeted_ratio))

            # Identify lowest-confidence token positions (ascending sort)
            sorted_indices = torch.argsort(r_with_bos, dim=-1)
            targeted_indices = sorted_indices[:, :k_targeted]

            # Construct boolean mask tensor
            mask_positions = torch.zeros((B_total, T), dtype=torch.bool, device=upcoming_seqs.device)
            row_indices = torch.arange(B_total, device=upcoming_seqs.device).unsqueeze(-1).expand(-1, k_targeted)
            mask_positions[row_indices, targeted_indices] = True
            mask_positions[:, 0] = False  # BOS invariant

        self.cached_masks.clear()
        self.cached_masks.append(mask_positions.detach())
        self.last_refresh_step = current_step

    def pop_or_generate_mask(
        self,
        batch_seqs: torch.Tensor,
        mask_token_id: int,
        mask_prob: float = 0.20,
        mask_blend: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs corrupted sequences using cached low-confidence masks blended
        with uniform random exploration based on mask_blend ratio.
        """
        B, T = batch_seqs.shape
        device = batch_seqs.device

        # If mask_blend <= 0.0 or no cached masks, use pure uniform random masking
        if mask_blend <= 0.0 or not self.cached_masks:
            rand_probs = torch.rand((B, T), device=device)
            mask_pos = rand_probs < mask_prob
            mask_pos[:, 0] = False  # Preserve BOS

            # Guarantee at least one masked position per sequence
            no_mask = ~mask_pos.any(dim=1, keepdim=True)
            fallback = torch.rand((B, T), device=device) < mask_prob
            fallback[:, 0] = False
            mask_pos = torch.where(no_mask, fallback, mask_pos)
        else:
            # Blend cached confidence-routed mask with uniform random
            full_cached = self.cached_masks[0]
            if full_cached.shape[0] >= B:
                cached_slice = full_cached[:B]
            else:
                # If cached slice is smaller, pad with uniform random
                cached_slice = torch.zeros((B, T), dtype=torch.bool, device=device)
                cached_slice[: full_cached.shape[0]] = full_cached

            total_k = max(1, int(T * mask_prob))

            # 100% Static Top-K Blending (TPU / XLA Safe):
            # Eliminates dynamic slicing and argsort by selecting top-k across blended priority scores.
            # Guarantees identical XLA tensor shapes (B, total_k) across all steps,
            # completely eliminating graph recompilations and host RAM leaks.
            rand_scores = torch.rand((B, T), device=device)
            priority = (1.0 - mask_blend) * rand_scores + mask_blend * cached_slice.float()
            priority[:, 0] = -1e9  # Never mask BOS token

            top_indices = torch.topk(priority, k=total_k, dim=-1).indices
            mask_pos = torch.zeros((B, T), dtype=torch.bool, device=device)
            mask_pos.scatter_(dim=-1, index=top_indices, value=True)

        mask_pos[:, 0] = False
        corrupted_seqs = torch.where(
            mask_pos,
            torch.full((B, T), mask_token_id, dtype=batch_seqs.dtype, device=device),
            batch_seqs,
        )
        return corrupted_seqs, mask_pos


def corosred_unified_step_pytorch(
    model,
    batch_seqs: torch.Tensor,
    vocab_size: int,
    schedule_weights: dict[str, float],
    mask_token_id: int = 1,
    mask_prob: float = 0.20,
    special_token_lut: torch.Tensor | None = None,
    k_amb: int = 5,
    causal_ratio: float = 0.60,
    routing_cache: RoutingMaskCache | None = None,
    metric_tracker=None,
    adaptive_rebalance: bool = False,
    compute_metrics: bool | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Executes a single unified COROSred training step using heterogeneous minibatches.
    
    Args:
        model: Transformer model
        batch_seqs: Tensor of shape (B, T)
        vocab_size: Model vocabulary size
        schedule_weights: Dict with keys "alpha", "beta", "gamma", "mask_blend"
        mask_token_id: Token ID for [MASK]
        mask_prob: Mask probability for infilling
        special_token_lut: Special token lookup table for exclusion
        k_amb: Ambiguity exclusion top-k for LRH
        causal_ratio: Fraction of batch sequences dedicated to causal pass (default 0.60)
        routing_cache: Optional RoutingMaskCache instance
        metric_tracker: Optional DynamicMetricTracker instance
        adaptive_rebalance: Whether to apply safe trust-region loss rebalancing
        
    Returns:
        total_loss: Differentiable scalar loss
        metrics: Dictionary of diagnostic metrics
    """
    B, T = batch_seqs.shape
    device = batch_seqs.device
    is_accelerator = (device.type in ["xla", "cuda"] or str(device).startswith(("xla", "cuda")))

    # 1. Microbatch Partitioning
    if B >= 2:
        B_c = max(1, min(B - 1, int(round(B * causal_ratio))))
        B_m = B - B_c
    else:
        B_c = 1
        B_m = 1

    causal_seqs = batch_seqs[:B_c]
    infill_seqs = batch_seqs[B_c:]

    # 2. Sub-batch 1: Causal Autoregressive Pass
    # Enables is_causal=True in SDPA (native fused FlashAttention-2 kernel)
    causal_logits, raw_r_scores = model(causal_seqs, mask_override=True, return_reliability=True)

    shift_logits = causal_logits[:, :-1, :]
    shift_targets = causal_seqs[:, 1:]

    # Causal next-token cross entropy
    causal_ce_flat = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_targets.reshape(-1),
        reduction="none",
    )
    causal_loss_sum = causal_ce_flat.sum()
    causal_token_count = float(shift_targets.numel())
    mean_causal_ce = causal_loss_sum / max(1.0, causal_token_count)

    # Teacher-forced LRH prediction labels and metric tracking (free from causal logits)
    with torch.no_grad():
        detached_logits = shift_logits.detach()
        argmax_indices = detached_logits.argmax(dim=-1)
        is_exact_match = (argmax_indices == shift_targets)

        # Memory-efficient top-k membership via torch.topk (allocates <1MB instead of 2.35GB broadcast tensor)
        topk_indices = torch.topk(detached_logits, k=k_amb, dim=-1).indices
        is_target_in_top_k = (topk_indices == shift_targets.unsqueeze(-1)).any(dim=-1)

        labels = is_exact_match.float()
        is_ambiguous = is_target_in_top_k & (~is_exact_match)
        valid_mask = ~is_ambiguous

        if special_token_lut is not None:
            content_mask = (shift_targets >= 4)
            valid_mask = valid_mask & content_mask

        # Batch classification accuracy (static shape scalar tensor on device)
        valid_preds = (raw_r_scores[:, :-1] > 0.0).float()
        correct_preds = (valid_preds == labels).float() * valid_mask.float()
        valid_count = valid_mask.sum().float().clamp(min=1.0)
        batch_lrh_acc = (correct_preds.sum() / valid_count)

        # Balanced classification accuracy: 0.5 * (TPR + TNR), fully immune to ~9:1 class imbalance
        # Evaluates to 0.50 under chance / constant predictions, establishing an uncorrupted discrimination baseline
        valid_count = valid_mask.float().sum().clamp(min=1.0)

        preds = (torch.sigmoid(raw_r_scores[:, :-1]) >= 0.5)
        correct_preds = (preds == (labels > 0.5)) & valid_mask
        batch_lrh_acc = correct_preds.float().sum() / valid_count

        pos_mask = (labels > 0.5) & valid_mask
        neg_mask = (labels <= 0.5) & valid_mask
        tpr = (preds & pos_mask).float().sum() / pos_mask.float().sum().clamp(min=1.0)
        tnr = ((~preds) & neg_mask).float().sum() / neg_mask.float().sum().clamp(min=1.0)
        batch_lrh_bal_acc = 0.5 * (tpr + tnr)

        if not is_accelerator:
            batch_lrh_acc_val = float(batch_lrh_acc.item())
            batch_lrh_bal_acc_val = float(batch_lrh_bal_acc.item())
            flat_r = raw_r_scores[:, :-1][valid_mask]
            flat_y = labels[valid_mask]
            batch_lrh_auc_val = compute_vectorized_roc_auc(flat_r, flat_y)
        else:
            # Use balanced accuracy as 0D tensor proxy on accelerators to avoid device sync
            batch_lrh_acc_val = batch_lrh_acc
            batch_lrh_bal_acc_val = batch_lrh_bal_acc
            batch_lrh_auc_val = batch_lrh_bal_acc

    # LRH Binary Cross Entropy Loss
    shift_r_scores = raw_r_scores[:, :-1]
    bce_raw = F.binary_cross_entropy_with_logits(shift_r_scores, labels, reduction="none")
    masked_bce = bce_raw * valid_mask.float()
    r_loss = masked_bce.sum() / valid_count

    # Bidirectional Infill Pass
    mask_blend = schedule_weights.get("mask_blend", 0.0)
    if routing_cache is not None:
        corrupted_infill, mask_positions = routing_cache.pop_or_generate_mask(
            infill_seqs,
            mask_token_id=mask_token_id,
            mask_prob=mask_prob,
            mask_blend=mask_blend,
        )
    else:
        rand_probs = torch.rand((B_m, T), device=device)
        mask_positions = rand_probs < mask_prob
        mask_positions[:, 0] = False
        no_mask = ~mask_positions.any(dim=1, keepdim=True)
        fallback = torch.rand((B_m, T), device=device) < mask_prob
        fallback[:, 0] = False
        mask_positions = torch.where(no_mask, fallback, mask_positions)
        corrupted_infill = torch.where(
            mask_positions,
            torch.full((B_m, T), mask_token_id, dtype=infill_seqs.dtype, device=device),
            infill_seqs,
        )

    infill_logits = model(corrupted_infill, mask_override=False, return_reliability=False)
    infill_ce_flat = F.cross_entropy(
        infill_logits.reshape(-1, vocab_size),
        infill_seqs.reshape(-1),
        reduction="none",
    ).view(B_m, T)

    masked_infill_ce = infill_ce_flat * mask_positions.float()
    infill_loss_sum = masked_infill_ce.sum()
    infill_token_count_t = mask_positions.sum().float().clamp(min=1.0)
    mean_infill_ce = infill_loss_sum / infill_token_count_t

    # Schedule Weight Extraction
    alpha_nom = float(schedule_weights.get("alpha", 0.85))
    beta_nom = float(schedule_weights.get("beta", 0.15))
    gamma_nom = float(schedule_weights.get("gamma", 0.0))

    if adaptive_rebalance and metric_tracker is not None:
        alpha, beta, rebal_telem = metric_tracker.get_balanced_weights(alpha_nom, beta_nom)
    else:
        alpha, beta = alpha_nom, beta_nom
        rebal_telem = {"nominal_ratio": beta_nom / max(1e-6, alpha_nom), "clamped": False}

    # 5. Per-Task Per-Token Mean Multi-Objective Loss Pooling
    # Normalizes by active task token counts so alpha and beta directly define loss weights
    causal_seq_norm = causal_loss_sum / max(1.0, causal_token_count)
    infill_seq_norm = infill_loss_sum / infill_token_count_t

    task_weight_sum = max(1e-6, alpha + beta)
    pooled_task_loss = (alpha * causal_seq_norm + beta * infill_seq_norm) / task_weight_sum
    total_loss = pooled_task_loss + gamma_nom * r_loss

    # 6. Metric Tracker Update
    if metric_tracker is not None:
        if not is_accelerator:
            metric_tracker.update_lrh(batch_lrh_acc_val, batch_lrh_auc_val)
            metric_tracker.update_losses(float(mean_causal_ce.item()), float(mean_infill_ce.item()))

    unweighted_ce = 0.5 * (mean_causal_ce.detach() + mean_infill_ce.detach())
    metrics = {
        "loss": total_loss.detach(),
        "unweighted_ce": unweighted_ce,
        "causal_ce": mean_causal_ce.detach(),
        "infill_ce": mean_infill_ce.detach(),
        "r_loss": r_loss.detach(),
        "lrh_acc": batch_lrh_acc_val,
        "lrh_bal_acc": batch_lrh_bal_acc_val,
        "lrh_auc": batch_lrh_auc_val,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma_nom,
        "mask_blend": mask_blend,
        "rebal_clamped": rebal_telem.get("clamped", False),
    }

    return total_loss, metrics
```


# Appendix C — Trainer initialization, update path and checkpoint handling

Full logical sections are provided as excerpts with original line numbers identified in headings. The helper imports and omitted generic reporting/data code can be inferred from the included CLI and model sources. This is not a claim that these excerpts alone reconstruct all historical execution.


### telos/training/trainer_pytorch.py, lines 43–225

SHA-256 of embedded excerpt/source: `affbad07b4413bf7e4a6e0f9d80c8834157dc0a8f5615c5fa5357939e3cc83f6`

```python


class UnifiedPyTorchTrainer:
    """Unified PyTorch Trainer orchestrator for all paradigms."""

    def __init__(self, paradigm: str, model, cfg: dict, device_type: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Cannot use UnifiedPyTorchTrainer.")

        if torch.cuda.is_available():
            # Enable TF32 for matrix multiplications on Ampere+ architectures
            torch.set_float32_matmul_precision("high")

        self.paradigm = paradigm.lower()
        self.crsr_cfg = cfg.get("crsr", cfg.get("corosred", {}))
        
        # Determine if COROSred should run in unified continuous mode (default) or legacy multi-phase
        if self.paradigm == "corosred":
            if self.crsr_cfg.get("unified", None) is not None:
                self.is_unified = bool(self.crsr_cfg["unified"])
            elif self.crsr_cfg.get("legacy_phases", False):
                self.is_unified = False
            elif "phase" in cfg and cfg["phase"] in ["A", "B", "C"]:
                self.is_unified = False
            else:
                self.is_unified = True
            raw_phase = self.crsr_cfg.get("phase", cfg.get("phase", "unified"))
            self.phase = str(raw_phase).upper() if raw_phase else "UNIFIED"
        else:
            self.is_unified = False
            self.phase = "A"
        self.model = model
        self.cfg = cfg
        self.m_cfg = cfg.setdefault("model", {})
        self.t_cfg = cfg.setdefault("training", {})
        self.c_cfg = cfg.setdefault("checkpoint", {})

        self.seed = int(self.t_cfg.get("seed", cfg.get("seed", 42)))
        from .core import set_global_seed
        set_global_seed(self.seed)

        # Ensure paradigm and architectural metadata are mirrored into configuration
        if "paradigm" not in self.cfg:
            self.cfg["paradigm"] = self.paradigm
        if "paradigm" not in self.m_cfg:
            self.m_cfg["paradigm"] = self.paradigm
        if "is_causal" not in self.m_cfg:
            self.m_cfg["is_causal"] = getattr(self.model, "is_causal", self.paradigm in ("ar", "corosred"))
        if "use_reliability_head" not in self.m_cfg:
            self.m_cfg["use_reliability_head"] = getattr(self.model, "use_reliability_head", self.paradigm == "corosred")

        self.vocab_size = self.m_cfg.get("vocab_size", 8192)
        self.seq_len = self.m_cfg.get("seq_len", 512)
        self.precision = self.t_cfg.get("precision", "bf16")
        
        # Build special token LUT manually for PyTorch
        self.special_lut = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.special_lut[:4] = True

        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if str(device_type).lower() in ["tpu", "xla"]:
            try:
                # Use cached XLA device and consistent xla_model context to guarantee
                # InitializeComputationClient() is invoked at most once per process lifetime.
                from .xla_utils import get_xla_device, get_xla_world_size, is_xla_master, get_xla_spmd_mesh
                self.device = get_xla_device()
                self.is_tpu = True
                self.spmd_mesh = get_xla_spmd_mesh()
                self.is_spmd = self.spmd_mesh is not None
                self.world_size = get_xla_world_size()
                self.is_master = is_xla_master()
                try:
                    import torch_xla.core.xla_model as xm
                    self.rank = xm.get_ordinal()
                except Exception:
                    self.rank = 0
                if self.is_spmd:
                    print(f"  [Hardware] Detected PyTorch-XLA SPMD Topology ({self.world_size} Cores).")
                else:
                    print(f"  [Hardware] Detected PyTorch-XLA TPU Topology ({self.world_size} Cores, Rank {self.rank}).")
            except ImportError as e:
                print(f"Warning: torch_xla not installed ({e}). Falling back to CPU.")
                self.device = torch.device("cpu")
                self.is_tpu = False
                self.spmd_mesh = None
                self.is_spmd = False
                self.world_size = 1
                self.rank = 0
                self.is_master = True
            except RuntimeError:
                # Re-raise hardware lock / initialization errors so users don't silently train on CPU
                raise
        elif str(device_type).lower() == "cuda":
            if local_rank != -1 and torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                self.device = torch.device(f"cuda:{local_rank}")
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend="nccl")
                self.world_size = int(os.environ.get("WORLD_SIZE", 1))
                self.rank = int(os.environ.get("RANK", 0))
                self.is_master = (self.rank == 0)
                self.is_ddp = True
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.world_size = 1
                self.rank = 0
                self.is_master = True
                self.is_ddp = False

            self.is_tpu = False
            self.spmd_mesh = None
            self.is_spmd = False
            self.n_gpus = torch.cuda.device_count()
            # Check native BF16 support (e.g. Turing T4 does not support BF16, Ampere/Ada/Hopper do)
            if not torch.cuda.is_bf16_supported() and self.precision == "bf16":
                print("  [Notice] Hardware lacks native BF16 support. Automatically falling back to FP16 with GradScaler.")
                self.precision = "fp16"
        else:
            self.device = torch.device(device_type)
            self.is_tpu = False
            self.spmd_mesh = None
            self.is_spmd = False
            self.world_size = 1
            self.rank = 0
            self.is_master = True
            self.is_ddp = False

        if getattr(self, "is_tpu", False):
            try:
                import torch_xla.core.xla_model as xm
                xm.set_rng_state(self.seed)
            except Exception:
                pass

        self.model.to(self.device)
        self.special_lut = self.special_lut.to(self.device)

        # Keep model parameters in float32 on TPU to prevent optimizer update underflow
        self.use_master_weights = False
        if self.is_tpu and self.precision in ["bfloat16", "bf16"]:
            self.model.to(dtype=torch.float32)
            print("  [Precision] TPU model parameters kept in float32 (AdamW updates preserved; forward pass autocast to bfloat16).")

        # Re-tie weights after moving to device for PyTorch-XLA compatibility
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()
        elif hasattr(getattr(self.model, "module", None), "tie_weights"):
            self.model.module.tie_weights()

        # Multi-GPU wrapping: Prefer DDP over deprecated DataParallel
        if getattr(self, "is_ddp", False):
            # Allow unused parameters in DDP only when reliability head is frozen
            find_unused = (
                self.paradigm == "corosred"
                and not getattr(self, "is_unified", False)
                and str(self.phase).upper() in ["B", "C"]
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                find_unused_parameters=find_unused
            )
            if self.is_master:
                print(f"  [Hardware] Multi-GPU DDP initialized across {self.world_size} processes.")
        elif getattr(self, "n_gpus", 1) > 1 and not self.is_tpu and self.device.type == "cuda":
            print("  [Hardware Notice] Multi-GPU detected without torchrun. For 1.8x-7x throughput scaling, launch with 'torchrun'. Falling back to DataParallel.")
            self.model = nn.DataParallel(self.model)

        # Select gradient checkpointing based on device VRAM and model dimension
        cuda_needs_chkpt = False
        if self.device.type == "cuda" and not self.is_tpu:
            cuda_gb = 0.0
            try:
                cuda_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            except Exception:
                pass
            if (cuda_gb > 0 and cuda_gb <= 16.0) or (self.m_cfg.get("d_model", 512) >= 1024):
                cuda_needs_chkpt = True

        # Enable checkpointing on TPU only for large models (d_model >= 1536)
        tpu_needs_chkpt = self.is_tpu and (self.m_cfg.get("d_model", 512) >= 1536 or self.t_cfg.get("batch_size", 32) > 128)
        auto_chkpt = (
            tpu_needs_chkpt
```


### telos/training/trainer_pytorch.py, lines 249–377

SHA-256 of embedded excerpt/source: `9bea9d139b94c77c70c8066f437da6c17b5b79b641c1d82692c1cf3a7c577417`

```python
        self.max_steps = int(self.t_cfg.get("max_steps", 5000))
        self.max_lr = float(self.t_cfg.get("max_lr", 3e-4))
        self.min_lr = float(self.t_cfg.get("min_lr", 3e-5))
        self.warmup_steps = int(self.t_cfg.get("warmup_steps", 100))
        self.weight_decay = float(self.t_cfg.get("weight_decay", 0.1))
        self.grad_clip = float(self.t_cfg.get("grad_clip", 1.0))

        if self.paradigm == "corosred":
            if getattr(self, "is_unified", False):
                self.schedule = COROSredSchedule(
                    max_steps=self.max_steps,
                    alpha_max=float(self.crsr_cfg.get("alpha_max", 0.85)),
                    alpha_min=float(self.crsr_cfg.get("alpha_min", 0.20)),
                    beta_min=float(self.crsr_cfg.get("beta_min", 0.15)),
                    beta_max=float(self.crsr_cfg.get("beta_max", 0.70)),
                    gamma_max=float(self.crsr_cfg.get("gamma_max", 0.10)),
                    hold_fraction=float(self.crsr_cfg.get("hold_fraction", 0.20)),
                    decay_power=float(self.crsr_cfg.get("decay_power", 2.5)),
                    gamma_gate_auc=float(self.crsr_cfg.get("gamma_gate_auc", 0.55)),
                    acc_gate_threshold=float(self.crsr_cfg.get("acc_gate_threshold", 0.65)),
                )
                default_cadence = int(self.crsr_cfg.get("default_cadence", self.t_cfg.get("default_cadence", 25 if self.is_tpu else 1)))
                cadence = int(self.crsr_cfg.get("cadence", self.t_cfg.get("cadence", default_cadence)) or default_cadence)
                self.metric_tracker = DynamicMetricTracker(
                    trust_region=float(self.crsr_cfg.get("trust_region", 0.20)),
                    rebalance_temp=float(self.crsr_cfg.get("rebalance_temp", 0.25)),
                    cadence=cadence,
                )
                self.routing_cache = RoutingMaskCache(
                    refresh_every_steps=int(self.crsr_cfg.get("routing_cache_steps", 50)),
                    k_targeted_ratio=float(self.crsr_cfg.get("k_targeted_ratio", 0.70)),
                )
                self.adaptive_rebalance = bool(self.crsr_cfg.get("adaptive_rebalance", False))
                self.causal_ratio = float(self.crsr_cfg.get("causal_ratio", 0.60))
                self.dual_monitor = None
                if self.is_master:
                    print(
                        f"  [COROSred Unified] Initialized continuous multi-objective loop: "
                        f"alpha=[{self.schedule.alpha_max:.2f}->{self.schedule.alpha_min:.2f} floor], "
                        f"beta=[{self.schedule.beta_min:.2f}->{self.schedule.beta_max:.2f}], "
                        f"gamma_max={self.schedule.gamma_max:.2f}, hold={self.schedule.hold_fraction*100:.0f}%, "
                        f"adaptive_rebalance={self.adaptive_rebalance}."
                    )

        # Separate parameters into decayed and non-decayed groups:
        # Standard transformer optimization applies 0 weight decay to 1D parameters (biases, layer norms, RMS norms)
        # and embedding lookup tables to avoid regularizing scale/shift parameters.
        self.param_to_master = []
        decay_params = []
        no_decay_params = []
        # In PyTorch, named_parameters() does not deduplicate if tensors are on separate memory addresses.
        # Track seen parameter object IDs to strictly prevent tied embeddings from being placed in both groups.
        seen_param_ids = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_id = id(param)
                if param_id in seen_param_ids:
                    continue
                seen_param_ids.add(param_id)

                if self.use_master_weights:
                    # Allocate fp32 master parameter copy to prevent updates from rounding to zero in bf16
                    mp = param.detach().clone().float().requires_grad_(True)
                    self.param_to_master.append((param, mp))
                    target_param = mp
                else:
                    target_param = param

                if param.ndim == 1 or "embed" in name:
                    no_decay_params.append(target_param)
                else:
                    decay_params.append(target_param)

        self.master_params = [mp for _, mp in self.param_to_master] if self.use_master_weights else None

        # Fused AdamW merges kernel operations for faster gradient updates on CUDA
        use_fused = (self.device.type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        opt_kwargs = {
            "lr": self.max_lr,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
        }
        if use_fused:
            opt_kwargs["fused"] = True

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            **opt_kwargs
        )

        # Quantize LR updates to cadence (defaults to default_cadence: 25 on TPU, 1 on CUDA/CPU)
        default_lr_cadence = int(self.t_cfg.get("default_cadence", 25 if self.is_tpu else 1))
        lr_cadence = int(self.t_cfg.get("lr_cadence", default_lr_cadence) or default_lr_cadence)
        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            min_lr=self.min_lr,
            update_cadence=lr_cadence
        )

        # Autocast AMP precision
        if self.is_tpu:
            # Enable hardware bfloat16 autocast on TPU to utilize systolic MXU arrays
            # while parameters in HBM remain in float32 for unadulterated AdamW updates.
            self.use_amp = (self.precision in ["fp16", "bf16", "bfloat16"])
            self.amp_device = "xla"
            self.amp_dtype = torch.bfloat16
        else:
            self.use_amp = (self.precision in ["fp16", "bf16"]) and (self.device.type in ["cuda", "mps"])
            self.amp_device = self.device.type
            self.amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16

        # Modern GradScaler API for FP16 training on CUDA
        self.use_scaler = (self.precision == "fp16") and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_scaler else None

        # Pre-sampled device-resident Beta(1.5, 1.5) buffer to eliminate per-microbatch host RNG & PCIe syncs
        if self.paradigm in ["mdlm", "undlm"]:
            self.beta_buffer_size = 1_000_000
            beta_np = sample_beta_timesteps(self.beta_buffer_size)
            # Retain float32 precision across all backends (MLX, CUDA, TPU) for exact masking probabilities
            self.beta_buffer = torch.from_numpy(beta_np).to(device=self.device, dtype=torch.float32)
            self.beta_counter = torch.zeros((), dtype=torch.int64, device=self.device)
```


### telos/training/trainer_pytorch.py, lines 403–488

SHA-256 of embedded excerpt/source: `fcc5726b63f3c4680f2f8d432de9cab854aa2a0c7f5ee7bec54c67a5ac7817d8`

```python
    def save_checkpoint(self, path: str | Path, sync: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Join any prior background save thread to ensure two saves never overlap or race
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()

        if self.is_tpu:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
            str_path = str(path)

            # Detach model parameters to host CPU on master rank without saving optimizer state
            raw_model = getattr(self.model, "module", self.model)
            cpu_model = {
                k.removeprefix("module."): v.detach().cpu().clone()
                for k, v in raw_model.state_dict().items()
            }
            checkpoint = {
                "global_step": self.global_step,
                "model_state_dict": cpu_model,
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.cfg,
                "lrh_acc_ema": getattr(self, "metric_tracker", None).lrh_acc_ema if getattr(self, "metric_tracker", None) is not None else None,
                "lrh_auc_ema": getattr(self, "metric_tracker", None).lrh_auc_ema if getattr(self, "metric_tracker", None) is not None else None,
            }
            torch.save(checkpoint, str_path)

            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"FATAL: Checkpoint file could not be created at {str_path}!")

            print(f"  [Checkpoint Verified] Successfully saved {str_path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
            return

        # Clone state dicts to detached CPU tensors to prevent torn-state corruption from concurrent in-place mutations
        cpu_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        raw_opt_state = self.optimizer.state_dict()
        cpu_opt_state = {
            "state": {
                k: {sk: sv.detach().cpu().clone() if isinstance(sv, torch.Tensor) else sv for sk, sv in v.items()}
                for k, v in raw_opt_state.get("state", {}).items()
            },
            "param_groups": raw_opt_state.get("param_groups", [])
        }

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": cpu_model_state,
            "optimizer_state_dict": cpu_opt_state,
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.cfg,
        }

        if sync:
            torch.save(checkpoint, path)
        else:
            # Asynchronous checkpoint saving on a background worker thread with cloned CPU tensors
            def _async_save():
                with self._save_lock:
                    torch.save(checkpoint, path)
            self._save_thread = threading.Thread(target=_async_save, daemon=True)
            self._save_thread.start()

    def load_checkpoint(self, path: str | Path):
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        # Unwrap DataParallel if state_dict keys match
        state_dict = checkpoint["model_state_dict"]
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        # Re-tie weights after loading checkpoint state dict
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()
        elif hasattr(getattr(self.model, "module", None), "tie_weights"):
            self.model.module.tie_weights()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)

        # Synchronize fp32 master weights from loaded model parameters to prevent overwriting with stale init values
        if getattr(self, "use_master_weights", False):
            for p, mp in self.param_to_master:
```


### telos/training/trainer_pytorch.py, lines 491–545

SHA-256 of embedded excerpt/source: `fcccb415a1e9a17c7bbf9966842a850eaa5552f3c4ae1e215e6d30f43085af90`

```python
    def _execute_microbatch(self, batch_seqs, compute_metrics: bool = False):
        """Executes a single microbatch and returns loss and metrics."""
        loss = None
        metrics = None

        if self.paradigm == "ar":
            logits = self.model(batch_seqs, mask_override=True)
            loss, metrics = ar_loss_fn_pytorch(logits, batch_seqs, special_token_lut=self.special_lut)

        elif self.paradigm == "mdlm":
            bs = batch_seqs.shape[0]
            t_vals = self._sample_beta_timesteps(bs)
            masked_ids, mask_pos, t_vals_out = apply_masking_pytorch(batch_seqs, t_vals, mask_token_id=1, special_token_lut=self.special_lut)
            logits = self.model(masked_ids, mask_override=False)
            loss, metrics = mdlm_loss_pytorch(logits, batch_seqs, mask_pos, t_vals_out)

        elif self.paradigm == "undlm":
            bs = batch_seqs.shape[0]
            t_vals = self._sample_beta_timesteps(bs)
            noisy_ids, corrupt_mask, t_vals_out = apply_uniform_noise_pytorch(batch_seqs, t_vals, self.vocab_size, special_token_lut=self.special_lut)
            logits = self.model(noisy_ids, mask_override=False)
            loss, metrics = undlm_loss_pytorch(logits, batch_seqs, t_vals_out, special_token_lut=self.special_lut)

        elif self.paradigm == "corosred":
            if getattr(self, "is_unified", False):
                mask_token_id = self.m_cfg.get("mask_token_id", 1)
                mask_prob = float(self.crsr_cfg.get("mask_prob", 0.20))
                k_amb = int(self.crsr_cfg.get("k_amb", 5))
                
                # Fetch continuous schedule weights driven by step progress and empirical EMAs.
                # Defaults to 25 on TPU (to avoid XLA recompilations) and 1 on CUDA/CPU unless manually specified.
                default_sched_cadence = int(self.crsr_cfg.get("default_cadence", self.t_cfg.get("default_cadence", 25 if self.is_tpu else 1)))
                sched_cadence = int(self.crsr_cfg.get("sched_cadence", self.t_cfg.get("sched_cadence", default_sched_cadence)) or default_sched_cadence)
                sched_step = (self.global_step // sched_cadence) * sched_cadence if sched_cadence > 1 else self.global_step
                lrh_acc_ema = self.metric_tracker.lrh_acc_ema if hasattr(self, "metric_tracker") else None
                lrh_auc_ema = self.metric_tracker.lrh_auc_ema if hasattr(self, "metric_tracker") else None
                sched_w = self.schedule.get_weights(sched_step, lrh_acc_ema, lrh_auc_ema)

                loss, metrics = corosred_unified_step_pytorch(
                    self.model,
                    batch_seqs,
                    self.vocab_size,
                    schedule_weights=sched_w,
                    mask_token_id=mask_token_id,
                    mask_prob=mask_prob,
                    special_token_lut=self.special_lut,
                    k_amb=k_amb,
                    causal_ratio=getattr(self, "causal_ratio", 0.60),
                    routing_cache=getattr(self, "routing_cache", None),
                    metric_tracker=getattr(self, "metric_tracker", None),
                    adaptive_rebalance=getattr(self, "adaptive_rebalance", False),
                    compute_metrics=compute_metrics,
                )
            elif self.phase == "A":
                # Phase A: Causal Autoregressive backbone pretraining + detachable Learned Reliability Head
```


### telos/training/trainer_pytorch.py, lines 775–915

SHA-256 of embedded excerpt/source: `af2ec248b79b7ca45ca4c2d6ca1f88e7e9b1ea21c037590ea14cc53ad5f763c5`

```python
                global_targets = item
            else:
                global_targets, _ = get_global_targets_contiguous_pytorch(
                    dataset_matrix, idx_ptr, local_step_seqs, self.seq_len, self.device, non_blocking=False
                )
                idx_ptr = (idx_ptr + cluster_step_seqs) % n_rows
            # Periodically replenish routing cache only once confidence-routed masking is active (after hold phase)
            has_routing = getattr(self, "is_unified", False) and hasattr(self, "routing_cache")
            if has_routing and self.routing_cache.should_refresh(step):
                sched_w_check = self.schedule.get_weights(step)
                if sched_w_check.get("mask_blend", 0.0) > 0.0:
                    self.routing_cache.refresh(
                        self.model,
                        upcoming_seqs=global_targets,
                        current_step=step,
                        mask_prob=float(self.crsr_cfg.get("mask_prob", 0.15)),
                    )

            last_metrics = None
            
            for i in range(grad_accum):
                batch_seqs = global_targets[i * bs : (i + 1) * bs]

                # In PyTorch-XLA SPMD mode, shard batch dimension across TPU chips so PJRT
                # produces valid PjRtShardedData buffers and avoids ExecuteReplicated NULL pointer SIGSEGV.
                if getattr(self, "is_spmd", False) and self.spmd_mesh is not None:
                    import torch_xla.distributed.spmd as xs
                    xs.mark_sharding(batch_seqs, self.spmd_mesh, ("data", None))

                is_log_step = (
                    step <= 5
                    or (step <= 50 and step % 10 == 0)
                    or step % 50 == 0
                    or step == self.max_steps
                    or (benchmark and step % 10 == 0)
                )
                eval_metrics = (i == grad_accum - 1) and is_log_step
                # DDP gradient accumulation optimization: disable all-reduce on non-final microbatches
                sync_ctx = self.model.no_sync() if (getattr(self, "is_ddp", False) and i < grad_accum - 1) else nullcontext()
                with sync_ctx:
                    if self.use_amp:
                        with torch.amp.autocast(device_type=getattr(self, "amp_device", self.device.type), dtype=self.amp_dtype):
                            loss, metrics = self._execute_microbatch(batch_seqs, compute_metrics=eval_metrics)
                    else:
                        loss, metrics = self._execute_microbatch(batch_seqs, compute_metrics=eval_metrics)

                    loss = loss / grad_accum
                    if self.use_scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                last_metrics = metrics

            if self.use_scaler:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            elif self.is_tpu:
                import torch_xla.core.xla_model as xm
                # Copy gradients from bfloat16 model parameters to float32 master parameters
                if self.use_master_weights:
                    for p, mp in self.param_to_master:
                        if p.grad is not None:
                            mp.grad = p.grad.float()

                # In SPMD mode, xs.mark_sharding on the batch automatically triggers the partitioner all-reduce.
                if not getattr(self, "is_spmd", False):
                    xm.reduce_gradients(self.optimizer)
                if self.grad_clip > 0:
                    target_params = self.master_params if self.use_master_weights else self.model.parameters()
                    nn.utils.clip_grad_norm_(target_params, self.grad_clip)
                # Call self.optimizer.step() directly: xm.optimizer_step() would redundantly invoke
                # reduce_gradients() a second time, multiplying gradients by 1/(world_size^2) (1/64x on 8 cores).
                self.optimizer.step()

                # Synchronize updated float32 master weights back to bfloat16 model parameters
                if self.use_master_weights:
                    for p, mp in self.param_to_master:
                        p.data.copy_(mp.data.to(p.dtype))

                xm.mark_step()

            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.optimizer.zero_grad()
            if self.use_master_weights:
                self.model.zero_grad()
            self.scheduler.step()
            self.global_step = step

            # Synchronized cadence all-reduce of LRH AUC / balanced accuracy across all 8 replicas.
            # Executed synchronously by ALL ranks outside `if self.is_master` to ensure 100% identical schedule EMAs.
            default_cadence = int(self.crsr_cfg.get("default_cadence", self.t_cfg.get("default_cadence", 25 if self.is_tpu else 1)))
            cadence = int(self.crsr_cfg.get("cadence", self.t_cfg.get("cadence", default_cadence)) or default_cadence)
            if getattr(self, "is_unified", False) and (step % cadence == 0 or step == 1) and last_metrics is not None:
                raw_auc = last_metrics.get("lrh_bal_acc", last_metrics.get("lrh_auc", None))
                raw_acc = last_metrics.get("lrh_acc", None)
                if raw_auc is not None:
                    if isinstance(raw_auc, torch.Tensor):
                        auc_t = raw_auc.detach().clone()
                        acc_t = raw_acc.detach().clone() if isinstance(raw_acc, torch.Tensor) else None

                        # Cheap scalar all-reduce of 0D tensors across all replicas
                        if self.is_tpu:
                            import torch_xla.core.xla_model as xm
                            # Use cached world_size or xworld_size() to avoid non-existent xrt_world_size attribute
                            world_size = xm.xworld_size() if hasattr(xm, "xworld_size") else getattr(self, "world_size", 1)
                            if world_size > 1:
                                reduce_op = getattr(xm, "REDUCE_SUM", "sum")
                                tensors_to_reduce = [auc_t] + ([acc_t] if acc_t is not None else [])
                                reduced = xm.all_reduce(reduce_op, tensors_to_reduce, scale=1.0 / float(world_size))
                                auc_t = reduced[0]
                                if acc_t is not None:
                                    acc_t = reduced[1]
                        elif getattr(self, "is_ddp", False):
                            import torch.distributed as dist
                            dist.all_reduce(auc_t, op=dist.ReduceOp.SUM)
                            auc_t /= dist.get_world_size()
                            if acc_t is not None:
                                dist.all_reduce(acc_t, op=dist.ReduceOp.SUM)
                                acc_t /= dist.get_world_size()

                        auc_val = float(auc_t.cpu().item())
                        acc_val = float(acc_t.cpu().item()) if acc_t is not None else auc_val
                    else:
                        auc_val = float(raw_auc)
                        acc_val = float(raw_acc) if raw_acc is not None else auc_val

                    # Update local metric tracker on ALL replicas with identical cluster-wide reduced scalar
                    if getattr(self, "metric_tracker", None) is not None:
                        self.metric_tracker.update_lrh(acc=acc_val, auc=auc_val, cadence=cadence)

            if self.is_tpu and step % 50 == 0:
                import gc
                gc.collect()
                try:
                    import ctypes
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
```


### telos/training/trainer_pytorch.py, lines 985–1004

SHA-256 of embedded excerpt/source: `259608f657004bdd696cf5491e67e644591177ad8f951d95e9d317a242de5e79`

```python
                if getattr(self, "is_unified", False) and getattr(self, "dual_monitor", None) is not None and self.is_master and not self.is_tpu:
                    probe_res = self.dual_monitor.evaluate(self.model, current_step=step)
                    print(
                        f"  [Dual Probe Step {step:>6d}] Causal Top-1: {probe_res['causal_top1']*100:.1f}% | "
                        f"C-CE: {probe_res['causal_ce']:.3f} | Infill Top-1: {probe_res['infill_top1']*100:.1f}% | "
                        f"I-CE: {probe_res['infill_ce']:.3f} | LRH AUC: {probe_res['lrh_auc']:.3f}",
                        flush=True
                    )
                    div_warn = self.dual_monitor.check_divergence()
                    if div_warn:
                        print(f"  {div_warn}", flush=True)
                    if getattr(self, "metric_tracker", None) is not None:
                        self.metric_tracker.update_lrh(probe_res["lrh_acc"], probe_res["lrh_auc"])

                # In PyTorch-XLA multi-processing, all workers must synchronize at step boundaries
                # so master rank can serialize model weights without worker desynchronization
                if self.is_tpu:
                    import torch_xla.core.xla_model as xm
                    xm.rendezvous(f"checkpoint_save_{step}")
```


### telos/train/cli.py

SHA-256 of embedded excerpt/source: `7f327d9f0c19665b67175a1dfd1af80418fc7ad173103dbebb7e2d4a391366ee`

```python
import os
import sys
import argparse
from pathlib import Path

# Allocator hygiene: Set expandable_segments BEFORE torch is imported
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Glibc memory allocator hygiene: Restrict glibc memory arenas to prevent multi-threaded RSS bloat
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# Thread hygiene: Restrict thread pools BEFORE torch / numpy load to stop multi-core thread storms
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from telos.configs import build_config


def _mp_train_worker(index, kwargs):
    """Worker entrypoint executed on each spawned TPU core."""
    import os
    os.environ["MALLOC_ARENA_MAX"] = "2"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    import torch
    # Set single intra-op thread per worker to prevent CPU thread contention on TPU
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    try:
        import torch_xla
        if not hasattr(torch, "xla"):
            torch.xla = torch_xla
    except ImportError:
        pass
    from telos.training import xla_utils
    xla_utils.clean_tpu_environment()
    # Reset singleton references in forked child process so xm.xla_device() binds to assigned core
    xla_utils._CACHED_XLA_DEVICE = None
    xla_utils._CACHED_XLA_WORLD_SIZE = None
    xla_utils._CACHED_IS_MASTER = None

    from telos.training import set_global_seed
    worker_seed = int(kwargs.get("seed", 42))
    set_global_seed(worker_seed)

    train(**kwargs)



def _load_checkpoint_into_mlx(model, ckpt_path: str | Path):
    """Loads weights from either a safetensors file or a PyTorch (.pt/.bin) checkpoint into an MLX model."""
    import mlx.core as mx
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"  [Init Warning] Checkpoint not found at {ckpt_path}, starting from scratch.")
        return

    if ckpt_path.suffix in (".safetensors", ".npz"):
        print(f"  [Init] Loading native MLX weights from {ckpt_path}...")
        model.load_weights(str(ckpt_path), strict=False)
        return

    # PyTorch (.pt / .bin) checkpoint translation
    import torch
    print(f"  [Init] Loading PyTorch weights into MLX model from {ckpt_path}...")
    ckpt_state = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt_state.get("model_state_dict", ckpt_state)

    mlx_weights = {}
    qkv = {}
    tied_embeddings = getattr(model, "tied_embeddings", True)

    for k, v in sd.items():
        k = k.removeprefix("module.")
        if not isinstance(v, torch.Tensor):
            continue
        arr = mx.array(v.float().numpy()).astype(mx.bfloat16)

        if k == "tok_embeddings.weight":
            mlx_weights["emb.weight"] = arr
        elif k == "final_norm.weight":
            mlx_weights["norm.weight"] = arr
        elif k == "output_projection.weight":
            if not tied_embeddings:
                mlx_weights["head.weight"] = arr
        elif ".attn_norm.weight" in k:
            layer_idx = k.split(".")[1]
            mlx_weights[f"layers.{layer_idx}.norm1.weight"] = arr
        elif ".mlp_norm.weight" in k:
            layer_idx = k.split(".")[1]
            mlx_weights[f"layers.{layer_idx}.norm2.weight"] = arr
        elif ".attn.out_proj.weight" in k:
            layer_idx = k.split(".")[1]
            mlx_weights[f"layers.{layer_idx}.out.weight"] = arr
        elif ".attn.q_proj.weight" in k:
            layer_idx = k.split(".")[1]
            qkv.setdefault(layer_idx, {})["q"] = arr
        elif ".attn.k_proj.weight" in k:
            layer_idx = k.split(".")[1]
            qkv.setdefault(layer_idx, {})["k"] = arr
        elif ".attn.v_proj.weight" in k:
            layer_idx = k.split(".")[1]
            qkv.setdefault(layer_idx, {})["v"] = arr
        elif ".mlp.w1.weight" in k:
            layer_idx = k.split(".")[1]
            mlx_weights[f"layers.{layer_idx}.mlp.w1.weight"] = arr
        elif ".mlp.v.weight" in k:
            layer_idx = k.split(".")[1]
            mlx_weights[f"layers.{layer_idx}.mlp.w2.weight"] = arr
        elif ".mlp.w2.weight" in k:
            layer_idx = k.split(".")[1]
            mlx_weights[f"layers.{layer_idx}.mlp.w3.weight"] = arr
        elif "reliability_head." in k:
            parts = k.split(".")
            mlx_weights[f"reliability_head.layers.{parts[1]}.{parts[2]}"] = arr

    for layer_idx, p in qkv.items():
        if "q" in p and "k" in p and "v" in p:
            mlx_weights[f"layers.{layer_idx}.qkv_proj.weight"] = mx.concatenate([p["q"], p["k"], p["v"]], axis=0)

    model.load_weights(list(mlx_weights.items()), strict=False)
    print(f"  [Init] Successfully loaded {len(mlx_weights)} parameter tensors into MLX model.")


def train(
    paradigm: str = "mdlm",
    phase: str | None = None,
    params: str | int | None = "12M",
    tokens: str | int | None = None,
    effective_batch: int | str | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    seq_len: int = 512,
    d_model: int | None = None,
    n_layers: int | None = None,
    tokenizer: str | None = None,
    vocab_size: int | None = None,
    hardware: str | None = "auto",
    devices: int | str | None = "auto",
    max_steps: int | None = None,
    max_lr: float | None = None,
    min_lr: float | None = None,
    warmup_steps: int | None = None,
    weight_decay: float | None = None,
    checkpoint_dir: str | None = None,
    save_every: int | None = None,
    config_path: str | Path | None = None,
    data_path: str | Path | None = None,
    synthetic: bool = False,
    resume_step: int = 0,
    eval_policy: str = "auto",
    benchmark: bool = False,
    benchmark_duration: float = 300.0,
    self_condition: bool = True,
    self_cond_prob: float = 0.5,
    init_checkpoint: str | Path | None = None,
    compile: bool | None = None,
    seed: int = 42,
    default_cadence: int | None = None,
    **kwargs
):
    """
    Programmatic entrypoint to train any Télos model.
    """
    cfg = build_config(
        default_cadence=default_cadence,
        paradigm=paradigm,
        phase=phase,
        params=params,
        tokens=tokens,
        effective_batch=effective_batch,
        batch_size=batch_size,
        grad_accum=grad_accum,
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        hardware=hardware,
        devices=devices,
        max_steps=max_steps,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir,
        save_every=save_every,
        config_path=config_path,
        data_path=data_path,
        synthetic=synthetic,
        self_condition=self_condition,
        self_cond_prob=self_cond_prob,
        compile=compile,
        seed=seed,
        **kwargs
    )

    backend = cfg["_backend"]
    device = cfg["_device"]
    dev_count = cfg.get("_device_count", 1)
    is_causal = paradigm.lower() in ["ar", "corosred"]

    m_cfg = cfg["model"]
    t_cfg = cfg["training"]

    if not kwargs.get("_is_spawned", False):

        print("=" * 76)
        phase_str = f" (Phase {phase.upper()})" if phase else (" (Unified Continuous)" if paradigm.lower() == "corosred" else "")
        print(f"  TÉLOS UNIFIED TRAINER  |  Paradigm: {paradigm.upper()}{phase_str}")
        print(f"  Hardware Backend:     {backend.upper()} ({device})")
        if "_resolved_params" in cfg:
            print(f"  Target Parameters:    {cfg['_resolved_params']:,} (~{params})")
        dev_multiplier = dev_count if dev_count > 1 else 1
        eff_batch = t_cfg["batch_size"] * t_cfg["gradient_accumulation"] * dev_multiplier
        dev_str = f", devices={dev_count}" if dev_count > 1 else ""
        print(f"  Batch Config:         batch_size={t_cfg['batch_size']}, grad_accum={t_cfg['gradient_accumulation']}{dev_str} (effective={eff_batch} seqs / {eff_batch * m_cfg['seq_len']:,} tok)")
        print(f"  Training Steps:       {t_cfg['max_steps']:,} steps | LR: {t_cfg['max_lr']:.2e} -> {t_cfg['min_lr']:.2e} (warmup={t_cfg['warmup_steps']})")
        print(f"  Checkpoint Dir:       {cfg['checkpoint']['checkpoint_dir']} (every {cfg['checkpoint']['save_every_steps']} steps)")
    if backend == "pytorch" and device == "xla" and dev_count > 1 and not kwargs.get("_is_spawned", False):
        try:
            from telos.training.xla_utils import clean_tpu_environment
            clean_tpu_environment()
            import torch_xla.distributed.xla_multiprocessing as xmp
            print(f"  [Hardware] Launching multi-core PyTorch-XLA execution across {dev_count} TPU cores via xmp.spawn...")
            spawn_args = dict(
                    paradigm=paradigm,
                    phase=phase,
                    params=params,
                    tokens=None,  # Pre-resolved cluster max_steps takes precedence in workers
                    effective_batch=None,
                    batch_size=t_cfg["batch_size"],
                    grad_accum=t_cfg["gradient_accumulation"],
                    seq_len=seq_len,
                    d_model=d_model,
                    n_layers=n_layers,
                    tokenizer=tokenizer,
                    vocab_size=vocab_size,
                    hardware=hardware,
                    devices=dev_count,
                    max_steps=t_cfg["max_steps"],
                    max_lr=t_cfg["max_lr"],
                    min_lr=t_cfg["min_lr"],
                    warmup_steps=t_cfg["warmup_steps"],
                    weight_decay=weight_decay,
                    checkpoint_dir=checkpoint_dir,
                    save_every=save_every,
                    config_path=config_path,
                    data_path=data_path,
                    synthetic=synthetic,
                    resume_step=resume_step,
                    eval_policy=eval_policy,
                    benchmark=benchmark,
                    benchmark_duration=benchmark_duration,
                    self_condition=self_condition,
                    self_cond_prob=self_cond_prob,
                    init_checkpoint=init_checkpoint,
                    compile=compile,
                    seed=t_cfg.get("seed", seed),
                    default_cadence=default_cadence,
                    _is_spawned=True,
                    **kwargs,
                )
            xmp.spawn(_mp_train_worker, args=(spawn_args,), nprocs=None)
            return None

        except Exception as e:
            print(f"  [Notice] Multi-core xmp.spawn skipped ({e}). Proceeding on single core.")

    from telos.training import set_global_seed
    resolved_seed = int(t_cfg.get("seed", kwargs.get("seed", seed)))
    set_global_seed(resolved_seed)

    if backend == "mlx":
        # Defer MLX imports so non-Apple-Silicon environments don't resolve MLX
        from telos.models import MLXTelosTransformer
        from telos.training import UnifiedMLXTrainer

        precision = t_cfg.get("precision", "bfloat16")
        model = MLXTelosTransformer(
            vocab_size=m_cfg.get("vocab_size", 8192),
            d_model=m_cfg.get("d_model", 512),
            n_layers=m_cfg.get("n_layers", 12),
            n_heads=m_cfg.get("n_heads", 16),
            n_kv_heads=m_cfg.get("n_kv_heads", None),
            is_causal=is_causal,
            use_reliability_head=bool(m_cfg.get("use_reliability_head", paradigm.lower() == "corosred")),
            use_grad_checkpoint=t_cfg.get("gradient_checkpointing", False) or m_cfg.get("use_grad_checkpoint", False),
            precision=precision
        )
        # Auto-detect or load initial checkpoint (e.g. chaining Phase A weights into Phase B or Phase C)
        init_ckpt_path = init_checkpoint
        is_unified = cfg.get("corosred", {}).get("unified", False)
        if init_ckpt_path is None and paradigm.lower() == "corosred" and not is_unified and str(phase).upper() in ["B", "C"]:
            cand = Path(cfg["checkpoint"]["checkpoint_dir"]).parent / "phase_a" / "checkpoint_final.pt"
            if cand.exists():
                init_ckpt_path = str(cand)

        if init_ckpt_path:
            ckpt_file = Path(init_ckpt_path)
            if not ckpt_file.exists():
                raise FileNotFoundError(
                    f"Initial checkpoint file '{init_ckpt_path}' does not exist! "
                    f"Aborting to avoid training from uninitialized random weights."
                )
            _load_checkpoint_into_mlx(model, init_ckpt_path)

        trainer = UnifiedMLXTrainer(paradigm=paradigm, model=model, cfg=cfg, eval_policy=eval_policy)
    else:
        from telos.models import TelosTransformer
        from telos.training import UnifiedPyTorchTrainer

        model = TelosTransformer(
            vocab_size=m_cfg.get("vocab_size", 8192),
            d_model=m_cfg.get("d_model", 512),
            n_layers=m_cfg.get("n_layers", 12),
            n_heads=m_cfg.get("n_heads", 16),
            n_kv_heads=m_cfg.get("n_kv_heads", None),
            is_causal=is_causal,
            use_reliability_head=(paradigm.lower() == "corosred"),
            use_grad_checkpoint=t_cfg.get("gradient_checkpointing", False) or m_cfg.get("use_grad_checkpoint", False)
        )


        # Auto-detect or load initial checkpoint (e.g. chaining Phase A weights into Phase B or Phase C)
        init_ckpt_path = init_checkpoint
        is_unified = cfg.get("corosred", {}).get("unified", False)
        if init_ckpt_path is None and paradigm.lower() == "corosred" and not is_unified and str(phase).upper() in ["B", "C"]:
            cand = Path(cfg["checkpoint"]["checkpoint_dir"]).parent / "phase_a" / "checkpoint_final.pt"
            if cand.exists():
                init_ckpt_path = str(cand)

        ckpt_state = None
        if init_ckpt_path:
            ckpt_file = Path(init_ckpt_path)
            if not ckpt_file.exists():
                raise FileNotFoundError(
                    f"Initial checkpoint file '{init_ckpt_path}' does not exist! "
                    f"Aborting to avoid training from uninitialized random weights."
                )
            import torch
            print(f"  [Init] Loading model weights from {init_ckpt_path}...")
            ckpt_state = torch.load(init_ckpt_path, map_location="cpu")
            sd = ckpt_state.get("model_state_dict", ckpt_state)
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            print(f"  [Init] Successfully loaded weights from {init_ckpt_path}.")

        trainer = UnifiedPyTorchTrainer(paradigm=paradigm, model=model, cfg=cfg, device_type=device)
        if ckpt_state is not None and resume_step > 0:
            # Restore optimizer state only on CUDA/CPU to avoid host-device tensor desync on XLA
            if device != "xla" and not getattr(trainer, "is_tpu", False) and "optimizer_state_dict" in ckpt_state:
                try:
                    trainer.optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
                    print(f"  [Resume] Successfully restored AdamW optimizer moment buffers from step {resume_step}.")
                except Exception as e:
                    print(f"  [Resume Notice] Could not restore optimizer state ({e}). Proceeding with fresh optimizer states.")
            elif getattr(trainer, "is_tpu", False):
                print("  [Resume TPU] Optimizer moments initialized natively on TPU device (avoiding cross-device bus stalls).")

            # Restore metric tracker EMAs if available
            tracker = getattr(trainer, "metric_tracker", None)
            if tracker is not None:
                acc_saved = ckpt_state.get("lrh_acc_ema", None)
                auc_saved = ckpt_state.get("lrh_auc_ema", None)
                if acc_saved is not None and auc_saved is not None:
                    tracker.update_lrh(acc=float(acc_saved), auc=float(auc_saved))
                    print(f"  [Resume] Restored LRH telemetry: Acc={acc_saved:.2f}, AUC={auc_saved:.2f}.")

    # Execute training or benchmark
    bench_dur = min(float(benchmark_duration), 300.0)
    trainer.train(
        resume_step=resume_step,
        benchmark=benchmark,
        benchmark_duration=bench_dur
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Télos Unified Dimensional Trainer (Zero Config)")
    
    # 6 Fundamental Dimensions
    parser.add_argument("--paradigm", type=str, default="mdlm", choices=["ar", "mdlm", "undlm", "corosred", "custom"], help="Training paradigm")
    parser.add_argument("--phase", type=str, default=None, help="Phase for legacy COROSred paradigm (A, B, C). If omitted, COROSred runs in unified continuous mode.")
    parser.add_argument("--legacy-phases", action="store_true", help="Force legacy sequential phase execution (A -> B -> C) instead of unified continuous loop")
    parser.add_argument("--alpha-max", type=float, default=0.85, help="Initial causal weight alpha_max (default: 0.85)")
    parser.add_argument("--alpha-min", type=float, default=0.50, help="Permanent floor for causal weight alpha_min (default: 0.50)")
    parser.add_argument("--beta-min", type=float, default=0.15, help="Initial infilling weight beta_min (default: 0.15)")
    parser.add_argument("--beta-max", type=float, default=0.35, help="Peak infilling weight beta_max (default: 0.35)")
    parser.add_argument("--gamma-max", type=float, default=0.10, help="Peak reliability head weight gamma_max (default: 0.10)")
    parser.add_argument("--hold-frac", type=float, default=0.20, help="Fraction of steps to hold alpha at alpha_max (default: 0.20)")
    parser.add_argument("--decay-power", type=float, default=2.5, help="Polynomial exponent p for hold-then-decay schedule (default: 2.5)")
    parser.add_argument("--acc-gate", type=float, default=0.65, help="LRH classification accuracy threshold to activate gamma (default: 0.65)")
    parser.add_argument("--causal-ratio", type=float, default=0.65, help="Fraction of microbatch sequences dedicated to causal pass (default: 0.65)")
    parser.add_argument("--mask-prob", type=float, default=0.20, help="Fraction of infill tokens to mask (default: 0.20)")
    parser.add_argument("--routing-cache-steps", type=int, default=50, help="Steps between pre-computed routing mask refreshes (default: 50)")
    parser.add_argument("--adaptive-rebalance", action="store_true", help="Enable trust-region dynamic loss rebalancing")
    parser.add_argument("--params", type=str, default="12M", help="Target parameter budget (e.g. 12M, 25M, 50M, 100M, 500M)")
    parser.add_argument("--d-model", type=int, default=None, help="Explicit hidden dimension d_model override")
    parser.add_argument("--n-layers", "--layers", dest="n_layers", type=int, default=None, help="Explicit layer count n_layers override")
    parser.add_argument("--tokens", type=str, default=None, help="Target total training tokens (e.g. 2.5B, 300M, 50M)")
    parser.add_argument("--effective-batch", type=str, default=None, help="Target effective batch size in sequences or tokens (e.g. 32, 64, 32k)")
    parser.add_argument("--batch-size", type=int, default=None, help="Microbatch size override")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps override")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to custom BPE tokenizer JSON or HF model")
    parser.add_argument("--vocab-size", type=int, default=None, help="Vocabulary size override")
    parser.add_argument("--hardware", type=str, default="auto", choices=["auto", "mlx", "cuda", "mps", "xla", "cpu", "pytorch"], help="Hardware backend")
    parser.add_argument("--devices", type=str, default="auto", help="Hardware device count (e.g. 1, 4, 8, auto)")

    # Training Dynamics & Checkpoints
    parser.add_argument("--seq-len", type=int, default=512, help="Context sequence length")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps override")
    parser.add_argument("--max-lr", type=float, default=None, help="Peak learning rate override")
    parser.add_argument("--min-lr", type=float, default=None, help="Floor learning rate override")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps override")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay regularization override")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save model checkpoints")
    parser.add_argument("--save-every", type=int, default=None, help="Save checkpoint cadence (steps)")
    parser.add_argument("--resume", "--resume-step", dest="resume", type=int, default=0, help="Step to resume training from")
    parser.add_argument("--eval-policy", type=str, default="auto", choices=["auto", "eager", "step", "lazy"], help="Memory evaluation policy for Apple Silicon MLX")

    # Data & Config Bypass
    parser.add_argument("--data", "--data-path", dest="data", type=str, default=None, help="Path to pre-tokenized binary dataset (.bin)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic token stream without loading disk dataset")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file bypass")

    # Benchmark options
    parser.add_argument("--benchmark", action="store_true", help="Run throughput benchmark (at most 5 minutes)")
    parser.add_argument("--benchmark-duration", type=int, default=300, help="Benchmark duration in seconds (capped at 300)")

    # COROSred Self-Conditioning options
    parser.add_argument("--self-condition", action=argparse.BooleanOptionalAction, default=True, help="Enable self-conditioned draft training in COROSred Phase B")
    parser.add_argument("--self-cond-prob", type=float, default=0.5, help="Probability of training on model drafts vs clean masks in Phase B")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Path to initial checkpoint to load weights from before training")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None, help="Enable torch.compile for model execution")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for initialization and training reproducibility (default: 42)")
    # Cadence & Dynamics Overrides
    parser.add_argument("--default-cadence", "--default_cadence", dest="default_cadence", type=int, default=None, help="Default cadence across all cadence operations (default: 25 on TPU, 1 on CUDA/CPU)")
    parser.add_argument("--sched-step", "--sched-cadence", dest="sched_step", type=int, default=None, help="Cadence for schedule updates (default: 25 on TPU, 1 on CUDA/CPU)")
    parser.add_argument("--lr-cadence", type=int, default=None, help="Cadence for learning rate updates (default: 10 on TPU, 1 on CUDA/CPU)")
    parser.add_argument("--cadence", type=int, default=None, help="Cadence for metric reduction and synchronization (default: 25 on TPU, 1 on CUDA/CPU)")

    args = parser.parse_args()

    try:
        train(
            default_cadence=args.default_cadence,
            sched_step=args.sched_step,
            lr_cadence=args.lr_cadence,
            cadence=args.cadence,
            paradigm=args.paradigm,
            phase=args.phase,
            params=args.params,
            tokens=args.tokens,
            effective_batch=args.effective_batch,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            tokenizer=args.tokenizer,
            vocab_size=args.vocab_size,
            hardware=args.hardware,
            devices=args.devices,
            max_steps=args.max_steps,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            save_every=args.save_every,
            config_path=args.config,
            data_path=args.data,
            synthetic=args.synthetic,
            resume_step=args.resume,
            eval_policy=args.eval_policy,
            benchmark=args.benchmark,
            benchmark_duration=args.benchmark_duration,
            self_condition=args.self_condition,
            self_cond_prob=args.self_cond_prob,
            init_checkpoint=args.init_checkpoint,
            compile=getattr(args, "compile", None),
            legacy_phases=args.legacy_phases,
            alpha_max=args.alpha_max,
            alpha_min=args.alpha_min,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            gamma_max=args.gamma_max,
            hold_fraction=args.hold_frac,
            decay_power=args.decay_power,
            acc_gate_threshold=args.acc_gate,
            causal_ratio=args.causal_ratio,
            mask_prob=args.mask_prob,
            routing_cache_steps=args.routing_cache_steps,
            adaptive_rebalance=args.adaptive_rebalance,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
```


### telos/training/xla_utils.py

SHA-256 of embedded excerpt/source: `b4403034981a39215167a46013828f7ef1637d7ce34c79adf316ff459776e7ba`

```python
"""
PyTorch-XLA device and runtime management utilities.
Caches device and topology singletons to prevent 'InitializeComputationClient() can only be called once'
fatal runtime assertions when training multiple phases or re-running in notebook/interactive environments.
"""

import os
from pathlib import Path

_CACHED_XLA_DEVICE = None
_CACHED_XLA_WORLD_SIZE = None
_CACHED_IS_MASTER = None
_CACHED_SPMD_MESH = None


def is_tpu_environment() -> bool:
    """Non-intrusively checks if the current runtime environment is a Google Cloud TPU or TPU VM."""
    return bool(
        os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
        or "TPU_NAME" in os.environ
        or "CLOUD_TPU_TASK_ID" in os.environ
        or "TPU_PROCESS_ADDRESSES" in os.environ
        or "COLAB_TPU_ADDR" in os.environ
    )


def is_xla_initialized() -> bool:
    """Returns True if the XLA device singleton has already been initialized in this process."""
    return _CACHED_XLA_DEVICE is not None


def clean_tpu_environment():
    """
    Cleans conflicting Kaggle/GCP environment variables that trigger fatal
    SliceBuilder port lookup crashes in PyTorch-XLA PJRT multiprocessing,
    and configures LibTPU runtime flags to eliminate host CPU busy-polling.
    """
    if os.environ.get("TPU_PROCESS_ADDRESSES", "").strip().lower() == "local":
        os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    if "CLOUD_TPU_TASK_ID" in os.environ and "TPU_NAME" not in os.environ:
        os.environ.pop("CLOUD_TPU_TASK_ID", None)

    # Purge any unrecognized spin-wait flags from LIBTPU_INIT_ARGS that abort gflags startup:
    if "LIBTPU_INIT_ARGS" in os.environ:
        cleaned_args = " ".join(arg for arg in os.environ["LIBTPU_INIT_ARGS"].split() if "spin_wait" not in arg)
        if cleaned_args:
            os.environ["LIBTPU_INIT_ARGS"] = cleaned_args
        else:
            os.environ.pop("LIBTPU_INIT_ARGS", None)

    # OpenMP / BLAS thread explosion and glibc arena bloat prevention across spawned processes:
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def get_xla_device():
    """
    Returns the XLA device singleton.
    Guarantees xm.xla_device() is called at most once per process lifetime,
    preventing 'InitializeComputationClient() can only be called once' fatal assertions.
    """
    global _CACHED_XLA_DEVICE
    if _CACHED_XLA_DEVICE is not None:
        return _CACHED_XLA_DEVICE

    clean_tpu_environment()
    import torch
    try:
        import torch_xla
        if not hasattr(torch, "xla"):
            torch.xla = torch_xla
    except ImportError:
        pass
    import torch_xla.core.xla_model as xm
    _CACHED_XLA_DEVICE = xm.xla_device()
    return _CACHED_XLA_DEVICE



def get_xla_spmd_mesh():
    """Returns the active SPMD Mesh singleton across addressable TPU chips, or None."""
    global _CACHED_SPMD_MESH
    if _CACHED_SPMD_MESH is not None:
        return _CACHED_SPMD_MESH

    try:
        import torch_xla.runtime as xr
        if xr.is_spmd():
            import torch_xla.distributed.spmd as xs
            import numpy as np
            n_dev = xr.global_runtime_device_count()
            _CACHED_SPMD_MESH = xs.Mesh(np.arange(n_dev), (n_dev,), ("data",))
            return _CACHED_SPMD_MESH
    except Exception:
        pass
    return None


def get_xla_world_size() -> int:
    """Returns the XLA world size (SPMD chip count or standard XRT world size)."""
    global _CACHED_XLA_WORLD_SIZE
    if _CACHED_XLA_WORLD_SIZE is not None:
        return _CACHED_XLA_WORLD_SIZE

    # If XLA is not yet initialized in this process, read environment variables first
    # to avoid premature ComputationClient initialization before the trainer starts.
    if not is_xla_initialized():
        for env_k in ("TPU_NUM_DEVICES", "WORLD_SIZE", "PJRT_LOCAL_PROCESS_COUNT"):
            if env_k in os.environ:
                try:
                    return max(1, int(os.environ[env_k]))
                except ValueError:
                    pass
        if is_tpu_environment():
            try:
                vfio_chips = list(Path("/dev/vfio").glob("[0-9]*"))
                if len(vfio_chips) >= 4:
                    return 8
                elif len(vfio_chips) > 0:
                    return len(vfio_chips) * 2
            except Exception:
                pass
            return 8

    try:
        import torch_xla.runtime as xr
        if hasattr(xr, "world_size"):
            ws = xr.world_size()
            if ws > 1:
                _CACHED_XLA_WORLD_SIZE = ws
                return _CACHED_XLA_WORLD_SIZE
        if xr.is_spmd():
            _CACHED_XLA_WORLD_SIZE = xr.global_runtime_device_count()
            return max(1, _CACHED_XLA_WORLD_SIZE)
    except Exception:
        pass

    try:
        import torch_xla.core.xla_model as xm
        if hasattr(xm, "xrt_world_size"):
            ws = xm.xrt_world_size()
            if ws > 1:
                _CACHED_XLA_WORLD_SIZE = ws
                return _CACHED_XLA_WORLD_SIZE
    except Exception:
        pass

    _CACHED_XLA_WORLD_SIZE = 1
    return _CACHED_XLA_WORLD_SIZE


def is_xla_master() -> bool:
    """Returns True if this process is the master ordinal (0)."""
    global _CACHED_IS_MASTER
    if _CACHED_IS_MASTER is not None:
        return _CACHED_IS_MASTER

    try:
        import torch_xla.runtime as xr
        if hasattr(xr, "process_index"):
            _CACHED_IS_MASTER = (xr.process_index() == 0)
            return _CACHED_IS_MASTER
    except Exception:
        pass

    try:
        import torch_xla.core.xla_model as xm
        if hasattr(xm, "is_master_ordinal"):
            _CACHED_IS_MASTER = xm.is_master_ordinal()
            return _CACHED_IS_MASTER
    except Exception:
        pass

    _CACHED_IS_MASTER = True
    return _CACHED_IS_MASTER
```


### telos/training/core.py, set_global_seed helper

SHA-256 of embedded excerpt/source: `9fe2504904255e8032767fac63224af250e5e3ee987defe14f91b5604a6ac0c3`

```python
def set_global_seed(seed: int = 42):
    """Sets random seed across Python, NumPy, PyTorch (CPU/CUDA/XLA), and MLX."""
    import random
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)
    except Exception:
        pass

    if MLX_AVAILABLE:
        try:
            mx.random.seed(seed)
        except Exception:
            pass
```


# Appendix D — Representative historical differences

The historical revision a2f09a0 is a verified 17 September 2026 repository revision. Its association with a particular saved training run is not proven. These diffs identify implementation evolution, not a controlled training experiment.


### a2f09a0 → inspected working tree: telos/diffusion/corosred_unified.py

SHA-256 of embedded excerpt/source: `427e0416581ca94bcef8ab9f0256edec0c558f1ead42a5d4384b183ad9ca4862`

```diff
diff --git a/telos/diffusion/corosred_unified.py b/telos/diffusion/corosred_unified.py
index 547682c..ac41328 100644
--- a/telos/diffusion/corosred_unified.py
+++ b/telos/diffusion/corosred_unified.py
@@ -252,12 +252,16 @@ def corosred_unified_step_pytorch(
 
         # Balanced classification accuracy: 0.5 * (TPR + TNR), fully immune to ~9:1 class imbalance
         # Evaluates to 0.50 under chance / constant predictions, establishing an uncorrupted discrimination baseline
-        pos_mask = (labels == 1.0) & valid_mask
-        neg_mask = (labels == 0.0) & valid_mask
-        pos_count = pos_mask.sum().float().clamp(min=1.0)
-        neg_count = neg_mask.sum().float().clamp(min=1.0)
-        tpr = ((valid_preds == 1.0) & pos_mask).sum().float() / pos_count
-        tnr = ((valid_preds == 0.0) & neg_mask).sum().float() / neg_count
+        valid_count = valid_mask.float().sum().clamp(min=1.0)
+
+        preds = (torch.sigmoid(raw_r_scores[:, :-1]) >= 0.5)
+        correct_preds = (preds == (labels > 0.5)) & valid_mask
+        batch_lrh_acc = correct_preds.float().sum() / valid_count
+
+        pos_mask = (labels > 0.5) & valid_mask
+        neg_mask = (labels <= 0.5) & valid_mask
+        tpr = (preds & pos_mask).float().sum() / pos_mask.float().sum().clamp(min=1.0)
+        tnr = ((~preds) & neg_mask).float().sum() / neg_mask.float().sum().clamp(min=1.0)
         batch_lrh_bal_acc = 0.5 * (tpr + tnr)
 
         if not is_accelerator:
@@ -267,21 +271,18 @@ def corosred_unified_step_pytorch(
             flat_y = labels[valid_mask]
             batch_lrh_auc_val = compute_vectorized_roc_auc(flat_r, flat_y)
         else:
-            # On TPU/GPU accelerators, maintain 100% static computation graph:
-            # Avoid mid-forward device-to-host .cpu() synchronization and dynamic-shape boolean masking.
-            # Balanced accuracy (0.50-1.00 scale) is a shape-stable 0D device tensor proxy for ROC-AUC.
+            # Use balanced accuracy as 0D tensor proxy on accelerators to avoid device sync
             batch_lrh_acc_val = batch_lrh_acc
             batch_lrh_bal_acc_val = batch_lrh_bal_acc
             batch_lrh_auc_val = batch_lrh_bal_acc
 
-    # LRH Binary Cross Entropy Loss (always evaluated to ensure 100% static XLA computation graph across all steps)
+    # LRH Binary Cross Entropy Loss
     shift_r_scores = raw_r_scores[:, :-1]
     bce_raw = F.binary_cross_entropy_with_logits(shift_r_scores, labels, reduction="none")
     masked_bce = bce_raw * valid_mask.float()
     r_loss = masked_bce.sum() / valid_count
 
-    # 3. Sub-batch 2: Bidirectional Infill Pass
-    # Enables is_causal=False in SDPA (native fused FlashAttention-2 full attention)
+    # Bidirectional Infill Pass
     mask_blend = schedule_weights.get("mask_blend", 0.0)
     if routing_cache is not None:
         corrupted_infill, mask_positions = routing_cache.pop_or_generate_mask(
@@ -316,7 +317,7 @@ def corosred_unified_step_pytorch(
     infill_token_count_t = mask_positions.sum().float().clamp(min=1.0)
     mean_infill_ce = infill_loss_sum / infill_token_count_t
 
-    # 4. Schedule Weight Extraction and Safe Dynamic Rebalancing
+    # Schedule Weight Extraction
     alpha_nom = float(schedule_weights.get("alpha", 0.85))
     beta_nom = float(schedule_weights.get("beta", 0.15))
     gamma_nom = float(schedule_weights.get("gamma", 0.0))
@@ -327,18 +328,13 @@ def corosred_unified_step_pytorch(
         alpha, beta = alpha_nom, beta_nom
         rebal_telem = {"nominal_ratio": beta_nom / max(1e-6, alpha_nom), "clamped": False}
 
-    # 5. Sequence-Normalized Multi-Objective Loss Pooling
-    # Eliminates the ~7.6x per-token gradient amplification caused by dividing
-    # sparse masked tokens (~1,945) vs dense causal tokens (~14,819).
-    # Normalizing by sequence count and sequence length ensures that the per-token gradient
-    # ratio strictly adheres to beta / alpha without artificial token-count distortion.
-    w_c = float(B_c) / float(B)
-    w_m = float(B_m) / float(B)
-    causal_seq_norm = causal_loss_sum / max(1.0, float(B_c * (T - 1)))
-    infill_seq_norm = infill_loss_sum / max(1.0, float(B_m * (T - 1)))
-
-    task_weight_sum = max(1e-6, alpha * w_c + beta * w_m)
-    pooled_task_loss = (alpha * w_c * causal_seq_norm + beta * w_m * infill_seq_norm) / task_weight_sum
+    # 5. Per-Task Per-Token Mean Multi-Objective Loss Pooling
+    # Normalizes by active task token counts so alpha and beta directly define loss weights
+    causal_seq_norm = causal_loss_sum / max(1.0, causal_token_count)
+    infill_seq_norm = infill_loss_sum / infill_token_count_t
+
+    task_weight_sum = max(1e-6, alpha + beta)
+    pooled_task_loss = (alpha * causal_seq_norm + beta * infill_seq_norm) / task_weight_sum
     total_loss = pooled_task_loss + gamma_nom * r_loss
 
     # 6. Metric Tracker Update
```


### a2f09a0 → inspected working tree: telos/training/schedule.py

SHA-256 of embedded excerpt/source: `fdce21da68eb20afe3480459a886296906990c47685c33dc4c6e4fe9cbb0e58a`

```diff
diff --git a/telos/training/schedule.py b/telos/training/schedule.py
index c29c322..e4a7900 100644
--- a/telos/training/schedule.py
+++ b/telos/training/schedule.py
@@ -142,38 +142,28 @@ class COROSredSchedule:
         alpha = max(self.alpha_min, alpha)
         beta = max(self.beta_min, min(self.beta_max, beta))
 
-        # 2. Compute gamma(t) (Head warmup gate)
-        # Gated on empirical classification ranking discrimination (AUC or balanced accuracy)
-        # rather than raw accuracy. Under severe ~9:1 class imbalance, trivial all-negative
-        # predictions yield ~90% raw accuracy with zero ranking power. ROC-AUC and balanced accuracy
-        # maintain an uncorrupted 0.50 chance baseline, ensuring gamma opens only when real signal exists.
+        # Gate gamma on ranking metric (AUC or balanced accuracy) to handle class imbalance
         head_is_competent = self.head_ready_override
         if not head_is_competent and lrh_auc_ema is not None:
-            # Primary: Gate opens when running ROC-AUC / balanced accuracy crosses gamma_gate_auc (e.g. 0.55-0.60)
             head_is_competent = (lrh_auc_ema >= self.gamma_gate_auc)
         elif not head_is_competent and lrh_acc_ema is not None and self.acc_gate_threshold is not None:
-            # Backward compatibility fallback if only raw accuracy telemetry is available
             head_is_competent = (lrh_acc_ema >= self.acc_gate_threshold)
         elif not head_is_competent and lrh_auc_ema is None and lrh_acc_ema is None:
-            # Fallback when no telemetry is provided: Keep gamma strictly at 0.0 until empirical signal arrives
             head_is_competent = False
 
         if head_is_competent:
-            # Smooth linear ramp-in over 5% of training steps once competence threshold is met
+            # Ramp gamma over warmup fraction
             warmup_frac = 0.05
             progress_after_gate = max(0.0, min(1.0, (t - 0.10) / warmup_frac)) if (lrh_auc_ema is None and lrh_acc_ema is None) else 1.0
             gamma = self.gamma_max * progress_after_gate
         else:
             gamma = 0.0
 
-        # 3. Compute mask_blend ratio (Uniform random vs LRH confidence-routed)
-        # Driven by ROC-AUC ranking fidelity (only if confidence routing / gamma_max is enabled)
+        # Compute mask blend ratio based on ranking performance
         if lrh_auc_ema is not None and self.gamma_max > 0.0:
-            # Blend ratio = clamp((AUC - 0.50) / (0.75 - 0.50), 0.0, 1.0)
             auc_range = max(1e-6, self.auc_gate_target - self.auc_gate_min)
             mask_blend = max(0.0, min(1.0, (lrh_auc_ema - self.auc_gate_min) / auc_range))
         else:
-            # When telemetry is unestablished or LRH routing is disabled, remain on pure uniform random masking
             mask_blend = 0.0
 
         return {
@@ -195,8 +185,13 @@ class DynamicMetricTracker:
         ema_decay: float = 0.99,
         trust_region: float = 0.20,
         rebalance_temp: float = 0.25,
+        cadence: int = 1,
     ):
-        self.decay = ema_decay
+        self.raw_decay = ema_decay
+        self.cadence = max(1, int(cadence))
+        # Effective decay: decay_eff = 1.0 - cadence * (1.0 - ema_decay)
+        # Keeps effective time constant identical regardless of whether updates occur every step or every 25 steps
+        self.decay = max(0.0, 1.0 - self.cadence * (1.0 - self.raw_decay))
         self.trust_region = trust_region
         self.rebalance_temp = rebalance_temp
 
@@ -212,31 +207,37 @@ class DynamicMetricTracker:
 
         self.step_count = 0
 
-    def update_lrh(self, acc: float, auc: float):
+    def update_lrh(self, acc: float, auc: float, cadence: int | None = None):
         """Updates LRH classification accuracy and ROC-AUC running EMAs."""
         if math.isnan(acc) or math.isnan(auc):
             return
 
+        cadence_step = self.cadence if cadence is None else max(1, int(cadence))
+        decay_eff = max(0.0, 1.0 - cadence_step * (1.0 - self.raw_decay))
+
         if self.lrh_acc_ema is None:
             self.lrh_acc_ema = float(acc)
             self.lrh_auc_ema = float(auc)
         else:
-            self.lrh_acc_ema = self.decay * self.lrh_acc_ema + (1.0 - self.decay) * float(acc)
-            self.lrh_auc_ema = self.decay * self.lrh_auc_ema + (1.0 - self.decay) * float(auc)
+            self.lrh_acc_ema = decay_eff * self.lrh_acc_ema + (1.0 - decay_eff) * float(acc)
+            self.lrh_auc_ema = decay_eff * self.lrh_auc_ema + (1.0 - decay_eff) * float(auc)
 
-    def update_losses(self, loss_c: float, loss_m: float):
+    def update_losses(self, loss_c: float, loss_m: float, cadence: int | None = None):
         """Updates causal and masked reconstruction per-token loss EMAs."""
         if math.isnan(loss_c) or math.isnan(loss_m):
             return
 
+        cadence_step = self.cadence if cadence is None else max(1, int(cadence))
+        decay_eff = max(0.0, 1.0 - cadence_step * (1.0 - self.raw_decay))
+
         if self.initial_loss_c is None:
             self.initial_loss_c = max(1e-4, float(loss_c))
             self.initial_loss_m = max(1e-4, float(loss_m))
             self.loss_c_ema = self.initial_loss_c
             self.loss_m_ema = self.initial_loss_m
         else:
-            self.loss_c_ema = self.decay * self.loss_c_ema + (1.0 - self.decay) * float(loss_c)
-            self.loss_m_ema = self.decay * self.loss_m_ema + (1.0 - self.decay) * float(loss_m)
+            self.loss_c_ema = decay_eff * self.loss_c_ema + (1.0 - decay_eff) * float(loss_c)
+            self.loss_m_ema = decay_eff * self.loss_m_ema + (1.0 - decay_eff) * float(loss_m)
 
         self.step_count += 1
```


### a2f09a0 → inspected working tree: telos/training/trainer_pytorch.py

SHA-256 of embedded excerpt/source: `9321b587c099ec2487ee4c616c40b9f96e8e7878edd571104065fe01df22236e`

```diff
diff --git a/telos/training/trainer_pytorch.py b/telos/training/trainer_pytorch.py
index afa37c4..b9da03a 100644
--- a/telos/training/trainer_pytorch.py
+++ b/telos/training/trainer_pytorch.py
@@ -77,6 +77,10 @@ class UnifiedPyTorchTrainer:
         self.t_cfg = cfg.setdefault("training", {})
         self.c_cfg = cfg.setdefault("checkpoint", {})
 
+        self.seed = int(self.t_cfg.get("seed", cfg.get("seed", 42)))
+        from .core import set_global_seed
+        set_global_seed(self.seed)
+
         # Ensure paradigm and architectural metadata are mirrored into configuration
         if "paradigm" not in self.cfg:
             self.cfg["paradigm"] = self.paradigm
@@ -163,22 +167,23 @@ class UnifiedPyTorchTrainer:
             self.is_master = True
             self.is_ddp = False
 
+        if getattr(self, "is_tpu", False):
+            try:
+                import torch_xla.core.xla_model as xm
+                xm.set_rng_state(self.seed)
+            except Exception:
+                pass
+
         self.model.to(self.device)
         self.special_lut = self.special_lut.to(self.device)
 
-        # In mixed-precision training (bfloat16), model parameters in memory MUST remain in float32
-        # so AdamW gradient updates (order 10^-5 to 10^-6) do not mathematically underflow to zero
-        # against bfloat16's 7-bit mantissa (machine epsilon 2^-7 ~ 0.0078).
-        # On TPU, Matrix Multiply Units (MXUs) execute systolic GEMMs in bfloat16 via hardware autocast
-        # while optimizer states and parameter weights remain unadulterated in float32.
+        # Keep model parameters in float32 on TPU to prevent optimizer update underflow
         self.use_master_weights = False
         if self.is_tpu and self.precision in ["bfloat16", "bf16"]:
             self.model.to(dtype=torch.float32)
             print("  [Precision] TPU model parameters kept in float32 (AdamW updates preserved; forward pass autocast to bfloat16).")
 
-        # Crucial for PyTorch-XLA / Cloud TPU: Parameter tensor memory handles decouple across child
-        # modules during .to(device). Explicitly re-tie output_projection to tok_embeddings so PyTorch-XLA
-        # maintains a single unified weight tensor and backward updates accumulate jointly.
+        # Re-tie weights after moving to device for PyTorch-XLA compatibility
         if hasattr(self.model, "tie_weights"):
             self.model.tie_weights()
         elif hasattr(getattr(self.model, "module", None), "tie_weights"):
@@ -186,10 +191,7 @@ class UnifiedPyTorchTrainer:
 
         # Multi-GPU wrapping: Prefer DDP over deprecated DataParallel
         if getattr(self, "is_ddp", False):
-            # In legacy COROSred Phase B and C, reliability_head is frozen / only used for routing,
-            # so find_unused_parameters=True prevents DDP unused parameter reduction assertions.
-            # In Unified COROSred, both causal backbone and reliability head receive gradients,
-            # so find_unused_parameters=False eliminates per-step DDP parameter tree traversal overhead.
+            # Allow unused parameters in DDP only when reliability head is frozen
             find_unused = (
                 self.paradigm == "corosred"
                 and not getattr(self, "is_unified", False)
@@ -206,9 +208,7 @@ class UnifiedPyTorchTrainer:
             print("  [Hardware Notice] Multi-GPU detected without torchrun. For 1.8x-7x throughput scaling, launch with 'torchrun'. Falling back to DataParallel.")
             self.model = nn.DataParallel(self.model)
 
-        # Gradient checkpointing activation:
-        # Avoid auto-enabling on high-VRAM CUDA GPUs (A100/H100/4090) for small/medium models (<=100M),
-        # preserving ~33% compute FLOPs by eliminating backward activation recomputation.
+        # Select gradient checkpointing based on device VRAM and model dimension
         cuda_needs_chkpt = False
         if self.device.type == "cuda" and not self.is_tpu:
             cuda_gb = 0.0
@@ -216,14 +216,10 @@ class UnifiedPyTorchTrainer:
                 cuda_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
             except Exception:
                 pass
-            # Auto-enable only if VRAM <= 16GB or model width >= 1024
             if (cuda_gb > 0 and cuda_gb <= 16.0) or (self.m_cfg.get("d_model", 512) >= 1024):
                 cuda_needs_chkpt = True
 
-        # On TPU (v3/v4/v5e with 16GB-32GB HBM per core), models up to ~300M (d_model < 1536)
-        # easily fit in memory (<5GB HBM). Gradient checkpointing recomputation on PyTorch-XLA
-        # triggers subgraph rematerialization overhead and numerical instability during backward passes.
-        # Only auto-enable for very large architectures (d_model >= 1536) or huge microbatches (>128).
+        # Enable checkpointing on TPU only for large models (d_model >= 1536)
         tpu_needs_chkpt = self.is_tpu and (self.m_cfg.get("d_model", 512) >= 1536 or self.t_cfg.get("batch_size", 32) > 128)
         auto_chkpt = (
             tpu_needs_chkpt
@@ -271,9 +267,12 @@ class UnifiedPyTorchTrainer:
                     gamma_gate_auc=float(self.crsr_cfg.get("gamma_gate_auc", 0.55)),
                     acc_gate_threshold=float(self.crsr_cfg.get("acc_gate_threshold", 0.65)),
                 )
+                default_cadence = int(self.crsr_cfg.get("default_cadence", self.t_cfg.get("default_cadence", 25 if self.is_tpu else 1)))
+                cadence = int(self.crsr_cfg.get("cadence", self.t_cfg.get("cadence", default_cadence)) or default_cadence)
                 self.metric_tracker = DynamicMetricTracker(
                     trust_region=float(self.crsr_cfg.get("trust_region", 0.20)),
                     rebalance_temp=float(self.crsr_cfg.get("rebalance_temp", 0.25)),
+                    cadence=cadence,
                 )
                 self.routing_cache = RoutingMaskCache(
                     refresh_every_steps=int(self.crsr_cfg.get("routing_cache_steps", 50)),
@@ -342,8 +341,8 @@ class UnifiedPyTorchTrainer:
             **opt_kwargs
         )
 
-        # On TPU, quantize LR updates to 10-step cadence by default to avoid XLA graph recompilations from Python float changes, unless overridden
-        default_lr_cadence = 10 if self.is_tpu else 1
+        # Quantize LR updates to cadence (defaults to default_cadence: 25 on TPU, 1 on CUDA/CPU)
+        default_lr_cadence = int(self.t_cfg.get("default_cadence", 25 if self.is_tpu else 1))
         lr_cadence = int(self.t_cfg.get("lr_cadence", default_lr_cadence) or default_lr_cadence)
         self.scheduler = WarmupCosineLR(
             self.optimizer,
@@ -414,9 +413,7 @@ class UnifiedPyTorchTrainer:
             xm.mark_step()
             str_path = str(path)
 
-            # Direct host CPU detachment on master rank:
-            # Avoids xm.save collective synchronization hangs across PJRT replicas and eliminates
-            # pulling ~400MB of unused AdamW momentum buffers across the TPU device bus.
+            # Detach model parameters to host CPU on master rank without saving optimizer state
             raw_model = getattr(self.model, "module", self.model)
             cpu_model = {
                 k.removeprefix("module."): v.detach().cpu().clone()
@@ -522,7 +519,7 @@ class UnifiedPyTorchTrainer:
                 
                 # Fetch continuous schedule weights driven by step progress and empirical EMAs.
                 # Defaults to 25 on TPU (to avoid XLA recompilations) and 1 on CUDA/CPU unless manually specified.
-                default_sched_cadence = 25 if self.is_tpu else 1
+                default_sched_cadence = int(self.crsr_cfg.get("default_cadence", self.t_cfg.get("default_cadence", 25 if self.is_tpu else 1)))
                 sched_cadence = int(self.crsr_cfg.get("sched_cadence", self.t_cfg.get("sched_cadence", default_sched_cadence)) or default_sched_cadence)
                 sched_step = (self.global_step // sched_cadence) * sched_cadence if sched_cadence > 1 else self.global_step
                 lrh_acc_ema = self.metric_tracker.lrh_acc_ema if hasattr(self, "metric_tracker") else None
@@ -637,7 +634,7 @@ class UnifiedPyTorchTrainer:
             }
         }
         report_file = log_dir / f"benchmark_{self.paradigm}_pytorch_{int(time.time())}.json"
-        with open(report_file, "w") as f:
+        with open(report_file, "w", encoding="utf-8") as f:
             json.dump(bench_payload, f, indent=2)
         print(f"  Saved benchmark metrics to {report_file}\n")
 
@@ -662,7 +659,7 @@ class UnifiedPyTorchTrainer:
             for meta_cand in [Path(str(train_bin) + ".json"), train_bin.with_suffix(".json")]:
                 if meta_cand.exists():
                     try:
-                        with open(meta_cand, "r") as mf:
+                        with open(meta_cand, "r", encoding="utf-8") as mf:
                             m_info = json.load(mf)
                             dt_str = m_info.get("dtype", "uint16")
                             dtype = np.int32 if dt_str == "int32" else np.uint16
@@ -685,9 +682,7 @@ class UnifiedPyTorchTrainer:
                 "or pass '--synthetic' to train on synthetic random tokens."
             )
 
-        # Initialize DualMetricMonitor on held-out validation data if running unified COROSred (master rank only)
-        # On TPU multi-core, in-loop probe evaluation causes cross-replica graph desynchronization and barrier deadlocks,
-        # so probe evaluation is deferred to standalone post-checkpoint evaluation suites.
+        # Initialize DualMetricMonitor on validation split for unified COROSred on single-node GPU/CPU
         if getattr(self, "is_unified", False) and self.is_master and not self.is_tpu:
             val_path = d_cfg.get("val_path", d_cfg.get("val_dataset_path", None))
             if val_path and Path(val_path).exists():
@@ -700,7 +695,7 @@ class UnifiedPyTorchTrainer:
                         print(f"  [Dual Monitor] Initialized on validation split ({val_seqs} sequences).")
                 except Exception as e:
                     if self.is_master:
-                        print(f"  [Dual Monitor] Could not load val data: {e}")
+                        print(f"  Notice: Failed to initialize DualMetricMonitor ({e}).")
             if getattr(self, "dual_monitor", None) is None and len(dataset_matrix) > 200:
                 self.dual_monitor = DualMetricMonitor(dataset_matrix[-200:], self.vocab_size, self.seq_len, self.device)
                 if self.is_master:
@@ -748,9 +743,7 @@ class UnifiedPyTorchTrainer:
         if self.use_master_weights:
             self.model.zero_grad()
 
-        # Background asynchronous batch prefetcher (CUDA only)
-        # PyTorch-XLA does NOT support multi-threaded tensor allocation/transfer to device;
-        # transfers on background worker threads cause heavy XLA runtime mutex serialization.
+        # Background asynchronous batch prefetcher (CUDA only; disabled on XLA)
         if self.is_tpu:
             prefetch_queue = None
         else:
@@ -874,7 +867,7 @@ class UnifiedPyTorchTrainer:
 
             # Synchronized cadence all-reduce of LRH AUC / balanced accuracy across all 8 replicas.
             # Executed synchronously by ALL ranks outside `if self.is_master` to ensure 100% identical schedule EMAs.
-            default_cadence = 25 if self.is_tpu else 1
+            default_cadence = int(self.crsr_cfg.get("default_cadence", self.t_cfg.get("default_cadence", 25 if self.is_tpu else 1)))
             cadence = int(self.crsr_cfg.get("cadence", self.t_cfg.get("cadence", default_cadence)) or default_cadence)
             if getattr(self, "is_unified", False) and (step % cadence == 0 or step == 1) and last_metrics is not None:
                 raw_auc = last_metrics.get("lrh_bal_acc", last_metrics.get("lrh_auc", None))
@@ -912,7 +905,7 @@ class UnifiedPyTorchTrainer:
 
                     # Update local metric tracker on ALL replicas with identical cluster-wide reduced scalar
                     if getattr(self, "metric_tracker", None) is not None:
-                        self.metric_tracker.update_lrh(acc=acc_val, auc=auc_val)
+                        self.metric_tracker.update_lrh(acc=acc_val, auc=auc_val, cadence=cadence)
 
             if self.is_tpu and step % 50 == 0:
                 import gc
@@ -1041,7 +1034,7 @@ class UnifiedPyTorchTrainer:
         if self.is_master:
             self.save_checkpoint(ckpt_dir / "checkpoint_final.pt", sync=True)
             # Write standalone config.json for eval loader and downstream tools
-            with open(ckpt_dir / "config.json", "w") as f:
+            with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
                 json.dump(self.cfg, f, indent=2, default=str)
             print("=" * 70)
             print(f"  {self.paradigm.upper()} PyTorch Training Complete! Total time: {total_time/60.0:.2f} minutes.")
```


### a2f09a0 → inspected working tree: telos/train/cli.py

SHA-256 of embedded excerpt/source: `a57d48ad1a3847c9b67aba1e78566f4a9e8d33d063f0fcdcbe13135e8eb412bb`

```diff
diff --git a/telos/train/cli.py b/telos/train/cli.py
index f971137..fe812ec 100644
--- a/telos/train/cli.py
+++ b/telos/train/cli.py
@@ -27,9 +27,7 @@ def _mp_train_worker(index, kwargs):
     os.environ["OMP_NUM_THREADS"] = "1"
     os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
     import torch
-    # Crucial for multi-core TPU VM: Restrict intra-op thread count to 1 per worker
-    # to stop 8 spawned processes from generating competing OpenMP spin-lock threads
-    # that pin CPU utilization at 100% per core.
+    # Set single intra-op thread per worker to prevent CPU thread contention on TPU
     torch.set_num_threads(1)
     if hasattr(torch, "set_num_interop_threads"):
         try:
@@ -49,6 +47,11 @@ def _mp_train_worker(index, kwargs):
     xla_utils._CACHED_XLA_DEVICE = None
     xla_utils._CACHED_XLA_WORLD_SIZE = None
     xla_utils._CACHED_IS_MASTER = None
+
+    from telos.training import set_global_seed
+    worker_seed = int(kwargs.get("seed", 42))
+    set_global_seed(worker_seed)
+
     train(**kwargs)
 
 
@@ -161,12 +164,15 @@ def train(
     self_cond_prob: float = 0.5,
     init_checkpoint: str | Path | None = None,
     compile: bool | None = None,
+    seed: int = 42,
+    default_cadence: int | None = None,
     **kwargs
 ):
     """
     Programmatic entrypoint to train any Télos model.
     """
     cfg = build_config(
+        default_cadence=default_cadence,
         paradigm=paradigm,
         phase=phase,
         params=params,
@@ -194,6 +200,7 @@ def train(
         self_condition=self_condition,
         self_cond_prob=self_cond_prob,
         compile=compile,
+        seed=seed,
         **kwargs
     )
 
@@ -258,6 +265,8 @@ def train(
                     self_cond_prob=self_cond_prob,
                     init_checkpoint=init_checkpoint,
                     compile=compile,
+                    seed=t_cfg.get("seed", seed),
+                    default_cadence=default_cadence,
                     _is_spawned=True,
                     **kwargs,
                 )
@@ -267,6 +276,10 @@ def train(
         except Exception as e:
             print(f"  [Notice] Multi-core xmp.spawn skipped ({e}). Proceeding on single core.")
 
+    from telos.training import set_global_seed
+    resolved_seed = int(t_cfg.get("seed", kwargs.get("seed", seed)))
+    set_global_seed(resolved_seed)
+
     if backend == "mlx":
         # Defer MLX imports so non-Apple-Silicon environments don't resolve MLX
         from telos.models import MLXTelosTransformer
@@ -344,9 +357,7 @@ def train(
 
         trainer = UnifiedPyTorchTrainer(paradigm=paradigm, model=model, cfg=cfg, device_type=device)
         if ckpt_state is not None and resume_step > 0:
-            # On TPU / XLA, restoring CPU-serialized optimizer state dictionaries strands tensors on CPU
-            # while model parameters reside on TPU device, forcing catastrophic 10s/step CPU-TPU synchronization fallbacks.
-            # Only restore optimizer state dict on CUDA/CPU where tensor devices are unified.
+            # Restore optimizer state only on CUDA/CPU to avoid host-device tensor desync on XLA
             if device != "xla" and not getattr(trainer, "is_tpu", False) and "optimizer_state_dict" in ckpt_state:
                 try:
                     trainer.optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
@@ -431,7 +442,10 @@ def main():
     parser.add_argument("--self-condition", action=argparse.BooleanOptionalAction, default=True, help="Enable self-conditioned draft training in COROSred Phase B")
     parser.add_argument("--self-cond-prob", type=float, default=0.5, help="Probability of training on model drafts vs clean masks in Phase B")
     parser.add_argument("--init-checkpoint", type=str, default=None, help="Path to initial checkpoint to load weights from before training")
+    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None, help="Enable torch.compile for model execution")
+    parser.add_argument("--seed", type=int, default=42, help="Global random seed for initialization and training reproducibility (default: 42)")
     # Cadence & Dynamics Overrides
+    parser.add_argument("--default-cadence", "--default_cadence", dest="default_cadence", type=int, default=None, help="Default cadence across all cadence operations (default: 25 on TPU, 1 on CUDA/CPU)")
     parser.add_argument("--sched-step", "--sched-cadence", dest="sched_step", type=int, default=None, help="Cadence for schedule updates (default: 25 on TPU, 1 on CUDA/CPU)")
     parser.add_argument("--lr-cadence", type=int, default=None, help="Cadence for learning rate updates (default: 10 on TPU, 1 on CUDA/CPU)")
     parser.add_argument("--cadence", type=int, default=None, help="Cadence for metric reduction and synchronization (default: 25 on TPU, 1 on CUDA/CPU)")
@@ -440,6 +454,7 @@ def main():
 
     try:
         train(
+            default_cadence=args.default_cadence,
             sched_step=args.sched_step,
             lr_cadence=args.lr_cadence,
             cadence=args.cadence,
@@ -474,7 +489,7 @@ def main():
             self_condition=args.self_condition,
             self_cond_prob=args.self_cond_prob,
             init_checkpoint=args.init_checkpoint,
-            compile=args.compile,
+            compile=getattr(args, "compile", None),
             legacy_phases=args.legacy_phases,
             alpha_max=args.alpha_max,
             alpha_min=args.alpha_min,
@@ -488,6 +503,7 @@ def main():
             mask_prob=args.mask_prob,
             routing_cache_steps=args.routing_cache_steps,
             adaptive_rebalance=args.adaptive_rebalance,
+            seed=args.seed,
         )
     except KeyboardInterrupt:
         print("\nTraining interrupted by user.")
```


# Appendix E — Diagnostic implementations for independent scrutiny

These sources produced the measurements above. Inspect them for confounders or errors. They are CPU diagnostics unless explicitly told otherwise; merely having a CUDA/XLA option is not evidence that option ran. The existing fixed-fixture backend probe is single-replica, uses a fresh optimizer, and explicitly disables fused AdamW. It does not validate distributed training or original CUDA/XLA kernels. Its initial FP32-first workflow predates the user constraints; do not treat that workflow as a requirement for a new scarce TPU allocation.


### scripts/compare_saved_50m_gradients.py

SHA-256 of embedded excerpt/source: `56ee4909e26f099d1e58edbf32b1ad15aa529df7ead37755d407e7d8999859a1`

```python
"""CPU endpoint comparison of saved 50M COROSred checkpoints, with explicit masks.

This describes learned checkpoints under one declared objective. It does not
compare accelerator kernels, reconstruct Adam state, or identify a training cause.
"""
import argparse
from dataclasses import fields
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from telos.models import TelosConfig, TelosTransformer

DEFAULT_CHECKPOINTS = {
    'cuda_lightning': 'checkpoints/corosred/50m_lightning/checkpoint_final.pt',
    'tpu_baseline': 'checkpoints/corosred/unified/50m_python/checkpoint_final.pt',
    'tpu_schedstep1': 'checkpoints/corosred/unified/50m_tpu_schedstep1/checkpoint_final.pt',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def block_name(name):
    parts = name.split('.')
    return '.'.join(parts[:2]) if parts[0] == 'layers' else parts[0]


def gradient_geometry(causal, infill, alpha, beta):
    """Compute task geometry without assuming equal gradient norms or directions."""
    if causal.keys() != infill.keys():
        raise ValueError('Causal and infill have different active parameter sets')
    accum = {}
    for name in causal:
        c, i = causal[name].double(), infill[name].double()
        values = (float((c * c).sum()), float((i * i).sum()), float((c * i).sum()))
        for group in ('all_backbone', block_name(name)):
            totals = accum.setdefault(group, [0.0, 0.0, 0.0])
            for index, value in enumerate(values):
                totals[index] += value
    a, b = alpha / (alpha + beta), beta / (alpha + beta)
    result = {}
    for name, (cc, ii, dot) in accum.items():
        nc, ni = cc ** .5, ii ** .5
        result[name] = {
            'causal_gradient_norm': nc, 'infill_gradient_norm': ni,
            'causal_infill_cosine': max(-1., min(1., dot / (nc * ni))) if nc and ni else None,
            'weighted_causal_norm': a * nc, 'weighted_infill_norm': b * ni,
            'weighted_causal_to_infill_ratio': a * nc / (b * ni) if b * ni else None,
            'combined_gradient_norm': max(0., a*a*cc + b*b*ii + 2*a*b*dot) ** .5,
            # -grad(total) has this first-order effect on the individual loss.
            'causal_loss_directional_derivative': -(a * cc + b * dot),
            'infill_loss_directional_derivative': -(a * dot + b * ii),
        }
    return result


def task_pass(model, batches, masks, *, infill):
    model.zero_grad(set_to_none=True)
    denominator = sum(int(mask.sum()) for mask in masks) if infill else sum(
        batch.shape[0] * (batch.shape[1] - 1) for batch in batches)
    total_loss, total_correct = 0.0, 0
    batch_metrics = []
    for index, (batch, mask) in enumerate(zip(batches, masks)):
        if infill:
            corrupted = torch.where(mask, torch.ones_like(batch), batch)
            logits = model(corrupted, mask_override=False, return_reliability=False)
            flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   batch.reshape(-1), reduction='none').view_as(batch)
            loss_sum = (flat * mask).sum()
            correct = int(((logits.detach().argmax(-1) == batch) & mask).sum())
            count = int(mask.sum())
        else:
            logits = model(batch, mask_override=True, return_reliability=False)
            shifted = logits[:, :-1, :]
            target = batch[:, 1:]
            loss_sum = F.cross_entropy(shifted.reshape(-1, shifted.shape[-1]),
                                       target.reshape(-1), reduction='sum')
            correct = int((shifted.detach().argmax(-1) == target).sum())
            count = target.numel()
        if not torch.isfinite(loss_sum):
            raise RuntimeError('Nonfinite loss')
        (loss_sum / denominator).backward()
        value = float(loss_sum.detach())
        total_loss += value
        total_correct += correct
        batch_metrics.append({'batch': index, 'tokens': count, 'ce': value / count,
                              'top1_pct': 100 * correct / count})
        del logits, loss_sum
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                 if p.grad is not None}
    if not all(torch.isfinite(g).all() for g in gradients.values()):
        raise RuntimeError('Nonfinite gradient')
    return {'ce': total_loss / denominator, 'top1_pct': 100 * total_correct / denominator,
            'tokens': denominator, 'batches': batch_metrics}, gradients


def compare_checkpoint(path, batches, masks, alpha, beta):
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    embedded = checkpoint['config']
    keys = {f.name for f in fields(TelosConfig)}
    config = {k: v for k, v in embedded['model'].items() if k in keys}
    config.update(dropout=0.0, use_grad_checkpoint=False)
    state = {k.removeprefix('module.').removeprefix('_orig_mod.'): v
             for k, v in checkpoint['model_state_dict'].items()}
    if config.get('tied_embeddings', True) and not torch.equal(
            state['tok_embeddings.weight'], state['output_projection.weight']):
        raise ValueError('Saved tied weights disagree; refusing to silently overwrite')
    if max(int(batch.max()) for batch in batches) >= config['vocab_size']:
        raise ValueError('Evaluation token ID outside checkpoint vocabulary')
    model = TelosTransformer(TelosConfig(**config)).float().eval()
    model.load_state_dict(state, strict=True)
    model.tie_weights()
    metadata = {'checkpoint': str(path), 'checkpoint_sha256': sha256(path),
                'step': checkpoint.get('global_step'), 'config': embedded,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'saved_tensor_dtypes': sorted({str(p.dtype) for p in state.values()})}
    del checkpoint, state
    causal_metrics, causal_gradients = task_pass(model, batches, masks, infill=False)
    infill_metrics, infill_gradients = task_pass(model, batches, masks, infill=True)
    geometry = gradient_geometry(causal_gradients, infill_gradients, alpha, beta)
    result = {'metadata': metadata, 'causal': causal_metrics, 'infill': infill_metrics,
              'gradient_geometry': geometry}
    del model, causal_gradients, infill_gradients
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/python_corpus_15b.bin')
    parser.add_argument('--offset-tokens', type=int, default=6_000_000_000)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=.50)
    parser.add_argument('--beta', type=float, default=.35)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.batches, args.batch_size, args.threads) < 1 or args.seq_len < 2:
        raise ValueError('Positive batch/thread sizes and sequence length >= 2 required')
    if args.alpha < 0 or args.beta < 0 or args.alpha + args.beta <= 0:
        raise ValueError('Nonnegative weights with a positive sum required')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    total = args.batches * args.batch_size * args.seq_len
    if args.offset_tokens < 0 or (args.offset_tokens + total) * 2 > args.data.stat().st_size:
        raise ValueError('Selected uint16 token slice is outside the data file')
    tokens = np.memmap(args.data, mode='r', dtype=np.uint16,
                       offset=args.offset_tokens * 2, shape=(total,))
    batch = torch.from_numpy(np.array(tokens, dtype=np.int64)).reshape(
        args.batches, args.batch_size, args.seq_len)
    generator = torch.Generator().manual_seed(args.seed)
    mask = torch.rand(batch.shape, generator=generator) < .20
    mask[:, :, 0] = False
    if not mask.flatten(1).any(1).all():
        raise ValueError('An evaluation batch contains no masked tokens')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture_path = args.output.with_suffix('.fixture.pt')
    torch.save({'tokens': batch, 'infill_mask': mask}, fixture_path)
    report = {'purpose': 'Saved endpoint diagnostic, not runtime backend parity or training causality',
              'device': 'cpu', 'compute_dtype': 'float32', 'torch': str(torch.__version__),
              'seed': args.seed, 'threads': args.threads, 'sequence_length': args.seq_len,
              'data_path': str(args.data), 'offset_tokens': args.offset_tokens,
              'data_dtype': 'uint16 (explicit assumption matching prior audit)',
              'evaluation_sequences': args.batches * args.batch_size,
              'evaluation_tokens': total,
              'input_sha256': hashlib.sha256(batch.numpy().tobytes()).hexdigest(),
              'mask_sha256': hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
              'fixture': str(fixture_path), 'alpha': args.alpha, 'beta': args.beta,
              'reliability_loss': 'excluded; compare causal and infill backbone gradients only',
              'gradient_aggregation': 'token-weighted mean over the complete fixture per task',
              'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                                [Path(__file__), ROOT / 'telos/models/transformer.py',
                                 ROOT / 'telos/models/components.py']},
              'models': {}}
    try:
        report['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except subprocess.SubprocessError:
        report['git_commit'] = None
    expected_architecture = None
    for label, relative in DEFAULT_CHECKPOINTS.items():
        started = time.perf_counter()
        print(f'[Starting] {label}', flush=True)
        result = compare_checkpoint(ROOT / relative, batch, mask, args.alpha, args.beta)
        architecture = {key: result['metadata']['config']['model'].get(key)
                        for key in ('vocab_size', 'd_model', 'n_layers', 'n_heads', 'n_kv_heads')}
        if expected_architecture is not None and architecture != expected_architecture:
            raise ValueError('Checkpoint architectures differ')
        expected_architecture = architecture
        result['seconds'] = time.perf_counter() - started
        report['models'][label] = result
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(args.output)
        geometry = result['gradient_geometry']['all_backbone']
        print(f'[Done] {label}: causal CE={result["causal"]["ce"]:.6f}, '
              f'infill CE={result["infill"]["ce"]:.6f}, '
              f'gradient cosine={geometry["causal_infill_cosine"]:.6f}, '
              f'{result["seconds"]:.1f}s', flush=True)
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
```


### scripts/audit_cadence_paths.py

SHA-256 of embedded excerpt/source: `e3cdb05e32c950aaa0aff6ab48610eec8f95bf5178f93d783d11807bccb36a32`

```python
"""CPU schedule replay and gradient-reachability audit; not a training-cause proof."""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from telos.training.schedule import COROSredSchedule,DynamicMetricTracker
from telos.training.lr_schedule import WarmupCosineLR
from telos.diffusion.corosred_unified import RoutingMaskCache,corosred_unified_step_pytorch
from telos.models import TelosConfig,TelosTransformer

HISTORICAL='a2f09a0'

def historical(path):
    source=subprocess.check_output(['git','show',f'{HISTORICAL}:{path}'],cwd=ROOT,text=True)
    ns={'__name__':'cadence_historical_probe'}
    exec(compile(source,f'{HISTORICAL}:{path}','exec'),ns)
    return ns,hashlib.sha256(source.encode()).hexdigest()


def schedule_replay(schedule_class,tracker_class,*,cadence,corrected,steps=12716):
    schedule=schedule_class(max_steps=steps,alpha_min=.5,beta_max=.35)
    tracker=tracker_class(cadence=cadence) if corrected else tracker_class()
    trace=[]
    for step in range(1,steps+1):
        # Match the trainer: weights use the previous step's metric state.
        weights=schedule.get_weights(step-1,tracker.lrh_acc_ema,tracker.lrh_auc_ema)
        trace.append({**weights,'auc_ema':tracker.lrh_auc_ema})
        # Synthetic step response, deliberately crossing the gamma threshold.
        observation=.50 if step<1000 else .60
        if step==1 or step%cadence==0:
            if corrected: tracker.update_lrh(observation,observation,cadence=cadence)
            else: tracker.update_lrh(observation,observation)
    active=[i+1 for i,x in enumerate(trace) if x['gamma']>0]
    return trace,{'first_gamma_step':active[0] if active else None,'gamma_active_steps':len(active),'final_auc_ema':tracker.lrh_auc_ema}


def gradient_probe(step_fn=corosred_unified_step_pytorch,cache_class=RoutingMaskCache):
    torch.manual_seed(42)
    template=TelosTransformer(TelosConfig(vocab_size=32,d_model=16,n_layers=1,n_heads=2,n_kv_heads=2,max_seq_len=16,use_reliability_head=True,dropout=0.)).eval()
    tokens=torch.randint(4,32,(6,16))
    outcomes=[]
    for gamma,blend in [(0.,0.),(.1,.8)]:
        model=copy.deepcopy(template)
        cache=cache_class()
        torch.manual_seed(123)
        loss,metrics=step_fn(model,tokens,32,{'alpha':.85,'beta':.15,'gamma':gamma,'mask_blend':blend},routing_cache=cache,causal_ratio=.65,k_amb=5)
        loss.backward()
        raw={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        clipped={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
        for k,p in model.named_parameters():
            if k in raw: p.grad=raw[k].clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(),.25)
        forced_clipped={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
        outcomes.append((raw,clipped,norm,forced_clipped))
    a,b=outcomes
    backbone=[k for k in a[0] if not k.startswith('reliability_head.')]
    head=[k for k in a[0] if k.startswith('reliability_head.')]
    return {'max_backbone_gradient_difference_before_clip':max(float((a[0][k]-b[0][k]).abs().max()) for k in backbone),
            'max_backbone_gradient_difference_after_clip':max(float((a[1][k]-b[1][k]).abs().max()) for k in backbone),
            'max_backbone_gradient_difference_with_forced_active_clip_point25':max(float((a[3][k]-b[3][k]).abs().max()) for k in backbone),
            'global_norm_gamma0':a[2],'global_norm_gamma_point1':b[2],
            'head_gradient_norm_gamma0':math.sqrt(sum(float(a[0][k].double().square().sum()) for k in head)),
            'head_gradient_norm_gamma_point1':math.sqrt(sum(float(b[0][k].double().square().sum()) for k in head))}


def main():
    torch.set_num_threads(1)
    old,oldhash=historical('telos/training/schedule.py')
    loss_old,losshash=historical('telos/diffusion/corosred_unified.py')
    report={'purpose':'Synthetic CPU replay and source reachability; does not identify cause of saved CUDA/TPU deficit',
            'historical_revision':subprocess.check_output(['git','rev-parse',HISTORICAL],text=True).strip(),
            'historical_source_sha256':{'schedule.py':oldhash,'corosred_unified.py':losshash},
            'current_source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['telos/training/schedule.py','telos/training/lr_schedule.py','telos/training/trainer_pytorch.py','telos/diffusion/corosred_unified.py','telos/models/transformer.py']},
            'metric_input':'Synthetic identical per-step .50 until step 999, .60 thereafter; not recovered training observations',
            'metric_replays':{},'lr_quantization':{}}
    traces={}
    for name,cls,tracker,cadence,corrected in [('historical_every1',old['COROSredSchedule'],old['DynamicMetricTracker'],1,False),('historical_every25',old['COROSredSchedule'],old['DynamicMetricTracker'],25,False),('current_every1',COROSredSchedule,DynamicMetricTracker,1,True),('current_every25',COROSredSchedule,DynamicMetricTracker,25,True)]:
        traces[name],report['metric_replays'][name]=schedule_replay(cls,tracker,cadence=cadence,corrected=corrected)
    reference=traces['current_every1']
    for name,trace in traces.items():
        report['metric_replays'][name]['max_difference_vs_current_every1']={key:max(abs(x[key]-y[key]) for x,y in zip(trace,reference)) for key in ['alpha','beta','gamma','mask_blend']}
    schedule=COROSredSchedule(max_steps=12716,alpha_min=.5,beta_max=.35)
    report['objective_step_quantization']={}
    for cadence in [10,25]:
        maxima={key:0. for key in ['alpha','beta']}
        for step in range(12716):
            fine=schedule.get_weights(step)
            coarse=schedule.get_weights((step//cadence)*cadence)
            for key in maxima: maxima[key]=max(maxima[key],abs(fine[key]-coarse[key]))
        report['objective_step_quantization'][str(cadence)]=maxima
    lrs={}
    for cadence in [1,10,25]:
        param=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([param],lr=.0004242640687119285)
        sched=WarmupCosineLR(opt,254,12716,min_lr=4.2426406871192855e-05,update_cadence=cadence)
        values=[]
        for step in range(12716):
            values.append(opt.param_groups[0]['lr']);opt.step();sched.step()
        lrs[cadence]=values
        report['lr_quantization'][str(cadence)]={'integrated_lr':sum(values),'max_absolute_difference_vs_1':max(abs(a-b) for a,b in zip(values,lrs[1])),'integrated_relative_difference_vs_1':sum(values)/sum(lrs[1])-1}
    report['gradient_paths_current']=gradient_probe()
    report['gradient_paths_historical_loss']=gradient_probe(loss_old['corosred_unified_step_pytorch'],loss_old['RoutingMaskCache'])
    sched=COROSredSchedule(max_steps=12716,alpha_min=.5,beta_max=.35)
    report['routing_refresh_gate']={'without_metrics':sched.get_weights(4000)['mask_blend'],'with_auc_point8':sched.get_weights(4000,.8,.8)['mask_blend']}
    path=ROOT/'logs/cadence_path_audit_20261018.json'
    path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print(path)
if __name__=='__main__':main()
```


### scripts/replay_saved_optimizer.py

SHA-256 of embedded excerpt/source: `e64e533c3fa3ec79eb46e695bf1d6af8328f1d987fe74d0e74ef0c0fe4d82b8c`

```python
"""Compare CPU AdamW implementations with identical saved 50M moments/gradients.

This tests CPU arithmetic only, not CUDA/XLA kernels or historical TPU updates.
"""
import copy
from dataclasses import fields
import gc
import hashlib
import json
import importlib
import math
from pathlib import Path
import sys
import torch
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from telos.models import TelosConfig,TelosTransformer
from scripts.compare_saved_50m_gradients import sha256


def build_optimizer(checkpoint,names,fused):
    state=checkpoint['model_state_dict']
    parameters={name:torch.nn.Parameter(state[name].clone()) for group in names for name in group}
    saved=copy.deepcopy(checkpoint['optimizer_state_dict'])
    groups=[]
    for keys,group in zip(names,saved['param_groups']):
        groups.append({'params':[parameters[name] for name in keys]})
        for name,index in zip(keys,group['params']):
            if tuple(saved['state'][index]['exp_avg'].shape)!=tuple(parameters[name].shape):
                raise ValueError('Saved optimizer mapping disagrees with parameter shape')
    optimizer=torch.optim.AdamW(groups,lr=1e-4,fused=fused,foreach=False)
    optimizer.load_state_dict(saved)
    for group in optimizer.param_groups:
        group.update(fused=fused,foreach=False,capturable=False,differentiable=False)
    return parameters,optimizer


@torch.no_grad()
def compare_tensors(a,b,initial=None):
    maximum=0.; squared=0.; reference=0.;updates=0.;changed=0;count=0
    for name,x in a.items():
        y=b[name];delta=x.double()-y.double()
        maximum=max(maximum,float(delta.abs().max()));squared+=float(delta.square().sum())
        reference+=float(x.double().square().sum());changed+=int((x!=y).sum());count+=x.numel()
        if initial is not None:updates+=float((x.double()-initial[name].double()).square().sum())
    return {'max_absolute':maximum,'relative_l2_vs_tensor':math.sqrt(squared/max(reference,1e-300)),
            'relative_l2_vs_plain_cumulative_update':math.sqrt(squared/max(updates,1e-300)) if initial is not None else None,
            'unequal_elements':changed,'elements':count}


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    checkpoint_path=ROOT/'checkpoints/corosred/50m_lightning/checkpoint_final.pt'
    fixture_path=ROOT/'scratch/backend_parity/fixture.pt'
    capture_path=ROOT/'scratch/backend_parity/cpu_fp32.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True,mmap=True)
    fixture=torch.load(fixture_path,map_location='cpu',weights_only=True,mmap=True)
    capture=torch.load(capture_path,map_location='cpu',weights_only=True,mmap=True)
    if fixture['checkpoint_sha256']!=sha256(checkpoint_path) or capture['metadata']['fixture_sha256']!=sha256(fixture_path):
        raise ValueError('Capture/checkpoint provenance mismatch')
    keys={f.name for f in fields(TelosConfig)}
    model=TelosTransformer(TelosConfig(**{k:v for k,v in checkpoint['config']['model'].items() if k in keys}))
    names=[[],[]]
    for name,p in model.named_parameters():
        names[1 if p.ndim==1 or 'embed' in name else 0].append(name)
    del model;gc.collect()
    if [len(g) for g in names]!=[len(g['params']) for g in checkpoint['optimizer_state_dict']['param_groups']]:
        raise ValueError('Optimizer parameter group sizes differ')
    gradients={name:capture['tensors']['gradient/'+name] for group in names for name in group}
    ordinary,opt_a=build_optimizer(checkpoint,names,False)
    fused,opt_b=build_optimizer(checkpoint,names,True)
    for name in ordinary:
        if ordinary[name].data_ptr()==fused[name].data_ptr():
            raise ValueError('Replay parameters alias each other')
        for key in ['exp_avg','exp_avg_sq','step']:
            if opt_a.state[ordinary[name]][key].data_ptr()==opt_b.state[fused[name]][key].data_ptr():
                raise ValueError('Replay optimizer states alias each other')
    initial=checkpoint['model_state_dict']
    report={'device':'cpu','torch':str(torch.__version__),'physical_cuda_xla_comparison':False,
            'checkpoint_sha256':sha256(checkpoint_path),'capture_sha256':sha256(capture_path),
            'capture_metadata':capture['metadata'],'source_sha256':sha256(__file__),
            'optimizer_state':'Saved CUDA final moments at step 12716, cloned identically for both CPU modes',
            'mapping_caveat':'Parameter group sizes and every moment shape verified against current named-parameter order; checkpoint has no saved parameter-name manifest',
            'gradient_stream':'Same captured FP32 gradients, repeated with prescribed sign/scale changes; not a continuation of training',
            'steps':[]}
    module=importlib.import_module(torch.optim.AdamW.step.__module__)
    report['dispatch_counts']={'ordinary':0,'fused':0}
    for index,scale in enumerate([1.,.5,1.5,-.25,1.,.75,-.5,.01],1):
        norms=[]
        for parameters,optimizer in [(ordinary,opt_a),(fused,opt_b)]:
            for name,p in parameters.items():p.grad=gradients[name].clone().mul_(scale)
            norm=torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.,foreach=False)
            norms.append(float(norm))
            kind='fused' if optimizer is opt_b else 'ordinary'
            prefix='_fused_' if kind=='fused' else '_single_tensor_'
            function=next(name for name in [prefix+'adamw',prefix+'adam'] if hasattr(module,name))
            with patch.object(module,function,wraps=getattr(module,function)) as call:
                optimizer.step()
                report['dispatch_counts'][kind]+=call.call_count
            optimizer.zero_grad(set_to_none=True)
        if norms[0]!=norms[1]:raise ValueError('Clipping inputs differed')
        row={'step':index,'prescribed_gradient_scale':scale,'preclip_norm':norms[0],
             'clip_coefficient':min(1.,1./(norms[0]+1e-6)),
             'parameters':compare_tensors(ordinary,fused,initial),
             'exp_avg':compare_tensors({n:opt_a.state[p]['exp_avg'] for n,p in ordinary.items()},{n:opt_b.state[p]['exp_avg'] for n,p in fused.items()}),
             'exp_avg_sq':compare_tensors({n:opt_a.state[p]['exp_avg_sq'] for n,p in ordinary.items()},{n:opt_b.state[p]['exp_avg_sq'] for n,p in fused.items()})}
        report['steps'].append(row)
        print(json.dumps(row),flush=True)
    path=ROOT/'logs/saved_optimizer_replay_20261018.json';path.write_text(json.dumps(report,indent=2));print(path)
if __name__=='__main__':main()
```


### scripts/diagnose_backend_parity.py

SHA-256 of embedded excerpt/source: `50225d3e219e69d62b5900f7891e9dab5b4d0487f1f36a5487029a39cb7749b1`

```python
"""Fixed-input COROSred backend probe; not a distributed trainer or exact resume.

prepare freezes one checkpoint, token batch, and infill mask. capture compares
the production objective with a fresh, identical AdamW state on one device.
Use FP32 first, then BF16. A successful single-device probe does not validate
distributed reduction, accumulation, data streaming, or long-run schedules.
"""
import argparse
from contextlib import nullcontext
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from telos.models import TelosConfig, TelosTransformer
from telos.diffusion.corosred_unified import corosred_unified_step_pytorch


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    import numpy as np
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    width = config['model'].get('seq_len') or config['model'].get('max_seq_len', 512)
    if args.batch_size < 2 or not 2 <= args.seq_len <= width or args.row < 0:
        raise ValueError('Need batch >= 2, 2 <= seq_len <= model context, and row >= 0')
    data = np.memmap(args.data, mode='r', dtype=np.uint16)
    end = (args.row + args.batch_size) * width
    if end > len(data):
        raise ValueError('Selected token rows exceed the data file')
    batch = torch.from_numpy(np.array(
        data[args.row * width:end].reshape(args.batch_size, width)[:, :args.seq_len],
        dtype=np.int64,
    ))
    ratio = config.get('corosred', {}).get('causal_ratio', 0.65)
    causal_rows = max(1, min(args.batch_size - 1, round(args.batch_size * ratio)))
    generator = torch.Generator().manual_seed(42)
    mask = torch.rand((args.batch_size - causal_rows, args.seq_len), generator=generator) < 0.2
    mask[:, 0] = False
    mask[:, 1] = True
    state = {k.removeprefix('module.').removeprefix('_orig_mod.'): v
             for k, v in checkpoint['model_state_dict'].items()}
    fixture = {
        'config': config, 'state': state, 'batch': batch, 'mask': mask,
        'causal_ratio': ratio,
        # Diagnostic weights exercise all components, independent of EMA gates.
        'weights': {'alpha': 0.7, 'beta': 0.3, 'gamma': 0.1, 'mask_blend': 0.0},
        'checkpoint_sha256': digest(args.checkpoint),
        'checkpoint_step': checkpoint.get('global_step'),
        'data_path': str(args.data), 'data_row': args.row,
        'optimizer': {'lr': 1e-4, 'betas': (0.9, 0.95), 'eps': 1e-8, 'weight_decay': 0.1},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, args.output)
    print(json.dumps({'fixture': str(args.output), 'sha256': digest(args.output),
                      'batch_shape': list(batch.shape), 'masked_tokens': int(mask.sum())}, indent=2))


def capture(args):
    fixture = torch.load(args.fixture, map_location='cpu', weights_only=True)
    if args.device == 'xla':
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        from telos.training.xla_utils import get_xla_device
        device = get_xla_device()
        if xr.is_spmd() or xr.world_size() != 1:
            raise RuntimeError('First probe requires one XLA replica with SPMD disabled')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    keys = {f.name for f in fields(TelosConfig)}
    config = {k: v for k, v in fixture['config']['model'].items() if k in keys}
    config.update(dropout=0.0, use_grad_checkpoint=False)
    model = TelosTransformer(TelosConfig(**config))
    model.load_state_dict(fixture['state'], strict=True)
    model.to(device=device, dtype=torch.float32)
    model.tie_weights()
    model.train()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim == 1 or 'embed' in name else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': 0.1}, {'params': no_decay, 'weight_decay': 0.0}],
        lr=fixture['optimizer']['lr'], betas=fixture['optimizer']['betas'],
        eps=fixture['optimizer']['eps'], foreach=False, fused=False,
    )
    batch = fixture['batch'].to(device)
    mask = fixture['mask'].to(device)

    class FixedMask:
        def pop_or_generate_mask(self, seqs, mask_token_id, **kwargs):
            if tuple(seqs.shape) != tuple(mask.shape):
                raise ValueError('Unexpected infill partition')
            return torch.where(mask, torch.full_like(seqs, mask_token_id), seqs), mask

    forward_outputs = []

    def record_forward(module, inputs, output):
        if isinstance(output, tuple):
            forward_outputs.extend(output)
        else:
            forward_outputs.append(output)

    hook = model.register_forward_hook(record_forward)
    context = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
               if args.precision == 'bf16' else nullcontext())
    special = (torch.arange(config['vocab_size'], device=device) < 4)
    with context:
        loss, metrics = corosred_unified_step_pytorch(
            model, batch, config['vocab_size'], fixture['weights'],
            special_token_lut=special, causal_ratio=fixture['causal_ratio'],
            routing_cache=FixedMask(), metric_tracker=None,
        )
    loss.backward()
    hook.remove()

    def cpu(tensor):
        return tensor.detach().float().cpu().clone()

    gradients = {name: cpu(p.grad) for name, p in model.named_parameters() if p.grad is not None}
    initial = {name: cpu(p) for name, p in model.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device.type == 'xla':
        xm.mark_step()
    updates = {name: cpu(p) - initial[name] for name, p in model.named_parameters()}
    tensors = {**{'gradient/' + k: v for k, v in gradients.items()},
               **{'update/' + k: v for k, v in updates.items()},
               **{f'forward/{i}': cpu(v) for i, v in enumerate(forward_outputs)}}
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f'Nonfinite capture: {name}')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                         cwd=Path(__file__).resolve().parents[1]).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = None
    metadata = {
        'fixture_sha256': digest(args.fixture), 'device': str(device),
        'precision': args.precision, 'torch': str(torch.__version__), 'git_commit': commit,
        'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
        'torch_xla': str(torch_xla.__version__) if device.type == 'xla' else None,
        'probe_source_sha256': digest(__file__),
        'objective_source_sha256': digest(Path(__file__).resolve().parents[1] / 'telos/diffusion/corosred_unified.py'),
        'model_source_sha256': digest(Path(__file__).resolve().parents[1] / 'telos/models/transformer.py'),
        'components_source_sha256': digest(Path(__file__).resolve().parents[1] / 'telos/models/components.py'),
        'fresh_optimizer': True, 'diagnostic_weights': fixture['weights'],
        'loss': float(loss.detach().cpu()), 'preclip_gradient_norm': float(norm.detach().cpu()),
        'metrics': {k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else v
                    for k, v in metrics.items()},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'metadata': metadata, 'tensors': tensors}, args.output)
    print(json.dumps(metadata, indent=2))


def compare(args):
    left, right = [torch.load(p, map_location='cpu', weights_only=True) for p in (args.left, args.right)]
    for key in ('fixture_sha256', 'probe_source_sha256', 'objective_source_sha256', 'model_source_sha256', 'components_source_sha256'):
        if left['metadata'][key] != right['metadata'][key]:
            raise ValueError(f'Unmatched probe: {key}')
    if left['tensors'].keys() != right['tensors'].keys():
        raise ValueError('Different gradient/forward/update keys')
    differences = {}
    for key, a in left['tensors'].items():
        b = right['tensors'][key]
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f'Invalid comparable tensors: {key}')
        a, b = a.double().flatten(), b.double().flatten()
        delta = a - b
        na, nb = a.norm().item(), b.norm().item()
        differences[key] = {
            'max_absolute': delta.abs().max().item(),
            'relative_l2': delta.norm().item() / max(na, 1e-30),
            'cosine': (a @ b).item() / (na * nb) if na and nb else None,
        }
    report = {'left': left['metadata'], 'right': right['metadata'], 'differences': differences}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    worst = sorted(differences.items(), key=lambda kv: kv[1]['relative_l2'], reverse=True)[:8]
    print(json.dumps({'worst_relative_l2': worst, 'report': str(args.output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--row', type=int, default=0)
    p = commands.add_parser('capture')
    p.add_argument('--fixture', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda', 'xla'], required=True)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p = commands.add_parser('compare')
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    for p in commands.choices.values():
        p.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    {'prepare': prepare, 'capture': capture, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
```


### scratch/backend_parity/inspect_intermediate_50m.py

SHA-256 of embedded excerpt/source: `1e60e1b220f1d9d2ffd4f039d9a83c94465068084b21f1a35850c5589abce196`

```python
"""Inspect pinned HF checkpoints locally; evaluate the prior fixed CPU fixture."""
import sys,json,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts.compare_saved_50m_gradients import compare_checkpoint,sha256
ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
inventory=json.loads((ROOT/'scratch/backend_parity/hf_50m_inventory.json').read_text())
remote={x['path']:x for x in inventory['files']}
fixture_path=ROOT/'logs/saved_50m_gradient_comparison_20261018.fixture.pt'
f=torch.load(fixture_path,weights_only=True,map_location='cpu')
out=ROOT/'logs/intermediate_50m_comparison_20261018.json'
report={'hf_repo':inventory['repo'],'hf_revision':inventory['revision'],'fixture':str(fixture_path),'fixture_sha256':sha256(fixture_path),'device':'cpu','compute_dtype':'float32','threads':2,'note':'Same common evaluator as endpoint audit; nearby steps are not identical and this does not identify backend causality. No historical objective or optimizer replay.','models':{}}
selected=[('cuda', 'checkpoints/corosred/50m_lightning',s) for s in [1271,3813,7626,11439]]+ [('tpu','checkpoints/corosred/unified/50m_python',s) for s in [4000,8000,12000]]
for label,folder,step in selected:
 relative=f'{folder}/checkpoint_step_{step}.pt'
 path=ROOT/relative
 digest=sha256(path)
 if digest!=remote[relative]['sha256']:
  raise ValueError(f'Local checkpoint differs from pinned HF revision: {relative}')
 print(f'Start {label} step {step}; HF SHA-256 verified',flush=True)
 t=time.monotonic()
 result=compare_checkpoint(path,f['tokens'],f['infill_mask'],.5,.35)
 payload=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
 result['saved_keys']=list(payload)
 result['scheduler_state']=payload.get('scheduler_state_dict')
 result['lrh_acc_ema']=payload.get('lrh_acc_ema')
 result['lrh_auc_ema']=payload.get('lrh_auc_ema')
 result['optimizer_present']='optimizer_state_dict' in payload
 if result['optimizer_present']:
  opt=payload['optimizer_state_dict']
  result['optimizer_param_groups']=[{k:v for k,v in g.items() if k!='params'} for g in opt['param_groups']]
  result['optimizer_steps']=sorted({float(x['step']) for x in opt['state'].values() if 'step' in x})
 result['seconds']=time.monotonic()-t
 report['models'][f'{label}_{step}']=result
 temp=out.with_suffix('.tmp')
 temp.write_text(json.dumps(report,indent=2,default=lambda x: x.item() if isinstance(x,torch.Tensor) and x.numel()==1 else str(x)))
 temp.replace(out)
 g=result['gradient_geometry']['all_backbone']
 print(f'Done {label} {step}: C={result["causal"]["ce"]:.5f}, I={result["infill"]["ce"]:.5f}, cos={g["causal_infill_cosine"]:.4f}, {result["seconds"]:.1f}s',flush=True)
 del payload,result
 gc.collect()
print(out,flush=True)
```


# Appendix F — Focused regression tests

Reported passing groups: four saved-checkpoint gradient tests; 20 cadence/schedule/loss tests; nine optimizer/cadence/gradient tests (overlapping groups, not 33 unique tests). Tests establish the named properties in their scope, not hardware parity.


### tests/test_saved_checkpoint_gradient_audit.py

SHA-256 of embedded excerpt/source: `9347323a5c31246c8f9672552f901020a71dc835084e59792334eebe914aa483`

```python
"""Contracts for the diagnostic that replaces the incomplete historical audit."""
from unittest.mock import patch

import pytest
import torch

from scripts.compare_saved_50m_gradients import gradient_geometry, task_pass
from telos.models import TelosConfig, TelosTransformer


def tiny_model():
    torch.manual_seed(42)
    return TelosTransformer(TelosConfig(vocab_size=16, d_model=16, n_layers=1,
                            n_heads=2, n_kv_heads=2, max_seq_len=8,
                            is_causal=True, use_reliability_head=True)).eval()


def test_audit_explicitly_changes_attention_for_infill():
    model = tiny_model()
    tokens = torch.tensor([[2, 4, 5, 6, 7, 8, 9, 3]])
    mask = torch.tensor([[False, False, True, True, False, False, False, False]])
    with patch.object(model, 'forward', wraps=model.forward) as calls:
        task_pass(model, [tokens], [mask], infill=False)
        assert calls.call_args.kwargs['mask_override'] is True
        task_pass(model, [tokens], [mask], infill=True)
        assert calls.call_args.kwargs['mask_override'] is False


@pytest.mark.parametrize('infill', [False, True])
def test_audit_microbatches_equal_full_fixture_gradient(infill):
    model = tiny_model()
    tokens = torch.tensor([[2, 4, 5, 6, 7, 8, 9, 3], [2, 9, 8, 7, 6, 5, 4, 3]])
    # Different mask counts catch accidental averaging of microbatch means.
    mask = torch.tensor([[False, True, False, False, False, False, False, False],
                         [False, True, True, True, True, False, False, False]])
    full_metrics, full_gradients = task_pass(model, [tokens], [mask], infill=infill)
    split_metrics, split_gradients = task_pass(
        model, list(tokens.split(1)), list(mask.split(1)), infill=infill)
    assert full_metrics['ce'] == pytest.approx(split_metrics['ce'], abs=1e-6)
    assert full_metrics['top1_pct'] == split_metrics['top1_pct']
    for key in full_gradients:
        torch.testing.assert_close(full_gradients[key], split_gradients[key], atol=1e-6, rtol=1e-5)


def test_gradient_geometry_detects_task_conflict():
    g = {'layers.0.weight': torch.tensor([3., 4.])}
    opposite = {'layers.0.weight': -g['layers.0.weight']}
    result = gradient_geometry(g, opposite, .5, .5)['all_backbone']
    assert result['causal_gradient_norm'] == 5.
    assert result['causal_infill_cosine'] == -1.
    assert result['combined_gradient_norm'] == 0.
```


### tests/test_cadence_path_audit.py

SHA-256 of embedded excerpt/source: `918c310e0581ec283a60a1506977d868e9f3058febbc8ac54fa4fe75456b99e1`

```python
"""Causal reachability matters more than observing different scheduler values."""
import torch
from scripts.audit_cadence_paths import gradient_probe
from telos.training.schedule import COROSredSchedule


def test_metric_feedback_does_not_change_nominal_backbone_weights():
    schedule=COROSredSchedule(max_steps=12716,alpha_min=.5,beta_max=.35)
    for step in [0,2543,4000,8000,12716]:
        below=schedule.get_weights(step,.5,.5)
        above=schedule.get_weights(step,.8,.8)
        assert below['alpha']==above['alpha']
        assert below['beta']==above['beta']
        assert below['gamma']==0.
        assert above['gamma']==.1


def test_routing_refresh_without_metrics_remains_disabled():
    schedule=COROSredSchedule(max_steps=12716)
    for step in [1,4000,8000,12716]:
        assert schedule.get_weights(step)['mask_blend']==0.
        assert schedule.get_weights(step,.8,.8)['mask_blend']==1.


def test_detached_head_can_affect_backbone_only_after_shared_clipping_here():
    torch.set_num_threads(1)
    result=gradient_probe()
    assert result['max_backbone_gradient_difference_before_clip']==0.
    assert result['head_gradient_norm_gamma0']==0.
    assert result['head_gradient_norm_gamma_point1']>0.
    assert result['max_backbone_gradient_difference_with_forced_active_clip_point25']>0.
```


### tests/test_saved_optimizer_replay.py

SHA-256 of embedded excerpt/source: `b61bef7c9c10618e500f65071dc9c8bcb008f5c05872b3dd6f999f675a9ceba3`

```python
"""Prevent a false optimizer-parity result from aliased replay states."""
import copy
import torch
from scripts.replay_saved_optimizer import build_optimizer,compare_tensors


def source_checkpoint():
    parameter=torch.nn.Parameter(torch.tensor([.5,-.2]))
    optimizer=torch.optim.AdamW([parameter],lr=.001,betas=(.9,.95))
    parameter.grad=torch.tensor([.2,-.3]);optimizer.step()
    return {'model_state_dict':{'weight':parameter.detach().clone()},
            'optimizer_state_dict':copy.deepcopy(optimizer.state_dict())}


def test_replay_optimizers_do_not_share_parameters_or_moments():
    checkpoint=source_checkpoint()
    a,oa=build_optimizer(checkpoint,[['weight']],False)
    b,ob=build_optimizer(checkpoint,[['weight']],True)
    before=b['weight'].detach().clone()
    moment=ob.state[b['weight']]['exp_avg'].clone()
    a['weight'].grad=torch.ones(2);oa.step()
    assert torch.equal(b['weight'],before)
    assert torch.equal(ob.state[b['weight']]['exp_avg'],moment)
    assert torch.equal(checkpoint['model_state_dict']['weight'],before)
    assert compare_tensors(a,b)['max_absolute']>0


def test_comparison_detects_different_prescribed_updates():
    checkpoint=source_checkpoint()
    a,oa=build_optimizer(checkpoint,[['weight']],False)
    b,ob=build_optimizer(checkpoint,[['weight']],True)
    a['weight'].grad=torch.ones(2)
    b['weight'].grad=-torch.ones(2)
    oa.step();ob.step()
    assert compare_tensors(a,b,checkpoint['model_state_dict'])['relative_l2_vs_plain_cumulative_update']>0
```


# End of supplied evidence

Please reason from the contents provided. You may propose a precise additional source excerpt or measurement if it is indispensable, but do not assume local paths can be opened. Report what is unknown rather than filling it with an invented explanation.
