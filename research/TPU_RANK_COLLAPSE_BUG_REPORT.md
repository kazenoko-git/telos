# Silent TPU replica-rank collapse in the Telos trainer

**Primary investigation report · 20 October 2026**

## Executive finding

Historical completed Telos TPU runs incorrectly identified all eight data-parallel workers as rank zero. The trainer uses this rank to choose each worker's input slice. In the inspected replicated training path, this collapses eight distinct slices into eight presentations of the same slice while retaining the eight-worker data-pointer stride and nominal token accounting.

This is a **confirmed historical rank-identity defect with a verified data-indexing consequence**, affecting the shared trainer used by both AR and unified COROSred. It is a strongly supported contributor to the historical training-quality deficit. The available evidence does **not** isolate how much of the deficit it caused or prove it was the only cause.

The new 100M-tier TPU run used a notebook-applied compatibility patch, logged distinct ranks 0–7, and performs broadly comparably to the H100 checkpoint on common losses, contextual probes, HumanEval and private functional suites.

## Mechanism

The original trainer obtains the TPU rank using:

```python
try:
    import torch_xla.core.xla_model as xm
    self.rank = xm.get_ordinal()
except Exception:
    self.rank = 0
```

The official PyTorch/XLA **v2.8.0** `xla_model.py` does not define `get_ordinal`. Its `runtime.py` provides `global_ordinal`. An unavailable API therefore becomes a plausible-looking integer rather than an initialization error. The report verifies this tagged version; it does not establish the first release that removed the API.

The replicated trainer then computes:

```python
local_step_seqs = batch_size * gradient_accumulation
cluster_step_seqs = local_step_seqs * world_size
idx_ptr = rank * local_step_seqs
# Each worker reads local_step_seqs rows from idx_ptr.
idx_ptr = (idx_ptr + cluster_step_seqs) % n_rows
```

For eight workers, batch 48 and accumulation 1:

| Worker | Correct first-step rows | Rows with silent rank-zero fallback |
|---|---|---|
| 0 | 0–47 | 0–47 |
| 1 | 48–95 | 0–47 |
| 2 | 96–143 | 0–47 |
| 3 | 144–191 | 0–47 |
| 4 | 192–239 | 0–47 |
| 5 | 240–287 | 0–47 |
| 6 | 288–335 | 0–47 |
| 7 | 336–383 | 0–47 |
| **Distinct sequence rows per update** | **384** | **48** |

At the next update, the buggy workers all start at row 384. Rows 48–383 were not consumed during the first update. Before wraparound, the pattern skips seven of every eight scheduled slices.

The same local-step calculation applies to the recovered AR run's batch 16 × accumulation 3: each worker processes 48 sequences per optimizer update.

### Why this impairs training

- An update receives much less varied raw input than intended.
- The corpus pointer advances as though all eight distinct slices were consumed.
- Training still stops at the planned number of optimizer updates, with the LR schedule progressing normally.
- Reported token throughput counts executed sequence presentations, including duplicates. It is not evidence that the intended distinct corpus positions were consumed.

If the defect persists throughout a nominal 5B-token pass before corpus wraparound, input exposure covers roughly **625M distinct-position tokens**, repeated across workers. This is not a claim of 625M unique token values or unique documents. The analogous nominal 2.5B-token budget is approximately 312.5M distinct-position tokens under the same assumptions.

Gradient averaging across repeated input does not multiply the update by eight. Random masks and dropout may still differ, so duplicate input does not prove identical gradients or stochastic computation. It is also **not exactly equivalent to running one eighth as many optimization steps**: update count, LR trajectory, moments and repeated exposure remain different.

## Direct historical evidence

Full logs were downloaded through Kaggle's visible UI or supplied by the user in `~/Downloads`, preserved locally, and hashed. These are runtime messages, not generated sample logs or code comments.

| Evidence | Logged run | Completion and rank evidence |
|---|---|---|
| `kazenoko/100m-ar-5b`, Version 4, script ID `350392839` | Unified 75M; `75m_python`; 19,074 steps | Completed and uploaded; **eight rank-zero messages** |
| Same notebook, Version 8, script ID `351499919` | Unified 50M; `50m_seed`; 12,716 steps | Completed and uploaded; **eight rank-zero messages** |
| Alternate-account download `telos-100m-ar (1).log` | Unified 50M; `50m_python`; 12,716 steps | Completed and uploaded; **eight rank-zero messages** |
| Alternate-account download `telos-100m-ar.log` | Two sequential unified 50M runs; both write/upload `50m_python` | Both completed; **eight rank-zero messages per run** |
| Alternate-account download `telos-100m-ar (2).log` | **AR 100M**; `ar/100m_5b`; 25,432 steps; batch 16 × accumulation 3 | Completed and uploaded; **eight rank-zero messages**; log records source commit **`b286a90`** |
| New successful TPU run, script ID `359630346` | Unified 100M; `100m_r50_tpu_v5e8_seed42_cad1`; 25,432 steps | Completed and uploaded; **one message for each rank 0–7** |

The user identifies the alternate-account downloads as Versions 2, 4 and 6. The exact filename-to-version mapping has not been independently recovered and is deliberately not guessed here. Likewise, a notebook titled `100M-AR-5B` can contain a 50M or 75M COROSred run; model identity above comes from the logged configuration and output path.

### Why the AR evidence is especially useful

The AR log explicitly prints:

```text
HEAD is now at b286a90 fix: eliminate PyTorch-XLA double gradient reduction and add final checkpoint rendezvous
```

Inspection of that Git revision confirms both the silent `xm.get_ordinal()` fallback and the use of `self.rank` for the data offset. This connects an observed completed AR run, its logged source revision and the defective indexing mechanism more directly than folder-name inference alone. A printed Git revision does not exclude every possible unrecorded runtime edit, but no such edit is evidenced in the log.

Worker output buffering explains why several rank messages appear at shutdown rather than next to the first rank message. The complete downloaded logs, rather than only Kaggle's first displayed log page, contain all eight.

## Fix used by the successful run

The fresh Kaggle notebook replaced rank acquisition with:

```python
import torch_xla.runtime as xr
self.rank = int(xr.global_ordinal())
```

Failure now propagates rather than silently impersonating rank zero. The notebook also disables silent single-core launch fallback.

Applying only this rank replacement to the inspected current trainer reproduces the exact trainer hash recorded by the successful TPU run:

```text
888d21b53e734192ed7cb560b352de8fc36125855c5bd81ddd9bc7263312381b
```

Recorded runtime: `telos-ml==1.0.0`, Torch `2.8.0+cpu`, Torch/XLA `2.8.0`, libtpu `0.0.17`. A pure index-arithmetic replay verifies 48 versus 384 distinct sequence rows under all-zero versus correct ranks. No new training was launched for that replay.

**Fix distribution status (updated 20 October):** the successful run used a notebook-installed source patch. The repository trainer and version **1.1.0** release sources now use `xr.global_ordinal()`, abort if rank lookup fails, and reject out-of-range ranks in replicated execution. Regression tests instantiate the trainer with an XLA runtime double lacking the removed API, verify eight distinct ranks and disjoint input slices, and verify lookup errors and invalid ranks abort. The rank and reproducibility tests pass: **8 passed**; the complete release test suite passes **100 tests**. These are CPU-based regression checks, not a new physical TPU run. Installing the older **1.0.0** wheel still requires the notebook compatibility patch.

## Outcome of the successful run

| Common-fixture metric | H100 | Patched TPU |
|---|---:|---:|
| Causal CE | 1.360218 | 1.356237 |
| Infill CE | 0.589983 | 0.593795 |
| Causal top-1 | 70.62% | 70.70% |
| Infill top-1 | 86.89% | 87.02% |
| Reliability correctness AUROC | 0.4957 | 0.8510 |

| Strict functional passes | H100 | Patched TPU |
|---|---:|---:|
| Full HumanEval | 4/164 | 3/164 |
| Private base | 10/512 | 10/512 |
| Private hint | 20/512 | 22/512 |

Both HumanEval reruns reproduced all raw completions exactly. The small task-count differences do not isolate backend effects or training-seed variability. The old large quality gap is absent in these measurements.

The successful run also changed cadence settings, objective recipe details and environment hygiene relative to older experiments. There is no otherwise-identical patched-versus-unpatched training ablation. Its success supports the rank diagnosis but does not independently measure the patch's isolated effect.

## Why earlier cadence experiments did not exclude this bug

Schedule cadence and LR cadence do not change replica rank acquisition. The historical 50M LR+schedule-one checkpoint really saves LR update cadence 1 and still underperforms. It does not establish every-step metric synchronization, but changing cadence could not repair the invalid rank call in any case.

Earlier debugging addressed many real issues: precision/storage, gradient reduction, weight tying, scheduling and nominal data accounting. The rank-partitioning code itself had already been introduced in commit `b19e133` on 7 September. That change intended to eliminate duplicate slices but relied on a rank API whose failure was hidden. Valid-looking world-size and throughput messages could therefore coexist with ineffective partitioning.

This distinction matters: configuration arithmetic saying “48 × 8 = 384” is insufficient when the eight worker identities are wrong.

## Separate Version 7 checkpoint-storage incident

Version 7, script ID `351383019`, planned a 50M run with 25,432 steps and checkpoint saves every 200 steps. It is **not a completed quality endpoint**.

Its log confirms 50 checkpoint writes through step 10,000 and ends while saving step 10,200 at 4,152.7 seconds. Kaggle records cancellation at 11,661.1 seconds. Approximately 10.1 GiB was free after downloading the dataset; each checkpoint was 214.5 MB. Fifty checkpoints occupy approximately 9.99 GiB, while 51 require approximately 10.19 GiB.

Disk exhaustion at the next write is a strong inference consistent with the observed stopping point. There is no explicit disk-full traceback or proven cancellation reason in the log. This storage incident must not be conflated with the completed runs' rank bug.

## Discovery credit and hypothesis boundaries

**Gemini 4 Argon deserves specific credit for identifying the decisive hypothesis early.** In the assessment supplied by the user, Hypothesis 2 identified the removed rank API, silent rank-zero fallback, duplicate worker slices and retained cluster-sized stride. That hypothesis directed the subsequent investigation toward the startup logs that corroborated the rank defect.

The user's quoted hypothesis was:

> Hypothesis 2: Silent `self.rank = 0` Fallback (`xm.get_ordinal()` Removal) & Un-Offset Worker RNG Seeds Causing 8x Replica Lockstep

The rank/data-slicing portion is now supported by recovered historical logs and code inspection. **The RNG-lockstep extension has not been measured**, and identical model initialization across data-parallel replicas is desirable. Neither exact eightfold CUDA-equivalent steps nor all stochastic replicas being identical should be presented as established findings.

The user reports that Gemini 3.8 Flash, Opus 4.6, Opus 5.5, DeepSeek V4.1 Flash and GLM 5.3 did not resolve the investigation. That is project-history attribution, not a controlled comparison of model capabilities. Research credit here attaches to the specific useful hypothesis and the evidence that subsequently supported it.

The 18 October Argon triage correctly treated the historical attribution as unconfirmed with the evidence then available. The newly recovered Kaggle logs materially strengthen that evidence; they do not retroactively validate every claim in Argon's assessment.

## Paper-ready statement

> We identified a silent replica-rank failure in the Telos PyTorch/XLA training path. Completed historical AR and unified COROSred runs logged all eight workers as rank zero. In the inspected trainer, this rank controls input-slice offsets while the data pointer retains an eight-worker stride, reducing distinct input positions per update by a factor of eight under the observed configuration. A subsequent TPU run using the supported global-rank API logged distinct worker ranks and achieved performance comparable to the CUDA reference. Because other training settings also changed, we attribute the indexing defect confidently but do not isolate its contribution to the full historical quality gap.

## Evidence index

- [Downloaded Kaggle logs and original-account manifest](tpu_success_vs_historical_2026-10-20/kaggle_logs/manifest.json)
- [Alternate-account log manifest](tpu_success_vs_historical_2026-10-20/kaggle_logs/alternate_account_manifest.json)
- [API source, trainer-hash and index-arithmetic verification](tpu_success_vs_historical_2026-10-20/rank_api_and_index_verification.json)
- [AR trainer at recorded commit b286a90](tpu_success_vs_historical_2026-10-20/trainer_b286a90.py)
- [Historical checkpoint inventory and hashes](tpu_success_vs_historical_2026-10-20/checkpoint_inventory.json)
- [Argon source attribution and assessment hash](tpu_success_vs_historical_2026-10-20/argon_attribution.json)
- [Completed functional comparison](tpu_h100_functional_2026-10-20/comparison.md)
- [Anti-cheat, probes and generation diagnostics](tpu_h100_functional_2026-10-20/anticheat_diagnostics_report.md)
- Official versioned API sources: [XLA v2.8 xla_model.py](https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/core/xla_model.py), [XLA v2.8 runtime.py](https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/runtime.py).

No new accelerator run was performed to prepare this report. The initial documentation pass left production code unchanged; the subsequent repository fix and 1.1.0 release preparation are recorded above. Research records describe the stages at which their evidence was collected; they are not live package-publication status indicators.
