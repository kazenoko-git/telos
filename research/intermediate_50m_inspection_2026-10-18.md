# Intermediate 50M checkpoint inspection — 18 October 2026

## Eligibility and provenance

The user explicitly warned against mixing old separate Phase A/B/C checkpoints into the unified COROSred comparison. The selected CUDA Lightning and TPU Python intermediates explicitly declare `corosred.unified=True` and `phase=unified`, with matching model architecture and main task settings. Their SHA-256 hashes match the pinned HF revision below. These flags are evidence of intended configuration, not proof of the exact historical training source.

- HF repo: https://huggingface.co/Kazenowoko/telos
- Revision: `d728840aada52094748903ddc56256807a0bfd61`
- CUDA files last uploaded 14 September 2026; TPU Python files uploaded 18 September in a commit describing LR and schedule cadence 1. Upload dates are not training dates.
- Older `50m_2.5b` experiment excluded: differing task settings and embedding/output tensors that disagree despite the tied-embedding flag. Its saved Adam state is not used to explain the later backend gap.
- Separate-phase legacy models excluded.

## Important lineage mismatch

The `50m_python` intermediates save scheduler cadence 1 and explicit cadence-1 configuration, while the final checkpoint in that same directory saves cadence 10 and lacks those explicit settings. Do not connect that final to the intermediates as a verified continuous run.

Relative weight L2 distance from the step-12,000 intermediate (output weight alias excluded):

| Candidate | Relative distance | Weight cosine |
|---|---:|---:|
| `checkpoints/corosred/unified/50m_python/checkpoint_step_8000.pt` | 0.136598 | 0.990639 |
| `checkpoints/corosred/unified/50m_python/checkpoint_final.pt` | 1.240007 | 0.228330 |
| `checkpoints/corosred/unified/50m_tpu_schedstep1/checkpoint_final.pt` | 1.241478 | 0.232037 |
| `checkpoints/corosred/unified/50m_tpu_lrcadence_schedstep1/checkpoint_final.pt` | 0.024999 | 0.999688 |

The local `50m_tpu_lrcadence_schedstep1` endpoint is strongly compatible with the intermediates by weight proximity and settings. Its SHA-256 (`5e62ca9fedc1aaf2ad3951c5a4f7da6085bbf661cf0b25b4cb28bda0c69ae781`) matches the pinned HF `50m_python/checkpoint_final.pt`; its contents are therefore present remotely under a different path. The local `50m_python/checkpoint_final.pt` used above has a different hash. The path/content distinction was missed in the original report and corrected after Argon’s review. Matching hashes establish file identity, not an independently verified uninterrupted training/resume chain or the reason for local renaming/overwriting.

## Common-fixture measurements

All evaluations reuse the prior exact seed-42 fixture: 16 sequences of 512 tokens (8,192 input tokens), four microbatches, CPU FP32, dropout disabled. Causal and bidirectional infill attention are explicit. Infill loss uses masked target positions only. These are endpoint gradients under the current common evaluator, excluding reliability loss; they do not reconstruct the historical mixed objective, clipping, or Adam updates. This slice is not established to be document-disjoint from training.

| Checkpoint | Step | Causal CE | Infill CE | Task-gradient cosine |
|---|---:|---:|---:|---:|
| cuda_1271 | 1271 | 2.20181 | 2.40955 | 0.1571 |
| cuda_3813 | 3813 | 1.74667 | 1.65492 | 0.1252 |
| cuda_7626 | 7626 | 1.55809 | 1.23012 | 0.0874 |
| cuda_11439 | 11439 | 1.48908 | 1.09837 | 0.0968 |
| tpu_4000 | 4000 | 2.43701 | 2.82393 | 0.2688 |
| tpu_8000 | 8000 | 2.11615 | 2.11556 | 0.2155 |
| tpu_12000 | 12000 | 1.99400 | 1.80241 | 0.1678 |
| TPU cadence-1 candidate final | 12716 | 1.99400 | 1.77821 | 0.1858 |

## Bounded interpretation

- The quality deficit is already present at TPU step 4,000: both losses are worse even than CUDA step 1,271 on these identical inputs. Nearby-step comparisons also favor CUDA. This establishes a deficit by the first eligible TPU save, not its onset or training cause.
- TPU continues improving between its observed intermediate saves. These sparse checkpoints cannot rule out instability between saves or prove a convergence floor.
- Overall causal/infill gradients are positively aligned at each measured intermediate. Layer-level geometry is retained in JSON; endpoint alignment does not rule out historical training-time conflicts.
- Historical moments are absent from the selected TPU intermediates. The older experiment with moments is excluded from causal comparison.
- No accelerator time or training was used. Production training code was not changed. Four existing regression tests of the shared evaluator passed.

## Artifacts

- `logs/intermediate_50m_comparison_20261018.json`: seven checkpoint measurements, configs, scheduler state, gradients and HF provenance.
- `logs/intermediate_50m_cadence1_final_20261018.json`: local candidate endpoint measurement.
- `logs/intermediate_50m_lineage_distances_20261018.json`: candidate weight-distance measurements.
- `scratch/backend_parity/hf_50m_inventory.json` and `hf_50m_file_history.json`: pinned HF inventory and file history.

The productive next target is the unified trainer path before step 4,000. Saved checkpoints narrow the interval; they do not yet distinguish BF16 kernels, optimizer execution, clipping, or other training behavior.
