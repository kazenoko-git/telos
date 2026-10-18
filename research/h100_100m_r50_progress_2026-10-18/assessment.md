# H100 100M-tier / 5B-token progress snapshot — 18 October 2026

Source: supplied log `/Users/ivansamuel/.codex/attachments/200f4d0b-3e5e-4b51-bea9-da12a07de8cf/Pasted text.txt`; parsed482 training rows, through step11825 of25432. This is a supplied-log snapshot, not live monitoring. No TPU v5e-8 observations exist yet; the user reports its run has not started.

## Progress and throughput

- 46.50% of planned steps; approximately2.325B scheduled input tokens.
- Latest reported cumulative throughput:228,022tokens/sec. Trainer logs nominal input-token throughput, not tokens generated at inference or an independent GPU utilization measurement.
- Logged ETA:195.5minutes from this snapshot. Constant-speed extrapolation:6.09hours training, approximately$26.98 at the supplied$4.43/hour rate; startup/download/upload and resource bursting can change actual charges.

## Learning evidence

- [Dual Probe Step   2000] Causal Top-1: 60.6% | C-CE: 1.994 | Infill Top-1: 76.9% | I-CE: 1.190 | LRH AUC: 0.515
- [Dual Probe Step   4000] Causal Top-1: 63.4% | C-CE: 1.787 | Infill Top-1: 80.4% | I-CE: 0.952 | LRH AUC: 0.507
- [Dual Probe Step   6000] Causal Top-1: 64.6% | C-CE: 1.691 | Infill Top-1: 82.6% | I-CE: 0.832 | LRH AUC: 0.505
- [Dual Probe Step   8000] Causal Top-1: 65.5% | C-CE: 1.629 | Infill Top-1: 83.3% | I-CE: 0.793 | LRH AUC: 0.507
- [Dual Probe Step  10000] Causal Top-1: 66.0% | C-CE: 1.594 | Infill Top-1: 83.9% | I-CE: 0.745 | LRH AUC: 0.491

The fixed-tail monitor improves in both tasks. The monitor explicitly uses the tail200 sequences of the training corpus; these are training-corpus probes, not established held-out validation. Printed minibatch losses fluctuate and isolated spikes near9550/9600 return promptly to the surrounding range. No traceback, OOM or nonfinite-loss markers were found in the supplied log.

All sampled training rows print gamma0.00; sampled LRH AUC remains around chance. This records a visible inactive/near-zero reliability contribution at those log points, not a proof of exact-zero gradients at every unlogged step, or a TPU/CUDA causal diagnosis. Do not change the recipe midway through the paired comparison. Preserve intermediate checkpoints and the full log for matching to the later TPU run under a common evaluator.
