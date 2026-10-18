# Completed H100 100M-tier / 5B-token run — supplied log

The full supplied log confirms completion of25,432 steps in365.48minutes (6.091hours), with all12 intermediate-save success messages and the final completion banner. The exact source copy is `full_training_log_user.txt`; parsed data are in `full_training_metrics.json`.

Final printed minibatch: loss1.0080, causal CE1.199, infill CE0.681; alpha0.60, beta0.35, gamma0.00. These are minibatch values, not a common-fixture final evaluation.

Last periodic fixed-tail probe (step24,000): causal CE1.446 / top1 68.6%; infill CE0.597 / top1 86.6%; LRH AUC0.487. Both CE trajectories improve at every logged periodic probe. These probes use the tail200 sequences from the training corpus, not an established held-out set.

Nominal input-token count from resolved steps/batch/context:5,000,134,656. Final reported cumulative throughput228,018tokens/sec. The supplied$4.43/hour price gives approximately$26.98 for training time alone; setup, transfers and actual resource billing may differ.

All1027 parsed sampled training rows print gamma0.00; the reliability signal remains near chance. This does not establish exact zero at every unlogged step. No nonfinite-loss/error markers were found. No runtime changes were made, and no paired TPU run has started per the user's latest status. These observations do not identify the historical TPU/CUDA gap's cause.
