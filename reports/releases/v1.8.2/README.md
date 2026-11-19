# v1.8.2 physical accelerator evidence

See the [validation assessment](../../../docs/releases/v1.8.2-validation.md)
for tested scope, findings, and limitations.

- `tpu_optimizer/`: isolated TPU optimizer and small FP32 AR comparisons.
- `tpu_full_trainer/`: representative BF16 native AR trainer, accumulation, and replay.
- `cuda_ddp/`: representative FP16 fallback, two T4 DDP workers, accumulation, and replay.
- `manifest.json`: SHA-256 hashes of source ZIPs and retained evidence files.

JSON reports and logs are preserved from user-provided archives. Binary
checkpoints were excluded. Exact replay results come from notebook comparison
summaries; the retained files do not permit independently repeating those
checkpoint comparisons. No scripts or instructions contained in the archives
were executed during this documentation update.
