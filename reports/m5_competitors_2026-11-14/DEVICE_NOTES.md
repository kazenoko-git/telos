# Native workflow feasibility

Source revisions are pinned in `upstream_manifest.json`.

- **MLX-Pretrain:** native Apple Silicon workflow. `Trainer.setup_model` loads
  MLX-LM architectures; the first controlled experiment compares that Llama
  backbone with Télos using identical tensors and a common optimizer loop.
  Its native `Trainer.train` instead uses document tokenization/padding, a
  different precision/optimizer setup, no equivalent gradient accumulation in
  the inspected loop, optional elementwise clipping, and per-step cache clearing.
  Testing its complete workflow requires a separately specified protocol.
- **MLX Examples:** native Metal transformer example, but its sinusoidal
  positions, LayerNorm, stock feed-forward blocks, untied output projection,
  and default FP32 training differ from the Télos AR workload. Treat a stock
  run as a separate workflow probe; do not put its tokens/s in a matched ranking.
- **nanochat:** source explicitly supports `--device-type mps` and autodetects
  MPS. It is eligible for an M5 native workflow test, not CUDA-only. Its declared
  dependencies pin PyTorch 2.9.1, requiring an environment separate from the
  controlled MLX test. The native base-training script uses CUDA synchronization
  only; MPS throughput instrumentation must explicitly synchronize the device.
  Its architecture, tokenizer, optimizer recipe, and evaluation target must be
  specified before a fair end-to-end comparison. No nanochat training has run yet.
- **LitGPT:** single-device pretraining uses Lightning Fabric auto accelerator
  selection. Its default precision helper is CUDA-oriented; test MPS with an
  explicit compatible precision and verify actual placement. A native MPS run
  has not yet been verified; do not label it unsupported from documentation alone.

These are source-inspection findings, not performance results or hardware passes.
