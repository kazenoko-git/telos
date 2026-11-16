# v1.7.0 local release validation

- Source base: v1.6.2 main `08e56fe6662ff7cc65ce253d27b20275cd25fab8`.
- Local runtime: Python 3.13.14, PyTorch 2.13.0, MLX 0.32.2, Apple M5 Pro/Metal.
- Final full suite: **277 passed, 7 skipped, 6 failed** in 11.48 seconds.
- All six remaining failures reproduced on untouched v1.6.2: five sandbox-executor tests fail because macOS rejects the sandbox child `execvp`; one MLX FP32 checkpoint-gradient test exceeds its unchanged tolerance.
- Baseline failure subset: 18 failed, 15 passed. Twelve of those were stale MLX experimental-accumulation wrapper API calls; the wrapper is fixed here and those twelve checks pass.
- New/changed functionality: CPU/Metal SFT/DPO with full weights, LoRA and 4-bit QLoRA; frozen bases; adapter and merged exports; same-directory MLX resume; nonzero dropout and BF16 master resume; packed projections/gradients/resume; tiled vocabulary CE/gradients; benchmark timing; input-pipeline cancellation; mocked XLA control flow and DDP group ownership.
- Both CPU and Metal ran normal AR, MDLM, UNDLM and COROSred benchmark loops: five warmups followed by five measured updates each. `benchmark_smoke.json` records the results.
- Synthetic Metal A/B/B/A end-to-end runs used 2,048 updates. `posttrain_abba.json` contains timings and source hashes. Interpret them with the caveats in `docs/optimization-v1.7.0.md`; repeated DPO inputs produce a best-case cache hit rate.
- `uv lock --offline`, wheel and source build passed. Extracted-wheel import/version, native LoRA/QLoRA update/reload and bundled 512-case document suite integrity checks passed (`package_validation.json`).
- No CUDA or TPU hardware was available. Physical copy-stream overlap, mixed-precision CUDA/scaler continuation, multi-GPU DDP, native GQA, TPU launcher/collectives/compilation and sustained convergence remain unverified. CPU simulations do not establish accelerator acceptance.

Command: `/Users/ivansamuel/Documents/Projects/telos/.venv/bin/python -m pytest -q` from the v1.7 worktree. The suite's nonzero exit is retained and reported; no failed tests were removed or relaxed.
