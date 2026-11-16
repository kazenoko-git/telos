# v1.7.1: six local failures fixed

Base: released v1.7.0 (`6037f123c39772f10b0bd966285acb5fa7e4b281`).
Runtime: Python 3.13.14, MLX 0.32.2, PyTorch 2.13.0, Apple M5 Pro/Metal,
macOS 27.2. No CUDA or TPU hardware was used.

## Results

- All five original sandbox tests pass through the real macOS sandbox.
- The original FP32 and BF16 checkpoint-gradient test function is preserved
  verbatim, including every numerical tolerance, and both cases pass.
- Three consecutive full local runs: **299 passed, 7 skipped, zero failures**
  each (14.30 s, 12.75 s, 13.16 s).
- After adding the independent Float64 transformer-gradient oracle, final full
  local run: **300 passed, 7 skipped, zero failures**, 14.22 s.
- Ruff's CI checks and `git diff --check` pass; wheel and source distribution
  build successfully.
- Extracted v1.7.1 wheel passes isolated version/import, real sandbox launch
  and exact repeated-token gradient smoke checks outside the source tree.

The seven skips remain hardware/dependency-gated checks, not skipped versions
of these six failures. CUDA/TPU acceptance is unchanged from v1.7.0.

## Sandbox launch root cause and fix

The virtualenv interpreter traversed an intermediate uv symlink outside the
readable runtime roots. The original temporary directory string also used the
`/var` alias, while the policy allowed its canonical `/private/var` path.
After resolving those paths, dyld separately aborted because opening `/` as
its directory anchor was denied.

Resolve the command entry point and scratch path before applying the policy.
Allow `file-read-data` on **literal `/` only**, not a root subpath. Keep the
original deny-default/network policy, narrow runtime reads, scratch-only writes
and no Python process-fork permission. The new real OS tests prove unrelated
file reads/writes, parent-network access and process forking remain denied,
while scratch aliases and runtime startup work. The fork test runs directly
under the policy without the executor's RLIMIT_NPROC, so a busy user's process
count cannot make the test pass accidentally.

## FP32 gradient root cause and fix

MLX 0.32.2's `env::enable_tf32()` defaults to 1. On M5, its matmul dispatch can
therefore use TF32 tensor operations for arrays typed `float32`. The original
test compared eager/plain against compiled/checkpointed; a four-way probe
showed eager/checkpointed matched eager/plain, and compiled/plain matched
compiled/checkpointed. The failing comparison was across compilation contexts
under reduced-precision FP32 GEMMs, not a disconnected checkpoint gradient.

Configure `MLX_ENABLE_TF32=0` before first MLX matmul by default on macOS.
An explicitly provided `MLX_ENABLE_TF32=1` is preserved. The original failing
comparison passes without changing a bound. Independent CPU/Float64 transformer
losses and every parameter gradient also pass the original FP32 bounds for
both eager and compiled execution.

`tf32_on_comparison.txt` and `tf32_off_comparison.txt` record the four-way
comparison. Tuple entries are `(checkpointing, compilation)`. With TF32 on,
11 elements of the first QKV weight gradient violate the original threshold,
with maximum absolute difference 6.649643e-6. With TF32 off, no comparisons
violate it. The test suite also compares a nontrivial-mantissa FP32 GEMM against
NumPy Float64, preventing a dtype-only claim of full precision.

MLX caches the setting at first matmul. Importing Télos cannot retroactively
reconfigure a process whose backend was already initialized by another library;
set the variable before starting Python in that situation. BF16/FP16 matmuls
retain their acceleration. Full FP32 can be slower, and corrected arithmetic
changes trajectories relative to v1.7.0. No speedup is claimed for this patch.

## Additional BF16 instability found and fixed

The full suite exposed a repeated-token embedding gradient issue. Native
low-precision scatter-add rounds individual updates, and GPU update ordering
can alter the result. Instead, the embedding lookup's custom VJP accumulates
in FP32 and casts back once. Parameter keys, shapes, forward lookup and tied
output projection are unchanged.

A regression uses 257 identical IDs with incoming gradients `1` followed by
256 copies of `1/256`. Every term is representable; the exact sum is `2`.
Native BF16 lookup backward produced `1.859375` in the recorded run; the fixed
path produces exactly `2`. `repeated_token_gradient.json` records the result.
Tests cover BF16/FP16, eager/compiled execution, and the original checkpoint
comparison plus matched eager/eager and compiled/compiled comparisons.

## Reproduction and source

```sh
MLX_ENABLE_TF32=0 /path/to/python -m pytest -q
/path/to/python -m pytest -q tests/test_eval_suite.py::TestSandboxExecutor \
  tests/test_sandbox_security.py tests/test_mlx_checkpoint_gradients.py \
  tests/test_mlx_precision.py
uvx ruff check telos/ tests/ --select E9,F63,F7,F82
uv build
```

- Original tests preserved: `original_tests_preserved.json`.
- Repeated suite results: `repeated_full_suite.json`.
- Final suite: `final_suite.txt`.
- [Installed-version TF32 default](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/utils.h#L214-L217).
- [Installed-version matmul selection](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/matmul.cpp#L917-L919).
