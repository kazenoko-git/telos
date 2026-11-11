# Neural Engine feasibility probe — 2026-11-11

Measured on Apple M5 Pro, Core ML Tools 9.0, MLX 0.32.2. Core ML Tools was installed in /tmp/telos-ane-deps, outside the project environment.

Core ML was configured CPU_AND_NE (GPU excluded). Its compute plan selected MLNeuralEngineComputeDevice for both a static 1x1 convolution and a matrix multiplication with runtime weight inputs. This is estimated placement, not hardware occupancy evidence.

The runtime-weight test changed the supplied weights and verified the new output without converting or loading another model. Thus recompilation on each update is not required for this particular dynamic-weight matrix program.

| Probe | Core ML median | MLX resident median |
|---|---:|---:|
| Fixed-weight projection | 0.178 ms | 0.209 ms |
| Runtime-weight projection | 0.226 ms | 0.205 ms |

Both implement (256,512) @ (512,512). Fixed-weight projection uses an ANE-friendly convolution layout; runtime weights use matmul. Host API and transfer costs differ between resident MLX and Core ML predict, so these rows are operational latency comparisons, not isolated accelerator kernel timings.

## Hybrid update

The hybrid test executes FP16 linear forward through Core ML, then analytic FP32 backward and a simple SGD update on the GPU. It sends updated weights on each prediction and returns the activation to MLX. The GPU-only comparison compiles forward, backward and update together; the hybrid comparison compiles its backward/update. Neither runs another GPU forward for differentiation.

Trial order GPU / hybrid / hybrid / GPU. Each trial starts from identical weights and uses 100 warmup updates plus 1,000 measured updates.

Trial medians (ms): [0.22045901278033853, 0.372291513485834, 0.34579201019369066, 0.17327099340036511]

Pooled step medians: GPU 0.192 ms; hybrid 0.354 ms (1.84x GPU latency). Both GPU trials were faster than both hybrid trials.

Final maximum weight difference: 1.64e-05. Final losses: GPU 0.05127796; hybrid 0.05128083. These are close numerical results, not exact parity. This probe uses FP16, not the production BF16 training format.

System swap: 5708.56 MiB before and 5708.56 MiB after. Final memory headroom: 57%. MLX peak: 11.75 MiB; this excludes Core ML allocations, so it is not a whole-process memory measurement.

## Scope

This is a small linear layer and synthetic input, not a 50M transformer, AdamW, SwiGLU, attention, or an ANE training occupancy measurement. Dynamic matmul support does not prove the entire transformer maps to ANE. Repeated timing and numerical quality evaluation would be required before using this in production. The production training engine does not use this hybrid path.

## Reproduction

```sh
PYTHONPATH=/tmp/telos-ane-deps:. .venv/bin/python scripts/probe_neural_engine.py
```

The script writes logs/neural_engine_probe_20261111.json and requires coremltools 9.0 plus its dependencies. Raw results include compute-plan operation placements and system memory samples.

Source: [Apple Core ML compute-plan API](https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.compute_plan).
