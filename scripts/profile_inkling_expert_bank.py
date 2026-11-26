"""Compare exact NVFP4 projection strategies on one Inkling expert bank."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from evaluate_inkling_window_adapter import guard_check, sample, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from telos.models.inkling2 import InklingTextLayout

    initial = sample()
    guard_check(initial, initial)
    mx.set_memory_limit(3072 * 2**20)
    mx.set_cache_limit(128 * 2**20)
    bank = InklingTextLayout.from_checkpoint(args.base).expert_banks[6]
    if bank.mode != "nvfp4" or bank.top_k != 8:
        raise ValueError("Expected Inkling Mini's eight-expert NVFP4 bank")

    packed = []
    for expert in range(bank.top_k):
        guard_check(sample(), initial)
        tensors = {key: entry.expert(expert).load(uncached=True) for key, entry in bank.ranges.items()}
        mx.eval(tensors)
        packed.append(tensors)
    expanded = [{name: mx.dequantize(
        tensors[name + ".weight"], tensors[name + ".scales"], mode="nvfp4",
        global_scale=tensors[name + ".global_scale"], dtype=mx.bfloat16,
    ) for name in ("gate_up_proj", "down_proj")} for tensors in packed]
    mx.eval(expanded)

    def dense_projection(hidden, weights):
        fused = (hidden @ weights["gate_up_proj"].T).astype(mx.float32)
        gate, up = fused[..., ::2], fused[..., 1::2]
        activated = (
            bank.gate_softcap * mx.tanh(gate / bank.gate_softcap) * mx.sigmoid(gate)
            * (bank.up_softcap * mx.tanh(up / bank.up_softcap))
        ).astype(hidden.dtype)
        return activated @ weights["down_proj"].T

    def project(hidden, weights, mode):
        out = mx.zeros(hidden.shape, dtype=mx.float32)
        for i, tensors in enumerate(packed):
            if mode == "scatter":
                row, slot = mx.array([0]), mx.array([i])
                values = bank.project(hidden[row], tensors).astype(mx.float32)
                out = out.at[row].add(values * weights[row, slot, None])
            else:
                values = (dense_projection(hidden, expanded[i]) if mode == "expanded"
                          else bank.project(hidden, tensors)).astype(mx.float32)
                out = out + values * weights[0, i]
        return out.astype(hidden.dtype)

    modes = ("scatter", "direct", "expanded")
    width = packed[0]["down_proj.weight"].shape[0]
    inputs = []
    for seed in range(3):
        hidden = mx.random.normal((1, width), key=mx.random.key(seed), dtype=mx.bfloat16)
        weights = mx.softmax(mx.random.normal((1, bank.top_k), key=mx.random.key(seed + 10)))
        mx.eval(hidden, weights)
        expected = project(hidden, weights, "scatter")
        for mode in modes[1:]:
            if not mx.array_equal(project(hidden, weights, mode), expected).item():
                raise RuntimeError(f"Exact expert output parity failed for {mode}")
        inputs.append((hidden, weights))

    trials = {mode: [] for mode in modes}
    for repeat in range(6):
        for mode in modes if repeat % 2 == 0 else modes[::-1]:
            guard_check(sample(), initial)
            started = perf_counter()
            for _ in range(8):
                for hidden, weights in inputs:
                    mx.eval(project(hidden, weights, mode))
            trials[mode].append((perf_counter() - started) / 24)
    final = sample()
    guard_check(final, initial)
    result = {
        "bank": bank.layer, "experts": bank.top_k, "exact_bf16_parity": True,
        "packed_bytes": sum(t.nbytes for expert in packed for t in expert.values()),
        "expanded_bytes": sum(t.nbytes for expert in expanded for t in expert.values()),
        "median_seconds_per_visit": {mode: median(times) for mode, times in trials.items()},
        "trials_seconds_per_visit": trials, "guard_initial": initial, "guard_final": final,
        "mlx_peak_mib": mx.get_peak_memory() / 2**20,
    }
    write_json(args.output, result)
    print(json.dumps(result["median_seconds_per_visit"]), flush=True)


if __name__ == "__main__":
    main()
