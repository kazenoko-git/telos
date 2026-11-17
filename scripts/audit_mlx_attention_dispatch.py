"""Export the installed MLX SDPA training graphs without changing dispatch policy."""
import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MLX_ENABLE_TF32", "0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.set_memory_limit(int(3.2 * 1024**3))
    mx.set_cache_limit(256 * 1024**2)
    mx.random.seed(42)
    results = []
    for length in (256, 2048):
        for force in (False, True):
            inputs = tuple(mx.random.normal((1, 8, length, 64)).astype(mx.bfloat16) for _ in range(3))
            cotangent = mx.ones_like(inputs[0])
            def attention(q, k, v):
                return mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=64**-.5, mask="causal", force_fused=force)
            graph = mx.vjp(attention, inputs, (cotangent,))
            arrays = [v for _, v in tree_flatten(graph)]
            path = args.output / f"t{length}-force{int(force)}.dot"
            mx.export_to_dot(str(path), *arrays)
            text = path.read_text()
            mx.eval(*arrays)
            results.append({"length": length, "force_fused": force,
                            "native_vjp_nodes": len(re.findall(r'label\s*=\s*"ScaledDotProductAttentionVJP"', text)),
                            "fused_forward_nodes": len(re.findall(r'label\s*=\s*"ScaledDotProductAttention"', text)),
                            "finite": all(mx.all(mx.isfinite(v)).item() for v in arrays),
                            "graph": path.name})
            del graph, arrays, inputs, cotangent
            mx.clear_cache()
    (args.output / "audit.json").write_text(json.dumps({
        "mlx_version": mx.__version__, "mlx_module": mx.__file__, "device": mx.device_info(),
        "attention_backward_env": os.environ.get("MLX_SDPA_VJP_FALLBACK"),
        "results": results,
        "limitations": "Graph dispatch audit for four shapes/options, not throughput or convergence."}, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
