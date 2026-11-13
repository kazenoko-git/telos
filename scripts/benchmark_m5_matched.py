"""Matched Télos / MLX-LM Llama backbones; not a native toolkit benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import resource
import subprocess
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np

from telos.models.mlx_transformer import MLXTelosTransformer
from mlx_lm.models.llama import Model, ModelArgs


def models(width=512, layers=14, vocab=8192):
    heads = width // 64
    hidden = math.ceil(int(width * 8 / 3) / 64) * 64
    telos = MLXTelosTransformer(vocab, width, layers, heads, heads,
                              is_causal=True, precision="float32")
    peer = Model(ModelArgs(model_type="llama", hidden_size=width,
                           num_hidden_layers=layers, intermediate_size=hidden,
                           num_attention_heads=heads, num_key_value_heads=heads,
                           vocab_size=vocab, rms_norm_eps=1e-5,
                           rope_traditional=True, tie_word_embeddings=True))
    return telos, peer


def map_to_peer(weights, layers):
    """Bijective mapping, including split QKV and packed gate/up tensors."""
    w = dict(weights)
    result = {"model.embed_tokens.weight": w.pop("emb.weight"),
              "model.norm.weight": w.pop("norm.weight")}
    for i in range(layers):
        t, p = f"layers.{i}.", f"model.layers.{i}."
        for source, target in [("norm1.weight", "input_layernorm.weight"),
                               ("norm2.weight", "post_attention_layernorm.weight"),
                               ("out.weight", "self_attn.o_proj.weight"),
                               ("mlp.w3.weight", "mlp.down_proj.weight")]:
            result[p + target] = w.pop(t + source)
        for name, value in zip(("q", "k", "v"), mx.split(w.pop(t + "qkv_proj.weight"), 3)):
            result[p + f"self_attn.{name}_proj.weight"] = value
        for name, value in zip(("gate", "up"), mx.split(w.pop(t + "mlp.gate_up.weight"), 2)):
            result[p + f"mlp.{name}_proj.weight"] = value
    if w:
        raise ValueError(f"Unmapped tensors: {list(w)}")
    return result


def loss_fn(model, batch):
    logits = model(batch)[:, :-1, :].astype(mx.float32)
    return nn.losses.cross_entropy(logits, batch[:, 1:], reduction="mean")


def parity():
    mx.random.seed(42)
    a, b = models(width=64, layers=2, vocab=128)
    mapped = map_to_peer(tree_flatten(a.parameters()), 2)
    b.load_weights(list(mapped.items()), strict=True)
    batch = mx.random.randint(0, 128, (2, 16))
    logits_a, logits_b = a(batch), b(batch)
    la, ga = nn.value_and_grad(a, loss_fn)(a, batch)
    lb, gb = nn.value_and_grad(b, loss_fn)(b, batch)
    ma, mb = map_to_peer(tree_flatten(ga), 2), dict(tree_flatten(gb))
    assert ma.keys() == mb.keys()
    errors = {}
    for key in ma:
        errors[key] = float(mx.sqrt(mx.sum((ma[key] - mb[key]) ** 2)) /
                            mx.maximum(mx.sqrt(mx.sum(ma[key] ** 2)), 1e-12))
    report = {"fp32_logits_max_abs": float(mx.max(mx.abs(logits_a - logits_b))),
              "fp32_loss_abs": abs(float(la) - float(lb)),
              "gradient_max_relative_l2": max(errors.values()),
              "gradient_relative_l2_by_tensor": errors}
    report["passed"] = (report["fp32_logits_max_abs"] < 1e-4 and
                        report["fp32_loss_abs"] < 1e-5 and
                        report["gradient_max_relative_l2"] < 1e-3)
    return report


def command(*args):
    p = subprocess.run(args, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=["telos", "mlx_lm"], default="telos")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.warmup < args.updates:
        parser.error("Require 0 <= warmup < updates")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    check = parity()
    (args.output / "parity.json").write_text(json.dumps(check, indent=2) + "\n")
    if not check["passed"]:
        raise RuntimeError("FP32 equivalence gate failed")
    if args.parity_only:
        print(json.dumps(check), flush=True)
        return
    if args.data is None:
        parser.error("--data required for training")

    import mlx_lm.models.llama as llama_source
    import telos.models.mlx_components as telos_source
    metadata = {"implementation": args.implementation, "updates": args.updates,
                "warmup": args.warmup, "python": platform.python_version(),
                "os": platform.platform(), "device": mx.device_info(),
                "packages": {n: importlib.metadata.version(n) for n in
                             ["mlx", "mlx-lm", "mlx-metal", "numpy"]},
                "telos_commit": command("git", "rev-parse", "HEAD"),
                "script_sha256": source_hash(__file__),
                "source_hashes": {"telos_components": source_hash(telos_source.__file__),
                                  "mlx_lm_llama": source_hash(llama_source.__file__)},
                "power": command("pmset", "-g", "custom"),
                "battery": command("pmset", "-g", "batt"),
                "thermal": command("pmset", "-g", "therm"),
                "initial_memory": command("memory_pressure"),
                "architecture": {"vocab": 8192, "width": 512, "layers": 14,
                                 "heads": 8, "mlp_hidden": 1408, "sequence": 256},
                "microbatch": 8, "accumulation": 8,
                "limits": "Common harness; not native trainer or time-to-quality."}
    if "AC Power" not in metadata["battery"]:
        raise RuntimeError("AC power required")
    mx.set_memory_limit(int(3.2 * 1024**3))
    mx.set_cache_limit(128 * 1024**2)
    mx.random.seed(42)
    a, b = models()
    weights = tree_flatten(a.parameters())
    parameter_count = sum(v.size for _, v in weights)
    peer_count = sum(v.size for _, v in tree_flatten(b.parameters()))
    assert parameter_count == peer_count
    b.load_weights(list(map_to_peer(weights, 14).items()), strict=True)
    model = a if args.implementation == "telos" else b
    model.update(tree_map(lambda v: v.astype(mx.bfloat16), model.parameters()))
    del a, b, weights
    mx.eval(model.parameters())
    mx.clear_cache()
    metadata["parameters"] = parameter_count
    count = args.updates * 16384
    raw = np.memmap(args.data, dtype=np.uint16, mode="r")
    tokens = np.array(raw[:count + 2048], dtype=np.int32)
    assert tokens.size == count + 2048 and int(tokens.max()) < 8192
    metadata["data_prefix_sha256"] = hashlib.sha256(tokens.tobytes()).hexdigest()
    metadata["data_dtype_on_disk"] = "uint16"
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    validation = mx.array(tokens[count:].reshape(8, 256))
    initial_loss = float(loss_fn(model, validation))
    optimizer = optim.AdamW(learning_rate=4e-4, betas=[0.9, 0.999], eps=1e-8,
                            weight_decay=0.01)
    master = tree_map(lambda v: v.astype(mx.float32), model.parameters())
    value_grad = nn.value_and_grad(model, loss_fn)
    compiled = mx.compile(lambda batch: value_grad(model, batch), inputs=[model.state])
    records = []
    mx.reset_peak_memory()
    with (args.output / "steps.jsonl").open("w", buffering=1) as stream:
        for update in range(args.updates):
            begin = time.perf_counter()
            batches = mx.array(tokens[update * 16384:(update + 1) * 16384].reshape(8, 8, 256))
            total, gradients = mx.array(0.0), None
            for micro in range(8):
                loss, grad = compiled(batches[micro])
                grad = tree_map(lambda v: v.astype(mx.float32), grad)
                gradients = grad if gradients is None else tree_map(lambda a, b: a + b, gradients, grad)
                total = total + loss
                mx.eval(gradients, total)
            gradients = tree_map(lambda v: v / 8, gradients)
            norm = mx.sqrt(sum(mx.sum(v * v) for _, v in tree_flatten(gradients)))
            scale = mx.minimum(1.0, 1.0 / (norm + 1e-6))
            gradients = tree_map(lambda v: v * scale, gradients)
            master = optimizer.apply_gradients(gradients, master)
            model.update(tree_map(lambda v: v.astype(mx.bfloat16), master))
            mx.eval(model.parameters(), master, optimizer.state, total)
            elapsed = time.perf_counter() - begin
            record = {"update": update + 1, "seconds": elapsed,
                      "loss": float(total / 8), "peak_mlx_bytes": mx.get_peak_memory()}
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            if not math.isfinite(record["loss"]):
                raise RuntimeError("Non-finite loss")
            if mx.get_peak_memory() > 3.2 * 1024**3:
                raise RuntimeError("MLX allocation guard exceeded")
            if update % 8 == 0:
                pressure = command("memory_pressure")
                import re
                free = re.search(r"free percentage: (\d+)%", pressure)
                if free and int(free[1]) < 20:
                    raise RuntimeError("System memory guard exceeded")
    warmed = records[args.warmup:]
    elapsed = sum(r["seconds"] for r in warmed)
    result = {"status": "complete", "implementation": args.implementation,
              "updates": len(records), "warmed_input_tokens_per_second": len(warmed) * 16384 / elapsed,
              "warmed_target_tokens_per_second": len(warmed) * 16320 / elapsed,
              "initial_heldout_loss": initial_loss, "final_heldout_loss": float(loss_fn(model, validation)),
              "training_peak_mlx_bytes": max(r["peak_mlx_bytes"] for r in records),
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "wall_seconds": time.perf_counter() - started,
              "final_memory": command("memory_pressure"), "final_thermal": command("pmset", "-g", "therm")}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
