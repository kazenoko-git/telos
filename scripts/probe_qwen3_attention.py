#!/usr/bin/env python3
"""Frozen Qwen3 attention experiment. Default: prepare data; NEVER load weights.

No prompts, FIM tokens, suffix rearrangement, optimizer, or candidate generation.
Only --run imports MLX and loads a model. --download-weights is also explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/qwen3_attention_probe_2026-10-18"
CACHE = ROOT / "scratch/qwen_attention_probe/hf_cache"
MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODES = ("causal", "full", "future_bias_4", "future_bias_2",
         "last_1_full", "last_4_full", "query_only")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def make_case(tokenizer, ident, text, start, end, group, pair=None):
    enc = tokenizer.encode(text, add_special_tokens=False)
    hits = [i for i, (a, b) in enumerate(enc.offsets) if a < end and b > start]
    if len(hits) != 1:
        raise ValueError(f"{ident}: hole is not one token: {hits}")
    i = hits[0]
    a, b = enc.offsets[i]
    if (a, b) != (start, end) or i == 0 or i >= len(enc.ids) - 3:
        raise ValueError(f"{ident}: invalid hole boundary/context {(a,b,start,end,i)}")
    return dict(id=ident, group=group, pair=pair, text=text, ids=enc.ids,
                hole=i, target=enc.ids[i], target_text=text[a:b])


def prepare(tokenizer, natural_count=64):
    cases = []
    # Expected answer changes only with the right-hand assertions. No answer
    # operator is repeated in the suffix. Both completed originals are executed.
    operands = [(a, b) for a in range(3, 8) for b in range(1, a) if a+b < 10][:12]
    for k, (a, b) in enumerate(operands):
        prefix = f"def calc_{k}(a, b):\n    return a"
        for op, result in [("+", a+b), ("-", a-b)]:
            text = prefix + f" {op} b\n\nassert calc_{k}({a}, {b}) == {result}\n"
            exec(compile(text, "<generated-arithmetic-probe>", "exec"), {})
            cases.append(make_case(tokenizer, f"arithmetic_{k}_{op}", text,
                         len(prefix), len(prefix)+2, "paired_arithmetic", f"arith_{k}"))
    for k in range(12):
        a, b = k+1, k+3
        prefix = f"def compare_{k}(a, b):\n    return a"
        for op, result in [("<", True), (">", False)]:
            text = prefix + f" {op} b\n\nassert compare_{k}({a}, {b}) is {result}\n"
            exec(compile(text, "<generated-comparison-probe>", "exec"), {})
            cases.append(make_case(tokenizer, f"comparison_{k}_{op}", text,
                         len(prefix), len(prefix)+2, "paired_comparison", f"compare_{k}"))
    pairs = {}
    for c in cases:
        pairs.setdefault(c["pair"], []).append(c)
    for pair in pairs.values():
        x, y = pair
        i = x["hole"]
        assert i == y["hole"] and x["ids"][:i] == y["ids"][:i]
        assert len(x["ids"]) == len(y["ids"]), "Counterfactual length confound"
        assert x["target"] != y["target"] and x["ids"][i+1:] != y["ids"][i+1:]
        x["alternative"] = y["target"]
        y["alternative"] = x["target"]

    source = ROOT / "evals/benchmarks/private_unseen_suite.json"
    # This is a local reconstruction sample, NOT a guarantee of exclusion from
    # Qwen pretraining. Never execute corpus code. One token per source task.
    rng = random.Random(20261018)
    added = 0
    for item in json.loads(source.read_text()):
        prefix = item.get("prompt_base", item["prompt"])
        text = prefix + item["ground_truth_solution"] + "\n"
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc.ids) > 512:
            continue
        eligible = []
        for i, (a, b) in enumerate(enc.offsets):
            word = text[a:b].strip()
            if a >= len(prefix) and 0 < i < len(enc.ids)-8 and word and (
                word.isidentifier() or word in {"+", "-", "*", "%", "<", ">", "=="}
            ):
                eligible.append((a, b))
        if not eligible:
            continue
        a, b = rng.choice(eligible)
        cases.append(make_case(tokenizer, "local_"+item["id"], text, a, b, "local_code"))
        added += 1
        if added == natural_count:
            break
    return cases, source


def attention_mask(length, hole, mode, layer, layers):
    """Pure CPU construction. Every mode excludes the unknown token as a key.

    The true token is also replaced BEFORE embedding. Original sequence order
    and RoPE positions are preserved. No cross-example attention or KV caches.
    """
    import numpy as np
    q = np.arange(length)[:, None]
    k = np.arange(length)[None, :]
    future = k > q
    mask = np.where(future, -np.inf, 0.0).astype(np.float32)
    active = mode == "full" or (
        mode.startswith("last_") and layer >= layers-int(mode.split("_")[1]))
    if active:
        mask[:] = 0
    elif mode.startswith("future_bias_"):
        mask[future] = -float(mode.rsplit("_", 1)[1])
    elif mode == "query_only":
        mask[[hole-1, hole], :] = 0
    elif mode != "causal" and not mode.startswith("last_"):
        raise ValueError(mode)
    mask[:, hole] = -np.inf
    return mask


def validate_masks():
    import numpy as np
    for mode in MODES:
        m = attention_mask(8, 3, mode, 35, 36)
        assert np.isneginf(m[:, 3]).all()
        assert np.isfinite(m).any(axis=1).all()
    causal = attention_mask(8, 3, "causal", 0, 36)
    assert np.isneginf(causal[2, 4])
    assert attention_mask(8, 3, "query_only", 0, 36)[2, 4] == 0
    assert np.isneginf(attention_mask(8, 3, "query_only", 0, 36)[1, 4])
    assert np.isneginf(attention_mask(8, 3, "last_4_full", 31, 36)[2, 4])
    assert attention_mask(8, 3, "last_4_full", 32, 36)[2, 4] == 0
    assert attention_mask(8, 3, "future_bias_4", 0, 36)[2, 4] == -4


def summarize(rows):
    result = {}
    keys = sorted({(r["mode"], r["placeholder"], r["readout"], r["group"]) for r in rows})
    for key in keys:
        subset = [r for r in rows if tuple(r[k] for k in
                  ("mode", "placeholder", "readout", "group")) == key]
        pairs = {}
        for r in subset:
            if r["pair"]:
                pairs.setdefault(r["pair"], []).append(r)
        complete = [p for p in pairs.values() if len(p) == 2]
        entry = dict(n=len(subset), top1=statistics.mean(r["correct"] for r in subset),
                     top5=statistics.mean(r["rank"] <= 5 for r in subset),
                     mean_nll=statistics.mean(-r["target_logp"] for r in subset))
        if complete:
            entry.update(pairs=len(complete),
                both_top1=sum(all(r["correct"] for r in p) for p in complete),
                both_prefer_required=sum(all(r["candidate_margin"] > 0 for r in p) for p in complete),
                prediction_changes=sum(p[0]["prediction"] != p[1]["prediction"] for p in complete),
                # A causal paired baseline should have zero distribution change.
                max_causal_control_delta=max(abs(p[0]["candidate_a_logp"]-p[1]["candidate_a_logp"])
                                             for p in complete))
        result["/".join(key)] = entry
    return result


def run(args, cases, tokenizer, model_dir):
    # Intentionally deferred: preparing the experiment must not initialize Metal.
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model
    import numpy as np
    model, config = load_model(model_dir, lazy=False, strict=True)
    assert config["model_type"] == "qwen3" and "quantization" not in config
    model.eval()
    model.freeze()
    assert not tree_flatten(model.trainable_parameters())
    underscore = tokenizer.encode("_", add_special_tokens=False).ids
    assert len(underscore) == 1
    rows = []
    start = time.monotonic()
    results_path = args.output / "results.jsonl"
    if results_path.exists():
        raise FileExistsError(f"Refusing to overwrite {results_path}; use a new --output")

    def embeddings(case, placeholder):
        i = case["hole"]
        ids = list(case["ids"])
        ids[i] = underscore[0]  # True target NEVER enters the model input.
        h = model.model.embed_tokens(mx.array([ids]))
        if placeholder == "zero":
            keep = mx.array([j != i for j in range(len(ids))])[None, :, None]
            h = mx.where(keep, h, mx.zeros_like(h))
        return ids, h

    def forward(case, placeholder, mode):
        _, h = embeddings(case, placeholder)
        for index, layer in enumerate(model.model.layers):
            mask = mx.array(attention_mask(len(case["ids"]), case["hole"], mode,
                                          index, len(model.model.layers)), dtype=h.dtype)
            h = layer(h, mask=mask, cache=None)
        selected = model.model.norm(h[:, [case["hole"]-1, case["hole"]], :])
        logits = (model.model.embed_tokens.as_linear(selected) if config["tie_word_embeddings"]
                  else model.lm_head(selected))
        mx.eval(logits)
        return logits

    # Necessary implementation check: custom causal forward must reproduce the
    # stock causal model at the shifted readout (before the hidden position).
    case = cases[0]
    ids, emb = embeddings(case, "zero")
    native = model(mx.array([ids]), input_embeddings=emb)[:, case["hole"]-1, :]
    custom = forward(case, "zero", "causal")[:, 0, :]
    mx.eval(native, custom)
    delta = float(mx.max(mx.abs(native.astype(mx.float32)-custom.astype(mx.float32))).item())
    if delta > 0.05 or int(mx.argmax(native).item()) != int(mx.argmax(custom).item()):
        raise AssertionError(f"Native/custom causal mismatch: {delta}")
    dump(args.output / "runtime.json", dict(model=MODEL, revision=REVISION,
         config=config, native_causal_max_logit_delta=delta,
         script_sha256=sha(__file__), packages={k: importlib.metadata.version(k)
         for k in ("mlx", "mlx-lm", "tokenizers", "numpy")},
         frozen=True, training=False, fim=False))
    del native, custom, emb
    with results_path.open("x") as output:
        for case in cases:
            for placeholder in ("zero", "underscore"):
                for mode in args.modes:
                    logits = np.asarray(forward(case, placeholder, mode).astype(mx.float32))[0]
                    if not np.isfinite(logits).all():
                        raise AssertionError(f"Non-finite logits: {case['id']}/{mode}/{placeholder}")
                    for rindex, readout in enumerate(("shifted", "same_position")):
                        v = logits[rindex]
                        logp = v - (float(v.max()) + math.log(float(np.exp(v-v.max()).sum())))
                        target = case["target"]
                        pred = int(v.argmax())
                        row = {k: case.get(k) for k in ("id", "group", "pair", "target", "target_text")}
                        row.update(mode=mode, placeholder=placeholder, readout=readout,
                             prediction=pred, prediction_text=tokenizer.decode([pred]),
                             target_logp=float(logp[target]), rank=int((v > v[target]).sum())+1,
                             correct=pred == target)
                        if case.get("alternative") is not None:
                            alt = case["alternative"]
                            row.update(candidate_margin=float(v[target]-v[alt]),
                                       candidate_a_logp=float(logp[min(target, alt)]))
                        output.write(json.dumps(row)+"\n")
                        rows.append(row)
                    output.flush()
            print(f"{case['id']}: {len(rows)} rows, {time.monotonic()-start:.1f}s", flush=True)
            dump(args.output / "summary.json", summarize(rows))
    for key, summary in summarize(rows).items():
        if key.startswith("causal/") and summary.get("max_causal_control_delta", 0) > 1e-4:
            raise AssertionError(f"Causal suffix-independence control failed: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--model-dir", type=Path, default=CACHE / "models--Qwen--Qwen3-4B/snapshots" / REVISION)
    parser.add_argument("--run", action="store_true", help="Explicitly load and run Qwen")
    parser.add_argument("--download-weights", action="store_true", help="Only with --run; download pinned weights")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    if args.download_weights and not args.run:
        parser.error("--download-weights requires --run")
    if (args.output / "results.jsonl").exists():
        parser.error("Results exist; choose a new --output to preserve the previous experiment")
    from tokenizers import Tokenizer
    tokenizer_path = args.model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        parser.error(f"Local tokenizer missing: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    validate_masks()
    cases, source = prepare(tokenizer)
    if args.max_cases is not None:
        if args.max_cases <= 0:
            parser.error("--max-cases must be positive")
        cases = cases[:args.max_cases]
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / "cases.json", cases)
    dump(args.output / "prepared.json", dict(status="prepared_only", model=MODEL, revision=REVISION,
         cases=len(cases), groups={g: sum(c["group"] == g for c in cases) for g in sorted({c["group"] for c in cases})},
         tokenizer_sha256=sha(tokenizer_path), corpus_path=str(source), corpus_sha256=sha(source),
         script_sha256=sha(__file__), modes=args.modes, model_loaded=False,
         mask_tests="passed", generated_code_assertions="passed",
         placeholder_tests=["zero_embedding", "underscore_embedding"],
         readouts=["shifted_primary", "same_position_diagnostic"]))
    print(f"Prepared {len(cases)} cases at {args.output}; no model loaded.", flush=True)
    if args.run:
        if (args.output / "results.jsonl").exists():
            parser.error("Results exist; choose a new --output")
        model_dir = args.model_dir
        if args.download_weights:
            from huggingface_hub import snapshot_download
            model_dir = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=str(CACHE),
                             allow_patterns=["*.json", "*.safetensors", "merges.txt"]))
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        missing = [f for f in set(index["weight_map"].values()) if not (model_dir / f).exists()]
        if missing:
            parser.error(f"Missing weights: {missing}. Download later with --run --download-weights")
        run(args, cases, tokenizer, model_dir)


if __name__ == "__main__":
    main()
