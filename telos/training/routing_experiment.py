"""Train and compare tiny causal prompt-routed and token-routed checkpoints."""

import argparse
import fcntl
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import numpy as np
from mlx import nn, optimizers

from telos.models.prompt_routing_mlx import RoutedTransformer
from telos.models.ssd_expert_io import LFMSSDExpertCache
from telos.models.ssd_experts import AppleMemoryGuard, tensor_ranges

from .mlx_step import build_training_step

VOCAB = 40
PROMPT = 5
ANSWER = 24
MICROBATCH = 16
ACCUMULATION = 2


def corpus():
    """Four symbol transformations with disjoint held-out seed/offset pairs."""
    rng = np.random.default_rng(2026)
    train, heldout = [], []
    for task in range(4):
        table = rng.permutation(16)
        for seed in range(16):
            for offset in range(16):
                row = [37, task+1, seed+5, offset+5, 38]
                value = (seed + offset) % 16
                for _ in range(ANSWER):
                    row.append(int(value)+21)
                    value = table[value]
                target = heldout if (seed + 3*offset + task) % 5 == 0 else train
                target.append(row)
    return np.array(train, np.int32), np.array(heldout, np.int32)


def _completion(model, ids):
    logits, balance = model(ids, PROMPT)
    return logits[:, PROMPT-1:-1], ids[:, PROMPT:], balance


def _quality(model, heldout):
    losses = correct = count = 0
    for start in range(0, len(heldout), MICROBATCH):
        logits, targets, _ = _completion(model, mx.array(heldout[start:start+MICROBATCH]))
        losses += nn.losses.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(axis=-1) == targets).sum().item()
        count += targets.size
    return {"examples": len(heldout), "tokens": count, "nll": losses/count,
            "perplexity": math.exp(losses/count), "token_accuracy": correct/count}


def _train(mode, train, heldout, *, steps, seed, path, guard):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = RoutedTransformer(routing=mode, vocab_size=VOCAB)
    optimizer = optimizers.AdamW(1e-3)
    optimizer.init(model.trainable_parameters())

    def loss_fn(model, ids):
        logits, labels, balance = _completion(model, ids)
        loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
        return loss + 0.02*balance, loss

    gradients = nn.value_and_grad(model, loss_fn)

    def microbatch(ids):
        (objective, nll), grads = gradients(model, ids)
        return objective, nll, grads

    train_step = build_training_step(model, optimizer, microbatch, [model.state], {}, grad_clip=1.0)
    started = time.perf_counter()
    curve = []
    for step in range(1, steps+1):
        guard.check()
        batches = ((mx.array(train[rng.integers(len(train), size=MICROBATCH)]),) for _ in range(ACCUMULATION))
        _, nll = train_step(batches, grad_accum=ACCUMULATION)
        if step == 1 or step % 50 == 0 or step == steps:
            curve.append({"step": step, "train_nll": float(nll.item())})
            print(f"{mode} step {step}/{steps} · train NLL {curve[-1]['train_nll']:.3f}", flush=True)
    train_step.synchronize()
    elapsed = time.perf_counter()-started
    model.eval()
    result = {"seconds": elapsed, "curve": curve, "quality": _quality(model, heldout),
              "parameters": sum(value.size for _,value in nn.utils.tree_flatten(model.parameters()))}
    model.save_weights(str(path))
    print(f"{mode} held-out NLL {result['quality']['nll']:.3f}", flush=True)
    return result


def _open(mode, checkpoint, guard, *, ssd):
    model = RoutedTransformer(routing=mode, vocab_size=VOCAB).eval()
    if not ssd:
        model.load_weights(str(checkpoint))
        return model, None, None
    ranges = tensor_ranges(checkpoint)
    experts = []
    for index, layer in enumerate(model.layers):
        prefix = f"layers.{index}.mlp."
        experts.append({name: ranges.pop(prefix+name) for name in ("gate_up","down")})
        del layer.mlp.gate_up, layer.mlp.down
    model.load_weights([(name, entry.load()) for name,entry in ranges.items()])
    mx.eval(model.parameters())
    sizes = {index: sum(entry.size // entry.shape[0] for entry in bank.values())
             for index, bank in enumerate(experts)}
    fragment = max(entry.size//entry.shape[0] for bank in experts for entry in bank.values())
    cache = LFMSSDExpertCache(sum(sizes.values()), guard, layer_sizes=sizes, fragment_size=fragment, top_k=1)

    def read(layer, expert):
        tensors = cache.get(layer, expert, experts[layer])
        return tensors["gate_up"], tensors["down"]

    return model, cache, read


def _greedy(model, cache, reader, prompt):
    started = time.perf_counter()
    logits, state = model.step(mx.array([prompt], dtype=mx.int32), read_expert=reader)
    token = int(logits.argmax().item())
    prefill = time.perf_counter()-started
    prefill_bytes = cache.bytes_read if cache is not None else 0
    remaining = []
    started = time.perf_counter()
    for _ in range(ANSWER-1):
        remaining.append(token)
        logits, state = model.step(mx.array([[token]], dtype=mx.int32), state, reader)
        token = int(logits.argmax().item())
    elapsed = time.perf_counter()-started
    tokens = [*remaining, token]
    return tokens, prefill, elapsed, prefill_bytes


def _parity(mode, path, prompts, guard):
    resident, _, _ = _open(mode, path, guard, ssd=False)
    ssd, cache, read = _open(mode, path, guard, ssd=True)
    for row in prompts:
        a = b = None
        with cache.request():
            ids = mx.array(row[None, :PROMPT])
            for _ in range(ANSWER):
                left, a = resident.step(ids, a)
                right, b = ssd.step(ids, b, read)
                if not mx.array_equal(left, right).item():
                    raise RuntimeError(f"{mode} resident and SSD logits differ")
                ids = left.argmax().reshape(1,1)
    return len(prompts)*ANSWER


def _benchmark(mode, path, examples, guard, disk_sample):
    model, cache, read = _open(mode, path, guard, ssd=True)
    counts = [Counter() for _ in model.layers]

    def record(layer, expert):
        counts[layer][expert] += 1
        return read(layer, expert)

    prefill = seconds = prefill_bytes = decode_bytes = correct = exact = 0
    disk_before = disk_sample()
    for row in examples:
        # Clear request residency so both arms start from the same empty cache.
        cache._entries.clear()
        cache.resident_bytes = 0
        before = cache.bytes_read
        with cache.request():
            tokens, first, rest, after_first = _greedy(model, cache, record, row[:PROMPT].tolist())
        prefill += first
        seconds += rest
        prefill_bytes += after_first-before
        decode_bytes += cache.bytes_read-after_first
        labels = row[PROMPT:].tolist()
        correct += sum(a == b for a, b in zip(tokens, labels, strict=True))
        exact += tokens == labels
    device_bytes = disk_sample()-disk_before
    if device_bytes < 0:
        raise RuntimeError("SSD read counter reset during the trial")
    return {"examples": len(examples), "generated_tokens": ANSWER*len(examples),
            "decode_tokens": (ANSWER-1)*len(examples), "decode_seconds": seconds,
            "decode_tok_s": (ANSWER-1)*len(examples)/seconds, "mean_ttft_s": prefill/len(examples),
            "prefill_expert_bytes": prefill_bytes, "decode_expert_bytes": decode_bytes,
            "device_request_read_bytes": device_bytes, "expert_cache_bytes": cache.byte_limit,
            "greedy_token_accuracy": correct/(ANSWER*len(examples)), "greedy_exact_sequences": exact,
            "expert_requests_by_layer": counts}


def _copy_uncached(checkpoint, target):
    with checkpoint.open("rb") as source, target.open("xb", buffering=0) as output:
        fcntl.fcntl(output.fileno(), fcntl.F_NOCACHE, 1)
        shutil.copyfileobj(source, output)


def _disk_sampler(path):
    """Whole-device IOKit read counter for the checkpoint volume on macOS."""
    import plistlib
    import subprocess

    def plist(*args):
        return plistlib.loads(subprocess.check_output(args, timeout=5))

    volume_device = subprocess.check_output(["df", "-P", str(path)], text=True, timeout=5).splitlines()[1].split()[0]
    volume = plist("diskutil", "info", "-plist", volume_device)
    disks = {plist("diskutil", "info", "-plist", store["APFSPhysicalStore"])["ParentWholeDisk"]
             for store in volume["APFSPhysicalStores"]}

    def sample():
        counts = {}
        for driver in plist("ioreg", "-a", "-l", "-r", "-c", "IOBlockStorageDriver"):
            for media in driver.get("IORegistryEntryChildren", []):
                if media.get("Whole") and media.get("BSD Name") in disks:
                    disk = media["BSD Name"]
                    if disk in counts:
                        raise RuntimeError("Duplicate SSD counter source")
                    counts[disk] = driver["Statistics"]["Bytes (Read)"]
        if set(counts) != disks or not counts:
            raise RuntimeError("Could not locate the checkpoint drive's read counter")
        return sum(counts.values())

    return sample


def main(argv=None):
    parser = argparse.ArgumentParser(prog="telos pretrain routing", description=__doc__)
    parser.add_argument("--routing", choices=("prompt","token","both"), default="both")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.seed < 0:
        parser.error("--steps must be positive and --seed non-negative")
    args.output.mkdir(parents=True, exist_ok=False)
    modes = ("prompt","token") if args.routing == "both" else (args.routing,)
    train, heldout = corpus()
    sample_rng = np.random.default_rng(2027)
    domains = [heldout[heldout[:, 1] == task+1] for task in range(4)]
    examples = np.concatenate([rows[sample_rng.choice(len(rows), 8, replace=False)] for rows in domains])
    guard = AppleMemoryGuard()
    disk_sample = _disk_sampler(args.output)
    checkpoints = {mode: args.output/f"{mode}.safetensors" for mode in modes}
    results = {"mlx": mx.__version__, "seed": args.seed, "steps": args.steps, "train_examples": len(train),
               "microbatch": MICROBATCH, "accumulation": ACCUMULATION, "effective_batch": MICROBATCH*ACCUMULATION,
               "prompt_tokens": PROMPT, "answer_tokens": ANSWER, "models": {}, "trials": []}

    def save():
        (args.output/"results.json").write_text(json.dumps(results, indent=2)+"\n")

    for mode in modes:
        results["models"][mode] = _train(mode, train, heldout, steps=args.steps, seed=args.seed,
                                         path=checkpoints[mode], guard=guard)
        save()
    for mode in modes:
        results["models"][mode]["exact_parity_logits"] = _parity(mode, checkpoints[mode], examples[::8], guard)
        save()
    before = disk_sample()
    started = time.perf_counter()
    time.sleep(1)
    results["idle"] = {"seconds": time.perf_counter()-started, "device_read_bytes": disk_sample()-before}
    with TemporaryDirectory(dir=args.output) as scratch:
        # Benchmark different files than the resident parity pass opened.
        for run, mode in enumerate((*modes, *reversed(modes))):
            path = Path(scratch)/f"{mode}-{run}.safetensors"
            _copy_uncached(checkpoints[mode], path)
            trial = _benchmark(mode, path, examples, guard, disk_sample)
            results["trials"].append({"mode": mode, **trial})
            save()
            print(f"{mode} SSD decode {trial['decode_tok_s']:.1f} tok/s", flush=True)
