"""Evaluate a local Inkling W=16 adapter with guarded, sequential SSD workers."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import plistlib
import signal
import subprocess
import sys
import time

from guarded_serve_benchmark import memory_sample, read_command, stop_process

BUDGET_MIB = 2048
CONTEXT = 512
WINDOW = 16
DECODE_TOKENS = 16
PERF_ROWS = 3
VARIANTS = ("stock", "untrained_w16", "trained_w16")
PINNING_VARIANTS = ("trained_w16", "pinned_w16")
PROFILING_VARIANTS = ("pinned_w16", "profiled_w16")
FUSED_NVFP4_VARIANTS = ("pinned_w16", "fused_nvfp4_w16")
FUSED_ATTENTION_VARIANTS = ("fused_nvfp4_w16", "profiled_fused_attention_w16")
GROUPED_QUERY_VARIANTS = ("fused_nvfp4_w16", "grouped_query_w16")
FUSED_MASK_VARIANTS = ("fused_nvfp4_w16", "fused_mask_w16")
WORKERS = (*VARIANTS, *PROFILING_VARIANTS, *FUSED_ATTENTION_VARIANTS, "grouped_query_w16", "fused_mask_w16")


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def disk_counters():
    drivers = plistlib.loads(subprocess.check_output(
        ["ioreg", "-a", "-l", "-r", "-c", "IOBlockStorageDriver"], timeout=3,
    ))
    for driver in drivers:
        if any(m.get("Whole") and m.get("BSD Name") == "disk0" for m in driver.get("IORegistryEntryChildren", ())):
            stats = driver["Statistics"]
            return {"read_bytes": stats["Bytes (Read)"], "write_bytes": stats["Bytes (Write)"]}
    raise RuntimeError("Cannot locate physical disk0 counters")


def disk_delta(before, after):
    return {key: after[key] - before[key] for key in before}


def benchmark(generator, prompt, *, token_cap, per_step=False, split_decode=False, split_attention=False,
              attention_stride=1, attention_components=False):
    from telos.models.ssd_profile import SSDGenerationProfile

    cache = generator.model._expert_cache
    cache._entries.clear()
    cache.resident_bytes = 0
    ids = generator.tokenizer.encode(prompt, add_special_tokens=False).ids
    if generator.add_bos and ids[0] != generator.bos_id:
        ids.insert(0, generator.bos_id)
    profile = SSDGenerationProfile(
        split_decode=split_decode, split_attention=split_attention, attention_stride=attention_stride,
        attention_components=attention_components,
    )
    initial = disk_counters()
    start = time.perf_counter()
    stream = generator._mlx_tokens(ids, token_cap, 0.0, 0, 1.0, None, "dynamic", ssd_profile=profile)
    with closing(stream):
        first = next(stream)
        ttft = time.perf_counter() - start
        after_prefill = disk_counters()
        start_decode = time.perf_counter()
        tokens = [first]
        if first not in generator.stop_ids:
            for token in stream:
                tokens.append(token)
                if token in generator.stop_ids:
                    break
        decode_seconds = time.perf_counter() - start_decode
    after_decode = disk_counters()
    result = {
        "prompt_tokens": len(ids), "token_ids": tokens,
        "text": generator.tokenizer.decode(tokens, skip_special_tokens=True),
        "ttft_seconds": ttft, "decode_seconds": decode_seconds,
        "decode_tok_s": (len(tokens) - 1) / decode_seconds if len(tokens) > 1 else None,
        "physical_prefill_io": disk_delta(initial, after_prefill),
        "physical_decode_io": disk_delta(after_prefill, after_decode),
        "profile": profile.report()["phases"],
    }
    if per_step:
        result["steps"] = profile.steps
    return result


def worker(args):
    import mlx.core as mx

    from telos.inference import TextGenerator
    from telos.models import inkling2_mlx
    from telos.posttrain.trainer import _encode, _special_token_ids

    mx.set_memory_limit(args.memory_budget_mib * 2**20)
    mx.set_cache_limit(128 * 2**20)
    plan_inkling = inkling2_mlx.plan_inkling
    pinned = args.worker not in VARIANTS
    decode_only = args.variants != VARIANTS

    def require_ssd(*plan_args):
        context, limit = plan_inkling(*plan_args)
        if limit is None:
            raise RuntimeError("SSD-only evaluation rejected a resident weight plan")
        if pinned:
            layout = plan_args[1]
            sizes = inkling2_mlx._layer_sizes(layout)
            if sum(sizes[b.layer] * b.top_k for b in layout.expert_banks) > limit:
                raise RuntimeError("The admitted expert cache cannot fit every selected window expert")
        return context, limit

    inkling2_mlx.plan_inkling = require_ssd
    rows = [json.loads(line) for line in args.validation.read_text().splitlines() if line.strip()]
    stock = args.worker == "stock"
    generator = TextGenerator.from_checkpoint(
        args.adapter if pinned or args.worker == "trained_w16" else args.base, device="mlx",
        context_length=CONTEXT, memory_budget_mib=args.memory_budget_mib, window_size=None if stock else WINDOW,
    )
    model = generator.model
    pinned_bytes = model.pin_window_experts() if pinned else 0
    model.fused_nvfp4_decode = pinned and args.worker not in PROFILING_VARIANTS
    model.fused_mask_decode = args.worker == "fused_mask_w16"
    grouped_query_banks = model.prepare_grouped_query_decode() if args.worker == "grouped_query_w16" else 0
    if args.worker == "grouped_query_w16" and grouped_query_banks == 0:
        raise RuntimeError("No attention banks could group their query projections")
    special = _special_token_ids(generator.tokenizer, args.base, json.loads((args.base / "config.json").read_text()))
    quality = []
    for index, row in enumerate([] if decode_only else rows):
        tokens, labels, _ = _encode(generator.tokenizer, row["prompt"], row["response"], 64, None, special_ids=special)
        start = next(i for i, label in enumerate(labels) if label != -100)
        ids = mx.array([tokens], dtype=mx.int32)
        with model._expert_cache.request():
            if not stock:
                model._window_controller.reset_sequence_student(WINDOW, [start])
            try:
                hidden = model.hidden_states(ids)
            finally:
                model._window_controller.clear_sequence()
            logits = model.logits_from_hidden(hidden[0, start - 1:-1]).astype(mx.float32)
            mx.eval(logits)
        target = mx.array(labels[start:])[:, None]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        ce = -mx.take_along_axis(logprobs, target, axis=-1).mean()
        teacher_file = args.output / f"teacher-{index:02}.safetensors"
        if stock:
            mx.save_safetensors(str(teacher_file), {"logits": logits})
            kl, match = 0.0, 1.0
        else:
            teacher = mx.load(str(teacher_file))["logits"]
            teacher_logprobs = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
            kl = (mx.exp(teacher_logprobs) * (teacher_logprobs - logprobs)).sum(axis=-1).mean().item()
            match = (logits.argmax(axis=-1) == teacher.argmax(axis=-1)).mean().item()
        quality.append({"response_tokens": len(labels) - start, "ce": ce.item(), "teacher_kl": kl,
                        "teacher_top1_match": match})
        write_json(args.output / f"{args.worker}-progress.json", quality)
        print(f"{args.worker}: held-out {index + 1}/{len(rows)}", flush=True)
    speed = []
    for index, row in enumerate(rows[:PERF_ROWS]):
        speed.append(benchmark(
            generator, row["prompt"], token_cap=args.max_decode_tokens, per_step=decode_only,
            split_decode=args.worker == "profiled_w16",
            split_attention=args.worker == "profiled_fused_attention_w16",
            attention_stride=args.attention_profile_stride,
            attention_components=args.attention_components and args.worker == "profiled_fused_attention_w16",
        ))
        print(f"{args.worker}: decode {index + 1}/{min(PERF_ROWS, len(rows))}", flush=True)
    write_json(args.output / f"{args.worker}.json", {
        "quality": quality, "decode": speed, "mlx_version": mx.__version__,
        "mlx_peak_mib": mx.get_peak_memory() / 2**20, "expert_cache_mib": model._expert_cache.byte_limit / 2**20,
        "pinned_expert_bytes": pinned_bytes,
        "fused_nvfp4_decode": model.fused_nvfp4_decode,
        "fused_mask_decode": model.fused_mask_decode,
        "grouped_query_banks": grouped_query_banks,
    })


def sample():
    from telos.models.ssd_experts import _swapout_bytes

    free, swap, free_swap = memory_sample()
    return {"free_percent": free, "swap_used_mib": swap, "free_swap_mib": free_swap,
            "swapout_bytes": _swapout_bytes()}


def guard_check(current, initial, *, preflight=False):
    if current["swap_used_mib"] > initial["swap_used_mib"] or current["swapout_bytes"] > initial["swapout_bytes"]:
        raise RuntimeError("Guard stopped after swap growth")
    if current["free_percent"] < (45 if preflight else 35):
        raise RuntimeError("Guard stopped for low memory headroom")
    if current["free_swap_mib"] is not None and current["free_swap_mib"] < 768:
        raise RuntimeError("Guard stopped for low free swap")


def supervise(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Use an empty output directory to preserve earlier evaluation artifacts")
    base_config = json.loads((args.base / "config.json").read_text())
    adapter_config = json.loads((args.adapter / "config.json").read_text())
    posttrain = adapter_config.get("posttrain", {})
    source = Path(posttrain.get("source_checkpoint", "")).resolve()
    if base_config.get("model_type") != "inkling2_loop" or adapter_config.get("model_type") != "inkling2_loop":
        raise ValueError("The base and adapter must both identify Inkling 2 before loading weights")
    if source != (args.base / "model.safetensors.index.json").resolve() or posttrain.get("window_size") != WINDOW:
        raise ValueError("The W=16 adapter must reference the supplied stock checkpoint")
    initial = sample()
    guard_check(initial, initial, preflight=True)
    with args.validation.open("rb") as validation, (args.adapter / "model.adapter.safetensors").open("rb") as adapter:
        validation_hash = hashlib.file_digest(validation, "sha256").hexdigest()
        adapter_hash = hashlib.file_digest(adapter, "sha256").hexdigest()
    report = {
        "commit": read_command("git", "rev-parse", "HEAD").strip(),
        "power": read_command("pmset", "-g"), "budget_mib": args.memory_budget_mib, "context": CONTEXT,
        "window": WINDOW, "decode_token_cap": args.max_decode_tokens,
        "attention_profile_stride": args.attention_profile_stride,
        "attention_components": args.attention_components,
        "reverse_order": args.reverse_order,
        "validation_sha256": validation_hash, "adapter_sha256": adapter_hash,
        "variants": {}, "guards": {},
    }
    variants = args.variants[::-1] if args.reverse_order else args.variants
    for variant in variants:
        start = sample()
        guard_check(start, initial, preflight=True)
        print(f"Starting guarded {variant} evaluation at {start['free_percent']}% free", flush=True)
        process = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--worker", variant],
                                   env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
        record = {"initial": start, "min_free_percent": start["free_percent"], "peak_rss_mib": 0.0, "samples": 0}
        report["guards"][variant] = record
        try:
            deadline = time.monotonic() + 600
            while process.poll() is None:
                current = sample()
                record["final"] = current
                record["samples"] += 1
                record["min_free_percent"] = min(record["min_free_percent"], current["free_percent"])
                guard_check(current, initial)
                rss = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True, timeout=2)
                if rss.stdout.strip():
                    record["peak_rss_mib"] = max(record["peak_rss_mib"], int(rss.stdout.strip()) / 1024)
                if record["peak_rss_mib"] >= args.memory_budget_mib or time.monotonic() >= deadline:
                    raise RuntimeError("Worker reached its RSS or time limit")
                time.sleep(0.10)
            record["final"] = sample()
            guard_check(record["final"], initial)
            if process.returncode != 0:
                raise RuntimeError(f"{variant} worker exited with status {process.returncode}")
            report["variants"][variant] = json.loads((args.output / f"{variant}.json").read_text())
        except (Exception, KeyboardInterrupt) as exc:
            record["stop_reason"] = str(exc)
            raise
        finally:
            stop_process(process)
            write_json(args.output / "comparison.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "adapter", "validation", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--memory-budget-mib", type=int, choices=(2048, 3072), default=BUDGET_MIB)
    parser.add_argument("--max-decode-tokens", type=int, choices=range(1, CONTEXT), default=DECODE_TOKENS, metavar="N")
    mode = parser.add_mutually_exclusive_group()
    for flag, variants, help_text in (
        ("--compare-pinning", PINNING_VARIANTS, "Compare the trained adapter with and without bank pinning"),
        ("--profile-pinned", PROFILING_VARIANTS, "Measure pinned decode with and without section timing fences"),
        ("--compare-fused-nvfp4", FUSED_NVFP4_VARIANTS, "Compare pinned decode with the NVFP4 Metal kernel"),
        ("--profile-fused-attention", FUSED_ATTENTION_VARIANTS, "Split fused pinned decode timing into attention and routing"),
        ("--compare-grouped-query", GROUPED_QUERY_VARIANTS, "Compare separate and grouped query-input projections"),
        ("--compare-fused-mask", FUSED_MASK_VARIANTS, "Compare relative attention mask construction with and without Metal fusion"),
    ):
        mode.add_argument(flag, dest="variants", action="store_const", const=variants, help=help_text)
    parser.set_defaults(variants=VARIANTS)
    parser.add_argument("--attention-profile-stride", type=int, choices=(1, 4, 12), default=1,
                        help="Under attention profiling, time one layer visit per N and rotate the sample each token")
    parser.add_argument("--attention-components", action="store_true", help="Split sampled attention into its constituent operations")
    parser.add_argument("--reverse-order", action="store_true", help="Run the decode comparison in reverse order")
    parser.add_argument("--worker", choices=WORKERS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reverse_order and args.variants == VARIANTS:
        parser.error("--reverse-order requires a decode comparison")
    if args.worker:
        worker(args)
    else:
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        supervise(args)


if __name__ == "__main__":
    main()
