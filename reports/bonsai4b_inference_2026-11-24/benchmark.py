"""Paired raw-prompt decode test. Run one engine per process."""

import argparse
import contextlib
import json
from pathlib import Path
import resource
import subprocess
import threading
import time

import psutil

MODEL_KEY = "ternary-bonsai-4b-mlx"
MODEL_DIR = Path.home() / ".lmstudio/models/prism-ml/Ternary-Bonsai-4B-mlx-2bit"
LMS = Path.home() / ".lmstudio/bin/lms"
IDENTIFIER = "telos-bonsai4b"
CONTEXT = 1024
BUDGET = 192
MIB = 2**20
WARMUP = "Education changes lives because"
PROMPT = """Workshop notes on writing useful tests for small Python programs

The intended reader can write functions and read error messages, but has never
designed a test suite. Examples should use only the standard library. Prefer
one clear function per example, with names that explain the inputs and outputs.
A good example includes an ordinary input, an empty input, and a boundary case
where a common implementation would produce an incorrect result. Use short
explanations and complete examples that a beginner could type and run.

For instance, a function that joins names should handle a list containing one
name, a list containing several names, and an empty list. Checking that a test
fails against the wrong implementation is as important as checking that it
passes against the right one. A test that passes regardless of the result is
an example of wasted work.

The following handout contains twenty numbered examples. Each entry introduces
a distinct principle, explains the mistake it prevents, and supplies a runnable
Python example. The handout begins with the first principle.

1."""


def direct():
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    from telos.eval.adapters import load_adapter

    started = time.perf_counter()
    adapter = load_adapter(str(MODEL_DIR), backend="mlx_lm")
    mx.eval(adapter.model.parameters())
    load = time.perf_counter() - started
    ids = adapter.tokenizer.encode(PROMPT, add_special_tokens=False)
    assert len(ids) + BUDGET <= CONTEXT

    adapter.generate(WARMUP, max_new_tokens=8, temperature=0)
    mx.synchronize()
    mx.reset_peak_memory()
    active = mx.get_active_memory()
    started = time.perf_counter()
    parts = []
    times = []
    for piece in mlx_lm.stream_generate(
        adapter.model, adapter.tokenizer, ids, max_tokens=BUDGET, max_kv_size=CONTEXT,
        sampler=make_sampler(temp=0),
    ):
        parts.append(piece)
        now = time.perf_counter()
        if piece.finish_reason != "stop":
            times.append(now)
    elapsed = now - started
    return {
        "engine": "telos_mlx_lm", "prompt_ids": ids,
        "checkpoint": str(MODEL_DIR), "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
        "context": CONTEXT, "load_seconds": load, "generated_tokens": len(times),
        "token_arrival_seconds": [t - started for t in times],
        "client_ttft_seconds": times[0] - started, "wall_seconds": elapsed,
        "engine_ttft_seconds": parts[-1].prompt_tokens / parts[-1].prompt_tps,
        "engine_decode_tok_s": parts[-1].generation_tps,
        "stop_reason": parts[-1].finish_reason,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MIB,
        "mlx_active_before_mib": active / MIB, "mlx_peak_mib": mx.get_peak_memory() / MIB,
        "text": "".join(piece.text for piece in parts),
    }


def lm_rss_mib():
    workers = {}
    for process in psutil.process_iter(["name"]):
        if process.info["name"] == "LM Studio":
            with contextlib.suppress(psutil.Error):
                workers[process.pid] = process
                workers.update({p.pid: p for p in process.children(recursive=True)})
    resident = 0
    for process in workers.values():
        with contextlib.suppress(psutil.Error):
            resident += process.memory_info().rss
    return resident / MIB


def lm_studio():
    import lmstudio

    def command(*args):
        subprocess.run([str(LMS), *args], check=True, capture_output=True, timeout=60)

    baseline = lm_rss_mib()
    peak = [baseline]
    end = threading.Event()

    def sample():
        while not end.wait(0.12):
            peak.append(lm_rss_mib())

    observer = threading.Thread(target=sample, daemon=True)
    observer.start()
    try:
        started = time.perf_counter()
        command("load", MODEL_KEY, "--identifier", IDENTIFIER, "--context-length", str(CONTEXT),
                "--parallel", "1", "--no-speculative-draft-mtp", "--ttl", "180", "--yes")
        load = time.perf_counter() - started
        settings = {
            "temperature": 0, "topKSampling": 0, "topPSampling": 1.0, "minPSampling": 0.0,
            "repeatPenalty": 1.0, "stopStrings": [], "maxTokens": BUDGET,
            "contextOverflowPolicy": "stopAtLimit",
        }
        with lmstudio.Client("127.0.0.1:12489") as client:
            model = client.llm.model(IDENTIFIER)
            assert model.get_context_length() == CONTEXT
            ids = model.tokenize(PROMPT)
            assert len(ids) + BUDGET <= CONTEXT
            model.complete(WARMUP, config={**settings, "maxTokens": 8})

            started = time.perf_counter()
            stream = model.complete_stream(PROMPT, config=settings)
            times = []
            drafted = 0
            for fragment in stream:
                now = time.perf_counter()
                times.extend([now] * fragment.tokens_count)
                drafted += fragment.tokens_count if fragment.contains_drafted else 0
            result = stream.result()
            stats = result.stats.to_dict()
            assert stats["promptTokensCount"] == len(ids)
            assert drafted == 0
            return {
                "engine": "lm_studio", "prompt_ids": ids, "context": CONTEXT,
                "load_seconds": load, "generated_tokens": len(times),
                "token_arrival_seconds": [t - started for t in times],
                "client_ttft_seconds": times[0] - started, "wall_seconds": time.perf_counter() - started,
                "engine_ttft_seconds": stats["timeToFirstTokenSec"],
                "engine_decode_tok_s": stats["tokensPerSecond"], "stats": stats,
                "sampled_process_tree_peak_rss_mib": max(peak),
                "sampled_process_tree_rss_delta_mib": max(peak) - baseline,
                "load_config": result.load_config.to_dict(),
                "prediction_config": result.prediction_config.to_dict(),
                "text": result.content,
            }
    finally:
        end.set()
        observer.join()
        command("unload", IDENTIFIER)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=["telos", "lmstudio"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = direct() if args.engine == "telos" else lm_studio()
    with args.output.open("a") as output:
        print(json.dumps(result, ensure_ascii=False), file=output)


if __name__ == "__main__":
    main()
