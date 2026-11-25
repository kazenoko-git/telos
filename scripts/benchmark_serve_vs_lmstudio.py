"""Head-to-head HTTP/SSE benchmark comparing `telos serve` against LM Studio.

Measures client-observed TTFT, sustained decode rate, total request wall time,
and greedy output parity across short, shared-prefix, and sampled workloads
using `/v1/completions` SSE streaming on both servers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


DEFAULT_PREFIX = (
    "You are a precise systems engineer reviewing a Python module for performance "
    "and correctness. Focus on cache locality, algorithmic complexity, memory "
    "allocations, and exact numerical parity. "
) * 12

DEFAULT_WORKLOADS = {
    "short_greedy": {
        "prompt": "Write a concise Python function `is_prime(n: int) -> bool` with O(sqrt(n)) complexity.\n\n```python\n",
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "shared_prefix_turn1": {
        "prompt": DEFAULT_PREFIX + "\nQuestion: Summarize why contiguous KV cache reservations reduce decode allocations.\nAnswer:",
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "shared_prefix_turn2": {
        "prompt": DEFAULT_PREFIX + "\nQuestion: Summarize how exact prefix KV reuse reduces multi-turn TTFT.\nAnswer:",
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "sampled_seeded": {
        "prompt": "List three concrete strategies to profile GPU kernel launch overhead on Apple Silicon:\n1.",
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42,
    },
}


def stream_completion(
    client: httpx.Client,
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 192,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Stream one `/v1/completions` SSE request and record client timing metrics."""
    url = base_url.rstrip("/") + "/completions"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        payload["seed"] = seed

    started = time.perf_counter()
    first_chunk_at: float | None = None
    last_chunk_at: float | None = None
    chunks: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    with client.stream("POST", url, json=payload) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            body = line.removeprefix("data: ").strip()
            if body == "[DONE]":
                break
            event = json.loads(body)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            text = choice.get("text") or ""
            if text:
                now = time.perf_counter() - started
                if first_chunk_at is None:
                    first_chunk_at = now
                last_chunk_at = now
                chunks.append(text)

    total_seconds = time.perf_counter() - started
    completion_tokens = (
        int(usage["completion_tokens"])
        if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int)
        else len(chunks)
    )
    decode_tok_s = None
    if completion_tokens > 1 and first_chunk_at is not None and last_chunk_at is not None:
        decode_window = last_chunk_at - first_chunk_at
        if decode_window > 0:
            decode_tok_s = (completion_tokens - 1) / decode_window

    return {
        "text": "".join(chunks),
        "finish_reason": finish_reason,
        "usage": usage,
        "completion_tokens": completion_tokens,
        "chunk_count": len(chunks),
        "ttft_ms": None if first_chunk_at is None else first_chunk_at * 1000.0,
        "wall_seconds": total_seconds,
        "decode_tok_s": decode_tok_s,
    }


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    ttfts = [t["ttft_ms"] for t in trials if t["ttft_ms"] is not None]
    rates = [t["decode_tok_s"] for t in trials if t["decode_tok_s"] is not None]
    walls = [t["wall_seconds"] for t in trials]
    return {
        "trials": len(trials),
        "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
        "median_decode_tok_s": statistics.median(rates) if rates else None,
        "median_wall_seconds": statistics.median(walls) if walls else None,
        "sample_text": trials[0]["text"] if trials else "",
    }


def run_benchmark(
    telos_url: str,
    telos_model: str,
    lmstudio_url: str,
    lmstudio_model: str,
    *,
    max_tokens: int = 192,
    trials: int = 4,
    warmup_tokens: int = 8,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        for url, model in [(telos_url, telos_model), (lmstudio_url, lmstudio_model)]:
            stream_completion(http, url, model, "Warmup prompt.", max_tokens=warmup_tokens)

        report: dict[str, Any] = {"max_tokens": max_tokens, "trials": trials, "workloads": {}}
        order = ["telos", "lmstudio", "lmstudio", "telos"]
        for name, spec in DEFAULT_WORKLOADS.items():
            telos_trials: list[dict[str, Any]] = []
            lms_trials: list[dict[str, Any]] = []
            for i in range(trials):
                first = order[i % len(order)]
                pair = [first, "lmstudio" if first == "telos" else "telos"]
                for engine in pair:
                    url = telos_url if engine == "telos" else lmstudio_url
                    model = telos_model if engine == "telos" else lmstudio_model
                    res = stream_completion(
                        http,
                        url,
                        model,
                        spec["prompt"],
                        max_tokens=max_tokens,
                        temperature=spec["temperature"],
                        top_p=spec["top_p"],
                        seed=spec.get("seed"),
                    )
                    (telos_trials if engine == "telos" else lms_trials).append(res)

            t_sum = summarize_trials(telos_trials)
            l_sum = summarize_trials(lms_trials)
            parity = t_sum["sample_text"].lstrip() == l_sum["sample_text"].lstrip()
            report["workloads"][name] = {
                "telos": t_sum,
                "lmstudio": l_sum,
                "greedy_text_parity": parity if spec["temperature"] == 0.0 else None,
            }
        return report
    finally:
        if owns_client:
            http.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark `telos serve` vs LM Studio over HTTP/SSE.")
    parser.add_argument("--telos-url", required=True)
    parser.add_argument("--telos-model", required=True)
    parser.add_argument("--lmstudio-url", required=True)
    parser.add_argument("--lmstudio-model", required=True)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_benchmark(
        args.telos_url,
        args.telos_model,
        args.lmstudio_url,
        args.lmstudio_model,
        max_tokens=args.max_tokens,
        trials=args.trials,
        warmup_tokens=args.warmup_tokens,
    )
    for name, item in report["workloads"].items():
        t_sum, l_sum = item["telos"], item["lmstudio"]
        print(
            f"[{name}] Telos: TTFT={t_sum['median_ttft_ms'] or 0:.1f}ms, "
            f"decode={t_sum['median_decode_tok_s'] or 0:.1f} tok/s | "
            f"LM Studio: TTFT={l_sum['median_ttft_ms'] or 0:.1f}ms, "
            f"decode={l_sum['median_decode_tok_s'] or 0:.1f} tok/s | "
            f"parity={item['greedy_text_parity']}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
