"""Run one local LM Studio model at a time on the localhost API."""

import argparse
import contextlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import psutil

from telos.eval.executor import execute_code_sandboxed

from telos_eval import evaluate as telos_evaluate

LMS = Path.home() / ".lmstudio/bin/lms"
IDENTIFIER = "telos-audit"
MiB = 1024 ** 2
CASES = {
    "fact": "Reply with exactly one word, no punctuation: What is the capital of France?",
    "json": 'Return only valid JSON with exactly these fields and values: city is "Pune", count is 3, open is true.',
    "python": "Write a single Python function named square(x) that returns the square of x. Return only the code.",
}


def lms(*args, timeout=90):
    result = subprocess.run([str(LMS), *args], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        def plain(value):
            return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", value).strip()

        error_lines = plain(result.stderr).splitlines()
        output_lines = plain(result.stdout).splitlines()
        details = [line for line in error_lines + output_lines
                   if line.strip() and not line.strip().startswith("Loading ")]
        raise RuntimeError(" | ".join(details[-5:]) or f"lms exited {result.returncode}")
    return result.stdout


def chat(base, prompt, limit, reasoning=None):
    payload = {"model": IDENTIFIER, "input": prompt, "stream": True, "store": False,
               "temperature": 0, "max_output_tokens": limit}
    if reasoning is not None:
        payload["reasoning"] = reasoning
    request = Request(base + "/api/v1/chat", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first_model = first_visible = None
    with urlopen(request, timeout=65) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = json.loads(line[5:])
            kind = data["type"]
            now = time.perf_counter()
            if first_model is None and kind in {"message.delta", "reasoning.delta"}:
                first_model = now
            if first_visible is None and kind == "message.delta":
                first_visible = now
            if kind == "error":
                raise RuntimeError(str(data.get("error", data)))
            if kind == "chat.end":
                result = data["result"]
                return {
                    "client_first_token_ms": (first_model - start) * 1e3 if first_model else None,
                    "client_first_visible_ms": (first_visible - start) * 1e3 if first_visible else None,
                    "total_ms": (now - start) * 1e3,
                    "text": "".join(x["content"] for x in result["output"] if x["type"] == "message"),
                    "stats": result["stats"],
                }
    raise RuntimeError("LM Studio stream closed without chat.end")


def grade(name, text):
    if not text.strip():
        return "no_visible_answer"
    if name == "fact":
        ok = text.strip() == "Paris"
    elif name == "json":
        try:
            value = json.loads(text)
            ok = value == {"city": "Pune", "count": 3, "open": True}
            ok = ok and type(value["count"]) is int and type(value["open"]) is bool
        except (json.JSONDecodeError, TypeError, KeyError):
            ok = False
    else:
        code = re.sub(r"\A```(?:python)?\s*(.*?)\s*```\Z", r"\1", text.strip(), flags=re.S)
        outcome, _ = execute_code_sandboxed(code, "assert square(-3) == 9 and square(0) == 0 and square(7) == 49")
        return outcome.value
    return "passed" if ok else "failed_contract"


def workers():
    processes = {}
    for candidate in psutil.process_iter(["name"]):
        if candidate.info["name"] == "LM Studio":
            with contextlib.suppress(psutil.Error):
                processes[candidate.pid] = candidate
                processes.update({child.pid: child for child in candidate.children(recursive=True)})
    return processes


def rss(proc_map):
    readings = {}
    for pid, process in proc_map.items():
        with contextlib.suppress(psutil.Error):
            readings[pid] = process.memory_info().rss
    return sum(readings.values()) / MiB


def performance_prompts():
    instructions = "Write twenty numbered, distinct, concrete Python programming tips. Include a short example for every tip. Start at number 1."
    yield "short", "Single request A. " + instructions
    for index, word in enumerate(["orchid", "copper"], 1):
        prefix = ("For this exercise, examples can use small lists, dictionaries, strings, and numeric values. "
                  "Avoid external libraries and make each example readable to a beginner. ") * 15
        yield "long", f"{word} exercise {index}. " + prefix + instructions


def available_percent():
    output = subprocess.check_output(["memory_pressure", "-Q"], text=True, timeout=10)
    match = re.search(r"free percentage: (\d+)%", output)
    return int(match[1]) if match else None


def audit_model(base, model, *, telos_only=False, context_length=2048):
    key = model["modelKey"]
    result = {"model": model, "context_length": context_length}
    large = model["sizeBytes"] >= 8 * 1024 ** 3 - 64 * MiB
    if large:
        result["memory_free_before_load_pct"] = free = available_percent()
        if free is not None and free < 40:
            result["error"] = {"type": "MemoryHeadroom", "message": f"Deferred 27B load at {free}% free memory."}
            return result
    baseline = rss(workers())
    samples = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.12):
            samples.append(rss(workers()))

    observer = threading.Thread(target=sample, daemon=True)
    observer.start()
    try:
        start = time.perf_counter()
        text_options = ("--parallel", "1", "--no-speculative-draft-mtp") if model["type"] == "llm" else ()
        lms("load", key, "--identifier", IDENTIFIER, "--context-length", str(context_length),
            *text_options, "--ttl", "240", "--yes")
        result["load_ms"] = (time.perf_counter() - start) * 1e3
        result["idle_rss_mib"] = rss(workers())
        if large:
            result["memory_free_after_load_pct"] = free = available_percent()
            if free is not None and free < 18:
                raise RuntimeError(f"Stopped 27B benchmark at {free}% free memory after load.")
        if model["type"] == "embedding":
            request = Request(base + "/v1/embeddings", data=json.dumps({
                "model": IDENTIFIER, "input": ["a list of Python values", "a list of Python values", "fresh orange juice"]
            }).encode(), headers={"Content-Type": "application/json"})
            start = time.perf_counter()
            with urlopen(request, timeout=60) as response:
                vectors = [x["embedding"] for x in json.load(response)["data"]]
            result["embedding"] = {"latency_ms": (time.perf_counter() - start) * 1e3,
                                   "dimensions": len(vectors[0]), "duplicates_equal": vectors[0] == vectors[1],
                                   "distinct_input_differs": vectors[0] != vectors[2]}
        elif not telos_only:
            result["performance"] = []
            result["quality"] = {}
            reasoning = "off"
            try:
                chat(base, "Return just OK.", 16, reasoning)
            except (HTTPError, RuntimeError):
                reasoning = None
                chat(base, "Return just OK.", 16, reasoning)
            result["reasoning_setting"] = reasoning or "model default"
            for group, prompt in performance_prompts():
                result["performance"].append({"group": group, **chat(base, prompt, 160, reasoning)})
            for name, prompt in CASES.items():
                answer = chat(base, prompt, 192, reasoning)
                result["quality"][name] = {**answer, "outcome": grade(name, answer["text"])}
        if model["type"] == "llm":
            result["telos_eval"] = telos_evaluate(IDENTIFIER, api_base=base + "/v1")
    except Exception as exc:
        if isinstance(exc, HTTPError):
            message = exc.read().decode(errors="replace").strip()
        else:
            message = str(exc)
        result["error"] = {"type": type(exc).__name__, "message": message[:700]}
    finally:
        stop.set()
        observer.join()
        result["baseline_rss_mib"] = baseline
        result["sampled_peak_rss_mib"] = max(samples)
        result["sampled_peak_rss_delta_mib"] = max(samples) - baseline
        try:
            lms("unload", IDENTIFIER, timeout=20)
        except Exception as exc:
            result["unload_error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:12489")
    parser.add_argument("--model", action="append")
    parser.add_argument("--telos-only", action="store_true")
    parser.add_argument("--context-length", type=int, default=2048)
    args = parser.parse_args()
    entries = json.loads(lms("ls", "--json"))
    entries.sort(key=lambda entry: entry["sizeBytes"])
    seen = set()
    if args.output.exists():
        seen = {json.loads(line)["model"]["modelKey"] for line in args.output.read_text().splitlines() if line.strip()}
    with args.output.open("a") as out:
        for entry in entries:
            key = entry["modelKey"]
            if key in seen or (args.model and key not in args.model):
                continue
            result = audit_model(args.base, entry, telos_only=args.telos_only, context_length=args.context_length)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{key}: {'ERROR ' + result['error']['message'] if 'error' in result else 'complete'}", flush=True)
            if result.get("unload_error"):
                remaining = json.loads(lms("ps", "--json"))
                if any(model.get("identifier") == IDENTIFIER for model in remaining):
                    raise RuntimeError("The audit model is still loaded. Stopping to avoid multiple resident models.")


if __name__ == "__main__":
    main()
