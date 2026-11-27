"""Out-of-distribution 96-token interactive & capability test suite comparing Stock vs CE-matched W=16."""
import json
from pathlib import Path
import resource
import threading
import time

import modal

VERSION = "1.9.1rc2"
TRANSFORMERS_REVISION = "8e094f8970a60ca83052e2c199783b247b8fc13e"
MODELS = {
    "inkling": ("thinkingmachines/Inkling-2-Mini", "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"),
    "qwen": ("Qwen/Qwen3.5-35B-A3B", "2508b2437e382c605eb0fc52b83f5545960f6b4a"),
    "gemma": ("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"),
}
DOLLARS_PER_SECOND = 0.0012
GPU_TIMEOUT = 1200
MAX_NEW_TOKENS = 96

TEST_PROMPTS = [
    {"id": "logic_apples", "category": "reasoning", "prompt": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Answer concisely."},
    {"id": "python_palindrome", "category": "coding", "prompt": "Write a short Python function `is_palindrome(s: str) -> bool` that ignores spaces and case."},
    {"id": "json_extract", "category": "formatting", "prompt": "Return only valid JSON with keys `city`, `country`, and `currency` for Tokyo, Japan."},
    {"id": "science_photosynthesis", "category": "science", "prompt": "Explain in 2-3 clear sentences what chlorophyll does during photosynthesis."},
    {"id": "history_printing", "category": "factual", "prompt": "Who invented the movable-type printing press in Europe, and around what year?"},
    {"id": "math_train", "category": "math", "prompt": "A train travels at 90 km/h for 2 hours and 20 minutes. How many kilometers does it cover?"},
    {"id": "summary_bullet", "category": "summarization", "prompt": "Summarize in three bullet points why indexes speed up relational database queries."},
    {"id": "creative_haiku", "category": "instruction", "prompt": "Write a haiku (5-7-5 syllables) about an autumn rainstorm."},
]

app = modal.App("telos-full-router-test-suite")
volume = modal.Volume.from_name("telos-router-pilots", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"telos-ml=={VERSION}", "accelerate>=1.2", "sentencepiece", "tiktoken",
        f"git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_REVISION}",
    )
)


def _format_prompt(tokenizer, name: str, user: str) -> list[int]:
    if name == "inkling":
        prompt = (
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
            f"<|message_user|><|content_text|>{user}<|end_message|>"
            "<|message_model|><|content_text|>"
        )
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
        )
    return tokenizer.encode(prompt, add_special_tokens=False)


def _stop_token_ids(tokenizer, model, name: str) -> set[int]:
    stop_ids = set()
    for cand in (tokenizer.eos_token_id, getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if isinstance(cand, int):
            stop_ids.add(cand)
        elif isinstance(cand, (list, tuple, set)):
            stop_ids.update(int(x) for x in cand if x is not None)
    if name == "inkling":
        end_id = tokenizer.convert_tokens_to_ids("<|end_message|>")
        if isinstance(end_id, int) and end_id >= 0:
            stop_ids.add(end_id)
    return stop_ids


def _distinct_4gram_ratio(token_ids: list[int]) -> float:
    if len(token_ids) < 4:
        return 1.0
    grams = [tuple(token_ids[i : i + 4]) for i in range(len(token_ids) - 3)]
    return len(set(grams)) / len(grams)


def _greedy_run(model, tokenizer, stop_ids: set[int], name: str, *, ctx=None):
    import torch

    model.eval()
    results = []
    with torch.no_grad():
        for item in TEST_PROMPTS:
            prompt_ids = _format_prompt(tokenizer, name, item["prompt"])
            prompt_len = len(prompt_ids)
            cur = list(prompt_ids)
            lengths = torch.tensor([prompt_len], device="cuda")
            t0 = time.monotonic()
            stopped_eos = False
            for _ in range(MAX_NEW_TOKENS):
                x = torch.tensor([cur], device="cuda")
                mask = torch.ones_like(x)
                if ctx is not None:
                    ctx.reset_forward("student", lengths)
                logits = model(input_ids=x, attention_mask=mask, use_cache=False).logits
                nxt = int(logits[0, -1].argmax().item())
                cur.append(nxt)
                if nxt in stop_ids:
                    stopped_eos = True
                    break
            gen_ids = cur[prompt_len:]
            results.append({
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "generated_tokens": len(gen_ids),
                "stopped_eos": stopped_eos,
                "distinct_4gram_ratio": _distinct_4gram_ratio(gen_ids),
                "elapsed_seconds": time.monotonic() - t0,
                "text": tokenizer.decode(gen_ids, skip_special_tokens=True),
            })
    return results


@app.function(
    image=image, gpu="H100", cpu=1, memory=8192, timeout=GPU_TIMEOUT,
    retries=0, scaledown_window=2, max_containers=1,
    volumes={"/vol": volume},
    env={"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "HF_ENABLE_PARALLEL_LOADING": "false",
         "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
)
def run_test_suite(name: str, adapter_relpath: str = "ce_match_80step/step_080.safetensors"):
    import torch
    import transformers
    import telos
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from telos.training.inkling2_window_distill import load_adapter, patch_moe_for_window_distill

    started = time.monotonic()
    out_dir = Path("/vol/results") / name
    adapter_path = out_dir / adapter_relpath
    meta = json.loads(adapter_path.with_suffix(".json").read_text())
    saved = load_file(str(adapter_path))
    lora_rank = next(t.shape[0] for k, t in saved.items() if k.endswith(".lora_a"))
    lora_alpha = float(meta.get("lora_alpha", 32.0))

    torch.set_num_threads(2)
    torch.manual_seed(73)
    torch.cuda.reset_peak_memory_stats()
    repo, revision = MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)
    cfg = AutoConfig.from_pretrained(repo, revision=revision, local_files_only=True)
    cls = next(
        (getattr(transformers, arch) for arch in cfg.architectures if hasattr(transformers, arch)),
        AutoModelForCausalLM,
    )
    model = cls.from_pretrained(
        repo, revision=revision, local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager",
    )
    stop_ids = _stop_token_ids(tokenizer, model, name)

    stock = _greedy_run(model, tokenizer, stop_ids, name, ctx=None)

    ctx = patch_moe_for_window_distill(
        model, window_size=16, lora_rank=lora_rank, lora_alpha=lora_alpha,
        selector_rank=8, router_training="full",
    )
    load_adapter(model, str(adapter_path))
    params = dict((n, p) for n, p in model.named_parameters() if p.requires_grad)
    reload_exact = set(saved.keys()) == set(params.keys()) and all(
        torch.equal(params[k].detach().cpu(), saved[k]) for k in saved
    )
    if not reload_exact:
        raise ValueError(f"Fresh-base reload failed for {adapter_path}")
    adapted = _greedy_run(model, tokenizer, stop_ids, name, ctx=ctx)

    summary_stats = {}
    for label, rows in (("stock", stock), ("adapted", adapted)):
        summary_stats[label] = {
            "mean_distinct_4gram": sum(r["distinct_4gram_ratio"] for r in rows) / len(rows),
            "eos_stop_count": sum(1 for r in rows if r["stopped_eos"]),
            "mean_tokens": sum(r["generated_tokens"] for r in rows) / len(rows),
        }

    result = {
        "name": name,
        "model": repo,
        "revision": revision,
        "adapter_path": str(adapter_path),
        "lora_rank": int(lora_rank),
        "lora_alpha": lora_alpha,
        "reload_exact": reload_exact,
        "summary": summary_stats,
        "stock": stock,
        "adapted": adapted,
        "elapsed_seconds": time.monotonic() - started,
        "estimated_compute_usd": (time.monotonic() - started) * DOLLARS_PER_SECOND,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rusage_maxrss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
        "status": "completed",
    }
    (out_dir / "test_suite_96tok.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return json.loads(json.dumps(result))


@app.local_entrypoint()
def main(
    output: str = "reports/full_router_h100_2026-11-27/test_suite_96tok",
    model: str = "all",
    adapter_relpath: str = "ce_match_80step/step_080.safetensors",
):
    if model != "all" and model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    names = list(MODELS) if model == "all" else [model]
    for name in names:
        summary[name] = run_test_suite.remote(name, adapter_relpath)
        summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v["summary"] for k, v in summary.items()}, indent=2))
