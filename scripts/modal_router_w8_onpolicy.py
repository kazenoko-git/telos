"""Resumable W=8 on-policy + mixed-domain H100 distillation from step 80 to step 320."""
import json
import math
from pathlib import Path
import random
import resource
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
WINDOW_SIZE = 8
START_STEP = 80
TARGET_STEP = 320
GRAD_ACCUM = 4
MAX_LENGTH = 256
LORA_RANK = 16
LORA_ALPHA = 32.0
LR_INIT = 3.0e-5
LR_PEAK = 2.0e-4
LR_MIN = 2.0e-5
WARMUP_STEPS = 10
COV_WEIGHT = 0.5
CE_WEIGHT = 0.05
KL_WEIGHT = 1.0
ROLLOUT_TARGET = 96
ROLLOUT_TOPUP_TIME_CAP_S = 120.0
TEST_MAX_NEW_TOKENS = 96

CHUNKS = {
    "inkling": [(80, 200), (200, 320)],
    "qwen": [(80, 200), (200, 320)],
    "gemma": [(80, 320)],
}

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

app = modal.App("telos-full-router-w8-onpolicy")
volume = modal.Volume.from_name("telos-router-pilots", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"telos-ml=={VERSION}", "accelerate>=1.2", "sentencepiece", "tiktoken",
        f"git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_REVISION}",
    )
    .add_local_python_source("telos")
)


def _format_prompt_ids(tokenizer, name: str, user: str) -> list[int]:
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


def _encode_pair(tokenizer, name: str, user: str, response: str, max_length: int):
    prompt_ids = _format_prompt_ids(tokenizer, name, user)
    eos = "<|end_message|>" if name == "inkling" else (tokenizer.eos_token or "")
    ids = (prompt_ids + tokenizer.encode(response + eos, add_special_tokens=False))[:max_length]
    if len(ids) < len(prompt_ids) + 4:
        return None
    return {"ids": ids, "prompt_len": len(prompt_ids), "user": user}


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


def _lr_for_step(step_num: int) -> float:
    rel = step_num - START_STEP - 1
    total = TARGET_STEP - START_STEP
    if rel < WARMUP_STEPS:
        return LR_INIT + (LR_PEAK - LR_INIT) * float(rel + 1) / float(WARMUP_STEPS)
    progress = float(rel - WARMUP_STEPS) / float(max(1, total - WARMUP_STEPS))
    return LR_MIN + 0.5 * (LR_PEAK - LR_MIN) * (1.0 + math.cos(math.pi * progress))


def _topup_stock_rollouts(model, tokenizer, stop_ids: set[int], name: str, existing: list[dict], mixed_rows: list[dict]):
    import torch

    seen = {r["user"] for r in existing}
    candidates = [r for r in mixed_rows if r["user"] not in seen]
    eos_list = sorted(stop_ids) if stop_ids else None
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_list[0] if eos_list else 0
    if isinstance(pad_id, list):
        pad_id = pad_id[0]

    model.eval()
    t0 = time.monotonic()
    rollouts = list(existing)
    idx = 0
    batch_size = 4
    with torch.no_grad():
        while len(rollouts) < ROLLOUT_TARGET and idx < len(candidates) and (time.monotonic() - t0) < ROLLOUT_TOPUP_TIME_CAP_S:
            batch_rows = candidates[idx : idx + batch_size]
            idx += len(batch_rows)
            prompts = []
            for row in batch_rows:
                p_ids = _format_prompt_ids(tokenizer, name, row["user"])
                if len(p_ids) < MAX_LENGTH - 16:
                    prompts.append((row, p_ids))
            if not prompts:
                continue
            max_p = max(len(p) for _, p in prompts)
            in_ids = torch.full((len(prompts), max_p), pad_id, dtype=torch.long, device="cuda")
            attn_mask = torch.zeros_like(in_ids)
            for bi, (_, p_ids) in enumerate(prompts):
                in_ids[bi, max_p - len(p_ids) :] = torch.tensor(p_ids, dtype=torch.long, device="cuda")
                attn_mask[bi, max_p - len(p_ids) :] = 1
            try:
                out = model.generate(
                    input_ids=in_ids, attention_mask=attn_mask, max_new_tokens=160,
                    do_sample=False, use_cache=True, eos_token_id=eos_list, pad_token_id=pad_id,
                )
                for bi, (row, p_ids) in enumerate(prompts):
                    gen_ids = out[bi, max_p:].tolist()
                    # Trim trailing pad/stop repeats after first stop token.
                    trimmed = []
                    for tok in gen_ids:
                        trimmed.append(tok)
                        if tok in stop_ids:
                            break
                    if len(trimmed) >= 6:
                        rollouts.append({
                            "source": f"onpolicy_{row.get('source', 'mixed')}",
                            "user": row["user"],
                            "prompt_len": len(p_ids),
                            "ids": (p_ids + trimmed)[: MAX_LENGTH],
                        })
            except Exception:
                # Fallback to single-example generation if batched left-pad fails on a custom cache.
                batch_size = 1
                row, p_ids = prompts[0]
                x = torch.tensor([p_ids], device="cuda")
                out = model.generate(
                    input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=144,
                    do_sample=False, use_cache=True, eos_token_id=eos_list, pad_token_id=pad_id,
                )
                gen_ids = out[0, len(p_ids) :].tolist()
                if len(gen_ids) >= 6:
                    rollouts.append({
                        "source": f"onpolicy_{row.get('source', 'mixed')}",
                        "user": row["user"],
                        "prompt_len": len(p_ids),
                        "ids": (p_ids + gen_ids)[: MAX_LENGTH],
                    })
    return rollouts, time.monotonic() - t0


@app.function(
    image=image, gpu="H100", cpu=1, memory=8192, timeout=GPU_TIMEOUT,
    retries=0, scaledown_window=2, max_containers=1,
    volumes={"/vol": volume},
    env={"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "HF_ENABLE_PARALLEL_LOADING": "false",
         "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
)
def train_w8_onpolicy_chunk(name: str, start_step: int, end_step: int, disjoint_prompts: list[str]):
    import torch
    import transformers
    import telos
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from telos.training.inkling2_window_distill import (
        WindowPinnedRouter, distill_step, export_adapter, load_adapter, patch_moe_for_window_distill,
    )

    started = time.monotonic()
    out_dir = Path("/vol/results") / name / "w8_onpolicy_320"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "name": name, "phase": f"w8_onpolicy_{start_step}_to_{end_step}",
        "start_step": start_step, "end_step": end_step,
        "window": WINDOW_SIZE, "adapt_resident_mlp": True,
        "telos": str(telos.__version__), "transformers": str(transformers.__version__),
        "torch": str(torch.__version__), "gpu": str(torch.cuda.get_device_name()),
        "model": MODELS[name][0], "revision": MODELS[name][1],
    }

    try:
        assert telos.__version__ == VERSION
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
        result["load_seconds"] = time.monotonic() - started
        stop_ids = _stop_token_ids(tokenizer, model, name)

        mixed_rows = json.loads(Path("/vol/stage2_mixed.json").read_text())
        rollout_path = Path("/vol/results") / name / "onpolicy_rollouts_v2.json"
        if not rollout_path.exists():
            old_path = Path("/vol/results") / name / "onpolicy_rollouts.json"
            existing = json.loads(old_path.read_text()) if old_path.exists() else []
            rollouts, r_secs = _topup_stock_rollouts(model, tokenizer, stop_ids, name, existing, mixed_rows[: 300])
            rollout_path.write_text(json.dumps(rollouts, indent=2))
            volume.commit()
            result["rollouts_topup"] = {"total": len(rollouts), "seconds": r_secs}
            print(name, "rollouts_v2", result["rollouts_topup"], flush=True)
        else:
            rollouts = json.loads(rollout_path.read_text())

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ctx = patch_moe_for_window_distill(
            model, window_size=WINDOW_SIZE, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
            selector_rank=8, router_training="full", adapt_resident_mlp=True,
        )
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        routers = [(n, m) for n, m in model.named_modules() if isinstance(m, WindowPinnedRouter)]

        if start_step == START_STEP:
            resume_adapter = Path("/vol/results") / name / "w8_80step" / "step_080.safetensors"
            resume_opt = None
        else:
            resume_adapter = out_dir / f"step_{start_step:03d}.safetensors"
            resume_opt = out_dir / f"optimizer_{start_step:03d}.pt"
        load_adapter(model, str(resume_adapter))
        optimizer = torch.optim.AdamW([p for _, p in params], lr=LR_INIT, weight_decay=0.0)
        if resume_opt is not None and resume_opt.is_file():
            opt_state = torch.load(str(resume_opt), map_location="cpu", weights_only=True)
            optimizer.load_state_dict(opt_state)
            del opt_state

        dolly_1024 = json.loads(Path("/vol/dolly_1024.json").read_text())
        by_user = {
            r["instruction"] + ("\n\n" + r["context"] if r.get("context") else ""): r
            for r in dolly_1024
        }
        disjoint_held = [
            _encode_pair(tokenizer, name, u, by_user[u]["response"], MAX_LENGTH)
            for u in disjoint_prompts
        ]
        assert all(x is not None for x in disjoint_held)

        # Build interleaved stream: 50% on-policy stock rollouts (cycled) + 50% fresh mixed rows.
        onpolicy_samples = [
            {"ids": r["ids"][: MAX_LENGTH], "prompt_len": r["prompt_len"], "user": r["user"], "onpolicy": True}
            for r in rollouts if len(r["ids"]) >= r["prompt_len"] + 4
        ]
        blocked = set(disjoint_prompts)
        offpolicy_samples = []
        for row in mixed_rows[520:]:
            u = row["user"]
            if u in blocked:
                continue
            enc = _encode_pair(tokenizer, name, u, row["response"], MAX_LENGTH)
            if enc is not None:
                enc["onpolicy"] = False
                offpolicy_samples.append(enc)
                blocked.add(u)
            if len(offpolicy_samples) == (TARGET_STEP - START_STEP) * GRAD_ACCUM:
                break
        rng = random.Random(73)
        rng.shuffle(onpolicy_samples)

        history = []
        for step_num in range(start_step + 1, end_step + 1):
            lr_now = _lr_for_step(step_num)
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            model.train()
            optimizer.zero_grad()
            aggregate = {}
            base_idx = step_num - START_STEP - 1
            batch_rows = [
                onpolicy_samples[(base_idx * 2) % len(onpolicy_samples)],
                offpolicy_samples[(base_idx * 2) % len(offpolicy_samples)],
                onpolicy_samples[(base_idx * 2 + 1) % len(onpolicy_samples)],
                offpolicy_samples[(base_idx * 2 + 1) % len(offpolicy_samples)],
            ]
            for row in batch_rows:
                x = torch.tensor([row["ids"]], device="cuda")
                lengths = torch.tensor([row["prompt_len"]], device="cuda")
                ce_w = 0.0 if row["onpolicy"] else CE_WEIGHT
                loss, metrics = distill_step(
                    model, ctx, x, lengths, torch.ones_like(x),
                    kl_weight=KL_WEIGHT, ce_weight=ce_w, cov_weight=COV_WEIGHT,
                )
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite loss at step {step_num}")
                (loss / GRAD_ACCUM).backward()
                for k, val in metrics.items():
                    aggregate[k] = aggregate.get(k, 0.0) + val / GRAD_ACCUM
            for n, router in routers:
                for field, w in (("selector", router.selector.selector_weight), ("token", router.token_weight)):
                    if w.grad is None or not torch.isfinite(w.grad).all():
                        raise ValueError(f"Missing/nonfinite {field} grad at {n}")
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0, error_if_nonfinite=True).item())
            optimizer.step()
            history.append({"step": step_num, "lr": lr_now, "grad_norm": grad_norm, **aggregate})
            if step_num == start_step + 1 or step_num % 20 == 0 or step_num == end_step:
                print(name, "w8_onpolicy", history[-1], flush=True)

        ckpt_path = out_dir / f"step_{end_step:03d}.safetensors"
        export_adapter(model, str(ckpt_path), metadata={
            "source_model": repo, "source_revision": revision,
            "lora_alpha": LORA_ALPHA, "step": end_step,
            "window": {"window_size": WINDOW_SIZE, "selector_alpha": 32.0, "token_router_scale": 2.0,
                       "router_training": "full", "adapt_resident_mlp": True},
        })
        if end_step < TARGET_STEP:
            opt_path = out_dir / f"optimizer_{end_step:03d}.pt"
            torch.save(optimizer.state_dict(), str(opt_path))
            if resume_opt is not None and resume_opt.is_file() and resume_opt != opt_path:
                resume_opt.unlink()
        else:
            saved = load_file(str(ckpt_path))
            with torch.no_grad():
                for _, p in params:
                    p.zero_()
            load_adapter(model, str(ckpt_path))
            result["reload_parameters_exact"] = all(torch.equal(p.detach().cpu(), saved[n]) for n, p in params)
            if resume_opt is not None and resume_opt.is_file():
                resume_opt.unlink()

        model.eval()
        eval_metrics = []
        with torch.no_grad():
            for row in disjoint_held:
                x = torch.tensor([row["ids"]], device="cuda")
                lengths = torch.tensor([row["prompt_len"]], device="cuda")
                _, m = distill_step(
                    model, ctx, x, lengths, torch.ones_like(x),
                    kl_weight=KL_WEIGHT, ce_weight=CE_WEIGHT, cov_weight=COV_WEIGHT,
                )
                eval_metrics.append(m)
        result["disjoint_64"] = {k: sum(m[k] for m in eval_metrics) / len(eval_metrics) for k in eval_metrics[0]}
        print(name, f"eval_w8_onpolicy_{end_step}", result["disjoint_64"], flush=True)

        if end_step == TARGET_STEP:
            test_rows = []
            with torch.no_grad():
                for item in TEST_PROMPTS:
                    p_ids = _format_prompt_ids(tokenizer, name, item["prompt"])
                    cur = list(p_ids)
                    lengths = torch.tensor([len(p_ids)], device="cuda")
                    stopped_eos = False
                    for _ in range(TEST_MAX_NEW_TOKENS):
                        x = torch.tensor([cur], device="cuda")
                        ctx.reset_forward("student", lengths)
                        logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
                        nxt = int(logits[0, -1].argmax().item())
                        cur.append(nxt)
                        if nxt in stop_ids:
                            stopped_eos = True
                            break
                    gen_ids = cur[len(p_ids) :]
                    test_rows.append({
                        "id": item["id"],
                        "generated_tokens": len(gen_ids),
                        "stopped_eos": stopped_eos,
                        "distinct_4gram_ratio": _distinct_4gram_ratio(gen_ids),
                        "text": tokenizer.decode(gen_ids, skip_special_tokens=True),
                    })
            result["test_suite_w8_step320"] = {
                "mean_distinct_4gram": sum(r["distinct_4gram_ratio"] for r in test_rows) / len(test_rows),
                "eos_stop_count": sum(1 for r in test_rows if r["stopped_eos"]),
                "rows": test_rows,
            }

        result["checkpoint_path"] = str(ckpt_path)
        result["checkpoint_bytes"] = ckpt_path.stat().st_size
        result["history"] = history
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["estimated_compute_usd"] = result["elapsed_seconds"] * DOLLARS_PER_SECOND
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        result["rusage_maxrss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
        (out_dir / f"result_{start_step}_{end_step}.json").write_text(json.dumps(result, indent=2))
        volume.commit()
    return json.loads(json.dumps(result))


@app.local_entrypoint()
def main(output: str = "reports/full_router_h100_2026-11-27/w8_onpolicy_320", model: str = "all"):
    if model != "all" and model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    phase1 = json.loads(Path("reports/full_router_h100_2026-11-27/eval_phase1/summary.json").read_text())
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    names = list(MODELS) if model == "all" else [model]
    for name in names:
        disjoint_prompts = phase1[name]["eval"]["disjoint_prompts"]
        model_entry = summary.get(name, {})
        for start_s, end_s in CHUNKS[name]:
            key = f"{start_s}_{end_s}"
            if model_entry.get(key, {}).get("status") == "completed":
                continue
            model_entry[key] = train_w8_onpolicy_chunk.remote(name, start_s, end_s, disjoint_prompts)
            summary[name] = model_entry
            summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        m: {
            k: {
                "disjoint_64": v.get("disjoint_64"),
                "test_d4": v.get("test_suite_w8_step320", {}).get("mean_distinct_4gram"),
                "test_eos": v.get("test_suite_w8_step320", {}).get("eos_stop_count"),
            }
            for k, v in chunks.items()
        }
        for m, chunks in summary.items()
    }, indent=2))
