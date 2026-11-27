"""Combined H100 diagnostic: generate stock on-policy rollouts + train/eval W=8 with resident MLP LoRA."""
import json
import math
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
WINDOW_SIZE = 8
TRAIN_STEPS = 80
GRAD_ACCUM = 4
MAX_LENGTH = 256
LORA_RANK = 16
LORA_ALPHA = 32.0
LR_MAX = 3.0e-4
LR_MIN = 3.0e-5
WARMUP_STEPS = 8
COV_WEIGHT = 0.5
CE_WEIGHT = 0.1
KL_WEIGHT = 1.0
ROLLOUT_TARGET = 96
ROLLOUT_TIME_CAP_S = 200.0
ROLLOUT_MAX_NEW_TOKENS = 160
TEST_MAX_NEW_TOKENS = 96

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

app = modal.App("telos-full-router-w8-diag")
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


def _cosine_lr(step_idx: int) -> float:
    if step_idx < WARMUP_STEPS:
        return LR_MAX * float(step_idx + 1) / float(WARMUP_STEPS)
    progress = float(step_idx - WARMUP_STEPS) / float(max(1, TRAIN_STEPS - WARMUP_STEPS))
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * progress))


def _generate_stock_rollouts(model, tokenizer, stop_ids: set[int], name: str, prompts: list[dict]):
    import torch

    model.eval()
    rollouts = []
    t0 = time.monotonic()
    eos_list = sorted(stop_ids) if stop_ids else None
    use_kv_cache = True
    with torch.no_grad():
        for row in prompts:
            if len(rollouts) >= ROLLOUT_TARGET or (time.monotonic() - t0) >= ROLLOUT_TIME_CAP_S:
                break
            user = row["user"]
            prompt_ids = _format_prompt_ids(tokenizer, name, user)
            if len(prompt_ids) >= MAX_LENGTH - 16:
                continue
            x = torch.tensor([prompt_ids], device="cuda")
            mask = torch.ones_like(x)
            try:
                out = model.generate(
                    input_ids=x, attention_mask=mask, max_new_tokens=ROLLOUT_MAX_NEW_TOKENS,
                    do_sample=False, use_cache=use_kv_cache, eos_token_id=eos_list,
                    pad_token_id=(eos_list[0] if eos_list else 0),
                )
                gen_ids = out[0, len(prompt_ids) :].tolist()
            except Exception:
                use_kv_cache = False
                cur = list(prompt_ids)
                for _ in range(96):
                    cur_x = torch.tensor([cur], device="cuda")
                    logits = model(input_ids=cur_x, attention_mask=torch.ones_like(cur_x), use_cache=False).logits
                    nxt = int(logits[0, -1].argmax().item())
                    cur.append(nxt)
                    if nxt in stop_ids:
                        break
                gen_ids = cur[len(prompt_ids) :]
            if len(gen_ids) < 6:
                continue
            full_ids = (prompt_ids + gen_ids)[: MAX_LENGTH + 64]
            rollouts.append({
                "source": row.get("source", "mixed"),
                "user": user,
                "prompt_len": len(prompt_ids),
                "ids": full_ids,
                "text": tokenizer.decode(gen_ids, skip_special_tokens=False),
            })
    return rollouts, use_kv_cache, time.monotonic() - t0


@app.function(
    image=image, gpu="H100", cpu=1, memory=8192, timeout=GPU_TIMEOUT,
    retries=0, scaledown_window=2, max_containers=1,
    volumes={"/vol": volume},
    env={"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "HF_ENABLE_PARALLEL_LOADING": "false",
         "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
)
def run_w8_diag(name: str, disjoint_prompts: list[str]):
    import torch
    import transformers
    import telos
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from telos.training.inkling2_window_distill import (
        WindowPinnedRouter, distill_step, export_adapter, load_adapter, patch_moe_for_window_distill,
    )

    started = time.monotonic()
    out_dir = Path("/vol/results") / name / "w8_80step"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "name": name, "phase": "w8_resident_mlp_diag",
        "window": WINDOW_SIZE, "adapt_resident_mlp": True,
        "telos": str(telos.__version__), "transformers": str(transformers.__version__),
        "torch": str(torch.__version__), "gpu": str(torch.cuda.get_device_name()),
        "model": MODELS[name][0], "revision": MODELS[name][1],
        "steps": TRAIN_STEPS, "grad_accum": GRAD_ACCUM, "lora_rank": LORA_RANK,
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

        # 1. Generate stock on-policy rollouts before patching.
        mixed_rows = json.loads(Path("/vol/stage2_mixed.json").read_text())
        rollout_path = Path("/vol/results") / name / "onpolicy_rollouts.json"
        rollouts, used_kv_cache, rollout_secs = _generate_stock_rollouts(
            model, tokenizer, stop_ids, name, mixed_rows[: 200],
        )
        rollout_path.write_text(json.dumps(rollouts, indent=2))
        result["rollouts"] = {"count": len(rollouts), "used_kv_cache": used_kv_cache, "seconds": rollout_secs}
        print(name, "rollouts", result["rollouts"], flush=True)

        # 2. Patch with W=8 and adapt_resident_mlp=True.
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ctx = patch_moe_for_window_distill(
            model, window_size=WINDOW_SIZE, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
            selector_rank=8, router_training="full", adapt_resident_mlp=True,
        )
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        routers = [(n, m) for n, m in model.named_modules() if isinstance(m, WindowPinnedRouter)]
        result["trainable_parameters"] = sum(p.numel() for _, p in params)
        result["resident_mlp_lora_tensors"] = sum(1 for n, _ in params if ".mlp." in n or "shared_expert" in n)

        dolly_1024 = json.loads(Path("/vol/dolly_1024.json").read_text())
        by_user = {
            r["instruction"] + ("\n\n" + r["context"] if r.get("context") else ""): r
            for r in dolly_1024
        }
        disjoint_held = []
        for u in disjoint_prompts:
            row = by_user[u]
            enc = _encode_pair(tokenizer, name, u, row["response"], MAX_LENGTH)
            if enc is None:
                raise ValueError("Locked disjoint prompt failed encoding")
            disjoint_held.append(enc)

        blocked = set(disjoint_prompts)
        train_samples = []
        needed = TRAIN_STEPS * GRAD_ACCUM
        for row in mixed_rows[200:]:
            u = row["user"]
            if u in blocked:
                continue
            enc = _encode_pair(tokenizer, name, u, row["response"], MAX_LENGTH)
            if enc is not None:
                train_samples.append(enc)
                blocked.add(u)
            if len(train_samples) == needed:
                break
        assert len(train_samples) == needed

        def eval_disjoint():
            model.eval()
            metrics = []
            with torch.no_grad():
                for row in disjoint_held:
                    x = torch.tensor([row["ids"]], device="cuda")
                    lengths = torch.tensor([row["prompt_len"]], device="cuda")
                    _, m = distill_step(
                        model, ctx, x, lengths, torch.ones_like(x),
                        kl_weight=KL_WEIGHT, ce_weight=CE_WEIGHT, cov_weight=COV_WEIGHT,
                    )
                    metrics.append(m)
            return {k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0]}

        result["untrained_w8"] = eval_disjoint()
        print(name, "untrained_w8", result["untrained_w8"], flush=True)

        optimizer = torch.optim.AdamW([p for _, p in params], lr=LR_MAX, weight_decay=0.0)
        history = []
        checkpoints = {}
        for step in range(TRAIN_STEPS):
            lr_now = _cosine_lr(step)
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            model.train()
            optimizer.zero_grad()
            aggregate = {}
            for row in train_samples[step * GRAD_ACCUM : (step + 1) * GRAD_ACCUM]:
                x = torch.tensor([row["ids"]], device="cuda")
                lengths = torch.tensor([row["prompt_len"]], device="cuda")
                loss, metrics = distill_step(
                    model, ctx, x, lengths, torch.ones_like(x),
                    kl_weight=KL_WEIGHT, ce_weight=CE_WEIGHT, cov_weight=COV_WEIGHT,
                )
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite loss at step {step + 1}")
                (loss / GRAD_ACCUM).backward()
                for k, v in metrics.items():
                    aggregate[k] = aggregate.get(k, 0.0) + v / GRAD_ACCUM
            for n, router in routers:
                for field, w in (("selector", router.selector.selector_weight), ("token", router.token_weight)):
                    if w.grad is None or not torch.isfinite(w.grad).all():
                        raise ValueError(f"Missing/nonfinite {field} grad at {n}")
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0, error_if_nonfinite=True).item())
            optimizer.step()
            step_num = step + 1
            history.append({"step": step_num, "lr": lr_now, "grad_norm": grad_norm, **aggregate})
            if step_num == 1 or step_num % 20 == 0:
                print(name, "w8", history[-1], flush=True)
            if step_num in (40, 80):
                ckpt_path = out_dir / f"step_{step_num:03d}.safetensors"
                export_adapter(model, str(ckpt_path), metadata={
                    "source_model": repo, "source_revision": revision,
                    "lora_alpha": LORA_ALPHA, "step": step_num,
                    "window": {"window_size": WINDOW_SIZE, "selector_alpha": 32.0, "token_router_scale": 2.0,
                               "router_training": "full", "adapt_resident_mlp": True},
                })
                checkpoints[str(step_num)] = {"path": str(ckpt_path), "disjoint_64": eval_disjoint()}
                print(name, f"eval_w8_{step_num}", checkpoints[str(step_num)]["disjoint_64"], flush=True)
                volume.commit()

        final_ckpt = out_dir / "step_080.safetensors"
        saved = load_file(str(final_ckpt))
        with torch.no_grad():
            for _, p in params:
                p.zero_()
        load_adapter(model, str(final_ckpt))
        result["reload_parameters_exact"] = all(torch.equal(p.detach().cpu(), saved[n]) for n, p in params)

        # 3. 8-prompt 96-token test suite on W=8 step_080.
        model.eval()
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
        result["test_suite_w8_step80"] = {
            "mean_distinct_4gram": sum(r["distinct_4gram_ratio"] for r in test_rows) / len(test_rows),
            "eos_stop_count": sum(1 for r in test_rows if r["stopped_eos"]),
            "rows": test_rows,
        }
        result["checkpoints"] = checkpoints
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
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        volume.commit()
    return json.loads(json.dumps(result))


@app.local_entrypoint()
def main(output: str = "reports/full_router_h100_2026-11-27/w8_diag", model: str = "all"):
    if model != "all" and model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    phase1 = json.loads(Path("reports/full_router_h100_2026-11-27/eval_phase1/summary.json").read_text())
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    names = list(MODELS) if model == "all" else [model]
    for name in names:
        summary[name] = run_w8_diag.remote(name, phase1[name]["eval"]["disjoint_prompts"])
        summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        k: {
            "untrained_w8": v.get("untrained_w8"),
            "step80": v.get("checkpoints", {}).get("80", {}).get("disjoint_64"),
            "test_d4": v.get("test_suite_w8_step80", {}).get("mean_distinct_4gram"),
            "rollouts": v.get("rollouts"),
        }
        for k, v in summary.items()
    }, indent=2))
