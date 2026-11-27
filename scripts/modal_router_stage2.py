"""Resumable mixed-domain H100 full-router training from step 80 to step 500."""
import json
import math
from pathlib import Path
import random
import re
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
START_STEP = 80
TARGET_STEP = 500
GRAD_ACCUM = 4
MAX_LENGTH = 256
LORA_RANK = 16
LORA_ALPHA = 32.0
LR_INIT = 3.0e-5
LR_PEAK = 2.0e-4
LR_MIN = 2.0e-5
WARMUP_STEPS = 12
COV_WEIGHT = 0.5
CE_WEIGHT = 0.1
KL_WEIGHT = 1.0
DISJOINT_EVAL_COUNT = 64

CHUNKS = {
    "inkling": [(80, 164), (164, 248), (248, 332), (332, 416), (416, 500)],
    "qwen": [(80, 220), (220, 360), (360, 500)],
    "gemma": [(80, 220), (220, 360), (360, 500)],
}

app = modal.App("telos-full-router-stage2")
volume = modal.Volume.from_name("telos-router-pilots", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"telos-ml=={VERSION}", "accelerate>=1.2", "sentencepiece", "tiktoken",
        f"git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_REVISION}",
    )
)


class RssMonitor:
    """Sample /proc/self/status VmRSS alongside resource.getrusage."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak_rss_kib = 0
        self.stages: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> int:
        rss = 0
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])
                break
        self.peak_rss_kib = max(self.peak_rss_kib, rss)
        return rss

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        self._sample()

    def mark(self, stage: str):
        self.stages[stage] = self._sample() / (1024 ** 2)

    def report(self) -> dict:
        rss = self._sample()
        ru_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "current_rss_gib": rss / (1024 ** 2),
            "sampled_peak_rss_gib": self.peak_rss_kib / (1024 ** 2),
            "rusage_maxrss_gib": ru_max / (1024 ** 2),
            "stages_rss_gib": dict(self.stages),
        }


def _format_user(row: dict) -> str:
    if "user" in row:
        return row["user"]
    return row["instruction"] + ("\n\n" + row["context"] if row.get("context") else "")


def _encode_row(tokenizer, name: str, row: dict, max_length: int):
    user = _format_user(row)
    if name == "inkling":
        prompt = (
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
            f"<|message_user|><|content_text|>{user}<|end_message|>"
            "<|message_model|><|content_text|>"
        )
        eos = "<|end_message|>"
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
        )
        eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = (prompt_ids + tokenizer.encode(row["response"] + eos, add_special_tokens=False))[:max_length]
    if len(ids) < len(prompt_ids) + 4:
        return None
    return {"ids": ids, "prompt_len": len(prompt_ids), "user": user, "source": row.get("source", "dolly")}


def _stage2_lr(step_num: int) -> float:
    rel = step_num - START_STEP - 1
    total = TARGET_STEP - START_STEP
    if rel < WARMUP_STEPS:
        return LR_INIT + (LR_PEAK - LR_INIT) * float(rel + 1) / float(WARMUP_STEPS)
    progress = float(rel - WARMUP_STEPS) / float(max(1, total - WARMUP_STEPS))
    return LR_MIN + 0.5 * (LR_PEAK - LR_MIN) * (1.0 + math.cos(math.pi * progress))


@app.function(
    image=image, timeout=600, cpu=1, memory=8192,
    volumes={"/vol": volume}, env={"HF_HOME": "/vol/hf"}, retries=0,
)
def prepare_mixed_dataset():
    from datasets import load_dataset

    out_path = Path("/vol/stage2_mixed.json")
    prior_1024 = json.loads(Path("/vol/dolly_1024.json").read_text())
    blocked_users = set()
    for r in prior_1024:
        blocked_users.add(r["instruction"] + ("\n\n" + r["context"] if r.get("context") else ""))
        blocked_users.add(r["instruction"].strip() + ("\n\n" + r.get("context", "").strip() if r.get("context", "").strip() else ""))
    seen_users = set(blocked_users)

    def keep(user: str, resp: str) -> bool:
        return (
            user not in seen_users
            and 4 <= len(user) <= 1000
            and len(resp.strip()) >= 16
        )

    buckets: dict[str, list[dict]] = {"dolly": [], "code": [], "math": [], "norobots": []}

    for row in load_dataset("databricks/databricks-dolly-15k", split="train"):
        user = row["instruction"].strip() + ("\n\n" + row.get("context", "").strip() if row.get("context", "").strip() else "")
        resp = row["response"].strip()
        if keep(user, resp):
            seen_users.add(user)
            buckets["dolly"].append({"user": user, "response": resp, "source": "dolly"})
        if len(buckets["dolly"]) == 1400:
            break

    for row in load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train"):
        inst = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        user = inst + (f"\n\n{inp}" if inp else "")
        resp = (row.get("output") or "").strip()
        if keep(user, resp):
            seen_users.add(user)
            buckets["code"].append({"user": user, "response": resp, "source": "python_code"})
        if len(buckets["code"]) == 600:
            break

    for row in load_dataset("openai/gsm8k", "main", split="train"):
        user = (row.get("question") or "").strip()
        resp = re.sub(r"<<[^<>]+>>", "", (row.get("answer") or "")).strip()
        if keep(user, resp):
            seen_users.add(user)
            buckets["math"].append({"user": user, "response": resp, "source": "gsm8k"})
        if len(buckets["math"]) == 600:
            break

    for row in load_dataset("HuggingFaceH4/no_robots", split="train"):
        msgs = row.get("messages") or []
        if len(msgs) != 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
            continue
        user = (msgs[0].get("content") or "").strip()
        resp = (msgs[1].get("content") or "").strip()
        if keep(user, resp):
            seen_users.add(user)
            buckets["norobots"].append({"user": user, "response": resp, "source": "no_robots"})
        if len(buckets["norobots"]) == 600:
            break

    rng = random.Random(73)
    for vals in buckets.values():
        rng.shuffle(vals)
    # Interleave 2 dolly : 1 code : 1 math : 1 norobots for balanced domain coverage.
    mixed = []
    idx = {k: 0 for k in buckets}
    pattern = ("dolly", "code", "dolly", "math", "norobots")
    while len(mixed) < 3000:
        progressed = False
        for key in pattern:
            if idx[key] < len(buckets[key]):
                mixed.append(buckets[key][idx[key]])
                idx[key] += 1
                progressed = True
        if not progressed:
            break

    out_path.write_text(json.dumps(mixed))
    volume.commit()
    return {
        "counts": {k: len(v) for k, v in buckets.items()},
        "mixed_total": len(mixed),
        "path": str(out_path),
    }


@app.function(
    image=image, gpu="H100", cpu=1, memory=8192, timeout=GPU_TIMEOUT,
    retries=0, scaledown_window=2, max_containers=1,
    volumes={"/vol": volume},
    env={"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "HF_ENABLE_PARALLEL_LOADING": "false",
         "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
)
def train_stage2_chunk(name: str, start_step: int, end_step: int, disjoint_prompts: list[str]):
    import torch
    import transformers
    import telos
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from telos.training.inkling2_window_distill import (
        WindowPinnedRouter, distill_step, export_adapter, load_adapter, patch_moe_for_window_distill,
    )

    started = time.monotonic()
    out_dir = Path("/vol/results") / name / "stage2_500step"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "name": name, "phase": f"stage2_{start_step}_to_{end_step}",
        "start_step": start_step, "end_step": end_step,
        "telos": str(telos.__version__), "transformers": str(transformers.__version__),
        "torch": str(torch.__version__), "gpu": str(torch.cuda.get_device_name()),
        "model": MODELS[name][0], "revision": MODELS[name][1],
        "batch_size": 1, "grad_accum": GRAD_ACCUM,
        "window": 16, "max_length": MAX_LENGTH,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
        "kl_weight": KL_WEIGHT, "ce_weight": CE_WEIGHT, "cov_weight": COV_WEIGHT,
        "host_vcpu": 2, "host_ram_gib": 8,
    }

    with RssMonitor() as rss:
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
            rss.mark("after_model_load")
            result["load_seconds"] = time.monotonic() - started
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            ctx = patch_moe_for_window_distill(
                model, window_size=16, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
                selector_rank=8, router_training="full",
            )
            params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            routers = [(n, m) for n, m in model.named_modules() if isinstance(m, WindowPinnedRouter)]

            if start_step == START_STEP:
                resume_adapter = Path("/vol/results") / name / "ce_match_80step" / "step_080.safetensors"
                resume_opt = None
            else:
                resume_adapter = out_dir / f"step_{start_step:03d}.safetensors"
                resume_opt = out_dir / f"optimizer_{start_step:03d}.pt"
            load_adapter(model, str(resume_adapter))
            result["resumed_from"] = str(resume_adapter)

            optimizer = torch.optim.AdamW([p for _, p in params], lr=LR_INIT, weight_decay=0.0)
            if resume_opt is not None and resume_opt.is_file():
                opt_state = torch.load(str(resume_opt), map_location="cpu", weights_only=True)
                optimizer.load_state_dict(opt_state)
                del opt_state
                result["optimizer_resumed"] = True
            else:
                result["optimizer_resumed"] = False

            dolly_1024 = json.loads(Path("/vol/dolly_1024.json").read_text())
            by_user = {_format_user(r): r for r in dolly_1024}
            disjoint_held = []
            for u in disjoint_prompts:
                row = by_user[u]
                enc = _encode_row(tokenizer, name, row, MAX_LENGTH)
                if enc is None:
                    raise ValueError("Locked disjoint prompt failed encoding")
                disjoint_held.append(enc)
            assert len(disjoint_held) == DISJOINT_EVAL_COUNT

            mixed_rows = json.loads(Path("/vol/stage2_mixed.json").read_text())
            blocked = set(disjoint_prompts)
            encoded_all = []
            total_needed = (TARGET_STEP - START_STEP) * GRAD_ACCUM
            for row in mixed_rows:
                if _format_user(row) in blocked:
                    continue
                enc = _encode_row(tokenizer, name, row, MAX_LENGTH)
                if enc is not None:
                    encoded_all.append(enc)
                if len(encoded_all) == total_needed:
                    break
            if len(encoded_all) < total_needed:
                raise ValueError(f"Only {len(encoded_all)}/{total_needed} stage2 examples encoded for {name}")
            result["train_disjoint_verified"] = not (set(r["user"] for r in encoded_all) & blocked)

            history = []
            for step_num in range(start_step + 1, end_step + 1):
                lr_now = _stage2_lr(step_num)
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
                model.train()
                optimizer.zero_grad()
                aggregate = {}
                offset = (step_num - START_STEP - 1) * GRAD_ACCUM
                for row in encoded_all[offset : offset + GRAD_ACCUM]:
                    x = torch.tensor([row["ids"]], device="cuda")
                    lengths = torch.tensor([row["prompt_len"]], device="cuda")
                    mask = torch.ones_like(x)
                    loss, metrics = distill_step(
                        model, ctx, x, lengths, mask,
                        kl_weight=KL_WEIGHT, ce_weight=CE_WEIGHT, cov_weight=COV_WEIGHT,
                    )
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite training loss at step {step_num}")
                    (loss / GRAD_ACCUM).backward()
                    for k, val in metrics.items():
                        aggregate[k] = aggregate.get(k, 0.0) + val / GRAD_ACCUM
                for n, router in routers:
                    for field, weight in (("selector", router.selector.selector_weight), ("token", router.token_weight)):
                        if weight.grad is None or not torch.isfinite(weight.grad).all():
                            raise ValueError(f"Missing/nonfinite {field} gradient at {n} step {step_num}")
                    if any(p.grad is not None for p in router.base_gate.parameters()):
                        raise ValueError("Frozen teacher received gradients")
                grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0, error_if_nonfinite=True).item())
                optimizer.step()
                history.append({"step": step_num, "lr": lr_now, "grad_norm": grad_norm, **aggregate})
                if step_num == start_step + 1 or step_num % 20 == 0 or step_num == end_step:
                    print(name, "stage2", history[-1], flush=True)

            rss.mark("after_train")
            ckpt_path = out_dir / f"step_{end_step:03d}.safetensors"
            export_adapter(model, str(ckpt_path), metadata={
                "source_model": repo, "source_revision": revision,
                "lora_alpha": LORA_ALPHA, "step": end_step,
                "window": {"window_size": 16, "selector_alpha": 32.0, "token_router_scale": 2.0, "router_training": "full"},
            })
            if end_step < TARGET_STEP:
                opt_path = out_dir / f"optimizer_{end_step:03d}.pt"
                torch.save(optimizer.state_dict(), str(opt_path))
                if resume_opt is not None and resume_opt.is_file() and resume_opt != opt_path:
                    resume_opt.unlink()

            if end_step == TARGET_STEP:
                saved = load_file(str(ckpt_path))
                with torch.no_grad():
                    for _, p in params:
                        p.zero_()
                load_adapter(model, str(ckpt_path))
                result["reload_parameters_exact"] = all(torch.equal(p.detach().cpu(), saved[n]) for n, p in params)
                if not result["reload_parameters_exact"]:
                    raise ValueError("Final step-500 reload parity failed")
                if resume_opt is not None and resume_opt.is_file():
                    resume_opt.unlink()

            model.eval()
            eval_metrics = []
            with torch.no_grad():
                for row in disjoint_held:
                    x = torch.tensor([row["ids"]], device="cuda")
                    lengths = torch.tensor([row["prompt_len"]], device="cuda")
                    mask = torch.ones_like(x)
                    _, m = distill_step(
                        model, ctx, x, lengths, mask,
                        kl_weight=KL_WEIGHT, ce_weight=CE_WEIGHT, cov_weight=COV_WEIGHT,
                    )
                    eval_metrics.append(m)
            result["disjoint_64"] = {k: sum(m[k] for m in eval_metrics) / len(eval_metrics) for k in eval_metrics[0]}
            print(name, f"eval_step_{end_step}", result["disjoint_64"], flush=True)
            rss.mark("after_eval")

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
            result["host_memory"] = rss.report()
            (out_dir / f"result_{start_step}_{end_step}.json").write_text(json.dumps(result, indent=2))
            volume.commit()
    return json.loads(json.dumps(result))


@app.local_entrypoint()
def main(
    output: str = "reports/full_router_h100_2026-11-27/stage2_500step",
    model: str = "all",
    prepare_only: bool = False,
):
    if model != "all" and model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    prep_path = folder / "dataset_prep.json"
    if not prep_path.exists():
        prep = prepare_mixed_dataset.remote()
        prep_path.write_text(json.dumps(prep, indent=2))
        print("Prepared mixed dataset:", prep, flush=True)
    if prepare_only:
        return

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
                print(f"Skipping completed {name} chunk {key}", flush=True)
                continue
            model_entry[key] = train_stage2_chunk.remote(name, start_s, end_s, disjoint_prompts)
            summary[name] = model_entry
            summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        m: {k: v.get("disjoint_64") for k, v in chunks.items()}
        for m, chunks in summary.items()
    }, indent=2))
