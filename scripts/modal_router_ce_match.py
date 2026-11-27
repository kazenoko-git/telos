"""Bounded 80-step H100 full-router training tuned to match stock CE/KL on disjoint eval."""
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
SAVE_STEPS = (20, 40, 60, 80)
EVAL_STEPS = (40, 80)
DISJOINT_EVAL_COUNT = 64

app = modal.App("telos-full-router-ce-match")
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
    return {"ids": ids, "prompt_len": len(prompt_ids), "user": user, "category": row.get("category", "")}


def _cosine_lr(step_idx: int) -> float:
    if step_idx < WARMUP_STEPS:
        return LR_MAX * float(step_idx + 1) / float(WARMUP_STEPS)
    progress = float(step_idx - WARMUP_STEPS) / float(max(1, TRAIN_STEPS - WARMUP_STEPS))
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * progress))


@app.function(
    image=image, gpu="H100", cpu=1, memory=8192, timeout=GPU_TIMEOUT,
    retries=0, scaledown_window=2, max_containers=1,
    volumes={"/vol": volume},
    env={"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "HF_ENABLE_PARALLEL_LOADING": "false",
         "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
)
def train_ce_match(name: str, blocked_prompts: list[str], disjoint_prompts: list[str]):
    import torch
    import transformers
    import telos
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from telos.training.inkling2_window_distill import (
        WindowPinnedRouter, distill_step, export_adapter, load_adapter, patch_moe_for_window_distill,
    )

    started = time.monotonic()
    out_dir = Path("/vol/results") / name / "ce_match_80step"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "name": name, "phase": "ce_match_80step",
        "telos": str(telos.__version__), "transformers": str(transformers.__version__),
        "torch": str(torch.__version__), "gpu": str(torch.cuda.get_device_name()),
        "model": MODELS[name][0], "revision": MODELS[name][1],
        "steps": TRAIN_STEPS, "batch_size": 1, "grad_accum": GRAD_ACCUM,
        "window": 16, "max_length": MAX_LENGTH,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
        "lr_max": LR_MAX, "lr_min": LR_MIN, "warmup_steps": WARMUP_STEPS,
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
            result["router_count"] = len(routers)
            result["trainable_parameters"] = sum(p.numel() for _, p in params)

            dolly_512 = json.loads(Path("/vol/dolly.json").read_text())
            dolly_1024 = json.loads(Path("/vol/dolly_1024.json").read_text())
            by_user = {_format_user(r): r for r in dolly_1024}
            disjoint_held = []
            for u in disjoint_prompts:
                row = by_user.get(u)
                if row is None:
                    raise ValueError("Disjoint prompt missing from dolly_1024.json")
                enc = _encode_row(tokenizer, name, row, MAX_LENGTH)
                if enc is None:
                    raise ValueError("Locked disjoint prompt failed encoding")
                disjoint_held.append(enc)
            assert len(disjoint_held) == DISJOINT_EVAL_COUNT

            blocked = set(blocked_prompts) | set(disjoint_prompts)
            train_candidates = dolly_512 + dolly_1024[len(dolly_512) :]
            train_samples = []
            needed = TRAIN_STEPS * GRAD_ACCUM
            for row in train_candidates:
                if _format_user(row) in blocked or len(row.get("response", "").strip()) < 16:
                    continue
                enc = _encode_row(tokenizer, name, row, MAX_LENGTH)
                if enc is not None:
                    train_samples.append(enc)
                    blocked.add(enc["user"])
                if len(train_samples) == needed:
                    break
            if len(train_samples) < needed:
                raise ValueError(f"Only {len(train_samples)} disjoint train examples for {name}")
            result["train_examples"] = len(train_samples)
            result["train_disjoint_verified"] = not (set(r["user"] for r in train_samples) & set(disjoint_prompts))

            def eval_disjoint():
                model.eval()
                metrics = []
                with torch.no_grad():
                    for row in disjoint_held:
                        x = torch.tensor([row["ids"]], device="cuda")
                        lengths = torch.tensor([row["prompt_len"]], device="cuda")
                        mask = torch.ones_like(x)
                        _, m = distill_step(
                            model, ctx, x, lengths, mask,
                            kl_weight=KL_WEIGHT, ce_weight=CE_WEIGHT, cov_weight=COV_WEIGHT,
                        )
                        metrics.append(m)
                return {k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0]}

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
                    mask = torch.ones_like(x)
                    loss, metrics = distill_step(
                        model, ctx, x, lengths, mask,
                        kl_weight=KL_WEIGHT, ce_weight=CE_WEIGHT, cov_weight=COV_WEIGHT,
                    )
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite training loss at step {step + 1}")
                    (loss / GRAD_ACCUM).backward()
                    for k, val in metrics.items():
                        aggregate[k] = aggregate.get(k, 0.0) + val / GRAD_ACCUM
                for n, router in routers:
                    for field, weight in (("selector", router.selector.selector_weight), ("token", router.token_weight)):
                        if weight.grad is None or not torch.isfinite(weight.grad).all():
                            raise ValueError(f"Missing/nonfinite {field} gradient at {n} step {step + 1}")
                    if any(p.grad is not None for p in router.base_gate.parameters()):
                        raise ValueError("Frozen teacher received gradients")
                grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0, error_if_nonfinite=True).item())
                optimizer.step()
                step_num = step + 1
                history.append({"step": step_num, "lr": lr_now, "grad_norm": grad_norm, **aggregate})
                if step_num == 1 or step_num % 10 == 0:
                    print(name, "ce_match", history[-1], flush=True)
                if step_num in SAVE_STEPS:
                    ckpt_path = out_dir / f"step_{step_num:03d}.safetensors"
                    export_adapter(model, str(ckpt_path), metadata={
                        "source_model": repo, "source_revision": revision,
                        "lora_alpha": LORA_ALPHA, "step": step_num,
                        "window": {"window_size": 16, "selector_alpha": 32.0, "token_router_scale": 2.0, "router_training": "full"},
                    })
                    ckpt_info = {"path": str(ckpt_path), "bytes": ckpt_path.stat().st_size}
                    if step_num in EVAL_STEPS:
                        ckpt_info["disjoint_64"] = eval_disjoint()
                        print(name, f"eval_step_{step_num}", ckpt_info["disjoint_64"], flush=True)
                    checkpoints[str(step_num)] = ckpt_info
                    volume.commit()

            rss.mark("after_train")
            final_ckpt = out_dir / f"step_{TRAIN_STEPS:03d}.safetensors"
            saved = load_file(str(final_ckpt))
            with torch.no_grad():
                for _, p in params:
                    p.zero_()
            load_adapter(model, str(final_ckpt))
            result["reload_parameters_exact"] = all(torch.equal(p.detach().cpu(), saved[n]) for n, p in params)
            if not result["reload_parameters_exact"]:
                raise ValueError("Final checkpoint reload parity failed")
            result["history"] = history
            result["checkpoints"] = checkpoints
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
            (out_dir / "result.json").write_text(json.dumps(result, indent=2))
            volume.commit()
    return json.loads(json.dumps(result))


@app.local_entrypoint()
def main(
    output: str = "reports/full_router_h100_2026-11-27/ce_match_80step",
    model: str = "all",
):
    if model != "all" and model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    phase1 = json.loads(Path("reports/full_router_h100_2026-11-27/eval_phase1/summary.json").read_text())
    rc2 = json.loads(Path("reports/full_router_h100_2026-11-27/rc2/summary.json").read_text())
    rc1 = json.loads(Path("reports/full_router_h100_2026-11-27/inkling-rc1-diagnostic.json").read_text())
    blocked = set()
    for entry in list(rc2.values()) + [rc1]:
        split = entry.get("split", {})
        blocked.update(split.get("train_prompts", []))
        blocked.update(split.get("heldout_prompts", []))

    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    names = list(MODELS) if model == "all" else [model]
    for name in names:
        disjoint_prompts = phase1[name]["eval"]["disjoint_prompts"]
        summary[name] = train_ce_match.remote(name, sorted(blocked), disjoint_prompts)
        summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v.get("status") for k, v in summary.items()}, indent=2))
