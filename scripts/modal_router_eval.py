"""Sequential H100 fresh-base reload verification, disjoint eval, and low-rank baseline."""
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
EVAL_MAX_LENGTH = 256
DISJOINT_EVAL_COUNT = 64
GEN_PROMPTS_COUNT = 4
GEN_MAX_NEW_TOKENS = 48

app = modal.App("telos-full-router-eval")
volume = modal.Volume.from_name("telos-router-pilots", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"telos-ml=={VERSION}", "accelerate>=1.2", "sentencepiece", "tiktoken",
        f"git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_REVISION}",
    )
)


def _rss_kib():
    status = Path("/proc/self/status").read_text()
    rss = hwm = 0
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1])
        elif line.startswith("VmHWM:"):
            hwm = int(line.split()[1])
    return rss, hwm


class RssMonitor:
    """Sample /proc/self/status RSS/HWM alongside resource.getrusage."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak_rss_kib = 0
        self.peak_hwm_kib = 0
        self.stages: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        rss, hwm = _rss_kib()
        self.peak_rss_kib = max(self.peak_rss_kib, rss)
        self.peak_hwm_kib = max(self.peak_hwm_kib, hwm)
        return rss, hwm

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
        rss, _ = self._sample()
        self.stages[stage] = rss / (1024 ** 2)

    def report(self) -> dict:
        rss, hwm = self._sample()
        ru_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "current_rss_gib": rss / (1024 ** 2),
            "sampled_peak_rss_gib": self.peak_rss_kib / (1024 ** 2),
            "proc_vmhwm_gib": max(hwm, self.peak_hwm_kib) / (1024 ** 2),
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


def _load_fresh_model(name: str):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    return model, tokenizer, cfg, repo, revision


def _build_splits(tokenizer, name: str, seen_prompts: set[str]):
    rows = json.loads(Path("/vol/dolly.json").read_text())
    orig_samples = []
    for row in rows:
        enc = _encode_row(tokenizer, name, row, 128)
        if enc is not None:
            orig_samples.append(enc)
        if len(orig_samples) == 24:
            break
    if len(orig_samples) < 24:
        raise ValueError("Insufficient original rc2 Dolly examples")

    # Build a larger, strictly disjoint held-out set from the tail of dolly.json.
    unseen_rows = [
        r for r in rows
        if _format_user(r) not in seen_prompts
        and len(r.get("response", "").strip()) >= 16
        and len(_format_user(r)) <= 400
    ]
    tail_candidates = unseen_rows[-128:]
    disjoint_samples = []
    for row in tail_candidates:
        enc = _encode_row(tokenizer, name, row, EVAL_MAX_LENGTH)
        if enc is not None:
            disjoint_samples.append(enc)
        if len(disjoint_samples) == DISJOINT_EVAL_COUNT:
            break
    if len(disjoint_samples) < DISJOINT_EVAL_COUNT:
        raise ValueError(f"Only {len(disjoint_samples)} disjoint samples encoded for {name}")
    # Pick 6 diverse categories/indices for readable greedy generation probes.
    seen_cats = set()
    gen_samples = []
    for sample in disjoint_samples:
        cat = sample["category"]
        if cat not in seen_cats:
            seen_cats.add(cat)
            gen_samples.append(sample)
        if len(gen_samples) == GEN_PROMPTS_COUNT:
            break
    for sample in disjoint_samples:
        if len(gen_samples) == GEN_PROMPTS_COUNT:
            break
        if sample not in gen_samples:
            gen_samples.append(sample)
    return orig_samples[:16], orig_samples[16:], disjoint_samples, gen_samples


def _eval_stock_unpatched(model, samples: list[dict]):
    import torch
    import torch.nn.functional as F

    model.eval()
    ces = []
    first_logits = []
    with torch.no_grad():
        for idx, row in enumerate(samples):
            x = torch.tensor([row["ids"]], device="cuda")
            mask = torch.ones_like(x)
            logits = model(input_ids=x, attention_mask=mask, use_cache=False).logits[:, :-1, :].float()
            if idx < 4:
                first_logits.append(logits[:, row["prompt_len"] - 1 :].cpu())
            targets = x[:, 1:]
            mask_f = (torch.arange(x.shape[1], device="cuda")[None, 1:] >= row["prompt_len"]).float()
            ce_tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(1, -1)
            ces.append(float(((ce_tok * mask_f).sum() / mask_f.sum().clamp_min(1.0)).item()))
    return {"ce": sum(ces) / len(ces), "kl": 0.0, "top1_match": 1.0, "count": len(ces)}, first_logits


def _eval_patched(model, ctx, samples: list[dict]):
    import torch
    from telos.training.inkling2_window_distill import distill_step

    model.eval()
    metrics = []
    with torch.no_grad():
        for row in samples:
            x = torch.tensor([row["ids"]], device="cuda")
            lengths = torch.tensor([row["prompt_len"]], device="cuda")
            mask = torch.ones_like(x)
            _, m = distill_step(model, ctx, x, lengths, mask)
            metrics.append(m)
    agg = {k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0]}
    agg["count"] = len(metrics)
    return agg


def _student_logits(model, ctx, samples: list[dict]):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for row in samples:
            x = torch.tensor([row["ids"]], device="cuda")
            lengths = torch.tensor([row["prompt_len"]], device="cuda")
            mask = torch.ones_like(x)
            ctx.reset_forward("student", lengths)
            logits = model(input_ids=x, attention_mask=mask, use_cache=False).logits
            out.append(logits[:, row["prompt_len"] - 1 : -1].float().cpu())
    return out


def _teacher_logits(model, ctx, samples: list[dict]):
    import torch
    from telos.training.inkling2_window_distill import _disabled_lora

    model.eval()
    out = []
    with torch.no_grad(), _disabled_lora(model):
        for row in samples:
            x = torch.tensor([row["ids"]], device="cuda")
            lengths = torch.tensor([row["prompt_len"]], device="cuda")
            mask = torch.ones_like(x)
            ctx.reset_forward("teacher", lengths)
            logits = model(input_ids=x, attention_mask=mask, use_cache=False).logits
            out.append(logits[:, row["prompt_len"] - 1 : -1].float().cpu())
    return out


def _greedy_generate(model, tokenizer, stop_ids: set[int], samples: list[dict], *, ctx=None, max_new_tokens: int = GEN_MAX_NEW_TOKENS):
    import torch

    model.eval()
    outputs = []
    with torch.no_grad():
        for row in samples:
            prompt_ids = row["ids"][: row["prompt_len"]]
            cur = list(prompt_ids)
            lengths = torch.tensor([row["prompt_len"]], device="cuda")
            t0 = time.monotonic()
            for _ in range(max_new_tokens):
                x = torch.tensor([cur], device="cuda")
                mask = torch.ones_like(x)
                if ctx is not None:
                    ctx.reset_forward("student", lengths)
                logits = model(input_ids=x, attention_mask=mask, use_cache=False).logits
                nxt = int(logits[0, -1].argmax().item())
                cur.append(nxt)
                if nxt in stop_ids:
                    break
            gen_ids = cur[row["prompt_len"] :]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            outputs.append({
                "category": row["category"],
                "prompt": row["user"],
                "generated_tokens": len(gen_ids),
                "elapsed_seconds": time.monotonic() - t0,
                "text": text,
            })
    return outputs


@app.function(
    image=image, gpu="H100", cpu=1, memory=8192, timeout=GPU_TIMEOUT,
    retries=0, scaledown_window=2, max_containers=1,
    volumes={"/vol": volume},
    env={"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "HF_ENABLE_PARALLEL_LOADING": "false",
         "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
)
def verify_and_eval(name: str, seen_prompts: list[str]):
    import torch
    import transformers
    import telos
    from safetensors.torch import load_file
    from telos.training.inkling2_window_distill import (
        WindowPinnedRouter, load_adapter, patch_moe_for_window_distill,
    )

    started = time.monotonic()
    out_dir = Path("/vol/results") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "name": name, "phase": "verify_and_eval",
        "telos": str(telos.__version__), "transformers": str(transformers.__version__),
        "torch": str(torch.__version__), "gpu": str(torch.cuda.get_device_name()),
        "model": MODELS[name][0], "revision": MODELS[name][1],
        "host_vcpu": 2, "host_ram_gib": 8, "eval_max_length": EVAL_MAX_LENGTH,
    }

    with RssMonitor() as rss:
        try:
            assert telos.__version__ == VERSION
            torch.set_num_threads(2)
            torch.manual_seed(73)
            torch.cuda.reset_peak_memory_stats()
            rss.mark("start")
            model, tokenizer, _, repo, revision = _load_fresh_model(name)
            rss.mark("after_model_load")
            result["load_seconds"] = time.monotonic() - started
            stop_ids = _stop_token_ids(tokenizer, model, name)
            _, orig_held, disjoint_held, gen_samples = _build_splits(tokenizer, name, set(seen_prompts))
            result["disjoint_prompts"] = [r["user"] for r in disjoint_held]
            result["gen_prompts"] = [r["user"] for r in gen_samples]

            # 1. Unpatched stock model evaluation.
            stock_orig, stock_probe_logits = _eval_stock_unpatched(model, orig_held)
            stock_disjoint, _ = _eval_stock_unpatched(model, disjoint_held)
            stock_gen = _greedy_generate(model, tokenizer, stop_ids, gen_samples, ctx=None)
            rss.mark("after_stock")
            result["stock"] = {"orig_8": stock_orig, "disjoint_64": stock_disjoint, "generations": stock_gen}

            # 2. Patch fresh base model for W=16 full-router mode (zero delta = untrained W=16).
            ctx = patch_moe_for_window_distill(
                model, window_size=16, lora_rank=8, selector_rank=8, router_training="full",
            )
            teacher_probe_logits = _teacher_logits(model, ctx, orig_held[:4])
            result["teacher_matches_unpatched_stock"] = all(
                torch.equal(a, b) for a, b in zip(stock_probe_logits, teacher_probe_logits)
            )
            result["teacher_stock_max_abs_diff"] = max(
                float((a - b).abs().max().item()) for a, b in zip(stock_probe_logits, teacher_probe_logits)
            )
            untrained_orig = _eval_patched(model, ctx, orig_held)
            untrained_disjoint = _eval_patched(model, ctx, disjoint_held)
            untrained_gen = _greedy_generate(model, tokenizer, stop_ids, gen_samples, ctx=ctx)
            rss.mark("after_untrained_w16")
            result["untrained_w16"] = {
                "orig_8": untrained_orig, "disjoint_64": untrained_disjoint, "generations": untrained_gen,
            }

            # 3. Fresh-base reload check for saved full_router.safetensors.
            adapter_path = out_dir / "full_router.safetensors"
            saved = load_file(str(adapter_path))
            load_adapter(model, str(adapter_path))
            params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            routers = [(n, m) for n, m in model.named_modules() if isinstance(m, WindowPinnedRouter)]
            param_map = dict(params)
            exact_tensors = set(saved.keys()) == set(param_map.keys()) and all(
                torch.equal(param_map[k].detach().cpu(), saved[k]) for k in saved
            )
            selector_norms = [float(m.selector.selector_weight.detach().float().norm().item()) for _, m in routers]
            token_norms = [float(m.token_weight.detach().float().norm().item()) for _, m in routers]
            base_frozen = all(
                not p.requires_grad
                for _, m in routers
                for p in list(m.base_gate.parameters()) + [m.selector.base_weight]
            )
            expected_logits = _student_logits(model, ctx, orig_held[:4])
            with torch.no_grad():
                for _, p in params:
                    p.zero_()
            load_adapter(model, str(adapter_path))
            reloaded_logits = _student_logits(model, ctx, orig_held[:4])
            logits_exact = all(torch.equal(a, b) for a, b in zip(expected_logits, reloaded_logits))
            result["fresh_reload_check"] = {
                "adapter_path": str(adapter_path),
                "tensor_count": len(saved),
                "exact_tensors": exact_tensors,
                "zero_reload_logits_exact": logits_exact,
                "base_gate_frozen": base_frozen,
                "all_selector_nonzero": all(v > 0 for v in selector_norms),
                "all_token_nonzero": all(v > 0 for v in token_norms),
                "selector_weight_l2_mean": sum(selector_norms) / len(selector_norms),
                "token_weight_l2_mean": sum(token_norms) / len(token_norms),
            }
            if not exact_tensors or not logits_exact or not base_frozen:
                raise ValueError("Fresh-base adapter reload verification failed")

            full_orig = _eval_patched(model, ctx, orig_held)
            full_disjoint = _eval_patched(model, ctx, disjoint_held)
            full_gen = _greedy_generate(model, tokenizer, stop_ids, gen_samples, ctx=ctx)
            rss.mark("after_full_router")
            result["full_router_8step"] = {
                "orig_8": full_orig, "disjoint_64": full_disjoint, "generations": full_gen,
            }
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
            (out_dir / "eval_full_router.json").write_text(json.dumps(result, indent=2))
            volume.commit()
    return json.loads(json.dumps(result))


@app.function(
    image=image, gpu="H100", cpu=1, memory=8192, timeout=GPU_TIMEOUT,
    retries=0, scaledown_window=2, max_containers=1,
    volumes={"/vol": volume},
    env={"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "HF_ENABLE_PARALLEL_LOADING": "false",
         "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
)
def lowrank_baseline(name: str, seen_prompts: list[str]):
    import torch
    import transformers
    import telos
    from safetensors.torch import load_file
    from telos.training.inkling2_window_distill import (
        WindowPinnedRouter, distill_step, export_adapter, load_adapter, patch_moe_for_window_distill,
    )

    started = time.monotonic()
    out_dir = Path("/vol/results") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "name": name, "phase": "lowrank_baseline", "router_training": "adapter",
        "telos": str(telos.__version__), "transformers": str(transformers.__version__),
        "model": MODELS[name][0], "revision": MODELS[name][1],
        "steps": 8, "batch_size": 1, "grad_accum": 2, "window": 16,
        "max_length": 128, "eval_max_length": EVAL_MAX_LENGTH,
        "host_vcpu": 2, "host_ram_gib": 8,
    }

    with RssMonitor() as rss:
        try:
            assert telos.__version__ == VERSION
            torch.set_num_threads(2)
            torch.manual_seed(73)
            torch.cuda.reset_peak_memory_stats()
            model, tokenizer, _, repo, revision = _load_fresh_model(name)
            rss.mark("after_model_load")
            result["load_seconds"] = time.monotonic() - started
            stop_ids = _stop_token_ids(tokenizer, model, name)
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            ctx = patch_moe_for_window_distill(
                model, window_size=16, lora_rank=8, selector_rank=8, router_training="adapter",
            )
            params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            routers = [(n, m) for n, m in model.named_modules() if isinstance(m, WindowPinnedRouter)]
            result["router_count"] = len(routers)
            result["trainable_parameters"] = sum(p.numel() for _, p in params)

            train, orig_held, disjoint_held, gen_samples = _build_splits(tokenizer, name, set(seen_prompts))
            optimizer = torch.optim.AdamW([p for _, p in params], lr=5e-5, weight_decay=0.0)
            history = []
            for step in range(8):
                model.train()
                optimizer.zero_grad()
                aggregate = {}
                for row in train[step * 2 : step * 2 + 2]:
                    x = torch.tensor([row["ids"]], device="cuda")
                    lengths = torch.tensor([row["prompt_len"]], device="cuda")
                    mask = torch.ones_like(x)
                    loss, metrics = distill_step(model, ctx, x, lengths, mask)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    (loss / 2).backward()
                    for k, value in metrics.items():
                        aggregate[k] = aggregate.get(k, 0.0) + value / 2
                torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0, error_if_nonfinite=True)
                optimizer.step()
                history.append({"step": step + 1, **aggregate})
                print(name, "lowrank", history[-1], flush=True)
            rss.mark("after_train")
            result["history"] = history

            adapter_path = out_dir / "lowrank_8step.safetensors"
            export_adapter(model, str(adapter_path), metadata={
                "source_model": repo, "source_revision": revision, "lora_alpha": 32.0,
                "window": {"window_size": 16, "selector_alpha": 32.0, "token_router_scale": 2.0, "router_training": "adapter"},
            })
            saved = load_file(str(adapter_path))
            expected_logits = _student_logits(model, ctx, orig_held[:4])
            with torch.no_grad():
                for _, p in params:
                    p.zero_()
            load_adapter(model, str(adapter_path))
            actual_logits = _student_logits(model, ctx, orig_held[:4])
            result["reload_parameters_exact"] = all(torch.equal(p.detach().cpu(), saved[n]) for n, p in params)
            result["reload_logits_exact"] = all(torch.equal(a, b) for a, b in zip(expected_logits, actual_logits))

            result["lowrank_8step"] = {
                "orig_8": _eval_patched(model, ctx, orig_held),
                "disjoint_64": _eval_patched(model, ctx, disjoint_held),
                "generations": _greedy_generate(model, tokenizer, stop_ids, gen_samples, ctx=ctx),
            }
            rss.mark("after_eval")
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
            (out_dir / "eval_lowrank_8step.json").write_text(json.dumps(result, indent=2))
            volume.commit()
    return json.loads(json.dumps(result))


def _collect_seen_prompts() -> list[str]:
    seen = set()
    rc2 = Path("reports/full_router_h100_2026-11-27/rc2/summary.json")
    rc1 = Path("reports/full_router_h100_2026-11-27/inkling-rc1-diagnostic.json")
    if rc2.is_file():
        data = json.loads(rc2.read_text())
        for entry in data.values():
            split = entry.get("split", {})
            seen.update(split.get("train_prompts", []))
            seen.update(split.get("heldout_prompts", []))
    if rc1.is_file():
        split = json.loads(rc1.read_text()).get("split", {})
        seen.update(split.get("train_prompts", []))
        seen.update(split.get("heldout_prompts", []))
    return sorted(seen)


@app.local_entrypoint()
def main(
    output: str = "reports/full_router_h100_2026-11-27/eval_phase1",
    model: str = "all",
    mode: str = "all",
):
    if model != "all" and model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if mode not in {"eval", "lowrank", "all"}:
        raise ValueError("mode must be eval, lowrank, or all")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    seen = _collect_seen_prompts()
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    names = list(MODELS) if model == "all" else [model]
    for name in names:
        entry = summary.get(name, {})
        if mode in {"eval", "all"}:
            entry["eval"] = verify_and_eval.remote(name, seen)
            summary[name] = entry
            summary_path.write_text(json.dumps(summary, indent=2))
        if mode in {"lowrank", "all"}:
            entry["lowrank"] = lowrank_baseline.remote(name, seen)
            summary[name] = entry
            summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: {kk: vv.get("status") for kk, vv in v.items()} for k, v in summary.items()}, indent=2))
