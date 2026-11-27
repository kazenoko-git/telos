"""Bounded H100 full-router pilots using the published Telos prerelease."""
import json
from pathlib import Path
import time

import modal

VERSION = "1.9.1rc1"
TRANSFORMERS_REVISION = "8e094f8970a60ca83052e2c199783b247b8fc13e"
MODELS = {
    "inkling": ("thinkingmachines/Inkling-2-Mini", "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"),
    "qwen": ("Qwen/Qwen3.5-35B-A3B", "2508b2437e382c605eb0fc52b83f5545960f6b4a"),
    "gemma": ("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"),
}
# Upper bound per H100 call, including requested CPU and RAM, rounded up.
DOLLARS_PER_SECOND = 0.0016
GPU_TIMEOUT = 1200
app = modal.App("telos-full-router-pilots")
volume = modal.Volume.from_name("telos-router-pilots", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"telos-ml=={VERSION}", "accelerate>=1.2", "sentencepiece", "tiktoken",
        f"git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_REVISION}",
    )
)


@app.function(image=image, timeout=1200, cpu=2, memory=8192,
              volumes={"/vol": volume}, env={"HF_HOME": "/vol/hf"}, retries=0)
def prepare(name: str):
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    repo, revision = MODELS[name]
    path = snapshot_download(repo, revision=revision,
                             ignore_patterns=["*.gguf", "*.bin", "*.h5", "*.msgpack", "*.onnx"])
    data = load_dataset("databricks/databricks-dolly-15k", split="train")
    # Split by prompt identity before tokenization to exclude duplicate prompts.
    unique, seen = [], set()
    for row in data:
        prompt = row["instruction"].strip() + "\n" + row.get("context", "").strip()
        if prompt not in seen:
            seen.add(prompt)
            unique.append(dict(row))
        if len(unique) == 512:
            break
    Path("/vol/dolly.json").write_text(json.dumps(unique))
    volume.commit()
    return {"model": repo, "revision": revision, "path": path}


@app.function(image=image, gpu="H100", cpu=4, memory=131072, timeout=GPU_TIMEOUT,
              retries=0, scaledown_window=0, max_containers=1,
              volumes={"/vol": volume}, env={"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1"})
def pilot(name: str):
    import math
    import torch
    import transformers
    import telos
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from telos.training.inkling2_window_distill import (
        WindowPinnedRouter, distill_step, export_adapter, load_adapter,
        patch_moe_for_window_distill,
    )
    from safetensors.torch import load_file

    started = time.monotonic()
    result = {"name": name, "telos": telos.__version__, "transformers": transformers.__version__,
              "model": MODELS[name][0], "revision": MODELS[name][1],
              "steps": 8, "batch_size": 1, "grad_accum": 2, "window": 16,
              "max_length": 128, "gpu": torch.cuda.get_device_name(),
              "gpu_timeout_seconds": GPU_TIMEOUT,
              "cost_upper_bound_usd": GPU_TIMEOUT * DOLLARS_PER_SECOND}
    out = Path("/vol/results") / name
    out.mkdir(parents=True, exist_ok=True)

    def save():
        result["elapsed_seconds"] = time.monotonic() - started
        result["estimated_compute_usd"] = result["elapsed_seconds"] * DOLLARS_PER_SECOND
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        (out / "result.json").write_text(json.dumps(result, indent=2))
        volume.commit()

    try:
        assert telos.__version__ == VERSION
        torch.manual_seed(73)
        repo, revision = MODELS[name]
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)
        cfg = AutoConfig.from_pretrained(repo, revision=revision, local_files_only=True)
        cls = next((getattr(transformers, arch) for arch in cfg.architectures
                    if hasattr(transformers, arch)), AutoModelForCausalLM)
        model = cls.from_pretrained(repo, revision=revision, local_files_only=True,
                                    dtype=torch.bfloat16, device_map="cuda",
                                    attn_implementation="eager")
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ctx = patch_moe_for_window_distill(model, window_size=16, lora_rank=8,
                                          selector_rank=8, router_training="full")
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        routers = [(n, m) for n, m in model.named_modules() if isinstance(m, WindowPinnedRouter)]
        result["router_count"] = len(routers)
        result["trainable_parameters"] = sum(p.numel() for _, p in params)
        result["load_seconds"] = time.monotonic() - started
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if isinstance(pad, list):
            pad = pad[0]
        if pad is None:
            pad = 0
        rows = json.loads(Path("/vol/dolly.json").read_text())
        samples = []
        for row in rows:
            user = row["instruction"] + ("\n\n" + row["context"] if row.get("context") else "")
            if name == "inkling":
                prompt = ("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
                          f"<|message_user|><|content_text|>{user}<|end_message|>"
                          "<|message_model|><|content_text|>")
                eos = "<|end_message|>"
            else:
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                                       tokenize=False, add_generation_prompt=True)
                eos = tokenizer.eos_token or ""
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            ids = (prompt_ids + tokenizer.encode(row["response"] + eos, add_special_tokens=False))[:128]
            if len(ids) >= len(prompt_ids) + 4:
                samples.append((ids, len(prompt_ids), user))
            if len(samples) == 24:
                break
        if len(samples) < 24:
            raise ValueError("Insufficient disjoint short Dolly examples")
        train, held = samples[:16], samples[16:]
        result["split"] = {"train_prompts": [r[2] for r in train], "heldout_prompts": [r[2] for r in held]}

        def batch(row):
            ids, prompt_len, _ = row
            x = torch.tensor([ids], device="cuda")
            return x, torch.tensor([prompt_len], device="cuda"), torch.ones_like(x)

        def evaluate():
            model.eval()
            metrics = []
            predictions = []
            with torch.no_grad():
                for row in held:
                    x, lengths, mask = batch(row)
                    _, m = distill_step(model, ctx, x, lengths, mask)
                    metrics.append(m)
                    ctx.reset_forward("student", lengths)
                    logits = model(x, attention_mask=mask, use_cache=False).logits
                    predictions.append(logits[:, lengths.item()-1:-1].float().cpu())
            return {k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0]}, predictions

        result["before"], _ = evaluate()
        optimizer = torch.optim.AdamW([p for _, p in params], lr=5e-5, weight_decay=0.0)
        history = []
        for step in range(8):
            model.train()
            optimizer.zero_grad()
            aggregate = {}
            for row in train[step*2:step*2+2]:
                x, lengths, mask = batch(row)
                loss, metrics = distill_step(model, ctx, x, lengths, mask)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                (loss / 2).backward()
                for k, value in metrics.items():
                    aggregate[k] = aggregate.get(k, 0) + value / 2
            gradients = {}
            for n, router in routers:
                gradients[n] = {}
                for field, weight in (("selector", router.selector.selector_weight), ("token", router.token_weight)):
                    grad = weight.grad
                    if grad is None or not torch.isfinite(grad).all():
                        raise ValueError(f"Missing/nonfinite {field} gradient at {n}")
                    gradients[n][field] = grad.float().norm().item()
                if any(p.grad is not None for p in router.base_gate.parameters()):
                    raise ValueError("Frozen teacher received gradients")
            if not any(g["selector"] > 0 for g in gradients.values()) or not any(g["token"] > 0 for g in gradients.values()):
                raise ValueError("All routing gradients were zero")
            if step == 0:
                result["first_step_router_gradients"] = gradients
            torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0, error_if_nonfinite=True)
            optimizer.step()
            history.append({"step": step+1, **aggregate})
            result["history"] = history
            print(name, history[-1], flush=True)
            save()
        result["after"], expected_logits = evaluate()
        path = out / "full_router.safetensors"
        export_adapter(model, str(path), metadata={
            "source_model": repo, "source_revision": revision, "lora_alpha": 32.0,
            "window": {"window_size": 16, "router_training": "full"},
        })
        saved = load_file(str(path))
        with torch.no_grad():
            for _, p in params:
                p.zero_()
        load_adapter(model, str(path))
        result["reload_parameters_exact"] = all(torch.equal(p.detach().cpu(), saved[n]) for n, p in params)
        _, actual_logits = evaluate()
        result["reload_logits_exact"] = all(torch.equal(a, b) for a, b in zip(expected_logits, actual_logits))
        if not result["reload_parameters_exact"] or not result["reload_logits_exact"]:
            raise ValueError("Export/reload parity failed")
        result["status"] = "completed"
        save()
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    return result


@app.local_entrypoint()
def main(output: str = "reports/full_router_h100_2026-11-27"):
    # Three bounded calls cost at most $5.76 at the conservative rate above.
    # Preparation and image builds use CPU. No automatic retries or longer runs.
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in MODELS:
        try:
            prep = prepare.remote(name)
            (folder / f"{name}-preparation.json").write_text(json.dumps(prep, indent=2))
            summary[name] = pilot.remote(name)
        except Exception as exc:
            summary[name] = {"status": "failed", "error": str(exc)}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
