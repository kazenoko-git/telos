"""Modal H100 window-pinned MoE distillation job across Inkling2, LFM2-MoE, Qwen3-MoE, and dense models.

Uploads only the lightweight `telos` Python package (`add_local_python_source("telos")`),
caches Hugging Face checkpoints on a persistent Modal Volume inside datacenter storage,
runs teacher-student distillation on an 80 GB H100, and writes the compact LoRA + window-selector
Safetensors adapter (plus companion JSON metadata for MLX loading) to `/vol/outputs`.
"""

import json
from pathlib import Path

import modal

APP_NAME = "telos-inkling2-w16-distill"
MODEL_ID = "thinkingmachines/Inkling-2-Mini"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("telos-inkling2-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.5",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.28",
        "accelerate>=1.2",
        "datasets>=3.0",
        "sentencepiece",
        "tiktoken",
        "git+https://github.com/huggingface/transformers.git",
    )
    .add_local_python_source("telos")
)


def _format_example(tokenizer, model_type: str, instruction: str, context: str, response: str, max_len: int):
    user = f"{instruction}\n\n{context}" if context else instruction
    if "inkling" in model_type:
        prompt = (
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
            f"<|message_user|><|content_text|>{user}<|end_message|>"
            "<|message_model|><|content_text|>"
        )
        eos = "<|end_message|>"
    elif getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        eos = tokenizer.eos_token or ""
    else:
        prompt = f"User: {user}\n\nAssistant: "
        eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    resp_ids = tokenizer.encode(response + eos, add_special_tokens=False)
    ids = (prompt_ids + resp_ids)[:max_len]
    if len(ids) <= len(prompt_ids) + 4:
        return None
    return ids, len(prompt_ids)


def _load_hf_model(model_id: str, dtype):
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id)
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="cuda"), cfg
    except (KeyError, ValueError):
        model_type = getattr(cfg, "model_type", "")
        if "inkling" in model_type and hasattr(transformers, "Inkling2LoopForConditionalGeneration"):
            cls = transformers.Inkling2LoopForConditionalGeneration
            return cls.from_pretrained(model_id, dtype=dtype, device_map="cuda"), cfg
        raise


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/vol": volume},
    env={"HF_HOME": "/vol/hf_cache"},
)
def distill_inkling_w16(
    *,
    model_id: str = MODEL_ID,
    steps: int = 200,
    batch_size: int = 2,
    grad_accum: int = 4,
    max_len: int = 256,
    window_size: int | None = 16,
    lr: float = 5e-4,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    selector_rank: int = 32,
    dataset_name: str = "databricks/databricks-dolly-15k",
    output_name: str = "inkling2_w16_adapter.safetensors",
):
    import time

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from telos.training.inkling2_window_distill import (
        distill_step,
        export_adapter,
        patch_moe_for_window_distill,
    )

    out_dir = Path("/vol/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model, cfg = _load_hf_model(model_id, torch.bfloat16)
    model_type = str(getattr(cfg, "model_type", "")).lower()

    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ctx = patch_moe_for_window_distill(
        model,
        window_size=window_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        selector_rank=selector_rank,
        require_moe=(window_size is not None),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"Loaded {model_id} ({model_type}) in {time.perf_counter() - t0:.1f}s | trainable params={n_params:,d}")

    ds = load_dataset(dataset_name, split="train")
    samples = []
    for row in ds:
        formatted = _format_example(
            tokenizer,
            model_type,
            row.get("instruction", ""),
            row.get("context", ""),
            row.get("response", ""),
            max_len,
        )
        if formatted is not None:
            samples.append(formatted)
        if len(samples) >= steps * batch_size * grad_accum:
            break
    print(f"Prepared {len(samples)} training sequences")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    cursor = 0
    history = []

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        agg = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "cov": 0.0, "top1_match": 0.0}
        for _ in range(grad_accum):
            batch = [samples[(cursor + i) % len(samples)] for i in range(batch_size)]
            cursor += batch_size
            width = max(len(ids) for ids, _ in batch)
            input_ids = torch.full((batch_size, width), pad_id, dtype=torch.long, device="cuda")
            attn_mask = torch.zeros((batch_size, width), dtype=torch.long, device="cuda")
            prompt_lens = torch.tensor([p_len for _, p_len in batch], dtype=torch.long, device="cuda")
            for row_i, (ids, _) in enumerate(batch):
                input_ids[row_i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device="cuda")
                attn_mask[row_i, :len(ids)] = 1

            loss, metrics = distill_step(model, ctx, input_ids, prompt_lens, attn_mask)
            (loss / grad_accum).backward()
            for k in agg:
                agg[k] += metrics[k] / grad_accum

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append({"step": step, **agg})
        if step == 1 or step % 10 == 0:
            print(
                f"step {step:4d}/{steps} | loss={agg['loss']:.4f} kl={agg['kl']:.4f} "
                f"ce={agg['ce']:.4f} cov={agg['cov']:.4f} top1={agg['top1_match']:.3f}",
                flush=True,
            )

    adapter_path = out_dir / output_name
    meta = {
        "model_type": model_type,
        "source_model": model_id,
        "lora_alpha": lora_alpha,
        "window": (
            {"window_size": window_size, "selector_alpha": 32.0, "token_router_scale": 2.0}
            if window_size is not None else None
        ),
    }
    keys = export_adapter(model, str(adapter_path), metadata=meta)
    (out_dir / f"{output_name}.metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    volume.commit()
    print(f"Saved {len(keys)} adapter tensors to {adapter_path}")
    return {"adapter": str(adapter_path), "final": history[-1], "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}


@app.local_entrypoint()
def main(
    model_id: str = MODEL_ID,
    steps: int = 200,
    batch_size: int = 2,
    grad_accum: int = 4,
    window_size: int = 16,
    output_name: str = "inkling2_w16_adapter.safetensors",
):
    result = distill_inkling_w16.remote(
        model_id=model_id,
        steps=steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        window_size=window_size,
        output_name=output_name,
    )
    print("Modal run finished:", result)
    print(f"Download adapter locally with: modal volume get telos-inkling2-cache outputs/{output_name} ./")
