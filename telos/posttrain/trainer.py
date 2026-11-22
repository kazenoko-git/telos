"""SFT, DPO and GRPO post-training for native Télos checkpoints."""

from __future__ import annotations

import copy
import json
import platform
import os
import random
import hashlib
import math
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from telos.data.tokenizer import load_tokenizer
from telos.eval.runner import load_model_from_checkpoint
from . import grpo
from .resume import PosttrainResume
from telos.training.checkpoint import atomic_write


def _seed_host_rng(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def _atomic_torch_save(state, path):
    atomic_write(path, lambda temporary: torch.save(state, temporary))


def _checkpoint_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        for name in ("checkpoint_final.pt", "model.safetensors"):
            candidate = path / name
            if candidate.exists():
                return candidate
        candidates = [(int(match.group(1)), candidate) for candidate in path.iterdir()
                      if (match := re.fullmatch(r"checkpoint_step_(\d+)\.(?:pt|safetensors)", candidate.name))]
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]
        raise FileNotFoundError(f"No supported checkpoint found in {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _checkpoint_config(path: Path) -> dict[str, Any]:
    config_path = path.parent / "config.json"
    if config_path.is_file():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid checkpoint config JSON: {config_path}") from exc
    if path.suffix in {".pt", ".bin"}:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return state.get("config", {}) if isinstance(state, dict) else {}
    return {}


def _read_records(path: str | Path, method: str, reward="exact") -> list[dict[str, Any]]:
    required = {
        "sft": {"prompt", "response"},
        "dpo": {"prompt", "chosen", "rejected"},
        "grpo": {"prompt", "answer"} if reward == "exact" else {"prompt"},
    }[method]
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"Line {line_number} must be an object with fields: {', '.join(sorted(required))}")
            if not all(isinstance(row[key], str) for key in required):
                raise ValueError(f"Line {line_number} fields {', '.join(sorted(required))} must be strings")
            records.append(row if method == "grpo" else {key: row[key] for key in required})
    if not records:
        raise ValueError(f"No usable records found in {path}")
    return records


def _encode(tokenizer, prompt: str, answer: str, max_length: int, device: torch.device):
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    if bos_id is None or eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must define [BOS], [EOS], and [PAD] tokens")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    answer_ids = tokenizer.encode(answer, add_special_tokens=False).ids
    if max_length < 3:
        raise ValueError("max_length must be at least 3 tokens")
    answer_ids = answer_ids[: max_length - 2]
    prompt_budget = max_length - len(answer_ids) - 2
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
    tokens = [bos_id, *prompt_ids, *answer_ids, eos_id]
    answer_start = 1 + len(prompt_ids)
    labels = [-100] * answer_start + tokens[answer_start:]
    return tokens, labels, pad_id


def _prepare_records(records, tokenizer, max_length, method):
    """Encode once, retaining exactly the original sampling order/length."""
    prepared = []
    for row in records:
        item = dict(row)
        for key in (("response",) if method == "sft" else ("chosen", "rejected")):
            item["_" + key] = _encode(tokenizer, row["prompt"], row[key], max_length, None)
        prepared.append(item)
    return prepared


def _response_rows(batch, key):
    return [{"_encoded": row["_" + key]} for row in batch]


def _collate(rows, tokenizer, max_length: int, device: torch.device, *, fixed=False):
    encoded = [row["_encoded"] if "_encoded" in row else
               _encode(tokenizer, row["prompt"], row["response"], max_length, None) for row in rows]
    width = max_length if fixed else max(len(tokens) for tokens, _, _ in encoded)
    # CPU staging avoids one device allocation/copy per example (and XLA slice graphs).
    input_ids = torch.full((len(rows), width), encoded[0][2], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    for index, (tokens, targets, _) in enumerate(encoded):
        input_ids[index, :len(tokens)] = torch.tensor(tokens)
        labels[index, :len(targets)] = torch.tensor(targets)
    if device.type == "cuda":
        input_ids, labels = input_ids.pin_memory(), labels.pin_memory()
    return (input_ids.to(device, non_blocking=device.type == "cuda"),
            labels.to(device, non_blocking=device.type == "cuda"))


def _adapter_options(finetune, rank, alpha, dropout, targets):
    if finetune not in {"full", "lora", "qlora"}:
        raise ValueError("finetune must be full, lora or qlora")
    if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("LoRA requires rank>0, alpha>0 and 0<=dropout<1")
    if isinstance(targets, str):
        targets = [target.strip() for target in targets.split(",") if target.strip()]
    return dict(finetune=finetune, lora_rank=rank, lora_alpha=alpha,
                lora_dropout=dropout, lora_targets=targets)


def _apply_adapters(model, options, backend):
    if getattr(model, "reliability_head", None) is not None:
        if backend == "mlx":
            model.reliability_head.freeze()
        else:
            model.reliability_head.requires_grad_(False)
    if options["finetune"] == "full":
        return
    if backend == "mlx":
        from .mlx_adapters import apply_lora
    else:
        from telos.models.packed import unpack_projections
        unpack_projections(model)
        from .adapters import apply_lora
    apply_lora(model, rank=options["lora_rank"], alpha=options["lora_alpha"],
               dropout=options["lora_dropout"], targets=options["lora_targets"],
               quantized=options["finetune"] == "qlora")


def _export_torch(model):
    from .adapters import merged_state_dict
    return {key: value.detach().cpu() for key, value in merged_state_dict(model).items()}


def _save_torch_adapter(model, path, options):
    if options["finetune"] == "full":
        return
    from .adapters import adapter_state_dict
    _atomic_torch_save({"format": "telos-lora-v1", "options": options,
                        "adapter_state_dict": _to_cpu(adapter_state_dict(model))},
                       path.with_suffix(".adapter.pt"))


def _torch_precision(device, precision):
    if precision not in {"auto", "float32", "bf16", "fp16"}:
        raise ValueError("precision must be auto, float32, bf16 or fp16")
    if device.type == "cpu":
        if precision not in {"auto", "float32"}:
            raise ValueError("CPU post-training supports float32 precision")
        return None
    if precision == "auto":
        from telos.training.hardware import cuda_has_native_bf16
        precision = "bf16" if device.type == "xla" or cuda_has_native_bf16(device) else "fp16"
    if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 was requested but this CUDA device does not support it")
    if device.type == "xla" and precision == "fp16":
        raise ValueError("XLA post-training supports bf16 or float32")
    return {"float32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]


def _autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()


def _sft_loss(logits, labels, device, world_size=1):
    targets = labels[:, 1:].reshape(-1)
    total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets, ignore_index=-100, reduction="sum")
    count = (targets != -100).sum().float()
    if device.type == "xla":
        import torch_xla.core.xla_model as xm
        count = xm.all_reduce(xm.REDUCE_SUM, count) / world_size
    elif torch.distributed.is_initialized():
        torch.distributed.all_reduce(count)
        count /= torch.distributed.get_world_size()
    return total / count.clamp_min(1)


def _pair_logps(model, cids, clabels, rids, rlabels):
    scores = _sequence_logps(model, torch.cat((cids, rids)), torch.cat((clabels, rlabels)))
    return scores.chunk(2)


class _CudaParallelModel(torch.nn.DataParallel):
    """Expose native model metadata while scattering forward batches."""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def _cuda_device_ids(devices, available):
    count = available if devices == "auto" else int(devices)
    if count < 1 or count > available:
        raise ValueError(f"Requested {count} CUDA devices but only {available} are available")
    return list(range(count))


def _torch_token_logps(model, input_ids, labels):
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    raw_model = getattr(model, "_orig_mod", model)
    raw_model = getattr(raw_model, "module", raw_model)
    can_sparse_project = (
        input_ids.device.type != "xla"
        and model is raw_model
        and hasattr(raw_model, "hidden_states")
        and hasattr(raw_model, "output_projection")
    )
    if can_sparse_project:
        hidden = raw_model.hidden_states(input_ids, mask_override=True)[:, :-1]
        out = torch.zeros(targets.shape, dtype=torch.float32, device=input_ids.device)
        if valid.any():
            valid_logits = raw_model.output_projection(hidden[valid]).float()
            valid_logps = -F.cross_entropy(valid_logits, targets[valid], reduction="none")
            out = out.masked_scatter(valid, valid_logps)
        return out
    logits = model(input_ids, mask_override=True, next_token_only=True).float()
    safe_targets = targets.masked_fill(~valid, 0)
    token_logps = -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), safe_targets.reshape(-1), reduction="none",
    ).reshape(targets.shape)
    return token_logps * valid


def _sequence_logps(model, input_ids, labels):
    return _torch_token_logps(model, input_ids, labels).sum(dim=-1)


def _resolve_backend(name: str) -> str:
    if name != "auto":
        if name not in {"mlx", "cuda", "xla", "cpu"}:
            raise ValueError("hardware must be auto, mlx, cuda, xla, or cpu; PyTorch MPS is not used")
        return name
    from telos.training.xla_utils import is_tpu_environment
    if is_tpu_environment():
        return "xla"
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        try:
            from telos.training.core import metal_usable
            if metal_usable():
                return "mlx"
        except ImportError:
            pass
        raise RuntimeError(
            "Apple Silicon was detected but MLX/Metal is unavailable; install telos-ml[mlx] "
            "or explicitly select --hardware cpu"
        )
    try:
        from telos.training.core import metal_usable
        if metal_usable():
            return "mlx"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _map_pytorch_parameter_names(state_dict):
    """Translate PyTorch tensor names to the current Telos MLX module names."""
    weights = {}
    qkv = {}
    gate_up = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.").removeprefix("_orig_mod.")
        if not isinstance(value, torch.Tensor):
            continue
        array = value.detach().float().cpu()
        if key == "tok_embeddings.weight":
            weights["emb.weight"] = array
        elif key == "output_projection.weight":
            weights["head.weight"] = array
        elif key == "final_norm.weight":
            weights["norm.weight"] = array
        elif key.startswith("layers."):
            parts = key.split(".")
            layer = parts[1]
            if ".attn.q_proj.weight" in key:
                qkv.setdefault(layer, {})["q"] = array
            elif ".attn.k_proj.weight" in key:
                qkv.setdefault(layer, {})["k"] = array
            elif ".attn.v_proj.weight" in key:
                qkv.setdefault(layer, {})["v"] = array
            elif ".attn.out_proj.weight" in key:
                weights[f"layers.{layer}.out.weight"] = array
            elif ".attn_norm.weight" in key:
                weights[f"layers.{layer}.norm1.weight"] = array
            elif ".mlp_norm.weight" in key:
                weights[f"layers.{layer}.norm2.weight"] = array
            elif ".mlp.w1.weight" in key:
                gate_up.setdefault(layer, {})["gate"] = array
            elif ".mlp.v.weight" in key:
                gate_up.setdefault(layer, {})["up"] = array
            elif ".mlp.w2.weight" in key:
                weights[f"layers.{layer}.mlp.w3.weight"] = array
        elif key.startswith("reliability_head."):
            parts = key.split(".")
            if len(parts) == 3:
                weights[f"reliability_head.layers.{parts[1]}.{parts[2]}"] = array

    for layer, projections in qkv.items():
        if set(projections) != {"q", "k", "v"}:
            raise ValueError(f"Incomplete q/k/v projections in PyTorch layer {layer}")
        weights[f"layers.{layer}.qkv_proj.weight"] = torch.cat(
            [projections["q"], projections["k"], projections["v"]], axis=0
        )
    for layer, projections in gate_up.items():
        if set(projections) != {"gate", "up"}:
            raise ValueError(f"Incomplete SwiGLU projections in PyTorch layer {layer}")
        weights[f"layers.{layer}.mlp.gate_up.weight"] = torch.cat(
            [projections["gate"], projections["up"]], axis=0
        )

    if not weights or "emb.weight" not in weights:
        raise ValueError("Could not map PyTorch checkpoint weights into the MLX Telos model")
    return weights


def _mlx_parameters_from_pytorch(state_dict):
    """Translate tensors and convert them to MLX arrays."""
    import mlx.core as mx

    return {key: mx.array(value.numpy()) for key, value in _map_pytorch_parameter_names(state_dict).items()}


def _mlx_token_logps(model, input_ids, labels):
    import mlx.core as mx

    from telos.training.mlx_losses import cross_entropy
    logits = model(input_ids, is_causal=True, next_token_only=True)
    targets = labels[:, 1:]
    valid = (targets != -100).astype(mx.float32)
    safe_targets = mx.maximum(targets, 0)
    return -cross_entropy(logits, safe_targets, reduction="none") * valid


def _mlx_logps(model, input_ids, labels):
    import mlx.core as mx
    return mx.sum(_mlx_token_logps(model, input_ids, labels), axis=-1)


def _mlx_loss(model, input_ids, labels):
    import mlx.core as mx

    logps = _mlx_logps(model, input_ids, labels)
    mask = (labels[:, 1:] != -100).astype(mx.float32)
    return -mx.sum(logps) / mx.maximum(mx.sum(mask), 1.0)


def _mlx_preference_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    import mlx.core as mx

    margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
    # The abs/maximum expansion has the wrong autodiff derivative at zero,
    # exactly where an initially identical policy/reference pair starts.
    return mx.mean(mx.logaddexp(0.0, -margin))


def _posttrain_mlx(source_model, source_backend, checkpoint_path, source_config, records, tokenizer,
                   out_dir, *, method, max_steps, batch_size, learning_rate, beta,
                   max_length, save_every, seed, resume=None, options=None):
    import mlx.core as mx
    import mlx.optimizers as optim
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map
    from telos.models import MLXTelosTransformer

    if source_backend == "mlx":
        model = source_model
    else:
        config = source_model.config
        model = MLXTelosTransformer(
            vocab_size=config.vocab_size, d_model=config.d_model,
            n_layers=config.n_layers, n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads, tied_embeddings=config.tied_embeddings,
            is_causal=True, use_reliability_head=config.use_reliability_head,
            precision="float32",
        )
        state = source_model.state_dict()
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(checkpoint_path), device="cpu")
            if "model_state_dict" in state:
                state = state["model_state_dict"]
        weights = _mlx_parameters_from_pytorch(state)
        expected = {name for name, _ in tree_flatten(model.parameters())}
        missing = expected - weights.keys()
        if missing:
            raise ValueError(f"PyTorch checkpoint is missing MLX model parameters: {sorted(missing)[:8]}")
        model.load_weights(list(weights.items()), strict=False)
    model.is_causal = True
    if options["precision"] == "fp16":
        raise ValueError("MLX post-training supports auto, float32 or bf16 precision")
    if options["precision"] != "auto":
        model.set_dtype(mx.bfloat16 if options["precision"] == "bf16" else mx.float32)
    mx.eval(model.parameters())

    _seed_host_rng(seed)
    mx.random.seed(seed)
    _apply_adapters(model, options, "mlx")
    # Rollouts and policy scoring share the same dropout-free distribution.
    model.train(method != "grpo")
    need_ref = method == "dpo" or (method == "grpo" and beta)
    use_lora_base_ref = need_ref and options["finetune"] in {"lora", "qlora"}
    reference = copy.deepcopy(model) if (need_ref and not use_lora_base_ref) else None
    if reference is not None:
        reference.eval()
    optimizer = optim.AdamW(learning_rate=learning_rate)
    mx.random.seed(seed)
    master_params = None
    if any(value.dtype == mx.bfloat16 for _, value in tree_flatten(model.trainable_parameters())):
        master_params = tree_map(lambda value: value.astype(mx.float32), model.trainable_parameters())
        optimizer.master_params = master_params
    randomizer = random.Random(seed)
    session = PosttrainResume(resume, 'mlx', records, tokenizer,
        dict(method=method, max_steps=max_steps, batch_size=batch_size, learning_rate=learning_rate,
             beta=beta, max_length=max_length, seed=seed, options=options))
    session.restore(model, optimizer, reference, randomizer)
    master_params = getattr(optimizer, 'master_params', master_params)
    final_path = out_dir / "model.safetensors"
    if final_path.resolve() == checkpoint_path.resolve():
        raise ValueError("Output checkpoint must not overwrite the starting checkpoint")
    if resume is None and (final_path.exists() or any(out_dir.glob("checkpoint_step_*.safetensors"))):
        raise FileExistsError(f"Output directory already contains MLX checkpoints: {out_dir}")

    def save(step, final=False):
        path = final_path if final else out_dir / f"checkpoint_step_{step}.safetensors"
        mx.eval(model.parameters(), optimizer.state)
        from telos.training.checkpoint import atomic_write
        from .mlx_adapters import merged_weights, adapter_weights
        atomic_write(path, lambda tmp: mx.save_safetensors(tmp, dict(merged_weights(model))))
        if options["finetune"] != "full":
            atomic_write(path.with_suffix(".adapter.safetensors"),
                         lambda tmp: mx.save_safetensors(tmp, dict(adapter_weights(model))))
            path.with_suffix(".adapter.json").write_text(json.dumps(options, indent=2))
        session.save(path.with_suffix('.resume.npz'), model, optimizer, step, reference, randomizer)
        cfg = copy.deepcopy(source_config)
        cfg.setdefault("model", {}).update({
            "vocab_size": model.vocab_size, "d_model": model.d_model,
            "n_layers": len(model.layers), "n_heads": model.layers[0].n_heads,
            "n_kv_heads": model.layers[0].n_kv_heads,
            "max_seq_len": int(source_config.get("model", {}).get("max_seq_len", max_length)),
            "tied_embeddings": model.tied_embeddings,
            "is_causal": True, "use_reliability_head": model.use_reliability_head,
        })
        cfg["paradigm"] = "ar"
        cfg["posttrain"] = {"method": method, "source_checkpoint": str(checkpoint_path), "step": step, "backend": "mlx", "options": options}
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
        return path

    rl = options.get("grpo")
    if rl:
        prepared, eos, pad = grpo.prepare_prompts(records, tokenizer, max_length, rl["max_new_tokens"])
        reward = grpo.reward_function(rl["reward"])
    else:
        prepared = _prepare_records(records, tokenizer, max_length, method)
    fixed = options["padding"] != "batch"
    from .mlx_batches import collate_mlx, ReferenceScores
    if method == "dpo":
        for index, row in enumerate(prepared):
            for response in ("chosen", "rejected"):
                # Encoding is immutable within this invocation; a row identity
                # avoids duplicating long token/label tuples for the cache.
                row["_" + response + "_key"] = (index, response)

    def dpo_loss(active_model, ids, labels, ref_chosen, ref_rejected):
        c_logps, r_logps = mx.split(_mlx_logps(active_model, ids, labels), 2)
        return _mlx_preference_loss(c_logps, r_logps, ref_chosen, ref_rejected, beta)

    def sample(ids, positions):
        hidden = model.hidden_states(mx.array(ids), is_causal=True)
        last = hidden[mx.arange(len(positions)), mx.array(positions)]
        logits = model.logits_from_hidden(last).astype(mx.float32)
        tokens = mx.random.categorical(logits)
        chosen = mx.take_along_axis(logits, tokens[:, None], axis=-1).squeeze(-1)
        logps = chosen - mx.logsumexp(logits, axis=-1)
        mx.eval(tokens, logps)
        return tokens.tolist(), logps.tolist()

    def grpo_loss(active_model, ids, labels, old, ref, advantages):
        policy = _mlx_token_logps(active_model, ids, labels)
        return grpo.mlx_loss(policy, old, ref, labels, advantages, beta, rl["clip_epsilon"])

    loss_fn = {"sft": _mlx_loss, "dpo": dpo_loss, "grpo": grpo_loss}[method]
    value_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer.init(master_params if master_params is not None else model.trainable_parameters())
    state = [model.state, optimizer.state, mx.random.state]
    if master_params is not None:
        state.append(master_params)

    def update(*args):
        loss, grads = value_and_grad(model, *args)
        if rl:
            grads, _ = optim.clip_grad_norm(grads, 1.0)
        if master_params is not None:
            master_params.update(optimizer.apply_gradients(grads, master_params))
            model.update(tree_map(lambda value: value.astype(mx.bfloat16), master_params))
        else:
            optimizer.update(model, grads)
        return loss

    step_fn = mx.compile(update, inputs=state, outputs=state) if options["compile_step"] else update
    ref_cache = ReferenceScores(
        options["reference_cache_size"] if options["cache_reference"] else 0, batch_size)
    for step in range(session.step + 1, max_steps + 1):
        batch = randomizer.choices(prepared, k=batch_size)
        metrics = ""
        if rl:
            rows, action_logps, advantages, mean_reward, uniform = grpo.collect_rollouts(
                batch, tokenizer, eos, pad, rl["group_size"], rl["max_new_tokens"], sample, reward,
            )
            ids, labels = collate_mlx(rows, max_length, fixed=fixed)
            old = mx.array(grpo.collate_old_logps(rows, action_logps, ids.shape[1]))
            if reference is not None:
                ref = _mlx_token_logps(reference, ids, labels)
            elif use_lora_base_ref:
                from .mlx_adapters import disable_lora_adapters
                with disable_lora_adapters(model):
                    ref = mx.stop_gradient(_mlx_token_logps(model, ids, labels))
            else:
                ref = old
            advantages = mx.array(advantages)
            mx.eval(old, ref, advantages)
            if beta == 0 and uniform >= 1.0:
                loss = mx.array(0.0, dtype=mx.float32)
            else:
                for _ in range(rl["policy_updates"]):
                    loss = step_fn(ids, labels, old, ref, advantages)
                    mx.eval(model.trainable_parameters(), optimizer.state, loss,
                            master_params if master_params is not None else [])
            metrics = f" reward={mean_reward:.4g} uniform_groups={uniform:.1%}"
        else:
            rows = _response_rows(batch, "response") if method == "sft" else (
                _response_rows(batch, "chosen") + _response_rows(batch, "rejected"))
            ids, labels = collate_mlx(rows, max_length, fixed=fixed)
            if method == "sft":
                loss = step_fn(ids, labels)
            else:
                for item, key in zip(rows, [row["_chosen_key"] for row in batch] +
                                          [row["_rejected_key"] for row in batch]):
                    item["_reference_key"] = key
                def _compute_mlx_ref():
                    if reference is not None:
                        return _mlx_logps(reference, ids, labels)
                    from .mlx_adapters import disable_lora_adapters
                    with disable_lora_adapters(model):
                        return mx.stop_gradient(_mlx_logps(model, ids, labels))
                ref = ref_cache.get(rows, ids.shape[1], _compute_mlx_ref)
                ref_c, ref_r = mx.split(ref, 2)
                loss = step_fn(ids, labels, ref_c, ref_r)
            mx.eval(model.trainable_parameters(), optimizer.state, loss,
                    master_params if master_params is not None else [])
        if step == 1 or step % 10 == 0 or step == max_steps:
            print(f"[{method.upper()}/MLX] step {step}/{max_steps} loss={loss.item():.4f}{metrics}", flush=True)
        if save_every and step % save_every == 0 and step != max_steps:
            save(step)
    result = save(max_steps, final=True)
    print(f"Saved MLX post-trained checkpoint: {result}", flush=True)
    return result


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _xla_topology(runtime, expected_devices="auto"):
    """Resolve XLA rank identity without defaulting errors to process zero."""
    rank = int(runtime.global_ordinal())
    device_count = int(runtime.global_runtime_device_count())
    process_count = int(runtime.world_size())
    if device_count < 1 or process_count < 1 or not 0 <= rank < device_count:
        raise RuntimeError(
            f"Invalid PyTorch/XLA topology: rank={rank}, devices={device_count}, processes={process_count}"
        )
    if expected_devices != "auto" and int(expected_devices) != device_count:
        raise RuntimeError(
            f"Requested {expected_devices} XLA devices but runtime exposes {device_count}; refusing partial-rank training"
        )
    return rank, device_count, process_count


def _xla_posttrain_worker(index, arguments):
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    rank, world_size, process_world = _xla_topology(xr, arguments["devices"])
    _seed_host_rng(arguments["seed"] + rank)
    torch.manual_seed(arguments["seed"] + rank)
    xm.set_rng_state(arguments["seed"] + rank)
    device = xm.xla_device()
    checkpoint_path = Path(arguments["checkpoint"])
    source_config = _checkpoint_config(checkpoint_path)
    model, backend, _ = load_model_from_checkpoint(checkpoint_path)
    if backend != "pytorch":
        raise ValueError("XLA post-training requires a native PyTorch checkpoint")
    model_config = source_config.get("model", source_config)
    model.config.max_seq_len = int(model_config.get("max_seq_len", model_config.get("seq_len", model.config.max_seq_len)))
    model.config.is_causal = True
    options = arguments["options"]
    # Every rank must start with identical LoRA factors before reducing gradients.
    torch.manual_seed(arguments["seed"])
    _apply_adapters(model, options, "xla")
    torch.manual_seed(arguments["seed"] + rank)
    model.to(device)
    amp_dtype = _torch_precision(device, options["precision"])
    model.train()
    reference = copy.deepcopy(model).eval() if arguments["method"] == "dpo" else None
    if reference is not None:
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=arguments["learning_rate"])
    tokenizer = load_tokenizer(arguments["tokenizer"])
    records = _read_records(arguments["data"], arguments["method"])
    randomizer = random.Random(arguments["seed"])
    max_length = min(arguments["max_length"], model.config.max_seq_len)
    batch_size = arguments["batch_size"]
    out_dir = Path(arguments["output"])
    session = PosttrainResume(arguments.get('resume'), 'xla', records, tokenizer,
        {k: arguments[k] for k in ('method', 'max_steps', 'batch_size', 'learning_rate', 'beta', 'max_length', 'seed', 'options')},
        rank=rank, world_size=world_size)
    session.restore(model, optimizer, reference, randomizer)
    output_collision = False
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        output_collision = not arguments.get("resume") and ((out_dir / "checkpoint_final.pt").exists() or any(out_dir.glob("checkpoint_step_*.pt")))
    collision = xm.mesh_reduce("telos_posttrain_output_collision", output_collision, any)
    if collision:
        raise FileExistsError(f"Output directory already contains checkpoints: {out_dir}")
    xm.rendezvous("telos_posttrain_output_ready")

    def save(step, final=False):
        xm.mark_step()
        xm.rendezvous(f"telos_posttrain_save_{step}_start")
        if rank == 0:
            config = copy.deepcopy(source_config)
            config.setdefault("model", {}).update({
                "vocab_size": model.config.vocab_size, "d_model": model.config.d_model,
                "n_layers": model.config.n_layers, "n_heads": model.config.n_heads,
                "n_kv_heads": model.config.n_kv_heads,
                "max_seq_len": model.config.max_seq_len, "is_causal": True,
                "packed_projections": model.config.packed_projections,
                "use_reliability_head": model.reliability_head is not None,
            })
            config["paradigm"] = "ar"
            config["posttrain"] = {
                "method": arguments["method"], "backend": "xla", "world_size": world_size,
                "process_world_size": process_world, "step": step,
            }
            path = out_dir / ("checkpoint_final.pt" if final else f"checkpoint_step_{step}.pt")
            _atomic_torch_save({
                "global_step": step,
                "model_state_dict": _export_torch(model),
                "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
                "config": config,
            }, path)
            _save_torch_adapter(model, path, options)
            (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
            print(f"Saved XLA post-trained checkpoint: {path}", flush=True)
        resume_path = out_dir / ("checkpoint_final.resume.pt" if final else f"checkpoint_step_{step}.resume.pt")
        session.save(resume_path, model, optimizer, step, reference, randomizer)
        xm.rendezvous(f"telos_posttrain_save_{step}_done")

    prepared = _prepare_records(records, tokenizer, max_length, arguments["method"])
    ref_cache = {}
    for step in range(session.step + 1, arguments["max_steps"] + 1):
        xm.set_rng_state(arguments["seed"] + step * world_size + rank)
        global_batch = randomizer.choices(prepared, k=batch_size * world_size)
        local_batch = global_batch[rank * batch_size:(rank + 1) * batch_size]
        optimizer.zero_grad(set_to_none=True)
        method = arguments["method"]
        rows = _response_rows(local_batch, "response") if method == "sft" else (
            _response_rows(local_batch, "chosen") + _response_rows(local_batch, "rejected"))
        ids, labels = _collate(rows, tokenizer, max_length, device, fixed=True)
        with _autocast(device, amp_dtype):
            if method == "sft":
                logits = model(ids, mask_override=True, next_token_only=True).float()
                loss = _sft_loss(logits, labels, device, world_size)
            else:
                key = tuple(tuple(row["_encoded"][0]) + tuple(row["_encoded"][1]) for row in rows)
                if not options["cache_reference"] or key not in ref_cache:
                    with torch.no_grad():
                        ref = _sequence_logps(reference, ids, labels)
                    # Materialize outside the policy graph, avoiding its recomputation.
                    xm.mark_step()
                    if options["cache_reference"] and len(ref_cache) < options["reference_cache_size"]:
                        ref_cache[key] = ref.detach()
                else:
                    ref = ref_cache[key]
                policy = _sequence_logps(model, ids, labels)
                ref_c, ref_r = ref.chunk(2)
                pi_c, pi_r = policy.chunk(2)
                loss = -F.logsigmoid(arguments["beta"] * ((pi_c - pi_r) - (ref_c - ref_r))).mean()
        loss.backward()
        xm.reduce_gradients(optimizer)
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        log_step = step == 1 or step % 10 == 0 or step == arguments["max_steps"]
        if log_step:
            # Identical collective order on every rank; no host loss read on other updates.
            mean_loss = xm.all_reduce(xm.REDUCE_SUM, loss.detach()) / world_size
        xm.mark_step()
        if log_step:
            mean_loss_value = mean_loss.item()
            if rank == 0:
                print(f"[{method.upper()}/XLA] step {step}/{arguments['max_steps']} loss={mean_loss_value:.4f} ranks={world_size}", flush=True)
        if arguments["save_every"] and step % arguments["save_every"] == 0 and step != arguments["max_steps"]:
            save(step)
    save(arguments["max_steps"], final=True)


def _posttrain_xla(arguments):
    # Always launch the complete XLA runtime. Each worker validates its ordinal;
    # there is no rank-zero fallback if runtime identity cannot be read.
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    from telos.training.xla_utils import clean_tpu_environment
    clean_tpu_environment()
    import torch_xla
    if hasattr(torch_xla, "launch"):
        torch_xla.launch(_xla_posttrain_worker, args=(arguments,))
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_xla_posttrain_worker, args=(arguments,), nprocs=None)
    return Path(arguments["output"]) / "checkpoint_final.pt"


def _posttrain(
    checkpoint: str | Path,
    data: str | Path,
    output: str | Path,
    *,
    method: str = "sft",
    tokenizer_path: str | Path | None = None,
    hardware: str = "auto",
    devices: str | int = "auto",
    max_steps: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    beta: float = 0.1,
    max_length: int | None = None,
    save_every: int = 250,
    seed: int = 42,
    resume: str | Path | None = None,
    finetune: str = "full",
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    lora_targets: list[str] | str | None = None,
    precision: str = "auto",
    compile_step: bool = True,
    padding: str = "auto",
    cache_reference: bool = True,
    reference_cache_size: int = 128,
    group_size: int = 4,
    max_new_tokens: int = 32,
    policy_updates: int = 2,
    clip_epsilon: float = 0.2,
    reward: str = "exact",
) -> Path:
    """Run SFT, DPO or online GRPO and export a native checkpoint."""
    method = method.lower()
    if method not in {"sft", "dpo", "grpo"}:
        raise ValueError("method must be 'sft', 'dpo' or 'grpo'")
    if (max_steps < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0
            or not math.isfinite(beta) or beta < 0 or (beta == 0 and method != "grpo") or save_every < 0):
        raise ValueError("Invalid steps, batch size, learning rate, beta, or save cadence")

    options = _adapter_options(finetune, lora_rank, lora_alpha, lora_dropout, lora_targets)
    if padding not in {"auto", "batch", "fixed"}:
        raise ValueError("padding must be auto, batch or fixed")
    if precision not in {"auto", "float32", "bf16", "fp16"} or reference_cache_size < 0:
        raise ValueError("Invalid precision or reference cache size")
    options.update(precision=precision, compile_step=compile_step, padding=padding,
                   cache_reference=cache_reference, reference_cache_size=reference_cache_size)
    backend_name = _resolve_backend(hardware)
    if method == "grpo":
        distributed_cuda = (
            backend_name == "cuda" and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        if backend_name == "xla" or distributed_cuda or (devices != "auto" and int(devices) != 1):
            raise ValueError("GRPO supports single-device MLX, CUDA and CPU")
        if finetune != "full" and lora_dropout:
            raise ValueError("GRPO requires zero LoRA dropout")
        if (group_size < 2 or max_new_tokens < 1 or policy_updates < 1
                or not 0 < clip_epsilon < 1 or (max_length is not None and max_length <= max_new_tokens)):
            raise ValueError("Invalid GRPO group size, completion budget, policy updates or clip epsilon")
        options["grpo"] = dict(group_size=group_size, max_new_tokens=max_new_tokens,
                               policy_updates=policy_updates, clip_epsilon=clip_epsilon, reward=reward)
    checkpoint_path = _checkpoint_file(checkpoint)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    options["base_sha256"] = digest.hexdigest()
    options["base_checkpoint"] = str(checkpoint_path.resolve())
    source_config = _checkpoint_config(checkpoint_path)
    options["quantization"] = "affine_int4_group64" if backend_name == "mlx" and finetune == "qlora" else ("nf4_block64" if finetune == "qlora" else None)
    print(f"[POSTTRAIN] method={method.upper()} backend={backend_name}", flush=True)
    if devices != "auto" and int(devices) < 1:
        raise ValueError("devices must be 'auto' or a positive integer")
    if backend_name not in {"xla", "cuda"} and devices != "auto" and int(devices) != 1:
        raise ValueError(f"{backend_name} post-training is single-device; use devices=1")
    if backend_name == "xla":
        if checkpoint_path.suffix == ".safetensors":
            raise ValueError("XLA post-training currently requires a native PyTorch .pt checkpoint")
        model_config = source_config.get("model", source_config)
        paradigm = str(source_config.get("paradigm", model_config.get("paradigm", ""))).lower()
        if paradigm and paradigm not in {"ar", "corosred"}:
            raise ValueError(f"Post-training requires a causal AR checkpoint; checkpoint paradigm is {paradigm!r}")
        data_config = source_config.get("data", {}) or {}
        tokenizer_ref = tokenizer_path or data_config.get("tokenizer")
        if tokenizer_ref is not None and not Path(tokenizer_ref).is_file():
            raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_ref}")
        _read_records(data, method)
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        if resume is None and ((output_path / "checkpoint_final.pt").exists() or any(output_path.glob("checkpoint_step_*.pt"))):
            raise FileExistsError(f"Output directory already contains checkpoints: {output_path}")
        arguments = {
            "checkpoint": str(checkpoint_path), "data": str(data), "output": str(output_path),
            "tokenizer": str(tokenizer_ref) if tokenizer_ref else None,
            "method": method, "devices": str(devices), "max_steps": max_steps,
            "batch_size": batch_size, "learning_rate": learning_rate, "beta": beta,
            "max_length": max_length or int(model_config.get("max_seq_len", model_config.get("seq_len", 512))),
            "save_every": save_every, "seed": seed, "resume": str(resume) if resume else None,
            "options": options,
        }
        return _posttrain_xla(arguments)

    model, backend, _ = load_model_from_checkpoint(checkpoint_path)
    model_config = source_config.get("model", source_config)
    paradigm = str(source_config.get("paradigm", model_config.get("paradigm", ""))).lower()
    if paradigm and paradigm not in {"ar", "corosred"}:
        raise ValueError(f"Post-training requires a causal AR checkpoint; checkpoint paradigm is {paradigm!r}")
    configured_length = int(model_config.get("max_seq_len", model_config.get("seq_len", getattr(getattr(model, "config", None), "max_seq_len", 512))))
    model_config.setdefault("max_seq_len", configured_length)
    if backend_name == "mlx":
        if backend == "mlx":
            model_config.setdefault("vocab_size", model.vocab_size)
            model_config.setdefault("d_model", model.d_model)
            model_config.setdefault("n_layers", len(model.layers))
            model_config.setdefault("n_heads", model.layers[0].n_heads)
            model_config.setdefault("n_kv_heads", model.layers[0].n_kv_heads)
            model_config.setdefault("max_seq_len", configured_length)
            model_config.setdefault("is_causal", True)
        else:
            model.config.max_seq_len = configured_length
        data_config = source_config.get("data", {}) or {}
        tokenizer_ref = tokenizer_path or data_config.get("tokenizer")
        if tokenizer_ref is not None and not Path(tokenizer_ref).is_file():
            raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_ref}")
        tokenizer = load_tokenizer(tokenizer_ref)
        vocab_size = model.vocab_size if backend == "mlx" else model.config.vocab_size
        if len(tokenizer.get_vocab()) != vocab_size:
            raise ValueError(f"Tokenizer has {len(tokenizer.get_vocab())} entries but checkpoint expects {vocab_size}")
        max_length = min(max_length or configured_length, configured_length)
        records = _read_records(data, method, reward)
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        if resume is None and ((out_dir / "model.safetensors").exists() or any(out_dir.glob("checkpoint_step_*.safetensors"))):
            raise FileExistsError(f"Output directory already contains MLX checkpoints: {out_dir}")
        if backend != "mlx" and backend != "pytorch":
            raise ValueError(f"MLX post-training cannot load checkpoint backend {backend!r}")
        return _posttrain_mlx(
            model, backend, checkpoint_path, source_config, records, tokenizer, out_dir,
            method=method, max_steps=max_steps, batch_size=batch_size,
            learning_rate=learning_rate, beta=beta, max_length=max_length,
            save_every=save_every, seed=seed, resume=resume, options=options,
        )

    if backend != "pytorch":
        raise ValueError("CUDA/CPU post-training requires a native PyTorch checkpoint")
    if backend_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model.config.max_seq_len = configured_length
    model.config.is_causal = True
    ddp = backend_name == "cuda" and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if ddp else 0
    world_size = torch.distributed.get_world_size() if ddp else 1
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if backend_name == "cuda" else "cpu")
    if ddp and devices != "auto" and int(devices) != world_size:
        raise ValueError("Requested devices must match the torchrun world size")
    # Adapt on CPU so NF4 quantization never synchronizes the training accelerator.

    data_config = source_config.get("data", {}) or {}
    tokenizer_ref = tokenizer_path or data_config.get("tokenizer")
    if tokenizer_ref is not None and not Path(tokenizer_ref).is_file():
        raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_ref}")
    tokenizer = load_tokenizer(tokenizer_ref)
    if len(tokenizer.get_vocab()) != model.config.vocab_size:
        raise ValueError(
            f"Tokenizer has {len(tokenizer.get_vocab())} entries but checkpoint expects {model.config.vocab_size}; "
            "use the tokenizer from pretraining"
        )
    max_length = min(max_length or model.config.max_seq_len, model.config.max_seq_len)
    records = _read_records(data, method, reward)
    _seed_host_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    _apply_adapters(model, options, backend_name)
    model.to(device)
    amp_dtype = _torch_precision(device, precision)
    need_ref = method == "dpo" or (method == "grpo" and beta)
    use_lora_base_ref = need_ref and options["finetune"] in {"lora", "qlora"} and not ddp
    reference = None
    if need_ref and not use_lora_base_ref:
        reference = copy.deepcopy(model).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
        # Frozen reference has no gradient synchronization and stays rank-local.
        device_ids = [device.index]
    elif backend_name == "cuda":
        device_ids = _cuda_device_ids(1 if method == "grpo" else devices, torch.cuda.device_count())
        if len(device_ids) > 1:
            model = _CudaParallelModel(model, device_ids=device_ids)
            if reference is not None:
                reference = _CudaParallelModel(reference, device_ids=device_ids)
            batch_size *= len(device_ids)
    model.train(method != "grpo")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate,
                                  **({"fused": True} if device.type == "cuda" else {}))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "checkpoint_final.pt"
    if final_path.resolve() == checkpoint_path.resolve():
        raise ValueError("Output checkpoint must not overwrite the starting checkpoint")
    collision = resume is None and (final_path.exists() or any(out_dir.glob("checkpoint_step_*.pt")))
    if ddp:
        flag = torch.tensor(int(collision), device=device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        collision = bool(flag.item())
    if collision:
        raise FileExistsError(f"Output directory already contains checkpoints: {out_dir}")

    randomizer = random.Random(seed)
    session = PosttrainResume(resume, backend_name, records, tokenizer,
        dict(method=method, max_steps=max_steps, batch_size=batch_size, learning_rate=learning_rate,
             beta=beta, max_length=max_length, seed=seed, options=options),
        rank=rank, world_size=world_size if ddp else (len(device_ids) if backend_name == "cuda" else 1))
    if ddp:
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
    session.restore(model, optimizer, reference, randomizer, scaler=scaler)

    def save(step: int, final: bool = False):
        output_path = out_dir / ("checkpoint_final.pt" if final else f"checkpoint_step_{step}.pt")
        backbone = getattr(model, "module", model)
        saved_config = copy.deepcopy(source_config)
        saved_config.setdefault("model", {}).update({
            "vocab_size": backbone.config.vocab_size,
            "d_model": backbone.config.d_model,
            "n_layers": backbone.config.n_layers,
            "n_heads": backbone.config.n_heads,
            "n_kv_heads": backbone.config.n_kv_heads,
            "max_seq_len": backbone.config.max_seq_len,
            "packed_projections": backbone.config.packed_projections,
            "is_causal": True,
            "use_reliability_head": backbone.reliability_head is not None,
        })
        saved_config["paradigm"] = "ar"
        if ddp:
            torch.distributed.barrier()
        if rank == 0:
            _atomic_torch_save({
                "global_step": step,
                "model_state_dict": _export_torch(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": saved_config,
                "posttrain": {"method": method, "source_checkpoint": str(checkpoint_path), "step": step, "options": options},
            }, output_path)
            _save_torch_adapter(model, output_path, options)
        session.save(output_path.with_suffix('.resume.pt'), model, optimizer, step, reference, randomizer, scaler=scaler)
        if ddp:
            torch.distributed.barrier()
        return output_path

    rl = options.get("grpo")
    if rl:
        prepared, eos, pad = grpo.prepare_prompts(records, tokenizer, max_length, rl["max_new_tokens"])
        reward_fn = grpo.reward_function(rl["reward"])
    else:
        prepared = _prepare_records(records, tokenizer, max_length, method)
    fixed = padding == "fixed" or (padding == "auto" and compile_step and device.type == "cuda")
    active_model = torch.compile(model) if compile_step and device.type == "cuda" else model
    ref_cache = {}

    def update(loss):
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        scaler.step(optimizer)
        scaler.update()

    @torch.no_grad()
    def sample(ids, positions):
        with _autocast(device, amp_dtype):
            logits = model(torch.as_tensor(ids, device=device, dtype=torch.long), mask_override=True,
                           logit_positions=torch.tensor(positions, device=device))
        logps = logits.float().log_softmax(dim=-1)
        tokens = torch.multinomial(logps.exp(), 1)
        return tokens.squeeze(-1).tolist(), logps.gather(-1, tokens).squeeze(-1).tolist()

    for step in range(session.step + 1, max_steps + 1):
        global_batch = randomizer.choices(prepared, k=batch_size * world_size)
        batch = global_batch[rank * batch_size:(rank + 1) * batch_size]
        metrics = ""
        if rl:
            rows, action_logps, advantages, mean_reward, uniform = grpo.collect_rollouts(
                batch, tokenizer, eos, pad, rl["group_size"], rl["max_new_tokens"], sample, reward_fn,
            )
            input_ids, labels = _collate(rows, tokenizer, max_length, device, fixed=fixed)
            advantages = torch.as_tensor(advantages, device=device)
            old = torch.as_tensor(grpo.collate_old_logps(rows, action_logps, input_ids.shape[1]), device=device)
            with torch.no_grad(), _autocast(device, amp_dtype):
                if reference is not None:
                    ref = _torch_token_logps(reference, input_ids, labels)
                elif use_lora_base_ref:
                    from .adapters import disable_lora_adapters
                    was_training = model.training
                    model.eval()
                    with disable_lora_adapters(model):
                        ref = _torch_token_logps(model, input_ids, labels)
                    model.train(was_training)
                else:
                    ref = old
            if beta == 0 and uniform >= 1.0:
                loss = torch.zeros((), device=device)
            else:
                for _ in range(rl["policy_updates"]):
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(device, amp_dtype):
                        policy = _torch_token_logps(active_model, input_ids, labels)
                        loss = grpo.torch_loss(policy, old, ref, labels, advantages, beta, rl["clip_epsilon"])
                    update(loss)
            metrics = f" reward={mean_reward:.4g} uniform_groups={uniform:.1%}"
        else:
            optimizer.zero_grad(set_to_none=True)
            rows = _response_rows(batch, "response") if method == "sft" else (
                _response_rows(batch, "chosen") + _response_rows(batch, "rejected"))
            input_ids, labels = _collate(rows, tokenizer, max_length, device, fixed=fixed)
            with _autocast(device, amp_dtype):
                if method == "sft":
                    logits = active_model(input_ids, mask_override=True, next_token_only=True).float()
                    loss = _sft_loss(logits, labels, device, world_size)
                else:
                    key = tuple(tuple(row["_encoded"][0]) + tuple(row["_encoded"][1]) for row in rows)
                    if not cache_reference or key not in ref_cache:
                        with torch.no_grad():
                            if reference is not None:
                                ref = _sequence_logps(reference, input_ids, labels)
                            else:
                                from .adapters import disable_lora_adapters
                                was_training = model.training
                                model.eval()
                                with disable_lora_adapters(model):
                                    ref = _sequence_logps(model, input_ids, labels)
                                model.train(was_training)
                        if cache_reference and len(ref_cache) < reference_cache_size:
                            ref_cache[key] = ref.detach()
                    else:
                        ref = ref_cache[key]
                    policy = _sequence_logps(active_model, input_ids, labels)
                    ref_chosen, ref_rejected = ref.chunk(2)
                    policy_chosen, policy_rejected = policy.chunk(2)
                    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
                    loss = -F.logsigmoid(logits).mean()
            update(loss)
        if rank == 0 and (step == 1 or step % 10 == 0 or step == max_steps):
            print(f"[{method.upper()}] step {step}/{max_steps} loss={loss.item():.4f}{metrics}", flush=True)
        if save_every and step % save_every == 0 and step != max_steps:
            save(step)

    result = save(max_steps, final=True)
    if rank == 0:
        print(f"Saved post-trained checkpoint: {result}", flush=True)
    return result


def posttrain(*args, **kwargs):
    """Post-training entry point; SFT/DPO can use one CUDA device per torchrun process."""
    owned_group = False
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1 and "LOCAL_RANK" in os.environ
    if distributed:
        if kwargs.get("method", "sft").lower() == "grpo":
            raise ValueError("GRPO supports single-device MLX, CUDA and CPU")
        backend = _resolve_backend(kwargs.get("hardware", "auto"))
        if backend != "cuda":
            raise ValueError("torchrun post-training requires the CUDA backend; XLA uses its native launcher")
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if not dist.is_initialized():
            dist.init_process_group("nccl")
            owned_group = True
    try:
        return _posttrain(*args, **kwargs)
    finally:
        if owned_group:
            torch.distributed.destroy_process_group()
