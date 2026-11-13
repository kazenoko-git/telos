"""SFT and DPO post-training for native Télos PyTorch checkpoints."""

from __future__ import annotations

import copy
import json
import platform
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from telos.data.tokenizer import load_tokenizer
from telos.eval.runner import load_model_from_checkpoint
from .resume import PosttrainResume
from telos.training.checkpoint import atomic_write


def _atomic_torch_save(state, path):
    atomic_write(path, lambda temporary: torch.save(state, temporary))


def _checkpoint_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        for name in ("checkpoint_final.pt", "model.safetensors"):
            candidate = path / name
            if candidate.exists():
                return candidate
        candidates = sorted(path.glob("checkpoint_step_*.pt"))
        if candidates:
            return candidates[-1]
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


def _read_records(path: str | Path, method: str) -> list[dict[str, str]]:
    required = {"prompt", "response"} if method == "sft" else {"prompt", "chosen", "rejected"}
    records: list[dict[str, str]] = []
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
            records.append({key: row[key] for key in required})
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


def _collate(rows, tokenizer, max_length: int, device: torch.device):
    encoded = [_encode(tokenizer, row["prompt"], row["response"], max_length, device) for row in rows]
    pad_id = encoded[0][2]
    width = max(len(tokens) for tokens, _, _ in encoded)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    for index, (tokens, targets, _) in enumerate(encoded):
        input_ids[index, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        labels[index, :len(targets)] = torch.tensor(targets, dtype=torch.long, device=device)
    return input_ids, labels


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


def _sequence_logps(model, input_ids, labels):
    logits = model(input_ids, mask_override=True)[:, :-1].float()
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    safe_targets = targets.masked_fill(~valid, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (token_logps * valid).sum(dim=-1)


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


def _mlx_logps(model, input_ids, labels):
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(input_ids, is_causal=True)[:, :-1, :].astype(mx.float32)
    targets = labels[:, 1:]
    valid = (targets != -100).astype(mx.float32)
    safe_targets = mx.maximum(targets, 0)
    token_logps = mx.take_along_axis(
        nn.log_softmax(logits, axis=-1), safe_targets[..., None], axis=-1
    ).squeeze(-1)
    return mx.sum(token_logps * valid, axis=-1)


def _mlx_loss(model, input_ids, labels):
    import mlx.core as mx

    logps = _mlx_logps(model, input_ids, labels)
    mask = (labels[:, 1:] != -100).astype(mx.float32)
    return -mx.sum(logps) / mx.maximum(mx.sum(mask), 1.0)


def _posttrain_mlx(source_model, source_backend, checkpoint_path, source_config, records, tokenizer,
                   out_dir, *, method, max_steps, batch_size, learning_rate, beta,
                   max_length, save_every, seed, resume=None):
    import mlx.core as mx
    import mlx.optimizers as optim
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
    mx.eval(model.parameters())

    reference = copy.deepcopy(model) if method == "dpo" else None
    optimizer = optim.AdamW(learning_rate=learning_rate)
    mx.random.seed(seed)
    master_params = None
    if any(value.dtype == mx.bfloat16 for _, value in tree_flatten(model.parameters())):
        master_params = tree_map(lambda value: value.astype(mx.float32), model.parameters())
        optimizer.master_params = master_params
    randomizer = random.Random(seed)
    session = PosttrainResume(resume, 'mlx', records, tokenizer,
        dict(method=method, max_steps=max_steps, batch_size=batch_size, learning_rate=learning_rate,
             beta=beta, max_length=max_length, seed=seed))
    session.restore(model, optimizer, reference, randomizer)
    master_params = getattr(optimizer, 'master_params', master_params)
    final_path = out_dir / "model.safetensors"
    if resume is None and (final_path.exists() or any(out_dir.glob("checkpoint_step_*.safetensors"))):
        raise FileExistsError(f"Output directory already contains MLX checkpoints: {out_dir}")

    def save(step, final=False):
        path = final_path if final else out_dir / f"checkpoint_step_{step}.safetensors"
        mx.eval(model.parameters(), optimizer.state)
        from telos.training.checkpoint import atomic_write
        atomic_write(path, lambda tmp: model.save_weights(tmp))
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
        cfg["posttrain"] = {"method": method, "source_checkpoint": str(checkpoint_path), "step": step, "backend": "mlx"}
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
        return path

    value_and_grad = mx.value_and_grad(_mlx_loss)
    for step in range(session.step + 1, max_steps + 1):
        batch = randomizer.choices(records, k=batch_size)
        if method == "sft":
            ids, labels = _collate(batch, tokenizer, max_length, torch.device("cpu"))
            ids, labels = mx.array(ids.numpy(), dtype=mx.int32), mx.array(labels.numpy(), dtype=mx.int32)
            loss, grads = value_and_grad(model, ids, labels)
        else:
            chosen = [{"prompt": row["prompt"], "response": row["chosen"]} for row in batch]
            rejected = [{"prompt": row["prompt"], "response": row["rejected"]} for row in batch]
            cids, clabels = _collate(chosen, tokenizer, max_length, torch.device("cpu"))
            rids, rlabels = _collate(rejected, tokenizer, max_length, torch.device("cpu"))
            cids, clabels = mx.array(cids.numpy(), dtype=mx.int32), mx.array(clabels.numpy(), dtype=mx.int32)
            rids, rlabels = mx.array(rids.numpy(), dtype=mx.int32), mx.array(rlabels.numpy(), dtype=mx.int32)
            ref_c, ref_r = _mlx_logps(reference, cids, clabels), _mlx_logps(reference, rids, rlabels)
            mx.eval(ref_c, ref_r)

            def dpo_loss(active_model, chosen_ids, chosen_labels, rejected_ids, rejected_labels, ref_chosen, ref_rejected):
                c_logps = _mlx_logps(active_model, chosen_ids, chosen_labels)
                r_logps = _mlx_logps(active_model, rejected_ids, rejected_labels)
                preference_margin = beta * ((c_logps - r_logps) - (ref_chosen - ref_rejected))
                log_sigmoid = -mx.maximum(-preference_margin, 0.0) - mx.log1p(mx.exp(-mx.abs(preference_margin)))
                return -mx.mean(log_sigmoid)

            loss, grads = mx.value_and_grad(dpo_loss)(model, cids, clabels, rids, rlabels, ref_c, ref_r)
        if master_params is not None:
            master_params = optimizer.apply_gradients(grads, master_params)
            model.update(tree_map(lambda value: value.astype(mx.bfloat16), master_params))
            optimizer.master_params = master_params
        else:
            optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 1 or step % 10 == 0 or step == max_steps:
            print(f"[{method.upper()}/MLX] step {step}/{max_steps} loss={loss.item():.4f}", flush=True)
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
    model.to(device)
    reference = copy.deepcopy(model).eval() if arguments["method"] == "dpo" else None
    if reference is not None:
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments["learning_rate"])
    tokenizer = load_tokenizer(arguments["tokenizer"])
    records = _read_records(arguments["data"], arguments["method"])
    randomizer = random.Random(arguments["seed"])
    max_length = min(arguments["max_length"], model.config.max_seq_len)
    batch_size = arguments["batch_size"]
    out_dir = Path(arguments["output"])
    session = PosttrainResume(arguments.get('resume'), 'xla', records, tokenizer,
        {k: arguments[k] for k in ('method', 'max_steps', 'batch_size', 'learning_rate', 'beta', 'max_length', 'seed')},
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
                "model_state_dict": _to_cpu(model.state_dict()),
                "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
                "config": config,
            }, path)
            (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
            print(f"Saved XLA post-trained checkpoint: {path}", flush=True)
        resume_path = out_dir / ("checkpoint_final.resume.pt" if final else f"checkpoint_step_{step}.resume.pt")
        session.save(resume_path, model, optimizer, step, reference, randomizer)
        xm.rendezvous(f"telos_posttrain_save_{step}_done")

    for step in range(session.step + 1, arguments["max_steps"] + 1):
        xm.set_rng_state(arguments["seed"] + step * world_size + rank)
        global_batch = randomizer.choices(records, k=batch_size * world_size)
        local_batch = global_batch[rank * batch_size:(rank + 1) * batch_size]
        optimizer.zero_grad(set_to_none=True)
        if arguments["method"] == "sft":
            ids, labels = _collate(local_batch, tokenizer, max_length, device)
            logits = model(ids, mask_override=True)[:, :-1].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        else:
            chosen = [{"prompt": row["prompt"], "response": row["chosen"]} for row in local_batch]
            rejected = [{"prompt": row["prompt"], "response": row["rejected"]} for row in local_batch]
            cids, clabels = _collate(chosen, tokenizer, max_length, device)
            rids, rlabels = _collate(rejected, tokenizer, max_length, device)
            with torch.no_grad():
                ref_c, ref_r = _sequence_logps(reference, cids, clabels), _sequence_logps(reference, rids, rlabels)
            pi_c, pi_r = _sequence_logps(model, cids, clabels), _sequence_logps(model, rids, rlabels)
            loss = -F.logsigmoid(arguments["beta"] * ((pi_c - pi_r) - (ref_c - ref_r))).mean()
        loss.backward()
        xm.reduce_gradients(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        xm.mark_step()
        mean_loss = xm.all_reduce(xm.REDUCE_SUM, loss.detach()) / world_size
        mean_loss_value = mean_loss.item()
        if rank == 0 and (step == 1 or step % 10 == 0 or step == arguments["max_steps"]):
            print(f"[{arguments['method'].upper()}/XLA] step {step}/{arguments['max_steps']} loss={mean_loss_value:.4f} ranks={world_size}", flush=True)
        if arguments["save_every"] and step % arguments["save_every"] == 0 and step != arguments["max_steps"]:
            save(step)
    save(arguments["max_steps"], final=True)


def _posttrain_xla(arguments):
    # Always launch the complete XLA runtime. Each worker validates its ordinal;
    # there is no rank-zero fallback if runtime identity cannot be read.
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    import torch_xla
    if hasattr(torch_xla, "launch"):
        torch_xla.launch(_xla_posttrain_worker, args=(arguments,))
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_xla_posttrain_worker, args=(arguments,), nprocs=None)
    return Path(arguments["output"]) / "checkpoint_final.pt"


def posttrain(
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
) -> Path:
    """Run full-parameter SFT or DPO and write a native Telos checkpoint."""
    method = method.lower()
    if method not in {"sft", "dpo"}:
        raise ValueError("method must be 'sft' or 'dpo'")
    if max_steps < 1 or batch_size < 1 or learning_rate <= 0 or beta <= 0 or save_every < 0:
        raise ValueError("steps, batch size, learning rate, beta, and save cadence must be valid positive values")

    checkpoint_path = _checkpoint_file(checkpoint)
    source_config = _checkpoint_config(checkpoint_path)
    backend_name = _resolve_backend(hardware)
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
        records = _read_records(data, method)
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "model.safetensors").exists() or any(out_dir.glob("checkpoint_step_*.safetensors")):
            raise FileExistsError(f"Output directory already contains MLX checkpoints: {out_dir}")
        if backend != "mlx" and backend != "pytorch":
            raise ValueError(f"MLX post-training cannot load checkpoint backend {backend!r}")
        return _posttrain_mlx(
            model, backend, checkpoint_path, source_config, records, tokenizer, out_dir,
            method=method, max_steps=max_steps, batch_size=batch_size,
            learning_rate=learning_rate, beta=beta, max_length=max_length,
            save_every=save_every, seed=seed, resume=resume,
        )

    if backend != "pytorch":
        raise ValueError("CUDA/CPU post-training requires a native PyTorch checkpoint")
    if backend_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model.config.max_seq_len = configured_length
    model.config.is_causal = True
    device = torch.device("cuda:0" if backend_name == "cuda" else "cpu")
    model.to(device)

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
    records = _read_records(data, method)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    reference = None
    if method == "dpo":
        reference = copy.deepcopy(model).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    if backend_name == "cuda":
        device_ids = _cuda_device_ids(devices, torch.cuda.device_count())
        if len(device_ids) > 1:
            model = _CudaParallelModel(model, device_ids=device_ids)
            if reference is not None:
                reference = _CudaParallelModel(reference, device_ids=device_ids)
            batch_size *= len(device_ids)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "checkpoint_final.pt"
    if final_path.resolve() == checkpoint_path.resolve():
        raise ValueError("Output checkpoint must not overwrite the starting checkpoint")
    if resume is None and (final_path.exists() or any(out_dir.glob("checkpoint_step_*.pt"))):
        raise FileExistsError(f"Output directory already contains checkpoints: {out_dir}")

    session = PosttrainResume(resume, backend_name, records, tokenizer,
        dict(method=method, max_steps=max_steps, batch_size=batch_size, learning_rate=learning_rate,
             beta=beta, max_length=max_length, seed=seed))
    session.restore(model, optimizer, reference)

    def save(step: int, final: bool = False):
        output_path = out_dir / ("checkpoint_final.pt" if final else f"checkpoint_step_{step}.pt")
        saved_config = copy.deepcopy(source_config)
        saved_config.setdefault("model", {}).update({
            "vocab_size": model.config.vocab_size,
            "d_model": model.config.d_model,
            "n_layers": model.config.n_layers,
            "n_heads": model.config.n_heads,
            "n_kv_heads": model.config.n_kv_heads,
            "max_seq_len": model.config.max_seq_len,
            "is_causal": True,
            "use_reliability_head": model.reliability_head is not None,
        })
        saved_config["paradigm"] = "ar"
        _atomic_torch_save({
            "global_step": step,
            "model_state_dict": {key: value.detach().cpu() for key, value in getattr(model, "module", model).state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "config": saved_config,
            "posttrain": {"method": method, "source_checkpoint": str(checkpoint_path), "step": step},
        }, output_path)
        session.save(output_path.with_suffix('.resume.pt'), model, optimizer, step, reference)
        return output_path

    for step in range(session.step + 1, max_steps + 1):
        batch = random.choices(records, k=batch_size)
        optimizer.zero_grad(set_to_none=True)
        if method == "sft":
            input_ids, labels = _collate(batch, tokenizer, max_length, device)
            logits = model(input_ids, mask_override=True)[:, :-1].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        else:
            chosen = [{"prompt": row["prompt"], "response": row["chosen"]} for row in batch]
            rejected = [{"prompt": row["prompt"], "response": row["rejected"]} for row in batch]
            chosen_ids, chosen_labels = _collate(chosen, tokenizer, max_length, device)
            rejected_ids, rejected_labels = _collate(rejected, tokenizer, max_length, device)
            with torch.no_grad():
                ref_chosen = _sequence_logps(reference, chosen_ids, chosen_labels)
                ref_rejected = _sequence_logps(reference, rejected_ids, rejected_labels)
            policy_chosen = _sequence_logps(model, chosen_ids, chosen_labels)
            policy_rejected = _sequence_logps(model, rejected_ids, rejected_labels)
            logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
            loss = -F.logsigmoid(logits).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == max_steps:
            print(f"[{method.upper()}] step {step}/{max_steps} loss={loss.item():.4f}", flush=True)
        if save_every and step % save_every == 0 and step != max_steps:
            save(step)

    result = save(max_steps, final=True)
    print(f"Saved post-trained checkpoint: {result}", flush=True)
    return result
