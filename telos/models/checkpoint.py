"""Load native model weights and their architecture metadata."""

import json
import re
from pathlib import Path

from .config import TelosConfig


def resolve_checkpoint(checkpoint_path: str | Path) -> Path:
    cp = Path(checkpoint_path)
    if cp.is_dir():
        mlx_file = cp / "model.safetensors"
        pt_file = cp / "checkpoint_final.pt"
        if not pt_file.exists():
            native_files = [(int(match.group(1)), path) for path in cp.iterdir()
                            if (match := re.fullmatch(r"checkpoint_step_(\d+)\.(?:pt|safetensors)", path.name))]
            pt_file = max(native_files, key=lambda item: item[0])[1] if native_files else pt_file

        if mlx_file.exists():
            cp = mlx_file
        elif pt_file.exists():
            cp = pt_file
        elif (cp / "model.safetensors.index.json").is_file():
            cp = cp / "model.safetensors.index.json"
        else:
            raise FileNotFoundError(f"No checkpoint file found in directory {checkpoint_path}")

    if not cp.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {cp}")
    return cp


def _load_mlx_arrays(checkpoint):
    import mlx.core as mx

    if checkpoint.suffix == ".safetensors":
        return mx.load(str(checkpoint))
    if checkpoint.name != "model.safetensors.index.json":
        raise ValueError("Native MLX inference requires a Safetensors checkpoint")
    index = json.loads(checkpoint.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map or not all(isinstance(p, str) for p in weight_map.values()):
        raise ValueError("Invalid Safetensors weight index")
    root = checkpoint.parent.resolve()
    arrays = {}
    for shard in sorted(set(weight_map.values())):
        path = (root / shard).resolve()
        if not path.is_relative_to(root) or path.suffix != ".safetensors":
            raise ValueError(f"Invalid Safetensors shard path: {shard}")
        tensors = mx.load(str(path))
        if tensors.keys() != {key for key, filename in weight_map.items() if filename == shard}:
            raise ValueError(f"Safetensors weight index does not match shard: {shard}")
        arrays.update(tensors)
    return arrays


EXTERNAL_MLX_MOE_ARCHITECTURES = frozenset({"qwen3_moe", "lfm2_moe", "inkling2_loop"})
EXTERNAL_MLX_ARCHITECTURES = EXTERNAL_MLX_MOE_ARCHITECTURES | {"qwen3", "qwen2", "llama", "mistral", "phi3"}


def _load_external_causal_checkpoint(checkpoint, config, *, device, context_length,
                                     memory_budget_mib, prefix_cache_entries):
    family = config["model_type"]
    if family in EXTERNAL_MLX_ARCHITECTURES and device not in ("mlx", "auto"):
        raise ValueError("Native MLX checkpoints require --device mlx or auto.")
    if family in EXTERNAL_MLX_MOE_ARCHITECTURES:
        from telos.training.core import metal_usable

        if not metal_usable():
            raise ValueError("Native MLX MoE requires Apple Silicon with Metal.")
        if family == "inkling2_loop":
            from .inkling2_mlx import load_inkling2

            model = load_inkling2(checkpoint, config, context_length=context_length,
                                  memory_budget_mib=memory_budget_mib)
        else:
            from .qwen3_moe_mlx import load_mlx_moe

            model = load_mlx_moe(checkpoint, config, context_length=context_length,
                                 memory_budget_mib=memory_budget_mib, prefix_cache_entries=prefix_cache_entries)
        return model, "mlx", model.vocab_size, config
    if memory_budget_mib is not None:
        raise ValueError("--memory-budget-mib is supported for native MLX MoE checkpoints.")
    if family == "qwen3":
        from .qwen3_mlx import Qwen3MLX as model_cls
    elif family in {"llama", "mistral"}:
        from .mlx_causal import LlamaMLX as model_cls
    elif family == "qwen2":
        from .mlx_causal import Qwen2MLX as model_cls
    elif family == "phi3":
        from .mlx_causal import Phi3MLX as model_cls
    else:
        raise ValueError(f"Unsupported native inference model architecture: {family}")

    model = model_cls(config, _load_mlx_arrays(checkpoint)).eval()
    return model, "mlx", model.vocab_size, config


def load_model_from_checkpoint(checkpoint_path: str | Path, config: dict | None = None):
    """Load a native model and return the legacy evaluation tuple."""
    model, backend, vocab_size, _ = load_checkpoint(checkpoint_path, config)
    return model, backend, vocab_size


def load_checkpoint(checkpoint_path: str | Path, config: dict | None = None, *, strict_metadata=False,
                    device="auto", context_length=None, memory_budget_mib=None, prefix_cache_entries=0):
    """Return native model, backend, vocabulary size and checkpoint metadata."""
    cp = resolve_checkpoint(checkpoint_path)
    cfg_file = cp.parent / "config.json"
    if not cfg_file.exists():
        cfg_file = cp.parent / "config.yaml"
    state = None
    full_cfg = {}
    if strict_metadata and config is not None:
        full_cfg = config
    elif cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                full_cfg = json.load(f)
        except (OSError, UnicodeError, json.JSONDecodeError):
            import yaml

            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    full_cfg = yaml.safe_load(f) or {}
            except (OSError, UnicodeError, yaml.YAMLError) as exc:
                if strict_metadata:
                    raise ValueError(f"Invalid checkpoint config: {cfg_file}") from exc
                full_cfg = {}
    elif config:
        full_cfg = config
    elif cp.suffix in [".pt", ".bin"]:
        try:
            import torch
            state = torch.load(str(cp), map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                full_cfg = state.get("config", {})
        except Exception:
            if strict_metadata:
                raise

    if strict_metadata and (not isinstance(full_cfg, dict) or not full_cfg):
        raise ValueError("Checkpoint architecture metadata is missing. Supply a config file.")
    m_cfg = full_cfg.get("model", full_cfg)
    if strict_metadata and not isinstance(m_cfg, dict):
        raise ValueError("Checkpoint model config must be a mapping.")
    if strict_metadata and m_cfg is full_cfg and full_cfg.get("model_type") not in (None, "telos"):
        return _load_external_causal_checkpoint(
            cp, full_cfg, device=device, context_length=context_length, memory_budget_mib=memory_budget_mib,
            prefix_cache_entries=prefix_cache_entries,
        )
    if memory_budget_mib is not None:
        raise ValueError("--memory-budget-mib is supported for native MLX MoE checkpoints.")

    vocab_size = m_cfg.get("vocab_size", 8192)
    d_model = m_cfg.get("d_model", 512)
    n_layers = m_cfg.get("n_layers", 12)
    n_heads = m_cfg.get("n_heads", 16)
    n_kv_heads = m_cfg.get("n_kv_heads", n_heads)

    paradigm = str(full_cfg.get("paradigm", "") or m_cfg.get("paradigm", "")).lower()
    if not paradigm and not strict_metadata:
        for p in ["corosred", "mdlm", "undlm", "ar"]:
            if p in str(cp).lower():
                paradigm = p
                break

    is_causal = bool(m_cfg.get("is_causal", paradigm == "ar"))
    if strict_metadata:
        if not is_causal:
            raise ValueError("Generation requires a native causal checkpoint.")
        if "n_heads" not in m_cfg:
            raise ValueError("Checkpoint config must specify n_heads.")
    use_reliability_head = bool(m_cfg.get("use_reliability_head", paradigm == "corosred"))

    if cp.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(cp), framework="pt") as f:
            keys = list(f.keys())
            layer_indices = [int(k.split(".")[1]) for k in keys if "layers." in k and k.split(".")[1].isdigit()]
            if layer_indices:
                n_layers = max(layer_indices) + 1
            if "emb.weight" in keys:
                emb_shape = f.get_slice("emb.weight").get_shape()
                vocab_size = emb_shape[0]
                d_model = emb_shape[1]
            if "layers.0.qkv_proj.weight" in keys and "n_heads" not in m_cfg:
                qkv_shape = f.get_slice("layers.0.qkv_proj.weight").get_shape()
                qkv_out = qkv_shape[0]
                # QKV projection = 3 * n_heads * head_dim (head_dim = 64)
                total_heads = qkv_out // 64
                n_heads = max(1, total_heads // 3)
                n_kv_heads = n_heads

        if not use_reliability_head and any("reliability_head" in k for k in keys):
            use_reliability_head = True

        is_pytorch_keys = any("tok_embeddings" in k or ".attn." in k or "final_norm" in k for k in keys)

        if is_pytorch_keys:
            from safetensors.torch import load_file
            from torch import nn

            sd = load_file(str(cp))
            sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}

            if not use_reliability_head and any("reliability_head" in k for k in sd):
                use_reliability_head = True

            tc = TelosConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                tied_embeddings=bool(m_cfg.get("tied_embeddings", True)),
                packed_projections=bool(m_cfg.get("packed_projections", False)),
                max_seq_len=int(m_cfg.get("max_seq_len", m_cfg.get("seq_len") or 512)),
                is_causal=is_causal,
                use_reliability_head=use_reliability_head,
            )
            from telos.models import TelosTransformer
            model = TelosTransformer(tc)
            if "reliability_head.weight" in sd and hasattr(model, "reliability_head") and isinstance(model.reliability_head, nn.Sequential):
                model.reliability_head = nn.Linear(tc.d_model, 1, bias=False)

            model.load_state_dict(sd, strict=True)
            model.eval()
            return model, "pytorch", vocab_size, full_cfg
        else:
            from telos.models import MLXTelosTransformer
            model = MLXTelosTransformer(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                tied_embeddings=bool(m_cfg.get("tied_embeddings", True)),
                max_seq_len=int(m_cfg.get("max_seq_len", m_cfg.get("seq_len") or 512)),
                is_causal=is_causal,
                use_reliability_head=use_reliability_head,
            )
            model.load_weights(str(cp))
            model.paradigm = paradigm
            return model, "mlx", vocab_size, full_cfg

    elif cp.suffix in [".pt", ".bin"]:
        import torch
        from torch import nn

        if state is None:
            state = torch.load(str(cp), map_location="cpu", weights_only=True)
        if strict_metadata and state.get("posttrain"):
            full_cfg = {**full_cfg, "posttrain": state["posttrain"]}
        sd = state.get("model_state_dict", state)
        sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}

        if not use_reliability_head and any("reliability_head" in k for k in sd):
            use_reliability_head = True

        layer_indices = [int(k.split(".")[1]) for k in sd if "layers." in k and k.split(".")[1].isdigit()]
        if layer_indices:
            n_layers = max(layer_indices) + 1

        tc = TelosConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            tied_embeddings=bool(m_cfg.get("tied_embeddings", True)),
            packed_projections=bool(m_cfg.get("packed_projections", False)),
            max_seq_len=int(m_cfg.get("max_seq_len", m_cfg.get("seq_len") or 512)),
            is_causal=is_causal,
            use_reliability_head=use_reliability_head,
        )
        from telos.models import TelosTransformer
        model = TelosTransformer(tc)
        if "reliability_head.weight" in sd and hasattr(model, "reliability_head") and isinstance(model.reliability_head, nn.Sequential):
            model.reliability_head = nn.Linear(tc.d_model, 1, bias=False)

        model.load_state_dict(sd, strict=True)
        model.eval()
        model.paradigm = paradigm
        return model, "pytorch", vocab_size, full_cfg

    else:
        raise ValueError(f"Unrecognized checkpoint format: {cp.name}")


