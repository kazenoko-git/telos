"""Post-training contract and CPU integration coverage."""
import json
import platform
import sys
import types

import pytest
import torch

from telos.data.tokenizer import load_tokenizer
from telos.eval.runner import load_model_from_checkpoint
from telos.models import TelosConfig, TelosTransformer
from telos.posttrain.trainer import (
    _encode,
    _map_pytorch_parameter_names,
    _resolve_backend,
    _xla_topology,
    posttrain,
)
from telos.training import xla_utils


def test_sft_encoding_masks_prompt_and_trains_eos():
    tokenizer = load_tokenizer()
    tokens, labels, _ = _encode(tokenizer, "Question", "Answer", 16, torch.device("cpu"))
    first_answer = 1 + len(tokenizer.encode("Question", add_special_tokens=False).ids)
    assert labels[:first_answer] == [-100] * first_answer
    assert labels[first_answer:] == tokens[first_answer:]
    assert labels[-1] == tokenizer.token_to_id("[EOS]")


def test_pytorch_checkpoint_mapping_packs_qkv_and_swiglu():
    state = {
        "tok_embeddings.weight": torch.zeros(8, 4),
        "layers.0.attn.q_proj.weight": torch.full((4, 4), 1.0),
        "layers.0.attn.k_proj.weight": torch.full((2, 4), 2.0),
        "layers.0.attn.v_proj.weight": torch.full((2, 4), 3.0),
        "layers.0.mlp.w1.weight": torch.full((8, 4), 4.0),
        "layers.0.mlp.v.weight": torch.full((8, 4), 5.0),
        "layers.0.mlp.w2.weight": torch.full((4, 8), 6.0),
    }
    mapped = _map_pytorch_parameter_names(state)
    assert mapped["layers.0.qkv_proj.weight"].shape == (8, 4)
    assert mapped["layers.0.qkv_proj.weight"][:4].eq(1).all()
    assert mapped["layers.0.mlp.gate_up.weight"].shape == (16, 4)
    assert mapped["layers.0.mlp.w3.weight"].eq(6).all()


def test_auto_backend_prefers_mlx_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    import telos.training.core as core
    monkeypatch.setattr(core, "metal_usable", lambda: True)
    assert _resolve_backend("auto") == "mlx"


def test_auto_backend_does_not_select_torch_mps():
    assert _resolve_backend("cpu") == "cpu"
    with pytest.raises(ValueError, match="MPS is not used"):
        _resolve_backend("mps")


@pytest.mark.parametrize(
    ("rank", "devices", "processes", "expected", "message"),
    [
        (0, 8, 8, "auto", None),
        (7, 8, 8, 8, None),
        (8, 8, 8, "auto", "Invalid PyTorch/XLA topology"),
        (0, 8, 4, 4, "Requested 4 XLA devices"),
        (0, 0, 0, "auto", "Invalid PyTorch/XLA topology"),
    ],
)
def test_xla_topology_is_explicit_and_never_defaults_to_rank_zero(rank, devices, processes, expected, message):
    runtime = types.SimpleNamespace(
        global_ordinal=lambda: rank,
        global_runtime_device_count=lambda: devices,
        world_size=lambda: processes,
    )
    if message:
        with pytest.raises(RuntimeError, match=message):
            _xla_topology(runtime, expected)
    else:
        actual = _xla_topology(runtime, expected)
        assert actual == (rank, devices, processes)


def test_xla_master_lookup_never_guesses_rank_zero(monkeypatch):
    xla = types.ModuleType("torch_xla")
    core = types.ModuleType("torch_xla.core")
    xm = types.ModuleType("torch_xla.core.xla_model")
    runtime = types.ModuleType("torch_xla.runtime")
    xla.core, xla.runtime, core.xla_model = core, runtime, xm
    for name, module in {
        "torch_xla": xla,
        "torch_xla.core": core,
        "torch_xla.core.xla_model": xm,
        "torch_xla.runtime": runtime,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    monkeypatch.setattr(xla_utils, "_CACHED_IS_MASTER", None)
    with pytest.raises(RuntimeError, match="Unable to determine.*master rank"):
        xla_utils.is_xla_master()


def test_xla_master_lookup_uses_verified_process_index(monkeypatch):
    xla = types.ModuleType("torch_xla")
    xla.__path__ = []
    runtime = types.ModuleType("torch_xla.runtime")
    runtime.process_index = lambda: 3
    xla.runtime = runtime
    monkeypatch.setitem(sys.modules, "torch_xla", xla)
    monkeypatch.setitem(sys.modules, "torch_xla.runtime", runtime)
    monkeypatch.setattr(xla_utils, "_CACHED_IS_MASTER", None)
    assert xla_utils.is_xla_master() is False


@pytest.mark.parametrize("method", ["sft", "dpo"])
def test_posttrain_cpu_writes_reloadable_checkpoint(tmp_path, method):
    tokenizer = load_tokenizer()
    checkpoint_dir = tmp_path / "pretrained"
    checkpoint_dir.mkdir()
    model_config = TelosConfig(
        vocab_size=len(tokenizer.get_vocab()), d_model=16, n_layers=1,
        n_heads=2, n_kv_heads=2, max_seq_len=16, is_causal=True,
    )
    model = TelosTransformer(model_config)
    checkpoint = checkpoint_dir / "checkpoint_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "paradigm": "ar",
            "model": {
                "vocab_size": model_config.vocab_size, "d_model": 16,
                "n_layers": 1, "n_heads": 2, "n_kv_heads": 2,
                "max_seq_len": 16, "is_causal": True,
            },
        },
    }, checkpoint)
    data = tmp_path / "examples.jsonl"
    row = {"prompt": "What is 1+1?", "response": "2"}
    if method == "dpo":
        row = {"prompt": "What is 1+1?", "chosen": "2", "rejected": "4"}
    data.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = posttrain(
        checkpoint, data, tmp_path / f"{method}-out", method=method,
        tokenizer_path=tokenizer_path(), hardware="cpu", max_steps=1,
        batch_size=1, max_length=8, save_every=0, seed=7,
    )
    assert result.is_file()
    reloaded, backend, vocab_size = load_model_from_checkpoint(result)
    assert backend == "pytorch"
    assert vocab_size == model_config.vocab_size
    assert reloaded.config.is_causal


def tokenizer_path():
    from importlib import resources

    asset = resources.files("telos.assets").joinpath("tokenizer_0.json")
    return str(asset)
