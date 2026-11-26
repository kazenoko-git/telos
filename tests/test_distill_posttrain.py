"""Tests for native --method distill post-training across PyTorch and MLX."""

import json
import sys

import pytest
import torch

from telos.data.tokenizer import load_tokenizer
from telos.eval.runner import load_model_from_checkpoint
from telos.models import TelosConfig, TelosTransformer
from telos.posttrain.cli import main as posttrain_cli
from telos.posttrain.trainer import posttrain


def _write_checkpoint(path, vocab_size, d_model, n_layers, seed):
    torch.manual_seed(seed)
    cfg = TelosConfig(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=2, n_kv_heads=2, max_seq_len=16, is_causal=True,
    )
    model = TelosTransformer(cfg)
    torch.save({"model_state_dict": model.state_dict(), "config": {"paradigm": "ar", "model": vars(cfg)}}, path)
    return cfg, model


@pytest.mark.parametrize("hardware", ["cpu", "mlx"])
@pytest.mark.parametrize("finetune", ["full", "lora"])
def test_distill_self_and_resume_parity(tmp_path, hardware, finetune):
    if hardware == "mlx":
        from telos.training.core import metal_usable

        if not metal_usable():
            pytest.skip("Metal required")

    tok = load_tokenizer()
    vocab = len(tok.get_vocab())
    source = tmp_path / "student.pt"
    _write_checkpoint(source, vocab, d_model=64, n_layers=1, seed=11)

    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"prompt": p, "response": "2"}) for p in ("1+1?", "2+2?")))

    opts = dict(
        method="distill", hardware=hardware, finetune=finetune, lora_rank=4,
        max_steps=3, batch_size=1, max_length=12, save_every=1, learning_rate=0.002,
        distill_alpha=0.5, distill_temperature=1.5,
        precision="bf16" if hardware == "mlx" else "auto",
    )
    final = posttrain(source, data, tmp_path / "run1", **opts)
    if hardware == "mlx":
        import mlx.core as mx

        expected_mlx = mx.load(str(final))
        mx.eval(expected_mlx)
    suffix = ".resume.npz" if hardware == "mlx" else ".resume.pt"
    resumed = posttrain(
        source, data, tmp_path / ("run1" if hardware == "mlx" else "run_resumed"),
        resume=tmp_path / "run1" / ("checkpoint_step_1" + suffix), **opts,
    )
    loaded, backend, _ = load_model_from_checkpoint(final)
    assert backend == ("mlx" if hardware == "mlx" else "pytorch")
    assert loaded is not None

    if hardware == "cpu":
        a, b = [torch.load(p, weights_only=True)["model_state_dict"] for p in (final, resumed)]
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    else:
        import mlx.core as mx

        b = mx.load(str(resumed))
        for key in expected_mlx:
            assert mx.allclose(expected_mlx[key], b[key], rtol=1e-5, atol=1e-6).item(), key


@pytest.mark.parametrize("hardware", ["cpu", "mlx"])
def test_distill_external_larger_teacher_and_cli(tmp_path, monkeypatch, hardware):
    if hardware == "mlx":
        from telos.training.core import metal_usable

        if not metal_usable():
            pytest.skip("Metal required")

    tok = load_tokenizer()
    vocab = len(tok.get_vocab())
    student = tmp_path / "student.pt"
    teacher = tmp_path / "teacher.pt"
    _write_checkpoint(student, vocab, d_model=64, n_layers=1, seed=17)
    _write_checkpoint(teacher, vocab, d_model=128, n_layers=2, seed=29)

    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"prompt": p, "response": "ok"}) for p in ("a?", "b?")))

    out = tmp_path / "distilled"
    monkeypatch.setattr(sys, "argv", [
        "telos posttrain",
        "--method", "distill",
        "--checkpoint", str(student),
        "--teacher", str(teacher),
        "--data", str(data),
        "--output", str(out),
        "--hardware", hardware,
        "--finetune", "lora",
        "--lora-rank", "4",
        "--distill-alpha", "0.75",
        "--distill-temperature", "2.0",
        "--max-steps", "2",
        "--batch-size", "1",
        "--max-length", "12",
        "--save-every", "0",
    ])
    posttrain_cli()

    if hardware == "mlx":
        meta = json.loads((out / "config.json").read_text(encoding="utf-8"))["posttrain"]
    else:
        meta = torch.load(out / "checkpoint_final.pt", weights_only=True)["posttrain"]
    assert meta["method"] == "distill"
    assert meta["options"]["distill"]["teacher"] == str(teacher.resolve())
    assert meta["options"]["distill"]["alpha"] == 0.75
    assert meta["options"]["distill"]["temperature"] == 2.0


def test_distill_validation_errors(tmp_path):
    tok = load_tokenizer()
    vocab = len(tok.get_vocab())
    student = tmp_path / "student.pt"
    bad_teacher = tmp_path / "bad_teacher.pt"
    _write_checkpoint(student, vocab, d_model=64, n_layers=1, seed=3)
    _write_checkpoint(bad_teacher, vocab + 16, d_model=64, n_layers=1, seed=4)

    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"prompt": "hi", "response": "there"}))

    with pytest.raises(ValueError, match="alpha"):
        posttrain(student, data, tmp_path / "o1", method="distill", hardware="cpu", distill_alpha=1.5)
    with pytest.raises(ValueError, match="temperature"):
        posttrain(student, data, tmp_path / "o2", method="distill", hardware="cpu", distill_temperature=0.0)
    with pytest.raises(ValueError, match="vocab size"):
        posttrain(student, data, tmp_path / "o3", method="distill", hardware="cpu", teacher=bad_teacher)
    with pytest.raises(ValueError, match="grad"):
        posttrain(student, data, tmp_path / "o4", method="distill", hardware="cpu", grad_accum=0)
    with pytest.raises(ValueError, match="GRPO"):
        posttrain(student, data, tmp_path / "o5", method="grpo", hardware="cpu", grad_accum=2)


@pytest.mark.parametrize("hardware", ["cpu", "mlx"])
def test_grad_accum_and_resume_parity(tmp_path, hardware):
    if hardware == "mlx":
        from telos.training.core import metal_usable

        if not metal_usable():
            pytest.skip("Metal required")

    tok = load_tokenizer()
    vocab = len(tok.get_vocab())
    source = tmp_path / "student.pt"
    _write_checkpoint(source, vocab, d_model=64, n_layers=1, seed=7)

    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"prompt": p, "response": "2"}) for p in ("1+1?", "2+2?", "3+3?")))

    opts = dict(
        method="distill", hardware=hardware, finetune="lora", lora_rank=4,
        max_steps=3, batch_size=1, grad_accum=2, max_length=12, save_every=1, learning_rate=0.002,
        precision="bf16" if hardware == "mlx" else "auto",
    )
    final = posttrain(source, data, tmp_path / "run1", **opts)
    suffix = ".resume.npz" if hardware == "mlx" else ".resume.pt"
    resumed = posttrain(
        source, data, tmp_path / ("run1" if hardware == "mlx" else "run2"),
        resume=tmp_path / "run1" / ("checkpoint_step_1" + suffix), **opts,
    )
    if hardware == "cpu":
        a, b = [torch.load(p, weights_only=True)["model_state_dict"] for p in (final, resumed)]
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    else:
        import mlx.core as mx

        a, b = mx.load(str(final)), mx.load(str(resumed))
        for key in a:
            assert mx.allclose(a[key], b[key], rtol=1e-5, atol=1e-6).item(), key
