"""Tests for native MLX Qwen 3.5 MoE and Gemma 4 MoE inference, SSD streaming, and window distillation."""

import json
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from telos.eval.runner import load_model_from_checkpoint
from telos.inference import TextGenerator
from telos.models import qwen3_moe_mlx as moe_loader
from telos.models.checkpoint import EXTERNAL_MLX_ARCHITECTURES, EXTERNAL_MLX_MOE_ARCHITECTURES, load_checkpoint
from telos.models.ssd_experts import expert_layer_sizes, is_expert_tensor
from telos.posttrain.trainer import posttrain
from telos.training.core import metal_usable

pytestmark = pytest.mark.skipif(not metal_usable(), reason="MLX Metal required")


def _write_tokenizer(path: Path, vocab_entries: int = 32):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "<|im_end|>": 1, **{f"t{i}": i for i in range(2, vocab_entries)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))


def _write_qwen3_5_moe_fixture(path: Path, mx):
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen3_5_moe",
        "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
        "eos_token_id": 1,
        "pad_token_id": 0,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 32,
            "vocab_size": 32,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "attn_output_gate": True,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "norm_topk_prob": True,
            "rope_parameters": {"rope_theta": 100000.0, "partial_rotary_factor": 0.5, "rope_type": "default"},
        },
    }
    arrays = {}

    def sample(shape, dtype=mx.bfloat16):
        return (mx.random.normal(shape, key=mx.random.key(41 + len(arrays))) * 0.1).astype(dtype)

    def proj(name, rows, cols):
        w, s, b = mx.quantize(sample((rows, cols)), group_size=32, bits=4)
        arrays.update({name + ".weight": w, name + ".scales": s, name + ".biases": b})

    root = "language_model.model"
    proj(f"{root}.embed_tokens", 32, 32)
    arrays[f"{root}.norm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
    arrays["vision_tower.dummy.weight"] = mx.ones((4, 4), dtype=mx.bfloat16)

    for i, kind in enumerate(config["text_config"]["layer_types"]):
        lp = f"{root}.layers.{i}"
        arrays[f"{lp}.input_layernorm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
        arrays[f"{lp}.post_attention_layernorm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
        if kind == "linear_attention":
            la = f"{lp}.linear_attn"
            # conv_dim = 2*(2*8) + 4*8 = 64
            proj(f"{la}.in_proj_qkv", 64, 32)
            proj(f"{la}.in_proj_z", 32, 32)
            proj(f"{la}.in_proj_b", 4, 32)
            proj(f"{la}.in_proj_a", 4, 32)
            proj(f"{la}.out_proj", 32, 32)
            arrays[f"{la}.conv1d.weight"] = sample((64, 4, 1))
            arrays[f"{la}.A_log"] = mx.zeros((4,), dtype=mx.float32)
            arrays[f"{la}.dt_bias"] = mx.ones((4,), dtype=mx.bfloat16)
            arrays[f"{la}.norm.weight"] = mx.ones((8,), dtype=mx.bfloat16)
        else:
            sa = f"{lp}.self_attn"
            proj(f"{sa}.q_proj", 64, 32)
            proj(f"{sa}.k_proj", 16, 32)
            proj(f"{sa}.v_proj", 16, 32)
            proj(f"{sa}.o_proj", 32, 32)
            arrays[f"{sa}.q_norm.weight"] = mx.ones((16,), dtype=mx.bfloat16)
            arrays[f"{sa}.k_norm.weight"] = mx.ones((16,), dtype=mx.bfloat16)

        mlp = f"{lp}.mlp"
        proj(f"{mlp}.gate", 4, 32)
        for p in ("gate_proj", "up_proj", "down_proj"):
            proj(f"{mlp}.shared_expert.{p}", 32, 32)
            proj(f"{mlp}.switch_mlp.{p}", 4, 32 if False else 32)
        # Replace switch_mlp with 3D stacked expert tensors
        for p in ("gate_proj", "up_proj", "down_proj"):
            for s in (".weight", ".scales", ".biases"):
                arrays.pop(f"{mlp}.switch_mlp.{p}{s}")
            w, sc, bi = mx.quantize(sample((4, 32, 32)), group_size=32, bits=4)
            arrays[f"{mlp}.switch_mlp.{p}.weight"] = w
            arrays[f"{mlp}.switch_mlp.{p}.scales"] = sc
            arrays[f"{mlp}.switch_mlp.{p}.biases"] = bi
        proj(f"{mlp}.shared_expert_gate", 1, 32)

    groups = [{k: v for k, v in arrays.items() if (".switch_mlp." in k) == is_exp} for is_exp in (False, True)]
    mapping = {}
    for fname, grp in zip(("resident.safetensors", "experts.safetensors"), groups, strict=True):
        mx.save_safetensors(str(path / fname), grp)
        mapping.update(dict.fromkeys(grp, fname))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    (path / "config.json").write_text(json.dumps(config))
    _write_tokenizer(path)
    return path


def _write_gemma4_fixture(path: Path, mx, *, enable_moe: bool = True):
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "gemma4",
        "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
        "eos_token_id": 1,
        "pad_token_id": 0,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 32,
            "vocab_size": 32,
            "num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "global_head_dim": 32,
            "attention_k_eq_v": True,
            "intermediate_size": 32,
            "moe_intermediate_size": 32,
            "hidden_activation": "gelu_pytorch_tanh",
            "enable_moe_block": enable_moe,
            "num_experts": 4 if enable_moe else None,
            "top_k_experts": 2 if enable_moe else None,
            "final_logit_softcapping": 30.0,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "proportional"},
            },
        },
    }
    arrays = {}

    def sample(shape, dtype=mx.bfloat16):
        return (mx.random.normal(shape, key=mx.random.key(67 + len(arrays))) * 0.1).astype(dtype)

    def proj(name, rows, cols):
        w, s, b = mx.quantize(sample((rows, cols)), group_size=32, bits=4)
        arrays.update({name + ".weight": w, name + ".scales": s, name + ".biases": b})

    root = "language_model.model"
    proj(f"{root}.embed_tokens", 32, 32)
    arrays[f"{root}.norm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
    arrays["vision_tower.dummy.weight"] = mx.ones((4, 4), dtype=mx.bfloat16)
    arrays["audio_tower.dummy.weight"] = mx.ones((4, 4), dtype=mx.bfloat16)

    for i, kind in enumerate(config["text_config"]["layer_types"]):
        lp = f"{root}.layers.{i}"
        arrays[f"{lp}.input_layernorm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
        arrays[f"{lp}.post_attention_layernorm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
        arrays[f"{lp}.pre_feedforward_layernorm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
        arrays[f"{lp}.post_feedforward_layernorm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
        arrays[f"{lp}.layer_scalar"] = mx.array(0.9, dtype=mx.bfloat16)

        hdim = 16 if kind == "sliding_attention" else 32
        sa = f"{lp}.self_attn"
        proj(f"{sa}.q_proj", 2 * hdim, 32)
        proj(f"{sa}.k_proj", hdim, 32)
        if kind == "sliding_attention":
            proj(f"{sa}.v_proj", hdim, 32)
        proj(f"{sa}.o_proj", 32, 2 * hdim)
        arrays[f"{sa}.q_norm.weight"] = mx.ones((hdim,), dtype=mx.bfloat16)
        arrays[f"{sa}.k_norm.weight"] = mx.ones((hdim,), dtype=mx.bfloat16)

        for p, r, c in (("gate_proj", 32, 32), ("up_proj", 32, 32), ("down_proj", 32, 32)):
            proj(f"{lp}.mlp.{p}", r, c)

        if enable_moe:
            arrays[f"{lp}.pre_feedforward_layernorm_2.weight"] = mx.ones((32,), dtype=mx.bfloat16)
            arrays[f"{lp}.post_feedforward_layernorm_1.weight"] = mx.ones((32,), dtype=mx.bfloat16)
            arrays[f"{lp}.post_feedforward_layernorm_2.weight"] = mx.ones((32,), dtype=mx.bfloat16)
            proj(f"{lp}.router.proj", 4, 32)
            arrays[f"{lp}.router.scale"] = mx.ones((32,), dtype=mx.bfloat16)
            arrays[f"{lp}.router.per_expert_scale"] = mx.linspace(0.8, 1.2, 4).astype(mx.bfloat16)
            for p in ("gate_proj", "up_proj", "down_proj"):
                w, sc, bi = mx.quantize(sample((4, 32, 32)), group_size=32, bits=4)
                arrays[f"{lp}.experts.switch_glu.{p}.weight"] = w
                arrays[f"{lp}.experts.switch_glu.{p}.scales"] = sc
                arrays[f"{lp}.experts.switch_glu.{p}.biases"] = bi

    groups = [{k: v for k, v in arrays.items() if (".experts.switch_glu." in k) == is_exp} for is_exp in (False, True)]
    mapping = {}
    for fname, grp in zip(("resident.safetensors", "experts.safetensors"), groups, strict=True):
        if grp:
            mx.save_safetensors(str(path / fname), grp)
            mapping.update(dict.fromkeys(grp, fname))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    (path / "config.json").write_text(json.dumps(config))
    _write_tokenizer(path)
    return path


def _patch_budget(monkeypatch, budget_bytes):
    class FixtureGuard:
        budget = budget_bytes

        def check(self):
            pass

    monkeypatch.setattr(moe_loader, "AppleMemoryGuard", FixtureGuard)


def test_architectures_registered():
    assert {"qwen3_5_moe", "gemma4"} <= EXTERNAL_MLX_MOE_ARCHITECTURES
    assert {"qwen3_5_moe", "gemma4"} <= EXTERNAL_MLX_ARCHITECTURES


def _find_ssd_budget(ckpt: Path) -> int:
    from telos.models.ssd_experts import tensor_ranges

    ranges = {k: v for k, v in tensor_ranges(ckpt / "model.safetensors.index.json").items() if not k.startswith(("vision_tower.", "audio_tower."))}
    cfg = json.loads((ckpt / "config.json").read_text())
    low, high = 0, 2**30
    while low < high:
        mid = (low + high) // 2
        try:
            moe_loader.plan_experts(cfg, ranges, mid, None, 0)
        except ValueError:
            low = mid + 1
        else:
            high = mid
    for delta in (4096, 8192, 16384):
        try:
            _, cache = moe_loader.plan_experts(cfg, ranges, high + delta, None, 0)
            if cache is not None:
                return high + delta
        except ValueError:
            pass
    _, cache = moe_loader.plan_experts(cfg, ranges, high, None, 0)
    assert cache is not None
    return high


@pytest.mark.parametrize("router_training", ["adapter", "full"])
def test_qwen3_5_moe_resident_ssd_and_window_distill(tmp_path, monkeypatch, router_training):
    import mlx.core as mx

    ckpt = _write_qwen3_5_moe_fixture(tmp_path / "q35", mx)
    _patch_budget(monkeypatch, 2**30)
    res, _, _, _ = load_checkpoint(ckpt, device="mlx")
    assert res.weight_storage == "resident"

    ssd_budget = _find_ssd_budget(ckpt)
    _patch_budget(monkeypatch, ssd_budget)
    ssd, _, _, _ = load_checkpoint(ckpt, device="mlx")
    assert ssd.weight_storage == "ssd_experts"

    ids = mx.array([[2, 3, 4, 5]], dtype=mx.int32)
    assert mx.allclose(res.hidden_states(ids), ssd.hidden_states(ids), atol=1e-4, rtol=1e-4).item()
    gen_res = TextGenerator.from_checkpoint(ckpt, device="mlx", window_size=4)
    assert gen_res.model.weight_storage == "ssd_experts"
    out = gen_res.generate("t2 t3", max_new_tokens=4, temperature=0.0, cache="dynamic")
    assert len(out) > 0

    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"prompt": "t2 t3", "response": "t4 t5 t6"}) + "\n")
    out_dir = tmp_path / "q35_distill"
    posttrain(
        ckpt, data, out_dir,
        method="distill", hardware="mlx", finetune="lora",
        lora_rank=4, lora_alpha=8.0, window_size=4, selector_rank=4, router_training=router_training,
        max_steps=2, batch_size=1, grad_accum=2, max_length=16,
    )
    _patch_budget(monkeypatch, 2**30)
    loaded, backend, _ = load_model_from_checkpoint(out_dir)
    assert backend == "mlx" and loaded._window_controller.window_size == 4
    if router_training == "full":
        from mlx.utils import tree_flatten
        saved = mx.load(str(out_dir / "model.adapter.safetensors"))
        weights = {k: v for k, v in saved.items() if k.endswith(("selector_weight", "token_weight"))}
        assert weights
        assert any(mx.any(v != 0).item() for k, v in weights.items() if k.endswith("selector_weight"))
        assert any(mx.any(v != 0).item() for k, v in weights.items() if k.endswith("token_weight"))
        reloaded = dict(tree_flatten(loaded.parameters()))
        for key, value in weights.items():
            assert mx.array_equal(value, reloaded[key]).item()


@pytest.mark.parametrize("router_training", ["adapter", "full"])
def test_gemma4_moe_and_dense_ssd_and_window_distill(tmp_path, monkeypatch, router_training):
    import mlx.core as mx

    # 1. Dense Gemma 4 (enable_moe_block=False)
    dense_ckpt = _write_gemma4_fixture(tmp_path / "g4_dense", mx, enable_moe=False)
    _patch_budget(monkeypatch, 2**30)
    dense_model, _, _, _ = load_checkpoint(dense_ckpt, device="mlx")
    assert dense_model.weight_storage == "resident"
    h = dense_model.hidden_states(mx.array([[2, 3, 4]], dtype=mx.int32))
    assert dense_model.logits_from_hidden(h).shape == (1, 3, 32)

    # 2. MoE Gemma 4 26B-A4B layout (sliding + full k_eq_v + router + experts.switch_glu + parallel mlp)
    ckpt = _write_gemma4_fixture(tmp_path / "g4_moe", mx, enable_moe=True)
    res, _, _, _ = load_checkpoint(ckpt, device="mlx")
    assert res.weight_storage == "resident"

    ssd_budget = _find_ssd_budget(ckpt)
    _patch_budget(monkeypatch, ssd_budget)
    ssd, _, _, _ = load_checkpoint(ckpt, device="mlx")
    assert ssd.weight_storage == "ssd_experts"

    ids = mx.array([[2, 3, 4, 5, 6]], dtype=mx.int32)
    assert mx.allclose(res.hidden_states(ids), ssd.hidden_states(ids), atol=1e-4, rtol=1e-4).item()
    gen = TextGenerator.from_checkpoint(ckpt, device="mlx", window_size=4)
    assert gen.model.weight_storage == "ssd_experts"
    out = gen.generate("t2 t3", max_new_tokens=4, temperature=0.0, cache="dynamic")
    assert len(out) > 0

    data = tmp_path / "g4_data.jsonl"
    data.write_text(json.dumps({"prompt": "t2 t3", "response": "t4 t5 t6"}) + "\n")
    out_dir = tmp_path / "g4_distill"
    posttrain(
        ckpt, data, out_dir,
        method="distill", hardware="mlx", finetune="lora",
        lora_rank=4, lora_alpha=8.0, window_size=4, selector_rank=4, router_training=router_training,
        max_steps=2, batch_size=1, grad_accum=2, max_length=16,
    )
    _patch_budget(monkeypatch, 2**30)
    loaded, backend, _ = load_model_from_checkpoint(out_dir)
    assert backend == "mlx" and loaded._window_controller.window_size == 4
    if router_training == "full":
        from mlx.utils import tree_flatten
        saved = mx.load(str(out_dir / "model.adapter.safetensors"))
        weights = {k: v for k, v in saved.items() if k.endswith(("selector_weight", "token_weight"))}
        assert weights
        assert any(mx.any(v != 0).item() for k, v in weights.items() if k.endswith("selector_weight"))
        assert any(mx.any(v != 0).item() for k, v in weights.items() if k.endswith("token_weight"))
        reloaded = dict(tree_flatten(loaded.parameters()))
        for key, value in weights.items():
            assert mx.array_equal(value, reloaded[key]).item()


def test_real_cached_qwen3_5_and_gemma4_expert_indices():
    from telos.models.ssd_experts import TensorRange

    q35_idx = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec/model.safetensors.index.json"
    if q35_idx.is_file():
        wm = json.loads(q35_idx.read_text())["weight_map"]
        dummy = {
            k: TensorRange(q35_idx, 0, 256 * 8, "U8", (256, 4, 2))
            for k in wm if is_expert_tensor(k)
        }
        assert len(expert_layer_sizes(dummy, 256)) == 40

    g26_idx = Path.home() / ".cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-4bit/snapshots/e05846c0573ede5ce8d281bff96b9f916198cd45/model.safetensors.index.json"
    if g26_idx.is_file():
        wm = json.loads(g26_idx.read_text())["weight_map"]
        dummy = {
            k: TensorRange(g26_idx, 0, 128 * 8, "U8", (128, 4, 2))
            for k in wm if is_expert_tensor(k)
        }
        assert len(expert_layer_sizes(dummy, 128)) == 30
