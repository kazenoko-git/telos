"""Regression tests for post-training external MLX dense and MoE checkpoints."""

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from telos.eval.runner import load_model_from_checkpoint
from telos.inference import TextGenerator
from telos.posttrain.trainer import posttrain
from telos.training.core import metal_usable

pytestmark = pytest.mark.skipif(not metal_usable(), reason="MLX Metal required")


def _write_tokenizer(path, *, vocab_entries=30):
    vocab = {"[UNK]": 0, "<|im_end|>": 1, **{f"t{i}": i for i in range(2, vocab_entries)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))


def _write_qwen3_fixture(path, mx, *, quantized):
    path.mkdir(parents=True, exist_ok=True)
    # Intentionally keep tokenizer vocab smaller than model vocab_size (HF padded vocab case).
    config = {
        "model_type": "qwen3", "hidden_size": 64, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 32, "num_hidden_layers": 1, "intermediate_size": 128,
        "vocab_size": 32, "max_position_embeddings": 32, "tie_word_embeddings": quantized,
        "rope_theta": 10000.0,
    }
    shapes = {
        "model.embed_tokens.weight": (32, 64),
        "model.norm.weight": (64,),
        "model.layers.0.input_layernorm.weight": (64,),
        "model.layers.0.post_attention_layernorm.weight": (64,),
        "model.layers.0.self_attn.q_norm.weight": (32,),
        "model.layers.0.self_attn.k_norm.weight": (32,),
        "model.layers.0.self_attn.q_proj.weight": (128, 64),
        "model.layers.0.self_attn.k_proj.weight": (64, 64),
        "model.layers.0.self_attn.v_proj.weight": (64, 64),
        "model.layers.0.self_attn.o_proj.weight": (64, 128),
        "model.layers.0.mlp.gate_proj.weight": (128, 64),
        "model.layers.0.mlp.up_proj.weight": (128, 64),
        "model.layers.0.mlp.down_proj.weight": (64, 128),
    }
    if not quantized:
        shapes["lm_head.weight"] = (32, 64)
    arrays = {
        name: (mx.linspace(0.8, 1.2, shape[0]) if len(shape) == 1
               else mx.random.normal(shape, key=mx.random.key(19 + i)) * 0.08)
        for i, (name, shape) in enumerate(shapes.items())
    }
    if quantized:
        config["quantization"] = {"group_size": 32, "bits": 4}
        for name, array in list(arrays.items()):
            if array.ndim == 2:
                weight, scales, biases = mx.quantize(array, group_size=32, bits=4)
                prefix = name.removesuffix(".weight")
                arrays[name] = weight
                arrays[prefix + ".scales"] = scales
                arrays[prefix + ".biases"] = biases
    mx.save_safetensors(str(path / "model.safetensors"), arrays)
    (path / "config.json").write_text(json.dumps(config))
    # Deliberately omit bos_token_id to cover BOS-less chat tokenizers like Qwen3.
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": [1], "pad_token_id": 0}))
    _write_tokenizer(path, vocab_entries=30)
    return path


def _write_lfm2_moe_fixture(path, mx):
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "lfm2_moe", "hidden_size": 64, "vocab_size": 32,
        "num_hidden_layers": 3, "num_dense_layers": 1, "intermediate_size": 64,
        "layer_types": ["conv", "full_attention", "conv"],
        "num_attention_heads": 4, "num_key_value_heads": 2, "rope_theta": 5_000_000,
        "max_position_embeddings": 24, "norm_eps": 1e-5, "conv_L_cache": 3,
        "conv_bias": False, "tie_word_embeddings": True,
        "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
        "use_expert_bias": True, "norm_topk_prob": True, "routed_scaling_factor": 1.5,
        "quantization": {"bits": 4, "group_size": 32, "model.embed_tokens": {"bits": 6, "group_size": 32}},
        "eos_token_id": 1, "pad_token_id": 0,
    }
    arrays = {}

    def sample(shape):
        return (mx.random.normal(shape, key=mx.random.key(29 + len(arrays))) * 0.12).astype(mx.bfloat16)

    def weight(name, shape, *, bits=4):
        packed, scales, biases = mx.quantize(sample(shape), group_size=32, bits=bits)
        arrays.update({name + ".weight": packed, name + ".scales": scales, name + ".biases": biases})

    def norm(name, width):
        arrays[name + ".weight"] = mx.ones((width,), dtype=mx.bfloat16)

    weight("model.embed_tokens", (32, 64), bits=6)
    norm("model.embedding_norm", 64)
    for i, kind in enumerate(config["layer_types"]):
        layer = f"model.layers.{i}"
        norm(layer + ".operator_norm", 64)
        norm(layer + ".ffn_norm", 64)
        if kind == "conv":
            conv = layer + ".conv"
            weight(conv + ".in_proj", (192, 64))
            weight(conv + ".out_proj", (64, 64))
            arrays[conv + ".conv.weight"] = sample((64, 3, 1))
        else:
            attn = layer + ".self_attn"
            for name, rows in (("q_proj", 64), ("k_proj", 32), ("v_proj", 32), ("out_proj", 64)):
                weight(attn + "." + name, (rows, 64))
            norm(attn + ".q_layernorm", 16)
            norm(attn + ".k_layernorm", 16)
        ffn = layer + ".feed_forward"
        if i:
            arrays[ffn + ".gate.weight"] = sample((4, 64)).astype(mx.float32)
            arrays[ffn + ".expert_bias"] = mx.array([-1.0, -1.0, 1.0, 1.0], dtype=mx.bfloat16)
            ffn += ".switch_mlp"
        for name in ("gate_proj", "up_proj", "down_proj"):
            weight(ffn + "." + name, (4, 64, 64) if i else (64, 64))

    groups = [{k: v for k, v in arrays.items() if (".switch_mlp." in k) == expert} for expert in (False, True)]
    mapping = {}
    for fname, group in zip(("resident.safetensors", "experts.safetensors"), groups, strict=True):
        mx.save_safetensors(str(path / fname), group)
        mapping.update(dict.fromkeys(group, fname))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    (path / "config.json").write_text(json.dumps(config))
    _write_tokenizer(path, vocab_entries=32)
    return path


def _write_inkling2_fixture(path, mx):
    path.mkdir(parents=True, exist_ok=True)
    local = [0, 2, 4, 6]
    config = {
        "model_type": "inkling2_loop",
        "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
        "eos_token_id": 1,
        "pad_token_id": 0,
        "text_config": {
            "num_hidden_layers": 8, "hidden_size": 32, "intermediate_size": 32,
            "n_routed_experts": 2, "num_experts_per_tok": 2, "route_scale": 8.0,
            "use_gate_bias": True, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,
            "model_max_length": 24, "vocab_size": 64, "unpadded_vocab_size": 32,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
            "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1, "swa_head_dim": 16,
            "d_rel": 2, "rel_extent": 3, "sliding_window_size": 3, "sconv_kernel_size": 3,
            "local_layer_ids": local, "rms_norm_eps": 1e-6, "use_value_norm_no_gamma": True,
            "log_scaling_n_floor": 2, "log_scaling_alpha": 0.1,
            "qk_l2_clip_threshold": 2.0, "rel_l2_clip_threshold": 0.5, "final_logit_softcapping": 10.0,
            "n_layers_per_attn_res_block": 2, "attn_res_normalize_values": True,
            "loop_config": {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 4, "is_varloop": False},
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    index = {}
    prefix = "model.language_model."
    for layer in range(8):
        nvfp4 = layer < 7
        one, two, columns, scales, unit_scale = (
            (0x22222222, 0x44444444, 4, 2, 0x38) if nvfp4
            else (0x38383838, 0x40404040, 8, 1, 127)
        )
        m_prefix = f"{prefix}layers.{layer}.mlp."
        tensors = {
            "gate_up_proj.weight": mx.tile(mx.array([[one], [two]], dtype=mx.uint32), (2, 32, columns)),
            "gate_up_proj.scales": mx.full((2, 64, scales), unit_scale, dtype=mx.uint8),
            "down_proj.weight": mx.full((2, 32, columns), one, dtype=mx.uint32),
            "down_proj.scales": mx.full((2, 32, scales), unit_scale, dtype=mx.uint8),
        }
        if nvfp4:
            tensors["gate_up_proj.global_scale"] = mx.array([2688.0, 1344.0], dtype=mx.float32)
            tensors["down_proj.global_scale"] = mx.array([2688.0, 1344.0], dtype=mx.float32)
        shard = f"model-{layer + 1:05d}-of-00008.safetensors"
        weights = {m_prefix + "experts." + k: v for k, v in tensors.items()}
        weights[m_prefix + "gate.weight"] = mx.broadcast_to(mx.array([[0.0], [1.0]]), (2, 32))
        weights[m_prefix + "gate.e_score_correction_bias"] = mx.array([1.0, 0.0])
        mx.save_safetensors(str(path / shard), weights)
        index.update(dict.fromkeys(weights, shard))

    tensors = {}

    def sample(shape, dtype=mx.bfloat16):
        return (mx.random.normal(shape, key=mx.random.key(37 + len(tensors))) * 0.12).astype(dtype)

    def projection(name, rows):
        packed, scales, biases = mx.quantize(sample((rows, 32)), group_size=32, bits=4)
        tensors.update({name + ".weight": packed, name + ".scales": scales, name + ".biases": biases})

    def norm(name):
        tensors[name + ".weight"] = mx.ones((32,), dtype=mx.bfloat16)

    projection(prefix + "embed_tokens", 64)
    projection("lm_head", 64)
    norm(prefix + "embed_norm")
    norm(prefix + "norm")
    tensors[prefix + "final_input.query"] = sample((32,), mx.float32)
    tensors[prefix + "message_input.query"] = sample((4, 4, 32), mx.float32)
    for layer in range(8):
        lkey = f"{prefix}layers.{layer}."
        attn = lkey + "self_attn."
        for name, rows in (("q_proj", 32), ("k_proj", 16), ("v_proj", 16), ("r_proj", 4),
                           ("o_proj", 32), ("gate_proj", 32), ("sink_proj", 2)):
            projection(attn + name, rows)
        for name, dim in (("q_l2_clip", 16), ("rel_l2_clip", 2)):
            tensors[f"{attn}{name}.scale"] = mx.ones((2, dim), dtype=mx.bfloat16)
        tensors[attn + "rel_logits_proj.proj"] = sample((2, 3))
        for name in ("k_sconv", "v_sconv"):
            if layer in local:
                tensors[f"{attn}{name}.conv1d.weight"] = sample((16, 3, 1))
            else:
                tensors[f"{attn}{name}.static_weight"] = sample((3, 16))
                projection(f"{attn}{name}.weight_proj", 3)
        if layer:
            tensors[lkey + "attention_input.query"] = sample((32,) if layer in local else (3, 32), mx.float32)
        inputs = ("input_layernorm",) if layer in local else tuple(f"{h}_input_layernorm" for h in "qkv")
        for name in (*inputs, "post_attention_layernorm"):
            norm(lkey + name)
        tensors[lkey + "mlp_input.query"] = sample((32,), mx.float32)
        for name in ("post_attention_convolution", "post_mlp_convolution"):
            tensors[f"{lkey}{name}.conv1d.weight"] = sample((32, 3, 1))
        if 2 <= layer < 6:
            for name in ("q", "k", "v", "mlp"):
                gate = f"{lkey}{name}_message_gate."
                tensors[gate + "down_proj.weight"] = sample((4, 64))
                tensors[gate + "forget_gate_proj.weight"] = sample((32, 4), mx.float32)
                tensors[gate + "message_gate_proj.weight"] = sample((32, 4), mx.float32)

    mx.save_safetensors(str(path / "backbone.safetensors"), tensors)
    index.update(dict.fromkeys(tensors, "backbone.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    _write_tokenizer(path, vocab_entries=32)
    return path


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("method", ["sft", "distill"])
def test_qwen3_external_mlx_posttrain_resume_and_reload(tmp_path, quantized, method):
    import mlx.core as mx

    ckpt = _write_qwen3_fixture(tmp_path / "base", mx, quantized=quantized)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"prompt": p, "response": "t4 t5"}) for p in ("t2 t3", "t6 t7")))
    opts = dict(
        method=method, hardware="mlx", finetune="qlora" if not quantized else "lora",
        lora_rank=4, max_steps=3, batch_size=1, max_length=12, save_every=1, learning_rate=0.002,
    )
    final = posttrain(ckpt, data, tmp_path / "out", **opts)
    assert final.name == "model.adapter.safetensors"
    expected = mx.load(str(final))
    mx.eval(expected)
    assert any(k.endswith("lora_b") and float(mx.sum(mx.abs(v)).item()) > 0 for k, v in expected.items())

    resumed = posttrain(
        ckpt, data, tmp_path / "out",
        resume=tmp_path / "out" / "checkpoint_step_1.adapter.resume.npz", **opts,
    )
    actual = mx.load(str(resumed))
    for key in expected:
        assert mx.allclose(expected[key], actual[key], rtol=1e-5, atol=1e-6).item(), key

    loaded, backend, vocab_size = load_model_from_checkpoint(tmp_path / "out")
    assert backend == "mlx" and vocab_size == 32
    assert loaded.hidden_states(mx.array([[2, 3]]), is_causal=True).shape == (1, 2, 64)
    gen = TextGenerator.from_checkpoint(tmp_path / "out")
    assert len(gen.generate("t2 t3", max_new_tokens=3, temperature=0.0)) > 0


def test_lfm2_moe_resident_and_ssd_posttrain(tmp_path, monkeypatch):
    import mlx.core as mx
    from telos.models import qwen3_moe_mlx as loader
    from telos.models.ssd_experts import is_expert_tensor, tensor_ranges

    ckpt = _write_lfm2_moe_fixture(tmp_path / "lfm2", mx)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"prompt": p, "response": "t4"}) for p in ("t2 t3", "t5 t6")))

    class FixtureGuard:
        budget = 2**30

        def check(self):
            pass

    monkeypatch.setattr(loader, "AppleMemoryGuard", FixtureGuard)
    resident_out = posttrain(
        ckpt, data, tmp_path / "resident_out",
        method="distill", hardware="mlx", finetune="lora", lora_rank=4,
        max_steps=2, batch_size=2, max_length=10, save_every=0, learning_rate=0.002,
    )
    assert any(float(mx.sum(mx.abs(v)).item()) > 0 for k, v in mx.load(str(resident_out)).items() if k.endswith("lora_b"))

    ranges = tensor_ranges(ckpt / "model.safetensors.index.json")
    cfg = json.loads((ckpt / "config.json").read_text())
    # Find the lowest admissible budget so LFM2-MoE must stream experts from SSD.
    low, high = 0, 2**30
    while low < high:
        midpoint = (low + high) // 2
        try:
            loader.plan_experts(cfg, ranges, midpoint, None, 0)
        except ValueError:
            low = midpoint + 1
        else:
            high = midpoint
    _, ssd_cache = loader.plan_experts(cfg, ranges, high, None, 0)
    assert ssd_cache is not None
    FixtureGuard.budget = high

    ssd_out = posttrain(
        ckpt, data, tmp_path / "ssd_out",
        method="sft", hardware="mlx", finetune="lora", lora_rank=4,
        max_steps=2, batch_size=1, max_length=10, save_every=0, learning_rate=0.002,
    )
    assert ssd_out.is_file()
    ssd_loaded, ssd_backend, _ = load_model_from_checkpoint(tmp_path / "ssd_out")
    assert ssd_backend == "mlx" and ssd_loaded.weight_storage == "ssd_experts"


def test_inkling2_loop_posttrain_dpo_and_guards(tmp_path, monkeypatch):
    import mlx.core as mx
    from telos.models import inkling2_mlx

    class FixtureGuard:
        budget = 2**30

        def check(self):
            pass

    monkeypatch.setattr(inkling2_mlx, "AppleMemoryGuard", FixtureGuard)
    ckpt = _write_inkling2_fixture(tmp_path / "inkling", mx)
    data = tmp_path / "dpo.jsonl"
    data.write_text("\n".join(
        json.dumps({"prompt": p, "chosen": "t4", "rejected": "t5"}) for p in ("t2 t3", "t6 t7")
    ))

    out = posttrain(
        ckpt, data, tmp_path / "dpo_out",
        method="dpo", hardware="mlx", finetune="lora", lora_rank=4,
        max_steps=2, batch_size=1, max_length=10, save_every=0, learning_rate=0.002,
    )
    assert out.name == "model.adapter.safetensors"
    loaded, backend, vocab_size = load_model_from_checkpoint(tmp_path / "dpo_out")
    assert backend == "mlx" and vocab_size == 32
    assert loaded.hidden_states(mx.array([[2, 3]]), is_causal=True).shape == (1, 2, 32)

    sft_data = tmp_path / "sft.jsonl"
    sft_data.write_text(json.dumps({"prompt": "t2", "response": "t4"}) + "\n")
    with pytest.raises(ValueError, match="lora or qlora"):
        posttrain(ckpt, sft_data, tmp_path / "bad_full", method="sft", hardware="mlx", finetune="full", max_steps=1)
    with pytest.raises(ValueError, match="GRPO"):
        posttrain(
            ckpt, tmp_path / "grpo.jsonl", tmp_path / "bad_grpo",
            method="grpo", hardware="mlx", finetune="lora", max_steps=1,
        )


def test_mlx_posttrain_shares_budget_with_separate_teacher(tmp_path, monkeypatch):
    import mlx.core as mx
    from telos.posttrain import memory, trainer

    mib = 2**20
    monkeypatch.setattr(memory, "_rss_bytes", lambda: 128 * mib)
    monkeypatch.setattr(memory, "_command", lambda *args: str(16 * 1024 * mib))
    monkeypatch.setattr(memory, "_free_percent", lambda: 90)
    monkeypatch.setattr(memory, "_swapout_bytes", lambda: 0)
    checkpoint = _write_lfm2_moe_fixture(tmp_path / "moe", mx)
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"prompt": "t2 t3", "response": "t4"}) + "\n")
    loads = []
    actual_load = trainer._load_mlx_compatible

    def record_load(path, **kwargs):
        loads.append((kwargs["memory_budget_mib"], kwargs["memory_guard"]))
        return actual_load(path, **kwargs)

    monkeypatch.setattr(trainer, "_load_mlx_compatible", record_load)
    out = posttrain(
        checkpoint, data, tmp_path / "out", method="distill", teacher=checkpoint,
        hardware="mlx", finetune="lora", lora_rank=4, memory_budget_mib=1536,
        max_steps=1, batch_size=1, max_length=10, save_every=0,
    )
    assert out.is_file()
    assert [share for share, _ in loads] == [704, 704]
    assert loads[0][1] is loads[1][1]

    from telos.models.ssd_experts import TensorRange

    def reject_payload(*args, **kwargs):
        pytest.fail("A rejected training plan tried to read weight payloads")

    monkeypatch.setattr(TensorRange, "load", reject_payload)
    with pytest.raises(ValueError, match="cannot fit"):
        posttrain(
            checkpoint, data, tmp_path / "rejected", method="distill", teacher=checkpoint,
            hardware="mlx", finetune="lora", memory_budget_mib=1000,
            max_steps=1, batch_size=1, max_length=10, save_every=0,
        )


def test_posttrain_memory_guard_checks_rss_and_swap(monkeypatch):
    from telos.posttrain import memory

    mib = 2**20
    rss, swap = 128 * mib, 0
    monkeypatch.setattr(memory, "_rss_bytes", lambda: rss)
    monkeypatch.setattr(memory, "_command", lambda *args: str(16 * 1024 * mib))
    monkeypatch.setattr(memory, "_free_percent", lambda: 90)
    monkeypatch.setattr(memory, "_swapout_bytes", lambda: swap)
    with memory.MLXPosttrainMemory(1536) as guard:
        rss = 1536 * mib
        with pytest.raises(RuntimeError, match="exceeded its 1536 MiB memory budget"):
            guard.check(force=True)
        rss, swap = 128 * mib, 16384
        with pytest.raises(RuntimeError, match="new swap writes"):
            guard.check(force=True)
