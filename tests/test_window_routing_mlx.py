"""Tests for shared MLX window-pinned MoE decode routing and H100 adapter loading."""

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from telos.inference import TextGenerator
from telos.models.window_routing import apply_window_adapter
from telos.training.core import metal_usable

pytestmark = pytest.mark.skipif(not metal_usable(), reason="MLX Metal required")


def _write_tokenizer(path, vocab_entries=32):
    vocab = {"[UNK]": 0, **{f"t{i}": i for i in range(1, vocab_entries)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))


def _write_inkling_fixture(path, mx):
    path.mkdir(parents=True, exist_ok=True)
    local = [0, 2, 4, 6]
    config = {
        "model_type": "inkling2_loop",
        "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
        "eos_token_id": 31,
        "pad_token_id": 0,
        "text_config": {
            "num_hidden_layers": 8, "hidden_size": 32, "intermediate_size": 32,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "route_scale": 8.0,
            "use_gate_bias": True, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,
            "model_max_length": 32, "vocab_size": 64, "unpadded_vocab_size": 32,
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
            "gate_up_proj.weight": mx.tile(mx.array([[one], [two]], dtype=mx.uint32), (4, 32, columns)),
            "gate_up_proj.scales": mx.full((4, 64, scales), unit_scale, dtype=mx.uint8),
            "down_proj.weight": mx.full((4, 32, columns), one, dtype=mx.uint32),
            "down_proj.scales": mx.full((4, 32, scales), unit_scale, dtype=mx.uint8),
        }
        if nvfp4:
            tensors["gate_up_proj.global_scale"] = mx.array([2688.0, 1344.0, 2688.0, 1344.0], dtype=mx.float32)
            tensors["down_proj.global_scale"] = mx.array([2688.0, 1344.0, 2688.0, 1344.0], dtype=mx.float32)
        shard = f"model-{layer + 1:05d}-of-00008.safetensors"
        weights = {m_prefix + "experts." + k: v for k, v in tensors.items()}
        weights[m_prefix + "gate.weight"] = mx.random.normal((4, 32), key=mx.random.key(101 + layer)) * 0.5
        weights[m_prefix + "gate.e_score_correction_bias"] = mx.array([0.1, -0.1, 0.2, -0.2])
        mx.save_safetensors(str(path / shard), weights)
        index.update(dict.fromkeys(weights, shard))

    tensors = {}

    def sample(shape, dtype=mx.bfloat16):
        return (mx.random.normal(shape, key=mx.random.key(41 + len(tensors))) * 0.12).astype(dtype)

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
    _write_tokenizer(path)
    return path


def _write_lfm2_fixture(path, mx):
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
        "eos_token_id": 31, "pad_token_id": 0,
    }
    arrays = {}

    def sample(shape):
        return (mx.random.normal(shape, key=mx.random.key(17 + len(arrays))) * 0.25).astype(mx.bfloat16)

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
            arrays[ffn + ".expert_bias"] = mx.zeros((4,), dtype=mx.bfloat16)
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
    _write_tokenizer(path)
    return path


def test_inkling2_window_pinned_decode_and_h100_adapter(tmp_path, monkeypatch):
    import mlx.core as mx
    from telos.models import inkling2_mlx

    class FixtureGuard:
        budget = 2**30

        def check(self):
            pass

    monkeypatch.setattr(inkling2_mlx, "AppleMemoryGuard", FixtureGuard)
    ckpt = _write_inkling_fixture(tmp_path / "inkling", mx)

    gen = TextGenerator.from_checkpoint(ckpt, window_size=4)
    gen.model.route_trace = []
    out = gen.generate("t1 t2", max_new_tokens=6, temperature=0.0, cache="dynamic")
    assert len(out) > 0

    decode_traces = [row for row in gen.model.route_trace if row["phase"] == "decode"]
    assert decode_traces
    # Decode steps 1..4 belong to window 0 and share exact pinned expert IDs per physical layer;
    # step 5 crosses the W=4 boundary and may refresh.
    by_layer_step = {}
    for row in decode_traces:
        by_layer_step.setdefault(row["layer"], {})[row["token_index"], row["loop_pass"]] = row["experts"]
    for layer, entries in by_layer_step.items():
        steps = sorted({tok for tok, _ in entries})
        assert len(steps) >= 5
        win0 = entries[steps[0], 0]
        # Loop passes 1 and 2 on token 0 match pass 0, and tokens 1..3 keep the same pinned experts.
        for tok in steps[:4]:
            for loop_pass in sorted({lp for t, lp in entries if t == tok}):
                assert entries[tok, loop_pass] == win0, (layer, tok, loop_pass)

    # Export a synthetic H100-schema window adapter and load it directly onto Inkling2TextMLX.
    adapter_tensors = {}
    for layer in range(8):
        m_prefix = f"model.language_model.layers.{layer}.mlp.gate."
        adapter_tensors[m_prefix + "selector.selector_a"] = mx.ones((4, 32), dtype=mx.float32) * 0.25
        adapter_tensors[m_prefix + "selector.selector_b"] = mx.random.normal((4, 4), key=mx.random.key(7 + layer))
        adapter_tensors[m_prefix + "selector.bias"] = mx.array([2.0, 2.0, -2.0, -2.0], dtype=mx.float32)
        adapter_tensors[m_prefix + "token_a"] = mx.ones((4, 32), dtype=mx.float32) * 0.1
        adapter_tensors[m_prefix + "token_b"] = mx.ones((4, 4), dtype=mx.float32) * 0.1
        a_prefix = f"model.language_model.layers.{layer}.self_attn."
        for proj, rows in (("q_proj", 32), ("k_proj", 16), ("v_proj", 16), ("o_proj", 32)):
            adapter_tensors[f"{a_prefix}{proj}.lora_a"] = mx.ones((2, 32), dtype=mx.float32) * 0.05
            adapter_tensors[f"{a_prefix}{proj}.lora_b"] = mx.ones((rows, 2), dtype=mx.float32) * 0.05
        # Include the nested HF sub-projection naming variant emitted when gate_proj wraps a sub-linear.
        adapter_tensors[f"{a_prefix}gate_proj.gate_proj.lora_a"] = mx.ones((2, 32), dtype=mx.float32) * 0.05
        adapter_tensors[f"{a_prefix}gate_proj.gate_proj.lora_b"] = mx.ones((32, 2), dtype=mx.float32) * 0.05

    adapter_path = tmp_path / "inkling2_w16_adapter.safetensors"
    mx.save_safetensors(str(adapter_path), adapter_tensors)

    adapted = TextGenerator.from_checkpoint(ckpt, window_adapter=adapter_path, window_size=4)
    adapted.model.route_trace = []
    _ = adapted.generate("t1 t2", max_new_tokens=5, temperature=0.0, cache="dynamic")
    # Strong [2, 2, -2, -2] selector bias must pin experts [0, 1] across window 0 decode steps.
    adapted_decode = [row for row in adapted.model.route_trace if row["phase"] == "decode"]
    assert all(row["experts"] == [0, 1] for row in adapted_decode if row["token_index"] == adapted_decode[0]["token_index"])


def test_lfm2_moe_ssd_window_pinning_reduces_reads_and_validates_args(tmp_path, monkeypatch):
    import mlx.core as mx
    from telos.models import qwen3_moe_mlx as loader
    from telos.models.ssd_experts import tensor_ranges

    ckpt = _write_lfm2_fixture(tmp_path / "lfm2", mx)

    class FixtureGuard:
        budget = 2**30

        def check(self):
            pass

    monkeypatch.setattr(loader, "AppleMemoryGuard", FixtureGuard)
    ranges = tensor_ranges(ckpt / "model.safetensors.index.json")
    cfg = json.loads((ckpt / "config.json").read_text())
    low, high = 0, 2**30
    while low < high:
        midpoint = (low + high) // 2
        try:
            loader.plan_experts(cfg, ranges, midpoint, None, 0)
        except ValueError:
            low = midpoint + 1
        else:
            high = midpoint
    # Give enough cache for one layer's pinned top_k experts (2 of 4 per MoE layer).
    _, ssd_cache = loader.plan_experts(cfg, ranges, high + 32768, None, 0)
    assert ssd_cache is not None
    FixtureGuard.budget = high + 32768

    gen = TextGenerator.from_checkpoint(ckpt, window_size=4)
    assert gen.model.weight_storage == "ssd_experts"
    res = gen.generate("t1 t2", max_new_tokens=6, temperature=0.0, cache="dynamic")
    assert len(res) > 0
    # Verify manual controller pinning keeps cache reads bounded to first step of each window.
    gen.model.set_window_size(4)
    hidden, cache = gen.model.prefill_last_hidden(mx.array([[1, 2]]))
    mx.eval(hidden, cache)
    gen.model._expert_cache.begin_decode()
    with gen.model._expert_cache.request():
        reads_before = gen.model._expert_cache.bytes_read
        for _ in range(4):
            h, cache = gen.model.hidden_states_with_cache(mx.array([[3]]), kv_cache=cache)
            mx.eval(h, cache)
        win_reads = gen.model._expert_cache.bytes_read - reads_before

    unpinned = TextGenerator.from_checkpoint(ckpt)
    h, cache = unpinned.model.prefill_last_hidden(mx.array([[1, 2]]))
    mx.eval(h, cache)
    unpinned.model._expert_cache.begin_decode()
    with unpinned.model._expert_cache.request():
        reads_before = unpinned.model._expert_cache.bytes_read
        # Alternate distinct tokens so unpinned routing keeps flipping experts vs a single window.
        for tok in (3, 4, 5, 6):
            h, cache = unpinned.model.hidden_states_with_cache(mx.array([[tok]]), kv_cache=cache)
            mx.eval(h, cache)
        unpinned_reads = unpinned.model._expert_cache.bytes_read - reads_before
    assert win_reads <= unpinned_reads

    with pytest.raises(ValueError, match="positive integer"):
        TextGenerator.from_checkpoint(ckpt, window_size=0)
    assert apply_window_adapter(gen.model, tmp_path / "missing" if False else _save_noop_lfm2_adapter(tmp_path, mx), window_size=4) is gen.model


def _save_noop_lfm2_adapter(tmp_path, mx):
    path = tmp_path / "lfm2_adapter.safetensors"
    mx.save_safetensors(str(path), {
        "model.layers.1.mlp.gate.selector.selector_a": mx.zeros((4, 64), dtype=mx.float32),
        "model.layers.1.mlp.gate.selector.selector_b": mx.zeros((4, 4), dtype=mx.float32),
        "model.layers.1.self_attn.q_proj.lora_a": mx.zeros((2, 64), dtype=mx.float32),
        "model.layers.1.self_attn.q_proj.lora_b": mx.zeros((64, 2), dtype=mx.float32),
    })
    return path


def test_local_mlx_window_distill_inkling2_resume_and_reload(tmp_path, monkeypatch):
    import mlx.core as mx
    from telos.eval.runner import load_model_from_checkpoint
    from telos.models import inkling2_mlx
    from telos.posttrain.trainer import posttrain

    class FixtureGuard:
        budget = 2**30

        def check(self):
            pass

    monkeypatch.setattr(inkling2_mlx, "AppleMemoryGuard", FixtureGuard)
    ckpt = _write_inkling_fixture(tmp_path / "inkling", mx)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"prompt": p, "response": "t4 t5 t6 t7"}) for p in ("t1 t2", "t3 t4")))

    opts = dict(
        method="distill", hardware="mlx", finetune="lora", lora_rank=4,
        max_steps=3, batch_size=2, max_length=12, save_every=1, learning_rate=0.005,
        window_size=4, selector_rank=4, window_cov_weight=0.25,
    )
    final = posttrain(ckpt, data, tmp_path / "win_out", **opts)
    saved = mx.load(str(final))
    mx.eval(saved)
    assert any(k.endswith("selector_b") and float(mx.sum(mx.abs(v)).item()) > 0 for k, v in saved.items())
    assert any(k.endswith("token_b") and float(mx.sum(mx.abs(v)).item()) > 0 for k, v in saved.items())
    assert any(k.endswith("lora_b") and float(mx.sum(mx.abs(v)).item()) > 0 for k, v in saved.items())

    resumed = posttrain(
        ckpt, data, tmp_path / "win_out",
        resume=tmp_path / "win_out" / "checkpoint_step_1.adapter.resume.npz", **opts,
    )
    actual = mx.load(str(resumed))
    for key, expected in saved.items():
        assert mx.allclose(expected, actual[key], rtol=1e-5, atol=1e-6).item(), key

    loaded, backend, _ = load_model_from_checkpoint(tmp_path / "win_out")
    assert backend == "mlx" and loaded._window_controller.window_size == 4
    gen = TextGenerator.from_checkpoint(tmp_path / "win_out")
    assert gen.model._window_controller.window_size == 4
    assert len(gen.generate("t1 t2", max_new_tokens=5, temperature=0.0)) > 0

    with pytest.raises(ValueError, match="distill"):
        posttrain(ckpt, data, tmp_path / "bad_sft", method="sft", hardware="mlx", finetune="lora", max_steps=1, window_size=4)


def test_local_mlx_window_distill_lfm2_moe_ssd_and_cli(tmp_path, monkeypatch):
    import sys
    import mlx.core as mx
    from telos.models import qwen3_moe_mlx as loader
    from telos.models.ssd_experts import tensor_ranges
    from telos.posttrain.cli import main as posttrain_cli

    ckpt = _write_lfm2_fixture(tmp_path / "lfm2", mx)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"prompt": p, "response": "t4 t5"}) for p in ("t1 t2", "t3")))

    class FixtureGuard:
        budget = 2**30

        def check(self):
            pass

    monkeypatch.setattr(loader, "AppleMemoryGuard", FixtureGuard)
    ranges = tensor_ranges(ckpt / "model.safetensors.index.json")
    cfg = json.loads((ckpt / "config.json").read_text())
    low, high = 0, 2**30
    while low < high:
        midpoint = (low + high) // 2
        try:
            loader.plan_experts(cfg, ranges, midpoint, None, 0)
        except ValueError:
            low = midpoint + 1
        else:
            high = midpoint
    FixtureGuard.budget = high + 32768

    out_dir = tmp_path / "lfm2_win"
    monkeypatch.setattr(sys, "argv", [
        "telos-posttrain",
        "--method", "distill",
        "--checkpoint", str(ckpt),
        "--data", str(data),
        "--output", str(out_dir),
        "--hardware", "mlx",
        "--finetune", "lora",
        "--lora-rank", "4",
        "--max-steps", "2",
        "--batch-size", "2",
        "--max-length", "10",
        "--save-every", "0",
        "--learning-rate", "0.005",
        "--window-size", "4",
        "--selector-rank", "4",
        "--window-cov-weight", "0.25",
    ])
    posttrain_cli()
    saved = mx.load(str(out_dir / "model.adapter.safetensors"))
    mx.eval(saved)
    assert any(k.endswith("selector_b") and float(mx.sum(mx.abs(v)).item()) > 0 for k, v in saved.items())
    gen = TextGenerator.from_checkpoint(out_dir)
    assert gen.model.weight_storage == "ssd_experts" and gen.model._window_controller.window_size == 4
    assert len(gen.generate("t1 t2", max_new_tokens=4, temperature=0.0)) > 0


def test_sequence_windows_keep_the_original_experts_across_chunks():
    import mlx.core as mx
    from telos.models.window_routing import WindowPinController, compute_route

    controller = WindowPinController(4)

    def route(values):
        hidden = mx.array(values)[..., None]
        ids, _ = compute_route(
            hidden, lambda x: mx.concatenate((x, -x), axis=-1),
            layer=0, top_k=1, controller=controller,
        )
        return ids.reshape(hidden.shape[:2]).tolist()

    controller.reset_sequence_student(4, [1, 5])
    assert route([[1.0] * 4, [1.0] * 4]) == [[0] * 4, [0] * 4]
    controller.chunk_offset = 4
    assert route([[-1.0] * 4, [-1.0, 1.0, -1.0, -1.0]]) == [[0, 1, 1, 1], [1, 0, 0, 0]]
    # A later visit of the same physical bank must also keep those experts.
    assert route([[1.0] * 4, [1.0] * 4]) == [[0, 1, 1, 1], [0, 0, 0, 0]]
    controller.chunk_offset = 8
    assert route([[1.0, 1.0], [-1.0, -1.0]]) == [[1, 0], [0, 1]]
    for row, windows in enumerate(((0, 1, 2), (0, 1))):
        anchors = mx.concatenate(controller.seq_selector_windows[0], axis=1).tolist()[row]
        assert tuple(w for w in anchors if w >= 0) == windows


def test_ssd_segmented_vjp_matches_exact_autograd(tmp_path, monkeypatch):
    """Verify NOGRAD forward + layerwise VJP matches full autograd identically."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from telos.models import inkling2_mlx, qwen3_moe_mlx as loader
    from telos.models.checkpoint import load_checkpoint
    from telos.models.ssd_experts import tensor_ranges
    from telos.models.window_routing import init_window_adapters
    from telos.posttrain.mlx_adapters import apply_lora
    from telos.posttrain.trainer import _mlx_distill_from_logits, _mlx_window_distill_loss, _run_window_teacher

    class FixtureGuard:
        budget = 2**30

        def check(self):
            pass

    monkeypatch.setattr(inkling2_mlx, "AppleMemoryGuard", FixtureGuard)
    monkeypatch.setattr(loader, "AppleMemoryGuard", FixtureGuard)

    ckpt = _write_inkling_fixture(tmp_path / "inkling_eq", mx)
    model, _, _, _ = load_checkpoint(ckpt, device="mlx")
    apply_lora(model, rank=4, alpha=16.0, targets=["q", "k", "v", "output", "gate"])
    init_window_adapters(model, selector_rank=4)
    # Give non-zero lora_b so attention projections receive distinct downstream gradients.
    for mod in model.layers:
        mod.attention.q.lora_b = mx.ones_like(mod.attention.q.lora_b) * 0.05
    mx.eval(model.trainable_parameters())

    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)
    labels = mx.array([[-100, -100, -100, 4, 5, 6, 7, 8]], dtype=mx.int32)
    pl = [3]
    t_logits, t_routes = _run_window_teacher(model, ids, pl)
    t_logits = mx.stop_gradient(t_logits)
    mx.eval(t_logits, t_routes)

    ref_loss, ref_grads = nn.value_and_grad(model, _mlx_window_distill_loss)(
        model, ids, labels, t_logits, t_routes, pl, 0.9, 1.0, 0.25, 4,
    )
    seg_loss, seg_grads = model.ssd_loss_and_grad(
        ids,
        lambda logits: _mlx_distill_from_logits(logits.astype(mx.float32), labels, t_logits, 0.9, 1.0),
        teacher_routes=t_routes,
        prompt_lengths=pl,
        response_mask=labels != -100,
        window_size=4,
        cov_weight=0.25,
    )
    mx.eval(ref_loss, ref_grads, seg_loss, seg_grads)
    assert mx.allclose(ref_loss, seg_loss, rtol=1e-4, atol=1e-5).item()
    ref_flat = dict(tree_flatten(ref_grads))
    seg_flat = dict(tree_flatten(seg_grads))
    assert ref_flat.keys() == seg_flat.keys()
    for name in ref_flat:
        assert mx.allclose(ref_flat[name], seg_flat[name], rtol=1e-3, atol=1e-4).item(), name
