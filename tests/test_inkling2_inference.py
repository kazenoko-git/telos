"""Tiny sharded fixtures for Inkling's fixed-loop layout and packed experts."""

import json
import math
from dataclasses import replace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from telos.inference import TextGenerator
from telos.models.inkling2 import InklingTextLayout
from telos.models.ssd_experts import SSDExpertCache, TensorRange, tensor_ranges
from telos.training.core import metal_usable

pytestmark = pytest.mark.skipif(not metal_usable(), reason="MLX Metal required")


def _write_fixture(path, mx, **text_options):
    config = {
        "model_type": "inkling2_loop",
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "intermediate_size": 32,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "route_scale": 8.0,
            "use_gate_bias": True,
            "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0,
            "loop_config": {"core_start": 0, "core_end": 1, "num_passes": 3, "is_varloop": False},
        },
    }
    config["text_config"].update(text_options)
    (path / "config.json").write_text(json.dumps(config))
    index = {}
    layers = config["text_config"]["num_hidden_layers"]
    for layer in range(layers):
        nvfp4 = layer < layers - 1
        one, two, columns, scales, unit_scale = (
            (0x22222222, 0x44444444, 4, 2, 0x38) if nvfp4  # E2M1 codes, E4M3 scale 1
            else (0x38383838, 0x40404040, 8, 1, 127)  # E4M3 codes, E8M0 scale 1
        )
        prefix = f"model.language_model.layers.{layer}.mlp."
        gate_up = mx.tile(mx.array([[one], [two]], dtype=mx.uint32), (2, 32, columns))
        tensors = {
            "gate_up_proj.weight": gate_up,
            "gate_up_proj.scales": mx.full((2, 64, scales), unit_scale, dtype=mx.uint8),
            "down_proj.weight": mx.full((2, 32, columns), one, dtype=mx.uint32),
            "down_proj.scales": mx.full((2, 32, scales), unit_scale, dtype=mx.uint8),
        }
        if nvfp4:
            tensors["gate_up_proj.global_scale"] = mx.array([2688.0, 1344.0], dtype=mx.float32)
            tensors["down_proj.global_scale"] = mx.array([2688.0, 1344.0], dtype=mx.float32)
        shard = f"model-{layer + 1:05d}-of-{layers:05d}.safetensors"
        weights = {prefix + "experts." + key: value for key, value in tensors.items()}
        weights[prefix + "gate.weight"] = mx.broadcast_to(mx.array([[0.0], [1.0]]), (2, 32))
        weights[prefix + "gate.e_score_correction_bias"] = mx.array([1.0, 0.0])
        mx.save_safetensors(str(path / shard), weights)
        index.update(dict.fromkeys(weights, shard))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return config


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_inkling2_fixed_loop_expert_projection_parity(tmp_path, dtype):
    import mlx.core as mx

    class FixtureGuard:
        def check(self):
            pass

    _write_fixture(tmp_path, mx)
    layout = InklingTextLayout.from_checkpoint(tmp_path)
    assert layout.visits == ((0, 0), (0, 2), (0, 3), (1, 1))
    assert tuple(bank.mode for bank in layout.expert_banks) == ("nvfp4", "mxfp8")
    first, second = layout.expert_banks
    first_bytes = sum(entry.size // 2 for entry in first.ranges.values())
    second_bytes = sum(entry.size // 2 for entry in second.ranges.values())
    cache = SSDExpertCache(max(first_bytes, second_bytes), FixtureGuard())
    x = mx.full((1, 32), 1 / 32, dtype=getattr(mx, dtype))

    for bank in (first, second):
        resident_tensors = mx.load(str(next(iter(bank.ranges.values())).path))
        for expert in (0, 1):
            prefix = f"model.language_model.layers.{bank.layer}.mlp.experts."
            resident = {key: resident_tensors[prefix + key][expert] for key in bank.ranges}
            streamed = cache.get(bank.layer, expert, bank.ranges)
            for key, array in resident.items():
                assert mx.array_equal(array, streamed[key]).item()
            projected = bank.project(x, streamed)
            assert mx.array_equal(projected, bank.project(x, resident)).item()
            outer = 0.5 if bank.mode == "nvfp4" and expert == 1 else 1.0
            expected = (32 * outer * 4 * math.tanh(outer / 4) / (1 + math.exp(-outer))
                        * 25 * math.tanh(2 * outer / 25))
            tolerance = 1e-5 if dtype == "float32" else 0.01
            assert mx.allclose(projected, mx.full((1, 32), expected), rtol=tolerance, atol=1e-5).item()
    assert cache.bytes_read == 2 * (first_bytes + second_bytes)
    assert cache.resident_bytes <= cache.byte_limit


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_nvfp4_projection_matches_dequantized_reference(tmp_path, dtype):
    import mlx.core as mx

    _write_fixture(tmp_path, mx)
    bank = InklingTextLayout.from_checkpoint(tmp_path).expert_banks[0]
    hidden = mx.random.normal((3, 32), key=mx.random.key(42)).astype(getattr(mx, dtype))
    packed, dense = {}, {}
    for i, (name, rows, scale) in enumerate((("gate_up_proj", 64, 1300.0), ("down_proj", 32, 2100.0))):
        weight = mx.random.normal((rows, 32), key=mx.random.key(43 + i))
        q, s = mx.quantize(weight, mode="nvfp4")
        global_scale = mx.array(scale, dtype=mx.float32)
        packed.update({name + ".weight": q, name + ".scales": s, name + ".global_scale": global_scale})
        dense[name] = mx.dequantize(q, s, mode="nvfp4", global_scale=global_scale, dtype=hidden.dtype)

    gate_up = (hidden @ dense["gate_up_proj"].T).astype(mx.float32)
    gate, up = gate_up[:, ::2], gate_up[:, 1::2]
    activation = (4 * mx.tanh(gate / 4) * mx.sigmoid(gate) * (25 * mx.tanh(up / 25))).astype(hidden.dtype)
    expected = activation @ dense["down_proj"].T
    assert mx.array_equal(bank.project(hidden, packed), expected).item()


def test_inkling2_rejects_invalid_loop_before_opening_weight_shards(tmp_path):
    config = {
        "model_type": "inkling2_loop",
        "text_config": {
            "num_hidden_layers": 2,
            "loop_config": {"core_start": 0, "core_end": 3, "num_passes": 2, "is_varloop": False},
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").touch()
    with pytest.raises(ValueError, match="loop bounds"):
        InklingTextLayout.from_checkpoint(tmp_path)


def test_inkling2_router_bias_affects_selection_but_not_mixture(tmp_path):
    import mlx.core as mx

    class FixtureGuard:
        def check(self):
            pass

    _write_fixture(tmp_path, mx)
    bank = InklingTextLayout.from_checkpoint(tmp_path).expert_banks[0]
    router = bank.load_router()
    x = mx.full((1, 32), 1 / 32, dtype=mx.float32)
    indices, weights = bank.route(x, router)
    assert indices.tolist() == [[0]]
    assert weights.tolist() == [[8.0]]

    bank = replace(bank, top_k=2)
    indices, weights = bank.route(x, router)
    scores = mx.array([0.5, 1 / (1 + math.exp(-1))])
    expected_weights = 8 * scores / scores.sum()
    assert indices.tolist() == [[0, 1]]
    assert mx.allclose(weights[0], expected_weights).item()

    one_expert = sum(entry.size // 2 for entry in bank.ranges.values())
    cache = SSDExpertCache(one_expert, FixtureGuard())
    expected = mx.zeros_like(x)
    for expert in (0, 1):
        resident = {key: entry.load()[expert] for key, entry in bank.ranges.items()}
        expected = expected + bank.project(x, resident) * weights[0, expert]
    assert mx.array_equal(bank.routed(x, router, cache), expected).item()
    assert cache.bytes_read == 2 * one_expert
    assert cache.resident_bytes == one_expert


def _write_text_fixture(path, mx):
    local = [0, 2, 4, 6]
    config = _write_fixture(
        path, mx, num_hidden_layers=8, num_experts_per_tok=2, model_max_length=24, vocab_size=64,
        unpadded_vocab_size=32, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        swa_num_attention_heads=2, swa_num_key_value_heads=1, swa_head_dim=16,
        d_rel=2, rel_extent=3, sliding_window_size=3, sconv_kernel_size=3, local_layer_ids=local,
        rms_norm_eps=1e-6, use_value_norm_no_gamma=True, log_scaling_n_floor=2, log_scaling_alpha=0.1,
        qk_l2_clip_threshold=2.0, rel_l2_clip_threshold=0.5, final_logit_softcapping=10.0,
        n_layers_per_attn_res_block=2, attn_res_normalize_values=True,
        loop_config={"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 4, "is_varloop": False},
    )
    config["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine"}
    (path / "config.json").write_text(json.dumps(config))
    tensors = {}
    prefix = "model.language_model."

    def sample(shape, dtype=mx.bfloat16):
        space = mx.random.normal(shape, key=mx.random.key(23 + len(tensors)))
        return (space * 0.12).astype(dtype)

    def weight(name, shape, dtype=mx.bfloat16):
        tensors[name] = sample(shape, dtype)

    def projection(name, rows):
        packed, scales, biases = mx.quantize(sample((rows, 32)), group_size=32, bits=4)
        tensors.update({name + ".weight": packed, name + ".scales": scales, name + ".biases": biases})

    def norm(name):
        tensors[name + ".weight"] = mx.ones((32,), dtype=mx.bfloat16)

    projection(prefix + "embed_tokens", 64)
    projection("lm_head", 64)
    norm(prefix + "embed_norm")
    norm(prefix + "norm")
    weight(prefix + "final_input.query", (32,), mx.float32)
    weight(prefix + "message_input.query", (4, 4, 32), mx.float32)
    for layer in range(8):
        layer_key = f"{prefix}layers.{layer}."
        attn = layer_key + "self_attn."
        for name, rows in (("q_proj", 32), ("k_proj", 16), ("v_proj", 16), ("r_proj", 4),
                           ("o_proj", 32), ("gate_proj", 32), ("sink_proj", 2)):
            projection(attn + name, rows)
        for name, dim in (("q_l2_clip", 16), ("rel_l2_clip", 2)):
            tensors[f"{attn}{name}.scale"] = mx.ones((2, dim), dtype=mx.bfloat16)
        weight(attn + "rel_logits_proj.proj", (2, 3))
        for name in ("k_sconv", "v_sconv"):
            if layer in local:
                weight(f"{attn}{name}.conv1d.weight", (16, 3, 1))
            else:
                weight(f"{attn}{name}.static_weight", (3, 16))
                projection(f"{attn}{name}.weight_proj", 3)
        if layer:
            shape = (32,) if layer in local else (3, 32)
            weight(layer_key + "attention_input.query", shape, mx.float32)
        inputs = ("input_layernorm",) if layer in local else tuple(f"{head}_input_layernorm" for head in "qkv")
        for name in (*inputs, "post_attention_layernorm"):
            norm(layer_key + name)
        weight(layer_key + "mlp_input.query", (32,), mx.float32)
        for name in ("post_attention_convolution", "post_mlp_convolution"):
            weight(f"{layer_key}{name}.conv1d.weight", (32, 3, 1))
        if 2 <= layer < 6:
            for name in ("q", "k", "v", "mlp"):
                gate = f"{layer_key}{name}_message_gate."
                weight(gate + "down_proj.weight", (4, 64))
                weight(gate + "forget_gate_proj.weight", (32, 4), mx.float32)
                weight(gate + "message_gate_proj.weight", (32, 4), mx.float32)

    shard = "backbone.safetensors"
    mx.save_safetensors(str(path / shard), tensors)
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].update(dict.fromkeys(tensors, shard))
    index_path.write_text(json.dumps(index))
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, **{f"t{i}": i for i in range(1, 32)}}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))
    return config


def test_inkling2_text_prefill_decode_and_ssd_parity(tmp_path, monkeypatch):
    import mlx.core as mx

    from telos.models import inkling2_mlx

    class FixtureGuard:
        budget = 2**30

        def check(self):
            pass

    config = _write_text_fixture(tmp_path, mx)
    text = config["text_config"]
    checkpoint = tmp_path / "model.safetensors.index.json"
    layout = InklingTextLayout.from_checkpoint(checkpoint)
    non_experts = sum(entry.size for name, entry in tensor_ranges(checkpoint).items() if ".mlp.experts." not in name)
    monkeypatch.setattr(inkling2_mlx, "AppleMemoryGuard", FixtureGuard)
    monkeypatch.setattr(inkling2_mlx, "_PREFILL_CHUNK", 3)
    resident = TextGenerator.from_checkpoint(tmp_path, memory_budget_mib=1024, prefix_cache_entries=2)
    assert resident.model.weight_storage == "resident"
    assert resident.prefix_cache.max_entries == 0
    assert resident.vocab_size == 32

    # Admit exactly one largest expert and reject anything smaller before loading payloads.
    low, high = 0, FixtureGuard.budget
    while low < high:
        mid = (low + high) // 2
        try:
            inkling2_mlx.plan_inkling(text, layout, non_experts, mid, 24)
        except ValueError:
            low = mid + 1
        else:
            high = mid
    FixtureGuard.budget = high - 1
    with monkeypatch.context() as patch:
        def reject_payload(*_args, **_kwargs):
            pytest.fail("A rejected plan attempted to open tensor payload")

        patch.setattr(TensorRange, "load", reject_payload)
        with pytest.raises(ValueError, match="cannot fit"):
            TextGenerator.from_checkpoint(tmp_path)
    FixtureGuard.budget = high
    streamed = TextGenerator.from_checkpoint(tmp_path, memory_budget_mib=1024)
    assert streamed.model.weight_storage == "ssd_experts"
    cache = streamed.model._expert_cache
    assert cache.byte_limit == max(inkling2_mlx._layer_sizes(layout).values())

    prompt = mx.array([[2, 3, 5, 7, 11, 13]], dtype=mx.int32)
    next_token = mx.array([[17]], dtype=mx.int32)
    left, left_state = resident.model.hidden_states_with_cache(prompt)
    with cache.request():
        right, right_state = streamed.model.hidden_states_with_cache(prompt)
        rd, right_next = streamed.model.hidden_states_with_cache(next_token, kv_cache=right_state)
    ld, left_next = resident.model.hidden_states_with_cache(next_token, kv_cache=left_state)
    assert mx.array_equal(left, right).item()
    assert mx.array_equal(ld, rd).item()
    assert mx.array_equal(resident.model.logits_from_hidden(ld), streamed.model.logits_from_hidden(rd)).item()
    assert len(left_state.visits) == len(right_state.visits) == 16
    assert left_state.length == right_state.length == 6
    assert left_next.length == right_next.length == 7
    for layer, slot in layout.visits:
        a, b = left_next.visits[slot], right_next.visits[slot]
        assert a.attention.kv.length == (2 if layer in text["local_layer_ids"] else 7)
        assert mx.array_equal(a.attention.kv.keys, b.attention.kv.keys).item()
        assert mx.array_equal(a.attention.kv.values, b.attention.kv.values).item()
        assert mx.array_equal(a.attention.key_history, b.attention.key_history).item()
        assert mx.array_equal(a.attention.value_history, b.attention.value_history).item()
        assert mx.array_equal(a.attention_history, b.attention_history).item()
        assert mx.array_equal(a.mlp_history, b.mlp_history).item()
    assert not mx.array_equal(left_state.visits[2].attention.kv.keys, left_state.visits[8].attention.kv.keys).item()

    options = (prompt[0].tolist(), 4, 0.0, 0, 1.0, None, "dynamic")
    assert list(resident._mlx_tokens(*options)) == list(streamed._mlx_tokens(*options))
    assert resident.generate("t2 t3 t5 t7 t11 t13", max_new_tokens=4) == streamed.generate(
        "t2 t3 t5 t7 t11 t13", max_new_tokens=4,
    )
    assert cache.bytes_read > cache.byte_limit and cache.resident_bytes <= cache.byte_limit


def test_inkling2_relative_attention_sink_and_sliding_history():
    import mlx.core as mx

    from telos.models.inkling2_attention import InklingAttention
    from telos.models.qwen3_mlx import _Weights

    config = {
        "swa_num_attention_heads": 1, "swa_num_key_value_heads": 1, "swa_head_dim": 2,
        "sliding_window_size": 2, "d_rel": 1, "hidden_size": 2, "sconv_kernel_size": 1,
        "rms_norm_eps": 1e-6, "qk_l2_clip_threshold": 10.0, "rel_l2_clip_threshold": 10.0,
        "use_value_norm_no_gamma": False,
    }
    tensors = {
        "attn.q_proj.weight": mx.zeros((2, 2)),
        "attn.k_proj.weight": mx.zeros((2, 2)),
        "attn.v_proj.weight": mx.eye(2),
        "attn.r_proj.weight": mx.array([[1.0, 0.0]]),
        "attn.gate_proj.weight": mx.zeros((2, 2)),
        "attn.sink_proj.weight": mx.zeros((1, 2)),
        "attn.o_proj.weight": mx.eye(2),
        "attn.q_l2_clip.scale": mx.ones((1, 2)),
        "attn.rel_l2_clip.scale": mx.ones((1, 1)),
        "attn.rel_logits_proj.proj": mx.array([[math.log(2), 0.0]]),
        "attn.k_sconv.conv1d.weight": mx.zeros((2, 1, 1)),
        "attn.v_sconv.conv1d.weight": mx.zeros((2, 1, 1)),
    }
    attention = InklingAttention(_Weights(tensors, {}), "attn", config, sliding=True)
    x = mx.array([[[1.0, 0.0], [1.0, 2.0], [1.0, 4.0]]])
    expected = mx.array([[[1 / 3, 0.0], [3 / 8, 4 / 8], [3 / 8, 10 / 8]]]) / math.sqrt(2)
    output, state = attention(x, x, x, None, 0, 8)
    assert mx.allclose(output, expected, rtol=1e-6, atol=1e-6).item()
    assert state.kv.length == 1
    prefill, tail = x[:, :2], x[:, 2:]
    _, prefix_state = attention(prefill, prefill, prefill, None, 0, 8)
    cached, _ = attention(tail, tail, tail, prefix_state, 2, 8)
    assert mx.allclose(cached, expected[:, 2:], rtol=1e-6, atol=1e-6).item()


def test_inkling2_dynamic_convolution_uses_per_head_lag_weights():
    import mlx.core as mx

    from telos.models.inkling2_attention import DynamicConvolution
    from telos.models.qwen3_mlx import _Weights

    weights = _Weights({
        "conv.static_weight": mx.array([[0.0, 0.5], [1.0, -0.5]]),
        "conv.weight_proj.weight": mx.array([[1.0], [2.0], [3.0], [4.0]]),
    }, {})
    convolution = DynamicConvolution(weights, "conv", width=1, channels=2, heads=2, kernel=2)
    x = mx.array([[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]])
    inputs = mx.array([[[1.0], [2.0], [0.0]]])
    expected = mx.array([[[2.0, 9.0], [14.0, 52.5], [10.0, 14.0]]]) / math.sqrt(5)
    full, _ = convolution(x, None, inputs)
    _, cache = convolution(x[:, :2], None, inputs[:, :2])
    tail, _ = convolution(x[:, 2:], cache, inputs[:, 2:])
    assert mx.allclose(full, expected, rtol=1e-6, atol=1e-6).item()
    assert mx.array_equal(tail, full[:, 2:]).item()


def test_inkling2_route_tracing_and_offline_replay_match_live(tmp_path, monkeypatch):
    import mlx.core as mx

    from telos.models import inkling2_mlx
    from telos.models.ssd_replay import replay_existing, replay_policy

    class FixtureGuard:
        budget = 2**30

        def check(self):
            pass

    config = _write_text_fixture(tmp_path, mx)
    text = config["text_config"]
    checkpoint = tmp_path / "model.safetensors.index.json"
    layout = InklingTextLayout.from_checkpoint(checkpoint)
    sizes = inkling2_mlx._layer_sizes(layout)
    visits = inkling2_mlx._layer_visits(layout)
    non_experts = sum(entry.size for name, entry in tensor_ranges(checkpoint).items() if ".mlp.experts." not in name)

    monkeypatch.setattr(inkling2_mlx, "AppleMemoryGuard", FixtureGuard)
    monkeypatch.setattr(inkling2_mlx, "_PREFILL_CHUNK", 3)
    resident = TextGenerator.from_checkpoint(tmp_path, memory_budget_mib=1024)
    assert resident.model.weight_storage == "resident"

    # Size the budget to fit ~13 NVFP4 experts so prefill + decode produce hits and evictions.
    low, high = 0, FixtureGuard.budget
    while low < high:
        mid = (low + high) // 2
        try:
            inkling2_mlx.plan_inkling(text, layout, non_experts, mid, 24)
        except ValueError:
            low = mid + 1
        else:
            high = mid
    FixtureGuard.budget = high + 12 * sizes[0]

    streamed = TextGenerator.from_checkpoint(tmp_path, memory_budget_mib=1024)
    assert streamed.model.weight_storage == "ssd_experts"
    assert resident.model.route_trace is None
    assert streamed.model.route_trace is None

    streamed.model.route_trace = []
    prompt = [2, 3, 5, 7, 11, 13]
    options = (prompt, 5, 0.0, 0, 1.0, None, "dynamic")
    resident_tokens = list(resident._mlx_tokens(*options))
    streamed_tokens = list(streamed._mlx_tokens(*options))
    assert streamed_tokens == resident_tokens

    trace = streamed.model.route_trace
    # 6 prefill tokens in chunks of 3 => 2 * 16 visits; 4 decode steps => 4 * 16 visits.
    assert len(trace) == 6 * len(layout.visits)
    prefill = [entry for entry in trace if entry["phase"] == "prefill"]
    decode = [entry for entry in trace if entry["phase"] == "decode"]
    assert len(prefill) == 2 * len(layout.visits)
    assert len(decode) == 4 * len(layout.visits)
    assert [entry["token_index"] for entry in decode[::len(layout.visits)]] == [6, 7, 8, 9]
    assert [entry["loop_pass"] for entry in trace if entry["layer"] == 2][:6] == [0, 1, 2, 0, 1, 2]
    for entry in trace:
        assert entry["experts"] == sorted(entry["experts"])
        assert entry["byte_sizes"] == [sizes[entry["layer"]]] * len(entry["experts"])

    cache = streamed.model._expert_cache
    assert cache.hits > 0 and cache.misses > 0
    replayed = replay_policy(
        trace, cache.byte_limit, layer_sizes=sizes,
        layer_weights={layer: count ** 2 for layer, count in visits.items()},
        decode_decay=0.5, protect_visit=True,
    )
    assert replayed["decode_tokens"] == 4
    assert replayed["total"]["hits"] == cache.hits
    assert replayed["total"]["misses"] == cache.misses
    assert replayed["total"]["bytes_read"] == cache.bytes_read
    assert replayed["prefill"]["hits"] == sum(entry["hits"] for entry in prefill)
    assert replayed["prefill"]["bytes_read"] == sum(entry["bytes_read"] for entry in prefill)
    assert replayed["decode"]["hits"] == sum(entry["hits"] for entry in decode)
    assert replayed["decode"]["bytes_read"] == sum(entry["bytes_read"] for entry in decode)

    # Also verify global-LRU SSDExpertCache matches legacy replay_existing.
    global_cache = SSDExpertCache(cache.byte_limit, FixtureGuard())
    for entry in trace:
        for expert in entry["experts"]:
            global_cache.get(entry["layer"], expert, layout.expert_banks[entry["layer"]].ranges)
    legacy = replay_existing(trace, global_cache.byte_limit, layer_sizes=sizes)
    assert legacy["total"]["hits"] == global_cache.hits
    assert legacy["total"]["misses"] == global_cache.misses
    assert legacy["total"]["bytes_read"] == global_cache.bytes_read


def test_inkling2_visit_protection_prevents_self_eviction(tmp_path):
    import struct

    class FixtureGuard:
        def check(self):
            pass

    path = tmp_path / "experts.bin"
    path.write_bytes(struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0))
    ranges = {"weight": TensorRange(path, 0, 16, "F32", (4, 1))}
    cache = SSDExpertCache(8, FixtureGuard(), layer_sizes={0: 4, 1: 4}, layer_weights={0: 1.0, 1: 0.0})
    assert cache.layer_slots == {0: 2, 1: 0}

    # Layer 0 holds (0, 1) and (0, 2); visit needs (0, 0), (0, 1), (0, 2).
    assert cache.get(0, 1, ranges)["weight"].item() == 2.0
    assert cache.get(0, 2, ranges)["weight"].item() == 3.0
    with cache.visit(0, [0, 1, 2]):
        assert cache.get(0, 0, ranges)["weight"].item() == 1.0
        assert cache.get(0, 1, ranges)["weight"].item() == 2.0
        assert cache.get(0, 2, ranges)["weight"].item() == 3.0
    assert cache.hits == 2 and cache.misses == 3 and cache.bytes_read == 12
    # Layer 1 has zero quota and bypasses caching without error.
    assert cache.get(1, 3, ranges)["weight"].item() == 4.0
    assert cache.resident_bytes == 8
