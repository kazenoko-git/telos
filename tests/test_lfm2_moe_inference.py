"""Tiny fixtures for LFM2.5 hybrid state and Flash-MoE parity."""

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from telos.inference import TextGenerator
from telos.training.core import metal_usable

pytestmark = pytest.mark.skipif(not metal_usable(), reason="MLX Metal required")


def write_checkpoint(path, mx, quantized):
    config = {
        "model_type": "lfm2_moe", "hidden_size": 64, "vocab_size": 32,
        "num_hidden_layers": 3, "num_dense_layers": 1, "intermediate_size": 64,
        "layer_types": ["conv", "full_attention", "conv"],
        "num_attention_heads": 4, "num_key_value_heads": 2, "rope_theta": 5_000_000,
        "max_position_embeddings": 24, "norm_eps": 1e-5, "conv_L_cache": 3,
        "conv_bias": not quantized, "tie_word_embeddings": True,
        "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
        "use_expert_bias": True, "norm_topk_prob": True, "routed_scaling_factor": 1.5,
    }
    arrays = {}
    dtype = mx.bfloat16 if quantized else mx.float32

    def sample(shape):
        return (mx.random.normal(shape, key=mx.random.key(19 + len(arrays))) * 0.12).astype(dtype)

    def weight(name, shape, *, bits=4):
        array = sample(shape)
        if quantized:
            packed, scales, biases = mx.quantize(array, group_size=32, bits=bits)
            arrays.update({name + ".weight": packed, name + ".scales": scales, name + ".biases": biases})
        else:
            arrays[name + ".weight"] = array

    def norm(name, width):
        arrays[name + ".weight"] = mx.ones((width,), dtype=dtype)

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
            if config["conv_bias"]:
                arrays[conv + ".in_proj.bias"] = sample((192,))
                arrays[conv + ".out_proj.bias"] = sample((64,))
                arrays[conv + ".conv.bias"] = sample((64,))
        else:
            attention = layer + ".self_attn"
            for name, rows in (("q_proj", 64), ("k_proj", 32), ("v_proj", 32), ("out_proj", 64)):
                weight(attention + "." + name, (rows, 64))
            norm(attention + ".q_layernorm", 16)
            norm(attention + ".k_layernorm", 16)
        ffn = layer + ".feed_forward"
        if i:
            arrays[ffn + ".gate.weight"] = sample((4, 64)).astype(mx.float32)
            arrays[ffn + ".expert_bias"] = mx.array([-1.0, -1.0, 1.0, 1.0], dtype=dtype)
            ffn += ".switch_mlp"
        for name in ("gate_proj", "up_proj", "down_proj"):
            weight(ffn + "." + name, (4, 64, 64) if i else (64, 64))

    if quantized:
        config["quantization"] = {"bits": 4, "group_size": 32,
                                  "model.embed_tokens": {"bits": 6, "group_size": 32}}
        groups = [{key: value for key, value in arrays.items() if (".switch_mlp." in key) == expert}
                  for expert in (False, True)]
        mapping = {}
        for name, group in zip(("resident.safetensors", "experts.safetensors"), groups, strict=True):
            mx.save_safetensors(str(path / name), group)
            mapping.update(dict.fromkeys(group, name))
        checkpoint = path / "model.safetensors.index.json"
        checkpoint.write_text(json.dumps({"weight_map": mapping}))
    else:
        checkpoint = path / "model.safetensors"
        mx.save_safetensors(str(checkpoint), arrays)
    (path / "config.json").write_text(json.dumps(config))
    vocab = {"[UNK]": 0, **{f"t{i}": i for i in range(1, 32)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))
    return checkpoint, config


@pytest.mark.parametrize("quantized", [False, True])
def test_lfm2_resident_and_ssd_parity(tmp_path, monkeypatch, quantized):
    import mlx.core as mx

    from telos.models import qwen3_moe_mlx as loader
    from telos.models.ssd_experts import TensorRange, is_expert_tensor, tensor_ranges

    checkpoint, config = write_checkpoint(tmp_path, mx, quantized)
    ranges = tensor_ranges(checkpoint)
    total = sum(item.size for item in ranges.values())
    expert_bytes = sum(item.size for name, item in ranges.items() if is_expert_tensor(name))
    one_expert = expert_bytes // (2 * config["num_experts"])

    class FixtureGuard:
        budget = 2**30

        def check(self):
            pass

    monkeypatch.setattr(loader, "AppleMemoryGuard", FixtureGuard)
    resident = TextGenerator.from_checkpoint(tmp_path, prefix_cache_entries=4)
    assert resident.model.weight_storage == "resident"
    assert resident.prefix_cache.max_entries == 0

    # Find the lowest admissible budget. This leaves exactly one expert in the cache.
    low, high = 0, 2**30
    while low < high:
        midpoint = (low + high) // 2
        try:
            loader.plan_experts(config, ranges, midpoint, None, 0)
        except ValueError as exc:
            assert "cannot fit" in str(exc)
            low = midpoint + 1
        else:
            high = midpoint
    FixtureGuard.budget = high
    reads = []
    original = TensorRange.load

    def record(entry, *, uncached=False):
        reads.append((entry.size, uncached))
        return original(entry, uncached=uncached)

    monkeypatch.setattr(TensorRange, "load", record)
    streamed = TextGenerator.from_checkpoint(tmp_path, prefix_cache_entries=4)
    cache = streamed.model._expert_cache
    assert streamed.model.weight_storage == "ssd_experts"
    assert streamed.prefix_cache.max_entries == 0
    assert cache.byte_limit == one_expert
    assert sum(size for size, _ in reads) == total - expert_bytes
    assert not any(uncached for _, uncached in reads)
    reads.clear()

    sequence = mx.array([[3, 5, 7, 9, 11]], dtype=mx.int32)
    full = resident.model.hidden_states(sequence)
    streamed_full = streamed.model.hidden_states(sequence)
    assert mx.array_equal(full, streamed_full).item()
    assert mx.array_equal(resident.model.logits_from_hidden(full),
                          streamed.model.logits_from_hidden(streamed_full)).item()
    assert cache.bytes_read == 4 * one_expert  # Two selected experts in each routed layer.
    assert cache.resident_bytes == one_expert
    assert all(uncached for _, uncached in reads)  # Streaming reads bypass the file cache.
    for generator in (resident, streamed):
        model = generator.model
        prefix, state = model.hidden_states_with_cache(sequence[:, :2])
        suffix, state = model.hidden_states_with_cache(sequence[:, 2:4], kv_cache=state)
        last, state = model.hidden_states_with_cache(sequence[:, 4:], kv_cache=state)
        tolerance = 0.025 if quantized else 2e-5
        assert mx.allclose(mx.concatenate((prefix, suffix, last), axis=1), full,
                           atol=tolerance, rtol=tolerance).item()
        assert [layer.length for layer in state] == [5, 5, 5]
        assert state[0].history.shape == state[2].history.shape == (1, 2, 64)
    options = ([3, 5, 7], 4, 0.0, 0, 1.0, None, "dynamic")
    assert list(streamed._mlx_tokens(*options)) == list(resident._mlx_tokens(*options))
    assert streamed.generate("t3 t5 t7", max_new_tokens=4) == resident.generate("t3 t5 t7", max_new_tokens=4)
    with pytest.raises(ValueError, match="dynamic KV cache"):
        streamed.generate("t3 t5", cache="off")


def test_lfm2_router_bias_changes_selection_only(tmp_path, monkeypatch):
    import mlx.core as mx

    from telos.models import qwen3_moe_mlx as loader

    write_checkpoint(tmp_path, mx, False)

    class FixtureGuard:
        budget = 2**30

        def check(self):
            pass

    monkeypatch.setattr(loader, "AppleMemoryGuard", FixtureGuard)
    generator = TextGenerator.from_checkpoint(tmp_path)
    router = generator.model.layers[1].mlp
    router.router.weight = mx.zeros((4, 64))
    router.bias = mx.array([0.0, 0.0, 1.0, 2.0])
    indices, weights = router._route(mx.ones((1, 64)))
    assert indices.tolist() == [[3, 2]]
    assert mx.allclose(weights, mx.full((1, 2), 0.75), atol=2e-6).item()
