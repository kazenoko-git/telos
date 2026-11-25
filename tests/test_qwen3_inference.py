"""Native Qwen3 checkpoint and cache regression tests."""

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from telos.inference import TextGenerator
from telos.training.core import metal_usable

pytestmark = pytest.mark.skipif(not metal_usable(), reason="MLX Metal required")


def fixture_weights(mx, *, tied):
    # The head dimension intentionally differs from hidden_size / num_heads.
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
        "model.layers.0.mlp.gate_proj.weight": (96, 64),
        "model.layers.0.mlp.up_proj.weight": (96, 64),
        "model.layers.0.mlp.down_proj.weight": (64, 96),
    }
    if not tied:
        shapes["lm_head.weight"] = (32, 64)
    return {name: (mx.linspace(0.8, 1.2, shape[0]) if len(shape) == 1
                   else mx.random.normal(shape, key=mx.random.key(19 + i)) * 0.08)
            for i, (name, shape) in enumerate(shapes.items())}


def fixture_config(tied):
    return {
        "model_type": "qwen3", "hidden_size": 64, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 32, "num_hidden_layers": 1, "intermediate_size": 96,
        "vocab_size": 32, "max_position_embeddings": 32, "tie_word_embeddings": tied,
        "eos_token_id": 1, "pad_token_id": 0, "bos_token_id": 3,
        "rope_theta": 10000.0, "rope_scaling": {"rope_type": "yarn", "factor": 2.0,
                                               "original_max_position_embeddings": 16},
    }


@pytest.mark.parametrize("quantized", [False, True])
def test_qwen3_checkpoint_load_and_cache_parity(tmp_path, quantized):
    import mlx.core as mx

    from telos.models.qwen3_mlx import Qwen3MLX

    tied = quantized
    config = fixture_config(tied)
    arrays = fixture_weights(mx, tied=tied)
    expected = dict(arrays)
    if quantized:
        config["quantization"] = {"group_size": 32, "bits": 2}
        for name, array in expected.items():
            if array.ndim == 2:
                weight, scales, biases = mx.quantize(array, group_size=32, bits=2)
                arrays[name] = weight
                prefix = name.removesuffix(".weight")
                arrays[prefix + ".scales"] = scales
                arrays[prefix + ".biases"] = biases
                expected[name] = mx.dequantize(weight, scales, biases, group_size=32, bits=2)
        first = dict(list(arrays.items())[:8])
        second = {name: array for name, array in arrays.items() if name not in first}
        mx.save_safetensors(str(tmp_path / "part-a.safetensors"), first)
        mx.save_safetensors(str(tmp_path / "part-b.safetensors"), second)
        weight_map = {**dict.fromkeys(first, "part-a.safetensors"), **dict.fromkeys(second, "part-b.safetensors")}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        mx.save_safetensors(str(tmp_path / "model.safetensors"), arrays)

    vocabulary = {"[PAD]": 0, "<|done|>": 1, "[UNK]": 2, **{f"t{i}": i for i in range(3, 32)}}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(json.dumps(config))
    generator = TextGenerator.from_checkpoint(tmp_path, context_length=12)
    assert generator.max_seq_len == 12
    assert generator.stop_ids == {0, 1}
    assert generator.bos_id == 3

    model = generator.model
    reference = Qwen3MLX(config, expected)
    sequence = mx.array([[3, 5, 7, 9, 12, 15]])
    full = model.logits_from_hidden(model.hidden_states(sequence))
    expected_logits = reference.logits_from_hidden(reference.hidden_states(sequence))
    assert mx.allclose(full, expected_logits, atol=2e-5, rtol=2e-5).item()
    prompt = sequence[:, :3]
    hidden, cache = model.prefill_last_hidden(prompt)
    assert hidden.shape == (1, 1, config["hidden_size"])
    prefill_logits = model.logits_from_hidden(hidden)[:, 0]
    assert mx.allclose(prefill_logits, full[:, 2], atol=2e-5, rtol=2e-5).item()
    assert all(layer.length == prompt.shape[1] for layer in cache)
    for offset in range(3, sequence.shape[1]):
        hidden, cache = model.hidden_states_with_cache(sequence[:, offset:offset + 1], kv_cache=cache)
        assert mx.allclose(model.logits_from_hidden(hidden)[:, 0], full[:, offset], atol=2e-5, rtol=2e-5).item()
    assert generator.generate("t5 t7", max_new_tokens=4, cache="auto") == generator.generate(
        "t5 t7", max_new_tokens=4, cache="off",
    )


def test_unsupported_qwen_family_reports_architecture(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_next"}')
    with pytest.raises(ValueError, match="Unsupported native inference model architecture: qwen3_next"):
        TextGenerator.from_checkpoint(tmp_path)


def test_qwen3_yarn_serves_sse_and_reuses_prefix():
    import mlx.core as mx

    from telos.inference import create_app
    from telos.models.qwen3_mlx import Qwen3MLX

    test_client = pytest.importorskip("fastapi.testclient").TestClient
    vocabulary = {"[UNK]": 0, **{f"t{i}": i for i in range(1, 32)}}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    model = Qwen3MLX(fixture_config(tied=True), fixture_weights(mx, tied=True))
    generator = TextGenerator(model, tokenizer, backend="mlx", max_seq_len=32,
                              stop_ids=set(), prefix_cache_entries=4)
    payload = {"model": "qwen3", "prompt": "t2 t3 t4 t5", "max_tokens": 4}

    with test_client(create_app(generator, model_id="qwen3")) as client:
        response = client.post("/v1/completions", json={**payload, "stream": True})
        assert response.status_code == 200
        events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[-1] == "[DONE]"
        streamed = "".join(json.loads(event)["choices"][0]["text"] for event in events[:-1])

        cached = client.post("/v1/completions", json=payload)
        assert cached.status_code == 200
        assert cached.headers["X-Telos-Cached-Tokens"] == "3"
        assert cached.json()["choices"][0]["text"] == streamed
        assert cached.json()["usage"]["completion_tokens"] == 4


def test_qwen3_indexed_cache_growth_preserves_prefix_and_context_bound():
    import mlx.core as mx

    from telos.models.qwen3_mlx import _LayerCache

    prefix = mx.arange(255 * 4, dtype=mx.float32).reshape(1, 1, 255, 4)
    current = _LayerCache(prefix, -prefix, 255)
    first = mx.full((1, 1, 1, 4), 42.0)
    last = mx.full((1, 1, 1, 4), 63.0)
    filled = current.append(first, -first, 260)
    grown = filled.append(last, -last, 260)

    assert current.keys.shape[2] == 255
    assert filled.length == filled.keys.shape[2] == 256
    assert grown.length == 257
    assert grown.keys.shape[2] == 260
    expected = mx.concatenate((prefix, first, last), axis=2)
    assert mx.array_equal(grown.keys[:, :, :grown.length], expected).item()
    assert mx.array_equal(grown.values[:, :, :grown.length], -expected).item()


@pytest.mark.parametrize("quantized", [False, True])
def test_qwen3_moe_automatically_streams_selected_experts(tmp_path, monkeypatch, quantized):
    import mlx.core as mx

    from telos.models import qwen3_moe_mlx
    from telos.models.ssd_experts import TensorRange, is_expert_tensor, tensor_ranges

    config = {
        **fixture_config(tied=True),
        "model_type": "qwen3_moe",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "norm_topk_prob": True,
    }
    arrays = fixture_weights(mx, tied=True)
    for key in list(arrays):
        if ".mlp." in key:
            del arrays[key]
    prefix = "model.layers.0.mlp"
    arrays[f"{prefix}.gate.weight"] = mx.eye(64)[:4] * 0.3
    banks = {}
    for index, name in enumerate(("gate", "up", "down")):
        key = f"{prefix}.switch_mlp.{name}_proj"
        matrix = mx.random.normal((4, 64, 64), key=mx.random.key(41 + index)) * 0.08
        if quantized:
            weight, scales, biases = mx.quantize(matrix.astype(mx.bfloat16), group_size=32, bits=4)
            banks.update({key + ".weight": weight, key + ".scales": scales, key + ".biases": biases})
        else:
            banks[key + ".weight"] = matrix
    if quantized:
        config["quantization"] = {"group_size": 32, "bits": 4}
        mx.save_safetensors(str(tmp_path / "resident.safetensors"), arrays)
        mx.save_safetensors(str(tmp_path / "experts.safetensors"), banks)
        index = {**dict.fromkeys(arrays, "resident.safetensors"), **dict.fromkeys(banks, "experts.safetensors")}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    else:
        mx.save_safetensors(str(tmp_path / "model.safetensors"), {**arrays, **banks})
    (tmp_path / "config.json").write_text(json.dumps(config))
    vocabulary = {"[UNK]": 0, **{f"t{i}": i for i in range(1, 32)}}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    class FixtureGuard:
        budget = 2**30

        def check(self):
            pass

    monkeypatch.setattr(qwen3_moe_mlx, "AppleMemoryGuard", FixtureGuard)
    baseline = TextGenerator.from_checkpoint(tmp_path, context_length=16)
    assert baseline.model.weight_storage == "resident"

    ranges = tensor_ranges(baseline.checkpoint_dir / (
        "model.safetensors.index.json" if quantized else "model.safetensors"
    ))
    expert_size = sum(r.size for name, r in ranges.items() if is_expert_tensor(name)) // 4
    resident_bytes = sum(r.size for name, r in ranges.items() if not is_expert_tensor(name))
    # Allow one expert after reserving room for model state, KV and staging.
    FixtureGuard.budget = 512 * 2**20 + 32 * 16 * 64 + 2 * 4 * 2 * 32 * 32 + resident_bytes + 3 * expert_size
    reads = []
    original = TensorRange.load

    def record(entry, *, uncached=False):
        reads.append((entry.size, uncached))
        return original(entry, uncached=uncached)

    monkeypatch.setattr(TensorRange, "load", record)
    streamed = TextGenerator.from_checkpoint(tmp_path, context_length=16, prefix_cache_entries=4)
    cache = streamed.model._expert_cache
    assert streamed.model.weight_storage == "ssd_experts"
    assert streamed.prefix_cache.max_entries == 0
    assert cache.byte_limit == expert_size
    assert cache.bytes_read == 0
    assert sum(size for size, _ in reads) == resident_bytes
    assert all(not uncached for _, uncached in reads)

    logits, _ = streamed.model.prefill_last_hidden(mx.array([[3]], dtype=mx.int32))
    mx.eval(logits)
    assert cache.bytes_read == 2 * expert_size

    ids = mx.array([[3, 5, 7, 9]], dtype=mx.int32)
    ref = baseline.model.logits_from_hidden(baseline.model.hidden_states(ids))
    out = streamed.model.logits_from_hidden(streamed.model.hidden_states(ids))
    assert mx.array_equal(ref, out).item()
    assert streamed.generate("t3 t5 t7 t9", max_new_tokens=3) == baseline.generate(
        "t3 t5 t7 t9", max_new_tokens=3
    )
    assert cache.resident_bytes == expert_size
    assert len(cache._entries) == 1
    assert cache.bytes_read >= 2 * expert_size
    assert all(uncached for _, uncached in reads[len(arrays):])

    with pytest.raises(ValueError, match="dynamic KV cache"):
        streamed.generate("t3 t5", cache="off")
    reads.clear()
    with pytest.raises(ValueError, match="cannot fit"):
        TextGenerator.from_checkpoint(tmp_path, context_length=16, memory_budget_mib=1)
    assert not reads

    if quantized:
        from telos.inference import create_app

        test_client = pytest.importorskip("fastapi.testclient").TestClient
        with test_client(create_app(streamed, model_id="moe")) as client:
            assert client.get("/health").json()["weight_storage"] == "ssd_experts"
            response = client.post("/v1/completions", json={
                "model": "moe", "prompt": "t3 t5 t7 t9", "max_tokens": 3, "stream": True,
            })
            assert response.status_code == 200
            assert "data: [DONE]" in response.text


def test_flash_moe_stops_on_cumulative_swap_writes(monkeypatch):
    from telos.models import ssd_experts

    monkeypatch.setattr(ssd_experts, "_command", lambda *_: str(24 * 2**30))
    monkeypatch.setattr(ssd_experts, "_free_percent", lambda: 70)
    samples = iter((0, 128 * 2**20))
    monkeypatch.setattr(ssd_experts, "_swapout_bytes", lambda: next(samples))
    guard = ssd_experts.AppleMemoryGuard()
    with pytest.raises(RuntimeError, match="new swap writes"):
        guard.check()
