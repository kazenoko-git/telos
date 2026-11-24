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
    _, cache = model.hidden_states_with_cache(sequence[:, :3])
    for offset in range(3, sequence.shape[1]):
        hidden, cache = model.hidden_states_with_cache(sequence[:, offset:offset + 1], kv_cache=cache)
        assert mx.allclose(model.logits_from_hidden(hidden)[:, 0], full[:, offset], atol=2e-5, rtol=2e-5).item()
    assert generator.generate("t5 t7", max_new_tokens=4, cache="auto") == generator.generate(
        "t5 t7", max_new_tokens=4, cache="off",
    )


def test_unsupported_qwen_family_reports_architecture(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_moe"}')
    with pytest.raises(ValueError, match="Unsupported native inference model architecture: qwen3_moe"):
        TextGenerator.from_checkpoint(tmp_path)
