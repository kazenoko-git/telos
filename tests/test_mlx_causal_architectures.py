"""Native MLX causal checkpoint and cache parity tests for Llama, Mistral, Qwen2, and Phi-3."""

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from telos.inference import TextGenerator
from telos.training.core import metal_usable


def _write_tokenizer(path, vocab_size=32):
    vocabulary = {"[PAD]": 0, "<|done|>": 1, "[UNK]": 2, **{f"t{i}": i for i in range(3, vocab_size)}}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))


def _standard_weights(mx, *, tied, qkv_bias):
    shapes = {
        "model.embed_tokens.weight": (32, 64),
        "model.norm.weight": (64,),
        "model.layers.0.input_layernorm.weight": (64,),
        "model.layers.0.post_attention_layernorm.weight": (64,),
        "model.layers.0.self_attn.q_proj.weight": (64, 64),
        "model.layers.0.self_attn.k_proj.weight": (32, 64),
        "model.layers.0.self_attn.v_proj.weight": (32, 64),
        "model.layers.0.self_attn.o_proj.weight": (64, 64),
        "model.layers.0.mlp.gate_proj.weight": (96, 64),
        "model.layers.0.mlp.up_proj.weight": (96, 64),
        "model.layers.0.mlp.down_proj.weight": (64, 96),
    }
    if qkv_bias:
        shapes["model.layers.0.self_attn.q_proj.bias"] = (64,)
        shapes["model.layers.0.self_attn.k_proj.bias"] = (32,)
        shapes["model.layers.0.self_attn.v_proj.bias"] = (32,)
    if not tied:
        shapes["lm_head.weight"] = (32, 64)
    return {
        name: (
            mx.linspace(0.8, 1.2, shape[0])
            if len(shape) == 1 and name.endswith("norm.weight")
            else mx.random.normal(shape, key=mx.random.key(31 + i)) * 0.08
        )
        for i, (name, shape) in enumerate(shapes.items())
    }


def _phi3_weights(mx, *, tied):
    shapes = {
        "model.embed_tokens.weight": (32, 64),
        "model.norm.weight": (64,),
        "model.layers.0.input_layernorm.weight": (64,),
        "model.layers.0.post_attention_layernorm.weight": (64,),
        "model.layers.0.self_attn.qkv_proj.weight": (128, 64),
        "model.layers.0.self_attn.o_proj.weight": (64, 64),
        "model.layers.0.mlp.gate_up_proj.weight": (192, 64),
        "model.layers.0.mlp.down_proj.weight": (64, 96),
    }
    if not tied:
        shapes["lm_head.weight"] = (32, 64)
    return {
        name: (
            mx.linspace(0.8, 1.2, shape[0])
            if len(shape) == 1
            else mx.random.normal(shape, key=mx.random.key(71 + i)) * 0.08
        )
        for i, (name, shape) in enumerate(shapes.items())
    }


def _quantize_2d(mx, arrays):
    packed = dict(arrays)
    dequantized = dict(arrays)
    for name, array in list(arrays.items()):
        if array.ndim == 2:
            weight, scales, biases = mx.quantize(array, group_size=32, bits=4)
            packed[name] = weight
            prefix = name.removesuffix(".weight")
            packed[prefix + ".scales"] = scales
            packed[prefix + ".biases"] = biases
            dequantized[name] = mx.dequantize(weight, scales, biases, group_size=32, bits=4)
    return packed, dequantized


def test_external_architecture_registry_and_unknown_rejection(tmp_path):
    from telos.models.checkpoint import EXTERNAL_MLX_ARCHITECTURES

    assert {"qwen3", "qwen2", "llama", "mistral", "phi3"} <= EXTERNAL_MLX_ARCHITECTURES
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "config.json").write_text('{"model_type":"unsupported_arch"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported native inference model architecture: unsupported_arch"):
        TextGenerator.from_checkpoint(tmp_path)


@pytest.mark.skipif(not metal_usable(), reason="MLX Metal required")
@pytest.mark.parametrize("model_type,quantized", [
    ("llama", False),
    ("llama", True),
    ("mistral", False),
    ("qwen2", False),
    ("qwen2", True),
    ("phi3", False),
    ("phi3", True),
])
def test_external_mlx_architecture_load_cache_and_suffix_parity(tmp_path, model_type, quantized):
    import mlx.core as mx

    from telos.models.mlx_causal import LlamaMLX, Phi3MLX, Qwen2MLX

    tied = quantized
    config = {
        "model_type": model_type,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "intermediate_size": 96,
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "tie_word_embeddings": tied,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "bos_token_id": 3,
        "rope_theta": 500000.0 if model_type == "llama" else 10000.0,
    }
    if model_type == "llama":
        config["rope_scaling"] = {
            "rope_type": "llama3",
            "factor": 2.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 16,
        }

    if model_type == "phi3":
        arrays = _phi3_weights(mx, tied=tied)
        ref_cls = Phi3MLX
    elif model_type == "qwen2":
        arrays = _standard_weights(mx, tied=tied, qkv_bias=True)
        ref_cls = Qwen2MLX
    else:
        arrays = _standard_weights(mx, tied=tied, qkv_bias=False)
        ref_cls = LlamaMLX

    expected = dict(arrays)
    if quantized:
        config["quantization"] = {"group_size": 32, "bits": 4}
        arrays, expected = _quantize_2d(mx, arrays)

    mx.save_safetensors(str(tmp_path / "model.safetensors"), arrays)
    _write_tokenizer(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    generator = TextGenerator.from_checkpoint(tmp_path, context_length=16, prefix_cache_entries=4)
    model = generator.model
    reference = ref_cls(config, expected)

    sequence = mx.array([[3, 5, 7, 9, 12, 15]])
    full = model.logits_from_hidden(model.hidden_states(sequence))
    ref_logits = reference.logits_from_hidden(reference.hidden_states(sequence))
    assert mx.allclose(full, ref_logits, atol=2e-5, rtol=2e-5).item()

    # Prefill + multi-token suffix append + single-token decode parity
    hidden_p, cache = model.prefill_last_hidden(sequence[:, :2])
    assert mx.allclose(model.logits_from_hidden(hidden_p)[:, 0], full[:, 1], atol=2e-5, rtol=2e-5).item()
    hidden_s, cache = model.prefill_last_hidden(sequence[:, 2:5], kv_cache=cache)
    assert mx.allclose(model.logits_from_hidden(hidden_s)[:, 0], full[:, 4], atol=2e-5, rtol=2e-5).item()
    hidden_d, cache = model.hidden_states_with_cache(sequence[:, 5:6], kv_cache=cache)
    assert mx.allclose(model.logits_from_hidden(hidden_d)[:, 0], full[:, 5], atol=2e-5, rtol=2e-5).item()

    assert generator.generate("t5 t7", max_new_tokens=4, cache="auto") == generator.generate(
        "t5 t7", max_new_tokens=4, cache="off",
    )
