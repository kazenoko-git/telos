"""Native generation checkpoint, cache, sampling and streaming contracts."""

import json
import sys
from dataclasses import asdict
from subprocess import run

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from telos.inference import TextGenerator
from telos.models import TelosConfig, TelosTransformer
from telos.training.core import metal_usable


@pytest.fixture
def tokenizer():
    tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "hello", "world", "end", "again"]
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.add_special_tokens(tokens[:4])
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer


@pytest.fixture(params=["pytorch", "mlx"])
def native_checkpoint(tmp_path, tokenizer, request):
    config = TelosConfig(
        vocab_size=8, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=16, is_causal=True,
    )
    metadata = {"paradigm": "ar", "model": asdict(config), "data": {"tokenizer": "custom.json"}}
    tokenizer.save(str(tmp_path / "custom.json"))
    if request.param == "mlx":
        if not metal_usable():
            pytest.skip("Metal required")
        import mlx.core as mx

        from telos.models import MLXTelosTransformer

        mx.random.seed(693)
        model = MLXTelosTransformer(**metadata["model"], precision="float32")
        model.save_weights(str(tmp_path / "model.safetensors"))
        (tmp_path / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    else:
        with torch.random.fork_rng():
            torch.manual_seed(693)
            model = TelosTransformer(config)
        torch.save({"config": metadata, "model_state_dict": model.state_dict()},
                   tmp_path / "checkpoint_final.pt")
    return tmp_path, request.param


def test_native_cached_generation_and_cli(native_checkpoint):
    path, backend = native_checkpoint
    model = TextGenerator.from_checkpoint(path)
    assert model.backend == backend
    ids = model.tokenizer.encode("hello world end again").ids
    args = (ids, 12, 0.0, 0, 1.0, None)
    tokens = model._mlx_tokens if backend == "mlx" else model._torch_tokens
    expected = list(tokens(*args, "off"))
    assert list(tokens(*args, "dynamic")) == expected
    if backend == "pytorch":
        assert list(tokens(*args, "static")) == expected

    options = {"temperature": 0.8, "top_k": 4, "top_p": 0.8, "seed": 312, "max_new_tokens": 8}
    result = model.generate("hello world end again", **options)
    assert result
    assert result == "".join(model.stream("hello world end again", **options))
    command = [sys.executable, "-m", "telos", "generate", "--checkpoint", str(path),
               "--prompt-file", "-", "--temperature", "0.8", "--top-k", "4",
               "--top-p", "0.8", "--seed", "312", "--max-new-tokens", "8", "--stream"]
    output = run(command, input="hello world end again", text=True, capture_output=True, check=True)
    assert output.stdout == result
    assert not output.stderr


def test_stream_stops_within_token_and_never_yields_stop_prefix(native_checkpoint, monkeypatch):
    path, backend = native_checkpoint
    model = TextGenerator.from_checkpoint(path)
    tokens = "_mlx_tokens" if backend == "mlx" else "_torch_tokens"
    monkeypatch.setattr(model, tokens, lambda *args: iter([4, 5, 6, 7, 2]))
    assert list(model.stream("hello", stop="world en")) == ["hello", " "]
    assert model.generate("hello", stop="world xyz") == "hello world end again"
    assert model.generate("hello", stop=["end", "world end"]) == "hello "
    assert model.generate("hello", max_new_tokens=0) == ""
    assert model.generate("hello " * 16) == ""
    with pytest.raises(ValueError, match="Checkpoint limit"):
        model.generate("hello " * 17)

    def inspect_prefix(ids, count, *args):
        assert ids == [1]
        assert count == 15
        return iter([4, 2, 7])

    monkeypatch.setattr(model, tokens, inspect_prefix)
    assert model.generate("") == "hello"


def test_missing_tokenizer_fails_even_with_a_same_size_override(native_checkpoint):
    path, _ = native_checkpoint
    original = path / "custom.json"
    relocated = original.rename(path / "relocated.json")
    with pytest.raises(FileNotFoundError, match="Tokenizer JSON not found"):
        TextGenerator.from_checkpoint(path)
    assert TextGenerator.from_checkpoint(path, tokenizer=relocated).vocab_size == 8
    with pytest.raises(FileNotFoundError, match="Tokenizer JSON not found"):
        TextGenerator.from_checkpoint(path, tokenizer=original)


def test_utf8_streaming_across_byte_tokens(monkeypatch):
    tokens = {"[BOS]": 0, "[EOS]": 1, "<0xC3>": 2, "<0xA9>": 3, "x": 4}
    tokenizer = Tokenizer(models.WordLevel(tokens))
    tokenizer.add_special_tokens(["[BOS]", "[EOS]"])
    tokenizer.decoder = decoders.ByteFallback()
    model = TelosTransformer(TelosConfig(
        vocab_size=5, d_model=16, n_layers=1, n_heads=2, max_seq_len=8, is_causal=True,
    ))
    inference = TextGenerator(model, tokenizer, backend="pytorch", max_seq_len=8)
    monkeypatch.setattr(inference, "_torch_tokens", lambda *args: iter([2, 3, 4, 2, 1]))
    assert list(inference.stream("")) == ["é", "x", "�"]
    assert inference.generate("", stop="éx") == ""


def test_torch_reserved_cache_matches_dynamic_logits_without_reallocating(tokenizer):
    model = TelosTransformer(TelosConfig(
        vocab_size=8, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=16, is_causal=True,
    )).eval()
    from telos.inference.generation import _reserve_torch_cache

    ids = torch.tensor([[4, 5, 6, 7, 4, 6, 5, 6]])
    with torch.inference_mode():
        _, cache = model.hidden_states_with_cache(ids[:, :3])
        static = _reserve_torch_cache(cache, ids.shape[1], 3)
        pointers = [(k.data_ptr(), v.data_ptr()) for k, v in static]
        for offset in range(3, ids.shape[1]):
            current = ids[:, offset:offset + 1]
            dynamic_hidden, cache = model.hidden_states_with_cache(current, cache)
            static_hidden, static = model.hidden_states_with_cache(current, static, cache_position=offset)
            torch.testing.assert_close(static_hidden, dynamic_hidden)
            assert [(k.data_ptr(), v.data_ptr()) for k, v in static] == pointers
