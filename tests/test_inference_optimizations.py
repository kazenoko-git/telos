"""Regression and parity tests for prefix KV reuse and pipelined MLX sampling."""

from dataclasses import asdict
import json

import pytest
import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from telos.inference import TextGenerator, create_app
from telos.models import TelosConfig, TelosTransformer
from telos.training.core import metal_usable


@pytest.fixture
def opt_checkpoint(tmp_path):
    tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.add_special_tokens(tokens[:4])
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tmp_path / "custom.json"))

    config = TelosConfig(
        vocab_size=len(tokens),
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        is_causal=True,
    )
    metadata = {"paradigm": "ar", "model": asdict(config), "data": {"tokenizer": "custom.json"}}
    with torch.random.fork_rng():
        torch.manual_seed(101)
        model = TelosTransformer(config)
    torch.save({"config": metadata, "model_state_dict": model.state_dict()}, tmp_path / "checkpoint_final.pt")
    return tmp_path


def test_pytorch_multi_token_suffix_append_matches_full_prefill():
    config = TelosConfig(
        vocab_size=16, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=32, is_causal=True,
    )
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = TelosTransformer(config).eval()

    ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    with torch.inference_mode():
        full_hidden = model.hidden_states(ids, mask_override=True)
        prefix_hidden, prefix_cache = model.hidden_states_with_cache(ids[:, :3], mask_override=True)
        suffix_hidden, full_cache = model.hidden_states_with_cache(
            ids[:, 3:], kv_cache=prefix_cache, mask_override=True,
        )
        torch.testing.assert_close(prefix_hidden, full_hidden[:, :3], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(suffix_hidden, full_hidden[:, 3:], atol=1e-5, rtol=1e-5)
        assert full_cache[0][0].shape[2] == ids.shape[1]


@pytest.mark.parametrize("cache_mode", ["static", "dynamic"])
def test_prefix_kv_cache_exact_parity_and_immutability(opt_checkpoint, cache_mode):
    cold = TextGenerator.from_checkpoint(opt_checkpoint, prefix_cache_entries=0)
    warm = TextGenerator.from_checkpoint(opt_checkpoint, prefix_cache_entries=4)

    base_prompt = "alpha beta gamma delta"
    ext_a = "alpha beta gamma delta epsilon zeta"
    ext_b = "alpha beta gamma delta zeta epsilon"

    r0 = warm.generate_with_stats(base_prompt, max_new_tokens=6, cache=cache_mode)
    assert r0.stats.cached_tokens == 0
    assert r0.text == cold.generate(base_prompt, max_new_tokens=6, cache=cache_mode)

    # Exact repeat reuses len(prompt) - 1 tokens so final logit is still evaluated.
    r_repeat = warm.generate_with_stats(base_prompt, max_new_tokens=6, cache=cache_mode)
    assert r_repeat.stats.cached_tokens == r0.stats.prompt_tokens - 1
    assert r_repeat.text == r0.text

    # Extending the shared prefix reuses all 4 base tokens.
    r_a = warm.generate_with_stats(ext_a, max_new_tokens=6, cache=cache_mode)
    assert r_a.stats.cached_tokens == 4
    assert r_a.text == cold.generate(ext_a, max_new_tokens=6, cache=cache_mode)

    # Divergent branch from the same shared prefix must not see mutated KV state.
    r_b = warm.generate_with_stats(ext_b, max_new_tokens=6, cache=cache_mode)
    assert r_b.stats.cached_tokens == 4
    assert r_b.text == cold.generate(ext_b, max_new_tokens=6, cache=cache_mode)


def test_serve_exposes_prefix_cache_header(opt_checkpoint):
    generator = TextGenerator.from_checkpoint(opt_checkpoint, prefix_cache_entries=4)
    app = create_app(generator, model_id="telos-cache")
    client = TestClient(app)

    first = client.post("/v1/completions", json={
        "model": "telos-cache",
        "prompt": "alpha beta gamma delta",
        "max_tokens": 4,
    })
    assert first.status_code == 200
    assert first.headers["X-Telos-Cached-Tokens"] == "0"

    second = client.post("/v1/completions", json={
        "model": "telos-cache",
        "prompt": "alpha beta gamma delta epsilon",
        "max_tokens": 4,
    })
    assert second.status_code == 200
    assert second.headers["X-Telos-Cached-Tokens"] == "4"


def test_mlx_pipelined_sampled_decode_seeded_parity(tmp_path):
    if not metal_usable():
        pytest.skip("Metal required")
    import mlx.core as mx

    from telos.models import MLXTelosTransformer

    tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "alpha", "beta", "gamma", "delta"]
    tokenizer = Tokenizer(models.WordLevel({t: i for i, t in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.add_special_tokens(tokens[:4])
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tmp_path / "custom.json"))

    cfg = {
        "vocab_size": 8, "d_model": 64, "n_layers": 2, "n_heads": 4,
        "n_kv_heads": 2, "max_seq_len": 24, "is_causal": True,
    }
    mx.random.seed(42)
    model = MLXTelosTransformer(**cfg, precision="float32")
    model.save_weights(str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"paradigm": "ar", "model": cfg, "data": {"tokenizer": "custom.json"}}),
        encoding="utf-8",
    )

    gen = TextGenerator.from_checkpoint(tmp_path, prefix_cache_entries=4)
    ids = gen.tokenizer.encode("alpha beta gamma").ids
    pipelined = list(gen._mlx_pipelined_cached_tokens(ids, 8, temperature=0.8, top_k=4, top_p=0.9, seed=1234))
    uncached = list(gen._mlx_tokens(ids, 8, 0.8, 4, 0.9, 1234, "off"))
    assert pipelined == uncached

    # Prefix reuse on MLX
    cold_text = gen.generate("alpha beta gamma delta", max_new_tokens=6, cache="off")
    r1 = gen.generate_with_stats("alpha beta gamma", max_new_tokens=4)
    assert r1.stats.cached_tokens == 0
    r2 = gen.generate_with_stats("alpha beta gamma delta", max_new_tokens=6)
    assert r2.stats.cached_tokens == 3
    assert r2.text == cold_text
