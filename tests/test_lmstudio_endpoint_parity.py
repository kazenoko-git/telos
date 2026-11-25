"""Wire-format parity tests for the `telos serve` vs LM Studio SSE benchmark harness."""

from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from telos.inference import TextGenerator, create_app
from telos.models import TelosConfig, TelosTransformer


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_serve_vs_lmstudio.py"
    spec = importlib.util.spec_from_file_location("benchmark_serve_vs_lmstudio", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def serve_client(tmp_path):
    tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "hello", "world", "end", "again"]
    tokenizer = Tokenizer(models.WordLevel({t: i for i, t in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.add_special_tokens(tokens[:4])
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tmp_path / "custom.json"))

    config = TelosConfig(
        vocab_size=len(tokens),
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=512,
        is_causal=True,
    )
    metadata = {"paradigm": "ar", "model": asdict(config), "data": {"tokenizer": "custom.json"}}
    with torch.random.fork_rng():
        torch.manual_seed(693)
        model = TelosTransformer(config)
    torch.save({"config": metadata, "model_state_dict": model.state_dict()}, tmp_path / "checkpoint_final.pt")

    generator = TextGenerator.from_checkpoint(tmp_path, prefix_cache_entries=4)
    app = create_app(generator, model_id="telos-bench")
    return TestClient(app)


def test_sse_benchmark_harness_against_telos_serve(serve_client):
    bench = _load_benchmark_module()
    result = bench.stream_completion(
        serve_client,
        "http://testserver/v1",
        "telos-bench",
        "hello world",
        max_tokens=6,
        temperature=0.0,
    )
    assert result["completion_tokens"] > 0
    assert result["finish_reason"] in {"stop", "length"}
    assert result["ttft_ms"] is not None and result["ttft_ms"] >= 0.0
    assert result["wall_seconds"] >= 0.0

    report = bench.run_benchmark(
        "http://testserver/v1",
        "telos-bench",
        "http://testserver/v1",
        "telos-bench",
        max_tokens=6,
        trials=2,
        warmup_tokens=2,
        client=serve_client,
    )
    assert set(report["workloads"]) == set(bench.DEFAULT_WORKLOADS)
    assert report["workloads"]["short_greedy"]["greedy_text_parity"] is True
    assert report["workloads"]["shared_prefix_turn1"]["greedy_text_parity"] is True
    assert report["workloads"]["shared_prefix_turn2"]["greedy_text_parity"] is True
    assert report["workloads"]["sampled_seeded"]["greedy_text_parity"] is None
