"""OpenAI-compatible HTTP/SSE serving, chat template, and adapter tests."""

from dataclasses import asdict
import io
import json
import subprocess
import sys

import pytest
import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from telos.eval.adapters.openai_api import OpenAIAPIAdapter
from telos.inference import ChatTemplateRenderer, TextGenerator, create_app
from telos.models import TelosConfig, TelosTransformer


@pytest.fixture
def serve_checkpoint(tmp_path):
    tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "hello", "world", "end", "again", "System:", "User:", "Assistant:"]
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
        torch.manual_seed(693)
        model = TelosTransformer(config)
    torch.save({"config": metadata, "model_state_dict": model.state_dict()}, tmp_path / "checkpoint_final.pt")
    return tmp_path


def _parse_sse(body: str):
    events = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ").strip()
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(payload))
    return events


def test_health_models_and_auth(serve_checkpoint):
    generator = TextGenerator.from_checkpoint(serve_checkpoint)
    app = create_app(generator, model_id="telos-test", api_key="test-token")
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model": "telos-test",
        "backend": "pytorch",
        "max_seq_len": 32,
    }

    assert client.get("/v1/models").status_code == 401
    models_resp = client.get("/v1/models", headers={"Authorization": "Bearer test-token"})
    assert models_resp.status_code == 200
    data = models_resp.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "telos-test"
    assert data["data"][0]["max_model_len"] == 32


def test_completions_sync_stream_and_validation(serve_checkpoint):
    generator = TextGenerator.from_checkpoint(serve_checkpoint)
    app = create_app(generator, model_id="telos-test")
    client = TestClient(app)

    expected = generator.generate("hello world", max_new_tokens=6, temperature=0.0)
    resp = client.post("/v1/completions", json={
        "model": "telos-test",
        "prompt": "hello world",
        "max_tokens": 6,
        "temperature": 0.0,
    })
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["object"] == "text_completion"
    assert payload["choices"][0]["text"] == expected
    assert payload["choices"][0]["finish_reason"] in {"stop", "length"}
    assert payload["usage"]["prompt_tokens"] == 2
    assert payload["usage"]["total_tokens"] == payload["usage"]["prompt_tokens"] + payload["usage"]["completion_tokens"]
    assert "X-Telos-Total-Ms" in resp.headers

    stream_resp = client.post("/v1/completions", json={
        "model": "telos-test",
        "prompt": "hello world",
        "max_tokens": 6,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert stream_resp.status_code == 200
    assert stream_resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(stream_resp.text)
    assert events[-1] == "[DONE]"
    chunks = [e["choices"][0]["text"] for e in events[:-1] if e.get("choices")]
    assert "".join(chunks) == expected
    usage_events = [e for e in events[:-1] if e.get("usage")]
    assert len(usage_events) == 1
    assert usage_events[0]["usage"]["prompt_tokens"] == 2

    assert client.post("/v1/completions", json={"model": "wrong", "prompt": "hello"}).status_code == 404
    assert client.post("/v1/completions", json={"model": "telos-test", "prompt": "hello " * 40}).status_code == 400


def test_chat_completions_jinja_template_and_fallback(serve_checkpoint):
    generator = TextGenerator.from_checkpoint(serve_checkpoint)
    fallback_renderer = ChatTemplateRenderer.from_checkpoint_dir(serve_checkpoint)
    rendered_fallback = fallback_renderer.render([
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "world"},
    ])
    assert "System: hello" in rendered_fallback
    assert "User: world" in rendered_fallback
    assert rendered_fallback.endswith("Assistant:")

    template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}Assistant:{% endif %}"
    )
    (serve_checkpoint / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template, "bos_token": "[BOS]", "eos_token": "[EOS]"}),
        encoding="utf-8",
    )
    app = create_app(generator, model_id="telos-chat")
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json={
        "model": "telos-chat",
        "messages": [{"role": "user", "content": "hello world"}],
        "max_tokens": 5,
        "temperature": 0.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(data["choices"][0]["message"]["content"], str)

    stream_resp = client.post("/v1/chat/completions", json={
        "model": "telos-chat",
        "messages": [{"role": "user", "content": "hello world"}],
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert stream_resp.status_code == 200
    events = _parse_sse(stream_resp.text)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[-1] == "[DONE]"
    streamed = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events[:-1]
        if e.get("choices")
    )
    assert streamed == data["choices"][0]["message"]["content"]


def test_openai_adapter_interoperates_with_serve_app(serve_checkpoint, monkeypatch):
    generator = TextGenerator.from_checkpoint(serve_checkpoint)
    app = create_app(generator, model_id="telos-live")
    client = TestClient(app)

    class _FakeResponse:
        def __init__(self, status, body):
            self.status = status
            self._body = io.BytesIO(body)

        def read(self):
            return self._body.read()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=60.0):
        path = req.full_url.removeprefix("http://telos.local")
        resp = client.post(path, content=req.data, headers={"Content-Type": "application/json"})
        return _FakeResponse(resp.status_code, resp.content)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenAIAPIAdapter(
        model_name="telos-live",
        api_base="http://telos.local/v1",
        use_chat_endpoint=False,
    )
    content, finish_reason = adapter.generate_with_finish_reason("hello world", max_new_tokens=5, temperature=0.0)
    assert finish_reason in {"stop", "length"}
    assert content == generator.generate("hello world", max_new_tokens=5, temperature=0.0)


def test_telos_serve_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "telos", "serve", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "telos serve" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--port" in result.stdout
