"""OpenAI-compatible HTTP and SSE inference server for native Télos checkpoints."""

import json
import threading
import time
import uuid
from typing import Any

from .chat import ChatTemplateRenderer
from .generation import GenerationStats, TextGenerator


def _require_fastapi():
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise ImportError(
            "telos serve requires the optional serve dependencies. "
            "Install them via `pip install 'telos-ml[serve]'`."
        ) from exc
    return FastAPI, Request, JSONResponse, StreamingResponse


def _normalize_stops(stop: Any) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,) if stop else ()
    if isinstance(stop, list) and all(isinstance(item, str) for item in stop):
        return tuple(item for item in stop if item)
    raise ValueError("stop must be a string or a list of strings.")


def _usage_payload(stats: GenerationStats) -> dict[str, int]:
    return {
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "total_tokens": stats.prompt_tokens + stats.completion_tokens,
    }


def _timing_headers(stats: GenerationStats) -> dict[str, str]:
    headers = {
        "X-Telos-Total-Ms": f"{stats.total_seconds * 1000:.2f}",
    }
    if stats.ttft_seconds is not None:
        headers["X-Telos-TTFT-Ms"] = f"{stats.ttft_seconds * 1000:.2f}"
    if stats.tokens_per_second is not None:
        headers["X-Telos-Decode-Tok-Per-Sec"] = f"{stats.tokens_per_second:.2f}"
    return headers


def _sse_line(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _parse_generation_options(body: dict[str, Any], default_cache: str) -> dict[str, Any]:
    max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_tokens", 128)
    if max_tokens is None:
        max_tokens = 128
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("max_tokens must be an integer.")

    temperature = body.get("temperature", 0.0)
    if temperature is None:
        temperature = 0.0
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a number.")

    top_p = body.get("top_p", 1.0)
    if top_p is None:
        top_p = 1.0
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise ValueError("top_p must be a number.")

    top_k = body.get("top_k", 0)
    if top_k is None:
        top_k = 0
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer.")

    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer.")

    cache = body.get("cache") or default_cache
    stops = _normalize_stops(body.get("stop"))
    return {
        "max_new_tokens": max_tokens,
        "temperature": float(temperature),
        "top_k": top_k,
        "top_p": float(top_p),
        "seed": seed,
        "stop": stops,
        "cache": cache,
    }


def create_app(
    generator: TextGenerator,
    *,
    model_id: str = "telos",
    chat_renderer: ChatTemplateRenderer | None = None,
    default_cache: str = "auto",
    api_key: str | None = None,
):
    """Build a FastAPI application serving OpenAI-compatible generation endpoints."""
    FastAPI, Request, JSONResponse, StreamingResponse = _require_fastapi()

    renderer = chat_renderer or ChatTemplateRenderer.from_checkpoint_dir(generator.checkpoint_dir)
    created_at = int(time.time())
    lock = threading.Lock()

    app = FastAPI(title="Télos Inference Server", docs_url=None, redoc_url=None)
    app.state.generator = generator
    app.state.model_id = model_id
    app.state.chat_renderer = renderer

    def error_response(status_code: int, message: str, *, error_type: str = "invalid_request_error", code: str | None = None):
        return JSONResponse(
            status_code=status_code,
            content={"error": {"message": message, "type": error_type, "param": None, "code": code}},
        )

    def check_auth(request: Request):
        if not api_key:
            return None
        header = request.headers.get("authorization", "")
        if header != f"Bearer {api_key}":
            return error_response(401, "Invalid or missing API key.", error_type="authentication_error", code="invalid_api_key")
        return None

    def check_model(body: dict[str, Any]):
        requested = body.get("model")
        if requested is not None and requested != model_id:
            return error_response(
                404,
                f"Model '{requested}' not found. Available model: '{model_id}'.",
                error_type="invalid_request_error",
                code="model_not_found",
            )
        return None

    async def parse_json_body(request: Request):
        try:
            body = await request.json()
        except Exception as exc:
            return None, error_response(400, f"Malformed JSON request body: {exc}")
        if not isinstance(body, dict):
            return None, error_response(400, "Request body must be a JSON object.")
        return body, None

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": model_id,
            "backend": generator.backend,
            "max_seq_len": generator.max_seq_len,
        }

    @app.get("/v1/models")
    def list_models(request: Request):
        auth_err = check_auth(request)
        if auth_err is not None:
            return auth_err
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": created_at,
                    "owned_by": "telos",
                    "max_model_len": generator.max_seq_len,
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(request: Request):
        auth_err = check_auth(request)
        if auth_err is not None:
            return auth_err
        body, parse_err = await parse_json_body(request)
        if parse_err is not None:
            return parse_err
        model_err = check_model(body)
        if model_err is not None:
            return model_err

        raw_prompt = body.get("prompt")
        if isinstance(raw_prompt, list):
            if len(raw_prompt) != 1 or not isinstance(raw_prompt[0], str):
                return error_response(400, "prompt must be a string or a single-element list of strings.")
            prompt = raw_prompt[0]
        elif isinstance(raw_prompt, str):
            prompt = raw_prompt
        else:
            return error_response(400, "prompt is required and must be a string.")

        echo = bool(body.get("echo", False))
        stream = bool(body.get("stream", False))
        stream_options = body.get("stream_options") or {}
        include_usage = bool(isinstance(stream_options, dict) and stream_options.get("include_usage"))

        try:
            options = _parse_generation_options(body, default_cache)
            # Validate inputs eagerly so errors return HTTP 400 before streaming starts.
            gen_stream = generator.stream_with_stats(prompt, **options)
        except ValueError as exc:
            return error_response(400, str(exc))

        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not stream:
            with lock:
                text = "".join(gen_stream)
            stats = gen_stream.stats
            output_text = (prompt + text) if echo else text
            return JSONResponse(
                content={
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "text": output_text,
                            "logprobs": None,
                            "finish_reason": stats.finish_reason,
                        }
                    ],
                    "usage": _usage_payload(stats),
                },
                headers=_timing_headers(stats),
            )

        def event_stream():
            with lock:
                if echo and prompt:
                    yield _sse_line({
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "text": prompt, "logprobs": None, "finish_reason": None}],
                    })
                for chunk in gen_stream:
                    yield _sse_line({
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "text": chunk, "logprobs": None, "finish_reason": None}],
                    })
                stats = gen_stream.stats
                yield _sse_line({
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "text": "", "logprobs": None, "finish_reason": stats.finish_reason}],
                })
                if include_usage:
                    yield _sse_line({
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_id,
                        "choices": [],
                        "usage": _usage_payload(stats),
                    })
                yield _sse_line("[DONE]")

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        auth_err = check_auth(request)
        if auth_err is not None:
            return auth_err
        body, parse_err = await parse_json_body(request)
        if parse_err is not None:
            return parse_err
        model_err = check_model(body)
        if model_err is not None:
            return model_err

        messages = body.get("messages")
        enable_thinking = body.get("enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            return error_response(400, "enable_thinking must be a boolean.")

        try:
            prompt = renderer.render(messages, add_generation_prompt=True, enable_thinking=enable_thinking)
            options = _parse_generation_options(body, default_cache)
            gen_stream = generator.stream_with_stats(prompt, **options)
        except (ValueError, ImportError) as exc:
            return error_response(400, str(exc))

        stream = bool(body.get("stream", False))
        stream_options = body.get("stream_options") or {}
        include_usage = bool(isinstance(stream_options, dict) and stream_options.get("include_usage"))
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not stream:
            with lock:
                text = "".join(gen_stream)
            stats = gen_stream.stats
            return JSONResponse(
                content={
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "logprobs": None,
                            "finish_reason": stats.finish_reason,
                        }
                    ],
                    "usage": _usage_payload(stats),
                },
                headers=_timing_headers(stats),
            )

        def event_stream():
            with lock:
                yield _sse_line({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}],
                })
                for chunk in gen_stream:
                    yield _sse_line({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": chunk}, "logprobs": None, "finish_reason": None}],
                    })
                stats = gen_stream.stats
                yield _sse_line({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": stats.finish_reason}],
                })
                if include_usage:
                    yield _sse_line({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [],
                        "usage": _usage_payload(stats),
                    })
                yield _sse_line("[DONE]")

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
