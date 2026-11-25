"""Cached autoregressive generation for native PyTorch and MLX checkpoints."""

from dataclasses import dataclass
import math
from pathlib import Path
import time

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from telos.data.tokenizer import checkpoint_tokenizer_path, load_tokenizer
from telos.models.checkpoint import load_checkpoint, resolve_checkpoint


@dataclass(frozen=True)
class GenerationStats:
    """Token counts, finish reason and wall-clock timing for one generation."""

    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    ttft_seconds: float | None
    total_seconds: float

    @property
    def tokens_per_second(self) -> float | None:
        if self.completion_tokens <= 1 or self.ttft_seconds is None:
            return None
        decode_seconds = self.total_seconds - self.ttft_seconds
        if decode_seconds <= 0:
            return None
        return (self.completion_tokens - 1) / decode_seconds


@dataclass(frozen=True)
class GenerationResult:
    """Generated continuation paired with its usage and timing statistics."""

    text: str
    stats: GenerationStats


class GenerationStream:
    """Iterable text stream that captures usage and timing stats on completion."""

    def __init__(self, iterator, finalize):
        self._iterator = iterator
        self._finalize = finalize
        self.stats: GenerationStats | None = None

    def __iter__(self):
        try:
            yield from self._iterator
        finally:
            if self.stats is None:
                self.stats = self._finalize()


def _stop_boundary(text, stops):
    """Return the visible boundary, holding back any partial stop string."""
    matches = [index for stop in stops if (index := text.find(stop)) >= 0]
    if matches:
        return min(matches), True

    for size in range(min(len(text), max(map(len, stops), default=0) - 1), 0, -1):
        if any(stop.startswith(text[-size:]) for stop in stops):
            return len(text) - size, False
    return len(text), False


def _torch_sample(logits, temperature, top_k, top_p, rng):
    import torch

    if temperature == 0:
        return int(logits.argmax().item())

    logits = logits.float()
    indices = None
    if top_k:
        logits, indices = torch.topk(logits, top_k)
    elif top_p < 1:
        logits, indices = logits.sort(descending=True)
    logits = (logits - logits.max()) / max(temperature, torch.finfo(torch.float32).tiny)
    probabilities = logits.softmax(dim=-1)
    if top_p < 1:
        preceding = probabilities.cumsum(dim=-1) - probabilities
        probabilities = probabilities.masked_fill(preceding >= top_p, 0)
    index = torch.multinomial(probabilities, 1, generator=rng)
    return int((indices[index] if indices is not None else index).item())


def _mlx_sample(logits, temperature, top_k, top_p, key):
    import mlx.core as mx

    if temperature == 0:
        return int(mx.argmax(logits).item())

    logits = logits.astype(mx.float32)
    indices = None
    if top_k:
        indices = mx.argpartition(-logits, top_k - 1)[:top_k]
        logits = logits[indices]
    if top_p < 1:
        order = mx.argsort(-logits)
        logits = logits[order]
        indices = order if indices is None else indices[order]
    logits = (logits - mx.max(logits)) / max(temperature, 1.17549435e-38)
    if top_p < 1:
        probabilities = mx.softmax(logits)
        preceding = mx.cumsum(probabilities) - probabilities
        logits = mx.where(preceding >= top_p, -mx.inf, logits)
    index = mx.random.categorical(logits, key=key)
    return int((indices[index] if indices is not None else index).item())


def _reserve_torch_cache(cache, capacity, length):
    reserved = []
    for keys, values in cache:
        key_storage = keys.new_empty((*keys.shape[:2], capacity, keys.shape[3]))
        value_storage = values.new_empty((*values.shape[:2], capacity, values.shape[3]))
        key_storage[:, :, :length].copy_(keys)
        value_storage[:, :, :length].copy_(values)
        reserved.append((key_storage, value_storage))
    return reserved


class TextGenerator:
    """Load once and generate independent continuations with per-request state."""

    def __init__(self, model, tokenizer, *, backend, max_seq_len, add_bos=False,
                 bos_id=None, stop_ids=None, allow_padded_vocab=False, checkpoint_dir=None):
        if backend not in {"mlx", "pytorch"}:
            raise ValueError(f"Unsupported inference backend: {backend}")
        if not isinstance(max_seq_len, int) or max_seq_len < 2:
            raise ValueError("Checkpoint context length must be at least two tokens.")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.backend = backend
        self.max_seq_len = max_seq_len
        self.add_bos = add_bos
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        checkpoint_vocab = model.vocab_size if backend == "mlx" else model.config.vocab_size
        self.vocab_size = tokenizer.get_vocab_size()
        if self.vocab_size != checkpoint_vocab:
            if (not allow_padded_vocab or not 0 < self.vocab_size < checkpoint_vocab
                    or max(tokenizer.get_vocab().values()) != self.vocab_size - 1):
                raise ValueError(f"Tokenizer vocabulary does not match the checkpoint ({checkpoint_vocab} tokens).")
        if not getattr(model, "is_causal", getattr(getattr(model, "config", None), "is_causal", False)):
            raise ValueError("Generation requires a native causal checkpoint.")
        self.bos_id = tokenizer.token_to_id("[BOS]") if bos_id is None else bos_id
        self.stop_ids = (set(stop_ids) if stop_ids is not None else {
            token_id for name in ("[EOS]", "[PAD]") if (token_id := tokenizer.token_to_id(name)) is not None
        })

    @classmethod
    def from_checkpoint(cls, checkpoint, *, tokenizer=None, config=None, device="auto", context_length=None):
        """Load native weights, tokenizer and context limit from a file or directory."""
        checkpoint = resolve_checkpoint(checkpoint)
        model, backend, _, metadata = load_checkpoint(checkpoint, config, strict_metadata=True)
        if backend == "mlx":
            if device not in {"auto", "mlx"}:
                raise ValueError("Native MLX checkpoints require --device mlx or auto.")
        else:
            import torch

            if device == "mlx":
                raise ValueError("Native PyTorch checkpoints require a PyTorch device.")
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device)
            if device.type not in {"cpu", "cuda"}:
                raise ValueError("Native PyTorch generation supports CPU and CUDA.")
            model.to(device)

        m_cfg = metadata.get("model", metadata)
        limit = m_cfg.get("max_seq_len") or m_cfg.get("seq_len") or getattr(model, "max_seq_len", 512)
        if backend == "pytorch":
            limit = model.config.max_seq_len
        if context_length is not None:
            if not isinstance(context_length, int) or not 2 <= context_length <= limit:
                raise ValueError(f"Context length must be between 2 and {limit}.")
            limit = context_length
        tokenizer_path = checkpoint_tokenizer_path(checkpoint, metadata, tokenizer)
        if tokenizer_path is None:
            resolved_tokenizer = load_tokenizer()
        else:
            try:
                resolved_tokenizer = Tokenizer.from_file(str(tokenizer_path))
            except Exception as exc:
                raise ValueError(f"Invalid tokenizer JSON: {tokenizer_path}") from exc
        options = {}
        if metadata.get("model_type") == "qwen3":
            from json import loads

            generation_path = checkpoint.with_name("generation_config.json")
            generation = loads(generation_path.read_text(encoding="utf-8")) if generation_path.is_file() else {}
            eos = generation.get("eos_token_id", metadata.get("eos_token_id"))
            stop_ids = set(eos if isinstance(eos, list) else [eos])
            stop_ids.add(generation.get("pad_token_id", metadata.get("pad_token_id")))
            options = {"allow_padded_vocab": True, "stop_ids": stop_ids - {None},
                       "bos_id": generation.get("bos_token_id", metadata.get("bos_token_id"))}
        return cls(model, resolved_tokenizer, backend=backend, max_seq_len=limit,
                   add_bos=bool(metadata.get("posttrain")), checkpoint_dir=checkpoint.parent, **options)

    def _torch_tokens(self, ids, count, temperature, top_k, top_p, seed, cache_mode):
        import torch

        device = next(self.model.parameters()).device
        rng = None
        if temperature:
            rng = torch.Generator(device=device)
            rng.seed() if seed is None else rng.manual_seed(seed)

        prompt_len = len(ids)
        cache = None
        for step in range(count):
            with torch.inference_mode():
                if step == 1 and cache_mode == "static":
                    cache = _reserve_torch_cache(cache, prompt_len + count - 1, prompt_len)
                input_ids = torch.tensor([ids], device=device, dtype=torch.long)
                if cache_mode == "off":
                    hidden = self.model.hidden_states(input_ids, mask_override=True)
                else:
                    position = prompt_len + step - 1 if step and cache_mode == "static" else None
                    hidden, cache = self.model.hidden_states_with_cache(
                        input_ids, kv_cache=cache, mask_override=True, cache_position=position,
                    )
                logits = self.model.output_projection(hidden[:, -1])[0]
                token = _torch_sample(logits, temperature, top_k, top_p, rng)
            yield token
            ids = [*ids, token] if cache_mode == "off" else [token]

    def _mlx_prefill(self, inputs):
        prefill = getattr(self.model, "prefill_last_hidden", None)
        if prefill is not None:
            return prefill(inputs)
        return self.model.hidden_states_with_cache(inputs, kv_cache=None, is_causal=True)

    def _mlx_tokens(self, ids, count, temperature, top_k, top_p, seed, cache_mode):
        import secrets

        import mlx.core as mx

        if temperature == 0 and cache_mode == "dynamic" and count > 1:
            yield from self._mlx_greedy_cached_tokens(ids, count)
            return

        key = None
        if temperature:
            key = mx.random.key((secrets.randbits(64) if seed is None else seed) & ((1 << 64) - 1))
        cache = None
        for _ in range(count):
            input_ids = mx.array([ids], dtype=mx.int32)
            if cache_mode == "off":
                hidden = self.model.hidden_states(input_ids, is_causal=True)
            elif cache is None:
                hidden, cache = self._mlx_prefill(input_ids)
            else:
                hidden, cache = self.model.hidden_states_with_cache(input_ids, kv_cache=cache, is_causal=True)
            subkey = None
            if key is not None:
                key, subkey = mx.random.split(key)
            logits = self.model.logits_from_hidden(hidden[:, -1])[0, :self.vocab_size]
            token = _mlx_sample(logits, temperature, top_k, top_p, subkey)
            yield token
            ids = [*ids, token] if cache_mode == "off" else [token]

    def _mlx_greedy_cached_tokens(self, ids, count):
        """Overlap one cached Metal decode step with host token delivery."""
        import mlx.core as mx

        def predict(inputs, cache):
            if cache is None:
                hidden, updated = self._mlx_prefill(inputs)
            else:
                hidden, updated = self.model.hidden_states_with_cache(inputs, kv_cache=cache, is_causal=True)
            logits = self.model.logits_from_hidden(hidden[:, -1])[0, :self.vocab_size]
            return mx.argmax(logits), updated

        token, cache = predict(mx.array([ids], dtype=mx.int32), None)
        mx.async_eval(token)
        for _ in range(count - 1):
            # Keep the greedy ID on device so the next graph can be submitted
            # before waiting for the current ID. At most one step is queued.
            next_token, cache = predict(token.reshape(1, 1), cache)
            mx.async_eval(next_token)
            yield int(token.item())
            token = next_token
        yield int(token.item())

    def stream_with_stats(self, prompt, *, max_new_tokens=128, temperature=0.0, top_k=0, top_p=1.0,
                          seed=None, stop=(), add_bos=None, cache="auto"):
        """Return a GenerationStream that yields text chunks and records usage/timing."""
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer.")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative.")
        if not isinstance(top_k, int) or not 0 <= top_k <= self.vocab_size:
            raise ValueError(f"top_k must be between 0 and {self.vocab_size}.")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if seed is not None and (not isinstance(seed, int) or not 0 <= seed < 1 << 64):
            raise ValueError("seed must be an unsigned 64-bit integer.")
        stops = (stop,) if isinstance(stop, str) else tuple(stop)
        if any(not isinstance(marker, str) or not marker for marker in stops):
            raise ValueError("Stop strings must be nonempty text.")
        if cache not in {"auto", "dynamic", "static", "off"}:
            raise ValueError("cache must be auto, dynamic, static or off.")
        if self.backend == "mlx" and cache == "static":
            raise ValueError("Static KV storage is available on PyTorch.")
        if cache == "auto":
            cache = "dynamic" if self.backend == "mlx" else "static"

        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if add_bos is None:
            add_bos = self.add_bos
        if not ids or (add_bos and ids[0] != self.bos_id):
            if self.bos_id is None:
                raise ValueError("An empty prompt or add_bos requires a [BOS] token.")
            ids = [self.bos_id, *ids]

        if len(ids) > self.max_seq_len:
            raise ValueError(f"Prompt has {len(ids)} tokens. Checkpoint limit is {self.max_seq_len}.")
        count = min(max_new_tokens, self.max_seq_len - len(ids))

        started = time.perf_counter()
        state = {
            "completion_tokens": 0,
            "finish_reason": "length",
            "ttft_seconds": None,
        }

        def finalize():
            return GenerationStats(
                prompt_tokens=len(ids),
                completion_tokens=state["completion_tokens"],
                finish_reason=state["finish_reason"],
                ttft_seconds=state["ttft_seconds"],
                total_seconds=time.perf_counter() - started,
            )

        def iterate():
            if count == 0:
                return

            tokens = self._mlx_tokens if self.backend == "mlx" else self._torch_tokens
            decoder = DecodeStream(skip_special_tokens=True)
            generated = []
            pending = ""
            decoded_length = 0
            for token in tokens(ids, count, temperature, top_k, top_p, seed, cache):
                if state["ttft_seconds"] is None:
                    state["ttft_seconds"] = time.perf_counter() - started
                if token in self.stop_ids:
                    state["finish_reason"] = "stop"
                    break
                generated.append(token)
                state["completion_tokens"] += 1
                chunk = decoder.step(self.tokenizer, token) or ""
                decoded_length += len(chunk)
                if chunk:
                    pending += chunk
                    boundary, stopped = _stop_boundary(pending, stops)
                    if boundary:
                        yield pending[:boundary]
                    if stopped:
                        state["finish_reason"] = "stop"
                        return
                    pending = pending[boundary:]

            # Flush any incomplete trailing byte sequence without re-decoding each step.
            pending += self.tokenizer.decode(generated, skip_special_tokens=True)[decoded_length:]
            matches = [index for stop in stops if (index := pending.find(stop)) >= 0]
            if matches:
                state["finish_reason"] = "stop"
                final = pending[:min(matches)]
            else:
                final = pending
            if final:
                yield final

        return GenerationStream(iterate(), finalize)

    def stream(self, prompt, **options):
        """Yield continuation chunks, withholding stop strings across token boundaries."""
        yield from self.stream_with_stats(prompt, **options)

    def generate_with_stats(self, prompt, **options):
        """Return the generated continuation along with token usage and timing."""
        stream = self.stream_with_stats(prompt, **options)
        text = "".join(stream)
        return GenerationResult(text=text, stats=stream.stats)

    def generate(self, prompt, **options):
        """Return only the generated continuation."""
        return "".join(self.stream(prompt, **options))
