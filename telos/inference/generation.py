"""Cached autoregressive generation for native PyTorch and MLX checkpoints."""

import math

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from telos.data.tokenizer import checkpoint_tokenizer_path, load_tokenizer
from telos.models.checkpoint import load_checkpoint, resolve_checkpoint


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

    def __init__(self, model, tokenizer, *, backend, max_seq_len, add_bos=False):
        if backend not in {"mlx", "pytorch"}:
            raise ValueError(f"Unsupported inference backend: {backend}")
        if not isinstance(max_seq_len, int) or max_seq_len < 2:
            raise ValueError("Checkpoint context length must be at least two tokens.")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.backend = backend
        self.max_seq_len = max_seq_len
        self.add_bos = add_bos
        self.vocab_size = model.vocab_size if backend == "mlx" else model.config.vocab_size
        if tokenizer.get_vocab_size() != self.vocab_size:
            raise ValueError(f"Tokenizer vocabulary does not match the checkpoint ({self.vocab_size} tokens).")
        if not getattr(model, "is_causal", getattr(getattr(model, "config", None), "is_causal", False)):
            raise ValueError("Generation requires a native causal checkpoint.")
        self.bos_id = tokenizer.token_to_id("[BOS]")
        self.stop_ids = {token_id for name in ("[EOS]", "[PAD]")
                         if (token_id := tokenizer.token_to_id(name)) is not None}

    @classmethod
    def from_checkpoint(cls, checkpoint, *, tokenizer=None, config=None, device="auto"):
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
        limit = m_cfg.get("max_seq_len") or m_cfg.get("seq_len") or 512
        if backend == "pytorch":
            limit = model.config.max_seq_len
        tokenizer_path = checkpoint_tokenizer_path(checkpoint, metadata, tokenizer)
        if tokenizer_path is None:
            resolved_tokenizer = load_tokenizer()
        else:
            try:
                resolved_tokenizer = Tokenizer.from_file(str(tokenizer_path))
            except Exception as exc:
                raise ValueError(f"Invalid tokenizer JSON: {tokenizer_path}") from exc
        return cls(model, resolved_tokenizer,
                   backend=backend, max_seq_len=limit, add_bos=bool(metadata.get("posttrain")))

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

    def _mlx_tokens(self, ids, count, temperature, top_k, top_p, seed, cache_mode):
        import secrets

        import mlx.core as mx

        key = None
        if temperature:
            key = mx.random.key((secrets.randbits(64) if seed is None else seed) & ((1 << 64) - 1))
        cache = None
        for _ in range(count):
            input_ids = mx.array([ids], dtype=mx.int32)
            if cache_mode == "off":
                hidden = self.model.hidden_states(input_ids, is_causal=True)
            else:
                hidden, cache = self.model.hidden_states_with_cache(input_ids, kv_cache=cache, is_causal=True)
            subkey = None
            if key is not None:
                key, subkey = mx.random.split(key)
            logits = self.model.logits_from_hidden(hidden[:, -1])[0]
            token = _mlx_sample(logits, temperature, top_k, top_p, subkey)
            yield token
            ids = [*ids, token] if cache_mode == "off" else [token]

    def stream(self, prompt, *, max_new_tokens=128, temperature=0.0, top_k=0, top_p=1.0,
               seed=None, stop=(), add_bos=None, cache="auto"):
        """Yield continuation chunks, withholding stop strings across token boundaries."""
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
        if count == 0:
            return

        tokens = self._mlx_tokens if self.backend == "mlx" else self._torch_tokens
        decoder = DecodeStream(skip_special_tokens=True)
        generated = []
        pending = ""
        decoded_length = 0
        for token in tokens(ids, count, temperature, top_k, top_p, seed, cache):
            if token in self.stop_ids:
                break
            generated.append(token)
            chunk = decoder.step(self.tokenizer, token) or ""
            decoded_length += len(chunk)
            if chunk:
                pending += chunk
                boundary, stopped = _stop_boundary(pending, stops)
                if boundary:
                    yield pending[:boundary]
                if stopped:
                    return
                pending = pending[boundary:]

        # Flush any incomplete trailing byte sequence without re-decoding each step.
        pending += self.tokenizer.decode(generated, skip_special_tokens=True)[decoded_length:]
        matches = [index for stop in stops if (index := pending.find(stop)) >= 0]
        final = pending[:min(matches)] if matches else pending
        if final:
            yield final

    def generate(self, prompt, **options):
        """Return only the generated continuation."""
        return "".join(self.stream(prompt, **options))
