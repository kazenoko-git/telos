"""Native, fixed-loop Inkling 2 Mini text inference on MLX."""

from typing import NamedTuple

import mlx.core as mx
from mlx import nn

from .inkling2 import _TEXT, InklingTextLayout, _integer, _positive_number
from .inkling2_attention import (
    AttentionState,
    InklingAttention,
    StaticConvolution,
    attention_dimensions,
)
from .qwen3_mlx import _Weights
from .ssd_expert_io import LFMSSDExpertCache
from .ssd_experts import AppleMemoryGuard, tensor_ranges
from .window_routing import (
    WindowPinController,
    add_grad_trees,
    compute_window_coverage_loss_and_grads,
    zero_grad_tree,
)

_MIB = 2**20
_PREFILL_CHUNK = 128


class _VisitState(NamedTuple):
    attention: AttentionState
    attention_history: mx.array
    mlp_history: mx.array


class _TextState(NamedTuple):
    length: int
    visits: tuple[_VisitState, ...]


def _mix(blocks, partial, query, normalize_values):
    states = mx.stack((*blocks, partial), axis=-2).astype(mx.float32)
    keys = mx.fast.rms_norm(states, None, 1e-6)
    scores = (query.astype(mx.float32) @ keys.swapaxes(-1, -2)) * states.shape[-1] ** -0.5
    return mx.softmax(scores, axis=-1) @ (keys if normalize_values else states)


def _norm(x, weight, eps, dtype):
    return mx.fast.rms_norm(x.astype(mx.float32), weight.astype(mx.float32), eps).astype(dtype)


class _MessageGate(nn.Module):
    def __init__(self, weights, prefix, width, rank, eps):
        super().__init__()
        self.down = weights.take(prefix + ".down_proj.weight", (rank, 2 * width))
        self.forget = weights.take(prefix + ".forget_gate_proj.weight", (width, rank))
        self.gain = weights.take(prefix + ".message_gate_proj.weight", (width, rank))
        self.eps = eps

    def __call__(self, x, message):
        pair = mx.concatenate(
            (mx.fast.rms_norm(x.astype(mx.float32), None, self.eps),
             mx.fast.rms_norm(message, None, self.eps)), axis=-1,
        )
        hidden = (pair.astype(self.down.dtype) @ self.down.T).astype(mx.float32)
        return ((1 - mx.tanh(hidden @ self.forget.T)) * x
                + mx.tanh(hidden @ self.gain.T) * message)


class _DecoderLayer(nn.Module):
    def __init__(self, weights, config, bank, block_size, sliding):
        super().__init__()
        index = bank.layer
        prefix = f"{_TEXT}layers.{index}"
        width = _integer(config, "hidden_size")
        kernel = _integer(config, "sconv_kernel_size")
        self.eps = _positive_number(config, "rms_norm_eps")
        self.normalize_values = config["attn_res_normalize_values"]
        self.boundary = index % block_size == 0
        self.bank = bank
        self.router_adapter = None
        self.router = (
            weights.take(prefix + ".mlp.gate.weight", bank.router_weight.shape),
            weights.take(prefix + ".mlp.gate.e_score_correction_bias", bank.router_bias.shape)
            if bank.router_bias is not None else None,
        )
        self.attention = InklingAttention(weights, prefix + ".self_attn", config, sliding=sliding)
        self.attn_query = (weights.take(prefix + ".attention_input.query", (width,) if sliding else (3, width))
                           if index else None)
        self.mlp_query = weights.take(prefix + ".mlp_input.query", (width,))
        if sliding:
            self.input_norms = (weights.take(prefix + ".input_layernorm.weight", (width,)),)
        else:
            self.input_norms = tuple(weights.take(f"{prefix}.{name}_input_layernorm.weight", (width,))
                                     for name in ("q", "k", "v"))
        self.mlp_norm = weights.take(prefix + ".post_attention_layernorm.weight", (width,))
        self.attention_conv = StaticConvolution(weights, prefix + ".post_attention_convolution", width, kernel)
        self.mlp_conv = StaticConvolution(weights, prefix + ".post_mlp_convolution", width, kernel)
        loop = config["loop_config"]
        self.gates = tuple(
            _MessageGate(weights, f"{prefix}.{name}_message_gate", width, loop["gate_rank"], self.eps)
            for name in ("q", "k", "v", "mlp")
        ) if loop["core_start"] <= index < loop["core_end"] and loop["num_passes"] > 1 else ()

    def __call__(self, x, blocks, messages, previous, offset, limit, experts, *, trace=None, controller=None):
        if self.attn_query is None:
            mixed = (x,) * 3
        else:
            mixed = _mix(blocks, x, self.attn_query.reshape(-1, x.shape[-1]), self.normalize_values)
            mixed = tuple(mixed[..., i, :] for i in range(mixed.shape[-2]))
            if len(mixed) == 1:
                mixed *= 3
        if messages is not None:
            mixed = tuple(self.gates[i](part, messages[..., i, :]) for i, part in enumerate(mixed))
        norms = self.input_norms if len(self.input_norms) == 3 else self.input_norms * 3
        inputs = [_norm(part, norm, self.eps, x.dtype) for part, norm in zip(mixed, norms, strict=True)]
        attention, state = self.attention(*inputs, None if previous is None else previous.attention, offset, limit)
        residual, attention_history = self.attention_conv(
            attention, None if previous is None else previous.attention_history,
        )
        residual = residual.astype(mx.float32)
        if self.boundary:
            blocks.append(x)
        else:
            residual = residual + x.astype(mx.float32)

        hidden = _mix(blocks, residual, self.mlp_query[None], self.normalize_values)[..., 0, :]
        if messages is not None:
            hidden = self.gates[3](hidden, messages[..., 3, :])
        hidden = self.bank.routed(
            _norm(hidden, self.mlp_norm, self.eps, x.dtype),
            self.router,
            experts,
            trace=trace,
            controller=controller,
            adapter=self.router_adapter,
        )
        output, mlp_history = self.mlp_conv(hidden, None if previous is None else previous.mlp_history)
        output = (output.astype(mx.float32) + residual).astype(x.dtype)
        return output, _VisitState(state, attention_history, mlp_history)


class _ResidentExperts:
    def __init__(self, layout, guard):
        self.banks = tuple({name: _load(entry, guard) for name, entry in bank.ranges.items()}
                           for bank in layout.expert_banks)

    def get(self, layer, expert, _ranges):
        return {name: weights[expert] for name, weights in self.banks[layer].items()}


class Inkling2TextMLX(nn.Module):
    is_causal = True
    paradigm = "ar"
    supports_prefix_cache = False
    supports_ssd_profiling = True
    max_batch_size = 1

    def __init__(self, config, arrays, layout, experts, cache, guard, context):
        super().__init__()
        text = config["text_config"]
        width = _integer(text, "hidden_size")
        padded_vocab = _integer(text, "vocab_size")
        self.vocab_size = _integer(text, "unpadded_vocab_size")
        if self.vocab_size > padded_vocab:
            raise ValueError("Inkling 2 tokenizer vocabulary exceeds the embedding table")
        self.max_seq_len = context
        self.weight_storage = "ssd_experts" if cache is not None else "resident"
        self._expert_cache = cache
        self._experts = experts
        self._guard = guard
        self._bank_top_k = {bank.layer: bank.top_k for bank in layout.expert_banks}
        self.route_trace = None
        self._window_controller = WindowPinController()
        self._visits = layout.visits
        self._loop_start = text["loop_config"]["core_start"]
        self._loop_end = text["loop_config"]["core_end"]
        block = _integer(text, "n_layers_per_attn_res_block")
        if self._loop_start % block or self._loop_end % block:
            raise ValueError("Inkling 2 loop does not align with residual block boundaries")
        self.eps = _positive_number(text, "rms_norm_eps")
        self.logit_cap = _positive_number(text, "final_logit_softcapping")
        self.normalize_values = text["attn_res_normalize_values"]
        weights = _Weights(arrays, config.get("quantization") or {})
        self.embed = weights.matrix(_TEXT + "embed_tokens", padded_vocab, width)
        self.embed_norm = weights.take(_TEXT + "embed_norm.weight", (width,))
        self.final_query = weights.take(_TEXT + "final_input.query", (width,))
        self.final_norm = weights.take(_TEXT + "norm.weight", (width,))
        self.head = weights.matrix("lm_head", padded_vocab, width)
        self.message_queries = (
            weights.take(_TEXT + "message_input.query", (self._loop_end - self._loop_start, 4, width))
            if text["loop_config"]["num_passes"] > 1 else None
        )
        if self.message_queries is not None:
            _integer(text["loop_config"], "gate_rank")
        local = set(text["local_layer_ids"])
        self.layers = [_DecoderLayer(weights, text, bank, block, bank.layer in local) for bank in layout.expert_banks]
        if weights.tensors:
            raise ValueError(f"Unexpected Inkling 2 text tensor {next(iter(weights.tensors))}")

    def set_window_size(self, window_size: int | None):
        if window_size is None and self._expert_cache is not None:
            self._expert_cache.reserve_window_slots(None)
        self._window_controller.set_window_size(window_size)

    def pin_window_experts(self):
        if self._expert_cache is None or self._window_controller.window_size is None:
            raise ValueError("Pinning window experts requires SSD weights and a decode window")
        return self._expert_cache.reserve_window_slots(self._bank_top_k)

    def _chunk(self, ids, cache, *, phase="prefill"):
        if phase == "decode":
            step_decode = getattr(self._expert_cache, "step_decode", None)
            if step_decode is not None:
                step_decode()
        embedded = self.embed.lookup(ids)
        x = _norm(embedded, self.embed_norm, self.eps, embedded.dtype)
        blocks = []
        offset = 0 if cache is None else cache.length
        depth = len(self.layers)
        states = [None] * len(self._visits)
        message_bank = None
        pass_counts = {}
        for layer, slot in self._visits:
            self._guard.check()
            loop_pass = pass_counts.get(layer, 0)
            pass_counts[layer] = loop_pass + 1
            if layer == self._loop_start:
                if slot == layer:
                    prelude, prelude_blocks = x, blocks.copy()
                else:
                    # The first appended block is the prelude partial, already shared by every pass.
                    message_bank = (blocks[len(prelude_blocks) + 1:], x)
                    x, blocks = prelude, prelude_blocks.copy()
            messages = None
            if slot >= depth:
                messages = _mix(*message_bank, self.message_queries[layer - self._loop_start], True)
            previous = None if cache is None else cache.visits[slot]
            trace = None
            if self.route_trace is not None:
                trace = {
                    "phase": phase,
                    "token_index": offset,
                    "token_count": ids.shape[1],
                    "layer": layer,
                    "loop_pass": loop_pass,
                }
            x, states[slot] = self.layers[layer](
                x, blocks, messages, previous, offset, self.max_seq_len, self._experts,
                trace=trace, controller=self._window_controller,
            )
            if trace is not None:
                self.route_trace.append(trace)
        hidden = _mix(blocks, x, self.final_query[None], self.normalize_values)[..., 0, :]
        hidden = _norm(hidden, self.final_norm, self.eps, embedded.dtype)
        # Finish outstanding state before the next chunk and keep cache graphs bounded.
        profile = self._expert_cache.profile if self._expert_cache is not None else None
        if profile is None:
            mx.eval(hidden, states)
        else:
            profile.evaluate(mx.eval, hidden, states)
        return hidden, _TextState(offset + ids.shape[1], tuple(states))

    def _chunks(self, ids, cache, is_causal):
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Inkling 2 text inference requires one token sequence per request")
        length = ids.shape[1]
        offset = 0 if cache is None else cache.length
        if not is_causal or not length or offset + length > self.max_seq_len:
            raise ValueError(f"Inkling 2 requires a non-empty causal input within {self.max_seq_len} tokens")
        phase = "prefill" if cache is None or length > 1 else "decode"
        seq_active = self._window_controller.mode in {"teacher", "student"}
        if not seq_active:
            if cache is None:
                self._window_controller.begin_prefill()
            elif phase == "decode":
                self._window_controller.begin_decode_step()
        for start in range(0, length, _PREFILL_CHUNK):
            if seq_active:
                self._window_controller.chunk_offset = start
            hidden, cache = self._chunk(ids[:, start:start + _PREFILL_CHUNK], cache, phase=phase)
            yield hidden, cache
        if seq_active:
            self._window_controller.chunk_offset = 0

    def hidden_states_with_cache(self, ids, kv_cache=None, *, is_causal=True):
        parts = []
        for hidden, state in self._chunks(ids, kv_cache, is_causal):
            parts.append(hidden)
        return (parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)), state

    def hidden_states(self, ids, *, is_causal=True):
        return self.hidden_states_with_cache(ids, is_causal=is_causal)[0]

    def prefill_last_hidden(self, ids, kv_cache=None):
        for hidden, state in self._chunks(ids, kv_cache, True):
            pass
        return hidden[:, -1:], state

    def logits_from_hidden(self, hidden):
        logits = self.head(hidden.astype(mx.float32))[..., :self.vocab_size]
        return mx.tanh(logits / self.logit_cap) * self.logit_cap

    def ssd_loss_and_grad(
        self,
        ids,
        loss_from_logits_fn,
        *,
        teacher_routes=None,
        prompt_lengths=None,
        response_mask=None,
        window_size=None,
        cov_weight=0.0,
    ):
        """Run NOGRAD forward + per-visit reverse-mode VJP to keep SSD expert memory bounded."""
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Inkling 2 text forward requires one token sequence per call")
        if ids.shape[1] > _PREFILL_CHUNK:
            raise ValueError(f"Inkling 2 SSD segmented backward supports sequences up to {_PREFILL_CHUNK} tokens")

        ctrl = self._window_controller
        accum = zero_grad_tree(self.trainable_parameters())
        if window_size is not None and prompt_lengths is not None:
            ctrl.reset_sequence_student(window_size, prompt_lengths)
        try:
            embedded = self.embed.lookup(ids)
            x0 = _norm(embedded, self.embed_norm, self.eps, embedded.dtype)
            mx.eval(x0)
            nodes = {0: x0}
            cur_id = 0
            next_id = 1
            blocks_ids = []
            message_bank_ids = None
            depth = len(self.layers)
            records = []

            for layer, slot in self._visits:
                self._guard.check()
                if layer == self._loop_start:
                    if slot == layer:
                        prelude_id = cur_id
                        prelude_blocks_len = len(blocks_ids)
                    else:
                        message_bank_ids = (tuple(blocks_ids[prelude_blocks_len + 1 :]), cur_id)
                        cur_id = prelude_id
                        blocks_ids = blocks_ids[:prelude_blocks_len]
                x_in_id = cur_id
                b_ids = tuple(blocks_ids)
                mb_ids = message_bank_ids if slot >= depth else None
                msgs = (_mix([nodes[i] for i in mb_ids[0]], nodes[mb_ids[1]],
                             self.message_queries[layer - self._loop_start], True)
                        if mb_ids is not None else None)
                local_blocks = [nodes[i] for i in b_ids]
                x_out, _ = self.layers[layer](
                    nodes[x_in_id], local_blocks, msgs, None, 0, self.max_seq_len, self._experts,
                    controller=ctrl,
                )
                mx.eval(x_out)
                if self.layers[layer].boundary:
                    blocks_ids.append(x_in_id)
                out_id = next_id
                next_id += 1
                nodes[out_id] = x_out
                records.append((layer, slot, x_in_id, b_ids, mb_ids, out_id))
                cur_id = out_id

            final_x_id = cur_id
            final_b_ids = tuple(blocks_ids)

            def _tail_loss(x_last, *b_tensors):
                h = _mix(b_tensors, x_last, self.final_query[None], self.normalize_values)[..., 0, :]
                h = _norm(h, self.final_norm, self.eps, embedded.dtype)
                logits = self.logits_from_hidden(h)[:, :-1]
                return loss_from_logits_fn(logits)

            argnums = list(range(1 + len(final_b_ids)))
            loss_val, tail_grads = mx.value_and_grad(_tail_loss, argnums=argnums)(
                nodes[final_x_id], *(nodes[i] for i in final_b_ids),
            )
            mx.eval(loss_val, tail_grads)

            cotangents = {}

            def _add_cot(nid, g):
                cotangents[nid] = g if nid not in cotangents else cotangents[nid] + g

            _add_cot(final_x_id, tail_grads[0])
            for bid, grad in zip(final_b_ids, tail_grads[1:], strict=True):
                _add_cot(bid, grad)

            if cov_weight > 0.0 and teacher_routes and prompt_lengths is not None and response_mask is not None:
                cov_loss, cov_grads = compute_window_coverage_loss_and_grads(
                    self, ctrl, teacher_routes, prompt_lengths, response_mask,
                )
                loss_val = loss_val + cov_weight * cov_loss
                for l_idx, (_, g_adapt) in cov_grads.items():
                    accum["layers"][l_idx]["router_adapter"] = add_grad_trees(
                        accum["layers"][l_idx]["router_adapter"],
                        {k: cov_weight * v for k, v in g_adapt.items()},
                    )
                mx.eval(loss_val)

            from mlx.utils import tree_flatten, tree_unflatten

            for layer, slot, x_in_id, b_ids, mb_ids, out_id in reversed(records):
                d_out = cotangents.pop(out_id, None)
                if d_out is None:
                    continue
                self._guard.check()
                mod = self.layers[layer]
                orig = mod.trainable_parameters()
                flat_items = tree_flatten(orig)
                keys = [k for k, _ in flat_items]
                vals = [v for _, v in flat_items]
                n_p = len(vals)
                mb_flat = (*mb_ids[0], mb_ids[1]) if mb_ids is not None else ()
                inputs_list = [*vals, nodes[x_in_id], *(nodes[i] for i in b_ids), *(nodes[i] for i in mb_flat)]
                n_b = len(b_ids)

                def _one_visit(*args):
                    l_params = tree_unflatten(list(zip(keys, args[:n_p], strict=True)))
                    x_a = args[n_p]
                    extra = args[n_p + 1 :]
                    mod.update(l_params)
                    try:
                        b_list = list(extra[:n_b])
                        msgs = None
                        if mb_ids is not None:
                            mb_args = extra[n_b:]
                            msgs = _mix(mb_args[:-1], mb_args[-1],
                                        self.message_queries[layer - self._loop_start], True)
                        res, _ = mod(x_a, b_list, msgs, None, 0, self.max_seq_len, self._experts, controller=ctrl)
                        return [res]
                    finally:
                        mod.update(orig)

                _, all_grads = mx.vjp(_one_visit, inputs_list, [d_out])
                mx.eval(all_grads)
                g_layer = tree_unflatten(list(zip(keys, all_grads[:n_p], strict=True))) if n_p else {}
                accum["layers"][layer] = add_grad_trees(accum["layers"][layer], g_layer)
                _add_cot(x_in_id, all_grads[n_p])
                for bid, g in zip(b_ids, all_grads[n_p + 1 : n_p + 1 + n_b], strict=True):
                    _add_cot(bid, g)
                for mid, g in zip(mb_flat, all_grads[n_p + 1 + n_b :], strict=True):
                    _add_cot(mid, g)

            return loss_val, accum
        finally:
            if window_size is not None:
                ctrl.clear_sequence()


def _layer_sizes(layout):
    return {bank.layer: sum(entry.size for entry in bank.ranges.values()) // bank.router_weight.shape[0]
            for bank in layout.expert_banks}


def _layer_visits(layout):
    counts = {}
    for layer, _ in layout.visits:
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def plan_inkling(text, layout, non_expert_bytes, budget, context_length):
    limit = _integer(text, "model_max_length", minimum=2)
    context = min(2048, limit) if context_length is None else context_length
    if type(context) is not int or not 2 <= context <= limit:
        raise ValueError(f"Inkling 2 context must be between 2 and {limit} tokens")
    sizes = _layer_sizes(layout)
    local = set(text["local_layer_ids"])
    kernel = _integer(text, "sconv_kernel_size")
    width = _integer(text, "hidden_size")
    state = mask_workspace = 0
    for layer, _ in layout.visits:
        sliding = layer in local
        heads, kv_heads, dim = attention_dimensions(text, sliding)
        key_length = min(context, _integer(text, "sliding_window_size")) if sliding else context
        state += 8 * kv_heads * dim * key_length + 8 * (kernel - 1) * (kv_heads * dim + width)
        mask_keys = min(context, key_length + _PREFILL_CHUNK - 1)
        mask_workspace = max(mask_workspace, 24 * heads * min(_PREFILL_CHUNK, context) * (mask_keys + 1))
    workspace = 512 * _MIB + 8 * max(sizes.values()) + 32 * context * width + mask_workspace
    available = budget - non_expert_bytes - state - workspace
    resident_bytes = sum(sizes[bank.layer] * bank.router_weight.shape[0] for bank in layout.expert_banks)
    if resident_bytes <= available:
        return context, None
    cache = min(1024 * _MIB, available)
    if cache < max(sizes.values()):
        raise ValueError("Inkling 2 cannot fit text weights, loop state and one expert in the RAM budget")
    return context, cache


def _load(entry, guard):
    guard.check()
    value = entry.load()
    mx.eval(value)
    return value


def load_inkling2(checkpoint, config, *, context_length, memory_budget_mib, memory_guard=None):
    ranges = tensor_ranges(checkpoint)
    layout = InklingTextLayout.from_checkpoint(checkpoint, config=config, ranges=ranges)
    expert_names = {f"{_TEXT}layers.{bank.layer}.mlp.experts.{name}"
                    for bank in layout.expert_banks for name in bank.ranges}
    non_experts = {name: entry for name, entry in ranges.items()
                   if name.startswith((_TEXT, "lm_head.")) and name not in expert_names}
    guard = memory_guard if memory_guard is not None else AppleMemoryGuard()
    budget = min(guard.budget, mx.device_info()["max_recommended_working_set_size"])
    if memory_budget_mib is not None:
        budget = min(budget, memory_budget_mib * _MIB)
    context, cache_limit = plan_inkling(config["text_config"], layout, sum(r.size for r in non_experts.values()),
                                       budget, context_length)
    arrays = {name: _load(entry, guard) for name, entry in non_experts.items()}
    cache = None
    if cache_limit is not None:
        visits = _layer_visits(layout)
        cache = LFMSSDExpertCache(
            cache_limit, guard, layer_sizes=_layer_sizes(layout),
            layer_weights={layer: count ** 2 for layer, count in visits.items()},
            decode_decay=0.5,
            fragment_size=max(entry.size // bank.router_weight.shape[0] for bank in layout.expert_banks
                              for entry in bank.ranges.values()),
            top_k=layout.expert_banks[0].top_k,
        )
    experts = _ResidentExperts(layout, guard) if cache is None else cache
    return Inkling2TextMLX(config, arrays, layout, experts, cache, guard, context).eval()
