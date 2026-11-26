"""Offline replay of expert route traces under bounded SSD cache policies."""

import math
from collections import OrderedDict, defaultdict

from .ssd_experts import allocate_layer_slots


def _phase_metrics(stats, decode_tokens):
    requests = stats["hits"] + stats["misses"]
    return {
        "hits": stats["hits"],
        "misses": stats["misses"],
        "requests": requests,
        "hit_rate": stats["hits"] / requests if requests else 0.0,
        "bytes_read": stats["bytes_read"],
        "bytes_per_decode_token": stats["bytes_read"] / decode_tokens if decode_tokens else 0.0,
    }


def _summarize(byte_limit, decode_tokens, phase_stats):
    total = {
        "hits": phase_stats["prefill"]["hits"] + phase_stats["decode"]["hits"],
        "misses": phase_stats["prefill"]["misses"] + phase_stats["decode"]["misses"],
        "bytes_read": phase_stats["prefill"]["bytes_read"] + phase_stats["decode"]["bytes_read"],
    }
    return {
        "byte_limit": byte_limit,
        "decode_tokens": decode_tokens,
        "prefill": _phase_metrics(phase_stats["prefill"], 0),
        "decode": _phase_metrics(phase_stats["decode"], decode_tokens),
        "total": _phase_metrics(total, decode_tokens),
    }


def replay_policy(records, byte_limit, *, layer_sizes=None, layer_weights=None,
                  use_frequency_decode=True, decode_decay=1.0, protect_visit=False,
                  equal_byte_quotas=False):
    """Simulate demand-only SSD expert caching on a recorded route trace."""
    entries = OrderedDict()
    resident_bytes = 0
    if equal_byte_quotas and layer_sizes:
        quotas = dict.fromkeys(layer_sizes, byte_limit // len(layer_sizes))
        slots = {layer: quotas[layer] // size for layer, size in layer_sizes.items()}
    else:
        slots = allocate_layer_slots(byte_limit, layer_sizes, layer_weights)
    decode_uses = None
    last_decode_token = None
    decode_tokens = {r["token_index"] for r in records if r["phase"] == "decode"}
    phase_stats = {
        "prefill": {"hits": 0, "misses": 0, "bytes_read": 0},
        "decode": {"hits": 0, "misses": 0, "bytes_read": 0},
    }

    for record in records:
        phase = record["phase"]
        layer = record["layer"]
        if phase == "decode":
            if decode_uses is None and use_frequency_decode and slots:
                decode_uses = defaultdict(float if decode_decay != 1.0 else int)
            token_idx = record["token_index"]
            if decode_uses is not None and token_idx != last_decode_token:
                if last_decode_token is not None and decode_decay != 1.0:
                    for key in list(decode_uses):
                        decode_uses[key] *= decode_decay
                last_decode_token = token_idx

        protected = {(layer, expert) for expert in record["experts"]} if protect_visit else set()
        stats = phase_stats[phase]
        for expert, size in zip(record["experts"], record["byte_sizes"], strict=True):
            key = (layer, expert)
            protected.discard(key)
            if decode_uses is not None:
                decode_uses[key] += 1
            if key in entries:
                stats["hits"] += 1
                entries.move_to_end(key)
                continue
            if size > byte_limit:
                raise ValueError(f"Expert ({layer}, {expert}) needs {size} bytes over {byte_limit}")
            stats["misses"] += 1
            if layer in slots:
                layer_limit = slots[layer]
                if layer_limit == 0:
                    stats["bytes_read"] += size
                    continue
                resident = [k for k in entries if k[0] == layer]
                if len(resident) >= layer_limit:
                    candidates = [k for k in resident if k not in protected]
                    if not candidates:
                        stats["bytes_read"] += size
                        continue
                    victim = (candidates[0] if decode_uses is None
                              else min(candidates, key=decode_uses.__getitem__))
                    if decode_uses is not None and decode_uses[key] < decode_uses[victim]:
                        stats["bytes_read"] += size
                        continue
                    resident_bytes -= entries.pop(victim)
            while resident_bytes + size > byte_limit:
                victim = next((k for k in entries if k not in protected), None)
                if victim is None:
                    break
                resident_bytes -= entries.pop(victim)
            if resident_bytes + size > byte_limit:
                stats["bytes_read"] += size
                continue
            entries[key] = size
            resident_bytes += size
            stats["bytes_read"] += size

    return _summarize(byte_limit, len(decode_tokens), phase_stats)


def replay_existing(records, byte_limit, *, layer_sizes=None, use_frequency_decode=True):
    """Reproduce Telos's legacy global/uniform-slot eviction behavior."""
    return replay_policy(
        records, byte_limit, layer_sizes=layer_sizes, layer_weights=None,
        use_frequency_decode=use_frequency_decode, decode_decay=1.0, protect_visit=False,
    )


def replay_belady(records, byte_limit, *, protect_visit=True):
    """Offline future-aware Belady-MIN reference allowing variable item sizes."""
    requests = [
        (visit_idx, r["phase"], r["layer"], expert, size)
        for visit_idx, r in enumerate(records)
        for expert, size in zip(r["experts"], r["byte_sizes"], strict=True)
    ]
    last_seen = {}
    next_use = [math.inf] * len(requests)
    for idx in range(len(requests) - 1, -1, -1):
        key = (requests[idx][2], requests[idx][3])
        if key in last_seen:
            next_use[idx] = last_seen[key]
        last_seen[key] = idx

    entries = OrderedDict()
    resident_bytes = 0
    last_req = {}
    decode_tokens = {r["token_index"] for r in records if r["phase"] == "decode"}
    phase_stats = {
        "prefill": {"hits": 0, "misses": 0, "bytes_read": 0},
        "decode": {"hits": 0, "misses": 0, "bytes_read": 0},
    }

    req_idx = 0
    for record in records:
        layer = record["layer"]
        protected = {(layer, expert) for expert in record["experts"]} if protect_visit else set()
        for expert, size in zip(record["experts"], record["byte_sizes"], strict=True):
            _, phase, layer, expert, size = requests[req_idx]
            key = (layer, expert)
            protected.discard(key)
            last_req[key] = req_idx
            stats = phase_stats[phase]
            if key in entries:
                stats["hits"] += 1
                entries.move_to_end(key)
                req_idx += 1
                continue
            stats["misses"] += 1
            stats["bytes_read"] += size
            upcoming = next_use[req_idx]
            req_idx += 1
            if math.isinf(upcoming) or size > byte_limit:
                continue
            candidates = sorted(
                (k for k in entries if k not in protected),
                key=lambda k: next_use[last_req[k]],
                reverse=True,
            )
            needed = resident_bytes + size - byte_limit
            victims = []
            freed = 0
            for cand in candidates:
                if freed >= needed or next_use[last_req[cand]] <= upcoming:
                    break
                victims.append(cand)
                freed += entries[cand]
            if freed < needed:
                continue
            for cand in victims:
                resident_bytes -= entries.pop(cand)
            entries[key] = size
            resident_bytes += size

    return _summarize(byte_limit, len(decode_tokens), phase_stats)
