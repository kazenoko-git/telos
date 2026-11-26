"""Offline replay of expert route traces under bounded SSD cache policies."""

from collections import Counter, OrderedDict


def _layer_slots(byte_limit, layer_sizes):
    if not layer_sizes or len(set(layer_sizes.values())) != 1:
        return {}
    slots = byte_limit // next(iter(layer_sizes.values()))
    n = len(layer_sizes)
    if slots < n:
        return {}
    return {layer: (i + 1) * slots // n - i * slots // n for i, layer in enumerate(sorted(layer_sizes))}


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


def replay_existing(records, byte_limit, *, layer_sizes=None, use_frequency_decode=True):
    """Reproduce Telos's current SSDExpertCache / LFMSSDExpertCache eviction behavior."""
    entries = OrderedDict()
    resident_bytes = 0
    slots = _layer_slots(byte_limit, layer_sizes)
    decode_uses = None
    decode_tokens = {
        record["token_index"] for record in records if record["phase"] == "decode"
    }
    phase_stats = {
        "prefill": {"hits": 0, "misses": 0, "bytes_read": 0},
        "decode": {"hits": 0, "misses": 0, "bytes_read": 0},
    }

    for record in records:
        phase = record["phase"]
        if phase == "decode" and decode_uses is None and use_frequency_decode and slots:
            decode_uses = Counter()
        layer = record["layer"]
        stats = phase_stats[phase]
        for expert, size in zip(record["experts"], record["byte_sizes"], strict=True):
            key = (layer, expert)
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
                resident = [k for k in entries if k[0] == layer]
                if len(resident) >= slots[layer]:
                    victim = (resident[0] if decode_uses is None
                              else min(resident, key=decode_uses.__getitem__))
                    if decode_uses is not None and decode_uses[key] < decode_uses[victim]:
                        stats["bytes_read"] += size
                        continue
                    resident_bytes -= entries.pop(victim)
            while resident_bytes + size > byte_limit:
                _, evicted_size = entries.popitem(last=False)
                resident_bytes -= evicted_size
            entries[key] = size
            resident_bytes += size
            stats["bytes_read"] += size

    total = {
        "hits": phase_stats["prefill"]["hits"] + phase_stats["decode"]["hits"],
        "misses": phase_stats["prefill"]["misses"] + phase_stats["decode"]["misses"],
        "bytes_read": phase_stats["prefill"]["bytes_read"] + phase_stats["decode"]["bytes_read"],
    }
    return {
        "byte_limit": byte_limit,
        "decode_tokens": len(decode_tokens),
        "prefill": _phase_metrics(phase_stats["prefill"], 0),
        "decode": _phase_metrics(phase_stats["decode"], len(decode_tokens)),
        "total": _phase_metrics(total, len(decode_tokens)),
    }
