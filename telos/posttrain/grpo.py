"""Collect on-policy completions and compute the GRPO token objective."""

import importlib
import math
import unicodedata

import numpy as np


def reward_function(spec):
    """Resolve an exact-match verifier or a Python callable (record, completion)."""
    if spec == "exact":
        return _exact_match
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("reward must be 'exact' or a Python module:function")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise ValueError(f"Reward {spec!r} must be callable")
    return function


def _exact_match(record, completion):
    def normalize(text):
        return " ".join(unicodedata.normalize("NFC", text).split())
    return float(normalize(completion) == normalize(record["answer"]))


def prepare_prompts(records, tokenizer, max_length, max_new_tokens):
    bos, eos, pad = (tokenizer.token_to_id(name) for name in ("[BOS]", "[EOS]", "[PAD]"))
    if bos is None or eos is None or pad is None:
        raise ValueError("Tokenizer must define [BOS], [EOS], and [PAD] tokens")
    prompt_budget = max_length - max_new_tokens - 1
    if prompt_budget < 0:
        raise ValueError("GRPO max_length must reserve space for BOS and max_new_tokens")
    prompts = []
    for record in records:
        tokens = tokenizer.encode(record["prompt"], add_special_tokens=False).ids
        tokens = tokens[-prompt_budget:] if prompt_budget else []
        prompts.append((record, [bos, *tokens]))
    return prompts, eos, pad


def collect_rollouts(batch, tokenizer, eos, pad, group_size, max_new_tokens, sample, reward):
    """Sample one group per prompt. EOS is a policy action; padding is ignored."""
    sequences = [prefix.copy() for _, prefix in batch for _ in range(group_size)]
    prefix_lengths = [len(sequence) for sequence in sequences]
    action_logps = [[] for _ in sequences]
    active = list(range(len(sequences)))
    for _ in range(max_new_tokens):
        if not active:
            break
        lengths = [len(sequences[index]) for index in active]
        ids = np.full((len(active), max(lengths)), pad, dtype=np.int32)
        for row, index in enumerate(active):
            ids[row, :lengths[row]] = sequences[index]
        tokens, logps = sample(ids, [length - 1 for length in lengths])
        unfinished = []
        for index, token, logp in zip(active, tokens, logps, strict=True):
            sequences[index].append(token)
            action_logps[index].append(logp)
            if token != eos:
                unfinished.append(index)
        active = unfinished

    rows, rewards = [], []
    for index, (sequence, prefix_length) in enumerate(zip(sequences, prefix_lengths)):
        record, _ = batch[index // group_size]
        completion = tokenizer.decode(sequence[prefix_length:], skip_special_tokens=True)
        score = float(reward(record, completion))
        if not math.isfinite(score):
            raise ValueError("GRPO reward functions must return a finite scalar")
        rewards.append(score)
        labels = [-100] * prefix_length + sequence[prefix_length:]
        rows.append({"_encoded": (sequence, labels, pad)})

    scores = np.asarray(rewards, dtype=np.float64).reshape(-1, group_size)
    # Scale before centering to handle large finite custom rewards safely.
    scale = np.maximum(np.max(np.abs(scores), axis=1, keepdims=True), 1.0)
    scaled = scores / scale
    centered = scaled - scaled.mean(axis=1, keepdims=True)
    spread = scaled.std(axis=1, keepdims=True)
    advantages = centered / np.maximum(spread, 1e-8 / scale)
    return (
        rows, action_logps, advantages.ravel().astype(np.float32),
        float((scores / scores.size).sum()), float((spread == 0).mean()),
    )


def collate_old_logps(rows, action_logps, width):
    """Align sampled action scores with shifted labels, leaving padding at zero."""
    old = np.zeros((len(rows), width - 1), dtype=np.float32)
    for i, (row, logps) in enumerate(zip(rows, action_logps, strict=True)):
        sequence, _, _ = row["_encoded"]
        end = len(sequence) - 1
        old[i, end - len(logps):end] = logps
    return old


def torch_loss(logps, old, reference, labels, advantages, beta, epsilon):
    import torch

    valid = labels[:, 1:].ne(-100)
    ratio = torch.exp(torch.where(valid, logps - old, 0.0))
    advantage = advantages[:, None]
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    if beta:
        delta = torch.where(valid, reference - logps, 0.0)
        objective = objective - beta * (torch.expm1(delta) - delta)
    return -((objective * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)).mean()


def mlx_loss(logps, old, reference, labels, advantages, beta, epsilon):
    import mlx.core as mx

    valid = labels[:, 1:] != -100
    ratio = mx.exp(mx.where(valid, logps - old, 0.0))
    advantage = advantages[:, None]
    objective = mx.minimum(ratio * advantage, mx.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage)
    if beta:
        delta = mx.where(valid, reference - logps, 0.0)
        objective = objective - beta * (mx.expm1(delta) - delta)
    return -mx.mean(mx.sum(objective * valid, axis=1) / mx.maximum(mx.sum(valid, axis=1), 1))
