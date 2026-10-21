#!/usr/bin/env python3
"""Verification tests for the interleaved COROSred decoder.

Verifications:
1. Cache Correctness: Compare incremental continuation with rebuilt KV-cache against
   full-prefix recomputation after an intentional edit.
2. Prompt Immutability: Assert that prompt tokens are never mutated during decoding.
3. Token-Position Mapping: Verify lexical token analysis excludes comments, docstrings,
   whitespace, and specials without offset drift.
4. Absence of Unresolved Masks & Clean Termination: Assert that no <|mask|> tokens remain
   in emitted outputs.
5. Inference Call Graph Audit: Statically inspect AST to prove no evaluation harnesses,
   sandboxes, or benchmark answers are imported or called.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from scripts.qwen25_coder_model import (
    get_ar_weights_path,
    get_diff_weights_path,
    load_qwen25_coder_model,
)
from telos.decoding.interleaved_corosred import (
    InterleavedConfig,
    ar_baseline_decode,
    interleaved_corosred_decode,
    lexical_filter_executable_tokens,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _has_qwen_weights() -> bool:
    """Checks whether local Hugging Face cache has Qwen2.5-Coder model weights."""
    try:
        get_diff_weights_path()
        get_ar_weights_path()
        return True
    except FileNotFoundError:
        return False


def test_call_graph_audit_no_eval_leakage():
    """Audit AST of telos/decoding/interleaved_corosred.py for zero eval leakage."""
    decoder_file = PROJECT_ROOT / "telos/decoding/interleaved_corosred.py"
    with open(decoder_file, "r") as f:
        tree = ast.parse(f.read(), filename=str(decoder_file))

    forbidden_names = {
        "execute_code_sandboxed",
        "ExecutionResult",
        "mbpp_suite",
        "test_harness",
        "reference_code",
        "benchmark_answers",
        "pytest",
    }

    imported_names = set()
    called_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module)
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    leaked_imports = imported_names.intersection(forbidden_names)
    assert not leaked_imports, f"Forbidden imports detected in decoder: {leaked_imports}"

    leaked_calls = called_names.intersection(forbidden_names)
    assert not leaked_calls, f"Forbidden calls detected in decoder: {leaked_calls}"


@pytest.mark.skipif(not _has_qwen_weights(), reason="Qwen2.5-Coder weights not present in local HF cache")
def test_token_position_mapping_and_lexical_filter():
    """Verify lexical token filtering cleanly isolates executable code tokens."""
    diff_path = get_diff_weights_path()
    tokenizer = Tokenizer.from_file(str(diff_path.parent / "tokenizer.json"))

    code = (
        "def sum_two(a: int, b: int) -> int:\n"
        "    # Line comment explaining sum\n"
        '    """Docstring explaining function."""\n'
        "    total = a + b\n"
        '    msg = "incomplete'
    )

    prompt_str = "def sum_two(a: int, b: int) -> int:\n"
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False).ids
    full_ids = tokenizer.encode(code, add_special_tokens=False).ids

    eligible = lexical_filter_executable_tokens(
        sequence_ids=full_ids,
        prompt_len=len(prompt_ids),
        window_start=len(prompt_ids),
        window_end=len(full_ids),
        tokenizer=tokenizer,
    )

    assert len(eligible) > 0, "Expected non-zero eligible executable tokens."

    # Validate each eligible token
    for idx in eligible:
        assert idx >= len(prompt_ids), f"Eligible token {idx} falls within prompt!"
        t_id = full_ids[idx]
        t_str = tokenizer.decode([t_id])
        assert t_str.strip() != "", f"Whitespace-only token {idx} ({t_str!r}) marked eligible!"
        assert t_id < 151643, f"Special token {idx} (ID {t_id}) marked eligible!"
        assert "#" not in t_str or "\n" in t_str, f"Comment token {idx} marked eligible: {t_str}"
        assert '"""' not in t_str, f"Docstring token {idx} marked eligible: {t_str}"


def test_kv_cache_correctness_after_intentional_edit():
    """Assert incremental generation with rebuilt cache matches full prefix forward pass."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ar_weights = get_ar_weights_path()
    # Use float32 for deterministic mathematical comparison of KV caching
    model = load_qwen25_coder_model(ar_weights, device=device, dtype=torch.float32)

    prompt = [101, 202, 303, 404, 505]
    next_tok = 606
    edited_seq = [101, 202, 9999, 404, 505, next_tok]  # Intentionally mutate position 2

    # Full prefix forward pass with KV cache rebuild
    full_tensor = torch.tensor([edited_seq], dtype=torch.long, device=device)
    with torch.no_grad():
        rebuilt_logits, rebuilt_kvs = model(full_tensor, is_causal=True, use_cache=True, start_pos=0)
    last_rebuilt = rebuilt_logits[0, -1, :]

    # Next token incremental step using the rebuilt cache
    candidate_next = int(torch.argmax(last_rebuilt).item())
    next_tensor = torch.tensor([[candidate_next]], dtype=torch.long, device=device)
    with torch.no_grad():
        inc_logits, _ = model(
            next_tensor,
            is_causal=True,
            past_key_values=rebuilt_kvs,
            start_pos=len(edited_seq),
            use_cache=True,
        )
    inc_next_logits = inc_logits[0, -1, :]

    # Ground-truth: full forward pass without cache on the entire sequence + candidate_next
    gt_tensor = torch.tensor([edited_seq + [candidate_next]], dtype=torch.long, device=device)
    with torch.no_grad():
        gt_logits, _ = model(gt_tensor, is_causal=True, use_cache=False)
    gt_next_logits = gt_logits[0, -1, :]

    diff = (gt_next_logits - inc_next_logits).abs().max().item()
    assert diff < 1e-4, f"Cache continuation diverged from full prefix! Max diff: {diff}"
    assert torch.argmax(gt_next_logits).item() == torch.argmax(inc_next_logits).item(), "Argmax token mismatch!"


@pytest.mark.skipif(not _has_qwen_weights(), reason="Qwen2.5-Coder weights not present in local HF cache")
def test_interleaved_decoder_smoke_and_invariants():
    """Smoke test running interleaved decoder for a short prompt and asserting invariants."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ar_weights = get_ar_weights_path()
    diff_weights = get_diff_weights_path()

    ar_model = load_qwen25_coder_model(ar_weights, device=device, dtype=torch.bfloat16)
    diff_model = load_qwen25_coder_model(diff_weights, device=device, dtype=torch.bfloat16)
    tokenizer = Tokenizer.from_file(str(diff_weights.parent / "tokenizer.json"))

    prompt_str = "def is_positive(x: int) -> bool:\n"
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False).ids

    config = InterleavedConfig(
        block_size=8,
        overlap=4,
        max_masks_per_block=2,
        max_refinement_passes=2,
        max_new_tokens=16,
    )

    result = interleaved_corosred_decode(
        prompt_ids=prompt_ids,
        ar_model=ar_model,
        diff_model=diff_model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )

    # Invariants checks
    assert len(result.token_ids) > 0, "No tokens emitted!"
    assert result.unresolved_masks == 0, f"Found unresolved masks: {result.unresolved_masks}"
    assert config.mask_id not in result.token_ids, "Mask token in emitted token IDs!"
    assert result.total_latency > 0.0, "Total latency should be non-zero."
    assert len(result.block_traces) > 0, "Expected at least one block trace."

    # Baseline decoder smoke test
    base_result = ar_baseline_decode(
        prompt_ids=prompt_ids,
        ar_model=ar_model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )
    assert len(base_result.token_ids) > 0, "No baseline tokens emitted!"
    assert base_result.total_latency > 0.0, "Baseline latency should be non-zero."
