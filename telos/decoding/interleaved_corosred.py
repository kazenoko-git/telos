#!/usr/bin/env python3
"""Interleaved COROSred decoder and causal AR baseline decoder.

This module implements the true interleaved COROSred state machine:
1. AR Drafting: Generates a short block (default: 32 tokens) autoregressively.
2. Bidirectional Inspection: Measures masked disagreement under MDLM context over
   the new block and bounded overlap (default: 8 tokens). Lexically filters for
   executable code tokens, excluding comments, docstrings, whitespace, and specials.
3. Remasking & Sequential Refinement: Deterministically selects top <=3 positions with
   positive disagreement, replaces them with MASK in the same sequence, and iteratively
   refines them (committing the highest-confidence prediction per forward pass).
4. Cache Rebuilding & AR Continuation: Rebuilds AR KV-cache when edits occur, maintains
   prompt immutability, and resumes drafting from the revised sequence.
5. Strict Decoupling: Zero benchmark test execution or reference code during inference.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from telos.eval.runner import clean_functional_completion

DEFAULT_STOP_STRINGS: Tuple[str, ...] = (
    "\nclass ",
    "\ndef ",
    "\nif __name__",
    "\nprint(",
    "<|im_end|>",
    "<|endoftext|>",
)


def sync_device(device: str) -> None:
    """Synchronize device queue for high-precision latency measurement."""
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def get_mask_token_id(tokenizer: Tokenizer) -> int:
    """Resolve and assert round-trip of diffusion mask token ID."""
    mask_id = tokenizer.token_to_id("<|mask|>")
    assert mask_id is not None, "Failed to resolve <|mask|> token ID."
    assert tokenizer.id_to_token(mask_id) == "<|mask|>", "Mask token round-trip failed."
    return mask_id


@dataclass
class InterleavedConfig:
    """Frozen configuration for interleaved COROSred decoding."""
    block_size: int = 32
    overlap: int = 8
    max_masks_per_block: int = 3
    max_refinement_passes: int = 3
    delta_threshold: float = 0.0
    max_new_tokens: int = 160
    stop_strings: Tuple[str, ...] = DEFAULT_STOP_STRINGS
    mask_id: int = 151665


@dataclass
class PerBlockTrace:
    """Trace of decoding, inspection, and refinement for a single block."""
    block_idx: int
    drafted_tokens: List[int]
    inspected_positions: List[Dict[str, Any]]
    remasked_positions: List[int]
    refinement_updates: List[Dict[str, Any]]
    cache_rebuilt: bool
    continued_draft_len: int


@dataclass
class InterleavedDecodeResult:
    """Complete inference artifact for an interleaved COROSred run."""
    token_ids: List[int]
    raw_output: str
    cleaned_output: str
    block_traces: List[PerBlockTrace]
    total_latency: float
    timing_breakdown: Dict[str, float]
    forward_passes: Dict[str, int]
    peak_memory_mb: float
    unresolved_masks: int


@dataclass
class BaselineDecodeResult:
    """Complete inference artifact for an AR baseline run."""
    token_ids: List[int]
    raw_output: str
    cleaned_output: str
    total_latency: float
    timing_breakdown: Dict[str, float]
    forward_passes: Dict[str, int]
    peak_memory_mb: float


def lexical_filter_executable_tokens(
    sequence_ids: List[int],
    prompt_len: int,
    window_start: int,
    window_end: int,
    tokenizer: Tokenizer,
) -> List[int]:
    """Identify token indices corresponding to executable code within a window.

    Excludes comments, docstrings, string literals (including unterminated literals),
    whitespace-only tokens, and special tokens. Uses documented fallback for incomplete code.
    """
    if window_start >= window_end:
        return []

    # Bounded candidate positions strictly in generated portion
    actual_start = max(prompt_len, window_start)
    actual_end = min(len(sequence_ids), window_end)
    if actual_start >= actual_end:
        return []

    try:
        full_code = tokenizer.decode(sequence_ids)

        # Lexical pattern spans for comments and strings
        comment_spans = [(m.start(), m.end()) for m in re.finditer(r"#.*$", full_code, re.MULTILINE)]
        doc_spans = [(m.start(), m.end()) for m in re.finditer(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"""[\s\S]*$|\'\'\'[\s\S]*$)', full_code)]
        str_spans = [(m.start(), m.end()) for m in re.finditer(r'("[^\n"]*"|\'[^\n\']*\'|"[^\n"]*$|\'[^\n\']*$)', full_code)]
        literal_spans = comment_spans + doc_spans + str_spans

        # Precompute character offset boundaries for each token
        curr_char = 0
        offsets: List[Tuple[int, int]] = []
        for tid in sequence_ids:
            t_str = tokenizer.decode([tid])
            s = curr_char
            e = curr_char + len(t_str)
            offsets.append((s, e))
            curr_char = e

        eligible: List[int] = []
        for i in range(actual_start, actual_end):
            tid = sequence_ids[i]
            t_str = tokenizer.decode([tid])

            # Exclude whitespace-only tokens
            if not t_str.strip():
                continue
            # Exclude special tokens (Qwen special IDs start >= 151643)
            if tid >= 151643:
                continue

            # Exclude tokens falling within comments or string literals
            s, e = offsets[i]
            if any(max(s, ls) < min(e, le) for ls, le in literal_spans):
                continue

            eligible.append(i)

        return eligible

    except Exception:
        # Documented fallback for incomplete code: exclude whitespace and specials
        fallback: List[int] = []
        for i in range(actual_start, actual_end):
            tid = sequence_ids[i]
            t_str = tokenizer.decode([tid])
            if t_str.strip() and tid < 151643:
                fallback.append(i)
        return fallback


def interleaved_corosred_decode(
    prompt_ids: List[int],
    ar_model,
    diff_model,
    tokenizer: Tokenizer,
    config: InterleavedConfig,
    device: str = "cpu",
) -> InterleavedDecodeResult:
    """Execute interleaved COROSred decoding maintaining exactly one live sequence.

    Constraints:
    - Pure decoding signature receiving only prompt, models, tokenizer, and config.
    - Zero execution feedback, benchmark tests, or reference solutions.
    - Exactly one evolving sequence with no candidate branching.
    - Prompt immutability and unresolved mask assertions enforced.
    """
    assert len(prompt_ids) > 0, "Prompt must not be empty."

    # Invariants and setup
    mask_id = config.mask_id
    eos_id = tokenizer.token_to_id("<|im_end|>")
    endoftext_id = tokenizer.token_to_id("<|endoftext|>")
    stop_ids: Set[int] = {s for s in (eos_id, endoftext_id) if s is not None}

    # Sequence state
    prompt_len = len(prompt_ids)
    current_sequence: List[int] = list(prompt_ids)

    # Timing metrics (non-overlapping)
    t_ar: float = 0.0
    t_probe: float = 0.0
    t_refine: float = 0.0
    t_rebuild: float = 0.0
    t_overhead: float = 0.0

    # Forward pass counters
    fp_counts = {
        "ar_draft": 0,
        "mdlm_probe_batches": 0,
        "mdlm_probe_positions": 0,
        "mdlm_refine_passes": 0,
        "ar_cache_rebuilds": 0,
    }

    block_traces: List[PerBlockTrace] = []

    # Initial KV-cache compilation on immutable prompt
    sync_device(device)
    t_start_rebuild = time.perf_counter()
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        prompt_logits, kv_cache = ar_model(prompt_tensor, is_causal=True, use_cache=True, start_pos=0)
    sync_device(device)
    t_rebuild += (time.perf_counter() - t_start_rebuild)
    fp_counts["ar_cache_rebuilds"] += 1

    last_ar_logits = prompt_logits[0, -1, :].float()

    block_idx = 0
    terminated = False

    while len(current_sequence) - prompt_len < config.max_new_tokens and not terminated:
        block_idx += 1
        block_draft_start = len(current_sequence)
        tokens_to_draft = min(config.block_size, config.max_new_tokens - (len(current_sequence) - prompt_len))
        drafted_in_block: List[int] = []

        # ==========================================
        # STEP 1: AR Drafting
        # ==========================================
        for _ in range(tokens_to_draft):
            sync_device(device)
            t0 = time.perf_counter()

            # Greedy argmax next token
            next_token = int(torch.argmax(last_ar_logits).item())
            current_pos = len(current_sequence)
            current_sequence.append(next_token)
            drafted_in_block.append(next_token)

            # Check stop token
            if next_token in stop_ids:
                sync_device(device)
                t_ar += (time.perf_counter() - t0)
                terminated = True
                break

            # Incremental AR step with KV-cache
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            with torch.no_grad():
                step_logits, kv_cache = ar_model(
                    next_tensor,
                    is_causal=True,
                    past_key_values=kv_cache,
                    start_pos=current_pos,
                    use_cache=True,
                )
            last_ar_logits = step_logits[0, -1, :].float()
            fp_counts["ar_draft"] += 1

            sync_device(device)
            t_ar += (time.perf_counter() - t0)

            # Check stop strings in decoded generated sequence
            sync_device(device)
            t_ov0 = time.perf_counter()
            gen_text = tokenizer.decode(current_sequence[prompt_len:])
            if any(stop_s in gen_text for stop_s in config.stop_strings):
                terminated = True
                t_overhead += (time.perf_counter() - t_ov0)
                break
            t_overhead += (time.perf_counter() - t_ov0)

        # ==========================================
        # STEP 2: Bidirectional Inspection
        # ==========================================
        sync_device(device)
        t_ov1 = time.perf_counter()

        # Bounded inspection window: newly drafted block plus bounded overlap of previous block
        window_start = max(prompt_len, block_draft_start - config.overlap)
        window_end = len(current_sequence)

        eligible_positions = lexical_filter_executable_tokens(
            current_sequence,
            prompt_len=prompt_len,
            window_start=window_start,
            window_end=window_end,
            tokenizer=tokenizer,
        )
        t_overhead += (time.perf_counter() - t_ov1)

        inspected_positions: List[Dict[str, Any]] = []
        if eligible_positions:
            sync_device(device)
            t_pr0 = time.perf_counter()

            # Batch masked MDLM evaluations across eligible positions
            batch_seqs: List[List[int]] = []
            for pos in eligible_positions:
                masked_seq = list(current_sequence)
                masked_seq[pos] = mask_id
                batch_seqs.append(masked_seq)

            batch_tensor = torch.tensor(batch_seqs, dtype=torch.long, device=device)
            with torch.no_grad():
                batch_logits, _ = diff_model(batch_tensor, is_causal=False)

            sync_device(device)
            t_probe += (time.perf_counter() - t_pr0)
            fp_counts["mdlm_probe_batches"] += 1
            fp_counts["mdlm_probe_positions"] += len(eligible_positions)

            sync_device(device)
            t_ov2 = time.perf_counter()
            for idx, pos in enumerate(eligible_positions):
                pos_logits = batch_logits[idx, pos, :].float()
                log_p = F.log_softmax(pos_logits, dim=-1)

                orig_id = current_sequence[pos]
                orig_logp = float(log_p[orig_id].item())

                log_p_alt = log_p.clone()
                log_p_alt[orig_id] = -float("inf")
                best_alt_id = int(torch.argmax(log_p_alt).item())
                best_alt_logp = float(log_p[best_alt_id].item())

                delta_logp = best_alt_logp - orig_logp

                inspected_positions.append({
                    "position": pos,
                    "orig_id": orig_id,
                    "orig_token": tokenizer.decode([orig_id]),
                    "best_alt_id": best_alt_id,
                    "best_alt_token": tokenizer.decode([best_alt_id]),
                    "delta_logp": delta_logp,
                })
            t_overhead += (time.perf_counter() - t_ov2)

        # ==========================================
        # STEP 3: Remasking and Sequential Refinement
        # ==========================================
        sync_device(device)
        t_ov3 = time.perf_counter()

        # Deterministic ranking rule: delta_logp > threshold, descending delta_logp, top <=3
        candidates = [c for c in inspected_positions if c["delta_logp"] > config.delta_threshold]
        candidates.sort(key=lambda c: (c["delta_logp"], -c["position"]), reverse=True)
        selected_candidates = candidates[: config.max_masks_per_block]
        remasked_positions = [c["position"] for c in selected_candidates]
        t_overhead += (time.perf_counter() - t_ov3)

        refinement_updates: List[Dict[str, Any]] = []
        cache_rebuilt = False

        if remasked_positions:
            # Replace selected positions with MASK in the SAME live sequence
            orig_tokens_map = {p: current_sequence[p] for p in remasked_positions}
            for p in remasked_positions:
                current_sequence[p] = mask_id

            remaining_masks = set(remasked_positions)

            # Refine sequentially over up to max_refinement_passes forward passes
            for _ in range(min(config.max_refinement_passes, len(remasked_positions))):
                if not remaining_masks:
                    break

                sync_device(device)
                t_ref0 = time.perf_counter()

                ref_tensor = torch.tensor([current_sequence], dtype=torch.long, device=device)
                with torch.no_grad():
                    ref_logits, _ = diff_model(ref_tensor, is_causal=False)

                sync_device(device)
                t_refine += (time.perf_counter() - t_ref0)
                fp_counts["mdlm_refine_passes"] += 1

                sync_device(device)
                t_ov4 = time.perf_counter()

                # Find highest-confidence prediction among remaining masked positions
                best_pos: Optional[int] = None
                best_token_id: Optional[int] = None
                best_confidence: float = -1.0

                # Sort remaining masks for deterministic iteration
                for p in sorted(remaining_masks):
                    p_logits = ref_logits[0, p, :].float()
                    p_dist = F.softmax(p_logits, dim=-1)
                    pred_id = int(torch.argmax(p_dist).item())
                    conf = float(p_dist[pred_id].item())
                    if conf > best_confidence:
                        best_confidence = conf
                        best_pos = p
                        best_token_id = pred_id

                assert best_pos is not None and best_token_id is not None

                # Commit one position per pass; others remain masked
                current_sequence[best_pos] = best_token_id
                remaining_masks.remove(best_pos)

                refinement_updates.append({
                    "position": best_pos,
                    "previous_token_id": orig_tokens_map[best_pos],
                    "committed_token_id": best_token_id,
                    "confidence": best_confidence,
                    "changed": (orig_tokens_map[best_pos] != best_token_id),
                })
                t_overhead += (time.perf_counter() - t_ov4)

            # ==========================================
            # STEP 4: AR KV-Cache Invalidation & Rebuild
            # ==========================================
            # Check if any committed token actually changed from the original sequence
            tokens_changed = any(u["changed"] for u in refinement_updates)
            if tokens_changed:
                sync_device(device)
                t_reb0 = time.perf_counter()

                # Full-prefix rebuild of AR KV-cache up to current sequence length
                full_tensor = torch.tensor([current_sequence], dtype=torch.long, device=device)
                with torch.no_grad():
                    rebuilt_logits, kv_cache = ar_model(
                        full_tensor,
                        is_causal=True,
                        use_cache=True,
                        start_pos=0,
                    )
                last_ar_logits = rebuilt_logits[0, -1, :].float()

                sync_device(device)
                t_rebuild += (time.perf_counter() - t_reb0)
                fp_counts["ar_cache_rebuilds"] += 1
                cache_rebuilt = True

        # Record per-block trace
        block_traces.append(PerBlockTrace(
            block_idx=block_idx,
            drafted_tokens=drafted_in_block,
            inspected_positions=inspected_positions,
            remasked_positions=remasked_positions,
            refinement_updates=refinement_updates,
            cache_rebuilt=cache_rebuilt,
            continued_draft_len=len(current_sequence),
        ))

    # ==========================================
    # STEP 5: Termination & Validation
    # ==========================================
    # Assert immutable prompt
    assert current_sequence[:prompt_len] == prompt_ids, "Prompt tokens were mutated during decoding!"

    # Assert zero unresolved mask tokens
    unresolved = sum(1 for tid in current_sequence[prompt_len:] if tid == mask_id)
    assert unresolved == 0, f"Encountered {unresolved} unresolved MASK tokens in generated output!"

    raw_output = tokenizer.decode(current_sequence[prompt_len:])
    cleaned_output = clean_functional_completion(tokenizer.decode(prompt_ids), raw_output)

    total_latency = t_ar + t_probe + t_refine + t_rebuild + t_overhead

    peak_mem_mb = 0.0
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        peak_mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)

    return InterleavedDecodeResult(
        token_ids=current_sequence[prompt_len:],
        raw_output=raw_output,
        cleaned_output=cleaned_output,
        block_traces=block_traces,
        total_latency=total_latency,
        timing_breakdown={
            "ar_draft_s": t_ar,
            "mdlm_probe_s": t_probe,
            "mdlm_refine_s": t_refine,
            "ar_rebuild_s": t_rebuild,
            "overhead_s": t_overhead,
        },
        forward_passes=fp_counts,
        peak_memory_mb=peak_mem_mb,
        unresolved_masks=unresolved,
    )


def ar_baseline_decode(
    prompt_ids: List[int],
    ar_model,
    tokenizer: Tokenizer,
    config: InterleavedConfig,
    device: str = "cpu",
) -> BaselineDecodeResult:
    """Execute untouched causal AR baseline decoding under identical environment."""
    assert len(prompt_ids) > 0, "Prompt must not be empty."

    prompt_len = len(prompt_ids)
    current_sequence: List[int] = list(prompt_ids)

    eos_id = tokenizer.token_to_id("<|im_end|>")
    endoftext_id = tokenizer.token_to_id("<|endoftext|>")
    stop_ids: Set[int] = {s for s in (eos_id, endoftext_id) if s is not None}

    t_ar: float = 0.0
    t_overhead: float = 0.0
    fp_counts = {"ar_draft": 0, "ar_cache_rebuilds": 1}

    # Initial prompt ingestion with KV-cache
    sync_device(device)
    t0 = time.perf_counter()
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        prompt_logits, kv_cache = ar_model(prompt_tensor, is_causal=True, use_cache=True, start_pos=0)
    sync_device(device)
    t_ar += (time.perf_counter() - t0)

    last_logits = prompt_logits[0, -1, :].float()

    for _ in range(config.max_new_tokens):
        sync_device(device)
        t_step0 = time.perf_counter()

        next_token = int(torch.argmax(last_logits).item())
        current_pos = len(current_sequence)
        current_sequence.append(next_token)

        if next_token in stop_ids:
            sync_device(device)
            t_ar += (time.perf_counter() - t_step0)
            break

        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        with torch.no_grad():
            step_logits, kv_cache = ar_model(
                next_tensor,
                is_causal=True,
                past_key_values=kv_cache,
                start_pos=current_pos,
                use_cache=True,
            )
        last_logits = step_logits[0, -1, :].float()
        fp_counts["ar_draft"] += 1

        sync_device(device)
        t_ar += (time.perf_counter() - t_step0)

        # Stop string check
        t_ov0 = time.perf_counter()
        gen_text = tokenizer.decode(current_sequence[prompt_len:])
        if any(stop_s in gen_text for stop_s in config.stop_strings):
            t_overhead += (time.perf_counter() - t_ov0)
            break
        t_overhead += (time.perf_counter() - t_ov0)

    # Invariants check
    assert current_sequence[:prompt_len] == prompt_ids, "Baseline prompt modified!"

    raw_output = tokenizer.decode(current_sequence[prompt_len:])
    cleaned_output = clean_functional_completion(tokenizer.decode(prompt_ids), raw_output)

    total_latency = t_ar + t_overhead

    peak_mem_mb = 0.0
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        peak_mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)

    return BaselineDecodeResult(
        token_ids=current_sequence[prompt_len:],
        raw_output=raw_output,
        cleaned_output=cleaned_output,
        total_latency=total_latency,
        timing_breakdown={
            "ar_draft_s": t_ar,
            "overhead_s": t_overhead,
        },
        forward_passes=fp_counts,
        peak_memory_mb=peak_mem_mb,
    )
