#!/usr/bin/env python3
"""Prepare 112 evaluation cases for DCLM-530M models.

Generates:
- 24 counterfactual pairs (48 cases): 12 arithmetic and 12 comparison pairs.
  Identical token prefixes and hole positions, different operators (+ vs - or < vs >),
  different assertions in suffixes, and equal token lengths.
- 64 local code reconstructions sampled deterministically from private_unseen_suite.json.
  All cases verified to have exact single-token hole boundaries in DCLM tokenizer.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "research/matched_530m_corosred_probe_2026-10-18"
CASES_FILE = OUT_DIR / "cases.json"
TOKENIZER_FILE = (
    Path.home()
    / ".cache/huggingface/hub/models--michaelchenkj--datadecide-dclm-530m-53b/snapshots/213f7e96745ee3d397834821fe2b397019bdde98/weights/step57786/tokenizer.json"
)
CORPUS_SOURCE = ROOT / "evals/benchmarks/private_unseen_suite.json"


def make_case(tokenizer: Tokenizer, ident: str, text: str, start: int, end: int, group: str, pair: str | None = None) -> dict:
    enc = tokenizer.encode(text, add_special_tokens=False)
    hits = [i for i, (a, b) in enumerate(enc.offsets) if a < end and b > start]
    if len(hits) != 1:
        raise ValueError(f"{ident}: hole is not exactly one token: {hits}")
    i = hits[0]
    a, b = enc.offsets[i]
    if (a, b) != (start, end) or i == 0 or i >= len(enc.ids) - 3:
        raise ValueError(f"{ident}: invalid hole boundary {(a, b, start, end, i)}")
    return {
        "id": ident,
        "group": group,
        "pair": pair,
        "text": text,
        "ids": enc.ids,
        "hole": i,
        "target": enc.ids[i],
        "target_text": text[a:b],
    }


def prepare_cases() -> list[dict]:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    cases = []

    # 1. 12 Arithmetic Pairs
    operands = [(a, b) for a in range(3, 8) for b in range(1, a) if a + b < 10][:12]
    for k, (a, b) in enumerate(operands):
        for op, result in [("+", a + b), ("-", a - b)]:
            prefix = f"def calc_{k}(a, b):\n    return a"
            op_text = f" {op}"
            suffix = f" b\n\nassert calc_{k}({a}, {b}) == {result}\n"
            text = prefix + op_text + suffix
            # Verify executable correctness
            exec(compile(text, f"<arithmetic_{k}_{op}>", "exec"), {})
            # In GPT-NeoX: 'def calc_k(a, b):\n return a' ends at len(prefix).
            # The operator is ' +' or ' -'. Let us find its exact token boundary:
            enc = tokenizer.encode(text, add_special_tokens=False)
            # Find the token index containing the operator character
            op_char_pos = len(prefix) + op_text.find(op)
            hits = [idx for idx, (ca, cb) in enumerate(enc.offsets) if ca <= op_char_pos < cb]
            assert len(hits) == 1, f"Multiple hits for {op} in {ident}"
            hole_idx = hits[0]
            start_pos, end_pos = enc.offsets[hole_idx]
            cases.append(
                make_case(
                    tokenizer,
                    f"arithmetic_{k}_{op}",
                    text,
                    start_pos,
                    end_pos,
                    "paired_arithmetic",
                    f"arith_{k}",
                )
            )

    # 2. 12 Comparison Pairs
    for k in range(12):
        a, b = k + 1, k + 3
        for op, result in [("<", True), (">", False)]:
            prefix = f"def compare_{k}(a, b):\n    return a"
            op_text = f" {op}"
            suffix = f" b\n\nassert compare_{k}({a}, {b}) is {result}\n"
            text = prefix + op_text + suffix
            exec(compile(text, f"<comparison_{k}_{op}>", "exec"), {})
            enc = tokenizer.encode(text, add_special_tokens=False)
            op_char_pos = len(prefix) + op_text.find(op)
            hits = [idx for idx, (ca, cb) in enumerate(enc.offsets) if ca <= op_char_pos < cb]
            assert len(hits) == 1
            hole_idx = hits[0]
            start_pos, end_pos = enc.offsets[hole_idx]
            cases.append(
                make_case(
                    tokenizer,
                    f"comparison_{k}_{op}",
                    text,
                    start_pos,
                    end_pos,
                    "paired_comparison",
                    f"compare_{k}",
                )
            )

    # Link pairs and set alternatives
    pairs: dict[str, list[dict]] = {}
    for c in cases:
        pairs.setdefault(c["pair"], []).append(c)
    for p_id, p_list in pairs.items():
        assert len(p_list) == 2, f"Pair {p_id} has {len(p_list)} members"
        x, y = p_list
        i = x["hole"]
        assert i == y["hole"], f"Mismatched hole position in {p_id}: {i} vs {y['hole']}"
        assert x["ids"][:i] == y["ids"][:i], f"Prefix token divergence in {p_id}"
        assert len(x["ids"]) == len(y["ids"]), f"Length confound in {p_id}"
        assert x["target"] != y["target"], f"Targets identical in {p_id}"
        assert x["ids"][i + 1 :] != y["ids"][i + 1 :], f"Suffix identical in {p_id}"
        x["alternative"] = y["target"]
        y["alternative"] = x["target"]

    # 3. 64 Local Code Reconstructions
    rng = random.Random(20261018)
    items = json.loads(CORPUS_SOURCE.read_text())
    added = 0
    for item in items:
        prefix = item.get("prompt_base", item["prompt"])
        text = prefix + item["ground_truth_solution"] + "\n"
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc.ids) > 512:
            continue
        eligible = []
        for i, (a, b) in enumerate(enc.offsets):
            word = text[a:b].strip()
            if a >= len(prefix) and 0 < i < len(enc.ids) - 8 and word and (
                word.isidentifier() or word in {"+", "-", "*", "%", "<", ">", "=="}
            ):
                eligible.append((a, b))
        if not eligible:
            continue
        a, b = rng.choice(eligible)
        try:
            case = make_case(tokenizer, "local_" + item["id"], text, a, b, "local_code")
            cases.append(case)
            added += 1
        except Exception:
            continue
        if added == 64:
            break

    assert len(cases) == 112, f"Expected 112 cases, got {len(cases)}"
    return cases


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = prepare_cases()
    CASES_FILE.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Successfully generated and validated {len(cases)} cases -> {CASES_FILE}")


if __name__ == "__main__":
    main()
