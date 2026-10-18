"""
Generates 1,000 verified, deterministic, authentic contextual probes for Télos.
8 categories x 125 probes = 1,000 probes.
Every single probe is strictly validated against configs/tokenizer_mac.json.
"""

import json
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "configs" / "tokenizer_mac.json"
OUTPUT_PATH = ROOT / "evals" / "benchmarks" / "contextual_probes_1000.json"
SCRIPT_TARGET = ROOT / "scripts" / "benchmark_generators" / "build_authentic_probes_1000.py"

tok = Tokenizer.from_file(str(TOKENIZER_PATH))


def verify(prompt: str, target: str, target_bpe: str):
    t_id = tok.token_to_id(target_bpe)
    if t_id is None:
        return False, f"target_bpe '{target_bpe}' not in vocab"
    sep = " " if (target_bpe.startswith("Ġ") and not prompt.endswith(" ")) else ""
    full_text = prompt + sep + target
    p_ids = tok.encode(prompt).ids
    full_ids = tok.encode(full_text).ids
    if len(full_ids) != len(p_ids) + 1:
        return False, f"Length mismatch: full={len(full_ids)}, p={len(p_ids)}"
    if full_ids[len(p_ids)] != t_id:
        actual_tok = tok.decode([full_ids[len(p_ids)]])
        return False, f"Token mismatch: expected {target_bpe} (ID {t_id}), got ID {full_ids[len(p_ids)]} ('{actual_tok}')"
    return True, t_id


def add_probe(probe_list, prompt, target, target_bpe, suffix="", mode="both", desc=""):
    ok, res = verify(prompt, target, target_bpe)
    if not ok:
        raise ValueError(f"Probe verification failed: {res} | prompt={repr(prompt)}, target={repr(target)}, tbpe={repr(target_bpe)}")
    probe_list.append({
        "prompt": prompt,
        "prefix": prompt,
        "target": target,
        "target_bpe": target_bpe,
        "suffix": suffix,
        "mode": mode,
        "description": desc
    })
