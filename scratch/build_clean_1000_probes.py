"""
Script to build and verify 1,000 authentic contextual probes (125 per category across 8 categories).
Ensures 100% alignment with configs/tokenizer_mac.json.
"""

import json
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "configs" / "tokenizer_mac.json"
OUTPUT_PATH = ROOT / "evals" / "benchmarks" / "contextual_probes_1000.json"
TARGET_SCRIPT = ROOT / "scripts" / "benchmark_generators" / "build_authentic_probes_1000.py"

tok = Tokenizer.from_file(str(TOKENIZER_PATH))
vocab = tok.get_vocab()

def check(prompt: str, target: str, target_bpe: str):
    t_id = tok.token_to_id(target_bpe)
    if t_id is None:
        return False, f"target_bpe '{target_bpe}' not in vocab"
    sep = " " if (target_bpe.startswith("Ġ") and not prompt.endswith(" ")) else ""
    full = prompt + sep + target
    p_ids = tok.encode(prompt).ids
    f_ids = tok.encode(full).ids
    if len(f_ids) != len(p_ids) + 1:
        return False, f"Length mismatch: p={len(p_ids)}, full={len(f_ids)}"
    if f_ids[len(p_ids)] != t_id:
        return False, f"ID mismatch: got {f_ids[len(p_ids)]} vs {t_id}"
    return True, "OK"
