"""
Generates 1,000 verified, deterministic, authentic contextual probes for Télos.
Ensures:
- Exact token alignment: len(full_ids) == len(p_ids) + 1 and full_ids[len(p_ids)] == target_token_id
- 125 distinct probes per category across 8 categories = 1000 probes total
- Clean Python code context (no dummy `# var_` comments)
- Tested with configs/tokenizer_mac.json
"""

import json
from pathlib import Path
from tokenizers import Tokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT_DIR / "configs" / "tokenizer_mac.json"
OUTPUT_PATH = ROOT_DIR / "evals" / "benchmarks" / "contextual_probes_1000.json"
SCRIPT_TARGET = ROOT_DIR / "scripts" / "benchmark_generators" / "build_authentic_probes_1000.py"

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))


def verify_probe(prompt: str, target: str, target_bpe: str):
    t_id = tokenizer.token_to_id(target_bpe)
    if t_id is None:
        return False, f"target_bpe '{target_bpe}' not in vocab"
    sep = " " if (target_bpe.startswith("Ġ") and not prompt.endswith(" ")) else ""
    full_text = prompt + sep + target
    p_ids = tokenizer.encode(prompt).ids
    full_ids = tokenizer.encode(full_text).ids
    if len(full_ids) != len(p_ids) + 1:
        return False, f"Length mismatch: full={len(full_ids)}, p={len(p_ids)}"
    if full_ids[len(p_ids)] != t_id:
        actual_tok = tokenizer.decode([full_ids[len(p_ids)]])
        return False, f"Token mismatch: expected {target_bpe} (ID {t_id}), got ID {full_ids[len(p_ids)]} ('{actual_tok}')"
    return True, t_id
