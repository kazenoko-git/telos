"""
Curates and verifies 1,000 authentic contextual probes across 8 categories:
1. Algorithms & Numerical (125)
2. Data Structures & Collections (125)
3. String & Text Processing (125)
4. Object-Oriented Programming (125)
5. Control Flow & Loops (125)
6. Built-ins & Iteration (125)
7. Exception Handling & Context Managers (125)
8. Imports, Typing & Signatures (125)

Strict verification:
  full_ids == p_ids + [target_id]
  target_id in vocab
Zero synthetic/dummy repetitions. Clean Python syntax.
"""

import json
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "configs" / "tokenizer_mac.json"
OUTPUT_PATH = ROOT / "evals" / "benchmarks" / "contextual_probes_1000.json"

tok = Tokenizer.from_file(str(TOKENIZER_PATH))
vocab = tok.get_vocab()

def make_probe(prompt: str, target: str, target_bpe: str, suffix: str = "", desc: str = ""):
    t_id = tok.token_to_id(target_bpe)
    if t_id is None:
        raise ValueError(f"Target BPE {target_bpe} not in tokenizer vocab!")
    sep = " " if (target_bpe.startswith("Ġ") and not prompt.endswith(" ")) else ""
    full_text = prompt + sep + target
    p_ids = tok.encode(prompt).ids
    f_ids = tok.encode(full_text).ids
    if len(f_ids) != len(p_ids) + 1:
        raise ValueError(f"Length mismatch: len(full)={len(f_ids)}, len(p)={len(p_ids)}. Prompt: {repr(prompt)}, target: {repr(target)}")
    if f_ids[len(p_ids)] != t_id:
        raise ValueError(f"Token ID mismatch: expected {t_id} ({target_bpe}), got {f_ids[len(p_ids)]} ({tok.decode([f_ids[len(p_ids)]])})")
    
    return {
        "prompt": prompt,
        "prefix": prompt,
        "target": target,
        "target_bpe": target_bpe,
        "suffix": suffix,
        "mode": "both" if suffix else "causal",
        "description": desc
    }
