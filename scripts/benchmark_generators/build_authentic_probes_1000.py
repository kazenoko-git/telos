"""
Builds and verifies the authentic 1,000 contextual probes suite for Télos.

Contains 125 distinct probes per category across 8 categories:
1. Algorithms & Numerical
2. Data Structures & Collections
3. String & Text Processing
4. Object-Oriented Programming
5. Control Flow & Loops
6. Built-ins & Iteration
7. Exception Handling & Context Managers
8. Imports, Typing & Signatures

Guarantees 100% single-token vocabulary alignment with configs/tokenizer_mac.json.
"""

import json
from pathlib import Path
from tokenizers import Tokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = ROOT_DIR / "configs" / "tokenizer_mac.json"
OUTPUT_PATH = ROOT_DIR / "evals" / "benchmarks" / "contextual_probes_1000.json"


def verify_all_probes() -> bool:
    """Verifies all 1,000 contextual probes against the BPE tokenizer."""
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Contextual probes benchmark not found at {OUTPUT_PATH}")

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    with open(OUTPUT_PATH, "r") as f:
        probes = json.load(f)

    if len(probes) != 1000:
        raise ValueError(f"Expected 1,000 probes, found {len(probes)}")

    category_counts = {}
    for i, p in enumerate(probes):
        cat = p.get("category", "General")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        prompt = p["prompt"]
        target = p["target"]
        tbpe = p["target_bpe"]

        t_id = tok.token_to_id(tbpe)
        if t_id is None:
            raise ValueError(f"Probe {i} ({cat}): target_bpe '{tbpe}' not in tokenizer vocabulary")

        sep = " " if (tbpe.startswith("Ġ") and not prompt.endswith(" ")) else ""
        full = prompt + sep + target
        p_ids = tok.encode(prompt).ids
        f_ids = tok.encode(full).ids

        if len(f_ids) != len(p_ids) + 1:
            raise ValueError(f"Probe {i} ({cat}): length mismatch: len(prompt)={len(p_ids)}, len(full)={len(f_ids)}")
        if f_ids[len(p_ids)] != t_id:
            raise ValueError(f"Probe {i} ({cat}): token ID mismatch: expected {t_id} ({tbpe}), got {f_ids[len(p_ids)]}")

    for cat, count in category_counts.items():
        if count != 125:
            raise ValueError(f"Category '{cat}' has {count} probes (expected 125)")

    print(f"✓ Successfully verified {len(probes)} contextual probes across {len(category_counts)} categories.")
    return True


if __name__ == "__main__":
    verify_all_probes()
