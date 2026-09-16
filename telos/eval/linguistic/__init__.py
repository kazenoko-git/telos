"""
Télos English Linguistic Evaluation Suite.

Exports:
- evaluate_linguistic: Master evaluation function for English linguistic benchmarks.
- load_english_probes: Loads deterministic 100-probe English benchmark dataset.
- ENGLISH_PROBES_100: Canonical 100 deterministic English linguistic probes.
"""

from .probes_en import load_english_probes, ENGLISH_PROBES_100
from .evaluator import evaluate_linguistic, evaluate_english_perplexity, evaluate_english_sample

__all__ = [
    "evaluate_linguistic",
    "load_english_probes",
    "ENGLISH_PROBES_100",
    "evaluate_english_perplexity",
    "evaluate_english_sample",
]
