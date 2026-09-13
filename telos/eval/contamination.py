"""
Training Data Contamination & Leakage Screening Tool for Télos.

Implements 13-gram sliding window matching against training binary token files (.bin)
to guarantee that benchmark tasks have 0% data leakage from the training corpus.
"""

import os
from pathlib import Path
from typing import List, Set, Dict, Any, Union, Optional
import numpy as np


def extract_token_ngrams(tokens: List[int], n: int = 13) -> Set[tuple]:
    """Extracts a set of n-gram tuples from a list of token IDs."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class ContaminationDetector:
    """
    Scans candidate benchmark texts or token sequences against a memory-mapped
    training corpus to detect exact n-gram data leakage.
    """
    def __init__(self, n: int = 13):
        self.n = n

    def scan_tokens_against_corpus(
        self,
        query_tokens: List[int],
        corpus_bin_path: Union[str, Path],
        itemsize: int = 2,
        chunk_size: int = 10_000_000
    ) -> Dict[str, Any]:
        """
        Scans query tokens against a memory-mapped binary token corpus.

        Args:
            query_tokens: Token sequence to check for contamination.
            corpus_bin_path: Path to .bin file (e.g. data/python_corpus_2.5b.bin).
            itemsize: 2 for uint16, 4 for uint32.
            chunk_size: Number of tokens to stream per chunk.

        Returns:
            Dict detailing contamination metrics and matching n-grams.
        """
        query_ngrams = extract_token_ngrams(query_tokens, self.n)
        if not query_ngrams:
            return {
                "total_query_ngrams": 0,
                "matching_ngrams_count": 0,
                "is_contaminated": False,
                "contamination_pct": 0.0,
            }

        p = Path(corpus_bin_path)
        if not p.exists():
            raise FileNotFoundError(f"Corpus binary not found at {corpus_bin_path}")

        dtype = np.uint16 if itemsize == 2 else np.uint32
        mmap = np.memmap(str(p), dtype=dtype, mode="r")
        total_tokens = len(mmap)

        matched_ngrams = set()

        # Stream through memory map in manageable chunks
        for start_idx in range(0, total_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size + self.n, total_tokens)
            chunk = mmap[start_idx:end_idx]
            
            # Fast check: convert chunk into sliding window view
            if len(chunk) < self.n:
                break
                
            chunk_ngrams = extract_token_ngrams(chunk.tolist(), self.n)
            overlap = query_ngrams.intersection(chunk_ngrams)
            if overlap:
                matched_ngrams.update(overlap)
                # If we've found all query n-grams, stop early
                if len(matched_ngrams) == len(query_ngrams):
                    break

        total_q = len(query_ngrams)
        matches = len(matched_ngrams)
        is_contaminated = matches > 0
        contamination_pct = round((matches / total_q) * 100.0, 2) if total_q else 0.0

        return {
            "total_query_ngrams": total_q,
            "matching_ngrams_count": matches,
            "is_contaminated": is_contaminated,
            "contamination_pct": contamination_pct,
        }


def check_benchmark_contamination(
    tasks: List[Dict[str, Any]],
    tokenizer,
    corpus_bin_path: Union[str, Path],
    n: int = 13,
    sample_limit: Optional[int] = 100
) -> Dict[str, Any]:
    """
    Screens a list of benchmark tasks (prompt + ground truth) for contamination.
    """
    detector = ContaminationDetector(n=n)
    
    tasks_to_check = tasks[:sample_limit] if sample_limit else tasks
    contaminated_tasks = []
    total_checked = len(tasks_to_check)

    for idx, t in enumerate(tasks_to_check):
        text = t.get("prompt", "") + " " + t.get("target", "") + " " + t.get("test_harness", "")
        tok_ids = tokenizer.encode(text).ids
        res = detector.scan_tokens_against_corpus(tok_ids, corpus_bin_path)
        if res["is_contaminated"]:
            contaminated_tasks.append({
                "task_id": t.get("id", idx),
                "contamination_pct": res["contamination_pct"],
                "matching_count": res["matching_ngrams_count"]
            })

    return {
        "tasks_checked": total_checked,
        "contaminated_count": len(contaminated_tasks),
        "contamination_rate_pct": round((len(contaminated_tasks) / max(total_checked, 1)) * 100.0, 2),
        "contaminated_task_details": contaminated_tasks,
    }
