"""
Analytical transformer geometry solver and unit parsers for Télos.
Resolves human-readable target parameter counts into optimal, balanced
transformer dimensions (d_model, n_layers, n_heads) adhering to standard scaling laws.
"""

import math
import re
from telos.models.param_counter import count_parameters
from telos.models.config import TelosConfig


def parse_human_number(val: str | int | float) -> int:
    """
    Parses human-readable numbers with standard SI suffixes into integers.
    
    Examples:
        '12M'  -> 12_000_000
        '2.5B' -> 2_500_000_000
        '32k'  -> 32_000
        '500'  -> 500
        250000 -> 250000
    """
    if isinstance(val, (int, float)):
        return int(val)
        
    s = str(val).strip().upper().replace(",", "")
    match = re.match(r"^([\d.]+)\s*([KMBT]?)$", s)
    if not match:
        raise ValueError(f"Cannot parse number from string: '{val}'")
        
    num_str, suffix = match.groups()
    num = float(num_str)
    
    multipliers = {
        "": 1,
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
        "T": 1_000_000_000_000,
    }
    return int(num * multipliers[suffix])


CANONICAL_TIERS = {
    "12M": {"d_model": 256, "n_layers": 8, "n_heads": 4, "n_kv_heads": 4},
    "25M": {"d_model": 384, "n_layers": 10, "n_heads": 6, "n_kv_heads": 6},
    "50M": {"d_model": 512, "n_layers": 14, "n_heads": 8, "n_kv_heads": 8},
    "100M": {"d_model": 768, "n_layers": 14, "n_heads": 12, "n_kv_heads": 12},
    "300M": {"d_model": 1024, "n_layers": 23, "n_heads": 16, "n_kv_heads": 16},
}


def solve_transformer_geometry(
    target_params: int | str,
    vocab_size: int = 8192,
    tied_embeddings: bool = True
) -> dict:
    """
    Finds the optimal transformer dimensions (d_model, n_layers, n_heads, n_kv_heads)
    that closest match a target parameter budget while maintaining optimal aspect ratios.
    
    Aspect ratio constraints:
    - head_dim = 64 (standard for modern LLMs)
    - n_heads = d_model // 64
    - d_model is a multiple of 64 (and multiple of 128 for TPU MXU systolic alignment)
    - n_layers typically ranges between 6 and 32 for models under 1B params
    """
    # Check canonical hardware-aligned tiers first
    target_key = str(target_params).strip().upper()
    if target_key in CANONICAL_TIERS and vocab_size == 8192 and tied_embeddings:
        tier = CANONICAL_TIERS[target_key]
        cfg = TelosConfig(
            vocab_size=vocab_size,
            d_model=tier["d_model"],
            n_layers=tier["n_layers"],
            n_heads=tier["n_heads"],
            n_kv_heads=tier["n_kv_heads"],
            tied_embeddings=tied_embeddings
        )
        actual_params = count_parameters(cfg)["total"]
        target = parse_human_number(target_params)
        return {
            "d_model": tier["d_model"],
            "n_layers": tier["n_layers"],
            "n_heads": tier["n_heads"],
            "n_kv_heads": tier["n_kv_heads"],
            "vocab_size": vocab_size,
            "tied_embeddings": tied_embeddings,
            "actual_params": actual_params,
            "target_params": target,
        }

    target = parse_human_number(target_params)
    
    # Candidate hidden dimensions (multiples of 64 from 128 to 2048)
    d_model_candidates = [128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 2048]
    
    best_candidate = None
    min_diff = float("inf")
    
    for d in d_model_candidates:
        # Standard head_dim = 64
        n_heads = max(2, d // 64)
        if d % n_heads != 0:
            continue
            
        # Explore reasonable layer depths
        for layers in range(4, 36):
            cfg = TelosConfig(
                vocab_size=vocab_size,
                d_model=d,
                n_layers=layers,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                tied_embeddings=tied_embeddings
            )
            actual_params = count_parameters(cfg)["total"]
            diff = abs(actual_params - target)
            
            # Prioritize matching target closely with balanced depth/width ratio
            aspect_ratio = layers / (d / 64)
            penalty = 0.0
            if aspect_ratio < 0.8 or aspect_ratio > 3.0:
                penalty = diff * 0.2  # Slight penalty for extreme aspect ratios
                
            score = diff + penalty
            if score < min_diff:
                min_diff = score
                best_candidate = {
                    "d_model": d,
                    "n_layers": layers,
                    "n_heads": n_heads,
                    "n_kv_heads": n_heads,
                    "vocab_size": vocab_size,
                    "tied_embeddings": tied_embeddings,
                    "actual_params": actual_params,
                    "target_params": target,
                }
                
    return best_candidate
