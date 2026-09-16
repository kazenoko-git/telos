"""
Statistical Utilities for Télos Model Evaluation.

Provides:
- bootstrap_confidence_interval: Empirical percentile bootstrap confidence intervals
  for binary classification and pass-fail benchmark outcomes.
"""

from typing import List, Tuple
import numpy as np


def bootstrap_confidence_interval(
    binary_outcomes: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Computes empirical bootstrap confidence interval for binary outcomes.
    
    Args:
        binary_outcomes: List of binary indicators (0.0 or 1.0).
        n_resamples: Number of bootstrap iterations (default: 1000).
        confidence_level: Confidence interval coverage (default: 0.95 for 95% CI).
        
    Returns:
        (ci_lower_pct, ci_upper_pct): Confidence interval bounds as percentages rounded to 2 decimals.
    """
    if not binary_outcomes:
        return 0.0, 0.0
    arr = np.array(binary_outcomes, dtype=np.float32)
    n = len(arr)
    if n <= 1:
        val = float(arr[0] * 100.0) if n == 1 else 0.0
        return val, val

    np.random.seed(42)
    # Resample with replacement to form empirical distribution of the mean
    resample_means = [
        float(np.mean(np.random.choice(arr, size=n, replace=True)))
        for _ in range(n_resamples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(resample_means, alpha * 100.0)) * 100.0
    upper = float(np.percentile(resample_means, (1.0 - alpha) * 100.0)) * 100.0
    return round(lower, 2), round(upper, 2)
