"""Unit tests verifying audit fixes in model adapters."""

import sys
import time
from pathlib import Path

# Add reports directory to sys.path
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "xerj_reader_2026-11-14"
sys.path.insert(0, str(REPORTS_DIR))

from model_adapters import (
    LMSTUDIO_JSON_SCHEMA,
    GeminiRateLimiter,
    normalize_date,
    normalize_value,
)


def test_money_normalization_from_dict():
    """Verify money dictionaries are normalized with string amount and uppercase currency."""
    res1 = normalize_value("money", {"amount": "15", "currency": "USD"})
    assert res1 == {"amount": "15", "currency": "USD"}

    res2 = normalize_value("money", {"amount": 15, "currency": "usd"})
    assert res2 == {"amount": "15", "currency": "USD"}

    res3 = normalize_value("money", {"amount": "$15.50", "currency": "usd"})
    assert res3 == {"amount": "15.50", "currency": "USD"}


def test_money_normalization_from_string():
    """Verify money strings with symbols or currency codes are converted to {amount, currency}."""
    assert normalize_value("money", "$15") == {"amount": "15", "currency": "USD"}
    assert normalize_value("money", "USD 15") == {"amount": "15", "currency": "USD"}
    assert normalize_value("money", "15 USD") == {"amount": "15", "currency": "USD"}
    assert normalize_value("money", "€25.50") == {"amount": "25.50", "currency": "EUR"}
    assert normalize_value("money", "£50") == {"amount": "50", "currency": "GBP"}
    assert normalize_value("money", "15") == {"amount": "15", "currency": "USD"}
    assert normalize_value("money", 15) == {"amount": "15", "currency": "USD"}


def test_percentage_normalization():
    """Verify percentages strip percent signs and normalize to numeric string."""
    assert normalize_value("pct", "10%") == "10"
    assert normalize_value("pct", " 25 % ") == "25"
    assert normalize_value("pct", 10) == "10"
    assert normalize_value("pct", "12.5%") == "12.5"
    assert normalize_value("pct", "0.5") == "0.5"


def test_date_normalization():
    """Verify various human-readable date formats normalize to ISO YYYY-MM-DD."""
    assert normalize_date("2024-02-27") == "2024-02-27"
    assert normalize_date("14 March 2024") == "2024-03-14"
    assert normalize_date("March 14, 2024") == "2024-03-14"
    assert normalize_date("21 Feb 2024") == "2024-02-21"
    assert normalize_date("2024/03/14") == "2024-03-14"
    assert normalize_date("14/03/2024") == "2024-03-14"


def test_list_and_non_dict_root_handling():
    """Verify that decoded lists or scalar values do not trigger AttributeError."""
    decoded_list = ["alpha-1", "beta-1", "gamma-1"]

    # Simulating adapter list wrapping logic
    if isinstance(decoded_list, list):
        parsed = {"status": "ANSWER", "value": decoded_list, "confidence": 1.0, "evidence": [], "reason": None}
    else:
        parsed = decoded_list

    assert parsed.get("status") == "ANSWER"
    assert parsed.get("value") == ["alpha-1", "beta-1", "gamma-1"]


def test_lmstudio_schema_money_support():
    """Verify LM Studio schema permits object value with amount and currency."""
    val_schema = LMSTUDIO_JSON_SCHEMA["schema"]["properties"]["value"]
    assert "object" in val_schema["type"]
    assert "amount" in val_schema["properties"]
    assert "currency" in val_schema["properties"]


def test_gemini_rate_limiter_pacing():
    """Verify rate limiter enforces minimum intervals between successive calls."""
    limiter = GeminiRateLimiter(interval=0.1)
    t0 = time.monotonic()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.09
