"""
telos.afm
Python access to Apple's on-device Foundation Models.

Usage:
    >>> import telos.afm
    >>> telos.afm.probe().describe()
    >>> telos.afm.generate("Write a haiku about gradients.")

Requires macOS 26+, an Apple Intelligence-eligible Apple Silicon Mac with it
enabled, and the Xcode Command Line Tools. The Swift bridge is built on first
use into ~/.cache/telos (override with TELOS_CACHE_DIR).
"""

from __future__ import annotations

from typing import Optional, Sequence

from .availability import (
    AFMAvailability,
    AFMUnavailableError,
    FALLBACK_CHIP_FAMILY,
    MIN_MACOS_MAJOR,
    MODEL_CORE,
    MODEL_CORE_ADVANCED,
    probe,
    require,
)
from .bridge import AFMBridge, cache_root, ensure_bridge_binary

__all__ = [
    "AFMAvailability",
    "AFMBridge",
    "AFMUnavailableError",
    "FALLBACK_CHIP_FAMILY",
    "MIN_MACOS_MAJOR",
    "MODEL_CORE",
    "MODEL_CORE_ADVANCED",
    "cache_root",
    "ensure_bridge_binary",
    "generate",
    "probe",
    "require",
]


def generate(
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    stop: Optional[Sequence[str]] = None,
    instructions: Optional[str] = None,
) -> str:
    """Generates one completion on the on-device model.

    Opens a bridge, generates, and closes it. Hold an AFMBridge open for
    repeated calls so the process and model stay warm.
    """
    with AFMBridge() as bridge:
        return bridge.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop,
            instructions=instructions,
        )
