"""
AFM Availability
Detects whether Apple's on-device Foundation Models are usable on this machine.

FoundationModels exposes only `SystemLanguageModel.default`, so this detects
eligibility and names what is called rather than switching models at runtime.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from .bridge import AFMUnavailableError, query_status

__all__ = [
    "AFMAvailability",
    "AFMUnavailableError",
    "MIN_MACOS_MAJOR",
    "FALLBACK_CHIP_FAMILY",
    "probe",
    "require",
]

MIN_MACOS_MAJOR = 26
FALLBACK_CHIP_FAMILY = "M3"

MODEL_CORE_ADVANCED = "afm-3-core-advanced"
MODEL_CORE = "afm-3-core"


@dataclass(frozen=True)
class AFMAvailability:
    """Outcome of an on-device Foundation Models probe."""

    available: bool
    #: Name of the model being called, or None when unavailable.
    model: Optional[str]
    #: Smaller model offered as fallback on M3+, or None on older silicon.
    fallback_model: Optional[str]
    #: "ok" when available, otherwise why not.
    reason: str
    macos_version: Optional[tuple[int, ...]]
    chip_family: Optional[str]
    framework_check: Optional[str]
    xcode_clt: bool
    swiftc: bool

    def describe(self) -> str:
        """One-line human summary."""
        if self.available:
            line = f"{self.model} available (Apple on-device Foundation Models)."
            if self.fallback_model:
                line += f" Fallback on this chip: {self.fallback_model}."
            return line
        return f"Apple on-device Foundation Models unavailable: {self.reason}"


def _parse_macos_major(ver: str) -> Optional[tuple[int, ...]]:
    parts = []
    for chunk in ver.split("."):
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts) if parts else None


def _chip_family() -> Optional[str]:
    # sysctl can be blocked by a sandbox; unknown is fine, runtime check decides.
    try:
        res = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    match = re.search(r"\bApple\s+(M\d+)", res.stdout)
    return match.group(1) if match else None


def _chip_at_least(family: Optional[str], minimum: str) -> bool:
    if family is None:
        return False
    m = re.fullmatch(r"M(\d+)", family)
    n = re.fullmatch(r"M(\d+)", minimum)
    if not m or not n:
        return False
    return int(m.group(1)) >= int(n.group(1))


def _has_xcode_clt() -> bool:
    try:
        res = subprocess.run(["xcode-select", "-p"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0


def probe(*, runtime: bool = True) -> AFMAvailability:
    """Detects whether the on-device models are usable. Never raises for an
    unsupported machine; inspect `available` and `reason` instead."""
    macos_ver = _parse_macos_major(platform.mac_ver()[0])
    chip = _chip_family()
    has_clt = _has_xcode_clt()
    has_swiftc = shutil.which("swiftc") is not None

    def unavailable(reason: str, framework_check: Optional[str] = None) -> AFMAvailability:
        return AFMAvailability(
            available=False,
            model=None,
            fallback_model=None,
            reason=reason,
            macos_version=macos_ver,
            chip_family=chip,
            framework_check=framework_check,
            xcode_clt=has_clt,
            swiftc=has_swiftc,
        )

    if platform.system() != "Darwin":
        return unavailable(f"Apple Foundation Models are macOS-only (this is {platform.system()}).")
    if platform.machine() != "arm64":
        return unavailable(f"Apple Foundation Models require Apple Silicon (this is {platform.machine()}).")
    if macos_ver is None:
        return unavailable("Could not determine the macOS version.")
    if macos_ver[0] < MIN_MACOS_MAJOR:
        return unavailable(
            f"Apple Foundation Models require macOS {MIN_MACOS_MAJOR} or later "
            f"(this is {'.'.join(map(str, macos_ver))})."
        )

    fallback = MODEL_CORE if _chip_at_least(chip, FALLBACK_CHIP_FAMILY) else None

    if not runtime:
        # Platform can host the framework; Apple Intelligence state is runtime-only.
        return AFMAvailability(
            available=True,
            model=MODEL_CORE_ADVANCED,
            fallback_model=fallback,
            reason="ok",
            macos_version=macos_ver,
            chip_family=chip,
            framework_check=None,
            xcode_clt=has_clt,
            swiftc=has_swiftc,
        )

    try:
        status = query_status()
    except AFMUnavailableError as exc:
        return unavailable(exc.reason)

    check = str(status.get("check", "unknown"))
    if not status.get("available"):
        detail = status.get("detail") or None
        return unavailable(detail or f"The on-device model is unavailable ({check}).", check)

    return AFMAvailability(
        available=True,
        model=MODEL_CORE_ADVANCED,
        fallback_model=fallback,
        reason="ok",
        macos_version=macos_ver,
        chip_family=chip,
        framework_check=check,
        xcode_clt=has_clt,
        swiftc=has_swiftc,
    )


def require(*, runtime: bool = True) -> AFMAvailability:
    """Like probe(), but raises when the models are unavailable."""
    avail = probe(runtime=runtime)
    if not avail.available:
        raise AFMUnavailableError(avail.reason)
    return avail
