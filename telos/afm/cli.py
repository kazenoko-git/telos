"""
AFM CLI
Subcommands for Apple on-device Foundation Models: status and generate.
"""

from __future__ import annotations

import argparse
import sys

from .availability import AFMUnavailableError, probe
from .bridge import AFMBridge


def _format_version(ver: tuple[int, ...] | None) -> str:
    return ".".join(map(str, ver)) if ver else "unknown"


def cmd_status(args: argparse.Namespace) -> int:
    """Prints the availability report. Exits 0 when usable, 1 when not."""
    avail = probe(runtime=not args.static_only)

    rows = [
        ("Available", "yes" if avail.available else "no"),
        ("Model", avail.model or "-"),
        ("Fallback model", avail.fallback_model or "-"),
        ("Reason", avail.reason),
        ("macOS", _format_version(avail.macos_version)),
        ("Chip family", avail.chip_family or "unknown"),
        ("FoundationModels check", avail.framework_check or "-"),
        ("Xcode Command Line Tools", "yes" if avail.xcode_clt else "no"),
        ("swiftc", "yes" if avail.swiftc else "no"),
    ]

    width = max(len(label) for label, _ in rows)
    print("Apple on-device Foundation Models")
    print("-" * (width + 2 + 40))
    for label, value in rows:
        print(f"  {label.ljust(width)}  {value}")

    if not avail.available and not avail.swiftc:
        print(
            "\nHint: the on-device models are reached through a small Swift bridge.\n"
            "      Install the Xcode Command Line Tools with `xcode-select --install`."
        )

    return 0 if avail.available else 1


def cmd_generate(args: argparse.Namespace) -> int:
    """Runs a single generation, printing only the completion to stdout."""
    try:
        with AFMBridge() as bridge:
            text = bridge.generate(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                stop=args.stop,
                instructions=args.instructions,
            )
    except AFMUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(text)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="telos afm",
        description="Access Apple's on-device Foundation Models from Python.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser(
        "status",
        help="Report whether the on-device model is usable on this machine.",
    )
    p_status.add_argument(
        "--static-only",
        action="store_true",
        help="Skip the runtime framework check (no Swift compile) and inspect only the platform.",
    )
    p_status.set_defaults(func=cmd_status)

    p_gen = sub.add_parser("generate", help="Run a single prompt through the on-device model.")
    p_gen.add_argument("prompt", help="The prompt to send.")
    p_gen.add_argument("--max-tokens", type=int, default=256, help="Response token ceiling (default: 256).")
    p_gen.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0, greedy).")
    p_gen.add_argument("--instructions", default=None, help="System instructions for the session.")
    p_gen.add_argument("--stop", nargs="*", default=None, help="Stop sequences trimmed from the response tail.")
    p_gen.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    sys.exit(args.func(args))
