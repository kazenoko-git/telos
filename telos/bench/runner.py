"""
Hardware Throughput and Latency Benchmark Engine for Télos.
Measures steps/sec, tokens/sec, step latency percentiles, and unified memory usage.
Enforces strict 5-minute (300-second) upper bound limit.
"""

import sys
import argparse
from pathlib import Path

from telos.train.cli import train


def benchmark(
    paradigm: str = "mdlm",
    params: str | int | None = "12M",
    duration: float = 30.0,
    hardware: str = "auto",
    devices: int | str = "auto",
    effective_batch: int | str | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    synthetic: bool = True,
    **kwargs
):
    """
    Executes a hardware throughput and latency benchmark capped at 5 minutes (300s).
    """
    # Strict 300 second ceiling
    bench_duration = min(float(duration), 300.0)

    print("\n" + "=" * 76)
    print(f"  TÉLOS HARDWARE BENCHMARK SUITE")
    print(f"  Target Paradigm:      {paradigm.upper()}")
    print(f"  Parameter Budget:     {params}")
    print(f"  Hardware Target:      {hardware.upper()}")
    print(f"  Duration Limit:       {bench_duration:.1f} seconds (strict limit: 300.0s / 5.0m)")
    print("=" * 76 + "\n")

    return train(
        paradigm=paradigm,
        params=params,
        hardware=hardware,
        devices=devices,
        effective_batch=effective_batch,
        batch_size=batch_size,
        grad_accum=grad_accum,
        synthetic=synthetic,
        benchmark=True,
        benchmark_duration=bench_duration,
        **kwargs
    )


def main():
    parser = argparse.ArgumentParser(description="Télos Hardware Throughput Benchmark")
    parser.add_argument("--paradigm", type=str, default="mdlm", choices=["ar", "mdlm", "undlm", "corosred"], help="Training paradigm")
    parser.add_argument("--params", type=str, default="12M", help="Target parameter budget (e.g. 12M, 25M, 50M, 100M)")
    parser.add_argument("--duration", type=float, default=15.0, help="Benchmark duration in seconds (capped at 300)")
    parser.add_argument("--hardware", type=str, default="auto", choices=["auto", "mlx", "cuda", "mps", "xla", "cpu"], help="Hardware backend")
    parser.add_argument("--devices", type=str, default="auto", help="Hardware device count (e.g. 1, 4, auto)")
    parser.add_argument("--effective-batch", type=str, default=None, help="Effective batch size")
    parser.add_argument("--batch-size", type=int, default=None, help="Microbatch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file bypass")

    args = parser.parse_args()

    benchmark(
        paradigm=args.paradigm,
        params=args.params,
        duration=args.duration,
        hardware=args.hardware,
        devices=args.devices,
        effective_batch=args.effective_batch,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
