"""
Master CLI Dispatcher for Télos.
Routes subcommands:
  telos dataprep ...  -> High-efficiency data preparation
  telos train ...     -> Zero-config dimensional model training
  telos eval ...      -> 100 contextual probes and model evaluation
  telos bench ...     -> 5-minute hardware throughput benchmark
  telos test ...      -> Unified test suite & model verification
"""

import sys
import argparse


def print_banner():
    banner = """
  ╔═══════════════════════════════════════════════════════════════╗
  ║   TÉLOS (τέλος) — Discrete Diffusion & Autoregressive LLM    ║
  ╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print_banner()
        print("Usage: telos <command> [options]\n")
        print("Available Commands:")
        print("  dataprep   Prepare raw text, code directories, JSONL, or HF datasets into binary token stream (.bin)")
        print("  train      Unified zero-config model trainer (AR, MDLM, UNDLM, COROSred, custom)")
        print("  eval       High-end evaluation suite (100 contextual probes, code sampling, perplexity)")
        print("  bench      Hardware throughput & step latency benchmark (strictly capped at 5 minutes)")
        print("  test       Run unified verification suite across model contracts, causality, and samplers")
        print("\nRun 'telos <command> --help' for detailed options on any command.\n")
        sys.exit(0)

    cmd = sys.argv[1].lower()
    # Strip the subcommand from argv so the sub-parser parses subsequent flags cleanly
    sys.argv.pop(1)

    if cmd == "train":
        from telos.train.cli import main as train_main
        train_main()
    elif cmd in ["dataprep", "prep"]:
        from telos.dataprep.prepare import main as prep_main
        prep_main()
    elif cmd in ["eval", "evaluate"]:
        from telos.eval.runner import main as eval_main
        eval_main()
    elif cmd in ["bench", "benchmark"]:
        from telos.bench.runner import main as bench_main
        bench_main()
    elif cmd in ["test", "verify"]:
        from telos.testing.suite import main as test_main
        test_main()
    else:
        print(f"Error: Unknown command '{cmd}'. Available: dataprep, train, eval, bench, test.")
        sys.exit(1)


if __name__ == "__main__":
    main()
