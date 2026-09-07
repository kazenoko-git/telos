"""
Unified Training Script for Telos paradigms.
Delegates to the unified telos.train CLI.
Supports zero-argument invocation for 15M COROSRED Phase B retraining on the dense 2.5B corpus.
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos.train.cli import main

if __name__ == "__main__":
    # If invoked without CLI arguments, supply default 15M COROSRED Phase B retrain configuration
    # using the dense, unpadded 2.5B token corpus with eager evaluation for memory stability.
    if len(sys.argv) == 1:
        dense_corpus = "data/python_corpus_2.5b.bin" if Path("data/python_corpus_2.5b.bin").exists() else "data/corpus.bin"
        sys.argv.extend([
            "--paradigm", "corosred",
            "--phase", "B",
            "--params", "15m",
            "--tokens", "41M",
            "--hardware", "mlx",
            "--batch-size", "16",
            "--grad-accum", "24",
            "--eval-policy", "eager",
            "--data", dense_corpus,
            "--init-checkpoint", "checkpoints/corosred/15m/phase_a/checkpoint_final.pt",
            "--checkpoint-dir", "checkpoints/corosred/15m/phase_b",
            "--self-condition",
            "--self-cond-prob", "0.5",
        ])
    main()

