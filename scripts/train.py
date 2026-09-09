"""
Unified Training Script for Télos paradigms.
Delegates directly to the unified telos.train CLI.
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos.train.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python scripts/train.py --paradigm <ar|mdlm|undlm|corosred> --params <50M|100M> ...")
        print("Run 'python scripts/train.py --help' for full dimensional options.")
        sys.exit(0)
    main()

