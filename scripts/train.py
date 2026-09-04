"""
Unified Training Script for Telos paradigms.
Delegates to the unified telos.train CLI.
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos.train.cli import main

if __name__ == "__main__":
    main()

