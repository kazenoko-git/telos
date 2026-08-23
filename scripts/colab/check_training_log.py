"""
Reads and prints the latest lines from Colab training log.
"""
from pathlib import Path

log_file = Path("/content/telos/training.log")
if log_file.exists():
    lines = log_file.read_text().splitlines()
    print("\n".join(lines[-40:]))
else:
    print("Training log not found yet.")
