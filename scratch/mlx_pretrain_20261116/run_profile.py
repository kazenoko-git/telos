"""Run the ordinary bounded real-corpus profiler with an optional dispatch probe."""
import runpy
import sys
from pathlib import Path
import pipeline

if '--pipeline' in sys.argv:
    sys.argv.remove('--pipeline')
    pipeline.install()
try:
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/profile_mlx_training.py'),run_name='__main__')
finally:
    pipeline.uninstall()
