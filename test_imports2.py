import sys
import importlib
import traceback

sys.modules["torch"] = None

try:
    import telos.testing
except Exception as e:
    traceback.print_exc()
