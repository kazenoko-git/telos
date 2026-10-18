# Replace the earlier rank-API guard with this block at the END of Cell 1.
# Applies the same fix on every saved-version execution after pip installs the wheel.
import hashlib
from importlib.metadata import distribution

trainer_path = Path(distribution("telos-ml").locate_file(
    "telos/training/trainer_pytorch.py"
))
source = trainer_path.read_text(encoding="utf-8")
old = """                try:
                    import torch_xla.core.xla_model as xm
                    self.rank = xm.get_ordinal()
                except Exception:
                    self.rank = 0"""
new = """                # Rank API compatibility: never silently treat every replica as rank zero.
                import torch_xla.runtime as xr
                self.rank = int(xr.global_ordinal())"""

if old in source:
    if source.count(old) != 1:
        raise RuntimeError("Unexpected trainer layout; rank patch not applied.")
    trainer_path.write_text(source.replace(old, new, 1), encoding="utf-8")
elif new not in source:
    raise RuntimeError("Unexpected trainer version; rank patch not applied.")

runtime["compatibility_patch"] = "XLA rank via torch_xla.runtime.global_ordinal"
runtime["trainer_sha256"] = hashlib.sha256(trainer_path.read_bytes()).hexdigest()
(WORK / "runtime.json").write_text(json.dumps(runtime, indent=2))

# Separate process keeps the notebook kernel from initializing the TPU runtime.
subprocess.run([
    sys.executable, "-c",
    "import torch_xla.runtime as xr; "
    "assert callable(getattr(xr, 'global_ordinal', None)), 'Unsupported XLA runtime'; "
    "print('XLA rank compatibility fix ready.')"
], check=True)
