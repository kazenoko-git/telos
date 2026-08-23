import os
import glob
import subprocess
from pathlib import Path

print("=" * 60)
print("PROCESS STATUS (ps aux | grep python):")
print("=" * 60)
ps_out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
for line in ps_out.splitlines():
    if "python" in line or "colab" in line or "train" in line:
        print(line)

print("\n" + "=" * 60)
print("CHECKPOINT FILES ON DISK (/content):")
print("=" * 60)
ckpts = glob.glob("/content/**/checkpoints/**/*.safetensors", recursive=True) + glob.glob("./checkpoints/**/*.safetensors", recursive=True)
if ckpts:
    for c in ckpts:
        size_mb = os.path.getsize(c) / (1024 * 1024)
        print(f"  {c} ({size_mb:.2f} MB)")
else:
    print("  No .safetensors files written to local disk yet.")

print("\n" + "=" * 60)
print("TRAINING LOG FILES:")
print("=" * 60)
for log_name in ["/content/telos/training.log", "/content/training.log", "training.log"]:
    p = Path(log_name)
    if p.exists():
        print(f"\n--- {log_name} (Last 30 lines) ---")
        lines = p.read_text().splitlines()
        print("\n".join(lines[-30:]))
