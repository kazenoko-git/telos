"""
Launches the 25M training suite in the background on Colab.
"""
import os
import subprocess
from pathlib import Path

work_dir = Path("/content/telos")
work_dir.mkdir(parents=True, exist_ok=True)
os.chdir(str(work_dir))

log_file = work_dir / "training.log"
runner_script = work_dir / "scripts" / "colab" / "remote_colab_runner.py"

# Clone/pull codebase if needed
if not runner_script.exists():
    subprocess.run(["git", "clone", "https://github.com/kazenoko-git/telos.git", "/content/telos_git"], check=False)
    import shutil
    if Path("/content/telos_git").exists():
        for item in ["mdiff", "undiff", "ar", "scripts", "notebooks", "evals"]:
            src = Path("/content/telos_git") / item
            dst = work_dir / item
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)

# Spawn training process detached
with open(log_file, "a") as f_log:
    f_log.write("\n=== LAUNCHING RETRAINING SESSION ===\n")

proc = subprocess.Popen(
    ["python3", "scripts/colab/remote_colab_runner.py"],
    stdout=open(log_file, "a"),
    stderr=subprocess.STDOUT,
    cwd=str(work_dir),
    env=os.environ.copy()
)

print(f"Background Training Process Spawned with PID: {proc.pid}")
print(f"Logging to: {log_file}")
