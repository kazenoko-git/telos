"""
Re-evaluates Apple AFM 3 Core Advanced on Cybersecurity with 1,024-token budget and patched criteria.
"""
import json
import time
from pathlib import Path
from telos.eval.runner import evaluate

LOGS_DIR = Path("logs")
out_file = LOGS_DIR / "eval_report_afm3_cyber_full.json"

print("\n" + "=" * 80)
print("  RE-RUNNING CYBERSECURITY (50 TASKS) ON APPLE AFM 3 CORE ADVANCED")
print("  Token Budget: 1,024 tokens | Patched Security Criteria")
print("=" * 80 + "\n", flush=True)

t0 = time.time()
report = evaluate(
    checkpoint="afm-3-core-advanced",
    mode="functional",
    suite="cyber",
    benchmark_type="cyber",
    max_tasks=None,
    backend="swift_afm",
    max_new_tokens=1024,
    output_path=str(out_file)
)
elapsed = time.time() - t0
data = report["cyber"]["functional"]
print(f"\n✓ Completed AFM 3 Cybersecurity in {elapsed/60.0:.2f} mins. Pass@1: {data['pass_at_1_pct']}%\n", flush=True)
