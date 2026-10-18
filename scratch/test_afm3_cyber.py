import json
import subprocess
from telos.eval.cyber_eval import evaluate_cybersecurity_challenge

bridge = "telos/eval/adapters/swift_afm/afm_bridge"
proc = subprocess.Popen(
    [bridge],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

with open("evals/benchmarks/cybersecurity_suite.json") as f:
    suite = json.load(f)

for task_idx in [0, 1, 2]:
    task = suite[task_idx]
    req = {
        "id": task_idx,
        "prompt": task["prompt"],
        "instructions": "You are an expert Application Security Engineer.",
        "maxTokens": 512,
        "temperature": 0.0
    }
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    resp = json.loads(line)
    content = resp.get("content", "")
    eval_res = evaluate_cybersecurity_challenge(content, task)
    print(f"=== Task {task['id']} ===")
    print("CWE Target:", task["evaluation_criteria"]["target_cwe"])
    print("Eval Result:", eval_res)
    print("AFM3 Content Snippet:\n", content[:300])
    print()

proc.terminate()
