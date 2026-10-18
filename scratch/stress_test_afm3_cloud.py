import subprocess
import time
import concurrent.futures
from pathlib import Path

def single_query(req_id: int, prompt: str) -> dict:
    in_file = Path(f"scratch/_afm_stress_in_{req_id}_{int(time.time()*1000)}.txt")
    out_file = Path(f"scratch/_afm_stress_out_{req_id}_{int(time.time()*1000)}.txt")
    
    in_file.write_text(prompt, encoding="utf-8")
    if out_file.exists():
        out_file.unlink()
        
    t0 = time.time()
    res = subprocess.run(
        ["shortcuts", "run", "AFM3Cloud", "-i", str(in_file), "-o", str(out_file)],
        capture_output=True,
        text=True
    )
    elapsed = time.time() - t0
    
    content = ""
    if out_file.exists():
        content = out_file.read_text(encoding="utf-8").strip()
        out_file.unlink()
    if in_file.exists():
        in_file.unlink()
        
    success = (res.returncode == 0) and (len(content) > 0)
    words = len(content.split())
    
    return {
        "id": req_id,
        "success": success,
        "elapsed": elapsed,
        "words": words,
        "content_snippet": content[:100],
        "returncode": res.returncode,
        "stderr": res.stderr.strip()
    }

print("=" * 70)
print("  AFM 3 CLOUD (PRIVATE CLOUD COMPUTE) RATE LIMIT & STRESS TEST")
print("=" * 70)

# 1. Test Sustained Sequential Requests (10 rapid requests, 0s delay)
print("\n[TEST 1] Sustained Back-to-Back Sequential Requests (10 Iterations)...")
seq_results = []
for i in range(1, 11):
    prompt = f"Give the capital of country #{i} in one single word: France, Germany, Japan, Brazil, India, Canada, Italy, Spain, Australia, Egypt (pick item {i})."
    r = single_query(i, prompt)
    seq_results.append(r)
    status = "✓ OK" if r["success"] else "✗ FAILED"
    print(f"  Req {i:02d}: {status} | Latency: {r['elapsed']:.2f}s | Output: {r['content_snippet']}")
    # 0s delay - test immediate back-to-back dispatch

passed_seq = sum(1 for r in seq_results if r["success"])
print(f"--> Sequential Stress Result: {passed_seq}/10 Succeeded ({passed_seq/10*100:.0f}%)")

# 2. Test Concurrency: 2 Parallel Requests
print("\n[TEST 2] Concurrency Test: 2 Simultaneous Requests...")
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
        executor.submit(single_query, j, f"What is 100 + {j}? Reply with number only.")
        for j in range(1, 3)
    ]
    par2_results = [f.result() for f in futures]
par2_time = time.time() - t0
for r in par2_results:
    st = "✓ OK" if r["success"] else "✗ FAILED"
    print(f"  Thread {r['id']}: {st} | Elapsed: {r['elapsed']:.2f}s | Output: {r['content_snippet']}")
print(f"--> Total Wall-Clock Time for 2 Parallel Requests: {par2_time:.2f}s")

# 3. Test Concurrency: 4 Parallel Requests
print("\n[TEST 3] Concurrency Test: 4 Simultaneous Requests...")
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(single_query, k, f"What is 200 + {k}? Reply with number only.")
        for k in range(1, 5)
    ]
    par4_results = [f.result() for f in futures]
par4_time = time.time() - t0
for r in par4_results:
    st = "✓ OK" if r["success"] else "✗ FAILED"
    print(f"  Thread {r['id']}: {st} | Elapsed: {r['elapsed']:.2f}s | Output: {r['content_snippet']}")
print(f"--> Total Wall-Clock Time for 4 Parallel Requests: {par4_time:.2f}s")

print("\n" + "=" * 70)
print("  RATE LIMIT STRESS TEST COMPLETE")
print("=" * 70)
