import subprocess
import time
import os
from pathlib import Path

def query_afm_cloud(prompt: str) -> dict:
    in_file = Path("scratch/_temp_afm_in.txt")
    out_file = Path("scratch/_temp_afm_out.txt")
    
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
        
    # Clean temporary files
    if in_file.exists(): in_file.unlink()
    if out_file.exists(): out_file.unlink()
    
    words = len(content.split())
    approx_tokens = int(words * 1.33)
    tok_per_sec = approx_tokens / elapsed if elapsed > 0 else 0
    
    return {
        "content": content,
        "elapsed": elapsed,
        "words": words,
        "approx_tokens": approx_tokens,
        "tok_per_sec": tok_per_sec,
        "returncode": res.returncode,
        "stderr": res.stderr
    }

if __name__ == "__main__":
    print("Testing AFM 3 Cloud Single-Turn Throughput...")
test_prompt = "Explain quantum entanglement in 150 words with scientific precision."
result = query_afm_cloud(test_prompt)
print(f"Elapsed Time:  {result['elapsed']:.2f} seconds")
print(f"Word Count:    {result['words']} words (~{result['approx_tokens']} tokens)")
print(f"Throughput:    {result['tok_per_sec']:.2f} tokens/sec")
print("\n=== Model Response ===")
print(result['content'])
print("======================\n")

print("Running 5-request burst test to measure rate limit and sustained speed...")
times = []
tok_speeds = []
for i in range(5):
    p = f"Write a haiku about computing and algorithms, poem #{i+1}."
    r = query_afm_cloud(p)
    times.append(r['elapsed'])
    tok_speeds.append(r['tok_per_sec'])
    print(f"  Request {i+1}: {r['elapsed']:.2f}s | {r['approx_tokens']} tokens | {r['tok_per_sec']:.1f} tok/s")
    time.sleep(0.5)

avg_lat = sum(times) / len(times)
avg_speed = sum(tok_speeds) / len(tok_speeds)
print(f"\nAverage Latency: {avg_lat:.2f}s")
print(f"Average Throughput: {avg_speed:.1f} tokens/sec")
