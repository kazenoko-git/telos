import os
import sys
import json
import time
import urllib.request
import gzip
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from telos.eval.runner import load_model_from_checkpoint, clean_functional_completion, bootstrap_confidence_interval
from telos.eval.syntax import check_ast_validity
from telos.eval.executor import execute_code_sandboxed, ExecutionResult
from telos.data.tokenizer import load_tokenizer

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

def fetch_humaneval():
    req = urllib.request.urlopen(HUMANEVAL_URL)
    with gzip.GzipFile(fileobj=req) as f:
        tasks = [json.loads(line) for line in f]
    return tasks

def generate_greedy(model, tokenizer, device, prompt: str, max_new_tokens: int = 128) -> str:
    p_ids = tokenizer.encode(prompt).ids
    max_seq_len = 512
    if len(p_ids) >= max_seq_len - 16:
        p_ids = p_ids[-(max_seq_len - 16):]
    curr_ids = list(p_ids)
    stop_set = {0, 3}
    stop_words = ["\ndef ", "\nclass ", "\nif __name__"]
    
    for _ in range(min(max_new_tokens, max_seq_len - len(curr_ids))):
        x = torch.tensor([curr_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        next_tok = int(torch.argmax(logits[0, -1]).item())
        if next_tok in stop_set:
            break
        curr_ids.append(next_tok)
        cur_text = tokenizer.decode(curr_ids[len(p_ids):])
        if any(sw in cur_text for sw in stop_words):
            break
    return tokenizer.decode(curr_ids[len(p_ids):])

def eval_model(model_name, ckpt_path, tasks, tokenizer, device, num_tasks=50):
    model, backend, _ = load_model_from_checkpoint(ckpt_path)
    model = model.to(device).eval()
    
    eval_tasks = tasks[:num_tasks]
    passed = 0
    ast_valid = 0
    outcomes = {k.value: 0 for k in ExecutionResult}
    
    t0 = time.time()
    for idx, t in enumerate(eval_tasks, 1):
        prompt = t["prompt"]
        entry = t["entry_point"]
        test_harness = f"{t['test']}\n\ncheck({entry})\n"
        
        raw_comp = generate_greedy(model, tokenizer, device, prompt, max_new_tokens=128)
        comp = clean_functional_completion(prompt, raw_comp)
        full_code = prompt + comp
        
        is_ast, _ = check_ast_validity(full_code)
        if is_ast:
            ast_valid += 1
            
        res, _ = execute_code_sandboxed(full_code, test_harness, timeout_seconds=1.5)
        outcomes[res.value] += 1
        if res == ExecutionResult.PASSED:
            passed += 1
            
    elapsed = time.time() - t0
    return {
        "model": model_name,
        "tasks_evaluated": len(eval_tasks),
        "pass_at_1": round(passed / len(eval_tasks) * 100, 2),
        "ast_validity": round(ast_valid / len(eval_tasks) * 100, 2),
        "syntax_error_rate": round(outcomes['SYNTAX_ERROR'] / len(eval_tasks) * 100, 2),
        "outcomes": outcomes,
        "elapsed_sec": round(elapsed, 1)
    }

def main():
    tasks = fetch_humaneval()
    tokenizer = load_tokenizer(str(PROJECT_ROOT / "configs" / "tokenizer_mac.json"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Loaded {len(tasks)} HumanEval tasks. Running comparative evaluation...")
    
    res_tpu = eval_model("50M AR (TPU 2B)", "checkpoints/ar/50m_2b/checkpoint_final.pt", tasks, tokenizer, device, num_tasks=50)
    print("TPU Results:", res_tpu)
    
    res_cuda = eval_model("50M AR (CUDA 2.5B)", "checkpoints/ar/50m_2.5b/checkpoint_step_12000.pt", tasks, tokenizer, device, num_tasks=50)
    print("CUDA Results:", res_cuda)
    
    out_file = PROJECT_ROOT / "logs" / "humaneval_50m_tpu_vs_cuda.json"
    with open(out_file, "w") as f:
        json.dump({"tpu": res_tpu, "cuda": res_cuda}, f, indent=2)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
