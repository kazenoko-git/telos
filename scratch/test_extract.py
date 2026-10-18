import ast
import json
import re
from telos.eval.adapters.gemini_api import GeminiAPIAdapter
from telos.eval.runner import execute_code_sandboxed

def clean_candidate(prompt: str, raw_completion: str) -> str:
    comp = raw_completion.strip()
    if "```" in comp:
        blocks = re.findall(r"```(?:python)?\s*(.*?)```", comp, re.DOTALL)
        if blocks:
            candidate = blocks[0].strip()
            # If candidate is a complete standalone python program defining the function
            try:
                tree = ast.parse(candidate)
                func_defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if func_defs:
                    return candidate
            except SyntaxError:
                pass

    # Fallback to appending to prompt
    return prompt + "\n" + comp

if __name__ == '__main__':
    adapter = GeminiAPIAdapter(
        model_name='gemma-4-26b-a4b-it',
        api_key=os.environ.get('GEMINI_API_KEY', '')
    )

    with open('evals/benchmarks/humaneval_suite.json') as f:
        tasks = json.load(f)[:5]

    for i, t in enumerate(tasks):
        raw = adapter.generate(t['prompt'], max_new_tokens=1536)
        full_code = clean_candidate(t['prompt'], raw)
        outcome, details = execute_code_sandboxed(full_code, t['test_harness'], 3.0)
        print(f"Task {t['id']}: {outcome} ({details})")
