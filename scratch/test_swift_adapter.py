import json
import subprocess
import os

bridge_path = "telos/eval/adapters/swift_afm/afm_bridge"
proc = subprocess.Popen(
    [bridge_path],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

def query(prompt, max_tokens=256, instructions=None):
    req = {
        "id": 1,
        "prompt": prompt,
        "instructions": instructions or "You are an expert programming and reasoning assistant. Complete the code or provide direct answer without conversational filler.",
        "maxTokens": max_tokens,
        "temperature": 0.0
    }
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    return json.loads(line)

# 1. Code test
p_code = 'def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n'
res_code = query(p_code)
print("--- Code Response ---")
print(res_code.get("content"))

# 2. Math test
p_math = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Provide the reasoning and final answer as: #### <number>"
res_math = query(p_math)
print("--- Math Response ---")
print(res_math.get("content"))

proc.terminate()
