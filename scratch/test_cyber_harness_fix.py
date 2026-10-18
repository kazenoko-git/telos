import re

def extract_remediation_code(resp_raw: str, req_rem_patterns: list) -> str:
    resp_lower = resp_raw.lower()
    
    # Check if an explicit remediation section marker exists
    remediation_markers = [
        "remediated code", "secure, remediated code", "secure code", 
        "fixed code", "remediation:", "solution:", "patched code"
    ]
    for marker in remediation_markers:
        if marker in resp_lower:
            idx = resp_lower.rfind(marker)
            section = resp_raw[idx:]
            sub_blocks = re.findall(r"```(?:python|javascript|csharp|java)?\s*(.*?)```", section, re.DOTALL)
            if sub_blocks:
                return sub_blocks[-1]
            return section

    # Extract all markdown code blocks
    code_blocks = re.findall(r"```(?:python|javascript|csharp|java)?\s*(.*?)```", resp_raw, re.DOTALL)
    if len(code_blocks) > 1:
        best_block = code_blocks[-1]
        best_score = -1
        for block in code_blocks:
            score = sum(1 for pat in req_rem_patterns if pat.lower() in block.lower())
            if score > best_score:
                best_score = score
                best_block = block
        return best_block
    elif code_blocks:
        return code_blocks[0]
    return resp_raw

# Test 1: Model quotes vulnerable code first, then provides remediated code
sample_1 = """
The vulnerability is SQL Injection (CWE-89).
Original flawed code:
```python
cursor.execute("SELECT * FROM users WHERE name = '" + name + "'")
```

Here is the secure remediated code:
```python
cursor.execute("SELECT * FROM users WHERE name = %s", (name,))
```
"""
rem_code = extract_remediation_code(sample_1, ["cursor.execute", "%s"])
print("Test 1 Extracted:")
print(rem_code.strip())
assert "+ name +" not in rem_code
assert "%s" in rem_code
print("✓ Test 1 Passed!")

# Test 2: Single code block
sample_2 = """
CWE-89. Fix:
```python
cursor.execute("SELECT * FROM users WHERE name = %s", (name,))
```
"""
rem_code_2 = extract_remediation_code(sample_2, ["cursor.execute", "%s"])
print("Test 2 Extracted:")
print(rem_code_2.strip())
assert "%s" in rem_code_2
print("✓ Test 2 Passed!")
