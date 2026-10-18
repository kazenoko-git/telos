"""
Comprehensive Verification Test Suite for Télos Multi-Domain Evaluation Ecosystem.
"""

from telos.eval.reasoning_eval import (
    extract_gsm8k_numeric_answer,
    evaluate_gsm8k_sample,
    extract_arc_choice,
    evaluate_arc_sample,
    extract_boxed_latex,
    evaluate_competition_math_sample
)
from telos.eval.cyber_eval import evaluate_cybersecurity_challenge
from telos.eval.polyglot_eval import (
    clean_polyglot_completion,
    validate_code_structural_contract,
    execute_polyglot_task
)
from telos.eval.adapters import load_adapter, OpenAIAPIAdapter, GeminiAPIAdapter

def run_tests():
    print("=== 1. VERIFYING REASONING EVALUATORS ===")
    # 1. GSM8K
    assert extract_gsm8k_numeric_answer("Total is <<10+2=12>> #### 12") == "12"
    assert extract_gsm8k_numeric_answer("The final answer is: 1,250.50 dollars.") == "1250.50"
    is_match, _ = evaluate_gsm8k_sample("Answer: #### 42", "42")
    assert is_match, "GSM8K match failed"
    is_match, _ = evaluate_gsm8k_sample("Result is $1,500.", "1500")
    assert is_match, "GSM8K currency match failed"
    print("  ✓ GSM8K evaluator verified")

    # 2. ARC-Challenge
    assert extract_arc_choice("Answer: (C)") == "C"
    assert extract_arc_choice("The correct choice is B.") == "B"
    is_match, _ = evaluate_arc_sample("Hence option: A", "A")
    assert is_match, "ARC match failed"
    print("  ✓ ARC-Challenge evaluator verified")

    # 3. Competition MATH
    assert extract_boxed_latex(r"Therefore, \boxed{\frac{1}{2}} is correct.") == r"\frac{1}{2}"
    assert extract_boxed_latex(r"So x = \boxed{14}.") == "14"
    is_match, _ = evaluate_competition_math_sample(r"We find \boxed{2}.", "2")
    assert is_match, "Competition MATH match failed"
    print("  ✓ Competition MATH evaluator verified")

    print("\n=== 2. VERIFYING CYBERSECURITY EVALUATOR ===")
    test_challenge = {
        "evaluation_criteria": {
            "target_cwe": "CWE-89",
            "required_audit_keywords": ["parameterized", "SQL injection"],
            "required_remediation_patterns": ["%s", "cursor.execute"],
            "forbidden_vulnerable_patterns": [r"f[\"'].*SELECT.*\{", r"\+ dept_name \+"],
        }
    }
    good_resp = (
        "This vulnerability is CWE-89, an instance of SQL injection. "
        "The query should be parameterized. "
        "Remediation:\n```python\ncursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))\n```"
    )
    res = evaluate_cybersecurity_challenge(good_resp, test_challenge)
    assert res["cwe_detected"] is True, f"CWE not detected: {res}"
    assert res["overall_passed"] is True, f"Overall pass failed: {res}"
    print("  ✓ Cybersecurity evaluator positive remediation verified")

    bad_resp = (
        "Vulnerability is CWE-89 SQL injection. Fix:\n"
        "```python\ncursor.execute(f'SELECT * FROM users WHERE id = {user_id}')\n```"
    )
    res_bad = evaluate_cybersecurity_challenge(bad_resp, test_challenge)
    assert res_bad["overall_passed"] is False, "Bad response incorrectly passed"
    assert res_bad["has_forbidden_pattern"] is True, "Forbidden pattern not detected"
    print("  ✓ Cybersecurity evaluator negative vulnerability blocking verified")

    print("\n=== 3. VERIFYING POLYGLOT EVALUATION ===")
    cs_code = "using System;\nclass Program {\n    public static int Add(int a, int b) {\n        return a + b;\n    }\n}"
    valid, msg = validate_code_structural_contract("csharp", cs_code)
    assert valid, f"C# contract check failed: {msg}"

    js_code = "function add(a, b) { return a + b; }\nassert(add(2, 3) === 5);"
    outcome, details = execute_polyglot_task("javascript", "", js_code, "")
    assert outcome.value == "PASSED", f"JS execution failed: {details}"
    print("  ✓ Polyglot C# structural validation & Node.js execution verified")

    print("\n=== 4. VERIFYING UNIVERSAL MODEL ADAPTER REGISTRY ===")
    afm_adapter = load_adapter("afm-3-core", api_base="http://localhost:8000/v1")
    assert isinstance(afm_adapter, OpenAIAPIAdapter)
    assert afm_adapter.model_name == "afm-3-core"
    print("  ✓ AFM 3 Core -> OpenAIAPIAdapter routing verified")

    gemini_adapter = load_adapter("gemini-4-26b-a4b", api_key="test-key")
    assert isinstance(gemini_adapter, GeminiAPIAdapter)
    print("  ✓ Gemini 4 26B A4B -> GeminiAPIAdapter routing verified")

    native_adapter = load_adapter("checkpoints/ar/model.safetensors")
    assert native_adapter.backend_name == "telos_native"
    assert hasattr(native_adapter, "model")
    print("  ✓ Local Checkpoint -> TelosNativeAdapter routing verified")

    print("\n=== ALL COMPONENT EVALUATORS AND ADAPTERS 100% VERIFIED ===")

if __name__ == "__main__":
    run_tests()
