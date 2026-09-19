"""
Cybersecurity & Vulnerability Remediation Evaluation Engine for Télos.

Evaluates:
1. CWE Identification Accuracy: Did the model accurately classify the vulnerability (e.g. CWE-89, CWE-78)?
2. Security Explanation Depth: Did the model explain the root cause and attack vector?
3. Secure Code Remediation: Did the proposed fix eliminate the vulnerability without introducing regressions?
"""

import re
from typing import Dict, Any, List, Optional, Tuple


def evaluate_cybersecurity_challenge(
    candidate_response: str,
    challenge_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Scores a single cybersecurity challenge response across:
    - cwe_detected (bool)
    - explanation_score (0.0 - 1.0)
    - remediation_score (0.0 - 1.0)
    - overall_passed (bool)
    """
    criteria = challenge_meta.get("evaluation_criteria", {})
    target_cwe = criteria.get("target_cwe", "").upper()
    req_audit_kw = criteria.get("required_audit_keywords", [])
    req_rem_patterns = criteria.get("required_remediation_patterns", [])
    forbidden_patterns = criteria.get("forbidden_vulnerable_patterns", [])

    resp_lower = candidate_response.lower()
    resp_raw = candidate_response

    # 1. CWE Identification Check
    # Look for exact CWE or normalized number (e.g. 'CWE-89' or 'cwe 89')
    cwe_num = target_cwe.replace("CWE-", "")
    cwe_pattern = rf"\bcwe[-_\s]?{cwe_num}\b"
    cwe_detected = bool(re.search(cwe_pattern, resp_lower))

    # 2. Audit Explanation Keywords
    matched_audit_kw = 0
    for kw in req_audit_kw:
        if kw.lower() in resp_lower:
            matched_audit_kw += 1
    explanation_score = matched_audit_kw / max(1, len(req_audit_kw))

    # 3. Remediation Code Check
    # Extract code blocks from candidate response
    code_blocks = re.findall(r"```(?:python|javascript|csharp|java)?\s*(.*?)```", resp_raw, re.DOTALL)
    remediation_code = "\n".join(code_blocks) if code_blocks else resp_raw

    # Check for forbidden patterns (e.g. recurring vulnerability)
    has_forbidden_pattern = False
    for pat in forbidden_patterns:
        try:
            if re.search(pat, remediation_code, re.IGNORECASE):
                has_forbidden_pattern = True
                break
        except re.error:
            if pat.lower() in remediation_code.lower():
                has_forbidden_pattern = True
                break

    # Check for required secure remediation constructs
    matched_rem = 0
    for pat in req_rem_patterns:
        if pat.lower() in remediation_code.lower():
            matched_rem += 1
    rem_coverage = matched_rem / max(1, len(req_rem_patterns))

    remediation_score = 0.0 if has_forbidden_pattern else rem_coverage
    overall_passed = cwe_detected and (remediation_score >= 0.5) and not has_forbidden_pattern

    return {
        "cwe_detected": cwe_detected,
        "explanation_score": explanation_score,
        "remediation_score": remediation_score,
        "has_forbidden_pattern": has_forbidden_pattern,
        "overall_passed": overall_passed,
        "target_cwe": target_cwe,
    }
