"""
Cybersecurity & Vulnerability Remediation Evaluation Engine for Télos.

Evaluates:
1. CWE Identification Accuracy: Did the model accurately classify the vulnerability (e.g. CWE-89, CWE-78)?
2. Security Explanation Depth: Did the model explain the root cause and attack vector?
3. Secure Code Remediation: Did the proposed fix eliminate the vulnerability without introducing regressions?
"""

import re
from typing import Dict, Any, List, Optional, Tuple


def extract_remediation_code(candidate_response: str, req_rem_patterns: List[str]) -> str:
    """
    Extracts the isolated remediation code block from a model completion.
    Prevents false positive forbidden pattern detection from quoted vulnerabilities.
    """
    resp_lower = candidate_response.lower()

    # Check if an explicit remediation section marker exists
    remediation_markers = [
        "remediated code", "secure, remediated code", "secure code",
        "fixed code", "remediation:", "solution:", "patched code"
    ]
    for marker in remediation_markers:
        if marker in resp_lower:
            idx = resp_lower.rfind(marker)
            section = candidate_response[idx:]
            # Extract code blocks within the remediation section
            sub_blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", section, re.DOTALL)
            if sub_blocks:
                return sub_blocks[-1]
            return section

    # Extract all markdown code blocks
    code_blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", candidate_response, re.DOTALL)
    if len(code_blocks) > 1:
        # Select the block matching the highest number of required remediation patterns
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

    return candidate_response


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

    # Match CWE identifier or normalized number
    cwe_num = target_cwe.replace("CWE-", "")
    cwe_pattern = rf"\bcwe[-_\s]?{cwe_num}\b"
    cwe_detected = bool(re.search(cwe_pattern, resp_lower))

    # Calculate audit keyword coverage score
    matched_audit_kw = 0
    for kw in req_audit_kw:
        if kw.lower() in resp_lower:
            matched_audit_kw += 1
    explanation_score = matched_audit_kw / max(1, len(req_audit_kw))

    # Extract isolated remediation code block to avoid evaluating quoted vulnerable code
    remediation_code = extract_remediation_code(resp_raw, req_rem_patterns)

    # Check for forbidden vulnerable code patterns
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
