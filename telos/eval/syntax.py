"""
Syntactic AST Analysis Engine for Télos Model Evaluations.

Provides inline, zero-overhead Abstract Syntax Tree (AST) validation and error
breakdown categorization for Python completions across all benchmarks.
"""

import ast
from typing import Tuple, Optional, Dict, Any


def check_ast_validity(code: str) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether generated Python code parses into a valid AST.
    
    Args:
        code: The Python code string to parse.
        
    Returns:
        A tuple (is_valid, error_description):
        - is_valid: True if ast.parse succeeds, False otherwise.
        - error_description: None if valid, else formatted string "ErrorClass: details".
    """
    try:
        # compile() verifies both syntax tree and bytecode legality (e.g. return inside function)
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as err:
        # Capture the specific error type (e.g. IndentationError, TabError, SyntaxError)
        error_type = type(err).__name__
        msg = err.msg or "invalid syntax"
        line_info = f" (line {err.lineno})" if err.lineno is not None else ""
        return False, f"{error_type}: {msg}{line_info}"


def categorize_syntax_error(error_str: Optional[str]) -> str:
    """
    Categorizes a SyntaxError message into standardized diagnostic buckets.
    
    Categories:
    - 'valid': Code parsed cleanly without error
    - 'indentation': IndentationError or TabError
    - 'unexpected_eof': Incomplete generation truncating mid-block or mid-expression
    - 'unclosed_delimiter': Unclosed parenthesis, bracket, brace, or triple-quote string
    - 'general_syntax': Other syntax errors (bad operators, invalid keywords, etc.)
    """
    if not error_str:
        return "valid"
    
    err_lower = error_str.lower()
    
    # Check for indentation issues first
    if "indentationerror" in err_lower or "taberror" in err_lower:
        return "indentation"
    
    # Check for unclosed delimiters or brackets
    if any(phrase in err_lower for phrase in ["was never closed", "unclosed", "closing parenthesis", "unterminated"]):
        return "unclosed_delimiter"

    # Check for truncated completions (EOF before expected block or token)
    if "unexpected eof" in err_lower or "unexpected end of file" in err_lower:
        return "unexpected_eof"
    
    return "general_syntax"


def analyze_syntax_batch(completions: list[str]) -> Dict[str, Any]:
    """
    Analyzes an array of code completions and returns summary metrics.
    
    Returns:
        Dictionary containing:
        - 'valid_count': Total valid completions
        - 'total_count': Total completions analyzed
        - 'valid_pct': Percentage of valid completions (0.0 to 100.0)
        - 'error_breakdown': Counts by categorized error bucket
    """
    total = len(completions)
    if total == 0:
        return {
            "valid_count": 0,
            "total_count": 0,
            "valid_pct": 0.0,
            "error_breakdown": {
                "valid": 0,
                "indentation": 0,
                "unexpected_eof": 0,
                "unclosed_delimiter": 0,
                "general_syntax": 0,
            }
        }
    
    valid_count = 0
    breakdown = {
        "valid": 0,
        "indentation": 0,
        "unexpected_eof": 0,
        "unclosed_delimiter": 0,
        "general_syntax": 0,
    }
    
    for code in completions:
        is_valid, err = check_ast_validity(code)
        bucket = categorize_syntax_error(err)
        breakdown[bucket] = breakdown.get(bucket, 0) + 1
        if is_valid:
            valid_count += 1
            
    return {
        "valid_count": valid_count,
        "total_count": total,
        "valid_pct": round((valid_count / total) * 100.0, 2),
        "error_breakdown": breakdown,
    }
