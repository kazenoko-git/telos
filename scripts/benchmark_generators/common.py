"""
Common utilities for building verified benchmark tasks with BASE and HINT prompts.
"""

from typing import Dict, Any, List


def create_task(
    name: str,
    signature: str,
    doc_desc: str,
    doctests: List[str],
    hint: str,
    solution: str,
    test: str,
    args_desc: str = "",
    returns_desc: str = "",
    prefix_code: str = ""
) -> Dict[str, Any]:
    """
    Constructs a standardized evaluation challenge with dual BASE and HINT prompts.
    
    Args:
        name: Function or class name.
        signature: Def or class line, e.g. "def foo(x: int) -> int:"
        doc_desc: Primary docstring description of what the function does.
        doctests: Example usage lines (e.g. [">>> foo(1)", "2"]).
        hint: Specific algorithmic guidance or architectural tip.
        solution: Python code body (indented).
        test: Assertion statements testing the function.
        args_desc: Optional description of arguments.
        returns_desc: Optional description of return value.
        prefix_code: Any imports or helper definitions needed before the signature.
    """
    # Build docstring body
    lines = [f"    {doc_desc}"]
    if args_desc:
        lines.append("")
        lines.append(f"    Args:\n        {args_desc}")
    if returns_desc:
        lines.append("")
        lines.append(f"    Returns:\n        {returns_desc}")
    if doctests:
        lines.append("")
        for dt in doctests:
            lines.append(f"    {dt}")
            
    doc_base_inner = "\n".join(lines)
    
    # Automatically ensure typing symbols used in signature or solution are imported
    typing_symbols = ["Any", "Optional", "Union", "Callable", "List", "Dict", "Tuple", "Set"]
    needed_typing = [s for s in typing_symbols if (s in signature or s in solution) and f"import {s}" not in prefix_code and f"typing" not in prefix_code]
    if needed_typing:
        import_stmt = f"from typing import {', '.join(needed_typing)}"
        if prefix_code:
            prefix_code = f"{import_stmt}\n{prefix_code}"
        else:
            prefix_code = import_stmt

    # Base prompt has no hints
    doc_base = f'    """\n{doc_base_inner}\n    """\n'
    if prefix_code:
        prompt_base = f"{prefix_code.strip()}\n\n{signature}\n{doc_base}"
    else:
        prompt_base = f"{signature}\n{doc_base}"

    # Hint prompt includes the hint section
    hint_lines = [
        "",
        "    Hint:",
        f"        {hint}"
    ]
    doc_hint_inner = doc_base_inner + "\n" + "\n".join(hint_lines)
    doc_hint = f'    """\n{doc_hint_inner}\n    """\n'
    if prefix_code:
        prompt_hint = f"{prefix_code.strip()}\n\n{signature}\n{doc_hint}"
    else:
        prompt_hint = f"{signature}\n{doc_hint}"

    task_dict = {
        "name": name,
        "prompt_base": prompt_base,
        "prompt_hint": prompt_hint,
        "hint": hint,
        "solution": solution,
        "test": test
    }

    # Verify that the solution passes the test harness
    full_code = prompt_base + solution + "\n\n" + test
    env = {"__name__": "__main__"}
    try:
        exec(full_code, env)
    except Exception as exc:
        raise RuntimeError(f"Self-verification failed for task '{name}': {type(exc).__name__}: {exc}") from exc

    return task_dict
