"""
Tool-Use & Function Calling Evaluation Engine for Télos.

Evaluates:
1. Tool Selection Accuracy: Whether the model invokes the correct functional tool.
2. Syntax & Schema Validity: Whether the generated call parses as valid JSON or function invocation.
3. Argument Correctness: Whether required parameters and argument values match ground truth expectations.
4. Category breakdown across Math, Weather, Retrieval, Filesystem, Communication, Database, and System.
"""

import re
import ast
import json
import math
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .suite import load_tooluse_suite, STANDARD_TOOL_DEFINITIONS, TOOLUSE_BENCHMARK_SUITE
from telos.eval.stats import bootstrap_confidence_interval


def format_tooluse_prompt(user_query: str, tools: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Formats the evaluation prompt containing available tool specifications and user query.
    """
    tools = tools or STANDARD_TOOL_DEFINITIONS
    tool_descriptions = []
    for t in tools:
        params_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in t["parameters"]["properties"].items())
        tool_descriptions.append(f"- {t['name']}({params_str}): {t['description']}")
    tools_block = "\n".join(tool_descriptions)

    prompt = (
        "You are an AI assistant with access to the following tools:\n"
        f"{tools_block}\n\n"
        "To invoke a tool, respond with a call in the format:\n"
        '{"tool": "<tool_name>", "arguments": {<key>: <value>}}\n\n'
        f"User: {user_query}\n"
        "Call: "
    )
    return prompt


def parse_tool_call(completion_text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    """
    Parses tool name and arguments from model completion.
    
    Supports:
    1. JSON format: {"tool": "...", "arguments": {...}} or {"name": "...", "parameters": {...}}
    2. Function call syntax: tool_name(key="val", ...)
    
    Returns:
        (tool_name, arguments_dict, error_status)
    """
    clean_text = completion_text.strip()
    if not clean_text:
        return None, None, "empty_completion"

    # Attempt 1: Look for JSON object enclosed in braces
    json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            if isinstance(payload, dict):
                tool_name = payload.get("tool") or payload.get("name") or payload.get("function")
                args = payload.get("arguments") or payload.get("parameters") or payload.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                if tool_name:
                    return str(tool_name).strip(), args if isinstance(args, dict) else {}, "valid"
        except Exception:
            pass

    # Attempt 2: Function call pattern: name(arg1=val, arg2=val)
    call_match = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)", clean_text, re.DOTALL)
    if call_match:
        fn_name = call_match.group(1).strip()
        raw_args = call_match.group(2).strip()
        # Parse arguments using ast.parse inside dummy function call
        try:
            parsed = ast.parse(f"{fn_name}({raw_args})")
            if parsed.body and isinstance(parsed.body[0], ast.Expr) and isinstance(parsed.body[0].value, ast.Call):
                call_node = parsed.body[0].value
                args_dict = {}
                for kw in call_node.keywords:
                    try:
                        args_dict[kw.arg] = ast.literal_eval(kw.value)
                    except Exception:
                        args_dict[kw.arg] = ast.unparse(kw.value)
                # Positional arguments if any
                for i, pos_arg in enumerate(call_node.args):
                    try:
                        args_dict[f"arg_{i}"] = ast.literal_eval(pos_arg)
                    except Exception:
                        args_dict[f"arg_{i}"] = ast.unparse(pos_arg)
                return fn_name, args_dict, "valid"
        except Exception:
            pass

    return None, None, "syntax_parse_error"


def _try_eval_math(expr: Any) -> Optional[float]:
    """Safely evaluates basic arithmetic expressions to float for equivalence checking."""
    try:
        clean = str(expr).strip().replace("^", "**")
        allowed_names = {"math": math, "sqrt": math.sqrt, "abs": abs}
        val = eval(clean, {"__builtins__": {}}, allowed_names)
        if isinstance(val, (int, float)):
            return float(val)
    except Exception:
        pass
    return None


def _compare_values(expected_val: Any, actual_val: Any) -> bool:
    """Compares two argument values across dictionaries, lists, math expressions, and strings."""
    if isinstance(expected_val, dict) and isinstance(actual_val, dict):
        return _compare_arguments(expected_val, actual_val)
    if isinstance(expected_val, list) and isinstance(actual_val, list):
        if len(expected_val) != len(actual_val):
            return False
        return all(_compare_values(e, a) for e, a in zip(expected_val, actual_val))

    # Mathematical expression evaluation
    math_exp = _try_eval_math(expected_val)
    math_act = _try_eval_math(actual_val)
    if math_exp is not None and math_act is not None:
        return abs(math_exp - math_act) < 1e-5

    # Direct float comparison for numeric strings (e.g. 84.50 vs 84.5)
    try:
        if abs(float(expected_val) - float(actual_val)) < 1e-5:
            return True
    except (ValueError, TypeError):
        pass

    # Normalized string and substring query matching
    exp_str = str(expected_val).strip().lower().replace(" ", "")
    act_str = str(actual_val).strip().lower().replace(" ", "")
    if exp_str == act_str:
        return True
    if exp_str in act_str or act_str in exp_str:
        return True
    return False


def _compare_arguments(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    """
    Compares expected ground-truth arguments with parsed actual arguments.
    Performs case-insensitive normalization, mathematical equivalence, and substring checks.
    """
    if not expected:
        return True
    if not actual:
        return False

    for key, expected_val in expected.items():
        if key not in actual:
            # Check if value matches under alternate positional key
            matched = any(_compare_values(expected_val, act_v) for act_v in actual.values())
            if not matched:
                return False
            continue

        if not _compare_values(expected_val, actual[key]):
            return False

    return True


def evaluate_tooluse(
    model,
    tokenizer,
    backend: str,
    max_tasks: Optional[int] = None,
    max_new_tokens: int = 64,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluates tool selection, schema conformance, and parameter extraction accuracy.
    
    Args:
        model: Loaded model instance.
        tokenizer: Tokenizer instance.
        backend: 'mlx' or 'pytorch'.
        max_tasks: Optional limit on number of tool tasks.
        max_new_tokens: Max tokens generated per tool call completion.
    """
    tasks = load_tooluse_suite(max_tasks)
    total_tasks = len(tasks)

    print("\n" + "=" * 85)
    print(f"  TÉLOS TOOL-USE & FUNCTION CALLING BENCHMARK ({total_tasks} TASKS)")
    print(f"  Decoding: Greedy (T=0.0) | Max Tokens: {max_new_tokens}")
    print("=" * 85)

    category_stats: Dict[str, Dict[str, Any]] = {}
    task_results = []
    correct_tools = 0
    valid_syntax_count = 0
    passed_tasks = 0

    stop_words = ["\nUser:", "\n\n", "User:"]

    for idx, task in enumerate(tasks, 1):
        cat = task.get("category", "General")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "tool_correct": 0, "syntax_valid": 0, "passed": 0}

        prompt = format_tooluse_prompt(task["user_prompt"])
        expected_tool = task["expected_tool"]
        expected_args = task["expected_args"]

        # Generate completion with model adapter or native autoregressive loop
        if hasattr(model, "generate"):
            completion = model.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                stop=stop_words
            ).strip()
        else:
            p_ids = tokenizer.encode(prompt).ids
            curr_ids = list(p_ids)
            stop_set = {0, 3}  # EOS / PAD

            for _ in range(max_new_tokens):
                if backend == "mlx":
                    import mlx.core as mx
                    x = mx.array([curr_ids], dtype=mx.int32)
                    logits = model(x)
                    next_tok = int(np.argmax(np.array(logits[0, -1].astype(mx.float32))))
                else:
                    import torch
                    device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
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

            completion = tokenizer.decode(curr_ids[len(p_ids):]).strip()

        # Parse generated tool call
        parsed_tool, parsed_args, parse_status = parse_tool_call(completion)
        is_syntax_valid = (parse_status == "valid")
        is_tool_correct = bool(parsed_tool and parsed_tool.lower() == expected_tool.lower())
        is_args_correct = bool(is_tool_correct and _compare_arguments(expected_args, parsed_args or {}))
        is_passed = (is_tool_correct and is_args_correct)

        category_stats[cat]["total"] += 1
        if is_syntax_valid:
            valid_syntax_count += 1
            category_stats[cat]["syntax_valid"] += 1
        if is_tool_correct:
            correct_tools += 1
            category_stats[cat]["tool_correct"] += 1
        if is_passed:
            passed_tasks += 1
            category_stats[cat]["passed"] += 1

        task_results.append({
            "id": task.get("id", idx),
            "category": cat,
            "prompt": task["user_prompt"],
            "expected_tool": expected_tool,
            "expected_args": expected_args,
            "parsed_tool": parsed_tool,
            "parsed_args": parsed_args,
            "completion": completion,
            "syntax_valid": is_syntax_valid,
            "tool_correct": is_tool_correct,
            "passed": is_passed,
        })

    tool_acc_pct = (correct_tools / max(total_tasks, 1)) * 100.0
    syntax_acc_pct = (valid_syntax_count / max(total_tasks, 1)) * 100.0
    overall_pass_pct = (passed_tasks / max(total_tasks, 1)) * 100.0

    ci_pass = bootstrap_confidence_interval([1.0 if r["passed"] else 0.0 for r in task_results])

    print("\n" + "-" * 85)
    print(f"  {'Category':<28} | {'Total':<5} | {'Tool Acc (%)':<13} | {'Syntax (%)':<11} | {'Pass Rate (%)'}")
    print("-" * 85)
    cat_summary = {}
    for cat, s in category_stats.items():
        cnt = s["total"]
        t_acc = round((s["tool_correct"] / cnt) * 100.0, 1) if cnt else 0.0
        syn_acc = round((s["syntax_valid"] / cnt) * 100.0, 1) if cnt else 0.0
        p_acc = round((s["passed"] / cnt) * 100.0, 1) if cnt else 0.0
        cat_summary[cat] = {
            "count": cnt,
            "tool_accuracy_pct": t_acc,
            "syntax_validity_pct": syn_acc,
            "pass_rate_pct": p_acc,
        }
        print(f"  {cat:<28} | {cnt:<5d} | {t_acc:>12.1f}% | {syn_acc:>10.1f}% | {p_acc:>12.1f}%")

    print("-" * 85)
    print(f"  OVERALL TOOL SELECTION: {tool_acc_pct:.1f}%")
    print(f"  OVERALL SYNTAX VALIDITY:{syntax_acc_pct:.1f}%")
    print(f"  OVERALL PASS RATE:      {overall_pass_pct:.1f}% (95% CI: [{ci_pass[0]}%, {ci_pass[1]}%])")
    print("=" * 85 + "\n")

    return {
        "benchmark_type": "tooluse",
        "total_tasks": total_tasks,
        "tool_accuracy_pct": round(tool_acc_pct, 2),
        "syntax_validity_pct": round(syntax_acc_pct, 2),
        "pass_rate_pct": round(overall_pass_pct, 2),
        "pass_rate_95ci": ci_pass,
        "categories": cat_summary,
        "tasks": task_results,
    }
