"""
Unit Test Suite for Télos Multi-Domain Evaluation Framework.

Tests:
1. English Linguistic Probes (100 deterministic probes across 5 linguistic categories).
2. Tool-Use Benchmark Suite (prompt formatting, JSON & function call parsing, schema comparison).
3. CLI Dispatch & Argument Parsing (--type all|code|linguistic|tooluse, --language auto|english|python).
"""

import pytest
from telos.eval.linguistic import load_english_probes, ENGLISH_PROBES_100
from telos.eval.tooluse import (
    load_tooluse_suite,
    TOOLUSE_BENCHMARK_SUITE,
    STANDARD_TOOL_DEFINITIONS,
    format_tooluse_prompt,
    parse_tool_call,
)
from telos.eval.tooluse.evaluator import _compare_arguments


class TestEnglishLinguisticSuite:
    """Validates English linguistic probes and benchmark dataset integrity."""

    def test_probe_suite_count_and_loading(self):
        probes = load_english_probes(100)
        assert len(probes) == 100
        assert len(ENGLISH_PROBES_100) == 100

    def test_probe_schema_fields(self):
        probes = load_english_probes()
        required_keys = {"id", "category", "prompt", "prefix", "target", "mode", "description"}
        for p in probes:
            missing = required_keys - set(p.keys())
            assert not missing, f"Probe {p.get('id')} missing required fields: {missing}"
            assert len(p["target"]) > 0, f"Probe {p.get('id')} has empty target"
            assert p["mode"] in ["both", "causal", "infill"]

    def test_category_distribution(self):
        probes = load_english_probes()
        categories = {}
        for p in probes:
            cat = p["category"]
            categories[cat] = categories.get(cat, 0) + 1

        expected_categories = [
            "Subject-Verb Agreement & Morphology",
            "Lexical Collocations & Prepositional Idioms",
            "Connectives & Paired Correlatives",
            "Common Sense & World Knowledge Cloze",
            "Suffix-Clued Bidirectional Infilling",
        ]
        for cat in expected_categories:
            assert cat in categories, f"Category '{cat}' missing from English probe suite"
            assert categories[cat] == 20, f"Expected exactly 20 probes for '{cat}', found {categories[cat]}"

    def test_bidirectional_infill_probes_have_suffixes(self):
        probes = load_english_probes()
        infill_probes = [p for p in probes if p["mode"] in ["infill", "both"]]
        for p in infill_probes:
            assert "suffix" in p and len(p["suffix"]) > 0, f"Infill probe {p['id']} must have non-empty suffix"


class TestToolUseSuite:
    """Validates Tool-Use dataset, prompt templating, and parser."""

    def test_suite_loading(self):
        tasks = load_tooluse_suite()
        assert len(tasks) >= 30
        assert len(TOOLUSE_BENCHMARK_SUITE) >= 30

    def test_task_schema(self):
        tasks = load_tooluse_suite()
        for t in tasks:
            assert "id" in t
            assert "category" in t
            assert "user_prompt" in t
            assert "expected_tool" in t
            assert "expected_args" in t
            assert isinstance(t["expected_args"], dict)

    def test_prompt_formatting(self):
        prompt = format_tooluse_prompt("What is 10 + 20?")
        assert "calculator" in prompt
        assert "get_weather" in prompt
        assert "User: What is 10 + 20?" in prompt
        assert "Call:" in prompt

    def test_json_tool_call_parsing(self):
        # Valid JSON call
        completion = '{"tool": "calculator", "arguments": {"expression": "10 + 20"}}'
        tool, args, status = parse_tool_call(completion)
        assert tool == "calculator"
        assert args == {"expression": "10 + 20"}
        assert status == "valid"

    def test_alternative_json_keys(self):
        # Alternate keys: name & parameters
        completion = '{"name": "get_weather", "parameters": {"location": "Tokyo", "unit": "celsius"}}'
        tool, args, status = parse_tool_call(completion)
        assert tool == "get_weather"
        assert args == {"location": "Tokyo", "unit": "celsius"}
        assert status == "valid"

    def test_function_syntax_tool_call_parsing(self):
        # Python function call syntax
        completion = 'calculator(expression="458 * 32")'
        tool, args, status = parse_tool_call(completion)
        assert tool == "calculator"
        assert args == {"expression": "458 * 32"}
        assert status == "valid"

    def test_argument_comparison(self):
        expected = {"expression": "458 * 32"}
        actual = {"expression": "458*32"}
        assert _compare_arguments(expected, actual) is True

        # Query substring matching
        expected_search = {"query": "latest Nobel Prize winners in physics"}
        actual_search = {"query": "Nobel Prize winners in physics"}
        assert _compare_arguments(expected_search, actual_search) is True

        # Mismatched tool arguments
        mismatched = {"expression": "100 + 200"}
        assert _compare_arguments(expected, mismatched) is False


class TestDispatcherRouting:
    """Validates dispatcher logic and CLI argument options."""

    def test_cli_parser_defaults(self):
        import argparse
        from telos.eval.runner import main

        # Create parser identical to runner.main
        parser = argparse.ArgumentParser()
        parser.add_argument("--type", "--benchmark-type", type=str, default="all", dest="benchmark_type")
        parser.add_argument("--language", "--lang", type=str, default="auto", dest="language")

        # Test defaults
        args = parser.parse_args([])
        assert args.benchmark_type == "all"
        assert args.language == "auto"

        # Test explicit linguistic English
        args_ling = parser.parse_args(["--type", "linguistic", "--language", "english"])
        assert args_ling.benchmark_type == "linguistic"
        assert args_ling.language == "english"

        # Test tooluse
        args_tool = parser.parse_args(["--type", "tooluse"])
        assert args_tool.benchmark_type == "tooluse"

        # Test code python
        args_code = parser.parse_args(["--type", "code", "--language", "python"])
        assert args_code.benchmark_type == "code"
        assert args_code.language == "python"
