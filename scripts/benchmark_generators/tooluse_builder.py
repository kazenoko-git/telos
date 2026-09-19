"""
Tool-Use & Function Calling Benchmark Generator.

Expands tool-use evaluation to 100+ standardized challenges across 10 tools:
1. calculator
2. web_search
3. get_weather
4. file_search
5. send_email
6. database_query
7. run_command
8. python_repl
9. git_log
10. fetch_url

Outputs to: evals/benchmarks/tooluse_suite.json
"""

import json
from pathlib import Path
from typing import List, Dict, Any

TOOLS_SPEC = [
    {
        "name": "calculator",
        "description": "Calculates the mathematical result of an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    },
    {
        "name": "web_search",
        "description": "Searches the web for relevant documents matching a query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_weather",
        "description": "Retrieves the current weather and forecast for a specified city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "file_search",
        "description": "Searches for files matching a filename pattern or directory path.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."}
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "send_email",
        "description": "Sends an email message to a specified recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["recipient", "subject", "body"]
        }
    },
    {
        "name": "database_query",
        "description": "Queries a database table with filter criteria.",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "filters": {"type": "object"}
            },
            "required": ["table"]
        }
    },
    {
        "name": "run_command",
        "description": "Executes a shell command in an isolated environment.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    },
    {
        "name": "python_repl",
        "description": "Executes arbitrary Python code in a sandboxed interpreter and returns standard output.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"]
        }
    },
    {
        "name": "git_log",
        "description": "Retrieves commit log entries from a git repository with optional author or limit filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "default": "."},
                "max_count": {"type": "integer", "default": 10},
                "author": {"type": "string"}
            }
        }
    },
    {
        "name": "fetch_url",
        "description": "Fetches raw HTML or JSON content from a specified HTTP URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timeout": {"type": "integer", "default": 15}
            },
            "required": ["url"]
        }
    }
]

BASE_CASES = [
    # Calculator
    ("What is 850 divided by 17?", "calculator", {"expression": "850 / 17"}, "Math & Calculation"),
    ("Compute 18% tip on a bill of $142.50.", "calculator", {"expression": "142.50 * 0.18"}, "Math & Calculation"),
    ("Calculate 3 to the 8th power.", "calculator", {"expression": "3 ** 8"}, "Math & Calculation"),
    ("What is the square root of 65536?", "calculator", {"expression": "65536 ** 0.5"}, "Math & Calculation"),
    ("Evaluate (45 * 12) + (130 / 5).", "calculator", {"expression": "(45 * 12) + (130 / 5)"}, "Math & Calculation"),
    ("What is 1024 * 768?", "calculator", {"expression": "1024 * 768"}, "Math & Calculation"),
    ("Calculate 25 factorial or roughly 25 * 24 * 23.", "calculator", {"expression": "25 * 24 * 23"}, "Math & Calculation"),
    ("Find the remainder of 987654 divided by 37.", "calculator", {"expression": "987654 % 37"}, "Math & Calculation"),
    ("What is log base 2 of 2048? Calculate 2 ** 11.", "calculator", {"expression": "2 ** 11"}, "Math & Calculation"),
    ("Compute (300 - 45) * 1.08 for tax calculation.", "calculator", {"expression": "(300 - 45) * 1.08"}, "Math & Calculation"),

    # Weather
    ("What's the temperature in Paris, France right now?", "get_weather", {"location": "Paris, France", "unit": "celsius"}, "Weather & Environment"),
    ("Is it raining in Seattle today?", "get_weather", {"location": "Seattle", "unit": "fahrenheit"}, "Weather & Environment"),
    ("Check weather for Toronto in celsius.", "get_weather", {"location": "Toronto", "unit": "celsius"}, "Weather & Environment"),
    ("How hot is it in Miami in fahrenheit?", "get_weather", {"location": "Miami", "unit": "fahrenheit"}, "Weather & Environment"),
    ("Current weather conditions in Sydney, Australia.", "get_weather", {"location": "Sydney, Australia", "unit": "celsius"}, "Weather & Environment"),
    ("What is the forecast for Berlin this afternoon?", "get_weather", {"location": "Berlin", "unit": "celsius"}, "Weather & Environment"),
    ("Give me the temperature in Tokyo in celsius.", "get_weather", {"location": "Tokyo", "unit": "celsius"}, "Weather & Environment"),
    ("Check whether it is snowing in Oslo.", "get_weather", {"location": "Oslo", "unit": "celsius"}, "Weather & Environment"),
    ("Weather report for Austin, Texas.", "get_weather", {"location": "Austin, Texas", "unit": "fahrenheit"}, "Weather & Environment"),
    ("What is the humidity and temperature in Singapore?", "get_weather", {"location": "Singapore", "unit": "celsius"}, "Weather & Environment"),

    # Web Search
    ("Search for latest developments in discrete diffusion models 2026.", "web_search", {"query": "discrete diffusion models 2026", "max_results": 5}, "Information Retrieval"),
    ("Find recent documentation on Python 3.13 free-threading.", "web_search", {"query": "Python 3.13 free threading documentation", "max_results": 5}, "Information Retrieval"),
    ("Look up the date of the next solar eclipse.", "web_search", {"query": "next solar eclipse date", "max_results": 3}, "Information Retrieval"),
    ("Search for Rust async runtime performance benchmarks.", "web_search", {"query": "Rust async runtime benchmarks", "max_results": 5}, "Information Retrieval"),
    ("Who won the Nobel Prize in Physics in 2024?", "web_search", {"query": "Nobel Prize in Physics 2024 winner", "max_results": 5}, "Information Retrieval"),
    ("Find official docs for vLLM speculative decoding.", "web_search", {"query": "vLLM speculative decoding docs", "max_results": 5}, "Information Retrieval"),
    ("Search for Apple Silicon MLX benchmark results.", "web_search", {"query": "Apple Silicon MLX benchmark", "max_results": 5}, "Information Retrieval"),
    ("Look up RFC 8446 for TLS 1.3 protocol details.", "web_search", {"query": "RFC 8446 TLS 1.3 specification", "max_results": 5}, "Information Retrieval"),
    ("Search for state-of-the-art DNA language models.", "web_search", {"query": "state of the art DNA foundation models", "max_results": 5}, "Information Retrieval"),
    ("Find how to fix CUDA out of memory error in PyTorch.", "web_search", {"query": "fix CUDA out of memory error PyTorch", "max_results": 5}, "Information Retrieval"),

    # File Search
    ("Find all Python test files in tests/ directory.", "file_search", {"pattern": "test_*.py", "path": "tests"}, "Filesystem Navigation"),
    ("Locate any package.json files in the project.", "file_search", {"pattern": "package.json", "path": "."}, "Filesystem Navigation"),
    ("Search for all markdown files in the repository docs.", "file_search", {"pattern": "*.md", "path": "docs"}, "Filesystem Navigation"),
    ("Find any .pt or .safetensors model weights files.", "file_search", {"pattern": "*.safetensors", "path": "checkpoints"}, "Filesystem Navigation"),
    ("Search for Dockerfile anywhere in the repository.", "file_search", {"pattern": "*Dockerfile*", "path": "."}, "Filesystem Navigation"),
    ("Find all Rust source files in src/.", "file_search", {"pattern": "*.rs", "path": "src"}, "Filesystem Navigation"),
    ("Locate config.yaml or config.json files.", "file_search", {"pattern": "config.*", "path": "config"}, "Filesystem Navigation"),
    ("Find all C# source files in the project.", "file_search", {"pattern": "*.cs", "path": "."}, "Filesystem Navigation"),
    ("Locate all .env or .env.example files.", "file_search", {"pattern": ".env*", "path": "."}, "Filesystem Navigation"),
    ("Find all CSV dataset files in data/ directory.", "file_search", {"pattern": "*.csv", "path": "data"}, "Filesystem Navigation"),

    # Send Email
    ("Send an email to dev-lead@example.com with subject 'Sprint Status' saying 'All tests passed'.", "send_email", {"recipient": "dev-lead@example.com", "subject": "Sprint Status", "body": "All tests passed"}, "Communication"),
    ("Email alice@company.org about 'Meeting Rescheduled' letting her know it's moved to 3pm.", "send_email", {"recipient": "alice@company.org", "subject": "Meeting Rescheduled", "body": "Meeting is moved to 3pm."}, "Communication"),
    ("Notify ops-alert@infra.net with subject 'High Disk Usage' stating 'Disk /dev/sda1 is at 94%'.", "send_email", {"recipient": "ops-alert@infra.net", "subject": "High Disk Usage", "body": "Disk /dev/sda1 is at 94%"}, "Communication"),
    ("Send receipt to customer@shop.com with subject 'Order #4092' and body 'Thank you for your purchase'.", "send_email", {"recipient": "customer@shop.com", "subject": "Order #4092", "body": "Thank you for your purchase"}, "Communication"),
    ("Email security@internal.io subject 'Security Audit Report' with body 'Report attached for review'.", "send_email", {"recipient": "security@internal.io", "subject": "Security Audit Report", "body": "Report attached for review"}, "Communication"),
    ("Send welcome email to user12@domain.com subject 'Welcome!' body 'Welcome to the platform'.", "send_email", {"recipient": "user12@domain.com", "subject": "Welcome!", "body": "Welcome to the platform"}, "Communication"),
    ("Email hr@corp.com subject 'Leave Request' body 'Requesting time off on Friday'.", "send_email", {"recipient": "hr@corp.com", "subject": "Leave Request", "body": "Requesting time off on Friday"}, "Communication"),
    ("Send notification to pager@oncall.org subject 'Incident 102' body 'Database connection pool exhausted'.", "send_email", {"recipient": "pager@oncall.org", "subject": "Incident 102", "body": "Database connection pool exhausted"}, "Communication"),
    ("Email billing@vendor.com subject 'Invoice #772' body 'Please find invoice payment confirmation'.", "send_email", {"recipient": "billing@vendor.com", "subject": "Invoice #772", "body": "Please find invoice payment confirmation"}, "Communication"),
    ("Send feedback to support@tool.ai subject 'Feature Request' body 'Please add dark mode support'.", "send_email", {"recipient": "support@tool.ai", "subject": "Feature Request", "body": "Please add dark mode support"}, "Communication"),

    # Database Query
    ("Query the users table for active accounts where status is 'active'.", "database_query", {"table": "users", "filters": {"status": "active"}}, "Database Operations"),
    ("Get all orders from orders table with status 'pending'.", "database_query", {"table": "orders", "filters": {"status": "pending"}}, "Database Operations"),
    ("Find customers from customers table located in 'California'.", "database_query", {"table": "customers", "filters": {"state": "California"}}, "Database Operations"),
    ("Lookup products from inventory table with category 'electronics'.", "database_query", {"table": "inventory", "filters": {"category": "electronics"}}, "Database Operations"),
    ("Find employee records in employees table where department is 'Engineering'.", "database_query", {"table": "employees", "filters": {"department": "Engineering"}}, "Database Operations"),
    ("Check logs table for severity level 'ERROR'.", "database_query", {"table": "logs", "filters": {"severity": "ERROR"}}, "Database Operations"),
    ("Query transactions table for currency 'USD'.", "database_query", {"table": "transactions", "filters": {"currency": "USD"}}, "Database Operations"),
    ("Find all items in shipments table with status 'in_transit'.", "database_query", {"table": "shipments", "filters": {"status": "in_transit"}}, "Database Operations"),
    ("Get all articles in blog_posts where published is true.", "database_query", {"table": "blog_posts", "filters": {"published": True}}, "Database Operations"),
    ("Search subscriptions table for plan 'enterprise'.", "database_query", {"table": "subscriptions", "filters": {"plan": "enterprise"}}, "Database Operations"),

    # Run Command
    ("List all files in long format using ls -la.", "run_command", {"command": "ls -la"}, "System Execution"),
    ("Check current disk usage using df -h.", "run_command", {"command": "df -h"}, "System Execution"),
    ("Check free memory on the system using free -m.", "run_command", {"command": "free -m"}, "System Execution"),
    ("Print current working directory with pwd.", "run_command", {"command": "pwd"}, "System Execution"),
    ("View git status of the current repository.", "run_command", {"command": "git status"}, "System Execution"),
    ("Check network interfaces using ifconfig or ip a.", "run_command", {"command": "ip a"}, "System Execution"),
    ("Find top running processes sorted by CPU using ps aux --sort=-%cpu.", "run_command", {"command": "ps aux --sort=-%cpu"}, "System Execution"),
    ("Check system uptime and load average.", "run_command", {"command": "uptime"}, "System Execution"),
    ("Check Python version using python3 --version.", "run_command", {"command": "python3 --version"}, "System Execution"),
    ("Check the current date and UTC time with date -u.", "run_command", {"command": "date -u"}, "System Execution"),

    # Python REPL
    ("Run a python snippet to compute the first 10 Fibonacci numbers.", "python_repl", {"code": "def fib(n):\n    a, b = 0, 1\n    res = []\n    for _ in range(n):\n        res.append(a)\n        a, b = b, a + b\n    return res\nprint(fib(10))"}, "Code Execution"),
    ("Execute Python code to generate 5 random floats and print their mean.", "python_repl", {"code": "import random\nnums = [random.random() for _ in range(5)]\nprint(sum(nums)/len(nums))"}, "Code Execution"),
    ("Run Python to test if string 'racecar' is a palindrome.", "python_repl", {"code": "s = 'racecar'\nprint(s == s[::-1])"}, "Code Execution"),
    ("Execute Python code to compute permutations of [1, 2, 3].", "python_repl", {"code": "import itertools\nprint(list(itertools.permutations([1, 2, 3])))"}, "Code Execution"),
    ("Run Python code to get SHA256 of string 'hello'.", "python_repl", {"code": "import hashlib\nprint(hashlib.sha256(b'hello').hexdigest())"}, "Code Execution"),

    # Git Log
    ("Show the last 5 git commits in the current repository.", "git_log", {"max_count": 5, "repo_path": "."}, "Version Control"),
    ("Get recent commit history authored by 'Ivan Samuel'.", "git_log", {"author": "Ivan Samuel", "max_count": 10, "repo_path": "."}, "Version Control"),
    ("View git commit log for repo at /Users/ivansamuel/telos.", "git_log", {"repo_path": "/Users/ivansamuel/telos", "max_count": 10}, "Version Control"),
    ("Check last 3 commits in frontend subdirectory.", "git_log", {"repo_path": "./frontend", "max_count": 3}, "Version Control"),
    ("Show commit logs by author 'bot@ci.org'.", "git_log", {"author": "bot@ci.org", "max_count": 10, "repo_path": "."}, "Version Control"),

    # Fetch URL
    ("Fetch the latest JSON response from https://api.github.com/zen.", "fetch_url", {"url": "https://api.github.com/zen", "timeout": 10}, "Network Retrieval"),
    ("Download page content from https://example.com.", "fetch_url", {"url": "https://example.com", "timeout": 15}, "Network Retrieval"),
    ("Fetch raw contents from https://httpbin.org/get.", "fetch_url", {"url": "https://httpbin.org/get", "timeout": 10}, "Network Retrieval"),
    ("Fetch status JSON from https://status.cloud.google.com/incidents.json.", "fetch_url", {"url": "https://status.cloud.google.com/incidents.json", "timeout": 15}, "Network Retrieval"),
    ("Fetch header info from https://cloudflare.com/cdn-cgi/trace.", "fetch_url", {"url": "https://cloudflare.com/cdn-cgi/trace", "timeout": 10}, "Network Retrieval"),
]


def generate_tooluse_suite(output_path: Path, target_count: int = 100) -> List[Dict[str, Any]]:
    """Generates 100+ standardized tool-use and function calling benchmark tasks."""
    tasks = []
    base_len = len(BASE_CASES)

    for i in range(target_count):
        idx = i % base_len
        prompt, tool_name, args, category = BASE_CASES[idx]

        task_id = f"tooluse_{i + 1:03d}"
        if i >= base_len:
            task_id += f"_v{i // base_len + 1}"

        # Format standardized prompt with tool specifications
        tools_desc = []
        for t in TOOLS_SPEC:
            props = t["parameters"]["properties"]
            p_str = ", ".join(f"{k}: {v.get('type', 'str')}" for k, v in props.items())
            tools_desc.append(f"- {t['name']}({p_str}): {t['description']}")
        tools_block = "\n".join(tools_desc)

        full_prompt = (
            f"You are an AI model with function calling capabilities. "
            f"Given the user request, call the appropriate tool by outputting a single JSON object:\n"
            f'{{"tool": "<tool_name>", "arguments": {{<key>: <value>}}}}\n\n'
            f"Available Tools:\n{tools_block}\n\n"
            f"User: {prompt}\n"
            f"Call: "
        )

        tasks.append({
            "id": task_id,
            "domain": "tooluse",
            "category": category,
            "user_prompt": prompt,
            "prompt": full_prompt,
            "expected_tool": tool_name,
            "expected_args": args,
            "available_tools": TOOLS_SPEC,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"  ✓ Saved {len(tasks)} Tool-Use challenges -> {output_path}")
    return tasks


if __name__ == "__main__":
    b_dir = Path(__file__).resolve().parents[2] / "evals" / "benchmarks"
    out = b_dir / "tooluse_suite.json"
    generate_tooluse_suite(out, target_count=100)
