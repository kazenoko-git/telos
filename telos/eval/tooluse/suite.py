"""
Standardized Tool-Use & Function Calling Benchmark Suite for Télos.

Contains tool schemas, query prompts, and canonical ground-truth tool invocations across:
1. Mathematical & Scientific Computing (`calculator`)
2. Information Retrieval (`web_search`)
3. Environmental & Weather Information (`get_weather`)
4. Filesystem & Code Repository Navigation (`file_search`)
5. Communication & Notification (`send_email`)
6. Structured Data Querying (`database_query`)
7. System Execution (`run_command`)
"""

from typing import List, Dict, Any

STANDARD_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "calculator",
        "description": "Calculates the mathematical result of an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression (e.g. '24 * 7 + 15')"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "web_search",
        "description": "Searches the web for relevant documents matching a query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search keywords or query."},
                "max_results": {"type": "integer", "description": "Maximum number of search results to return.", "default": 5}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_weather",
        "description": "Retrieves the current weather and forecast for a specified city or location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state/country name (e.g. 'San Francisco, CA')."},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "file_search",
        "description": "Searches for files matching a filename glob pattern or path.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py', 'model.pt')."},
                "path": {"type": "string", "description": "Root directory path to search within.", "default": "."}
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "send_email",
        "description": "Sends an email message to a specified recipient address.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Subject line of the email."},
                "body": {"type": "string", "description": "Body text of the message."}
            },
            "required": ["recipient", "subject", "body"]
        }
    },
    {
        "name": "database_query",
        "description": "Queries a database table with filter conditions.",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Name of the database table."},
                "filters": {"type": "object", "description": "Key-value dictionary of column filter criteria."}
            },
            "required": ["table"]
        }
    },
    {
        "name": "run_command",
        "description": "Executes a shell command in an isolated environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command line to execute."}
            },
            "required": ["command"]
        }
    }
]

TOOLUSE_BENCHMARK_SUITE: List[Dict[str, Any]] = [
    # 1. Calculator
    {
        "id": "tool_001",
        "category": "Math & Calculation",
        "user_prompt": "What is 458 multiplied by 32?",
        "expected_tool": "calculator",
        "expected_args": {"expression": "458 * 32"},
    },
    {
        "id": "tool_002",
        "category": "Math & Calculation",
        "user_prompt": "Compute 15% tip on a bill of $84.50.",
        "expected_tool": "calculator",
        "expected_args": {"expression": "84.50 * 0.15"},
    },
    {
        "id": "tool_003",
        "category": "Math & Calculation",
        "user_prompt": "Calculate 2 to the power of 16.",
        "expected_tool": "calculator",
        "expected_args": {"expression": "2 ** 16"},
    },
    {
        "id": "tool_004",
        "category": "Math & Calculation",
        "user_prompt": "Find the square root of 144 plus 25.",
        "expected_tool": "calculator",
        "expected_args": {"expression": "144 ** 0.5 + 25"},
    },
    {
        "id": "tool_005",
        "category": "Math & Calculation",
        "user_prompt": "Calculate (120 - 45) / 5.",
        "expected_tool": "calculator",
        "expected_args": {"expression": "(120 - 45) / 5"},
    },

    # 2. Weather
    {
        "id": "tool_006",
        "category": "Weather & Environment",
        "user_prompt": "What is the weather like in Tokyo right now?",
        "expected_tool": "get_weather",
        "expected_args": {"location": "Tokyo", "unit": "celsius"},
    },
    {
        "id": "tool_007",
        "category": "Weather & Environment",
        "user_prompt": "Give me the current temperature in New York in fahrenheit.",
        "expected_tool": "get_weather",
        "expected_args": {"location": "New York", "unit": "fahrenheit"},
    },
    {
        "id": "tool_008",
        "category": "Weather & Environment",
        "user_prompt": "Is it raining in London today?",
        "expected_tool": "get_weather",
        "expected_args": {"location": "London", "unit": "celsius"},
    },
    {
        "id": "tool_009",
        "category": "Weather & Environment",
        "user_prompt": "Check the temperature in Paris, France.",
        "expected_tool": "get_weather",
        "expected_args": {"location": "Paris, France", "unit": "celsius"},
    },
    {
        "id": "tool_010",
        "category": "Weather & Environment",
        "user_prompt": "Check the weather forecast for Sydney, Australia in celsius.",
        "expected_tool": "get_weather",
        "expected_args": {"location": "Sydney, Australia", "unit": "celsius"},
    },

    # 3. Web Search
    {
        "id": "tool_011",
        "category": "Information Retrieval",
        "user_prompt": "Search the web for the latest Nobel Prize winners in physics.",
        "expected_tool": "web_search",
        "expected_args": {"query": "latest Nobel Prize winners in physics"},
    },
    {
        "id": "tool_012",
        "category": "Information Retrieval",
        "user_prompt": "Find recent research papers on discrete diffusion models for code generation.",
        "expected_tool": "web_search",
        "expected_args": {"query": "discrete diffusion models code generation"},
    },
    {
        "id": "tool_013",
        "category": "Information Retrieval",
        "user_prompt": "Who won the most recent FIFA World Cup?",
        "expected_tool": "web_search",
        "expected_args": {"query": "most recent FIFA World Cup winner"},
    },
    {
        "id": "tool_014",
        "category": "Information Retrieval",
        "user_prompt": "Search for Python 3.12 release highlights and breaking changes.",
        "expected_tool": "web_search",
        "expected_args": {"query": "Python 3.12 release highlights breaking changes"},
    },
    {
        "id": "tool_015",
        "category": "Information Retrieval",
        "user_prompt": "Look up the distance from Earth to Mars in kilometers.",
        "expected_tool": "web_search",
        "expected_args": {"query": "distance from Earth to Mars kilometers"},
    },

    # 4. File Search
    {
        "id": "tool_016",
        "category": "Filesystem Operations",
        "user_prompt": "Find all Python test files in the project.",
        "expected_tool": "file_search",
        "expected_args": {"pattern": "test_*.py", "path": "."},
    },
    {
        "id": "tool_017",
        "category": "Filesystem Operations",
        "user_prompt": "Search for any saved safetensors checkpoint files in the checkpoints directory.",
        "expected_tool": "file_search",
        "expected_args": {"pattern": "*.safetensors", "path": "checkpoints"},
    },
    {
        "id": "tool_018",
        "category": "Filesystem Operations",
        "user_prompt": "Locate tokenizer.json in the configs folder.",
        "expected_tool": "file_search",
        "expected_args": {"pattern": "tokenizer.json", "path": "configs"},
    },
    {
        "id": "tool_019",
        "category": "Filesystem Operations",
        "user_prompt": "Find all YAML configuration files across the repository.",
        "expected_tool": "file_search",
        "expected_args": {"pattern": "*.yaml", "path": "."},
    },
    {
        "id": "tool_020",
        "category": "Filesystem Operations",
        "user_prompt": "Find markdown files in the docs directory.",
        "expected_tool": "file_search",
        "expected_args": {"pattern": "*.md", "path": "docs"},
    },

    # 5. Send Email
    {
        "id": "tool_021",
        "category": "Communication",
        "user_prompt": "Send an email to alice@example.com with subject 'Meeting Rescheduled' saying 'The team meeting is moved to 3pm.'",
        "expected_tool": "send_email",
        "expected_args": {"recipient": "alice@example.com", "subject": "Meeting Rescheduled", "body": "The team meeting is moved to 3pm."},
    },
    {
        "id": "tool_022",
        "category": "Communication",
        "user_prompt": "Email bob@telos.ai saying 'Code review requested' with body 'Please review pull request 42.'",
        "expected_tool": "send_email",
        "expected_args": {"recipient": "bob@telos.ai", "subject": "Code review requested", "body": "Please review pull request 42."},
    },
    {
        "id": "tool_023",
        "category": "Communication",
        "user_prompt": "Send a status update email to dev-team@company.org subject 'Sprint 10 Complete' body 'All tasks are deployed.'",
        "expected_tool": "send_email",
        "expected_args": {"recipient": "dev-team@company.org", "subject": "Sprint 10 Complete", "body": "All tasks are deployed."},
    },

    # 6. Database Query
    {
        "id": "tool_024",
        "category": "Database Operations",
        "user_prompt": "Query the 'users' table for all active accounts with role 'admin'.",
        "expected_tool": "database_query",
        "expected_args": {"table": "users", "filters": {"status": "active", "role": "admin"}},
    },
    {
        "id": "tool_025",
        "category": "Database Operations",
        "user_prompt": "Look up orders in table 'orders' where customer_id is 1042.",
        "expected_tool": "database_query",
        "expected_args": {"table": "orders", "filters": {"customer_id": 1042}},
    },
    {
        "id": "tool_026",
        "category": "Database Operations",
        "user_prompt": "Get all records from table 'products' with category 'hardware'.",
        "expected_tool": "database_query",
        "expected_args": {"table": "products", "filters": {"category": "hardware"}},
    },

    # 7. Shell Command Execution
    {
        "id": "tool_027",
        "category": "System Execution",
        "user_prompt": "Run pytest on the test_eval_suite.py file.",
        "expected_tool": "run_command",
        "expected_args": {"command": "pytest tests/test_eval_suite.py"},
    },
    {
        "id": "tool_028",
        "category": "System Execution",
        "user_prompt": "Check git status in the current repository.",
        "expected_tool": "run_command",
        "expected_args": {"command": "git status"},
    },
    {
        "id": "tool_029",
        "category": "System Execution",
        "user_prompt": "List files with human readable sizes in the data directory.",
        "expected_tool": "run_command",
        "expected_args": {"command": "ls -lh data"},
    },
    {
        "id": "tool_030",
        "category": "System Execution",
        "user_prompt": "Show the disk usage of the current workspace.",
        "expected_tool": "run_command",
        "expected_args": {"command": "df -h ."},
    },
]


def load_tooluse_suite(num_tasks: int = 30) -> List[Dict[str, Any]]:
    """
    Loads tool-use benchmark tasks for Télos evaluation.
    
    Args:
        num_tasks: Maximum number of tasks to return.
        
    Returns:
        List of tool-use benchmark task dictionaries.
    """
    if num_tasks and num_tasks < len(TOOLUSE_BENCHMARK_SUITE):
        return TOOLUSE_BENCHMARK_SUITE[:num_tasks]
    return TOOLUSE_BENCHMARK_SUITE
