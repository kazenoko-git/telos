"""
Contextual Probes Suite for Télos Models.
Contains 100 benchmark prompts across 8 syntactic code categories:
1. Identifier recovery (13)
2. Function names (13)
3. Keywords (13)
4. Operators (12)
5. Literals (12)
6. Imports (13)
7. Class names (12)
8. Attribute names (13)
"""

PROBE_SUITE_100 = [
    # --- 1. Identifier Recovery (13) ---
    {"category": "Identifier recovery", "prompt": "return a +", "target": "b", "target_bpe": "b"},
    {"category": "Identifier recovery", "prompt": "x = 10\nprint(", "target": "x", "target_bpe": "x"},
    {"category": "Identifier recovery", "prompt": "def __init__(self,", "target": "name", "target_bpe": "Ġname"},
    {"category": "Identifier recovery", "prompt": "self.name =", "target": "name", "target_bpe": "Ġname"},
    {"category": "Identifier recovery", "prompt": "total = sum(", "target": "items", "target_bpe": "Ġitems"},
    {"category": "Identifier recovery", "prompt": "for elem in", "target": "lst", "target_bpe": "Ġlst"},
    {"category": "Identifier recovery", "prompt": "res = val *", "target": "factor", "target_bpe": "Ġfactor"},
    {"category": "Identifier recovery", "prompt": "msg = str(", "target": "err", "target_bpe": "Ġerr"},
    {"category": "Identifier recovery", "prompt": "dx = x2 -", "target": "x1", "target_bpe": "Ġx1"},
    {"category": "Identifier recovery", "prompt": "data = json.loads(", "target": "text", "target_bpe": "Ġtext"},
    {"category": "Identifier recovery", "prompt": "res = []\nfor x in", "target": "items", "target_bpe": "Ġitems"},
    {"category": "Identifier recovery", "prompt": "left +", "target": "right", "target_bpe": "Ġright"},
    {"category": "Identifier recovery", "prompt": "width *", "target": "height", "target_bpe": "Ġheight"},

    # --- 2. Function Names (13) ---
    {"category": "Function names", "prompt": "def get_", "target": "name", "target_bpe": "name"},
    {"category": "Function names", "prompt": "def set_", "target": "val", "target_bpe": "val"},
    {"category": "Function names", "prompt": "def parse_", "target": "data", "target_bpe": "data"},
    {"category": "Function names", "prompt": "def build_", "target": "model", "target_bpe": "model"},
    {"category": "Function names", "prompt": "def test_", "target": "func", "target_bpe": "func"},
    {"category": "Function names", "prompt": "def process_", "target": "request", "target_bpe": "request"},
    {"category": "Function names", "prompt": "def validate_", "target": "input", "target_bpe": "input"},
    {"category": "Function names", "prompt": "def load_", "target": "config", "target_bpe": "config"},
    {"category": "Function names", "prompt": "def save_", "target": "file", "target_bpe": "file"},
    {"category": "Function names", "prompt": "def calculate_", "target": "total", "target_bpe": "total"},
    {"category": "Function names", "prompt": "def convert_", "target": "type", "target_bpe": "type"},
    {"category": "Function names", "prompt": "def read_", "target": "bytes", "target_bpe": "bytes"},
    {"category": "Function names", "prompt": "def create_", "target": "instance", "target_bpe": "instance"},

    # --- 3. Keywords (13) ---
    {"category": "Keywords", "prompt": "if x == 1:\n    pass\n", "target": "else", "target_bpe": "else"},
    {"category": "Keywords", "prompt": "try:\n    pass\n", "target": "except", "target_bpe": "except"},
    {"category": "Keywords", "prompt": "for i in", "target": "range", "target_bpe": "Ġrange"},
    {"category": "Keywords", "prompt": "with open(path) ", "target": "as", "target_bpe": "Ġas"},
    {"category": "Keywords", "prompt": "if not", "target": "found", "target_bpe": "Ġfound"},
    {"category": "Keywords", "prompt": "while", "target": "True", "target_bpe": "ĠTrue"},
    {"category": "Keywords", "prompt": "from typing", "target": "import", "target_bpe": "Ġimport"},
    {"category": "Keywords", "prompt": "assert x is", "target": "not", "target_bpe": "Ġnot"},
    {"category": "Keywords", "prompt": "if item", "target": "in", "target_bpe": "Ġin"},
    {"category": "Keywords", "prompt": "def func():\n   ", "target": "return", "target_bpe": "Ġreturn"},
    {"category": "Keywords", "prompt": "raise ValueError(", "target": "msg", "target_bpe": "msg"},
    {"category": "Keywords", "prompt": "class MyClass(", "target": "object", "target_bpe": "object"},
    {"category": "Keywords", "prompt": "yield", "target": "from", "target_bpe": "Ġfrom"},

    # --- 4. Operators (12) ---
    {"category": "Operators", "prompt": "x = a +", "target": "b", "target_bpe": "Ġb"},
    {"category": "Operators", "prompt": "if a ==", "target": "b", "target_bpe": "Ġb"},
    {"category": "Operators", "prompt": "x +=", "target": "1", "target_bpe": "Ġ1"},
    {"category": "Operators", "prompt": "a >", "target": "0", "target_bpe": "Ġ0"},
    {"category": "Operators", "prompt": "x = y *", "target": "z", "target_bpe": "Ġz"},
    {"category": "Operators", "prompt": "a !=", "target": "None", "target_bpe": "ĠNone"},
    {"category": "Operators", "prompt": "count = len(arr) -", "target": "1", "target_bpe": "Ġ1"},
    {"category": "Operators", "prompt": "if x <=", "target": "max_val", "target_bpe": "Ġmax_val"},
    {"category": "Operators", "prompt": "idx = (i + 1) %", "target": "n", "target_bpe": "Ġn"},
    {"category": "Operators", "prompt": "res = a &", "target": "b", "target_bpe": "Ġb"},
    {"category": "Operators", "prompt": "flags = A |", "target": "B", "target_bpe": "ĠB"},
    {"category": "Operators", "prompt": "val = x **", "target": "2", "target_bpe": "Ġ2"},

    # --- 5. Literals (12) ---
    {"category": "Literals", "prompt": "if x is", "target": "None", "target_bpe": "ĠNone"},
    {"category": "Literals", "prompt": "flag =", "target": "True", "target_bpe": "ĠTrue"},
    {"category": "Literals", "prompt": "status =", "target": "False", "target_bpe": "ĠFalse"},
    {"category": "Literals", "prompt": "count =", "target": "0", "target_bpe": "Ġ0"},
    {"category": "Literals", "prompt": "name =", "target": "\"\"", "target_bpe": "Ġ\"\""},
    {"category": "Literals", "prompt": "items =", "target": "[]", "target_bpe": "Ġ[]"},
    {"category": "Literals", "prompt": "data =", "target": "{}", "target_bpe": "Ġ{}"},
    {"category": "Literals", "prompt": "rate =", "target": "0.0", "target_bpe": "Ġ0.0"},
    {"category": "Literals", "prompt": "idx =", "target": "-1", "target_bpe": "Ġ-1"},
    {"category": "Literals", "prompt": "pi =", "target": "3.14", "target_bpe": "Ġ3.14"},
    {"category": "Literals", "prompt": "res =", "target": "1", "target_bpe": "Ġ1"},
    {"category": "Literals", "prompt": "msg =", "target": "\"hello\"", "target_bpe": "Ġ\"hello\""},

    # --- 6. Imports (13) ---
    {"category": "Imports", "prompt": "import", "target": "os", "target_bpe": "Ġos"},
    {"category": "Imports", "prompt": "import", "target": "sys", "target_bpe": "Ġsys"},
    {"category": "Imports", "prompt": "import", "target": "json", "target_bpe": "Ġjson"},
    {"category": "Imports", "prompt": "import", "target": "time", "target_bpe": "Ġtime"},
    {"category": "Imports", "prompt": "import", "target": "math", "target_bpe": "Ġmath"},
    {"category": "Imports", "prompt": "import", "target": "re", "target_bpe": "Ġre"},
    {"category": "Imports", "prompt": "import", "target": "random", "target_bpe": "Ġrandom"},
    {"category": "Imports", "prompt": "from typing import", "target": "List", "target_bpe": "ĠList"},
    {"category": "Imports", "prompt": "from pathlib import", "target": "Path", "target_bpe": "ĠPath"},
    {"category": "Imports", "prompt": "import numpy as", "target": "np", "target_bpe": "Ġnp"},
    {"category": "Imports", "prompt": "import torch.nn as", "target": "nn", "target_bpe": "Ġnn"},
    {"category": "Imports", "prompt": "from collections import", "target": "defaultdict", "target_bpe": "Ġdefaultdict"},
    {"category": "Imports", "prompt": "import logging", "target": "as", "target_bpe": "Ġas"},

    # --- 7. Class Names (12) ---
    {"category": "Class names", "prompt": "class", "target": "Base", "target_bpe": "ĠBase"},
    {"category": "Class names", "prompt": "class", "target": "Model", "target_bpe": "ĠModel"},
    {"category": "Class names", "prompt": "class", "target": "Config", "target_bpe": "ĠConfig"},
    {"category": "Class names", "prompt": "class", "target": "Trainer", "target_bpe": "ĠTrainer"},
    {"category": "Class names", "prompt": "class", "target": "User", "target_bpe": "ĠUser"},
    {"category": "Class names", "prompt": "class", "target": "Dataset", "target_bpe": "ĠDataset"},
    {"category": "Class names", "prompt": "class", "target": "Engine", "target_bpe": "ĠEngine"},
    {"category": "Class names", "prompt": "class", "target": "Handler", "target_bpe": "ĠHandler"},
    {"category": "Class names", "prompt": "class", "target": "Session", "target_bpe": "ĠSession"},
    {"category": "Class names", "prompt": "class", "target": "Exception", "target_bpe": "ĠException"},
    {"category": "Class names", "prompt": "class", "target": "Node", "target_bpe": "ĠNode"},
    {"category": "Class names", "prompt": "class", "target": "Server", "target_bpe": "ĠServer"},

    # --- 8. Attribute Names (13) ---
    {"category": "Attribute names", "prompt": "self.", "target": "name", "target_bpe": "name"},
    {"category": "Attribute names", "prompt": "self.", "target": "value", "target_bpe": "value"},
    {"category": "Attribute names", "prompt": "self.", "target": "config", "target_bpe": "config"},
    {"category": "Attribute names", "prompt": "self.", "target": "device", "target_bpe": "device"},
    {"category": "Attribute names", "prompt": "self.", "target": "logger", "target_bpe": "logger"},
    {"category": "Attribute names", "prompt": "self.", "target": "state", "target_bpe": "state"},
    {"category": "Attribute names", "prompt": "obj.", "target": "data", "target_bpe": "data"},
    {"category": "Attribute names", "prompt": "req.", "target": "json", "target_bpe": "json"},
    {"category": "Attribute names", "prompt": "path.", "target": "exists", "target_bpe": "exists"},
    {"category": "Attribute names", "prompt": "res.", "target": "status_code", "target_bpe": "status_code"},
    {"category": "Attribute names", "prompt": "torch.", "target": "cuda", "target_bpe": "cuda"},
    {"category": "Attribute names", "prompt": "os.", "target": "path", "target_bpe": "path"},
    {"category": "Attribute names", "prompt": "sys.", "target": "path", "target_bpe": "path"},
]
