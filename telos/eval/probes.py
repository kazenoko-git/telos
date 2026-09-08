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

Each probe contains:
- category: Syntactic category name
- prompt: Prefix context (for backward compatibility and causal evaluation)
- prefix: Alias for prompt
- target: Target token string to predict
- target_bpe: Expected BPE token representation
- suffix: Subsequent context enabling realistic bidirectional [MASK] infilling
"""

PROBE_SUITE_100 = [
    # --- 1. Identifier Recovery (13) ---
    {"category": "Identifier recovery", "prompt": "return a +", "prefix": "return a +", "target": "b", "target_bpe": "b", "suffix": "\n"},
    {"category": "Identifier recovery", "prompt": "x = 10\nprint(", "prefix": "x = 10\nprint(", "target": "x", "target_bpe": "x", "suffix": ")\n"},
    {"category": "Identifier recovery", "prompt": "def __init__(self,", "prefix": "def __init__(self,", "target": "name", "target_bpe": "Ġname", "suffix": "):\n        self.name = name"},
    {"category": "Identifier recovery", "prompt": "self.name =", "prefix": "self.name =", "target": "name", "target_bpe": "Ġname", "suffix": "\n        self.value = value"},
    {"category": "Identifier recovery", "prompt": "total = sum(", "prefix": "total = sum(", "target": "items", "target_bpe": "Ġitems", "suffix": ")\n"},
    {"category": "Identifier recovery", "prompt": "for elem in", "prefix": "for elem in", "target": "lst", "target_bpe": "Ġlst", "suffix": ":\n        process(elem)"},
    {"category": "Identifier recovery", "prompt": "res = val *", "prefix": "res = val *", "target": "factor", "target_bpe": "Ġfactor", "suffix": "\n"},
    {"category": "Identifier recovery", "prompt": "msg = str(", "prefix": "msg = str(", "target": "err", "target_bpe": "Ġerr", "suffix": ")\n"},
    {"category": "Identifier recovery", "prompt": "dx = x2 -", "prefix": "dx = x2 -", "target": "x1", "target_bpe": "Ġx1", "suffix": "\n"},
    {"category": "Identifier recovery", "prompt": "data = json.loads(", "prefix": "data = json.loads(", "target": "text", "target_bpe": "Ġtext", "suffix": ")\n"},
    {"category": "Identifier recovery", "prompt": "res = []\nfor x in", "prefix": "res = []\nfor x in", "target": "items", "target_bpe": "Ġitems", "suffix": ":\n    res.append(x)"},
    {"category": "Identifier recovery", "prompt": "left +", "prefix": "left +", "target": "right", "target_bpe": "Ġright", "suffix": "\n"},
    {"category": "Identifier recovery", "prompt": "width *", "prefix": "width *", "target": "height", "target_bpe": "Ġheight", "suffix": "\n"},

    # --- 2. Function Names (13) ---
    {"category": "Function names", "prompt": "def get_", "prefix": "def get_", "target": "name", "target_bpe": "name", "suffix": "(self):\n    return self._name"},
    {"category": "Function names", "prompt": "def set_", "prefix": "def set_", "target": "val", "target_bpe": "val", "suffix": "(self, val):\n    self._val = val"},
    {"category": "Function names", "prompt": "def parse_", "prefix": "def parse_", "target": "data", "target_bpe": "data", "suffix": "(raw_str):\n    return json.loads(raw_str)"},
    {"category": "Function names", "prompt": "def build_", "prefix": "def build_", "target": "model", "target_bpe": "model", "suffix": "(config):\n    return Model(config)"},
    {"category": "Function names", "prompt": "def test_", "prefix": "def test_", "target": "func", "target_bpe": "func", "suffix": "():\n    assert True"},
    {"category": "Function names", "prompt": "def process_", "prefix": "def process_", "target": "request", "target_bpe": "request", "suffix": "(req):\n    return req.json()"},
    {"category": "Function names", "prompt": "def validate_", "prefix": "def validate_", "target": "input", "target_bpe": "input", "suffix": "(data):\n    assert data is not None"},
    {"category": "Function names", "prompt": "def load_", "prefix": "def load_", "target": "config", "target_bpe": "config", "suffix": "(path):\n    with open(path) as f:\n        return yaml.safe_load(f)"},
    {"category": "Function names", "prompt": "def save_", "prefix": "def save_", "target": "file", "target_bpe": "file", "suffix": "(path, content):\n    with open(path, 'w') as f:\n        f.write(content)"},
    {"category": "Function names", "prompt": "def calculate_", "prefix": "def calculate_", "target": "total", "target_bpe": "total", "suffix": "(items):\n    return sum(items)"},
    {"category": "Function names", "prompt": "def convert_", "prefix": "def convert_", "target": "type", "target_bpe": "type", "suffix": "(val, target_type):\n    return target_type(val)"},
    {"category": "Function names", "prompt": "def read_", "prefix": "def read_", "target": "bytes", "target_bpe": "bytes", "suffix": "(path):\n    with open(path, 'rb') as f:\n        return f.read()"},
    {"category": "Function names", "prompt": "def create_", "prefix": "def create_", "target": "instance", "target_bpe": "instance", "suffix": "(cls, *args):\n    return cls(*args)"},

    # --- 3. Keywords (13) ---
    {"category": "Keywords", "prompt": "if x == 1:\n    pass\n", "prefix": "if x == 1:\n    pass\n", "target": "else", "target_bpe": "else", "suffix": ":\n    pass"},
    {"category": "Keywords", "prompt": "try:\n    pass\n", "prefix": "try:\n    pass\n", "target": "except", "target_bpe": "except", "suffix": " Exception:\n    pass"},
    {"category": "Keywords", "prompt": "for i in", "prefix": "for i in", "target": "range", "target_bpe": "Ġrange", "suffix": "(10):\n    print(i)"},
    {"category": "Keywords", "prompt": "with open(path) ", "prefix": "with open(path) ", "target": "as", "target_bpe": "Ġas", "suffix": " f:\n    data = f.read()"},
    {"category": "Keywords", "prompt": "if not", "prefix": "if not", "target": "found", "target_bpe": "Ġfound", "suffix": ":\n    raise KeyError('Not found')"},
    {"category": "Keywords", "prompt": "while", "prefix": "while", "target": "True", "target_bpe": "ĠTrue", "suffix": ":\n    break"},
    {"category": "Keywords", "prompt": "from typing", "prefix": "from typing", "target": "import", "target_bpe": "Ġimport", "suffix": " List, Dict, Optional"},
    {"category": "Keywords", "prompt": "assert x is", "prefix": "assert x is", "target": "not", "target_bpe": "Ġnot", "suffix": " None"},
    {"category": "Keywords", "prompt": "if item", "prefix": "if item", "target": "in", "target_bpe": "Ġin", "suffix": " items:\n    return True"},
    {"category": "Keywords", "prompt": "def func():\n   ", "prefix": "def func():\n   ", "target": "return", "target_bpe": "Ġreturn", "suffix": " 42"},
    {"category": "Keywords", "prompt": "raise ValueError(", "prefix": "raise ValueError(", "target": "msg", "target_bpe": "msg", "suffix": ")\n"},
    {"category": "Keywords", "prompt": "class MyClass(", "prefix": "class MyClass(", "target": "object", "target_bpe": "object", "suffix": "):\n    pass"},
    {"category": "Keywords", "prompt": "yield", "prefix": "yield", "target": "from", "target_bpe": "Ġfrom", "suffix": " sub_generator()"},

    # --- 4. Operators (12) ---
    {"category": "Operators", "prompt": "x = a +", "prefix": "x = a +", "target": "b", "target_bpe": "Ġb", "suffix": "\n"},
    {"category": "Operators", "prompt": "if a ==", "prefix": "if a ==", "target": "b", "target_bpe": "Ġb", "suffix": ":\n    return True"},
    {"category": "Operators", "prompt": "x +=", "prefix": "x +=", "target": "1", "target_bpe": "Ġ1", "suffix": "\n"},
    {"category": "Operators", "prompt": "a >", "prefix": "a >", "target": "0", "target_bpe": "Ġ0", "suffix": "\n"},
    {"category": "Operators", "prompt": "x = y *", "prefix": "x = y *", "target": "z", "target_bpe": "Ġz", "suffix": "\n"},
    {"category": "Operators", "prompt": "a !=", "prefix": "a !=", "target": "None", "target_bpe": "ĠNone", "suffix": ":\n    return a"},
    {"category": "Operators", "prompt": "count = len(arr) -", "prefix": "count = len(arr) -", "target": "1", "target_bpe": "Ġ1", "suffix": "\n"},
    {"category": "Operators", "prompt": "if x <=", "prefix": "if x <=", "target": "max_val", "target_bpe": "Ġmax_val", "suffix": ":\n    return x"},
    {"category": "Operators", "prompt": "idx = (i + 1) %", "prefix": "idx = (i + 1) %", "target": "n", "target_bpe": "Ġn", "suffix": "\n"},
    {"category": "Operators", "prompt": "res = a &", "prefix": "res = a &", "target": "b", "target_bpe": "Ġb", "suffix": "\n"},
    {"category": "Operators", "prompt": "flags = A |", "prefix": "flags = A |", "target": "B", "target_bpe": "ĠB", "suffix": "\n"},
    {"category": "Operators", "prompt": "val = x **", "prefix": "val = x **", "target": "2", "target_bpe": "Ġ2", "suffix": "\n"},

    # --- 5. Literals (12) ---
    {"category": "Literals", "prompt": "if x is", "prefix": "if x is", "target": "None", "target_bpe": "ĠNone", "suffix": ":\n    return None"},
    {"category": "Literals", "prompt": "flag =", "prefix": "flag =", "target": "True", "target_bpe": "ĠTrue", "suffix": "\n"},
    {"category": "Literals", "prompt": "status =", "prefix": "status =", "target": "False", "target_bpe": "ĠFalse", "suffix": "\n"},
    {"category": "Literals", "prompt": "count =", "prefix": "count =", "target": "0", "target_bpe": "Ġ0", "suffix": "\n"},
    {"category": "Literals", "prompt": "name =", "prefix": "name =", "target": "\"\"", "target_bpe": "Ġ\"\"", "suffix": "\n"},
    {"category": "Literals", "prompt": "items =", "prefix": "items =", "target": "[]", "target_bpe": "Ġ[]", "suffix": "\n"},
    {"category": "Literals", "prompt": "data =", "prefix": "data =", "target": "{}", "target_bpe": "Ġ{}", "suffix": "\n"},
    {"category": "Literals", "prompt": "rate =", "prefix": "rate =", "target": "0.0", "target_bpe": "Ġ0.0", "suffix": "\n"},
    {"category": "Literals", "prompt": "idx =", "prefix": "idx =", "target": "-1", "target_bpe": "Ġ-1", "suffix": "\n"},
    {"category": "Literals", "prompt": "pi =", "prefix": "pi =", "target": "3.14", "target_bpe": "Ġ3.14", "suffix": "\n"},
    {"category": "Literals", "prompt": "res =", "prefix": "res =", "target": "1", "target_bpe": "Ġ1", "suffix": "\n"},
    {"category": "Literals", "prompt": "msg =", "prefix": "msg =", "target": "\"hello\"", "target_bpe": "Ġ\"hello\"", "suffix": "\n"},

    # --- 6. Imports (13) ---
    {"category": "Imports", "prompt": "import", "prefix": "import", "target": "os", "target_bpe": "Ġos", "suffix": "\n\npath = os.path.join(base, name)"},
    {"category": "Imports", "prompt": "import", "prefix": "import", "target": "sys", "target_bpe": "Ġsys", "suffix": "\n\nsys.exit(0)"},
    {"category": "Imports", "prompt": "import", "prefix": "import", "target": "json", "target_bpe": "Ġjson", "suffix": "\n\ndata = json.loads(text)"},
    {"category": "Imports", "prompt": "import", "prefix": "import", "target": "time", "target_bpe": "Ġtime", "suffix": "\n\nstart = time.time()"},
    {"category": "Imports", "prompt": "import", "prefix": "import", "target": "math", "target_bpe": "Ġmath", "suffix": "\n\nx = math.sqrt(val)"},
    {"category": "Imports", "prompt": "import", "prefix": "import", "target": "re", "target_bpe": "Ġre", "suffix": "\n\npattern = re.compile(r\"\\d+\")"},
    {"category": "Imports", "prompt": "import", "prefix": "import", "target": "random", "target_bpe": "Ġrandom", "suffix": "\n\nchoice = random.choice(items)"},
    {"category": "Imports", "prompt": "from typing import", "prefix": "from typing import", "target": "List", "target_bpe": "ĠList", "suffix": ", Dict, Optional"},
    {"category": "Imports", "prompt": "from pathlib import", "prefix": "from pathlib import", "target": "Path", "target_bpe": "ĠPath", "suffix": "\n\np = Path('data.txt')"},
    {"category": "Imports", "prompt": "import numpy as", "prefix": "import numpy as", "target": "np", "target_bpe": "Ġnp", "suffix": "\n\narr = np.zeros(10)"},
    {"category": "Imports", "prompt": "import torch.nn as", "prefix": "import torch.nn as", "target": "nn", "target_bpe": "Ġnn", "suffix": "\n\nclass Net(nn.Module): pass"},
    {"category": "Imports", "prompt": "from collections import", "prefix": "from collections import", "target": "defaultdict", "target_bpe": "Ġdefaultdict", "suffix": "\n\nd = defaultdict(list)"},
    {"category": "Imports", "prompt": "import logging", "prefix": "import logging", "target": "as", "target_bpe": "Ġas", "suffix": " log\nlog.basicConfig()"},

    # --- 7. Class Names (12) ---
    {"category": "Class names", "prompt": "class", "prefix": "class", "target": "Base", "target_bpe": "ĠBase", "suffix": ":\n    def __init__(self):\n        pass"},
    {"category": "Class names", "prompt": "class", "prefix": "class", "target": "Model", "target_bpe": "ĠModel", "suffix": "(nn.Module):\n    def forward(self, x):\n        return x"},
    {"category": "Class names", "prompt": "class", "prefix": "class", "target": "Config", "target_bpe": "ĠConfig", "suffix": ":\n    def __init__(self, lr=1e-3):\n        self.lr = lr"},
    {"category": "Class names", "prompt": "class", "prefix": "class", "target": "Trainer", "target_bpe": "ĠTrainer", "suffix": ":\n    def train(self, steps):\n        pass"},
    {"category": "Class names", "prompt": "class", "prefix": "class", "target": "User", "target_bpe": "ĠUser", "suffix": ":\n    def __init__(self, username, email):\n        pass"},
    {"category": "Class names", "prompt": "class", "prefix": "class", "target": "Dataset", "target_bpe": "ĠDataset", "suffix": ":\n    def __len__(self):\n        return len(self.data)"},
    {"category": "Class names", "prompt": "class", "prefix": "class", "target": "Engine", "target_bpe": "ĠEngine", "suffix": ":\n    def step(self):\n        pass"},
    {"category": "Class names", "prompt": "class", "prefix": "class", "target": "Handler", "target_bpe": "ĠHandler", "suffix": ":\n    def handle(self, request):\n        pass"},
    {"category": "Class names", "prompt": "class", "prefix": "class", "target": "Session", "target_bpe": "ĠSession", "suffix": ":\n    def close(self):\n        pass"},
    {"category": "Class names", "prompt": "class", "prefix": "class", "target": "Exception", "target_bpe": "ĠException", "suffix": "(BaseException):\n    pass"},
    {"category": "Class names", "prompt": "class", "prefix": "class", "target": "Node", "target_bpe": "ĠNode", "suffix": ":\n    def __init__(self, val, next=None):\n        self.val = val"},
    {"category": "Class names", "prompt": "class", "prefix": "class", "target": "Server", "target_bpe": "ĠServer", "suffix": ":\n    def listen(self, port=8080):\n        pass"},

    # --- 8. Attribute Names (13) ---
    {"category": "Attribute names", "prompt": "self.", "prefix": "self.", "target": "name", "target_bpe": "name", "suffix": " = name\n        self.age = age"},
    {"category": "Attribute names", "prompt": "self.", "prefix": "self.", "target": "value", "target_bpe": "value", "suffix": " = value\n        return self.value"},
    {"category": "Attribute names", "prompt": "self.", "prefix": "self.", "target": "config", "target_bpe": "config", "suffix": " = config\n        self.lr = self.config.lr"},
    {"category": "Attribute names", "prompt": "self.", "prefix": "self.", "target": "device", "target_bpe": "device", "suffix": " = device\n        self.model.to(self.device)"},
    {"category": "Attribute names", "prompt": "self.", "prefix": "self.", "target": "logger", "target_bpe": "logger", "suffix": ".info('Starting run')"},
    {"category": "Attribute names", "prompt": "self.", "prefix": "self.", "target": "state", "target_bpe": "state", "suffix": " = 'running'\n        return self.state"},
    {"category": "Attribute names", "prompt": "obj.", "prefix": "obj.", "target": "data", "target_bpe": "data", "suffix": " = [1, 2, 3]"},
    {"category": "Attribute names", "prompt": "req.", "prefix": "req.", "target": "json", "target_bpe": "json", "suffix": "()\n    return response"},
    {"category": "Attribute names", "prompt": "path.", "prefix": "path.", "target": "exists", "target_bpe": "exists", "suffix": "():\n    return True"},
    {"category": "Attribute names", "prompt": "res.", "prefix": "res.", "target": "status_code", "target_bpe": "status_code", "suffix": " == 200:\n    return True"},
    {"category": "Attribute names", "prompt": "torch.", "prefix": "torch.", "target": "cuda", "target_bpe": "cuda", "suffix": ".is_available():\n    device = 'cuda'"},
    {"category": "Attribute names", "prompt": "os.", "prefix": "os.", "target": "path", "target_bpe": "path", "suffix": ".join(root, file)"},
    {"category": "Attribute names", "prompt": "sys.", "prefix": "sys.", "target": "path", "target_bpe": "path", "suffix": ".append(module_dir)"},
]
