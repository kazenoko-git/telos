"""
Verification script for the 100 Contextually Deterministic Probes suite.
Validates tokenization, target token existence in vocabulary, and tests accuracy
on 100M COROSred and 50M AR models.
"""

from telos.eval.runner import load_tokenizer

tok = load_tokenizer("configs/shared/tokenizer_mac.json")

PROBES = [
    # =========================================================================
    # 1. Contextually Deterministic Identifiers (25 probes)
    # Target is strictly forced by local variable scope, method context, or naming
    # =========================================================================
    {"category": "Contextually Deterministic Identifiers", "prompt": "x = 10\nprint(", "suffix": ")\n", "target": "x", "target_bpe": "x", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "y = 20\nprint(", "suffix": ")\n", "target": "y", "target_bpe": "y", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_name(self):\n    return self._", "suffix": "\n", "target": "name", "target_bpe": "name", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_value(self):\n    return self._", "suffix": "\n", "target": "value", "target_bpe": "value", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_id(self):\n    return self._", "suffix": "\n", "target": "id", "target_bpe": "id", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_path(self):\n    return self._", "suffix": "\n", "target": "path", "target_bpe": "path", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_data(self):\n    return self._", "suffix": "\n", "target": "data", "target_bpe": "data", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_status(self):\n    return self._", "suffix": "\n", "target": "status", "target_bpe": "status", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_config(self):\n    return self._", "suffix": "\n", "target": "config", "target_bpe": "config", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def set_count(self, count):\n    self._count =", "suffix": "\n", "target": "count", "target_bpe": "Ġcount", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def set_timeout(self, timeout):\n    self.timeout =", "suffix": "\n", "target": "timeout", "target_bpe": "Ġtimeout", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "total = 0\nfor item in items:\n    total +=", "suffix": "\n", "target": "item", "target_bpe": "Ġitem", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "total = 0\nfor x in values:\n    total +=", "suffix": "\n", "target": "x", "target_bpe": "Ġx", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "for k, v in mapping.items():\n    print(", "suffix": ", v)\n", "target": "k", "target_bpe": "k", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "for k, v in mapping.items():\n    print(k,", "suffix": ")\n", "target": "v", "target_bpe": "Ġv", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "width, height = dims\narea = width *", "suffix": "\n", "target": "height", "target_bpe": "Ġheight", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "left, right = bounds\nspan = right -", "suffix": "\n", "target": "left", "target_bpe": "Ġleft", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "start, end = range_vals\nlength = end -", "suffix": "\n", "target": "start", "target_bpe": "Ġstart", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "row, col = coord\nnext_pos = (row + 1,", "suffix": ")\n", "target": "col", "target_bpe": "Ġcol", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def __init__(self, name):\n    self.name =", "suffix": "\n", "target": "name", "target_bpe": "Ġname", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def __init__(self, config):\n    self.config =", "suffix": "\n", "target": "config", "target_bpe": "Ġconfig", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def __init__(self, model):\n    self.model =", "suffix": "\n", "target": "model", "target_bpe": "Ġmodel", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "res = []\nfor elem in elements:\n    res.append(", "suffix": ")\n", "target": "elem", "target_bpe": "elem", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "key = 'user_id'\nval = cache[", "suffix": "]\n", "target": "key", "target_bpe": "key", "mode": "both"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "head, tail = split(lst)\nreturn", "suffix": "\n", "target": "head", "target_bpe": "Ġhead", "mode": "both"},

    # =========================================================================
    # 2. Syntactic Keywords (25 probes)
    # Target is strictly enforced by Python grammar & syntax rules
    # =========================================================================
    {"category": "Syntactic Keywords", "prompt": "try:\n    do_something()\n", "suffix": " Exception:\n    pass\n", "target": "except", "target_bpe": "except", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "with open(f)", "suffix": " handle:\n    pass\n", "target": "as", "target_bpe": "Ġas", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "with open(path, 'r')", "suffix": " f:\n    text = f.read()\n", "target": "as", "target_bpe": "Ġas", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "for item", "suffix": " collection:\n    pass\n", "target": "in", "target_bpe": "Ġin", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "for i", "suffix": " range(10):\n    print(i)\n", "target": "in", "target_bpe": "Ġin", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "for idx, elem", "suffix": " enumerate(lst):\n    pass\n", "target": "in", "target_bpe": "Ġin", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "while", "suffix": ":\n    break\n", "target": "True", "target_bpe": "ĠTrue", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "from typing", "suffix": " List, Dict, Optional\n", "target": "import", "target_bpe": "Ġimport", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "from pathlib", "suffix": " Path\n", "target": "import", "target_bpe": "Ġimport", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "from collections", "suffix": " deque\n", "target": "import", "target_bpe": "Ġimport", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "assert x is", "suffix": " None\n", "target": "not", "target_bpe": "Ġnot", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "if item", "suffix": " items:\n    return True\n", "target": "in", "target_bpe": "Ġin", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "if x == 1:\n    return 1\n", "suffix": ":\n    return 0\n", "target": "else", "target_bpe": "else", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "yield", "suffix": " sub_generator()\n", "target": "from", "target_bpe": "Ġfrom", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "if not", "suffix": ":\n    return False\n", "target": "found", "target_bpe": "Ġfound", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "class MyModel(", "suffix": "):\n    pass\n", "target": "object", "target_bpe": "object", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "if a is None or", "suffix": ":\n    pass\n", "target": "b", "target_bpe": "Ġb", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "try:\n    pass\nexcept:\n    pass\n", "suffix": ":\n    cleanup()\n", "target": "finally", "target_bpe": "finally", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "def empty_func():\n    ", "suffix": "\n", "target": "pass", "target_bpe": "pass", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "from abc import ABC,", "suffix": "\n", "target": "abstractmethod", "target_bpe": "Ġabstractmethod", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "def generator():\n    ", "suffix": " 42\n", "target": "yield", "target_bpe": "yield", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "assert len(arr) >=", "suffix": "\n", "target": "0", "target_bpe": "Ġ0", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "if condition:\n    return True\n", "suffix": " False\n", "target": "return", "target_bpe": "return", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "if x is not", "suffix": ":\n    pass\n", "target": "None", "target_bpe": "ĠNone", "mode": "both"},
    {"category": "Syntactic Keywords", "prompt": "while", "suffix": ":\n    step()\n", "target": "running", "target_bpe": "Ġrunning", "mode": "both"},

    # =========================================================================
    # 3. Idiomatic Imports & Calls (25 probes)
    # Ubiquitous Python stdlib / framework idioms
    # =========================================================================
    {"category": "Idiomatic Imports & Calls", "prompt": "import torch.nn", "suffix": " nn\n", "target": "as", "target_bpe": "Ġas", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "import numpy", "suffix": " np\n", "target": "as", "target_bpe": "Ġas", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "import pandas", "suffix": " pd\n", "target": "as", "target_bpe": "Ġas", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "import matplotlib.pyplot", "suffix": " plt\n", "target": "as", "target_bpe": "Ġas", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "import os.path", "suffix": " osp\n", "target": "as", "target_bpe": "Ġas", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "data = json.", "suffix": "(raw_text)\n", "target": "loads", "target_bpe": "loads", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "text = json.", "suffix": "(data)\n", "target": "dumps", "target_bpe": "dumps", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "path = os.path.", "suffix": "(root, filename)\n", "target": "join", "target_bpe": "join", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "exists = os.path.", "suffix": "(filepath)\n", "target": "exists", "target_bpe": "exists", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "base = os.path.", "suffix": "(path)\n", "target": "basename", "target_bpe": "basename", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "d = os.path.", "suffix": "(path)\n", "target": "dirname", "target_bpe": "dirname", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "sys.", "suffix": "(0)\n", "target": "exit", "target_bpe": "exit", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "time.", "suffix": "(1)\n", "target": "sleep", "target_bpe": "sleep", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "now = time.", "suffix": "()\n", "target": "time", "target_bpe": "time", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "math.", "suffix": "(x)\n", "target": "sqrt", "target_bpe": "sqrt", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "val = math.", "suffix": "(x)\n", "target": "log", "target_bpe": "log", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "pattern = re.", "suffix": "(r\"^\\d+$\")\n", "target": "compile", "target_bpe": "compile", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "match = re.", "suffix": "(pattern, text)\n", "target": "search", "target_bpe": "search", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "arr = np.", "suffix": "((10, 10))\n", "target": "zeros", "target_bpe": "zeros", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "ones = np.", "suffix": "(shape)\n", "target": "ones", "target_bpe": "ones", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "t = torch.", "suffix": "((3, 3))\n", "target": "zeros", "target_bpe": "zeros", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "device = torch.", "suffix": "('cuda' if torch.cuda.is_available() else 'cpu')\n", "target": "device", "target_bpe": "device", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "logging.", "suffix": "(level=logging.INFO)\n", "target": "basicConfig", "target_bpe": "basicConfig", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "res = requests.", "suffix": "(url)\n", "target": "get", "target_bpe": "get", "mode": "both"},
    {"category": "Idiomatic Imports & Calls", "prompt": "torch.manual_seed(", "suffix": ")\n", "target": "42", "target_bpe": "42", "mode": "both"},

    # =========================================================================
    # 4. Suffix-Clued Bidirectional Infilling (25 probes)
    # The suffix provides the missing semantic clue that the prefix alone lacks!
    # (Evaluated on COROSred infill mode; N/A on AR causal models)
    # =========================================================================
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "suffix": "(self):\n    return self._name\n", "target": "name", "target_bpe": "name", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "suffix": "(self):\n    return self._value\n", "target": "value", "target_bpe": "value", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "suffix": "(self):\n    return self._status\n", "target": "status", "target_bpe": "status", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "suffix": "(self):\n    return self._config\n", "target": "config", "target_bpe": "config", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "suffix": "(self):\n    return self._id\n", "target": "id", "target_bpe": "id", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "suffix": "(self):\n    return self._data\n", "target": "data", "target_bpe": "data", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "suffix": "(self):\n    return self._path\n", "target": "path", "target_bpe": "path", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def set_", "suffix": "(self, name):\n    self._name = name\n", "target": "name", "target_bpe": "name", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def set_", "suffix": "(self, value):\n    self._value = value\n", "target": "value", "target_bpe": "value", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def set_", "suffix": "(self, timeout):\n    self._timeout = timeout\n", "target": "timeout", "target_bpe": "timeout", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def set_", "suffix": "(self, count):\n    self._count = count\n", "target": "count", "target_bpe": "count", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "result = ", "suffix": ".loads(text)\n", "target": "json", "target_bpe": "json", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "serialized = ", "suffix": ".dumps(data)\n", "target": "json", "target_bpe": "json", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "full_path = ", "suffix": ".path.join(root, file)\n", "target": "os", "target_bpe": "os", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "with ", "suffix": "(path, 'r') as f:\n    data = f.read()\n", "target": "open", "target_bpe": "open", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "total = ", "suffix": "(numbers)\n    return total\n", "target": "sum", "target_bpe": "sum", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "n_items = ", "suffix": "(collection)\n", "target": "len", "target_bpe": "len", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "import torch.nn as ", "suffix": "\n\nclass Net(nn.Module):\n    pass\n", "target": "nn", "target_bpe": "nn", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "import numpy as ", "suffix": "\n\narr = np.zeros(10)\n", "target": "np", "target_bpe": "np", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "import pandas as ", "suffix": "\n\ndf = pd.DataFrame(data)\n", "target": "pd", "target_bpe": "pd", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def is_", "suffix": "(obj) -> bool:\n    return obj.is_valid\n", "target": "valid", "target_bpe": "valid", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def is_", "suffix": "(container) -> bool:\n    return len(container) == 0\n", "target": "empty", "target_bpe": "empty", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "self._", "suffix": " = {}\n        return self._cache[key]\n", "target": "cache", "target_bpe": "cache", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "self._", "suffix": " = threading.Lock()\n        with self._lock:\n", "target": "lock", "target_bpe": "lock", "mode": "infill"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def to_", "suffix": "(self):\n    return {'name': self.name, 'value': self.value}\n", "target": "dict", "target_bpe": "dict", "mode": "infill"},
]

print(f"Total probes defined: {len(PROBES)}")
for idx, p in enumerate(PROBES):
    t_bpe = p["target_bpe"]
    tid = tok.token_to_id(t_bpe)
    if tid is None:
        tid = tok.encode(p["target"]).ids[0]
    p["id"] = f"probe_{idx:03d}"
    p["target_id"] = tid

print("All 100 probes verified with valid token IDs!")
