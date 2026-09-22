"""
Contextual Probes Suite for Télos Models.
Contains 100 deterministic benchmark prompts across 4 core syntactic & semantic categories:
1. Contextually Deterministic Identifiers (25 probes)
   - Deducible from local scope, method context, class structure, or function naming.
2. Syntactic Keywords (25 probes)
   - Strictly enforced by Python grammar & syntax rules (try/except, with/as, for/in, etc.).
3. Idiomatic Imports & Calls (25 probes)
   - Ubiquitous Python stdlib & framework idioms (json.loads, os.path.join, torch.nn as nn, etc.).
4. Suffix-Clued Bidirectional Infilling (25 probes)
   - Where the suffix provides the essential semantic clue that disambiguates the missing token.
   - Evaluated on bidirectional models (COROSred); N/A for pure causal AR models.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

PROBE_SUITE_100: List[Dict[str, Any]] = [
    # =========================================================================
    # 1. Contextually Deterministic Identifiers (25 probes)
    # Target is strictly forced by local variable scope, method context, or naming
    # =========================================================================
    {"category": "Contextually Deterministic Identifiers", "prompt": "x = 10\nprint(", "prefix": "x = 10\nprint(", "suffix": ")\n", "target": "x", "target_bpe": "x", "mode": "both", "description": "Local scope variable resolution"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "y = 20\nprint(", "prefix": "y = 20\nprint(", "suffix": ")\n", "target": "y", "target_bpe": "y", "mode": "both", "description": "Local scope variable resolution"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_name(self):\n    return self._", "prefix": "def get_name(self):\n    return self._", "suffix": "\n", "target": "name", "target_bpe": "name", "mode": "both", "description": "Getter attribute resolution from function name"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_value(self):\n    return self._", "prefix": "def get_value(self):\n    return self._", "suffix": "\n", "target": "value", "target_bpe": "value", "mode": "both", "description": "Getter attribute resolution from function name"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_id(self):\n    return self._", "prefix": "def get_id(self):\n    return self._", "suffix": "\n", "target": "id", "target_bpe": "id", "mode": "both", "description": "Getter attribute resolution from function name"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_path(self):\n    return self._", "prefix": "def get_path(self):\n    return self._", "suffix": "\n", "target": "path", "target_bpe": "path", "mode": "both", "description": "Getter attribute resolution from function name"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_data(self):\n    return self._", "prefix": "def get_data(self):\n    return self._", "suffix": "\n", "target": "data", "target_bpe": "data", "mode": "both", "description": "Getter attribute resolution from function name"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_status(self):\n    return self._", "prefix": "def get_status(self):\n    return self._", "suffix": "\n", "target": "status", "target_bpe": "status", "mode": "both", "description": "Getter attribute resolution from function name"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def get_config(self):\n    return self._", "prefix": "def get_config(self):\n    return self._", "suffix": "\n", "target": "config", "target_bpe": "config", "mode": "both", "description": "Getter attribute resolution from function name"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def set_count(self, count):\n    self._count =", "prefix": "def set_count(self, count):\n    self._count =", "suffix": "\n", "target": "count", "target_bpe": "Ġcount", "mode": "both", "description": "Setter parameter resolution"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def set_timeout(self, timeout):\n    self.timeout =", "prefix": "def set_timeout(self, timeout):\n    self.timeout =", "suffix": "\n", "target": "timeout", "target_bpe": "Ġtimeout", "mode": "both", "description": "Setter parameter resolution"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "total = 0\nfor item in items:\n    total +=", "prefix": "total = 0\nfor item in items:\n    total +=", "suffix": "\n", "target": "item", "target_bpe": "Ġitem", "mode": "both", "description": "Loop accumulation variable"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "total = 0\nfor x in values:\n    total +=", "prefix": "total = 0\nfor x in values:\n    total +=", "suffix": "\n", "target": "x", "target_bpe": "Ġx", "mode": "both", "description": "Loop accumulation variable"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "for k, v in mapping.items():\n    print(", "prefix": "for k, v in mapping.items():\n    print(", "suffix": ", v)\n", "target": "k", "target_bpe": "k", "mode": "both", "description": "Dictionary key variable in iteration"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "for k, v in mapping.items():\n    print(k,", "prefix": "for k, v in mapping.items():\n    print(k,", "suffix": ")\n", "target": "v", "target_bpe": "Ġv", "mode": "both", "description": "Dictionary value variable in iteration"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "width, height = dims\narea = width *", "prefix": "width, height = dims\narea = width *", "suffix": "\n", "target": "height", "target_bpe": "Ġheight", "mode": "both", "description": "Paired dimensional identifier"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "left, right = bounds\nspan = right -", "prefix": "left, right = bounds\nspan = right -", "suffix": "\n", "target": "left", "target_bpe": "Ġleft", "mode": "both", "description": "Boundary interval subtraction"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "start, end = range_vals\nlength = end -", "prefix": "start, end = range_vals\nlength = end -", "suffix": "\n", "target": "start", "target_bpe": "Ġstart", "mode": "both", "description": "Range span subtraction"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "row, col = coord\nnext_pos = (row + 1,", "prefix": "row, col = coord\nnext_pos = (row + 1,", "suffix": ")\n", "target": "col", "target_bpe": "Ġcol", "mode": "both", "description": "Coordinate tuple completion"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def __init__(self, name):\n    self.name =", "prefix": "def __init__(self, name):\n    self.name =", "suffix": "\n", "target": "name", "target_bpe": "Ġname", "mode": "both", "description": "Constructor attribute assignment"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def __init__(self, config):\n    self.config =", "prefix": "def __init__(self, config):\n    self.config =", "suffix": "\n", "target": "config", "target_bpe": "Ġconfig", "mode": "both", "description": "Constructor attribute assignment"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "def __init__(self, model):\n    self.model =", "prefix": "def __init__(self, model):\n    self.model =", "suffix": "\n", "target": "model", "target_bpe": "Ġmodel", "mode": "both", "description": "Constructor attribute assignment"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "res = []\nfor elem in elements:\n    res.append(", "prefix": "res = []\nfor elem in elements:\n    res.append(", "suffix": ")\n", "target": "elem", "target_bpe": "elem", "mode": "both", "description": "List accumulation variable"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "key = 'user_id'\nval = cache[", "prefix": "key = 'user_id'\nval = cache[", "suffix": "]\n", "target": "key", "target_bpe": "key", "mode": "both", "description": "Dictionary lookup key"},
    {"category": "Contextually Deterministic Identifiers", "prompt": "head, tail = split(lst)\nreturn", "prefix": "head, tail = split(lst)\nreturn", "suffix": "\n", "target": "head", "target_bpe": "Ġhead", "mode": "both", "description": "Unpacked head return"},

    # =========================================================================
    # 2. Syntactic Keywords (25 probes)
    # Target is strictly enforced by Python grammar & syntax rules
    # =========================================================================
    {"category": "Syntactic Keywords", "prompt": "try:\n    do_something()\n", "prefix": "try:\n    do_something()\n", "suffix": " Exception:\n    pass\n", "target": "except", "target_bpe": "except", "mode": "both", "description": "try-except block syntax"},
    {"category": "Syntactic Keywords", "prompt": "with open(f)", "prefix": "with open(f)", "suffix": " handle:\n    pass\n", "target": "as", "target_bpe": "Ġas", "mode": "both", "description": "with-as context manager syntax"},
    {"category": "Syntactic Keywords", "prompt": "with open(path, 'r')", "prefix": "with open(path, 'r')", "suffix": " f:\n    text = f.read()\n", "target": "as", "target_bpe": "Ġas", "mode": "both", "description": "with-as context manager syntax"},
    {"category": "Syntactic Keywords", "prompt": "for item", "prefix": "for item", "suffix": " collection:\n    pass\n", "target": "in", "target_bpe": "Ġin", "mode": "both", "description": "for-in iteration syntax"},
    {"category": "Syntactic Keywords", "prompt": "for i", "prefix": "for i", "suffix": " range(10):\n    print(i)\n", "target": "in", "target_bpe": "Ġin", "mode": "both", "description": "for-in iteration syntax"},
    {"category": "Syntactic Keywords", "prompt": "for idx, elem", "prefix": "for idx, elem", "suffix": " enumerate(lst):\n    pass\n", "target": "in", "target_bpe": "Ġin", "mode": "both", "description": "for-in iteration syntax"},
    {"category": "Syntactic Keywords", "prompt": "while", "prefix": "while", "suffix": ":\n    break\n", "target": "True", "target_bpe": "ĠTrue", "mode": "both", "description": "Infinite loop condition"},
    {"category": "Syntactic Keywords", "prompt": "from typing", "prefix": "from typing", "suffix": " List, Dict, Optional\n", "target": "import", "target_bpe": "Ġimport", "mode": "both", "description": "from-import statement"},
    {"category": "Syntactic Keywords", "prompt": "from pathlib", "prefix": "from pathlib", "suffix": " Path\n", "target": "import", "target_bpe": "Ġimport", "mode": "both", "description": "from-import statement"},
    {"category": "Syntactic Keywords", "prompt": "from collections", "prefix": "from collections", "suffix": " deque\n", "target": "import", "target_bpe": "Ġimport", "mode": "both", "description": "from-import statement"},
    {"category": "Syntactic Keywords", "prompt": "assert x is", "prefix": "assert x is", "suffix": " None\n", "target": "not", "target_bpe": "Ġnot", "mode": "both", "description": "Identity assertion negation"},
    {"category": "Syntactic Keywords", "prompt": "if item", "prefix": "if item", "suffix": " items:\n    return True\n", "target": "in", "target_bpe": "Ġin", "mode": "both", "description": "Membership condition"},
    {"category": "Syntactic Keywords", "prompt": "if x == 1:\n    return 1\n", "prefix": "if x == 1:\n    return 1\n", "suffix": ":\n    return 0\n", "target": "else", "target_bpe": "else", "mode": "both", "description": "if-else branch"},
    {"category": "Syntactic Keywords", "prompt": "yield", "prefix": "yield", "suffix": " sub_generator()\n", "target": "from", "target_bpe": "Ġfrom", "mode": "both", "description": "yield-from delegating generator"},
    {"category": "Syntactic Keywords", "prompt": "if not", "prefix": "if not", "suffix": ":\n    return False\n", "target": "found", "target_bpe": "Ġfound", "mode": "both", "description": "Boolean flag check"},
    {"category": "Syntactic Keywords", "prompt": "class MyModel(", "prefix": "class MyModel(", "suffix": "):\n    pass\n", "target": "object", "target_bpe": "object", "mode": "both", "description": "Base object inheritance"},
    {"category": "Syntactic Keywords", "prompt": "if a is None or", "prefix": "if a is None or", "suffix": ":\n    pass\n", "target": "b", "target_bpe": "Ġb", "mode": "both", "description": "Boolean disjunction"},
    {"category": "Syntactic Keywords", "prompt": "try:\n    pass\nexcept:\n    pass\n", "prefix": "try:\n    pass\nexcept:\n    pass\n", "suffix": ":\n    cleanup()\n", "target": "finally", "target_bpe": "finally", "mode": "both", "description": "try-except-finally cleanup"},
    {"category": "Syntactic Keywords", "prompt": "def empty_func():\n    ", "prefix": "def empty_func():\n    ", "suffix": "\n", "target": "pass", "target_bpe": "pass", "mode": "both", "description": "No-op pass statement"},
    {"category": "Syntactic Keywords", "prompt": "from abc import ABC,", "prefix": "from abc import ABC,", "suffix": "\n", "target": "abstractmethod", "target_bpe": "Ġabstractmethod", "mode": "both", "description": "Abstract method decorator import"},
    {"category": "Syntactic Keywords", "prompt": "def generator():\n    ", "prefix": "def generator():\n    ", "suffix": " 42\n", "target": "yield", "target_bpe": "yield", "mode": "both", "description": "Generator yield"},
    {"category": "Syntactic Keywords", "prompt": "assert len(arr) >=", "prefix": "assert len(arr) >=", "suffix": "\n", "target": "0", "target_bpe": "Ġ0", "mode": "both", "description": "Non-negative length assertion"},
    {"category": "Syntactic Keywords", "prompt": "if condition:\n    return True\n", "prefix": "if condition:\n    return True\n", "suffix": " False\n", "target": "return", "target_bpe": "return", "mode": "both", "description": "Fallback return statement"},
    {"category": "Syntactic Keywords", "prompt": "if x is not", "prefix": "if x is not", "suffix": ":\n    pass\n", "target": "None", "target_bpe": "ĠNone", "mode": "both", "description": "Non-None identity check"},
    {"category": "Syntactic Keywords", "prompt": "while", "prefix": "while", "suffix": ":\n    step()\n", "target": "running", "target_bpe": "Ġrunning", "mode": "both", "description": "Loop flag condition"},

    # =========================================================================
    # 3. Idiomatic Imports & Calls (25 probes)
    # Ubiquitous Python stdlib / framework idioms
    # =========================================================================
    {"category": "Idiomatic Imports & Calls", "prompt": "import torch.nn", "prefix": "import torch.nn", "suffix": " nn\n", "target": "as", "target_bpe": "Ġas", "mode": "both", "description": "PyTorch nn alias"},
    {"category": "Idiomatic Imports & Calls", "prompt": "import numpy", "prefix": "import numpy", "suffix": " np\n", "target": "as", "target_bpe": "Ġas", "mode": "both", "description": "NumPy alias"},
    {"category": "Idiomatic Imports & Calls", "prompt": "import pandas", "prefix": "import pandas", "suffix": " pd\n", "target": "as", "target_bpe": "Ġas", "mode": "both", "description": "Pandas alias"},
    {"category": "Idiomatic Imports & Calls", "prompt": "import matplotlib.pyplot", "prefix": "import matplotlib.pyplot", "suffix": " plt\n", "target": "as", "target_bpe": "Ġas", "mode": "both", "description": "Matplotlib alias"},
    {"category": "Idiomatic Imports & Calls", "prompt": "import os.path", "prefix": "import os.path", "suffix": " osp\n", "target": "as", "target_bpe": "Ġas", "mode": "both", "description": "os.path alias"},
    {"category": "Idiomatic Imports & Calls", "prompt": "data = json.", "prefix": "data = json.", "suffix": "(raw_text)\n", "target": "loads", "target_bpe": "loads", "mode": "both", "description": "JSON string deserialize"},
    {"category": "Idiomatic Imports & Calls", "prompt": "text = json.", "prefix": "text = json.", "suffix": "(data)\n", "target": "dumps", "target_bpe": "dumps", "mode": "both", "description": "JSON object serialize"},
    {"category": "Idiomatic Imports & Calls", "prompt": "path = os.path.", "prefix": "path = os.path.", "suffix": "(root, filename)\n", "target": "join", "target_bpe": "join", "mode": "both", "description": "File path concatenation"},
    {"category": "Idiomatic Imports & Calls", "prompt": "exists = os.path.", "prefix": "exists = os.path.", "suffix": "(filepath)\n", "target": "exists", "target_bpe": "exists", "mode": "both", "description": "File existence check"},
    {"category": "Idiomatic Imports & Calls", "prompt": "base = os.path.", "prefix": "base = os.path.", "suffix": "(path)\n", "target": "basename", "target_bpe": "basename", "mode": "both", "description": "Filename extraction"},
    {"category": "Idiomatic Imports & Calls", "prompt": "d = os.path.", "prefix": "d = os.path.", "suffix": "(path)\n", "target": "dirname", "target_bpe": "dirname", "mode": "both", "description": "Directory path extraction"},
    {"category": "Idiomatic Imports & Calls", "prompt": "sys.", "prefix": "sys.", "suffix": "(0)\n", "target": "exit", "target_bpe": "exit", "mode": "both", "description": "Process termination"},
    {"category": "Idiomatic Imports & Calls", "prompt": "time.", "prefix": "time.", "suffix": "(1)\n", "target": "sleep", "target_bpe": "sleep", "mode": "both", "description": "Process pause"},
    {"category": "Idiomatic Imports & Calls", "prompt": "now = time.", "prefix": "now = time.", "suffix": "()\n", "target": "time", "target_bpe": "time", "mode": "both", "description": "Timestamp retrieval"},
    {"category": "Idiomatic Imports & Calls", "prompt": "math.", "prefix": "math.", "suffix": "(x)\n", "target": "sqrt", "target_bpe": "sqrt", "mode": "both", "description": "Square root"},
    {"category": "Idiomatic Imports & Calls", "prompt": "val = math.", "prefix": "val = math.", "suffix": "(x)\n", "target": "log", "target_bpe": "log", "mode": "both", "description": "Natural logarithm"},
    {"category": "Idiomatic Imports & Calls", "prompt": "pattern = re.", "prefix": "pattern = re.", "suffix": "(r\"^\\d+$\")\n", "target": "compile", "target_bpe": "compile", "mode": "both", "description": "Regex pattern compile"},
    {"category": "Idiomatic Imports & Calls", "prompt": "match = re.", "prefix": "match = re.", "suffix": "(pattern, text)\n", "target": "search", "target_bpe": "search", "mode": "both", "description": "Regex pattern search"},
    {"category": "Idiomatic Imports & Calls", "prompt": "arr = np.", "prefix": "arr = np.", "suffix": "((10, 10))\n", "target": "zeros", "target_bpe": "zeros", "mode": "both", "description": "NumPy zeros initialization"},
    {"category": "Idiomatic Imports & Calls", "prompt": "ones = np.", "prefix": "ones = np.", "suffix": "(shape)\n", "target": "ones", "target_bpe": "ones", "mode": "both", "description": "NumPy ones initialization"},
    {"category": "Idiomatic Imports & Calls", "prompt": "t = torch.", "prefix": "t = torch.", "suffix": "((3, 3))\n", "target": "zeros", "target_bpe": "zeros", "mode": "both", "description": "PyTorch tensor zeros"},
    {"category": "Idiomatic Imports & Calls", "prompt": "device = torch.", "prefix": "device = torch.", "suffix": "('cuda' if torch.cuda.is_available() else 'cpu')\n", "target": "device", "target_bpe": "device", "mode": "both", "description": "PyTorch device selector"},
    {"category": "Idiomatic Imports & Calls", "prompt": "logging.", "prefix": "logging.", "suffix": "(level=logging.INFO)\n", "target": "basicConfig", "target_bpe": "basicConfig", "mode": "both", "description": "Logging configuration"},
    {"category": "Idiomatic Imports & Calls", "prompt": "res = requests.", "prefix": "res = requests.", "suffix": "(url)\n", "target": "get", "target_bpe": "get", "mode": "both", "description": "HTTP GET request"},
    {"category": "Idiomatic Imports & Calls", "prompt": "torch.manual_seed(", "prefix": "torch.manual_seed(", "suffix": ")\n", "target": "42", "target_bpe": "42", "mode": "both", "description": "RNG deterministic seed"},

    # =========================================================================
    # 4. Suffix-Clued Bidirectional Infilling (25 probes)
    # The suffix provides the essential semantic clue that the prefix alone lacks!
    # (Evaluated on COROSred infill mode; N/A on AR causal models)
    # =========================================================================
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "prefix": "def get_", "suffix": "(self):\n    return self._name\n", "target": "name", "target_bpe": "name", "mode": "infill", "description": "Getter name derived from returned attribute"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "prefix": "def get_", "suffix": "(self):\n    return self._value\n", "target": "value", "target_bpe": "value", "mode": "infill", "description": "Getter name derived from returned attribute"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "prefix": "def get_", "suffix": "(self):\n    return self._status\n", "target": "status", "target_bpe": "status", "mode": "infill", "description": "Getter name derived from returned attribute"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "prefix": "def get_", "suffix": "(self):\n    return self._config\n", "target": "config", "target_bpe": "config", "mode": "infill", "description": "Getter name derived from returned attribute"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "prefix": "def get_", "suffix": "(self):\n    return self._id\n", "target": "id", "target_bpe": "id", "mode": "infill", "description": "Getter name derived from returned attribute"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "prefix": "def get_", "suffix": "(self):\n    return self._data\n", "target": "data", "target_bpe": "data", "mode": "infill", "description": "Getter name derived from returned attribute"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def get_", "prefix": "def get_", "suffix": "(self):\n    return self._path\n", "target": "path", "target_bpe": "path", "mode": "infill", "description": "Getter name derived from returned attribute"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def set_", "prefix": "def set_", "suffix": "(self, name):\n    self._name = name\n", "target": "name", "target_bpe": "name", "mode": "infill", "description": "Setter name derived from mutated attribute"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def set_", "prefix": "def set_", "suffix": "(self, value):\n    self._value = value\n", "target": "value", "target_bpe": "value", "mode": "infill", "description": "Setter name derived from mutated attribute"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def set_", "prefix": "def set_", "suffix": "(self, timeout):\n    self._timeout = timeout\n", "target": "timeout", "target_bpe": "timeout", "mode": "infill", "description": "Setter name derived from mutated attribute"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def set_", "prefix": "def set_", "suffix": "(self, count):\n    self._count = count\n", "target": "count", "target_bpe": "count", "mode": "infill", "description": "Setter name derived from mutated attribute"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "result = ", "prefix": "result = ", "suffix": ".loads(text)\n", "target": "json", "target_bpe": "json", "mode": "infill", "description": "Module name derived from .loads() call"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "serialized = ", "prefix": "serialized = ", "suffix": ".dumps(data)\n", "target": "json", "target_bpe": "json", "mode": "infill", "description": "Module name derived from .dumps() call"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "full_path = ", "prefix": "full_path = ", "suffix": ".path.join(root, file)\n", "target": "os", "target_bpe": "os", "mode": "infill", "description": "Module name derived from .path.join call"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "with ", "prefix": "with ", "suffix": "(path, 'r') as f:\n    data = f.read()\n", "target": "open", "target_bpe": "open", "mode": "infill", "description": "Built-in function derived from file read context"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "total = ", "prefix": "total = ", "suffix": "(numbers)\n    return total\n", "target": "sum", "target_bpe": "sum", "mode": "infill", "description": "Aggregation function derived from variable name & argument"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "n_items = ", "prefix": "n_items = ", "suffix": "(collection)\n", "target": "len", "target_bpe": "len", "mode": "infill", "description": "Collection length function derived from variable context"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "import torch.nn as ", "prefix": "import torch.nn as ", "suffix": "\n\nclass Net(nn.Module):\n    pass\n", "target": "nn", "target_bpe": "nn", "mode": "infill", "description": "Import alias derived from subclass inheritance nn.Module"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "import numpy as ", "prefix": "import numpy as ", "suffix": "\n\narr = np.zeros(10)\n", "target": "np", "target_bpe": "np", "mode": "infill", "description": "Import alias derived from np.zeros usage"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "import pandas as ", "prefix": "import pandas as ", "suffix": "\n\ndf = pd.DataFrame(data)\n", "target": "pd", "target_bpe": "pd", "mode": "infill", "description": "Import alias derived from pd.DataFrame usage"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def is_", "prefix": "def is_", "suffix": "(obj) -> bool:\n    return obj.is_valid\n", "target": "valid", "target_bpe": "valid", "mode": "infill", "description": "Predicate name derived from returned boolean attribute"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def is_", "prefix": "def is_", "suffix": "(container) -> bool:\n    return len(container) == 0\n", "target": "empty", "target_bpe": "empty", "mode": "infill", "description": "Predicate name derived from zero-length check"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "self._", "prefix": "self._", "suffix": " = {}\n        return self._cache[key]\n", "target": "cache", "target_bpe": "cache", "mode": "infill", "description": "Private dictionary attribute derived from indexed lookup"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "self._", "prefix": "self._", "suffix": " = threading.Lock()\n        with self._lock:\n", "target": "lock", "target_bpe": "lock", "mode": "infill", "description": "Private synchronization attribute derived from threading.Lock()"},
    {"category": "Suffix-Clued Bidirectional Infill", "prompt": "def to_", "prefix": "def to_", "suffix": "(self):\n    return {'name': self.name, 'value': self.value}\n", "target": "dict", "target_bpe": "dict", "mode": "infill", "description": "Serialization method derived from returned dictionary"},
]


def load_contextual_probes(num_probes: int = 100) -> List[Dict[str, Any]]:
    """
    Loads deterministic contextual probes for Télos evaluation.
    Loads from evals/benchmarks/contextual_probes_1000.json if num_probes > 100 and the file exists,
    otherwise returns the standardized PROBE_SUITE_100.
    """
    if num_probes and num_probes > 100:
        bench_file = Path(__file__).resolve().parents[2] / "evals" / "benchmarks" / "contextual_probes_1000.json"
        if bench_file.exists():
            try:
                with open(bench_file, "r") as f:
                    data = json.load(f)
                if data:
                    return data[:num_probes]
            except Exception:
                pass

    return PROBE_SUITE_100[:num_probes] if num_probes else PROBE_SUITE_100
