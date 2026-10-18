"""
Complete builder script for 1,000 authentic contextual probes.
Takes the best authentic probes from build_authentic_probes_1000.py,
fixes formatting and aligns with configs/tokenizer_mac.json,
adds curated authentic probes to reach exactly 125 per category,
and validates 100% of all 1,000 probes.
"""

import json
from pathlib import Path
from tokenizers import Tokenizer
import scripts.benchmark_generators.build_authentic_probes_1000 as orig

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "configs" / "tokenizer_mac.json"
OUTPUT_PATH = ROOT / "evals" / "benchmarks" / "contextual_probes_1000.json"
TARGET_FILE = ROOT / "scripts" / "benchmark_generators" / "build_authentic_probes_1000.py"

tok = Tokenizer.from_file(str(TOKENIZER_PATH))

def verify(p, t, tb):
    t_id = tok.token_to_id(tb)
    if t_id is None:
        return False, f"target_bpe '{tb}' not in vocab"
    sep = " " if (tb.startswith("Ġ") and not p.endswith(" ")) else ""
    full = p + sep + t
    p_ids = tok.encode(p).ids
    f_ids = tok.encode(full).ids
    if len(f_ids) != len(p_ids) + 1:
        return False, f"Length mismatch: p={len(p_ids)}, full={len(f_ids)}"
    if f_ids[len(p_ids)] != t_id:
        return False, f"ID mismatch: got {f_ids[len(p_ids)]} vs {t_id}"
    return True, t_id

def get_valid_from_orig(fn):
    probes = fn()
    valid = []
    seen = set()
    for p in probes:
        key = (p['prompt'], p['target'])
        if key in seen:
            continue
        # Test original
        ok, _ = verify(p['prompt'], p['target'], p['target_bpe'])
        if ok:
            seen.add(key)
            valid.append(p)
            continue
        # Test stripped prompt + Ġ
        p_strip = p['prompt'].rstrip(' ')
        tb_sp = 'Ġ' + p['target']
        ok, _ = verify(p_strip, p['target'], tb_sp)
        if ok:
            p['prompt'] = p_strip
            p['prefix'] = p_strip
            p['target_bpe'] = tb_sp
            seen.add((p_strip, p['target']))
            valid.append(p)
            continue
        # Test raw
        ok, _ = verify(p['prompt'], p['target'], p['target'])
        if ok:
            p['target_bpe'] = p['target']
            seen.add(key)
            valid.append(p)
            continue
        ok, _ = verify(p_strip, p['target'], p['target'])
        if ok:
            p['prompt'] = p_strip
            p['prefix'] = p_strip
            p['target_bpe'] = p['target']
            seen.add((p_strip, p['target']))
            valid.append(p)
            continue
    return valid

def add_curated(valid_list, candidates, target_count=125):
    for p, t, tb, sfx, desc in candidates:
        if len(valid_list) >= target_count:
            break
        ok, res = verify(p, t, tb)
        if not ok:
            raise ValueError(f"Curated probe invalid: {res} -> {p}, {t}, {tb}")
        valid_list.append({
            "prompt": p,
            "prefix": p,
            "target": t,
            "target_bpe": tb,
            "suffix": sfx,
            "mode": "both" if sfx else "causal",
            "description": desc
        })
    if len(valid_list) < target_count:
        raise ValueError(f"Only reached {len(valid_list)} / {target_count}")
    return valid_list[:target_count]

CAT1_EXTRA = [
    ("import math\nval = math.", "comb", "comb", "(n, k)\n", "Combinations"),
    ("import math\nval = math.", "perm", "perm", "(n, k)\n", "Permutations"),
    ("import math\nval = math.", "prod", "prod", "(items)\n", "Product of iterable"),
    ("import math\nval = math.", "trunc", "trunc", "(val)\n", "Truncate value"),
    ("import math\nval = math.", "dist", "dist", "(p1, p2)\n", "Euclidean distance"),
    ("import math\nval = math.", "pow", "pow", "(x, y)\n", "Power"),
    ("def square(x):\n    return x *", "x", "Ġx", "\n", "Square function"),
    ("def cube(x):\n    return x **", "3", "Ġ3", "\n", "Cube function"),
    ("def double(x):\n    return x *", "2", "Ġ2", "\n", "Double function"),
    ("def half(x):\n    return x /", "2", "Ġ2", "\n", "Half function"),
    ("def negate(x):\n    return -", "x", "x", "\n", "Negate function"),
    ("def reciprocal(x):\n    return 1.0 /", "x", "Ġx", "\n", "Reciprocal function"),
    ("def is_divisible(n, d):\n    return n % d ==", "0", "Ġ0", "\n", "Divisibility check"),
    ("def area_circle(r):\n    return math.pi * r **", "2", "Ġ2", "\n", "Circle area"),
    ("def kinetic_energy(m, v):\n    return 0.5 * m * v **", "2", "Ġ2", "\n", "Kinetic energy"),
    ("def fahrenheit_to_celsius(f):\n    return (f - 32) * 5 /", "9", "Ġ9", "\n", "Temperature conversion"),
    ("def celsius_to_fahrenheit(c):\n    return c * 9 / 5 +", "32", "Ġ32", "\n", "Temperature conversion"),
    ("def circum(r):\n    return 2.0 * math.pi *", "r", "Ġr", "\n", "Circumference"),
    ("def mod_add(a, b, m):\n    return (a + b) %", "m", "Ġm", "\n", "Modular addition"),
    ("def mod_sub(a, b, m):\n    return (a - b) %", "m", "Ġm", "\n", "Modular subtraction"),
    ("def mod_mul(a, b, m):\n    return (a * b) %", "m", "Ġm", "\n", "Modular multiplication"),
    ("def dot2(x, y, a, b):\n    return x * a + y *", "b", "Ġb", "\n", "Dot product 2D"),
    ("def dist_sq(x, y, a, b):\n    return (x - a)**2 + (y -", "b", "Ġb", ")**2\n", "Squared distance"),
    ("def slope(x, y, a, b):\n    return (b - y) / (a -", "x", "Ġx", ")\n", "Slope calculation"),
    ("def area_rect(w, h):\n    return w *", "h", "Ġh", "\n", "Rectangle area"),
    ("def volume_box(w, h, d):\n    return w * h *", "d", "Ġd", "\n", "Box volume"),
    ("def clamp(val, low, high):\n    return max(low, min(val,", "high", "Ġhigh", "))\n", "Clamp value"),
    ("def lerp(a, b, t):\n    return a + t * (b -", "a", "Ġa", ")\n", "Linear interpolation"),
]

CAT2_EXTRA = [
    ("queue = deque([1, 2, 3])\nitem = queue.", "pop", "pop", "()\n", "Queue pop"),
    ("stack = [10, 20, 30]\nval = stack.", "pop", "pop", "()\n", "Stack pop"),
    ("items = [1, 2, 3]\nitems.", "append", "append", "(4)\n", "List append"),
    ("items = [1, 2, 3]\nitems.", "clear", "clear", "()\n", "List clear"),
    ("items = [1, 2, 3]\nitems.", "reverse", "reverse", "()\n", "List reverse"),
    ("items = [3, 1, 2]\nitems.", "sort", "sort", "()\n", "List sort"),
    ("items = [1, 2, 3]\ncount = items.", "count", "count", "(2)\n", "List count"),
    ("items = [1, 2, 3]\nidx = items.", "index", "index", "(2)\n", "List index"),
    ("mapping = {'a': 1}\nmapping.", "update", "update", "({'b': 2})\n", "Dict update"),
    ("mapping = {'a': 1}\nval = mapping.", "get", "get", "('a')\n", "Dict get"),
    ("seen = set()\nseen.", "add", "add", "(item)\n", "Set add"),
    ("seen = set([1, 2])\nseen.", "remove", "remove", "(1)\n", "Set remove"),
    ("items = [1, 2, 3]\ncp = items.", "copy", "copy", "()\n", "List copy"),
]

CAT3_EXTRA = [
    ("s = text.", "strip", "strip", "()\n", "String strip"),
    ("s = text.", "rstrip", "rstrip", "()\n", "String rstrip"),
    ("words = text.", "split", "split", "(' ')\n", "String split"),
    ("joined = ','.", "join", "join", "(items)\n", "String join"),
    ("clean = text.", "replace", "replace", "('old', 'new')\n", "String replace"),
    ("idx = text.", "find", "find", "('sub')\n", "String find"),
    ("idx = text.", "index", "index", "('sub')\n", "String index"),
    ("is_match = text.", "startswith", "startswith", "('prefix')\n", "String startswith"),
    ("is_match = text.", "endswith", "endswith", "('suffix')\n", "String endswith"),
    ("low = text.", "lower", "lower", "()\n", "String lower"),
    ("up = text.", "upper", "upper", "()\n", "String upper"),
    ("t = text.", "title", "title", "()\n", "String title"),
    ("b = text.", "encode", "encode", "('utf-8')\n", "String encode"),
    ("s = data.", "decode", "decode", "('utf-8')\n", "String decode"),
    ("num_a = text.", "count", "count", "('a')\n", "String count"),
    ("msg = '{}: {}'.", "format", "format", "(k, v)\n", "String format"),
    ("ch = text[", "0", "0", "]\n", "First character"),
    ("last_ch = text[-", "1", "1", "]\n", "Last character"),
    ("prefix = text[:", "5", "5", "]\n", "Prefix slice"),
    ("suffix = text[-", "5", "5", ":]\n", "Suffix slice"),
    ("reversed_text = text[::-", "1", "1", "]\n", "Reverse string"),
    ("is_empty = (len(text) ==", "0", "Ġ0", ")\n", "Check empty string"),
    ("path = root + '/' +", "name", "Ġname", "\n", "Path concatenation"),
    ("greeting = 'Hello, ' +", "name", "Ġname", "\n", "Greeting concatenation"),
    ("first_word = text.split()[", "0", "0", "]\n", "First word"),
    ("last_word = text.split()[-", "1", "1", "]\n", "Last word"),
    ("s1, s2 = text.split(',',", "1", "Ġ1", ")\n", "Split maxsplit 1"),
    ("char_list = list(", "text", "text", ")\n", "String to list"),
    ("length = len(", "text", "text", ")\n", "String length"),
    ("has_space = ' ' in", "text", "Ġtext", "\n", "Check space in string"),
    ("has_comma = ',' in", "text", "Ġtext", "\n", "Check comma in string"),
    ("has_dot = '.' in", "text", "Ġtext", "\n", "Check dot in string"),
    ("ext = filename.split('.')[-", "1", "1", "]\n", "File extension"),
    ("base = filename.split('.')[", "0", "0", "]\n", "File basename"),
    ("padded = text + ' ' *", "pad", "Ġpad", "\n", "Space padding"),
    ("repeated = text *", "count", "Ġcount", "\n", "String repetition"),
    ("sub = text[start:", "end", "end", "]\n", "Substring slice"),
    ("token = tokens[", "idx", "idx", "]\n", "Token index"),
    ("line = lines[", "idx", "idx", "]\n", "Line index"),
    ("full = first +", "second", "Ġsecond", "\n", "String full concat"),
    ("filename = base +", "ext", "Ġext", "\n", "String filename concat"),
    ("prefix = 'data_' +", "name", "Ġname", "\n", "Prefix concat"),
]

CAT4_EXTRA = [
    ("def __init__(self, name):\n    self.name =", "name", "Ġname", "\n", "Constructor name assignment"),
    ("def __init__(self, size):\n    self.size =", "size", "Ġsize", "\n", "Constructor size assignment"),
    ("def __init__(self, model):\n    self.model =", "model", "Ġmodel", "\n", "Constructor model assignment"),
    ("def __init__(self, loss):\n    self.loss =", "loss", "Ġloss", "\n", "Constructor loss assignment"),
    ("def __init__(self, step):\n    self.step =", "step", "Ġstep", "\n", "Constructor step assignment"),
    ("def __init__(self, state):\n    self.state =", "state", "Ġstate", "\n", "Constructor state assignment"),
    ("def __init__(self, device):\n    self.device =", "device", "Ġdevice", "\n", "Constructor device assignment"),
    ("def __init__(self, input):\n    self.input =", "input", "Ġinput", "\n", "Constructor input assignment"),
    ("def __init__(self, output):\n    self.output =", "output", "Ġoutput", "\n", "Constructor output assignment"),
    ("def __init__(self, target):\n    self.target =", "target", "Ġtarget", "\n", "Constructor target assignment"),
    ("def __init__(self, weight):\n    self.weight =", "weight", "Ġweight", "\n", "Constructor weight assignment"),
    ("def __init__(self, flag):\n    self.flag =", "flag", "Ġflag", "\n", "Constructor flag assignment"),
    ("def get_name(self):\n    return self.", "name", "name", "\n", "Getter name"),
    ("def get_value(self):\n    return self.", "value", "value", "\n", "Getter value"),
    ("def get_data(self):\n    return self.", "data", "data", "\n", "Getter data"),
    ("def get_config(self):\n    return self.", "config", "config", "\n", "Getter config"),
    ("def get_path(self):\n    return self.", "path", "path", "\n", "Getter path"),
]

CAT5_EXTRA = [
    ("while count > 0:\n    count -= 1\n    if count == 5:\n       ", "break", "Ġbreak", "\n", "Loop break"),
    ("for i in range(10):\n    if i % 2 == 0:\n       ", "continue", "Ġcontinue", "\n", "Loop continue"),
    ("if not initialized:\n   ", "pass", "Ġpass", "\n", "Conditional pass"),
    ("def placeholder():\n   ", "pass", "Ġpass", "\n", "Function pass"),
    ("assert condition,", "msg", "Ġmsg", "\n", "Assertion with msg"),
    ("assert x > 0,", "err", "Ġerr", "\n", "Assertion with err"),
    ("assert valid,", "error", "Ġerror", "\n", "Assertion with error"),
    ("if val is None:\n   ", "return", "Ġreturn", "\n", "Early return"),
    ("if not items:\n   ", "return", "Ġreturn", " []\n", "Return empty list"),
    ("if a > b:\n    result =", "a", "Ġa", "\nelse:\n    result = b\n", "Ternary high"),
    ("if a < b:\n    result =", "a", "Ġa", "\nelse:\n    result = b\n", "Ternary low"),
    ("for item in", "items", "Ġitems", ":\n    process(item)\n", "For in items"),
    ("for elem in", "collection", "Ġcollection", ":\n    pass\n", "For in collection"),
    ("while", "True", "ĠTrue", ":\n    line = f.readline()\n", "While True"),
    ("while", "running", "Ġrunning", ":\n    tick()\n", "While running"),
    ("while queue:\n    node = queue.", "pop", "pop", "(0)\n", "While queue pop"),
    ("if flag:\n    do_a()\nelif", "condition", "Ġcondition", ":\n    do_b()\n", "Elif condition"),
    ("if key in", "cache", "Ġcache", ":\n    return cache[key]\n", "If in cache"),
    ("if val not in", "seen", "Ġseen", ":\n    seen.add(val)\n", "If not in seen"),
    ("for k, v in mapping.", "items", "items", "():\n    pass\n", "Mapping items loop"),
]

CAT6_EXTRA = [
    ("items = list(range(10))\nlength =", "len", "Ġlen", "(items)\n", "Length of range"),
    ("total =", "sum", "Ġsum", "(values)\n", "Sum of values"),
    ("lowest =", "min", "Ġmin", "(scores)\n", "Min of scores"),
    ("highest =", "max", "Ġmax", "(scores)\n", "Max of scores"),
    ("mag =", "abs", "Ġabs", "(diff)\n", "Abs of diff"),
    ("for idx, item in", "enumerate", "Ġenumerate", "(items):\n", "Enumerate items"),
    ("pairs =", "zip", "Ġzip", "(xs, ys)\n", "Zip lists"),
]

CAT7_EXTRA = [
    ("try:\n    val = int(text)\nexcept", "ValueError", "ĠValueError", ":\n    val = 0\n", "Except ValueError"),
    ("try:\n    val = items[key]\nexcept", "KeyError", "ĠKeyError", ":\n    val = None\n", "Except KeyError"),
    ("try:\n    val = items[idx]\nexcept", "IndexError", "ĠIndexError", ":\n    val = None\n", "Except IndexError"),
    ("try:\n    val = getattr(obj, attr)\nexcept", "AttributeError", "ĠAttributeError", ":\n    val = None\n", "Except AttributeError"),
    ("try:\n    execute()\nexcept", "RuntimeError", "ĠRuntimeError", ":\n    pass\n", "Except RuntimeError"),
    ("try:\n    execute()\nexcept", "TypeError", "ĠTypeError", ":\n    pass\n", "Except TypeError"),
    ("try:\n    execute()\nexcept", "ImportError", "ĠImportError", ":\n    pass\n", "Except ImportError"),
    ("try:\n    execute()\nexcept", "IOError", "ĠIOError", ":\n    pass\n", "Except IOError"),
    ("try:\n    execute()\nexcept", "OSError", "ĠOSError", ":\n    pass\n", "Except OSError"),
    ("try:\n    execute()\nexcept", "Exception", "ĠException", " as e:\n    pass\n", "Except Exception as e"),
    ("if not valid:\n    raise", "ValueError", "ĠValueError", "('Invalid input')\n", "Raise ValueError"),
    ("if not found:\n    raise", "KeyError", "ĠKeyError", "(key)\n", "Raise KeyError"),
    ("if idx >= len(arr):\n    raise", "IndexError", "ĠIndexError", "('Out of bounds')\n", "Raise IndexError"),
    ("if not hasattr(obj, name):\n    raise", "AttributeError", "ĠAttributeError", "(name)\n", "Raise AttributeError"),
    ("def abstract_method():\n    raise", "NotImplementedError", "ĠNotImplementedError", "()\n", "Raise NotImplementedError"),
    ("if not isinstance(val, int):\n    raise", "TypeError", "ĠTypeError", "('Expected int')\n", "Raise TypeError"),
    ("with open(path)", "as", "Ġas", " f:\n    text = f.read()\n", "With open as f"),
    ("with open(dest, 'w')", "as", "Ġas", " out:\n    out.write(text)\n", "With open as out"),
    ("with open(filename, 'rb')", "as", "Ġas", " stream:\n    data = stream.read()\n", "With open as stream"),
    ("with lock:\n   ", "pass", "Ġpass", "\n", "With lock pass"),
    ("try:\n    setup()\nfinally:\n   ", "close", "Ġclose", "()\n", "Finally close"),
    ("try:\n    start()\nexcept Exception as", "err", "Ġerr", ":\n    log(err)\n", "Except Exception as err"),
    ("try:\n    start()\nexcept Exception as", "e", "Ġe", ":\n    print(e)\n", "Except Exception as e"),
    ("try:\n    connect()\nexcept", "OSError", "ĠOSError", " as err:\n    pass\n", "Except OSError as err"),
    ("try:\n    parse()\nexcept", "ValueError", "ĠValueError", " as exc:\n    pass\n", "Except ValueError as exc"),
    ("if size < 0:\n    raise", "ValueError", "ĠValueError", "('size must be >= 0')\n", "Raise ValueError size"),
    ("if key not in d:\n    raise", "KeyError", "ĠKeyError", "(f'missing {key}')\n", "Raise KeyError missing"),
    ("if item is None:\n    raise", "ValueError", "ĠValueError", "('item cannot be None')\n", "Raise ValueError None"),
    ("def step():\n    raise", "NotImplementedError", "ĠNotImplementedError", "('subclasses must implement')\n", "Raise NotImplementedError step"),
    ("if count == 0:\n    raise", "ValueError", "ĠValueError", "('count must be non-zero')\n", "Raise ValueError count"),
    ("try:\n    d[k]\nexcept", "KeyError", "ĠKeyError", ":\n    d[k] = default\n", "KeyError default"),
    ("try:\n    f = open(path)\nexcept", "IOError", "ĠIOError", ":\n    f = None\n", "IOError open"),
    ("try:\n    load_module()\nexcept", "ImportError", "ĠImportError", ":\n    fallback()\n", "ImportError fallback"),
    ("with open(log_path, 'a')", "as", "Ġas", " log_file:\n    log_file.write(entry)\n", "With open append"),
    ("with tempfile.NamedTemporaryFile()", "as", "Ġas", " tmp:\n    tmp.write(b'data')\n", "With tempfile"),
    ("try:\n    calc()\nexcept", "RuntimeError", "ĠRuntimeError", " as err:\n    handle(err)\n", "RuntimeError handle"),
    ("if val < 0:\n    raise", "ValueError", "ĠValueError", "('Negative value not allowed')\n", "Raise ValueError negative"),
    ("if not isinstance(name, str):\n    raise", "TypeError", "ĠTypeError", "('Name must be string')\n", "Raise TypeError name"),
]

CAT8_EXTRA = [
    ("import", "os", "Ġos", "\n", "Import os"),
    ("import", "sys", "Ġsys", "\n", "Import sys"),
    ("import", "math", "Ġmath", "\n", "Import math"),
    ("import", "json", "Ġjson", "\n", "Import json"),
    ("import", "time", "Ġtime", "\n", "Import time"),
    ("import", "re", "Ġre", "\n", "Import re"),
    ("import", "random", "Ġrandom", "\n", "Import random"),
    ("import", "collections", "Ġcollections", "\n", "Import collections"),
    ("import", "itertools", "Ġitertools", "\n", "Import itertools"),
    ("import", "copy", "Ġcopy", "\n", "Import copy"),
    ("import", "io", "Ġio", "\n", "Import io"),
    ("import", "logging", "Ġlogging", "\n", "Import logging"),
    ("import", "csv", "Ġcsv", "\n", "Import csv"),
    ("import", "subprocess", "Ġsubprocess", "\n", "Import subprocess"),
    ("import", "threading", "Ġthreading", "\n", "Import threading"),
    ("import", "socket", "Ġsocket", "\n", "Import socket"),
    ("import", "unittest", "Ġunittest", "\n", "Import unittest"),
    ("import numpy as", "np", "Ġnp", "\n", "Import numpy as np"),
    ("from typing import", "List", "ĠList", "\n", "Import List"),
    ("from typing import", "Dict", "ĠDict", "\n", "Import Dict"),
    ("from typing import", "Set", "ĠSet", "\n", "Import Set"),
    ("from typing import", "Optional", "ĠOptional", "\n", "Import Optional"),
    ("from typing import", "Any", "ĠAny", "\n", "Import Any"),
    ("def process(items:", "List", "ĠList", "[str]) -> None:\n", "Type annotation List"),
    ("def lookup(table:", "Dict", "ĠDict", "[str, int]) -> int:\n", "Type annotation Dict"),
    ("def find_user(user_id: int) ->", "Optional", "ĠOptional", "[dict]:\n", "Type annotation Optional"),
    ("def handle(data:", "Any", "ĠAny", ") -> None:\n", "Type annotation Any"),
    ("def get_status() ->", "bool", "Ġbool", ":\n", "Return bool"),
    ("def to_text(val: int) ->", "str", "Ġstr", ":\n", "Return str"),
    ("def get_count() ->", "int", "Ġint", ":\n", "Return int"),
]

def generate_all_1000():
    cats = [
        ("Algorithms & Numerical", orig.build_category_1_algorithms, CAT1_EXTRA),
        ("Data Structures & Collections", orig.build_category_2_data_structures, CAT2_EXTRA),
        ("String & Text Processing", orig.build_category_3_strings, CAT3_EXTRA),
        ("Object-Oriented Programming", orig.build_category_4_oop, CAT4_EXTRA),
        ("Control Flow & Loops", orig.build_category_5_control_flow, CAT5_EXTRA),
        ("Built-ins & Iteration", orig.build_category_6_builtins, CAT6_EXTRA),
        ("Exception Handling & Context Managers", orig.build_category_7_exceptions, CAT7_EXTRA),
        ("Imports, Typing & Signatures", orig.build_category_8_imports_typing, CAT8_EXTRA),
    ]

    all_probes = []
    for cat_name, orig_fn, extra in cats:
        base_list = get_valid_from_orig(orig_fn)
        final_list = add_curated(base_list, extra, target_count=125)
        print(f"✓ {cat_name}: exactly {len(final_list)} verified probes")
        for p in final_list:
            p["category"] = cat_name
            p["id"] = f"probe_{len(all_probes):04d}"
            p["prefix"] = p["prompt"]
            # Strict verification
            ok, res = verify(p["prompt"], p["target"], p["target_bpe"])
            assert ok, f"Probe {p['id']} failed verification: {res}"
            all_probes.append(p)

    assert len(all_probes) == 1000, f"Expected 1000, got {len(all_probes)}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_probes, f, indent=2)

    print(f"\n=======================================================")
    print(f"✓ ALL 1,000 PROBES 100% VERIFIED AND SAVED TO:")
    print(f"  {OUTPUT_PATH}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    generate_all_1000()
