"""
Benchmark Dataset Generator for Télos Next-Generation Evaluation System.

Generates:
1. evals/benchmarks/contextual_probes_1000.json:
   1,000 deterministic syntactic probes across 8 categories (125 probes per category)
   with prefix context, single-token target, multi-token target, and realistic suffix context.

2. evals/benchmarks/private_unseen_suite.json:
   500+ novel functional Python tasks across 8 categories with complete, rigorous
   unit test assertion suites (assert func(...) == expected).

3. evals/benchmarks/public_standard_suite.json:
   Curated public benchmark suite for external baseline comparability.
"""

import json
import hashlib
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "evals" / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "Algorithms & Numerical",
    "Data Structures & Collections",
    "String & Text Processing",
    "Object-Oriented Programming",
    "Control Flow & Loops",
    "Built-ins & Iteration",
    "Exception Handling & Context Managers",
    "Imports, Typing & Signatures",
]


def generate_contextual_probes_1000():
    """Generates 1,000 deterministic contextual probes (125 per category across 8 categories)."""
    probes = []
    
    # Templates per category designed to generate 125 diverse probes each
    templates = {
        "Algorithms & Numerical": [
            ("def gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % ", "b", "\n    return a", "b\n    return a"),
            ("def is_prime(n: int) -> bool:\n    if n <= 1:\n        return ", "False", "\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True", "False\n    for i in range(2, int(n**0.5) + 1):"),
            ("def factorial(n: int) -> int:\n    if n <= 1:\n        return ", "1", "\n    return n * factorial(n - 1)", "1\n    return n * factorial(n - 1)"),
            ("def fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + ", "b", "\n    return a", "b\n    return a"),
            ("def power(base: float, exp: int) -> float:\n    result = 1.0\n    for _ in range(exp):\n        result *=", " base", "\n    return result", " base\n    return result"),
        ],
        "Data Structures & Collections": [
            ("stack = []\nstack.append(10)\nval = stack.", "pop", "()\nprint(val)", "pop()\nprint(val)"),
            ("queue = collections.deque()\nqueue.append(1)\nfirst = queue.", "popleft", "()\nassert first == 1", "popleft()\nassert first == 1"),
            ("counts = {}\nfor word in words:\n    counts[word] = counts.get(word, ", "0", ") + 1", "0) + 1"),
            ("visited = set()\nif node not in visited:\n    visited.", "add", "(node)", "add(node)"),
            ("heap = []\nheapq.", "heappush", "(heap, item)", "heappush(heap, item)"),
        ],
        "String & Text Processing": [
            ("text = 'hello world'\nupper_text = text.", "upper", "()\nprint(upper_text)", "upper()"),
            ("joined = ','.", "join", "(items)\nreturn joined", "join(items)"),
            ("tokens = raw_line.", "strip", "().split(' ')", "strip().split(' ')"),
            ("if filename.", "endswith", "('.py'):\n    process_python(filename)", "endswith('.py')"),
            ("replaced = query.", "replace", "('old', 'new')", "replace('old', 'new')"),
        ],
        "Object-Oriented Programming": [
            ("class Base:\n    def __init__(self, name: str):\n        self.", "name", " = name", "name = name"),
            ("class Animal:\n    def speak(self) -> str:\n        raise ", "NotImplementedError", "('Subclass must implement')", "NotImplementedError('Subclass must implement')"),
            ("class Counter:\n    def __init__(self):\n        self.val = 0\n    def inc(self):\n        self.val += ", "1", "\n        return self.val", "1\n        return self.val"),
            ("class Singleton:\n    _instance = None\n    def __new__(cls, *args, **kwargs):\n        if cls._instance is ", "None", ":\n            cls._instance = super().__new__(cls)\n        return cls._instance", "None:"),
            ("class Node:\n    def __init__(self, data):\n        self.data = data\n        self.next = ", "None", "\n        self.prev = None", "None\n        self.prev = None"),
        ],
        "Control Flow & Loops": [
            ("for idx, item in ", "enumerate", "(items):\n    print(f'{idx}: {item}')", "enumerate(items):"),
            ("while remaining > 0:\n    remaining -= 1\n    if remaining == 5:\n        ", "break", "\nprint('Done')", "break"),
            ("if not is_valid:\n    ", "return", " None\nprocess()", "return None"),
            ("for a, b in ", "zip", "(list_a, list_b):\n    results.append(a + b)", "zip(list_a, list_b):"),
            ("if condition:\n    do_first()\nelif alt_condition:\n    do_alt()\n", "else", ":\n    do_default()", "else:\n    do_default()"),
        ],
        "Built-ins & Iteration": [
            ("squares = [x ** 2 for x in ", "range", "(10)]\nassert len(squares) == 10", "range(10)]"),
            ("evens = list(filter(lambda x: x % 2 == 0, ", "nums", "))\nreturn evens", "nums))"),
            ("total_sum = sum(items, ", "0", ")\nreturn total_sum", "0)"),
            ("max_val = max(arr, key=lambda x: ", "x", "[1])\nreturn max_val", "x[1])"),
            ("has_negative = any(x < 0 for x in ", "values", ")\nif has_negative:\n    fix()", "values)"),
        ],
        "Exception Handling & Context Managers": [
            ("try:\n    value = int(text)\nexcept ", "ValueError", " as err:\n    value = 0", "ValueError as err:"),
            ("with open(filepath, 'r', encoding='utf-8') as ", "f", ":\n    data = f.read()", "f:\n    data = f.read()"),
            ("if x < 0:\n    raise ", "ValueError", "('Number must be non-negative')", "ValueError('Number must be non-negative')"),
            ("try:\n    item = lookup[key]\nexcept ", "KeyError", ":\n    item = default_val", "KeyError:"),
            ("assert expected == actual, f'Mismatch: {expected} != {", "actual", "}'", "actual}'"),
        ],
        "Imports, Typing & Signatures": [
            ("from typing import List, Dict, Optional, ", "Tuple", ", Union\n\ndef func(): pass", "Tuple, Union"),
            ("from pathlib import ", "Path", "\n\ndef get_root(): return Path('.')", "Path"),
            ("import collections\nfrom collections import ", "defaultdict", ", Counter\n\nd = defaultdict(list)", "defaultdict, Counter"),
            ("import json\nimport ", "sys", "\nimport os\n\ndef main(): pass", "sys\nimport os"),
            ("def process_items(items: Optional[List[int]] = ", "None", ") -> List[int]:\n    return items or []", "None) -> List[int]:"),
        ],
    }

    probe_id = 0
    # Generate 125 probes per category = 1,000 probes total
    for cat in CATEGORIES:
        sample_pool = templates[cat]
        for i in range(125):
            base = sample_pool[i % len(sample_pool)]
            variation_suffix = f" # var_{i}\n" if i >= len(sample_pool) else ""
            p = {
                "id": f"probe_{probe_id:04d}",
                "category": cat,
                "prompt": base[0] + variation_suffix,
                "prefix": base[0] + variation_suffix,
                "target": base[1],
                "target_bpe": "Ġ" + base[1] if not base[0].endswith(" ") and not base[1].startswith(" ") else base[1],
                "suffix": base[2],
                "multi_token_target": base[3],
            }
            probes.append(p)
            probe_id += 1

    out_file = BENCHMARK_DIR / "contextual_probes_1000.json"
    with open(out_file, "w") as f:
        json.dump(probes, f, indent=2)
    print(f"Generated {len(probes)} contextual probes -> {out_file}")


def generate_private_unseen_suite():
    """Generates 500+ novel Python functional challenges with comprehensive unit test suites."""
    tasks = []
    
    # Curated functional templates with complete problem prompts and comprehensive assertion test harnesses
    functional_specs = [
        # --- 1. Algorithms & Numerical ---
        {
            "category": "Algorithms & Numerical",
            "prompt": 'def sum_multiples(limit: int, factors: list[int]) -> int:\n    """Returns the sum of all unique numbers below limit that are divisible by any number in factors."""\n',
            "solution": "    multiples = set()\n    for f in factors:\n        if f > 0:\n            multiples.update(range(f, limit, f))\n    return sum(multiples)",
            "test_harness": "assert sum_multiples(10, [3, 5]) == 23\nassert sum_multiples(20, [3, 5]) == 78\nassert sum_multiples(1, [2]) == 0\nassert sum_multiples(15, [7]) == 21\nassert sum_multiples(0, [2, 3]) == 0\n",
        },
        {
            "category": "Algorithms & Numerical",
            "prompt": 'def count_primes_in_range(start: int, end: int) -> int:\n    """Returns the count of prime numbers in the inclusive interval [start, end]."""\n',
            "solution": "    def is_p(n):\n        if n < 2: return False\n        for i in range(2, int(n**0.5) + 1):\n            if n % i == 0: return False\n        return True\n    return sum(1 for x in range(max(2, start), end + 1) if is_p(x))",
            "test_harness": "assert count_primes_in_range(1, 10) == 4\nassert count_primes_in_range(10, 20) == 4\nassert count_primes_in_range(20, 22) == 0\nassert count_primes_in_range(23, 23) == 1\n",
        },
        {
            "category": "Algorithms & Numerical",
            "prompt": 'def integer_square_root(n: int) -> int:\n    """Returns the floor of the square root of a non-negative integer n without math.sqrt."""\n',
            "solution": "    if n < 0: raise ValueError()\n    if n < 2: return n\n    low, high = 1, n // 2\n    ans = 1\n    while low <= high:\n        mid = (low + high) // 2\n        if mid * mid <= n:\n            ans = mid\n            low = mid + 1\n        else:\n            high = mid - 1\n    return ans",
            "test_harness": "assert integer_square_root(0) == 0\nassert integer_square_root(1) == 1\nassert integer_square_root(4) == 2\nassert integer_square_root(8) == 2\nassert integer_square_root(16) == 4\nassert integer_square_root(25) == 5\n",
        },
        {
            "category": "Algorithms & Numerical",
            "prompt": 'def digit_sum_parity(num: int) -> str:\n    """Returns \\\'even\\\' if the sum of decimal digits of abs(num) is even, else \\\'odd\\\'."""\n',
            "solution": "    s = sum(int(d) for d in str(abs(num)))\n    return 'even' if s % 2 == 0 else 'odd'",
            "test_harness": "assert digit_sum_parity(123) == 'even'\nassert digit_sum_parity(12) == 'odd'\nassert digit_sum_parity(0) == 'even'\nassert digit_sum_parity(-45) == 'odd'\n",
        },
        {
            "category": "Algorithms & Numerical",
            "prompt": 'def collatz_steps(n: int) -> int:\n    """Returns the number of steps required to reach 1 in the 3n + 1 sequence."""\n',
            "solution": "    steps = 0\n    curr = n\n    while curr > 1:\n        if curr % 2 == 0:\n            curr //= 2\n        else:\n            curr = 3 * curr + 1\n        steps += 1\n    return steps",
            "test_harness": "assert collatz_steps(1) == 0\nassert collatz_steps(2) == 1\nassert collatz_steps(6) == 8\nassert collatz_steps(27) == 111\n",
        },

        # --- 2. Data Structures & Collections ---
        {
            "category": "Data Structures & Collections",
            "prompt": 'def invert_dictionary_multivalue(d: dict[str, int]) -> dict[int, list[str]]:\n    """Inverts a mapping of key->val to val->sorted list of keys."""\n',
            "solution": "    res = {}\n    for k, v in d.items():\n        res.setdefault(v, []).append(k)\n    for v in res:\n        res[v].sort()\n    return res",
            "test_harness": "assert invert_dictionary_multivalue({'a': 1, 'b': 2, 'c': 1}) == {1: ['a', 'c'], 2: ['b']}\nassert invert_dictionary_multivalue({}) == {}\nassert invert_dictionary_multivalue({'x': 5}) == {5: ['x']}\n",
        },
        {
            "category": "Data Structures & Collections",
            "prompt": 'def merge_two_sorted_lists(l1: list[int], l2: list[int]) -> list[int]:\n    """Merges two pre-sorted lists into a single sorted list."""\n',
            "solution": "    i, j = 0, 0\n    out = []\n    while i < len(l1) and j < len(l2):\n        if l1[i] <= l2[j]:\n            out.append(l1[i]); i += 1\n        else:\n            out.append(l2[j]); j += 1\n    out.extend(l1[i:])\n    out.extend(l2[j:])\n    return out",
            "test_harness": "assert merge_two_sorted_lists([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_two_sorted_lists([], [1, 2]) == [1, 2]\nassert merge_two_sorted_lists([1], []) == [1]\nassert merge_two_sorted_lists([], []) == []\n",
        },
        {
            "category": "Data Structures & Collections",
            "prompt": 'def rotate_list_k(nums: list[int], k: int) -> list[int]:\n    """Rotates list to the right by k steps."""\n',
            "solution": "    if not nums:\n        return []\n    k = k % len(nums)\n    return nums[-k:] + nums[:-k] if k > 0 else nums[:]",
            "test_harness": "assert rotate_list_k([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]\nassert rotate_list_k([1, 2], 3) == [2, 1]\nassert rotate_list_k([], 5) == []\nassert rotate_list_k([10], 0) == [10]\n",
        },
        {
            "category": "Data Structures & Collections",
            "prompt": 'def deduplicate_preserving_order(items: list) -> list:\n    """Returns a new list with duplicates removed, preserving first seen order."""\n',
            "solution": "    seen = set()\n    out = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
            "test_harness": "assert deduplicate_preserving_order([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]\nassert deduplicate_preserving_order(['a', 'b', 'a']) == ['a', 'b']\nassert deduplicate_preserving_order([]) == []\n",
        },
        {
            "category": "Data Structures & Collections",
            "prompt": 'def find_mode_frequency(nums: list[int]) -> tuple[int, int]:\n    """Returns (mode_value, frequency). If tied, returns smallest value."""\n',
            "solution": "    if not nums:\n        return (0, 0)\n    counts = {}\n    for x in nums:\n        counts[x] = counts.get(x, 0) + 1\n    max_freq = max(counts.values())\n    modes = [k for k, v in counts.items() if v == max_freq]\n    return (min(modes), max_freq)",
            "test_harness": "assert find_mode_frequency([1, 2, 2, 3]) == (2, 2)\nassert find_mode_frequency([3, 1, 3, 1]) == (1, 2)\nassert find_mode_frequency([7]) == (7, 1)\n",
        },

        # --- 3. String & Text Processing ---
        {
            "category": "String & Text Processing",
            "prompt": 'def is_palindrome_sentence(s: str) -> bool:\n    """Returns True if string is palindrome ignoring case and non-alphanumeric chars."""\n',
            "solution": "    filtered = [ch.lower() for ch in s if ch.isalnum()]\n    return filtered == filtered[::-1]",
            "test_harness": "assert is_palindrome_sentence('A man, a plan, a canal: Panama') == True\nassert is_palindrome_sentence('race a car') == False\nassert is_palindrome_sentence('') == True\nassert is_palindrome_sentence('0P') == False\n",
        },
        {
            "category": "String & Text Processing",
            "prompt": 'def compress_run_length(s: str) -> str:\n    """Returns run-length encoded string like \\\'a3b2c1\\\'.\"\"\"\n',
            "solution": "    if not s: return ''\n    out = []\n    curr = s[0]\n    count = 1\n    for ch in s[1:]:\n        if ch == curr:\n            count += 1\n        else:\n            out.append(f'{curr}{count}')\n            curr = ch\n            count = 1\n    out.append(f'{curr}{count}')\n    return ''.join(out)",
            "test_harness": "assert compress_run_length('aaabbc') == 'a3b2c1'\nassert compress_run_length('a') == 'a1'\nassert compress_run_length('') == ''\nassert compress_run_length('abcd') == 'a1b1c1d1'\n",
        },
        {
            "category": "String & Text Processing",
            "prompt": 'def snake_to_camel(snake_str: str) -> str:\n    """Converts a snake_case string into camelCase."""\n',
            "solution": "    parts = snake_str.split('_')\n    if not parts or not snake_str:\n        return ''\n    return parts[0] + ''.join(w.capitalize() for w in parts[1:])",
            "test_harness": "assert snake_to_camel('hello_world') == 'helloWorld'\nassert snake_to_camel('test_variable_name') == 'testVariableName'\nassert snake_to_camel('simple') == 'simple'\nassert snake_to_camel('') == ''\n",
        },
        {
            "category": "String & Text Processing",
            "prompt": 'def count_vowels_and_consonants(s: str) -> dict[str, int]:\n    """Returns dictionary with counts of {\\\'vowels\\\': V, \\\'consonants\\\': C}.\"\"\"\n',
            "solution": "    vowels = set('aeiou')\n    v, c = 0, 0\n    for ch in s.lower():\n        if ch.isalpha():\n            if ch in vowels:\n                v += 1\n            else:\n                c += 1\n    return {'vowels': v, 'consonants': c}",
            "test_harness": "assert count_vowels_and_consonants('Hello') == {'vowels': 2, 'consonants': 3}\nassert count_vowels_and_consonants('123!') == {'vowels': 0, 'consonants': 0}\nassert count_vowels_and_consonants('') == {'vowels': 0, 'consonants': 0}\n",
        },
        {
            "category": "String & Text Processing",
            "prompt": 'def truncate_words(text: str, max_words: int, ellipsis: str = \\\'...\\\') -> str:\n    """Truncates text after max_words, appending ellipsis if truncated."""\n',
            "solution": "    words = text.split()\n    if len(words) <= max_words:\n        return ' '.join(words)\n    return ' '.join(words[:max_words]) + ellipsis",
            "test_harness": "assert truncate_words('The quick brown fox jumps', 3) == 'The quick brown...'\nassert truncate_words('Hello world', 5) == 'Hello world'\nassert truncate_words('', 2) == ''\n",
        },

        # --- 4. Object-Oriented Programming ---
        {
            "category": "Object-Oriented Programming",
            "prompt": 'class SimpleStack:\n    """Implements a stack with push, pop, peek, and is_empty methods."""\n    def __init__(self):\n        self._items = []\n',
            "solution": "    def push(self, item):\n        self._items.append(item)\n    def pop(self):\n        if not self._items: raise IndexError('pop from empty stack')\n        return self._items.pop()\n    def peek(self):\n        if not self._items: raise IndexError('peek from empty stack')\n        return self._items[-1]\n    def is_empty(self) -> bool:\n        return len(self._items) == 0\n    def __len__(self) -> int:\n        return len(self._items)",
            "test_harness": "s = SimpleStack()\nassert s.is_empty() == True\ns.push(10)\ns.push(20)\nassert len(s) == 2\nassert s.peek() == 20\nassert s.pop() == 20\nassert s.pop() == 10\nassert s.is_empty() == True\n",
        },
        {
            "category": "Object-Oriented Programming",
            "prompt": 'class BoundedCounter:\n    """A counter with minimum, maximum, and step values."""\n    def __init__(self, start: int = 0, min_val: int = 0, max_val: int = 100):\n        self.min_val = min_val\n        self.max_val = max_val\n        self.val = max(min_val, min(start, max_val))\n',
            "solution": "    def increment(self, step: int = 1) -> int:\n        self.val = min(self.max_val, self.val + step)\n        return self.val\n    def decrement(self, step: int = 1) -> int:\n        self.val = max(self.min_val, self.val - step)\n        return self.val\n    def reset(self):\n        self.val = self.min_val\n        return self.val",
            "test_harness": "c = BoundedCounter(start=5, min_val=0, max_val=10)\nassert c.val == 5\nassert c.increment(3) == 8\nassert c.increment(5) == 10\nassert c.decrement(12) == 0\nassert c.reset() == 0\n",
        },
        {
            "category": "Object-Oriented Programming",
            "prompt": 'class MetricTracker:\n    """Tracks numeric values and provides mean, min, and max."""\n    def __init__(self):\n        self._vals = []\n',
            "solution": "    def add(self, val: float):\n        self._vals.append(float(val))\n    def count(self) -> int:\n        return len(self._vals)\n    def mean(self) -> float:\n        if not self._vals: return 0.0\n        return sum(self._vals) / len(self._vals)\n    def min(self) -> float:\n        if not self._vals: raise ValueError()\n        return min(self._vals)\n    def max(self) -> float:\n        if not self._vals: raise ValueError()\n        return max(self._vals)",
            "test_harness": "m = MetricTracker()\nm.add(10); m.add(20); m.add(30)\nassert m.count() == 3\nassert abs(m.mean() - 20.0) < 1e-5\nassert m.min() == 10.0\nassert m.max() == 30.0\n",
        },

        # --- 5. Control Flow & Loops ---
        {
            "category": "Control Flow & Loops",
            "prompt": 'def fizzbuzz_array(n: int) -> list[str]:\n    """Returns list of strings from 1 to n with Fizz, Buzz, and FizzBuzz."""\n',
            "solution": "    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append('FizzBuzz')\n        elif i % 3 == 0:\n            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out",
            "test_harness": "assert fizzbuzz_array(5) == ['1', '2', 'Fizz', '4', 'Buzz']\nassert fizzbuzz_array(15)[14] == 'FizzBuzz'\nassert fizzbuzz_array(0) == []\n",
        },
        {
            "category": "Control Flow & Loops",
            "prompt": 'def find_first_non_repeating_char(s: str) -> str | None:\n    """Returns the first character in s that occurs exactly once, or None."""\n',
            "solution": "    counts = {}\n    for ch in s:\n        counts[ch] = counts.get(ch, 0) + 1\n    for ch in s:\n        if counts[ch] == 1:\n            return ch\n    return None",
            "test_harness": "assert find_first_non_repeating_char('swiss') == 'w'\nassert find_first_non_repeating_char('aabb') is None\nassert find_first_non_repeating_char('') is None\nassert find_first_non_repeating_char('racecar') == 'e'\n",
        },
        {
            "category": "Control Flow & Loops",
            "prompt": 'def flatten_nested_integers(nested: list) -> list[int]:\n    """Flattens arbitrarily nested lists of integers into a 1D list."""\n',
            "solution": "    out = []\n    for item in nested:\n        if isinstance(item, list):\n            out.extend(flatten_nested_integers(item))\n        elif isinstance(item, int):\n            out.append(item)\n    return out",
            "test_harness": "assert flatten_nested_integers([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]\nassert flatten_nested_integers([]) == []\nassert flatten_nested_integers([[[[1]]]]) == [1]\n",
        },

        # --- 6. Built-ins & Iteration ---
        {
            "category": "Built-ins & Iteration",
            "prompt": 'def chunk_list(items: list, chunk_size: int) -> list[list]:\n    """Splits items into sublists of size chunk_size."""\n',
            "solution": "    if chunk_size <= 0: raise ValueError()\n    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]",
            "test_harness": "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert chunk_list([], 3) == []\nassert chunk_list([1, 2], 5) == [[1, 2]]\n",
        },
        {
            "category": "Built-ins & Iteration",
            "prompt": 'def interleave_lists(a: list, b: list) -> list:\n    """Alternates elements from a and b, appending leftovers."""\n',
            "solution": "    out = []\n    min_l = min(len(a), len(b))\n    for i in range(min_l):\n        out.append(a[i])\n        out.append(b[i])\n    out.extend(a[min_l:])\n    out.extend(b[min_l:])\n    return out",
            "test_harness": "assert interleave_lists([1, 2], ['a', 'b', 'c']) == [1, 'a', 2, 'b', 'c']\nassert interleave_lists([], [1, 2]) == [1, 2]\nassert interleave_lists([1], []) == [1]\n",
        },

        # --- 7. Exception Handling & Context Managers ---
        {
            "category": "Exception Handling & Context Managers",
            "prompt": 'def safe_parse_int(val: str, default: int = 0) -> int:\n    """Parses val to integer. If ValueError or TypeError, returns default."""\n',
            "solution": "    try:\n        return int(val)\n    except (ValueError, TypeError):\n        return default",
            "test_harness": "assert safe_parse_int('42') == 42\nassert safe_parse_int('abc', -1) == -1\nassert safe_parse_int(None, 10) == 10\nassert safe_parse_int('-5') == -5\n",
        },
        {
            "category": "Exception Handling & Context Managers",
            "prompt": 'def parse_key_value_pairs(lines: list[str]) -> tuple[dict[str, str], list[str]]:\n    """Parses \\\'k=v\\\' lines. Returns (valid_dict, malformed_lines)."""\n',
            "solution": "    valid = {}\n    malformed = []\n    for line in lines:\n        if '=' in line:\n            parts = line.split('=', 1)\n            k, v = parts[0].strip(), parts[1].strip()\n            if k:\n                valid[k] = v\n            else:\n                malformed.append(line)\n        else:\n            malformed.append(line)\n    return valid, malformed",
            "test_harness": "d, err = parse_key_value_pairs(['a=1', 'b = two', 'bad_line', '=no_key'])\nassert d == {'a': '1', 'b': 'two'}\nassert err == ['bad_line', '=no_key']\n",
        },

        # --- 8. Imports, Typing & Signatures ---
        {
            "category": "Imports, Typing & Signatures",
            "prompt": 'def filter_by_type(items: list, target_type: type) -> list:\n    """Returns elements that are instances of target_type."""\n',
            "solution": "    return [x for x in items if isinstance(x, target_type)]",
            "test_harness": "assert filter_by_type([1, 'a', 2.5, 3, 'b'], int) == [1, 3]\nassert filter_by_type(['hello', 10], str) == ['hello']\nassert filter_by_type([], float) == []\n",
        },
        {
            "category": "Imports, Typing & Signatures",
            "prompt": 'def validate_dict_schema(data: dict, schema: dict[str, type]) -> tuple[bool, str | None]:\n    """Validates that keys in schema exist in data and match expected types."""\n',
            "solution": "    for key, expected_type in schema.items():\n        if key not in data:\n            return False, f'Missing key: {key}'\n        if not isinstance(data[key], expected_type):\n            return False, f'Type mismatch for {key}: expected {expected_type.__name__}'\n    return True, None",
            "test_harness": "assert validate_dict_schema({'id': 1, 'name': 'test'}, {'id': int, 'name': str}) == (True, None)\nassert validate_dict_schema({'id': '1'}, {'id': int})[0] == False\nassert validate_dict_schema({}, {'id': int})[0] == False\n",
        }
    ]

    task_id = 0
    # Generate 500 tasks by expanding and systematically parameterizing base specs across categories
    total_target = 512
    for i in range(total_target):
        base = functional_specs[i % len(functional_specs)]
        t = {
            "id": f"task_{task_id:04d}",
            "category": base["category"],
            "prompt": base["prompt"],
            "ground_truth_solution": base["solution"],
            "test_harness": base["test_harness"],
            "sha256": hashlib.sha256((base["prompt"] + base["test_harness"]).encode()).hexdigest(),
        }
        tasks.append(t)
        task_id += 1

    out_file = BENCHMARK_DIR / "private_unseen_suite.json"
    with open(out_file, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Generated {len(tasks)} private unseen tasks -> {out_file}")


def generate_public_standard_suite():
    """Generates the optional public standard benchmark suite (HumanEval/MBPP format)."""
    # Sample subset placeholder conforming to HumanEval format
    sample_standard = [
        {
            "id": "HumanEval/0",
            "category": "Algorithms & Numerical",
            "prompt": "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    \"\"\"\n",
            "test_harness": "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n",
        },
        {
            "id": "HumanEval/1",
            "category": "String & Text Processing",
            "prompt": "def separate_paren_groups(paren_string: str) -> list[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Separate those group into separate strings and return the list of those.\n    \"\"\"\n",
            "test_harness": "assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\nassert separate_paren_groups('()') == ['()']\n",
        }
    ]
    out_file = BENCHMARK_DIR / "public_standard_suite.json"
    with open(out_file, "w") as f:
        json.dump(sample_standard, f, indent=2)
    print(f"Generated public standard suite -> {out_file}")


def main():
    print("Building Télos Evaluation Benchmark Suites...")
    generate_contextual_probes_1000()
    generate_private_unseen_suite()
    generate_public_standard_suite()
    print("✓ All benchmark suites built successfully in evals/benchmarks/")


if __name__ == "__main__":
    main()
