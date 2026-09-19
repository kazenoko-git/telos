"""
Category 7: Exception Handling & Context Managers (64 distinct challenges).
"""

from typing import List, Dict, Any
from .common import create_task


def build_exceptions_context_category() -> List[Dict[str, Any]]:
    tasks = []

    # 1. Safe Integer Cast
    tasks.append(create_task(
        name="safe_int_cast",
        signature="def safe_int_cast(val: Any, default: int = 0) -> int:",
        doc_desc="Safely attempts to convert val to an integer. Catches ValueError and TypeError, returning default if conversion fails.",
        doctests=[">>> safe_int_cast('123')", "123", ">>> safe_int_cast('abc', -1)", "-1"],
        hint="Wrap int(val) in a try-except block catching (ValueError, TypeError).",
        solution="    try:\n        return int(val)\n    except (ValueError, TypeError):\n        return default",
        test="assert safe_int_cast('123') == 123\nassert safe_int_cast('abc', -1) == -1\nassert safe_int_cast(None, 42) == 42\nassert safe_int_cast(3.14) == 3\n"
    ))

    # 2. Safe Division
    tasks.append(create_task(
        name="safe_divide",
        signature="def safe_divide(a: Any, b: Any, fallback: float = 0.0) -> float:",
        doc_desc="Performs float division a / b. Returns fallback if ZeroDivisionError or TypeError is encountered.",
        doctests=[">>> safe_divide(10, 2)", "5.0", ">>> safe_divide(5, 0, -1.0)", "-1.0"],
        hint="Use try/except (ZeroDivisionError, TypeError) around float(a) / float(b).",
        solution="    try:\n        return float(a) / float(b)\n    except (ZeroDivisionError, TypeError, ValueError):\n        return fallback",
        test="assert safe_divide(10, 2) == 5.0\nassert safe_divide(5, 0, -1.0) == -1.0\nassert safe_divide('10', '2') == 5.0\nassert safe_divide('x', 2, 99.0) == 99.0\n"
    ))

    # 3. Parse Key-Value String with Validation
    tasks.append(create_task(
        name="parse_key_value_string",
        signature="def parse_key_value_string(text: str) -> dict[str, str]:",
        doc_desc="Parses a semicolon-delimited string of key=value pairs into a dict. Raises ValueError if any pair lacks '=' or contains multiple '='.",
        doctests=[">>> parse_key_value_string('a=1;b=2')", "{'a': '1', 'b': '2'}"],
        hint="Split by ';', strip whitespace. For each segment, check if it contains '=' and splits into exactly 2 parts, otherwise raise ValueError.",
        solution="    result = {}\n    if not text.strip():\n        return result\n    for part in text.split(';'):\n        part = part.strip()\n        if not part:\n            continue\n        pieces = part.split('=')\n        if len(pieces) != 2:\n            raise ValueError(f'Malformed pair: {part}')\n        result[pieces[0].strip()] = pieces[1].strip()\n    return result",
        test="assert parse_key_value_string('a=1;b=2') == {'a': '1', 'b': '2'}\nassert parse_key_value_string('') == {}\ntry:\n    parse_key_value_string('a=1;broken')\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 4. Age Validator
    tasks.append(create_task(
        name="validate_age",
        signature="def validate_age(age: Any) -> int:",
        doc_desc="Validates age: must be an integer (not bool) between 0 and 150 inclusive. Raises TypeError if not int, ValueError if out of bounds.",
        doctests=[">>> validate_age(25)", "25"],
        hint="Check isinstance(age, int) and not isinstance(age, bool). Then check 0 <= age <= 150.",
        solution="    if not isinstance(age, int) or isinstance(age, bool):\n        raise TypeError('Age must be an integer')\n    if age < 0 or age > 150:\n        raise ValueError('Age must be between 0 and 150')\n    return age",
        test="assert validate_age(25) == 25\ntry:\n    validate_age(True)\n    assert False\nexcept TypeError:\n    pass\ntry:\n    validate_age(-5)\n    assert False\nexcept ValueError:\n    pass\ntry:\n    validate_age(200)\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 5. Bank Account with Custom Exception
    tasks.append(create_task(
        name="BankAccount",
        signature="class BankAccount:",
        prefix_code="class InsufficientFundsError(Exception):\n    def __init__(self, balance: float, amount: float):\n        super().__init__(f'Cannot withdraw {amount} with balance {balance}')\n        self.balance = balance\n        self.amount = amount\n",
        doc_desc="Bank account tracking float balance. withdraw(amount) raises InsufficientFundsError if amount > balance, ValueError if amount <= 0.",
        doctests=[">>> acc = BankAccount(100.0)", ">>> acc.deposit(50.0)", "150.0"],
        hint="In withdraw(), check amount <= 0 first (raise ValueError), then check amount > self.balance (raise InsufficientFundsError).",
        solution="    def __init__(self, initial_balance: float = 0.0):\n        self.balance = float(initial_balance)\n    def deposit(self, amount: float) -> float:\n        if amount <= 0:\n            raise ValueError('Deposit amount must be positive')\n        self.balance += amount\n        return self.balance\n    def withdraw(self, amount: float) -> float:\n        if amount <= 0:\n            raise ValueError('Withdrawal amount must be positive')\n        if amount > self.balance:\n            raise InsufficientFundsError(self.balance, amount)\n        self.balance -= amount\n        return self.balance",
        test="acc = BankAccount(100.0)\nassert acc.deposit(50.0) == 150.0\nassert acc.withdraw(70.0) == 80.0\ntry:\n    acc.withdraw(100.0)\n    assert False\nexcept InsufficientFundsError as e:\n    assert e.balance == 80.0 and e.amount == 100.0\ntry:\n    acc.withdraw(-10.0)\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 6. Retry Operation with Allowed Exceptions
    tasks.append(create_task(
        name="retry_operation",
        signature="def retry_operation(func: Any, max_attempts: int, allowed_exceptions: tuple) -> Any:",
        doc_desc="Calls zero-argument func up to max_attempts times. If an allowed exception is raised, retries. If max_attempts exhausted, re-raises.",
        doctests=[">>> retry_operation(lambda: 42, 3, (ValueError,))", "42"],
        hint="Loop range(max_attempts). Inside try/except allowed_exceptions: if attempt == max_attempts - 1, re-raise.",
        solution="    for attempt in range(max_attempts):\n        try:\n            return func()\n        except allowed_exceptions as exc:\n            if attempt == max_attempts - 1:\n                raise exc",
        test="calls = [0]\ndef flaky():\n    calls[0] += 1\n    if calls[0] < 3:\n        raise ValueError('Temporary')\n    return 'success'\nassert retry_operation(flaky, 5, (ValueError,)) == 'success'\nassert calls[0] == 3\n"
    ))

    # 7. Timer Context Manager
    tasks.append(create_task(
        name="TimerContext",
        signature="class TimerContext:",
        prefix_code="import time\n",
        doc_desc="Context manager measuring execution time of a code block. Exposes elapsed property (float seconds).",
        doctests=[">>> with TimerContext() as t:", "...     x = sum(range(1000))", ">>> t.elapsed >= 0.0", "True"],
        hint="Use time.perf_counter() in __enter__ and __exit__. Store end - start in self.elapsed.",
        solution="    def __init__(self):\n        self.start_time = 0.0\n        self.elapsed = 0.0\n    def __enter__(self):\n        self.start_time = time.perf_counter()\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        self.elapsed = time.perf_counter() - self.start_time\n        return False",
        test="with TimerContext() as timer:\n    _ = [i * 2 for i in range(10000)]\nassert timer.elapsed > 0.0\n"
    ))

    # 8. Suppress Exceptions Context Manager
    tasks.append(create_task(
        name="SuppressExceptions",
        signature="class SuppressExceptions:",
        doc_desc="Context manager that suppresses specified exception types raised within its block.",
        doctests=[">>> with SuppressExceptions(KeyError):", "...     d = {}", "...     _ = d['missing']"],
        hint="In __init__, accept *exceptions. In __exit__, check if exc_type is not None and issubclass(exc_type, self.exceptions). Return True to suppress.",
        solution="    def __init__(self, *exceptions):\n        self.exceptions = exceptions\n    def __enter__(self):\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if exc_type is not None and issubclass(exc_type, self.exceptions):\n            return True\n        return False",
        test="with SuppressExceptions(KeyError, ValueError):\n    raise KeyError('test')\ntry:\n    with SuppressExceptions(KeyError):\n        raise ValueError('test')\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 9. Temporary Attribute Context Manager
    tasks.append(create_task(
        name="TemporaryAttribute",
        signature="class TemporaryAttribute:",
        doc_desc="Context manager temporarily setting an attribute on an object, restoring the previous value or deleting it on exit.",
        doctests=[">>> class Obj: pass", ">>> o = Obj()", ">>> with TemporaryAttribute(o, 'tag', 'temp'):", "...     val = o.tag", ">>> hasattr(o, 'tag')", "False"],
        hint="In __enter__, check hasattr. Save whether it existed and old value. Set new value. In __exit__, restore or delattr.",
        solution="    def __init__(self, target: Any, attr_name: str, temp_value: Any):\n        self.target = target\n        self.attr_name = attr_name\n        self.temp_value = temp_value\n        self.had_attr = False\n        self.prev_value = None\n    def __enter__(self):\n        self.had_attr = hasattr(self.target, self.attr_name)\n        if self.had_attr:\n            self.prev_value = getattr(self.target, self.attr_name)\n        setattr(self.target, self.attr_name, self.temp_value)\n        return self.target\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if self.had_attr:\n            setattr(self.target, self.attr_name, self.prev_value)\n        else:\n            delattr(self.target, self.attr_name)\n        return False",
        test="class Target:\n    val = 10\nt = Target()\nwith TemporaryAttribute(t, 'val', 99):\n    assert t.val == 99\nassert t.val == 10\nwith TemporaryAttribute(t, 'new_attr', 'hello'):\n    assert t.new_attr == 'hello'\nassert not hasattr(t, 'new_attr')\n"
    ))

    # 10. Scoped List Append Context Manager
    tasks.append(create_task(
        name="ScopedListAppend",
        signature="class ScopedListAppend:",
        doc_desc="Context manager that appends an item to a list on enter, and guarantees its removal on exit.",
        doctests=[">>> lst = [1, 2]", ">>> with ScopedListAppend(lst, 3):", "...     n = len(lst)", ">>> len(lst)", "2"],
        hint="In __enter__, lst.append(item). In __exit__, if item in lst, lst.remove(item).",
        solution="    def __init__(self, target_list: list, item: Any):\n        self.target_list = target_list\n        self.item = item\n    def __enter__(self):\n        self.target_list.append(self.item)\n        return self.target_list\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if self.item in self.target_list:\n            self.target_list.remove(self.item)\n        return False",
        test="my_list = [10, 20]\nwith ScopedListAppend(my_list, 30):\n    assert my_list == [10, 20, 30]\nassert my_list == [10, 20]\n"
    ))

    # 11. Rollback List Context Manager
    tasks.append(create_task(
        name="RollbackList",
        signature="class RollbackList:",
        doc_desc="Context manager taking a snapshot of a list. If an exception occurs, restores the list to its original state.",
        doctests=[">>> lst = [1, 2]", ">>> try:", "...     with RollbackList(lst):", "...         lst.append(3)", "...         raise RuntimeError()", ">>> lst", "[1, 2]"],
        hint="In __enter__, save self.target_list[:]. In __exit__, if exc_type is not None: clear and extend with snapshot.",
        solution="    def __init__(self, target_list: list):\n        self.target_list = target_list\n        self.snapshot = []\n    def __enter__(self):\n        self.snapshot = list(self.target_list)\n        return self.target_list\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if exc_type is not None:\n            self.target_list.clear()\n            self.target_list.extend(self.snapshot)\n        return False",
        test="nums = [1, 2, 3]\ntry:\n    with RollbackList(nums):\n        nums.append(4)\n        nums.append(5)\n        raise ValueError('abort')\nexcept ValueError:\n    pass\nassert nums == [1, 2, 3]\nwith RollbackList(nums):\n    nums.append(99)\nassert nums == [1, 2, 3, 99]\n"
    ))

    # 12. Mock Environment Context Manager
    tasks.append(create_task(
        name="MockEnvironment",
        signature="class MockEnvironment:",
        prefix_code="import os\n",
        doc_desc="Context manager temporarily overriding environment variables in os.environ, restoring previous state on exit.",
        doctests=[">>> with MockEnvironment({'TEST_KEY': 'val'}):", "...     v = os.environ.get('TEST_KEY')", ">>> 'TEST_KEY' in os.environ", "False"],
        hint="Save original values or whether keys existed. In __enter__, set new values. In __exit__, restore or del os.environ[k].",
        solution="    def __init__(self, overrides: dict[str, str]):\n        self.overrides = overrides\n        self.saved = {}\n    def __enter__(self):\n        for k, v in self.overrides.items():\n            self.saved[k] = os.environ.get(k)\n            os.environ[k] = v\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        for k, prev in self.saved.items():\n            if prev is None:\n                os.environ.pop(k, None)\n            else:\n                os.environ[k] = prev\n        return False",
        test="os.environ['EXISTING_KEY'] = 'orig'\nwith MockEnvironment({'EXISTING_KEY': 'mocked', 'NEW_KEY': 'created'}):\n    assert os.environ['EXISTING_KEY'] == 'mocked'\n    assert os.environ['NEW_KEY'] == 'created'\nassert os.environ['EXISTING_KEY'] == 'orig'\nassert 'NEW_KEY' not in os.environ\n"
    ))

    # 13. Safe ISO Date Parser
    tasks.append(create_task(
        name="parse_iso_date_safe",
        signature="def parse_iso_date_safe(date_str: str) -> tuple[int, int, int] | None:",
        doc_desc="Parses a date string 'YYYY-MM-DD' into (year, month, day). Returns None if format or values are invalid.",
        doctests=[">>> parse_iso_date_safe('2026-09-19')", "(2026, 9, 19)", ">>> parse_iso_date_safe('invalid')", "None"],
        hint="Split by '-', check len == 3, cast to int. Validate 1 <= month <= 12 and 1 <= day <= 31. Catch ValueError, TypeError.",
        solution="    try:\n        parts = date_str.split('-')\n        if len(parts) != 3:\n            return None\n        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])\n        if not (1 <= m <= 12 and 1 <= d <= 31 and y > 0):\n            return None\n        return (y, m, d)\n    except (ValueError, TypeError, AttributeError):\n        return None",
        test="assert parse_iso_date_safe('2026-09-19') == (2026, 9, 19)\nassert parse_iso_date_safe('2026-13-01') is None\nassert parse_iso_date_safe('bad') is None\nassert parse_iso_date_safe(123) is None\n"
    ))

    # 14. Nested Key Lookup with Custom KeyError
    tasks.append(create_task(
        name="fetch_nested_key",
        signature="def fetch_nested_key(data: dict, path: list[str]) -> Any:",
        prefix_code="class NestedKeyNotFoundError(KeyError):\n    def __init__(self, key: str, path: list[str]):\n        super().__init__(f'Key {key!r} not found along path {path}')\n        self.key = key\n        self.path = path\n",
        doc_desc="Traverses nested dictionaries along path keys. Raises NestedKeyNotFoundError if any key is missing.",
        doctests=[">>> fetch_nested_key({'a': {'b': 42}}, ['a', 'b'])", "42"],
        hint="Iterate over path. In each step, if key not in current or not isinstance(current, dict), raise NestedKeyNotFoundError(key, path).",
        solution="    curr = data\n    for k in path:\n        if not isinstance(curr, dict) or k not in curr:\n            raise NestedKeyNotFoundError(k, path)\n        curr = curr[k]\n    return curr",
        test="data = {'server': {'database': {'host': 'localhost'}}}\nassert fetch_nested_key(data, ['server', 'database', 'host']) == 'localhost'\ntry:\n    fetch_nested_key(data, ['server', 'port'])\n    assert False\nexcept NestedKeyNotFoundError as exc:\n    assert exc.key == 'port'\n"
    ))

    # 15. Validation Error Accumulator
    tasks.append(create_task(
        name="ValidationAccumulator",
        signature="class ValidationAccumulator:",
        prefix_code="class AggregateValidationError(Exception):\n    def __init__(self, errors: list[str]):\n        super().__init__(f'{len(errors)} validation errors occurred: ' + '; '.join(errors))\n        self.errors = errors\n",
        doc_desc="Collects field validation errors and raises AggregateValidationError with all collected errors when check() is called.",
        doctests=[">>> v = ValidationAccumulator()", ">>> v.add_error('Name required')", ">>> v.has_errors()", "True"],
        hint="Store self.errors = []. In check(), if self.errors is non-empty, raise AggregateValidationError(list(self.errors)).",
        solution="    def __init__(self):\n        self.errors: list[str] = []\n    def add_error(self, message: str) -> None:\n        self.errors.append(message)\n    def has_errors(self) -> bool:\n        return len(self.errors) > 0\n    def check(self) -> None:\n        if self.errors:\n            raise AggregateValidationError(list(self.errors))",
        test="v = ValidationAccumulator()\nassert not v.has_errors()\nv.check()\nv.add_error('Invalid email')\nv.add_error('Password too short')\nassert v.has_errors()\ntry:\n    v.check()\n    assert False\nexcept AggregateValidationError as err:\n    assert len(err.errors) == 2\n"
    ))

    # 16. Safe Index Access
    tasks.append(create_task(
        name="safe_index_access",
        signature="def safe_index_access(lst: list, idx: int, default: Any = None) -> Any:",
        doc_desc="Returns lst[idx] safely. If IndexError or TypeError occurs, returns default.",
        doctests=[">>> safe_index_access([10, 20], 1)", "20", ">>> safe_index_access([10, 20], 5, -1)", "-1"],
        hint="try/except (IndexError, TypeError) returning lst[idx] or default.",
        solution="    try:\n        return lst[idx]\n    except (IndexError, TypeError):\n        return default",
        test="assert safe_index_access([10, 20], 1) == 20\nassert safe_index_access([10, 20], 5, -1) == -1\nassert safe_index_access([10, 20], 'bad', 0) == 0\nassert safe_index_access(None, 0, 99) == 99\n"
    ))

    # 17. Execution Lifecycle Logger
    tasks.append(create_task(
        name="ExecutionLogger",
        signature="class ExecutionLogger:",
        doc_desc="Context manager that appends lifecycle events ('ENTER', 'SUCCESS', 'ERROR: <exc_type>') to a provided log list.",
        doctests=[">>> log = []", ">>> with ExecutionLogger(log):", "...     pass", ">>> log", "['ENTER', 'SUCCESS']"],
        hint="In __enter__, log.append('ENTER'). In __exit__, if exc_type is None append 'SUCCESS', else append f'ERROR: {exc_type.__name__}'.",
        solution="    def __init__(self, log_list: list[str]):\n        self.log_list = log_list\n    def __enter__(self):\n        self.log_list.append('ENTER')\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if exc_type is None:\n            self.log_list.append('SUCCESS')\n        else:\n            self.log_list.append(f'ERROR: {exc_type.__name__}')\n        return False",
        test="log = []\nwith ExecutionLogger(log):\n    pass\nassert log == ['ENTER', 'SUCCESS']\ntry:\n    with ExecutionLogger(log):\n        raise KeyError('oops')\nexcept KeyError:\n    pass\nassert log == ['ENTER', 'SUCCESS', 'ENTER', 'ERROR: KeyError']\n"
    ))

    # 18. Safe JSON Decode
    tasks.append(create_task(
        name="safe_json_decode",
        signature="def safe_json_decode(s: str, default: Any = None) -> Any:",
        prefix_code="import json\n",
        doc_desc="Decodes JSON string using json.loads. Returns default if JSONDecodeError or TypeError is raised.",
        doctests=[">>> safe_json_decode('{\"a\": 1}')", "{'a': 1}", ">>> safe_json_decode('invalid', {})", "{}"],
        hint="Wrap json.loads(s) in try-except catching (json.JSONDecodeError, TypeError).",
        solution="    try:\n        return json.loads(s)\n    except (json.JSONDecodeError, TypeError):\n        return default",
        test="assert safe_json_decode('{\"a\": 1}') == {'a': 1}\nassert safe_json_decode('invalid', {}) == {}\nassert safe_json_decode(None, 42) == 42\n"
    ))

    # 19. Nested Parentheses Depth Validator
    tasks.append(create_task(
        name="check_parentheses_depth",
        signature="def check_parentheses_depth(s: str, max_allowed: int) -> int:",
        prefix_code="class DepthLimitExceededError(Exception):\n    pass\n",
        doc_desc="Calculates max nesting depth of '()'. Raises DepthLimitExceededError if depth > max_allowed, ValueError if brackets are unbalanced.",
        doctests=[">>> check_parentheses_depth('((()))', 5)", "3"],
        hint="Track current_depth and max_depth. If current_depth > max_allowed, raise DepthLimitExceededError. If current_depth < 0 or final != 0, raise ValueError.",
        solution="    current_depth = 0\n    max_seen = 0\n    for char in s:\n        if char == '(':\n            current_depth += 1\n            if current_depth > max_allowed:\n                raise DepthLimitExceededError(f'Max depth {max_allowed} exceeded')\n            max_seen = max(max_seen, current_depth)\n        elif char == ')':\n            current_depth -= 1\n            if current_depth < 0:\n                raise ValueError('Unbalanced closing parenthesis')\n    if current_depth != 0:\n        raise ValueError('Unbalanced open parenthesis')\n    return max_seen",
        test="assert check_parentheses_depth('((()))', 5) == 3\ntry:\n    check_parentheses_depth('((()))', 2)\n    assert False\nexcept DepthLimitExceededError:\n    pass\ntry:\n    check_parentheses_depth('())', 5)\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 20. Re-raise As Custom Exception
    tasks.append(create_task(
        name="re_raise_as_custom",
        signature="def re_raise_as_custom(func: Any, custom_exc_cls: type, *args, **kwargs) -> Any:",
        doc_desc="Calls func(*args, **kwargs). If any Exception occurs, catches it and raises custom_exc_cls(str(exc)) from exc.",
        doctests=[">>> re_raise_as_custom(lambda: 42, RuntimeError)", "42"],
        hint="Inside try/except Exception as exc: raise custom_exc_cls(str(exc)) from exc.",
        solution="    try:\n        return func(*args, **kwargs)\n    except Exception as exc:\n        raise custom_exc_cls(str(exc)) from exc",
        test="class CustomAppError(Exception):\n    pass\ndef failing():\n    raise ValueError('Original issue')\ntry:\n    re_raise_as_custom(failing, CustomAppError)\n    assert False\nexcept CustomAppError as err:\n    assert 'Original issue' in str(err)\n    assert isinstance(err.__cause__, ValueError)\n"
    ))

    # 21. Managed Counter Context
    tasks.append(create_task(
        name="ManagedCounter",
        signature="class ManagedCounter:",
        doc_desc="Context manager that increments an internal count property upon entering, and decrements upon exiting.",
        doctests=[">>> c = ManagedCounter()", ">>> with c:", "...     v = c.count", ">>> c.count", "0"],
        hint="Initialize self.count = 0. In __enter__, self.count += 1; return self. In __exit__, self.count -= 1.",
        solution="    def __init__(self):\n        self.count = 0\n    def __enter__(self):\n        self.count += 1\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        self.count -= 1\n        return False",
        test="c = ManagedCounter()\nassert c.count == 0\nwith c:\n    assert c.count == 1\n    with c:\n        assert c.count == 2\n    assert c.count == 1\nassert c.count == 0\n"
    ))

    # 22. Capture Stdout Context Manager
    tasks.append(create_task(
        name="CaptureStdout",
        signature="class CaptureStdout:",
        prefix_code="import sys\nimport io\n",
        doc_desc="Context manager temporarily redirecting sys.stdout to an internal StringIO, exposing captured output as string via get_output().",
        doctests=[">>> with CaptureStdout() as cap:", "...     print('hello')", ">>> cap.get_output().strip()", "'hello'"],
        hint="Save sys.stdout, replace with io.StringIO(). In __exit__, restore sys.stdout.",
        solution="    def __init__(self):\n        self.buffer = io.StringIO()\n        self.old_stdout = None\n    def __enter__(self):\n        self.old_stdout = sys.stdout\n        sys.stdout = self.buffer\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        sys.stdout = self.old_stdout\n        return False\n    def get_output(self) -> str:\n        return self.buffer.getvalue()",
        test="with CaptureStdout() as cap:\n    print('test line 1')\n    print('test line 2')\nassert cap.get_output() == 'test line 1\\ntest line 2\\n'\n"
    ))

    # 23. Positive Integer List Validator
    tasks.append(create_task(
        name="validate_positive_integers",
        signature="def validate_positive_integers(items: list[Any]) -> list[int]:",
        doc_desc="Validates that all elements are positive ints (> 0, not bool). Raises ValueError with index and item on failure.",
        doctests=[">>> validate_positive_integers([1, 2, 3])", "[1, 2, 3]"],
        hint="Loop with enumerate. If not isinstance(x, int) or isinstance(x, bool) or x <= 0, raise ValueError(f'Invalid item at index {i}: {x}').",
        solution="    result = []\n    for i, x in enumerate(items):\n        if not isinstance(x, int) or isinstance(x, bool) or x <= 0:\n            raise ValueError(f'Invalid item at index {i}: {x}')\n        result.append(x)\n    return result",
        test="assert validate_positive_integers([1, 5, 10]) == [1, 5, 10]\ntry:\n    validate_positive_integers([1, 0, 3])\n    assert False\nexcept ValueError as err:\n    assert 'index 1' in str(err)\ntry:\n    validate_positive_integers([1, True, 3])\n    assert False\nexcept ValueError as err:\n    assert 'index 1' in str(err)\n"
    ))

    # 24. Resource Pool Context Manager
    tasks.append(create_task(
        name="ResourcePool",
        signature="class ResourcePool:",
        doc_desc="Manages a pool of string resource IDs. acquire() checks out an ID, returning a context manager that releases it back on exit.",
        doctests=[">>> pool = ResourcePool(['R1', 'R2'])", ">>> with pool.acquire() as res:", "...     res == 'R1'", "True"],
        hint="Store available set. acquire() pops one, yields in a context manager, adds back in finally.",
        solution="    class _Lease:\n        def __init__(self, pool, resource):\n            self.pool = pool\n            self.resource = resource\n        def __enter__(self):\n            return self.resource\n        def __exit__(self, exc_type, exc_val, exc_tb):\n            self.pool.available.append(self.resource)\n            return False\n    def __init__(self, resources: list[str]):\n        self.available = list(resources)\n    def acquire(self):\n        if not self.available:\n            raise RuntimeError('No resources available')\n        res = self.available.pop(0)\n        return ResourcePool._Lease(self, res)",
        test="pool = ResourcePool(['A', 'B'])\nwith pool.acquire() as r1:\n    assert r1 == 'A'\n    with pool.acquire() as r2:\n        assert r2 == 'B'\n        try:\n            with pool.acquire():\n                pass\n            assert False\n        except RuntimeError:\n            pass\nassert 'A' in pool.available and 'B' in pool.available\n"
    ))

    # 25. Transaction Context
    tasks.append(create_task(
        name="TransactionContext",
        signature="class TransactionContext:",
        doc_desc="Context manager simulating a transaction with commit/rollback callbacks. Calls commit() if clean exit, rollback() if exception occurs.",
        doctests=[">>> tx = TransactionContext()", ">>> with tx: pass", ">>> tx.state", "'committed'"],
        hint="In __exit__, if exc_type is None set self.state = 'committed', else self.state = 'rolled_back' and return False.",
        solution="    def __init__(self):\n        self.state = 'idle'\n    def __enter__(self):\n        self.state = 'active'\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if exc_type is None:\n            self.state = 'committed'\n        else:\n            self.state = 'rolled_back'\n        return False",
        test="tx = TransactionContext()\nwith tx:\n    assert tx.state == 'active'\nassert tx.state == 'committed'\ntry:\n    with tx:\n        raise RuntimeError('fail')\nexcept RuntimeError:\n    pass\nassert tx.state == 'rolled_back'\n"
    ))

    # 26. Value In Range Assertion
    tasks.append(create_task(
        name="assert_in_range",
        signature="def assert_in_range(val: float, low: float, high: float, param_name: str) -> None:",
        doc_desc="Checks low <= val <= high. Raises ValueError with descriptive message f'{param_name} ({val}) out of bounds [{low}, {high}]' if outside.",
        doctests=[">>> assert_in_range(5.0, 0.0, 10.0, 'alpha')", "None"],
        hint="Check val < low or val > high, and raise ValueError(f'{param_name} ({val}) out of bounds [{low}, {high}]').",
        solution="    if val < low or val > high:\n        raise ValueError(f'{param_name} ({val}) out of bounds [{low}, {high}]')",
        test="assert_in_range(5.0, 0.0, 10.0, 'alpha')\ntry:\n    assert_in_range(15.0, 0.0, 10.0, 'alpha')\n    assert False\nexcept ValueError as e:\n    assert 'alpha (15.0) out of bounds [0.0, 10.0]' in str(e)\n"
    ))

    # 27. Strict Dictionary (Immutable Keys)
    tasks.append(create_task(
        name="StrictDict",
        signature="class StrictDict(dict):",
        prefix_code="class KeyOverwriteError(KeyError):\n    pass\n",
        doc_desc="Dict subclass that raises KeyOverwriteError if attempting to assign a key that already exists.",
        doctests=[">>> sd = StrictDict({'a': 1})", ">>> sd['b'] = 2", ">>> sd['b']", "2"],
        hint="Override __setitem__(self, key, value). If key in self, raise KeyOverwriteError. Otherwise super().__setitem__(key, value).",
        solution="    def __setitem__(self, key, value):\n        if key in self:\n            raise KeyOverwriteError(f'Key {key!r} already exists')\n        super().__setitem__(key, value)",
        test="sd = StrictDict({'a': 1})\nsd['b'] = 2\nassert sd['b'] == 2\ntry:\n    sd['a'] = 99\n    assert False\nexcept KeyOverwriteError:\n    pass\n"
    ))

    # 28. Partition Integers and Failures
    tasks.append(create_task(
        name="safe_int_partition",
        signature="def safe_int_partition(items: list[str]) -> tuple[list[int], list[str]]:",
        doc_desc="Attempts to convert each string to an int. Returns a tuple (success_ints, failed_strings).",
        doctests=[">>> safe_int_partition(['10', 'abc', '25', 'xyz'])", "([10, 25], ['abc', 'xyz'])"],
        hint="Loop through items, try int(x), catch (ValueError, TypeError) to sort into two lists.",
        solution="    successes = []\n    failures = []\n    for item in items:\n        try:\n            successes.append(int(item))\n        except (ValueError, TypeError):\n            failures.append(item)\n    return (successes, failures)",
        test="assert safe_int_partition(['10', 'abc', '25', 'xyz']) == ([10, 25], ['abc', 'xyz'])\nassert safe_int_partition([]) == ([], [])\n"
    ))

    # 29. Custom Error with Numeric Code
    tasks.append(create_task(
        name="CustomErrorCode",
        signature="class CustomErrorCode(Exception):",
        doc_desc="Custom exception storing error_code: int and message: str. __str__ formats as '[ERR-{code}] {message}'.",
        doctests=[">>> err = CustomErrorCode(404, 'Not Found')", ">>> str(err)", "'[ERR-404] Not Found'"],
        hint="In __init__, call super().__init__(f'[ERR-{code}] {message}'). Set self.code = code and self.message = message.",
        solution="    def __init__(self, code: int, message: str):\n        super().__init__(f'[ERR-{code}] {message}')\n        self.code = code\n        self.message = message",
        test="err = CustomErrorCode(404, 'Not Found')\nassert err.code == 404\nassert err.message == 'Not Found'\nassert str(err) == '[ERR-404] Not Found'\n"
    ))

    # 30. Safe Float Filter
    tasks.append(create_task(
        name="safe_float_filter",
        signature="def safe_float_filter(items: list[Any]) -> list[float]:",
        doc_desc="Converts valid elements to float, silently omitting elements that raise ValueError or TypeError.",
        doctests=[">>> safe_float_filter(['3.14', 'bad', 42, None])", "[3.14, 42.0]"],
        hint="For item in items: inside try float(item), append; catch (ValueError, TypeError). Note: bools are valid floats in python, so check not isinstance(item, bool).",
        solution="    result = []\n    for x in items:\n        if isinstance(x, bool):\n            continue\n        try:\n            result.append(float(x))\n        except (ValueError, TypeError):\n            continue\n    return result",
        test="assert safe_float_filter(['3.14', 'bad', 42, None, True]) == [3.14, 42.0]\nassert safe_float_filter([]) == []\n"
    ))

    # 31. Pipeline Runner with Error Context
    tasks.append(create_task(
        name="run_pipeline_with_context",
        signature="def run_pipeline_with_context(stages: list[Any], initial_val: Any) -> Any:",
        prefix_code="class PipelineStageError(Exception):\n    def __init__(self, stage_idx: int, cause: Exception):\n        super().__init__(f'Pipeline failed at stage {stage_idx}: {cause}')\n        self.stage_idx = stage_idx\n        self.cause = cause\n",
        doc_desc="Executes functions in stages sequentially. Wraps any exception into PipelineStageError with stage_idx and original exception.",
        doctests=[">>> run_pipeline_with_context([lambda x: x + 1, lambda x: x * 2], 5)", "12"],
        hint="Loop with enumerate(stages). Try current = stage(current); except Exception as e: raise PipelineStageError(i, e) from e.",
        solution="    curr = initial_val\n    for i, stage in enumerate(stages):\n        try:\n            curr = stage(curr)\n        except Exception as exc:\n            raise PipelineStageError(i, exc) from exc\n    return curr",
        test="def step1(x): return x + 1\ndef step2(x): raise ZeroDivisionError('boom')\ntry:\n    run_pipeline_with_context([step1, step2], 10)\n    assert False\nexcept PipelineStageError as err:\n    assert err.stage_idx == 1\n    assert isinstance(err.cause, ZeroDivisionError)\n"
    ))

    # 32. Mutual Exclusion Lock Simulator
    tasks.append(create_task(
        name="LockSimulator",
        signature="class LockSimulator:",
        doc_desc="Context manager simulating a non-reentrant lock. Raises RuntimeError if acquired while already locked.",
        doctests=[">>> lock = LockSimulator()", ">>> with lock:", "...     lock.is_locked", "True"],
        hint="Maintain self.is_locked bool. In __enter__, if self.is_locked raise RuntimeError('Lock already acquired'). Set True. In __exit__, set False.",
        solution="    def __init__(self):\n        self.is_locked = False\n    def __enter__(self):\n        if self.is_locked:\n            raise RuntimeError('Lock already acquired')\n        self.is_locked = True\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        self.is_locked = False\n        return False",
        test="l = LockSimulator()\nassert not l.is_locked\nwith l:\n    assert l.is_locked\n    try:\n        with l:\n            pass\n        assert False\n    except RuntimeError:\n        pass\nassert not l.is_locked\n"
    ))

    # 33. Unique Key Extractor with Duplication Error
    tasks.append(create_task(
        name="extract_unique_keys",
        signature="def extract_unique_keys(items: list[dict], key: str) -> set[Any]:",
        prefix_code="class DuplicateKeyError(ValueError):\n    pass\n",
        doc_desc="Extracts values of key from items dicts. Raises KeyError if missing, DuplicateKeyError if value was already seen.",
        doctests=[">>> extract_unique_keys([{'id': 1}, {'id': 2}], 'id')", "{1, 2}"],
        hint="Check d in items: if key not in d, raise KeyError(key). If val in seen, raise DuplicateKeyError(f'Duplicate: {val}').",
        solution="    seen = set()\n    for d in items:\n        if key not in d:\n            raise KeyError(f'Missing key: {key}')\n        val = d[key]\n        if val in seen:\n            raise DuplicateKeyError(f'Duplicate value: {val}')\n        seen.add(val)\n    return seen",
        test="assert extract_unique_keys([{'id': 1}, {'id': 2}], 'id') == {1, 2}\ntry:\n    extract_unique_keys([{'id': 1}, {'id': 1}], 'id')\n    assert False\nexcept DuplicateKeyError:\n    pass\ntry:\n    extract_unique_keys([{'other': 1}], 'id')\n    assert False\nexcept KeyError:\n    pass\n"
    ))

    # 34. Assert Raises Context Manager
    tasks.append(create_task(
        name="AssertRaisesContext",
        signature="class AssertRaisesContext:",
        doc_desc="Testing context manager verifying that an expected exception type is raised inside the block. Raises AssertionError if not raised.",
        doctests=[">>> with AssertRaisesContext(ZeroDivisionError):", "...     _ = 1 / 0"],
        hint="In __exit__, if exc_type is None, raise AssertionError(f'{self.expected.__name__} not raised'). If issubclass(exc_type, self.expected), return True. Else return False.",
        solution="    def __init__(self, expected_exc: type):\n        self.expected_exc = expected_exc\n    def __enter__(self):\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if exc_type is None:\n            raise AssertionError(f'Expected {self.expected_exc.__name__} was not raised')\n        if issubclass(exc_type, self.expected_exc):\n            return True\n        return False",
        test="with AssertRaisesContext(KeyError):\n    _ = {}['missing']\ntry:\n    with AssertRaisesContext(ValueError):\n        pass\n    assert False\nexcept AssertionError:\n    pass\n"
    ))

    # 35. Circuit Breaker Simulator
    tasks.append(create_task(
        name="CircuitBreaker",
        signature="class CircuitBreaker:",
        prefix_code="class CircuitOpenError(Exception):\n    pass\n",
        doc_desc="Tracks consecutive failures. If failure_count >= threshold, call(func) raises CircuitOpenError without calling func. Resets on success.",
        doctests=[">>> cb = CircuitBreaker(threshold=2)"],
        hint="If self.failure_count >= threshold, raise CircuitOpenError. Try func(), reset count to 0, return result. Except: count += 1, re-raise.",
        solution="    def __init__(self, threshold: int = 3):\n        self.threshold = threshold\n        self.failures = 0\n    def call(self, func: Any, *args, **kwargs) -> Any:\n        if self.failures >= self.threshold:\n            raise CircuitOpenError('Circuit breaker is open')\n        try:\n            res = func(*args, **kwargs)\n            self.failures = 0\n            return res\n        except Exception as exc:\n            self.failures += 1\n            raise exc",
        test="cb = CircuitBreaker(threshold=2)\ndef fail(): raise ValueError('err')\ndef ok(): return 'ok'\ntry: cb.call(fail)\nexcept ValueError: pass\ntry: cb.call(fail)\nexcept ValueError: pass\nassert cb.failures == 2\ntry:\n    cb.call(ok)\n    assert False\nexcept CircuitOpenError:\n    pass\n"
    ))

    # 36. Parse Strict JSON Object
    tasks.append(create_task(
        name="parse_strict_json_object",
        signature="def parse_strict_json_object(s: str) -> dict[str, Any]:",
        prefix_code="import json\n",
        doc_desc="Parses JSON string. Raises ValueError if malformed JSON, TypeError if root structure is not a dict.",
        doctests=[">>> parse_strict_json_object('{\"a\": 1}')", "{'a': 1}"],
        hint="try json.loads(s). Catch json.JSONDecodeError -> raise ValueError from exc. If not isinstance(res, dict) -> raise TypeError.",
        solution="    try:\n        parsed = json.loads(s)\n    except (json.JSONDecodeError, TypeError) as exc:\n        raise ValueError(f'Invalid JSON string: {exc}') from exc\n    if not isinstance(parsed, dict):\n        raise TypeError(f'Expected dict, got {type(parsed).__name__}')\n    return parsed",
        test="assert parse_strict_json_object('{\"k\": \"v\"}') == {'k': 'v'}\ntry:\n    parse_strict_json_object('[1, 2, 3]')\n    assert False\nexcept TypeError:\n    pass\ntry:\n    parse_strict_json_object('bad')\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 37. Scoped Configuration Override
    tasks.append(create_task(
        name="ScopedConfig",
        signature="class ScopedConfig:",
        doc_desc="Context manager temporarily overriding key-value pairs in a config dict, restoring original values on exit.",
        doctests=[">>> cfg = {'debug': False}", ">>> with ScopedConfig(cfg, {'debug': True}):", "...     cfg['debug']", "True"],
        hint="In __enter__, save original values of keys in overrides. In __exit__, restore or remove newly created keys.",
        solution="    def __init__(self, config: dict, overrides: dict):\n        self.config = config\n        self.overrides = overrides\n        self.saved = {}\n    def __enter__(self):\n        for k, v in self.overrides.items():\n            self.saved[k] = (k in self.config, self.config.get(k))\n            self.config[k] = v\n        return self.config\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        for k, (existed, val) in self.saved.items():\n            if existed:\n                self.config[k] = val\n            else:\n                self.config.pop(k, None)\n        return False",
        test="cfg = {'host': 'localhost', 'port': 80}\nwith ScopedConfig(cfg, {'port': 8080, 'env': 'test'}):\n    assert cfg['port'] == 8080\n    assert cfg['env'] == 'test'\nassert cfg['port'] == 80\nassert 'env' not in cfg\n"
    ))

    # 38. Safe Schema Type Validator
    tasks.append(create_task(
        name="validate_dict_schema",
        signature="def validate_dict_schema(data: dict, schema: dict[str, type]) -> bool:",
        doc_desc="Validates that data contains all keys in schema with corresponding instance types. Returns True if valid, raises KeyError or TypeError.",
        doctests=[">>> validate_dict_schema({'name': 'Alice', 'age': 30}, {'name': str, 'age': int})", "True"],
        hint="For k, expected_type in schema.items(): if k not in data raise KeyError(k); if not isinstance(data[k], expected_type) raise TypeError.",
        solution="    for k, expected_type in schema.items():\n        if k not in data:\n            raise KeyError(f'Missing required field: {k}')\n        if not isinstance(data[k], expected_type):\n            raise TypeError(f'Field {k} must be of type {expected_type.__name__}, got {type(data[k]).__name__}')\n    return True",
        test="assert validate_dict_schema({'x': 1, 'y': 'str'}, {'x': int, 'y': str}) is True\ntry:\n    validate_dict_schema({'x': 1}, {'x': int, 'y': str})\n    assert False\nexcept KeyError:\n    pass\ntry:\n    validate_dict_schema({'x': 'bad', 'y': 'str'}, {'x': int, 'y': str})\n    assert False\nexcept TypeError:\n    pass\n"
    ))

    # 39. Try-Finally Lifecycle Tracer
    tasks.append(create_task(
        name="trace_try_finally",
        signature="def trace_try_finally(body_fn: Any, on_success_fn: Any, cleanup_fn: Any) -> tuple[Any, list[str]]:",
        doc_desc="Runs body_fn. Appends 'SUCCESS' if no error, appends 'CLEANUP' in finally. Returns (result, trace_log).",
        doctests=[">>> res, trace = trace_try_finally(lambda: 10, lambda: None, lambda: None)", ">>> trace", "['SUCCESS', 'CLEANUP']"],
        hint="Initialize trace = []. In try: res = body_fn(); on_success_fn(); trace.append('SUCCESS'). In finally: cleanup_fn(); trace.append('CLEANUP').",
        solution="    trace = []\n    try:\n        res = body_fn()\n        on_success_fn()\n        trace.append('SUCCESS')\n        return (res, trace)\n    finally:\n        cleanup_fn()\n        trace.append('CLEANUP')",
        test="r, tr = trace_try_finally(lambda: 42, lambda: None, lambda: None)\nassert r == 42 and tr == ['SUCCESS', 'CLEANUP']\ntr2 = []\ntry:\n    trace_try_finally(lambda: 1/0, lambda: None, lambda: None)\nexcept ZeroDivisionError:\n    pass\n"
    ))

    # 40. Safe URL Scheme Parser
    tasks.append(create_task(
        name="validate_url_scheme",
        signature="def validate_url_scheme(url: str, allowed_schemes: tuple[str, ...] = ('http', 'https')) -> str:",
        doc_desc="Validates that url starts with '<scheme>://' where scheme is in allowed_schemes. Raises ValueError if missing or invalid.",
        doctests=[">>> validate_url_scheme('https://example.com')", "'https://example.com'"],
        hint="Split by '://' on 1 occurrence. Check if scheme in allowed_schemes. If not or no '://', raise ValueError.",
        solution="    if '://' not in url:\n        raise ValueError('URL missing scheme delimiter ://')\n    scheme, rest = url.split('://', 1)\n    if scheme not in allowed_schemes or not rest:\n        raise ValueError(f'Invalid scheme {scheme!r}, allowed: {allowed_schemes}')\n    return url",
        test="assert validate_url_scheme('http://foo.bar') == 'http://foo.bar'\nassert validate_url_scheme('https://foo.bar') == 'https://foo.bar'\ntry:\n    validate_url_scheme('ftp://foo.bar')\n    assert False\nexcept ValueError:\n    pass\ntry:\n    validate_url_scheme('just_a_string')\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 41. Event Listener Context Manager
    tasks.append(create_task(
        name="EventListenerContext",
        signature="class EventListenerContext:",
        doc_desc="Context manager that registers callback with an emitter list on enter, and unregisters it on exit.",
        doctests=[">>> listeners = []", ">>> with EventListenerContext(listeners, 'fn'):", "...     len(listeners)", "1", ">>> len(listeners)", "0"],
        hint="In __enter__, listeners.append(cb). In __exit__, if cb in listeners: listeners.remove(cb).",
        solution="    def __init__(self, listeners: list[Any], callback: Any):\n        self.listeners = listeners\n        self.callback = callback\n    def __enter__(self):\n        self.listeners.append(self.callback)\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if self.callback in self.listeners:\n            self.listeners.remove(self.callback)\n        return False",
        test="cbs = []\ndef my_cb(): pass\nwith EventListenerContext(cbs, my_cb):\n    assert my_cb in cbs\nassert my_cb not in cbs\n"
    ))

    # 42. Dynamic Exception Class Factory
    tasks.append(create_task(
        name="create_custom_exception",
        signature="def create_custom_exception(name: str, base_cls: type = Exception) -> type:",
        doc_desc="Dynamically constructs a new Exception subclass with the given name and base class using type().",
        doctests=[">>> MyErr = create_custom_exception('MyErr')", ">>> issubclass(MyErr, Exception)", "True"],
        hint="Use type(name, (base_cls,), {}).",
        solution="    return type(name, (base_cls,), {})",
        test="CustomErr = create_custom_exception('CustomErr', ValueError)\nassert issubclass(CustomErr, ValueError)\nassert CustomErr.__name__ == 'CustomErr'\ntry:\n    raise CustomErr('test')\nexcept ValueError as e:\n    assert str(e) == 'test'\n"
    ))

    # 43. Safe Hex to RGB Color
    tasks.append(create_task(
        name="safe_hex_to_rgb",
        signature="def safe_hex_to_rgb(hex_str: str, default: tuple[int, int, int] = (0, 0, 0)) -> tuple[int, int, int]:",
        doc_desc="Parses '#RRGGBB' hex color code into (R, G, B) ints. Returns default on invalid format, bad hex chars, or length mismatch.",
        doctests=[">>> safe_hex_to_rgb('#FFFFFF')", "(255, 255, 255)", ">>> safe_hex_to_rgb('invalid')", "(0, 0, 0)"],
        hint="Check hex_str starts with '#', len == 7. Parse slices [1:3], [3:5], [5:7] with int(x, 16). Catch ValueError, TypeError, AttributeError.",
        solution="    try:\n        if not isinstance(hex_str, str) or not hex_str.startswith('#') or len(hex_str) != 7:\n            return default\n        r = int(hex_str[1:3], 16)\n        g = int(hex_str[3:5], 16)\n        b = int(hex_str[5:7], 16)\n        return (r, g, b)\n    except (ValueError, TypeError):\n        return default",
        test="assert safe_hex_to_rgb('#FF0000') == (255, 0, 0)\nassert safe_hex_to_rgb('#00FF00') == (0, 255, 0)\nassert safe_hex_to_rgb('not-a-color') == (0, 0, 0)\nassert safe_hex_to_rgb('#ZZZZZZ', (1, 1, 1)) == (1, 1, 1)\n"
    ))

    # 44. Rollback Counter Context Manager
    tasks.append(create_task(
        name="AtomicCounterWithRollback",
        signature="class AtomicCounterWithRollback:",
        doc_desc="Context manager that increments counter.value by delta on enter, but restores previous value if exception occurs.",
        doctests=[">>> c = AtomicCounterWithRollback(10)", ">>> with c.increment(5): pass", ">>> c.value", "15"],
        hint="In lease __enter__, save old value, add delta. In __exit__, if exc_type is not None: restore old value.",
        solution="    class _Lease:\n        def __init__(self, parent, delta):\n            self.parent = parent\n            self.delta = delta\n            self.prev = 0\n        def __enter__(self):\n            self.prev = self.parent.value\n            self.parent.value += self.delta\n            return self.parent\n        def __exit__(self, exc_type, exc_val, exc_tb):\n            if exc_type is not None:\n                self.parent.value = self.prev\n            return False\n    def __init__(self, initial: int = 0):\n        self.value = initial\n    def increment(self, delta: int = 1):\n        return AtomicCounterWithRollback._Lease(self, delta)",
        test="c = AtomicCounterWithRollback(10)\nwith c.increment(5):\n    assert c.value == 15\nassert c.value == 15\ntry:\n    with c.increment(10):\n        assert c.value == 25\n        raise ValueError('error')\nexcept ValueError:\n    pass\nassert c.value == 15\n"
    ))

    # 45. Exception Swallower with Predicate
    tasks.append(create_task(
        name="ExceptionSwallowPredicate",
        signature="class ExceptionSwallowPredicate:",
        doc_desc="Context manager that swallows an exception only if predicate(exc) returns True.",
        doctests=[">>> with ExceptionSwallowPredicate(lambda e: isinstance(e, ValueError)):", "...     raise ValueError()"],
        hint="In __exit__, if exc_val is not None and self.predicate(exc_val) is True, return True. Else return False.",
        solution="    def __init__(self, predicate: Any):\n        self.predicate = predicate\n    def __enter__(self):\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if exc_val is not None and self.predicate(exc_val):\n            return True\n        return False",
        test="with ExceptionSwallowPredicate(lambda e: 'ignore' in str(e)):\n    raise ValueError('please ignore me')\ntry:\n    with ExceptionSwallowPredicate(lambda e: 'ignore' in str(e)):\n        raise ValueError('do not catch')\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 46. Timeout Operation Step Simulator
    tasks.append(create_task(
        name="StepLimitSimulator",
        signature="class StepLimitSimulator:",
        prefix_code="class StepLimitExceededError(Exception):\n    pass\n",
        doc_desc="Context manager that counts operation steps via step(). Raises StepLimitExceededError if step() is called more than max_steps.",
        doctests=[">>> with StepLimitSimulator(2) as sim:", "...     sim.step()", "...     sim.step()"],
        hint="Store steps_taken = 0. In step(), self.steps_taken += 1; if self.steps_taken > self.max_steps raise StepLimitExceededError.",
        solution="    def __init__(self, max_steps: int):\n        self.max_steps = max_steps\n        self.steps_taken = 0\n    def __enter__(self):\n        return self\n    def step(self) -> None:\n        self.steps_taken += 1\n        if self.steps_taken > self.max_steps:\n            raise StepLimitExceededError(f'Exceeded limit of {self.max_steps} steps')\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        return False",
        test="with StepLimitSimulator(3) as sim:\n    sim.step()\n    sim.step()\n    sim.step()\ntry:\n    with StepLimitSimulator(1) as sim:\n        sim.step()\n        sim.step()\n    assert False\nexcept StepLimitExceededError:\n    pass\n"
    ))

    # 47. Safe Port Range Parser
    tasks.append(create_task(
        name="parse_port_number",
        signature="def parse_port_number(val: Any) -> int:",
        doc_desc="Converts val to int port (1-65535). Raises TypeError if bool/invalid type, ValueError if not in 1..65535.",
        doctests=[">>> parse_port_number('8080')", "8080"],
        hint="Check not isinstance(val, bool). Try int(val). If port < 1 or port > 65535, raise ValueError.",
        solution="    if isinstance(val, bool):\n        raise TypeError('Port cannot be boolean')\n    try:\n        port = int(val)\n    except (ValueError, TypeError) as exc:\n        raise TypeError(f'Invalid port value: {val}') from exc\n    if not (1 <= port <= 65535):\n        raise ValueError(f'Port {port} out of range (1-65535)')\n    return port",
        test="assert parse_port_number('80') == 80\nassert parse_port_number(443) == 443\ntry:\n    parse_port_number(70000)\n    assert False\nexcept ValueError:\n    pass\ntry:\n    parse_port_number(True)\n    assert False\nexcept TypeError:\n    pass\n"
    ))

    # 48. Multiple Error Accumulator Context
    tasks.append(create_task(
        name="ErrorCollectorContext",
        signature="class ErrorCollectorContext:",
        doc_desc="Context manager providing collect(func) that executes func, capturing any raised exception without aborting.",
        doctests=[">>> with ErrorCollectorContext() as ec:", "...     ec.run_safe(lambda: 1/0)", ">>> len(ec.errors)", "1"],
        hint="Store self.errors = []. In run_safe(fn), try: fn(); except Exception as e: self.errors.append(e).",
        solution="    def __init__(self):\n        self.errors: list[Exception] = []\n    def __enter__(self):\n        return self\n    def run_safe(self, func: Any, *args, **kwargs) -> Any:\n        try:\n            return func(*args, **kwargs)\n        except Exception as exc:\n            self.errors.append(exc)\n            return None\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        return False",
        test="with ErrorCollectorContext() as ec:\n    ec.run_safe(lambda: int('abc'))\n    ec.run_safe(lambda: 10 + 20)\n    ec.run_safe(lambda: [][0])\nassert len(ec.errors) == 2\nassert isinstance(ec.errors[0], ValueError)\nassert isinstance(ec.errors[1], IndexError)\n"
    ))

    # 49. Call with Fallback Function
    tasks.append(create_task(
        name="call_with_fallback",
        signature="def call_with_fallback(primary_fn: Any, fallback_fn: Any, *args, **kwargs) -> Any:",
        doc_desc="Executes primary_fn(*args, **kwargs). If an Exception is raised, executes fallback_fn(*args, **kwargs) instead.",
        doctests=[">>> call_with_fallback(lambda: 1/0, lambda: 42)", "42"],
        hint="try primary_fn; except Exception: return fallback_fn(*args, **kwargs).",
        solution="    try:\n        return primary_fn(*args, **kwargs)\n    except Exception:\n        return fallback_fn(*args, **kwargs)",
        test="assert call_with_fallback(lambda x: x * 2, lambda x: 0, 5) == 10\nassert call_with_fallback(lambda x: 1/x, lambda x: -1, 0) == -1\n"
    ))

    # 50. State Guard Context Manager
    tasks.append(create_task(
        name="StateGuard",
        signature="class StateGuard:",
        doc_desc="Context manager asserting obj.state == required_state on enter, transitioning to running_state, and setting finished_state on exit.",
        doctests=[">>> class Machine: state = 'idle'", ">>> m = Machine()", ">>> with StateGuard(m, 'idle', 'running', 'done'):", "...     m.state", "'running'", ">>> m.state", "'done'"],
        hint="In __enter__, check getattr(obj, 'state') == required. If not, raise RuntimeError. Set running. In __exit__, set finished.",
        solution="    def __init__(self, obj: Any, required_state: str, running_state: str, finished_state: str):\n        self.obj = obj\n        self.required_state = required_state\n        self.running_state = running_state\n        self.finished_state = finished_state\n    def __enter__(self):\n        if getattr(self.obj, 'state', None) != self.required_state:\n            raise RuntimeError(f'Expected state {self.required_state}, got {getattr(self.obj, \"state\", None)}')\n        self.obj.state = self.running_state\n        return self.obj\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        self.obj.state = self.finished_state\n        return False",
        test="class Worker:\n    state = 'ready'\nw = Worker()\nwith StateGuard(w, 'ready', 'working', 'idle'):\n    assert w.state == 'working'\nassert w.state == 'idle'\ntry:\n    with StateGuard(w, 'ready', 'working', 'idle'):\n        pass\n    assert False\nexcept RuntimeError:\n    pass\n"
    ))

    # 51. Safe Dictionary Lookup
    tasks.append(create_task(
        name="safe_dict_lookup",
        signature="def safe_dict_lookup(data: dict, *keys: str, default: Any = None) -> Any:",
        doc_desc="Navigates nested dictionaries along keys. Returns default if KeyError or TypeError is encountered at any stage.",
        doctests=[">>> safe_dict_lookup({'a': {'b': 42}}, 'a', 'b')", "42", ">>> safe_dict_lookup({'a': 10}, 'a', 'b', default=0)", "0"],
        hint="Iterate over keys in a try/except (KeyError, TypeError) block.",
        solution="    curr = data\n    try:\n        for k in keys:\n            curr = curr[k]\n        return curr\n    except (KeyError, TypeError):\n        return default",
        test="assert safe_dict_lookup({'a': {'b': 42}}, 'a', 'b') == 42\nassert safe_dict_lookup({'a': {'b': 42}}, 'a', 'c', default=-1) == -1\nassert safe_dict_lookup({}, 'x', default='none') == 'none'\n"
    ))

    # 52. Multi-Context Stack Simulator
    tasks.append(create_task(
        name="MultiContextStack",
        signature="class MultiContextStack:",
        doc_desc="Context manager entering multiple sub-contexts in order and guaranteeing exit in reverse order.",
        doctests=[">>> class Ctx: pass"],
        hint="Store contexts list. In __enter__, enter each and store in entered stack. In __exit__, pop and call __exit__ on each in reverse.",
        solution="    def __init__(self, *contexts):\n        self.contexts = contexts\n        self.entered = []\n    def __enter__(self):\n        for ctx in self.contexts:\n            ctx.__enter__()\n            self.entered.append(ctx)\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        suppress = False\n        while self.entered:\n            ctx = self.entered.pop()\n            try:\n                if ctx.__exit__(exc_type, exc_val, exc_tb):\n                    suppress = True\n                    exc_type, exc_val, exc_tb = None, None, None\n            except Exception as e:\n                exc_type, exc_val, exc_tb = type(e), e, e.__traceback__\n        return suppress",
        test="events = []\nclass Probe:\n    def __init__(self, name):\n        self.name = name\n    def __enter__(self):\n        events.append(f'ENTER {self.name}')\n    def __exit__(self, *args):\n        events.append(f'EXIT {self.name}')\nwith MultiContextStack(Probe('A'), Probe('B')):\n    events.append('INSIDE')\nassert events == ['ENTER A', 'ENTER B', 'INSIDE', 'EXIT B', 'EXIT A']\n"
    ))

    # 53. Safe Hex String Decoder
    tasks.append(create_task(
        name="safe_hex_to_bytes",
        signature="def safe_hex_to_bytes(hex_str: str, default: bytes = b'') -> bytes:",
        doc_desc="Converts hex string to bytes using bytes.fromhex. Returns default if ValueError or TypeError occurs.",
        doctests=[">>> safe_hex_to_bytes('deadbeef')", "b'\\xde\\xad\\xbe\\xef'", ">>> safe_hex_to_bytes('xyz', b'err')", "b'err'"],
        hint="Wrap bytes.fromhex(hex_str) in try-except catching (ValueError, TypeError, AttributeError).",
        solution="    try:\n        return bytes.fromhex(hex_str)\n    except (ValueError, TypeError, AttributeError):\n        return default",
        test="assert safe_hex_to_bytes('48656c6c6f') == b'Hello'\nassert safe_hex_to_bytes('invalid', b'') == b''\nassert safe_hex_to_bytes(None, b'none') == b'none'\n"
    ))

    # 54. Atomic File Write Simulator
    tasks.append(create_task(
        name="AtomicStorageSimulator",
        signature="class AtomicStorageSimulator:",
        doc_desc="Simulates atomic store. During block, changes go to staging. On clean exit, committed to live. On error, staging discarded.",
        doctests=[">>> store = AtomicStorageSimulator({'a': 1})", ">>> with store as staging:", "...     staging['a'] = 2", ">>> store.live['a']", "2"],
        hint="In __enter__, create self.staging = dict(self.live); return self.staging. In __exit__, if exc_type is None: self.live = dict(self.staging).",
        solution="    def __init__(self, initial: dict | None = None):\n        self.live = dict(initial or {})\n        self.staging = {}\n    def __enter__(self):\n        self.staging = dict(self.live)\n        return self.staging\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if exc_type is None:\n            self.live = dict(self.staging)\n        self.staging = {}\n        return False",
        test="store = AtomicStorageSimulator({'x': 10})\nwith store as staging:\n    staging['x'] = 20\n    staging['y'] = 30\nassert store.live == {'x': 20, 'y': 30}\ntry:\n    with store as staging:\n        staging['x'] = 999\n        raise RuntimeError('failed')\nexcept RuntimeError:\n    pass\nassert store.live == {'x': 20, 'y': 30}\n"
    ))

    # 55. Safe Matrix Value Access
    tasks.append(create_task(
        name="safe_matrix_get",
        signature="def safe_matrix_get(matrix: list[list[Any]], row: int, col: int, default: Any = None) -> Any:",
        doc_desc="Accesses matrix[row][col] safely. Returns default on IndexError or TypeError.",
        doctests=[">>> safe_matrix_get([[1, 2], [3, 4]], 0, 1)", "2", ">>> safe_matrix_get([[1, 2]], 2, 0, -1)", "-1"],
        hint="try matrix[row][col] except (IndexError, TypeError): return default.",
        solution="    try:\n        if row < 0 or col < 0:\n            return default\n        return matrix[row][col]\n    except (IndexError, TypeError):\n        return default",
        test="m = [[1, 2], [3, 4]]\nassert safe_matrix_get(m, 0, 0) == 1\nassert safe_matrix_get(m, 1, 1) == 4\nassert safe_matrix_get(m, 2, 0, 0) == 0\nassert safe_matrix_get(m, -1, 0, 0) == 0\n"
    ))

    # 56. Exception Cause Formatter
    tasks.append(create_task(
        name="format_exception_chain",
        signature="def format_exception_chain(exc: Exception) -> list[str]:",
        doc_desc="Returns a list of exception class names in an exception cause/context chain starting from exc.",
        doctests=[">>> format_exception_chain(ValueError('base'))", "['ValueError']"],
        hint="Loop while exc is not None. Append type(exc).__name__. Next exc is exc.__cause__ or exc.__context__.",
        solution="    chain = []\n    curr = exc\n    while curr is not None:\n        chain.append(type(curr).__name__)\n        curr = curr.__cause__ or curr.__context__\n    return chain",
        test="e1 = ValueError('bottom')\ne2 = RuntimeError('top')\ne2.__cause__ = e1\nassert format_exception_chain(e2) == ['RuntimeError', 'ValueError']\n"
    ))

    # 57. Temporary Set Item Context Manager
    tasks.append(create_task(
        name="TemporarySetItem",
        signature="class TemporarySetItem:",
        doc_desc="Context manager adding an element to a set on enter, and removing it on exit if it wasn't originally present.",
        doctests=[">>> s = {1, 2}", ">>> with TemporarySetItem(s, 3):", "...     3 in s", "True", ">>> 3 in s", "False"],
        hint="In __enter__, save was_present = item in target_set; target_set.add(item). In __exit__, if not was_present: target_set.discard(item).",
        solution="    def __init__(self, target_set: set, item: Any):\n        self.target_set = target_set\n        self.item = item\n        self.was_present = False\n    def __enter__(self):\n        self.was_present = self.item in self.target_set\n        self.target_set.add(self.item)\n        return self.target_set\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if not self.was_present:\n            self.target_set.discard(self.item)\n        return False",
        test="s = {1, 2}\nwith TemporarySetItem(s, 3):\n    assert s == {1, 2, 3}\nassert s == {1, 2}\nwith TemporarySetItem(s, 2):\n    assert s == {1, 2}\nassert s == {1, 2}\n"
    ))

    # 58. Parse Integer with Custom Bounds
    tasks.append(create_task(
        name="parse_bounded_int",
        signature="def parse_bounded_int(val: str, min_val: int, max_val: int) -> int:",
        doc_desc="Parses integer string and checks min_val <= x <= max_val. Raises ValueError on non-digit or out of bounds.",
        doctests=[">>> parse_bounded_int('50', 0, 100)", "50"],
        hint="Try int(val). If outside [min_val, max_val], raise ValueError(f'Value {x} not in [{min_val}, {max_val}]').",
        solution="    x = int(val)\n    if x < min_val or x > max_val:\n        raise ValueError(f'Value {x} out of range [{min_val}, {max_val}]')\n    return x",
        test="assert parse_bounded_int('50', 0, 100) == 50\ntry:\n    parse_bounded_int('150', 0, 100)\n    assert False\nexcept ValueError:\n    pass\ntry:\n    parse_bounded_int('bad', 0, 100)\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 59. Read-Only Object Guard Context
    tasks.append(create_task(
        name="ReadOnlyGuard",
        signature="class ReadOnlyGuard:",
        doc_desc="Context manager that intercepts attribute assignment on target object, raising RuntimeError if an attribute modification is attempted.",
        doctests=[">>> class Data: x = 1", ">>> d = Data()", ">>> with ReadOnlyGuard(d): pass"],
        hint="In __enter__, dynamically subclass target.__class__ overriding __setattr__ to raise RuntimeError, and assign using object.__setattr__(self.target, '__class__', _RO). In __exit__, restore.",
        solution="    def __init__(self, target: Any):\n        self.target = target\n        self.orig_cls = target.__class__\n    def __enter__(self):\n        orig = self.orig_cls\n        class _RO(orig):\n            def __setattr__(self, name, val):\n                raise RuntimeError('Object is currently read-only')\n        object.__setattr__(self.target, '__class__', _RO)\n        return self.target\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        object.__setattr__(self.target, '__class__', self.orig_cls)\n        return False",
        test="class Item:\n    def __init__(self):\n        self.x = 10\nit = Item()\nwith ReadOnlyGuard(it):\n    try:\n        it.x = 20\n        assert False\n    except RuntimeError:\n        pass\nit.x = 20\nassert it.x == 20\n"
    ))

    # 60. Safe CSV Row Parser
    tasks.append(create_task(
        name="parse_csv_row_strict",
        signature="def parse_csv_row_strict(row: str, expected_cols: int) -> list[str]:",
        doc_desc="Splits row by comma, strips whitespace. Raises ValueError if column count does not match expected_cols.",
        doctests=[">>> parse_csv_row_strict('alice, 30, engineer', 3)", "['alice', '30', 'engineer']"],
        hint="cols = [c.strip() for c in row.split(',')]. If len(cols) != expected_cols raise ValueError.",
        solution="    cols = [c.strip() for c in row.split(',')]\n    if len(cols) != expected_cols:\n        raise ValueError(f'Expected {expected_cols} columns, found {len(cols)}')\n    return cols",
        test="assert parse_csv_row_strict('a, b, c', 3) == ['a', 'b', 'c']\ntry:\n    parse_csv_row_strict('a, b', 3)\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 61. Safe Expression Evaluator
    tasks.append(create_task(
        name="safe_eval_math_expression",
        signature="def safe_eval_math_expression(expr: str, default: float = 0.0) -> float:",
        prefix_code="import ast\nimport operator as op\n",
        doc_desc="Safely evaluates basic arithmetic expressions (+, -, *, /) using ast. Returns default on syntax error or disallowed nodes.",
        doctests=[">>> safe_eval_math_expression('2 + 3 * 4')", "14.0", ">>> safe_eval_math_expression('__import__', -1.0)", "-1.0"],
        hint="Parse ast.parse(expr, mode='eval'). Walk AST, only allow ast.BinOp, ast.UnaryOp, ast.Constant, ast.Expression. If illegal, return default.",
        solution="    operators = {\n        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,\n        ast.USub: op.neg\n    }\n    def _eval(node):\n        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):\n            return float(node.value)\n        elif isinstance(node, ast.BinOp) and type(node.op) in operators:\n            return operators[type(node.op)](_eval(node.left), _eval(node.right))\n        elif isinstance(node, ast.UnaryOp) and type(node.op) in operators:\n            return operators[type(node.op)](_eval(node.operand))\n        raise ValueError('Disallowed node')\n    try:\n        parsed = ast.parse(expr, mode='eval')\n        return float(_eval(parsed.body))\n    except Exception:\n        return default",
        test="assert safe_eval_math_expression('10 + 20 * 2') == 50.0\nassert safe_eval_math_expression('100 / 4') == 25.0\nassert safe_eval_math_expression('__import__(\"os\")', -1.0) == -1.0\nassert safe_eval_math_expression('bad expression', 99.0) == 99.0\n"
    ))

    # 62. Safe Slice Access
    tasks.append(create_task(
        name="safe_slice_extract",
        signature="def safe_slice_extract(seq: Any, start: int, end: int) -> list[Any]:",
        doc_desc="Safely slices sequence from start to end as a list. Returns [] if seq is None or cannot be sliced.",
        doctests=[">>> safe_slice_extract([1, 2, 3, 4], 1, 3)", "[2, 3]", ">>> safe_slice_extract(None, 0, 2)", "[]"],
        hint="Wrap list(seq[start:end]) in try-except catching (TypeError, KeyError).",
        solution="    try:\n        return list(seq[start:end])\n    except (TypeError, KeyError):\n        return []",
        test="assert safe_slice_extract([1, 2, 3, 4], 1, 3) == [2, 3]\nassert safe_slice_extract('hello', 1, 4) == ['e', 'l', 'l']\nassert safe_slice_extract(None, 0, 2) == []\nassert safe_slice_extract(12345, 0, 2) == []\n"
    ))

    # 63. State Rollback Context Manager
    tasks.append(create_task(
        name="StateRollbackContext",
        signature="class StateRollbackContext:",
        doc_desc="Context manager taking a snapshot of an object's __dict__. If an exception occurs, restores attributes to snapshot values.",
        doctests=[">>> class Config: x = 1", ">>> c = Config()", ">>> try:", "...     with StateRollbackContext(c):", "...         c.x = 2", "...         raise RuntimeError()", ">>> c.x", "1"],
        hint="In __enter__, save self.snapshot = dict(self.obj.__dict__). In __exit__, if exc_type is not None: self.obj.__dict__.clear(); self.obj.__dict__.update(self.snapshot).",
        solution="    def __init__(self, obj: Any):\n        self.obj = obj\n        self.snapshot = {}\n    def __enter__(self):\n        self.snapshot = dict(getattr(self.obj, '__dict__', {}))\n        return self.obj\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if exc_type is not None:\n            self.obj.__dict__.clear()\n            self.obj.__dict__.update(self.snapshot)\n        return False",
        test="class App:\n    def __init__(self):\n        self.status = 'idle'\n        self.count = 0\napp = App()\ntry:\n    with StateRollbackContext(app):\n        app.status = 'running'\n        app.count = 5\n        raise ValueError('crash')\nexcept ValueError:\n    pass\nassert app.status == 'idle' and app.count == 0\nwith StateRollbackContext(app):\n    app.status = 'completed'\nassert app.status == 'completed'\n"
    ))

    # 64. Exception Counter Tracker
    tasks.append(create_task(
        name="ExceptionTracker",
        signature="class ExceptionTracker:",
        doc_desc="Context manager tracking counts of exception types encountered via handle_exception(exc).",
        doctests=[">>> tracker = ExceptionTracker()", ">>> tracker.record(ValueError())", ">>> tracker.get_count('ValueError')", "1"],
        hint="Use a dictionary self.counts: dict[str, int] initialized in __init__. Increment in record(exc).",
        solution="    def __init__(self):\n        self.counts: dict[str, int] = {}\n    def record(self, exc: Exception) -> None:\n        name = type(exc).__name__\n        self.counts[name] = self.counts.get(name, 0) + 1\n    def get_count(self, exc_name: str) -> int:\n        return self.counts.get(exc_name, 0)\n    def total(self) -> int:\n        return sum(self.counts.values())",
        test="t = ExceptionTracker()\nt.record(ValueError('err 1'))\nt.record(ValueError('err 2'))\nt.record(KeyError('k'))\nassert t.get_count('ValueError') == 2\nassert t.get_count('KeyError') == 1\nassert t.get_count('IndexError') == 0\nassert t.total() == 3\n"
    ))

    return tasks
