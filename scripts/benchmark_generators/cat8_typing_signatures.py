"""
Category 8: Imports, Typing & Signatures (64 distinct challenges).
"""

from typing import List, Dict, Any
from .common import create_task


def build_typing_signatures_category() -> List[Dict[str, Any]]:
    tasks = []

    # 1. Validate Primitive Type
    tasks.append(create_task(
        name="validate_primitive_type",
        signature="def validate_primitive_type(val: Any, expected_type: type) -> bool:",
        doc_desc="Validates that val is an instance of expected_type. Disallows bool being treated as int.",
        doctests=[">>> validate_primitive_type(42, int)", "True", ">>> validate_primitive_type(True, int)", "False"],
        hint="If expected_type is int and isinstance(val, bool), return False. Otherwise return isinstance(val, expected_type).",
        solution="    if expected_type is int and isinstance(val, bool):\n        return False\n    return isinstance(val, expected_type)",
        test="assert validate_primitive_type(42, int) is True\nassert validate_primitive_type(True, int) is False\nassert validate_primitive_type('hello', str) is True\nassert validate_primitive_type(3.14, int) is False\n"
    ))

    # 2. Function Parameter Names
    tasks.append(create_task(
        name="get_func_param_names",
        signature="def get_func_param_names(func: Callable) -> list[str]:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Returns a list of parameter names for func in order of declaration using inspect.signature.",
        doctests=[">>> def add(a, b=1): pass", ">>> get_func_param_names(add)", "['a', 'b']"],
        hint="Use list(inspect.signature(func).parameters.keys()).",
        solution="    return list(inspect.signature(func).parameters.keys())",
        test="def foo(x, y, z=10): pass\nassert get_func_param_names(foo) == ['x', 'y', 'z']\ndef bar(): pass\nassert get_func_param_names(bar) == []\n"
    ))

    # 3. Enforce Integer Arguments Decorator
    tasks.append(create_task(
        name="enforce_int_args",
        signature="def enforce_int_args(func: Callable) -> Callable:",
        prefix_code="import functools\nfrom typing import Callable\n",
        doc_desc="Decorator verifying that all positional arguments passed to func are ints (and not bools). Raises TypeError otherwise.",
        doctests=[">>> @enforce_int_args\n... def add(a, b): return a + b", ">>> add(1, 2)", "3"],
        hint="In wrapper(*args, **kwargs): check any(isinstance(a, bool) or not isinstance(a, int) for a in args). If so, raise TypeError.",
        solution="    @functools.wraps(func)\n    def wrapper(*args, **kwargs):\n        for a in args:\n            if isinstance(a, bool) or not isinstance(a, int):\n                raise TypeError(f'Argument {a} is not an integer')\n        return func(*args, **kwargs)\n    return wrapper",
        test="@enforce_int_args\ndef mul(a, b): return a * b\nassert mul(3, 4) == 12\ntry:\n    mul(3, True)\n    assert False\nexcept TypeError:\n    pass\ntry:\n    mul(3, '4')\n    assert False\nexcept TypeError:\n    pass\n"
    ))

    # 4. Count Required Parameters
    tasks.append(create_task(
        name="count_required_parameters",
        signature="def count_required_parameters(func: Callable) -> int:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Counts parameters that have no default value and are not *args or **kwargs using inspect.",
        doctests=[">>> def f(a, b, c=1, *args, **kwargs): pass", ">>> count_required_parameters(f)", "2"],
        hint="Inspect params. Count p where p.default is inspect.Parameter.empty and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY).",
        solution="    sig = inspect.signature(func)\n    count = 0\n    for p in sig.parameters.values():\n        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):\n            if p.default is inspect.Parameter.empty:\n                count += 1\n    return count",
        test="def f1(a, b, c=10): pass\nassert count_required_parameters(f1) == 2\ndef f2(*args, **kwargs): pass\nassert count_required_parameters(f2) == 0\n"
    ))

    # 5. Bind Arguments
    tasks.append(create_task(
        name="bind_and_map_args",
        signature="def bind_and_map_args(func: Callable, *args: Any, **kwargs: Any) -> dict[str, Any]:",
        prefix_code="import inspect\nfrom typing import Callable, Any\n",
        doc_desc="Binds args and kwargs to func's signature and returns a dict mapping parameter names to bound values.",
        doctests=[">>> def f(x, y=2): pass", ">>> bind_and_map_args(f, 10, y=20)", "{'x': 10, 'y': 20}"],
        hint="Use inspect.signature(func).bind(*args, **kwargs).arguments.",
        solution="    sig = inspect.signature(func)\n    bound = sig.bind(*args, **kwargs)\n    return dict(bound.arguments)",
        test="def calc(a, b, op='add'): pass\nassert bind_and_map_args(calc, 5, 10) == {'a': 5, 'b': 10}\nassert bind_and_map_args(calc, 5, 10, op='mul') == {'a': 5, 'b': 10, 'op': 'mul'}\n"
    ))

    # 6. Safe Module Import
    tasks.append(create_task(
        name="safe_import_module",
        signature="def safe_import_module(module_name: str) -> Any | None:",
        prefix_code="import importlib\nfrom typing import Any\n",
        doc_desc="Attempts to import module_name dynamically. Returns the module object or None if ModuleNotFoundError or ImportError.",
        doctests=[">>> m = safe_import_module('math')", ">>> m.__name__", "'math'", ">>> safe_import_module('non_existent_module_xyz') is None", "True"],
        hint="Wrap importlib.import_module(module_name) in try-except (ModuleNotFoundError, ImportError).",
        solution="    try:\n        return importlib.import_module(module_name)\n    except (ModuleNotFoundError, ImportError):\n        return None",
        test="m = safe_import_module('json')\nassert m is not None and m.__name__ == 'json'\nassert safe_import_module('unlikely_module_123_xyz') is None\n"
    ))

    # 7. Point2D NamedTuple
    tasks.append(create_task(
        name="Point2D",
        signature="class Point2D(NamedTuple):",
        prefix_code="import math\nfrom typing import NamedTuple\n",
        doc_desc="NamedTuple with fields x: float, y: float, and method distance_to(other: Point2D) -> float.",
        doctests=[">>> p1 = Point2D(0.0, 0.0)", ">>> p2 = Point2D(3.0, 4.0)", ">>> p1.distance_to(p2)", "5.0"],
        hint="Define x: float and y: float. distance_to computes math.hypot(self.x - other.x, self.y - other.y).",
        solution="    x: float\n    y: float\n    def distance_to(self, other: 'Point2D') -> float:\n        return math.hypot(self.x - other.x, self.y - other.y)",
        test="p1 = Point2D(0.0, 0.0)\np2 = Point2D(3.0, 4.0)\nassert p1.distance_to(p2) == 5.0\nassert p1.x == 0.0 and p1.y == 0.0\n"
    ))

    # 8. Validate Homogeneous List
    tasks.append(create_task(
        name="validate_homogeneous_list",
        signature="def validate_homogeneous_list(items: list[Any], expected_type: type) -> bool:",
        doc_desc="Verifies all elements in items are instances of expected_type (strict for int vs bool). Empty list returns True.",
        doctests=[">>> validate_homogeneous_list([1, 2, 3], int)", "True", ">>> validate_homogeneous_list([1, True], int)", "False"],
        hint="For x in items: if expected_type is int and isinstance(x, bool), return False. If not isinstance(x, expected_type), return False. Return True.",
        solution="    for x in items:\n        if expected_type is int and isinstance(x, bool):\n            return False\n        if not isinstance(x, expected_type):\n            return False\n    return True",
        test="assert validate_homogeneous_list([1, 2, 3], int) is True\nassert validate_homogeneous_list([1, True], int) is False\nassert validate_homogeneous_list(['a', 'b'], str) is True\nassert validate_homogeneous_list([], int) is True\n"
    ))

    # 9. Return Type Annotation Inspector
    tasks.append(create_task(
        name="inspect_return_type",
        signature="def inspect_return_type(func: Callable) -> type | None:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Extracts the return type annotation of func. Returns None if unannotated or inspect.Signature.empty.",
        doctests=[">>> def f() -> int: return 0", ">>> inspect_return_type(f)", "<class 'int'>"],
        hint="sig = inspect.signature(func); if sig.return_annotation is inspect.Signature.empty, return None; else return sig.return_annotation.",
        solution="    sig = inspect.signature(func)\n    ret = sig.return_annotation\n    if ret is inspect.Signature.empty:\n        return None\n    return ret",
        test="def f1() -> str: pass\nassert inspect_return_type(f1) is str\ndef f2(): pass\nassert inspect_return_type(f2) is None\n"
    ))

    # 10. Immutable Config Dataclass
    tasks.append(create_task(
        name="ImmutableConfig",
        signature="class ImmutableConfig:",
        prefix_code="from dataclasses import dataclass, FrozenInstanceError\n\n@dataclass(frozen=True)\n",
        doc_desc="Frozen dataclass with fields name: str, version: int, debug: bool = False. Disallows attribute assignment after creation.",
        doctests=[">>> cfg = ImmutableConfig('app', 1)", ">>> cfg.name", "'app'"],
        hint="Declare fields name: str, version: int, debug: bool = False.",
        solution="    name: str\n    version: int\n    debug: bool = False",
        test="cfg = ImmutableConfig('app', 1)\nassert cfg.name == 'app' and cfg.version == 1 and cfg.debug is False\ntry:\n    cfg.name = 'new'\n    assert False\nexcept FrozenInstanceError:\n    pass\n"
    ))

    # 11. Validate Dict Key-Value Types
    tasks.append(create_task(
        name="validate_dict_types",
        signature="def validate_dict_types(d: dict[Any, Any], key_type: type, val_type: type) -> bool:",
        doc_desc="Validates that every key is of type key_type and every value is of type val_type (strict for int vs bool).",
        doctests=[">>> validate_dict_types({'a': 1, 'b': 2}, str, int)", "True"],
        hint="Check isinstance for each k, v. Disallow bool for int.",
        solution="    for k, v in d.items():\n        if key_type is int and isinstance(k, bool):\n            return False\n        if val_type is int and isinstance(v, bool):\n            return False\n        if not isinstance(k, key_type) or not isinstance(v, val_type):\n            return False\n    return True",
        test="assert validate_dict_types({'a': 1, 'b': 2}, str, int) is True\nassert validate_dict_types({'a': True}, str, int) is False\nassert validate_dict_types({1: 'a'}, int, str) is True\nassert validate_dict_types({}, str, int) is True\n"
    ))

    # 12. Extract Default Parameter Values
    tasks.append(create_task(
        name="extract_default_kwargs",
        signature="def extract_default_kwargs(func: Callable) -> dict[str, Any]:",
        prefix_code="import inspect\nfrom typing import Callable, Any\n",
        doc_desc="Returns a dict {param_name: default_value} for all parameters of func that define a default value.",
        doctests=[">>> def f(a, b=10, c='hi'): pass", ">>> extract_default_kwargs(f)", "{'b': 10, 'c': 'hi'}"],
        hint="Iterate through inspect.signature(func).parameters.items(). Check p.default is not inspect.Parameter.empty.",
        solution="    sig = inspect.signature(func)\n    return {name: p.default for name, p in sig.parameters.items() if p.default is not inspect.Parameter.empty}",
        test="def f(x, y=20, z=None): pass\nassert extract_default_kwargs(f) == {'y': 20, 'z': None}\ndef g(a, b): pass\nassert extract_default_kwargs(g) == {}\n"
    ))

    # 13. Detect Variable Arguments
    tasks.append(create_task(
        name="detect_varargs",
        signature="def detect_varargs(func: Callable) -> tuple[bool, bool]:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Returns (has_var_positional, has_var_keyword) indicating if func accepts *args or **kwargs.",
        doctests=[">>> def f(*args, **kwargs): pass", ">>> detect_varargs(f)", "(True, True)"],
        hint="Check for p.kind == inspect.Parameter.VAR_POSITIONAL and p.kind == inspect.Parameter.VAR_KEYWORD.",
        solution="    sig = inspect.signature(func)\n    has_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())\n    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())\n    return (has_args, has_kwargs)",
        test="assert detect_varargs(lambda *args: None) == (True, False)\nassert detect_varargs(lambda **kwargs: None) == (False, True)\nassert detect_varargs(lambda a, b: None) == (False, False)\n"
    ))

    # 14. Result Monad / Container
    tasks.append(create_task(
        name="ResultContainer",
        signature="class ResultContainer:",
        prefix_code="from typing import Any\n",
        doc_desc="Container representing Ok(value) or Err(error). Exposes is_ok(), is_err(), and unwrap() which raises ValueError if Err.",
        doctests=[">>> r = ResultContainer.ok(42)", ">>> r.is_ok()", "True", ">>> r.unwrap()", "42"],
        hint="Store self.success: bool, self.value, self.error. In unwrap(), raise ValueError(self.error) if not self.success.",
        solution="    def __init__(self, success: bool, value: Any = None, error: Any = None):\n        self.success = success\n        self.value = value\n        self.error = error\n    @classmethod\n    def ok(cls, value: Any) -> 'ResultContainer':\n        return cls(True, value=value)\n    @classmethod\n    def err(cls, error: Any) -> 'ResultContainer':\n        return cls(False, error=error)\n    def is_ok(self) -> bool:\n        return self.success\n    def is_err(self) -> bool:\n        return not self.success\n    def unwrap(self) -> Any:\n        if not self.success:\n            raise ValueError(f'Cannot unwrap Err: {self.error}')\n        return self.value",
        test="r1 = ResultContainer.ok(100)\nassert r1.is_ok() and r1.unwrap() == 100\nr2 = ResultContainer.err('database offline')\nassert r2.is_err()\ntry:\n    r2.unwrap()\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 15. Validate Tuple Types
    tasks.append(create_task(
        name="validate_tuple_types",
        signature="def validate_tuple_types(tup: tuple[Any, ...], expected_types: list[type]) -> bool:",
        doc_desc="Validates that tup has the same length as expected_types and each element matches the corresponding expected type.",
        doctests=[">>> validate_tuple_types((1, 'a'), [int, str])", "True", ">>> validate_tuple_types((1,), [int, str])", "False"],
        hint="If len(tup) != len(expected_types) return False. For (elem, exp) in zip(tup, expected_types), check isinstance (strict int vs bool).",
        solution="    if len(tup) != len(expected_types):\n        return False\n    for elem, exp in zip(tup, expected_types):\n        if exp is int and isinstance(elem, bool):\n            return False\n        if not isinstance(elem, exp):\n            return False\n    return True",
        test="assert validate_tuple_types((1, 'hello', 3.14), [int, str, float]) is True\nassert validate_tuple_types((True, 'hello'), [int, str]) is False\nassert validate_tuple_types((1,), [int, int]) is False\n"
    ))

    # 16. Validated User Dataclass
    tasks.append(create_task(
        name="ValidatedUser",
        signature="class ValidatedUser:",
        prefix_code="from dataclasses import dataclass\n\n@dataclass\n",
        doc_desc="Dataclass with fields name: str, age: int. __post_init__ raises ValueError if name is empty or age < 0, TypeError if wrong types.",
        doctests=[">>> u = ValidatedUser('Alice', 25)", ">>> u.name", "'Alice'"],
        hint="In __post_init__, check isinstance(self.name, str) and not empty; isinstance(self.age, int) and not bool and self.age >= 0.",
        solution="    name: str\n    age: int\n    def __post_init__(self):\n        if not isinstance(self.name, str):\n            raise TypeError('name must be str')\n        if not self.name.strip():\n            raise ValueError('name cannot be empty')\n        if isinstance(self.age, bool) or not isinstance(self.age, int):\n            raise TypeError('age must be int')\n        if self.age < 0:\n            raise ValueError('age cannot be negative')",
        test="u = ValidatedUser('Bob', 30)\nassert u.name == 'Bob' and u.age == 30\ntry:\n    ValidatedUser('', 20)\n    assert False\nexcept ValueError:\n    pass\ntry:\n    ValidatedUser('Alice', -1)\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 17. Check Arity of Callable
    tasks.append(create_task(
        name="is_callable_with_n_args",
        signature="def is_callable_with_n_args(obj: Any, n: int) -> bool:",
        prefix_code="import inspect\nfrom typing import Any\n",
        doc_desc="Returns True if obj is callable and can be called with exactly n positional arguments without error.",
        doctests=[">>> is_callable_with_n_args(lambda x, y: x + y, 2)", "True", ">>> is_callable_with_n_args(lambda x: x, 3)", "False"],
        hint="Check callable(obj). Inspect signature. Try sig.bind(*([None] * n)). Return True on success, False on TypeError.",
        solution="    if not callable(obj):\n        return False\n    try:\n        sig = inspect.signature(obj)\n        sig.bind(*([None] * n))\n        return True\n    except TypeError:\n        return False",
        test="assert is_callable_with_n_args(lambda x, y: x, 2) is True\nassert is_callable_with_n_args(lambda x, y=1: x, 1) is True\nassert is_callable_with_n_args(lambda x: x, 3) is False\nassert is_callable_with_n_args('not_a_func', 1) is False\n"
    ))

    # 18. Serialize Dataclass to Dict
    tasks.append(create_task(
        name="serialize_dataclass",
        signature="def serialize_dataclass(dc_obj: Any) -> dict[str, Any]:",
        prefix_code="import dataclasses\nfrom typing import Any\n",
        doc_desc="Converts a dataclass instance into a dictionary using dataclasses.asdict.",
        doctests=[">>> @dataclasses.dataclass\n... class Item: id: int\n>>> serialize_dataclass(Item(1))", "{'id': 1}"],
        hint="Use dataclasses.asdict(dc_obj).",
        solution="    return dataclasses.asdict(dc_obj)",
        test="@dataclasses.dataclass\nclass Product:\n    sku: str\n    price: float\np = Product('ABC', 9.99)\nassert serialize_dataclass(p) == {'sku': 'ABC', 'price': 9.99}\n"
    ))

    # 19. Check Signature Identity
    tasks.append(create_task(
        name="check_signatures_identical",
        signature="def check_signatures_identical(f1: Callable, f2: Callable) -> bool:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Checks if f1 and f2 have identical parameters (names, kinds, defaults, and annotations) using inspect.signature.",
        doctests=[">>> def a(x: int): pass\n>>> def b(x: int): pass", ">>> check_signatures_identical(a, b)", "True"],
        hint="Compare inspect.signature(f1) == inspect.signature(f2).",
        solution="    return inspect.signature(f1) == inspect.signature(f2)",
        test="def f1(x: int, y: str = 'a') -> bool: pass\ndef f2(x: int, y: str = 'a') -> bool: pass\ndef f3(x: int, y: int = 1) -> bool: pass\nassert check_signatures_identical(f1, f2) is True\nassert check_signatures_identical(f1, f3) is False\n"
    ))

    # 20. String Key Dictionary Subclass
    tasks.append(create_task(
        name="StringKeyDict",
        signature="class StringKeyDict(dict):",
        doc_desc="Dictionary subclass enforcing that all keys must be of type str. Raises TypeError on non-string key assignment.",
        doctests=[">>> d = StringKeyDict()", ">>> d['k'] = 1", ">>> d['k']", "1"],
        hint="Override __setitem__(self, key, value). Check isinstance(key, str), otherwise raise TypeError.",
        solution="    def __setitem__(self, key: Any, value: Any) -> None:\n        if not isinstance(key, str):\n            raise TypeError(f'Key must be str, got {type(key).__name__}')\n        super().__setitem__(key, value)",
        test="d = StringKeyDict()\nd['name'] = 'test'\nassert d['name'] == 'test'\ntry:\n    d[123] = 'num'\n    assert False\nexcept TypeError:\n    pass\n"
    ))

    # 21. Safe Typed Getattr
    tasks.append(create_task(
        name="safe_getattr_typed",
        signature="def safe_getattr_typed(obj: Any, attr: str, expected_type: type, default: Any = None) -> Any:",
        doc_desc="Returns getattr(obj, attr) if attribute exists and is an instance of expected_type, otherwise returns default.",
        doctests=[">>> class O: val = 42\n>>> safe_getattr_typed(O(), 'val', int)", "42"],
        hint="Check hasattr(obj, attr). If val is instance of expected_type (strict int vs bool), return it, else default.",
        solution="    if not hasattr(obj, attr):\n        return default\n    val = getattr(obj, attr)\n    if expected_type is int and isinstance(val, bool):\n        return default\n    if isinstance(val, expected_type):\n        return val\n    return default",
        test="class Item:\n    count = 10\n    tag = 'A'\nit = Item()\nassert safe_getattr_typed(it, 'count', int) == 10\nassert safe_getattr_typed(it, 'count', str, 'missing') == 'missing'\nassert safe_getattr_typed(it, 'unknown', int, 0) == 0\n"
    ))

    # 22. Inspect Docstring Summary
    tasks.append(create_task(
        name="inspect_docstring_summary",
        signature="def inspect_docstring_summary(func: Callable) -> str:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Returns the first non-empty line of func's docstring stripped. Returns empty string if no docstring exists.",
        doctests=[">>> def f():\\n...     '''First line.\\n\\nSecond line.'''\\n...     pass", ">>> inspect_docstring_summary(f)", "'First line.'"],
        hint="doc = inspect.getdoc(func) or ''. Split lines, return first line stripped, or empty string.",
        solution="    doc = inspect.getdoc(func)\n    if not doc:\n        return ''\n    for line in doc.splitlines():\n        line = line.strip()\n        if line:\n            return line\n    return ''",
        test="def f1():\n    '''Compute the sum.\n    More details.\n    '''\n    pass\nassert inspect_docstring_summary(f1) == 'Compute the sum.'\ndef f2(): pass\nassert inspect_docstring_summary(f2) == ''\n"
    ))

    # 23. Filter Callables From Dict
    tasks.append(create_task(
        name="filter_callables",
        signature="def filter_callables(data: dict[str, Any]) -> dict[str, Callable]:",
        prefix_code="from typing import Callable, Any\n",
        doc_desc="Returns a sub-dictionary containing only items where callable(value) is True.",
        doctests=[">>> filter_callables({'a': lambda: 1, 'b': 42})", "{'a': ...}"],
        hint="Use dict comprehension {k: v for k, v in data.items() if callable(v)}.",
        solution="    return {k: v for k, v in data.items() if callable(v)}",
        test="res = filter_callables({'fn': len, 'val': 100, 'cls': int})\nassert 'fn' in res and 'cls' in res and 'val' not in res\n"
    ))

    # 24. Validated RangeBound Dataclass
    tasks.append(create_task(
        name="RangeBound",
        signature="class RangeBound:",
        prefix_code="from dataclasses import dataclass\n\n@dataclass\n",
        doc_desc="Dataclass with low: float and high: float. __post_init__ asserts low <= high, raising ValueError otherwise.",
        doctests=[">>> r = RangeBound(0.0, 10.0)", ">>> r.high", "10.0"],
        hint="In __post_init__, if self.low > self.high, raise ValueError(f'low ({self.low}) > high ({self.high})').",
        solution="    low: float\n    high: float\n    def __post_init__(self):\n        if self.low > self.high:\n            raise ValueError(f'low ({self.low}) cannot exceed high ({self.high})')\n    def contains(self, val: float) -> bool:\n        return self.low <= val <= self.high",
        test="r = RangeBound(1.0, 5.0)\nassert r.contains(3.0) is True\nassert r.contains(6.0) is False\ntry:\n    RangeBound(10.0, 2.0)\n    assert False\nexcept ValueError:\n    pass\n"
    ))

    # 25. Dynamic NamedTuple Factory
    tasks.append(create_task(
        name="create_named_tuple_record",
        signature="def create_named_tuple_record(type_name: str, field_names: list[str], values: list[Any]) -> tuple[Any, ...]:",
        prefix_code="from collections import namedtuple\nfrom typing import Any\n",
        doc_desc="Dynamically constructs a namedtuple class and instantiates it with values.",
        doctests=[">>> r = create_named_tuple_record('Person', ['name', 'age'], ['Bob', 25])", ">>> r.name", "'Bob'"],
        hint="cls = namedtuple(type_name, field_names); return cls(*values).",
        solution="    cls = namedtuple(type_name, field_names)\n    return cls(*values)",
        test="rec = create_named_tuple_record('Coord', ['lat', 'lon'], [37.77, -122.41])\nassert rec.lat == 37.77 and rec.lon == -122.41\nassert type(rec).__name__ == 'Coord'\n"
    ))

    # 26. Validate Optional Type
    tasks.append(create_task(
        name="validate_optional_type",
        signature="def validate_optional_type(val: Any, base_type: type) -> bool:",
        doc_desc="Checks if val is None or an instance of base_type (strict for int vs bool).",
        doctests=[">>> validate_optional_type(None, int)", "True", ">>> validate_optional_type(42, int)", "True"],
        hint="If val is None return True. Else check base_type is int vs bool, then isinstance.",
        solution="    if val is None:\n        return True\n    if base_type is int and isinstance(val, bool):\n        return False\n    return isinstance(val, base_type)",
        test="assert validate_optional_type(None, str) is True\nassert validate_optional_type('abc', str) is True\nassert validate_optional_type(123, str) is False\nassert validate_optional_type(True, int) is False\n"
    ))

    # 27. Has Parameter Name
    tasks.append(create_task(
        name="has_parameter_name",
        signature="def has_parameter_name(func: Callable, param_name: str) -> bool:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Returns True if param_name is in func's inspect.signature parameters.",
        doctests=[">>> def f(alpha, beta=1): pass", ">>> has_parameter_name(f, 'alpha')", "True"],
        hint="Return param_name in inspect.signature(func).parameters.",
        solution="    return param_name in inspect.signature(func).parameters",
        test="def my_func(a, b, timeout=10): pass\nassert has_parameter_name(my_func, 'timeout') is True\nassert has_parameter_name(my_func, 'missing') is False\n"
    ))

    # 28. Typed Stack Container
    tasks.append(create_task(
        name="TypedStack",
        signature="class TypedStack:",
        prefix_code="from typing import Any\n",
        doc_desc="Stack enforcing that all pushed elements match item_type. push() raises TypeError on mismatch.",
        doctests=[">>> s = TypedStack(int)", ">>> s.push(10)", ">>> s.pop()", "10"],
        hint="Check item is not bool if item_type is int, and isinstance(item, self.item_type). Raise TypeError on failure.",
        solution="    def __init__(self, item_type: type):\n        self.item_type = item_type\n        self.items: list[Any] = []\n    def push(self, item: Any) -> None:\n        if self.item_type is int and isinstance(item, bool):\n            raise TypeError('Expected int, got bool')\n        if not isinstance(item, self.item_type):\n            raise TypeError(f'Expected {self.item_type.__name__}, got {type(item).__name__}')\n        self.items.append(item)\n    def pop(self) -> Any:\n        if not self.items:\n            raise IndexError('Pop from empty stack')\n        return self.items.pop()\n    def __len__(self) -> int:\n        return len(self.items)",
        test="s = TypedStack(str)\ns.push('hello')\ns.push('world')\nassert len(s) == 2\nassert s.pop() == 'world'\ntry:\n    s.push(123)\n    assert False\nexcept TypeError:\n    pass\n"
    ))

    # 29. Cast or Raise Type Exception
    tasks.append(create_task(
        name="cast_or_raise",
        signature="def cast_or_raise(val: Any, target_type: type) -> Any:",
        doc_desc="Attempts to cast val to target_type. Raises TypeError with clear message if cast fails.",
        doctests=[">>> cast_or_raise('123', int)", "123"],
        hint="Wrap target_type(val) in try-except (ValueError, TypeError). Raise TypeError(f'Failed to cast {val} to {target_type.__name__}') from exc.",
        solution="    try:\n        return target_type(val)\n    except (ValueError, TypeError) as exc:\n        raise TypeError(f'Cannot cast {val!r} to {target_type.__name__}') from exc",
        test="assert cast_or_raise('42', int) == 42\nassert cast_or_raise([1, 2], tuple) == (1, 2)\ntry:\n    cast_or_raise('abc', int)\n    assert False\nexcept TypeError:\n    pass\n"
    ))

    # 30. Inspect Direct Class Methods
    tasks.append(create_task(
        name="inspect_class_methods",
        signature="def inspect_class_methods(cls: type) -> list[str]:",
        prefix_code="import inspect\n",
        doc_desc="Returns a sorted list of method names defined directly in cls (excluding dunder methods).",
        doctests=[">>> class A: def run(self): pass", ">>> inspect_class_methods(A)", "['run']"],
        hint="Iterate cls.__dict__.items(). If callable(val) or isinstance(val, (staticmethod, classmethod)) and not k.startswith('__'): add k. Return sorted list.",
        solution="    methods = []\n    for k, v in cls.__dict__.items():\n        if k.startswith('__') and k.endswith('__'):\n            continue\n        if callable(v) or isinstance(v, (staticmethod, classmethod)):\n            methods.append(k)\n    return sorted(methods)",
        test="class Demo:\n    def foo(self): pass\n    def bar(self): pass\n    x = 10\nassert inspect_class_methods(Demo) == ['bar', 'foo']\n"
    ))

    # 31. Parse Annotated Function Parameters
    tasks.append(create_task(
        name="parse_annotated_parameters",
        signature="def parse_annotated_parameters(func: Callable) -> dict[str, type]:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Returns a dict {param_name: annotation_type} for parameters with explicit type annotations.",
        doctests=[">>> def f(a: int, b: str, c): pass", ">>> parse_annotated_parameters(f)", "{'a': <class 'int'>, 'b': <class 'str'>}"],
        hint="Iterate inspect.signature(func).parameters.items(). If p.annotation is not inspect.Parameter.empty, record {name: p.annotation}.",
        solution="    sig = inspect.signature(func)\n    return {name: p.annotation for name, p in sig.parameters.items() if p.annotation is not inspect.Parameter.empty}",
        test="def calc(x: float, y: float, flag=True) -> float: pass\nassert parse_annotated_parameters(calc) == {'x': float, 'y': float}\n"
    ))

    # 32. Validate Union Type
    tasks.append(create_task(
        name="validate_union_type",
        signature="def validate_union_type(val: Any, allowed_types: tuple[type, ...]) -> bool:",
        doc_desc="Validates that val matches at least one of allowed_types (strict for int vs bool).",
        doctests=[">>> validate_union_type(42, (int, str))", "True", ">>> validate_union_type(True, (int, str))", "False"],
        hint="Loop over allowed_types. For t in allowed_types: if t is int and isinstance(val, bool): continue; if isinstance(val, t): return True. Return False.",
        solution="    for t in allowed_types:\n        if t is int and isinstance(val, bool):\n            continue\n        if isinstance(val, t):\n            return True\n    return False",
        test="assert validate_union_type(10, (int, str)) is True\nassert validate_union_type('abc', (int, str)) is True\nassert validate_union_type(True, (int, str)) is False\nassert validate_union_type(3.14, (int, str)) is False\n"
    ))

    # 33. Key-Value Schema Validator
    tasks.append(create_task(
        name="KeyValueSchemaValidator",
        signature="class KeyValueSchemaValidator:",
        doc_desc="Validates dict against expected {key: type}. validate(d) returns (is_valid: bool, error_messages: list[str]).",
        doctests=[">>> v = KeyValueSchemaValidator({'name': str, 'age': int})", ">>> ok, errs = v.validate({'name': 'A', 'age': 20})", ">>> ok", "True"],
        hint="Check missing keys and mismatched types (strict for int vs bool). Collect error strings.",
        solution="    def __init__(self, schema: dict[str, type]):\n        self.schema = schema\n    def validate(self, d: dict) -> tuple[bool, list[str]]:\n        errors = []\n        for k, exp_type in self.schema.items():\n            if k not in d:\n                errors.append(f'Missing field: {k}')\n                continue\n            val = d[k]\n            if exp_type is int and isinstance(val, bool):\n                errors.append(f'Field {k} expected int, got bool')\n            elif not isinstance(val, exp_type):\n                errors.append(f'Field {k} expected {exp_type.__name__}, got {type(val).__name__}')\n        return (len(errors) == 0, errors)",
        test="v = KeyValueSchemaValidator({'id': int, 'title': str})\nok, errs = v.validate({'id': 1, 'title': 'Test'})\nassert ok and not errs\nok2, errs2 = v.validate({'id': 'bad'})\nassert not ok2 and len(errs2) == 2\n"
    ))

    # 34. Dataclass to JSON String
    tasks.append(create_task(
        name="dataclass_to_json_string",
        signature="def dataclass_to_json_string(dc_instance: Any) -> str:",
        prefix_code="import dataclasses\nimport json\nfrom typing import Any\n",
        doc_desc="Converts dataclass instance into a sorted JSON string using dataclasses.asdict and json.dumps.",
        doctests=[">>> @dataclasses.dataclass\n... class P: x: int\n>>> dataclass_to_json_string(P(1))", "'{\"x\": 1}'"],
        hint="Use json.dumps(dataclasses.asdict(dc_instance), sort_keys=True).",
        solution="    return json.dumps(dataclasses.asdict(dc_instance), sort_keys=True)",
        test="@dataclasses.dataclass\nclass Point:\n    x: int\n    y: int\nassert dataclass_to_json_string(Point(2, 1)) == '{\"x\": 2, \"y\": 1}'\n"
    ))

    # 35. Copy Function Metadata Decorator
    tasks.append(create_task(
        name="copy_func_metadata",
        signature="def copy_func_metadata(source: Callable, target: Callable) -> Callable:",
        prefix_code="from typing import Callable\n",
        doc_desc="Copies __name__, __doc__, and __annotations__ from source callable to target callable and returns target.",
        doctests=[">>> def src(): pass", ">>> def dst(): pass", ">>> copy_func_metadata(src, dst).__name__", "'src'"],
        hint="Copy target.__name__ = source.__name__, target.__doc__ = source.__doc__, target.__annotations__ = dict(getattr(source, '__annotations__', {})).",
        solution="    target.__name__ = source.__name__\n    target.__doc__ = source.__doc__\n    target.__annotations__ = dict(getattr(source, '__annotations__', {}))\n    return target",
        test="def original(a: int) -> str:\n    '''Original doc.'''\n    pass\ndef wrapper(*args): pass\ncopy_func_metadata(original, wrapper)\nassert wrapper.__name__ == 'original'\nassert wrapper.__doc__ == 'Original doc.'\nassert wrapper.__annotations__ == {'a': int, 'return': str}\n"
    ))

    # 36. Recursive Subclass Finder
    tasks.append(create_task(
        name="get_all_subclasses",
        signature="def get_all_subclasses(cls: type) -> set[type]:",
        doc_desc="Recursively discovers all subclasses of cls using cls.__subclasses__().",
        doctests=[">>> class Base: pass\n>>> class Child(Base): pass", ">>> Child in get_all_subclasses(Base)", "True"],
        hint="Use BFS or DFS with seen set on cls.__subclasses__().",
        solution="    result = set()\n    stack = list(cls.__subclasses__())\n    while stack:\n        sub = stack.pop()\n        if sub not in result:\n            result.add(sub)\n            stack.extend(sub.__subclasses__())\n    return result",
        test="class Root: pass\nclass Level1(Root): pass\nclass Level2(Level1): pass\nsubs = get_all_subclasses(Root)\nassert Level1 in subs and Level2 in subs\n"
    ))

    # 37. Typed Property Descriptor
    tasks.append(create_task(
        name="TypedPropertyDescriptor",
        signature="class TypedPropertyDescriptor:",
        prefix_code="from typing import Any\n",
        doc_desc="Descriptor enforcing that assigned values match declared type. Raises TypeError on assignment mismatch.",
        doctests=[">>> class Item:\n...     x = TypedPropertyDescriptor(int)", ">>> it = Item()\n>>> it.x = 10\n>>> it.x", "10"],
        hint="In __init__, save self.expected_type. In __set_name__(self, owner, name), self.name = name. In __set__, check type and set in instance.__dict__.",
        solution="    def __init__(self, expected_type: type):\n        self.expected_type = expected_type\n        self.name = ''\n    def __set_name__(self, owner, name):\n        self.name = name\n    def __get__(self, instance, owner):\n        if instance is None:\n            return self\n        return instance.__dict__.get(self.name)\n    def __set__(self, instance, value):\n        if self.expected_type is int and isinstance(value, bool):\n            raise TypeError('Expected int, got bool')\n        if not isinstance(value, self.expected_type):\n            raise TypeError(f'Expected {self.expected_type.__name__}, got {type(value).__name__}')\n        instance.__dict__[self.name] = value",
        test="class Model:\n    age = TypedPropertyDescriptor(int)\nm = Model()\nm.age = 25\nassert m.age == 25\ntry:\n    m.age = 'bad'\n    assert False\nexcept TypeError:\n    pass\n"
    ))

    # 38. Check Coroutine Function
    tasks.append(create_task(
        name="is_coroutine_function",
        signature="def is_coroutine_function(obj: Any) -> bool:",
        prefix_code="import inspect\nfrom typing import Any\n",
        doc_desc="Returns True if obj is defined as an async coroutine function using inspect.iscoroutinefunction.",
        doctests=[">>> async def f(): pass\n>>> is_coroutine_function(f)", "True"],
        hint="Return inspect.iscoroutinefunction(obj).",
        solution="    return inspect.iscoroutinefunction(obj)",
        test="async def afn(): pass\ndef sfn(): pass\nassert is_coroutine_function(afn) is True\nassert is_coroutine_function(sfn) is False\nassert is_coroutine_function(123) is False\n"
    ))

    # 39. Extract Dataclass Field Names
    tasks.append(create_task(
        name="extract_dataclass_fields",
        signature="def extract_dataclass_fields(dc_cls: type) -> list[str]:",
        prefix_code="import dataclasses\n",
        doc_desc="Returns list of field names defined in dataclass cls in order of definition.",
        doctests=[">>> @dataclasses.dataclass\n... class U: a: int; b: str\n>>> extract_dataclass_fields(U)", "['a', 'b']"],
        hint="Use [f.name for f in dataclasses.fields(dc_cls)].",
        solution="    return [f.name for f in dataclasses.fields(dc_cls)]",
        test="@dataclasses.dataclass\nclass Config:\n    host: str\n    port: int\n    debug: bool\nassert extract_dataclass_fields(Config) == ['host', 'port', 'debug']\n"
    ))

    # 40. Version Tuple NamedTuple
    tasks.append(create_task(
        name="VersionTuple",
        signature="class VersionTuple(NamedTuple):",
        prefix_code="from typing import NamedTuple\n",
        doc_desc="NamedTuple with fields major: int, minor: int, patch: int. Formats __str__ as 'major.minor.patch'.",
        doctests=[">>> v = VersionTuple(1, 2, 3)", ">>> str(v)", "'1.2.3'"],
        hint="Declare major: int, minor: int, patch: int. In __str__, return f'{self.major}.{self.minor}.{self.patch}'.",
        solution="    major: int\n    minor: int\n    patch: int\n    def __str__(self) -> str:\n        return f'{self.major}.{self.minor}.{self.patch}'",
        test="v = VersionTuple(2, 0, 1)\nassert str(v) == '2.0.1'\nassert v.major == 2 and v.minor == 0 and v.patch == 1\nassert VersionTuple(1, 0, 0) < VersionTuple(1, 1, 0)\n"
    ))

    # 41. Inspect Keyword-Only Parameters
    tasks.append(create_task(
        name="inspect_keyword_only_params",
        signature="def inspect_keyword_only_params(func: Callable) -> list[str]:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Returns list of parameter names that are keyword-only (*, arg) in order of definition.",
        doctests=[">>> def f(a, *, b, c=1): pass", ">>> inspect_keyword_only_params(f)", "['b', 'c']"],
        hint="Filter inspect.signature(func).parameters.values() for p.kind == inspect.Parameter.KEYWORD_ONLY.",
        solution="    sig = inspect.signature(func)\n    return [p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.KEYWORD_ONLY]",
        test="def f(x, *, opt1=1, opt2=2): pass\nassert inspect_keyword_only_params(f) == ['opt1', 'opt2']\ndef g(a, b): pass\nassert inspect_keyword_only_params(g) == []\n"
    ))

    # 42. Strict Type Mapping
    tasks.append(create_task(
        name="StrictTypeMapping",
        signature="class StrictTypeMapping(dict):",
        prefix_code="from typing import Any\n",
        doc_desc="Dictionary enforcing that keys match key_type and values match val_type on assignment. Raises TypeError on violation.",
        doctests=[">>> sm = StrictTypeMapping(str, int)", ">>> sm['a'] = 1", ">>> sm['a']", "1"],
        hint="In __init__, store key_type and val_type. In __setitem__, check types (strict int vs bool). Raise TypeError.",
        solution="    def __init__(self, key_type: type, val_type: type):\n        super().__init__()\n        self.key_type = key_type\n        self.val_type = val_type\n    def __setitem__(self, key: Any, val: Any) -> None:\n        if self.key_type is int and isinstance(key, bool):\n            raise TypeError('Expected int key, got bool')\n        if self.val_type is int and isinstance(val, bool):\n            raise TypeError('Expected int val, got bool')\n        if not isinstance(key, self.key_type):\n            raise TypeError(f'Invalid key type: {type(key).__name__}')\n        if not isinstance(val, self.val_type):\n            raise TypeError(f'Invalid val type: {type(val).__name__}')\n        super().__setitem__(key, val)",
        test="m = StrictTypeMapping(str, float)\nm['pi'] = 3.14\nassert m['pi'] == 3.14\ntry:\n    m[1] = 2.0\n    assert False\nexcept TypeError:\n    pass\ntry:\n    m['x'] = 'str'\n    assert False\nexcept TypeError:\n    pass\n"
    ))

    # 43. Resolve Builtin Type From Name
    tasks.append(create_task(
        name="resolve_type_from_name",
        signature="def resolve_type_from_name(type_name: str) -> type | None:",
        doc_desc="Resolves type names ('int', 'str', 'float', 'bool', 'list', 'dict', 'set', 'tuple') to their Python type object. Returns None if unknown.",
        doctests=[">>> resolve_type_from_name('int')", "<class 'int'>", ">>> resolve_type_from_name('unknown') is None", "True"],
        hint="Lookup type_name in a mapping dict: {'int': int, 'str': str, 'float': float, 'bool': bool, 'list': list, 'dict': dict, 'set': set, 'tuple': tuple}.",
        solution="    types_map = {\n        'int': int, 'str': str, 'float': float, 'bool': bool,\n        'list': list, 'dict': dict, 'set': set, 'tuple': tuple\n    }\n    return types_map.get(type_name.strip())",
        test="assert resolve_type_from_name('int') is int\nassert resolve_type_from_name('str') is str\nassert resolve_type_from_name('dict') is dict\nassert resolve_type_from_name('foobar') is None\n"
    ))

    # 44. Dataclass Clone With Modifications
    tasks.append(create_task(
        name="dataclass_clone_with",
        signature="def dataclass_clone_with(dc_instance: Any, **changes: Any) -> Any:",
        prefix_code="import dataclasses\nfrom typing import Any\n",
        doc_desc="Creates a copy of dc_instance with specified field changes using dataclasses.replace.",
        doctests=[">>> @dataclasses.dataclass\n... class D: x: int; y: int\n>>> d2 = dataclass_clone_with(D(1, 2), y=99)\n>>> d2.y", "99"],
        hint="Use dataclasses.replace(dc_instance, **changes).",
        solution="    return dataclasses.replace(dc_instance, **changes)",
        test="@dataclasses.dataclass\nclass User:\n    id: int\n    role: str\nu1 = User(1, 'guest')\nu2 = dataclass_clone_with(u1, role='admin')\nassert u2.id == 1 and u2.role == 'admin'\nassert u1.role == 'guest'\n"
    ))

    # 45. Check Duck Type Attributes
    tasks.append(create_task(
        name="check_duck_type_attributes",
        signature="def check_duck_type_attributes(obj: Any, required_attrs: list[str]) -> bool:",
        doc_desc="Returns True if obj possesses all attributes or methods named in required_attrs.",
        doctests=[">>> class Duck: def quack(self): pass\n>>> check_duck_type_attributes(Duck(), ['quack'])", "True"],
        hint="Return all(hasattr(obj, attr) for attr in required_attrs).",
        solution="    return all(hasattr(obj, attr) for attr in required_attrs)",
        test="class Reader:\n    def read(self): pass\n    def close(self): pass\nassert check_duck_type_attributes(Reader(), ['read', 'close']) is True\nassert check_duck_type_attributes(Reader(), ['read', 'write']) is False\n"
    ))

    # 46. Validate String Literal Set
    tasks.append(create_task(
        name="validate_string_literal",
        signature="def validate_string_literal(val: Any, allowed: set[str]) -> bool:",
        doc_desc="Returns True if val is a string and exists in the allowed set of string literals.",
        doctests=[">>> validate_string_literal('asc', {'asc', 'desc'})", "True", ">>> validate_string_literal('other', {'asc', 'desc'})", "False"],
        hint="Check isinstance(val, str) and val in allowed.",
        solution="    return isinstance(val, str) and val in allowed",
        test="assert validate_string_literal('GET', {'GET', 'POST', 'PUT'}) is True\nassert validate_string_literal('DELETE', {'GET', 'POST'}) is False\nassert validate_string_literal(123, {'123'}) is False\n"
    ))

    # 47. Count Positional-Only Parameters
    tasks.append(create_task(
        name="count_positional_only_params",
        signature="def count_positional_only_params(func: Callable) -> int:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Counts parameters that are positional-only (def f(a, /, b)) using inspect.",
        doctests=[">>> def f(a, /, b): pass\n>>> count_positional_only_params(f)", "1"],
        hint="Count parameters where p.kind == inspect.Parameter.POSITIONAL_ONLY.",
        solution="    sig = inspect.signature(func)\n    return sum(1 for p in sig.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_ONLY)",
        test="def f(x, y, /, z): pass\nassert count_positional_only_params(f) == 2\ndef g(a, b): pass\nassert count_positional_only_params(g) == 0\n"
    ))

    # 48. Type-Guarded List
    tasks.append(create_task(
        name="TypeGuardedList",
        signature="class TypeGuardedList(list):",
        prefix_code="from typing import Any\n",
        doc_desc="List subclass enforcing that appended or extended items match elem_type. Raises TypeError on violation.",
        doctests=[">>> tl = TypeGuardedList(int)\n>>> tl.append(1)\n>>> tl", "[1]"],
        hint="In __init__, store self.elem_type. In append(x) and extend(items), validate isinstance (strict int vs bool).",
        solution="    def __init__(self, elem_type: type):\n        super().__init__()\n        self.elem_type = elem_type\n    def _check(self, x: Any) -> None:\n        if self.elem_type is int and isinstance(x, bool):\n            raise TypeError('Expected int, got bool')\n        if not isinstance(x, self.elem_type):\n            raise TypeError(f'Expected {self.elem_type.__name__}, got {type(x).__name__}')\n    def append(self, item: Any) -> None:\n        self._check(item)\n        super().append(item)\n    def extend(self, iterable: Any) -> None:\n        items = list(iterable)\n        for x in items:\n            self._check(x)\n        super().extend(items)",
        test="tl = TypeGuardedList(str)\ntl.append('a')\ntl.extend(['b', 'c'])\nassert tl == ['a', 'b', 'c']\ntry:\n    tl.append(123)\n    assert False\nexcept TypeError:\n    pass\n"
    ))

    # 49. Inspect Local Variable Names
    tasks.append(create_task(
        name="inspect_local_variable_names",
        signature="def inspect_local_variable_names(func: Callable) -> tuple[str, ...]:",
        prefix_code="from typing import Callable\n",
        doc_desc="Returns a tuple of local variable and argument names defined in func using its code object (__code__.co_varnames).",
        doctests=[">>> def f(a, b): c = 1", ">>> inspect_local_variable_names(f)", "('a', 'b', 'c')"],
        hint="Return func.__code__.co_varnames.",
        solution="    return func.__code__.co_varnames",
        test="def calc(x, y):\n    z = x + y\n    return z\nassert inspect_local_variable_names(calc) == ('x', 'y', 'z')\n"
    ))

    # 50. Validate Shape and Element Type
    tasks.append(create_task(
        name="validate_shape_and_type",
        signature="def validate_shape_and_type(arr: list[Any], expected_len: int, elem_type: type) -> bool:",
        doc_desc="Validates that arr is a list of exact length expected_len and all elements are of elem_type (strict int vs bool).",
        doctests=[">>> validate_shape_and_type([1, 2], 2, int)", "True", ">>> validate_shape_and_type([1], 2, int)", "False"],
        hint="If not isinstance(arr, list) or len(arr) != expected_len, return False. For x in arr: check elem_type.",
        solution="    if not isinstance(arr, list) or len(arr) != expected_len:\n        return False\n    for x in arr:\n        if elem_type is int and isinstance(x, bool):\n            return False\n        if not isinstance(x, elem_type):\n            return False\n    return True",
        test="assert validate_shape_and_type([10, 20, 30], 3, int) is True\nassert validate_shape_and_type([10, True, 30], 3, int) is False\nassert validate_shape_and_type([10], 3, int) is False\n"
    ))

    # 51. Read-Only Mapping Wrapper
    tasks.append(create_task(
        name="ReadOnlyMapping",
        signature="class ReadOnlyMapping:",
        prefix_code="from typing import Any\n",
        doc_desc="Read-only dictionary wrapper implementing __getitem__, __iter__, __len__, and raising TypeError on __setitem__ or __delitem__.",
        doctests=[">>> rom = ReadOnlyMapping({'a': 1})", ">>> rom['a']", "1"],
        hint="Wrap self._data = dict(data). __setitem__ and __delitem__ raise TypeError('ReadOnlyMapping cannot be modified').",
        solution="    def __init__(self, data: dict):\n        self._data = dict(data)\n    def __getitem__(self, key: Any) -> Any:\n        return self._data[key]\n    def __iter__(self):\n        return iter(self._data)\n    def __len__(self) -> int:\n        return len(self._data)\n    def __setitem__(self, key: Any, val: Any) -> None:\n        raise TypeError('ReadOnlyMapping is immutable')\n    def __delitem__(self, key: Any) -> None:\n        raise TypeError('ReadOnlyMapping is immutable')",
        test="r = ReadOnlyMapping({'x': 10, 'y': 20})\nassert r['x'] == 10 and len(r) == 2\nassert set(r) == {'x', 'y'}\ntry:\n    r['x'] = 99\n    assert False\nexcept TypeError:\n    pass\n"
    ))

    # 52. Unwrap Function Wrappers
    tasks.append(create_task(
        name="unwrap_function_chain",
        signature="def unwrap_function_chain(func: Callable) -> Callable:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Unwraps all layers of decorators applied with functools.wraps using inspect.unwrap.",
        doctests=[">>> def f(): pass\n>>> unwrap_function_chain(f) is f", "True"],
        hint="Return inspect.unwrap(func).",
        solution="    return inspect.unwrap(func)",
        test="import functools\ndef base(): pass\n@functools.wraps(base)\ndef wrap1(): pass\n@functools.wraps(wrap1)\ndef wrap2(): pass\nassert unwrap_function_chain(wrap2) is base\n"
    ))

    # 53. Validate Matrix Dimensions and Element Types
    tasks.append(create_task(
        name="validate_matrix_dimensions_typed",
        signature="def validate_matrix_dimensions_typed(matrix: list[list[Any]], rows: int, cols: int, elem_type: type) -> bool:",
        doc_desc="Validates that matrix is a list of rows lists, each of length cols, and all elements match elem_type.",
        doctests=[">>> validate_matrix_dimensions_typed([[1, 2], [3, 4]], 2, 2, int)", "True"],
        hint="Check len(matrix) == rows. For row in matrix: check len(row) == cols, check elements match elem_type (strict int vs bool).",
        solution="    if not isinstance(matrix, list) or len(matrix) != rows:\n        return False\n    for r in matrix:\n        if not isinstance(r, list) or len(r) != cols:\n            return False\n        for x in r:\n            if elem_type is int and isinstance(x, bool):\n                return False\n            if not isinstance(x, elem_type):\n                return False\n    return True",
        test="m = [[1, 2], [3, 4]]\nassert validate_matrix_dimensions_typed(m, 2, 2, int) is True\nassert validate_matrix_dimensions_typed(m, 2, 3, int) is False\nassert validate_matrix_dimensions_typed([[1, True]], 1, 2, int) is False\n"
    ))

    # 54. Simple Singleton Metaclass
    tasks.append(create_task(
        name="SingletonMeta",
        signature="class SingletonMeta(type):",
        prefix_code="from typing import Any\n",
        doc_desc="Metaclass ensuring that classes using it only ever instantiate a single shared instance.",
        doctests=[">>> class S(metaclass=SingletonMeta): pass", ">>> S() is S()", "True"],
        hint="Store _instances = {} on metaclass. In __call__, if cls not in _instances: _instances[cls] = super().__call__(*args, **kwargs); return _instances[cls].",
        solution="    _instances = {}\n    def __call__(cls, *args: Any, **kwargs: Any) -> Any:\n        if cls not in cls._instances:\n            cls._instances[cls] = super().__call__(*args, **kwargs)\n        return cls._instances[cls]",
        test="class AppConfig(metaclass=SingletonMeta):\n    def __init__(self):\n        self.counter = 0\na1 = AppConfig()\na2 = AppConfig()\nassert a1 is a2\na1.counter = 42\nassert a2.counter == 42\n"
    ))

    # 55. Generate Function Signature String
    tasks.append(create_task(
        name="format_signature_string",
        signature="def format_signature_string(func_name: str, params: list[tuple[str, str | None]], return_type: str | None = None) -> str:",
        doc_desc="Builds a Python function header string e.g. 'def add(a: int, b: int) -> int:'.",
        doctests=[">>> format_signature_string('add', [('a', 'int'), ('b', 'int')], 'int')", "'def add(a: int, b: int) -> int:'"],
        hint="Format each param as f'{name}: {t}' if t else name. Join with ', '. Append f' -> {return_type}:' if return_type else ':'.",
        solution="    param_strs = []\n    for name, p_type in params:\n        if p_type:\n            param_strs.append(f'{name}: {p_type}')\n        else:\n            param_strs.append(name)\n    params_formatted = ', '.join(param_strs)\n    ret_formatted = f' -> {return_type}:' if return_type else ':'\n    return f'def {func_name}({params_formatted}){ret_formatted}'",
        test="assert format_signature_string('run', [('cmd', 'str'), ('timeout', 'int')], 'bool') == 'def run(cmd: str, timeout: int) -> bool:'\nassert format_signature_string('noop', []) == 'def noop():'\n"
    ))

    # 56. Extract Return Value Annotation String
    tasks.append(create_task(
        name="get_return_annotation_name",
        signature="def get_return_annotation_name(func: Callable) -> str:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Returns the string name of the return type annotation of func (e.g. 'int', 'NoneType'), or 'empty' if unannotated.",
        doctests=[">>> def f() -> int: pass\n>>> get_return_annotation_name(f)", "'int'"],
        hint="ret = inspect.signature(func).return_annotation. If inspect.Signature.empty, return 'empty'. If isinstance(ret, type), return ret.__name__, else str(ret).",
        solution="    ret = inspect.signature(func).return_annotation\n    if ret is inspect.Signature.empty:\n        return 'empty'\n    if isinstance(ret, type):\n        return ret.__name__\n    return str(ret)",
        test="def f1() -> str: pass\nassert get_return_annotation_name(f1) == 'str'\ndef f2(): pass\nassert get_return_annotation_name(f2) == 'empty'\n"
    ))

    # 57. Check Function Accepts Kwargs
    tasks.append(create_task(
        name="accepts_arbitrary_kwargs",
        signature="def accepts_arbitrary_kwargs(func: Callable) -> bool:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Returns True if func has a **kwargs parameter (inspect.Parameter.VAR_KEYWORD).",
        doctests=[">>> def f(**kwargs): pass\n>>> accepts_arbitrary_kwargs(f)", "True"],
        hint="Return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(func).parameters.values()).",
        solution="    sig = inspect.signature(func)\n    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())",
        test="assert accepts_arbitrary_kwargs(lambda **kwargs: None) is True\nassert accepts_arbitrary_kwargs(lambda a, b: None) is False\n"
    ))

    # 58. Validate Nested List Depth
    tasks.append(create_task(
        name="get_nested_list_depth",
        signature="def get_nested_list_depth(item: Any) -> int:",
        doc_desc="Returns the maximum nesting depth of lists. Returns 0 if item is not a list. [] has depth 1.",
        doctests=[">>> get_nested_list_depth([[1], [2, [3]]])", "3", ">>> get_nested_list_depth(42)", "0"],
        hint="If not isinstance(item, list) return 0. If len(item) == 0 return 1. Return 1 + max(get_nested_list_depth(x) for x in item).",
        solution="    if not isinstance(item, list):\n        return 0\n    if not item:\n        return 1\n    return 1 + max(get_nested_list_depth(x) for x in item)",
        test="assert get_nested_list_depth(42) == 0\nassert get_nested_list_depth([]) == 1\nassert get_nested_list_depth([1, 2]) == 1\nassert get_nested_list_depth([[1], [2, [3]]]) == 3\n"
    ))

    # 59. Dynamic Attribute Loader
    tasks.append(create_task(
        name="load_attribute_from_module",
        signature="def load_attribute_from_module(module_name: str, attr_name: str, default: Any = None) -> Any:",
        prefix_code="import importlib\nfrom typing import Any\n",
        doc_desc="Dynamically imports module_name and retrieves attr_name. Returns default if module or attribute does not exist.",
        doctests=[">>> load_attribute_from_module('math', 'pi')", "3.141592653589793"],
        hint="try importlib.import_module(module_name); getattr(mod, attr_name). Catch (ImportError, AttributeError).",
        solution="    try:\n        mod = importlib.import_module(module_name)\n        return getattr(mod, attr_name)\n    except (ImportError, AttributeError):\n        return default",
        test="assert load_attribute_from_module('math', 'sqrt')(4.0) == 2.0\nassert load_attribute_from_module('math', 'non_existent_fn', 'fallback') == 'fallback'\nassert load_attribute_from_module('non_module', 'xyz', None) is None\n"
    ))

    # 60. Auto-Repr Generator Class
    tasks.append(create_task(
        name="AutoReprBase",
        signature="class AutoReprBase:",
        doc_desc="Base class providing __repr__ that outputs f'{ClassName}(key1=val1, key2=val2)' sorted by key from __dict__.",
        doctests=[">>> class P(AutoReprBase):\n...     def __init__(self, x, y): self.x = x; self.y = y\n>>> str(P(1, 2))", "'P(x=1, y=2)'"],
        hint="In __repr__, pairs = [f'{k}={v!r}' for k, v in sorted(self.__dict__.items()) if not k.startswith('_')]. Return f'{self.__class__.__name__}({', '.join(pairs)})'.",
        solution="    def __repr__(self) -> str:\n        pairs = [f'{k}={v!r}' for k, v in sorted(self.__dict__.items()) if not k.startswith('_')]\n        return f'{self.__class__.__name__}({\", \".join(pairs)})'",
        test="class Point(AutoReprBase):\n    def __init__(self, a, b):\n        self.a = a\n        self.b = b\np = Point(10, 'test')\nassert repr(p) == \"Point(a=10, b='test')\"\n"
    ))

    # 61. Validate Callable Parameter Count Range
    tasks.append(create_task(
        name="can_accept_arg_count",
        signature="def can_accept_arg_count(func: Callable, arg_count: int) -> bool:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Checks if func can be invoked with exactly arg_count positional arguments without TypeError.",
        doctests=[">>> can_accept_arg_count(lambda a, b=1: None, 1)", "True", ">>> can_accept_arg_count(lambda a, b=1: None, 2)", "True"],
        hint="Check callable(func). sig = inspect.signature(func). Try sig.bind(*([None] * arg_count)). Return True on success, False on TypeError.",
        solution="    if not callable(func):\n        return False\n    try:\n        sig = inspect.signature(func)\n        sig.bind(*([None] * arg_count))\n        return True\n    except TypeError:\n        return False",
        test="def f(x, y=2, z=3): pass\nassert can_accept_arg_count(f, 1) is True\nassert can_accept_arg_count(f, 2) is True\nassert can_accept_arg_count(f, 3) is True\nassert can_accept_arg_count(f, 4) is False\nassert can_accept_arg_count(f, 0) is False\n"
    ))

    # 62. Check Subclass Relationship Safely
    tasks.append(create_task(
        name="safe_issubclass",
        signature="def safe_issubclass(candidate: Any, base_class: type) -> bool:",
        doc_desc="Returns issubclass(candidate, base_class) if candidate is a class, otherwise returns False.",
        doctests=[">>> safe_issubclass(int, object)", "True", ">>> safe_issubclass('not_a_class', object)", "False"],
        hint="if not isinstance(candidate, type): return False. Return issubclass(candidate, base_class).",
        solution="    if not isinstance(candidate, type):\n        return False\n    return issubclass(candidate, base_class)",
        test="assert safe_issubclass(bool, int) is True\nassert safe_issubclass(int, str) is False\nassert safe_issubclass(123, int) is False\n"
    ))

    # 63. Inspect Function Has Default Value
    tasks.append(create_task(
        name="has_default_value",
        signature="def has_default_value(func: Callable, param_name: str) -> bool:",
        prefix_code="import inspect\nfrom typing import Callable\n",
        doc_desc="Returns True if parameter param_name in func has a default value defined.",
        doctests=[">>> def f(a, b=10): pass\n>>> has_default_value(f, 'b')", "True"],
        hint="sig = inspect.signature(func). If param_name not in sig.parameters: return False. Return sig.parameters[param_name].default is not inspect.Parameter.empty.",
        solution="    sig = inspect.signature(func)\n    if param_name not in sig.parameters:\n        return False\n    return sig.parameters[param_name].default is not inspect.Parameter.empty",
        test="def demo(x, y=5): pass\nassert has_default_value(demo, 'y') is True\nassert has_default_value(demo, 'x') is False\nassert has_default_value(demo, 'missing') is False\n"
    ))

    # 64. Extract Module Export Names (__all__ or public)
    tasks.append(create_task(
        name="get_module_exports",
        signature="def get_module_exports(mod: Any) -> list[str]:",
        prefix_code="from typing import Any\n",
        doc_desc="Returns sorted list of exported names: mod.__all__ if defined, otherwise all attributes not starting with '_'.",
        doctests=[">>> import math\n>>> 'sin' in get_module_exports(math)", "True"],
        hint="If hasattr(mod, '__all__'): return sorted(list(mod.__all__)). Else return sorted([k for k in dir(mod) if not k.startswith('_')]).",
        solution="    if hasattr(mod, '__all__'):\n        return sorted(list(mod.__all__))\n    return sorted([k for k in dir(mod) if not k.startswith('_')])",
        test="class FakeMod:\n    __all__ = ['b', 'a']\nassert get_module_exports(FakeMod) == ['a', 'b']\nclass FakeMod2:\n    x = 1\n    _private = 2\nassert get_module_exports(FakeMod2) == ['x']\n"
    ))

    return tasks
